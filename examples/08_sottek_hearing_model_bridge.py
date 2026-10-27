"""ROTA and the Sottek hearing model: ECMA-418-2 metrics along rpm, attributed to orders.

ROTA computes no psychoacoustic metric (it is out of scope); it prepares the signal for
one, and it bins the time-varying result along rpm. This script links it to the Python
implementation of the Sottek hearing model of ECMA-418-2, ``sottek-hearing-model``
(import name ``sottek_hearing_model``, PyPI 0.1.14, repository
github.com/mlotinga/sottek-hearing-model). The package implements ECMA-418-2:2025 (4th
edition) time-dependent tonality (with the frequency of the dominant tonal component),
loudness and roughness. Fluctuation strength is not part of the release yet (the package
README announces it for a future version), so it is not used here.

Optional external dependency, not part of ROTA. The package is licensed AGPL-3.0-only
(package metadata) and needs Python 3.11 or later; it pulls in PyQt6, Matplotlib,
Bottleneck and tqdm. ROTA does not import it, vendor it or copy its code; only the
documented public functions ``shm_tonality_ecma``, ``shm_loudness_ecma_from_comp`` and
``shm_roughness_ecma`` are called. Install it yourself if you accept its licence:
``pip install sottek-hearing-model``. Without it this script stops with that message.

The signals come from ``rota.synth.synth_scenario``: ``"ice_runup"`` (a four-stroke
four-cylinder engine, firing order 2, no gear shift in the first 6 s) and ``"ev_runup"``
(motor orders 2, 4, 8, 10, 12, 14, gear-mesh order 6.7, gear whine at order 40, a dithered
10 kHz inverter carrier and two fixed tones, 8 s), both seeded, mono, 32 kHz.

Chain:

1. Speed from the tachometer channel (``rota.speed.tacho_to_rpm``, conditioning delay
   compensated) and calibrated pressure at 48 kHz (``rota.tonal.to_calibrated_48k``,
   which also fixes the unit: Pa, calibration factor 1 because the synthetic signal is
   already in Pa). The model expects sound pressure in Pa at its own rate.
2. The model: overall time-dependent tonality (187.5 Hz frame rate) with the frequency of
   its dominant tonal component, the specific (per half critical band) tonality with its
   tonal frequencies, loudness (derived from the tonal and noise loudness of the tonality
   run) and roughness (50 Hz frame rate). Free-frontal sound field, the package default.
3. Binning along rpm with ``rota.slices.metric_vs_rpm``. Tonality: the mean of the frames
   above 0.02 tu_HMS (frames below are set to NaN, which ``metric_vs_rpm`` discards), the
   rule ECMA-418-2 uses for its single value. Loudness: arithmetic mean of the
   time-dependent loudness (the package's ``loudness_powavg`` is a power average and is
   only printed for comparison). Roughness: 90th percentile, the standard's single value.
   ``skip_start`` drops the start-up block of each metric: 0.30 s (57 frames) for
   tonality and loudness, 0.31 s (the first 16 of the 50 Hz frames) for roughness, the
   exclusions that the package applies to its own single values. The script checks that
   ``metric_vs_rpm`` with one bin spanning the whole run reproduces the package's
   ``tonality_avg`` and ``roughness90pc``, so the per-bin values are those single values
   restricted to a speed bin.
4. Order attribution of the model's tonal frequencies: ``rota.slices.frequency_to_order``
   (frequency / (rpm / 60), nearest candidate order, 10 % tolerance) and
   ``rota.slices.order_share`` (share on a base order and its harmonics, optionally
   weighted by tonality). Two views are reported: the frequency of the overall tonality
   (one dominant tone per frame) and all half-band tonal components above the threshold,
   weighted by their specific tonality. The overall tonality is the largest specific
   tonality of the bands, so for the electric drive it follows the low-frequency rumble,
   not the weaker whine; the band-wise view shows the whine. A tone can appear in several
   neighbouring half bands, so band-wise shares count bands weighted by tonality, not
   tones. The 10 % window is wide because the model's frequency reading is coarse in its
   lowest bands (the engine's firing frequency reads 7 % low below 50 Hz, about 1 % low
   above 70 Hz; the script prints both); an order without a source (11 for the engine, 34
   for the drive) gives the chance level of the attribution.
5. Order contribution: the whine (order 40) is removed with ``rota.extract.remove_orders``
   (tracked harmonics, 4 revolutions, as in example 06) and the model runs again.
   The overall tonality along rpm is compared with and without it, and so is the
   order-attributed tonality: the largest specific tonality of a half band whose tonal
   frequency lies on the order (0 where there is none).

Units and conventions. The model's outputs keep the standard's units: tonality in tu_HMS,
loudness in sone_HMS, roughness in asper_HMS; ``metric_vs_rpm`` bins them as given. The
signal is sound pressure in Pa (the synthetic source orders are peak amplitudes, noise is
RMS; the model works on the pressure waveform, so no amplitude convention enters). Speed
in rpm, frequency in Hz, time in s.

The truth decides what the numbers mean. The scenario knows its orders, so the share of
the model's tonal frequencies on the true orders, the deviation of those frequencies from
``order * rpm / 60`` and the effect of removing the whine can be judged.

Run:  python examples/08_sottek_hearing_model_bridge.py
Needs matplotlib and sottek-hearing-model. About 100 s on four cores (the model dominates;
by default it uses all cores but one, ``PARALLEL_CORES = 1`` is about three times slower).
Writes docs/images/08_sottek_hearing_model_bridge.png.

Toolboxes used (cite them as their authors ask; ROTA does not include or redistribute
their code, which keeps its own licence):

* Lotinga, M. J. B., Torjussen, M., & Felix Greco, G. (2025). Verified implementations of
  the Sottek psychoacoustic Hearing Model standardised sound quality metrics (ECMA-418-2
  loudness, tonality and roughness). Proceedings of Forum Acusticum / Euronoise 2025,
  Malaga, Spain. Software: sottek-hearing-model 0.1.14,
  https://github.com/mlotinga/sottek-hearing-model (AGPL-3.0-only).

"""

import sys
import time
from pathlib import Path

import numpy as np

try:
    from sottek_hearing_model import (
        shm_loudness_ecma_from_comp,
        shm_roughness_ecma,
        shm_tonality_ecma,
    )
except ImportError:
    sys.exit(
        "This example needs the optional package sottek-hearing-model (ECMA-418-2 Sottek hearing\n"
        "model, AGPL-3.0-only, Python >= 3.11): pip install sottek-hearing-model"
    )

import matplotlib.pyplot as plt

import rota

FS = 32000.0  # sampling rate of the scenarios, Hz
FS_MODEL = 48000.0  # rate the model is run at, Hz
SEED = 3
ICE_DURATION = 6.0  # s; the first upshift of the default scenario is later than this
EV_DURATION = 8.0  # s
EV_OVERRIDES = {"tacho": {"ppr": 24}}  # 48 pulses would be shorter than 2 samples at 32 kHz above 10000 rpm
WHINE_ORDER = 40.0
WHINE_WINDOW_REVS = 4.0  # tracked-harmonics frame length for the whine, revolutions
SOUNDFIELD = "free_frontal"  # package default
PARALLEL_CORES = None  # None: all cores but one; 1: no parallel processing

TONALITY_THRESHOLD = 0.02  # tu_HMS; frames below are not averaged (ECMA-418-2 single value)
SKIP_TONALITY = 0.30  # s; the model's 57 start-up frames end at 0.304 s, so frame 57 is kept
SKIP_ROUGHNESS = 0.31  # s; the first 16 frames of 50 Hz end at 0.32 s
SHOW_THRESHOLD = 0.15  # tu_HMS; weaker components are left out of the scatter plots only
TOLERANCE = 0.10  # relative tolerance of the order attribution (see the module docstring)
BIN_ICE, BIN_EV = 300.0, 1000.0  # rpm bin widths
MIN_COUNT = 5  # samples a bin needs
ROUGHNESS_PERCENTILE = 90
CONTROL_ORDER = {"ice_runup": 11.0, "ev_runup": 34.0}  # no source at these orders (chance level)

ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = ROOT / "docs" / "images" / "08_sottek_hearing_model_bridge.png"
BLUE, ORANGE, GREEN, GREY, RED = "#0072B2", "#D55E00", "#009E73", "#555555", "#CC3311"


# ---- model runs ----------------------------------------------------------------------------------


def run_model(p, roughness=True):
    """Tonality, loudness and (optionally) roughness of a mono pressure signal at 48 kHz, in Pa."""
    kw = {"soundfield": SOUNDFIELD, "wait_bar": False, "parallel_cores": PARALLEL_CORES}
    ton = shm_tonality_ecma(p, FS_MODEL, **kw)
    lou = shm_loudness_ecma_from_comp(ton["spec_tonal_loudness"], ton["spec_noise_loudness"])
    rou = shm_roughness_ecma(p, FS_MODEL, **kw) if roughness else None
    return {"ton": ton, "lou": lou, "rou": rou}


def prepare(sc, ppr, removed_orders=None):
    """Tacho speed, and the 48 kHz pressure of the recording (and of the recording without orders)."""
    delay = float(sc.config["resolved"]["tacho_delay"])
    rpm = rota.speed.tacho_to_rpm(sc.tacho, sc.fs, ppr=ppr, t_out=sc.t + delay)
    x = sc.x
    if removed_orders is not None:
        x, _ = rota.extract.remove_orders(
            x, sc.fs, rpm, removed_orders, method="tracked-harmonics", window_revs=WHINE_WINDOW_REVS
        )
    # calibration=1.0: the synthetic signal is already in Pa. Sample 0 stays at t = 0.
    return rpm, rota.tonal.to_calibrated_48k(x, sc.fs, calibration=1.0, fs_out=FS_MODEL)


def speed_axis(sc, rpm):
    """Time axis and speed for ``metric_vs_rpm``, held for 10 ms beyond the last sample.

    The model's last frame sits at the duration of the record, one sample after the last speed
    sample, and ``metric_vs_rpm`` never extrapolates the speed: without this hold that frame would
    be dropped from the bins.
    """
    return np.append(sc.t, sc.t[-1] + 0.01), np.append(rpm, rpm[-1])


# ---- binning along rpm -----------------------------------------------------------------------------


def vs_rpm(y, t_y, rpm, t_rpm, width, skip, aggregate):
    return rota.slices.metric_vs_rpm(
        y, t_y, rpm, t_rpm, bin_width=width, skip_start=skip, aggregate=aggregate, min_count=MIN_COUNT
    )


def one_value(y, t_y, rpm, t_rpm, skip, aggregate):
    """metric_vs_rpm with a single bin over the whole run: the single value of the metric."""
    res = rota.slices.metric_vs_rpm(
        y, t_y, rpm, t_rpm, segment="all", bins=[0.0, 1.0e6], skip_start=skip, aggregate=aggregate, min_count=1
    )
    return float(res.value[0])


def above_threshold(tonality):
    return np.where(tonality > TONALITY_THRESHOLD, tonality, np.nan)


def order_tonality(spec, freqs, rpm_m, order):
    """Largest specific tonality among the half bands whose tonal frequency lies on ``order`` (else 0)."""
    f = np.where(spec > TONALITY_THRESHOLD, freqs, np.nan)
    fo = rota.slices.frequency_to_order(f, rpm_m[:, None], tolerance=TOLERANCE, orders=[order])
    return np.max(np.where(fo.matched, spec, 0.0), axis=1)


# ---- order attribution ---------------------------------------------------------------------------------


def band_attribution(spec, freqs, rpm_m, t_m, candidates):
    """Share of the tonality of all half-band tonal components on each candidate order.

    Components: specific tonality above the threshold after the start-up frames. Weight: the
    specific tonality. Returns {order: share}, the share on no candidate ("other") and the
    median relative deviation of the matched frequencies from the order lines.
    """
    keep = t_m >= SKIP_TONALITY
    s, f, r = spec[keep], freqs[keep], rpm_m[keep]
    on = s > TONALITY_THRESHOLD
    fo = rota.slices.frequency_to_order(np.where(on, f, np.nan), r[:, None], tolerance=TOLERANCE, orders=candidates)
    total = float(s[on & np.isfinite(fo.order)].sum())
    share = {o: float(s[on & fo.matched & (fo.nearest == o)].sum() / total) for o in candidates}
    other = 1.0 - sum(share.values())
    dev = np.abs(fo.deviation[on & fo.matched])
    return share, other, float(np.median(dev))


def dominant_attribution(res, rpm_m, t_m, base, harmonics):
    """Share of the frames' dominant tonal frequency (overall tonality) on ``base`` and its harmonics."""
    ton = res["ton"]
    keep = (t_m >= SKIP_TONALITY) & (ton["tonality_t"] > TONALITY_THRESHOLD)
    f = np.where(keep, ton["tonality_t_freqs"], np.nan)
    return rota.slices.order_share(f, rpm_m, base, harmonics=harmonics, tolerance=TOLERANCE), float(np.nanmedian(f))


def main():
    plt.switch_backend("Agg")  # file output only; no display needed
    t_start = time.time()

    # ---- signals ----------------------------------------------------------------------------------
    ice = rota.synth.synth_scenario("ice_runup", fs=FS, duration=ICE_DURATION, seed=SEED)
    ev = rota.synth.synth_scenario("ev_runup", fs=FS, duration=EV_DURATION, seed=SEED, **EV_OVERRIDES)
    ice_ppr, ev_ppr = int(ice.config["parameters"]["tacho"]["ppr"]), int(ev.config["parameters"]["tacho"]["ppr"])
    firing = float(ice.config["parameters"]["combustion"]["per_rev"])
    assert ice.events["shift_times"].size == 0, "the engine record must stay in one gear"
    rpm_ice, p_ice = prepare(ice, ice_ppr)
    rpm_ev, p_ev = prepare(ev, ev_ppr)
    rpm_ev_nw, p_ev_nw = prepare(ev, ev_ppr, removed_orders=[WHINE_ORDER])
    print(f"Signals (synth_scenario, seed {SEED}, mono, {FS:g} Hz -> to_calibrated_48k -> {FS_MODEL:g} Hz, Pa):")
    print(f"  ice_runup {ICE_DURATION:g} s, {rpm_ice[50]:.0f} -> {rpm_ice[-50]:.0f} rpm, firing order {firing:g}, "
          f"RMS {20 * np.log10(np.std(p_ice) / 20e-6):.1f} dB re 20 uPa")
    print(f"  ev_runup  {EV_DURATION:g} s, {np.nanmin(rpm_ev[50:]):.0f} -> {np.nanmax(rpm_ev):.0f} rpm, "
          f"orders {ev.orders.tolist()}, RMS {20 * np.log10(np.std(p_ev) / 20e-6):.1f} dB re 20 uPa")

    # ---- model ------------------------------------------------------------------------------------
    print("Sottek hearing model (ECMA-418-2:2025), free-frontal field:")
    runs = {}
    for name, p in (("ice", p_ice), ("ev", p_ev), ("ev_nw", p_ev_nw)):
        t0 = time.time()
        runs[name] = run_model(p, roughness=name != "ev_nw")
        print(f"  {name:6s} tonality, loudness{'' if name == 'ev_nw' else ', roughness'}: {time.time() - t0:5.1f} s")

    def prep_run(name, sc, rpm):
        r = runs[name]
        t_m = np.asarray(r["ton"]["time_out"], float)
        rpm_m = np.interp(t_m, sc.t, rpm)  # speed on the model's 187.5 Hz frame grid
        t_rpm, rpm_h = speed_axis(sc, rpm)
        r["t_m"], r["rpm_m"], r["rpm"], r["t_rpm"] = t_m, rpm_m, rpm_h, t_rpm
        r["t_r"] = np.asarray(r["rou"]["time_out"], float) if r["rou"] else None
        return r

    R = {"ice": prep_run("ice", ice, rpm_ice), "ev": prep_run("ev", ev, rpm_ev), "ev_nw": prep_run("ev_nw", ev, rpm_ev_nw)}
    for r in R.values():
        assert r["ton"]["tonality_t"].ndim == 1, "the example is for one channel"

    # ---- single values: does metric_vs_rpm with one bin reproduce the package's? -----------------------
    print("Single values: metric_vs_rpm with one bin over the whole run against the package's own:")
    for name in ("ice", "ev"):
        r = R[name]
        t_avg = one_value(above_threshold(r["ton"]["tonality_t"]), r["t_m"], r["rpm"], r["t_rpm"], SKIP_TONALITY, "mean")
        r90 = one_value(r["rou"]["roughness_t"], r["t_r"], r["rpm"], r["t_rpm"], SKIP_ROUGHNESS,
                        ("percentile", ROUGHNESS_PERCENTILE))
        l_mean = one_value(r["lou"]["loudness_t"], r["t_m"], r["rpm"], r["t_rpm"], SKIP_TONALITY, "mean")
        print(f"  {name:3s} tonality   {t_avg:.4f} tu_HMS     package tonality_avg {float(r['ton']['tonality_avg'][0]):.4f}")
        print(f"      roughness  {r90:.4f} asper_HMS  package roughness90pc {float(r['rou']['roughness90pc'][0]):.4f}")
        print(f"      loudness   {l_mean:.3f} sone_HMS (mean)   package loudness_powavg "
              f"{float(r['lou']['loudness_powavg'][0]):.3f} (power average)")
        assert np.isclose(t_avg, r["ton"]["tonality_avg"][0], rtol=1e-6), "tonality single value differs"
        assert np.isclose(r90, r["rou"]["roughness90pc"][0], rtol=1e-6), "roughness single value differs"

    # ---- metrics along rpm ----------------------------------------------------------------------------------
    for name, width in (("ice", BIN_ICE), ("ev", BIN_EV), ("ev_nw", BIN_EV)):
        r = R[name]
        r["tonality"] = vs_rpm(above_threshold(r["ton"]["tonality_t"]), r["t_m"], r["rpm"], r["t_rpm"],
                               width, SKIP_TONALITY, "mean")
        r["loudness"] = vs_rpm(r["lou"]["loudness_t"], r["t_m"], r["rpm"], r["t_rpm"], width, SKIP_TONALITY, "mean")
        if r["rou"]:
            r["roughness"] = vs_rpm(r["rou"]["roughness_t"], r["t_r"], r["rpm"], r["t_rpm"], width, SKIP_ROUGHNESS,
                                    ("percentile", ROUGHNESS_PERCENTILE))
    order_of = {"ice": firing, "ev": WHINE_ORDER, "ev_nw": WHINE_ORDER}
    for name in R:
        r = R[name]
        a = order_tonality(r["ton"]["spec_tonality"], r["ton"]["spec_tonality_freqs"], r["rpm_m"], order_of[name])
        r["order_tonality"] = vs_rpm(a, r["t_m"], r["rpm"], r["t_rpm"], BIN_ICE if name == "ice" else BIN_EV,
                                     SKIP_TONALITY, "mean")

    def table(label, centres, columns):
        """Print one row per rpm bin; ``columns`` is a list of (header, width, MetricVsRpm)."""
        print(f"{label}: bins of {centres[1] - centres[0]:g} rpm (tonality: mean of frames above "
              f"{TONALITY_THRESHOLD:g} tu_HMS; loudness: mean; roughness: P{ROUGHNESS_PERCENTILE}; "
              f"skip {SKIP_TONALITY:g} s / {SKIP_ROUGHNESS:g} s)")
        print("  rpm bin centre " + "".join(f"{h:>{w}}" for h, w, _ in columns))
        for c in centres:
            cells = []
            for _, w, res in columns:
                j = np.flatnonzero(np.isclose(res.centres, c))
                v = res.value[j[0]] if j.size else np.nan
                cells.append(f"{v:{w}.3f}" if np.isfinite(v) else f"{'-':>{w}}")
            print(f"  {c:14.0f} " + "".join(cells))

    print()
    ri, re_, rn = R["ice"], R["ev"], R["ev_nw"]
    table("Engine run-up", ri["tonality"].centres,
          [("tonality", 11, ri["tonality"]), ("loudness", 11, ri["loudness"]), ("roughness", 11, ri["roughness"]),
           (f"order {firing:g} tonal.", 17, ri["order_tonality"])])
    print()
    table("Electric drive", re_["tonality"].centres,
          [("tonality", 11, re_["tonality"]), ("loudness", 11, re_["loudness"]), ("roughness", 11, re_["roughness"]),
           (f"order {WHINE_ORDER:g} tonal.", 17, re_["order_tonality"]),
           ("tonality, no whine", 20, rn["tonality"]), (f"order {WHINE_ORDER:g}, no whine", 20, rn["order_tonality"])])

    # ---- order attribution -------------------------------------------------------------------------------------
    print("\nOrder attribution of the tonal frequencies (frequency / (rpm / 60), tolerance "
          f"{TOLERANCE * 100:g} %, frames after {SKIP_TONALITY:g} s):")
    att = {}
    for name, sc, cand in (("ice", ice, ice.orders), ("ev", ev, ev.orders)):
        r = R[name]
        cand = np.unique(np.append(cand, CONTROL_ORDER[sc.kind]))
        share, other, dev = band_attribution(r["ton"]["spec_tonality"], r["ton"]["spec_tonality_freqs"],
                                             r["rpm_m"], r["t_m"], cand)
        att[name] = (cand, share, other)
        print(f"  {sc.kind}, half-band components above threshold, weighted by specific tonality "
              f"(median |deviation| of matched frequencies {100 * dev:.1f} %):")
        line = ", ".join(f"{o:g}: {100 * s:.1f} %" + (" (control)" if o == CONTROL_ORDER[sc.kind] else "")
                         for o, s in share.items())
        print(f"    {line}, on no candidate: {100 * other:.1f} %")
    print()
    ri_ = R["ice"]
    tonal = (ri_["t_m"] >= SKIP_TONALITY) & (ri_["ton"]["tonality_t"] > TONALITY_THRESHOLD)
    dev = ri_["ton"]["tonality_t_freqs"] / (firing * ri_["rpm_m"] / 60.0) - 1.0
    lo, hi = tonal & (ri_["rpm_m"] < 1500.0), tonal & (ri_["rpm_m"] >= 2100.0)
    print(f"  ice_runup, dominant tonal frequency against the order-{firing:g} line (truth, median deviation): "
          f"{100 * np.median(dev[lo]):+.1f} % below 1500 rpm (under 50 Hz), {100 * np.median(dev[hi]):+.1f} % above "
          f"2100 rpm (over 70 Hz); the lowest bands of the model resolve frequency coarsely, hence the wide window")
    sh_ice, med_ice = dominant_attribution(R["ice"], R["ice"]["rpm_m"], R["ice"]["t_m"], firing, 8)
    print(f"  ice_runup, dominant tonal frequency of the overall tonality ({sh_ice.n_valid} tonal frames, median "
          f"{med_ice:.0f} Hz): on order {firing:g} and harmonics (up to order {firing * 8:g}) "
          f"{100 * sh_ice.share:.1f} %; per harmonic "
          + ", ".join(f"{k * firing:g}: {100 * s:.1f} %" for k, s in zip(sh_ice.harmonics, sh_ice.share_per_harmonic)))
    sh_ev, med_ev = dominant_attribution(R["ev"], R["ev"]["rpm_m"], R["ev"]["t_m"], 2.0, 7)
    sh_w, _ = dominant_attribution(R["ev"], R["ev"]["rpm_m"], R["ev"]["t_m"], WHINE_ORDER, 1)
    print(f"  ev_runup, dominant tonal frequency of the overall tonality ({sh_ev.n_valid} tonal frames, median "
          f"{med_ev:.0f} Hz, the low-frequency rumble band): motor family 2k (k = 1..7; order 6 has no source) "
          f"{100 * sh_ev.share:.1f} %, whine (order {WHINE_ORDER:g}) {100 * sh_w.share:.1f} %")
    kk = R["ev"]
    keep = kk["t_m"] >= SKIP_TONALITY
    for lbl, key in (("with the whine", "ev"), ("whine removed", "ev_nw")):
        q = R[key]
        sw = rota.slices.order_share(
            np.where(q["ton"]["spec_tonality"] > TONALITY_THRESHOLD, q["ton"]["spec_tonality_freqs"], np.nan)[keep].ravel(),
            np.repeat(q["rpm_m"][keep][:, None], q["ton"]["spec_tonality"].shape[1], axis=1).ravel(),
            WHINE_ORDER, harmonics=1, tolerance=TOLERANCE, weights=q["ton"]["spec_tonality"][keep].ravel())
        print(f"  ev_runup {lbl:15s}: tonality-weighted share of the half-band components on the whine: "
              f"{100 * sw.share:.1f} %")

    # ---- order contribution --------------------------------------------------------------------------------------
    t_with, t_without = float(R["ev"]["ton"]["tonality_avg"][0]), float(R["ev_nw"]["ton"]["tonality_avg"][0])
    print(f"\nWhine contribution (ev_runup, order {WHINE_ORDER:g} removed with remove_orders, tracked harmonics, "
          f"{WHINE_WINDOW_REVS:g} revolutions):")
    print(f"  overall tonality, single value: {t_with:.3f} -> {t_without:.3f} tu_HMS "
          f"({100 * (t_without / t_with - 1):+.1f} %); loudness (power average) "
          f"{float(R['ev']['lou']['loudness_powavg'][0]):.2f} -> {float(R['ev_nw']['lou']['loudness_powavg'][0]):.2f} sone_HMS")
    a_with = np.nanmean(R["ev"]["order_tonality"].value[R["ev"]["order_tonality"].valid])
    a_without = np.nanmean(R["ev_nw"]["order_tonality"].value[R["ev_nw"]["order_tonality"].valid])
    print(f"  order-{WHINE_ORDER:g} tonality, mean over the speed bins: {a_with:.3f} -> {a_without:.3f} tu_HMS")

    # ---- figure ---------------------------------------------------------------------------------------------------------
    # Importing the model package changes Matplotlib's global settings (font family and size);
    # return to the defaults before styling this figure.
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8})
    fig, axes = plt.subplots(4, 2, figsize=(8.0, 10.6), dpi=150, constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.15, 0.8, 0.9, 0.9]})
    titles = {"ice": "Engine run-up (ice_runup)", "ev": "Electric drive (ev_runup)"}
    for col, name in enumerate(("ice", "ev")):
        sc = ice if name == "ice" else ev
        r = R[name]
        spec, freqs, t_m = r["ton"]["spec_tonality"], r["ton"]["spec_tonality_freqs"], r["t_m"]
        keep = (spec > SHOW_THRESHOLD) & (t_m[:, None] >= SKIP_TONALITY) & (freqs > 0)
        # (a, b) tonal components of the model with the order lines
        ax = axes[0, col]
        tt = np.broadcast_to(t_m[:, None], spec.shape)[keep]
        ss, ff = spec[keep], freqs[keep]
        idx = np.argsort(ss)
        sca = ax.scatter(tt[idx], ff[idx], c=ss[idx], s=2.0, cmap="viridis", vmin=0.0, vmax=2.0, rasterized=True,
                         linewidths=0)
        lines = sc.orders if name == "ev" else np.arange(1, 5) * firing
        for o in lines:
            is_whine = name == "ev" and o == WHINE_ORDER
            ax.plot(sc.t, o * sc.rpm_mean / 60.0, color=RED if is_whine else "k", lw=0.9 if is_whine else 0.5,
                    ls=(0, (3, 3)), alpha=0.9)
        ax.set_yscale("log")
        ax.set_xlim(0.0, sc.t[-1])
        ax.set_ylim(20.0, 20000.0 if name == "ev" else 5000.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tonal frequency (Hz)")
        ax.set_title(f"({'ab'[col]}) {titles[name]}: tonal components", loc="left")
        box = {"fc": "w", "ec": "none", "alpha": 0.85, "pad": 1}
        i5 = int(np.searchsorted(sc.t, 0.8 * sc.t[-1]))
        if name == "ice":
            ax.annotate(f"firing order {firing:g}", (sc.t[i5], firing * sc.rpm_mean[i5] / 60.0),
                        xytext=(sc.t[i5] - 1.6, 400.0), fontsize=7, bbox=box,
                        arrowprops={"arrowstyle": "->", "lw": 0.6})
            ax.text(0.15, 3000.0, "1.9 kHz structural resonance\n(combustion bursts, not an order)", ha="left",
                    va="center", fontsize=7, bbox=box)
            ax.text(0.15, 21.0, "dashed: orders 2, 4, 6, 8", ha="left", va="bottom", fontsize=7, bbox=box)
        else:
            ax.annotate("whine (order 40)", (sc.t[i5], WHINE_ORDER * sc.rpm_mean[i5] / 60.0),
                        xytext=(sc.t[i5] - 3.4, 2.2 * WHINE_ORDER * sc.rpm_mean[i5] / 60.0), color=RED, fontsize=7,
                        bbox=box, arrowprops={"arrowstyle": "->", "lw": 0.6, "color": RED})
            ax.text(7.9, 21.0, "rumble band", ha="right", va="bottom", fontsize=7, bbox=box)
            ax.text(0.15, 21.0, "dashed: source orders", ha="left", va="bottom", fontsize=7, bbox=box)
        if col == 1:
            cb = fig.colorbar(sca, ax=axes[0, :], location="right", fraction=0.025, pad=0.01)
            cb.set_label(f"Specific tonality (tu$_{{HMS}}$), components above {SHOW_THRESHOLD:g}")

        # (c, d) attribution to candidate orders
        ax = axes[1, col]
        cand, share, other = att[name]
        labels = [f"{o:g}" for o in cand] + ["none"]
        vals = [100 * share[o] for o in cand] + [100 * other]
        cols = [GREY if o == CONTROL_ORDER[sc.kind] else (RED if (name == "ev" and o == WHINE_ORDER) else BLUE)
                for o in cand] + ["#BBBBBB"]
        ax.bar(range(len(vals)), vals, color=cols, width=0.75)
        ax.set_xticks(range(len(vals)), labels)
        ax.set_xlabel("Candidate order (grey: no source, chance level)")
        ax.set_ylabel("Tonality-weighted share (%)")
        ax.set_title(f"({'cd'[col]}) Attribution with frequency_to_order, ±{TOLERANCE * 100:g} %", loc="left")

        # (e, f) tonality and loudness along rpm
        ax = axes[2, col]
        ton, lou = r["tonality"], r["loudness"]
        ax.fill_between(ton.centres, ton.value - ton.std, ton.value + ton.std, color=BLUE, alpha=0.15, lw=0)
        ax.plot(ton.centres, ton.value, "-o", color=BLUE, ms=3, lw=1.4, label="tonality, mean above 0.02")
        if name == "ev":
            nw = R["ev_nw"]["tonality"]
            ax.plot(nw.centres, nw.value, "--o", color="k", ms=2.5, lw=1.0, mfc="w", label="tonality, whine removed")
        ax.set_ylabel("Tonality (tu$_{HMS}$)", color=BLUE)
        ax.tick_params(axis="y", colors=BLUE)
        ax2 = ax.twinx()
        ax2.plot(lou.centres, lou.value, "-s", color=ORANGE, ms=3, lw=1.2, label="loudness, mean")
        ax2.set_ylabel("Loudness (sone$_{HMS}$)", color=ORANGE)
        ax2.tick_params(axis="y", colors=ORANGE)
        for a in (ax, ax2):  # zero at the bottom, headroom for the legend
            a.set_ylim(0.0, a.get_ylim()[1] * 1.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.5, frameon=True)
        ax.set_title(f"({'ef'[col]}) Tonality and loudness along rpm (metric_vs_rpm)", loc="left")
        ax.set_xlabel("Engine speed (rpm)" if name == "ice" else "Motor speed (rpm)")

        # (g, h) order-attributed tonality and roughness along rpm
        ax = axes[3, col]
        ot = r["order_tonality"]
        ax.plot(ot.centres, ot.value, "-o", color=RED if name == "ev" else BLUE, ms=3, lw=1.4,
                label=f"order-{order_of[name]:g} tonality")
        if name == "ev":
            onw = R["ev_nw"]["order_tonality"]
            ax.plot(onw.centres, onw.value, "--o", color="k", ms=2.5, lw=1.0, mfc="w",
                    label=f"order-{WHINE_ORDER:g} tonality, whine removed")
        ax.set_ylabel(f"Order-{order_of[name]:g} tonality (tu$_{{HMS}}$)")
        ax2 = ax.twinx()
        rou = r["roughness"]
        ax2.fill_between(rou.centres, rou.value - 0.5 * rou.iqr, rou.value + 0.5 * rou.iqr, color=GREEN, alpha=0.15, lw=0)
        ax2.plot(rou.centres, rou.value, "-^", color=GREEN, ms=3, lw=1.2, label=f"roughness, P{ROUGHNESS_PERCENTILE}")
        ax2.set_ylabel("Roughness (asper$_{HMS}$)", color=GREEN)
        ax2.tick_params(axis="y", colors=GREEN)
        for a in (ax, ax2):  # zero at the bottom, headroom for the legend
            a.set_ylim(0.0, a.get_ylim()[1] * 1.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.5, frameon=True)
        ax.set_title(f"({'gh'[col]}) Order contribution and roughness along rpm", loc="left")
        ax.set_xlabel("Engine speed (rpm)" if name == "ice" else "Motor speed (rpm)")

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH)
    plt.close(fig)
    print(f"\nFigure: {FIG_PATH.relative_to(ROOT)}    total run time {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
