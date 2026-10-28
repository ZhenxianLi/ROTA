"""rota-order and the Sottek hearing model: ECMA-418-2 metrics along rpm, attributed to orders.

rota-order computes no psychoacoustic metric (it is out of scope); it prepares the signal for
one, and it bins the time-varying result along rpm. This script links it to the Python
implementation of the Sottek hearing model of ECMA-418-2, ``sottek-hearing-model``
(import name ``sottek_hearing_model``, PyPI 0.1.14, repository
github.com/mlotinga/sottek-hearing-model), through ``rota_order.bridges.sottek_hearing_model``.
The package implements ECMA-418-2:2025 (4th edition) time-dependent tonality (with the
frequency of the dominant tonal component), loudness and roughness. Fluctuation strength
is not part of the release yet (the package README announces it for a future version), so
it is not used here.

Optional external dependency, not part of rota-order. The package is licensed AGPL-3.0-only
(package metadata) and needs Python 3.11 or later; it pulls in PyQt6, Matplotlib,
Bottleneck and tqdm. rota-order does not depend on it, vendor it or copy its code: the bridge
imports it only inside its functions (and undoes the change of Matplotlib's settings that
the import makes) and calls only the documented public functions ``shm_tonality_ecma``,
``shm_loudness_ecma_from_comp`` and ``shm_roughness_ecma``. Install it yourself if you
accept its licence: ``pip install sottek-hearing-model`` (or
``pip install "rota-order[sottek-hearing-model]"``). Without it, or if its import fails, this
script stops with that message (and the import error) when it first calls the bridge.

The signals come from ``rota_order.synth.synth_scenario``: ``"ice_runup"`` (a four-stroke
four-cylinder engine, firing order 2, no gear shift in the first 6 s) and ``"ev_runup"``
(motor orders 2, 4, 8, 10, 12, 14, gear-mesh order 6.7, gear whine at order 40, a dithered
10 kHz inverter carrier and two fixed tones, 8 s), both seeded, mono, 32 kHz.

Chain:

1. Speed from the tachometer channel (``rota_order.speed.tacho_to_rpm``, conditioning delay
   compensated). The signal goes to the bridge as sound pressure in Pa at 32 kHz
   (calibration factor 1, the default, because the synthetic signal is already in Pa); the
   bridge brings it to the model's rate of 48 kHz with ``rota_order.tonal.to_calibrated_48k``.
2. The model, through the bridge: overall time-dependent tonality (187.5 Hz frame rate)
   with the frequency of its dominant tonal component, and the specific (per half critical
   band) tonality with its tonal frequencies, both in the result of ``tonality``; loudness
   (``loudness`` given that result, which derives it from the tonal and noise loudness of
   the tonality run) and roughness (50 Hz frame rate). Free-frontal sound field, the
   package default.
3. Binning along rpm with the ``vs_rpm`` method of each result, which calls
   ``rota_order.slices.metric_vs_rpm`` with what the bridge recommends for the metric. Tonality:
   the mean of the frames above 0.02 tu_HMS (the threshold of the result; frames below are
   discarded), the rule ECMA-418-2 uses for its single value. Loudness: arithmetic mean of
   the time-dependent loudness (the package's ``loudness_powavg``, the ``single_value`` of
   the result, is a power average and is only printed for comparison). Roughness: 90th
   percentile, the standard's single value. The start-up block of each metric is skipped:
   0.30 s (57 frames) for tonality and loudness, 0.31 s (the first 16 of the 50 Hz frames)
   for roughness, the exclusions that the package applies to its own single values. The
   model puts its last tonality and loudness frame at the signal length rounded up to a
   multiple of 256 samples at 48 kHz; for these records (6 s and 8 s) that is exactly the
   duration, one sample after the last speed sample, and ``vs_rpm`` holds the last speed
   for it. The script checks that ``vs_rpm`` with
   one bin spanning the whole run reproduces the package's ``tonality_avg`` and
   ``roughness90pc``, so the per-bin values are those single values restricted to a speed
   bin.
4. Order attribution of the model's tonal frequencies: ``rota_order.slices.frequency_to_order``
   (frequency / (rpm / 60), nearest candidate order, 10 % tolerance) and
   ``rota_order.slices.order_share`` (share on a base order and its harmonics, optionally
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
5. Order contribution: the whine (order 40) is removed with ``rota_order.extract.remove_orders``
   (tracked harmonics, 4 revolutions, as in example 06) and the model runs again.
   The overall tonality along rpm is compared with and without it, and so is the
   order-attributed tonality: the largest specific tonality of a half band whose tonal
   frequency lies on the order (0 where there is none).

Units and conventions. The model's outputs keep the standard's units: tonality in tu_HMS,
loudness in sone_HMS, roughness in asper_HMS; ``vs_rpm`` bins them as given. The
signal is sound pressure in Pa (the synthetic source orders are peak amplitudes, noise is
RMS; the model works on the pressure waveform, so no amplitude convention enters). Speed
in rpm, frequency in Hz, time in s.

The truth decides what the numbers mean. The scenario knows its orders, so the share of
the model's tonal frequencies on the true orders, the deviation of those frequencies from
``order * rpm / 60`` and the effect of removing the whine can be judged.

Run:  python examples/08_sottek_hearing_model_bridge.py
Needs matplotlib and sottek-hearing-model. About 2 minutes on four cores (the model dominates;
by default it uses all cores but one, ``PARALLEL_CORES = 1`` is about three times slower).
Writes docs/images/08_sottek_hearing_model_bridge.png.

Toolboxes used (cite them as their authors ask; rota-order does not include or redistribute
their code, which keeps its own licence):

* Lotinga, M. J. B., Torjussen, M., & Felix Greco, G. (2025). Verified implementations of
  the Sottek psychoacoustic Hearing Model standardised sound quality metrics (ECMA-418-2
  loudness, tonality and roughness). Proceedings of Forum Acusticum / Euronoise 2025,
  Malaga, Spain. https://doi.org/10.61782/fa.2025.0951. Software: sottek-hearing-model 0.1.14,
  https://github.com/mlotinga/sottek-hearing-model (AGPL-3.0-only).

"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import rota_order as ro
from rota_order import bridges

NEEDS_MODEL = (  # the message the script stops with if the bridge cannot import the package
    "This example needs the optional package sottek-hearing-model (ECMA-418-2 Sottek hearing\n"
    "model, AGPL-3.0-only, Python >= 3.11): pip install sottek-hearing-model"
)

FS = 32000.0  # sampling rate of the scenarios, Hz
SEED = 3
ICE_DURATION = 6.0  # s; the first upshift of the default scenario is later than this
EV_DURATION = 8.0  # s
EV_OVERRIDES = {"tacho": {"ppr": 24}}  # 48 pulses would be shorter than 2 samples at 32 kHz above 10000 rpm
WHINE_ORDER = 40.0
WHINE_WINDOW_REVS = 4.0  # tracked-harmonics frame length for the whine, revolutions
PARALLEL_CORES = None  # None: all cores but one; 1: no parallel processing

SHOW_THRESHOLD = 0.15  # tu_HMS; weaker components are left out of the scatter plots only
TOLERANCE = 0.10  # relative tolerance of the order attribution (see the module docstring)
BIN_ICE, BIN_EV = 300.0, 1000.0  # rpm bin widths
MIN_COUNT = 5  # samples a bin needs
CONTROL_ORDER = {"ice_runup": 11.0, "ev_runup": 34.0}  # no source at these orders (chance level)

ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = ROOT / "docs" / "images" / "08_sottek_hearing_model_bridge.png"
BLUE, ORANGE, GREEN, GREY, RED = "#0072B2", "#D55E00", "#009E73", "#555555", "#CC3311"


# ---- model runs ----------------------------------------------------------------------------------


def run_model(x, fs, roughness=True):
    """Tonality, loudness and (optionally) roughness of a mono signal in Pa, through the bridge.

    Free-frontal sound field (the default); the loudness is derived from the tonality run.
    """
    shm = bridges.sottek_hearing_model
    ton = shm.tonality(x, fs, parallel_cores=PARALLEL_CORES)
    lou = shm.loudness(None, None, tonality=ton)
    rou = shm.roughness(x, fs, parallel_cores=PARALLEL_CORES) if roughness else None
    return {"ton": ton, "lou": lou, "rou": rou}


def prepare(sc, ppr, removed_orders=None):
    """Tacho speed, and the recording in Pa at its own rate (without ``removed_orders`` if given)."""
    delay = float(sc.config["resolved"]["tacho_delay"])
    rpm = ro.speed.tacho_to_rpm(sc.tacho, sc.fs, ppr=ppr, t_out=sc.t + delay)
    x = sc.x  # Pa: the bridge's default calibration factor 1 applies
    if removed_orders is not None:
        x, _ = ro.extract.remove_orders(
            x, sc.fs, rpm, removed_orders, method="tracked-harmonics", window_revs=WHINE_WINDOW_REVS
        )
    return rpm, x


# ---- binning along rpm -----------------------------------------------------------------------------


def one_value(metric, rpm, t_rpm):
    """``vs_rpm`` with a single bin over the whole run: the single value of the metric."""
    res = metric.vs_rpm(rpm, t_rpm, segment="all", bins=[0.0, 1.0e6], min_count=1)
    return float(res.value[0])


def order_tonality(ton, rpm_m, order):
    """Largest specific tonality among the half bands whose tonal frequency lies on ``order`` (else 0).

    ``ton`` is a tonality result of the bridge, ``rpm_m`` the speed on its frames. The result is a
    ``TimeMetric`` on the same frames with the start-up exclusion of the tonality and no threshold
    (the zeros count), so that ``vs_rpm`` bins it like the tonality.
    """
    spec = ton.extra["spec_tonality"]
    f = np.where(spec > ton.threshold, ton.extra["spec_tonality_freqs"], np.nan)
    fo = ro.slices.frequency_to_order(f, rpm_m[:, None], tolerance=TOLERANCE, orders=[order])
    return bridges.TimeMetric(
        name=f"order-{order:g} tonality", t=ton.t, values=np.max(np.where(fo.matched, spec, 0.0), axis=1),
        unit=ton.unit, skip_start=ton.skip_start, source=ton.source,
    )


# ---- order attribution ---------------------------------------------------------------------------------


def band_attribution(ton, rpm_m, candidates):
    """Share of the tonality of all half-band tonal components on each candidate order.

    Components: specific tonality above the threshold of the tonality result after its start-up
    frames. Weight: the specific tonality. Returns {order: share}, the share on no candidate
    ("other") and the median relative deviation of the matched frequencies from the order lines.
    """
    keep = ton.t >= ton.skip_start
    s, f, r = ton.extra["spec_tonality"][keep], ton.extra["spec_tonality_freqs"][keep], rpm_m[keep]
    on = s > ton.threshold
    fo = ro.slices.frequency_to_order(np.where(on, f, np.nan), r[:, None], tolerance=TOLERANCE, orders=candidates)
    total = float(s[on & np.isfinite(fo.order)].sum())
    share = {o: float(s[on & fo.matched & (fo.nearest == o)].sum() / total) for o in candidates}
    other = 1.0 - sum(share.values())
    dev = np.abs(fo.deviation[on & fo.matched])
    return share, other, float(np.median(dev))


def dominant_attribution(ton, rpm_m, base, harmonics):
    """Share of the frames' dominant tonal frequency (overall tonality) on ``base`` and its harmonics."""
    keep = (ton.t >= ton.skip_start) & (ton.values > ton.threshold)
    f = np.where(keep, ton.extra["tonality_freqs"], np.nan)
    return ro.slices.order_share(f, rpm_m, base, harmonics=harmonics, tolerance=TOLERANCE), float(np.nanmedian(f))


def main():
    plt.switch_backend("Agg")  # file output only; no display needed
    t_start = time.time()

    # ---- signals ----------------------------------------------------------------------------------
    ice = ro.synth.synth_scenario("ice_runup", fs=FS, duration=ICE_DURATION, seed=SEED)
    ev = ro.synth.synth_scenario("ev_runup", fs=FS, duration=EV_DURATION, seed=SEED, **EV_OVERRIDES)
    ice_ppr, ev_ppr = int(ice.config["parameters"]["tacho"]["ppr"]), int(ev.config["parameters"]["tacho"]["ppr"])
    firing = float(ice.config["parameters"]["combustion"]["per_rev"])
    assert ice.events["shift_times"].size == 0, "the engine record must stay in one gear"
    rpm_ice, x_ice = prepare(ice, ice_ppr)
    rpm_ev, x_ev = prepare(ev, ev_ppr)
    rpm_ev_nw, x_ev_nw = prepare(ev, ev_ppr, removed_orders=[WHINE_ORDER])
    print(f"Signals (synth_scenario, seed {SEED}, mono, {FS:g} Hz -> to_calibrated_48k -> 48000 Hz, Pa):")
    print(f"  ice_runup {ICE_DURATION:g} s, {rpm_ice[50]:.0f} -> {rpm_ice[-50]:.0f} rpm, firing order {firing:g}, "
          f"RMS {20 * np.log10(np.std(x_ice) / 20e-6):.1f} dB re 20 uPa")
    print(f"  ev_runup  {EV_DURATION:g} s, {np.nanmin(rpm_ev[50:]):.0f} -> {np.nanmax(rpm_ev):.0f} rpm, "
          f"orders {ev.orders.tolist()}, RMS {20 * np.log10(np.std(x_ev) / 20e-6):.1f} dB re 20 uPa")

    # ---- model ------------------------------------------------------------------------------------
    print("Sottek hearing model (ECMA-418-2:2025), free-frontal field:")
    R = {}
    for name, sc, rpm, x in (("ice", ice, rpm_ice, x_ice), ("ev", ev, rpm_ev, x_ev), ("ev_nw", ev, rpm_ev_nw, x_ev_nw)):
        t0 = time.time()
        try:  # the bridge imports the package at its first call
            r = run_model(x, sc.fs, roughness=name != "ev_nw")
        except ImportError as err:  # not installed, or its import fails; the bridge chains the original error
            sys.exit(f"{NEEDS_MODEL}\nImport failed: {err.__cause__ or err}")
        print(f"  {name:6s} tonality, loudness{'' if name == 'ev_nw' else ', roughness'}: {time.time() - t0:5.1f} s")
        # speed for vs_rpm, and on the model's 187.5 Hz frame grid for the order attribution
        r.update(rpm=rpm, t_rpm=sc.t, rpm_m=np.interp(r["ton"].t, sc.t, rpm))
        R[name] = r

    # ---- single values: does metric_vs_rpm with one bin reproduce the package's? -----------------------
    print("Single values: metric_vs_rpm with one bin over the whole run against the package's own:")
    for name in ("ice", "ev"):
        r = R[name]
        ton, lou, rou = r["ton"], r["lou"], r["rou"]
        t_avg, r90, l_mean = (one_value(m, r["rpm"], r["t_rpm"]) for m in (ton, rou, lou))
        print(f"  {name:3s} tonality   {t_avg:.4f} tu_HMS     package tonality_avg {ton.single_value:.4f}")
        print(f"      roughness  {r90:.4f} asper_HMS  package roughness90pc {rou.single_value:.4f}")
        print(f"      loudness   {l_mean:.3f} sone_HMS (mean)   package loudness_powavg "
              f"{lou.single_value:.3f} (power average)")
        assert np.isclose(t_avg, ton.single_value, rtol=1e-6), "tonality single value differs"
        assert np.isclose(r90, rou.single_value, rtol=1e-6), "roughness single value differs"

    # ---- metrics along rpm ----------------------------------------------------------------------------------
    order_of = {"ice": firing, "ev": WHINE_ORDER, "ev_nw": WHINE_ORDER}
    for name, width in (("ice", BIN_ICE), ("ev", BIN_EV), ("ev_nw", BIN_EV)):
        r = R[name]
        kw = {"bin_width": width, "min_count": MIN_COUNT}
        r["tonality"] = r["ton"].vs_rpm(r["rpm"], r["t_rpm"], **kw)
        r["loudness"] = r["lou"].vs_rpm(r["rpm"], r["t_rpm"], **kw)
        if r["rou"] is not None:
            r["roughness"] = r["rou"].vs_rpm(r["rpm"], r["t_rpm"], **kw)
        r["order_tonality"] = order_tonality(r["ton"], r["rpm_m"], order_of[name]).vs_rpm(r["rpm"], r["t_rpm"], **kw)

    def table(label, r, columns):
        """Print one row per rpm bin of run ``r``; ``columns`` is a list of (header, width, MetricVsRpm)."""
        centres = r["tonality"].centres
        print(f"{label}: bins of {centres[1] - centres[0]:g} rpm (tonality: mean of frames above "
              f"{r['ton'].threshold:g} tu_HMS; loudness: mean; roughness: P{r['roughness'].q:g}; "
              f"skip {r['ton'].skip_start:g} s / {r['rou'].skip_start:g} s)")
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
    table("Engine run-up", ri,
          [("tonality", 11, ri["tonality"]), ("loudness", 11, ri["loudness"]), ("roughness", 11, ri["roughness"]),
           (f"order {firing:g} tonal.", 17, ri["order_tonality"])])
    print()
    table("Electric drive", re_,
          [("tonality", 11, re_["tonality"]), ("loudness", 11, re_["loudness"]), ("roughness", 11, re_["roughness"]),
           (f"order {WHINE_ORDER:g} tonal.", 17, re_["order_tonality"]),
           ("tonality, no whine", 20, rn["tonality"]), (f"order {WHINE_ORDER:g}, no whine", 20, rn["order_tonality"])])

    # ---- order attribution -------------------------------------------------------------------------------------
    print("\nOrder attribution of the tonal frequencies (frequency / (rpm / 60), tolerance "
          f"{TOLERANCE * 100:g} %, frames after {ri['ton'].skip_start:g} s):")
    att = {}
    for name, sc, cand in (("ice", ice, ice.orders), ("ev", ev, ev.orders)):
        r = R[name]
        cand = np.unique(np.append(cand, CONTROL_ORDER[sc.kind]))
        share, other, dev = band_attribution(r["ton"], r["rpm_m"], cand)
        att[name] = (cand, share, other)
        print(f"  {sc.kind}, half-band components above threshold, weighted by specific tonality "
              f"(median |deviation| of matched frequencies {100 * dev:.1f} %):")
        line = ", ".join(f"{o:g}: {100 * s:.1f} %" + (" (control)" if o == CONTROL_ORDER[sc.kind] else "")
                         for o, s in share.items())
        print(f"    {line}, on no candidate: {100 * other:.1f} %")
    print()
    ton_ice, rpm_m = ri["ton"], ri["rpm_m"]
    tonal = (ton_ice.t >= ton_ice.skip_start) & (ton_ice.values > ton_ice.threshold)
    dev = ton_ice.extra["tonality_freqs"] / (firing * rpm_m / 60.0) - 1.0
    lo, hi = tonal & (rpm_m < 1500.0), tonal & (rpm_m >= 2100.0)
    print(f"  ice_runup, dominant tonal frequency against the order-{firing:g} line (truth, median deviation): "
          f"{100 * np.median(dev[lo]):+.1f} % below 1500 rpm (under 50 Hz), {100 * np.median(dev[hi]):+.1f} % above "
          f"2100 rpm (over 70 Hz); the lowest bands of the model resolve frequency coarsely, hence the wide window")
    sh_ice, med_ice = dominant_attribution(ri["ton"], ri["rpm_m"], firing, 8)
    print(f"  ice_runup, dominant tonal frequency of the overall tonality ({sh_ice.n_valid} tonal frames, median "
          f"{med_ice:.0f} Hz): on order {firing:g} and harmonics (up to order {firing * 8:g}) "
          f"{100 * sh_ice.share:.1f} %; per harmonic "
          + ", ".join(f"{k * firing:g}: {100 * s:.1f} %" for k, s in zip(sh_ice.harmonics, sh_ice.share_per_harmonic)))
    sh_ev, med_ev = dominant_attribution(re_["ton"], re_["rpm_m"], 2.0, 7)
    sh_w, _ = dominant_attribution(re_["ton"], re_["rpm_m"], WHINE_ORDER, 1)
    print(f"  ev_runup, dominant tonal frequency of the overall tonality ({sh_ev.n_valid} tonal frames, median "
          f"{med_ev:.0f} Hz, the low-frequency rumble band): motor family 2k (k = 1..7; order 6 has no source) "
          f"{100 * sh_ev.share:.1f} %, whine (order {WHINE_ORDER:g}) {100 * sh_w.share:.1f} %")
    for lbl, key in (("with the whine", "ev"), ("whine removed", "ev_nw")):
        ton, rpm_m = R[key]["ton"], R[key]["rpm_m"]
        spec, keep = ton.extra["spec_tonality"], ton.t >= ton.skip_start
        sw = ro.slices.order_share(
            np.where(spec > ton.threshold, ton.extra["spec_tonality_freqs"], np.nan)[keep].ravel(),
            np.repeat(rpm_m[keep][:, None], spec.shape[1], axis=1).ravel(),
            WHINE_ORDER, harmonics=1, tolerance=TOLERANCE, weights=spec[keep].ravel())
        print(f"  ev_runup {lbl:15s}: tonality-weighted share of the half-band components on the whine: "
              f"{100 * sw.share:.1f} %")

    # ---- order contribution --------------------------------------------------------------------------------------
    t_with, t_without = re_["ton"].single_value, rn["ton"].single_value
    print(f"\nWhine contribution (ev_runup, order {WHINE_ORDER:g} removed with remove_orders, tracked harmonics, "
          f"{WHINE_WINDOW_REVS:g} revolutions):")
    print(f"  overall tonality, single value: {t_with:.3f} -> {t_without:.3f} tu_HMS "
          f"({100 * (t_without / t_with - 1):+.1f} %); loudness (power average) "
          f"{re_['lou'].single_value:.2f} -> {rn['lou'].single_value:.2f} sone_HMS")
    a_with = np.nanmean(re_["order_tonality"].value[re_["order_tonality"].valid])
    a_without = np.nanmean(rn["order_tonality"].value[rn["order_tonality"].valid])
    print(f"  order-{WHINE_ORDER:g} tonality, mean over the speed bins: {a_with:.3f} -> {a_without:.3f} tu_HMS")

    # ---- figure ---------------------------------------------------------------------------------------------------------
    plt.rcdefaults()  # Matplotlib's default style, whatever a local matplotlibrc sets
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8})
    fig, axes = plt.subplots(4, 2, figsize=(8.0, 10.6), dpi=150, constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.15, 0.8, 0.9, 0.9]})
    titles = {"ice": "Engine run-up (ice_runup)", "ev": "Electric drive (ev_runup)"}
    for col, name in enumerate(("ice", "ev")):
        sc = ice if name == "ice" else ev
        r = R[name]
        tm = r["ton"]
        spec, freqs, t_m = tm.extra["spec_tonality"], tm.extra["spec_tonality_freqs"], tm.t
        keep = (spec > SHOW_THRESHOLD) & (t_m[:, None] >= tm.skip_start) & (freqs > 0)
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
        ax.plot(ton.centres, ton.value, "-o", color=BLUE, ms=3, lw=1.4, label=f"tonality, mean above {tm.threshold:g}")
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
        ax2.plot(rou.centres, rou.value, "-^", color=GREEN, ms=3, lw=1.2, label=f"roughness, P{rou.q:g}")
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
