"""ROTA and MoSQITo: psychoacoustic metrics of a run-up along rpm, and what an order contributes.

ROTA does not compute loudness, sharpness or roughness (they are out of scope); the open-source
toolbox MoSQITo (Apache-2.0, ``pip install mosqito``, https://github.com/Eomys/MoSQITo) does. This
script shows the hand-over: ROTA prepares the signal and the speed, MoSQITo computes the metrics, ROTA
puts the metrics back on a speed axis and attributes them to an order. MoSQITo stays an optional,
external dependency: nothing in ``src/rota`` imports it, and only its documented public functions
are called here.

The recording is ``rota.synth.synth_scenario("ice_runup")`` (a synthetic four-cylinder engine in a
vehicle, firing order 2), the first 6 s of the default full-load run-up (about 1200 to 4400 rpm in
second gear; the first upshift comes later), one receiver channel, seed fixed. Nothing is measured.

MoSQITo functions used (version 1.2.1, ``mosqito.sq_metrics``) and the standards behind them, as
catalogued at https://zhenxianli.github.io/LISQM/ (projects/mosqito.html):

* ``loudness_zwtv``: ISO 532-1:2017 time-varying loudness (Zwicker), in sone, one value per 2 ms.
* ``sharpness_din_from_loudness``: DIN 45692 sharpness (``weighting="din"``), in acum, from the
  specific loudness of ``loudness_zwtv`` (the result is identical to ``sharpness_din_tv``, which
  would compute the loudness a second time).
* ``roughness_dw``: Daniel & Weber roughness, in asper, one value per 0.1 s (200 ms windows, 50 %
  overlap).
* ``tnr_ecma_st``: tone-to-noise ratio of the ECMA-74 Annex D method (the catalogue gives
  ECMA-74:2019, 17th edition; the docstring says "ECMA 418-1" without an edition).
* Not used here: ``loudness_ecma`` and ``roughness_ecma`` implement ECMA-418-2 in its 2nd edition
  (2022) only, which is not the current edition (the 4th); its Clause 8 loudness is missing. A
  result must not be quoted as "ECMA-418-2" without that qualification.

The script follows the steps below.

1. Preparation with ROTA. ``rota.tonal.to_calibrated_48k`` brings the signal to Pa at 48 kHz (the
   rate MoSQITo's time-varying loudness requires; the polyphase filter keeps the level, which is
   checked) and ``tacho_to_rpm`` gives the speed from the tachometer channel.
2. MoSQITo computes loudness, sharpness and roughness of the run-up (``field_type="free"``).
3. ``rota.slices.metric_vs_rpm`` bins each metric along the speed in 250 rpm bins on the monotonic
   run-up, after skipping the start-up of the metric: 0.6 s for loudness and sharpness (which comes
   from the same filters) and 1.5 s for roughness, the rules of thumb in the docstring of
   ``metric_vs_rpm``. The aggregation follows the usage for time-varying sounds: the 95th percentile
   of loudness (N5, the loudness exceeded in 5 % of the time; Fastl and Zwicker), the median of the
   sharpness and the 90th percentile of the roughness (R10). A metric is never averaged in dB.
4. Order contribution. Step 2 and 3 are repeated on the signal with order 2 removed
   (``rota.extract.remove_orders``, tracked harmonics) and with order 2 raised by 6 dB
   (``rota.resynth.modify_orders`` on the extracted components, then ``rota.resynth.resynthesize``
   with the unchanged residual), and the differences to the original are plotted against the speed.
   Order 2 is checked with an order slice (RMS, Pa) before MoSQITo sees the signals.
5. Tone prominence of order 2 against the speed: ROTA's ``order_prominence`` works in the angle
   domain on the run-up; MoSQITo's ``tnr_ecma_st`` is run on steady excerpts at three speeds, because
   it takes a signal of fixed duration as one spectrum and has no speed input. The two differ in
   method, not only in implementation:

   * ROTA resamples to uniform shaft angle (computed order tracking) and transforms frames of 50
     revolutions (the 1 % line spacing that ECMA-418-1 requires for the tone frequency), so order 2
     is one stationary line in every frame although the speed rises by about 500 rpm/s. It evaluates
     the critical band of each frame at the frame's mean speed (ECMA-418-1, clauses 10 and 11) and
     averages frames in speed bins. A fixed time window with the same line spacing at 100 Hz lasts
     1 s, and order 2 then sweeps 17 to 18 Hz, which is more than the 15 % of the critical band
     below which the standard regards a tone as resolved: ROTA flags such a window ("resolved: no"
     in the table; its TNR comes out close here only because ``tone_prominence`` gathers the swept
     lines into the tone, a choice that the standard leaves to the analyst).
   * MoSQITo takes the whole signal as one spectrum and finds the tones itself. It therefore needs a
     steady signal: ``rota.resynth.stationary_excerpt`` renders the orders as steady sinusoids with
     the amplitudes found at the reference speed, plus the broadband residual of the recording (2 s,
     five seeds, so that the scatter of the noise shows). ROTA's ``tone_prominence`` is run on the
     same excerpts.
   * Editions: ROTA follows ECMA-418-1 (3rd edition, 2024); MoSQITo is catalogued against
     ECMA-74:2019 (17th edition, Annex D), whose method is that of the 2nd edition of ECMA-418-1.
     Both define the TNR as the tone power over the noise power of the critical band, rescaled to
     the critical bandwidth, but the 3rd edition changed how the tone band is delimited and how the
     masking noise is estimated, so the two need not agree on noisy or sloping spectra. On a steady
     tone in noise with an exact flat spectrum they agree within about 0.5 dB from 130 Hz to 4 kHz
     (``tests/test_bridge_mosqito.py``). ROTA is given the tone frequency; MoSQITo detects the peaks
     itself.

   On the excerpts of this run-up, which are noisy and have a sloping spectrum, the two do not agree
   (see the table): ROTA reads 10 to 20 dB, as the excerpts are built, while MoSQITo 1.2.1 returns
   0.1 to 1 dB in most of them and about 11 dB in the rest, with the same tone and a different noise
   seed. Pure sinusoids without any noise give about 0.2 dB, as if the lines next to the peak were
   counted as noise. A tone of 12 dB in white Gaussian noise between about 130 and 380 Hz gives
   either the right value or 2 to 3 dB, depending on the noise realisation. We have not diagnosed
   it, and the table prints every MoSQITo value so that nothing is averaged away. Order 2 stays
   below 150 Hz in this run-up, in the range where this occurs. The difference of editions
   may explain part of it; we have not investigated MoSQITo's implementation.

Amplitude conventions. Signals are sound pressure in Pa; levels are in dB re 20 uPa. The RMS
convention is used for order slices and levels; the extracted order envelopes are peak values, but only
gains (ratios) are applied to them. MoSQITo takes the signal in Pa and returns sone, acum and asper.

Run:  python examples/07_mosqito_bridge.py     (needs matplotlib and mosqito; about 1 minute on 4 cores, 2 on one)
Writes docs/images/07_mosqito_bridge.png.
"""

import contextlib
import io
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import rota

try:  # MoSQITo is an optional, external dependency of this example only
    import mosqito
    from mosqito.sq_metrics import tnr_ecma_st
except ImportError as err:  # pragma: no cover
    raise SystemExit("This example needs MoSQITo: pip install mosqito") from err

P_REF = 20e-6  # reference pressure in Pa (dB re 20 uPa), passed explicitly
FS = 32000.0  # sampling rate of the scenario, Hz
FS_MQ = 48000.0  # sampling rate MoSQITo needs, Hz
DURATION = 6.0  # s
SEED = 3
ORDER = 2.0  # firing order of the four-cylinder engine
BOOST_DB = 6.0  # gain applied to order 2 in the "raised" variant
WINDOW_REVS = 8.0  # tracked-harmonics frame length, revolutions
EDGES = np.arange(1500.0, 4501.0, 250.0)  # speed bins, rpm
SKIP_S = {"loudness": 0.6, "sharpness": 0.6, "roughness": 1.5}  # start-up of each metric, s
AGGREGATE = {"loudness": ("percentile", 95), "sharpness": "median", "roughness": ("percentile", 90)}
EXC_RPM = (3000.0, 3500.0, 4000.0)  # reference speeds of the steady excerpts, rpm
EXC_DURATION = 2.0  # s
N_EXC = 5  # excerpts (seeds) per reference speed
WINDOW_S = 1.0  # fixed time window cut from the run-up for the comparison, s
AGREE_DB = 1.5  # agreement band between the two TNR values, dB
ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = ROOT / "docs" / "images" / "07_mosqito_bridge.png"
VARIANTS = ("original", "order 2 removed", f"order 2 {BOOST_DB:+g} dB")
COLOURS = {VARIANTS[0]: "#0072B2", VARIANTS[1]: "#D55E00", VARIANTS[2]: "#009E73"}


def mosqito_metrics(x48):
    """Loudness, sharpness and roughness of a signal in Pa at 48 kHz, with MoSQITo.

    Module-level so that it can run in a worker process. MoSQITo prints a warning about the first
    points of sharpness computed from a loudness track; those points are skipped by ``skip_start``
    later, so the output is silenced here.
    """
    from mosqito.sq_metrics import loudness_zwtv, roughness_dw, sharpness_din_from_loudness

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_sone, n_spec, _bark, t_n = loudness_zwtv(x48, FS_MQ, field_type="free")  # ISO 532-1:2017
        s_acum = sharpness_din_from_loudness(n_sone, n_spec, weighting="din")  # DIN 45692
        r_asper, _r_spec, _bark_r, t_r = roughness_dw(x48, FS_MQ, overlap=0.5)  # Daniel & Weber
    return {"t_n": np.asarray(t_n), "loudness": np.asarray(n_sone), "sharpness": np.asarray(s_acum),
            "t_r": np.asarray(t_r), "roughness": np.asarray(r_asper)}


def bin_metrics(m, t, rpm):
    """Bin the MoSQITo tracks along the speed with ``metric_vs_rpm`` (identical bins for every call)."""
    out = {}
    for name in ("loudness", "sharpness", "roughness"):
        t_m = m["t_r"] if name == "roughness" else m["t_n"]
        out[name] = rota.slices.metric_vs_rpm(
            m[name], t_m, rpm, t, bins=EDGES, skip_start=SKIP_S[name], aggregate=AGGREGATE[name],
            min_count=3,
        )
    return out


def mosqito_tnr(x, fs, f0, tol_hz=2.0):
    """TNR in dB of the tone MoSQITo finds nearest to ``f0`` (NaN if it finds none within ``tol_hz``)."""
    x48 = rota.tonal.to_calibrated_48k(x, fs)
    with contextlib.redirect_stdout(io.StringIO()):
        _t_tnr, tnr, _prominent, freqs = tnr_ecma_st(x48, FS_MQ, prominence=False)
    tnr, freqs = np.asarray(tnr, dtype=float).ravel(), np.asarray(freqs, dtype=float).ravel()
    if freqs.size == 0:
        return np.nan
    k = int(np.argmin(np.abs(freqs - f0)))
    return float(tnr[k]) if abs(freqs[k] - f0) <= tol_hz else np.nan


def rota_tone(x, fs, f0, search_hz=None):
    """ROTA's ``tone_prominence`` of the tone at ``f0`` on a Hann periodogram (PSD) of ``x``."""
    f, psd = signal.welch(x, fs, window="hann", nperseg=min(x.size, int(fs)), scaling="density")
    return rota.tonal.tone_prominence(f, psd, f0, search_hz=search_hz)


def order_level_change(variant, reference, fs, rpm, order, t_min):
    """Median change of the order's RMS level (dB) from ``reference`` to ``variant`` over the run-up."""
    sl = [rota.slices.order_slice(
        rota.maps.rpm_order_map(v, fs, rpm, order_resolution=0.125, max_order=6.0, amplitude="rms"), order)
        for v in (variant, reference)]
    keep = sl[0].t > t_min
    return float(np.median(20.0 * np.log10(sl[0].amplitude[keep] / sl[1].amplitude[keep])))


def main():
    plt.switch_backend("Agg")  # file output only; no display needed
    t0 = time.time()
    print(f"MoSQITo {mosqito.__version__}, ROTA {rota.__version__}")

    # ---- 1. the signal, prepared with ROTA ----------------------------------------------------
    sc = rota.synth.synth_scenario("ice_runup", fs=FS, duration=DURATION, seed=SEED)
    ppr = int(sc.config["parameters"]["tacho"]["ppr"])
    delay = float(sc.config["resolved"]["tacho_delay"])
    rpm = rota.speed.tacho_to_rpm(sc.tacho, FS, ppr=ppr, t_out=sc.t + delay)  # delay compensated
    err = np.abs(rpm / sc.rpm_mean - 1.0)
    x = sc.x  # Pa, one channel
    print(f"Scenario ice_runup: {DURATION:g} s at {FS:g} Hz, seed {SEED}, {rpm[0]:.0f} -> {rpm[-1]:.0f} rpm "
          f"(tacho, median error {100 * np.median(err):.3f} %), firing order {ORDER:g}")

    # ---- 4a. the variants (built first, so that MoSQITo can start on all of them at once) ------------
    y, comp = rota.extract.remove_orders(x, FS, rpm, [ORDER], method="tracked-harmonics", window_revs=WINDOW_REVS)
    raised = rota.resynth.resynthesize(rota.resynth.modify_orders(comp, gain_db={ORDER: BOOST_DB}), y)["mix"]
    signals = dict(zip(VARIANTS, (x, y, raised)))
    # mix = 2 * order + (x - order) = x + order in amplitude: +6 dB means a factor 10**(6/20) = 1.995
    gain = 10.0 ** (BOOST_DB / 20.0)
    assert np.allclose(raised, x + (gain - 1.0) * comp.total, atol=1e-9)
    x48 = {k: rota.tonal.to_calibrated_48k(v, FS) for k, v in signals.items()}  # Pa at 48 kHz
    rms_in, rms_out = np.sqrt(np.mean(x ** 2)), np.sqrt(np.mean(x48[VARIANTS[0]] ** 2))
    print(f"to_calibrated_48k: {FS:g} -> {FS_MQ:g} Hz, RMS {20 * np.log10(rms_in / P_REF):.2f} -> "
          f"{20 * np.log10(rms_out / P_REF):.2f} dB re 20 uPa")

    workers = max(1, min(len(VARIANTS), os.cpu_count() or 1))
    pool = ProcessPoolExecutor(max_workers=workers)
    futures = {k: pool.submit(mosqito_metrics, v) for k, v in x48.items()}  # one process per variant

    # ---- 4b. what the variants are, in ROTA's terms (while MoSQITo works) ------------------------
    print("Order 2 and the overall level of the three signals (ROTA, RMS, run-up after 1 s):")
    lvl = {}
    for k, v in signals.items():
        ol = rota.slices.overall_level(v, FS, rpm, frame=0.1, weighting="A", ref=P_REF)
        sel = ol.t > 1.0
        lvl[k] = float(10.0 * np.log10(np.mean(10.0 ** (ol.level[sel] / 10.0))))
        d_ord = "" if k == VARIANTS[0] else (
            f"   order {ORDER:g} {order_level_change(v, x, FS, rpm, ORDER, 1.0):+6.1f} dB (median slice)")
        print(f"  {k:18s} L_A {lvl[k]:5.1f} dB ({lvl[k] - lvl[VARIANTS[0]]:+5.1f}){d_ord}")

    # ---- 5. tone prominence of order 2 ---------------------------------------------------------------
    op = rota.tonal.order_prominence(x, FS, rpm, [ORDER], rpm_bins=250.0)
    print(f"order_prominence (angle domain): frames of {op.frame_revs:g} revolutions "
          f"({op.order_resolution:g} order lines), {op.rpm.size} speed bins, ECMA-418-1 TNR criterion defined above "
          f"89.1 Hz")
    rows = []
    for r_ref in EXC_RPM:
        f0 = ORDER * r_ref / 60.0
        tnr_angle = float(np.interp(r_ref, op.rpm, op.tnr[:, 0]))
        i = int(np.argmin(np.abs(rpm - r_ref)))  # fixed 1 s time window of the run-up around r_ref
        h = int(WINDOW_S * FS / 2)
        w = x[i - h:i + h]
        sweep_hz = ORDER * (rpm[i + h] - rpm[i - h]) / 60.0
        win = rota_tone(w, FS, f0, search_hz=0.5 * sweep_hz + 2.0)
        ex_rota, ex_mq = [], []
        for seed in range(N_EXC):
            ex = rota.resynth.stationary_excerpt(x, FS, rpm, r_ref, EXC_DURATION, orders=sc.orders, seed=seed)
            ex_rota.append(rota_tone(ex.x, FS, f0).tnr)
            ex_mq.append(mosqito_tnr(ex.x, FS, f0))
        rows.append({"rpm": r_ref, "f0": f0, "angle": tnr_angle, "window": float(win.tnr), "sweep": sweep_hz,
                     "unresolved": bool(win.tone_band_exceeded), "cb": float(win.critical_bandwidth),
                     "rota": np.array(ex_rota), "mq": np.array(ex_mq)})
    print(f"TNR of order {ORDER:g} in dB (ROTA: tone_prominence / order_prominence; MoSQITo: tnr_ecma_st):")
    print(f"  {'rpm':>5s} {'Hz':>4s} | {'angle domain':>12s} | {f'{WINDOW_S:g} s window':>11s} {'sweep':>6s} {'of band':>7s} "
          f"{'resolved':>8s} | {f'ROTA, {N_EXC} steady excerpts':>27s} | {f'MoSQITo, same {N_EXC}':>27s} | agree")
    for r in rows:
        agree = int(np.sum(np.abs(r["mq"] - r["rota"]) <= AGREE_DB))
        fmt = lambda a: " ".join("  -  " if np.isnan(v) else f"{v:5.1f}" for v in a)  # noqa: E731
        print(f"  {r['rpm']:5.0f} {r['f0']:4.0f} | {r['angle']:12.1f} | {r['window']:11.1f} {r['sweep']:4.0f} Hz "
              f"{100 * r['sweep'] / r['cb']:6.0f}% {'no' if r['unresolved'] else 'yes':>8s} | {fmt(r['rota']):>27s} | "
              f"{fmt(r['mq']):>27s} | {agree}/{N_EXC}")
    print(f"  (sweep: how far order {ORDER:g} moves during the window, as a share of the critical band; the standard "
          f"calls a tone wider than 15 % unresolved; '-': MoSQITo found no tone within 2 Hz; agree: |MoSQITo - ROTA| "
          f"<= {AGREE_DB:g} dB on the same excerpt)")

    # ---- 2. + 3. MoSQITo metrics, binned along rpm -------------------------------------------------
    metrics = {k: f.result() for k, f in futures.items()}
    pool.shutdown()
    binned = {k: bin_metrics(m, sc.t, rpm) for k, m in metrics.items()}
    base = binned[VARIANTS[0]]
    seg = base["loudness"].segment
    print(f"MoSQITo on the run-up ({seg}, {EDGES[0]:.0f} to {EDGES[-1]:.0f} rpm in {EDGES[1] - EDGES[0]:.0f} rpm bins): "
          f"loudness N5 (ISO 532-1), sharpness median (DIN 45692), roughness R10 (Daniel & Weber)")
    print(f"  {'rpm':>5s} | {'N5':>5s} {'dN rem':>7s} {'dN +6':>6s} | {'S':>5s} {'dS rem':>7s} {'dS +6':>6s} | "
          f"{'R10':>6s} {'dR rem':>7s} {'dR +6':>6s}   (sone, acum, asper; d = variant - original)")
    for j in range(0, EDGES.size - 1):
        if not np.isfinite(base["roughness"].value[j]) or j % 2 == 0:
            continue
        cells = []
        for name, w in (("loudness", 5), ("sharpness", 5), ("roughness", 6)):
            v0 = base[name].value[j]
            d = [binned[k][name].value[j] - v0 for k in VARIANTS[1:]]
            cells.append(f"{v0:{w}.2f} {d[0]:+7.2f} {d[1]:+6.2f}" if name != "roughness" else
                         f"{v0:{w}.3f} {d[0]:+7.3f} {d[1]:+6.3f}")
        print(f"  {base['loudness'].centres[j]:5.0f} | " + " | ".join(cells))
    n_valid = {n: int(np.sum(base[n].valid)) for n in base}
    print(f"  valid bins: {n_valid}; samples per bin: loudness {int(np.median(base['loudness'].count))}, "
          f"roughness {int(np.median(base['roughness'].count))}")

    # ---- figure -----------------------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig = plt.figure(figsize=(8.0, 8.2), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.15])
    labels = (("loudness", "Loudness N5 (sone)", "ISO 532-1"), ("sharpness", "Sharpness, median (acum)", "DIN 45692"),
              ("roughness", "Roughness R10 (asper)", "Daniel & Weber"))
    for c, (name, ylabel, std) in enumerate(labels):
        ax = fig.add_subplot(gs[0, c])
        for k in VARIANTS:
            ax.plot(binned[k][name].centres, binned[k][name].value, "o-", ms=2.5, lw=1.2, color=COLOURS[k], label=k)
        ax.set_title(f"({'abc'[c]}) {std}", loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(frameon=True, fontsize=7, loc="upper left")
        ax = fig.add_subplot(gs[1, c])
        ax.axhline(0.0, color="0.4", lw=0.6)
        for k in VARIANTS[1:]:
            ax.plot(binned[k][name].centres, binned[k][name].value - base[name].value, "o-", ms=2.5, lw=1.2,
                    color=COLOURS[k], label=k)
        ax.set_title(f"({'def'[c]}) difference to the original", loc="left")
        ax.set_ylabel("Difference " + ylabel.split("(")[1].rstrip(")"))
        ax.set_xlabel("Engine speed (rpm)")
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(frameon=True, fontsize=7, loc="upper left")
        if name == "roughness":  # the roughness change is not resolved by a few samples per bin
            ax.text(0.97, 0.03, f"median IQR within a bin: {np.nanmedian(base[name].iqr):.3f} asper\n"
                    f"({int(np.median(base[name].count))} samples per bin)", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=6.5, color="0.25")
    for ax in fig.axes:
        ax.set_xlim(EDGES[0], EDGES[-1])

    ax = fig.add_subplot(gs[2, :])
    crit = op.tnr_criterion[:, 0]
    ok = np.isfinite(crit)
    ax.plot(op.rpm, op.tnr[:, 0], "o-", color="#0072B2", lw=1.8, ms=4,
            label="ROTA order_prominence, angle domain, run-up")
    ax.plot(op.rpm[ok], crit[ok], color="0.3", lw=1.0, ls="--", label="ECMA-418-1 criterion (defined above 89.1 Hz)")
    ax.plot([r["rpm"] for r in rows], [r["window"] for r in rows], "s", mfc="none", mec="#D55E00", ms=7, mew=1.3,
            label=f"ROTA tone_prominence, fixed {WINDOW_S:g} s window of the run-up")
    ax.plot([r["rpm"] for r in rows], [np.mean(r["rota"]) for r in rows], "x", color="k", ms=8, mew=1.5,
            label=f"ROTA tone_prominence, steady excerpts (mean of {N_EXC})")
    floor = np.floor(min(np.nanmin(op.tnr), min(np.nanmin(r["rota"]) for r in rows),
                         min(np.nanmin(r["mq"]) if np.isfinite(r["mq"]).any() else 99.0 for r in rows),
                         min(r["window"] for r in rows)) / 5.0) * 5.0 - 5.0
    missing = False
    for j, r in enumerate(rows):
        xs = r["rpm"] + 40.0 * (np.arange(N_EXC) - 0.5 * (N_EXC - 1)) / N_EXC * 2.0
        found = np.isfinite(r["mq"])
        ax.plot(xs[found], r["mq"][found], "D", color="#009E73", ms=4.5, mec="k", mew=0.4,
                label=f"MoSQITo tnr_ecma_st, steady excerpts ({N_EXC} seeds)" if j == 0 else None)
        ax.plot(xs[~found], np.full(int((~found).sum()), floor + 1.5), "v", color="#CC79A7", ms=5, mec="k", mew=0.4)
        missing |= bool((~found).any())
    if missing:
        ax.plot([], [], "v", color="#CC79A7", ms=5, mec="k", mew=0.4, label="MoSQITo found no tone (drawn at the bottom)")
    ax.set_ylim(floor, np.ceil(max(np.nanmax(op.tnr), max(np.nanmax(r["rota"]) for r in rows)) / 5.0) * 5.0 + 12.0)
    ax.set_xlim(EDGES[0], EDGES[-1])
    ax.set_xlabel("Engine speed (rpm); frequency of order 2 = rpm / 30 Hz")
    ax.set_ylabel("Tone-to-noise ratio of order 2 (dB)")
    ax.set_title("(g) Tone prominence of order 2: angle domain against steady excerpts", loc="left")
    ax.grid(alpha=0.3)
    ax.legend(frameon=True, fontsize=7, loc="upper left", ncol=2)

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Figure written to {FIG_PATH.relative_to(ROOT)}   (total {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
