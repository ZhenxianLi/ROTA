"""A worked analysis: from a run-up recording with a tonal complaint to an engineering conclusion.

The recording comes from ``rota.synth.synth_scenario("ev_runup")`` (8 s, 32 kHz, one channel, seed
fixed): a synthetic electric drive with four pole pairs behind a 12:1 reduction, accelerating at full
load from walking pace to 109 km/h (200 to 10900 rpm). Nothing is measured; all of it comes with
exact truth. The complaint is a tone, the gear whine at order 40, which is the typical tonal
problem of an electric drive. In the defaults of the scenario no order is prominent, so the script
builds one problem in, and says so:

* a structural resonance is added to the transfer path (``PROBLEM_MODE``: 2900 Hz, quality factor
  20, static gain 0.5, added to the default path modes), so the whine is amplified when it passes
  2900 Hz, which is at 4350 rpm;
* the cabin has no diffuse-field fading up to 7.9 kHz (``CABIN``: ``crossover_hz``). With the
  default cabin the level of every order above about 340 Hz also fades in and out by several dB on a
  scale of tens of Hz, which would hide the resonance behind a ragged curve. Everything else is a
  default of ``synth_scenario``, except the 24-pulse tachometer (``tacho.ppr``: 48 pulses would be
  shorter than 2 samples at 32 kHz above 10000 rpm).

The analysis does not use this knowledge. It finds the order, the speed range and the size of the
problem from the signal, and the script compares them with the truth at each step.

Steps (the printed output follows them):

1. Preparation. ``tacho_to_rpm`` gives the speed from the tachometer (the conditioning delay is
   compensated); ``rpm_from_ridge`` estimates it from the signal alone, with a comb of the orders of
   the drive, and the two are compared. ``to_calibrated_48k`` makes the Pa signal at 48 kHz that the
   WAV files need, and that the bridges of step 5 make for the toolboxes (the level is unchanged,
   which is printed).
2. Overview. ``overall_level`` (A-weighted) and ``band_levels_vs_rpm`` (1/3-octave bands) against
   rpm, and ``rpm_order_map``. Candidate orders are the peaks of the average order spectrum
   (``order_spectrum``) that stand out of its running median.
3. Diagnosis. ``order_slice`` and ``order_prominence`` of every candidate (tone-to-noise ratio and
   prominence ratio of ECMA-418-1, 3rd edition 2024, against ``prominence_criterion``);
   ``order_share`` of the strongest narrowband line of each frame of the Campbell diagram
   (``rpm_frequency_map``) on each candidate order. Result: which order is prominent, between which
   speeds and by how much above the criterion, compared with the truth from the stems of the scenario.
4. Extraction and what-if. ``resolution_check`` justifies the Vold-Kalman bandwidth;
   ``order_waveform`` (through ``extract.residual``) extracts the order and the residual;
   ``modify_orders`` lowers the order by the reduction that the excess calls for (checked, and raised
   by 1 dB until the check passes) and by a round 10 dB; ``resynthesize`` adds the residual back. The
   new signals go through the same analysis. ``modify_orders`` scales the order over the whole run;
   a real fix, a damping of the resonance, would act only near it.
5. Psychoacoustics, with optional toolboxes, through ``rota.bridges``. MoSQITo (Apache-2.0,
   ``pip install mosqito``): ``bridges.mosqito.loudness`` and ``bridges.mosqito.sharpness``, ISO 532-1
   loudness (``loudness_zwtv``) and DIN 45692 sharpness (``sharpness_din_from_loudness``). The Sottek
   hearing model (``pip install sottek-hearing-model``, AGPL-3.0-only, Python 3.11 or later):
   ``bridges.sottek_hearing_model.tonality``, ECMA-418-2 tonality (``shm_tonality_ecma``). The bridges
   take the signals in Pa at 32 kHz and resample them to 48 kHz; the ``vs_rpm`` method of each result
   bins it along rpm with the start-up exclusion and the aggregation that the bridge recommends
   (loudness N5, the 95th percentile; sharpness, the median; tonality, the mean of the frames above
   0.02 tu_HMS), and the values are compared for the original and the two reduced signals. ROTA does
   not depend on either package: a bridge imports its toolbox only inside the function that calls it
   and calls only documented public functions. Without a package, or if its import fails, the script
   runs and says which part it skipped. Other open-source implementations of psychoacoustic metrics
   are catalogued at https://zhenxianli.github.io/LISQM/. Editions: ROTA's prominence follows
   ECMA-418-1 (3rd edition, 2024); the Sottek model implements ECMA-418-2:2025 (4th edition).
6. Conclusion: problem order, speed range, excess over the criterion, reduction needed, the effect
   on the A-weighted level, loudness, sharpness and tonality, next to the truth.

What the standard leaves to the listener is not assessed: ECMA-418-1 calls a tone prominent when its
ratio reaches the criterion and it is audible (an aural examination).

Amplitude conventions: maps, slices and levels are RMS in Pa, levels in dB re 20 uPa; extracted order
envelopes are peak values, but only gains (ratios) are applied to them; the truth amplitudes of the
scenario are peak values (divided by sqrt(2) for the comparison). Tone and noise powers of
``order_prominence`` are mean-square values (RMS convention). The toolboxes take Pa and return sone,
acum and tu_HMS. Speed in rpm, frequency in Hz, time in s.

Audio: three mono 16-bit WAV files at 48 kHz in ``docs/audio/`` (the recording and the two reduced
versions), scaled with the same ``FULL_SCALE_PA`` as ``05_realistic_engine_audio.py`` and
``06_electric_drive_audio.py`` so that their levels can be compared directly. The script refuses to
write a file that would clip.

Run:  python examples/10_worked_analysis.py     (needs matplotlib; about 2 minutes on four cores with
both toolboxes, about 25 s without them)
Writes docs/images/10_worked_overview.png, 10_worked_diagnosis.png, 10_worked_whatif.png and
docs/audio/worked_*.wav. The narrative is in docs/worked_example.md.

Toolboxes used (cite them as their authors ask; ROTA does not include or redistribute
their code, which keeps its own licence):

* Green Forge Coop (2024). MOSQITO (version 1.2.1) [software]. Zenodo.
  https://doi.org/10.5281/zenodo.11026796 (Apache-2.0).
* Lotinga, M. J. B., Torjussen, M., & Felix Greco, G. (2025). Verified implementations of
  the Sottek psychoacoustic Hearing Model standardised sound quality metrics (ECMA-418-2
  loudness, tonality and roughness). Proceedings of Forum Acusticum / Euronoise 2025,
  Malaga, Spain. https://doi.org/10.61782/fa.2025.0951. Software: sottek-hearing-model 0.1.14,
  https://github.com/mlotinga/sottek-hearing-model (AGPL-3.0-only).

"""

import importlib.metadata
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage, signal
from scipy.io import wavfile

import rota
from rota import bridges

P_REF = 20e-6  # reference pressure in Pa (dB re 20 uPa), passed explicitly
FULL_SCALE_PA = 4.0  # sound pressure in Pa that maps to digital full scale in every WAV file
FADE_S = 0.02  # raised-cosine fade-in and fade-out of every WAV file, s
FS = 32000.0  # sampling rate of the scenario and of the analysis, Hz
FS_TOOL = 48000.0  # sampling rate of the WAV files (and of the toolboxes, to which the bridges resample), Hz
DURATION = 8.0  # s
SEED = 3
PPR = 24  # tachometer pulses per revolution (the default 48 is too short at 32 kHz above 10000 rpm)

# The problem we built in (see the module docstring): one structural resonance added to the
# transfer path, and a cabin without the diffuse-field fading. Everything else is a default.
PROBLEM_MODE = {"freq": 2900.0, "q": 20.0, "gain": 0.5}  # added path mode: Hz, quality factor, static gain
CABIN = {"crossover_hz": 7900.0}  # smooth direct arrival (no diffuse fading) up to this frequency, Hz

# 1. speed
RIDGE_RANGE = (2000.0, 11500.0)  # search range of the tacholess estimate, rpm
RIDGE_ORDERS = (2.0, 4.0, 6.7, 8.0, 10.0, 12.0, 14.0, 40.0)  # comb: the orders of the drive (design data)
RIDGE_WINDOW = 0.2  # s
RIDGE_FROM_RPM = 2500.0  # the two speed estimates are compared above this speed, rpm
EDGE_S = 0.3  # start of the record left out of the statistics (start-up of the path), s

# 2. overview
BIN_RPM = 250.0  # speed bins of the overview and of the toolbox metrics, rpm
EDGES = np.arange(1000.0, 11001.0, BIN_RPM)
FRAME_S = 0.05  # frame of the level and band-level analyses, s
BAND_FROM_HZ = 500.0  # the band-level bump is searched in bands from this frequency (below: rumble and thumps), Hz
ORDER_RES = 0.125  # bin spacing of the order map, orders (8 revolutions per frame)
MAX_ORDER_MAP = 44.0  # top of the order map, orders
CAND_RPM = (2500.0, 11000.0)  # speed range of the average order spectrum that nominates candidates, rpm
CAND_EXCESS_DB = 6.0  # a candidate stands this far above the running median of the order spectrum
CAND_FLOOR_BINS = 49  # width of the running median, in order-map bins (+-3 orders)
CAND_WEAK_DB = 2.0  # weaker peaks from this height are listed, but are not candidates

# 3. diagnosis
MIN_RPM_PROM = 1500.0  # order_prominence leaves out frames below this speed, rpm
BIN_PROM = 100.0  # speed bins of the analysis of the problem order, rpm
BIN_SCREEN = 250.0  # speed bins when the long frames of the other candidates are screened, rpm
LINE_RES_HZ = 20.0  # bin spacing of the Campbell diagram used to find the strongest line, Hz
LINE_BAND = (300.0, 9000.0)  # the strongest line is searched in this band, Hz
LINE_EXCESS_DB = 6.0  # ... and counts if it is this far above the local median, dB
LINE_TOL = 0.02  # relative tolerance of the attribution of a line to an order

# 4. extraction and what-if
VK_BANDWIDTH = 20.0  # Vold-Kalman -3 dB bandwidth, Hz
ROUND_REDUCTION_DB = 10.0  # the round-valued what-if, dB

# 5. toolbox metrics (start-up exclusion, aggregation and tonality threshold: those of each bridge result)
TONALITY_TOL = 0.10  # relative tolerance when the model's tonal frequency is attributed to an order
MIN_COUNT = 3

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "images"
AUDIO_DIR = ROOT / "docs" / "audio"
BLUE, ORANGE, GREEN, GREY, RED, PURPLE = "#0072B2", "#D55E00", "#009E73", "#555555", "#CC3311", "#7B3294"


def db(amplitude):
    """Level in dB re P_REF of an amplitude in Pa (RMS in, RMS level out)."""
    return 20.0 * np.log10(np.maximum(amplitude, 1e-12) / P_REF)


def energy_mean_db(level):
    """Energy mean of levels in dB."""
    return float(10.0 * np.log10(np.mean(10.0 ** (np.asarray(level, dtype=float) / 10.0))))


# ---- signal and truth -----------------------------------------------------------------------------


def build_scenario(duration=DURATION):
    """The seeded ``ev_runup`` recording with the problem built in (mono, ``FS``, Pa).

    The default transients (a thump and a click) sit at fractions of the record; they are given in
    seconds here, at the times they have in the full ``DURATION`` record, so that a shortened record
    (the test) keeps them where they are and a transient does not land on the passage of the resonance.
    """
    defaults = rota.synth.scenario_defaults("ev_runup")
    modes = defaults["path"]["modes"] + [dict(PROBLEM_MODE)]
    transients = []
    for tr in defaults["transients"]:
        at = tr["at"] * DURATION
        if at < duration:
            transients.append({**{k: v for k, v in tr.items() if k != "at"}, "time": at})
    return rota.synth.synth_scenario(
        "ev_runup", fs=FS, duration=duration, seed=SEED, tacho={"ppr": PPR}, path={"modes": modes},
        cabin=dict(CABIN), transients=transients,
    )


def truth_of(sc):
    """What was built in, from the scenario: the whine order, the resonance and where the order meets it."""
    teeth = float(sc.config["parameters"]["gear_mesh"]["teeth"])
    k = sc.order_names.index(f"gear_mesh_{teeth:g}")
    f0, q = PROBLEM_MODE["freq"], PROBLEM_MODE["q"]
    return {
        "k": k, "order": teeth, "f_res": f0, "q": q,
        "rpm_res": 60.0 * f0 / teeth,  # speed at which the order is at the resonance
        "rpm_band": (60.0 * (f0 - 0.5 * f0 / q) / teeth, 60.0 * (f0 + 0.5 * f0 / q) / teeth),  # half-power band
    }


def true_tnr(sc, rpm, op, k, tr):
    """Tone-to-noise ratio of order ``k`` from the stems, on the points of an ``OrderProminence`` ``op``.

    Tone: the quasi-static peak amplitude at the receiver (``quasi_static_amplitude``, peak -> RMS),
    squared. Noise: the PSD of everything but that order (``x_clean`` minus its stem through the
    path), from an order map with the frame length and overlap of ``op``, averaged over the lines of
    the critical band around the tone and the frames of the speed bin, times the critical bandwidth.
    Same units as ``tone_prominence`` (mean-square values; RMS convention).
    """
    stem = sc.through_path(sc.order_stems[:, k])
    rest = sc.x_clean - stem
    m = rota.maps.rpm_order_map(
        rest, sc.fs, rpm, order_resolution=op.order_resolution, amplitude="rms", min_rpm=MIN_RPM_PROM,
        max_order=1.5 * tr["order"], overlap=op.params["overlap"],
    )
    tone = (sc.quasi_static_amplitude(k) / np.sqrt(2.0)) ** 2
    idx = np.digitize(m.rpm, op.rpm_bins) - 1
    out = []
    for b in range(op.rpm_bins.size - 1):
        sel = np.flatnonzero(idx == b)
        if sel.size == 0:
            continue
        r = float(np.mean(m.rpm[sel])) / 60.0
        ft = tr["order"] * r
        dfc = rota.tonal.critical_bandwidth(ft)
        psd = np.mean(np.abs(m.values[:, sel]) ** 2, axis=1) / (m.enbw_bins * m.resolution * r)
        band = np.abs(m.axis * r - ft) <= 0.5 * dfc
        out.append(10.0 * np.log10(np.interp(np.mean(m.t[sel]), sc.t, tone) / (np.mean(psd[band]) * dfc)))
    if len(out) != op.rpm.size:
        raise RuntimeError("the truth frames do not match the analysis frames")
    return np.array(out)


def prominent_range(rpm, excess):
    """Speed range (rpm) of the contiguous run around the maximum of ``excess`` where it is >= 0.

    The edges are linear interpolations of the zero crossings; NaN, NaN if the maximum is below 0.
    """
    e = np.where(np.isfinite(excess), excess, -np.inf)
    i = int(np.argmax(e))
    if not e[i] >= 0.0:
        return float("nan"), float("nan")
    a = i
    while a > 0 and e[a - 1] >= 0.0:
        a -= 1
    b = i
    while b < e.size - 1 and e[b + 1] >= 0.0:
        b += 1
    lo = rpm[a] if a == 0 or not np.isfinite(e[a - 1]) else np.interp(0.0, [e[a - 1], e[a]], [rpm[a - 1], rpm[a]])
    hi = rpm[b] if b == e.size - 1 or not np.isfinite(e[b + 1]) else np.interp(0.0, [e[b + 1], e[b]], [rpm[b + 1], rpm[b]])
    return float(lo), float(hi)


def default_reference(duration):
    """Largest TNR excess over the criterion of any source order in the unmodified scenario.

    The same recording without the added resonance and with the default cabin, analysed with
    ``order_prominence`` in ``BIN_SCREEN`` rpm bins. Returns (excess in dB, order, rpm).
    """
    sc = rota.synth.synth_scenario("ev_runup", fs=FS, duration=duration, seed=SEED, tacho={"ppr": PPR})
    rpm = rota.speed.tacho_to_rpm(sc.tacho, FS, ppr=PPR, t_out=sc.t + float(sc.config["resolved"]["tacho_delay"]))
    best = (-np.inf, None, None)
    for o in sc.orders:
        pr = prominence(sc.x, FS, rpm, float(o), BIN_SCREEN)
        e, r = max_of(pr["rpm"], pr["tnr_excess"])
        if e > best[0]:
            best = (e, float(o), r)
    return best


# ---- 1. preparation ---------------------------------------------------------------------------------


def speed_references(sc, x):
    """Speed from the tachometer and from the signal alone, on the sample grid of ``sc``.

    Returns ``rpm`` (tachometer, conditioning delay compensated), ``rpm_ridge`` (tacholess,
    ``rpm_from_ridge`` with a comb of the drive's orders) and the ridge result.
    """
    delay = float(sc.config["resolved"]["tacho_delay"])
    rpm = rota.speed.tacho_to_rpm(sc.tacho, sc.fs, ppr=PPR, t_out=sc.t + delay)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ridge = rota.speed.rpm_from_ridge(
            x, sc.fs, RIDGE_RANGE, orders=RIDGE_ORDERS, window=RIDGE_WINDOW, moving=True
        )
    return rpm, np.interp(sc.t, ridge.t, ridge.rpm), ridge


# ---- 2. overview ------------------------------------------------------------------------------------


def find_candidate_orders(om, rpm_range=CAND_RPM):
    """Orders at which the average order spectrum stands out of its surroundings.

    The spectrum is the energy average of the order map over ``rpm_range``
    (``rota.slices.order_spectrum``, RMS, dB re 20 uPa). A candidate is a peak of at least
    ``CAND_EXCESS_DB`` above the running median of the spectrum over +-3 orders, above order 1.
    The candidate orders are the peaks rounded to the nearest half order. Returns the candidate
    orders, their excess in dB, the orders of the peaks as found, and the weaker peaks (from
    ``CAND_WEAK_DB``) as a list of (order, excess).
    """
    sp = rota.slices.order_spectrum(om, rpm_range=rpm_range)
    level = db(sp.amplitude)
    excess = level - ndimage.median_filter(level, size=CAND_FLOOR_BINS, mode="nearest")
    pk, _ = signal.find_peaks(excess, height=CAND_WEAK_DB, distance=4)
    pk = pk[sp.axis[pk] >= 1.0]
    strong = excess[pk] >= CAND_EXCESS_DB
    orders = np.round(2.0 * sp.axis[pk[strong]]) / 2.0
    keep = np.r_[True, np.diff(orders) > 0]
    weak = [(float(sp.axis[i]), float(excess[i])) for i in pk[~strong]]
    return orders[keep], excess[pk[strong]][keep], sp.axis[pk[strong]][keep], weak


# ---- 3. diagnosis -----------------------------------------------------------------------------------


def strongest_lines(x, fs, rpm):
    """Frequency of the strongest narrowband line per frame of the Campbell diagram, and its excess.

    ``rpm_frequency_map`` (``LINE_RES_HZ`` bins, RMS); the excess of a bin is its level (dB) minus the
    median level of the 81 bins around it. The strongest line is the bin of largest excess in
    ``LINE_BAND``; frames in which it is less than ``LINE_EXCESS_DB`` high get NaN. Returns t, rpm,
    frequency (Hz) and excess (dB).
    """
    fmap = rota.maps.rpm_frequency_map(x, fs, rpm, resolution=LINE_RES_HZ, amplitude="rms")
    level = db(np.abs(fmap.values))
    excess = level - ndimage.median_filter(level, size=(81, 1), mode="nearest")
    in_band = (fmap.axis >= LINE_BAND[0]) & (fmap.axis <= LINE_BAND[1])
    excess = np.where(in_band[:, None], excess, -np.inf)
    i = np.argmax(excess, axis=0)
    peak = excess[i, np.arange(excess.shape[1])]
    return fmap.t, fmap.rpm, np.where(peak >= LINE_EXCESS_DB, fmap.axis[i], np.nan), peak


def prominence(x, fs, rpm, order, bin_width):
    """``order_prominence`` of one order in speed bins, as a plain dictionary (RMS, dB)."""
    op = rota.tonal.order_prominence(x, fs, rpm, [order], rpm_bins=bin_width, min_rpm=MIN_RPM_PROM, ref=P_REF)
    return {
        "op": op, "rpm": op.rpm, "freq": op.freq[:, 0], "tnr": op.tnr[:, 0], "pr": op.pr[:, 0],
        "tnr_crit": op.tnr_criterion[:, 0], "pr_crit": op.pr_criterion[:, 0],
        "tnr_excess": op.tnr[:, 0] - op.tnr_criterion[:, 0], "pr_excess": op.pr[:, 0] - op.pr_criterion[:, 0],
    }


def max_of(rpm, v):
    """(maximum, rpm of the maximum) ignoring NaN."""
    i = int(np.nanargmax(v))
    return float(v[i]), float(rpm[i])


# ---- 4. extraction and what-if ----------------------------------------------------------------------


def reduce_order(x, fs, rpm, order, reduction_db, comp=None, resid=None):
    """Extract ``order`` with Vold-Kalman and rebuild the signal with it reduced by ``reduction_db``.

    ``modify_orders`` scales the extracted order (peak envelope) by ``10 ** (-reduction_db / 20)`` over
    the whole run and ``resynthesize`` adds the unchanged residual. Returns the new signal, the
    extracted components and the residual.
    """
    if comp is None:
        resid, comp = rota.extract.residual(x, fs, rpm, [order], method="vold-kalman", bandwidth=VK_BANDWIDTH)
    mod = rota.resynth.modify_orders(comp, gain_db={order: -float(reduction_db)})
    y = rota.resynth.resynthesize(mod, resid)["mix"]
    return y, comp, resid


# ---- 5. toolbox metrics -------------------------------------------------------------------------------


def mosqito_metrics(x):
    """ISO 532-1 loudness and DIN 45692 sharpness of a signal in Pa at ``FS``, with ``rota.bridges.mosqito``.

    Module-level so that it can run in a worker process. The bridge resamples the signal to 48 kHz,
    calls MoSQITo (free field) and silences its output; the sharpness is computed from the specific
    loudness of the loudness result. Returns ``TimeMetric`` results.
    """
    loud = bridges.mosqito.loudness(x, FS)
    return {"loudness": loud, "sharpness": bridges.mosqito.sharpness(None, None, loudness=loud)}


def order_tonality(ton, t, rpm, order):
    """Largest specific tonality among the half bands whose tonal frequency lies on ``order`` (else 0).

    ``ton`` is a tonality result of ``rota.bridges.sottek_hearing_model`` and ``rpm`` the speed at the
    times ``t``. The result is a ``TimeMetric`` on the frames of ``ton`` with its start-up exclusion
    and no threshold (the zeros count), so that ``vs_rpm`` bins it like the tonality.
    """
    spec = ton.extra["spec_tonality"]
    f = np.where(spec > ton.threshold, ton.extra["spec_tonality_freqs"], np.nan)
    rpm_m = np.interp(ton.t, t, rpm)  # speed on the model's frames
    fo = rota.slices.frequency_to_order(f, rpm_m[:, None], tolerance=TONALITY_TOL, orders=[order])
    return bridges.TimeMetric(
        name=f"order-{order:g} tonality", t=ton.t, values=np.max(np.where(fo.matched, spec, 0.0), axis=1),
        unit=ton.unit, skip_start=ton.skip_start, source=ton.source,
    )


def bin_mosqito(m, t, rpm):
    """Loudness (N5, 95th percentile) and sharpness (median) of ``mosqito_metrics`` along rpm.

    ``vs_rpm`` skips the start-up and aggregates as the bridge recommends.
    """
    return {name: metric.vs_rpm(rpm, t, bins=EDGES, min_count=MIN_COUNT) for name, metric in m.items()}


def bin_sottek(ton, t, rpm, order):
    """Overall tonality and order tonality of a tonality result of the Sottek model along rpm.

    ``vs_rpm`` averages the tonality frames above the threshold of the result after its start-up and
    gives the last frame the last speed (the model puts it at the signal length rounded up to a multiple
    of 256 samples at 48 kHz, here one sample after the last speed sample).
    """
    kw = {"bins": EDGES, "min_count": MIN_COUNT}
    return {
        "tonality": ton.vs_rpm(rpm, t, **kw),
        "order_tonality": order_tonality(ton, t, rpm, order).vs_rpm(rpm, t, **kw),
    }


# ---- audio ------------------------------------------------------------------------------------------------


def write_wav(name, x, fs):
    """Write ``x`` (Pa, shape (n,)) as 16-bit PCM, scaled by FULL_SCALE_PA.

    Sample value = round(32767 * p / FULL_SCALE_PA) after a raised-cosine fade of FADE_S at both
    ends; nothing else is applied (no normalisation, no limiter).
    """
    x = np.array(x, dtype=float)
    n_fade = round(FADE_S * fs)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_fade) / n_fade))
    x[:n_fade] *= ramp
    x[-n_fade:] *= ramp[::-1]
    peak = float(np.max(np.abs(x)))
    if peak >= FULL_SCALE_PA:
        raise ValueError(f"{name}: peak {peak:.3f} Pa would clip; raise FULL_SCALE_PA")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIO_DIR / name
    wavfile.write(path, int(fs), np.round(x / FULL_SCALE_PA * 32767.0).astype(np.int16))
    rms = float(np.sqrt(np.mean(x**2)))
    print(f"  {name:36s} {db(rms):5.1f} dB SPL (RMS)   peak {20 * np.log10(peak / FULL_SCALE_PA):6.1f} dBFS"
          f"   {path.stat().st_size / 1e6:.2f} MB")


# ---- the analysis chain (steps 1 to 4, ROTA only) ------------------------------------------------------


def analyse(duration=DURATION, log=print, reference=True):
    """Steps 1 to 4 of the worked analysis with ROTA alone; returns every result in a dictionary.

    ``log`` receives the printed lines (``print`` in the script; a no-op in the test); ``reference=False``
    skips the check of the unmodified scenario. Nothing about
    the built-in problem is used before the diagnosis has named the order, the speed range and the
    reduction; the truth (``truth_of``, the stems) is only used to compare with what was found.
    """
    R = {}
    sc = build_scenario(duration)
    tr = truth_of(sc)
    x = sc.x
    R.update(sc=sc, tr=tr, x=x)

    # ---- 1. preparation ----------------------------------------------------------------------------
    log("== 1. Preparation ==")
    log(f"Scenario ev_runup: {duration:g} s at {FS:g} Hz, seed {SEED}, mono, {PPR}-pulse tachometer; "
        f"{sc.rpm_mean[0]:.0f} -> {sc.rpm_mean[-1]:.0f} rpm, {sc.speed_kmh[0]:.0f} -> {sc.speed_kmh[-1]:.0f} km/h; "
        f"RMS {db(np.sqrt(np.mean(x**2))):.1f} dB re 20 uPa, peak {np.max(np.abs(x)):.2f} Pa")
    log(f"The problem we built in: a structural resonance at {tr['f_res']:g} Hz (Q {tr['q']:g}, static gain "
        f"{PROBLEM_MODE['gain']:g}) added to the transfer path, and a cabin without diffuse-field fading up to "
        f"{CABIN['crossover_hz']:g} Hz. The gear whine, order {tr['order']:g}, is at the resonance at "
        f"{tr['rpm_res']:.0f} rpm (half-power band {tr['rpm_band'][0]:.0f} to {tr['rpm_band'][1]:.0f} rpm).")
    rpm, rpm_ridge, ridge = speed_references(sc, x)
    sel = sc.t > EDGE_S
    err_t = np.abs(rpm / sc.rpm_mean - 1.0)
    modes0 = rota.synth.scenario_defaults("ev_runup")["path"]["modes"]
    f_low = np.array([100.0, 500.0, 1000.0, 2000.0])
    d_gain = 20.0 * np.log10(np.abs(rota.synth.modal_response(f_low, modes0 + [dict(PROBLEM_MODE)], FS))
                             / np.abs(rota.synth.modal_response(f_low, modes0, FS)))
    log("The added mode also raises the modal path gain below the resonance (static gain): "
        + ", ".join(f"{g:+.1f} dB at {f:g} Hz" for f, g in zip(f_low, d_gain)) + ".")
    if reference:
        e0, o0, r0 = default_reference(duration)
        log(f"For reference, the recording without these two changes (all defaults): no source order reaches its "
            f"TNR criterion; the closest, order {o0:g}, is {-e0:.1f} dB below it (at {r0:.0f} rpm, {BIN_SCREEN:g} rpm bins).")
        R["default_excess"] = (e0, o0, r0)
    log(f"Speed from the tachometer (tacho_to_rpm, delay compensated) against the true mean speed: "
        f"median {100 * np.median(err_t[sel]):.4f} %, 95th percentile {100 * np.percentile(err_t[sel], 95):.3f} %")
    above = rpm > RIDGE_FROM_RPM
    dif = rpm_ridge / rpm - 1.0
    log(f"Tacholess speed (rpm_from_ridge, comb of orders {', '.join(f'{o:g}' for o in RIDGE_ORDERS)}, search range "
        f"{RIDGE_RANGE[0]:.0f} to {RIDGE_RANGE[1]:.0f} rpm, window {ridge.window:g} s, moving): reliable "
        f"{ridge.quality['reliable']}, frames at a bound {int(np.sum(ridge.at_bound))}")
    log(f"  against the tachometer above {RIDGE_FROM_RPM:.0f} rpm: median |difference| "
        f"{100 * np.median(np.abs(dif[above])):.3f} %, 95th percentile {100 * np.percentile(np.abs(dif[above]), 95):.2f} %, "
        f"maximum {100 * np.max(np.abs(dif[above])):.2f} %")
    x48 = rota.tonal.to_calibrated_48k(x, FS, calibration=1.0, fs_out=FS_TOOL)
    log(f"to_calibrated_48k: {FS:g} -> {FS_TOOL:g} Hz, RMS {db(np.sqrt(np.mean(x**2))):.2f} -> "
        f"{db(np.sqrt(np.mean(x48**2))):.2f} dB re 20 uPa (calibration factor 1, the signal is in Pa)")
    R.update(rpm=rpm, rpm_ridge=rpm_ridge, ridge_dif=dif, ridge_mask=above, x48=x48)
    R["speed_stats"] = {
        "tacho_median": float(np.median(err_t[sel])), "ridge_median": float(np.median(np.abs(dif[above]))),
        "ridge_p95": float(np.percentile(np.abs(dif[above]), 95)), "ridge_max": float(np.max(np.abs(dif[above]))),
    }

    # ---- 2. overview -------------------------------------------------------------------------------
    log("\n== 2. Overview ==")
    ol = rota.slices.overall_level(x, FS, rpm, frame=FRAME_S, weighting="A", ref=P_REF, rpm_bins=EDGES)
    fit = np.polyfit(ol.centres[ol.counts > 0], ol.binned_level[ol.counts > 0], 3)
    resid_a = ol.binned_level - np.polyval(fit, ol.centres)
    j = int(np.nanargmax(np.abs(resid_a)))
    log(f"A-weighted level in {BIN_RPM:g} rpm bins: {np.nanmin(ol.binned_level):.1f} dB at {ol.centres[np.nanargmin(ol.binned_level)]:.0f} rpm "
        f"to {np.nanmax(ol.binned_level):.1f} dB at {ol.centres[np.nanargmax(ol.binned_level)]:.0f} rpm; largest departure "
        f"from a smooth cubic fit {resid_a[j]:+.2f} dB at {ol.centres[j]:.0f} rpm (frames of {FRAME_S * 1e3:.0f} ms)")
    bl = rota.tonal.band_levels_vs_rpm(
        x, FS, rpm, rpm_bins=EDGES, frame=FRAME_S, fraction=3, fmin=100.0, fmax=12500.0, ref=P_REF, min_count=2)
    lv = bl.levels
    bump = np.full(lv.shape, np.nan)
    with warnings.catch_warnings():  # empty bins (a shorter record) give all-NaN neighbours
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(lv.shape[0]):  # level minus the median of the neighbouring speed bins (+-3)
            nb = [k for k in range(max(0, i - 3), min(lv.shape[0], i + 4)) if k != i]
            bump[i] = lv[i] - np.nanmedian(lv[nb], axis=0)
    bump[:, bl.nominal < BAND_FROM_HZ] = np.nan
    ib, jb = np.unravel_index(np.nanargmax(bump), bump.shape)
    log(f"1/3-octave band levels in the same bins ({bl.nominal[0]:g} to {bl.nominal[-1]:g} Hz): among the bands from "
        f"{BAND_FROM_HZ:g} Hz, the one that rises most above the median of its neighbouring speed bins (+-3) is "
        f"{bl.nominal[jb]:g} Hz at {bl.rpm_centres[ib]:.0f} rpm, +{bump[ib, jb]:.1f} dB (level {lv[ib, jb]:.1f} dB re 20 uPa)")
    om = rota.maps.rpm_order_map(x, FS, rpm, order_resolution=ORDER_RES, max_order=MAX_ORDER_MAP, amplitude="rms",
                                 min_rpm=500.0)
    cand, cand_excess, cand_peak, weak = find_candidate_orders(om)
    log(f"Order map ({ORDER_RES:g} order bins, RMS): peaks of the average order spectrum {CAND_RPM[0]:.0f} to "
        f"{CAND_RPM[1]:.0f} rpm at least {CAND_EXCESS_DB:g} dB above the running median: "
        + ", ".join(f"{p:g} (+{e:.0f} dB)" for p, e in zip(cand_peak, cand_excess))
        + f"  ->  candidate orders {', '.join(f'{o:g}' for o in cand)} (the peaks are rounded to the nearest half order)")
    log(f"  weaker peaks, {CAND_WEAK_DB:g} to {CAND_EXCESS_DB:g} dB, are not taken as candidates: "
        + ", ".join(f"{o:g} (+{e:.0f} dB)" for o, e in weak))
    R.update(ol=ol, resid_a=resid_a, bl=bl, bump=bump, om=om, cand=cand, cand_excess=cand_excess, weak=weak)

    # ---- 3. diagnosis ------------------------------------------------------------------------------
    log("\n== 3. Diagnosis ==")
    screen = {}
    log(f"order_prominence of each candidate ({BIN_SCREEN:g} rpm bins; frames of the length the standard's line "
        f"spacing needs, so the lowest orders have a coarse resolution in rpm) -- maximum of TNR and PR over the criterion:")
    for o in cand:
        pr = prominence(x, FS, rpm, float(o), BIN_SCREEN)
        e_t, r_t = max_of(pr["rpm"], pr["tnr_excess"])
        e_p, r_p = max_of(pr["rpm"], pr["pr_excess"])
        sl = rota.slices.order_slice(om, float(o))
        ok = (sl.rpm > CAND_RPM[0]) & np.isfinite(sl.amplitude)
        screen[float(o)] = {"pr": pr, "tnr_excess": e_t, "tnr_rpm": r_t, "pr_excess": e_p, "pr_rpm": r_p,
                            "slice": sl, "level_max": float(np.max(db(sl.amplitude[ok])))}
        log(f"  order {o:4g}: frames of {pr['op'].frame_revs:g} revolutions; TNR {e_t:+6.1f} dB at {r_t:5.0f} rpm, "
            f"PR {e_p:+6.1f} dB at {r_p:5.0f} rpm; order-slice maximum {screen[float(o)]['level_max']:.1f} dB re 20 uPa")
    problem = max(screen, key=lambda o: screen[o]["tnr_excess"])
    log(f"-> order {problem:g} is the only candidate above its criterion.")
    R.update(screen=screen, problem=problem)

    # fine analysis of the problem order
    fine = prominence(x, FS, rpm, problem, BIN_PROM)
    op = fine["op"]
    e_t, r_t = max_of(fine["rpm"], fine["tnr_excess"])
    e_p, r_p = max_of(fine["rpm"], fine["pr_excess"])
    lo, hi = prominent_range(fine["rpm"], fine["tnr_excess"])
    lo_p, hi_p = prominent_range(fine["rpm"], fine["pr_excess"])
    f_lo, f_hi = problem * lo / 60.0, problem * hi / 60.0
    lo6, hi6 = prominent_range(fine["rpm"], fine["tnr_excess"] - 6.0)
    log(f"Order {problem:g} alone: frames of {op.frame_revs:g} revolutions ({op.order_resolution:g} order lines), "
        f"{fine['rpm'].size} bins of {BIN_PROM:g} rpm, window {op.window}")
    log(f"  TNR above the ECMA-418-1 criterion between {lo:.0f} and {hi:.0f} rpm ({f_lo:.0f} to {f_hi:.0f} Hz), "
        f"maximum {fine['tnr'][np.nanargmax(fine['tnr_excess'])]:.1f} dB against {fine['tnr_crit'][np.nanargmax(fine['tnr_excess'])]:.1f} dB, "
        f"i.e. +{e_t:.1f} dB at {r_t:.0f} rpm; more than 6 dB above it between {lo6:.0f} and {hi6:.0f} rpm")
    log(f"  PR above its criterion between {lo_p:.0f} and {hi_p:.0f} rpm, maximum +{e_p:.1f} dB at {r_p:.0f} rpm; "
        f"tone wider than 15 % of the critical band in {int(np.sum(op.tone_band_exceeded[:, 0]))} of {op.rpm.size} bins")
    tnr_true = true_tnr(sc, rpm, op, tr["k"], tr)
    ex_true = tnr_true - fine["tnr_crit"]
    lo_t, hi_t = prominent_range(fine["rpm"], ex_true)
    lo6_t, hi6_t = prominent_range(fine["rpm"], ex_true - 6.0)
    e_true, r_true = max_of(fine["rpm"], ex_true)
    near = (fine["rpm"] > lo - 200) & (fine["rpm"] < hi + 200)
    log(f"  truth (tone from the scenario, noise from the stems): range {lo_t:.0f} to {hi_t:.0f} rpm, "
        f"maximum +{e_true:.1f} dB at {r_true:.0f} rpm, more than 6 dB above the criterion between {lo6_t:.0f} and "
        f"{hi6_t:.0f} rpm; difference measured - truth near the range: "
        f"median {np.nanmedian(fine['tnr'][near] - tnr_true[near]):+.2f} dB, largest |{np.nanmax(np.abs(fine['tnr'][near] - tnr_true[near])):.2f}| dB")
    log(f"  {'rpm':>6s} {'Hz':>5s} | {'TNR':>5s} {'crit':>5s} {'excess':>6s} {'truth':>6s} | {'PR':>5s} {'crit':>5s} {'excess':>6s}"
        "   (dB; truth: excess of the TNR from the stems; every second bin near the range)")
    for i in np.flatnonzero((fine["rpm"] >= lo - 150.0) & (fine["rpm"] <= hi + 150.0))[::2]:
        log(f"  {fine['rpm'][i]:6.0f} {fine['freq'][i]:5.0f} | {fine['tnr'][i]:5.1f} {fine['tnr_crit'][i]:5.1f} "
            f"{fine['tnr_excess'][i]:+6.1f} {ex_true[i]:+6.1f} | {fine['pr'][i]:5.1f} {fine['pr_crit'][i]:5.1f} "
            f"{fine['pr_excess'][i]:+6.1f}")
    log(f"  built-in resonance: order {tr['order']:g} reaches {tr['f_res']:g} Hz at {tr['rpm_res']:.0f} rpm "
        f"(measured maximum at {r_t:.0f} rpm, bins of {BIN_PROM:g} rpm)")
    R.update(fine=fine, tnr_true=tnr_true, ex_true=ex_true, window=(lo, hi), window6=(lo6, hi6), window_truth=(lo_t, hi_t),
             window6_truth=(lo6_t, hi6_t),
             max_excess=e_t, max_excess_rpm=r_t, max_excess_pr=e_p, max_excess_truth=e_true)

    # slice of the order against the quasi-static truth and the strongest line
    sl = screen[problem]["slice"]
    truth_rms = np.interp(sl.t, sc.t, sc.quasi_static_amplitude(tr["k"]) / np.sqrt(2.0))
    inw = (sl.rpm >= lo) & (sl.rpm <= hi) & np.isfinite(sl.amplitude)
    err_sl = db(sl.amplitude[inw]) - db(truth_rms[inw])
    log(f"  order slice in the range: maximum {np.max(db(sl.amplitude[inw])):.1f} dB re 20 uPa (RMS) against the "
        f"quasi-static truth {np.max(db(truth_rms[inw])):.1f} dB; difference median {np.median(err_sl):+.2f} dB, "
        f"largest |{np.max(np.abs(err_sl)):.2f}| dB ({int(inw.sum())} frames)")
    tl, rl, fl, pl = strongest_lines(x, FS, rpm)
    log("Strongest narrowband line per frame of the Campbell diagram "
        f"({LINE_RES_HZ:g} Hz bins, {LINE_BAND[0]:.0f} to {LINE_BAND[1]:.0f} Hz, at least {LINE_EXCESS_DB:g} dB over its local median), "
        "share of frames on each candidate order (order_share, weighted by the excess):")
    in_w = (rl >= lo) & (rl <= hi)
    shares = {}
    for label, m_ in (("whole run", np.ones(rl.size, bool)), ("range", in_w)):
        row = []
        for o in list(cand) + [34.0]:
            s = rota.slices.order_share(fl[m_], rl[m_], float(o), harmonics=1, tolerance=LINE_TOL, weights=pl[m_])
            shares[(label, float(o))] = s.share
            row.append(f"{o:g}: {100 * s.share:.0f} %" + (" (control, no source)" if o == 34.0 else ""))
        n_line = int(np.sum(np.isfinite(fl[m_])))
        log(f"  {label:9s} ({int(m_.sum())} frames, {n_line} with a line): " + ", ".join(row))
    on_tone = np.isfinite(fl) & (np.abs(fl - 6300.0) <= LINE_RES_HZ)
    log(f"  (the fixed 6.3 kHz tone, which is on no order, is the strongest line in {100 * np.mean(on_tone[np.isfinite(fl)]):.0f} % "
        f"of the frames with a line, {100 * np.mean(rl[on_tone] < lo):.0f} % of them below {lo:.0f} rpm)")
    R.update(shares=shares, lines=(tl, rl, fl, pl))

    # ---- 4. extraction and what-if -----------------------------------------------------------------
    log("\n== 4. Extraction and what-if ==")
    with warnings.catch_warnings():  # pairs of the other candidates at idle are not extracted: not our concern
        warnings.simplefilter("ignore")
        rep = rota.extract.resolution_check(rpm, FS, list(cand), bandwidth=VK_BANDWIDTH)
    pairs = [p for p in rep.pairs if problem in p["orders"]]
    nearest = min(pairs, key=lambda p: p["min_separation_hz"])
    log(f"resolution_check, Vold-Kalman bandwidth {VK_BANDWIDTH:g} Hz (pass band {2 * VK_BANDWIDTH:g} Hz) against the "
        f"candidates {', '.join(f'{o:g}' for o in cand)}: all pairs with order {problem:g} resolved "
        f"{all(p['resolved'] for p in pairs)}; nearest is order {next(o for o in nearest['orders'] if o != problem):g}, "
        f"at least {nearest['min_separation_hz']:.0f} Hz away, largest admissible bandwidth {min(p['required_bandwidth_hz'] for p in pairs):.0f} Hz")
    resid, comp = rota.extract.residual(x, FS, rpm, [problem], method="vold-kalman", bandwidth=VK_BANDWIDTH)
    stem = sc.through_path(sc.order_stems[:, tr["k"]])
    s_ok = sc.t > EDGE_S
    sim = rota.metrics.waveform_similarity(comp.waveforms[s_ok, 0], stem[s_ok])
    win = (rpm >= lo) & (rpm <= hi)
    lvl = lambda a: db(np.sqrt(np.mean(a[win] ** 2)))
    log(f"extracted order {problem:g} against its true stem through the path: correlation {sim['corr']:.4f}, "
        f"nrmse {sim['nrmse']:.3f}; RMS level in the range {lvl(comp.waveforms[:, 0]):.2f} dB (truth {lvl(stem):.2f} dB); "
        f"residual {db(np.sqrt(np.mean(resid ** 2))):.1f} dB re 20 uPa overall")
    tries_bw = []
    for bw in (10.0, 20.0, 30.0, 50.0):  # the choice of the bandwidth: separation against following the envelope
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep_b = rota.extract.resolution_check(rpm, FS, list(cand), bandwidth=bw)
        ok_b = all(q["resolved"] for q in rep_b.pairs if problem in q["orders"])
        c_b = rota.extract.order_waveform(x, FS, rpm, [problem], bandwidth=bw)
        sim_b = rota.metrics.waveform_similarity(c_b.waveforms[s_ok, 0], stem[s_ok])
        tries_bw.append((bw, ok_b, sim_b["corr"]))
    log("  other bandwidths, order resolved from its neighbours / correlation with the true stem: "
        + "; ".join(f"{bw:g} Hz: {'yes' if ok_b else 'no'} / {corr:.4f}" for bw, ok_b, corr in tries_bw))
    predicted = float(np.ceil(max(e_t, e_p)))
    log(f"Reduction needed to reach the criterion: the larger of the TNR and PR excess, +{max(e_t, e_p):.1f} dB, "
        f"rounded up to {predicted:g} dB (truth: +{e_true:.1f} dB). What-if, each checked with the same analysis:")
    required, tries = predicted, []
    while True:
        y_req, _, _ = reduce_order(x, FS, rpm, problem, required, comp=comp, resid=resid)
        pr_req = prominence(y_req, FS, rpm, problem, BIN_PROM)
        left = max(max_of(pr_req["rpm"], pr_req["tnr_excess"])[0], max_of(pr_req["rpm"], pr_req["pr_excess"])[0])
        tries.append((required, left))
        log(f"  order {problem:g} reduced by {required:g} dB: largest remaining excess {left:+.1f} dB")
        if left < 0.0 or required >= predicted + 5.0:
            break
        required += 1.0
    variants = {"original": x, "required": y_req}
    y_round, _, _ = reduce_order(x, FS, rpm, problem, ROUND_REDUCTION_DB, comp=comp, resid=resid)
    variants["round"] = y_round
    reductions = {"required": required, "round": ROUND_REDUCTION_DB}
    for key, red in reductions.items():  # modify_orders + resynthesize: mix = x - (1 - gain) * order
        gain = 10.0 ** (-red / 20.0)
        assert np.allclose(variants[key], x + (gain - 1.0) * comp.total, atol=1e-9)
    R.update(resid=resid, comp=comp, stem=stem, variants=variants, reductions=reductions, required=required,
             predicted=predicted, tries=tries)

    log("Verification with the same analysis (order_prominence in 100 rpm bins, order slice, A-weighted level):")
    checks = {}
    for key, y in variants.items():
        pr = fine if key == "original" else prominence(y, FS, rpm, problem, BIN_PROM)
        e_t_y, _ = max_of(pr["rpm"], pr["tnr_excess"])
        e_p_y, _ = max_of(pr["rpm"], pr["pr_excess"])
        rng = prominent_range(pr["rpm"], pr["tnr_excess"])
        om_y = om if key == "original" else rota.maps.rpm_order_map(
            y, FS, rpm, order_resolution=ORDER_RES, max_order=MAX_ORDER_MAP, amplitude="rms", min_rpm=500.0)
        sl_y = rota.slices.order_slice(om_y, problem)
        d_slice = np.median(db(sl_y.amplitude[inw]) - db(sl.amplitude[inw]))
        ol_y = rota.slices.overall_level(y, FS, rpm, frame=FRAME_S, weighting="A", ref=P_REF, rpm_bins=EDGES)
        la = energy_mean_db(ol_y.level[(ol_y.rpm >= lo) & (ol_y.rpm <= hi)])
        checks[key] = {"prom": pr, "slice": sl_y, "ol": ol_y, "la_window": la, "slice_change": float(d_slice),
                       "excess": e_t_y, "excess_pr": e_p_y, "range": rng}
        rng_txt = "none" if not np.isfinite(rng[0]) else f"{rng[0]:.0f} to {rng[1]:.0f} rpm"
        log(f"  {key:9s}: order-{problem:g} slice {d_slice:+6.2f} dB in the range; TNR excess maximum "
            f"{e_t_y:+6.1f} dB (PR {e_p_y:+6.1f} dB), prominent range {rng_txt}; "
            f"L_A in the range {la:.2f} dB ({la - checks['original']['la_window'] if key != 'original' else 0.0:+.2f} dB)")
    R["checks"] = checks
    base_a = checks["original"]["ol"].binned_level
    for key in ("required", "round"):
        d_a = checks[key]["ol"].binned_level - base_a
        j = int(np.nanargmin(d_a))
        log(f"  {key:9s}: A-weighted level, largest change of the {BIN_RPM:g} rpm bins {d_a[j]:+.2f} dB at "
            f"{checks[key]['ol'].centres[j]:.0f} rpm (from {base_a[j]:.1f} dB)")
    return R


# ---- step 5: the toolboxes (optional) --------------------------------------------------------------------


def toolbox_metrics(R, log=print, parallel=True):
    """Step 5: loudness and sharpness (MoSQITo) and tonality (Sottek hearing model) along rpm, if available.

    The bridges take the three signals in Pa at ``FS`` and return ``TimeMetric`` results; ``vs_rpm``
    bins them along rpm. MoSQITo runs in worker processes (``parallel``) while the Sottek model runs
    here; the lines are logged when both are done, with the start-up exclusion and the threshold of the
    results. A toolbox that is not installed, or whose import fails, is skipped with a message: its
    bridge imports it at the first call and raises ImportError in both cases (with the install command
    and the original error), here or in a worker. Returns
    ``{"mosqito": {variant: binned}, "sottek": {variant: binned}}`` (empty if skipped); a binned entry
    of the Sottek model also holds the tonality result itself, as ``"ton"``.
    """
    sc, rpm, problem, variants = R["sc"], R["rpm"], R["problem"], R["variants"]
    out = {"mosqito": {}, "sottek": {}}
    futures, pool = {}, None
    if parallel:
        pool = ProcessPoolExecutor(max_workers=max(1, min(len(variants), os.cpu_count() or 1)))
        futures = {k: pool.submit(mosqito_metrics, v) for k, v in variants.items()}  # one process per variant
    tonality, seconds, unavailable = {}, {}, {}
    try:
        for k, v in variants.items():
            t0 = time.time()
            tonality[k] = bridges.sottek_hearing_model.tonality(v, FS)  # free-frontal field
            seconds[k] = time.time() - t0
    except ImportError as err:  # not installed, or its import fails
        unavailable["sottek"] = err
    try:
        raw = {k: f.result() for k, f in futures.items()} if parallel else {k: mosqito_metrics(v) for k, v in variants.items()}
    except ImportError as err:  # not installed, or its import fails (raised in a worker, result() raises it here)
        unavailable["mosqito"] = err
    finally:
        if pool is not None:
            pool.shutdown()
    if "mosqito" in unavailable:
        log("MoSQITo is not installed or cannot be imported: loudness and sharpness skipped")
        log(f"  {unavailable['mosqito']}")
    else:
        log(f"MoSQITo {importlib.metadata.version('mosqito')}: loudness_zwtv (ISO 532-1:2017, free field) and "
            f"sharpness_din_from_loudness (DIN 45692), {raw['original']['loudness'].skip_start:g} s start-up skipped")
        out["mosqito"] = {k: bin_mosqito(m, sc.t, rpm) for k, m in raw.items()}
    if "sottek" in unavailable:
        log("sottek-hearing-model is not installed or cannot be imported: ECMA-418-2 tonality skipped")
        log(f"  {unavailable['sottek']}")
    else:
        ton = tonality["original"]
        log("sottek-hearing-model: shm_tonality_ecma (ECMA-418-2, 4th edition, free-frontal field), "
            f"{ton.skip_start:g} s start-up skipped, frames above {ton.threshold:g} tu_HMS averaged")
        for k, s in seconds.items():
            log(f"  {k:9s} {s:5.1f} s")
        out["sottek"] = {k: {"ton": m, **bin_sottek(m, sc.t, rpm, problem)} for k, m in tonality.items()}
    return out


def window_mean(res, key, lo, hi, base=None):
    """Mean over the speed bins whose centre lies in [lo, hi] of a binned metric (minus ``base``, if given)."""
    c = res.centres
    v = res.value if base is None else res.value - base.value
    sel = (c >= lo) & (c <= hi) & np.isfinite(v)
    return float(np.mean(v[sel])) if sel.any() else float("nan")


def print_metric_tables(R, M, log=print):
    """Print the binned toolbox metrics around the prominent range, and the means over the range."""
    lo, hi = R["window"]
    keys = ("original", "required", "round")
    red = R["reductions"]
    names = {"original": "original", "required": f"-{red['required']:g} dB", "round": f"-{red['round']:g} dB"}
    if M["mosqito"]:
        log(f"MoSQITo, {BIN_RPM:g} rpm bins (loudness N5 = 95th percentile in sone, sharpness = median in acum; "
            f"d = variant - original):")
        log(f"  {'rpm':>5s} | {'N5':>6s} {'d ' + names['required']:>9s} {'d ' + names['round']:>9s} | "
            f"{'S':>6s} {'d ' + names['required']:>9s} {'d ' + names['round']:>9s}")
        base = M["mosqito"]["original"]
        for j, c in enumerate(base["loudness"].centres):
            if c < lo - 750.0 or c > hi + 750.0 or not np.isfinite(base["loudness"].value[j]):
                continue
            cells = []
            for name in ("loudness", "sharpness"):
                v0 = base[name].value[j]
                d = [M["mosqito"][k][name].value[j] - v0 for k in keys[1:]]
                cells.append(f"{v0:6.2f} {d[0]:+9.3f} {d[1]:+9.3f}")
            mark = "  <- in the range" if lo <= c <= hi else ""
            log(f"  {c:5.0f} | " + " | ".join(cells) + mark)
    if M["sottek"]:
        log(f"Sottek hearing model, {BIN_RPM:g} rpm bins (tonality in tu_HMS: mean of the frames above "
            f"{M['sottek']['original']['ton'].threshold:g}; order {R['problem']:g}: largest specific tonality of a "
            f"half band whose tonal frequency lies on the order, {100 * TONALITY_TOL:g} % window, 0 where there is none):")
        log(f"  {'rpm':>5s} | " + " ".join(f"{'tonality ' + names[k]:>17s}" for k in keys) + " | "
            + " ".join(f"{'order ' + format(R['problem'], 'g') + ' ' + names[k]:>17s}" for k in keys))
        base = M["sottek"]["original"]
        for j, c in enumerate(base["tonality"].centres):
            if c < lo - 750.0 or c > hi + 750.0:
                continue
            f_ = lambda v: f"{v:17.3f}" if np.isfinite(v) else f"{'-':>17s}"
            log(f"  {c:5.0f} | " + " ".join(f_(M["sottek"][k]["tonality"].value[j]) for k in keys) + " | "
                + " ".join(f_(M["sottek"][k]["order_tonality"].value[j]) for k in keys)
                + ("  <- in the range" if lo <= c <= hi else ""))


def summary(R, M):
    """Effects of the two reductions in the prominent range (means over the bins inside it)."""
    lo, hi = R["window"]
    out = {}
    for key in ("required", "round"):
        row = {"L_A": R["checks"][key]["la_window"] - R["checks"]["original"]["la_window"]}
        if M["mosqito"]:
            for name in ("loudness", "sharpness"):
                b, v = M["mosqito"]["original"][name], M["mosqito"][key][name]
                row[name] = window_mean(v, name, lo, hi, base=b)
                row[name + "_orig"] = window_mean(b, name, lo, hi)
        if M["sottek"]:
            for name in ("tonality", "order_tonality"):
                b, v = M["sottek"]["original"][name], M["sottek"][key][name]
                row[name] = window_mean(v, name, lo, hi, base=b)
                row[name + "_orig"] = window_mean(b, name, lo, hi)
        out[key] = row
    return out


# ---- step 6: conclusion -------------------------------------------------------------------------------------


def print_conclusion(R, S, log=print):
    """The conclusion as a short table, next to what was built in."""
    tr, p, red = R["tr"], R["problem"], R["reductions"]
    lo, hi = R["window"]
    lo6, hi6 = R["window6"]
    lo_t, hi_t = R["window_truth"]
    lo6_t, hi6_t = R["window6_truth"]

    def row(name, found, truth):
        log(f"  {name:36s} {found:46s} {truth}")

    log("== 6. Conclusion ==")
    row("", "found by the analysis", "built in / truth")
    row("problem order", f"{p:g}", f"{tr['order']:g} (gear whine; resonance at {tr['f_res']:g} Hz)")
    row("range where TNR >= criterion", f"{lo:.0f} to {hi:.0f} rpm ({p * lo / 60:.0f} to {p * hi / 60:.0f} Hz)",
        f"{lo_t:.0f} to {hi_t:.0f} rpm from the stems")
    row("range more than 6 dB above it", f"{lo6:.0f} to {hi6:.0f} rpm",
        f"{lo6_t:.0f} to {hi6_t:.0f} rpm from the stems")
    row("maximum TNR excess over criterion", f"+{R['max_excess']:.1f} dB at {R['max_excess_rpm']:.0f} rpm",
        f"+{R['max_excess_truth']:.1f} dB; resonance at {tr['rpm_res']:.0f} rpm (half-power band "
        f"{tr['rpm_band'][0]:.0f} to {tr['rpm_band'][1]:.0f} rpm)")
    row("maximum PR excess over criterion", f"+{R['max_excess_pr']:.1f} dB", "-")
    row("reduction needed", f"{R['required']:g} dB ({R['predicted']:g} dB from the excess, +1 dB checked)"
        if R["required"] > R["predicted"] else f"{R['required']:g} dB (from the excess, checked)",
        f"{np.ceil(R['max_excess_truth']):g} dB (excess rounded up)")
    for key in ("required", "round"):
        s, c = S[key], R["checks"][key]
        left = ("below the criterion" if c["excess"] < 0
                else f"still prominent, {c['range'][0]:.0f} to {c['range'][1]:.0f} rpm")
        log(f"  Order {p:g} reduced by {red[key]:g} dB: largest TNR excess left {c['excess']:+.1f} dB ({left}). "
            "Mean change in the range:")
        parts = [f"L_A {s['L_A']:+.2f} dB"]
        if "loudness" in s:
            parts.append(f"loudness N5 {s['loudness']:+.2f} sone ({100 * s['loudness'] / s['loudness_orig']:+.1f} %, "
                         f"from {s['loudness_orig']:.1f})")
            parts.append(f"sharpness {s['sharpness']:+.3f} acum (from {s['sharpness_orig']:.2f})")
        if "tonality" in s:
            parts.append(f"ECMA-418-2 tonality {s['tonality']:+.3f} tu_HMS (from {s['tonality_orig']:.3f}), "
                         f"order-{p:g} tonality {s['order_tonality']:+.3f} (from {s['order_tonality_orig']:.3f})")
        log("    " + "; ".join(parts))
    if not ("loudness" in S["required"] and "tonality" in S["required"]):
        log("  (loudness and sharpness, or tonality, are missing: the toolbox is not installed or cannot be imported)")


# ---- figures ------------------------------------------------------------------------------------------------


def _style(plt):
    plt.rcdefaults()  # Matplotlib's default style, whatever a local matplotlibrc sets
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8})


def _shade(ax, window, label=None):
    ax.axvspan(window[0], window[1], color="0.85", alpha=0.6, lw=0, label=label, zorder=0)


def figure_overview(R, path):
    """Figure 1: speed references, A-weighted level, band levels, order map (all against rpm)."""
    import matplotlib.pyplot as plt

    _style(plt)
    sc, om, bl, ol = R["sc"], R["om"], R["bl"], R["ol"]
    fig = plt.figure(figsize=(8.0, 10.4), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[0.95, 0.8, 1.0, 1.25])
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(sc.t, R["rpm"] / 1e3, color=BLUE, lw=1.6, label="tachometer (tacho_to_rpm)")
    m = R["ridge_mask"]
    ax.plot(sc.t[m], R["rpm_ridge"][m] / 1e3, color=ORANGE, lw=1.0, ls=(0, (3, 2)), label="tacholess (rpm_from_ridge)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (1000 rpm)")
    ax.set_title("(a) Speed from the tachometer and from the signal", loc="left")
    ax.legend(frameon=True, fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.plot(R["rpm"][m], 100.0 * R["ridge_dif"][m], color=ORANGE, lw=0.8)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel("Speed (rpm)")
    ax.set_ylabel("Tacholess minus tachometer (%)")
    ax.set_title(f"(b) Agreement above {RIDGE_FROM_RPM:.0f} rpm", loc="left")
    st = R["speed_stats"]
    ax.text(0.97, 0.05, f"median |difference| {100 * st['ridge_median']:.2f} %\n95th percentile {100 * st['ridge_p95']:.2f} %",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(ol.rpm / 1e3, ol.level, ".", color="0.6", ms=1.5, label=f"frames of {FRAME_S * 1e3:.0f} ms")
    ok = ol.counts > 0
    ax.plot(ol.centres[ok] / 1e3, ol.binned_level[ok], "o-", color=BLUE, ms=2.5, lw=1.2,
            label=f"energy mean per {BIN_RPM:g} rpm bin")
    j = int(np.nanargmax(np.abs(R["resid_a"])))
    ax.annotate(f"{R['resid_a'][j]:+.1f} dB above\nthe smooth trend", (ol.centres[j] / 1e3, ol.binned_level[j]),
                xytext=(ol.centres[j] / 1e3 - 2.6, ol.binned_level[j] + 3.0), fontsize=7,
                arrowprops={"arrowstyle": "->", "lw": 0.6})
    ax.set_xlim(EDGES[0] / 1e3, EDGES[-1] / 1e3)
    ax.set_ylim(40.0, 80.0)
    ax.set_ylabel("A-weighted level (dB re 20 µPa)")
    ax.set_title("(c) Overall A-weighted level (RMS)", loc="left")
    ax.legend(frameon=True, fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    lv = bl.levels
    im = ax.pcolormesh(bl.rpm_centres / 1e3, np.arange(lv.shape[1]), lv.T, cmap="viridis", shading="nearest",
                       vmin=np.floor(np.nanpercentile(lv, 3) / 5) * 5, vmax=np.ceil(np.nanmax(lv) / 5) * 5, rasterized=True)
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04, label="dB re 20 µPa (RMS)")
    ticks = np.arange(0, lv.shape[1], 3)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{bl.nominal[i]:g}" for i in ticks])
    ax.set_ylabel("1/3-octave band (Hz)")
    ax.set_title("(d) Band levels against speed (band_levels_vs_rpm)", loc="left")
    ib, jb = np.unravel_index(np.nanargmax(R["bump"]), R["bump"].shape)
    ax.plot(bl.rpm_centres[ib] / 1e3, jb, "o", mfc="none", mec="w", ms=11, mew=1.2)
    ax.text(bl.rpm_centres[ib] / 1e3 + 0.35, jb - 1.0, f"{bl.nominal[jb]:g} Hz band, +{R['bump'][ib, jb]:.0f} dB",
            color="w", fontsize=7, va="center")
    ax.set_xlim(EDGES[0] / 1e3, EDGES[-1] / 1e3)

    ax = fig.add_subplot(gs[3, :])
    d = db(om.magnitude)
    vmax = np.ceil(np.percentile(d[om.axis > 6.0], 99.95) / 5.0) * 5.0  # the lines above order 6 set the scale
    im = ax.pcolormesh(om.rpm / 1e3, om.axis, d, cmap="viridis", vmin=vmax - 35.0, vmax=vmax, shading="nearest",
                       rasterized=True)
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04, label=f"dB re 20 µPa (RMS), {ORDER_RES:g} order bins")
    ax.set_xlim(EDGES[0] / 1e3, EDGES[-1] / 1e3)
    ax.set_ylim(0.0, MAX_ORDER_MAP)
    ax.set_facecolor(plt.get_cmap("viridis")(0.0))
    ax.set_xlabel("Speed (1000 rpm)")
    ax.set_ylabel("Order (cycles per revolution)")
    ax.set_title("(e) Order map (rpm_order_map), candidates found in its average spectrum", loc="left")
    for o in R["cand"]:
        ax.text(1.004, o, f"{o:g}", transform=ax.get_yaxis_transform(), fontsize=7, va="center", clip_on=False)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def figure_diagnosis(R, path):
    """Figure 2: order slices, TNR and PR of the problem order, screening of the candidates."""
    import matplotlib.pyplot as plt

    _style(plt)
    sc, tr, p, fine, win = R["sc"], R["tr"], R["problem"], R["fine"], R["window"]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.4), dpi=150, constrained_layout=True)
    pal = {8.0: "#999999", 10.0: "#CC79A7", 12.0: GREEN, 14.0: ORANGE, p: BLUE}
    ax = axes[0, 0]
    for o, s in R["screen"].items():
        sl = s["slice"]
        ax.plot(sl.rpm / 1e3, db(sl.amplitude), color=pal.get(o, "k"), lw=1.6 if o == p else 0.9, label=f"order {o:g}")
    truth = sc.quasi_static_amplitude(tr["k"]) / np.sqrt(2.0)
    ax.plot(sc.rpm_mean / 1e3, db(truth), color="k", lw=0.8, ls=(0, (4, 3)), label=f"truth, order {p:g}")
    ax.axvspan(win[0] / 1e3, win[1] / 1e3, color="0.85", alpha=0.6, lw=0, zorder=0)
    ax.set_xlim(2.0, 11.0)
    ax.set_ylim(15.0, 70.0)
    ax.set_xlabel("Speed (1000 rpm)")
    ax.set_ylabel("Order level (dB re 20 µPa, RMS)")
    ax.set_title("(a) Order slices of the candidates", loc="left")
    ax.legend(frameon=True, fontsize=6.5, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)

    r = fine["rpm"] / 1e3
    ax = axes[0, 1]
    _shade(ax, np.array(win) / 1e3, "range above the criterion")
    ax.plot(r, fine["tnr"], "o-", color=BLUE, ms=2.5, lw=1.2, label="TNR, order_prominence")
    ax.plot(r, R["tnr_true"], color="k", lw=0.8, ls=(0, (4, 3)), label="TNR from the stems (truth)")
    ax.plot(r, fine["tnr_crit"], color=RED, lw=1.0, label="ECMA-418-1 criterion")
    ax.axvline(tr["rpm_res"] / 1e3, color=GREY, lw=0.6, ls=":")
    ax.text(tr["rpm_res"] / 1e3 + 0.08, -12.0, f"built-in resonance\n{tr['f_res']:g} Hz", fontsize=6.5, color=GREY, va="bottom")
    ax.set_xlim(2.0, 8.0)
    ax.set_ylim(-15.0, 28.0)
    ax.set_xlabel("Speed (1000 rpm)")
    ax.set_ylabel("Tone-to-noise ratio (dB)")
    ax.set_title(f"(b) TNR of order {p:g}", loc="left")
    ax.legend(frameon=True, fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    _shade(ax, np.array(win) / 1e3)
    ax.plot([], [], color="0.85", lw=6, label="range above the TNR criterion")
    ax.plot(r, fine["pr"], "o-", color=BLUE, ms=2.5, lw=1.2, label="PR, order_prominence")
    ax.plot(r, fine["pr_crit"], color=RED, lw=1.0, label="ECMA-418-1 criterion")
    ax.set_xlim(2.0, 8.0)
    ax.set_ylim(-5.0, 22.0)
    ax.set_xlabel("Speed (1000 rpm)")
    ax.set_ylabel("Prominence ratio (dB)")
    ax.set_title(f"(c) PR of order {p:g}", loc="left")
    ax.legend(frameon=True, fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axhline(0.0, color=RED, lw=1.0)
    for o, s in R["screen"].items():
        pr = s["pr"]
        ax.plot(pr["rpm"] / 1e3, pr["tnr_excess"], "o-" if o == p else "-", color=pal.get(o, "k"),
                ms=2.5, lw=1.4 if o == p else 0.9, label=f"order {o:g}")
    ax.set_xlim(2.0, 11.0)
    ax.set_ylim(-30.0, 18.0)
    ax.set_xlabel("Speed (1000 rpm)")
    ax.set_ylabel("TNR minus criterion (dB)")
    ax.set_title("(d) The candidates against their criteria", loc="left")
    ax.legend(frameon=True, fontsize=6.5, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def figure_whatif(R, M, S, path):
    """Figure 3: the what-if. ROTA's check of the reduced signals and the toolbox metrics along rpm."""
    import matplotlib.pyplot as plt

    _style(plt)
    p, win, red, checks = R["problem"], R["window"], R["reductions"], R["checks"]
    keys = ("original", "required", "round")
    label = {"original": "original", "required": f"order {p:g} -{red['required']:g} dB",
             "round": f"order {p:g} -{red['round']:g} dB"}
    col = {"original": BLUE, "required": ORANGE, "round": GREEN}
    fig, axes = plt.subplots(3, 2, figsize=(8.0, 9.4), dpi=150, constrained_layout=True)
    w = np.array(win) / 1e3

    def skipped(ax, what):
        ax.text(0.5, 0.5, f"skipped: {what} not available", ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color=GREY)

    ax = axes[0, 0]
    _shade(ax, w)
    for k in keys:
        sl = checks[k]["slice"]
        ax.plot(sl.rpm / 1e3, db(sl.amplitude), color=col[k], lw=1.2, label=label[k])
    ax.set_xlim(2.0, 8.0)
    ax.set_ylim(25.0, 65.0)
    ax.set_xlabel("Speed (1000 rpm)")
    ax.set_ylabel("Order level (dB re 20 µPa, RMS)")
    ax.set_title(f"(a) Order {p:g} slice (ROTA)", loc="left")
    ax.legend(frameon=True, fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    _shade(ax, w)
    ax.axhline(0.0, color=RED, lw=1.0, label="criterion")
    for k in keys:
        pr = checks[k]["prom"]
        ax.plot(pr["rpm"] / 1e3, pr["tnr_excess"], color=col[k], lw=1.2, label=label[k])
    ax.set_xlim(2.0, 8.0)
    ax.set_ylim(-32.0, 18.0)
    ax.set_xlabel("Speed (1000 rpm)")
    ax.set_ylabel("TNR minus criterion (dB)")
    ax.set_title("(b) Tone prominence after the change (ROTA)", loc="left")
    ax.legend(frameon=True, fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    _shade(ax, w)
    ax.axhline(0.0, color="0.4", lw=0.6)
    base = checks["original"]["ol"].binned_level
    for k in keys[1:]:
        ax.plot(checks[k]["ol"].centres / 1e3, checks[k]["ol"].binned_level - base, "o-", color=col[k], ms=2.5, lw=1.2,
                label=label[k])
    ax.set_xlim(2.0, 8.0)
    ax.set_xlabel("Speed (1000 rpm)")
    ax.set_ylabel("A-weighted level change (dB)")
    ax.set_title("(c) Overall level (ROTA, overall_level)", loc="left")
    ax.legend(frameon=True, fontsize=6.5, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if M["mosqito"]:
        _shade(ax, w)
        ax.axhline(0.0, color="0.4", lw=0.6)
        b = M["mosqito"]["original"]["loudness"]
        for k in keys[1:]:
            v = M["mosqito"][k]["loudness"]
            ax.plot(v.centres / 1e3, v.value - b.value, "o-", color=col[k], ms=2.5, lw=1.2, label=label[k])
        ax.legend(frameon=True, fontsize=6.5, loc="lower right")
    else:
        skipped(ax, "MoSQITo")
    ax.set_xlim(2.0, 8.0)
    ax.set_xlabel("Speed (1000 rpm)")
    ax.set_ylabel("Loudness N5 change (sone)")
    ax.set_title("(d) Loudness, ISO 532-1 (MoSQITo)", loc="left")
    ax.grid(alpha=0.3)

    ax = axes[2, 0]
    if M["mosqito"]:
        _shade(ax, w)
        ax.axhline(0.0, color="0.4", lw=0.6)
        b = M["mosqito"]["original"]["sharpness"]
        for k in keys[1:]:
            v = M["mosqito"][k]["sharpness"]
            ax.plot(v.centres / 1e3, v.value - b.value, "o-", color=col[k], ms=2.5, lw=1.2, label=label[k])
        ax.legend(frameon=True, fontsize=6.5, loc="lower right")
    else:
        skipped(ax, "MoSQITo")
    ax.set_xlim(2.0, 8.0)
    ax.set_xlabel("Speed (1000 rpm)")
    ax.set_ylabel("Sharpness change (acum)")
    ax.set_title("(e) Sharpness, DIN 45692 (MoSQITo)", loc="left")
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    if M["sottek"]:
        _shade(ax, w)
        for k in keys:
            t_ = M["sottek"][k]["tonality"]
            o_ = M["sottek"][k]["order_tonality"]
            ax.plot(t_.centres / 1e3, t_.value, "o-", color=col[k], ms=2.5, lw=1.2, label=f"overall, {label[k]}")
            ax.plot(o_.centres / 1e3, o_.value, color=col[k], lw=0.9, ls=(0, (3, 2)))
        ax.plot([], [], color="k", lw=0.9, ls=(0, (3, 2)), label=f"order {p:g} only (dashed)")
        ax.legend(frameon=True, fontsize=6.0, loc="upper right")
    else:
        skipped(ax, "sottek-hearing-model")
    ax.set_xlim(2.0, 8.0)
    ax.set_xlabel("Speed (1000 rpm)")
    ax.set_ylabel("Tonality (tu$_{HMS}$)")
    ax.set_title("(f) Tonality, ECMA-418-2 (Sottek model)", loc="left")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


# ---- main ---------------------------------------------------------------------------------------------------------


def main():
    t_start = time.time()
    R = analyse()
    print("\n== 5. Psychoacoustics (optional toolboxes, called only through their documented functions) ==")
    print("ROTA itself computes none of these metrics. Editions: ROTA's prominence follows ECMA-418-1 (3rd edition, 2024); "
          "the Sottek model implements ECMA-418-2 (4th edition, 2025); more implementations: "
          "https://zhenxianli.github.io/LISQM/")
    M = toolbox_metrics(R)
    print_metric_tables(R, M)
    S = summary(R, M)
    print()
    print_conclusion(R, S)

    print(f"\nWAV files (16-bit PCM, {int(FS_TOOL)} Hz, mono, full scale = {FULL_SCALE_PA:g} Pa):")
    red = R["reductions"]
    write_wav("worked_original.wav", rota.tonal.to_calibrated_48k(R["variants"]["original"], FS, fs_out=FS_TOOL), FS_TOOL)
    write_wav(f"worked_order{R['problem']:g}_minus{red['required']:g}db.wav",
              rota.tonal.to_calibrated_48k(R["variants"]["required"], FS, fs_out=FS_TOOL), FS_TOOL)
    write_wav(f"worked_order{R['problem']:g}_minus{red['round']:g}db.wav",
              rota.tonal.to_calibrated_48k(R["variants"]["round"], FS, fs_out=FS_TOOL), FS_TOOL)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")  # file output only; no display needed
    for name, fn, args in (("10_worked_overview.png", figure_overview, ()),
                           ("10_worked_diagnosis.png", figure_diagnosis, ()),
                           ("10_worked_whatif.png", figure_whatif, (M, S))):
        path = FIG_DIR / name
        fn(R, *args, path)
        print(f"Figure written to {path.relative_to(ROOT)}")
    print(f"Total run time {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
