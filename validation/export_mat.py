"""Export validation cases to MATLAB v5 files for the black-box reference run.

ROADMAP section 5.5, step 2: the external implementation is run on exactly the
signals that ROTA is validated on, through public functions with every
parameter given explicitly (section 5.1). This script writes one
``validation/signals/<case>.mat`` per case (the directory is git-ignored):

    python validation/export_mat.py             # T1 T2 T3 T5 T6 T14 VKBW
    python validation/export_mat.py T1 T14      # selected cases
    python validation/export_mat.py --out DIR   # another output directory

Cases (``validation/make_signals.py``): T1, T2, T3, T5 (60 pulses per
revolution), T6 and T14; and VKBW, a probe of the Vold-Kalman -3 dB
bandwidth built here with ``rota.synth.synth_runup`` (constant speed, one
order and fixed tones at known offsets from it, see :func:`vk_probe`).

Variables in each file (units: time s, frequency Hz, speed rpm, order in
cycles per revolution; amplitudes are PEAK values in Pa, as in the truth of
``rota.synth``, unless a field says otherwise):

``x``                 signal, column (n, 1)
``fs``                sampling rate, Hz
``t``                 sample times, column (n, 1)
``rpm``               exact speed that built the signal, column (n, 1)
``tacho``             tachometer pulses, column (n, 1); only in T5
``orders``            order numbers, row (1, K)
``order_amplitude``   truth: peak amplitude of every order at every sample (n, K)
``order_stems``       truth: every order as rendered, ``order_amplitude * cos(phase)`` (n, K)
``par``               struct of analysis parameters (see :func:`analysis_parameters`)

``par`` carries the settings that both sides must use, so that the two
implementations estimate the same physical quantity: order resolution,
frequency resolution, window, overlap, Vold-Kalman bandwidth and decoupling,
the orders to track, the tachometer settings and the initial ridge point of
the tacholess tracker. ``validation/compare.py`` reads the same table
(:func:`analysis_parameters`), so the parameters cannot drift between the
MATLAB script and the ROTA side. Flags ``par.do_*`` (0 or 1) say which
analyses apply to a case.

References
----------
ROTA ROADMAP, sections 5.1 (principles), 5.3 (test signals) and 5.5
(directory and workflow).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.signal import get_window

from rota.synth import synth_runup

__all__ = ["CASE_NAMES", "SIGNALS_DIR", "VK_PROBE", "analysis_parameters", "build_case", "main", "vk_probe"]

HERE = Path(__file__).resolve().parent
SIGNALS_DIR = HERE / "signals"

# Cases of the black-box comparison. All but VKBW are keys of make_signals.CASES; T5 is the
# 60-pulse-per-revolution variant (the default of make_signals.t5). VKBW is the bandwidth probe.
CASE_NAMES = ("T1", "T2", "T3", "T5", "T6", "T14", "VKBW")

# Vold-Kalman bandwidth probe (ROADMAP 5.4, "Vold-Kalman -3 dB bandwidth"): constant speed, so
# that the carrier of the order is a fixed line, and unit tones at known offsets (Hz) from it.
# The filters are linear: the gain at each offset is read off a least-squares fit of the output
# on the known frequencies (compare.py). Offsets are dense around the set -3 dB offset (2 Hz).
# The sampling rate is low (fs / bandwidth = 100) so that iterative Vold-Kalman solvers converge
# quickly on this probe.
VK_PROBE = {
    "fs": 200.0, "duration": 30.0, "rpm": 1500.0, "order": 1.0, "bandwidth": 2.0, "seed": 21,
    "offsets": (-5.0, -4.0, -3.0, -2.5, -2.25, -2.0, -1.75, -1.5, -1.0, -0.5,
                0.5, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0, 5.0),
}

# Meaning of the bandwidth arguments. ROTA's ``bandwidth`` (``par.vk_bandwidth``) is the offset
# from the carrier at which the single-order filter attenuates a tone by 3 dB; its pass band is
# twice that wide. MATLAB's ``Bandwidth`` is taken as the full width of the pass band between the
# two -3 dB points, so it is given ``par.vk_bandwidth_matlab = 2 * par.vk_bandwidth``. The probe
# VKBW measures both (compare.py, table 10), with this value and with the same number as ROTA.
MATLAB_BANDWIDTH_FACTOR = 2.0

# Meaning of the resolution arguments of the maps. ROTA's resolution (``par.order_resolution``,
# ``par.freq_resolution``) is the bin spacing, one over the window length. MATLAB's ``res`` is
# the resolution bandwidth, the equivalent noise bandwidth of the window times the bin spacing
# For the same window length on both sides MATLAB
# is given ``res = ENBW * resolution`` (``par.order_resolution_matlab``,
# ``par.freq_resolution_matlab``), with the ENBW of the window in bins (Hann 1.5, flat top 3.77).
MATLAB_TO_SCIPY_WINDOW = {"hann": "hann", "flattopwin": "flattop"}


def window_enbw(matlab_name, n=8192):
    """Equivalent noise bandwidth of a window in bins, ``n sum(w^2) / (sum w)^2``.

    Parameters
    ----------
    matlab_name : str
        ``"hann"`` or ``"flattopwin"``; evaluated with the periodic SciPy window of the same
        definition (``hann``, ``flattop``).
    n : int, default 8192
        Window length used for the evaluation (the value converges for long windows).

    Returns
    -------
    float
        ENBW in bins (Hann 1.5, flat top about 3.77).

    References
    ----------
    Harris, F. J. (1978). On the use of windows for harmonic analysis with the discrete Fourier
    transform. Proceedings of the IEEE, 66(1), 51-83.
    """
    w = get_window(MATLAB_TO_SCIPY_WINDOW[matlab_name], n, fftbins=True)
    return float(n * np.sum(w**2) / np.sum(w) ** 2)

# Settings shared by all cases.
#   window_order_map    Hann: the main lobe is two bins wide each side, so orders half an
#                       order apart are separated at the resolutions below (T14);
#                       integer and half-integer orders fall on bin centres.
#   window_freq_map     flat top: the peak reading of a line is free of scalloping loss,
#                       so no assumption on the window length of the other side is needed.
#   overlap_percent     75 %, amplitude RMS (a unit sine of peak A reads A / sqrt(2)).
#   vk_filter_order     MATLAB 'FilterOrder' whose response matches the second-difference
#                       structure of rota.extract.order_waveform, |H| = 1 / (1 + c d^4) at an
#                       offset d from the carrier; the probe VKBW checks the response law.
_COMMON = {
    "window_order_map": "hann",
    "window_freq_map": "flattopwin",
    "overlap_percent": 75.0,
    "amplitude_map": "rms",
    "vk_filter_order": 1.0,
    "vk_amplitude": "peak",
}

# Per-case settings. A frequency resolution of 0 means "no frequency map for this case":
# a frequency map resolves only lines that are further apart than the flat-top main lobe
# (about 10 bins), which holds for the single order of T1 only.
#   order_resolution     bin spacing of the order map, orders (window of 1 / resolution revolutions)
#   freq_resolution      bin spacing / resolution bandwidth of the frequency map, Hz
#   vk_bandwidth         Vold-Kalman -3 dB bandwidth, Hz (pass band about twice that)
#   rpmtrack_order       order whose ridge the tacholess trackers follow (0: not run)
#   rpmtrack_resolution  frequency resolution of the tacholess ridge map of the other side, Hz
#   rpm_range            search range of rota.speed.rpm_from_ridge, rpm
_SPECS = {
    "T1": {
        "order_resolution": 0.25, "freq_resolution": 8.0, "vk_bandwidth": 2.0,
        "rpmtrack_order": 2.0, "rpmtrack_resolution": 4.0, "rpm_range": (800.0, 4500.0),
    },
    "T2": {
        "order_resolution": 0.25, "freq_resolution": 0.0, "vk_bandwidth": 2.0,
        "rpmtrack_order": 2.0, "rpmtrack_resolution": 4.0, "rpm_range": (800.0, 5500.0),
    },
    "T3": {
        "order_resolution": 0.25, "freq_resolution": 0.0, "vk_bandwidth": 2.0,
        "rpmtrack_order": 2.0, "rpmtrack_resolution": 6.0, "rpm_range": (800.0, 4500.0),
    },
    "T5": {
        "order_resolution": 0.0, "freq_resolution": 0.0, "vk_bandwidth": 0.0,
        "rpmtrack_order": 0.0, "rpmtrack_resolution": 0.0, "rpm_range": (0.0, 0.0),
    },
    # 1 s run-up: the tacholess estimators need windows of a few tenths of a second, so none here.
    "T6": {
        "order_resolution": 0.25, "freq_resolution": 0.0, "vk_bandwidth": 16.0,
        "rpmtrack_order": 0.0, "rpmtrack_resolution": 0.0, "rpm_range": (0.0, 0.0),
    },
    "T14": {
        "order_resolution": 0.125, "freq_resolution": 0.0, "vk_bandwidth": 2.0,
        "rpmtrack_order": 1.0, "rpmtrack_resolution": 3.0, "rpm_range": (600.0, 3500.0),
    },
    # Bandwidth probe only: none of the comparisons above.
    "VKBW": {
        "order_resolution": 0.0, "freq_resolution": 0.0, "vk_bandwidth": VK_PROBE["bandwidth"],
        "rpmtrack_order": 0.0, "rpmtrack_resolution": 0.0, "rpm_range": (0.0, 0.0), "vk_probe": True,
    },
}

# Tachometer settings of T5: 60 pulses per revolution, output of the speed curve at 1 kHz,
# 60 spline break points over the 15 s record (one per 0.25 s).
_TACHO = {"ppr": 60.0, "output_fs": 1000.0, "fit_points": 60.0}


def _load_module(name):
    """Import ``validation/<name>.py`` by path (the folder is not a package)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def vk_probe():
    """Probe of the Vold-Kalman -3 dB bandwidth (case VKBW).

    Constant speed ``VK_PROBE["rpm"]`` (rpm), one order ``VK_PROBE["order"]`` of peak
    amplitude 1 Pa (the carrier, 25 Hz) and one fixed tone of peak amplitude 1 Pa at every
    offset of ``VK_PROBE["offsets"]`` (Hz) from the carrier, random phases from the seed; no
    noise. A linear filter tracking the order passes each tone with its gain at that offset,
    which ``compare.py`` reads off a least-squares fit on the known frequencies.

    Returns
    -------
    rota.synth.SynthSignal
        Signal and analytic truth. Amplitudes are peak, in Pa.

    References
    ----------
    ROTA ROADMAP, section 5.4 (Vold-Kalman -3 dB bandwidth).
    Vold, H., Mains, M., & Blough, J. (1997). Theoretical foundations for high performance
    order tracking with the Vold-Kalman tracking filter. SAE Technical Paper 972007.
    """
    p = VK_PROBE
    n = int(round(p["duration"] * p["fs"]))
    f0 = p["order"] * p["rpm"] / 60.0
    tones = [{"freq": f0 + d, "amplitude": 1.0} for d in p["offsets"]]
    return synth_runup(p["fs"], np.full(n, p["rpm"]), [{"order": p["order"], "amplitude": 1.0}],
                       tones=tones, seed=p["seed"])


def build_case(name):
    """Build one case with ``make_signals`` (VKBW: :func:`vk_probe`).

    Parameters
    ----------
    name : str
        One of :data:`CASE_NAMES`.

    Returns
    -------
    rota.synth.SynthSignal
        Signal and analytic truth. Amplitudes are peak, in Pa (noise: RMS).

    References
    ----------
    ROTA ROADMAP, section 5.3.
    """
    if name not in CASE_NAMES:
        raise KeyError(f"unknown case {name!r}; available: {list(CASE_NAMES)}")
    if name == "VKBW":
        return vk_probe()
    return _load_module("make_signals").CASES[name]()


def _row(values):
    return np.asarray(values, dtype=float).reshape(1, -1)


def analysis_parameters(name, sig):
    """Analysis parameters that the MATLAB script and ``compare.py`` both use.

    Every value is a float, a float row vector or a string, so that
    ``scipy.io.savemat`` writes it as a MATLAB double, row vector or char array.

    Parameters
    ----------
    name : str
        Case name, one of :data:`CASE_NAMES`.
    sig : rota.synth.SynthSignal
        The case signal (used for the tachometer levels and the initial point of the
        tacholess ridge, which is read off the exact speed).

    Returns
    -------
    dict
        Keys: ``order_resolution`` (orders), ``freq_resolution`` (Hz; both bin spacings),
        ``order_resolution_matlab``, ``freq_resolution_matlab`` (MATLAB ``res``: window ENBW
        times the bin spacing),
        ``window_order_map``, ``window_freq_map`` (MATLAB window names),
        ``overlap_percent``, ``amplitude_map`` (``"rms"``), ``vk_bandwidth`` (Hz, ROTA:
        -3 dB offset from the carrier), ``vk_bandwidth_matlab`` (Hz, MATLAB ``Bandwidth``:
        full -3 dB width, twice ``vk_bandwidth``), ``vk_filter_order``, ``vk_decouple``
        (0 or 1), ``vk_amplitude`` (``"peak"``), ``probe_order`` and ``probe_offsets``
        (row, Hz; VKBW only, else 0 and empty),
        ``track_orders`` (row), ``rpmtrack_order``, ``rpmtrack_resolution`` (Hz),
        ``rpmtrack_t0`` (s), ``rpmtrack_f0`` (Hz, ``order * rpm / 60`` of the exact
        speed at ``t0``), ``rpm_range`` (row, rpm; ROTA search range), ``ppr``,
        ``state_levels`` (row, low and high level of the tachometer), ``tacho_output_fs``
        (Hz), ``fit_points``, and the flags ``do_tachorpm``, ``do_maps``, ``do_freqmap``,
        ``do_vk``, ``do_rpmtrack``, ``do_vkprobe``. Orders are cycles per revolution; speed
        in rpm.

    References
    ----------
    ROTA ROADMAP, sections 5.1 and 5.4.
    """
    if name not in _SPECS:
        raise KeyError(f"unknown case {name!r}; available: {list(CASE_NAMES)}")
    spec = _SPECS[name]
    orders = np.asarray(sig.orders, dtype=float)
    has_tacho = sig.tacho is not None
    probe = bool(spec.get("vk_probe", False))
    par = dict(_COMMON)
    par.update(
        order_resolution=float(spec["order_resolution"]),
        freq_resolution=float(spec["freq_resolution"]),
        order_resolution_matlab=float(window_enbw(_COMMON["window_order_map"]) * spec["order_resolution"]),
        freq_resolution_matlab=float(window_enbw(_COMMON["window_freq_map"]) * spec["freq_resolution"]),
        vk_bandwidth=float(spec["vk_bandwidth"]),
        vk_bandwidth_matlab=float(MATLAB_BANDWIDTH_FACTOR * spec["vk_bandwidth"]),
        vk_decouple=float(orders.size > 1),
        probe_order=float(VK_PROBE["order"]) if probe else 0.0,
        probe_offsets=_row(VK_PROBE["offsets"] if probe else []),
        track_orders=_row(orders),
        rpmtrack_order=float(spec["rpmtrack_order"]),
        rpmtrack_resolution=float(spec["rpmtrack_resolution"]),
        rpm_range=_row(spec["rpm_range"]),
        ppr=_TACHO["ppr"] if has_tacho else 0.0,
        tacho_output_fs=_TACHO["output_fs"] if has_tacho else 0.0,
        fit_points=_TACHO["fit_points"] if has_tacho else 0.0,
        state_levels=_row([sig.tacho.min(), sig.tacho.max()] if has_tacho else [0.0, 0.0]),
        do_tachorpm=float(has_tacho),
        do_maps=float(spec["order_resolution"] > 0),
        do_freqmap=float(spec["freq_resolution"] > 0),
        do_vk=float(spec["vk_bandwidth"] > 0 and not probe),
        do_rpmtrack=float(spec["rpmtrack_order"] > 0),
        do_vkprobe=float(probe),
    )
    # Initial ridge point of the tacholess tracker: the middle sample, on the ridge of the
    # tracked order, taken from the exact speed.
    i0 = sig.t.size // 2
    par["rpmtrack_t0"] = float(sig.t[i0])
    par["rpmtrack_f0"] = float(spec["rpmtrack_order"] * np.asarray(sig.rpm, dtype=float)[i0] / 60.0)
    return par


def _to_mat(name, sig, par):
    """Dictionary for ``savemat``: column vectors, a row of orders and the ``par`` struct."""
    def col(a):
        return np.asarray(a, dtype=float).reshape(-1, 1)

    out = {
        "case_name": name,
        "source": f"validation/make_signals.py, CASES[{name!r}]",
        "x": col(sig.x),
        "fs": np.float64(sig.fs),
        "t": col(sig.t),
        "rpm": col(sig.rpm),
        "orders": _row(sig.orders),
        "order_amplitude": np.asarray(sig.order_amplitude, dtype=float),
        "order_stems": np.asarray(sig.order_stems, dtype=float),
        "par": par,
    }
    if sig.tacho is not None:
        out["tacho"] = col(sig.tacho)
        out["pulse_times"] = col(sig.pulse_times)  # truth: rendered pulses (dropped ones absent), s
    return out


def main(outdir=None, names=None):
    """Write ``<outdir>/<case>.mat`` (MATLAB v5, compressed) for each case.

    Parameters
    ----------
    outdir : path-like, optional
        Output directory (created if needed). Default: ``validation/signals``.
    names : sequence of str, optional
        Cases to write. Default: all of :data:`CASE_NAMES`.

    Returns
    -------
    dict
        ``{case: path}`` of the files written.

    References
    ----------
    ROTA ROADMAP, section 5.5.
    """
    outdir = SIGNALS_DIR if outdir is None else Path(outdir)
    names = list(CASE_NAMES) if not names else list(names)
    unknown = [n for n in names if n not in CASE_NAMES]
    if unknown:
        raise KeyError(f"unknown case(s) {unknown}; available: {list(CASE_NAMES)}")
    outdir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in names:
        sig = build_case(name)
        par = analysis_parameters(name, sig)
        path = outdir / f"{name}.mat"
        savemat(path, _to_mat(name, sig, par), format="5", do_compression=True, oned_as="column")
        written[name] = path
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases", nargs="*", help=f"cases to write (default: {list(CASE_NAMES)})")
    ap.add_argument("--out", default=None, help="output directory (default: validation/signals)")
    args = ap.parse_args()
    bad = [c for c in args.cases if c not in CASE_NAMES]
    if bad:
        ap.error(f"unknown case(s) {bad}; available: {list(CASE_NAMES)}")
    for case, path in main(args.out, args.cases).items():
        print(f"{case}: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
