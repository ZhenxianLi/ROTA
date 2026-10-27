"""Compare ROTA, the analytic truth and the black-box MATLAB reference (ROADMAP 5.1, 5.2, 5.4).

For every case written by ``validation/export_mat.py`` the script computes the ROTA result
with the parameters stored in the signal file (``par``), reads the MATLAB output
``validation/reference/<case>_ref.mat`` if it exists, and writes a markdown report with one
table per comparison. Each row gives the deviation of ROTA from the truth, of MATLAB from the
truth, of ROTA from MATLAB, the tolerance of ROADMAP section 5.4 and a verdict::

    python validation/compare.py --out validation/reference/report.md
    python validation/compare.py --cases T1 T5 --out report.md

Without MATLAB output the script still runs: the MATLAB cells then read "MATLAB output
missing" and only the ROTA-truth criterion decides. Analytic truth is the first standard
(ROADMAP 5.1): when ROTA and MATLAB disagree, the row says whether MATLAB is also away from
the truth.

What is compared (physical quantities, never map cells; units: speed rpm, frequency Hz,
orders in cycles per revolution, amplitude in dB re the truth; map amplitudes are RMS, the
truth being the peak amplitude over sqrt(2); Vold-Kalman amplitudes are peak):

* speed curves: ``tacho_to_rpm`` vs ``tachorpm`` (T5) and ``rpm_from_ridge`` vs ``rpmtrack``,
  interpolated onto a common time grid, away from the record ends;
* amplitude of an order along its ridge, read from the order map (``rpm_order_map`` +
  ``order_slice`` vs ``rpmordermap`` and ``ordertrack`` of the map) and from the frequency map
  (``rpm_frequency_map`` vs ``rpmfreqmap``): on interior frames (3 frames dropped at each end)
  where the truth amplitude varies by at most 2 % over the frame and the line does not sweep
  across the resolution; the maps are interpolated in time onto the ROTA frames;
* the average order spectrum: peak order, and peak amplitude where the truth amplitude is
  constant;
* Vold-Kalman: correlation and least-squares gain of the order waveforms
  (``order_waveform`` vs ``orderwaveform``) and the envelope amplitude
  (``order_waveform`` vs ``ordertrack`` in its Vold-Kalman form), away from the record ends
  (3 / bandwidth seconds);
* the -3 dB bandwidth of the Vold-Kalman filter on the probe signal VKBW (constant speed, one
  order and tones at known offsets): the offset at which each filter attenuates a tone by 3 dB.

Conventions that differ between the two sides (the parameters are mapped, never the results):

* Vold-Kalman bandwidth: ROTA's ``bandwidth`` (``par.vk_bandwidth``) is the -3 dB offset from
  the carrier (pass band twice as wide); MATLAB's ``Bandwidth`` is given as
  ``par.vk_bandwidth_matlab`` = 2 x ``par.vk_bandwidth``, the full width between the -3 dB points.
  Table 10 measures both filters, and MATLAB also with the same number as ROTA.
* Amplitudes: map readings RMS (``Amplitude`` 'rms'), Vold-Kalman envelopes peak (``Amplitude``
  'peak'); the truth is peak, divided by sqrt(2) for RMS rows.
* Time axes: MATLAB's ``time`` / ``t`` / ``tout`` outputs are used as returned (s from the first
  sample); MATLAB map readings are compared with the truth at the speed MATLAB returns for each
  frame, and with ROTA by interpolation in time.

References
----------
ROTA ROADMAP, sections 5.1 (principles), 5.2 (comparison table), 5.4 (tolerances), 5.5.
Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
Peeters, C., et al. (2019). Mechanical Systems and Signal Processing, 129, 407-436.
"""

import argparse
import importlib.util
import math
import sys
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import rota
from rota.extract import order_waveform
from rota.maps import rpm_frequency_map, rpm_order_map
from rota.slices import order_slice, order_spectrum
from rota.speed import rpm_from_ridge, rpm_to_phase, tacho_to_rpm

HERE = Path(__file__).resolve().parent
SIGNALS_DIR = HERE / "signals"
REFERENCE_DIR = HERE / "reference"

SQRT2 = math.sqrt(2.0)
MISSING = "MATLAB output missing"

# ROADMAP 5.4 (initial tolerances).
TOL_RPM_MEDIAN = 0.1  # %, median relative rpm error, truth and MATLAB
TOL_AMP_TRUTH = 0.2  # dB, order amplitude, steady segments
TOL_AMP_MATLAB = 0.5  # dB
TOL_CORR = 0.99  # waveform correlation, ends removed
TOL_WITHIN3_TRUTH = 90.0  # %, tacholess frames within 3 %
TOL_WITHIN3_MATLAB = 5.0  # percentage points between the two trackers
TOL_BW_PCT = 5.0  # %, Vold-Kalman -3 dB bandwidth, against the set value and against MATLAB
# Not in 5.4: the order of the average-spectrum peak must agree to within half a bin, and the
# axis spacing with the requested resolution to within 5 %.
TOL_AXIS_PCT = 5.0

# Choices of this script (documented in the report).
EDGE_FRAMES = 3  # map frames dropped at each end
STEADY_TOL = 0.02  # largest relative variation of the truth amplitude over a frame
STEADY_SPAN = 1.5  # frame length considered, in nominal frame lengths
VK_EDGE_BANDWIDTHS = 3.0  # Vold-Kalman ends dropped: this / bandwidth seconds
EDGE_SECONDS = 0.5  # speed curves: ends dropped
ROTA_MAX_ORDER = 16.0  # highest order of the ROTA order maps (ROTA only)
RIDGE_WINDOW, RIDGE_HOP = 0.4, 0.05  # rpm_from_ridge, s (ROTA only)
ROTA_WINDOWS = {"hann": "hann", "flattopwin": "flattop"}  # MATLAB window name -> ROTA name


def _load_module(name):
    """Import ``validation/<name>.py`` by path (the folder is not a package)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------
# Small numerical helpers
# --------------------------------------------------------------------------------------------


def _db(a, ref):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20.0 * np.log10(np.asarray(a, dtype=float) / np.asarray(ref, dtype=float))


def _max_abs(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.max(np.abs(v))) if v.size else float("nan")


def _corr(a, b):
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    den = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def _gain_db(w, s):
    """Least-squares gain of ``w`` on ``s`` in dB: ``w = g s`` -> ``20 log10 g``."""
    den = float(np.dot(s, s))
    num = float(np.dot(w, s))
    return float(20.0 * math.log10(num / den)) if den > 0 and num > 0 else float("nan")


def _rel_pct(est, ref):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * np.abs(np.asarray(est, dtype=float) - ref) / np.abs(ref)


def _median(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def _percentile(v, q):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else float("nan")


def _interp_valid(t_new, t, y):
    """``np.interp`` that gives NaN outside the span of ``t`` (t sorted ascending)."""
    out = np.interp(t_new, t, y)
    out[(t_new < t[0]) | (t_new > t[-1])] = np.nan
    return out


def _sorted_unique(t, *arrays):
    t = np.asarray(t, dtype=float).ravel()
    order = np.argsort(t, kind="stable")
    t = t[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    return (t[keep],) + tuple(np.asarray(a, dtype=float).ravel()[order][keep] for a in arrays)


def _clean(v):
    if isinstance(v, (str, np.str_)):
        return str(v)
    a = np.asarray(v)
    if a.dtype.kind in "US":
        return str(a.reshape(-1)[0]) if a.size else ""
    return float(a) if a.ndim == 0 else a.astype(float)


def _text(v):
    if v is None:
        return ""
    a = np.asarray(v)
    if a.size == 0:
        return ""
    return str(a.reshape(-1)[0]) if a.ndim else str(a)


def _orient(M, n_rows, n_cols):
    """``M`` as (n_rows, n_cols), transposing if needed; None if the shape does not fit."""
    M = np.asarray(M, dtype=float)
    if M.ndim == 1:
        if n_rows == 1 and M.size == n_cols:
            return M.reshape(1, -1)
        if n_cols == 1 and M.size == n_rows:
            return M.reshape(-1, 1)
        return None
    if M.shape == (n_rows, n_cols):
        return M
    if M.shape == (n_cols, n_rows):
        return M.T
    return None


# --------------------------------------------------------------------------------------------
# MATLAB reference file
# --------------------------------------------------------------------------------------------


class Ref:
    """Contents of ``<case>_ref.mat`` written by ``validation/matlab/run_reference.m``."""

    def __init__(self, path):
        self.path = Path(path)
        self.d = None
        self.load_error = ""
        # <case>_ref.mat, and the parts <case>_ref_<part>.mat that CI jobs of their own wrote
        # (ROTA_REF_PART in run_reference.m); the calls of different parts do not overlap.
        files = [self.path] if self.path.exists() else []
        files += sorted(self.path.parent.glob(f"{self.path.stem}_*.mat"))
        for f in files:
            try:
                raw = loadmat(f, squeeze_me=True)
                self.d = (self.d or {}) | {k: v for k, v in raw.items() if not k.startswith("__")}
            except Exception as err:  # noqa: BLE001 - report any unreadable file
                self.load_error += f"{f.name}: {type(err).__name__}: {err}. "

    @property
    def present(self):
        return self.d is not None

    def ran(self, cid):
        return self.present and f"{cid}_ok" in self.d

    def ok(self, cid):
        return self.ran(cid) and float(np.asarray(self.d[f"{cid}_ok"]).reshape(-1)[0]) == 1.0

    def get(self, cid, name):
        return None if not self.present else self.d.get(f"{cid}_{name}")

    def text(self, cid, name):
        return _text(self.get(cid, name))

    def var(self, name):
        """A top-level variable of the file (None if absent)."""
        return None if not self.present else self.d.get(name)

    def reason(self, cid):
        """Why the output of call ``cid`` cannot be used (cell text), or ''."""
        if not self.present:
            return MISSING
        if not self.ran(cid):
            return MISSING + " (call not run for this case)"
        if not self.ok(cid):
            msg = self.text(cid, "msg")
            if msg.startswith("skipped"):
                return f"MATLAB call skipped ({msg[len('skipped:'):].strip()[:120] or 'an input call failed'})"
            return "MATLAB call failed"
        return ""

    def detail(self, cid):
        """Message of a failed or degraded call (first 160 characters)."""
        msg = self.text(cid, "msg").strip()
        return (msg[:160] + "...") if len(msg) > 160 else msg

    def warning(self, cid):
        """Warning MATLAB issued during the call that ran (e.g. an iterative solver that did not
        converge), or ''."""
        msg = self.text(cid, "msg")
        i = msg.find("warning: ")
        return f"MATLAB {msg[i:i + 150].strip()}" if i >= 0 else ""

    def degraded(self, cid):
        """Note if a later variant (fewer arguments) of the call was the one that ran, and any
        warning of that call."""
        notes = []
        if self.ok(cid):
            var = int(np.asarray(self.get(cid, "variant")).reshape(-1)[0])
            if var > 1:
                notes.append(f"MATLAB variant {var} ran (fewer arguments: {self.text(cid, 'call')[:140]}).")
            w = self.warning(cid)
            if w:
                notes.append(w.rstrip(".") + ".")
        return " ".join(notes)


# --------------------------------------------------------------------------------------------
# Case: signal, truth, parameters, ROTA results (computed lazily, once)
# --------------------------------------------------------------------------------------------


class Case:
    """One validation case: the exported signal and truth, parameters and the ROTA results."""

    def __init__(self, name, signals_dir, reference_dir):
        path = Path(signals_dir) / f"{name}.mat"
        if not path.exists():
            _load_module("export_mat").main(signals_dir, [name])
        d = loadmat(path, squeeze_me=True, struct_as_record=False)
        self.name = name
        self.fs = float(d["fs"])
        self.x = np.asarray(d["x"], dtype=float)
        self.t = np.asarray(d["t"], dtype=float)
        self.rpm = np.asarray(d["rpm"], dtype=float)
        self.tacho = np.asarray(d["tacho"], dtype=float) if "tacho" in d else None
        self.pulse_times = np.atleast_1d(d["pulse_times"]).astype(float) if "pulse_times" in d else None
        self.orders = np.atleast_1d(d["orders"]).astype(float)
        K = self.orders.size
        self.amp = np.asarray(d["order_amplitude"], dtype=float).reshape(self.x.size, K)
        self.stems = np.asarray(d["order_stems"], dtype=float).reshape(self.x.size, K)
        self.par = {k: _clean(getattr(d["par"], k)) for k in d["par"]._fieldnames}
        self.ref = Ref(Path(reference_dir) / f"{name}_ref.mat")
        self.errors = []  # ROTA failures, for the report

    def flag(self, key):
        return self.par.get(key, 0.0) != 0.0

    @cached_property
    def rev(self):
        return rpm_to_phase(self.rpm, self.fs)

    def truth_vs_rpm(self, k):
        """Peak amplitude of order ``k`` as a function of speed (amplitude laws depend on rpm only)."""
        u, inv = np.unique(self.rpm, return_inverse=True)
        mean = np.bincount(inv, weights=self.amp[:, k]) / np.bincount(inv)
        return lambda q: np.interp(q, u, mean)

    # ---- ROTA results (None plus an entry in self.errors if a call fails) ----------------------

    def _try(self, label, func):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = func()
            self.warnings_of[label] = [f"{w.category.__name__}: {str(w.message)[:110]}" for w in caught]
            return value
        except Exception as err:  # noqa: BLE001 - a failing ROTA call is a result, not a crash
            self.errors.append(f"{label}: {type(err).__name__}: {str(err)[:140]}")
            return None

    @cached_property
    def warnings_of(self):
        return {}

    @cached_property
    def rota_tacho(self):
        return self._try(
            "tacho_to_rpm",
            lambda: tacho_to_rpm(self.tacho, self.fs, int(self.par["ppr"]), smooth=1.0, return_info=True),
        )

    @cached_property
    def rota_order_map(self):
        p = self.par
        return self._try(
            "rpm_order_map",
            lambda: rpm_order_map(
                self.x, self.fs, self.rpm, order_resolution=p["order_resolution"],
                max_order=ROTA_MAX_ORDER, overlap=p["overlap_percent"] / 100.0,
                window=ROTA_WINDOWS[p["window_order_map"]], amplitude=p["amplitude_map"],
            ),
        )

    @cached_property
    def rota_freq_map(self):
        p = self.par
        return self._try(
            "rpm_frequency_map",
            lambda: rpm_frequency_map(
                self.x, self.fs, self.rpm, resolution=p["freq_resolution"],
                overlap=p["overlap_percent"] / 100.0, window=ROTA_WINDOWS[p["window_freq_map"]],
                amplitude=p["amplitude_map"],
            ),
        )

    @cached_property
    def rota_vk(self):
        p = self.par
        return self._try(
            "order_waveform",
            lambda: order_waveform(
                self.x, self.fs, self.rpm, self.orders, bandwidth=p["vk_bandwidth"],
                joint=bool(p["vk_decouple"]),
            ),
        )

    @cached_property
    def rota_ridge(self):
        p = self.par
        lo, hi = (float(v) for v in p["rpm_range"])
        return self._try(
            "rpm_from_ridge",
            lambda: rpm_from_ridge(
                self.x, self.fs, (lo, hi), orders=tuple(self.orders), window=RIDGE_WINDOW,
                hop=RIDGE_HOP,
            ),
        )


# --------------------------------------------------------------------------------------------
# Report tables
# --------------------------------------------------------------------------------------------


@dataclass
class Row:
    """One comparison. Values are floats, or text (a reason why the value is missing)."""

    case: str
    quantity: str
    rota_truth: object = None
    matlab_truth: object = None
    rota_matlab: object = None
    unit: str = ""
    tol_truth: tuple = None  # ("<=", limit): |value| <= limit; (">=", limit): value >= limit
    tol_matlab: tuple = None
    digits: int = 3
    note: str = ""
    unit_m: str = None  # unit of the ROTA-MATLAB column and its tolerance, if not ``unit``

    def _check(self, v, tol):
        if tol is None or isinstance(v, str) or v is None or not np.isfinite(v):
            return None
        return abs(v) <= tol[1] if tol[0] == "<=" else v >= tol[1]

    def verdict(self):
        """('pass' | 'pass (truth decides)' | 'pass (ROTA only)' | 'FAIL' | 'ERROR' | 'info' | 'n/a', comment).

        Analytic truth is the first standard (ROADMAP 5.1): when ROTA is within the truth
        tolerance and MATLAB is outside it, the two differ because of MATLAB, and the row passes
        with the verdict 'pass (truth decides)'.
        """
        if isinstance(self.rota_truth, str) and self.rota_truth.startswith("ROTA error"):
            return "ERROR", ""
        t_ok = self._check(self.rota_truth, self.tol_truth)
        m_ok = self._check(self.rota_matlab, self.tol_matlab)
        mt_ok = self._check(self.matlab_truth, self.tol_truth)
        if t_ok is False:
            return "FAIL", "ROTA is outside the truth tolerance."
        if m_ok is False:
            if mt_ok is False and t_ok is True:
                return "pass (truth decides)", (
                    "ROTA and MATLAB differ; ROTA is within the truth tolerance and MATLAB is not, so the truth "
                    "decides for ROTA."
                )
            if mt_ok is False:
                return "FAIL", "ROTA and MATLAB differ; MATLAB is also outside the truth tolerance, so the truth decides."
            if mt_ok is True:
                return "FAIL", "ROTA and MATLAB differ although both are within the truth tolerance."
            return "FAIL", "ROTA and MATLAB differ beyond the tolerance."
        if self.tol_truth is None and self.tol_matlab is None:
            return "info", ""
        if t_ok is None and m_ok is None:
            return "n/a", ""
        v = self.rota_matlab
        no_matlab = isinstance(v, str) or v is None or not np.isfinite(v)
        if self.tol_matlab is not None and no_matlab:
            return "pass (ROTA only)", ""
        return "pass", ""

    def tolerance_text(self):
        parts = []
        for label, tol, unit in (("ROTA-truth", self.tol_truth, self.unit),
                                 ("ROTA-MATLAB", self.tol_matlab, self.unit if self.unit_m is None else self.unit_m)):
            if tol is not None:
                op = "≤" if tol[0] == "<=" else "≥"
                parts.append(f"{label} {op} {tol[1]:g}{unit}")
        return "; ".join(parts) if parts else "none"

    def cell(self, v, unit=None):
        if isinstance(v, str):
            return v
        if v is None or not np.isfinite(v):
            return "n/a"
        return f"{v:.{self.digits}g}{self.unit if unit is None else unit}"


@dataclass
class Table:
    title: str
    description: str
    rows: list = field(default_factory=list)

    def markdown(self):
        out = [f"### {self.title}", "", self.description, ""]
        out.append("| Case | Quantity | ROTA–truth | MATLAB–truth | ROTA–MATLAB | Tolerance | Result | Notes |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in self.rows:
            status, comment = r.verdict()
            note = " ".join(s for s in (r.note, comment) if s)
            cells = [
                r.case, r.quantity, r.cell(r.rota_truth), r.cell(r.matlab_truth), r.cell(r.rota_matlab, r.unit_m),
                r.tolerance_text(), status, note,
            ]
            out.append("| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in cells) + " |")
        out.append("")
        return "\n".join(out)


def _rota_err(case):
    return "ROTA error: " + (case.errors[-1] if case.errors else "unknown")


def _warn_note(case, label):
    w = case.warnings_of.get(label) or []
    return ("ROTA warning: " + w[0]) if w else ""


# --------------------------------------------------------------------------------------------
# 1. Tachometer speed (T5)
# --------------------------------------------------------------------------------------------


def _matlab_speed(case, cid, t_default_fs):
    """(t, rpm, note) of a MATLAB speed output, or (reason, None, '')."""
    ref = case.ref
    if not ref.ok(cid):
        return ref.reason(cid), None, ref.detail(cid)
    rpm = np.atleast_1d(ref.get(cid, "rpm")).astype(float).ravel()
    t = ref.get(cid, "t") if cid.startswith("tachorpm") else ref.get(cid, "tout")
    note = ref.degraded(cid)
    if t is not None and np.atleast_1d(t).size == rpm.size:
        t = np.atleast_1d(t).astype(float).ravel()
    else:
        t = np.arange(rpm.size) / t_default_fs
        note = (note + " Time axis assumed: " + f"{t_default_fs:g} Hz from the first sample.").strip()
    t, rpm = _sorted_unique(t, rpm)
    return t, rpm, note


def tacho_table(cases):
    """Table 1: ``tacho_to_rpm`` against ``tachorpm`` (T5).

    Speed in rpm; errors are relative, in percent. No amplitude is involved.

    Returns
    -------
    Table
        Rows for the median and the 95th percentile of the relative error.

    References
    ----------
    ROTA ROADMAP, sections 5.1, 5.2 and 5.4.
    """
    tab = Table(
        "1. Tachometer speed: `tacho_to_rpm` vs `tachorpm` (T5)",
        "Speed curves (rpm) interpolated onto a 1 ms grid from 0.5 s to 0.5 s before the end of the "
        "record; relative error in percent. ROTA: `smooth` = 1 revolution. MATLAB: `FitType` smooth with the "
        "break points of `par.fit_points` (primary row) and `FitType` linear (per-pulse values, jitter not "
        "smoothed; informational).",
    )
    for case in cases:
        if case.name != "T5" or case.tacho is None:
            continue
        tg = np.arange(EDGE_SECONDS, case.t[-1] - EDGE_SECONDS, 1e-3)
        truth = np.interp(tg, case.t, case.rpm)
        res = case.rota_tacho
        if res is None:
            tab.rows.append(Row(case.name, "Median relative rpm error", _rota_err(case)))
            continue
        rota = np.interp(tg, case.t, res[0])
        e_rt = _rel_pct(rota, truth)
        note_r = f"ROTA: {res[1]['n_missing']} missing pulse(s) filled."
        out_fs = case.par["tacho_output_fs"]
        for cid, label, tol in (("tachorpm_smooth", "smooth fit", True), ("tachorpm_linear", "linear fit", False)):
            t_m, rpm_m, note_m = _matlab_speed(case, cid, out_fs)
            if isinstance(t_m, str):
                mt = rm = t_m
                n_m = ""
                e_mt = e_rm = None
            else:
                mat = _interp_valid(tg, t_m, rpm_m)
                e_mt = _rel_pct(mat, truth)
                e_rm = _rel_pct(rota, mat)
                mt, rm = _median(e_mt), _median(e_rm)
                n_m = f"n={int(np.isfinite(e_mt).sum())}."
                tp = case.ref.get(cid, "tp")
                if tol and tp is not None and np.size(tp) > 2:
                    # Black-box view of the pulses MATLAB used: intervals of about twice their
                    # neighbours are dropped pulses that were not filled in.
                    dtp = np.diff(np.atleast_1d(tp).astype(float).ravel())
                    nb = np.minimum(np.r_[dtp[1:], dtp[-1]], np.r_[dtp[0], dtp[:-1]])
                    gaps = int(np.sum(dtp > 1.5 * nb))
                    rendered = "" if case.pulse_times is None else f", {case.pulse_times.size} rendered (truth)"
                    n_m += (f" MATLAB `tp`: {np.size(tp)} pulses{rendered}; {gaps} intervals of about twice their "
                            "neighbours (dropped pulses, not filled).")
            tab.rows.append(Row(
                case.name, f"Median relative rpm error ({label})", _median(e_rt), mt, rm, " %",
                ("<=", TOL_RPM_MEDIAN) if tol else None, ("<=", TOL_RPM_MEDIAN) if tol else None,
                note=" ".join(s for s in (note_r if tol else "", n_m, note_m) if s),
            ))
            if tol:
                tab.rows.append(Row(
                    case.name, f"95th percentile relative rpm error ({label})", _percentile(e_rt, 95),
                    _percentile(e_mt, 95) if e_mt is not None else mt,
                    _percentile(e_rm, 95) if e_rm is not None else rm, " %",
                ))
    return tab


# --------------------------------------------------------------------------------------------
# 2. Tacholess speed
# --------------------------------------------------------------------------------------------


def ridge_table(cases):
    """Table 2: ``rpm_from_ridge`` against ``rpmtrack``.

    Speed in rpm, shares in percent of the frames (percentage points for the difference between
    the two trackers). No amplitude is involved.

    Returns
    -------
    Table

    References
    ----------
    ROTA ROADMAP, sections 5.1, 5.2 and 5.4.
    """
    tab = Table(
        "2. Tacholess speed: `rpm_from_ridge` vs `rpmtrack`",
        "Speed (rpm) at the ROTA frame centres between 0.5 s and 0.5 s before the end of the record; "
        "MATLAB's curve is interpolated onto them. ROTA: comb of the case orders, window "
        f"{RIDGE_WINDOW:g} s, hop {RIDGE_HOP:g} s, search range `par.rpm_range`. MATLAB: the order "
        "`par.rpmtrack_order`, an initial point (time, frequency) on its ridge taken from the exact "
        "speed at mid-record, `Method` stft and `par.rpmtrack_resolution`; `rpmtrack` has no argument "
        "for a frequency range. ROTA-MATLAB for the 3 % row is the difference of the two shares in "
        "percentage points.",
    )
    for case in cases:
        if not case.flag("do_rpmtrack"):
            continue
        res = case.rota_ridge
        if res is None:
            tab.rows.append(Row(case.name, "Frames within 3 % of the true rpm", _rota_err(case)))
            continue
        inside = (res.t >= EDGE_SECONDS) & (res.t <= case.t[-1] - EDGE_SECONDS)
        tt = res.t[inside]
        truth = np.interp(tt, case.t, case.rpm)
        e_r = _rel_pct(res.rpm[inside], truth)
        within_r = 100.0 * float(np.mean(e_r <= 3.0))
        at_bound = 100.0 * float(np.mean(res.at_bound[inside]))
        comb = ", ".join(f"{o:g}" for o in case.orders)
        lo_hi = " to ".join(f"{v:g}" for v in case.par["rpm_range"])
        note_r = (f"ROTA: n={tt.size} frames, comb {comb}, range {lo_hi} rpm. "
                  + _warn_note(case, "rpm_from_ridge")).strip()
        t_m, rpm_m, note_m = _matlab_speed(case, "rpmtrack", case.fs)
        if isinstance(t_m, str):
            within_m = within_d = med_m = med_d = t_m
        else:
            mat = _interp_valid(tt, t_m, rpm_m)
            e_m, e_d = _rel_pct(mat, truth), _rel_pct(res.rpm[inside], mat)
            good = np.isfinite(e_m)
            within_m = 100.0 * float(np.mean(e_m[good] <= 3.0)) if good.any() else float("nan")
            # Share ROTA within 3 % on the same frames, so the difference is like for like.
            within_r_same = 100.0 * float(np.mean(e_r[good] <= 3.0)) if good.any() else float("nan")
            within_d = abs(within_r_same - within_m)
            med_m, med_d = _median(e_m), _median(e_d)
            note_m = (f"MATLAB: n={int(good.sum())} frames compared. " + note_m).strip()
        tab.rows.append(Row(
            case.name, "Frames within 3 % of the true rpm", within_r, within_m, within_d, " %",
            (">=", TOL_WITHIN3_TRUTH), ("<=", TOL_WITHIN3_MATLAB), digits=4, note=f"{note_r} {note_m}".strip(),
            unit_m=" pp",
        ))
        tab.rows.append(Row(
            case.name, "Median relative rpm error", _median(e_r), med_m, med_d, " %", digits=3,
            note="Informational: the criterion of ROADMAP 5.4 for tacholess speed is the 3 % share.",
        ))
        tab.rows.append(Row(
            case.name, "ROTA frames flagged `at_bound` (boundary hit)", at_bound, "n/a (no counterpart)",
            "n/a (no counterpart)", " %", ("<=", 0.0), None,
            note="The truth lies inside the search range, so no frame may hit a bound.",
        ))
    return tab


# --------------------------------------------------------------------------------------------
# Amplitude along a ridge (order map, order track, frequency map)
# --------------------------------------------------------------------------------------------


def _steady_by_rev(case, k, centre_rev, half_rev):
    """Frames whose truth amplitude varies by at most STEADY_TOL over +-half_rev revolutions."""
    rev, a = case.rev, case.amp[:, k]
    lo, hi = centre_rev - half_rev, centre_rev + half_rev
    i0, i1 = np.searchsorted(rev, lo), np.searchsorted(rev, hi)
    ok = (lo >= rev[0]) & (hi <= rev[-1]) & (i1 - i0 >= 2)
    out = np.zeros(lo.size, dtype=bool)
    for j in np.flatnonzero(ok):
        seg = a[i0[j]:i1[j]]
        out[j] = (seg.max() - seg.min()) <= STEADY_TOL * seg.mean()
    return out


def _steady_by_time(case, k, t_c, half_s, order, max_sweep_hz):
    """As above in time, and the line sweeps by at most ``max_sweep_hz`` over the span."""
    a = case.amp[:, k]
    f = order * case.rpm / 60.0
    i0, i1 = np.searchsorted(case.t, t_c - half_s), np.searchsorted(case.t, t_c + half_s)
    ok = (t_c - half_s >= case.t[0]) & (t_c + half_s <= case.t[-1]) & (i1 - i0 >= 2)
    out = np.zeros(t_c.size, dtype=bool)
    for j in np.flatnonzero(ok):
        seg, fs_ = a[i0[j]:i1[j]], f[i0[j]:i1[j]]
        out[j] = ((seg.max() - seg.min()) <= STEADY_TOL * seg.mean()) and (np.ptp(fs_) <= max_sweep_hz)
    return out


def _rota_series(case, m, k, steady, bandwidth=None):
    """ROTA reading of order ``k`` from map ``m``: t, amplitude (RMS) and the frames to use."""
    kwargs = {} if bandwidth is None else {"bandwidth": bandwidth}
    amp = np.asarray(order_slice(m, case.orders[k], **kwargs).amplitude, dtype=float)
    n = m.t.size
    interior = np.zeros(n, dtype=bool)
    interior[EDGE_FRAMES:n - EDGE_FRAMES] = True
    return {"t": np.asarray(m.t, dtype=float), "amp": amp, "valid": interior & steady & np.isfinite(amp) & (amp > 0)}


def _to_rms(a, convention):
    return a / SQRT2 if convention == "peak" else a


def _matlab_map_series(case, cid, k, kind):
    """RMS reading of order ``k`` from a MATLAB map; returns (series | reason, note)."""
    ref = case.ref
    if not ref.ok(cid):
        return ref.reason(cid), ref.detail(cid)
    note = ref.degraded(cid)
    M = ref.get(cid, "map")
    ax = np.atleast_1d(ref.get(cid, "order" if kind == "order" else "freq")).astype(float).ravel()
    t = ref.get(cid, "time")
    rpm = ref.get(cid, "rpm")
    if M is None or t is None or rpm is None:
        return "MATLAB output incomplete (map, time or rpm missing)", note
    t, rpm = np.atleast_1d(t).astype(float).ravel(), np.atleast_1d(rpm).astype(float).ravel()
    M = _orient(M, ax.size, t.size)
    if M is None or rpm.size != t.size:
        return "MATLAB output unusable (unexpected array shapes)", note
    conv = ref.text(cid, "amplitude") or "rms"
    mag = np.abs(M)
    order = case.orders[k]
    if kind == "order":
        j = int(np.argmin(np.abs(ax - order)))
        off = (ax[j] - order) / case.par["order_resolution"]
        amp = mag[j]
        if abs(off) > 1e-3:
            note = (note + f" Order is {off:+.3g} bins from the nearest MATLAB bin (scalloping possible).").strip()
    else:
        df = float(np.median(np.diff(ax))) if ax.size > 1 else case.par["freq_resolution"]
        hw = max(case.par["freq_resolution"], 2.0 * df)  # half width of the peak search, Hz
        amp = np.empty(t.size)
        for j in range(t.size):
            band = np.abs(ax - order * rpm[j] / 60.0) <= hw
            amp[j] = mag[band, j].max() if band.any() else np.nan
    t, rpm, amp = _sorted_unique(t, rpm, _to_rms(amp, conv))
    return {"t": t, "rpm": rpm, "amp": amp}, note


def _matlab_track_series(case, cid, k):
    """RMS series of order ``k`` from MATLAB ``ordertrack`` (rows: orders, columns: time)."""
    ref = case.ref
    if not ref.ok(cid):
        return ref.reason(cid), ref.detail(cid)
    note = ref.degraded(cid)
    mag, t, rpm = ref.get(cid, "mag"), ref.get(cid, "time"), ref.get(cid, "rpm")
    if mag is None or t is None or rpm is None:
        return "MATLAB output incomplete (mag, time or rpm missing)", note
    t, rpm = np.atleast_1d(t).astype(float).ravel(), np.atleast_1d(rpm).astype(float).ravel()
    M = _orient(mag, case.orders.size, t.size)
    if M is None or rpm.size != t.size:
        return "MATLAB output unusable (unexpected array shapes)", note
    conv = ref.text(cid, "amplitude") or "rms"
    t, rpm, amp = _sorted_unique(t, rpm, _to_rms(M[k], conv))
    return {"t": t, "rpm": rpm, "amp": amp}, note


def _amp_compare(case, k, R, M):
    """(rota_truth, matlab_truth, rota_matlab, n_rota, n_matlab, n_both) in dB for order ``k``.

    ``R`` is the ROTA series and ``M`` the MATLAB series or the text of the reason it is missing.
    The truth is the peak amplitude over sqrt(2) (RMS), at the frame centre time for ROTA and at
    the frame speed for MATLAB (amplitude laws are functions of speed).
    """
    nan = float("nan")
    okR = R["valid"]
    truth_t = np.interp(R["t"], case.t, case.amp[:, k]) / SQRT2
    rt = _max_abs(_db(R["amp"][okR], truth_t[okR])) if okR.any() else nan
    if isinstance(M, str):
        return rt, M, M, int(okR.sum()), 0, 0
    tM, rM, aM = M["t"], M["rpm"], M["amp"]
    validM = (np.interp(tM, R["t"], R["valid"].astype(float)) >= 0.999) & np.isfinite(aM) & (aM > 0)
    truthM = case.truth_vs_rpm(k)(rM) / SQRT2
    mt = _max_abs(_db(aM[validM], truthM[validM])) if validM.any() else nan
    rm, n_both = nan, 0
    if tM.size >= 2:
        j = np.clip(np.searchsorted(tM, R["t"]), 1, tM.size - 1)
        both = okR & (R["t"] >= tM[0]) & (R["t"] <= tM[-1]) & validM[j - 1] & validM[j]
        if both.any():
            dbM = np.where(validM, _db(np.where(validM, aM, 1.0), 1.0), 0.0)
            w = (R["t"] - tM[j - 1]) / (tM[j] - tM[j - 1])
            diff = _db(R["amp"], 1.0) - (dbM[j - 1] * (1.0 - w) + dbM[j] * w)
            rm, n_both = _max_abs(diff[both]), int(both.sum())
    return rt, mt, rm, int(okR.sum()), int(validM.sum()), n_both


def _amp_rows(tab, case, cid, kind, series_of, rota_map, bandwidth=None, label=""):
    """Rows of an amplitude-along-the-ridge table for one case and one MATLAB call."""
    m = rota_map
    if m is None:
        tab.rows.append(Row(case.name, f"Order amplitude {label}", _rota_err(case)))
        return
    res = case.par["order_resolution"] if kind == "order" else case.par["freq_resolution"]
    centre_rev = np.interp(m.t, case.t, case.rev)
    for k, order in enumerate(case.orders):
        if kind == "order":
            steady = _steady_by_rev(case, k, centre_rev, 0.5 * STEADY_SPAN / res)
        else:
            steady = _steady_by_time(case, k, m.t, 0.5 * STEADY_SPAN / res, order, 0.5 * res)
        R = _rota_series(case, m, k, steady, bandwidth)
        M, note_m = series_of(case, cid, k)
        rt, mt, rm, nR, nM, nB = _amp_compare(case, k, R, M)
        note = f"n={nR} ROTA frames" + (f", {nM} MATLAB frames, {nB} matched." if nM else ".")
        note = f"{note} {note_m}".strip()
        if nR == 0:
            note += " No steady interior frames."
        tab.rows.append(Row(
            case.name, f"Order {order:g} {label}".strip(), rt, mt, rm, " dB",
            ("<=", TOL_AMP_TRUTH), ("<=", TOL_AMP_MATLAB), digits=3, note=note,
        ))


_STEADY_TEXT = (
    "Largest absolute dB difference over the interior frames (3 dropped at each end) where the truth "
    f"amplitude varies by at most {100 * STEADY_TOL:g} % over {STEADY_SPAN:g} frame lengths; RMS amplitude, "
    "the truth being the peak amplitude over sqrt(2). ROTA-MATLAB: the MATLAB readings are interpolated "
    "linearly in time (dB) onto the ROTA frames."
)


def order_map_table(cases):
    """Table 3: order amplitude along the ridge, ``rpm_order_map`` + ``order_slice`` against ``rpmordermap``.

    Amplitudes are RMS (the truth is the peak amplitude over sqrt(2)); differences in dB.

    Returns
    -------
    Table

    References
    ----------
    ROTA ROADMAP, sections 5.1, 5.2 and 5.4.
    """
    tab = Table(
        "3. Order amplitude along the ridge: `rpm_order_map` + `order_slice` vs `rpmordermap`",
        "ROTA: `order_slice` (band power) of an angle-domain map, Hann window, 75 % overlap. MATLAB: the "
        "value of the bin nearest the order, same window and overlap, resolution `par.order_resolution`. "
        + _STEADY_TEXT,
    )
    for case in cases:
        if case.flag("do_maps"):
            _amp_rows(tab, case, "rpmordermap", "order",
                      lambda c, cid, k: _matlab_map_series(c, cid, k, "order"), case.rota_order_map)
    return tab


def order_track_table(cases):
    """Table 4: ``order_slice`` against ``ordertrack`` applied to the MATLAB order map.

    Amplitudes are RMS; differences in dB.

    Returns
    -------
    Table

    References
    ----------
    ROTA ROADMAP, sections 5.1, 5.2 and 5.4.
    """
    tab = Table(
        "4. Order tracks: `order_slice` vs `ordertrack` (order map input)",
        "The same ROTA readings as in table 3 against the magnitude matrix that `ordertrack` returns for "
        "the MATLAB order map and the case orders. " + _STEADY_TEXT,
    )
    for case in cases:
        if case.flag("do_maps"):
            _amp_rows(tab, case, "ordertrack_map", "order",
                      lambda c, cid, k: _matlab_track_series(c, cid, k), case.rota_order_map)
    return tab


def freq_map_table(cases):
    """Table 5: ``rpm_frequency_map`` + ``order_slice`` against ``rpmfreqmap``.

    Amplitudes are RMS; differences in dB. Flat-top window, so that the peak reading of a line
    is free of scalloping loss.

    Returns
    -------
    Table

    References
    ----------
    ROTA ROADMAP, sections 5.1, 5.2 and 5.4.
    """
    tab = Table(
        "5. Order amplitude along the ridge: `rpm_frequency_map` + `order_slice` vs `rpmfreqmap`",
        "Flat-top window (no scalloping), 75 % overlap, resolution `par.freq_resolution`. ROTA: `order_slice` "
        "(band power over the whole main lobe). MATLAB: the largest value within one resolution of the "
        "ridge frequency `order * rpm / 60` of each frame. Frames are used only where the line moves by at "
        f"most half a resolution within {STEADY_SPAN:g} frame lengths and is clear of 0 Hz. " + _STEADY_TEXT,
    )
    for case in cases:
        if case.flag("do_freqmap"):
            _amp_rows(tab, case, "rpmfreqmap", "freq",
                      lambda c, cid, k: _matlab_map_series(c, cid, k, "freq"), case.rota_freq_map,
                      bandwidth=11.0 * case.par["freq_resolution"])
    return tab


def axis_table(cases):
    """Table 6: spacing of the order and frequency axes against the requested resolution (percent).

    Returns
    -------
    Table

    References
    ----------
    ROTA ROADMAP, sections 5.1, 5.2 and 5.4.
    """
    tab = Table(
        "6. Order and frequency axes: spacing against the requested resolution",
        "Median spacing of the bin axis relative to `par.order_resolution` / `par.freq_resolution` (the bin "
        "spacing, one over the window length), in percent (signed; the tolerance applies to the absolute "
        "value). ROTA-MATLAB is the relative difference of the two spacings. MATLAB's `res` is a resolution "
        "bandwidth, the window ENBW times the bin spacing, so it is given `par.order_resolution_matlab` / "
        "`par.freq_resolution_matlab` = ENBW x the resolution (Hann 1.5, flat top 3.77 bins); its `res` output "
        "is listed in the notes. The time-axis rows check MATLAB's frame times against its speed output.",
    )
    for case in cases:
        for do, kind, cid, rota_map, req in (
            ("do_maps", "order", "rpmordermap", "rota_order_map", case.par["order_resolution"]),
            ("do_freqmap", "freq", "rpmfreqmap", "rota_freq_map", case.par["freq_resolution"]),
        ):
            if not case.flag(do):
                continue
            m = getattr(case, rota_map)
            name = "Order-axis spacing" if kind == "order" else "Frequency-axis spacing"
            if m is None:
                tab.rows.append(Row(case.name, name, _rota_err(case)))
                continue
            sp_r = float(m.resolution)
            if not case.ref.ok(cid):
                mt = rm = case.ref.reason(cid)
            else:
                ax = np.atleast_1d(case.ref.get(cid, "order" if kind == "order" else "freq")).astype(float)
                sp_m = float(np.median(np.diff(ax))) if ax.size > 1 else float("nan")
                mt, rm = 100.0 * (sp_m / req - 1.0), 100.0 * (sp_r / sp_m - 1.0)
            tab.rows.append(Row(
                case.name, name, 100.0 * (sp_r / req - 1.0), mt, rm, " %",
                ("<=", TOL_AXIS_PCT), ("<=", TOL_AXIS_PCT),
                note=f"requested {req:g}, ROTA {sp_r:g}" + (
                    f", MATLAB {sp_m:.6g} (`res` given {_text(case.ref.get(cid, 'res_in')) or 'n/a'}, returned "
                    f"{float(np.asarray(case.ref.get(cid, 'res'), dtype=float).reshape(-1)[0]):.6g})"
                    if case.ref.ok(cid) and case.ref.get(cid, "res") is not None else "") + ".",
            ))
            # Time axis of the MATLAB map: the exact speed at the returned frame times against the
            # speed returned for the same frames (small if the times are the frame centres).
            if case.ref.ok(cid):
                t_m, rpm_m = case.ref.get(cid, "time"), case.ref.get(cid, "rpm")
                if t_m is not None and rpm_m is not None and np.size(t_m) == np.size(rpm_m) and np.size(t_m) > 0:
                    t_m = np.atleast_1d(t_m).astype(float).ravel()
                    rpm_m = np.atleast_1d(rpm_m).astype(float).ravel()
                    dev = _rel_pct(np.interp(t_m, case.t, case.rpm), rpm_m)
                    tab.rows.append(Row(
                        case.name, f"MATLAB {cid} time axis: exact rpm at `time` vs returned `rpm`",
                        "n/a (MATLAB only)", _median(dev), "n/a (MATLAB only)", " %", digits=3,
                        note=(f"Informational: median relative difference over {t_m.size} frames (95th percentile "
                              f"{_percentile(dev, 95):.3g} %); `time` from {t_m[0]:.4g} s to {t_m[-1]:.4g} s."),
                    ))
    return tab


# --------------------------------------------------------------------------------------------
# 7. Average order spectrum
# --------------------------------------------------------------------------------------------


def spectrum_table(cases):
    """Table 7: average order spectrum, ``order_spectrum`` against ``orderspectrum``.

    Peak order in orders; peak amplitude RMS, differences in dB.

    Returns
    -------
    Table

    References
    ----------
    ROTA ROADMAP, sections 5.1, 5.2 and 5.4.
    """
    tab = Table(
        "7. Average order spectrum: `order_spectrum` vs `orderspectrum`",
        "Peak of the spectrum above order 0.5 of the order map of table 3. Truth peak: the order with "
        "the largest angle-averaged power. Peak order: absolute difference in orders, tolerance half the order "
        "resolution (a choice of this report). Peak amplitude (RMS, dB) only where the truth amplitude of "
        "that order is constant (varies by less than 1 %), because the way the two averages weight frames "
        "differs.",
    )
    for case in cases:
        if not case.flag("do_maps"):
            continue
        res = case.par["order_resolution"]
        # angle-averaged power of every order: weights ~ rpm (d rev = rpm dt / 60)
        w = case.rpm / case.rpm.sum()
        power = (w[:, None] * case.amp**2).sum(axis=0) / 2.0
        kt = int(np.argmax(power))
        order_t = float(case.orders[kt])
        const = np.ptp(case.amp[:, kt]) <= 0.01 * case.amp[:, kt].mean()
        amp_t = math.sqrt(power[kt])
        m = case.rota_order_map
        if m is None:
            tab.rows.append(Row(case.name, "Peak order", _rota_err(case)))
            continue
        sp = order_spectrum(m)
        sel = sp.axis >= 0.5
        i = int(np.argmax(np.where(sel, sp.amplitude, -np.inf)))
        order_r, amp_r = float(sp.axis[i]), float(sp.amplitude[i])
        order_m = amp_m = None
        reason = ""
        if not case.ref.ok("orderspectrum"):
            reason = case.ref.reason("orderspectrum")
        else:
            ax = np.atleast_1d(case.ref.get("orderspectrum", "order")).astype(float).ravel()
            spec = np.atleast_1d(case.ref.get("orderspectrum", "spec")).astype(float).ravel()
            if ax.size != spec.size:
                reason = "MATLAB output unusable (spec and order differ in length)"
            else:
                sel_m = ax >= 0.5
                j = int(np.argmax(np.where(sel_m, spec, -np.inf)))
                order_m = float(ax[j])
                conv = case.ref.text("orderspectrum", "amplitude") or "rms"
                amp_m = float(_to_rms(spec[j], conv))
        note = f"Truth peak: order {order_t:g}. " + case.ref.degraded("orderspectrum")
        tab.rows.append(Row(
            case.name, "Peak order", abs(order_r - order_t), reason or abs(order_m - order_t),
            reason or abs(order_r - order_m), " orders", ("<=", 0.5 * res), ("<=", 0.5 * res), digits=3,
            note=f"{note} ROTA at {order_r:g}" + ("" if reason else f", MATLAB at {order_m:g}") + ".",
        ))
        if const:
            truth_rms = amp_t
            tab.rows.append(Row(
                case.name, f"Peak amplitude (order {order_t:g})", float(_db(amp_r, truth_rms)),
                reason or float(_db(amp_m, truth_rms)), reason or float(_db(amp_r, amp_m)), " dB",
                ("<=", TOL_AMP_TRUTH), ("<=", TOL_AMP_MATLAB), digits=3,
                note=f"Truth {truth_rms:.4g} (RMS, constant).",
            ))
    return tab


# --------------------------------------------------------------------------------------------
# 8-9. Vold-Kalman
# --------------------------------------------------------------------------------------------


def _vk_slice(case):
    e = math.ceil(VK_EDGE_BANDWIDTHS * case.fs / case.par["vk_bandwidth"])
    return slice(e, case.t.size - e)


def vk_tables(cases):
    """Tables 8 and 9: Vold-Kalman waveforms and envelopes.

    ``order_waveform`` against ``orderwaveform`` (waveform correlation and least-squares gain in
    dB) and against ``ordertrack`` in its Vold-Kalman form (envelope amplitude, PEAK convention,
    differences in dB).

    Returns
    -------
    (Table, Table)
        Waveform table and envelope table.

    References
    ----------
    ROTA ROADMAP, sections 5.1, 5.2 and 5.4.
    Vold, H., Mains, M., & Blough, J. (1997). Theoretical foundations for high performance
    order tracking with the Vold-Kalman tracking filter. SAE Technical Paper 972007.
    """
    wave = Table(
        "8. Vold-Kalman waveform: `order_waveform` vs `orderwaveform`",
        "Order waveforms against the truth stems (`order_amplitude * cos(phase)`), away from the record ends "
        f"({VK_EDGE_BANDWIDTHS:g} / bandwidth seconds). Correlation: Pearson coefficient. Gain: least-squares "
        "gain of the waveform on the stem in dB (amplitude of the waveform, independent of any envelope "
        "definition). ROTA: bandwidth `par.vk_bandwidth` (-3 dB offset from the carrier), joint solution when "
        "there are several orders. MATLAB: `FilterOrder` 2, `Bandwidth` `par.vk_bandwidth_matlab` = 2 x "
        "`par.vk_bandwidth` (full width between the -3 dB points, so that both filters have the same pass band; "
        "measured in table 10), `Decouple` as `par.vk_decouple`.",
    )
    env = Table(
        "9. Vold-Kalman envelope amplitude: `order_waveform` vs `ordertrack` (Vold-Kalman form)",
        "Peak envelope against the truth `order_amplitude`; the MATLAB track (amplitude type peak) is "
        "interpolated onto the sample times. Same interior and bandwidth as table 8. Largest absolute dB "
        "difference.",
    )
    for case in cases:
        if not case.flag("do_vk"):
            continue
        comp = case.rota_vk
        if comp is None:
            wave.rows.append(Row(case.name, "Waveform correlation", _rota_err(case)))
            env.rows.append(Row(case.name, "Envelope amplitude", _rota_err(case)))
            continue
        sl = _vk_slice(case)
        if sl.stop <= sl.start:
            wave.rows.append(Row(case.name, "Waveform correlation", "record too short for the bandwidth"))
            continue
        ref = case.ref
        K = case.orders.size
        # MATLAB waveform
        xrec = None
        if ref.ok("orderwaveform"):
            xrec = _orient(ref.get("orderwaveform", "xrec"), case.t.size, K)
            reason_w = "" if xrec is not None else "MATLAB output unusable (unexpected array shape)"
        else:
            reason_w = ref.reason("orderwaveform")
        note_w = " ".join(s for s in (ref.degraded("orderwaveform"), "" if xrec is not None else ref.detail("orderwaveform")) if s)
        # MATLAB envelope
        mag = None
        reason_e = ""
        if ref.ok("ordertrack_vk"):
            t_e = ref.get("ordertrack_vk", "time")
            m_e = ref.get("ordertrack_vk", "mag")
            if t_e is None or m_e is None:
                reason_e = "MATLAB output incomplete (mag or time missing)"
            else:
                t_e = np.atleast_1d(t_e).astype(float).ravel()
                M = _orient(m_e, K, t_e.size)
                if M is None:
                    reason_e = "MATLAB output unusable (unexpected array shapes)"
                else:
                    conv = ref.text("ordertrack_vk", "amplitude") or "peak"
                    f = 1.0 if conv == "peak" else SQRT2
                    mag = np.vstack([_interp_valid(case.t, *_sorted_unique(t_e, f * M[k])) for k in range(K)]).T
        else:
            reason_e = ref.reason("ordertrack_vk")
        note_e = " ".join(s for s in (ref.degraded("ordertrack_vk"), "" if not reason_e else ref.detail("ordertrack_vk")) if s)
        for k, order in enumerate(case.orders):
            s = case.stems[sl, k]
            w_r = comp.waveforms[sl, k]
            qty = f"Order {order:g}"
            base = f"n={s.size}."
            if reason_w:
                c_m = g_m = c_rm = g_rm = reason_w
            else:
                w_m = xrec[sl, k]
                c_m, c_rm = _corr(w_m, s), _corr(w_r, w_m)
                g_m, g_rm = _gain_db(w_m, s), _gain_db(w_r, w_m)
            wave.rows.append(Row(case.name, f"{qty} waveform correlation", _corr(w_r, s), c_m, c_rm, "",
                                 (">=", TOL_CORR), (">=", TOL_CORR), digits=6, note=f"{base} {note_w}".strip()))
            wave.rows.append(Row(case.name, f"{qty} waveform gain", _gain_db(w_r, s), g_m, g_rm, " dB",
                                 ("<=", TOL_AMP_TRUTH), ("<=", TOL_AMP_MATLAB), digits=3))
            a_r = np.abs(comp.envelopes[sl, k])
            e_rt = _max_abs(_db(a_r, case.amp[sl, k]))
            if reason_e:
                e_mt = e_rm = reason_e
            else:
                a_m = mag[sl, k]
                e_mt = _max_abs(_db(a_m, case.amp[sl, k]))
                e_rm = _max_abs(_db(a_r, a_m))
            env.rows.append(Row(case.name, f"{qty} envelope amplitude", e_rt, e_mt, e_rm, " dB",
                                ("<=", TOL_AMP_TRUTH), ("<=", TOL_AMP_MATLAB), digits=3,
                                note=f"{base} {note_e}".strip()))
    return wave, env


# --------------------------------------------------------------------------------------------
# 10. Vold-Kalman -3 dB bandwidth (probe VKBW)
# --------------------------------------------------------------------------------------------


def _probe_gains(case, w, sl):
    """Amplitude gains of a filter output ``w`` at the carrier and at every probe offset.

    The input and the output are fitted (least squares, interior samples ``sl``) on cosines and
    sines at the known frequencies: the carrier ``probe_order * rpm / 60`` (constant speed) and
    the carrier plus each offset of ``par.probe_offsets`` (Hz). Gain = |output| / |input| of
    each line; it does not depend on the amplitude convention of the filter.

    Returns
    -------
    (ndarray, float, ndarray)
        Offsets (Hz), gain at the carrier and gains at the offsets (linear).
    """
    offsets = np.atleast_1d(case.par["probe_offsets"]).astype(float)
    f0 = case.par["probe_order"] * float(np.median(case.rpm)) / 60.0
    freqs = np.concatenate(([f0], f0 + offsets))
    t = case.t[sl]
    arg = 2.0 * np.pi * t[:, None] * freqs[None, :]
    A = np.hstack([np.cos(arg), np.sin(arg)])
    F = freqs.size
    cx = np.linalg.lstsq(A, case.x[sl], rcond=None)[0]
    cw = np.linalg.lstsq(A, np.asarray(w, dtype=float)[sl], rcond=None)[0]
    g = np.abs(cw[:F] + 1j * cw[F:]) / np.abs(cx[:F] + 1j * cx[F:])
    return offsets, float(g[0]), g[1:]


def _offset_3db(offsets, gains, side):
    """Offset (Hz, positive) at which the gain falls to 1/sqrt(2), above (side +1) or below (-1).

    Between the two offsets that bracket the crossing, ``log(1 / g - 1)`` is interpolated
    linearly in ``log |offset|``; this is exact for any response ``1 / (1 + c |offset|^p)``,
    whatever ``p`` (the structure order). NaN if the gain does not cross 1/sqrt(2) on the grid.
    """
    sel = offsets * side > 0
    d, g = np.abs(offsets[sel]), gains[sel]
    order = np.argsort(d)
    d, g = d[order], g[order]
    target = 1.0 / SQRT2
    below = np.flatnonzero(g < target)
    if below.size == 0 or below[0] == 0:
        return float("nan")
    j = int(below[0])
    d0, d1, g0, g1 = d[j - 1], d[j], g[j - 1], g[j]
    if 0.0 < g1 < g0 < 1.0:
        y0, y1, yt = math.log(1.0 / g0 - 1.0), math.log(1.0 / g1 - 1.0), math.log(1.0 / target - 1.0)
        return float(math.exp(math.log(d0) + (yt - y0) / (y1 - y0) * (math.log(d1) - math.log(d0))))
    db0, db1, dbt = 20.0 * math.log10(g0), 20.0 * math.log10(g1), 20.0 * math.log10(target)
    return float(d0 + (dbt - db0) / (db1 - db0) * (d1 - d0))


def probe_table(cases):
    """Table 10: -3 dB bandwidth of the Vold-Kalman filter on the probe signal VKBW.

    The set value is ROTA's ``bandwidth`` (``par.vk_bandwidth``, Hz): the offset from the carrier
    at which a tone is attenuated by 3 dB. ROTA is run with it, MATLAB ``orderwaveform`` with
    ``Bandwidth`` = ``par.vk_bandwidth_matlab`` (twice the set value, taken as the full width
    between the two -3 dB points); the measured offsets are compared in percent. An
    informational row gives the offset MATLAB reaches when its ``Bandwidth`` is the same number
    as ROTA's, which shows what the argument means. Gains are amplitude ratios (dB).

    Returns
    -------
    Table

    References
    ----------
    ROTA ROADMAP, sections 5.1, 5.2 and 5.4 (Vold-Kalman -3 dB bandwidth: within 5 % of the set
    value and of MATLAB).
    """
    tab = Table(
        "10. Vold-Kalman -3 dB bandwidth: `order_waveform` vs `orderwaveform` (probe VKBW)",
        "Constant speed, one order (carrier 25 Hz, fs 200 Hz) and unit tones at known offsets from it "
        "(`par.probe_offsets`). "
        "Single-order filters, second-order structure. The gain at each offset is |output| / |input| of a "
        "least-squares fit on the known frequencies over the interior "
        f"({VK_EDGE_BANDWIDTHS:g} / bandwidth seconds dropped at each end); the -3 dB offset is interpolated "
        "between the two offsets that bracket the gain 1/sqrt(2) (exact for any response 1 / (1 + c d^p)). "
        "Set value: ROTA's `bandwidth` (-3 dB offset from the carrier). MATLAB `Bandwidth` = "
        "`par.vk_bandwidth_matlab` (twice the set value: the full width between the -3 dB points). "
        "Offsets in percent of the set value; ROTA-MATLAB in percent of the MATLAB offset.",
    )
    for case in cases:
        if not case.flag("do_vkprobe"):
            continue
        B = float(case.par["vk_bandwidth"])
        sl = _vk_slice(case)
        comp = case._try(
            "order_waveform (probe)",
            lambda: order_waveform(case.x, case.fs, case.rpm, [case.par["probe_order"]], bandwidth=B, joint=False),
        )
        if comp is None:
            tab.rows.append(Row(case.name, "-3 dB offset above the carrier", _rota_err(case)))
            continue
        offs, g0_r, g_r = _probe_gains(case, comp.waveforms[:, 0], sl)
        ref = case.ref
        res_m = {}
        for cid in ("vkprobe", "vkprobe_same"):
            if ref.ok(cid):
                w = np.asarray(ref.get(cid, "xrec"), dtype=float)
                w = _orient(w, case.t.size, 1)
                res_m[cid] = (_probe_gains(case, w[:, 0], sl) if w is not None
                              else "MATLAB output unusable (unexpected array shape)")
            else:
                res_m[cid] = ref.reason(cid)
        m = res_m["vkprobe"]
        bw_m = float(case.par["vk_bandwidth_matlab"])
        note_m = "" if not isinstance(m, str) else ref.detail("vkprobe")
        db = lambda g: 20.0 * math.log10(g) if g > 0 else float("nan")  # noqa: E731
        tab.rows.append(Row(
            case.name, "Gain at the carrier", db(g0_r), m if isinstance(m, str) else db(m[1]),
            m if isinstance(m, str) else db(g0_r) - db(m[1]), " dB", ("<=", TOL_AMP_TRUTH), ("<=", TOL_AMP_MATLAB),
            digits=3, note=note_m,
        ))
        for side, label in ((1, "above"), (-1, "below")):
            f_r = _offset_3db(offs, g_r, side)
            if isinstance(m, str):
                f_m, mt, rm = None, m, m
            else:
                f_m = _offset_3db(offs, m[2], side)
                mt, rm = 100.0 * (f_m / B - 1.0), 100.0 * (f_r / f_m - 1.0)
            twice = np.flatnonzero(np.isclose(offs, 2.0 * side * B))
            g2 = (f" Gain at {2 * side * B:+g} Hz: ROTA {db(g_r[twice[0]]):.2f} dB"
                  + ("" if isinstance(m, str) else f", MATLAB {db(m[2][twice[0]]):.2f} dB") + "."
                  ) if twice.size else ""
            tab.rows.append(Row(
                case.name, f"-3 dB offset {label} the carrier", 100.0 * (f_r / B - 1.0), mt, rm, " %",
                ("<=", TOL_BW_PCT), ("<=", TOL_BW_PCT), digits=3,
                note=(f"Set {B:g} Hz; ROTA {f_r:.4g} Hz" + ("" if f_m is None else f", MATLAB {f_m:.4g} Hz")
                      + f" (MATLAB `Bandwidth` {bw_m:g} Hz)." + g2 + (" " + note_m if note_m else "")).strip(),
            ))
        s = res_m["vkprobe_same"]
        if isinstance(s, str):
            tab.rows.append(Row(case.name, f"MATLAB with `Bandwidth` {B:g} Hz: -3 dB offset above the carrier",
                                "n/a (MATLAB only)", s, s, " %", note=ref.detail("vkprobe_same")))
        else:
            f_s = _offset_3db(offs, s[2], 1)
            tab.rows.append(Row(
                case.name, f"MATLAB with `Bandwidth` {B:g} Hz: -3 dB offset above the carrier", "n/a (MATLAB only)",
                100.0 * (f_s / B - 1.0), "n/a (MATLAB only)", " %", digits=3,
                note=(f"Informational: MATLAB reaches {f_s:.4g} Hz, i.e. {f_s / B:.3f} times the number given; "
                      "a ratio of 0.5 means that `Bandwidth` is the full -3 dB width."),
            ))
    return tab


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


def _calls_appendix(cases):
    lines = []
    for case in cases:
        d = case.ref.d
        if d is None:
            continue
        for key in sorted(d):
            if not key.endswith("_call"):
                continue
            cid = key[: -len("_call")]
            ok = case.ref.ok(cid)
            status = f"variant {int(np.asarray(d[cid + '_variant']).reshape(-1)[0])}" if ok else "FAILED / not run"
            lines.append(f"- {case.name} `{cid}` ({status}): `{case.ref.text(cid, 'call')}`")
            msg = case.ref.text(cid, "msg").strip()
            if msg:
                lines.append(f"  - message: {msg[:300]}")
        if "case_error" in d:
            lines.append(f"- {case.name}: case error: {_text(d['case_error'])[:300]}")
    return lines


def build_report(cases, reference_dir):
    """Run all comparisons and assemble the markdown report.

    Returns
    -------
    (str, dict)
        The report text and the number of rows per verdict.

    References
    ----------
    ROTA ROADMAP, sections 5.1, 5.2 and 5.4.
    """
    tables = [
        tacho_table(cases), ridge_table(cases), order_map_table(cases), order_track_table(cases),
        freq_map_table(cases), axis_table(cases), spectrum_table(cases),
    ]
    tables.extend(vk_tables(cases))
    tables.append(probe_table(cases))

    counts = {}
    for tab in tables:
        for r in tab.rows:
            s = r.verdict()[0]
            counts[s] = counts.get(s, 0) + 1

    have = [c.name for c in cases if c.ref.present]
    versions = Path(reference_dir) / "versions.txt"
    lines = [
        "# ROTA black-box reference comparison",
        "",
        f"ROTA {rota.__version__}; NumPy {np.__version__}; cases: {', '.join(c.name for c in cases)}.",
    ]
    if have:
        rel = next((_text(c.ref.var("matlab_release")) for c in cases if c.ref.var("matlab_release") is not None), "")
        lines.append(f"MATLAB output present for: {', '.join(have)}" + (f" (release {rel})." if rel else "."))
        if versions.exists():
            head = [ln.strip() for ln in versions.read_text(errors="replace").splitlines()[:3] if ln.strip()]
            lines.append("Versions: " + "; ".join(head))
    else:
        lines.append(f"{MISSING}: no `<case>_ref.mat` in `{Path(reference_dir).name}/`; only ROTA against the truth is evaluated.")
    lines += [
        "",
        (
            "Analytic truth is the first standard (ROADMAP 5.1): the MATLAB output is a second reference, and "
            "when the two disagree the truth decides. All parameters are given explicitly on both sides "
            "(`validation/export_mat.py`, field `par`); physical quantities are compared, never map cells. "
            "Tolerances are the initial values of ROADMAP 5.4. The MATLAB outputs are not part of the "
            "repository (licence terms to be confirmed before publishing them, ROADMAP 5.5)."
        ),
        "",
        "Verdicts: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) + ".",
        "",
    ]
    for tab in tables:
        lines.append(tab.markdown())
    errs = [f"- {c.name}: {e}" for c in cases for e in c.errors]
    if errs:
        lines += ["### ROTA errors", ""] + errs + [""]
    calls = _calls_appendix(cases)
    if calls:
        lines += ["### MATLAB calls recorded", ""] + calls + [""]
    lines += [
        "### Notes",
        "",
        "- Not compared here: the cases T4, T8, T10 and T13. VKBW is a probe of the Vold-Kalman bandwidth only.",
        (
            "- Rows say `MATLAB output missing` when `validation/reference/<case>_ref.mat` does not exist or the "
            "call was not run for the case, and `MATLAB call failed` when the call raised an error in every "
            "variant (the message is in the notes and in `log.txt`)."
        ),
        "",
    ]
    return "\n".join(lines), counts


def main(argv=None):
    """Command line entry point: write the report, return the exit status.

    Returns
    -------
    int
        0, or 1 with ``--strict`` if any row failed.

    References
    ----------
    ROTA ROADMAP, sections 5.1, 5.2 and 5.4.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(REFERENCE_DIR / "report.md"), help="markdown report to write")
    ap.add_argument("--signals", default=str(SIGNALS_DIR), help="folder with <case>.mat")
    ap.add_argument("--reference", default=str(REFERENCE_DIR), help="folder with <case>_ref.mat")
    ap.add_argument("--cases", nargs="*", default=None, help="cases to compare (default: all exported)")
    ap.add_argument("--strict", action="store_true", help="exit with status 1 if any row fails")
    args = ap.parse_args(argv)

    export_mat = _load_module("export_mat")
    names = args.cases or list(export_mat.CASE_NAMES)
    bad = [n for n in names if n not in export_mat.CASE_NAMES]
    if bad:
        ap.error(f"unknown case(s) {bad}; available: {list(export_mat.CASE_NAMES)}")
    cases = [Case(n, args.signals, args.reference) for n in names]
    report, counts = build_report(cases, args.reference)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"report: {out}  ({', '.join(f'{v} {k}' for k, v in sorted(counts.items()))})")
    return 1 if args.strict and counts.get("FAIL", 0) + counts.get("ERROR", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
