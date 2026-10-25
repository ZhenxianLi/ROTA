"""Resolvability diagnostic for order extraction.

Before extracting orders, :func:`resolution_check` tells whether the requested
components can be told apart with a chosen analysis resolution, and which
setting would do. It works on frequencies only (``f = order * rpm / 60``), so
no amplitude convention is involved: the result depends on the speed curves,
the order list and the resolution setting, not on the signal.

Three kinds of setting are understood, all reduced to a frequency resolution
in Hz that may vary with time:

* ``window_revs``: window of fixed length in shaft revolutions (angle-domain
  analysis such as :func:`rota.maps.rpm_order_map` or a phase-locked
  short-time extractor). Resolution ``c * rpm / (60 * window_revs)`` Hz: the
  same in orders at every speed.
* ``bandwidth``: Vold-Kalman -3 dB filter bandwidth in Hz. Resolution
  ``VK_SEPARATION_FACTOR * bandwidth`` Hz at every speed.
* ``time_window``: fixed window in seconds (equal-bandwidth STFT). Resolution
  ``c / time_window`` Hz at every speed, so at low speed it spans several
  orders (ROADMAP section 6, items 20 and 21).

Constants (module level, used by the check and documented in
:func:`resolution_check`):

``HANN_MAINLOBE_HALF_WIDTH`` = 2
    Hann window main-lobe half-width in bins (null-to-null width 4 bins). Two
    lines at least this many bins apart are separated down to the Hann
    side-lobe level (-31.5 dB).
``HANN_ENVELOPE_CUTOFF`` = 0.72
    Half of the Hann -3 dB bandwidth (1.44 bins): the envelope modulation
    frequency, in bins, below which a window follows an amplitude change.
``VK_SEPARATION_FACTOR`` = 1
    Required line separation in units of the Vold-Kalman -3 dB bandwidth.
"""

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ._utils import runs

__all__ = [
    "resolution_check",
    "ResolutionReport",
    "ResolutionWarning",
    "HANN_MAINLOBE_HALF_WIDTH",
    "HANN_ENVELOPE_CUTOFF",
    "VK_SEPARATION_FACTOR",
]

HANN_MAINLOBE_HALF_WIDTH = 2.0
HANN_ENVELOPE_CUTOFF = 0.72
VK_SEPARATION_FACTOR = 1.0

# Pairs whose frequencies come within this relative distance count as
# "near equal" (twin shafts, neighbouring orders).
NEAR_EQUAL_RELATIVE = 0.10

_KINDS = ("window_revs", "bandwidth", "time_window")
_UNITS = {"window_revs": "revolutions", "bandwidth": "Hz", "time_window": "s"}
# Relative slack on every resolved / not-resolved comparison, so that a
# setting exactly at the limit (or a rounded recommendation) is resolved.
_TOL = 1e-6
# At most this many time points are evaluated (a strided grid for longer records).
_MAX_POINTS = 50_000


class ResolutionWarning(UserWarning):
    """Issued by :func:`resolution_check` when components are not resolved."""


@dataclass
class ResolutionReport:
    """Outcome of :func:`resolution_check`.

    Attributes
    ----------
    ok : bool
        True if every pair of components is resolved at every evaluated time
        with the checked setting and, if ``envelope_rate`` was given, no
        envelope-following problem was found.
    warnings : list of str
        One human-readable message per problem, naming orders, shafts and
        times (s). Empty if ``ok``.
    notes : list of str
        Informational messages that do not make the check fail, e.g. how
        slowly an envelope may change for a resolved near-equal pair.
    pairs : list of dict
        One entry per pair of components (``i < j``, indices into ``orders``).
        Keys: ``i``, ``j``, ``orders`` (pair of order numbers), ``shafts``
        (pair of shaft columns), ``resolved`` (bool), ``fraction_unresolved``
        (fraction of the evaluated times with separation below the
        resolution), ``n_evaluated``, ``min_separation_hz`` (smallest
        ``|f_i - f_j|``; 0 if the lines cross), ``worst_time`` (s, where
        separation relative to resolution is smallest), ``beat_frequency_hz``
        (``|f_i - f_j|`` there, the beat of the two lines), ``resolution_hz``
        (resolution there), ``unresolved_intervals`` (list of ``(t0, t1)`` in
        s), ``crossings`` (times in s where the lines cross),
        ``near_equal`` (bool), ``max_envelope_rate_hz`` (highest envelope
        modulation frequency a setting that just resolves the pair can
        follow), ``required_window_revs``, ``required_bandwidth_hz`` and
        ``required_time_window_s`` (the coarsest setting of each kind that
        resolves this pair alone; ``inf`` / 0 if none exists). A pair with no
        common active time has ``n_evaluated == 0`` and counts as resolved.
    images : list of dict
        One entry per component: its separation from its own
        negative-frequency image (``2 f``). Keys ``i``, ``order``, ``shaft``
        and the statistics of ``pairs`` (``resolved`` to
        ``required_time_window_s``; ``beat_frequency_hz`` is ``2 f`` at the
        worst time). A line closer to zero frequency than half the resolution
        cannot be told from its image.
    recommended_window_revs : float or None
        Smallest angle-domain window in revolutions that resolves every pair
        and every line from its image everywhere (rounded up to three
        significant figures), or None if there is nothing to resolve or no
        window does (identical frequencies, a crossing, a window longer than
        the record, or an envelope conflict).
    recommended_bandwidth : float or None
        Largest Vold-Kalman -3 dB bandwidth in Hz that does the same (rounded
        down), or None.
    recommended_time_window : float or None
        Shortest fixed window in s that does the same (rounded up), or None.
    setting : dict
        The checked setting, ``{"kind": ..., "value": ...}``.
    orders, shafts : ndarray
        Parsed order numbers and shaft columns.
    """

    ok: bool
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    pairs: list = field(default_factory=list)
    images: list = field(default_factory=list)
    recommended_window_revs: float | None = None
    recommended_bandwidth: float | None = None
    recommended_time_window: float | None = None
    setting: dict = field(default_factory=dict)
    orders: np.ndarray = field(default_factory=lambda: np.empty(0))
    shafts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))


# --------------------------------------------------------------------------
# argument handling
# --------------------------------------------------------------------------


def _parse_orders(orders, shafts, n_shafts):
    """Order numbers and shaft columns as arrays of length K."""
    if isinstance(orders, Mapping) or np.ndim(orders) == 0:
        orders = [orders]
    seq = list(orders)
    if not seq:
        raise ValueError("orders must not be empty")
    is_map = [isinstance(o, Mapping) for o in seq]
    if any(is_map):
        if not all(is_map):
            raise ValueError("orders must be all numbers or all dicts")
        if shafts is not None:
            raise ValueError("give the shaft in the order dicts or in shafts, not both")
        values = [o.get("order") for o in seq]
        shaft_in = [0 if o.get("shaft") is None else o.get("shaft") for o in seq]
    else:
        values = seq
        shaft_in = shafts
    try:
        o = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("orders must be numbers (or dicts with an 'order' key)") from None
    if o.ndim != 1 or not np.all(np.isfinite(o)) or np.any(o <= 0):
        raise ValueError("orders must be a 1-D sequence of positive, finite order numbers")
    k = o.size
    if shaft_in is None:
        s = np.zeros(k, dtype=int)
    else:
        s_arr = np.asarray(shaft_in)
        if s_arr.ndim == 0:
            s_arr = np.full(k, s_arr)
        if s_arr.shape != (k,):
            raise ValueError("shafts must be an integer or have one entry per order")
        try:
            s = s_arr.astype(float)
        except (TypeError, ValueError):
            raise ValueError("shafts must be integers") from None
        if not np.all(s == np.round(s)):
            raise ValueError("shafts must be integers")
        s = s.astype(int)
    if np.any(s < 0) or np.any(s >= n_shafts):
        raise ValueError(f"shafts must be columns of rpm (0..{n_shafts - 1})")
    return o, s


def _parse_setting(window_revs, bandwidth, time_window):
    given = [(k, v) for k, v in zip(_KINDS, (window_revs, bandwidth, time_window)) if v is not None]
    if len(given) != 1:
        raise ValueError(
            "give exactly one of window_revs, bandwidth and time_window "
            f"(got {len(given)})"
        )
    kind, value = given[0]
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{kind} must be a number") from None
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{kind} must be positive and finite")
    return kind, value


def _resolution_hz(kind, value, r_max):
    """Frequency resolution in Hz; ``r_max`` the coarser shaft speed (rpm)."""
    if kind == "window_revs":
        return HANN_MAINLOBE_HALF_WIDTH * r_max / (60.0 * value)
    if kind == "bandwidth":
        return np.full(np.shape(r_max), VK_SEPARATION_FACTOR * value)
    return np.full(np.shape(r_max), HANN_MAINLOBE_HALF_WIDTH / value)


def _round_sig(x, sig, up):
    """Round a positive number to ``sig`` significant figures, up or down.

    Relative noise of 1e-9 is ignored so that exact values such as 4.000...01
    do not jump to the next step.
    """
    e = math.floor(math.log10(x)) - sig + 1
    q = x / 10.0**e
    q = math.ceil(q * (1.0 - 1e-9)) if up else math.floor(q * (1.0 + 1e-9))
    return float(f"{q:d}e{e:d}")  # decimal string: 10.2, not 10.200000000000001


# --------------------------------------------------------------------------
# evaluation of one pair (or one line against its image)
# --------------------------------------------------------------------------


def _evaluate(kind, value, t, d, r_max, r_min, f_max, valid, crossings):
    """Statistics of one separation curve ``d`` (Hz) on the time grid ``t``.

    ``valid`` marks the evaluated samples; the other arrays hold values there
    (NaN elsewhere is fine). ``crossings`` are times where the two lines cross.
    """
    n_eval = int(valid.sum())
    out = {"n_evaluated": n_eval, "crossings": list(crossings)}
    if n_eval == 0:
        out.update(
            resolved=True, fraction_unresolved=0.0, min_separation_hz=float("nan"),
            worst_time=float("nan"), beat_frequency_hz=float("nan"),
            resolution_hz=float("nan"), unresolved_intervals=[], near_equal=False,
            max_envelope_rate_hz=float("inf"), required_window_revs=0.0,
            required_bandwidth_hz=float("inf"), required_time_window_s=0.0,
        )
        return out
    dv, rmax_v, rmin_v = d[valid], r_max[valid], r_min[valid]
    tv = t[valid]
    res = _resolution_hz(kind, value, rmax_v)
    ratio = dv / res
    bad = ratio < 1.0 - _TOL
    cand = np.flatnonzero(ratio <= ratio.min() * (1.0 + 1e-9))
    worst = cand[np.argmin(dv[cand])]  # ties: the smallest separation
    d_min = 0.0 if crossings else float(dv.min())

    bad_full = np.zeros(t.size, dtype=bool)
    bad_full[valid] = bad
    starts, stops = runs(bad_full)
    intervals = [(float(t[a]), float(t[b - 1])) for a, b in zip(starts, stops)]

    c = HANN_MAINLOBE_HALF_WIDTH
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "bandwidth":
            env = dv / (2.0 * VK_SEPARATION_FACTOR)
        elif kind == "time_window":  # same window for both lines
            env = (HANN_ENVELOPE_CUTOFF / c) * dv
        else:  # window_revs: the slower shaft has the longer window, the lower cut-off
            env = (HANN_ENVELOPE_CUTOFF / c) * (rmin_v / rmax_v) * dv
        req_w = float(np.max(c * rmax_v / (60.0 * dv)))
        rel = dv / f_max[valid]
    env_max = 0.0 if crossings else float(env.min())
    rel_min = 0.0 if crossings else float(rel.min())
    out.update(
        resolved=not bool(bad.any()),
        fraction_unresolved=float(bad.mean()),
        min_separation_hz=d_min,
        worst_time=float(tv[worst]),
        beat_frequency_hz=float(dv[worst]),
        resolution_hz=float(res[worst]),
        unresolved_intervals=intervals,
        near_equal=bool(rel_min <= NEAR_EQUAL_RELATIVE),
        max_envelope_rate_hz=env_max,
        required_window_revs=math.inf if d_min == 0.0 else req_w,
        required_bandwidth_hz=d_min / VK_SEPARATION_FACTOR,
        required_time_window_s=math.inf if d_min == 0.0 else c / d_min,
    )
    return out


def _crossing_times(t, d_signed):
    """Times where a signed difference changes sign (linear interpolation)."""
    a, b = d_signed[:-1], d_signed[1:]
    with np.errstate(invalid="ignore"):
        k = np.flatnonzero(a * b < 0)
    if k.size == 0:
        return []
    frac = np.abs(a[k]) / (np.abs(a[k]) + np.abs(b[k]))
    return [float(x) for x in t[k] + frac * (t[k + 1] - t[k])]


# --------------------------------------------------------------------------
# public function
# --------------------------------------------------------------------------


def resolution_check(
    rpm,
    fs,
    orders,
    *,
    shafts=None,
    window_revs=None,
    bandwidth=None,
    time_window=None,
    envelope_rate=None,
):
    """Check whether orders can be separated with a given analysis resolution.

    Every component ``k`` is a line of frequency ``f_k(t) = order_k *
    rpm_{shaft_k}(t) / 60`` Hz. For every pair of components the frequency
    separation ``|f_i(t) - f_j(t)|`` (which is also the beat frequency of the
    two lines) is compared at every time with the effective resolution of the
    analysis, and the report says where, for how long and by how much a pair
    is not resolved, and which setting would resolve all pairs. Use it before
    :func:`rota.maps.rpm_order_map` or an order extractor, in particular for
    half orders at idle, orders of two shafts with nearly equal speed, and
    crossing orders.

    Exactly one resolution setting is given:

    ``window_revs``
        Angle-domain window of ``window_revs`` revolutions of each
        component's own shaft (Hann window, as in :func:`rota.maps.rpm_order_map`
        with ``order_resolution = 1 / window_revs``, or a phase-locked
        short-time extractor). The window lasts ``window_revs * 60 / rpm``
        seconds, so the resolution is ``c / window_revs`` orders at every
        speed, in Hz ``c * rpm / (60 * window_revs)``, with ``c =
        HANN_MAINLOBE_HALF_WIDTH = 2`` the Hann main-lobe half-width in bins
        (first null of the window's transform; lines at least that far apart
        are separated down to the first side lobe, -31.5 dB). For a pair on
        two shafts the coarser (faster) shaft sets the resolution.
    ``bandwidth``
        Vold-Kalman -3 dB bandwidth in Hz, taken as the full width of the pass
        band centred on each order. Two lines are resolved if they are at
        least ``VK_SEPARATION_FACTOR * bandwidth`` apart (factor 1: the other
        line lies at least half a pass-band width outside the edge). The exact
        roll-off depends on the filter order and weighting, so this is a rule
        of thumb, constant in Hz at every speed.
    ``time_window``
        Fixed window of ``time_window`` seconds (equal-bandwidth STFT, Hann).
        Resolution ``c / time_window`` Hz, constant in Hz. At low speed the
        line spacing in Hz shrinks, so the window spans several orders and
        neighbouring (half) orders merge, which an angle window of the same
        duration at top speed resolves.

    A pair is resolved at a time if its separation is at least the resolution.
    Lines are evaluated only where their shaft speed is positive and finite and
    their frequency is below ``fs / 2``. Each line is also checked against its
    own negative-frequency image (separation ``2 f``), which matters near zero
    frequency. Records longer than 50 000 samples are evaluated on a strided
    time grid; crossings are located by interpolation.

    Envelope variation (bias-variance wall). A narrow analysis separates
    near-equal lines but cannot follow fast envelope changes, a wide one
    follows the envelope but lets the neighbour leak in. A setting that just
    resolves a pair follows envelope changes only up to
    ``max_envelope_rate_hz``: for ``bandwidth`` the envelope cut-off is half
    the -3 dB bandwidth, so at most ``separation / (2 * VK_SEPARATION_FACTOR)``;
    for the windows it is ``HANN_ENVELOPE_CUTOFF / c * separation`` (0.36
    times the separation), where ``HANN_ENVELOPE_CUTOFF = 0.72`` is half the
    Hann -3 dB bandwidth of 1.44 bins. For twin shafts differing by
    0.5 % at 100 Hz this is 0.25 Hz or less: amplitude changes faster than
    that (resonance crossings, modulations) are smoothed. If the highest
    envelope modulation frequency expected is given as ``envelope_rate``, pairs
    whose bound is below it are reported as a wall (no setting both resolves
    them and follows the envelope), and a setting that is itself too narrow to
    follow it is reported too.

    Parameters
    ----------
    rpm : array_like, shape (n,) or (n, n_shafts)
        Speed in rpm at sampling rate ``fs``, one column per shaft. Samples
        that are not finite or not positive are treated as inactive.
    fs : float
        Sampling rate of ``rpm`` in Hz.
    orders : sequence of float, or sequence of dict
        Order numbers (cycles per revolution of the component's shaft,
        positive, may be fractional). Alternatively dicts with an ``"order"``
        key and an optional ``"shaft"`` key, as in :func:`rota.synth.synth_runup`.
    shafts : int or sequence of int, optional
        Column of ``rpm`` of each order (default: all on column 0).
    window_revs : float, optional
        Angle-domain window length in revolutions (see above).
    bandwidth : float, optional
        Vold-Kalman -3 dB bandwidth in Hz.
    time_window : float, optional
        Fixed time window in s.
    envelope_rate : float, optional
        Highest envelope modulation frequency in Hz that the extraction must
        follow (for example a modulation frequency, or the inverse of the time
        an order takes to sweep through a resonance).

    Returns
    -------
    ResolutionReport
        See its attributes. If ``ok`` is False, a
        :class:`ResolutionWarning` (a :class:`UserWarning`) is also issued with
        the messages and the recommended settings. The recommended settings
        are the coarsest of each kind that resolve every pair and every line
        from its image at every time, and re-running the check with one of
        them gives ``ok == True``; they are None if no setting does, in
        particular for identical or crossing frequencies, for a window longer
        than the record, for a conflict with ``envelope_rate``, and when there
        is nothing to resolve (a single line is only checked against its
        image).

    Raises
    ------
    ValueError
        For anything other than exactly one setting, non-positive values,
        invalid orders or shafts, an ``rpm`` of the wrong shape, or no active
        component.

    Notes
    -----
    No amplitude is involved: only frequencies (Hz), orders (cycles per
    revolution), speeds (rpm) and times (s). The check assumes lines that are
    stationary within a window; it does not model the smearing of a line that
    sweeps quickly through the window.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 9 and 12 (frequency
    resolution, equivalent noise bandwidth, order tracking). Wiley.
    Harris, F. J. (1978). On the use of windows for harmonic analysis with the
    discrete Fourier transform. Proceedings of the IEEE, 66(1), 51-83.
    Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order
    tracking. Mechanical Systems and Signal Processing, 11(2), 187-205.
    Tuma, J. (2005). The passband width of the Vold-Kalman order tracking
    filter. Sbornik vedeckych praci VSB-TU Ostrava, Rada strojni, 51(2),
    149-154.
    Pan, M.-C., & Lin, Y.-F. (2006). Further exploration of Vold-Kalman-filtering
    order tracking with shaft-speed information - I: Theoretical part,
    numerical implementation and parameter investigations. Mechanical Systems
    and Signal Processing, 20(5), 1134-1154.
    """
    fs = float(fs)
    if not (math.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a positive sampling rate in Hz")
    R_in = np.asarray(rpm, dtype=float)
    if R_in.ndim not in (1, 2):
        raise ValueError("rpm must have shape (n,) or (n, n_shafts)")
    if R_in.ndim == 1:
        R_in = R_in[:, None]
    n, n_shafts = R_in.shape
    if n < 2:
        raise ValueError("rpm must have at least 2 samples")
    o_arr, s_arr = _parse_orders(orders, shafts, n_shafts)
    kind, value = _parse_setting(window_revs, bandwidth, time_window)
    if envelope_rate is not None:
        envelope_rate = float(envelope_rate)
        if not (math.isfinite(envelope_rate) and envelope_rate > 0):
            raise ValueError("envelope_rate must be positive and finite (Hz)")

    stride = max(1, -(-n // _MAX_POINTS))
    idx = np.arange(0, n, stride)
    t = idx / fs
    R = R_in[idx]
    duration = (n - 1) / fs
    K = o_arr.size

    # Component frequencies and the samples where they are observable.
    freq = np.full((t.size, K), np.nan)
    speed = np.full((t.size, K), np.nan)
    active = np.zeros((t.size, K), dtype=bool)
    for k in range(K):
        r = R[:, s_arr[k]]
        f = o_arr[k] * r / 60.0
        with np.errstate(invalid="ignore"):
            a = np.isfinite(r) & (r > 0) & (f < 0.5 * fs)
        active[:, k] = a
        freq[a, k] = f[a]
        speed[a, k] = r[a]
    if not active.any():
        raise ValueError(
            "no component is active: speeds must be positive and finite and "
            "the frequencies below fs / 2 somewhere"
        )

    def label(k):
        return f"order {o_arr[k]:g} (shaft {s_arr[k]})"

    notes = []
    for k in range(K):
        if not active[:, k].any():
            notes.append(
                f"{label(k)} is never active (speed not positive, not finite, or "
                "frequency at or above fs / 2) and is not checked"
            )

    pairs = []
    for i, j in combinations(range(K), 2):
        valid = active[:, i] & active[:, j]
        sd = freq[:, i] - freq[:, j]
        d = np.abs(sd)
        r_max = np.fmax(speed[:, i], speed[:, j])
        r_min = np.fmin(speed[:, i], speed[:, j])
        f_max = np.fmax(freq[:, i], freq[:, j])
        crossings = _crossing_times(t, sd)
        p = _evaluate(kind, value, t, d, r_max, r_min, f_max, valid, crossings)
        p.update(
            i=i, j=j,
            orders=(float(o_arr[i]), float(o_arr[j])),
            shafts=(int(s_arr[i]), int(s_arr[j])),
        )
        pairs.append(p)

    images = []
    for k in range(K):
        valid = active[:, k]
        p = _evaluate(
            kind, value, t, 2.0 * freq[:, k], speed[:, k], speed[:, k], freq[:, k],
            valid, [],
        )
        p.pop("crossings")
        p.update(i=k, order=float(o_arr[k]), shaft=int(s_arr[k]))
        images.append(p)

    # ---- messages --------------------------------------------------------
    msgs = []
    unit = _UNITS[kind]
    setting_txt = f"{kind}={value:g} {unit}"

    def intervals_txt(p, limit=3):
        iv = p["unresolved_intervals"]
        txt = ", ".join(f"{a:.2f}-{b:.2f} s" for a, b in iv[:limit])
        return txt + (f" and {len(iv) - limit} more" if len(iv) > limit else "")

    for p in pairs:
        if p["resolved"]:
            continue
        i, j = p["i"], p["j"]
        text = (
            f"{label(i)} and {label(j)} are not resolved with {setting_txt} during "
            f"{100.0 * p['fraction_unresolved']:.3g} % of the record "
            f"({intervals_txt(p)}); worst at t = {p['worst_time']:.2f} s: separation "
            f"(beat frequency) {p['beat_frequency_hz']:.4g} Hz against a resolution "
            f"of {p['resolution_hz']:.4g} Hz"
        )
        if p["crossings"]:
            cr = ", ".join(f"{c:.2f}" for c in p["crossings"][:3])
            text += f"; the lines cross at t = {cr} s"
        elif p["min_separation_hz"] == 0.0:
            text += "; the frequencies are identical"
        msgs.append(text)
    for p in images:
        if p["resolved"]:
            continue
        msgs.append(
            f"{label(p['i'])} is not separated from its negative-frequency image with "
            f"{setting_txt} during {100.0 * p['fraction_unresolved']:.3g} % of the "
            f"record ({intervals_txt(p)}); worst at t = {p['worst_time']:.2f} s: "
            f"2 f = {p['beat_frequency_hz']:.4g} Hz against a resolution of "
            f"{p['resolution_hz']:.4g} Hz"
        )

    # Envelope following (bias-variance wall).
    walls = []
    if envelope_rate is not None:
        walls = [p for p in pairs if p["n_evaluated"] and p["max_envelope_rate_hz"] < envelope_rate * (1 - _TOL)]
        for p in walls:
            msgs.append(
                f"{label(p['i'])} and {label(p['j'])}: separating them limits the "
                f"extraction to envelope changes slower than "
                f"{p['max_envelope_rate_hz']:.4g} Hz, below the required "
                f"{envelope_rate:g} Hz (beat frequency down to "
                f"{p['min_separation_hz']:.4g} Hz): bias-variance wall, no "
                "setting both resolves the pair and follows the envelope"
            )
        if kind == "bandwidth":
            setting_env = 0.5 * value
        elif kind == "time_window":
            setting_env = HANN_ENVELOPE_CUTOFF / value
        else:
            setting_env = HANN_ENVELOPE_CUTOFF * float(np.nanmin(speed)) / (60.0 * value)
        if setting_env < envelope_rate * (1 - _TOL):
            msgs.append(
                f"{setting_txt} follows envelope changes only up to "
                f"{setting_env:.4g} Hz, below the required {envelope_rate:g} Hz: "
                "amplitude changes faster than that are smoothed (bias)"
            )

    for p in pairs:
        if p["n_evaluated"] and p["near_equal"] and p["resolved"]:
            rate = p["max_envelope_rate_hz"]
            notes.append(
                f"{label(p['i'])} and {label(p['j'])} come within "
                f"{100.0 * NEAR_EQUAL_RELATIVE:g} % in frequency (beat down to "
                f"{p['min_separation_hz']:.4g} Hz): resolving them limits the "
                f"extraction to envelope changes slower than {rate:.4g} Hz (time "
                f"scale {1.0 / rate:.3g} s); pass envelope_rate to check this "
                "against the expected envelope variation"
            )

    # ---- recommendation ---------------------------------------------------
    rec_w = rec_b = rec_t = None
    items = [p for p in pairs + images if p["n_evaluated"]]
    if items and not walls:
        w = max(p["required_window_revs"] for p in items)
        b = min(p["required_bandwidth_hz"] for p in items)
        tw = max(p["required_time_window_s"] for p in items)
        rev_total = [
            float(np.sum(np.where(np.isfinite(R[:, s]) & (R[:, s] > 0), R[:, s], 0.0)) * stride / fs / 60.0)
            for s in sorted(set(int(s) for s in s_arr))
        ]
        if math.isfinite(w) and w > 0 and w <= min(rev_total):
            rec_w = _round_sig(w, 3, up=True)
        if b > 0 and b * duration >= 1.0:
            rec_b = _round_sig(b, 3, up=False)
        if math.isfinite(tw) and tw <= duration:
            rec_t = _round_sig(tw, 3, up=True)
        if envelope_rate is not None:
            # The recommendation must itself follow the envelope.
            floor = envelope_rate * (1 - _TOL)
            if rec_w is not None and (
                HANN_ENVELOPE_CUTOFF * float(np.nanmin(speed)) / (60.0 * rec_w) < floor
            ):
                rec_w = None
            if rec_b is not None and 0.5 * rec_b < floor:
                rec_b = None
            if rec_t is not None and HANN_ENVELOPE_CUTOFF / rec_t < floor:
                rec_t = None

    ok = not msgs
    report = ResolutionReport(
        ok=ok,
        warnings=msgs,
        notes=notes,
        pairs=pairs,
        images=images,
        recommended_window_revs=rec_w,
        recommended_bandwidth=rec_b,
        recommended_time_window=rec_t,
        setting={"kind": kind, "value": value},
        orders=o_arr,
        shafts=s_arr,
    )
    if not ok:
        rec = []
        if rec_w is not None:
            rec.append(f"window_revs >= {rec_w:g} revolutions (angle domain)")
        if rec_b is not None:
            rec.append(f"bandwidth <= {rec_b:g} Hz (Vold-Kalman)")
        if rec_t is not None:
            rec.append(f"time_window >= {rec_t:g} s (fixed window)")
        tail = (
            "Recommended: " + "; or ".join(rec)
            if rec
            else "No setting resolves all components everywhere (identical or crossing "
            "frequencies, a window longer than the record, or an envelope conflict)"
        )
        text = "\n  - ".join(
            [f"resolution_check: {len(msgs)} problem(s) with {setting_txt}"] + msgs + [tail]
        )
        warnings.warn(text, ResolutionWarning, stacklevel=2)
    return report
