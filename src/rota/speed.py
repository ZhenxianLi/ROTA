"""Speed references: tachometer pulses, shaft phase and control signals.

Units: time in s, rotational speed in rev/min (rpm), shaft angle in
revolutions, vehicle speed in km/h.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from ._utils import runs, time_axis

__all__ = [
    "rpm_to_phase",
    "tacho_to_rpm",
    "regularize_control",
    "speed_to_rpm",
    "virtual_rpm",
    "fit_speed_to_rpm",
    "steady_segments",
    "Segment",
    "rpm_from_ridge",
    "RidgeResult",
    "WhiteningWarning",
    "RidgeQualityWarning",
]


def rpm_to_phase(rpm, fs=None, *, t=None, delay=0.0, min_rpm=None, rev0=0.0):
    """Integrate a speed curve to cumulative shaft angle.

    The angle is ``rev(t) = rev0 + integral of rpm(t)/60 dt``, evaluated with
    the trapezoidal rule on the sample grid, so it is exact for speed curves
    that are piecewise linear between samples.

    Parameters
    ----------
    rpm : array_like, shape (n,) or (n, k)
        Rotational speed in rev/min, one column per shaft.
    fs : float, optional
        Sampling rate in Hz of a uniformly sampled ``rpm``.
    t : array_like, shape (n,), optional
        Sample times in s (instead of ``fs``).
    delay : float, default 0
        Delay in s by which the speed reference lags the signal it is used
        with (positive when the reference is late, e.g. tachometer
        conditioning delay). Compensated by evaluating ``rpm`` at
        ``t + delay``, holding the end values.
    min_rpm : float, optional
        Lower bound applied to ``rpm`` before integration. Prevents a
        stationary or reversing reference from producing a non-increasing
        angle, which angle-domain resampling cannot invert.
    rev0 : float, default 0
        Angle at the first sample, in revolutions.

    Returns
    -------
    rev : ndarray, same shape as ``rpm``
        Cumulative shaft angle in revolutions. Multiply by ``2*pi*order`` to
        obtain the phase in rad of a given order. No amplitudes are involved.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis: Signal Analysis and
    Experimental Procedures, ch. 12. Wiley.
    """
    rpm = np.asarray(rpm, dtype=float)
    if rpm.ndim not in (1, 2):
        raise ValueError("rpm must have shape (n,) or (n, k)")
    if not np.all(np.isfinite(rpm)):
        raise ValueError("rpm contains non-finite values; regularise it first")
    tt = time_axis(rpm.shape[0], fs, t)
    if delay:
        tq = tt + delay
        if rpm.ndim == 1:
            rpm = np.interp(tq, tt, rpm)
        else:
            rpm = np.column_stack([np.interp(tq, tt, c) for c in rpm.T])
    if min_rpm is not None:
        rpm = np.maximum(rpm, float(min_rpm))
    dt = np.diff(tt)
    if rpm.ndim == 2:
        dt = dt[:, None]
    inc = 0.5 * (rpm[1:] + rpm[:-1]) / 60.0 * dt
    rev = np.empty_like(rpm)
    rev[0] = rev0
    np.cumsum(inc, axis=0, out=rev[1:])
    rev[1:] += rev0
    return rev


def _schmitt_edges(x, fs, level, hysteresis):
    """Rising-edge times (s) with sub-sample linear interpolation."""
    hi = level + 0.5 * hysteresis
    lo = level - 0.5 * hysteresis
    state = np.zeros(x.shape, dtype=np.int8)
    state[x >= hi] = 1
    state[x <= lo] = -1
    idx = np.where(state != 0, np.arange(x.size), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = state[idx]
    trans = np.flatnonzero((filled[1:] == 1) & (filled[:-1] == -1)) + 1
    cross = np.flatnonzero((x[:-1] < level) & (x[1:] >= level))
    j = np.searchsorted(cross, trans, side="left") - 1
    c = np.unique(cross[j[j >= 0]])
    frac = (level - x[c]) / (x[c + 1] - x[c])
    return (c + frac) / fs


def tacho_to_rpm(
    tacho,
    fs,
    ppr=1,
    *,
    level=None,
    hysteresis=None,
    edge="rising",
    smooth=1.0,
    t_out=None,
    gap_factor=1.5,
    return_info=False,
):
    """Convert a tachometer pulse signal to a speed curve.

    Pulse edges are found with a Schmitt trigger and located to sub-sample
    precision by linear interpolation of the threshold crossing. Missing
    pulses (intervals close to an integer multiple of the local median
    interval) are filled in, and spurious pulses (intervals much shorter than
    the local median) are discarded, so that each remaining pulse has a known
    shaft angle ``k/ppr``. Pulse time is then treated as a function of angle,
    ``t(rev)``, sampled uniformly in angle; its derivative is estimated with a
    local quadratic (Savitzky-Golay) fit, which also averages pulse timing
    jitter, and ``rpm = 60 / (dt/drev)``.

    Parameters
    ----------
    tacho : array_like, shape (n,)
        Tachometer signal (any unit).
    fs : float
        Sampling rate in Hz.
    ppr : int, default 1
        Pulses per revolution of the shaft of interest.
    level : float, optional
        Trigger level. Default: midway between the 1st and 99th percentiles.
    hysteresis : float, optional
        Schmitt-trigger hysteresis (full width). Default: 20 % of the
        1st-99th percentile range.
    edge : {"rising", "falling"}
        Edge used as pulse time.
    smooth : float, default 1.0
        Length in revolutions of the local fit window. The window holds at
        least 5 pulses. Larger values suppress jitter but smooth fast speed
        changes. Use 0 for a plain central difference (no smoothing).
    t_out : array_like, optional
        Output times in s. Default: the sample times of ``tacho``. Outside the
        span of detected pulses the first or last estimate is held.
    gap_factor : float, default 1.5
        Interval ratio (to the local median) above which pulses are treated as
        missing and below the inverse of which a pulse is treated as spurious.
    return_info : bool, default False
        Also return a dict with ``pulse_times`` (s), ``pulse_revs`` (shaft
        angle of each detected pulse, rev), ``n_missing``, ``n_spurious`` and
        ``extrapolated`` (bool mask over ``t_out``).

    Returns
    -------
    rpm : ndarray
        Speed in rev/min at ``t_out``. The tachometer amplitude only sets the
        trigger levels; no amplitude convention applies.
    info : dict
        Only if ``return_info``.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, sec. 12.2 (speed from
    tachometer signals). Wiley.
    Savitzky, A., & Golay, M. J. E. (1964). Smoothing and differentiation of
    data by simplified least squares procedures. Analytical Chemistry,
    36(8), 1627-1639.
    """
    x = np.asarray(tacho, dtype=float).ravel()
    if edge not in ("rising", "falling"):
        raise ValueError("edge must be 'rising' or 'falling'")
    if ppr < 1 or int(ppr) != ppr:
        raise ValueError("ppr must be a positive integer")
    p_lo, p_hi = np.percentile(x, [1, 99])
    if level is None:
        level = 0.5 * (p_lo + p_hi)
    if hysteresis is None:
        hysteresis = 0.2 * (p_hi - p_lo)
    if edge == "falling":
        x, level = -x, -level
    te = _schmitt_edges(x, fs, level, hysteresis)
    if te.size < 4:
        raise ValueError("fewer than 4 pulses detected; check level and hysteresis")

    # Discard spurious pulses, then count missing ones.
    n_spurious = 0
    while te.size >= 4:
        d = np.diff(te)
        ref = ndimage.median_filter(d, size=min(9, d.size), mode="nearest")
        short = np.flatnonzero(d < ref / gap_factor)
        if short.size == 0:
            break
        # Interval i is too short: drop whichever of its two pulses leaves the
        # merged interval closest to the local median.
        i = int(short[0])
        cost = []
        for drop, lo, hi in ((i + 1, i, i + 2), (i, i - 1, i + 1)):
            ok = 0 <= lo and hi < te.size
            cost.append(abs(np.log((te[hi] - te[lo]) / ref[i])) if ok else np.inf)
        te = np.delete(te, i + 1 if cost[0] <= cost[1] else i)
        n_spurious += 1
    d = np.diff(te)
    ref = ndimage.median_filter(d, size=min(9, d.size), mode="nearest")
    m = np.where(d > gap_factor * ref, np.rint(d / ref), 1.0).astype(int)
    counts = np.concatenate(([0], np.cumsum(m)))
    n_missing = int(counts[-1] - (te.size - 1))

    full = np.arange(counts[-1] + 1)
    t_full = np.interp(full, counts, te)
    drev = 1.0 / ppr
    win = int(round(smooth * ppr)) if smooth else 0
    if win > 0:
        win = max(win, 5)
        win += 1 - win % 2
        win = min(win, full.size - (1 - full.size % 2))
    if win >= 5:
        dtdrev = signal.savgol_filter(t_full, win, 2, deriv=1, delta=drev, mode="interp")
    else:
        dtdrev = np.gradient(t_full, drev)
    rpm_p = 60.0 / dtdrev

    if t_out is None:
        t_out = np.arange(x.size) / fs
    t_out = np.asarray(t_out, dtype=float)
    rpm = np.interp(t_out, t_full, rpm_p)
    if not return_info:
        return rpm
    info = {
        "pulse_times": te,
        "pulse_revs": counts * drev,
        "n_missing": n_missing,
        "n_spurious": n_spurious,
        "extrapolated": (t_out < t_full[0]) | (t_out > t_full[-1]),
    }
    return rpm, info


def regularize_control(
    x,
    t=None,
    fs=None,
    *,
    fs_out=100.0,
    t_out=None,
    offset=0.0,
    despike_tol=None,
    despike_window=1.5,
    max_gap=None,
    antialias=True,
):
    """Clean a slow control signal and resample it to a regular time grid.

    Intended for rpm, vehicle speed, throttle and similar channels that are
    logged at their own rate (possibly irregular, possibly an integer
    fraction of the acoustic rate) and must be put on a common time axis.

    Steps: (1) shift times by ``offset``; (2) optional despiking, which marks
    as missing every sample deviating from a running median by more than
    ``despike_tol`` times the median's magnitude; (3) gaps (NaN) are bridged
    by linear interpolation for filtering only; (4) if the input rate exceeds
    ``fs_out``, a zero-phase 4th-order Butterworth low-pass at
    ``0.4 * fs_out`` is applied; (5) linear interpolation onto the output
    grid; (6) output samples farther than ``max_gap`` from a valid input
    sample, or outside the input span, are set to NaN.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Control signal(s), in their physical unit. NaN marks missing samples.
    t, fs : array_like or float
        Input sample times in s, or input sampling rate in Hz.
    fs_out : float, default 100
        Output rate in Hz. The default grid holds the multiples of
        ``1/fs_out`` inside the input span, so it lines up with an acoustic
        time axis that starts at 0.
    t_out : array_like, optional
        Explicit output times in s (overrides ``fs_out``).
    offset : float, default 0
        Added to the input times, e.g. a measured inter-system delay.
    despike_tol : float, optional
        Relative tolerance for despiking (e.g. 0.07). None disables it.
    despike_window : float, default 1.5
        Running-median window in s for despiking.
    max_gap : float, optional
        Largest distance in s to a valid input sample for an output sample to
        be kept. Default: twice the larger of the median input interval and
        ``1/fs_out``.
    antialias : bool, default True
        Apply the low-pass of step (4).

    Returns
    -------
    t_out : ndarray
        Output times in s.
    y : ndarray
        Resampled signal, same unit as ``x``; NaN in gaps. Values are
        resampled, not scaled, so no amplitude convention applies.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 3 (anti-aliasing and
    resampling). Wiley.
    """
    x = np.asarray(x, dtype=float)
    one_d = x.ndim == 1
    X = (x[:, None] if one_d else x).copy()
    n = X.shape[0]
    tt = time_axis(n, fs, t) + offset
    dt_med = float(np.median(np.diff(tt)))

    if despike_tol is not None:
        w = max(3, int(round(despike_window / dt_med)) | 1)
        for c in range(X.shape[1]):
            col = X[:, c]
            ok = np.isfinite(col)
            if ok.sum() < 2:
                continue
            filled = np.interp(tt, tt[ok], col[ok])
            med = ndimage.median_filter(filled, size=w, mode="nearest")
            col[np.abs(col - med) > despike_tol * np.abs(med)] = np.nan

    if t_out is None:
        k0 = int(np.ceil(tt[0] * fs_out - 1e-9))
        k1 = int(np.floor(tt[-1] * fs_out + 1e-9))
        t_out = np.arange(k0, k1 + 1) / fs_out
    t_out = np.asarray(t_out, dtype=float)
    if max_gap is None:
        max_gap = 2.0 * max(dt_med, 1.0 / fs_out)

    Y = np.full((t_out.size, X.shape[1]), np.nan)
    fs_in = 1.0 / dt_med
    for c in range(X.shape[1]):
        col = X[:, c]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            continue
        tv = tt[ok]
        if antialias and fs_in > fs_out * 1.0001:
            tu = tt[0] + np.arange(int(np.floor((tt[-1] - tt[0]) * fs_in)) + 1) / fs_in
            u = np.interp(tu, tv, col[ok])
            sos = signal.butter(4, 0.4 * fs_out, fs=fs_in, output="sos")
            if u.size > 3 * 2 * sos.shape[0]:
                u = signal.sosfiltfilt(sos, u)
            yc = np.interp(t_out, tu, u)
        else:
            yc = np.interp(t_out, tv, col[ok])
        i = np.clip(np.searchsorted(tv, t_out), 1, tv.size - 1)
        dist = np.minimum(np.abs(t_out - tv[i - 1]), np.abs(tv[i] - t_out))
        bad = (dist > max_gap) | (t_out < tv[0] - 1e-12) | (t_out > tv[-1] + 1e-12)
        yc[bad] = np.nan
        Y[:, c] = yc
    return t_out, (Y[:, 0] if one_d else Y)


def speed_to_rpm(
    speed_kmh,
    *,
    rpm_per_kmh=None,
    wheel_circumference=None,
    gear_ratios=None,
    final_drive=1.0,
    gear=None,
    shift_speeds=None,
    idle=None,
    redline=None,
):
    """Convert vehicle speed to engine or motor speed.

    Two models:

    * Lumped slope: ``rpm = rpm_per_kmh * v``. Suits fixed-ratio drives
      (single-speed electric drives) or a single gear.
    * Driveline chain: ``rpm = v / 3.6 / C * 60 * G[g] * final_drive``, with
      wheel circumference ``C`` in m and gear ratio ``G[g]`` of the engaged
      gear ``g`` (1-based; 0 means neutral and yields ``idle``).

    In both models the result is clipped to ``[idle, redline]`` when these
    are given, which models clutch slip at launch and the speed limiter.

    Parameters
    ----------
    speed_kmh : array_like
        Vehicle speed in km/h.
    rpm_per_kmh : float, optional
        Slope of the lumped model in rpm per km/h.
    wheel_circumference : float, optional
        Rolling circumference in m (chain model).
    gear_ratios : sequence of float, optional
        Gear ratios, first gear first (chain model).
    final_drive : float, default 1
        Final-drive ratio (chain model).
    gear : int or array_like, optional
        Engaged gear (1-based, 0 = neutral), scalar or per sample.
    shift_speeds : sequence of float, optional
        Upshift speeds in km/h, ``len(gear_ratios) - 1`` values; the gear is
        then selected from speed (used when ``gear`` is None).
    idle, redline : float, optional
        Lower and upper rpm limits.

    Returns
    -------
    rpm : ndarray
        Rotational speed in rev/min. No amplitudes are involved.

    References
    ----------
    Gillespie, T. D. (1992). Fundamentals of Vehicle Dynamics, ch. 2
    (driveline ratios and tyre rolling radius). SAE International.
    """
    v = np.asarray(speed_kmh, dtype=float)
    if rpm_per_kmh is not None:
        rpm = rpm_per_kmh * v
    else:
        if wheel_circumference is None or gear_ratios is None:
            raise ValueError("give rpm_per_kmh, or wheel_circumference and gear_ratios")
        ratios = np.concatenate(([0.0], np.asarray(gear_ratios, dtype=float)))
        if gear is None:
            if shift_speeds is None:
                if len(gear_ratios) != 1:
                    raise ValueError("give gear or shift_speeds for a multi-speed gearbox")
                g = np.ones(v.shape, dtype=int)
            else:
                if len(shift_speeds) != len(gear_ratios) - 1:
                    raise ValueError("shift_speeds needs len(gear_ratios) - 1 values")
                g = np.searchsorted(np.asarray(shift_speeds, float), v, side="right") + 1
        else:
            g = np.broadcast_to(np.asarray(gear, dtype=int), v.shape)
        if np.any((g < 0) | (g >= ratios.size)):
            raise ValueError("gear out of range")
        rpm = v / 3.6 / wheel_circumference * 60.0 * ratios[g] * final_drive
        if idle is not None:
            rpm = np.where(g == 0, idle, rpm)
    if idle is not None or redline is not None:
        rpm = np.clip(rpm, idle if idle is not None else -np.inf,
                      redline if redline is not None else np.inf)
    return rpm


def virtual_rpm(speed_kmh, rpm_per_kmh, *, idle, redline, smooth=None, fs=None):
    """Virtual engine or motor speed from vehicle speed, for listening studies.

    Decouples the sound from the vehicle dynamics: whatever the real
    driveline did, the stimulus follows ``rpm = clip(rpm_per_kmh * v, idle,
    redline)``, a single-slope map with the engine held at ``idle`` at
    standstill and at the ``redline`` limiter. The result drives order
    synthesis or :func:`rota.angle.stationarize` as an ordinary speed curve.

    Smoothing. With ``smooth`` the clipped target ``r`` is passed through a
    causal first-order low-pass (one pole), the engine-inertia model a
    real-time sound engine would run sample by sample:
    ``y[n] = a y[n-1] + (1 - a) r[n]`` with ``a = exp(-1 / (smooth fs))``.
    A step in the target is followed as ``1 - exp(-t / smooth)`` (the
    recursion equals the continuous response at the end of each sample
    interval, for a target held constant over the interval), so ``smooth`` is
    the time to reach 63.2 % of a step. The filter is causal, not zero-phase:
    the output lags a ramp by ``smooth - 0.5 / fs`` seconds (about
    ``smooth``), as an inertial engine would. It starts at ``r[0]`` (no
    start-up transient) and, being a convex combination of values within
    ``[idle, redline]``, never leaves that range. Clipping is applied before
    smoothing, so the limits are respected without a kink.

    Parameters
    ----------
    speed_kmh : array_like, shape (), (n,) or (n, ch)
        Vehicle speed in km/h (not m/s). Smoothing runs along the first axis,
        channel by channel. NaN marks a gap: it stays NaN in the output and is
        bridged by linear interpolation (ends held) for the smoothing only.
        Negative speeds (reversing) map to ``idle``.
    rpm_per_kmh : float
        Slope in rpm per km/h, positive (e.g. the value fitted by
        :func:`fit_speed_to_rpm`).
    idle, redline : float
        Lower and upper rpm limits, ``0 <= idle <= redline`` (``redline`` may
        be ``inf`` for no upper limit).
    smooth : float, optional
        Time constant in s of the causal first-order smoothing (None or 0:
        none). Requires ``fs``.
    fs : float, optional
        Sampling rate in Hz of ``speed_kmh``; used only for smoothing.

    Returns
    -------
    rpm : ndarray, same shape as ``speed_kmh``
        Rotational speed in rev/min. No amplitudes are involved.

    References
    ----------
    Gillespie, T. D. (1992). Fundamentals of Vehicle Dynamics, ch. 2
    (driveline ratios and rolling radius). SAE International.
    Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal
    Processing, 3rd ed., ch. 2 (linear constant-coefficient difference
    equations; first-order recursive systems). Pearson.
    """
    k = float(rpm_per_kmh)
    if not (np.isfinite(k) and k > 0):
        raise ValueError("rpm_per_kmh must be a positive slope in rpm per km/h")
    idle, redline = float(idle), float(redline)
    if not (np.isfinite(idle) and idle >= 0):
        raise ValueError("idle must be a finite, non-negative speed in rpm")
    if np.isnan(redline) or redline < idle:
        raise ValueError("redline must be a speed in rpm not below idle")
    v = np.asarray(speed_kmh, dtype=float)
    if v.ndim > 2:
        raise ValueError("speed_kmh must have shape (), (n,) or (n, ch)")
    rpm = np.clip(k * v, idle, redline)
    if smooth is None or smooth == 0:
        return rpm
    smooth = float(smooth)
    if not (np.isfinite(smooth) and smooth > 0):
        raise ValueError("smooth must be a non-negative time constant in s")
    if fs is None or not (np.isfinite(fs) and fs > 0):
        raise ValueError("smooth needs the sampling rate fs in Hz")
    if rpm.ndim == 0:
        raise ValueError("smooth needs a time series, not a scalar speed")
    alpha = float(np.exp(-1.0 / (smooth * fs)))
    b, a = [1.0 - alpha], [1.0, -alpha]
    cols = rpm[:, None] if rpm.ndim == 1 else rpm
    out = np.full(cols.shape, np.nan)
    idx = np.arange(cols.shape[0])
    for c in range(cols.shape[1]):
        r = cols[:, c]
        ok = np.isfinite(r)
        if not ok.any():
            continue
        if not ok.all():
            r = np.interp(idx, idx[ok], r[ok])
        # Filter the deviation from the first value with zero initial state: a
        # constant input then passes exactly, whatever the SciPy version.
        y = r[0] + signal.lfilter(b, a, r - r[0])
        y[~ok] = np.nan
        out[:, c] = y
    return out[:, 0] if rpm.ndim == 1 else out


def fit_speed_to_rpm(speed_kmh, rpm, mask=None):
    """Fit the lumped slope ``rpm = k * v`` by least squares through the origin.

    Parameters
    ----------
    speed_kmh, rpm : array_like
        Paired vehicle speed (km/h) and rotational speed (rpm), e.g. the
        medians of steady segments. NaN pairs are ignored.
    mask : array_like of bool, optional
        Pairs to use.

    Returns
    -------
    dict
        ``rpm_per_kmh`` (k = sum(v*rpm)/sum(v**2)), ``residual_rms`` (rpm),
        ``max_rel_residual`` (largest |rpm - k v| / rpm) and ``n``.
        No amplitudes are involved.

    References
    ----------
    Draper, N. R., & Smith, H. (1998). Applied Regression Analysis, 3rd ed.,
    sec. 3.4 (regression through the origin). Wiley.
    """
    v = np.asarray(speed_kmh, dtype=float).ravel()
    r = np.asarray(rpm, dtype=float).ravel()
    ok = np.isfinite(v) & np.isfinite(r)
    if mask is not None:
        ok &= np.asarray(mask, bool).ravel()
    v, r = v[ok], r[ok]
    if v.size == 0 or not np.any(v):
        raise ValueError("no usable speed values")
    k = float(np.dot(v, r) / np.dot(v, v))
    res = r - k * v
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(res) / np.abs(r)
    return {
        "rpm_per_kmh": k,
        "residual_rms": float(np.sqrt(np.mean(res**2))),
        "max_rel_residual": float(np.nanmax(rel)) if rel.size else np.nan,
        "n": int(v.size),
    }


@dataclass
class Segment:
    """A steady segment of a control signal.

    ``start`` and ``stop`` are times in s; ``i_start``/``i_stop`` are sample
    indices (stop exclusive). ``mean`` and ``std`` describe the control signal
    inside the segment.
    """

    target: float
    start: float
    stop: float
    i_start: int
    i_stop: int
    mean: float
    std: float

    @property
    def duration(self):
        return self.stop - self.start


def steady_segments(
    x, targets, tolerance, *, t=None, fs=None, relative=False, min_duration=0.0, length=None
):
    """Find the steady segment of a control signal around each target value.

    For each target the longest contiguous run with
    ``|x - target| <= tolerance`` is taken. If ``length`` is given the
    segment is shortened to that length, centred on the run.

    Parameters
    ----------
    x : array_like, shape (n,)
        Control signal, e.g. vehicle speed in km/h or rpm. NaN never matches.
    targets : float or sequence of float
        Target values, same unit as ``x``.
    tolerance : float
        Allowed deviation, same unit as ``x`` (or a fraction of the target if
        ``relative``).
    t, fs : array_like or float
        Sample times in s or sampling rate in Hz.
    relative : bool, default False
        Interpret ``tolerance`` as a fraction of the target.
    min_duration : float, default 0
        Runs shorter than this (s) are rejected.
    length : float, optional
        Length in s of the returned centred excerpt. Runs shorter than
        ``length`` are rejected.

    Returns
    -------
    list of Segment or None
        One entry per target, None where no run qualifies. No amplitudes are
        involved.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12 (stationary
    and run-up measurements). Wiley.
    """
    x = np.asarray(x, dtype=float).ravel()
    tt = time_axis(x.size, fs, t)
    dt = float(np.median(np.diff(tt))) if x.size > 1 else 0.0
    out = []
    for target in np.atleast_1d(np.asarray(targets, dtype=float)):
        tol = tolerance * abs(target) if relative else tolerance
        with np.errstate(invalid="ignore"):
            mask = np.abs(x - target) <= tol
        starts, stops = runs(mask)
        if starts.size == 0:
            out.append(None)
            continue
        dur = tt[stops - 1] - tt[starts] + dt
        best = int(np.argmax(dur))
        i0, i1 = int(starts[best]), int(stops[best])
        need = max(min_duration, length or 0.0)
        if dur[best] + 1e-12 < need:
            out.append(None)
            continue
        if length is not None:
            centre = 0.5 * (tt[i0] + tt[i1 - 1] + dt)
            j0 = int(np.searchsorted(tt, centre - 0.5 * length - 1e-12))
            j1 = j0 + max(1, int(round(length / dt))) if dt else j0 + 1
            i0, i1 = max(i0, j0), min(i1, j1)
        seg = x[i0:i1]
        out.append(
            Segment(
                target=float(target),
                start=float(tt[i0]),
                stop=float(tt[i1 - 1] + dt),
                i_start=i0,
                i_stop=i1,
                mean=float(seg.mean()),
                std=float(seg.std()),
            )
        )
    return out


from ._ridge import RidgeQualityWarning, RidgeResult, WhiteningWarning, rpm_from_ridge  # noqa: E402  (tacholess estimation)
