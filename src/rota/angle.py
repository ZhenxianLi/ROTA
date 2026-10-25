"""Angle-domain resampling (computed order tracking).

Units: time in s, speed in rpm, angle in revolutions, order in cycles per
revolution.

:func:`angular_resample` maps a time signal to uniform shaft-angle increments
(computed order tracking); :func:`angular_to_time` is its inverse, and
:func:`stationarize` / :func:`destationarize` use the pair as a time warp that
makes order frequencies constant.
"""

from dataclasses import dataclass

import numpy as np
from scipy import interpolate, signal

from ._utils import as_2d, broadcast_rpm, restore
from .speed import rpm_to_phase

__all__ = [
    "angular_resample",
    "AngularSignal",
    "angular_to_time",
    "stationarize",
    "destationarize",
    "StationarizedSignal",
]

# Cap on the intermediate (oversampled) angle-domain length, in samples.
_MAX_INTERMEDIATE = 200_000_000


@dataclass
class AngularSignal:
    """Signal sampled uniformly in shaft angle.

    Attributes
    ----------
    x : ndarray, shape (m,) or (m, ch)
        Signal values, same unit and amplitude as the input.
    rev : ndarray, shape (m,)
        Shaft angle of each sample in revolutions (``k / samples_per_rev``).
    t : ndarray, shape (m,)
        Time of each sample in s.
    rpm : ndarray, shape (m,)
        Speed at each sample in rpm.
    samples_per_rev : int
        Angular sampling rate in samples per revolution.
    max_order : float
        Highest order kept by the anti-aliasing filter.
    """

    x: np.ndarray
    rev: np.ndarray
    t: np.ndarray
    rpm: np.ndarray
    samples_per_rev: int
    max_order: float


def angular_resample(
    x,
    fs,
    rpm,
    *,
    samples_per_rev=None,
    max_order=None,
    min_rpm=None,
    delay=0.0,
    attenuation=80.0,
    upsample=4,
):
    """Resample a time signal to uniform shaft-angle increments.

    The shaft angle is obtained from ``rpm`` with :func:`rota.speed.rpm_to_phase`.
    The signal is first upsampled in time by the integer factor ``upsample``
    with a linear-phase polyphase FIR (pass band to ``0.45 fs``, stop band
    from ``0.55 fs``), so that the subsequent cubic-spline interpolation only
    sees content below about ``0.14`` of its sampling rate, where its gain is
    flat and its images are small. To avoid aliasing whatever the speed, the
    spline is evaluated on an intermediate angle grid dense enough that
    ``0.55 fs`` at the *lowest* speed maps below the intermediate angular
    Nyquist order; there the interpolation is a local upsampling. The
    intermediate signal is then
    low-pass filtered *in the angle domain* (linear-phase Kaiser FIR) and
    decimated to ``samples_per_rev``. Because the filter acts on order, not
    frequency, the anti-aliasing cut-off follows the speed: orders above
    ``max_order`` are removed at every speed, and nothing below it is lost
    at high speed.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal sampled at ``fs``. Channels are processed identically.
    fs : float
        Sampling rate in Hz.
    rpm : float or array_like, shape (n,)
        Shaft speed in rpm at the signal samples.
    samples_per_rev : int, optional
        Output angular sampling rate. Default: ``ceil(2.5 * max_order)``
        rounded up to an even number.
    max_order : float, optional
        Highest order to keep (pass band edge). Default:
        ``0.45 * fs / (max(rpm)/60)``, the highest order representable at the
        top speed; if ``samples_per_rev`` is given, ``0.4 * samples_per_rev``.
    min_rpm : float, optional
        Floor applied to ``rpm`` before integration. Default: 2 % of the
        maximum speed. The angle is wrong wherever the true speed is below the
        floor.
    delay : float, default 0
        Speed-reference delay in s, see :func:`rota.speed.rpm_to_phase`.
    attenuation : float, default 80
        Stop-band attenuation in dB of the angle-domain anti-aliasing filter
        and of the time-domain upsampling filter.
    upsample : int, default 4
        Time-domain upsampling factor before spline interpolation (1 disables
        it; faster, but high orders then droop by up to about 1-2 dB near
        ``0.4 fs`` and content near ``fs/2`` can leak at about -45 dB).

    Returns
    -------
    AngularSignal
        Amplitudes are preserved for content below ``0.45 fs`` (peak and RMS
        conventions are unchanged); content between ``0.45 fs`` and ``fs/2``
        lies in the upsampling filter's transition band. The first and last
        few revolutions carry filter edge effects.

    References
    ----------
    Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order
    tracking. Mechanical Systems and Signal Processing, 11(2), 187-205.
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    Kaiser, J. F. (1974). Nonrecursive digital filter design using the
    I0-sinh window function. Proc. IEEE Int. Symp. Circuits and Systems,
    20-23.
    """
    X, was_1d = as_2d(x)
    n = X.shape[0]
    rpm = broadcast_rpm(rpm, n)
    rpm_max = float(rpm.max())
    if rpm_max <= 0:
        raise ValueError("rpm must be positive somewhere")
    if min_rpm is None:
        min_rpm = 0.02 * rpm_max
    if min_rpm <= 0:
        raise ValueError("min_rpm must be positive")

    if max_order is None:
        if samples_per_rev is not None:
            max_order = 0.4 * samples_per_rev
        else:
            max_order = 0.45 * fs / (rpm_max / 60.0)
    if samples_per_rev is None:
        samples_per_rev = int(np.ceil(2.5 * max_order))
        samples_per_rev += samples_per_rev % 2
    samples_per_rev = int(samples_per_rev)
    return _resample_to_angle(
        X, was_1d, fs, rpm, samples_per_rev, max_order, min_rpm, delay, attenuation, upsample
    )


def _resample_to_angle(
    X, was_1d, fs, rpm, samples_per_rev, max_order, min_rpm, delay, attenuation, upsample
):
    """Core of :func:`angular_resample` for a validated speed and angle grid.

    ``samples_per_rev`` is the angular sampling rate in samples per revolution;
    it may be a non-integer float (used by :func:`stationarize`), since only
    the integer decimation factor ``M`` of the intermediate grid has to be an
    integer. ``X`` has shape ``(n, ch)``, ``rpm`` shape ``(n,)``.
    """
    n = X.shape[0]
    if max_order >= 0.5 * samples_per_rev:
        raise ValueError("max_order must be below samples_per_rev / 2")

    rpm_eff = np.maximum(rpm, min_rpm)
    rev = rpm_to_phase(rpm_eff, fs, delay=delay)
    rpm_eff = np.maximum(
        np.interp(np.arange(n) / fs + delay, np.arange(n) / fs, rpm), min_rpm
    )
    rpm_lo = float(rpm_eff.min())
    upsample = int(upsample)
    if upsample < 1:
        raise ValueError("upsample must be a positive integer")

    # Intermediate angular rate: an integer multiple M of the output rate with
    # at least 1.2 * fs * 60 / rpm_lo samples per revolution, so that the
    # upsampler's transition band (up to 0.55 fs) does not fold.
    M = max(1, int(np.ceil(1.2 * fs * 60.0 / rpm_lo / samples_per_rev)))
    spr_hi = samples_per_rev * M
    m_hi = int(np.floor(rev[-1] * spr_hi)) + 1
    if m_hi * X.shape[1] > _MAX_INTERMEDIATE:
        raise MemoryError(
            "intermediate angle grid too large; raise min_rpm or lower fs"
        )
    rev_hi = np.arange(m_hi) / spr_hi
    t_hi = np.interp(rev_hi, rev, np.arange(n) / fs)
    if upsample > 1:
        # Normalised to the upsampled Nyquist upsample * fs / 2: cut-off fs/2,
        # transition band 0.45 fs to 0.55 fs.
        numtaps, beta = signal.kaiserord(attenuation, 0.2 / upsample)
        numtaps = 2 * (numtaps // 2) + 1
        h = signal.firwin(numtaps, 1.0 / upsample, window=("kaiser", beta))
        h /= h.sum()  # resample_poly applies the factor `upsample` itself
        X_up = signal.resample_poly(X, upsample, 1, axis=0, window=h)
    else:
        X_up = X
    spline = interpolate.make_interp_spline(np.arange(X_up.shape[0]), X_up, k=3, axis=0)
    x_hi = spline(t_hi * (fs * upsample))

    if M > 1:
        nyq_hi = 0.5 * spr_hi
        edge = 0.5 * samples_per_rev
        width = (edge - max_order) / nyq_hi
        numtaps, beta = signal.kaiserord(attenuation, width)
        numtaps += 1 - numtaps % 2
        h = signal.firwin(numtaps, 0.5 * (max_order + edge) / nyq_hi, window=("kaiser", beta))
        h /= h.sum()
        y = signal.resample_poly(x_hi, 1, M, axis=0, window=h)
    else:
        y = x_hi
    m = y.shape[0]
    rev_out = np.arange(m) / samples_per_rev
    t_out = t_hi[::M][:m]
    rpm_out = np.interp(t_out, np.arange(n) / fs, rpm_eff)
    return AngularSignal(
        x=restore(y, was_1d),
        rev=rev_out,
        t=t_out,
        rpm=rpm_out,
        samples_per_rev=samples_per_rev,
        max_order=float(max_order),
    )


# --------------------------------------------------------------------------
# inverse transform and time warp
# --------------------------------------------------------------------------


def _uniform_grid(t, fs):
    """Validate an explicit uniform time grid at spacing ``1/fs``."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2 or not np.all(np.isfinite(t)):
        raise ValueError("t must be a finite 1-D array of at least 2 times in s")
    if not np.allclose(np.diff(t), 1.0 / fs, rtol=1e-6, atol=0.0):
        raise ValueError("t must be uniformly spaced at 1/fs")
    return t


def angular_to_time(
    a,
    fs,
    rpm=None,
    *,
    t=None,
    samples_per_rev=None,
    max_order=None,
    min_rpm=None,
    delay=0.0,
    attenuation=80.0,
    upsample=4,
):
    """Map an angle-domain signal back to uniform time samples.

    Inverse of :func:`angular_resample`: the angle-domain signal ``y(theta)``
    (uniform in shaft angle) is read at the shaft angle ``theta(t)`` of each
    output sample, ``x(t) = y(theta(t))``. The shaft angle comes from ``rpm``
    with :func:`rota.speed.rpm_to_phase`, or, if ``rpm`` is None, from the
    ``(t, rev)`` pairs stored in the :class:`AngularSignal` (cubic-spline
    interpolation). Giving an ``rpm`` different from the one used for the
    forward transform re-times the angle-domain signal on a new speed curve
    (orders then appear at ``order * rpm / 60`` for the new speed).

    Interpolation. The angle-domain signal is first upsampled by the integer
    factor ``upsample`` with a linear-phase polyphase FIR whose pass band
    reaches ``max_order`` and whose stop band starts at the first image,
    ``samples_per_rev - max_order``, and then evaluated at the (non-uniform)
    angles with a cubic spline. With the default ``upsample=4`` content up to
    ``0.4 * samples_per_rev`` is interpolated from a grid on which it lies
    below 0.1 of the sampling rate, where the cubic spline has a gain error
    below about ``3e-4`` (-70 dB) and its images are below -75 dB.

    Anti-aliasing. An order ``o`` is heard at ``o * rpm(t) / 60`` Hz, so the
    highest frequency present is ``f_hi = max_order * max(rpm) / 60``. If
    ``f_hi <= 0.45 fs`` nothing can fold and the signal is only interpolated,
    with no further filtering. Otherwise the interpolated signal is evaluated
    on a time grid ``U`` times denser than ``fs``, ``U = max(2, ceil(2.2 f_hi
    / fs))`` (so that ``U fs / 2 > f_hi``), low-pass filtered by a
    linear-phase Kaiser FIR (pass band to ``0.45 fs``, stop band from
    ``0.55 fs``, ``attenuation`` dB) and decimated by ``U``. This is the dual
    of the angle-domain filter of :func:`angular_resample`: the cut-off is a
    fixed *frequency*, hence at every instant the content below ``0.45 fs`` is
    kept (including high orders at low speed) and the content above ``0.55 fs``
    is removed, whatever the order. It happens when ``fs`` is lower than the
    rate the signal was analysed at, when ``max_order`` is large, or when
    ``rpm`` is faster than the analysis speed.

    Parameters
    ----------
    a : AngularSignal or array_like, shape (m,) or (m, ch)
        Angle-domain signal, or a plain array of samples taken at angles
        ``rev0 + k / samples_per_rev`` with ``rev0 = 0``. Channels are
        processed identically.
    fs : float
        Sampling rate in Hz of the output time signal.
    rpm : float or array_like, shape (n,), optional
        Shaft speed in rpm *at the output samples*, defining the angle
        ``theta(t) = a.rev[0] + integral of rpm/60 dt`` (``a.rev[0]`` is 0 for
        plain arrays): the first output sample is read at the first angle
        sample. Required for plain arrays. If None, the angle is taken from
        ``a.t`` and ``a.rev`` and ``min_rpm`` and ``delay`` are not used.
    t : array_like, shape (n,), optional
        Output sample times in s. They must be uniformly spaced at ``1/fs``
        (the anti-aliasing decimation needs a uniform grid). Default: with
        ``rpm`` an array, ``a.t[0] + arange(len(rpm)) / fs`` (``a.t[0]`` is 0 for
        plain arrays); otherwise the whole span of ``a.t`` at ``fs``.
    samples_per_rev : float, optional
        Angular sampling rate of a plain array, in samples per revolution (may
        be non-integer). Not allowed with an :class:`AngularSignal`.
    max_order : float, optional
        Highest order present in the angle-domain signal (cycles per
        revolution), below ``samples_per_rev / 2``. Sets the upsampling filter
        and the anti-aliasing test. Default: ``a.max_order``, or
        ``0.4 * samples_per_rev`` for a plain array.
    min_rpm : float, optional
        Floor applied to ``rpm`` before integration, as in
        :func:`angular_resample` (default: 2 % of the maximum speed). Use the
        same value as in the forward transform for a round trip.
    delay : float, default 0
        Speed-reference delay in s, see :func:`rota.speed.rpm_to_phase`; use
        the value of the forward transform for a round trip.
    attenuation : float, default 80
        Stop-band attenuation in dB of the angle-domain upsampling filter and
        of the time-domain anti-aliasing filter.
    upsample : int, default 4
        Angle-domain upsampling factor before spline interpolation (1 disables
        it; faster, but then content near ``0.4 * samples_per_rev`` is
        interpolated with an error of several percent).

    Returns
    -------
    x : ndarray, shape (n,) or (n, ch)
        Time signal at ``t``, same unit and amplitude convention (peak or RMS)
        as ``a``: amplitudes of content below ``0.45 fs`` are preserved. The
        first and last few revolutions carry the edge effects of the filters,
        and a sample whose angle lies up to three angle samples outside the
        data takes the end value; a sample farther outside raises ValueError.

    References
    ----------
    Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order
    tracking. Mechanical Systems and Signal Processing, 11(2), 187-205.
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    Crochiere, R. E., & Rabiner, L. R. (1983). Multirate Digital Signal
    Processing, ch. 2 (interpolation and decimation). Prentice-Hall.
    de Boor, C. (1978). A Practical Guide to Splines, ch. 4 (cubic spline
    interpolation). Springer.
    Kaiser, J. F. (1974). Nonrecursive digital filter design using the
    I0-sinh window function. Proc. IEEE Int. Symp. Circuits and Systems,
    20-23.
    """
    fs = float(fs)
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a positive sampling rate in Hz")
    is_angular = isinstance(a, AngularSignal)
    if is_angular:
        if samples_per_rev is not None:
            raise ValueError("samples_per_rev is only for plain arrays; an AngularSignal has its own")
        Y, was_1d = as_2d(a.x)
        spr = float(a.samples_per_rev)
        rev0 = float(a.rev[0])
        if max_order is None:
            max_order = a.max_order
    else:
        if samples_per_rev is None:
            raise ValueError("a plain array needs samples_per_rev")
        if rpm is None:
            raise ValueError("a plain array needs rpm")
        Y, was_1d = as_2d(a)
        spr = float(samples_per_rev)
        rev0 = 0.0
    if not (np.isfinite(spr) and spr > 0):
        raise ValueError("samples_per_rev must be positive")
    if max_order is None:
        max_order = 0.4 * spr
    max_order = float(max_order)
    if not (0.0 < max_order < 0.5 * spr):
        raise ValueError("max_order must be positive and below samples_per_rev / 2")
    m = Y.shape[0]
    if m < 4 or not np.all(np.isfinite(Y)):
        raise ValueError("the angle-domain signal needs at least 4 finite samples")
    L = int(upsample)
    if L < 1:
        raise ValueError("upsample must be a positive integer")

    # Output time grid.
    rpm_arr = None if rpm is None else np.asarray(rpm, dtype=float)
    if t is not None:
        tt = _uniform_grid(t, fs)
    elif rpm_arr is not None and rpm_arr.ndim == 1:
        t0 = float(a.t[0]) if is_angular else 0.0
        tt = t0 + np.arange(rpm_arr.size) / fs
    elif is_angular:
        tt = float(a.t[0]) + np.arange(int(np.floor((a.t[-1] - a.t[0]) * fs + 1e-9)) + 1) / fs
    else:
        raise ValueError("give t or an rpm array for a plain array")
    n = tt.size

    # Shaft angle at the output samples and its maximum speed.
    if rpm_arr is None:
        if n < 2:
            raise ValueError("the output grid has fewer than 2 samples")
        rpm_max = float(np.max(a.rpm))
        rev_map = interpolate.make_interp_spline(a.t, a.rev, k=3)
        rpm_f = None
    else:
        rpm_arr = broadcast_rpm(rpm_arr, n)
        rpm_max = float(rpm_arr.max())
        if rpm_max <= 0:
            raise ValueError("rpm must be positive somewhere")
        if min_rpm is None:
            min_rpm = 0.02 * rpm_max
        if min_rpm <= 0:
            raise ValueError("min_rpm must be positive")
        rpm_f = np.maximum(rpm_arr, min_rpm)
        rpm_max = float(rpm_f.max())

    f_hi = max_order * rpm_max / 60.0
    U = 1 if f_hi <= 0.45 * fs else max(2, int(np.ceil(2.2 * f_hi / fs)))
    n_hi = (n - 1) * U + 1
    if max(n_hi, m * L) * Y.shape[1] > _MAX_INTERMEDIATE:
        raise MemoryError("intermediate grid too large; lower upsample or the speed range")
    t_hi = tt[0] + np.arange(n_hi) / (U * fs)
    if rpm_f is None:
        theta = rev_map(t_hi)
    else:
        rpm_hi = np.interp(t_hi + delay, tt, rpm_f) if (U > 1 or delay) else rpm_f
        theta = rev0 + rpm_to_phase(rpm_hi, t=t_hi)

    # Angle-domain upsampling, then cubic-spline evaluation at the angles.
    if L > 1:
        # transition band from max_order to its first image, spr - max_order
        width = max(1.0 - 2.0 * max_order / spr, 0.02)
        numtaps, beta = signal.kaiserord(attenuation, 2.0 * width / L)
        numtaps = 2 * (numtaps // 2) + 1
        h = signal.firwin(numtaps, 1.0 / L, window=("kaiser", beta))
        h /= h.sum()  # resample_poly applies the factor L itself
        Y_up = signal.resample_poly(Y, L, 1, axis=0, window=h)
    else:
        Y_up = Y
    u = (theta - rev0) * (spr * L)
    u_max = float((m - 1) * L)
    slack = 3.0 * L
    if u[0] < -slack or u[-1] > u_max + slack:
        raise ValueError(
            "the output samples reach beyond the angle-domain data; "
            "check rpm, t and samples_per_rev"
        )
    spline = interpolate.make_interp_spline(np.arange(Y_up.shape[0]), Y_up, k=3, axis=0)
    x_hi = spline(np.clip(u, 0.0, u_max))

    if U > 1:
        numtaps, beta = signal.kaiserord(attenuation, 0.2 / U)
        numtaps = 2 * (numtaps // 2) + 1
        h = signal.firwin(numtaps, 1.0 / U, window=("kaiser", beta))
        h /= h.sum()
        x_out = signal.resample_poly(x_hi, 1, U, axis=0, window=h)
    else:
        x_out = x_hi
    return restore(x_out, was_1d)


@dataclass
class StationarizedSignal(AngularSignal):
    """Result of :func:`stationarize`: an :class:`AngularSignal` read at ``fs``.

    ``x`` is the warped signal. Read at the rate ``fs``, an order ``o`` of the
    shaft sits at the constant frequency ``o * rpm_ref / 60`` Hz.
    ``samples_per_rev`` equals ``fs * 60 / rpm_ref`` and is in general not an
    integer (a float here). ``rev``, ``t`` and ``rpm`` describe each sample as
    in :class:`AngularSignal` (``t`` is the time in the *original* signal).

    Attributes
    ----------
    fs : float
        Rate in Hz at which ``x`` is to be read.
    rpm_ref : float
        Reference speed in rpm that sets the constant order frequencies.
    """

    fs: float
    rpm_ref: float

    @property
    def t_warped(self):
        """Time axis in s of the warped signal read at ``fs`` (``k / fs``)."""
        return np.arange(self.x.shape[0]) / self.fs


_STATIONARIZE_KEYS = ("max_order", "min_rpm", "delay", "attenuation", "upsample")


def stationarize(x, fs, rpm, *, rpm_ref=None, **kw):
    """Warp time so that order frequencies become constant.

    The signal is resampled to uniform shaft-angle increments with
    ``samples_per_rev = fs * 60 / rpm_ref`` (the same anti-aliased resampler
    as :func:`angular_resample`) and the result is then read at the original
    rate ``fs``. A component of order ``o`` has angle-domain phase
    ``2 pi o k / samples_per_rev``, i.e. frequency ``o * rpm_ref / 60`` Hz in
    the warped signal, whatever the actual speed. Long-window spectral metrics
    (tone-to-noise ratio, prominence ratio, long-window tonality) then see a
    drifting order as a stationary tone, instead of a smeared one.

    The warp is exact for any speed curve; it is the time warp of computed
    order tracking read as a time signal, not an approximation for constant
    gear ratios or linear speed-up. Fixed-frequency content (resonances,
    tonal noise) is not stationary after warping, it is swept instead.

    ``samples_per_rev`` is the *exact* float ``fs * 60 / rpm_ref``, not a
    rounded integer: the resampler of :func:`angular_resample` only needs an
    integer decimation factor on an internal grid, so ``rpm_ref`` is honoured
    to rounding error and the warped order frequencies are exactly
    ``order * rpm_ref / 60``. :func:`angular_resample` itself is unchanged
    (integer ``samples_per_rev``).

    Bandwidth. At speed ``rpm`` the angle-domain cut-off ``max_order`` (default
    ``0.4 * samples_per_rev``) corresponds to ``0.4 fs * rpm / rpm_ref`` Hz:
    below ``rpm_ref`` the warped signal has fewer samples per second than the
    original, and content above that frequency is attenuated, completely from
    ``0.5 fs * rpm / rpm_ref`` Hz (anti-aliasing: it is removed, not folded);
    above ``rpm_ref`` the warped signal has more samples and nothing is lost. Choose ``rpm_ref`` at
    or below the lowest speed of interest to keep the full bandwidth (at the
    price of a longer warped signal, about ``mean(rpm) / rpm_ref`` times the
    original length). The default is the median speed.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal sampled at ``fs``, in any unit; channels are processed
        identically.
    fs : float
        Sampling rate in Hz.
    rpm : float or array_like, shape (n,)
        Shaft speed in rpm at the signal samples.
    rpm_ref : float, optional
        Reference speed in rpm of the warped signal. Default: median of
        ``rpm``.
    **kw
        ``max_order``, ``min_rpm``, ``delay``, ``attenuation`` and
        ``upsample`` as in :func:`angular_resample` (``samples_per_rev`` is set
        by ``rpm_ref``).

    Returns
    -------
    StationarizedSignal
        An :class:`AngularSignal` (amplitudes unchanged, peak or RMS as in
        ``x``) that also holds ``fs`` and ``rpm_ref``. The first and last few
        revolutions carry filter edge effects.

    References
    ----------
    Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order
    tracking. Mechanical Systems and Signal Processing, 11(2), 187-205.
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    """
    bad = sorted(set(kw) - set(_STATIONARIZE_KEYS))
    if bad:
        hint = " (set by rpm_ref)" if "samples_per_rev" in bad else ""
        raise TypeError(f"stationarize() got unexpected keyword argument(s) {bad}{hint}")
    fs = float(fs)
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a positive sampling rate in Hz")
    X, was_1d = as_2d(x)
    rpm = broadcast_rpm(rpm, X.shape[0])
    rpm_max = float(rpm.max())
    if rpm_max <= 0:
        raise ValueError("rpm must be positive somewhere")
    rpm_ref = float(np.median(rpm)) if rpm_ref is None else float(rpm_ref)
    if not (np.isfinite(rpm_ref) and rpm_ref > 0):
        raise ValueError("rpm_ref must be a positive speed in rpm")
    min_rpm = kw.get("min_rpm")
    if min_rpm is None:
        min_rpm = 0.02 * rpm_max
    if min_rpm <= 0:
        raise ValueError("min_rpm must be positive")
    spr = fs * 60.0 / rpm_ref
    max_order = kw.get("max_order")
    if max_order is None:
        max_order = 0.4 * spr
    upsample = int(kw.get("upsample", 4))
    if upsample < 1:
        raise ValueError("upsample must be a positive integer")
    a = _resample_to_angle(
        X, was_1d, fs, rpm, spr, float(max_order), min_rpm,
        kw.get("delay", 0.0), kw.get("attenuation", 80.0), upsample,
    )
    return StationarizedSignal(
        x=a.x, rev=a.rev, t=a.t, rpm=a.rpm, samples_per_rev=spr, max_order=a.max_order,
        fs=fs, rpm_ref=rpm_ref,
    )


def destationarize(y, fs, rpm, *, rpm_ref=None, **kw):
    """Undo :func:`stationarize`: map a warped signal back to the real time axis.

    Inverse time warp, implemented with :func:`angular_to_time`: the warped
    signal is read at the shaft angle ``theta(t)`` of each output sample, so
    that orders return to their true frequencies ``order * rpm(t) / 60``. The
    output has ``len(rpm)`` samples at ``fs``. ``rpm`` may differ from the one
    used for stationarizing (processing in the warped domain and returning to
    a new speed curve re-times the signal); to recover the original, pass the
    same speed, and for a plain array the same ``rpm_ref``.

    Parameters
    ----------
    y : StationarizedSignal, AngularSignal or array_like, shape (m,) or (m, ch)
        Warped signal. A :class:`StationarizedSignal` (or any
        :class:`AngularSignal`) carries its own ``samples_per_rev``,
        ``max_order`` and angle origin. For a plain array, the angular rate is
        ``fs * 60 / rpm_ref`` and ``max_order`` defaults to
        ``0.4 * samples_per_rev``, as in :func:`stationarize`; the array is
        taken to start at angle 0.
    fs : float
        Sampling rate in Hz of the output time signal.
    rpm : array_like, shape (n,)
        Shaft speed in rpm at the output samples (a scalar is allowed only
        with a :class:`AngularSignal` and ``t``).
    rpm_ref : float, optional
        Reference speed in rpm of a plain-array ``y``. Default: median of
        ``rpm``. Not allowed with an :class:`AngularSignal`.
    **kw
        ``t``, ``max_order``, ``min_rpm``, ``delay``, ``attenuation`` and
        ``upsample`` as in :func:`angular_to_time`. For a round trip use the
        same ``min_rpm`` and ``delay`` as in :func:`stationarize`.

    Returns
    -------
    x : ndarray, shape (n,) or (n, ch)
        Time signal; amplitudes preserved (peak or RMS as in ``y``) for content
        below ``0.45 fs``. Anti-aliasing and edge behaviour are those of
        :func:`angular_to_time`.

    References
    ----------
    Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order
    tracking. Mechanical Systems and Signal Processing, 11(2), 187-205.
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    """
    if isinstance(y, AngularSignal):
        if rpm_ref is not None:
            raise ValueError("rpm_ref is only for plain arrays; an AngularSignal has its own rate")
        return angular_to_time(y, fs, rpm, **kw)
    if "samples_per_rev" in kw:
        raise TypeError("destationarize() got unexpected keyword argument 'samples_per_rev' "
                        "(set by rpm_ref)")
    fs = float(fs)
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a positive sampling rate in Hz")
    rpm_arr = np.asarray(rpm, dtype=float)
    if rpm_arr.ndim != 1:
        raise ValueError("a plain array needs rpm as an array of the output length")
    if not np.all(np.isfinite(rpm_arr)):
        raise ValueError("rpm contains non-finite values; regularise it first")
    rpm_ref = float(np.median(rpm_arr)) if rpm_ref is None else float(rpm_ref)
    if not (np.isfinite(rpm_ref) and rpm_ref > 0):
        raise ValueError("rpm_ref must be a positive speed in rpm")
    return angular_to_time(y, fs, rpm_arr, samples_per_rev=fs * 60.0 / rpm_ref, **kw)
