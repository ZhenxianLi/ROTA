"""Levels, frequency weightings, calibrated resampling and octave bands.

A thin bridge between order analysis and psychoacoustic tools: sound
pressure levels with explicit reference, A and C frequency weighting after
IEC 61672-1, exponential time weighting (Fast, Slow), calibration-preserving
polyphase resampling to the 48 kHz rate expected by external psychoacoustic
toolboxes, and octave and fractional-octave band filter banks after
IEC 61260-1, with band levels over time and along rpm.

Conventions
-----------
Signals are arrays of shape ``(n,)`` or ``(n, ch)`` in Pa (or any unit, as
long as ``ref`` is given in the same unit); every operator acts on each
channel independently. Levels are RMS levels in dB re ``ref``; the reference
is always an explicit argument (20 µPa by default, the reference sound
pressure in air). Time in s, frequency in Hz, speed in rpm. No gain or
calibration factor is applied unless passed explicitly.

References
----------
IEC 61672-1:2013. Electroacoustics - Sound level meters - Part 1:
Specifications.
IEC 61260-1:2014. Electroacoustics - Octave-band and fractional-octave-band
filters - Part 1: Specifications.
ANSI S1.11-2004. Specification for octave-band and fractional-octave-band
analog and digital filters.
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy import signal

from ._utils import as_2d, broadcast_rpm, restore

__all__ = [
    "weighting_db",
    "weighting_filter",
    "weight",
    "level_db",
    "level_vs_time",
    "combine_channels",
    "to_calibrated_48k",
    "octave_band_centres",
    "octave_filterbank",
    "octave_levels",
    "band_levels_vs_rpm",
    "OctaveFilterbank",
    "BandLevels",
    "BandRpmLevels",
]

_KINDS = ("A", "C", "Z")


# --------------------------------------------------------------------------
# Frequency weightings (IEC 61672-1)
# --------------------------------------------------------------------------


def _weighting_poles():
    """Pole frequencies f1, f2, f3, f4 in Hz from the IEC 61672-1 expressions.

    f1 and f4 follow from the -3 dB frequencies fL = 10**1.5 Hz and
    fH = 10**3.9 Hz of the C weighting relative to fr = 1 kHz with
    D**2 = 1/2; f2 and f3 from fA = 10**2.45 Hz. The rounded values are
    20.60, 107.7, 737.9 and 12194 Hz.
    """
    fr, fl, fh, fa = 1000.0, 10**1.5, 10**3.9, 10**2.45
    d = math.sqrt(0.5)
    b = (fr**2 + fl**2 * fh**2 / fr**2 - d * (fl**2 + fh**2)) / (1.0 - d)
    c = fl**2 * fh**2
    root = math.sqrt(b * b - 4.0 * c)
    f1 = math.sqrt((-b - root) / 2.0)
    f4 = math.sqrt((-b + root) / 2.0)
    f2 = (3.0 - math.sqrt(5.0)) / 2.0 * fa
    f3 = (3.0 + math.sqrt(5.0)) / 2.0 * fa
    return f1, f2, f3, f4


def _check_kind(kind):
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    return kind


def _analogue_db_unnormalised(f, kind):
    f1, f2, f3, f4 = _weighting_poles()
    f2_ = np.asarray(f, dtype=float) ** 2
    with np.errstate(divide="ignore"):
        c = 20.0 * np.log10(f4**2 * f2_ / ((f2_ + f1**2) * (f2_ + f4**2)))
        if kind == "C":
            return c
        return c + 20.0 * np.log10(f2_ / np.sqrt((f2_ + f2**2) * (f2_ + f3**2)))


def weighting_db(f, kind="A"):
    """Analytic A, C or Z frequency weighting in dB after IEC 61672-1.

    ``C(f) = 20 log10[f4² f² / ((f² + f1²)(f² + f4²))] - C1000`` and
    ``A(f) = C(f) + C1000 + 20 log10[f² / sqrt((f² + f2²)(f² + f3²))] - A1000``
    with the pole frequencies f1 = 20.60 Hz, f2 = 107.7 Hz, f3 = 737.9 Hz,
    f4 = 12194 Hz computed from the defining expressions of the standard,
    and C1000, A1000 (about -0.062 dB and -2.000 dB) evaluated exactly so
    that the weighting is 0 dB at 1 kHz. ``Z`` is 0 dB at all frequencies.

    Use it to weight a power or magnitude spectrum in the frequency domain
    (add the dB values to a level spectrum, or multiply magnitudes by
    ``10**(w / 20)``), the alternative to :func:`weight` in the time domain.
    Apply one or the other, once.

    Parameters
    ----------
    f : array_like
        Frequencies in Hz (>= 0; 0 Hz gives -inf for A and C).
    kind : {"A", "C", "Z"}, default "A"

    Returns
    -------
    ndarray
        Weighting in dB (a gain on levels, i.e. on RMS or peak amplitude
        alike), same shape as ``f``.

    References
    ----------
    IEC 61672-1:2013. Electroacoustics - Sound level meters - Part 1:
    Specifications (analytical expressions for the frequency weightings).
    """
    _check_kind(kind)
    f = np.asarray(f, dtype=float)
    if np.any(f < 0):
        raise ValueError("frequencies must be non-negative")
    if kind == "Z":
        return np.zeros_like(f)
    return _analogue_db_unnormalised(f, kind) - _analogue_db_unnormalised(1000.0, kind)


def weighting_filter(kind, fs):
    """Digital A, C or Z weighting filter as second-order sections.

    The analogue weighting of IEC 61672-1 (A: four zeros at s = 0, double
    poles at 2π·f1 and 2π·f4, single poles at 2π·f2 and 2π·f3; C: two zeros
    at s = 0, double poles at 2π·f1 and 2π·f4; f1 = 20.60 Hz, f2 = 107.7 Hz,
    f3 = 737.9 Hz, f4 = 12194 Hz) is mapped to the z-plane with the bilinear
    transform at the rate ``fs`` (no pre-warping) and scaled so that the
    digital response is exactly 0 dB at 1 kHz.

    The bilinear transform compresses the whole analogue frequency axis into
    0 ... fs/2: the digital response at f equals the analogue one at
    ``(fs/π)·tan(π f / fs)``, so above a few kHz the digital filter
    attenuates more than the analogue weighting, without limit towards
    fs/2. At fs = 48 kHz the deviation is about -0.2 dB at 6.3 kHz, -0.5 dB
    at 8 kHz, -1.2 dB at 10 kHz, -2.7 dB at 12.5 kHz, -6.2 dB at 16 kHz and
    -16 dB at 20 kHz; at 44.1 kHz -0.7 dB at 8 kHz, -1.5 dB at 10 kHz,
    -3.5 dB at 12.5 kHz, -8.2 dB at 16 kHz and -24 dB at 20 kHz. Both rates
    stay inside the class-1 acceptance limits of IEC 61672-1 at all nominal
    frequencies 10 Hz - 20 kHz (whose lower limits are -3, -5, -16 dB and
    -inf at 10, 12.5, 16 and 20 kHz; worst margin 1.5 dB at 10-12.5 kHz,
    44.1 kHz); lower rates do not cover the range, so resample to 48 kHz
    first (:func:`to_calibrated_48k`). Below 1 kHz the deviation is below
    0.01 dB at fs >= 44.1 kHz.

    Parameters
    ----------
    kind : {"A", "C", "Z"}
        Frequency weighting. ``"Z"`` returns the identity section.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    sos : ndarray, shape (n_sections, 6)
        Second-order sections for :func:`scipy.signal.sosfilt`; the gain is
        a pure number (the weighting does not change units).

    The filter gain is a dimensionless amplitude response (0 dB at 1 kHz); no amplitude convention applies.

    References
    ----------
    IEC 61672-1:2013. Electroacoustics - Sound level meters - Part 1:
    Specifications.
    Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal
    Processing (3rd ed.), sec. 7.2 (bilinear transformation). Pearson.
    """
    _check_kind(kind)
    fs = float(fs)
    if not fs > 0:
        raise ValueError("fs must be positive")
    if kind == "Z":
        return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    f1, f2, f3, f4 = _weighting_poles()
    w = 2.0 * np.pi * np.array([f1, f2, f3, f4])
    if kind == "A":
        z = np.zeros(4)
        p = -np.array([w[0], w[0], w[1], w[2], w[3], w[3]])
    else:
        z = np.zeros(2)
        p = -np.array([w[0], w[0], w[3], w[3]])
    zd, pd, kd = signal.bilinear_zpk(z, p, 1.0, fs)
    sos = signal.zpk2sos(zd, pd, kd)
    _, h = signal.sosfreqz(sos, worN=[1000.0], fs=fs)
    sos[0, :3] /= np.abs(h[0])
    return sos


def weight(x, fs, kind="A", *, axis=0):
    """Apply an A, C or Z frequency weighting once, causally.

    The signal is filtered a single time with the second-order sections of
    :func:`weighting_filter` (``scipy.signal.sosfilt``, zero initial state),
    so the magnitude response is that of the weighting. Zero-phase
    forward-backward filtering is deliberately not offered: it applies the
    weighting twice (the dB values double). The weighting is linear and acts
    on each channel independently; it leaves the unit and the amplitude
    convention of the signal unchanged.

    The first samples contain the filter's start-up transient; its slowest
    component is that of the 20.6 Hz double pole (time constant
    1/(2π·20.6 Hz) = 7.7 ms). After 0.1 s, steady sines from 20 Hz upwards
    read within 0.001 dB of their steady level.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal (e.g. sound pressure in Pa).
    fs : float
        Sampling rate in Hz (>= 44.1 kHz for class-1 accuracy up to 20 kHz;
        see :func:`weighting_filter`).
    kind : {"A", "C", "Z"}, default "A"
    axis : int, default 0
        Time axis.

    Returns
    -------
    ndarray
        Weighted signal, float64, same shape and unit as ``x``.

    References
    ----------
    IEC 61672-1:2013. Electroacoustics - Sound level meters - Part 1:
    Specifications.
    """
    x = np.asarray(x, dtype=float)
    sos = weighting_filter(kind, fs)
    if kind == "Z":
        return x.copy()
    return signal.sosfilt(sos, x, axis=axis)


# --------------------------------------------------------------------------
# Levels
# --------------------------------------------------------------------------


def _check_ref(ref):
    ref = float(ref)
    if not ref > 0:
        raise ValueError("ref must be positive")
    return ref


def _power_db(p, ref):
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.asarray(p, dtype=float) / ref**2)


def level_db(x, *, ref=20e-6, axis=0):
    """RMS level in dB re ``ref``.

    ``L = 10 log10(mean(x**2) / ref**2)`` along ``axis``: the equivalent
    continuous (energy-average) level of the whole signal. A sine of peak
    amplitude ``A`` gives ``20 log10(A / (sqrt(2) ref))``. No frequency or
    time weighting is applied (weight the signal first with :func:`weight`).

    Parameters
    ----------
    x : array_like
        Signal in the unit of ``ref`` (Pa by default).
    ref : float, default 20e-6
        Reference RMS value, in the unit of ``x`` (20 µPa for sound
        pressure in air).
    axis : int, default 0
        Time axis.

    Returns
    -------
    ndarray or float
        Level in dB re ``ref`` per channel (``-inf`` for an all-zero
        channel).

    References
    ----------
    IEC 61672-1:2013. Electroacoustics - Sound level meters - Part 1:
    Specifications (time-averaged sound level).
    """
    ref = _check_ref(ref)
    x = np.asarray(x, dtype=float)
    return _power_db(np.mean(x**2, axis=axis), ref)


_TAU = {"F": 0.125, "S": 1.0}


def _frame_samples(frame, fs):
    nf = int(round(float(frame) * fs))
    if nf < 1:
        raise ValueError("frame must be at least one sample long")
    return nf


def level_vs_time(x, fs, *, time_weighting="F", frame=None, ref=20e-6):
    """Time-weighted or block-averaged sound level versus time.

    ``time_weighting="F"`` or ``"S"`` gives the exponentially time-weighted
    level of IEC 61672-1, ``L(t) = 10 log10[(1/τ) ∫ x²(ξ) e^{-(t-ξ)/τ} dξ /
    ref²]`` with τ = 0.125 s (Fast) or 1 s (Slow); a positive number is used
    as τ in s. The integral is the one-pole recursion
    ``y[n] = a y[n-1] + (1 - a) x[n]²`` with ``a = exp(-1 / (τ fs))``, zero
    initial state (a meter switched on at t = 0, so the first few τ read low).
    After a steady signal stops, the level decays at 10 log10(e) / τ dB/s
    (34.7 dB/s Fast, 4.3 dB/s Slow). Without ``frame`` the level is returned
    at every sample; with ``frame`` it is read at the last sample of each
    consecutive frame of ``frame`` s (as a meter display would be).

    ``time_weighting=None`` gives block RMS levels (equivalent continuous
    level over each non-overlapping frame of ``frame`` s, which is then
    required); an incomplete last frame is dropped.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal in the unit of ``ref`` (Pa by default).
    fs : float
        Sampling rate in Hz.
    time_weighting : {"F", "S"}, float or None, default "F"
    frame : float, optional
        Frame length in s (rounded to whole samples).
    ref : float, default 20e-6
        Reference RMS value in the unit of ``x``.

    Returns
    -------
    t : ndarray, shape (m,)
        Times in s: sample times (exponential, no frame), the reading
        instants (exponential with frame) or the frame centres (block RMS).
    levels : ndarray, shape (m,) or (m, ch)
        RMS levels in dB re ``ref``.

    References
    ----------
    IEC 61672-1:2013. Electroacoustics - Sound level meters - Part 1:
    Specifications (time weightings F and S; time-averaged sound level).
    """
    ref = _check_ref(ref)
    fs = float(fs)
    if not fs > 0:
        raise ValueError("fs must be positive")
    x2, one = as_2d(x)
    n = x2.shape[0]
    if time_weighting is None:
        if frame is None:
            raise ValueError("block RMS levels (time_weighting=None) need a frame length")
        nf = _frame_samples(frame, fs)
        m = n // nf
        if m < 1:
            raise ValueError("signal shorter than one frame")
        p = np.mean(x2[: m * nf].reshape(m, nf, -1) ** 2, axis=1)
        t = (np.arange(m) * nf + 0.5 * nf) / fs
        return t, restore(_power_db(p, ref), one)
    if isinstance(time_weighting, str):
        if time_weighting not in _TAU:
            raise ValueError(
                f"time_weighting must be 'F', 'S', a time constant or None, got {time_weighting!r}"
            )
        tau = _TAU[time_weighting]
    else:
        tau = float(time_weighting)
        if not tau > 0:
            raise ValueError("time constant must be positive")
    a = math.exp(-1.0 / (tau * fs))
    p = signal.lfilter([1.0 - a], [1.0, -a], x2**2, axis=0)
    if frame is None:
        idx = np.arange(n)
    else:
        nf = _frame_samples(frame, fs)
        idx = np.arange(nf - 1, n, nf)
        if idx.size == 0:
            raise ValueError("signal shorter than one frame")
    return idx / fs, restore(_power_db(p[idx], ref), one)


def combine_channels(levels_db, method="energy", *, axis=-1):
    """Combine levels of several channels (e.g. two ears) into one.

    ``"energy"``: energy (power) average, ``10 log10(mean(10**(L/10)))``,
    the level of the mean square pressure over the channels (binaural
    energy averaging). ``"sum"``: energy sum ``10 log10(sum(10**(L/10)))``,
    the level of uncorrelated sources acting together. ``"max"``: the
    largest channel level. NaN in any channel propagates.

    Parameters
    ----------
    levels_db : array_like
        Levels in dB (any common reference), channels along ``axis``.
    method : {"energy", "sum", "max"}, default "energy"
    axis : int, default -1
        Channel axis.

    Returns
    -------
    ndarray or float
        Combined level in dB, same reference as the input.

    Inputs and output are levels in dB of power (RMS) quantities; no peak/RMS choice applies.

    References
    ----------
    IEC 61672-1:2013. Electroacoustics - Sound level meters - Part 1:
    Specifications (levels as logarithms of mean-square quantities).
    """
    L = np.asarray(levels_db, dtype=float)
    if method == "max":
        return np.max(L, axis=axis)
    if method not in ("energy", "sum"):
        raise ValueError(f"method must be 'energy', 'sum' or 'max', got {method!r}")
    p = 10.0 ** (L / 10.0)
    p = np.mean(p, axis=axis) if method == "energy" else np.sum(p, axis=axis)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(p)


# --------------------------------------------------------------------------
# Calibrated resampling
# --------------------------------------------------------------------------

_RESAMPLE_PASS = 0.45   # pass-band edge, fraction of the lower of the two rates
_RESAMPLE_STOP_DB = 100.0
_MAX_RATE_TERM = 2**16


def _rational_ratio(fs, fs_out):
    ratio = Fraction(fs_out) / Fraction(fs)
    approx = ratio.limit_denominator(_MAX_RATE_TERM)
    if approx.numerator > _MAX_RATE_TERM or abs(float(approx) - fs_out / fs) > 1e-12 * fs_out / fs:
        raise ValueError(
            f"fs_out / fs = {fs_out / fs!r} is not a ratio of integers up to {_MAX_RATE_TERM}"
        )
    return approx.numerator, approx.denominator


def _resampling_fir(up, down):
    """Kaiser low-pass at the up-sampled rate for rational resampling.

    Pass band 0 ... 0.45 min(fs_in, fs_out), stop band from
    0.5 min(fs_in, fs_out) with 100 dB attenuation, unit DC gain
    (``scipy.signal.resample_poly`` multiplies it by ``up``). Frequencies
    are normalised to the input rate (fs_in = 1).
    """
    lo = min(1.0, up / down)                 # min(fs_in, fs_out) / fs_in
    nyq_up = 0.5 * up                        # Nyquist of the up-sampled rate
    width = (0.5 - _RESAMPLE_PASS) * lo / nyq_up
    numtaps, beta = signal.kaiserord(_RESAMPLE_STOP_DB, width)
    numtaps |= 1                             # odd: integer group delay
    cutoff = 0.5 * (_RESAMPLE_PASS + 0.5) * lo / nyq_up
    h = signal.firwin(numtaps, cutoff, window=("kaiser", beta))
    return h


def to_calibrated_48k(x, fs, *, calibration=1.0, fs_out=48000.0):
    """Calibrate to Pa and resample to 48 kHz (or ``fs_out``) by polyphase filtering.

    Returns ``calibration * x`` resampled from ``fs`` to ``fs_out`` with a
    rational factor ``up / down = fs_out / fs`` (reduced fraction, i.e. from
    the greatest common divisor for integer rates; 44.1 -> 48 kHz uses
    160 / 147). The polyphase filter is a linear-phase Kaiser-window
    low-pass designed here: pass band 0 ... 0.45·min(fs, fs_out) with ripple
    below 0.001 dB, so amplitudes (peak and RMS alike) and levels of
    components in the pass band are preserved; stop band from
    0.5·min(fs, fs_out) with at least 100 dB attenuation, so images (up-
    sampling) and aliases (down-sampling) are suppressed by 100 dB; the
    band 0.45 ... 0.5·min(fs, fs_out) is a roll-off. The output is aligned
    with the input (the filter delay is compensated; sample 0 is at t = 0).
    The ends are computed with zero padding, so about half a filter length
    at each end (1.5 ms for 44.1 -> 48 kHz) contains the filter's edge
    transient.

    No gain is applied other than ``calibration``, which converts the file
    unit to Pa (e.g. Pa per full-scale unit, from a calibrator recording);
    pass it explicitly. The default 1.0 states that ``x`` is already in Pa.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal in file units.
    fs : float
        Sampling rate of ``x`` in Hz.
    calibration : float, default 1.0
        Factor from file units to Pa.
    fs_out : float, default 48000.0
        Output rate in Hz.

    Returns
    -------
    ndarray, shape (ceil(n · up / down),) or (ceil(n · up / down), ch)
        Calibrated signal in Pa at ``fs_out``, float64.

    References
    ----------
    Crochiere, R. E., & Rabiner, L. R. (1983). Multirate Digital Signal
    Processing, ch. 2-4. Prentice-Hall.
    Kaiser, J. F. (1974). Nonrecursive digital filter design using the
    I0-sinh window function. Proc. IEEE Int. Symp. Circuits and Systems,
    20-23.
    """
    fs, fs_out, calibration = float(fs), float(fs_out), float(calibration)
    if not (fs > 0 and fs_out > 0):
        raise ValueError("sampling rates must be positive")
    if not np.isfinite(calibration):
        raise ValueError("calibration must be finite")
    x = np.asarray(x, dtype=float)
    if x.ndim not in (1, 2):
        raise ValueError("signal must have shape (n,) or (n, ch)")
    y = calibration * x
    up, down = _rational_ratio(fs, fs_out)
    if up == down:
        return y
    h = _resampling_fir(up, down)
    return signal.resample_poly(y, up, down, axis=0, window=h)


# --------------------------------------------------------------------------
# Octave and fractional-octave bands (IEC 61260-1)
# --------------------------------------------------------------------------

_R10 = (1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0)
_FR = 1000.0


def _ratio(base):
    if base == 10:
        return 10.0 ** 0.3
    if base == 2:
        return 2.0
    raise ValueError(f"base must be 10 or 2, got {base!r}")


def _check_fraction(fraction):
    b = int(fraction)
    if b != fraction or b < 1:
        raise ValueError("fraction must be a positive integer (1 = octave, 3 = third octave)")
    return b


def _band_exponent(index, b):
    """Exponent e of fm = fr G**e for band index x (IEC 61260-1)."""
    index = np.asarray(index, dtype=float)
    return index / b if b % 2 else (2.0 * index + 1.0) / (2.0 * b)


def _nominal(index, b, exact):
    if b in (1, 3):
        k = np.asarray(index, dtype=int) * (3 // b)   # position in the R10 series
        mant = np.array([_R10[i] for i in np.mod(k, 10)])
        return mant * 10.0 ** (3 + np.floor_divide(k, 10))
    out = np.empty_like(exact)
    for i, f in enumerate(exact):
        digits = 3 - int(math.floor(math.log10(f))) - 1
        out[i] = round(f, digits)
    return out


def octave_band_centres(fraction=3, fmin=20.0, fmax=20000.0, base=10):
    """Exact and nominal mid-band frequencies of octave or fractional-octave bands.

    After IEC 61260-1 with reference frequency fr = 1 kHz and octave
    frequency ratio G = 10**(3/10) (``base=10``, the standard's preferred
    choice) or G = 2 (``base=2``). For a 1/b-octave band of index x the exact
    mid-band frequency is ``fm = fr G**(x/b)`` for odd b and
    ``fm = fr G**((2x + 1)/(2b))`` for even b; the band edges are
    ``fm G**(∓1/(2b))``. Nominal labels: for octave and one-third-octave
    bands the preferred numbers of the R10 series (..., 20, 25, 31.5, 40,
    ..., 16000, 20000 Hz; ISO 266) at the same position, for both bases;
    for other fractions the exact value rounded to three significant
    figures (a label only, not a standardised nominal value).

    The bands returned are those whose exact mid-band frequencies are
    nearest (on a log scale) to ``fmin`` and ``fmax``, and all in between,
    so nominal values such as 20 Hz and 20 kHz select the bands labelled so.

    Parameters
    ----------
    fraction : int, default 3
        b in 1/b octave (1 = octave, 3 = one-third octave, 6, 12, 24, ...).
    fmin, fmax : float, default 20.0, 20000.0
        Range of mid-band frequencies in Hz.
    base : {10, 2}, default 10

    Returns
    -------
    exact : ndarray
        Exact mid-band frequencies in Hz, increasing.
    nominal : ndarray
        Nominal mid-band frequencies in Hz (labels).

    No amplitudes are involved (frequencies in Hz).

    References
    ----------
    IEC 61260-1:2014. Electroacoustics - Octave-band and fractional-octave-
    band filters - Part 1: Specifications.
    ISO 266:1997. Acoustics - Preferred frequencies.
    """
    exact, nominal, _ = _bands(fraction, fmin, fmax, base)
    return exact, nominal


def _bands(fraction, fmin, fmax, base):
    b = _check_fraction(fraction)
    G = _ratio(base)
    fmin, fmax = float(fmin), float(fmax)
    if not (0 < fmin <= fmax):
        raise ValueError("need 0 < fmin <= fmax")

    def nearest(f):
        e = math.log(f / _FR) / math.log(G)
        return int(round(e * b)) if b % 2 else int(round(e * b - 0.5))

    index = np.arange(nearest(fmin), nearest(fmax) + 1)
    exact = _FR * G ** _band_exponent(index, b)
    return exact, _nominal(index, b, exact), index


@dataclass
class OctaveFilterbank:
    """Octave or fractional-octave Butterworth filter bank (see :func:`octave_filterbank`).

    Attributes
    ----------
    fs : float
        Input sampling rate in Hz.
    fraction, base, order : int
        Bandwidth 1/fraction octave, G = 10**0.3 (base 10) or 2, Butterworth
        prototype order (each band filter has ``2 order`` poles).
    exact, nominal, lower, upper : ndarray, shape (n_bands,)
        Exact and nominal mid-band frequencies and the band-edge
        frequencies (-3 dB points) in Hz.
    enbw : ndarray, shape (n_bands,)
        Effective (noise) bandwidth of the analogue design in Hz,
        ``(upper - lower) (π/(2 order)) / sin(π/(2 order))``.
    stages : ndarray of int, shape (n_bands,)
        Rate changes before each band filter: ``k > 0`` decimate-by-2
        stages, ``k < 0`` interpolate-by-2 stages, 0 none.
    rates : ndarray, shape (n_bands,)
        Sampling rate of each band filter, ``fs / 2**stages``, in Hz.
    sos : list of ndarray
        Band-pass second-order sections, one array per band, at ``rates``.
    decimator : ndarray
        Linear-phase FIR of every decimate-by-2 stage (unit DC gain).
    interpolators : list of ndarray
        Linear-phase FIR of interpolate-by-2 stage j = 1, 2, ... (unit DC
        gain, applied with gain 2 to the zero-stuffed signal).
    """

    fs: float
    fraction: int
    base: int
    order: int
    exact: np.ndarray
    nominal: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    enbw: np.ndarray
    stages: np.ndarray
    rates: np.ndarray
    sos: list = field(repr=False)
    decimator: np.ndarray = field(repr=False)
    interpolators: list = field(repr=False)

    def _components(self, f):
        """Per stage k: list of (frequency, gain) reaching rate fs / 2**k for a unit sine at f."""
        comps = {0: [(f, np.ones_like(f))]}
        rate, fc, gc = self.fs, f, np.ones_like(f)
        for k in range(1, max(0, int(self.stages.max())) + 1):
            _, h = signal.freqz(self.decimator, worN=fc, fs=rate)
            rate /= 2.0
            fc = np.abs((fc + 0.5 * rate) % rate - 0.5 * rate)   # alias
            gc = gc * np.abs(h)
            comps[k] = [(fc, gc)]
        rate, cur = self.fs, comps[0]
        for j in range(1, max(0, -int(self.stages.min())) + 1):
            new = []
            for fc, gc in cur:
                for fi in (fc, rate - fc):                          # signal and image
                    _, h = signal.freqz(self.interpolators[j - 1], worN=fi, fs=2.0 * rate)
                    new.append((fi, gc * np.abs(h)))
            rate *= 2.0
            comps[-j] = cur = new
        return comps

    def frequency_response(self, f):
        """Magnitude gain of each band for a steady sine at frequency ``f``.

        Includes the rate-change stages. Below the Nyquist frequency of a
        decimated band this is the cascade's transfer function; above it,
        the gain with which the sine's alias reaches the band output (the
        anti-alias attenuation times the band filter at the alias
        frequency). For an interpolated band, the image components created
        by each interpolation stage are added in power.

        Parameters
        ----------
        f : array_like
            Frequencies in Hz, 0 ... fs/2.

        Returns
        -------
        ndarray, shape (len(f), n_bands)
            Linear magnitude gain (output RMS / input RMS of the sine; the
            same ratio for peak amplitudes).
        """
        f = np.atleast_1d(np.asarray(f, dtype=float))
        if np.any(f < 0) or np.any(f > 0.5 * self.fs * (1 + 1e-12)):
            raise ValueError("frequencies must lie in 0 ... fs/2")
        comps = self._components(f)
        out = np.empty((f.size, self.exact.size))
        for i, (sos, k, r) in enumerate(zip(self.sos, self.stages, self.rates)):
            p = np.zeros_like(f)
            for fc, gc in comps[int(k)]:
                _, h = signal.sosfreqz(sos, worN=fc, fs=r)
                p += (gc * np.abs(h)) ** 2
            out[:, i] = np.sqrt(p)
        return out

    def apply(self, x):
        """Filter ``x`` (shape ``(n,)`` or ``(n, ch)``) through every band.

        Returns
        -------
        list of ndarray
            Band signals, band ``i`` at rate ``rates[i]`` (sample j at time
            ``j / rates[i]``), same unit and channels as ``x``.
        """
        x2, one = as_2d(x)
        return [restore(y, one) for y in self._bands_2d(x2)]

    def _bands_2d(self, x2):
        sig = {0: x2}
        for k in range(1, max(0, int(self.stages.max())) + 1):
            if sig[k - 1].shape[0] < 2:
                raise ValueError("signal too short for the decimation stages")
            sig[k] = signal.resample_poly(sig[k - 1], 1, 2, axis=0, window=self.decimator)
        for j in range(1, max(0, -int(self.stages.min())) + 1):
            sig[-j] = signal.resample_poly(sig[-j + 1], 2, 1, axis=0,
                                           window=self.interpolators[j - 1])
        for sos, k in zip(self.sos, self.stages):
            yield signal.sosfilt(sos, sig[int(k)], axis=0)


_STAGE_STOP_DB = 100.0


def _decimator_fir():
    """Decimate-by-2 FIR: pass band to 0.2 fs_in, stop band from 0.3 fs_in, 100 dB."""
    numtaps, beta = signal.kaiserord(_STAGE_STOP_DB, 0.2)   # width 0.1 fs_in in Nyquist units
    return signal.firwin(numtaps | 1, 0.5, window=("kaiser", beta))


def _interpolator_fir(fp, rate_in):
    """Interpolate-by-2 FIR at 2 rate_in: pass band to fp, stop band from rate_in - fp, 100 dB."""
    width = 1.0 - 2.0 * fp / rate_in                         # Nyquist units of 2 rate_in
    numtaps, beta = signal.kaiserord(_STAGE_STOP_DB, width)
    return signal.firwin(numtaps | 1, 0.5, window=("kaiser", beta))


def octave_filterbank(fs, fraction=3, fmin=20.0, fmax=20000.0, order=3, *, base=10,
                      multirate=True, oversampling=(5.0, 20.0)):
    """Design an octave or fractional-octave band filter bank after IEC 61260-1.

    Bands as in :func:`octave_band_centres`. Each band is a digital
    Butterworth band-pass of prototype order ``order`` (``2 order`` poles,
    ``order`` second-order sections) whose -3 dB points are the band edges
    ``fm G**(∓1/(2b))``, obtained by the bilinear transform with
    pre-warped edges (so the edges are exact at every rate). The analogue
    magnitude is ``1 / sqrt(1 + Q**(2 order))`` with
    ``Q = (Ω - 1/Ω) / (G**(1/(2b)) - G**(-1/(2b)))``, Ω = f / fm; with the
    default order 3 and multirate processing it lies inside the class-1
    acceptance limits for relative attenuation of IEC 61260-1 (verified in
    the test suite at 48 and 44.1 kHz for octave and one-third-octave
    bands from 20 Hz to the highest band below Nyquist). The effective
    bandwidth is ``(π/(2 order)) / sin(π/(2 order))`` times the nominal
    bandwidth (1.047, i.e. +0.20 dB, for order 3).

    Multirate (``multirate=True``, ``oversampling=(lo, hi)``): each band
    filter runs at a rate ``fs 2**-k`` chosen from the ratio of that rate to
    the band's upper edge.

    * Low bands (``fs / upper >= hi``) are run at the lowest rate whose
      ratio is still >= ``hi`` (decimation by 2**k): at a high rate their
      poles crowd the unit circle. The input is decimated by 2 repeatedly
      (stages shared by all bands) with a linear-phase Kaiser FIR (pass band
      to 0.2, stop band from 0.3 of the stage input rate, 100 dB, delay
      compensated). The band lies far inside the decimator's pass band
      (ripple < 0.001 dB); every alias is either suppressed by 100 dB or
      lands above 0.4 times the band rate, on the band filter's far upper
      skirt (the class-1 mask, aliases included, is verified through
      :meth:`OctaveFilterbank.frequency_response`). The response then
      matches the single-rate design within 0.03 dB down to -10 dB and
      0.1 dB down to -20 dB (bilinear warping at the lower rate).
    * High bands (``fs / upper < lo``) are run at the lowest rate
      ``fs 2**u`` whose ratio is >= ``lo`` (interpolation by 2**u). Near
      Nyquist the bilinear transform compresses the lower skirt of a band
      filter so much that it violates the class-1 stop-band limits (by
      ~10 dB for the 20 kHz third-octave band at 48 kHz); at a ratio of 5
      the warping is harmless. Each interpolation stage is a linear-phase
      Kaiser FIR with pass band to the highest upper edge it serves and
      stop band from the mirror frequency, 100 dB, so images are 100 dB
      down except those of components between that edge and Nyquist,
      which fall on the band's upper skirt.
    * Bands with ``lo <= fs / upper < hi`` run at ``fs``.

    ``multirate=False`` runs every band at ``fs`` (then bands with
    ``fs / upper`` below about 5 do not meet the class-1 limits). A band
    whose upper edge is not below fs/2 cannot be realised and raises
    ``ValueError`` (e.g. the 20 kHz third-octave band at 44.1 kHz; lower
    ``fmax``).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    fraction : int, default 3
        b in 1/b octave.
    fmin, fmax : float, default 20.0, 20000.0
        Range of mid-band frequencies in Hz (nominal values select their
        bands).
    order : int, default 3
        Butterworth prototype order.
    base : {10, 2}, default 10
        Octave ratio G = 10**0.3 or 2.
    multirate : bool, default True
    oversampling : (float, float), default (5.0, 20.0)
        ``(lo, hi)`` limits on the ratio of a band's filter rate to its
        upper edge frequency, ``2 < lo <= hi``, ``hi >= 8``.

    Returns
    -------
    OctaveFilterbank

    The filters have a dimensionless amplitude response; band levels computed with them are RMS.

    References
    ----------
    IEC 61260-1:2014. Electroacoustics - Octave-band and fractional-octave-
    band filters - Part 1: Specifications.
    ANSI S1.11-2004. Specification for octave-band and fractional-octave-band
    analog and digital filters.
    Crochiere, R. E., & Rabiner, L. R. (1983). Multirate Digital Signal
    Processing. Prentice-Hall.
    Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal
    Processing (3rd ed.), sec. 4.6 and 7.2. Pearson.
    """
    fs = float(fs)
    if not fs > 0:
        raise ValueError("fs must be positive")
    order = int(order)
    if order < 1:
        raise ValueError("order must be a positive integer")
    lo_ratio, hi_ratio = (float(v) for v in oversampling)
    if multirate and not (2.0 < lo_ratio <= hi_ratio and hi_ratio >= 8.0):
        raise ValueError("oversampling must be (lo, hi) with 2 < lo <= hi and hi >= 8")
    b = _check_fraction(fraction)
    exact, nominal, _ = _bands(b, fmin, fmax, base)
    G = _ratio(base)
    half = G ** (1.0 / (2 * b))
    lower, upper = exact / half, exact * half
    bad = upper >= 0.5 * fs
    if np.any(bad):
        raise ValueError(
            f"band {nominal[bad][0]:g} Hz has its upper edge {upper[bad][0]:.1f} Hz at or "
            f"above fs/2 = {0.5 * fs:g} Hz; lower fmax"
        )
    stages = np.zeros(exact.size, dtype=int)
    if multirate:
        ratio = fs / upper
        eps = 1e-12
        down = ratio >= hi_ratio
        stages[down] = np.floor(np.log2(ratio[down] / hi_ratio) + eps).astype(int)
        up = ratio < lo_ratio
        stages[up] = -np.ceil(np.log2(lo_ratio / ratio[up]) - eps).astype(int)
    rates = fs / 2.0 ** stages
    sos = [
        signal.butter(order, [lo, hi], btype="bandpass", fs=r, output="sos")
        for lo, hi, r in zip(lower, upper, rates)
    ]
    interpolators = []
    for j in range(1, max(0, -int(stages.min())) + 1):
        interpolators.append(_interpolator_fir(np.max(upper[stages <= -j]), fs * 2.0 ** (j - 1)))
    x = math.pi / (2 * order)
    enbw = (upper - lower) * x / math.sin(x)
    return OctaveFilterbank(
        fs=fs, fraction=b, base=int(base), order=order, exact=exact, nominal=nominal,
        lower=lower, upper=upper, enbw=enbw, stages=stages, rates=rates, sos=sos,
        decimator=_decimator_fir(), interpolators=interpolators,
    )


@dataclass
class BandLevels:
    """Band levels from :func:`octave_levels`.

    Attributes
    ----------
    exact, nominal, lower, upper : ndarray, shape (n_bands,)
        Exact and nominal mid-band frequencies and band edges in Hz.
    levels : ndarray
        RMS band levels in dB re ``ref``: shape ``(n_bands,)`` or
        ``(n_bands, ch)`` overall, ``(n_frames, n_bands)`` or
        ``(n_frames, n_bands, ch)`` per frame.
    t : ndarray or None
        Frame centre times in s (None for overall levels).
    frame : float or None
        Frame length actually used, in s (whole input samples).
    ref : float
        Reference RMS value of the levels.
    filterbank : OctaveFilterbank
    """

    exact: np.ndarray
    nominal: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    levels: np.ndarray
    t: np.ndarray
    frame: float
    ref: float
    filterbank: OctaveFilterbank = field(repr=False)


def _resolve_bank(fs, filterbank, kw):
    if filterbank is None:
        return octave_filterbank(fs, **kw)
    if not math.isclose(filterbank.fs, float(fs)):
        raise ValueError("filterbank was designed for a different sampling rate")
    return filterbank


def _band_power_frames(bank, x2, fs, i0, nf):
    """Mean square of every band over frames [i0 + j nf, i0 + (j+1) nf) of the input grid.

    nf=None: one frame from i0 to the end. Returns (n_frames, n_bands, ch).
    """
    n = x2.shape[0]
    if nf is None:
        edges = np.array([i0, n], dtype=float)
    else:
        m = (n - i0) // nf
        if m < 1:
            raise ValueError("signal (after skip) shorter than one frame")
        edges = i0 + nf * np.arange(m + 1, dtype=float)
    out = np.empty((edges.size - 1, bank.exact.size, x2.shape[1]))
    for i, (y, r) in enumerate(zip(bank._bands_2d(x2), bank.rates)):
        e = np.round(edges * (r / fs)).astype(int)
        e = np.clip(e, 0, y.shape[0])
        c = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y**2, axis=0)])
        cnt = np.diff(e)
        if np.any(cnt < 1):
            raise ValueError("frame shorter than one sample of a decimated band; "
                             "use a longer frame")
        out[:, i, :] = (c[e[1:]] - c[e[:-1]]) / cnt[:, None]
    return out, edges


def octave_levels(x, fs, fraction=3, fmin=20.0, fmax=20000.0, order=3, *, base=10,
                  frame=None, skip=0.0, ref=20e-6, multirate=True, filterbank=None):
    """Octave or fractional-octave band levels after IEC 61260-1.

    The signal is passed once (causally) through the filter bank of
    :func:`octave_filterbank` and the mean square of each band output is
    taken over the whole signal after ``skip`` (overall band levels, i.e.
    band equivalent continuous levels) or over consecutive non-overlapping
    frames of ``frame`` s starting at ``skip``. Band levels are RMS levels in
    dB re ``ref``: a sine of peak amplitude A at a mid-band frequency gives
    ``20 log10(A / (sqrt(2) ref))`` in its band; stationary white noise of
    one-sided power spectral density S gives ``10 log10(S enbw / ref²)``
    with ``enbw`` the effective bandwidth (1.047 times the nominal
    bandwidth for order 3). No frequency weighting is applied (weight the
    signal first if needed).

    Edge handling: filters start from zero state, so each band output
    begins with a transient of a few times 1 / bandwidth (a steady sine
    reads within 0.02 dB after 0.5 s in the 20 Hz third-octave band, after
    a few ms at 1 kHz); pass ``skip`` (s) to exclude it from all bands. Frame levels are additionally delayed by the
    band filter's group delay (of the same order) relative to the input. In
    decimated bands, frame boundaries are rounded to the band's sample
    grid.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal in the unit of ``ref`` (Pa by default).
    fs : float
        Sampling rate in Hz.
    fraction, fmin, fmax, order, base, multirate
        Filter bank, see :func:`octave_filterbank`.
    frame : float, optional
        Frame length in s (rounded to whole input samples).
    skip : float, default 0.0
        Initial duration in s excluded from the levels.
    ref : float, default 20e-6
        Reference RMS value in the unit of ``x``.
    filterbank : OctaveFilterbank, optional
        Precomputed bank for rate ``fs`` (then the design arguments are
        ignored).

    Returns
    -------
    BandLevels

    References
    ----------
    IEC 61260-1:2014. Electroacoustics - Octave-band and fractional-octave-
    band filters - Part 1: Specifications.
    ANSI S1.11-2004. Specification for octave-band and fractional-octave-band
    analog and digital filters.
    """
    ref = _check_ref(ref)
    fs = float(fs)
    bank = _resolve_bank(fs, filterbank, dict(fraction=fraction, fmin=fmin, fmax=fmax,
                                               order=order, base=base, multirate=multirate))
    x2, one = as_2d(x)
    i0 = int(round(float(skip) * fs))
    if not 0 <= i0 < x2.shape[0]:
        raise ValueError("skip must be non-negative and shorter than the signal")
    nf = None if frame is None else _frame_samples(frame, fs)
    p, edges = _band_power_frames(bank, x2, fs, i0, nf)
    L = _power_db(p, ref)
    if nf is None:
        L, t, frame_used = L[0], None, None
    else:
        t = 0.5 * (edges[:-1] + edges[1:]) / fs
        frame_used = nf / fs
    return BandLevels(
        exact=bank.exact.copy(), nominal=bank.nominal.copy(), lower=bank.lower.copy(),
        upper=bank.upper.copy(), levels=restore(L, one), t=t, frame=frame_used, ref=ref,
        filterbank=bank,
    )


@dataclass
class BandRpmLevels:
    """Band levels binned along rpm, from :func:`band_levels_vs_rpm`.

    Attributes
    ----------
    rpm_edges : ndarray, shape (n_bins + 1,)
        Rpm bin edges.
    rpm_centres : ndarray, shape (n_bins,)
        Bin centres in rpm.
    exact, nominal : ndarray, shape (n_bands,)
        Exact and nominal mid-band frequencies in Hz.
    levels : ndarray, shape (n_bins, n_bands) or (n_bins, n_bands, ch)
        Energy-mean RMS band level of the frames in each bin, dB re
        ``ref``; NaN for bins with fewer than ``min_count`` frames.
    counts : ndarray of int, shape (n_bins,)
        Number of frames in each bin.
    frames : BandLevels
        The frame levels that were binned.
    frame_rpm : ndarray, shape (n_frames,)
        Mean rpm over each frame.
    ref : float
        Reference RMS value of the levels.
    """

    rpm_edges: np.ndarray
    rpm_centres: np.ndarray
    exact: np.ndarray
    nominal: np.ndarray
    levels: np.ndarray
    counts: np.ndarray
    frames: BandLevels = field(repr=False)
    frame_rpm: np.ndarray = field(repr=False)
    ref: float = 20e-6


def band_levels_vs_rpm(x, fs, rpm, *, rpm_bins, frame, fraction=3, fmin=20.0, fmax=20000.0,
                       order=3, base=10, skip=0.0, ref=20e-6, min_count=1, multirate=True,
                       filterbank=None):
    """Octave or fractional-octave band levels binned along rpm.

    Frame band levels are computed with :func:`octave_levels` (frames of
    ``frame`` s from ``skip``); each frame is assigned the mean of ``rpm``
    over its samples, and frames are grouped by the bins ``rpm_bins``
    (edges; a frame belongs to ``[edge_i, edge_i+1)``, the last bin also
    includes its upper edge; frames outside all bins are ignored). The
    level of a bin is the energy mean of its frames,
    ``10 log10(mean(10**(L/10)))``, per band and channel, i.e. the band
    level of all the bin's frames taken together. Binning along rpm is
    meaningful on a monotonic segment at fixed gear (pass the segment
    only); the band filters' group delay (see :func:`octave_levels`) shifts
    low bands slightly towards later frames in a fast run-up.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal in the unit of ``ref`` (Pa by default).
    fs : float
        Sampling rate in Hz.
    rpm : array_like, shape (n,) or float
        Speed in rpm at every sample.
    rpm_bins : array_like
        Increasing rpm bin edges.
    frame : float
        Frame length in s.
    fraction, fmin, fmax, order, base, multirate, skip, ref, filterbank
        See :func:`octave_levels`.
    min_count : int, default 1
        Bins with fewer frames are set to NaN.

    Returns
    -------
    BandRpmLevels

    Band levels are RMS levels in dB re ``ref`` (energy means per rpm bin).

    References
    ----------
    IEC 61260-1:2014. Electroacoustics - Octave-band and fractional-octave-
    band filters - Part 1: Specifications.
    Brandt, A. (2011). Noise and Vibration Analysis: Signal Analysis and
    Experimental Procedures (analysis of rotating machinery). Wiley.
    """
    fs = float(fs)
    edges_rpm = np.asarray(rpm_bins, dtype=float)
    if edges_rpm.ndim != 1 or edges_rpm.size < 2 or np.any(np.diff(edges_rpm) <= 0):
        raise ValueError("rpm_bins must be at least two strictly increasing edges")
    min_count = int(min_count)
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    x2, _ = as_2d(x)
    rpm = broadcast_rpm(rpm, x2.shape[0])
    bl = octave_levels(x, fs, fraction, fmin, fmax, order, base=base, frame=frame, skip=skip,
                       ref=ref, multirate=multirate, filterbank=filterbank)
    nf = int(round(bl.frame * fs))
    i0 = int(round(float(skip) * fs))
    m = bl.t.size
    frame_rpm = rpm[i0: i0 + m * nf].reshape(m, nf).mean(axis=1)
    idx = np.searchsorted(edges_rpm, frame_rpm, side="right") - 1
    idx[frame_rpm == edges_rpm[-1]] = edges_rpm.size - 2
    nb = edges_rpm.size - 1
    valid = (idx >= 0) & (idx < nb)
    counts = np.bincount(idx[valid], minlength=nb)
    L = bl.levels  # (frames, bands[, ch])
    p = 10.0 ** (L / 10.0)
    psum = np.zeros((nb,) + p.shape[1:])
    np.add.at(psum, idx[valid], p[valid])
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = psum / counts.reshape((nb,) + (1,) * (p.ndim - 1))
        levels = 10.0 * np.log10(mean)
    levels[counts < min_count] = np.nan
    return BandRpmLevels(
        rpm_edges=edges_rpm, rpm_centres=0.5 * (edges_rpm[:-1] + edges_rpm[1:]),
        exact=bl.exact, nominal=bl.nominal, levels=levels, counts=counts, frames=bl,
        frame_rpm=frame_rpm, ref=bl.ref,
    )


# Tone and order prominence (ECMA-418-1 / ECMA-74), implemented in _prominence.
from ._prominence import (  # noqa: E402
    OrderComponent,
    OrderProminence,
    ToneProminence,
    components,
    critical_band_edges,
    critical_bandwidth,
    order_prominence,
    prominence_criterion,
    tone_prominence,
)

__all__ += [
    "OrderComponent",
    "OrderProminence",
    "ToneProminence",
    "components",
    "critical_band_edges",
    "critical_bandwidth",
    "order_prominence",
    "prominence_criterion",
    "tone_prominence",
]
