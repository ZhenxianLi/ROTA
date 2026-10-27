"""Transfer-path, background-noise and measurement-chain models for test signals.

The functions here sit between a clean synthetic source (for example
:func:`rota.synth.synth_runup`) and the signal an analysis chain sees:

* :func:`modal_path` and :func:`modal_response` -- a structural or acoustic
  transfer path as a parallel bank of second-order resonant modes. Unlike the
  quasi-static resonances of ``synth_runup`` it has memory: a component that
  sweeps through a mode shows the ring-down and the phase rotation of a real
  resonator.
* :func:`room_impulse_response` and :func:`apply_ir` -- a statistical
  (exponentially decaying Gaussian noise) reverberant impulse response with a
  stated reverberation time, and its application by fast convolution.
* :func:`speed_noise` -- coloured broadband noise (rolling or aerodynamic
  noise) whose level follows the vehicle speed through a log-speed law, built
  with :func:`rota._broadband.synthesize_broadband`.
* :func:`transient_events` -- discrete thumps and clicks (road bumps, door or
  switch noise).
* :func:`measurement_chain` and :func:`delay_signal` -- the recorder: gain,
  self-noise, DC offset, hard clipping, quantisation and sampling-clock drift,
  and the fractional delay that misaligns two channels in time.

Units: time in s, frequency in Hz, vehicle speed in km/h. Signals have shape
``(n,)`` or ``(n, ch)`` and every operator acts on each channel independently
and linearly where it is linear at all (design principle 6). Each docstring
states whether an amplitude is a peak or an RMS value. Randomness is
seedable: an integer seed, a :class:`numpy.random.Generator` or ``None`` (fresh
entropy, recorded in the configuration so that the result can be rebuilt).

Return convention. The generators (:func:`modal_path`,
:func:`room_impulse_response`, :func:`speed_noise`, :func:`transient_events`)
return ``(array, config)``, where ``config`` is a JSON-serialisable ``dict`` of
the resolved inputs (and, where there is any, of the resolved truth), suitable
for a provenance sidecar. :func:`measurement_chain` returns a ``dict`` that
contains the same ``config``. :func:`modal_response`, :func:`apply_ir` and
:func:`delay_signal` are plain transformations and return an array.
"""

import math
from collections.abc import Mapping
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from ._broadband import synthesize_broadband
from ._provenance import jsonable
from ._utils import as_2d, restore

__all__ = [
    "modal_path",
    "modal_response",
    "room_impulse_response",
    "apply_ir",
    "speed_noise",
    "transient_events",
    "measurement_chain",
    "delay_signal",
]

_TWO_PI = 2.0 * math.pi
_LN_1000 = math.log(1000.0)  # 60 dB of amplitude decay
_DB_PER_OCTAVE = 10.0 * math.log10(2.0)  # a power ratio of 2, in dB
_MODE_KEYS = ("freq", "zeta", "q", "gain")
_EVENT_KEYS = ("time", "kind", "amplitude", "decay", "freq")
_EVENT_KINDS = ("thump", "click")
_LEVEL_REFS = ("rms", "psd")
_SPEED_GRID_STEP = 0.03  # ln of the largest ratio of neighbouring speed rows

# Band-limited interpolation kernel: Kaiser-windowed sinc with its cut-off at
# the Nyquist frequency, 2 * _HALF_WIDTH taps, tabulated on _PHASES + 1
# fractional offsets and linearly interpolated between them.
_HALF_WIDTH = 32
_KAISER_BETA = 10.0
_PHASES = 1024
_INTERP_CHUNK = 1 << 21  # kernel-times-signal elements per block


# ==========================================================================
# Shared helpers
# ==========================================================================


def _version():
    from . import __version__

    return __version__


def _finite(value, what, *, lo=None, hi=None, lo_open=False, hi_open=False):
    """Float from ``value`` that must be finite (and within the optional limits)."""
    try:
        v = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{what} must be a real number, got {value!r}") from err
    ok = bool(np.isfinite(v))
    if ok and lo is not None:
        ok = v > lo if lo_open else v >= lo
    if ok and hi is not None:
        ok = v < hi if hi_open else v <= hi
    if not ok:
        raise ValueError(f"{what}={value!r} is out of range")
    return v


def _check_fs(fs):
    return _finite(fs, "fs", lo=0.0, lo_open=True)


def _check_count(value, what, minimum=1):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{what} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _generator(seed):
    """Generator and the integer seed behind it (None for a user-supplied generator)."""
    if isinstance(seed, np.random.Generator):
        return seed, None
    if seed is None:
        used = int(np.random.default_rng().integers(0, 2**63 - 1))
    elif isinstance(seed, (int, np.integer)) and not isinstance(seed, bool):
        used = int(seed)
    else:
        raise TypeError("seed must be an int, a numpy.random.Generator or None")
    return np.random.default_rng(used), used


def _child_generators(rng, count):
    """``count`` independent generators; generator ``i`` depends only on ``rng`` and ``i``."""
    base = int(rng.integers(0, 2**63 - 1))
    return [np.random.default_rng(np.random.SeedSequence([base, i])) for i in range(count)]


def _per_channel(value, channels, what, *, lo=None, lo_open=False):
    """Scalar or one-value-per-channel input as a float array of shape ``(channels,)``."""
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        a = np.full(channels, float(a))
    elif a.shape != (channels,):
        raise ValueError(f"{what} must be a scalar or have one value per channel ({channels})")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{what} must be finite")
    if lo is not None and not (np.all(a > lo) if lo_open else np.all(a >= lo)):
        raise ValueError(f"{what} must be {'>' if lo_open else '>='} {lo}")
    return a


def _signal_2d(x, what="x"):
    x2, was_1d = as_2d(x)
    if not np.all(np.isfinite(x2)):
        raise ValueError(f"{what} contains non-finite values")
    return x2, was_1d


def _config(function, seed_used, **fields):
    cfg = {"rota_version": _version(), "function": function, "seed": seed_used}
    cfg.update(fields)
    return jsonable(cfg)


# ==========================================================================
# Band-limited interpolation (shared by delay_signal and the clock drift)
# ==========================================================================


@lru_cache(maxsize=1)
def _kernel_table():
    """Interpolation kernels for the fractional offsets ``p / _PHASES``, ``p = 0 .. _PHASES``.

    Row ``p`` holds the weights ``h(frac - j)`` of the samples ``x[i0 + j]``,
    ``j = -M + 1 .. M``, that give ``x(i0 + frac)``, where ``h`` is the sinc
    function (cut-off at the Nyquist frequency) times a Kaiser window of
    half-width ``M``. Every row is normalised to a sum of one, so a constant
    (DC) is interpolated exactly. Rows 0 and ``_PHASES`` are exact unit
    samples (an integer shift is lossless).
    """
    m = _HALF_WIDTH
    j = np.arange(-m + 1, m + 1)
    frac = np.arange(_PHASES + 1) / _PHASES
    tau = frac[:, None] - j[None, :]
    arg = np.clip(1.0 - (tau / m) ** 2, 0.0, None)
    table = np.sinc(tau) * np.i0(_KAISER_BETA * np.sqrt(arg)) / np.i0(_KAISER_BETA)
    table[0] = 0.0
    table[0, m - 1] = 1.0
    table[_PHASES] = 0.0
    table[_PHASES, m] = 1.0
    table /= table.sum(axis=1, keepdims=True)
    table.setflags(write=False)
    return table


def _interpolate(x2, pos):
    """Values of the band-limited signal behind ``x2`` at fractional sample positions.

    Parameters
    ----------
    x2 : ndarray, shape (n, ch)
        Samples at the integer positions ``0 .. n - 1``; zero outside.
    pos : ndarray, shape (m,)
        Positions in samples (any real values, increasing or not).

    Returns
    -------
    ndarray, shape (m, ch)
        ``sum_j x[i0 + j] h(pos - i0 - j)`` with ``i0 = floor(pos)``.
    """
    n, ch = x2.shape
    width = 2 * _HALF_WIDTH
    pad = np.zeros((width, ch))
    xp = np.concatenate((pad, x2, pad))
    windows = sliding_window_view(xp, width, axis=0)  # (n + 2 width - width + 1, ch, width)
    table = _kernel_table()
    base = np.floor(pos)
    q = (pos - base) * _PHASES
    p0 = np.minimum(q.astype(np.int64), _PHASES - 1)
    a = (q - p0)[:, None]
    start = np.clip(base.astype(np.int64) - _HALF_WIDTH + 1 + width, 0, n + width)
    out = np.empty((pos.size, ch))
    step = max(1, _INTERP_CHUNK // (width * ch))
    for s0 in range(0, pos.size, step):
        s1 = min(pos.size, s0 + step)
        kern = (1.0 - a[s0:s1]) * table[p0[s0:s1]] + a[s0:s1] * table[p0[s0:s1] + 1]
        out[s0:s1] = np.einsum("st,sct->sc", kern, windows[start[s0:s1]])
    return out


# ==========================================================================
# 1. Modal transfer path
# ==========================================================================


def _parse_modes(modes, fs):
    """Validate the mode list; ``fs=None`` skips the Nyquist check (analog use)."""
    if isinstance(modes, Mapping):
        modes = [modes]
    try:
        modes = list(modes)
    except TypeError as err:
        raise TypeError("modes must be a sequence of dicts") from err
    if not modes:
        raise ValueError("modes must contain at least one mode")
    nyq = np.inf if fs is None else 0.5 * fs
    out = []
    for i, m in enumerate(modes):
        what = f"modes[{i}]"
        if not isinstance(m, Mapping):
            raise TypeError(f"{what} must be a dict, got {type(m).__name__}")
        extra = sorted(set(m) - set(_MODE_KEYS))
        if extra:
            raise ValueError(f"unknown key(s) {extra} in {what}; allowed: {list(_MODE_KEYS)}")
        if m.get("freq") is None:
            raise ValueError(f"{what} is missing the required key 'freq'")
        f0 = _finite(m["freq"], f"{what}: freq", lo=0.0, lo_open=True, hi=nyq, hi_open=True)
        zeta, q = m.get("zeta"), m.get("q")
        if (zeta is None) == (q is None):
            raise ValueError(f"{what}: give exactly one of 'zeta' and 'q'")
        if zeta is not None:
            zeta = _finite(zeta, f"{what}: zeta", lo=0.0, hi=1.0, lo_open=True, hi_open=True)
            q = 0.5 / zeta
        else:
            q = _finite(q, f"{what}: q", lo=0.5, lo_open=True)
            zeta = 0.5 / q
        gain = _finite(1.0 if m.get("gain") is None else m["gain"], f"{what}: gain")
        out.append({"freq": f0, "zeta": zeta, "q": q, "gain": gain})
    return out


def _mode_coefficients(mode, fs):
    """Impulse-invariant resonator ``b1 z^-1 / (1 + a1 z^-1 + a2 z^-2)`` of one mode."""
    w0 = _TWO_PI * mode["freq"]
    sigma = mode["zeta"] * w0
    wd = w0 * math.sqrt(1.0 - mode["zeta"] ** 2)
    dt = 1.0 / fs
    r = math.exp(-sigma * dt)
    theta = wd * dt
    b1 = dt * mode["gain"] * w0**2 / wd * r * math.sin(theta)
    return b1, -2.0 * r * math.cos(theta), r * r


def modal_path(x, fs, modes, *, gain=1.0):
    """Filter a signal through a transfer path made of second-order resonant modes.

    The path is the sum of single-degree-of-freedom modes (Ewins 2000), each
    with the receptance-type transfer function

        ``H_k(s) = g_k w_k**2 / (s**2 + 2 zeta_k w_k s + w_k**2)``,

    ``w_k = 2 pi f_k``, so that ``g_k`` is the *static gain* (``H_k(0)``) and
    the gain at the resonance is ``g_k / (2 zeta_k) = g_k Q_k`` with a phase of
    -90 degrees. The path is ``H = gain * sum_k H_k``. The impulse response of
    a mode is ``g_k w_k**2 / w_dk exp(-zeta_k w_k t) sin(w_dk t)`` with
    ``w_dk = w_k sqrt(1 - zeta_k**2)``: after an excitation it rings at the
    damped frequency and its envelope decays with the time constant
    ``1 / (zeta_k w_k)`` (a 1/e fall of the amplitude).

    Realisation. Each mode is a digital resonator in a parallel bank, designed
    by *impulse invariance* (Oppenheim & Schafer 2010, sec. 7.1.1): the poles
    are exactly ``exp(s T)``, so the resonance frequency, the damped frequency
    and the ring-down time constant are the analog ones with no frequency
    warping, and the digital impulse response is ``T * h_k(n T)``. The
    transfer function is
    ``T g w**2 / w_d * r sin(th) z^-1 / (1 - 2 r cos(th) z^-1 + r**2 z^-2)``
    with ``r = exp(-zeta w T)`` and ``th = w_d T``. The price of impulse
    invariance is aliasing of the analog response (Poisson sum): the static
    gain of a mode is low by ``(pi**2 / 3) (f_k / fs)**2`` (0.12 % for
    ``f_k = 0.019 fs``), the error at other frequencies is of the same order
    and is relatively larger where the response is small, far above ``f_k``.
    Use :func:`modal_response` with ``fs`` for the response of the filter that
    is actually applied, and without ``fs`` for the analog ideal. A bilinear
    design with prewarping would match the response at ``f_k`` but warp the
    damping and therefore the ring-down.

    Because the filter has memory, a component that sweeps through a mode
    shows the true ring-down and phase rotation, which the quasi-static
    resonances of :func:`rota.synth.synth_runup` do not (they are only valid
    for a sweep slow compared with ``1 / (zeta_k w_k)``). The signal is
    assumed to be zero before the first sample (zero initial state), so the
    path starts to respond at ``n = 0``.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Input signal in any unit; every channel passes through the same path.
    fs : float
        Sampling rate in Hz.
    modes : sequence of dict
        One dict per mode: ``freq`` (Hz, ``0 < freq < fs/2``), exactly one of
        ``zeta`` (damping ratio, ``0 < zeta < 1``) and ``q`` (quality factor,
        ``q > 1/2``; ``zeta = 1/(2 q)``) and optionally ``gain`` (modal static
        gain ``g_k``, any finite real number, default 1; a negative value is a
        mode of opposite sign). Unknown keys raise an error.
    gain : float, default 1.0
        Overall linear gain of the path.

    Returns
    -------
    y : ndarray, same shape as ``x``
        Filtered signal in the unit of ``x``. The filter is linear: peak and
        RMS amplitudes of a stationary sinusoid are both multiplied by
        ``|H(f)|`` (the amplitude convention of the input is preserved).
    config : dict
        JSON-serialisable record: ``fs``, ``gain``, the realisation, and per
        mode the resolved ``freq``, ``zeta``, ``q``, ``gain``, the damped
        frequency ``damped_freq`` (Hz) and ``decay_time`` (s, ``1/(zeta w)``).

    Raises
    ------
    ValueError
        For an invalid ``fs``, ``gain``, mode list or non-finite input.

    References
    ----------
    Ewins, D. J. (2000). Modal Testing: Theory, Practice and Application,
    2nd ed., ch. 2. Research Studies Press.
    Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal
    Processing, 3rd ed., sec. 7.1.1 (impulse invariance). Pearson.
    """
    fs = _check_fs(fs)
    x2, was_1d = _signal_2d(x)
    ms = _parse_modes(modes, fs)
    gain = _finite(gain, "gain")
    y = np.zeros_like(x2)
    for m in ms:
        b1, a1, a2 = _mode_coefficients(m, fs)
        y += signal.lfilter([0.0, b1], [1.0, a1, a2], x2, axis=0)
    y *= gain
    resolved = [
        {
            **m,
            "damped_freq": m["freq"] * math.sqrt(1.0 - m["zeta"] ** 2),
            "decay_time": 1.0 / (m["zeta"] * _TWO_PI * m["freq"]),
        }
        for m in ms
    ]
    cfg = _config(
        "modal_path", None, fs=fs, gain=gain, realisation="impulse_invariant", modes=resolved,
        n=int(x2.shape[0]), n_channels=int(x2.shape[1]),
    )
    return restore(y, was_1d), cfg


def modal_response(f, modes, fs=None):
    """Complex frequency response of a modal transfer path.

    With ``fs`` the response is that of the digital filter realised by
    :func:`modal_path` with ``gain=1`` (so a sinusoid of frequency ``f`` and
    amplitude ``A`` comes out, in steady state, as ``A |H| cos(2 pi f t +
    arg H)``); without it, the analog response ``sum_k g_k w_k**2 /
    (w_k**2 - w**2 + 2j zeta_k w_k w)`` of the ideal modes. The overall
    ``gain`` of :func:`modal_path` is a plain multiplier and is not included.

    Parameters
    ----------
    f : array_like
        Frequencies in Hz (any shape; negative values give the conjugate).
    modes : sequence of dict
        As for :func:`modal_path`.
    fs : float, optional
        Sampling rate in Hz of the digital realisation; ``None`` for the
        analog response (then the modes are not required to lie below
        ``fs/2``).

    Returns
    -------
    ndarray of complex, same shape as ``f``
        Dimensionless transfer function: the ratio of the complex amplitude of
        the output to that of the input (the ratio is the same for peak and
        RMS amplitudes).

    References
    ----------
    Ewins, D. J. (2000). Modal Testing: Theory, Practice and Application,
    2nd ed., ch. 2. Research Studies Press.
    Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal
    Processing, 3rd ed., sec. 7.1.1. Pearson.
    """
    f = np.asarray(f, dtype=float)
    if not np.all(np.isfinite(f)):
        raise ValueError("f must be finite")
    if fs is not None:
        fs = _check_fs(fs)
    ms = _parse_modes(modes, fs)
    h = np.zeros(f.shape, dtype=complex)
    if fs is None:
        s = 1j * _TWO_PI * f
        for m in ms:
            w0 = _TWO_PI * m["freq"]
            h = h + m["gain"] * w0**2 / (s * s + 2.0 * m["zeta"] * w0 * s + w0**2)
    else:
        z1 = np.exp(-1j * _TWO_PI * f / fs)
        for m in ms:
            b1, a1, a2 = _mode_coefficients(m, fs)
            h = h + b1 * z1 / (1.0 + a1 * z1 + a2 * z1 * z1)
    return h


# ==========================================================================
# 2. Room impulse response
# ==========================================================================


def room_impulse_response(
    fs, rt60, *, length=None, n_channels=1, direct_delay=0.0, interchannel_delay=0.0, seed=None
):
    """Statistical impulse response of a reverberant space (decaying Gaussian noise).

    The late, diffuse part of a room response is modelled as stationary
    Gaussian noise under an exponential envelope (Schroeder 1965; Polack
    1993; Kuttruff 2009): ``h(t) = exp(-3 ln(10) (t - d) / rt60) w(t)`` for
    ``t >= d``, with ``w`` white Gaussian noise and ``d`` the delay of the
    channel. The *amplitude* envelope falls by 60 dB in ``rt60`` seconds, so
    the mean energy decay curve (the backward integral of ``h**2``; Schroeder
    1965) falls by 60 dB in ``rt60`` too, which is the definition of the
    reverberation time (Sabine 1922; ISO 3382-2). Only this diffuse tail is
    modelled: there is no separate direct-sound impulse, and no early
    reflections or frequency-dependent decay (``rt60`` is the same at all
    frequencies).

    Channels (microphones, ears) are *independent* realisations of the noise,
    so their expected coherence is zero; they differ in the onset delay
    ``d_c = direct_delay + interchannel_delay * c`` (or
    ``direct_delay + interchannel_delay[c]``). The delay is applied to the
    envelope origin and may be fractional (the first non-zero sample is the
    first one at or after ``d_c``, with the envelope evaluated at its exact
    distance from ``d_c``). A coherent delay of one signal to another, as for
    a plane wave arriving at two sensors, is :func:`delay_signal`.

    Normalisation. Every channel is scaled to *unit energy*,
    ``sum_n h[n]**2 = 1``, after generation and truncation. A stationary
    white input therefore keeps its power through :func:`apply_ir` in
    expectation (the discrete convolution ``y = h * x`` is dimensionless),
    and the reverberant level is set by scaling the signal, not the IR.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    rt60 : float
        Reverberation time in s (time for a 60 dB decay), positive.
    length : int, optional
        Length of the response in samples. Default:
        ``ceil(fs * (max_delay + 1.5 * rt60))``, at which the envelope has
        fallen by 90 dB (the truncation does not affect the decay measured
        between -5 and -35 dB). A shorter length truncates the tail before the
        unit-energy scaling.
    n_channels : int, default 1
        Number of channels.
    direct_delay : float, default 0
        Delay in s of the first arrival (source-to-receiver propagation
        time), non-negative.
    interchannel_delay : float or array_like, shape (n_channels,), default 0
        A scalar is the delay increment in s per channel (channel ``c`` is
        later by ``c * interchannel_delay``; negative values make later
        channels earlier). An array gives the extra delay of each channel.
        Every channel's total delay must be non-negative.
    seed : int, numpy.random.Generator or None
        Random source. Each channel uses its own stream derived from it, so
        channel ``c`` does not depend on ``n_channels``.

    Returns
    -------
    ir : ndarray, shape (m,) for ``n_channels == 1``, else (m, n_channels)
        Impulse response, dimensionless, normalised to unit energy per
        channel (an energy normalisation, so neither a peak nor an RMS
        convention applies).
    config : dict
        JSON-serialisable record, including the per-channel ``delays`` (s),
        the first non-zero sample ``onset_samples`` and the resolved ``seed``.

    Raises
    ------
    ValueError
        For invalid arguments, or a ``length`` that does not extend beyond the
        delay of every channel.

    References
    ----------
    Schroeder, M. R. (1965). New method of measuring reverberation time.
    Journal of the Acoustical Society of America, 37(3), 409-412.
    Sabine, W. C. (1922). Collected Papers on Acoustics. Harvard University
    Press.
    Polack, J.-D. (1993). Playing billiards in the concert hall: the
    mathematical foundations of geometrical room acoustics. Applied
    Acoustics, 38(2-4), 235-244.
    Kuttruff, H. (2009). Room Acoustics, 5th ed. Spon Press.
    ISO 3382-2:2008. Acoustics - Measurement of room acoustic parameters -
    Part 2: Reverberation time in ordinary rooms.
    """
    fs = _check_fs(fs)
    rt60 = _finite(rt60, "rt60", lo=0.0, lo_open=True)
    channels = _check_count(n_channels, "n_channels")
    direct = _finite(direct_delay, "direct_delay", lo=0.0)
    inter = np.asarray(interchannel_delay, dtype=float)
    if inter.ndim == 0:
        extra = float(inter) * np.arange(channels)
    elif inter.shape == (channels,):
        extra = inter
    else:
        raise ValueError("interchannel_delay must be a scalar or have one value per channel")
    if not np.all(np.isfinite(extra)):
        raise ValueError("interchannel_delay must be finite")
    delays = direct + extra
    if np.any(delays < 0.0):
        raise ValueError("every channel's total delay must be non-negative")
    if length is None:
        length = math.ceil(fs * (float(delays.max()) + 1.5 * rt60))
    length = _check_count(length, "length")
    onset = np.ceil(delays * fs - 1e-9).astype(np.int64)
    if np.any(onset >= length):
        raise ValueError("length must extend beyond the delay of every channel")

    rng, used = _generator(seed)
    streams = _child_generators(rng, channels)
    h = np.zeros((length, channels))
    decay = _LN_1000 / rt60  # amplitude decay rate, 1/s
    for c in range(channels):
        tau = np.maximum(np.arange(onset[c], length) / fs - delays[c], 0.0)
        h[onset[c]:, c] = np.exp(-decay * tau) * streams[c].standard_normal(length - onset[c])
    h /= np.sqrt(np.sum(h * h, axis=0))

    cfg = _config(
        "room_impulse_response", used, fs=fs, rt60=rt60, length=length, n_channels=channels,
        direct_delay=direct, delays=delays, onset_samples=onset,
        amplitude_decay_time=1.0 / decay, normalisation="unit energy per channel",
    )
    return (h[:, 0] if channels == 1 else h), cfg


def apply_ir(x, ir):
    """Convolve a signal with an impulse response (fast convolution, same length).

    The output is the first ``n`` samples of the full linear convolution
    ``y[k] = sum_m ir[m] x[k - m]`` (computed by overlap-add FFT convolution),
    i.e. causal and of the same length as ``x``: the reverberant tail that
    runs beyond the end of the record is dropped, and the response starts at
    sample 0 with a zero initial state.

    Channel handling: with equal channel counts each channel of ``x`` is
    convolved with the same channel of ``ir``; a single signal channel (1-D or
    one column) is rendered through every impulse-response channel (a mono
    source to several receivers); a single impulse-response channel is applied
    to every signal channel. Any other combination raises an error. The result
    is 1-D only if both inputs are 1-D.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal in any unit.
    ir : array_like, shape (m,) or (m, ch)
        Impulse response, e.g. from :func:`room_impulse_response`
        (dimensionless; unit energy keeps the power of a white input).

    Returns
    -------
    ndarray, shape (n,) or (n, ch_out)
        Convolved signal in the unit of ``x``. The operation is linear in both
        arguments; peak and RMS amplitudes are scaled alike.

    References
    ----------
    Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal
    Processing, 3rd ed., sec. 8.7 (block convolution; overlap-add). Pearson.
    """
    x2, x_1d = _signal_2d(x, "x")
    h2, h_1d = _signal_2d(ir, "ir")
    if h2.shape[0] < 1:
        raise ValueError("ir must have at least one sample")
    n, cx = x2.shape
    m, ci = h2.shape
    if cx == ci:
        pass
    elif cx == 1:
        x2 = np.broadcast_to(x2, (n, ci))
    elif ci == 1:
        h2 = np.broadcast_to(h2, (m, cx))
    else:
        raise ValueError(
            f"x has {cx} channels and ir has {ci}; they must be equal, or one of them must be 1"
        )
    if n == 0:
        y = np.zeros(x2.shape)
    else:
        y = signal.oaconvolve(x2, h2, axes=0)[:n]
    return y[:, 0] if (x_1d and h_1d) else y


# ==========================================================================
# 3. Speed-dependent coloured noise
# ==========================================================================


def _speed_rows(v, slope):
    """Control axis in km/h for the rows of the speed-dependent PSD.

    Returns a strictly increasing axis (at least two rows) that covers the
    speeds in ``v`` on a geometric grid whose neighbouring ratio is at most
    ``exp(_SPEED_GRID_STEP)``, so that the linear-in-power blend of
    :func:`synthesize_broadband` follows the power law ``v**k`` to a few
    hundredths of a dB. For a rising level law the grid starts at 0.1 % of the
    largest speed, and a row at 0 km/h (silence) is added in front of it
    wherever speeds fall below that.
    """
    vmax = float(v.max())
    positive = v[v > 0.0]
    vmin = float(v.min())
    if slope > 0.0:
        lo = max(float(positive.min()), vmax * 1e-3)
    else:
        lo = float(positive.min())
    hi = max(vmax, lo * math.exp(_SPEED_GRID_STEP))
    rows = math.ceil(math.log(hi / lo) / _SPEED_GRID_STEP) + 1
    axis = lo * (hi / lo) ** (np.arange(rows) / (rows - 1))
    if slope > 0.0 and vmin < lo:
        axis = np.concatenate(([0.0], axis))
    return axis


def speed_noise(
    fs,
    n,
    speed_kmh,
    *,
    level_db_at_ref,
    ref_speed_kmh,
    level_slope_db_per_doubling,
    tilt_db_per_octave,
    f_ref=1000.0,
    highpass=20.0,
    level_ref="rms",
    n_channels=1,
    grain=0.05,
    seed=None,
):
    """Coloured broadband noise whose spectrum tilts and whose level follows vehicle speed.

    Rolling (tyre/road) noise and aerodynamic noise are broadband processes
    with a smooth spectrum whose level grows with the vehicle speed ``v`` as a
    power law, a straight line in ``a log10(v / v_ref)`` for the level in dB
    (for tyre/road noise see Sandberg & Ejsmont 2002). ROTA models both with
    the same parametric power spectral density

        ``S(f, v) = S_0 (f / f_ref)**b  *  (v / v_ref)**k``  for ``f >= highpass``,

    ``b = tilt_db_per_octave / 3.0103`` and ``k = level_slope_db_per_doubling /
    3.0103``, and zero below ``highpass``. The two exponents are the user's
    data, not constants of the function: a level slope of ``s`` dB per
    doubling of speed is the power law ``v**(s / 3.0103)`` of the power. For
    orientation, the sound power of a dipole-type aerodynamic source grows as
    ``v**6`` (Curle 1955), which is 18.06 dB per doubling; the coefficient of
    tyre/road noise depends on tyre and surface (Sandberg & Ejsmont 2002). The
    noise is rendered by :func:`rota._broadband.synthesize_broadband` with
    ``v(t)`` as the control variable, so the spectrum follows the speed
    without frame-rate pumping (power-complementary grains).

    Conventions (all explicit; no hidden constants):

    * Tilt is that of the *power spectral density* (power per Hz), the narrow
      band spectrum an FFT or Welch estimate shows. ``-3.01`` dB/octave is
      pink noise (equal power per octave), ``0`` is white; the slope of
      octave-band levels is the tilt plus 3.01 dB/octave. Below the first
      retained grain bin the shape is held constant (a negative tilt diverges
      at 0 Hz).
    * Level, ``level_ref="rms"`` (default): ``level_db_at_ref`` is the RMS of
      the noise at ``ref_speed_kmh`` (its PSD lies between ``highpass`` and
      ``fs/2``, apart from the leakage described under ``highpass``), as
      ``20 log10(rms / 1 unit)`` in decibels re 1 *signal unit*: re 1 Pa for
      a pressure in Pa (an SPL re 20 uPa is 93.98 dB higher), re 1 m/s**2 for
      an acceleration in m/s**2. ``f_ref`` then only sets the pivot of the
      tilt and does not change the level.
    * Level, ``level_ref="psd"``: ``level_db_at_ref`` is the PSD at ``f_ref``
      and ``ref_speed_kmh``, ``10 log10(S / 1 unit**2/Hz)``.
    * At speed ``v`` the level changes by
      ``level_slope_db_per_doubling * log2(v / ref_speed_kmh)`` dB at every
      frequency (the spectral shape does not change with speed).
    * The expected power is computed on the grain's frequency bins, exactly
      as :func:`rota._broadband.synthesize_broadband` synthesises them, so the
      realised RMS equals the stated level in expectation, not only for a
      band wider than the bin spacing ``1 / grain``.

    A speed of 0 gives silence for a positive level slope; a level slope of 0
    makes the noise independent of speed (and ``speed_kmh`` may then contain
    zeros).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    n : int
        Number of samples.
    speed_kmh : float or array_like, shape (n,)
        Vehicle speed in km/h at every sample (a scalar is a constant speed);
        finite and non-negative. Speeds must be regularised to the sample grid
        first (:func:`rota.speed.regularize_control`).
    level_db_at_ref : float
        Level at ``ref_speed_kmh`` in dB re 1 signal unit (RMS, or PSD
        for ``level_ref="psd"``); see above.
    ref_speed_kmh : float
        Reference speed in km/h, positive.
    level_slope_db_per_doubling : float
        Level change in dB per doubling of speed (negative is allowed for
        positive speeds).
    tilt_db_per_octave : float
        Spectral tilt of the PSD in dB per octave (positive tilts upward).
    f_ref : float, default 1000.0
        Tilt pivot in Hz; the frequency of the level for ``level_ref="psd"``
        (then it must lie in ``[max(highpass, 1/grain), fs/2)``).
    highpass : float, default 20.0
        Cut-off in Hz, passed to :func:`rota._broadband.synthesize_broadband`:
        grain bins below it are zeroed, so the cut-off is only as sharp as the
        grain's resolution ``1 / grain`` (20 Hz for the default grain) and the
        leakage of the grain window. With the defaults it removes the DC bin
        and attenuates the range below 10 Hz by only about 5 dB; a sharp
        cut-off at a low frequency needs a longer ``grain``. A cut-off well
        above ``1 / grain`` is sharp: for ``highpass=200`` and a flat
        spectrum the PSD is 35 dB down at 100 to 150 Hz and 45 dB down below
        60 Hz.
    level_ref : {"rms", "psd"}, default "rms"
        What ``level_db_at_ref`` refers to.
    n_channels : int, default 1
        Number of independent noise channels (same spectrum and level law;
        zero expected coherence, as for a diffuse field).
    grain : float, default 0.05
        Grain length in s of the synthesis (the grain rate is ``2 / grain``).
    seed : int, numpy.random.Generator or None
        Random source; each channel has its own stream.

    Returns
    -------
    noise : ndarray, shape (n,) for ``n_channels == 1``, else (n, n_channels)
        Noise in the signal unit. Amplitude convention: RMS (a random signal,
        characterised by its PSD, not by peaks).
    config : dict
        JSON-serialisable record, including ``rms_at_ref`` (the expected RMS
        at ``ref_speed_kmh``), ``psd_at_f_ref`` (unit**2/Hz) and the resolved
        seed.

    Raises
    ------
    ValueError
        For invalid arguments, or zero speeds with a negative level slope.

    References
    ----------
    Sandberg, U., & Ejsmont, J. A. (2002). Tyre/Road Noise Reference Book.
    Informex, Kisa, Sweden.
    Curle, N. (1955). The influence of solid boundaries upon aerodynamic
    sound. Proceedings of the Royal Society A, 231, 505-514.
    Serra, X., & Smith, J. O. (1990). Spectral modeling synthesis. Computer
    Music Journal, 14(4), 12-24 (the stochastic component).
    Welch, P. D. (1967). The use of fast Fourier transform for the estimation
    of power spectra. IEEE Transactions on Audio and Electroacoustics, 15(2),
    70-73 (the PSD estimator the conventions refer to).
    """
    fs = _check_fs(fs)
    n = _check_count(n, "n")
    channels = _check_count(n_channels, "n_channels")
    v = np.asarray(speed_kmh, dtype=float)
    if v.ndim == 0:
        v = np.full(n, float(v))
    if v.shape != (n,):
        raise ValueError(f"speed_kmh must be a scalar or have shape ({n},)")
    if not np.all(np.isfinite(v)) or np.any(v < 0.0):
        raise ValueError("speed_kmh must be finite and non-negative")
    level = _finite(level_db_at_ref, "level_db_at_ref")
    v_ref = _finite(ref_speed_kmh, "ref_speed_kmh", lo=0.0, lo_open=True)
    slope = _finite(level_slope_db_per_doubling, "level_slope_db_per_doubling")
    tilt = _finite(tilt_db_per_octave, "tilt_db_per_octave")
    f_ref = _finite(f_ref, "f_ref", lo=0.0, lo_open=True)
    highpass = _finite(highpass, "highpass", lo=0.0, hi=0.5 * fs, hi_open=True)
    grain = _finite(grain, "grain", lo=0.0, lo_open=True)
    if level_ref not in _LEVEL_REFS:
        raise ValueError(f"level_ref must be one of {_LEVEL_REFS}, got {level_ref!r}")
    if slope < 0.0 and np.any(v <= 0.0):
        raise ValueError("a negative level slope diverges at zero speed; speeds must be positive")

    # Frequency bins of one grain, as synthesize_broadband lays them out.
    length = 2 * round(grain * fs / 2.0)
    if length < 8:
        raise ValueError("grain is shorter than 8 samples; increase grain or fs")
    hop = length // 2
    df = fs / length
    f_bins = np.arange(hop + 1) * df
    keep = f_bins >= highpass
    f_min = max(highpass, df)
    beta = tilt / _DB_PER_OCTAVE
    kappa = slope / _DB_PER_OCTAVE

    def shape(f):
        return (np.maximum(f, f_min) / f_ref) ** beta

    weights = shape(f_bins) * keep
    weights[0] *= 0.5  # DC and Nyquist bins cover half a bin
    weights[-1] *= 0.5
    shape_power = float(np.sum(weights) * df)  # expected mean square for S_0 = 1
    if level_ref == "rms":
        s0 = 10.0 ** (level / 10.0) / shape_power
    else:
        if not max(highpass, df) <= f_ref < 0.5 * fs:
            raise ValueError("level_ref='psd' needs f_ref in [max(highpass, 1/grain), fs/2)")
        s0 = 10.0 ** (level / 10.0)
    rms_ref = math.sqrt(s0 * shape_power)

    rng, used = _generator(seed)
    streams = _child_generators(rng, channels)
    out = np.zeros((n, channels))
    if slope == 0.0 or float(v.max()) > 0.0:
        f_grid = np.arange(4 * hop + 1) * (df / 4.0)
        base = s0 * shape(f_grid)
        if slope == 0.0:
            psd, control, axis = base, None, None
        else:
            axis = _speed_rows(v, slope)
            with np.errstate(divide="ignore"):
                power_gain = np.where(axis > 0.0, (axis / v_ref) ** kappa, 0.0)
            psd, control = np.outer(power_gain, base), v
        for c in range(channels):
            out[:, c] = synthesize_broadband(
                psd, f_grid, fs, n, control=control, control_axis=axis, grain=grain,
                highpass=highpass, seed=streams[c],
            )
    psd_at_f_ref = s0 * float(shape(f_ref)) if f_ref >= f_min else None
    cfg = _config(
        "speed_noise", used, fs=fs, n=n, n_channels=channels, level_db_at_ref=level,
        level_ref=level_ref, ref_speed_kmh=v_ref, level_slope_db_per_doubling=slope,
        tilt_db_per_octave=tilt, f_ref=f_ref, highpass=highpass, grain=grain,
        grain_rate=fs / hop, rms_at_ref=rms_ref, psd_at_f_ref=psd_at_f_ref,
        speed_min=float(v.min()), speed_max=float(v.max()),
        amplitude="rms", level_unit="dB re 1 signal unit",
    )
    return (out[:, 0] if channels == 1 else out), cfg


# ==========================================================================
# 4. Transient events
# ==========================================================================


def transient_events(fs, n, events, *, seed=None):
    """Stem of discrete transient events: thumps (bumps) and clicks (switches).

    Each event starts at its ``time`` and decays exponentially with the time
    constant ``decay``:

    * ``"thump"``: a low-frequency damped burst,
      ``a exp(-tau / decay) sin(2 pi freq tau)`` with ``tau = t - time``
      (sine start: no step at the onset). It is scaled so that the peak of the
      continuous-time waveform, ``max |.|`` at ``tan(w tau) = w decay``, is
      ``amplitude``; sampled at a rate well above ``freq`` the largest
      sample is within a fraction of a percent of it.
    * ``"click"``: a short broadband burst, Gaussian white noise times
      ``exp(-tau / decay)``, scaled so that the largest absolute *sample* is
      exactly ``amplitude``.

    Events are placed at exact (fractional) times: the waveform is evaluated
    analytically at the sample instants, and the first non-zero sample is the
    first one at or after ``time``. An event is rendered for ``20 decay`` (the
    envelope has fallen by 87 dB) and may be cut off by the end of the
    record; the amplitude scaling refers to the whole event. Overlapping
    events add. Each event draws its noise from its own stream, so adding an
    event does not change the others.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    n : int
        Number of samples of the stem.
    events : sequence of dict (or one dict)
        Required keys: ``time`` (s, ``0 <= time < n / fs``), ``kind``
        (``"thump"`` or ``"click"``), ``amplitude`` (peak value in the signal
        unit, a negative value inverts the event) and ``decay`` (s, the time
        constant of the 1/e amplitude decay, positive). A thump also needs
        ``freq`` (Hz, ``0 < freq < fs/2``); a click must not have one.
    seed : int, numpy.random.Generator or None
        Random source of the click noise.

    Returns
    -------
    stem : ndarray, shape (n,)
        Sum of the events in the signal unit. Amplitude convention: peak
        (the stated ``amplitude`` is the largest absolute value of an
        isolated event).
    config : dict
        JSON-serialisable record with the resolved events, each with its
        ``onset_sample`` (first sample at or after ``time``), ``peak_sample``
        and ``peak_time`` (s) of its largest absolute value inside the record,
        and ``n_samples`` (samples inside the record), plus the resolved seed.

    Raises
    ------
    ValueError
        For invalid arguments or unknown keys.

    References
    ----------
    Ewins, D. J. (2000). Modal Testing: Theory, Practice and Application,
    2nd ed., ch. 2 (free decay of a single-degree-of-freedom system). Research
    Studies Press.
    Brandt, A. (2011). Noise and Vibration Analysis: Signal Analysis and
    Experimental Procedures, ch. 5-6. Wiley.
    """
    fs = _check_fs(fs)
    n = _check_count(n, "n")
    if isinstance(events, Mapping):
        events = [events]
    try:
        events = list(events)
    except TypeError as err:
        raise TypeError("events must be a sequence of dicts") from err
    rng, used = _generator(seed)
    streams = _child_generators(rng, len(events))
    stem = np.zeros(n)
    resolved = []
    for i, ev in enumerate(events):
        what = f"events[{i}]"
        if not isinstance(ev, Mapping):
            raise TypeError(f"{what} must be a dict, got {type(ev).__name__}")
        extra = sorted(set(ev) - set(_EVENT_KEYS))
        if extra:
            raise ValueError(f"unknown key(s) {extra} in {what}; allowed: {list(_EVENT_KEYS)}")
        missing = [k for k in ("time", "kind", "amplitude", "decay") if ev.get(k) is None]
        if missing:
            raise ValueError(f"{what} is missing required key(s) {missing}")
        time = _finite(ev["time"], f"{what}: time", lo=0.0, hi=n / fs, hi_open=True)
        kind = ev["kind"]
        if kind not in _EVENT_KINDS:
            raise ValueError(f"{what}: kind must be one of {_EVENT_KINDS}, got {kind!r}")
        amp = _finite(ev["amplitude"], f"{what}: amplitude")
        decay = _finite(ev["decay"], f"{what}: decay", lo=0.0, lo_open=True)
        freq = ev.get("freq")
        if kind == "thump":
            if freq is None:
                raise ValueError(f"{what}: a thump needs 'freq'")
            freq = _finite(freq, f"{what}: freq", lo=0.0, lo_open=True, hi=0.5 * fs, hi_open=True)
        elif freq is not None:
            raise ValueError(f"{what}: 'freq' is only used by thumps")

        i0 = math.ceil(time * fs - 1e-9)
        span = math.ceil(20.0 * decay * fs) + 1
        tau = np.maximum((i0 + np.arange(span)) / fs - time, 0.0)
        env = np.exp(-tau / decay)
        if kind == "thump":
            w = _TWO_PI * freq
            tau_peak = math.atan(w * decay) / w
            peak = math.exp(-tau_peak / decay) * math.sin(w * tau_peak)
            wave = amp / peak * env * np.sin(w * tau)
        else:
            wave = env * streams[i].standard_normal(span)
            wave *= amp / np.max(np.abs(wave))
        inside = wave[: max(0, min(span, n - i0))]
        stem[i0 : i0 + inside.size] += inside
        k = int(np.argmax(np.abs(inside))) if inside.size else None
        resolved.append({
            "time": time, "kind": kind, "amplitude": amp, "decay": decay, "freq": freq,
            "onset_sample": i0, "n_samples": int(inside.size),
            "peak_sample": None if k is None else i0 + k,
            "peak_time": None if k is None else (i0 + k) / fs,
        })
    cfg = _config("transient_events", used, fs=fs, n=n, events=resolved, amplitude="peak")
    return stem, cfg


# ==========================================================================
# 5. Measurement chain
# ==========================================================================


def measurement_chain(
    x,
    fs,
    *,
    gain=1.0,
    noise_floor_rms=0.0,
    dc=0.0,
    clip=None,
    bits=None,
    full_scale=None,
    clock_drift_ppm=0.0,
    seed=None,
):
    """Impair a signal with a recorder: clock drift, gain, self-noise, DC, clipping, quantisation.

    The stages act in this order, as in an analogue-to-digital recorder:

    1. *Time base.* The recorder's clock runs at ``fs * (1 + eps)``, with
       ``eps = clock_drift_ppm * 1e-6``, while the user of the data assumes
       ``fs``. The input ``x`` is the signal at the true instants ``k / fs``;
       sample ``k`` of the output is the band-limited signal at the true time
       ``t_true[k] = k / (fs * (1 + eps))``. **Sign convention:** ``eps > 0``
       is a *fast* clock. It takes samples closer together in true time, so a
       tone of true frequency ``f`` appears at ``f / (1 + eps)`` (about
       ``f (1 - eps)``) when the data are analysed at the nominal ``fs``, and
       a true duration ``D`` is recorded as ``D fs (1 + eps)`` samples. The
       output holds all samples whose true time falls inside the input
       record, ``n_out = floor((n - 1)(1 + eps)) + 1``, so its length differs
       from ``n`` by about ``n eps`` samples; ``t_true`` relates it to the
       truth. With ``eps = 0`` the signal is passed unchanged. The
       interpolation is a Kaiser-windowed sinc (64 taps, ``beta = 10``,
       tabulated at 1/1024 sample and linearly interpolated): amplitude and
       phase error below 2e-5 for content up to 0.45 ``fs``, 5 % at 0.47
       ``fs`` and attenuation close to ``fs/2``. The signal is taken as zero
       outside the record, so the first and last 32 samples carry edge
       effects.
    2. *Gain*: ``y = gain * y``.
    3. *Self-noise*: add independent white Gaussian noise of RMS
       ``noise_floor_rms`` (in the unit of the output, i.e. after the gain).
    4. *DC offset*: add ``dc``.
    5. *Hard clipping* at ``+-clip`` (symmetric).
    6. *Quantisation* to ``bits`` bits over ``+-full_scale``: a uniform
       mid-tread quantiser with step ``2 full_scale / 2**bits``, rounding to
       the nearest code (ties to even), two's-complement code range
       ``-2**(bits-1) .. 2**(bits-1) - 1`` (so the positive end is
       ``full_scale`` minus one step), saturating beyond it. For a signal
       that spans many steps the error is uniform in ``+-step/2`` with RMS
       ``step / sqrt(12)`` (Widrow & Kollar 2008).

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Input signal in any unit (the output is in the same unit).
    fs : float
        Nominal sampling rate in Hz.
    gain, noise_floor_rms, dc, clip, full_scale : float or array_like, shape (ch,)
        Scalars or one value per channel. ``noise_floor_rms`` is the RMS of
        the added noise (>= 0), ``clip`` and ``full_scale`` are peak limits
        (> 0; ``clip=None`` disables clipping). ``gain`` and ``dc`` are any
        finite real number.
    bits : int, optional
        Number of bits of the quantiser, 2 to 32; ``None`` disables
        quantisation. Requires ``full_scale``, which is otherwise not allowed.
    clock_drift_ppm : float, default 0
        Relative clock error ``eps`` in parts per million (one value for all
        channels: one recorder, one clock); ``|eps| < 0.1``.
    seed : int, numpy.random.Generator or None
        Random source of the self-noise (independent streams per channel).

    Returns
    -------
    dict
        ``"x"``: ndarray, shape (n_out,) or (n_out, ch), the impaired signal
        in the unit of ``x`` (amplitude convention of the input is kept; the
        noise floor is RMS, the limits are peak). ``"t_true"``: ndarray, shape
        (n_out,), the true time in s of every output sample.
        ``"clipped"``: bool ndarray of the shape of ``"x"``, True where the
        hard clip or the quantiser saturation limited the sample.
        ``"config"``: JSON-serialisable record of the resolved settings
        (including ``n_out``, the quantiser ``step`` per channel and the
        resolved seed).

    Raises
    ------
    ValueError
        For invalid arguments (``bits`` without ``full_scale`` and vice versa,
        negative noise, clip or full scale, ``|clock_drift_ppm|`` too large).

    References
    ----------
    Widrow, B., & Kollar, I. (2008). Quantization Noise: Roundoff Error in
    Digital Computation, Signal Processing, Control, and Communications.
    Cambridge University Press.
    IEEE Std 1241-2010. Terminology and Test Methods for Analog-to-Digital
    Converters.
    Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal
    Processing, 3rd ed., sec. 4.8 (A/D conversion, quantisation error) and
    sec. 4.6 (changing the sampling rate). Pearson.
    Laakso, T. I., Valimaki, V., Karjalainen, M., & Laine, U. K. (1996).
    Splitting the unit delay: tools for fractional delay filter design. IEEE
    Signal Processing Magazine, 13(1), 30-60.
    """
    fs = _check_fs(fs)
    x2, was_1d = _signal_2d(x)
    n_in, ch = x2.shape
    if n_in < 1:
        raise ValueError("x must have at least one sample")
    gain_c = _per_channel(gain, ch, "gain")
    noise_c = _per_channel(noise_floor_rms, ch, "noise_floor_rms", lo=0.0)
    dc_c = _per_channel(dc, ch, "dc")
    clip_c = None if clip is None else _per_channel(clip, ch, "clip", lo=0.0, lo_open=True)
    ppm = _finite(clock_drift_ppm, "clock_drift_ppm", lo=-1e5, hi=1e5, lo_open=True, hi_open=True)
    if bits is None:
        if full_scale is not None:
            raise ValueError("full_scale is only used together with bits")
        fs_c = None
    else:
        bits = _check_count(bits, "bits", minimum=2)
        if bits > 32:
            raise ValueError("bits must be at most 32")
        if full_scale is None:
            raise ValueError("bits needs full_scale (the quantiser range is +-full_scale)")
        fs_c = _per_channel(full_scale, ch, "full_scale", lo=0.0, lo_open=True)

    eps = ppm * 1e-6
    if eps != 0.0:
        n_out = math.floor((n_in - 1) * (1.0 + eps)) + 1
        k = np.arange(n_out)
        y = _interpolate(x2, k / (1.0 + eps))
        t_true = k / (fs * (1.0 + eps))
    else:
        n_out = n_in
        y = x2.copy()
        t_true = np.arange(n_out) / fs

    rng, used = _generator(seed)
    y *= gain_c
    if np.any(noise_c > 0.0):
        for c, stream in enumerate(_child_generators(rng, ch)):
            y[:, c] += noise_c[c] * stream.standard_normal(n_out)
    y += dc_c
    clipped = np.zeros(y.shape, dtype=bool)
    if clip_c is not None:
        clipped |= np.abs(y) > clip_c
        y = np.clip(y, -clip_c, clip_c)
    step = None
    if bits is not None:
        step = 2.0 * fs_c / 2.0**bits
        codes = np.rint(y / step)
        lo, hi = -(2.0 ** (bits - 1)), 2.0 ** (bits - 1) - 1.0
        clipped |= (codes < lo) | (codes > hi)
        y = np.clip(codes, lo, hi) * step

    cfg = _config(
        "measurement_chain", used, fs=fs, n_in=n_in, n_out=n_out, n_channels=ch, gain=gain_c,
        noise_floor_rms=noise_c, dc=dc_c, clip=clip_c, bits=bits, full_scale=fs_c, step=step,
        clock_drift_ppm=ppm, clock_ratio=1.0 + eps,
        order=["time_base", "gain", "noise", "dc", "clip", "quantise"],
    )
    return {
        "x": restore(y, was_1d),
        "t_true": t_true,
        "clipped": restore(clipped, was_1d),
        "config": cfg,
    }


def delay_signal(x, fs, delay):
    """Delay a signal by a fractional number of samples (band-limited interpolation).

    Returns ``y(t) = x(t - delay)`` on the same sample grid, for the
    misalignment of two recordings (a tachometer against audio, one
    acquisition system against another). A positive ``delay`` makes the signal
    later, a negative one earlier. The signal is taken as zero outside the
    record, so the first ``delay * fs`` samples of a delayed signal are zero
    and the end of the record is cut off, with an edge effect over the 32
    samples around each end of the record.

    The value between samples is computed with a Kaiser-windowed sinc kernel
    (64 taps, ``beta = 10``; cut-off at ``fs/2``). For a band-limited signal
    the delay is accurate to better than 1e-4 samples (relative amplitude and
    phase error below 2e-5) for content up to 0.45 ``fs``; the error is 5 %
    at 0.47 ``fs`` and the gain falls off towards ``fs/2``, which is inherent
    to a real fractional delay. An integer number of samples is an exact shift. The
    ringing of the interpolation limits the use for signals with sharp edges
    (a rectangular pulse train shows a few per cent of overshoot).

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal in any unit.
    fs : float
        Sampling rate in Hz.
    delay : float or array_like, shape (ch,)
        Delay in s (one value, or one per channel).

    Returns
    -------
    ndarray, same shape as ``x``
        Delayed signal in the unit of ``x``; linear, so peak and RMS
        amplitudes of the interior are unchanged.

    Raises
    ------
    ValueError
        For an invalid ``fs``, non-finite input, or a ``delay`` that is not a
        finite scalar or a vector of one value per channel.

    References
    ----------
    Laakso, T. I., Valimaki, V., Karjalainen, M., & Laine, U. K. (1996).
    Splitting the unit delay: tools for fractional delay filter design. IEEE
    Signal Processing Magazine, 13(1), 30-60.
    Kaiser, J. F. (1974). Nonrecursive digital filter design using the
    I0-sinh window function. Proceedings of the IEEE International Symposium
    on Circuits and Systems, 20-23.
    Crochiere, R. E., & Rabiner, L. R. (1983). Multirate Digital Signal
    Processing, ch. 2. Prentice-Hall.
    """
    fs = _check_fs(fs)
    x2, was_1d = _signal_2d(x)
    n, ch = x2.shape
    d = _per_channel(delay, ch, "delay")
    k = np.arange(n, dtype=float)
    if np.all(d == d[0]):
        y = _interpolate(x2, k - d[0] * fs)
    else:
        y = np.empty_like(x2)
        for c in range(ch):
            y[:, c] = _interpolate(x2[:, c : c + 1], k - d[c] * fs)[:, 0]
    return restore(y, was_1d)
