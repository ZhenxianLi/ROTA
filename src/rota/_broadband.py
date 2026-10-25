"""Broadband noise resynthesis from a spectral envelope.

:func:`synthesize_broadband` renders stationary or operating-point dependent
noise (the stochastic part of a deterministic-plus-stochastic model, Serra &
Smith 1990) from a one-sided power spectral density. The noise is built from
overlapping *grains*: each grain is the inverse FFT of a spectrum with the
magnitude of the envelope and random phases, multiplied by a window and
overlap-added (Allen & Rabiner 1977).

The window decides whether the result is free of level pumping at the grain
rate. Grains with independent random phases are uncorrelated, so they add in
*power*, not in amplitude. The window must therefore be *power
complementary*, ``sum_k w(n - k R)**2 = 1``, which the square root of the
periodic Hann window satisfies at 50 % overlap. The amplitude-COLA window
that is the right choice for a deterministic signal (``sum_k w = 1``, plain
Hann) makes the power of independent grains dip by 3 dB at every grain
boundary (ROADMAP section 6, item 11; Griffin & Lim 1984 discuss the two
conditions for signal estimation from modified short-time spectra).

Units: the output has the unit of ``sqrt(psd)`` (e.g. Pa for a PSD in
Pa^2/Hz). Amplitude convention: the output is a random signal characterised by
its PSD and RMS (``mean(y**2) = integral of psd``), not by peak amplitudes.
"""

import numpy as np

__all__ = ["synthesize_broadband"]

_WINDOWS = ("sqrt_hann", "hann")
_CHUNK_ELEMENTS = 1 << 20  # complex values per block of grains (bounds memory)


def _grain_window(length, kind):
    """Periodic grain window of ``length`` samples.

    ``"sqrt_hann"`` is ``sin(pi m / L)``, the square root of the periodic Hann
    window (power complementary at 50 % overlap); ``"hann"`` is the periodic
    Hann window ``sin(pi m / L)**2`` (amplitude COLA at 50 % overlap).
    """
    s = np.sin(np.pi * np.arange(length) / length)
    return s if kind == "sqrt_hann" else s * s


def _check_psd_inputs(psd, freqs, fs, n, control, control_axis, grain, highpass):
    """Validate and normalise the arguments; return arrays and the grain length."""
    fs = float(fs)
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be positive and finite")
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer number of samples")
    n = int(n)

    freqs = np.asarray(freqs, dtype=float)
    if freqs.ndim != 1 or freqs.size < 2:
        raise ValueError("freqs must be a 1-D array of at least two frequencies")
    if not np.all(np.isfinite(freqs)) or freqs[0] < 0 or np.any(np.diff(freqs) <= 0):
        raise ValueError("freqs must be finite, non-negative and strictly increasing")

    psd = np.asarray(psd, dtype=float)
    if psd.ndim not in (1, 2):
        raise ValueError("psd must have shape (n_freq,) or (n_control, n_freq)")
    if psd.shape[-1] != freqs.size:
        raise ValueError("psd must have one column per entry of freqs")
    if not np.all(np.isfinite(psd)) or np.any(psd < 0):
        raise ValueError("psd must be finite and non-negative")

    if psd.ndim == 1:
        if control is not None or control_axis is not None:
            raise ValueError("control and control_axis apply only to a 2-D psd")
        control_axis = None
        control = None
    else:
        if control is None or control_axis is None:
            raise ValueError("a 2-D psd needs control_axis and control")
        control_axis = np.asarray(control_axis, dtype=float)
        if control_axis.shape != (psd.shape[0],) or control_axis.size < 2:
            raise ValueError("control_axis must have one entry per row of psd (at least two rows)")
        if not np.all(np.isfinite(control_axis)) or np.any(np.diff(control_axis) <= 0):
            raise ValueError("control_axis must be finite and strictly increasing")
        control = np.asarray(control, dtype=float)
        if control.shape != (n,):
            raise ValueError(f"control must have shape ({n},), one value per output sample")
        if not np.all(np.isfinite(control)):
            raise ValueError("control contains non-finite values; regularise it first")

    grain = float(grain)
    if not (np.isfinite(grain) and grain > 0):
        raise ValueError("grain must be positive and finite")
    length = 2 * round(grain * fs / 2.0)
    if length < 8:
        raise ValueError("grain is shorter than 8 samples; increase grain or fs")

    highpass = float(highpass)
    if not (np.isfinite(highpass) and 0.0 <= highpass < fs / 2.0):
        raise ValueError("highpass must lie in [0, fs/2)")
    return psd, freqs, fs, n, control, control_axis, length, highpass


def _blend_rows(table, axis, c):
    """Power-linear blend of the rows of ``table`` at control values ``c``.

    ``table`` has shape ``(n_control, n_bins)`` on the increasing ``axis``;
    values of ``c`` outside the axis are held at the end rows.
    """
    i = np.clip(np.searchsorted(axis, c, side="right") - 1, 0, axis.size - 2)
    a = np.clip((c - axis[i]) / (axis[i + 1] - axis[i]), 0.0, 1.0)[:, None]
    return (1.0 - a) * table[i] + a * table[i + 1]


def _synthesize(psd, freqs, fs, n, *, control, control_axis, grain, highpass, seed, window):
    """Grain synthesis with a selectable window (``"sqrt_hann"`` or ``"hann"``).

    ``"hann"`` exists only to demonstrate the pumping of an amplitude-COLA
    window with independent random-phase grains (ROADMAP section 6, item 11):
    the scaling is that of ``"sqrt_hann"`` and is not renormalised, so the mean
    power is 1.25 dB low.
    """
    if window not in _WINDOWS:
        raise ValueError(f"window must be one of {_WINDOWS}, got {window!r}")
    psd, freqs, fs, n, control, control_axis, length, highpass = _check_psd_inputs(
        psd, freqs, fs, n, control, control_axis, grain, highpass
    )
    rng = np.random.default_rng(seed)

    hop = length // 2
    n_bins = hop + 1
    f_bins = np.arange(n_bins) * (fs / length)
    keep = f_bins >= highpass

    # Envelope on the grain's frequency bins (linear in frequency and power).
    rows = np.atleast_2d(psd)
    table = np.stack([np.interp(f_bins, freqs, row) for row in rows])

    # Grain k starts at sample (k - 1) * hop and is centred on sample k * hop,
    # so that every output sample lies in the fully overlapped region.
    n_grains = -(-n // hop) + 1
    if control is not None:
        centres = np.minimum(np.arange(n_grains) * hop, n - 1)
        grain_control = control[centres]

    win = _grain_window(length, window)
    # |G_k| = sqrt(psd_k * fs * L / 2) for every bin k (see the Notes of
    # synthesize_broadband).
    scale = np.sqrt(0.5 * fs * length)
    buf = np.zeros((n_grains + 1, hop))
    step = max(1, _CHUNK_ELEMENTS // n_bins)
    for k0 in range(0, n_grains, step):
        k1 = min(n_grains, k0 + step)
        if control is None:
            power = np.broadcast_to(table[0], (k1 - k0, n_bins))
        else:
            power = _blend_rows(table, control_axis, grain_control[k0:k1])
        mag = scale * np.sqrt(power) * keep
        u = rng.random((k1 - k0, n_bins))
        spec = mag * np.exp(2j * np.pi * u)
        # DC and Nyquist bins are real: a random sign instead of a phase.
        spec[:, 0] = np.where(u[:, 0] < 0.5, mag[:, 0], -mag[:, 0])
        spec[:, -1] = np.where(u[:, -1] < 0.5, mag[:, -1], -mag[:, -1])
        frames = np.fft.irfft(spec, n=length, axis=1) * win
        buf[k0:k1] += frames[:, :hop]
        buf[k0 + 1 : k1 + 1] += frames[:, hop:]
    return buf.reshape(-1)[hop : hop + n].copy()


def synthesize_broadband(
    psd, freqs, fs, n, *, control=None, control_axis=None, grain=0.05, highpass=20.0, seed=None
):
    """Broadband noise with a given power spectral density, optionally operating-point dependent.

    The noise is made of grains of ``grain`` seconds (rounded to an even
    number of samples ``L``) at a hop of ``L/2``. Each grain is the inverse
    real FFT of a spectrum whose magnitude follows ``psd`` at the grain's
    control value and whose phases are independent and uniformly random;
    bins below ``highpass`` are zeroed. The grains are multiplied by the
    square root of the periodic Hann window and overlap-added. Because
    ``sin**2 + cos**2 = 1`` this window is *power complementary* at 50 %
    overlap, so the power of the independent grains adds up to a constant and
    the output has no level pumping at the grain rate (ROADMAP section 6,
    item 11).

    With a 2-D ``psd`` the envelope depends on a control variable (engine
    speed in rpm, vehicle speed in km/h, load, ...). Each grain uses the
    control value at its centre sample, located between two neighbouring rows
    of ``psd``; the *power* spectrum is interpolated linearly between those
    rows, ``(1 - a) psd_i + a psd_{i+1}``. This is a crossfade of the two
    operating points with weights ``sqrt(1 - a)`` and ``sqrt(a)`` on the
    amplitude: the power stays between the two operating points (a linear
    amplitude crossfade of two independent noises dips by 3 dB at the middle).
    Between grains the window crossfades the power in the same
    power-complementary way, so the level follows the control smoothly.

    Parameters
    ----------
    psd : array_like, shape (n_freq,) or (n_control, n_freq)
        One-sided power spectral density in unit**2/Hz, non-negative, on
        ``freqs``. A 1-D array is a stationary envelope; a 2-D array holds one
        envelope per operating point (row) for ``control_axis``. The envelope
        is interpolated linearly in frequency and in power onto the grain's
        bins and held constant beyond the ends of ``freqs`` (add zero-valued
        end points to band-limit; give a dense grid for a steep envelope).
    freqs : array_like, shape (n_freq,)
        Strictly increasing frequencies in Hz, at least two.
    fs : float
        Sampling rate in Hz.
    n : int
        Number of output samples.
    control : array_like, shape (n,), optional
        Control value for every output sample, in the unit of
        ``control_axis`` (rpm, km/h, ...). Required for a 2-D ``psd``. Values
        outside ``control_axis`` use the first or last row (no
        extrapolation).
    control_axis : array_like, shape (n_control,), optional
        Strictly increasing control values of the rows of ``psd`` (at least
        two). Required for a 2-D ``psd``.
    grain : float, default 0.05
        Grain length in s. The hop is half of it, so the grain rate (the
        frequency at which pumping would appear) is ``fs / hop = 2 / grain``
        (40 Hz by default). The grain is rounded to an even number of samples
        and must have at least 8. The frequency resolution of the envelope is
        about ``1 / grain``; a shorter grain follows a changing control more
        closely, a longer one resolves narrower spectral features.
    highpass : float, default 20.0
        Cut-off in Hz: grain bins with a centre frequency below it are set to
        zero, which removes the sub-audio content whose offsets at the window
        switches cause clicks (ROADMAP section 6, item 11). The cut-off
        is a bin-wise zeroing at the grain's resolution ``1 / grain``; energy
        within about that distance below the cut-off leaks in through the
        window. 0 keeps the DC bin.
    seed : int, numpy.random.Generator or None
        Seed or generator for the random phases. The same integer seed and
        arguments give bit-identical output; None draws fresh entropy.

    Returns
    -------
    ndarray, shape (n,)
        Noise in the unit of ``sqrt(psd)`` (e.g. Pa). It is a random signal
        characterised by its PSD, not by peak amplitudes: the amplitude
        convention is RMS, with ``mean(y**2)`` equal to the integral of the
        (high-passed) ``psd`` over 0 to ``fs/2`` at every instant, in the
        sense of the expectation over the random phases.

    Raises
    ------
    ValueError
        For inconsistent shapes, non-finite or negative ``psd``, non-increasing
        ``freqs`` or ``control_axis``, a ``control`` of the wrong length or
        with non-finite values, a ``grain`` shorter than 8 samples, or a
        ``highpass`` outside ``[0, fs/2)``.

    Notes
    -----
    Scaling. Let ``L`` be the grain length, ``R = L/2`` the hop, ``df = fs/L``
    the bin spacing, ``f_k = k df`` and ``G_k`` the grain's spectrum
    (``k = 0 .. L/2``). With ``g = irfft(G)``, Parseval gives the mean square
    of a grain ``(|G_0|**2 + |G_{L/2}|**2 + 2 sum_{0<k<L/2} |G_k|**2) / L**2``.
    The target is ``integral psd df ~ sum_{0<k<L/2} psd_k df + (psd_0 +
    psd_{L/2}) df / 2`` (trapezoid rule; DC and Nyquist cover half a bin).
    Equating the two term by term gives, for every bin including DC and
    Nyquist, ``|G_k| = sqrt(psd_k fs L / 2)``. With independent uniform phases
    (a random sign for the real DC and Nyquist bins) ``E[g[m]**2]`` equals that
    mean square for every sample ``m`` of the grain, not only on average over
    ``m``. Independent zero-mean grains add in power, so
    ``E[y[n]**2] = sum_k w(n - kR)**2 E[g**2] = integral psd df`` for every
    ``n``, since ``sum_k w**2 = 1``. The long-term PSD of ``y`` is the
    envelope smoothed by the normalised power transfer function of the window
    (width about ``1 / grain``), which leaves a locally smooth envelope
    unchanged. Within a grain the magnitudes are deterministic, so the sample
    values approach a Gaussian distribution only by the central limit theorem
    over the bins; an envelope only a few bins wide gives a sinusoid-like
    rather than a Gaussian narrow-band noise.

    References
    ----------
    Serra, X., & Smith, J. O. (1990). Spectral modeling synthesis: A sound
    analysis/synthesis system based on a deterministic plus stochastic
    decomposition. Computer Music Journal, 14(4), 12-24.
    Allen, J. B., & Rabiner, L. R. (1977). A unified approach to short-time
    Fourier analysis and synthesis. Proceedings of the IEEE, 65(11), 1558-1564.
    Griffin, D. W., & Lim, J. S. (1984). Signal estimation from modified
    short-time Fourier transform. IEEE Transactions on Acoustics, Speech, and
    Signal Processing, 32(2), 236-243.
    Welch, P. D. (1967). The use of fast Fourier transform for the estimation
    of power spectra. IEEE Transactions on Audio and Electroacoustics, 15(2),
    70-73.
    """
    return _synthesize(
        psd, freqs, fs, n, control=control, control_axis=control_axis,
        grain=grain, highpass=highpass, seed=seed, window="sqrt_hann",
    )
