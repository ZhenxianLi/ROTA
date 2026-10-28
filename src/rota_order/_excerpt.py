"""Stationary listening excerpt from a run-up or steady recording.

:func:`stationary_excerpt` turns the part of a recording where the engine runs
at a given speed into an excerpt of any length with *constant* order
frequencies and *steady* order amplitudes. The signal is split into orders
(deterministic part) and residual (stochastic part) as in spectral modelling
synthesis (Serra & Smith 1990); each order is rendered at the amplitude it has
at the reference speed, and the residual is rendered as noise with the power
spectral density it has there.

Units: time in s, frequency in Hz, speed in rpm. Order amplitudes are *peak*
amplitudes in the unit of the recording (for example Pa); the residual PSD is
one-sided, in unit**2/Hz. No gain is applied anywhere: the excerpt has the
level of the recording at the reference speed.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from ._broadband import synthesize_broadband
from ._provenance import jsonable
from ._tracked import tracked_harmonics
from ._utils import runs
from ._vk import order_waveform
from .angle import stationarize
from .metrics import envelope_stats

__all__ = ["stationary_excerpt", "ExcerptResult"]

_TWO_PI = 2.0 * np.pi
_MODELS = ("sinusoid", "cluster")
_EXTRACTORS = ("tracked-harmonics", "vold-kalman")
_PSD_RESOLUTION_HZ = 5.0  # target bin spacing of the Welch estimate of the residual
_STATIONARIZE_ABOVE = 1e-3  # relative speed range in the window above which the residual is warped
_MIN_WINDOW_SAMPLES = 16
_TRIM_REVS = 2.0  # revolutions dropped at each end of the warped residual (filter edge effects)


@dataclass(eq=False)
class ExcerptResult:
    """Result of :func:`stationary_excerpt`.

    Attributes
    ----------
    x : ndarray, shape (n,)
        The excerpt, ``orders + residual``, in the unit of the recording.
    orders : ndarray, shape (n,)
        Deterministic part: the sum of the steady order sinusoids (or
        clusters), with the fades applied.
    residual : ndarray, shape (n,)
        Stochastic part: broadband noise with the residual PSD, with the fades
        applied.
    fs : float
        Sampling rate in Hz.
    rpm_ref : float
        Reference speed in rpm at which the orders sit.
    amplitudes : dict
        ``{order: peak amplitude}`` in the unit of the recording, the
        energy-mean ``sqrt(mean(|envelope|**2))`` of the extracted envelope in
        the operating window.
    psd : tuple of ndarray
        ``(freqs, values)``: one-sided power spectral density of the residual in
        the operating window, in unit**2/Hz (Welch, Hann window, 50 % overlap),
        at frequencies in Hz.
    info : dict
        JSON-serialisable record: the parameters, the operating window
        (``window``), the random phases and seed, how the PSD was obtained
        (``psd``) and the envelope statistics (``order_stem``, ``mix``,
        ``per_order_swing_db``; see :func:`stationary_excerpt`).
    """

    x: np.ndarray
    orders: np.ndarray
    residual: np.ndarray
    fs: float
    rpm_ref: float
    amplitudes: dict
    psd: tuple
    info: dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"ExcerptResult(n={self.x.size}, fs={self.fs:g}, rpm_ref={self.rpm_ref:g}, "
            f"orders={list(self.amplitudes)}, model={self.info.get('model')!r})"
        )


def _check_scalar(value, name, *, positive=True, finite=True):
    v = float(value)
    if not np.isfinite(v) or (positive and v <= 0.0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite, got {value!r}")
    return v


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


def _residual_psd(res_w, rpm_w, fs, rpm_ref):
    """One-sided Welch PSD of the residual in the operating window.

    If the speed varies in the window by more than ``_STATIONARIZE_ABOVE``
    (relative to ``rpm_ref``), the residual is first warped with
    :func:`rota_order.angle.stationarize` at ``rpm_ref`` so that any shaft-locked
    line left in it becomes stationary, and the revolutions at both ends that
    carry filter edge effects are dropped.
    """
    spread = float(rpm_w.max() - rpm_w.min()) / rpm_ref
    stationarized = spread > _STATIONARIZE_ABOVE
    y = res_w
    if stationarized:
        spr = fs * 60.0 / rpm_ref
        st = stationarize(res_w, fs, rpm_w, rpm_ref=rpm_ref, max_order=0.45 * spr)
        y = st.x
        trim = min(int(round(_TRIM_REVS * spr)), y.size // 10)
        if trim:
            y = y[trim:-trim]
    if y.size < _MIN_WINDOW_SAMPLES:
        raise ValueError("the operating window is too short to estimate the residual PSD")
    nperseg = int(min(y.size, round(fs / _PSD_RESOLUTION_HZ)))
    f, p = signal.welch(
        y, fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant",
        scaling="density",
    )
    return f, p, {
        "stationarized": bool(stationarized),
        "speed_spread": spread,
        "n_samples": int(y.size),
        "nperseg": nperseg,
        "resolution_hz": float(fs / nperseg),
    }


def _fade_window(n, n_fade):
    """Unit window with raised-cosine ramps of ``n_fade`` samples at both ends (0 at the ends)."""
    w = np.ones(n)
    if n_fade:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_fade) / n_fade))
        w[:n_fade] = ramp
        w[n - n_fade :] = ramp[::-1]
    return w


def _cluster_stem(freq, amp, width, n, fs, rng):
    """One order as a random-phase line cluster of total width ``width`` Hz.

    Lines on the bins of an ``n``-point FFT within ``freq +- width / 2`` (at
    least the nearest bin), equal amplitudes, independent uniform phases;
    ``amp / sqrt(lines)`` each so that the power equals that of a sinusoid of
    peak amplitude ``amp``. Periodic over the excerpt.
    """
    df = fs / n
    b_max = n // 2 - 1
    lo = max(1, int(np.ceil((freq - 0.5 * width) / df)))
    hi = min(b_max, int(np.floor((freq + 0.5 * width) / df)))
    if hi < lo:
        lo = hi = int(np.clip(round(freq / df), 1, b_max))
    bins = np.arange(lo, hi + 1)
    phi = rng.uniform(0.0, _TWO_PI, size=bins.size)
    spec = np.zeros(n // 2 + 1, dtype=complex)
    spec[bins] = (amp / np.sqrt(bins.size)) * (0.5 * n) * np.exp(1j * phi)
    return np.fft.irfft(spec, n=n)


def stationary_excerpt(
    x,
    fs,
    rpm,
    rpm_ref,
    duration,
    *,
    orders,
    model="sinusoid",
    tolerance=0.03,
    extractor="tracked-harmonics",
    window_revs=8.0,
    cluster_width_hz=6.25,
    fade=0.02,
    seed=None,
    **kw,
):
    """Steady listening excerpt at a reference speed from a run-up or steady recording.

    Procedure (ROADMAP section 2.6; section 6, items 9 and 13):

    1. *Operating window.* The samples whose speed is within ``+-tolerance``
       (relative) of ``rpm_ref``; the longest contiguous run is used.
    2. *Orders.* ``orders`` are extracted from the whole signal with
       :func:`rota_order.extract.tracked_harmonics` or
       :func:`rota_order.extract.order_waveform`. The peak amplitude of each order at
       ``rpm_ref`` is the energy mean of its envelope in the window,
       ``A_k = sqrt(mean(|c_k|**2))``. The residual is the signal minus the
       extracted orders.
    3. *Residual PSD.* Welch estimate (Hann window, 50 % overlap, about 5 Hz
       bins) of the residual in the window. If the speed varies in the window by
       more than 0.1 % of ``rpm_ref`` the window is first made stationary with
       :func:`rota_order.angle.stationarize` at ``rpm_ref`` (anti-aliasing cut-off
       ``0.45`` of the angular rate, edge revolutions dropped), so that
       shaft-locked lines left in the residual are not smeared; the PSD level
       is then scaled by the local speed ratio (at most ``tolerance``) and the
       top ~10 % of the band is attenuated where the speed is below
       ``rpm_ref``. Otherwise the window is used as it is.
    4. *Synthesis* over ``duration`` s. ``model="sinusoid"``: the orders are
       true sinusoids ``A_k cos(2 pi order_k rpm_ref / 60 t + phi_k)`` with
       seeded random phases, so there are no beats and the envelope of each
       order is constant. ``model="cluster"``: each order is a random-phase
       cluster of equal lines of total width ``cluster_width_hz`` and the same
       power, the legacy way of making a tone out of a spectral peak; it is
       narrow-band noise rather than a tone and its envelope swings by
       ~18 dB (5th to 95th percentile), which is why it exists here only for
       comparison. The residual is broadband noise with the measured PSD from
       :func:`rota_order.resynth.synthesize_broadband`. Orders at or above
       ``fs / 2`` are muted. Raised-cosine fades of ``fade`` s are applied to
       both parts, so that ``x = orders + residual`` exactly. No other gain.

    The excerpt is mono. The extractor is run on the whole signal, so its edge
    regions (``window_revs`` revolutions at either end for the tracked
    harmonics) should not contain the operating window.

    Parameters
    ----------
    x : array_like, shape (n,)
        Recording in any unit (for example Pa); mono.
    fs : float
        Sampling rate in Hz; the excerpt is rendered at the same rate.
    rpm : array_like, shape (n,)
        Shaft speed in rpm at the samples of ``x``.
    rpm_ref : float
        Reference speed in rpm at which the excerpt is made.
    duration : float
        Length of the excerpt in s.
    orders : sequence of float
        Orders (cycles per revolution, positive, distinct) to model as
        steady tones. Everything else becomes part of the residual.
    model : {"sinusoid", "cluster"}, default "sinusoid"
        Rendering of the orders (see above).
    tolerance : float, default 0.03
        Relative half-width of the operating window: ``|rpm / rpm_ref - 1| <=
        tolerance``.
    extractor : {"tracked-harmonics", "vold-kalman"}
        Order extractor.
    window_revs : float, default 8
        Window length in revolutions of the tracked harmonics (ignored by the
        Vold-Kalman filter).
    cluster_width_hz : float, default 6.25
        Total width in Hz of the clusters of ``model="cluster"``.
    fade : float, default 0.02
        Length in s of the fade-in and fade-out (0 for none); must satisfy
        ``2 * fade < duration``.
    seed : int, numpy.random.Generator or None
        Source of the random phases and of the noise; the same integer seed
        gives bit-identical output. ``None`` draws a seed, recorded in
        ``info["seed"]``.
    **kw
        Passed to the extractor (for example ``bandwidth`` for Vold-Kalman or
        ``overlap`` for the tracked harmonics).

    Returns
    -------
    ExcerptResult
        ``x``, the stems ``orders`` and ``residual``, ``amplitudes`` (peak, unit
        of ``x``), ``psd`` and ``info``. ``info`` holds the parameters and the
        envelope statistics of :func:`rota_order.metrics.envelope_stats`, computed on
        the steady part between the fades: ``order_stem`` (the whole order stem,
        which includes the beating between different orders), ``mix``, and
        ``per_order_swing_db`` (the swing of each order on its own: ~0 dB for
        the sinusoid model). Each has ``swing_db`` and ``crest_factor_db``.

    Raises
    ------
    ValueError
        If the speed never comes within ``tolerance`` of ``rpm_ref``, if the
        longest run is shorter than one revolution (or 16 samples), if the
        fades do not fit in ``duration``, or for inconsistent shapes and invalid
        parameters.

    Notes
    -----
    Amplitude convention: the orders are peak amplitudes, the residual is
    specified by its PSD (RMS, unit**2/Hz). With stationary orders at the
    reference speed, the level of the excerpt is that of the recording in the
    operating window; Welch and the grain synthesis share the same convention
    (``mean(y**2)`` equals the integral of the PSD).

    References
    ----------
    Serra, X., & Smith, J. O. (1990). Spectral modeling synthesis: a sound
    analysis/synthesis system based on a deterministic plus stochastic
    decomposition. Computer Music Journal, 14(4), 12-24.
    Welch, P. D. (1967). The use of fast Fourier transform for the estimation of
    power spectra: a method based on time averaging over short, modified
    periodograms. IEEE Transactions on Audio and Electroacoustics, 15(2), 70-73.
    Antoni, J. (2009). Cyclostationarity by examples. Mechanical Systems and
    Signal Processing, 23(4), 987-1036.
    Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order
    tracking. Mechanical Systems and Signal Processing, 11(2), 187-205.
    Boashash, B. (1992). Estimating and interpreting the instantaneous frequency
    of a signal. I. Fundamentals. Proceedings of the IEEE, 80(4), 520-538.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be mono with shape (n,); process channels one at a time")
    n = x.size
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")
    fs = _check_scalar(fs, "fs")
    rpm = np.asarray(rpm, dtype=float)
    if rpm.shape != (n,):
        raise ValueError(f"rpm must have shape ({n},) like x, got {rpm.shape}")
    if not np.all(np.isfinite(rpm)):
        raise ValueError("rpm contains non-finite values; regularise it first")
    rpm_ref = _check_scalar(rpm_ref, "rpm_ref")
    duration = _check_scalar(duration, "duration")
    orders = np.atleast_1d(np.asarray(orders, dtype=float))
    if orders.ndim != 1 or orders.size == 0 or not np.all(np.isfinite(orders)) or np.any(orders <= 0):
        raise ValueError("orders must be a non-empty 1-D sequence of positive orders")
    if np.unique(orders).size != orders.size:
        raise ValueError("orders must be distinct")
    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}, got {model!r}")
    if extractor not in _EXTRACTORS:
        raise ValueError(f"extractor must be one of {_EXTRACTORS}, got {extractor!r}")
    tolerance = _check_scalar(tolerance, "tolerance")
    if tolerance >= 1.0:
        raise ValueError("tolerance is relative and must be below 1")
    window_revs = _check_scalar(window_revs, "window_revs")
    cluster_width_hz = _check_scalar(cluster_width_hz, "cluster_width_hz")
    fade = _check_scalar(fade, "fade", positive=False)
    if fade < 0.0:
        raise ValueError("fade must be non-negative (s)")
    n_out = int(round(duration * fs))
    n_fade = int(round(fade * fs))
    if n_out - 2 * n_fade < 16:
        raise ValueError(
            "duration is too short for the fades (need 2 * fade < duration, leaving at least "
            "16 samples)"
        )

    # 1. operating window: the longest run within the tolerance of rpm_ref
    starts, stops = runs(np.abs(rpm / rpm_ref - 1.0) <= tolerance)
    if starts.size == 0:
        raise ValueError(
            f"the speed never comes within +-{tolerance:.3g} (relative) of rpm_ref = "
            f"{rpm_ref:g} rpm (it ranges from {rpm.min():g} to {rpm.max():g} rpm)"
        )
    i = int(np.argmax(stops - starts))
    a, b = int(starts[i]), int(stops[i])
    revs = float(np.mean(rpm[a:b])) / 60.0 * (b - a) / fs
    if b - a < _MIN_WINDOW_SAMPLES or revs < 1.0:
        raise ValueError(
            f"the longest run within +-{tolerance:.3g} of rpm_ref lasts {(b - a) / fs:.3g} s "
            f"({revs:.2g} revolutions); increase tolerance"
        )

    # 2. orders over the whole signal, amplitude at rpm_ref in the window
    if extractor == "tracked-harmonics":
        comp = tracked_harmonics(x, fs, rpm, orders, window_revs=window_revs, **kw)
    else:
        comp = order_waveform(x, fs, rpm, orders, **kw)
    amp = np.sqrt(np.mean(np.abs(comp.envelopes[a:b]) ** 2, axis=0))
    resid = x - comp.total

    # 3. residual PSD in the window
    freqs, psd, psd_info = _residual_psd(resid[a:b], rpm[a:b], fs, rpm_ref)

    # 4. synthesis
    rng, used_seed = _generator(seed)
    phases = rng.uniform(0.0, _TWO_PI, size=orders.size)
    noise_seed = int(rng.integers(0, 2**63 - 1))
    t = np.arange(n_out) / fs
    freq = orders * rpm_ref / 60.0
    live = freq < 0.5 * fs  # muted, never clipped (ROADMAP section 6, item 16)
    w = _fade_window(n_out, n_fade)
    interior = slice(n_fade, n_out - n_fade)
    stem = np.zeros(n_out)
    swings = {}
    for k in range(orders.size):
        key = format(float(orders[k]), ".12g")
        if not live[k] or not amp[k] > 0.0:
            swings[key] = None
            continue
        if model == "sinusoid":
            s = amp[k] * np.cos(_TWO_PI * freq[k] * t + phases[k])
        else:
            s = _cluster_stem(freq[k], amp[k], cluster_width_hz, n_out, fs, rng)
        s *= w
        stem += s
        swings[key] = float(envelope_stats(s[interior], fs)["swing_db"])
    noise = synthesize_broadband(psd, freqs, fs, n_out, seed=noise_seed) * w
    mix = stem + noise

    def stats(v):
        v = v[interior]
        if not np.any(v):
            return None
        s = envelope_stats(v, fs)
        return {"swing_db": s["swing_db"], "crest_factor_db": s["crest_factor_db"]}

    e_o, e_r = float(np.sum(stem**2)), float(np.sum(noise**2))
    info = {
        "model": model,
        "extractor": extractor,
        "fs": fs,
        "rpm_ref": rpm_ref,
        "duration": duration,
        "n": int(n_out),
        "orders": orders,
        "frequencies_hz": freq,
        "muted_orders": orders[~live],
        "tolerance": tolerance,
        "window_revs": window_revs,
        "extractor_options": dict(kw),
        "cluster_width_hz": cluster_width_hz,
        "fade": fade,
        "n_fade": n_fade,
        "seed": used_seed,
        "phases": phases,
        "amplitude": "peak",
        "window": {
            "start": a,
            "stop": b,
            "t_start": a / fs,
            "t_stop": b / fs,
            "n_samples": b - a,
            "revolutions": revs,
            "rpm_mean": float(np.mean(rpm[a:b])),
            "rpm_min": float(np.min(rpm[a:b])),
            "rpm_max": float(np.max(rpm[a:b])),
        },
        "psd": psd_info,
        "order_stem": stats(stem),
        "mix": stats(mix),
        "per_order_swing_db": swings,
        "orders_to_residual_db": float(10.0 * np.log10(e_o / e_r)) if e_o > 0 and e_r > 0 else None,
    }
    return ExcerptResult(
        x=mix,
        orders=stem,
        residual=noise,
        fs=fs,
        rpm_ref=rpm_ref,
        amplitudes={float(o): float(v) for o, v in zip(orders, amp)},
        psd=(freqs, psd),
        info=jsonable(info, "info"),
    )
