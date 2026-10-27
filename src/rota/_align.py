"""Time alignment of recordings made by unsynchronised acquisition systems.

:func:`align_signals` estimates the time offset between two recordings of the
same sound field made by independent systems (different clocks, possibly
different sampling rates) and, when two or more markers such as hand claps are
available, the clock drift between them. The delay is found by the generalised
cross-correlation with phase transform (GCC-PHAT, Knapp & Carter 1976), whose
sharp peak does not depend on the spectra of the two microphones or
accelerometers, and refined to a fraction of a sample by maximising the
band-limited (sinc) interpolation of the correlogram.

Conventions
-----------
Times are in s, rates in Hz, drift in parts per million (ppm). The offset
and drift define the map from the time axis of the reference recording
``ref`` to that of the second recording ``other``::

    t_other = offset + (1 + drift_ppm * 1e-6) * t_ref

so that ``offset > 0`` means that ``other`` lags (its events occur later in
its own time stamps), the sign of ``rota.synth.delay_signal(x, fs, delay)``
with ``delay = offset``, and ``drift_ppm > 0`` means that the clock of
``other`` runs *fast* (its samples are closer together in true time), the
sign of ``clock_drift_ppm`` in :func:`rota.synth.measurement_chain`. Time
zero of each recording is its first sample.
"""

import warnings
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy import fft as sfft
from scipy import optimize, signal
from scipy.interpolate import CubicSpline

__all__ = ["AlignResult", "align_signals", "detect_onsets"]

_PHAT_FLOOR = 1e-2  # relative floor of the PHAT weighting (see _phat_weights)
_MIN_SEGMENT = 32  # shortest segment correlated, in samples
_SAMPLE_FLOOR = 0.01  # uncertainty floor in common-rate samples
_AMBIGUITY = 0.8  # second/first peak ratio above which the lag is ambiguous


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------


@dataclass
class AlignResult:
    """Result of :func:`align_signals`.

    The time map is ``t_other = offset + (1 + drift_ppm * 1e-6) * t_ref``
    (see :meth:`to_other` and :meth:`to_ref`). If ``ok`` is False the
    numerical fields hold NaN (``uncertainty`` is infinite) and the maps
    return NaN.

    Attributes
    ----------
    ok : bool
        True if every correlation peak stood out of the correlogram floor
        (see ``psr`` and ``threshold``). False flags a failure, for instance
        when the recordings share no common content or the true lag exceeds
        ``max_lag``; the result must not be used.
    message : str
        Human-readable summary of what was done or why it failed.
    offset : float
        Offset in s of ``other`` against ``ref`` at ``t_ref = 0`` (positive:
        ``other`` lags).
    drift_ppm : float
        Clock drift of ``other`` against ``ref`` in ppm (positive: ``other``
        runs fast). 0 when it was not estimated (``drift_estimated`` False).
    drift_estimated : bool
        True if two or more markers allowed estimating the drift.
    uncertainty : float
        Standard uncertainty in s of the time map, averaged (root mean
        square) over the marker times (the whole record for a global
        correlation), see :meth:`uncertainty_at`. Without a drift estimate the
        drift itself is not part of it.
    offset_uncertainty, drift_uncertainty_ppm : float
        Standard uncertainties of ``offset`` (s) and ``drift_ppm``.
    fs : float
        Common sampling rate in Hz at which the correlation was done.
    marker_times : ndarray
        Marker times in the time axis of ``ref``, in s (empty for a global
        correlation).
    marker_lags : ndarray
        Measured lag ``t_other - t_ref`` at each marker in s.
    marker_uncertainty : ndarray
        Standard uncertainty in s of each lag.
    psr : ndarray
        Peak-to-floor ratio of each correlation: the correlogram peak divided
        by the robust standard deviation of the correlogram floor.
    ambiguity : ndarray
        Height of the second highest correlation peak (outside the main
        peak) relative to the highest, one value per correlation; close to 1
        for periodic content, which has no unique lag.
    threshold : float
        Smallest ``psr`` accepted.
    polarity : ndarray
        +1 if the correlation peak was positive, -1 if the signals have
        opposite polarity (a reversed sensor), one value per correlation.
    """

    ok: bool
    message: str
    offset: float
    drift_ppm: float
    drift_estimated: bool
    uncertainty: float
    offset_uncertainty: float
    drift_uncertainty_ppm: float
    fs: float
    marker_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    marker_lags: np.ndarray = field(default_factory=lambda: np.zeros(0))
    marker_uncertainty: np.ndarray = field(default_factory=lambda: np.zeros(0))
    psr: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ambiguity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    threshold: float = float("nan")
    polarity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _cov: np.ndarray = field(default_factory=lambda: np.full((2, 2), np.nan), repr=False)

    @property
    def eps(self):
        """Relative clock error ``drift_ppm * 1e-6``."""
        return self.drift_ppm * 1e-6

    def to_other(self, t_ref):
        """Time stamps in ``other`` of events at times ``t_ref`` of ``ref`` (s)."""
        return self.offset + (1.0 + self.eps) * np.asarray(t_ref, dtype=float)

    def to_ref(self, t_other):
        """Times in ``ref`` (s) of the events stamped ``t_other`` in ``other``."""
        return (np.asarray(t_other, dtype=float) - self.offset) / (1.0 + self.eps)

    def uncertainty_at(self, t_ref):
        """Standard uncertainty in s of :meth:`to_other` at reference time ``t_ref``.

        Propagates the covariance of ``offset`` and ``eps`` from the marker
        fit, so it grows away from the markers when the drift is estimated.
        """
        t = np.asarray(t_ref, dtype=float)
        c = self._cov
        var = c[0, 0] + 2.0 * t * c[0, 1] + t**2 * c[1, 1]
        return np.sqrt(np.maximum(var, 0.0))

    def resample_other(self, other, fs_other, fs_out, n_out=None):
        """Resample ``other`` onto the time grid of ``ref``.

        Evaluates ``other`` (cubic-spline interpolation, which is accurate
        for content well below ``fs_other / 2``; oversample first for
        wideband signals) at the instants ``to_other(k / fs_out)``, ``k = 0,
        1, ...``, so that sample ``k`` of the output is simultaneous with
        sample ``k`` of a ``ref`` sampled at ``fs_out``. Instants outside the
        record of ``other`` give NaN.

        Parameters
        ----------
        other : array_like, shape (n,)
            The second recording, in any unit.
        fs_other : float
            Its sampling rate in Hz.
        fs_out : float
            Sampling rate of the output in Hz.
        n_out : int, optional
            Number of output samples (default: all that fit in ``other``).

        Returns
        -------
        ndarray, shape (n_out,)
            ``other`` on the ``ref`` time grid, in the unit of ``other``.
        """
        if not self.ok:
            raise ValueError("alignment failed; there is no valid time map")
        y = np.asarray(other, dtype=float)
        if y.ndim != 1:
            raise ValueError("other must be one-dimensional")
        t_o = np.arange(y.size) / float(fs_other)
        if n_out is None:
            t_end = t_o[-1]
            n_out = int(np.floor((self.to_ref(t_end)) * fs_out)) + 1
            n_out = max(n_out, 0)
        t_q = self.to_other(np.arange(n_out) / float(fs_out))
        out = CubicSpline(t_o, y)(t_q)
        out[(t_q < t_o[0]) | (t_q > t_o[-1])] = np.nan
        return out


def _failed(message, fs, **kw):
    nan = float("nan")
    return AlignResult(
        ok=False, message=message, offset=nan, drift_ppm=nan, drift_estimated=False,
        uncertainty=float("inf"), offset_uncertainty=float("inf"),
        drift_uncertainty_ppm=float("inf"), fs=fs, **kw,
    )


# --------------------------------------------------------------------------
# onset detection
# --------------------------------------------------------------------------


def detect_onsets(x, fs, *, threshold_db=15.0, frame=0.005, min_separation=1.0, highpass=100.0):
    """Locate transient onsets (hand claps) from the short-time energy.

    The signal is high-passed (second-order Butterworth, zero phase) to ignore
    rumble, its energy is averaged over frames of ``frame`` seconds
    (``mean(x**2)`` over the frame starting at each sample) and expressed in
    dB. The background is the median of this envelope; an onset is a
    crossing of ``background + threshold_db`` from below. Crossings closer
    than ``min_separation`` to the previous accepted one belong to the same
    event (reverberation, double claps) and are dropped. Each onset time is
    then refined to the first sample of the frame whose magnitude exceeds
    25 % of the largest magnitude in the frame.

    The result is an approximate marker: it only has to be accurate to
    within the marker window of :func:`align_signals`, which then locates the
    event to a fraction of a sample by cross-correlation.

    Parameters
    ----------
    x : array_like, shape (n,)
        Signal in any unit.
    fs : float
        Sampling rate in Hz.
    threshold_db : float, default 15
        Level above the median frame energy in dB (energy ratio, 10 log10).
    frame : float, default 0.005
        Energy frame length in s.
    min_separation : float, default 1.0
        Minimum time between two reported events in s.
    highpass : float or None, default 100
        High-pass corner frequency in Hz; None disables the filter.

    Returns
    -------
    ndarray, shape (k,)
        Onset times in s from the first sample, increasing; empty if none.

    Detection uses relative energy levels only; no amplitude convention applies.

    References
    ----------
    Bello, J. P., Daudet, L., Abdallah, S., Duxbury, C., Davies, M., &
    Sandler, M. B. (2005). A tutorial on onset detection in music signals.
    IEEE Transactions on Speech and Audio Processing, 13(5), 1035-1047.
    """
    x = _check_signal(x, "x")
    fs = _check_fs(fs, "fs")
    nf = max(2, int(round(frame * fs)))
    if x.size < 2 * nf:
        return np.zeros(0)
    x = x - np.mean(x)
    if highpass is not None and 0 < highpass < 0.45 * fs:
        sos = signal.butter(2, highpass, "highpass", fs=fs, output="sos")
        x = signal.sosfiltfilt(sos, x)
    c = np.concatenate(([0.0], np.cumsum(x * x)))
    n_env = x.size - nf + 1
    env = (c[nf : nf + n_env] - c[:n_env]) / nf
    env_db = 10.0 * np.log10(env + 1e-300)
    level = np.median(env_db) + threshold_db
    above = env_db > level
    cross = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    if above[0]:
        cross = np.concatenate(([0], cross))
    out = []
    last = -np.inf
    sep = int(round(min_separation * fs))
    for i in cross:
        if i - last < sep and out:
            continue
        seg = np.abs(x[i : i + 2 * nf])
        j = int(np.argmax(seg > 0.25 * seg.max())) if seg.size else 0
        out.append((i + j) / fs)
        last = i
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _check_fs(fs, name):
    fs = float(fs)
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError(f"{name} must be a positive sampling rate in Hz")
    return fs


def _check_signal(x, name):
    x = np.asarray(x, dtype=float)
    if x.ndim == 2 and 1 in x.shape:
        x = x.reshape(-1)
    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional (one channel)")
    if x.size < _MIN_SEGMENT:
        raise ValueError(f"{name} is too short")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values")
    return x


def _to_rate(x, fs, fs_target):
    """Resample ``x`` from ``fs`` to ``fs_target`` by a rational factor.

    The factor is the fraction ``up/down`` closest to ``fs_target/fs`` with a
    denominator up to 10000; the effective rate ``fs * up / down`` is returned
    (equal to ``fs_target`` unless the ratio is not representable), so the
    time axis of the output stays exact.
    """
    x = x - np.mean(x)
    if abs(fs - fs_target) <= 1e-9 * fs:
        return x, fs
    ratio = (Fraction(float(fs_target)) / Fraction(float(fs))).limit_denominator(10000)
    up, down = ratio.numerator, ratio.denominator
    return signal.resample_poly(x, up, down), fs * up / down


def _phat_weights(G, band_mask):
    """PHAT weighting ``G / (|G| + floor)`` limited to a band.

    The floor, a small fraction of the mean magnitude inside the band, keeps
    frequency bins that carry no signal (below the noise) from being raised
    to unit magnitude, which would add noise to the correlogram; for bins well
    above it the weighting is the pure phase transform of Knapp & Carter.
    """
    mag = np.abs(G)
    ref = mag[band_mask].mean() if np.any(band_mask) else mag.mean()
    W = G / (mag + _PHAT_FLOOR * ref + 1e-300)
    W[~band_mask] = 0.0
    return W


def _correlate(r, o, ts_r, ts_o, fs, band, delay_lo, delay_hi, refine):
    """GCC-PHAT delay of ``o`` against ``r`` within ``[delay_lo, delay_hi]`` s.

    ``ts_r`` and ``ts_o`` are the times in s of the first samples of the
    segments. Returns a dict with the delay in s (``t_o - t_r``), its standard
    uncertainty, the peak-to-floor ratio ``psr``, its acceptance threshold and
    the polarity; or None when the search range is empty.
    """
    nr, no = r.size, o.size
    if nr < _MIN_SEGMENT or no < _MIN_SEGMENT:
        return None
    n_fft = sfft.next_fast_len(nr + no - 1, real=True)
    Xr = sfft.rfft(r - r.mean(), n_fft)
    Xo = sfft.rfft(o - o.mean(), n_fft)
    G = Xo * np.conj(Xr)
    f = np.arange(G.size) * fs / n_fft
    lo_f, hi_f = (0.0, 0.5 * fs) if band is None else band
    mask = (f >= lo_f) & (f <= hi_f) & (f > 0)
    if not np.any(mask):
        return None
    W = _phat_weights(G, mask)
    c = sfft.irfft(W, n_fft)

    shift = (ts_o - ts_r) * fs  # delay = (lag + shift) / fs
    l_lo = max(int(np.ceil(delay_lo * fs - shift)), -(nr - 1))
    l_hi = min(int(np.floor(delay_hi * fs - shift)), no - 1)
    if l_hi < l_lo:
        return None
    lags = np.arange(l_lo, l_hi + 1)
    cand = c[lags % n_fft]
    k0 = int(np.argmax(np.abs(cand)))
    l0 = int(lags[k0])
    peak = cand[k0]
    pol = 1.0 if peak >= 0 else -1.0

    # robust floor: MAD of the whole correlogram, leaving out the peak region
    b_width = max(hi_f - lo_f, fs / n_fft)
    guard = int(np.ceil(4.0 * fs / b_width)) + 2
    idx = np.arange(n_fft)
    d = np.minimum((idx - l0) % n_fft, (l0 - idx) % n_fft)
    far = np.abs(lags - l0) > guard
    ambiguity = float(np.max(np.abs(cand[far])) / abs(peak)) if np.any(far) else 0.0
    floor_vals = c[d > guard]
    if floor_vals.size < 16:
        floor_vals = c[d > 0]
    sigma_bg = 1.4826 * np.median(np.abs(floor_vals - np.median(floor_vals))) + 1e-300
    psr = float(np.abs(peak) / sigma_bg)
    n_search = max(lags.size, 2)
    threshold = float(np.sqrt(2.0 * np.log(2.0 * n_search)) + 4.0)

    k = np.flatnonzero(W != 0)
    wgt = np.where((k == 0) | ((n_fft % 2 == 0) & (k == n_fft // 2)), 1.0, 2.0)
    Wr, Wi = (wgt * W[k].real) / n_fft, (wgt * W[k].imag) / n_fft
    kk = 2.0 * np.pi * k / n_fft  # rad per sample

    lag = float(l0)
    if refine:

        def neg(tau):
            th = kk * tau
            return -pol * (Wr @ np.cos(th) - Wi @ np.sin(th))

        res = optimize.minimize_scalar(
            neg, bounds=(l0 - 1.0, l0 + 1.0), method="bounded", options={"xatol": 1e-5}
        )
        lag = float(res.x)
    # Delay uncertainty from the curvature of the peak and the noise of the
    # slope of the correlogram: sigma = std(c') / |c''| (in samples).
    th = kk * lag
    curv = abs(Wr @ (kk**2 * np.cos(th)) - Wi @ (kk**2 * np.sin(th)))
    dW = np.zeros_like(W)
    dW[k] = 1j * kk * W[k]
    dc = sfft.irfft(dW, n_fft)
    dvals = dc[d > guard]
    if dvals.size < 16:
        dvals = dc[d > 0]
    sigma_d = 1.4826 * np.median(np.abs(dvals - np.median(dvals))) + 1e-300
    sigma = float(sigma_d / max(curv, 1e-300) / fs)
    sigma = float(np.hypot(sigma, _SAMPLE_FLOOR / fs))
    return {
        "delay": (lag + shift) / fs,
        "sigma": sigma,
        "psr": psr,
        "ambiguity": ambiguity,
        "threshold": threshold,
        "polarity": pol,
    }


def _parse_markers(markers, ref, fs_ref):
    """Marker spec -> (ref_times, other_times_or_None)."""
    if isinstance(markers, str):
        if markers != "auto":
            raise ValueError("markers must be None, 'auto' or marker times in s")
        t = detect_onsets(ref, fs_ref)
        if t.size == 0:
            return None, None
        if t.size == 1:
            return t, None
        return np.array([t[0], t[-1]]), None
    m = np.asarray(markers, dtype=float)
    if m.ndim == 0:
        m = m[None]
    if m.ndim == 1:
        if m.size < 1:
            raise ValueError("markers is empty")
        if not np.all(np.isfinite(m)):
            raise ValueError("markers must be finite")
        return m, None
    if m.ndim == 2 and m.shape[0] == 2 and m.shape[1] >= 1:
        if not np.all(np.isfinite(m)):
            raise ValueError("markers must be finite")
        return m[0], m[1]
    raise ValueError("markers must be marker times in ref (s), or (ref_times, other_times)")


# --------------------------------------------------------------------------
# main function
# --------------------------------------------------------------------------


def align_signals(
    ref,
    other,
    fs_ref,
    fs_other=None,
    *,
    markers=None,
    max_lag=10.0,
    band=None,
    refine=True,
    marker_window=0.25,
    fs_common=None,
):
    """Time offset and clock drift between two unsynchronised recordings.

    Two acquisition systems record the same sound field with independent
    clocks (and possibly different sampling rates). The delay of ``other``
    against ``ref`` is the lag of the peak of their generalised
    cross-correlation with phase transform (GCC-PHAT; Knapp & Carter 1976),
    whose cross-spectrum is normalised to unit magnitude so that the peak is
    sharp whatever the spectra of the two channels. The peak is refined to a
    fraction of a sample (``refine=True``) by maximising the band-limited
    interpolation of the correlogram (the sinc interpolation implied by the
    discrete spectrum; a parabola through three points would be biased by
    about 0.1 sample for a sinc-shaped peak).

    Procedure.

    1. *Common rate.* Both recordings are brought to the lower of the two
       rates (or ``fs_common``) by polyphase resampling with
       :func:`scipy.signal.resample_poly` (zero-phase anti-aliasing FIR; the
       ratio is the closest fraction with denominator up to 10000, so 5:1 and
       44.1 : 48 are exact). Only content below the Nyquist frequency of the
       slower system can be common, so nothing is lost. The mean is removed.
    2. *Markers.* With ``markers=None`` one global lag is estimated from the
       whole records (no drift). With marker times (or ``"auto"``, which
       takes the first and the last onset found by :func:`detect_onsets` in
       ``ref``), the lag is estimated in a window of ``+-marker_window``
       around each marker, searching ``other`` within ``+-max_lag``. Two
       markers (a start and an end event) give the drift: with the lag
       ``tau_k = t_other - t_ref`` at ``t_ref = t_k``, the clock error is
       ``eps = (tau_2 - tau_1) / (t_2 - t_1)`` and
       ``offset = tau_1 - eps t_1``. More than two markers are fitted by
       weighted least squares. One marker gives an offset only. The
       markers should be as far apart as possible: the uncertainty of the
       drift falls as ``1 / (t_2 - t_1)``. A recommended practice is a clap at
       the start and one at the end of every recording.
    3. *Quality.* The peak-to-floor ratio ``psr`` (peak over the robust
       standard deviation of the correlogram floor) must exceed
       ``sqrt(2 ln(2 n)) + 4``, ``n`` the number of lags searched, i.e. the
       expected extreme of a Gaussian floor plus four floor deviations (the
       margin allows for the floor of a correlogram with transients not being
       exactly Gaussian); if not, the result is flagged ``ok=False`` with NaN values and a
       ``RuntimeWarning`` is issued. No common content, or a lag beyond
       ``max_lag``, fail this way.
       A second peak above 0.8 of the highest (periodic content such as a
       steady tone) is flagged the same way, because the lag is then
       ambiguous.
    4. *Uncertainty.* The standard uncertainty of each lag is the
       standard deviation of the slope of the correlogram floor divided by the
       curvature of the peak, ``std(c') / |c''|`` (a peak-location error for
       a smooth peak in noise, which is the classical time-delay bound
       ``1 / (2 pi f_rms sqrt(SNR))`` of Skolnik 2008 with the effective
       bandwidth measured instead of assumed), added in quadrature to 0.01
       sample as a floor. The covariance of offset and drift follows from the
       fit and is exposed by :meth:`AlignResult.uncertainty_at`.

    The map from the time axis of ``ref`` to that of ``other`` is
    ``t_other = offset + (1 + drift_ppm 1e-6) t_ref``; see the module
    docstring for the sign conventions and :class:`AlignResult` for
    :meth:`~AlignResult.to_ref` and :meth:`~AlignResult.resample_other`.

    Parameters
    ----------
    ref, other : array_like, shape (n,)
        The two recordings of one channel each, in any unit (the result does
        not depend on their amplitudes). ``ref`` defines the time axis.
    fs_ref : float
        Sampling rate of ``ref`` in Hz.
    fs_other : float, optional
        Sampling rate of ``other`` in Hz (default ``fs_ref``).
    markers : None, "auto", array_like of shape (k,) or (2, k), optional
        None: one global correlation. ``"auto"``: onsets detected in ``ref``.
        Array of ``k`` approximate marker times in s on the time axis of
        ``ref``. Array of shape (2, k): approximate times of the same events
        in ``ref`` (first row) and in ``other`` (second row); ``max_lag`` is
        then the search half-range around the difference of the two.
    max_lag : float, default 10
        Largest lag searched in s, either side (about 0 or about the marker
        difference).
    band : (float, float), optional
        Frequency band in Hz used for the correlation (default: 0 to the
        Nyquist frequency of the common rate). Restricting it to the band
        where both channels have the common signal raises the peak in noise.
    refine : bool, default True
        Sub-sample refinement; False returns integer-sample lags of the
        common rate.
    marker_window : float, default 0.25
        Half-length in s of the window around each marker.
    fs_common : float, optional
        Common rate in Hz; default the lower of the two rates. A higher rate
        is useless because the slower recording has no content above its
        Nyquist frequency.

    Returns
    -------
    AlignResult
        Offset in s, drift in ppm, standard uncertainties, quality ratios and
        the time map. Time quantities are in s; amplitudes are not used.

    Raises
    ------
    ValueError
        For invalid arguments (non-positive rates, wrong shapes, bad markers
        or band, a common rate above the lower rate).

    References
    ----------
    Knapp, C. H., & Carter, G. C. (1976). The generalized correlation method
    for estimation of time delay. IEEE Transactions on Acoustics, Speech, and
    Signal Processing, 24(4), 320-327.
    Carter, G. C. (1987). Coherence and time delay estimation. Proceedings of
    the IEEE, 75(2), 236-255.
    Skolnik, M. I. (2008). Radar Handbook, 3rd ed., ch. 1 (accuracy of range
    measurement). McGraw-Hill.
    Crochiere, R. E., & Rabiner, L. R. (1983). Multirate Digital Signal
    Processing, ch. 2. Prentice-Hall.
    """
    ref = _check_signal(ref, "ref")
    other = _check_signal(other, "other")
    fs_ref = _check_fs(fs_ref, "fs_ref")
    fs_other = fs_ref if fs_other is None else _check_fs(fs_other, "fs_other")
    max_lag = float(max_lag)
    if not (np.isfinite(max_lag) and max_lag > 0):
        raise ValueError("max_lag must be a positive time in s")
    marker_window = float(marker_window)
    if not (np.isfinite(marker_window) and marker_window > 0):
        raise ValueError("marker_window must be a positive time in s")
    fs_c = min(fs_ref, fs_other) if fs_common is None else _check_fs(fs_common, "fs_common")
    if fs_c > min(fs_ref, fs_other) * (1 + 1e-9):
        raise ValueError("fs_common must not exceed the lower sampling rate")
    if band is not None:
        band = (float(band[0]), float(band[1]))
        if not (0 <= band[0] < band[1]):
            raise ValueError("band must be (low, high) in Hz with 0 <= low < high")
        band = (band[0], min(band[1], 0.5 * fs_c))

    ref_c, fs_r = _to_rate(ref, fs_ref, fs_c)
    oth_c, fs_o = _to_rate(other, fs_other, fs_c)
    fs_c = fs_r  # effective rate; fs_o differs from it by at most ~1e-8 relative

    t_ref_m, t_oth_m = (None, None) if markers is None else _parse_markers(markers, ref, fs_ref)
    notes = []
    if markers is not None and t_ref_m is None:
        notes.append("no onset found; used one global correlation")

    taus, sigmas, psrs, amb, thr, pols, tm_used = [], [], [], [], [], [], []
    if t_ref_m is None:
        res = _correlate(ref_c, oth_c, 0.0, 0.0, fs_c, band, -max_lag, max_lag, refine)
        if res is not None:
            taus.append(res["delay"]); sigmas.append(res["sigma"])
            psrs.append(res["psr"]); amb.append(res["ambiguity"]); thr.append(res["threshold"]); pols.append(res["polarity"])
        global_mode = True
    else:
        global_mode = False
        w = marker_window
        for k, tm in enumerate(t_ref_m):
            guess = 0.0 if t_oth_m is None else float(t_oth_m[k] - tm)
            i0, i1 = int(round((tm - w) * fs_r)), int(round((tm + w) * fs_r))
            i0, i1 = max(i0, 0), min(i1, ref_c.size)
            lo_t, hi_t = tm - w + guess - max_lag, tm + w + guess + max_lag
            j0, j1 = int(round(lo_t * fs_o)), int(round(hi_t * fs_o))
            j0, j1 = max(j0, 0), min(j1, oth_c.size)
            if i1 - i0 < _MIN_SEGMENT or j1 - j0 < _MIN_SEGMENT:
                notes.append(f"marker at {tm:.3f} s is outside the records")
                continue
            res = _correlate(
                ref_c[i0:i1], oth_c[j0:j1], i0 / fs_r, j0 / fs_o, fs_c, band,
                guess - max_lag, guess + max_lag, refine,
            )
            if res is None:
                notes.append(f"marker at {tm:.3f} s: empty search range")
                continue
            taus.append(res["delay"]); sigmas.append(res["sigma"])
            psrs.append(res["psr"]); amb.append(res["ambiguity"]); thr.append(res["threshold"]); pols.append(res["polarity"])
            tm_used.append(tm)

    psr = np.asarray(psrs)
    tm_arr = np.asarray(tm_used)
    n_expected = 1 if global_mode else len(t_ref_m)
    common = dict(
        marker_times=tm_arr, marker_lags=np.asarray(taus), marker_uncertainty=np.asarray(sigmas),
        psr=psr, ambiguity=np.asarray(amb),
        threshold=float(max(thr)) if thr else float("nan"), polarity=np.asarray(pols),
    )
    if len(taus) < n_expected or psr.size == 0:
        msg = "; ".join(notes) or "no correlation could be computed"
        warnings.warn(f"align_signals failed: {msg}", RuntimeWarning, stacklevel=2)
        return _failed(f"failed: {msg}", fs_c, **common)
    weak = [i for i in range(len(psrs)) if psrs[i] < thr[i]]
    ambiguous = [i for i in range(len(psrs)) if i not in weak and amb[i] > _AMBIGUITY]
    if weak or ambiguous:
        bad = sorted(weak + ambiguous)
        where = "global" if global_mode else ", ".join(f"{tm_arr[i]:.2f} s" for i in bad)
        if weak:
            msg = (
                f"no clear correlation peak ({where}: peak-to-floor ratio "
                f"{', '.join(f'{psrs[i]:.1f}' for i in weak)} below {max(thr):.1f}); "
                "the recordings may share no common content, or the lag exceeds max_lag"
            )
        else:
            msg = (
                f"ambiguous correlation peak ({where}: a second peak reaches "
                f"{', '.join(f'{amb[i]:.2f}' for i in ambiguous)} of the highest); "
                "periodic content only: use a transient marker"
            )
        warnings.warn(f"align_signals failed: {msg}", RuntimeWarning, stacklevel=2)
        return _failed(f"failed: {msg}", fs_c, **common)

    tau = np.asarray(taus)
    sig = np.asarray(sigmas)
    if global_mode or tau.size == 1:
        offset, eps = float(tau[0]), 0.0
        cov = np.array([[sig[0] ** 2, 0.0], [0.0, 0.0]])
        est = False
        msg = "global correlation" if global_mode else "one marker: offset only"
    else:
        A = np.column_stack([np.ones(tau.size), tm_arr])
        Wt = 1.0 / sig**2
        N = A.T @ (A * Wt[:, None])
        if abs(np.linalg.det(N)) < 1e-300 or np.ptp(tm_arr) <= 0:
            return _failed("failed: markers coincide", fs_c, **common)
        cov = np.linalg.inv(N)
        offset, eps = (cov @ (A.T @ (Wt * tau))).tolist()
        est = True
        msg = f"{tau.size} markers: offset and drift"
    drift_ppm = float(eps * 1e6)
    # relation between the lag fit and the time map: lag = offset + eps t_ref
    t_eval = tm_arr if tm_arr.size else np.zeros(1)
    var = cov[0, 0] + 2.0 * t_eval * cov[0, 1] + t_eval**2 * cov[1, 1]
    unc = float(np.sqrt(np.mean(np.maximum(var, 0.0))))
    if notes:
        msg += "; " + "; ".join(notes)
    if np.any(np.asarray(pols) < 0):
        msg += "; opposite polarity"
    return AlignResult(
        ok=True, message=msg, offset=float(offset), drift_ppm=drift_ppm, drift_estimated=est,
        uncertainty=unc, offset_uncertainty=float(np.sqrt(cov[0, 0])),
        drift_uncertainty_ppm=float(np.sqrt(cov[1, 1]) * 1e6), fs=fs_c, _cov=cov, **common,
    )
