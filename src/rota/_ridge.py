"""Tacholess speed estimation: comb scoring on a whitened spectrogram and
dynamic-programming ridge tracking.

Units: time in s, frequency in Hz, rotational speed in rev/min (rpm). The
estimator uses only ratios of spectral levels, so the unit and scale of the
input signal do not matter (no amplitude convention is involved).
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from ._utils import as_2d

__all__ = ["RidgeResult", "WhiteningWarning", "rpm_from_ridge"]


class WhiteningWarning(RuntimeWarning):
    """The whitening width is smaller than the spectral lines it must keep.

    Issued by :func:`rpm_from_ridge` when, along the tracked path, the
    frequency range one comb line occupies within a window (Hann main lobe
    plus the sweep of the highest comb order) is wider than the running
    median that whitens the spectrum. The median then follows the lines and
    erases them, and the speed estimate cannot be trusted there (its
    ``confidence`` is lowered on those frames). It is a ``RuntimeWarning``
    but is not the search-range warning: the true speed need not be outside
    ``rpm_range``.
    """


# Decimated Nyquist frequency over the highest comb frequency.
_NYQUIST_MARGIN = 1.25
# Zero-padding of the FFT relative to the window length.
_PAD = 4
# Default grid step = 1/_GRID_FRACTION of the relative Hann main-lobe FWHM at
# the highest comb frequency; clipped to the interval below.
_GRID_FRACTION = 8.0
_GRID_LIMITS = (1e-4, 5e-3)
# Default penalty = typical score prominence / _PENALTY_DIVISOR per % jump.
_PENALTY_DIVISOR = 10.0
# Confidence: alternatives closer than this relative distance (+-5 %) to the
# chosen speed are not counted as competitors.
_CONF_EXCLUDE = 0.05
# Out-of-range evidence: the comb is also scored up to this factor beyond
# either end of the range (not searched by the path), and a frame is flagged
# if such a candidate beats the chosen one by more than _BEYOND_MARGIN times
# the typical score prominence.
_GUARD = 2.0
_BEYOND_MARGIN = 0.25
# Largest score matrix (frames x grid, guard bands included), in elements.
_MAX_SCORE_SIZE = 150_000_000
# Whitening. One comb line of frequency f occupies (null to null)
# _LINE_BINS / window Hz for a Hann window (Harris 1978), plus the frequency it
# sweeps during the window, f * rate * window (rate = |d rpm/dt| / rpm). The
# running median must be at least that wide at the line's frequency: that is the
# condition checked after tracking. The automatic width is the larger of a static
# floor, _STATIC_BINS * _LINE_BINS / window (6 / window, 15 Hz for 0.4 s: the
# classical default, kept wherever it already satisfies the condition), and the
# line extent itself:
#   width(f) = max(_STATIC_BINS * _LINE_BINS / window,
#                  _AUTO_FACTOR * (_LINE_BINS / window + rate * window * f)).
_LINE_BINS = 4.0
_STATIC_BINS = 1.5
_AUTO_FACTOR = 1.0
# A frame violates the condition if its width at the top comb frequency is below
# this fraction of the line extent (calibrated on synthetic run-ups: see the
# Notes of rpm_from_ridge).
_VIOLATION_RATIO = 0.8
# Growth of the automatic width with frequency (Hz per Hz): rounded up to a
# ladder of _AUTO_STEPS steps per octave from _Q_MIN (below which it is 0),
# capped at _Q_MAX; frames of equal growth are whitened together.
_AUTO_STEPS = 3
_Q_MIN = 0.01
_Q_MAX = 1.0
# First pass of the automatic width: coarse grid step (relative), this many
# times the final step, but within [_PASS1_MIN_GRID, _PASS1_MAX_GRID] (and never
# finer than the final step).
_PASS1_GRID_FACTOR = 4.0
_PASS1_MIN_GRID = 0.002
_PASS1_MAX_GRID = 0.01


@dataclass
class RidgeResult:
    """Result of :func:`rpm_from_ridge`.

    Units: time in s, speed in rpm. ``T`` is the number of frames and ``G``
    the number of grid speeds.

    Attributes
    ----------
    t : ndarray, shape (T,)
        Frame centre times in s. Frames lie fully inside the signal, so the
        first centre is half a window after the first sample.
    rpm : ndarray, shape (T,)
        Estimated speed in rpm at each frame (sub-grid refined if requested).
        Frames flagged in ``at_bound`` hold the search-range boundary (or a
        value next to it) and must not be trusted.
    confidence : ndarray, shape (T,)
        Value in [0, 1]. ``(s_sel - s_alt) / (s_sel - s_med)``, with ``s_sel``
        the comb score at the chosen speed, ``s_alt`` the best score at
        speeds more than 5 % away from it and ``s_med`` the median score over
        the grid, all taken from that frame's own score profile (not the
        path total). 1: no competitor stands out; 0: a distant competitor is
        as good as the choice (or the choice is not above the median).
        Octave candidates (half or double speed) score part of the true comb
        and typically hold the value below 1. On frames flagged in
        ``whiten_violated`` it is multiplied by ``(whiten_width / (0.8 *
        line_width)) ** 2``: the score profile there is that of a spectrum
        whose lines were erased and says nothing about the speed.
    at_bound : ndarray of bool, shape (T,)
        True where the estimate sits at or evidently beyond a bound of
        ``rpm_range``: the chosen grid point lies within one grid step of
        either end, or ``beyond`` is set. The true speed may then be outside
        the range, and ``rpm`` there must not be trusted.
    beyond : ndarray of bool, shape (T,)
        True where a candidate outside ``rpm_range`` (up to a factor 2 above
        the upper or below the lower end, as far as ``fs`` allows) scores
        higher than the chosen point by more than a quarter of the median
        score prominence (maximum minus median of the frame's scores, median
        over frames). The path is never allowed to leave the range; this only
        reports that the data prefer a speed beyond it. It catches speeds
        that are well outside the range but whose comb partly matches an
        in-range candidate (octave or other ratios), which the end test
        cannot. At very low signal-to-noise ratio it also fires by chance
        (with very low ``confidence``).
    rpm_grid : ndarray, shape (G,)
        Candidate speeds in rpm: logarithmically spaced from ``rpm_range[0]``
        to ``rpm_range[1]`` inclusive, i.e. a constant relative step.
    score : ndarray, shape (T, G)
        Per-frame comb score (weighted mean of the whitened level in dB,
        floored at 0 dB), before path tracking.
    orders : ndarray, shape (K,)
        Comb orders used.
    weights : ndarray, shape (K,)
        Comb weights, normalised to sum 1.
    penalty : float
        Path penalty used, in score units per % speed change per frame.
    rpm_range : tuple of float
        Search range in rpm.
    guard_range : tuple of float
        Speed interval in rpm over which the comb was scored: ``rpm_range``
        extended by the guard bands (down to half the lower end, up to twice
        the upper end). The upper side is cut short when the comb would come
        too close to the Nyquist frequency of ``fs`` (then ``beyond`` cannot
        see speeds above ``guard_range[1]``; only the end test applies
        there).
    t_out, rpm_out : ndarray or None
        Times requested through ``t_out`` and the speed interpolated there
        (see :meth:`rpm_at`); None if ``t_out`` was not given.
    whiten_width : ndarray, shape (T,)
        Width in Hz of the running median that whitened each frame, at the
        frequency of the highest comb order at the estimated speed (a number
        given as ``whiten``: that number; ``whiten='auto'``: it grows with
        frequency and differs from frame to frame). ``inf`` for
        ``whiten=None`` (the frame's median over all bins was used).
    line_width : ndarray, shape (T,)
        Frequency in Hz that one line of the highest comb order occupies
        within the frame's window: ``4 / window`` (Hann main lobe) plus its
        sweep ``max(orders) * rpm / 60 * (|d rpm/dt| / rpm) * window``. The
        rate comes from the tracked path (not on frames flagged in
        ``at_bound``) and from the first-pass path, which does not depend on
        the whitening, whichever is larger (see Notes of
        :func:`rpm_from_ridge`).
    whiten_violated : ndarray of bool, shape (T,)
        True where ``whiten_width < 0.8 * line_width``: the running median is
        narrower than the lines and erases them. A :class:`WhiteningWarning`
        is issued if any is set, and ``confidence`` is multiplied there by
        ``(whiten_width / (0.8 * line_width)) ** 2``.
    """

    t: np.ndarray
    rpm: np.ndarray
    confidence: np.ndarray
    at_bound: np.ndarray
    beyond: np.ndarray
    rpm_grid: np.ndarray
    score: np.ndarray
    orders: np.ndarray
    weights: np.ndarray
    penalty: float
    rpm_range: tuple
    guard_range: tuple
    t_out: np.ndarray | None = None
    rpm_out: np.ndarray | None = None
    whiten_width: np.ndarray | None = None
    line_width: np.ndarray | None = None
    whiten_violated: np.ndarray | None = None

    @property
    def bound_fraction(self):
        """Fraction of frames with ``at_bound`` set (0 for no frames)."""
        return float(np.mean(self.at_bound)) if self.at_bound.size else 0.0

    def rpm_at(self, t):
        """Speed in rpm at times ``t`` (s), linearly interpolated over frames.

        Outside the span of the frame centres the first or last estimate is
        held. This is not an extrapolation: within half a window of the
        signal ends the speed may differ from the held value if the signal is
        accelerating.
        """
        return np.interp(np.asarray(t, dtype=float), self.t, self.rpm)


def _l1_max_sum(a, c):
    """Max-sum transform with an L1 cost, in linear time.

    Returns ``m`` and ``src`` with ``m[g] = max_k (a[k] - c |g - k|)`` and
    ``src[g]`` the maximising ``k``. Two passes (forward, backward) of a
    running maximum, written with cumulative maxima so that no Python loop
    over ``g`` is needed (Felzenszwalb & Huttenlocher 2012, sec. 3 for the
    L1 distance transform, here in its max-sum form).
    """
    n = a.size
    idx = np.arange(n)
    ramp = c * idx
    # Forward pass: best predecessor k <= g.
    u = a + ramp
    run = np.maximum.accumulate(u)
    f = run - ramp
    src_f = np.maximum.accumulate(np.where(u >= run, idx, 0))
    # Backward pass over f: best k >= g (reverse, accumulate, reverse back).
    v = (f - ramp)[::-1]
    run_b = np.maximum.accumulate(v)
    src_b = np.maximum.accumulate(np.where(v >= run_b, idx, 0))
    m = run_b[::-1] + ramp
    src_b = (n - 1 - src_b)[::-1]
    return m, src_f[src_b]


def _viterbi(S, c):
    """Maximum-sum path through ``S`` (frames x grid) with L1 jump cost ``c``.

    ``c`` is the cost in score units per grid step of change between
    consecutive frames. Cost per frame is linear in the grid, so one frame
    takes O(G) operations, not O(G^2).
    """
    T, G = S.shape
    back = np.empty((T, G), dtype=np.int32)
    delta = S[0].astype(float)
    for t in range(1, T):
        m, back[t] = _l1_max_sum(delta, c)
        delta = S[t] + m
        delta -= delta.max()
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _running_median_freq(P, width_bins):
    """Running median along axis 1 over ``width_bins`` bins.

    The median is evaluated on every ``stride``-th bin (adjacent zero-padded
    bins are strongly correlated, so nothing is lost) and interpolated back.
    """
    nb = P.shape[1]
    stride = max(1, width_bins // 16)
    sub = P[:, ::stride]
    k = max(3, round(width_bins / stride) | 1)
    med = ndimage.median_filter(sub, size=(1, k), mode="reflect")
    pos = np.arange(nb) / stride
    i0 = np.minimum(pos.astype(int), sub.shape[1] - 2)
    fr = np.minimum(pos - i0, 1.0)
    return med[:, i0] * (1.0 - fr) + med[:, i0 + 1] * fr


def _running_median_warped(P, df, w0_hz, w1_hz, q):
    """Running median along axis 1 whose width grows with frequency.

    The median width at frequency ``f`` is ``max(w0_hz, w1_hz + q * f)``: the
    constant ``w0_hz`` below the knee ``(w0_hz - w1_hz) / q`` and growing in
    proportion to the frequency (constant Q) above it (``w1_hz <= w0_hz``). The
    spectrum is read at points equally spaced in ``u(f) = integral of df /
    width(f)``, so that a median over a fixed number of those points spans the
    local width at every frequency; the result is interpolated back to the
    bins. Like :func:`_running_median_freq`, the median is evaluated on a
    sub-sampled set (16 points per width).
    """
    nb = P.shape[1]
    per = 16  # points per median width
    w0_hz = max(w0_hz, w1_hz)
    f_c = (w0_hz - w1_hz) / q  # knee
    u_c = f_c / w0_hz

    def u_of(f):
        f = np.asarray(f, dtype=float)
        return np.where(f <= f_c, f / w0_hz, u_c + np.log((w1_hz + q * f) / w0_hz) / q)

    u_max = float(u_of((nb - 1) * df))
    n_pt = int(np.ceil(u_max * per)) + 2
    uu = np.arange(n_pt) / per
    f_pt = np.where(uu <= u_c, uu * w0_hz, (w0_hz * np.exp(q * (uu - u_c)) - w1_hz) / q)
    sub = P[:, np.minimum(np.rint(f_pt / df).astype(int), nb - 1)]
    med = ndimage.median_filter(sub, size=(1, per + 1), mode="reflect")
    pos = u_of(np.arange(nb) * df) * per
    i0 = np.minimum(pos.astype(int), n_pt - 2)
    fr = np.minimum(pos - i0, 1.0)
    return med[:, i0] * (1.0 - fr) + med[:, i0 + 1] * fr


def _floor_of(P, df, w0_hz, w1_hz, q):
    """Noise-floor estimate of each row of ``P`` (frames x bins).

    ``w0_hz = 0``: the frame's median over all bins. Otherwise the running
    median along frequency, of width ``max(w0_hz, w1_hz + q * f)`` (constant
    ``w0_hz`` for ``q = 0``).
    """
    if not w0_hz:
        return np.median(P, axis=1, keepdims=True)
    if not q:
        return _running_median_freq(P, max(3, round(w0_hz / df)))
    return _running_median_warped(P, df, w0_hz, w1_hz, q)


def _whitened_levels(frames_all, starts, win, nfft, nb, df, w0_hz, w1_hz, q, moving):
    """Whitened level (dB, floored at 0) of the Hann-STFT frames, float32.

    ``w0_hz`` and ``q`` give, per frame, the running-median width
    ``max(w0_hz, w1_hz + q f)`` (see :func:`_floor_of`; ``w1_hz`` is a
    scalar); ``w0_hz = 0`` selects the frame median over the ``nb`` retained
    bins. Frames are processed in chunks, grouped by width.
    """
    n_fr = starts.size
    level = np.empty((n_fr, nb), dtype=np.float32)
    chunk = max(1, int(2**22 // nfft))
    tiny = np.finfo(float).tiny
    for a in range(0, n_fr, chunk):
        fr = frames_all[starts[a : a + chunk]]
        fr = (fr - fr.mean(axis=1, keepdims=True)) * win
        F = np.fft.rfft(fr, n=nfft, axis=1)[:, :nb]
        P = F.real**2 + F.imag**2
        w0_c, q_c = w0_hz[a : a + chunk], q[a : a + chunk]
        groups = np.unique(np.column_stack([w0_c, q_c]), axis=0)
        if len(groups) == 1:
            ratio = P / np.maximum(_floor_of(P, df, groups[0, 0], w1_hz, groups[0, 1]), tiny)
        else:
            ratio = np.empty_like(P)
            for g0, gq in groups:
                sel = (w0_c == g0) & (q_c == gq)
                Ps = P[sel]
                ratio[sel] = Ps / np.maximum(_floor_of(Ps, df, g0, w1_hz, gq), tiny)
        level[a : a + chunk] = 10.0 * np.log10(np.maximum(ratio, 1e-12))
    if moving:
        level -= np.median(level, axis=0).astype(np.float32)
    np.maximum(level, 0.0, out=level)
    return level


def _comb_scores(level, df, rpm_values, ords, w):
    """Comb score (frames x speeds): weighted mean of the interpolated level."""
    nb = level.shape[1]
    S = np.zeros((level.shape[0], rpm_values.size))
    for m, wm in zip(ords, w):
        if wm == 0.0:
            continue
        pos = m * rpm_values / 60.0 / df
        i0 = np.minimum(pos.astype(int), nb - 2)
        fr = pos - i0
        S += wm * (level[:, i0] * (1.0 - fr) + level[:, i0 + 1] * fr)
    return S


def _path_rate(rpm, hop, window, spread=False):
    """Relative rate of change ``|d rpm/dt| / rpm`` (1/s) along a speed path.

    ``rpm`` has one value per frame (step ``hop`` s). The slope of each frame
    is the least-squares slope over the frames whose centres lie within one
    window of it (at least two on either side; odd reflection at the ends
    keeps a linear trend), which averages the jitter of the path down to a few
    % of the slope. A short median first removes isolated outliers of a poor
    path. With ``spread`` the largest rate within half a window is returned (a
    conservative envelope for choosing a width).

    One line of order ``m`` at frequency ``f = m rpm / 60`` sweeps ``f * rate
    * window`` Hz within a window.
    """
    n = rpm.size
    if n < 3:
        return np.zeros(n)
    h_fit = max(2, int(np.ceil(window / hop - 1e-9)))
    r = ndimage.median_filter(np.asarray(rpm, dtype=float), size=3, mode="nearest")
    pad = min(h_fit, n - 1)
    rp = np.pad(r, pad, mode="reflect", reflect_type="odd")
    u = np.arange(-pad, pad + 1, dtype=float)
    slope = np.correlate(rp, u, mode="valid") / (np.sum(u * u) * hop)  # rpm/s
    rate = np.abs(slope) / r
    if spread:
        h = max(1, int(np.floor(0.5 * window / hop + 1e-9)))
        rate = ndimage.maximum_filter1d(rate, size=2 * h + 1, mode="nearest")
    return rate


def _line_extent(rpm, hop, window, max_order):
    """Frequency in Hz one line of the highest comb order occupies in a window.

    ``_LINE_BINS / window`` (Hann main lobe) plus its sweep within the window,
    ``max_order * rpm / 60 * rate * window``, with the relative rate of the
    speed path ``rpm`` (:func:`_path_rate`).
    """
    return _LINE_BINS / window + max_order * rpm / 60.0 * _path_rate(rpm, hop, window) * window


def _first_pass_path(
    frames_all, starts, win, nfft, df, ords, w, rpm_range, grid, moving, f_top
):
    """Coarse speed path (rpm per frame) that does not depend on any width.

    Frames are normalised by their own median (there is no running-median
    width to get wrong), the comb is scored on a coarse grid and a Viterbi
    path with the default penalty is taken. It is accurate to a fraction of a
    percent in the absence of coloured noise or fixed lines (which a frame
    median does not remove), which is all that the slope of the path needs.
    """
    lo, hi = rpm_range
    n_fr = starts.size
    nb1 = min(nfft // 2 + 1, int(np.ceil(f_top / df)) + 3)
    level = _whitened_levels(
        frames_all, starts, win, nfft, nb1, df, np.zeros(n_fr), 0.0, np.zeros(n_fr), moving
    )
    g1 = max(grid, _PASS1_MIN_GRID, min(_PASS1_GRID_FACTOR * grid, _PASS1_MAX_GRID))
    n1 = max(3, int(np.ceil(np.log(hi / lo) / np.log1p(g1))) + 1)
    rpm1 = np.geomspace(lo, hi, n1)
    S1 = _comb_scores(level, df, rpm1, ords, w)
    del level
    prominence = S1.max(axis=1) - np.median(S1, axis=1)
    cost = float(np.median(prominence)) / _PENALTY_DIVISOR * 100.0 * np.log(hi / lo) / (n1 - 1)
    return rpm1[_viterbi(S1, cost)]


def _auto_slope(path1, win_s, hop_s):
    """Growth ``q`` of the running-median width with frequency, per frame.

    The relative rate of the first-pass path (:func:`_path_rate`, largest
    value within half a window) gives the sweep of a line at frequency ``f``
    in a window, ``f * rate * window``; the median width must follow it, so
    ``q = _AUTO_FACTOR * rate * window``. ``q`` is capped at ``_Q_MAX`` and
    rounded up to a ladder of ``_AUTO_STEPS`` steps per octave from ``_Q_MIN``
    (below which it is 0), so that frames fall into a few groups.
    """
    q = _AUTO_FACTOR * win_s * _path_rate(path1, hop_s, win_s, spread=True)
    rung = np.ceil(_AUTO_STEPS * np.log2(np.maximum(q, 1e-12) / _Q_MIN) - 1e-9)
    q = np.where(q > _Q_MIN, _Q_MIN * 2.0 ** (np.maximum(rung, 0.0) / _AUTO_STEPS), 0.0)
    return np.minimum(q, _Q_MAX)


def _check_positive(value, name):
    value = float(value)
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive finite number")
    return value


def rpm_from_ridge(
    x,
    fs,
    rpm_range,
    *,
    orders=(1, 2, 3, 4),
    weights=None,
    window=0.4,
    hop=0.05,
    whiten="auto",
    moving=False,
    penalty=None,
    grid=None,
    refine=True,
    t_out=None,
):
    """Estimate the rotational speed from the signal alone (no tachometer).

    Method (after the reviews of tacholess instantaneous-speed estimators,
    Peeters et al. 2019, and the subharmonic-summation pitch detector of
    Hermes 1988):

    1. The channels are averaged to one signal and decimated (polyphase
       low-pass) to the lowest rate whose Nyquist frequency is at least 1.25
       times the highest comb frequency ``max(orders) * rpm_range[1] / 60``.
    2. A Hann-window STFT (``window`` s, step ``hop`` s, zero-padded 4x)
       gives the power spectrum of each frame. It is whitened: divided by its
       running median along frequency (width ``whiten``, see below), then
       expressed in dB and floored at 0 dB, so the noise floor reads about
       0 dB and a spectral line reads its height above the local floor. With
       ``moving=True`` each bin's median over time is also subtracted (before
       the floor), which removes lines at fixed frequencies. The median must
       be wider than the lines it leaves in place (Notes, "Whitening
       width"); with ``whiten='auto'`` a first, coarse pass tracks the speed
       without any running median (each frame divided by its median over
       all bins), and the sweep of its path sets the width of the final
       pass.
    3. For every candidate speed ``r`` on a logarithmic grid over
       ``rpm_range``, the comb score of a frame is
       ``sum_m w_m * L(order_m * r / 60)``, with ``L`` the whitened level
       (linearly interpolated) and ``w`` the weights (normalised to sum 1).
    4. A Viterbi (max-sum dynamic programming) path through the frames x grid
       score matrix maximises the total score minus ``penalty`` per % of
       speed change between consecutive frames (L1 penalty on the relative
       jump; for the grid used this is linear in the index difference). Each
       frame is processed in linear time in the grid size with a two-pass L1
       distance transform (Felzenszwalb & Huttenlocher 2012), not O(G^2).
    5. With ``refine`` the chosen grid point is moved to the vertex of the
       parabola through the frame's own scores at it and its two neighbours
       (only if that parabola has a maximum; shift limited to half a step).
    6. Range check (ROADMAP T10): the comb is also scored, but not tracked,
       in guard bands up to a factor 2 beyond either end of ``rpm_range``.
       Frames whose path ends within one grid step of an end, or in which a
       guard candidate clearly out-scores the path (``beyond``), are flagged
       in ``at_bound``.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal in any unit (the estimator is scale invariant). Channels are
        averaged before analysis; they should be in phase (use one channel
        otherwise).
    fs : float
        Sampling rate in Hz.
    rpm_range : (float, float)
        Search range ``(lo, hi)`` in rpm, ``0 < lo < hi``. The comb
        ``max(orders) * hi / 60`` must stay below ``fs / 2``.
    orders : sequence of float, default (1, 2, 3, 4)
        Comb orders (cycles per revolution). They may be fractional: half
        orders for a three-cylinder four-stroke engine
        (``(1.5, 3, 4.5, 6)``), motor pole-pair or gear-mesh harmonics.
    weights : sequence of float, optional
        One non-negative weight per order; normalised to sum 1. Default
        ``order ** -0.5``: lower orders count more. This breaks the tie of a
        single visible order with its sub-harmonics (speed / 2, / 3, / 4 put
        a higher comb tooth on that order: with equal weights all of them
        score alike), as in subharmonic summation (Hermes 1988). The price
        is a bias towards twice the speed when only the even orders are
        present (see Notes).
    window : float, default 0.4
        STFT window length in s. Frequency resolution is about ``1.5 /
        window`` Hz (Hann main lobe).
    hop : float, default 0.05
        Frame step in s.
    whiten : "auto", float or None, default "auto"
        Width of the running median along frequency that whitens the
        spectrum. ``"auto"``: chosen per frame and growing with frequency,
        ``max(6 / window, 4 / window + rate * window * f)`` Hz at frequency
        ``f`` (``rate`` is the relative speed rate ``|d rpm/dt| / rpm`` of
        the first-pass path), i.e. 15 Hz for the default window wherever the
        sweep is small and the width of a swept line where it is not; see
        the Notes. A number: that constant width in Hz, which must satisfy
        the condition of the Notes (it is checked after tracking, see
        Warns). None: no running median; levels are then relative to the
        frame's median over all bins, which is robust against any sweep but
        keeps coloured noise (resonances, humps) in the levels.
    moving : bool, default False
        Subtract each frequency bin's median over time, to suppress lines at
        fixed frequency (resonances, tones, mains hum). Valid only when the
        speed sweeps: a bin must be hit by an order for less than half of the
        frames, so at near-constant speed this removes the orders too. A fixed
        tone also puts a stripe into the guard bands, which can flag a few
        ``beyond`` frames on the weakest frames of a correct run (a window
        across the start of a ramp, ``confidence`` near 0; measured with a
        250 Hz tone and the comb 1.5 to 6, whose first tooth sits on it at
        10000 rpm): ``moving=True`` removes it.
    penalty : float, optional
        Path penalty in score units (dB of the weighted mean level) per %
        speed change per frame; 0 gives independent per-frame maxima. Default:
        the median over frames of (maximum - median of the frame's scores),
        divided by 10. A 10 % jump in one frame then costs as much as a
        typical ridge is high, while the steepest run-ups (a few % per frame)
        cost a fraction of it. Chosen on synthetic run-ups at 8 kHz: divisors
        of 5 or less lose sweeps beyond about 1000 rpm/s at ``hop=0.05``,
        20 or more let the path wander in noise; 10 followed 2000 rpm/s
        (window 0.2 s, 95 % of frames within 3 %) and held 1000 rpm/s at 99 %.
        It is in the result as ``penalty``.
    grid : float, optional
        Relative step of the speed grid (0.001 is 0.1 %). Default: 1/8 of the
        relative Hann main-lobe FWHM at the highest comb frequency,
        ``0.18 / (window * max(orders) * rpm_range[1] / 60)``, clipped to
        [0.01 %, 0.5 %] (about 0.1 % for a 0.4 s window and a 400 Hz comb).
    refine : bool, default True
        Sub-grid parabolic refinement.
    t_out : array_like, optional
        Times in s (for instance the signal sample times) at which the speed
        is interpolated into ``rpm_out`` of the result.

    Returns
    -------
    RidgeResult
        ``t`` (s), ``rpm`` (rpm), ``confidence`` (0 to 1), ``at_bound``,
        ``beyond``, ``rpm_grid``, ``score``, ``bound_fraction``,
        :meth:`~RidgeResult.rpm_at`, ``whiten_width``, ``line_width``,
        ``whiten_violated`` and, if ``t_out`` was given, ``rpm_out``.
        A ``RuntimeWarning`` is issued when ``bound_fraction > 0``: the
        estimator never returns a boundary value silently.

    Warns
    -----
    RuntimeWarning
        If any frame has ``at_bound`` set (search-range warning).
    WhiteningWarning
        (A subclass of ``RuntimeWarning``.) If on any frame the running
        median is narrower than 0.8 times the frequency a comb line occupies
        within the window (``whiten_violated``). It names the width, the line
        extent and the worst time; ``confidence`` is lowered on those frames.
        This is not about the search range: if both warnings appear and the
        speed is believed to be in range, fix the whitening first.

    Notes
    -----
    * The estimate is scale invariant; ``x`` has no amplitude convention.
      Output speeds are in rpm.
    * Whitening width. Whitening the spectrum so that discrete components stand
      out against a coloured background is routine in diagnostics (Randall &
      Antoni 2011; the cepstral form for variable speed, Borghesani et al.
      2013); here it is a running median of the power spectrum along
      frequency. A line of frequency ``f`` occupies ``L(f) = 4 / window
      + rate * window * f`` Hz in a window: the Hann main lobe (Harris 1978)
      plus the sweep, which is proportional to the frequency (``rate`` is the
      relative speed rate ``|d rpm/dt| / rpm``; at 750 rpm/s the sixth order
      sweeps 30 Hz at any speed, three times the main lobe). A running median
      erases what fills more than half of it (it ignores up to half of its
      samples as outliers; Huber 1981), so its width must reach ``L(f)`` at
      every comb frequency, the highest order being the binding one.
      Measured on synthetic half-order run-ups (8 kHz, windows of 0.4 s and
      0.25 s, 330 to 1500 rpm/s, low and moderate noise: 12 cases), a
      constant width fails abruptly, at a point between 0.4 and 1.1 ``L`` of
      the top order depending on the case, and works from 1.2 ``L`` in every
      case. ``'auto'`` therefore takes ``max(6 / window, L(f))`` per frame,
      with the rate rounded up on a ladder of three steps per octave (0 to
      26 % wider than ``L``). Below the knee ``f = 2 / (rate * window**2)``
      that is the classical 15 Hz (for 0.4 s), because the narrower the
      median the closer it follows coloured noise: a 60 Hz wide noise hump
      is removed by widths up to 60 Hz (0.86 of the frames within 3 %,
      against 0.89 at 15 Hz), not by the frame median (0.62). Above the knee
      the width grows in proportion to ``f``, the growth ``rate * window``
      capped at 1.
    * Sweep range, measured with ``'auto'`` and all other defaults (8 kHz,
      900 to 4500 rpm, comb 1.5 to 6): 97 to 100 % of the frames within 3 %
      at 158, 333, 750 and 1500 rpm/s up, 99 to 100 % at 333 and 750 rpm/s
      down, and 100 % for a comb of ten integer orders at 1500 rpm/s up.
      Two limits are not about the median. Run-downs faster than about 1000
      rpm/s lose frames at low speed to the path penalty (at 1500 rpm/s 94 %
      for the half-order comb and 79 % for ten orders, where the relative rate
      is 5 % per frame; ``penalty=0``: 98 to 100 %; the frame median or
      another width: 79 to 95 %). And above about 1500 rpm/s at the default
      window the lines blur into each other (the sixth order sweeps 80 Hz
      per window at 2000 rpm/s) whatever the width: the estimate fails with
      about a quarter of the frames in ``at_bound``, and a shorter window
      cures it (0.25 s: 100 % at 2000 rpm/s). Lines that overlap at low speed
      (harmonic spacing below the line extent) are the same limit.
    * Whitening check. After tracking, the line extent ``L`` of the top comb
      order is computed per frame from the relative rate of the path. A path
      that was tracked on erased lines is itself wrong, and so is its slope
      (with ``whiten=30`` on a 750 rpm/s run-up the path reads an extent of 20
      Hz against a true 40 Hz), so the rate is
      also taken from the first-pass path (frame-median normalisation, which
      does not depend on the width), median-filtered over 2.5 windows because
      that path has spikes where it is wrong (coloured noise without
      ``moving``), and the larger extent counts; the final path is left out
      on frames in ``at_bound``, where it is not valid. A frame violates the
      condition when the width is below 0.8 ``L`` (``whiten_violated``): 0.8
      is the tolerance for the scatter of the slope and lies where the
      measured failures begin; on a clean run-up the extent is within 1 % of
      the true one. The first pass costs up to about 15 % of the run time for an
      explicit width (none for ``None``); with ``'auto'`` the total is below
      that of the former constant 15 Hz median (0.6 to 0.85 times at 12 to
      50 s of signal), whose 25-bin median is dearer than the sub-sampled one
      used here. Frames with different widths are whitened in groups; with
      ``moving=True`` the time median of a bin then mixes them, which leaves a
      small residue at an extreme fixed tone (the stripe score is 2.6 times
      lower than without ``moving``, against 4.5 for a constant width).
    * Subharmonic ambiguity. A comb also scores part of its teeth at the
      ratios k/m of the true speed (k an order present in the signal, m a
      comb order): 1/2, 2/3, 3/2, 2, ... The default weights make the true
      speed win when the orders 1 to 4 are visible with comparable or
      decreasing amplitude, or when a single order dominates. They do not
      resolve spectra with only even orders (four-cylinder engines: orders 2,
      4 with little 1, 3), where twice the speed scores as well as the
      truth, nor half-order machines: for the half orders of a three-cylinder
      four-stroke engine an integer comb locks at exactly 1.5 times the speed.
      Give the comb that matches the machine (``orders=(1.5, 3, 4.5)``, or
      ``(2, 4, 6)``), read ``confidence`` (it is low where an octave or other
      ratio competes) and use ``rpm_range`` from a priori knowledge.
    * Resolution. The Hann line is about ``1.5 / window`` Hz wide (3.6 Hz
      at 0.4 s), which is 10 % of an order at 37 Hz: one low-frequency order
      alone is located to a few % only (80 % of frames within 3 % in tests),
      precision comes from the higher orders of the comb. Use a longer
      ``window`` for a single order or for low speeds.
    * Search-range bounds. If the true speed leaves ``rpm_range``, the best
      in-range candidate is the nearest end only just beyond it (a few %,
      because the high orders are narrow); further out it is usually an
      octave or another ratio of the truth inside the range, which the end
      test does not see. The guard bands (``beyond``) flag those frames as
      long as the true speed is within a factor 2 of the range; further out
      the lock may go unflagged, with low ``confidence``. Set ``rpm_range``
      from a priori knowledge (idle speed, redline).
    * Time axis and ends. Frames lie fully inside the signal, so ``t`` runs
      from ``window / 2`` to the duration minus ``window / 2`` and
      :meth:`~RidgeResult.rpm_at` holds the end values outside. The L1 path
      penalty is not balanced at the first and last frame (one neighbour
      only): where the score peak is wide they can be pulled by up to about
      1 % towards their neighbour (measured 1.0 % at 25 Hz fundamental,
      0.25 % in the interior).
    * Memory: the score and back-pointer matrices have ``T x G`` entries
      (about 12 bytes each); ``T`` is the frame count.

    References
    ----------
    Peeters, C., Leclere, Q., Antoni, J., Lindahl, P., Donnal, J., Leeb, S.,
    & Helsen, J. (2019). Review and comparison of tacholess instantaneous
    speed estimation methods on experimental vibration data. Mechanical
    Systems and Signal Processing, 129, 407-436.
    Hermes, D. J. (1988). Measurement of pitch by subharmonic summation.
    Journal of the Acoustical Society of America, 83(1), 257-264.
    Viterbi, A. J. (1967). Error bounds for convolutional codes and an
    asymptotically optimum decoding algorithm. IEEE Transactions on
    Information Theory, 13(2), 260-269.
    Felzenszwalb, P. F., & Huttenlocher, D. P. (2012). Distance transforms of
    sampled functions. Theory of Computing, 8(19), 415-428.
    Harris, F. J. (1978). On the use of windows for harmonic analysis with the
    discrete Fourier transform. Proceedings of the IEEE, 66(1), 51-83.
    Huber, P. J. (1981). Robust Statistics. Wiley.
    Randall, R. B., & Antoni, J. (2011). Rolling element bearing diagnostics -
    A tutorial. Mechanical Systems and Signal Processing, 25(2), 485-520.
    Borghesani, P., Pennacchi, P., Randall, R. B., Sawalhi, N., & Ricci, R.
    (2013). Application of cepstrum pre-whitening for the diagnosis of
    bearing faults under variable speed conditions. Mechanical Systems and
    Signal Processing, 36(2), 370-384.
    """
    # ---- validation -------------------------------------------------------
    fs = _check_positive(fs, "fs")
    X, _ = as_2d(x)
    if X.shape[0] < 2:
        raise ValueError("x needs at least 2 samples")
    if not np.all(np.isfinite(X)):
        raise ValueError("x contains non-finite values")
    try:
        lo, hi = (float(v) for v in rpm_range)
    except (TypeError, ValueError):
        raise ValueError("rpm_range must be a pair (lo, hi) in rpm") from None
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        raise ValueError("rpm_range must satisfy 0 < lo < hi (rpm)")
    ords = np.asarray(orders, dtype=float).ravel()
    if ords.size == 0 or not (np.all(np.isfinite(ords)) and np.all(ords > 0)):
        raise ValueError("orders must be a non-empty sequence of positive numbers")
    if weights is None:
        w = ords**-0.5
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != ords.shape:
            raise ValueError("weights must have one entry per order")
        if not (np.all(np.isfinite(w)) and np.all(w >= 0) and w.sum() > 0):
            raise ValueError("weights must be finite and non-negative, not all zero")
    w = w / w.sum()
    window = _check_positive(window, "window")
    hop = _check_positive(hop, "hop")
    auto = isinstance(whiten, str)
    if auto:
        if whiten.strip().lower() != "auto":
            raise ValueError("whiten must be 'auto', None or a positive width in Hz")
        whiten = None
    elif whiten is not None:
        whiten = _check_positive(whiten, "whiten")
    if penalty is not None:
        penalty = float(penalty)
        if not (np.isfinite(penalty) and penalty >= 0):
            raise ValueError("penalty must be a non-negative finite number")
    if grid is not None:
        grid = float(grid)
        if not (np.isfinite(grid) and 0 < grid < 0.5):
            raise ValueError("grid must be a relative step in (0, 0.5)")
    if t_out is not None:
        t_out = np.asarray(t_out, dtype=float)
        if t_out.ndim != 1 or not np.all(np.isfinite(t_out)):
            raise ValueError("t_out must be a finite 1-D array of times in s")

    f_top = float(ords.max()) * hi / 60.0
    if f_top >= 0.5 * fs:
        raise ValueError(
            f"the comb reaches {f_top:.4g} Hz at rpm_range[1], not below fs/2 = "
            f"{0.5 * fs:.4g} Hz; lower the orders or the range, or raise fs"
        )

    # Guard bands: the comb is also scored up to a factor _GUARD beyond either
    # end of the range, as far as the sampling rate allows.
    max_order = float(ords.max())
    hi_guard = max(hi, min(_GUARD * hi, 0.5 * fs * 60.0 / (_NYQUIST_MARGIN * max_order)))
    f_top_guard = max_order * hi_guard / 60.0

    # ---- mono mix and decimation -----------------------------------------
    y = X.mean(axis=1)
    q = max(1, int(np.floor(0.5 * fs / (_NYQUIST_MARGIN * f_top_guard))))
    if q > 1:
        y = signal.resample_poly(y, 1, q)
    fs_d = fs / q

    # ---- STFT and whitening ----------------------------------------------
    n_win = round(window * fs_d)
    n_hop = max(1, round(hop * fs_d))
    if n_win < 8:
        raise ValueError("window is shorter than 8 samples at the analysis rate")
    if y.size < n_win:
        raise ValueError("signal is shorter than one window")
    starts = np.arange(0, y.size - n_win + 1, n_hop)
    n_fr = starts.size
    nfft = 1 << int(np.ceil(np.log2(_PAD * n_win)))
    df = fs_d / nfft
    win_s, hop_s = n_win / fs_d, n_hop / fs_d  # window and step as realised, s
    win = signal.get_window("hann", n_win)
    frames_all = np.lib.stride_tricks.sliding_window_view(y, n_win)
    nb_max = nfft // 2 + 1

    # ---- speed grid (checked before any large array is built) --------------
    if grid is None:
        grid = float(np.clip(_GRID_FRACTION**-1 * 1.44 / (window * f_top), *_GRID_LIMITS))
    ln_ratio = np.log(hi / lo)
    n_grid = max(3, int(np.ceil(ln_ratio / np.log1p(grid))) + 1)
    step = ln_ratio / (n_grid - 1)  # logarithmic step, rpm_grid[-1] == hi
    n_low = int(np.floor(np.log(_GUARD) / step))
    n_up = int(np.floor(np.log(hi_guard / hi) / step + 1e-9))
    if n_fr * (n_low + n_grid + n_up) > _MAX_SCORE_SIZE:
        raise ValueError(
            f"score matrix of {n_fr} frames x {n_low + n_grid + n_up} speeds is too "
            "large; increase grid or hop, or shorten the signal"
        )

    # ---- width of the running median: w0 + q f per frame (w0 = 0: frame median)
    path1 = None  # first-pass path: sets the automatic width, and checks any width
    w1 = 0.0
    if whiten is None and not auto:
        w0, q_f = np.zeros(n_fr), np.zeros(n_fr)
    else:
        path1 = _first_pass_path(
            frames_all, starts, win, nfft, df, ords, w, (lo, hi), grid, moving, f_top
        )
        if auto:
            w0 = np.full(n_fr, _STATIC_BINS * _LINE_BINS / win_s)
            w1 = _AUTO_FACTOR * _LINE_BINS / win_s
            q_f = _auto_slope(path1, win_s, hop_s)
        else:
            w0, q_f = np.full(n_fr, whiten), np.zeros(n_fr)
    widest = np.maximum(w0, w1 + q_f * f_top_guard)[w0 > 0]
    f_keep = f_top_guard + (0.5 * widest.max() if widest.size else 0.0)
    nb = min(nb_max, int(np.ceil(f_keep / df)) + 3)
    level = _whitened_levels(frames_all, starts, win, nfft, nb, df, w0, w1, q_f, moving)

    # ---- comb score on a logarithmic speed grid ---------------------------
    rpm_ext = lo * np.exp(step * np.arange(-n_low, n_grid + n_up))
    rpm_ext[n_low + n_grid - 1] = hi
    S_ext = _comb_scores(level, df, rpm_ext, ords, w)
    del level
    rpm_grid = rpm_ext[n_low : n_low + n_grid].copy()
    S = np.ascontiguousarray(S_ext[:, n_low : n_low + n_grid])
    s_out = np.full(n_fr, -np.inf)
    if n_low:
        s_out = np.maximum(s_out, S_ext[:, :n_low].max(axis=1))
    if n_up:
        s_out = np.maximum(s_out, S_ext[:, n_low + n_grid :].max(axis=1))
    del S_ext

    # ---- path -------------------------------------------------------------
    if penalty is None:
        prominence = S.max(axis=1) - np.median(S, axis=1)
        penalty = float(np.median(prominence)) / _PENALTY_DIVISOR
    cost = penalty * 100.0 * step  # score units per grid step
    path = _viterbi(S, cost)

    # ---- refinement, flags, confidence ------------------------------------
    rows = np.arange(n_fr)
    pos = path.astype(float)
    if refine:
        g = np.clip(path, 1, n_grid - 2)
        s_m, s_0, s_p = S[rows, g - 1], S[rows, g], S[rows, g + 1]
        den = s_m - 2.0 * s_0 + s_p
        ok = (den < 0) & (path == g)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(ok, 0.5 * (s_m - s_p) / den, 0.0)
        pos = pos + np.clip(d, -0.5, 0.5)
    rpm = lo * np.exp(step * pos)
    s_sel = S[rows, path]
    s_med = np.median(S, axis=1)
    prominence = S.max(axis=1) - s_med
    beyond = (s_out - s_sel) > _BEYOND_MARGIN * np.median(prominence)
    at_bound = (path <= 1) | (path >= n_grid - 2) | beyond

    half = int(np.ceil(np.log1p(_CONF_EXCLUDE) / step))
    pre = np.maximum.accumulate(S, axis=1)
    suf = np.maximum.accumulate(S[:, ::-1], axis=1)[:, ::-1]
    s_alt = np.full(n_fr, -np.inf)
    left = path - half - 1
    right = path + half + 1
    has_l, has_r = left >= 0, right < n_grid
    s_alt[has_l] = pre[rows[has_l], left[has_l]]
    s_alt[has_r] = np.maximum(s_alt[has_r], suf[rows[has_r], right[has_r]])
    s_alt = np.where(np.isfinite(s_alt), s_alt, s_med)
    height = s_sel - s_med
    with np.errstate(divide="ignore", invalid="ignore"):
        conf = np.where(height > 1e-9, (s_sel - s_alt) / height, 0.0)
    conf = np.clip(conf, 0.0, 1.0)

    # Whitening check: does the running median that was used keep the lines?
    # Width and line extent are both linear in frequency, so the highest comb
    # order is the binding one. The extent comes from the sweep of the final
    # path (frames flagged at_bound excluded: their path is not valid) and from
    # that of the first-pass path, which does not depend on the width: when the
    # width is too small the final path is itself wrong and its slope with it.
    # The first-pass extent is median-filtered over 2.5 windows (it has spikes
    # where that path is wrong, e.g. in coloured noise).
    if path1 is not None:
        span = max(3, 2 * round(1.25 * win_s / hop_s) + 1)
        line = ndimage.median_filter(
            _line_extent(path1, hop_s, win_s, max_order), size=span, mode="nearest"
        )
        line = np.maximum(line, np.where(at_bound, 0.0, _line_extent(rpm, hop_s, win_s, max_order)))
    else:
        line = _line_extent(rpm, hop_s, win_s, max_order)
    width_top = np.where(w0 > 0, np.maximum(w0, w1 + q_f * max_order * rpm / 60.0), np.inf)
    violated = width_top < _VIOLATION_RATIO * line
    if violated.any():
        ratio = width_top[violated] / (_VIOLATION_RATIO * line[violated])
        conf[violated] *= np.minimum(1.0, ratio) ** 2

    res = RidgeResult(
        t=(starts + 0.5 * (n_win - 1)) / fs_d,
        rpm=rpm,
        confidence=conf,
        at_bound=at_bound,
        beyond=beyond,
        rpm_grid=rpm_grid,
        score=S,
        orders=ords,
        weights=w,
        penalty=float(penalty),
        rpm_range=(lo, hi),
        guard_range=(float(rpm_ext[0]), float(rpm_ext[-1])),
        whiten_width=width_top,
        line_width=line,
        whiten_violated=violated,
    )
    if t_out is not None:
        res.t_out = t_out
        res.rpm_out = res.rpm_at(t_out)
    if violated.any():
        k = int(np.argmax(np.where(violated, line / width_top, 0.0)))
        warnings.warn(
            f"rpm_from_ridge: the whitening width ({width_top[k]:.3g} Hz) is smaller than "
            f"the frequency one comb line occupies within a window ({line[k]:.3g} Hz = "
            f"4/window + sweep of order {max_order:g}) in {100.0 * violated.mean():.1f} % "
            f"of the frames (worst at t = {res.t[k]:.3g} s). The running median follows "
            "the lines and erases them, so the speed there is unreliable (confidence "
            "lowered). Use whiten='auto' (default) or a larger whiten, or a shorter "
            "window.",
            WhiteningWarning,
            stacklevel=2,
        )
    frac = res.bound_fraction
    if frac > 0:
        cause = ""
        if violated.any():
            cause = (
                f" The whitening width is too small on {100.0 * violated.mean():.1f} % of "
                "the frames (see the other warning), which can by itself cause these flags."
            )
        warnings.warn(
            f"rpm_from_ridge: {100.0 * frac:.1f} % of the frames sit at or beyond a "
            f"bound of the search range ({lo:g}, {hi:g}) rpm; the true speed may be "
            "outside it (or the data too noisy to tell: see confidence). Widen "
            "rpm_range and check RidgeResult.at_bound." + cause,
            RuntimeWarning,
            stacklevel=2,
        )
    return res
