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

__all__ = ["RidgeQualityWarning", "RidgeResult", "WhiteningWarning", "rpm_from_ridge"]


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


class RidgeQualityWarning(RuntimeWarning):
    """The tracked speed is likely wrong as a whole.

    Issued by :func:`rpm_from_ridge` when ``RidgeResult.quality["reliable"]``
    is False: the median ``confidence`` is below 0.2, more than 40 % of the
    frames have a ``confidence`` below 0.1, more than 25 % are flagged in
    ``at_bound``, more than 10 % jump by more than 2 % away from their
    neighbours, or an octave of the track (half or double speed) scores at
    least as high as the track itself on the median frame. The
    message names the criteria that failed. It is a ``RuntimeWarning``.
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
# Confidence: the peak of the chosen speed (at least +-5 %) and those of its
# sub-multiples 1/k (k in _CONF_HARMONIC) are not counted as rivals: a comb on
# the harmonic family of the true speed always scores there in part. Multiples
# k times the speed are rivals: that is where the truth lies when the track
# has locked onto a sub-multiple.
_CONF_EXCLUDE = 0.05
_CONF_HARMONIC = (2.0, 3.0, 4.0)
# Confidence = detection * distinctness. Detection: the height of the chosen
# score above the noise in standard deviations of a noise-only comb score
# (_detection_z), mapped linearly from _Z0 (0) to _Z1 (1); the tracked score
# of a noise-only frame reads z = 2 to 2.7 (median over frames; synthetic
# white and steeply coloured noise, combs of one to four orders), a clean
# run-up 10 to 12. Distinctness: (chosen - best rival) / (chosen - median),
# mapped linearly from 0 (0) to _D1 (1).
_Z0, _Z1 = 3.0, 8.0
_D1 = 0.3
# Mean and standard deviation in dB of the whitened level of a noise-only bin,
# max(0, 10 log10(E / ln 2)) with E standard exponential (numerical
# integration); see _detection_z.
_NOISE_MEAN_DB = 1.6445
_NOISE_SD_DB = 2.2505
# Out-of-range evidence: the comb is also scored up to this factor beyond
# either end of the range, and the path is tracked a second time over that
# extended grid; frames where that unconstrained path leaves the range by more
# than one grid step are flagged in ``beyond``.
_GUARD = 2.0
# Track quality (RidgeResult.quality): limits of a reliable track.
_Q_MIN_CONFIDENCE = 0.2  # median confidence
_Q_LOW_CONFIDENCE = 0.1  # a frame of low confidence ...
_Q_MAX_LOW = 0.4  # ... and the largest fraction of them
_Q_MAX_BOUND = 0.25  # fraction of frames in at_bound
_Q_JUMP = 0.02  # a jump: |ln rpm - its running median over _Q_JUMP_SPAN frames|
_Q_JUMP_SPAN = 5
_Q_MAX_JUMP = 0.10  # fraction of frames that jump
_Q_OCTAVE = 1.0  # octave score over track score (heights above the median)
# WhiteningWarning only if at least this fraction of the frames violate the
# width condition (isolated frames are counted in whiten_violated only).
_WHITEN_WARN_FRACTION = 0.05
# Automatic window: Hann window of the least line extent 4 / T + fdot * T for
# the sweep rate fdot (Hz/s) of the top comb order, T = 2 / sqrt(fdot), with
# fdot the _WINDOW_PERCENTILE-th percentile over frames of the first-pass path
# (slope fitted over at least _WINDOW_FIT s); clipped to [_WINDOW_MIN,
# _WINDOW_MAX] and to the resolution of the closest comb orders at the
# _WINDOW_LOW_PERCENTILE-th percentile speed (spacing >= 4 / T). A track at
# _WINDOW_MAX gives the first path; the window is shortened only if the rule
# asks for more than _WINDOW_TOL less, and re-chosen from each new track (a
# track that fails at too long a window shows a sweep too slow).
_WINDOW_MAX = 0.4
_WINDOW_MIN = 0.1
_WINDOW_PERCENTILE = 75.0
_WINDOW_LOW_PERCENTILE = 10.0
_WINDOW_FIT = 1.0
_WINDOW_TOL = 0.05
_WINDOW_ITER = 3  # tracks after the first: the window is re-chosen from each
_WINDOW_SHRINK = 0.7  # step of the fallback search of an unreliable track
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
        Value in [0, 1], the product of two terms taken from the frame's
        own scores (not the path total). Detection: ``z = (s_sel - max(s_med,
        1.64)) / (2.25 * sqrt(sum(weights**2)))``, the height of the score at
        the chosen speed above the frame's median score (or the mean noise
        level, 1.64 dB, if higher) in standard deviations of a noise-only comb
        score, mapped linearly from ``z = 3`` (0) to ``z = 8`` (1); a
        noise-only frame reads ``z`` of 2 to 2.7, a clean run-up 10 to 12.
        Distinctness: ``(s_sel - s_alt) / (s_sel - s_med)``, mapped linearly
        from 0 (0) to 0.3 (1), with ``s_alt`` the best score outside the peak
        of the chosen speed (from valley to valley, at least +-5 %) and
        outside the peaks at 1/2, 1/3 and 1/4 of it: any rival speed that
        scores nearly as high lowers it, a rational neighbour (7/6, 5/4, 3/2)
        or a multiple (where the truth lies after a lock onto a sub-multiple)
        included. The sub-multiples score part of every comb on the harmonics
        of the truth; the octaves are reported in ``quality["octave_ratio"]``.
        A clean, correct track reads near 1, a noise-only frame near 0 (see
        the Notes of :func:`rpm_from_ridge`). On frames flagged in
        ``whiten_violated`` it is multiplied by ``(whiten_width / (0.8 *
        line_width)) ** 2``: the score profile there is that of a spectrum
        whose lines were erased and says nothing about the speed.
    at_bound : ndarray of bool, shape (T,)
        True where the estimate sits at or evidently beyond a bound of
        ``rpm_range``: the chosen grid point lies within one grid step of
        either end, or ``beyond`` is set. The true speed may then be outside
        the range, and ``rpm`` there must not be trusted.
    beyond : ndarray of bool, shape (T,)
        True where the data prefer a speed outside ``rpm_range``: the path is
        tracked a second time, with the same penalty, over the grid extended
        by the guard bands (up to a factor 2 above the upper or below the
        lower end, as far as ``fs`` allows), and that unconstrained path lies
        more than one grid step outside the range. The returned path is never
        allowed to leave the range; this only reports that a continuous track
        beyond it explains the data better. It catches speeds well outside
        the range whose comb partly matches an in-range candidate (octave or
        other ratios), which the end test cannot, and, being a path and not
        a per-frame maximum, it does not fire on chance peaks of the noise.
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
        is issued if at least 5 % of the frames are set, and ``confidence``
        is multiplied there by ``(whiten_width / (0.8 * line_width)) ** 2``.
        Only frames on which a path is credible count (the first-pass or the
        final path detected, see ``confidence``): on noise the slope of a path
        is meaningless.
    window : float
        Window length in s that was used: ``window`` as given, or the one
        chosen by ``window="auto"``.
    quality : dict
        Summary of the whole track. ``median_confidence`` (median of
        ``confidence``), ``low_confidence_fraction`` (fraction of frames with
        ``confidence`` below 0.1), ``bound_fraction`` (fraction of frames in
        ``at_bound``), ``jump_fraction`` (fraction of frames that deviate by
        more than 2 % from the running median of the track over 5 frames: a
        frame that jumps away from its neighbours, which a ramp never does,
        however steep; a plain frame-to-frame change would count a fast
        run-up, 3.5 % per frame at 850 rpm/s from 1200 rpm with the default
        ``hop``), ``octave_ratio`` (pair: median over the detected frames of the
        height above the frame median of the score at half and at twice the
        tracked speed, over that of the track itself; NaN where not
        available), ``octave_ambiguous`` (either ratio at least 1: the octave scores at
        least as high as the track on the median detected frame),
        ``window`` (s), ``reliable`` (median confidence at least 0.2, at most
        40 % of the frames below 0.1, at most 25 % in ``at_bound``, at most
        10 % jumping, and no octave ambiguity) and ``problems`` (list of the failed criteria, in
        words). A :class:`RidgeQualityWarning` is issued when ``reliable``
        is False.
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
    window: float | None = None
    quality: dict | None = None

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

    ``frames_all`` holds one frame view per channel; the power spectra of the
    channels are averaged (incoherent averaging: channels need not be in
    phase, and uncorrelated noise averages down in power). ``w0_hz`` and
    ``q`` give, per frame, the running-median width ``max(w0_hz, w1_hz + q
    f)`` (see :func:`_floor_of`; ``w1_hz`` is a scalar); ``w0_hz = 0``
    selects the frame median over the ``nb`` retained bins. Frames are
    processed in chunks, grouped by width.
    """
    n_fr = starts.size
    level = np.empty((n_fr, nb), dtype=np.float32)
    chunk = max(1, int(2**22 // nfft))
    tiny = np.finfo(float).tiny
    for a in range(0, n_fr, chunk):
        P = 0.0
        for frames in frames_all:
            fr = frames[starts[a : a + chunk]]
            fr = (fr - fr.mean(axis=1, keepdims=True)) * win
            F = np.fft.rfft(fr, n=nfft, axis=1)[:, :nb]
            P = P + (F.real**2 + F.imag**2)
        P = P / len(frames_all)
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


def _detection_z(S, sel, w):
    """Height of the score ``sel`` (per frame) above the noise, in noise
    standard deviations of a comb score with weights ``w``.

    A noise-only bin, whitened by the median of its neighbours, has the power
    ratio ``E / ln 2`` with ``E`` standard exponential (chi-square with 2
    degrees of freedom; Kay 1993), so its level ``max(0, 10 log10(E / ln
    2))`` has mean ``_NOISE_MEAN_DB`` and standard deviation ``_NOISE_SD_DB``,
    and a comb score of independent teeth the standard deviation
    ``_NOISE_SD_DB * sqrt(sum w**2)``. The reference is the larger of that
    mean and the frame's median score (coloured noise that the whitening did
    not flatten raises the median). Unlike the spread of the frame's own
    scores, this does not grow with the number of lines in the spectrum.
    Returns ``(z, s_med)``.
    """
    s_med = np.median(S, axis=1)
    sigma = _NOISE_SD_DB * float(np.sqrt(np.sum(np.square(w))))
    z = (sel - np.maximum(s_med, _NOISE_MEAN_DB)) / sigma
    return np.maximum(z, 0.0), s_med


def _rival_max(S, centres, half):
    """Best score of each row of ``S`` outside the peaks at ``centres``.

    ``centres`` is a list of per-row index arrays (out-of-grid entries are
    ignored). Around each centre the excluded zone is its peak: from the
    nearest local minimum of the row below it to the nearest one above it,
    and at least ``+-half`` grid steps, so that the shoulder of a wide peak
    (a low order at a short window) is not taken for a rival. Returns -inf
    where nothing is left. Processed in row chunks to bound the memory.
    """
    n_fr, G = S.shape
    out = np.full(n_fr, -np.inf)
    idx = np.arange(G)
    chunk = max(1, int(2**21 // max(G, 1)))
    for a in range(0, n_fr, chunk):
        b = min(n_fr, a + chunk)
        Sc = S[a:b]
        r = np.arange(b - a)
        lm = np.ones(Sc.shape, dtype=bool)
        lm[:, 1:-1] = (Sc[:, 1:-1] <= Sc[:, :-2]) & (Sc[:, 1:-1] <= Sc[:, 2:])
        last = np.maximum.accumulate(np.where(lm, idx, 0), axis=1)
        nxt = np.minimum.accumulate(np.where(lm, idx, G - 1)[:, ::-1], axis=1)[:, ::-1]
        keep = np.ones(Sc.shape, dtype=bool)
        for c in centres:
            c = c[a:b]
            inside = (c >= 0) & (c < G)
            cc = np.clip(c, 0, G - 1)
            lo_i = np.where(inside, np.minimum(last[r, cc], c - half), G)
            hi_i = np.where(inside, np.maximum(nxt[r, cc], c + half), -1)
            keep &= (idx[None, :] < lo_i[:, None]) | (idx[None, :] > hi_i[:, None])
        out[a:b] = np.where(keep, Sc, -np.inf).max(axis=1)
    return out


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
    Returns the path and its detection ``z`` per frame (see
    :func:`_detection_z`), which tells where the path is credible.
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
    p = _viterbi(S1, cost)
    z, _ = _detection_z(S1, S1[np.arange(n_fr), p], w)
    return rpm1[p], z


def _stft_frames(Y, fs_d, window, hop):
    """Frame layout of the Hann STFT of the decimated channels ``Y`` (n, ch)."""
    n_win = round(window * fs_d)
    n_hop = max(1, round(hop * fs_d))
    if n_win < 8:
        raise ValueError("window is shorter than 8 samples at the analysis rate")
    if Y.shape[0] < n_win:
        raise ValueError("signal is shorter than one window")
    nfft = 1 << int(np.ceil(np.log2(_PAD * n_win)))
    return dict(
        n_win=n_win,
        starts=np.arange(0, Y.shape[0] - n_win + 1, n_hop),
        nfft=nfft,
        df=fs_d / nfft,
        win_s=n_win / fs_d,  # window and step as realised, s
        hop_s=n_hop / fs_d,
        win=signal.get_window("hann", n_win),
        frames=[np.lib.stride_tricks.sliding_window_view(Y[:, c], n_win) for c in range(Y.shape[1])],
    )


def _auto_window(rpm, credible, hop_s, win_s, ords):
    """Window length (s) of the least line extent for the sweep of a track.

    A line sweeping ``fdot`` Hz/s occupies ``4 / T + fdot T`` Hz in a Hann
    window of ``T`` s (main lobe plus sweep), least for ``T = 2 /
    sqrt(fdot)``: the classical trade-off between the time and frequency
    spread of a chirp in the spectrogram (Cohen 1995; Kwok & Jones 2000).
    ``fdot`` is that of the top comb order, from a high percentile of the
    track's slope over its credible frames (slope fitted over at least one
    second); the result is clipped to ``[_WINDOW_MIN, _WINDOW_MAX]`` and kept
    long enough for the closest comb orders to stay apart (``4 / T`` below
    their spacing) at a low percentile of the speed. With fewer than a quarter
    of the frames credible the longest window is kept.
    """
    if credible.mean() < 0.25 or rpm.size < 3:
        return _WINDOW_MAX
    fit = max(win_s, _WINDOW_FIT)
    slope = (_path_rate(rpm, hop_s, fit) * rpm)[credible]  # |d rpm/dt|, rpm/s
    u = np.unique(ords)
    gap = float(np.min(np.diff(u))) if u.size > 1 else float(u[0])
    r_low = float(np.percentile(rpm[credible], _WINDOW_LOW_PERCENTILE))
    t_res = _LINE_BINS / (gap * r_low / 60.0)
    fdot = float(u.max()) * float(np.percentile(slope, _WINDOW_PERCENTILE)) / 60.0
    t_opt = 2.0 / np.sqrt(fdot) if fdot > 0 else np.inf
    return float(min(_WINDOW_MAX, max(t_opt, _WINDOW_MIN, t_res)))


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


def _quality(rpm, conf, at_bound, octave, window):
    """Summary of a whole track (``RidgeResult.quality``); see there."""
    n_fr = rpm.size
    lr = np.log(rpm)
    if n_fr >= 3:
        local = ndimage.median_filter(lr, size=min(_Q_JUMP_SPAN, n_fr) | 1, mode="nearest")
        jump = float(np.mean(np.abs(lr - local) > _Q_JUMP))
    else:
        jump = 0.0
    med_conf = float(np.median(conf))
    low_frac = float(np.mean(conf < _Q_LOW_CONFIDENCE))
    bound_frac = float(np.mean(at_bound))
    octave_arr = np.asarray(octave, dtype=float)
    finite = np.isfinite(octave_arr)
    octave_amb = bool(finite.any() and np.max(octave_arr[finite]) >= _Q_OCTAVE)
    problems = []
    if med_conf < _Q_MIN_CONFIDENCE:
        problems.append(f"median confidence {med_conf:.2f} < {_Q_MIN_CONFIDENCE:g}")
    if low_frac > _Q_MAX_LOW:
        problems.append(
            f"{100 * low_frac:.0f} % of the frames with confidence below "
            f"{_Q_LOW_CONFIDENCE:g} (> {100 * _Q_MAX_LOW:.0f} %)"
        )
    if bound_frac > _Q_MAX_BOUND:
        problems.append(
            f"{100 * bound_frac:.0f} % of the frames at a bound (> {100 * _Q_MAX_BOUND:.0f} %)"
        )
    if jump > _Q_MAX_JUMP:
        problems.append(
            f"{100 * jump:.0f} % of the frames jump by more than {100 * _Q_JUMP:.0f} % "
            f"(> {100 * _Q_MAX_JUMP:.0f} %)"
        )
    if octave_amb:
        problems.append(
            f"an octave of the track scores {np.max(octave_arr[finite]):.2f} of the track "
            f"(half speed {octave_arr[0]:.2f}, double speed {octave_arr[1]:.2f}; "
            f">= {_Q_OCTAVE:g})"
        )
    return dict(
        median_confidence=med_conf,
        low_confidence_fraction=low_frac,
        bound_fraction=bound_frac,
        jump_fraction=jump,
        octave_ratio=tuple(float(v) for v in octave_arr),
        octave_ambiguous=octave_amb,
        window=float(window),
        reliable=not problems,
        problems=problems,
    )


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
    window="auto",
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

    1. Each channel is decimated (polyphase low-pass) to the lowest rate
       whose Nyquist frequency is at least 1.25 times the highest comb
       frequency ``max(orders) * rpm_range[1] / 60``.
    2. A Hann-window STFT (``window`` s, step ``hop`` s, zero-padded 4x)
       gives the power spectrum of each frame, averaged over the channels
       (incoherent averaging: the channels need not be in phase, and their
       noise averages down in power). It is whitened: divided by its
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
    6. Range check (ROADMAP T10): the comb is also scored in guard bands up
       to a factor 2 beyond either end of ``rpm_range`` and the path is
       tracked a second time over that extended grid. Frames whose path ends
       within one grid step of an end, or where the unconstrained path lies
       outside the range (``beyond``), are flagged in ``at_bound``.
    7. Confidence per frame (detection above the noise times distinctness
       from rival speeds) and a summary ``quality`` of the whole track; see
       :class:`RidgeResult`.

    With ``window="auto"`` (default) steps 2 to 7 run first at a 0.4 s
    window; if the sweep of that track calls for a shorter window (Notes,
    "Window"), they run again at the shorter one, with the whitening width
    set from the first track.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal in any unit (the estimator is scale invariant). The power
        spectra of the channels are averaged (step 2): two ear or microphone
        signals, nearly uncorrelated at high frequency, add their evidence
        instead of cancelling.
    fs : float
        Sampling rate in Hz.
    rpm_range : (float, float)
        Search range ``(lo, hi)`` in rpm, ``0 < lo < hi``. The comb
        ``max(orders) * hi / 60`` must stay below ``fs / 2``.
    orders : sequence of float, default (1, 2, 3, 4)
        Comb orders (cycles per revolution) of the shaft whose speed is
        returned. They may be fractional. Give the orders the machine
        actually shows: the comb, not the data, decides between a speed and
        its octaves (Notes, "Subharmonic ambiguity"). For a four-stroke
        engine with ``N`` cylinders the firing order ``N / 2`` and its
        harmonics dominate: ``(2, 4, 6)`` for four cylinders (the default
        comb then reads twice the crankshaft speed), ``(1.5, 3, 4.5)`` for
        three, ``(3, 6, 9)`` for six; for an electric machine the pole-pair
        or slot harmonics, for a gearbox the mesh orders. The default suits a
        shaft with a visible first order.
    weights : sequence of float, optional
        One non-negative weight per order; normalised to sum 1. Default
        ``order ** -0.5``: lower orders count more. This breaks the tie of a
        single visible order with its sub-harmonics (speed / 2, / 3, / 4 put
        a higher comb tooth on that order: with equal weights all of them
        score alike), as in subharmonic summation (Hermes 1988). The price
        is a bias towards twice the speed when only the even orders are
        present (see Notes).
    window : "auto" or float, default "auto"
        STFT window length in s. Frequency resolution is about ``1.5 /
        window`` Hz (Hann main lobe). ``"auto"``: 0.4 s, shortened for fast
        sweeps to ``2 / sqrt(fdot)`` (``fdot`` the sweep rate of the top comb
        order in Hz/s) but not below 0.1 s or the resolution of the closest
        comb orders (Notes, "Window"); the value used is in the result as
        ``window``.
    hop : float, default 0.05
        Frame step in s.
    whiten : "auto", float or None, default "auto"
        Width of the running median along frequency that whitens the
        spectrum. ``"auto"``: chosen per frame and growing with frequency,
        ``max(6 / window, 4 / window + rate * window * f)`` Hz at frequency
        ``f`` (``rate`` is the relative speed rate ``|d rpm/dt| / rpm`` of
        the first-pass path), i.e. 15 Hz for a 0.4 s window wherever the
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
        ``whiten_violated``, ``window``, ``quality`` and, if ``t_out`` was
        given, ``rpm_out``. A ``RuntimeWarning`` is issued when
        ``bound_fraction > 0``: the estimator never returns a boundary value
        silently.

    Warns
    -----
    RuntimeWarning
        If any frame has ``at_bound`` set (search-range warning).
    WhiteningWarning
        (A subclass of ``RuntimeWarning``.) If on at least 5 % of the frames
        the running median is narrower than 0.8 times the frequency a comb
        line occupies within the window (``whiten_violated``). It names the
        width, the line extent and the worst time; ``confidence`` is lowered
        on those frames. This is not about the search range: if both warnings
        appear and the speed is believed to be in range, fix the whitening
        first.
    RidgeQualityWarning
        (A subclass of ``RuntimeWarning``.) If ``quality["reliable"]`` is
        False: the track as a whole is likely wrong (median confidence below
        0.2, more than 40 % of the frames below 0.1, more than 25 % at a
        bound, more than 10 % jumping, or an octave scoring at least as high
        as the track). It names the failed criteria.

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
      resolve spectra with only even orders, nor half-order machines: for the
      half orders of a three-cylinder four-stroke engine an integer comb
      locks at exactly 1.5 times the speed. The even-order case cannot be
      resolved by any test on the data: the orders 2, 4, 6, 8 of a
      four-cylinder four-stroke engine at speed ``r`` (firing order 2
      dominant, odd and half orders at the noise floor) are, line for line,
      the orders 1, 2, 3, 4 of a machine at ``2 r``, so the default comb
      reads twice the crankshaft speed, confidently and smoothly (measured
      on synthetic interior-noise-like part-load ramps, coasts and cruises:
      0 % of the frames within 3 %, ratio 2.00; with ``(2, 4, 6)``: 100 %). Give
      the comb of the machine (``orders`` above) and set ``rpm_range`` from
      a priori knowledge (idle speed, redline): where twice the speed leaves
      the range, ``beyond`` flags it. ``quality["octave_ratio"]`` reports
      how high half and twice the tracked speed score relative to the track
      (a correct even-comb track reads about 0.3 and 0.7); it flags
      (``octave_ambiguous``) only an octave that scores at least as high as
      the track itself.
    * Window. A line sweeping ``fdot`` Hz/s occupies ``4 / T + fdot T`` Hz in
      a Hann window of ``T`` s; the extent is least, and the line highest
      above the floor, at ``T = 2 / sqrt(fdot)`` (the matched window of a
      chirp in the spectrogram; Cohen 1995, Kwok & Jones 2000). The order 6
      of a run-up at 850 rpm/s sweeps 85 Hz/s at any speed, so ``T = 0.22``
      s, against 34 Hz of sweep in the former fixed 0.4 s. ``window="auto"``
      takes ``fdot`` of the top comb order from the 75th percentile of the
      slope of a first track at 0.4 s (credible frames only, slope fitted
      over 1 s), clips ``T`` to [0.1, 0.4] s and to ``4 / T`` below the
      spacing of the closest comb orders at the 10th-percentile speed, and
      tracks again if ``T`` is more than 5 % shorter (up to three times). A
      track locked on a sub-multiple of a fast sweep reads too slow a slope;
      if the result is still unreliable, windows 0.7, 0.49, ... times as
      long are tried and the most reliable track is kept. Measured on a
      synthetic four-cylinder-like run-up at 850 rpm/s from 1200 rpm with a
      45 Hz boom and road noise 10 dB above the orders (RMS): 0.4 s put 99 %
      of the frames within 3 % but at a median confidence of 0.03, with a
      warning; ``"auto"`` (0.22 s) 100 % at 1.0, without; on the half-order
      run-ups of the whitening notes the 2000 rpm/s case, which fails at
      0.4 s, is tracked at 0.22 s (100 % within 3 %). Slow sweeps and steady
      speed keep 0.4 s, so their results are those of the former default.
    * Confidence and quality. The detection term compares the tracked score
      with the distribution of a noise-only comb score, known in closed
      form (the whitened level of a noise bin is that of an exponential
      power over its median, ``chi^2`` with 2 degrees of freedom; Kay 1993):
      unlike the spread of the frame's own scores it does not shrink when
      the spectrum holds many lines. The distinctness term counts every
      rival peak (rational neighbours such as 7/6 or 5/4 included: a
      single-order comb that has fallen onto a neighbouring line reads low,
      because the true line is a rival) except the sub-multiples 1/2, 1/3,
      1/4 of the tracked speed, at which every comb on the harmonics of the
      truth scores in part. Measured (8 kHz): clean correct tracks 0.7 to 1
      (median), white or steeply coloured noise only 0, an untrackable
      cruise (orders 30 dB below the noise, RMS) 0. The quality limits are
      those observed to separate failed from correct tracks; a fixed tone
      that a single-order comb locks onto is not detected (it is a strong,
      steady line): use ``moving=True``.
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
    Cohen, L. (1995). Time-Frequency Analysis. Prentice Hall.
    Kwok, H. K., & Jones, D. L. (2000). Improved instantaneous frequency
    estimation using an adaptive short-time Fourier transform. IEEE
    Transactions on Signal Processing, 48(10), 2964-2972.
    Kay, S. M. (1993). Fundamentals of Statistical Signal Processing, Vol. 2:
    Detection Theory. Prentice Hall.
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
    auto_window = isinstance(window, str)
    if auto_window:
        if window.strip().lower() != "auto":
            raise ValueError("window must be 'auto' or a positive length in s")
    else:
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

    # ---- decimation of each channel ----------------------------------------
    q = max(1, int(np.floor(0.5 * fs / (_NYQUIST_MARGIN * f_top_guard))))
    Y = signal.resample_poly(X, 1, q, axis=0) if q > 1 else X
    fs_d = fs / q

    def default_grid(win_len):
        return float(np.clip(_GRID_FRACTION**-1 * 1.44 / (win_len * f_top), *_GRID_LIMITS))

    def grid_layout(g, n_frames):
        ln_ratio = np.log(hi / lo)
        n_g = max(3, int(np.ceil(ln_ratio / np.log1p(g))) + 1)
        stp = ln_ratio / (n_g - 1)  # logarithmic step, rpm_grid[-1] == hi
        n_lo = int(np.floor(np.log(_GUARD) / stp))
        n_hi = int(np.floor(np.log(hi_guard / hi) / stp + 1e-9))
        if n_frames * (n_lo + n_g + n_hi) > _MAX_SCORE_SIZE:
            raise ValueError(
                f"score matrix of {n_frames} frames x {n_lo + n_g + n_hi} speeds is too "
                "large; increase grid or hop, or shorten the signal"
            )
        return n_g, stp, n_lo, n_hi

    grid_layout(default_grid(_WINDOW_MAX if auto_window else window) if grid is None else grid,
                _stft_frames(Y, fs_d, _WINDOW_MAX if auto_window else window, hop)["starts"].size)

    def track(window_used, hint):
        """Whole estimate at one window; ``hint`` = (t, rpm, credible) of an
        earlier track (window="auto"), which then replaces the first pass."""
        st = _stft_frames(Y, fs_d, window_used, hop)
        starts, n_win, nfft, df = st["starts"], st["n_win"], st["nfft"], st["df"]
        win, frames_all, win_s, hop_s = st["win"], st["frames"], st["win_s"], st["hop_s"]
        n_fr = starts.size
        nb_max = nfft // 2 + 1
        t_fr = (starts + 0.5 * (n_win - 1)) / fs_d

        # ---- speed grid (checked before any large array is built) ----------
        grid_used = default_grid(window_used) if grid is None else grid
        n_grid, step, n_low, n_up = grid_layout(grid_used, n_fr)

        # ---- width of the running median: w0 + q f per frame (w0 = 0: frame
        # median). The first-pass path sets the automatic width and checks any
        # width; an earlier track (hint) stands in for it when given.
        path1 = z1 = None
        w1 = 0.0
        if whiten is None and not auto:
            w0, q_f = np.zeros(n_fr), np.zeros(n_fr)
        else:
            if hint is None:
                path1, z1 = _first_pass_path(
                    frames_all, starts, win, nfft, df, ords, w, (lo, hi), grid_used, moving,
                    f_top,
                )
            else:
                path1 = np.interp(t_fr, hint[0], hint[1])
                z1 = np.where(np.interp(t_fr, hint[0], hint[2].astype(float)) > 0.5, np.inf, 0.0)
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

        # ---- comb score on a logarithmic speed grid -------------------------
        rpm_ext = lo * np.exp(step * np.arange(-n_low, n_grid + n_up))
        rpm_ext[n_low + n_grid - 1] = hi
        S_ext = _comb_scores(level, df, rpm_ext, ords, w)
        del level
        rpm_grid = rpm_ext[n_low : n_low + n_grid].copy()
        S = np.ascontiguousarray(S_ext[:, n_low : n_low + n_grid])

        # ---- path -----------------------------------------------------------
        pen = penalty
        if pen is None:
            prominence = S.max(axis=1) - np.median(S, axis=1)
            pen = float(np.median(prominence)) / _PENALTY_DIVISOR
        cost = pen * 100.0 * step  # score units per grid step
        path = _viterbi(S, cost)
        # Out-of-range evidence: the same path search over the grid extended by
        # the guard bands; where it leaves the range, the data prefer a speed
        # outside it.
        if n_low or n_up:
            p_ext = _viterbi(S_ext, cost) - n_low
            beyond = (p_ext < -1) | (p_ext > n_grid)
        else:
            beyond = np.zeros(n_fr, dtype=bool)

        # ---- refinement, flags, confidence ----------------------------------
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
        at_bound = (path <= 1) | (path >= n_grid - 2) | beyond

        z, s_med = _detection_z(S, s_sel, w)
        detect = np.clip((z - _Z0) / (_Z1 - _Z0), 0.0, 1.0)
        half = int(np.ceil(np.log1p(_CONF_EXCLUDE) / step))
        n_oct = int(round(np.log(2.0) / step))
        centres = [path] + [path - int(round(np.log(k) / step)) for k in _CONF_HARMONIC]
        s_alt = _rival_max(S, centres, half)
        s_alt = np.where(np.isfinite(s_alt), s_alt, s_med)
        height = s_sel - s_med
        with np.errstate(divide="ignore", invalid="ignore"):
            distinct = np.where(height > 1e-9, (s_sel - s_alt) / height, 0.0)
        conf = detect * np.clip(distinct / _D1, 0.0, 1.0)

        # Octave check: the score at half and at twice the tracked speed (from
        # the guard-extended grid), as heights above the frame median relative
        # to that of the track, on the frames where the track is detected.
        octave = []
        for i_k in (n_low + path - n_oct, n_low + path + n_oct):
            ok = (i_k >= 0) & (i_k < rpm_ext.size) & (detect > 0) & (height > 1e-9)
            if ok.any():
                h_k = S_ext[rows[ok], i_k[ok]] - s_med[ok]
                octave.append(float(np.median(h_k / height[ok])))
            else:
                octave.append(float("nan"))
        del S_ext

        # Whitening check: does the running median that was used keep the
        # lines? Width and line extent are both linear in frequency, so the
        # highest comb order is the binding one. The extent comes from the
        # sweep of the final path and from that of the first-pass path, which
        # does not depend on the width: when the width is too small the final
        # path is itself wrong and its slope with it. Each counts only where
        # its path is credible (detected and, for the final path, not
        # at_bound): the slope of a path through noise is meaningless. The
        # first-pass extent is median-filtered over 2.5 windows (it has spikes
        # where that path is wrong, e.g. in coloured noise).
        line_final = _line_extent(rpm, hop_s, win_s, max_order)
        cred_final = ~at_bound & (detect > 0)
        if path1 is not None:
            span = max(3, 2 * round(1.25 * win_s / hop_s) + 1)
            line1 = ndimage.median_filter(
                _line_extent(path1, hop_s, win_s, max_order), size=span, mode="nearest"
            )
            line = np.maximum(np.where(z1 > _Z0, line1, 0.0), np.where(cred_final, line_final, 0.0))
            credible = cred_final | (z1 > _Z0)
            # where neither path is credible, report the final path's extent
            line = np.where(credible, line, line_final)
        else:
            line = line_final
            credible = cred_final
        width_top = np.where(w0 > 0, np.maximum(w0, w1 + q_f * max_order * rpm / 60.0), np.inf)
        violated = (width_top < _VIOLATION_RATIO * line) & credible
        if violated.any():
            ratio = width_top[violated] / (_VIOLATION_RATIO * line[violated])
            conf[violated] *= np.minimum(1.0, ratio) ** 2
        return dict(
            t=t_fr, rpm=rpm, conf=conf, detect=detect, at_bound=at_bound, beyond=beyond,
            rpm_grid=rpm_grid, S=S, penalty=float(pen), rpm_ext=rpm_ext, width_top=width_top,
            line=line, violated=violated, octave=octave, cred=cred_final, hop_s=hop_s,
            win_s=win_s, window=float(window_used),
        )

    # ---- window="auto": a track at the longest window, then, if its sweep
    # calls for a shorter one, the final track at that window (its whitening
    # set by the first track, which is more robust in coloured noise than a
    # first pass without a running median).
    if auto_window:

        def merit(o):
            q_o = _quality(o["rpm"], o["conf"], o["at_bound"], o["octave"], o["window"])
            m = q_o["median_confidence"] * (1 - q_o["jump_fraction"]) * (1 - q_o["bound_fraction"])
            return (q_o["reliable"], m)

        out = track(_WINDOW_MAX, None)
        for _ in range(_WINDOW_ITER):
            chosen = _auto_window(out["rpm"], out["cred"], out["hop_s"], out["win_s"], ords)
            if chosen > out["window"] * (1.0 - _WINDOW_TOL):
                break  # only ever shortened: a track at a longer window was had
            out = track(chosen, (out["t"], out["rpm"], out["cred"]))
        # A track that is still unreliable may follow a sub-harmonic of a sweep
        # too fast for its window, whose slope then reads too low: try
        # shorter windows and keep the best track.
        best, m_best = out, merit(out)
        trial = out
        while not m_best[0] and trial["window"] * _WINDOW_SHRINK >= _WINDOW_MIN * (1 - 1e-9):
            trial = track(trial["window"] * _WINDOW_SHRINK, (out["t"], out["rpm"], out["cred"]))
            m_trial = merit(trial)
            if m_trial > m_best:
                best, m_best = trial, m_trial
        if best is not out:  # whiten once more from the better track
            again = track(best["window"], (best["t"], best["rpm"], best["cred"]))
            best = again if merit(again) >= m_best else best
        out = best
    else:
        out = track(window, None)
    rpm, conf, at_bound, violated = out["rpm"], out["conf"], out["at_bound"], out["violated"]
    line, width_top, octave, window_used = out["line"], out["width_top"], out["octave"], out["window"]
    n_fr = rpm.size

    quality = _quality(rpm, conf, at_bound, octave, window_used)
    problems = quality["problems"]

    res = RidgeResult(
        t=out["t"],
        rpm=rpm,
        confidence=conf,
        at_bound=at_bound,
        beyond=out["beyond"],
        rpm_grid=out["rpm_grid"],
        score=out["S"],
        orders=ords,
        weights=w,
        penalty=out["penalty"],
        rpm_range=(lo, hi),
        guard_range=(float(out["rpm_ext"][0]), float(out["rpm_ext"][-1])),
        whiten_width=width_top,
        line_width=line,
        whiten_violated=violated,
        window=float(window_used),
        quality=quality,
    )
    if t_out is not None:
        res.t_out = t_out
        res.rpm_out = res.rpm_at(t_out)
    warn_whiten = bool(violated.mean() >= _WHITEN_WARN_FRACTION)
    if warn_whiten:
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
        if warn_whiten:
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
    if problems:
        warnings.warn(
            "rpm_from_ridge: the track is likely unreliable: " + "; ".join(problems) + ". "
            "Check that orders match the machine (four-stroke engine with N cylinders: "
            "orders=(N/2, N, 3N/2), e.g. (2, 4, 6) for four cylinders; see the Notes), "
            "the window, the search range and RidgeResult.confidence.",
            RidgeQualityWarning,
            stacklevel=2,
        )
    return res
