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

__all__ = ["RidgeResult", "rpm_from_ridge"]

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
        and typically hold the value below 1.
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
    whiten=15.0,
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
       running median along frequency over ``whiten`` Hz, then expressed in dB
       and floored at 0 dB, so the noise floor reads about 0 dB and a
       spectral line reads its height above the local floor. With
       ``moving=True`` each bin's median over time is also subtracted (before
       the floor), which removes lines at fixed frequencies.
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
    whiten : float or None, default 15.0
        Width in Hz of the running median along frequency. It must be larger
        than the width a spectral line occupies, about ``4 / window`` plus
        the frequency the highest order sweeps during a window,
        ``max(orders) * |d rpm/dt| / 60 * window``; and the lines must cover
        less than half of the width, i.e. harmonic spacing (``rpm_range[0] /
        60``) well above ``4 / window``. Otherwise whitening erases the
        orders (very low idle speed or fast run-ups: use a longer window or a
        larger ``whiten``). None disables whitening: levels are then relative
        to the frame's median over all bins.
    moving : bool, default False
        Subtract each frequency bin's median over time, to suppress lines at
        fixed frequency (resonances, tones, mains hum). Valid only when the
        speed sweeps: a bin must be hit by an order for less than half of the
        frames, so at near-constant speed this removes the orders too.
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
        :meth:`~RidgeResult.rpm_at` and, if ``t_out`` was given, ``rpm_out``.
        A ``RuntimeWarning`` is issued when ``bound_fraction > 0``: the
        estimator never returns a boundary value silently.

    Warns
    -----
    RuntimeWarning
        If any frame has ``at_bound`` set.

    Notes
    -----
    * The estimate is scale invariant; ``x`` has no amplitude convention.
      Output speeds are in rpm.
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
    if whiten is not None:
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
    f_keep = f_top_guard + (0.5 * whiten if whiten else 0.0)
    nb = min(nfft // 2 + 1, int(np.ceil(f_keep / df)) + 3)
    width_bins = max(3, round(whiten / df)) if whiten else 0

    win = signal.get_window("hann", n_win)
    frames_all = np.lib.stride_tricks.sliding_window_view(y, n_win)
    level = np.empty((n_fr, nb), dtype=np.float32)
    chunk = max(1, int(2**22 // nfft))
    for a in range(0, n_fr, chunk):
        fr = frames_all[starts[a : a + chunk]]
        fr = (fr - fr.mean(axis=1, keepdims=True)) * win
        F = np.fft.rfft(fr, n=nfft, axis=1)[:, :nb]
        P = F.real**2 + F.imag**2
        if width_bins:
            med = _running_median_freq(P, width_bins)
        else:
            med = np.median(P, axis=1, keepdims=True)
        ratio = P / np.maximum(med, np.finfo(float).tiny)
        level[a : a + chunk] = 10.0 * np.log10(np.maximum(ratio, 1e-12))
    if moving:
        level -= np.median(level, axis=0).astype(np.float32)
    np.maximum(level, 0.0, out=level)

    # ---- comb score on a logarithmic speed grid ---------------------------
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
    rpm_ext = lo * np.exp(step * np.arange(-n_low, n_grid + n_up))
    rpm_ext[n_low + n_grid - 1] = hi
    S_ext = np.zeros((n_fr, rpm_ext.size))
    for m, wm in zip(ords, w):
        if wm == 0.0:
            continue
        pos = m * rpm_ext / 60.0 / df
        i0 = np.minimum(pos.astype(int), nb - 2)
        fr = pos - i0
        S_ext += wm * (level[:, i0] * (1.0 - fr) + level[:, i0 + 1] * fr)
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
    )
    if t_out is not None:
        res.t_out = t_out
        res.rpm_out = res.rpm_at(t_out)
    frac = res.bound_fraction
    if frac > 0:
        warnings.warn(
            f"rpm_from_ridge: {100.0 * frac:.1f} % of the frames sit at or beyond a "
            f"bound of the search range ({lo:g}, {hi:g}) rpm; the true speed may be "
            "outside it (or the data too noisy to tell: see confidence). Widen "
            "rpm_range and check RidgeResult.at_bound.",
            RuntimeWarning,
            stacklevel=2,
        )
    return res
