"""Phase-locked short-time harmonic extraction (angle-domain frames).

Units: time in s, rotational speed in rpm, shaft angle in revolutions, order
in cycles per revolution. Amplitudes are *peak* amplitudes in the unit of the
analysed signal.
"""

import warnings

import numpy as np
from scipy import sparse

from ._components import OrderComponents
from ._utils import as_2d
from .speed import rpm_to_phase

__all__ = ["tracked_harmonics"]

# Samples handled per block. Bounds the size of the temporaries; module level so that
# the tests can shrink it and check that the result does not depend on the blocking.
_CHUNK = 1 << 15

# Cosine-sum windows w(u) = sum_p a_p cos(2 pi p u) on |u| <= 1/2 (zero or nearly
# zero at the ends) and the half width of their main lobe in bins of 1/window_revs
# orders (Harris 1978).
_COSINE_WINDOWS = {
    "hann": ((0.5, 0.5), 2.0),
    "hamming": ((0.54, 0.46), 2.0),
    "blackman": ((0.42, 0.5, 0.08), 3.0),
    "blackmanharris": ((0.35875, 0.48829, 0.14128, 0.01168), 4.0),
}
_WINDOWS = (*_COSINE_WINDOWS, "bartlett", "boxcar")
_HALF_WIDTH = {**{k: v[1] for k, v in _COSINE_WINDOWS.items()}, "bartlett": 2.0, "boxcar": 1.0}

# Significance test: _PROBE_COUNT probe orders on each side of the tested order, the
# first one bin beyond the main lobe, spaced by _PROBE_STEP bins (1 bin = 1/window_revs
# orders). A probe is used in a frame only if at least _MIN_COVERAGE of the frame's
# angle weight is below the Nyquist frequency for the probe.
_PROBE_COUNT = 4
_PROBE_STEP = 1.5
_MIN_COVERAGE = 0.5


def _window_function(name):
    """Return ``w(u)`` for ``u = (angle - centre) / window_revs`` (support ``|u| < 1/2``)."""
    if name in _COSINE_WINDOWS:
        a = _COSINE_WINDOWS[name][0]

        def w(u):
            out = np.full(u.shape, a[0])
            for p in range(1, len(a)):
                out += a[p] * np.cos(2.0 * np.pi * p * u)
            return out

        return w
    if name == "bartlett":
        return lambda u: 1.0 - 2.0 * np.abs(u)
    return lambda u: np.ones(u.shape)


def _angle_increments(rev):
    """Trapezoid-rule weights of the samples in angle (they sum to the total angle)."""
    d = np.empty(rev.shape)
    d[1:-1] = 0.5 * (rev[2:] - rev[:-2])
    d[0] = 0.5 * (rev[1] - rev[0])
    d[-1] = 0.5 * (rev[-1] - rev[-2])
    return d


class _Frames:
    """Frames of one shaft, uniformly spaced in shaft angle.

    Frame ``j`` is centred at ``j * hop`` revolutions after the first sample, for
    ``j = 0 .. n_frames - 1`` (every centre lies inside the record), and weights a
    sample at angle ``rho`` by ``w((rho - j * hop) / window_revs)``.
    """

    def __init__(self, rev, window_revs, hop_revs, window):
        self.rel = rev - rev[0]
        self.delta = _angle_increments(self.rel)
        self.span = float(window_revs)
        self.hop = float(hop_revs)
        self.n_frames = int(np.floor(self.rel[-1] / self.hop + 1e-9)) + 1
        self.m = self.span / self.hop  # frames overlapping an angle
        self.n_over = int(np.ceil(self.m - 1e-9))
        self.window = _window_function(window)

    def block(self, a, b):
        """Frame indices ``(nc, n_over)`` and window weights of samples ``a:b``.

        Entries that fall outside the support or outside the existing frames have
        weight 0 (their index is clipped to a valid frame).
        """
        r = self.rel[a:b]
        j0 = np.floor(r / self.hop - 0.5 * self.m).astype(np.int64) + 1
        j = j0[:, None] + np.arange(self.n_over)
        u = (r[:, None] - j * self.hop) / self.span
        ok = (np.abs(u) < 0.5) & (j >= 0) & (j < self.n_frames)
        w = np.where(ok, self.window(u), 0.0)
        return np.clip(j, 0, self.n_frames - 1).astype(np.int32), w


def _sparse(jj, data, n_frames):
    """CSR matrix (samples x frames) with ``n_over`` stored entries per row."""
    nc, k = jj.shape
    indptr = np.arange(0, nc * k + 1, k, dtype=np.int32)
    return sparse.csr_matrix((data.ravel(), jj.ravel(), indptr), shape=(nc, n_frames))


def _frame_coefficients(X, rev, speed, fs, frames, orders):
    """Complex peak amplitude of each order in each frame.

    Returns ``c`` of shape ``(n_frames, L, ch)`` and the coverage ``cov`` of shape
    ``(n_frames, L)``: the fraction of the frame's angle weight that lies below the
    Nyquist frequency for that order (1 unless the order crosses ``fs / 2``).
    """
    n, ch = X.shape
    L = orders.size
    J = frames.n_frames
    nyq = 0.5 * fs
    muted = orders * float(speed.max()) / 60.0 >= nyq  # orders that cross Nyquist somewhere
    acc = np.zeros((J, 2 * L * ch))
    den = np.zeros(J)
    den_m = np.zeros((J, int(muted.sum())))
    for a in range(0, n, _CHUNK):
        b = min(a + _CHUNK, n)
        jj, w = frames.block(a, b)
        wd = w * frames.delta[a:b, None]
        flat = jj.ravel()
        den += np.bincount(flat, weights=wd.ravel(), minlength=J)
        z = X[a:b, None, :] * np.exp(-2j * np.pi * (rev[a:b, None] * orders))[:, :, None]
        if muted.any():
            ok = orders * speed[a:b, None] / 60.0 < nyq
            z *= ok[:, :, None]
            for i, col in enumerate(np.flatnonzero(muted)):
                den_m[:, i] += np.bincount(
                    flat, weights=(wd * ok[:, col, None]).ravel(), minlength=J
                )
        acc += _sparse(jj, wd, J).T @ z.reshape(b - a, L * ch).view(np.float64)
    num = acc.view(np.complex128).reshape(J, L, ch)
    den_l = np.repeat(den[:, None], L, axis=1)
    den_l[:, muted] = den_m
    live = den_l > 0
    c = np.zeros_like(num)
    np.divide(2.0 * num, den_l[:, :, None], out=c, where=live[:, :, None])
    cov = np.zeros((J, L))
    np.divide(den_l, den[:, None], out=cov, where=den[:, None] > 0)
    return c, cov


def _synthesise(C, frames, rev, speed, fs, orders, env, wav, cols):
    """Normalised overlap-add of the frame coefficients into ``env`` and ``wav``."""
    n, ch = env.shape[0], C.shape[2]
    L = orders.size
    J = frames.n_frames
    Cf = np.ascontiguousarray(C.reshape(J, L * ch)).view(np.float64)
    for a in range(0, n, _CHUNK):
        b = min(a + _CHUNK, n)
        jj, w = frames.block(a, b)
        total = w.sum(axis=1)
        scale = np.divide(1.0, total, out=np.zeros_like(total), where=total > 1e-12)
        e = (_sparse(jj, w, J) @ Cf).view(np.complex128).reshape(b - a, L, ch)
        e *= scale[:, None, None]
        e *= (orders * speed[a:b, None] / 60.0 < 0.5 * fs)[:, :, None]
        carrier = np.exp(2j * np.pi * (rev[a:b, None] * orders))
        env[a:b, cols] = e
        wav[a:b, cols] = (e * carrier[:, :, None]).real


def _plan_probes(orders, half_width, window_revs):
    """Probe orders for the significance test.

    For each order, up to ``2 * _PROBE_COUNT`` orders on both sides, starting one
    bin beyond the main lobe of the window. A probe is dropped if it is within
    the main lobe (plus half a bin) of any requested order or of order 0, or is
    not positive. Returns the sorted unique probe orders and, per requested
    order, the indices of its probes in that array.
    """
    unit = 1.0 / window_revs
    guard = (half_width + 0.5) * unit
    offs = (half_width + 1.0 + _PROBE_STEP * np.arange(_PROBE_COUNT)) * unit
    cand = np.concatenate((orders[:, None] - offs, orders[:, None] + offs), axis=1)
    ok = cand > guard
    near = np.abs(cand[:, :, None] - orders[None, None, :]) < guard
    ok &= ~near.any(axis=2)
    probes = np.unique(np.round(cand[ok], 9))
    index = [np.searchsorted(probes, np.round(cand[k][ok[k]], 9)) for k in range(orders.size)]
    return probes, index


def _significance_gate(C, P, cov, index, p, orders):
    """Zero the coefficients that are not significant against their probes.

    ``C`` is ``(J, K, ch)``, ``P`` the probe coefficients ``(J, Lp, ch)`` and ``cov``
    their coverage ``(J, Lp)``. Under the null hypothesis (noise only) ``|c|^2`` over
    the mean probe power of ``M`` probes follows an F(2, 2M) law with the survival
    function ``(1 + x / M) ** -M``; ``C`` is modified in place. Returns the fraction
    of coefficients kept per order (1 for orders that cannot be tested).
    """
    power = np.abs(P) ** 2
    kept = np.ones(C.shape[1])
    for k, idx in enumerate(index):
        if len(idx) < 2:
            warnings.warn(
                f"significance: order {orders[k]:g} has fewer than 2 probe orders clear of "
                "the other requested orders; it is left ungated",
                stacklevel=3,
            )
            continue
        good = cov[:, idx] >= _MIN_COVERAGE  # (J, m)
        count = good.sum(axis=1)  # (J,)
        noise = (power[:, idx, :] * good[:, :, None]).sum(axis=1) / np.maximum(count, 1)[:, None]
        m = np.maximum(count, 1).astype(float)
        threshold = m * ((1.0 - p) ** (-1.0 / m) - 1.0)
        keep = (np.abs(C[:, k, :]) ** 2 > threshold[:, None] * noise) | (count < 2)[:, None]
        C[:, k, :] *= keep
        kept[k] = keep.mean()
    return kept


def tracked_harmonics(
    x,
    fs,
    rpm,
    orders,
    *,
    window_revs=8.0,
    overlap=0.75,
    window="hann",
    shafts=None,
    delay=0.0,
    significance=None,
):
    """Extract order waveforms by phase-locked demodulation over angle-domain frames.

    For each order ``k`` the signal is multiplied by the conjugate carrier
    ``exp(-1j * theta_k)``, ``theta_k = 2 pi order_k rev(t)``, with the shaft angle
    ``rev`` from :func:`rota_order.speed.rpm_to_phase`. Whatever the speed does, a
    component locked to the shaft becomes a slowly varying (complex) constant,
    and everything else is moved away from zero frequency. The slowly varying part
    is then estimated by windowed averaging over frames that are *uniform in shaft
    angle*, not in time:

    * Frame ``j`` is centred at ``rho_j = j * hop_revs`` revolutions, with
      ``hop_revs = window_revs * (1 - overlap)``, and weights a sample at angle
      ``rev(n)`` by ``w((rev(n) - rho_j) / window_revs)``.
    * The frame coefficient is an integral over angle,
      ``c_kj = 2 sum_n z_k(n) w_j(n) d(n) / sum_n w_j(n) d(n)``, where ``d(n)`` is the
      angle increment of sample ``n`` (trapezoid weights). The factor 2 gives the
      peak amplitude of a real component.
    * The envelope is the normalised overlap-add
      ``c_k(n) = sum_j c_kj w_j(n) / sum_j w_j(n)`` (for a periodic Hann window
      at 75 % overlap the denominator is constant), and the waveform is
      ``Re{c_k(n) exp(1j theta_k(n))}``.

    Because the frame length is a number of revolutions, the order resolution is
    the same at every speed: the main lobe of a Hann window is
    ``+-2 / window_revs`` orders wide at 800 rpm as at 4000 rpm, whereas a
    time-domain window of fixed duration (fixed bandwidth in Hz) spans several
    orders at idle. Orders closer than about ``2 / window_revs`` cannot be
    separated; with ``window_revs = 8`` orders 0.5 apart are separated (the
    leakage of one into the other is a null for stationary components). The price
    is time resolution: amplitude changes faster than about one cycle per
    ``window_revs`` revolutions are smoothed. Phase variations that are due to the
    speed itself are demodulated exactly and cost nothing.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal sampled at ``fs``, in any unit. All channels use the same
        frame weights and are processed independently and linearly.
    fs : float
        Sampling rate in Hz.
    rpm : array_like, shape (n,) or (n, n_shafts)
        Shaft speed in rpm at the signal samples, non-negative, one column per
        shaft.
    orders : array_like, shape (K,)
        Orders to extract, in cycles per revolution of their shaft (positive,
        may be fractional). Keep them above about ``1 / window_revs``: a lower
        order cannot be separated from its own negative-frequency image.
    window_revs : float, default 8
        Window length in revolutions of the order's shaft.
    overlap : float, default 0.75
        Fraction of the window by which successive frames overlap, in
        ``[0, 1)``. The normalised overlap-add keeps the amplitude exact for
        any value; the window sum is constant (the usual COLA condition) for
        a periodic Hann window when ``1 / (1 - overlap)`` is an integer.
    window : str, default "hann"
        ``"hann"``, ``"hamming"``, ``"blackman"``, ``"blackmanharris"``,
        ``"bartlett"`` or ``"boxcar"``. The window is a continuous function of
        angle, evaluated at each sample's angle, over exactly ``window_revs``
        revolutions (cosine-sum windows in their periodic form, so that copies
        shifted by ``window_revs / m`` add up to a constant for integer ``m``).
        The significance test assumes a tapered window.
    shafts : int or array_like of int, shape (K,), optional
        Column of ``rpm`` each order refers to. Default: 0 for all orders (as
        in :func:`rota_order.synth.synth_runup`).
    delay : float, default 0
        Speed-reference delay in s, see :func:`rota_order.speed.rpm_to_phase`.
    significance : float in (0, 1), optional
        Confidence level of an optional per-frame test that discards noise.
        The power ``|c_kj|^2`` of each coefficient (each channel separately) is
        compared with the local noise level, estimated as the mean power of up
        to eight probe coefficients at orders ``order_k +- (3, 4.5, 6, 7.5) /
        window_revs`` (for a Hann window; the first probe is one bin beyond the
        main lobe, the spacing is 1.5 bins so the probes are roughly
        independent). A probe is dropped if it lies in the main lobe of another
        requested order or too close to order 0, or when less than half of its
        frame is below the Nyquist frequency. For noise only, ``|c|^2`` over a
        mean of ``M`` probe powers is F(2, 2M) distributed (``|c|^2 / E|c|^2`` is
        ``chi2(2) / 2``), and the coefficient is set to 0 unless it exceeds the
        ``significance`` quantile, ``M ((1 - p)^(-1/M) - 1)`` times the noise
        level. A frame with fewer than 2 probes is left unchanged, and an order
        with fewer than 2 probe orders at all (for example inside a comb of
        orders every 0.5 with an 8-revolution window) is left ungated with a
        warning. The probes are extra demodulations (up to nine times the cost).
        The null hypothesis assumes noise that is locally smooth over the probe
        orders and an order times ``window_revs`` of at least about 2 (so that
        ``|c|^2`` is exponentially distributed); other lines inside the probe
        range raise the noise estimate and make the test conservative.

    Returns
    -------
    OrderComponents
        ``waveforms`` and ``envelopes`` have shape ``(n, K)`` (or
        ``(n, K, ch)`` for multichannel input) in the unit of ``x``;
        ``envelopes[:, k]`` is the complex *peak* envelope, so that
        ``waveforms[:, k] = Re{envelopes[:, k] exp(1j 2 pi orders[k] rev)}``
        with ``rev`` the shaft angle(s) returned in ``rev``. ``method`` is
        ``"tracked-harmonics"``; ``params`` records the arguments, the
        number of frames per shaft and, with ``significance``, the fraction of
        coefficients kept per order.

    Notes
    -----
    *Edges.* Frames are centred only inside the record, so those centred within
    ``window_revs / 2`` revolutions of the first or last sample are incomplete:
    they see only part of their window. Such a coefficient is still normalised
    by the window weight actually available, and the overlap-add by the frame
    weights actually present, so a stationary component keeps its exact
    amplitude. The effective window is shorter and truncated, however, so
    leakage from neighbouring orders and noise suppression degrade; the
    incomplete frames overlap their neighbours, so this reaches up to
    ``window_revs`` revolutions from each end (``window_revs - hop_revs`` at the
    start), the distance reported as ``params["edge_revs"]``. Trust the output
    beyond it, and treat the first and last ``window_revs / 2`` revolutions as
    indicative only.

    *Nyquist.* Where ``order * rpm / 60 >= fs / 2`` the envelope and waveform are
    0 (never clipped to ``fs / 2``), and those samples are left out of the frame
    averages, so the envelope just below the crossing is not biased by the
    muted part of the window. A real component close to ``fs / 2`` cannot be
    told from its own aliased image, which falls near order 0 after
    demodulation: the amplitude is biased where ``fs / 2 - f`` is below a few
    main-lobe widths, ``3 * rpm / (60 * window_revs)`` Hz, and in the frames
    that see such samples, that is within about ``window_revs`` revolutions of
    a crossing of ``fs / 2``. Keep the orders of interest well below ``fs / 2``
    (and above ``1 / window_revs``).

    *Cost.* A few operations per sample, order, channel and overlapping frame
    (``1 / (1 - overlap)`` of them), by sparse accumulation in blocks of
    samples. It grows linearly with the record length and does not depend on
    the window length: of the order of one second per order and per minute of
    audio at 48 kHz for a mono signal. Memory is dominated by the outputs
    (24 bytes per sample, order and channel).

    References
    ----------
    Antoni, J. (2009). Cyclostationarity by examples. Mechanical Systems and
    Signal Processing, 23(4), 987-1036.
    Brandt, A. (2011). Noise and Vibration Analysis: Signal Analysis and
    Experimental Procedures, ch. 12. Wiley.
    Allen, J. B., & Rabiner, L. R. (1977). A unified approach to short-time
    Fourier analysis and synthesis. Proceedings of the IEEE, 65(11), 1558-1564.
    Harris, F. J. (1978). On the use of windows for harmonic analysis with the
    discrete Fourier transform. Proceedings of the IEEE, 66(1), 51-83.
    Finn, H. M., & Johnson, R. S. (1968). Adaptive detection mode with threshold
    control as a function of spatially sampled clutter-level estimates. RCA
    Review, 29(3), 414-464.
    """
    X, was_1d = as_2d(x)
    n, ch = X.shape
    if n < 2 or ch < 1:
        raise ValueError("the signal needs at least 2 samples and 1 channel")
    if not np.all(np.isfinite(X)):
        raise ValueError("x contains non-finite values")
    fs = float(fs)
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a positive sampling rate in Hz")

    rpm = np.asarray(rpm, dtype=float)
    if rpm.ndim not in (1, 2):
        raise ValueError("rpm must have shape (n,) or (n, n_shafts)")
    if rpm.shape[0] != n:
        raise ValueError(f"rpm must have {n} samples like the signal, got {rpm.shape[0]}")
    if not np.all(np.isfinite(rpm)):
        raise ValueError("rpm contains non-finite values; regularise it first")
    if np.any(rpm < 0):
        raise ValueError("rpm must be non-negative (the shaft angle must not decrease)")
    R = rpm[:, None] if rpm.ndim == 1 else rpm
    n_shafts = R.shape[1]

    orders = np.atleast_1d(np.asarray(orders, dtype=float))
    if orders.ndim != 1 or orders.size == 0:
        raise ValueError("orders must be a non-empty 1-D sequence")
    if not (np.all(np.isfinite(orders)) and np.all(orders > 0)):
        raise ValueError("orders must be finite and positive")
    if shafts is None:
        shafts = np.zeros(orders.size, dtype=int)
    else:
        sh = np.asarray(shafts)
        if sh.ndim == 0:
            sh = np.full(orders.size, sh)
        if sh.shape != orders.shape or not np.all(sh == np.round(sh)):
            raise ValueError("shafts must be an int or one int per order")
        shafts = sh.astype(int)
    if shafts.min() < 0 or shafts.max() >= n_shafts:
        raise ValueError(f"shafts must be columns of rpm (0..{n_shafts - 1})")

    window_revs = float(window_revs)
    if not (np.isfinite(window_revs) and window_revs > 0):
        raise ValueError("window_revs must be positive (revolutions)")
    overlap = float(overlap)
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if window not in _WINDOWS:
        raise ValueError(f"window must be one of {_WINDOWS}, got {window!r}")
    delay = float(delay)
    if not np.isfinite(delay):
        raise ValueError("delay must be finite (s)")
    if significance is not None:
        significance = float(significance)
        if not 0.0 < significance < 1.0:
            raise ValueError("significance must be in (0, 1)")
    hop_revs = window_revs * (1.0 - overlap)

    rev = rpm_to_phase(rpm, fs, delay=delay)
    REV = rev[:, None] if rev.ndim == 1 else rev
    if delay:
        t = np.arange(n) / fs
        R = np.column_stack([np.interp(t + delay, t, c) for c in R.T])

    K = orders.size
    env = np.zeros((n, K, ch), dtype=complex)
    wav = np.zeros((n, K, ch))
    n_frames = {}
    kept = np.ones(K)
    for s in np.unique(shafts):
        cols = np.flatnonzero(shafts == s)
        if np.array_equal(cols, np.arange(cols[0], cols[-1] + 1)):
            cols = slice(int(cols[0]), int(cols[-1]) + 1)  # basic indexing is faster
        r = REV[:, s]
        if not r[-1] > r[0]:
            raise ValueError(f"shaft {int(s)} does not turn (its angle is constant)")
        if r[-1] - r[0] < window_revs:
            warnings.warn(
                f"shaft {int(s)} turns {r[-1] - r[0]:.3g} revolutions, fewer than "
                f"window_revs = {window_revs:g}: the whole record is edge region",
                stacklevel=2,
            )
        frames = _Frames(r, window_revs, hop_revs, window)
        n_frames[int(s)] = frames.n_frames
        o = orders[cols]
        speed = R[:, s]
        if significance is None:
            C, _ = _frame_coefficients(X, r, speed, fs, frames, o)
        else:
            probes, index = _plan_probes(o, _HALF_WIDTH[window], window_revs)
            CP, cov = _frame_coefficients(X, r, speed, fs, frames, np.concatenate((o, probes)))
            C = np.ascontiguousarray(CP[:, : o.size])
            kept[cols] = _significance_gate(
                C, CP[:, o.size :], cov[:, o.size :], index, significance, o
            )
        _synthesise(C, frames, r, speed, fs, o, env, wav, cols)

    params = {
        "fs": fs,
        "window_revs": window_revs,
        "overlap": overlap,
        "window": window,
        "hop_revs": hop_revs,
        "edge_revs": window_revs,
        "delay": delay,
        "significance": significance,
        "shafts": [int(s) for s in shafts],
        "n_frames": {str(k): v for k, v in n_frames.items()},
        "amplitude": "peak",
    }
    if significance is not None:
        params["kept_fraction"] = [float(k) for k in kept]
    return OrderComponents(
        orders=orders,
        shafts=shafts,
        waveforms=wav[..., 0] if was_1d else wav,
        envelopes=env[..., 0] if was_1d else env,
        rev=rev,
        method="tracked-harmonics",
        params=params,
    )
