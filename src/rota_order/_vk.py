"""Vold-Kalman order filter (second generation, least-squares form).

Units: time in s, frequency and bandwidth in Hz, rotational speed in rpm,
shaft angle in revolutions, order in cycles per revolution. Amplitudes are
peak amplitudes in the unit of the input signal.
"""

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded

from ._components import OrderComponents
from ._utils import as_2d, broadcast_rpm
from .speed import rpm_to_phase

__all__ = ["order_waveform"]

# Half-power point of the envelope response H = 1 / (1 + lam s^4): |H| = 1/sqrt(2)
# <=> lam s^4 = sqrt(2) - 1.
_HALF_POWER = np.sqrt(2.0) - 1.0

# Working-set budget of one solved block, bytes (banded matrix + vectors).
_BUDGET = 200_000_000
# Block seams, in units of 1 / bandwidth seconds: samples within _MARGIN of a
# block end are discarded (the free-end transient of the smoother has decayed
# to exp(-5.5 * 3) ~ 6e-8 beyond it), the two blocks are cross-faded over _FADE.
_MARGIN, _FADE = 3.0, 1.0
# Iterative refinement: stop at _TOL, accept a stalled floor below _FLOOR.
_MAX_REFINE, _TOL, _FLOOR = 50, 1e-12, 1e-8


def _bandwidth_to_weight(bandwidth, fs):
    """Structure weight r for a -3 dB envelope bandwidth (Hz): H(bandwidth) = 1/sqrt(2)."""
    s = 2.0 * np.sin(np.pi * np.asarray(bandwidth, dtype=float) / fs)
    return np.sqrt(0.5 * _HALF_POWER) / s**2


def _weight_to_bandwidth(weight, fs):
    """Inverse of :func:`_bandwidth_to_weight`."""
    s = np.sqrt(np.sqrt(0.5 * _HALF_POWER) / np.asarray(weight, dtype=float))
    return fs / np.pi * np.arcsin(0.5 * s)


def _assemble(g, r2):
    """Lower banded storage of ``G^T G + diag(r2) (D2^T D2)`` for interleaved unknowns.

    ``g`` has shape (L, m) with m = 2K, row n the data row of sample n; unknown
    ``m * n + j`` belongs to sample n. Entry ``ab[d, c] = A[c + d, c]``; the
    half bandwidth is 2m (the structure couples sample n with n + 1, n + 2).
    """
    L, m = g.shape
    ab = np.zeros((2 * m + 1, L * m))
    for d in range(m):  # rank-one data term of each sample
        ab[d].reshape(L, m)[:, : m - d] = g[:, d:] * g[:, : m - d]
    # D2^T D2: diagonal 1 5 6 ... 6 5 1, first off-diagonal -2 -4 ... -4 -2, second 1
    s0 = np.zeros(L)
    s0[:-2] += 1.0
    s0[1:-1] += 4.0
    s0[2:] += 1.0
    s1 = np.zeros(L - 1)
    s1[:-1] -= 2.0
    s1[1:] -= 2.0
    ab[0].reshape(L, m)[:] += s0[:, None] * r2
    ab[m, : m * (L - 1)].reshape(L - 1, m)[:] += s1[:, None] * r2
    ab[2 * m, : m * (L - 2)].reshape(L - 2, m)[:] += r2
    return ab


def _solve_block(X, theta, r):
    """Solve the Vold-Kalman least-squares problem on one block.

    X : (L, ch) signal; theta : (L, K) carrier angles (rad); r : (K,) structure
    weights. Returns the complex envelopes, shape (L, K, ch).

    The normal equations are solved with a banded Cholesky factorisation and
    refined iteratively. Their condition number is about ``32 r^2``, so the
    plain solution loses ``log10(32 r^2)`` digits; refinement with a residual
    evaluated from differences of neighbouring values (each difference is
    accurate relative to itself, so the tiny second differences of the smooth
    solution keep their precision) recovers them while ``32 r^2 eps < 1``.
    """
    L, ch = X.shape
    K = theta.shape[1]
    m = 2 * K
    g = np.empty((L, m))
    g[:, 0::2] = np.cos(theta)
    g[:, 1::2] = -np.sin(theta)
    r2 = np.repeat(np.asarray(r, dtype=float) ** 2, 2)
    fail = (
        "the Vold-Kalman normal equations are numerically singular: the bandwidth "
        "is too narrow for this sampling rate (fs / bandwidth above about 5e4; "
        "decimate the signal and speed first) or two orders have identical carriers"
    )
    try:
        c = cholesky_banded(_assemble(g, r2), lower=True, overwrite_ab=True)
    except np.linalg.LinAlgError as err:
        raise ValueError(fail) from err
    b = (g[:, :, None] * X[:, None, :]).reshape(L * m, ch)
    u = cho_solve_banded((c, True), b)
    prev = np.inf
    for it in range(_MAX_REFINE):
        U = u.reshape(L, m, ch)
        y = (g[:, :, None] * U).sum(axis=1)  # sum_k a_k cos - b_k sin
        res = g[:, :, None] * (X - y)[:, None, :]
        w = (U[:-2] - U[1:-1]) - (U[1:-1] - U[2:])  # D2 u, accurate to its own size
        z = np.zeros_like(U)
        z[:-2] += w
        z[1:-1] -= 2.0 * w
        z[2:] += w
        res -= r2[None, :, None] * z
        d = cho_solve_banded((c, True), res.reshape(L * m, ch))
        u = u + d
        scale = np.abs(u).max(axis=0)
        rel = float(np.max(np.abs(d).max(axis=0) / np.where(scale > 0, scale, 1.0)))
        if not np.isfinite(rel):
            raise ValueError(fail)
        if rel <= _TOL:
            break
        if rel >= 0.5 * prev:  # stalled at the rounding floor, or diverging
            if rel > _FLOOR:
                raise ValueError(fail)
            break
        prev = rel
    else:
        if rel > _FLOOR:
            raise ValueError(fail)
    U = u.reshape(L, K, 2, ch)
    return U[:, :, 0, :] + 1j * U[:, :, 1, :], it + 1


def _fade_in(o, mg):
    """Weights of the later block over an overlap of ``o`` samples: 0, raised cosine up, 1."""
    q = o - 2 * mg
    w = np.empty(o)
    w[:mg] = 0.0
    w[o - mg :] = 1.0
    w[mg : o - mg] = 0.5 * (1.0 - np.cos(np.pi * (np.arange(q) + 0.5) / q))
    return w


def _solve_system(X, theta, r, bw_min, fs, block):
    """Solve one coupled system, in overlapping cross-faded blocks if it is long.

    Returns the envelopes (n, K, ch) and a dict with the blocking used.
    """
    n, ch = X.shape
    K = theta.shape[1]
    per_sample = 8 * 2 * K * ((4 * K + 1) + 8 * ch)  # matrix + working vectors
    mg = max(1, int(np.ceil(_MARGIN * fs / bw_min)))
    fade = max(2, int(np.ceil(_FADE * fs / bw_min)))
    ov = 2 * mg + fade
    info = {"block": None, "overlap": None, "n_blocks": 1}
    if block is None:
        # memory budget, but blocks of at least 4 overlaps (accuracy at crossings)
        block = max(_BUDGET // per_sample, 4 * ov)
        if min(n, block) * per_sample > 8 * _BUDGET:
            raise MemoryError(
                "the bandwidth is too narrow to solve this many orders within the memory "
                "budget; raise the bandwidth, use joint=False, decimate, or pass block="
            )
    elif n > block and block < 2 * ov:
        raise ValueError(
            f"block must be at least {2 * ov} samples (twice the {ov}-sample overlap of "
            f"{2 * _MARGIN + _FADE:g} / bandwidth) when the signal is longer than one block"
        )
    if n <= block:
        env, it = _solve_block(X, theta, r)
        info["iterations"] = it
        return env, info
    hop = block - ov
    nb = max(2, int(np.ceil((n - ov) / hop)))
    starts = np.rint(np.linspace(0, n - block, nb)).astype(int)
    acc = np.zeros((n, K, ch), dtype=complex)
    wsum = np.zeros(n)
    iters = 0
    for j, s in enumerate(starts):
        env, it = _solve_block(X[s : s + block], theta[s : s + block], r)
        iters = max(iters, it)
        w = np.ones(block)
        if j > 0:
            o = starts[j - 1] + block - s
            w[:o] *= _fade_in(o, mg)
        if j < nb - 1:
            o = s + block - starts[j + 1]
            w[block - o :] *= 1.0 - _fade_in(o, mg)
        acc[s : s + block] += w[:, None, None] * env
        wsum[s : s + block] += w
    acc /= wsum[:, None, None]
    info.update(block=int(block), overlap=int(ov), n_blocks=int(nb), iterations=iters)
    return acc, info


def order_waveform(
    x,
    fs,
    rpm,
    orders,
    *,
    bandwidth=1.0,
    shafts=None,
    joint=True,
    block=None,
    return_components=True,
):
    """Extract order waveforms and complex envelopes with a Vold-Kalman filter.

    Second-generation Vold-Kalman filter in its least-squares form. Each order
    ``k`` is modelled as ``Re{c_k(n) exp(i theta_k(n))}`` with the carrier
    angle ``theta_k = 2 pi orders[k] rev_s`` (``rev_s``: angle in revolutions of
    the order's shaft, :func:`rota_order.speed.rpm_to_phase`) and a slowly varying
    complex peak envelope ``c_k = a_k + i b_k``. The envelopes minimise

    ``sum_n (x(n) - sum_k [a_k(n) cos theta_k(n) - b_k(n) sin theta_k(n)])^2
    + r_k^2 sum_k (|D2 a_k|^2 + |D2 b_k|^2)``

    where ``D2`` is the second difference (the "structure equation": the
    envelope is a piece-wise straight line plus a small innovation). The data
    term is the sum of squared residuals, the second term makes the envelopes
    smooth; the weights ``r_k`` are set from ``bandwidth``. The minimiser is
    the exact solution of the normal equations, which are symmetric positive
    definite and banded when the unknowns are interleaved sample by sample;
    they are solved with a banded Cholesky factorisation
    (``scipy.linalg.cholesky_banded``), in ``O(n K^3)`` operations.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal sampled at ``fs`` (any unit). All channels share the system
        matrix and are solved together; channels do not interact.
    fs : float
        Sampling rate in Hz.
    rpm : float or array_like, shape (n,) or (n, n_shafts)
        Shaft speed in rpm at the signal samples, one column per shaft.
    orders : float or sequence of float, length K
        Orders to extract, in cycles per revolution of their shaft (positive,
        may be fractional). The carrier of each order must stay below ``fs / 2``.
    bandwidth : float or sequence of float, default 1
        -3 dB bandwidth of the envelope in Hz, scalar or one value per order.
        It is the offset from the carrier at which a tone is attenuated by
        3 dB (amplitude gain ``1 / sqrt(2)``, half power) by the single-order
        filter; the pass band around the carrier is therefore ``2 * bandwidth``
        wide. Smaller values reject more noise and neighbouring orders but
        follow amplitude and phase changes more slowly (the envelope response
        decays as ``exp(-5.5 bandwidth t)``, a time constant of about
        ``0.18 / bandwidth`` s).
    shafts : sequence of int, optional
        Column of ``rpm`` each order refers to. Default: all 0.
    joint : bool, default True
        Solve all orders in one coupled system. This is required whenever two
        carriers come closer than about ``2 * bandwidth`` (crossing orders,
        two shafts at nearly the same speed): the coupled solution attributes
        the signal to the orders jointly. With ``False`` each order is solved
        alone (``O(n)``), and any neighbouring order passing the pass band
        leaks into the result.
    block : int, optional
        Longest system, in samples, solved at once. Longer signals are split
        into overlapping blocks that are cross-faded, see Notes. Default: the
        largest length whose banded matrix and working vectors need about
        200 MB.
    return_components : bool, default True
        If False, return only the order waveforms array.

    Returns
    -------
    OrderComponents or ndarray
        ``orders``, ``shafts``, ``waveforms`` (shape (n, K) or (n, K, ch)),
        complex ``envelopes`` of the same shape, ``rev`` (shaft angle(s) in
        revolutions) and ``params``, with ``method="vold-kalman"``. Amplitudes
        are peak, in the unit of ``x``: ``waveforms = Re{envelopes *
        exp(1j * 2 pi * orders * rev)}`` and ``|envelopes|`` is the peak
        amplitude of the order. With ``return_components=False`` only the
        ``waveforms`` array.

    Raises
    ------
    ValueError
        On invalid arguments, a carrier at or above ``fs / 2``, or a
        bandwidth so narrow relative to ``fs`` (``fs / bandwidth`` above about
        5e4) that the system is numerically singular.
    MemoryError
        If the overlap needed for the bandwidth makes even the shortest
        admissible block far larger than the memory budget.

    Notes
    -----
    *Bandwidth.* For one order with a constant carrier the envelope of an
    input tone at ``f_carrier + df`` is multiplied by

    ``H(df) = 1 / (1 + 2 r^2 (2 sin(pi df / fs))^4)``.

    Derivation: with ``x = Re{X exp(i theta)} + ...`` the data term averages
    (the ``exp(2 i theta)`` part oscillates and cancels in the sum) to
    ``|X - c|^2 / 2`` per sample, so the cost is ``|X - c|^2 / 2 + r^2 |D2 c|^2``
    and the minimiser is the low-pass ``C = X / (1 + 2 r^2 |D2(w)|^2)``, with
    ``|D2(w)|^2 = (2 sin(w / 2))^4`` the squared magnitude of the second
    difference at ``w = 2 pi df / fs``. (The factor 2 comes from the real
    formulation: ``(x - Re{c e^{i theta}})^2`` has the mean ``|x_c - c|^2 / 2``.
    If the data residual is written for the demodulated complex envelope
    instead, the factor is absent and ``H = 1 / (1 + r^2 (...)^4)``; the
    weights of the two conventions differ by ``sqrt 2``.) Setting
    ``|H(bandwidth)| = 1 / sqrt(2)`` gives ``2 r^2 (2 sin(pi B / fs))^4 =
    sqrt(2) - 1`` and

    ``r = sqrt((sqrt(2) - 1) / 2) / (2 sin(pi B / fs))^2``,

    which is ``0.455 (fs / (2 pi B))^2`` for ``B << fs``. The measured -3 dB
    offset equals ``B`` to better than 1e-4 when the record is several time
    constants long (tests: 5 %, the ROADMAP tolerance; measured 2e-5); the
    only approximation is the image term of the real formulation (the
    ``exp(2 i theta)`` part dropped above), of the order ``(B / f_carrier)^4``.
    An order whose carrier is within about ``bandwidth`` of 0 Hz or of
    ``fs / 2`` overlaps its own mirror image and cannot be resolved (for
    example at the start of a run-up from standstill).

    *Resolution.* Two orders can be separated only if their carriers differ
    by more than about ``2 * bandwidth`` for a time of several time constants
    (``joint=True`` extends this to crossings). A bandwidth wider than the
    carrier spacing cannot separate them: the smoother is free to explain
    one order by the envelope of the other.

    *Numerics.* The condition number of the normal equations is about
    ``32 r^2``, up to ``1e17`` for narrow bands. The factorisation is
    therefore followed by iterative refinement with a residual evaluated
    from differences of neighbouring values (Higham 2002, ch. 12), which
    keeps the envelope accurate to about ``1e-8`` of its size for
    ``fs / bandwidth`` up to ``5e4``. Beyond that the factorisation breaks
    down and a ``ValueError`` asks for decimation.

    *Blocks.* The banded matrix has ``2K n`` unknowns and half bandwidth
    ``4K``. A long signal is split into blocks of ``block`` samples that
    overlap by ``7 / bandwidth`` s (``bandwidth``: the smallest of the system).
    Each block is solved with free ends; the ``3 / bandwidth`` s next to a
    block end, where the free-end transient of the smoother has decayed by
    ``exp(-5.5 * 3)``, are given zero weight and the rest of the overlap is
    cross-faded with a raised cosine. The result differs from the single
    solution by the truncated transient, below ``1e-6`` of the amplitude for
    stationary and slowly varying envelopes (about ``1e-10`` in the tests).
    The default block is the larger of the 200 MB working-set limit and
    four overlaps; an explicit ``block`` must be at least two overlaps. For
    crossing or nearly coincident carriers this is an approximation: the
    information that separates the orders is less local than the envelope
    response and is not shared across a seam beyond the overlap. In the tests
    a seam at a crossing gave differences of ``2e-2`` with blocks of two
    overlaps, ``7e-6`` with 3.6 and ``9e-7`` with 5.7; use long blocks there.

    References
    ----------
    Vold, H., & Leuridan, J. (1993). High resolution order tracking at
    extreme slew rates, using Kalman tracking filters. SAE Technical Paper
    931288.
    Vold, H., Mains, M., & Blough, J. (1997). Theoretical foundations for high
    performance order tracking with the Vold-Kalman tracking filter. SAE
    Technical Paper 972007.
    Feldbauer, C., & Hoeldrich, R. (2000). Realization of Vold-Kalman tracking
    filter: a least squares problem. Proc. COST G-6 Conference on Digital
    Audio Effects (DAFx-00), Verona.
    Tuma, J. (2005). The passband width of the Vold-Kalman order tracking
    filter. Sbornik vedeckych praci VSB-TU Ostrava, Rada strojni, 51(2),
    149-154.
    Pan, M.-C., & Lin, Y.-F. (2006). Further exploration of Vold-Kalman
    filtering order tracking with shaft-speed information. I: Theoretical
    part, numerical implementation and parameter investigations. Mechanical
    Systems and Signal Processing, 20(5), 1134-1154.
    Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms, 2nd
    ed., ch. 12 (iterative refinement). SIAM.
    """
    fs = float(fs)
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a positive sampling rate in Hz")
    X, was_1d = as_2d(x)
    n, ch = X.shape
    if n < 3:
        raise ValueError("the signal needs at least 3 samples")
    if ch < 1:
        raise ValueError("the signal needs at least one channel")
    if not np.all(np.isfinite(X)):
        raise ValueError("x contains non-finite values")

    rpm = np.asarray(rpm, dtype=float)
    if rpm.ndim < 2:
        rpm = broadcast_rpm(rpm, n)
    elif rpm.ndim != 2 or rpm.shape[0] != n:
        raise ValueError(f"rpm must have shape ({n},) or ({n}, n_shafts)")
    R = rpm[:, None] if rpm.ndim == 1 else rpm
    n_shafts = R.shape[1]

    orders = np.atleast_1d(np.asarray(orders, dtype=float))
    if orders.ndim != 1 or orders.size < 1:
        raise ValueError("orders must be a non-empty 1-D sequence")
    if not (np.all(np.isfinite(orders)) and np.all(orders > 0)):
        raise ValueError("orders must be finite and positive")
    K = orders.size

    if shafts is None:
        shafts = np.zeros(K, dtype=int)
    else:
        sh = np.atleast_1d(np.asarray(shafts))
        if sh.shape != (K,) or not np.all(np.isfinite(sh.astype(float))):
            raise ValueError(f"shafts must have one entry per order ({K})")
        if not np.all(sh == np.round(sh)):
            raise ValueError("shafts must be integer column indices of rpm")
        shafts = sh.astype(int)
    if np.any(shafts < 0) or np.any(shafts >= n_shafts):
        raise ValueError(f"shafts must be columns of rpm (0..{n_shafts - 1})")

    bw = np.asarray(bandwidth, dtype=float)
    if bw.ndim == 0:
        bw = np.full(K, float(bw))
    if bw.shape != (K,):
        raise ValueError(f"bandwidth must be a scalar or have one entry per order ({K})")
    if not (np.all(np.isfinite(bw)) and np.all(bw > 0) and np.all(bw < 0.5 * fs)):
        raise ValueError("bandwidth must be in (0, fs / 2) Hz")

    joint = bool(joint)
    if block is not None:
        if int(block) != block or block < 1:
            raise ValueError("block must be a positive integer number of samples")
        block = int(block)

    f_max = orders * np.abs(R[:, shafts]).max(axis=0) / 60.0
    if np.any(f_max >= 0.5 * fs):
        k = int(np.argmax(f_max))
        raise ValueError(
            f"order {orders[k]:g} reaches {f_max[k]:g} Hz, at or above fs / 2 = {0.5 * fs:g} Hz"
        )
    if joint and K > 1:
        pairs = set(zip(orders.tolist(), shafts.tolist()))
        if len(pairs) < K:
            raise ValueError("joint=True needs distinct (order, shaft) pairs")

    rev = rpm_to_phase(rpm, fs)
    REV = rev[:, None] if rev.ndim == 1 else rev
    cyc = REV[:, shafts] * orders[None, :]
    theta = 2.0 * np.pi * (cyc - np.floor(cyc))  # carrier angle, rad
    r = _bandwidth_to_weight(bw, fs)

    groups = [np.arange(K)] if joint else [np.array([k]) for k in range(K)]
    env = np.empty((n, K, ch), dtype=complex)
    infos = []
    for gi in groups:
        env[:, gi], info = _solve_system(X, theta[:, gi], r[gi], float(bw[gi].min()), fs, block)
        infos.append(info)
    wave = env.real * np.cos(theta)[:, :, None] - env.imag * np.sin(theta)[:, :, None]

    if was_1d:
        env, wave = env[..., 0], wave[..., 0]
    if not return_components:
        return wave
    blocks = [i["block"] for i in infos if i["block"] is not None]
    params = {
        "fs": fs,
        "bandwidth": [float(v) for v in bw],
        "weight": [float(v) for v in r],
        "joint": joint,
        "block": max(blocks) if blocks else None,
        "overlap": max((i["overlap"] for i in infos if i["overlap"] is not None), default=None),
        "n_blocks": max(i["n_blocks"] for i in infos),
        "refinement_iterations": max(i["iterations"] for i in infos),
        "structure": "second difference",
        "amplitude": "peak",
    }
    return OrderComponents(
        orders=orders,
        shafts=shafts,
        waveforms=wave,
        envelopes=env,
        rev=rev,
        method="vold-kalman",
        params=params,
    )
