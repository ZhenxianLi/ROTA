"""Editing, speed remapping and recombination of orders and residual.

:func:`modify_orders` applies per-order gain, mute, solo, relabelling and
frequency offsets to an :class:`~rota_order.resynth.OrderTable` or to extracted
:class:`~rota_order.extract.OrderComponents`; :func:`remap_rpm` renders either of them
along a *new* speed curve by remodulating the orders (the correct way to change
the apparent speed of a recording: no resampling or pitch shifting of the
audio); :func:`resynthesize` recombines the orders and the broadband residual
with explicit gains and an explicit normalisation.

Units: time in s, frequency in Hz, rotational speed in rpm, shaft angle in
revolutions, order in cycles per revolution, gains in dB (``20 log10`` of an
amplitude ratio). Amplitudes are in the unit of the analysed signal (for
example Pa). Order components are *peak* amplitudes (the convention of
:class:`~rota_order.extract.OrderComponents`), tables carry their own convention in
``OrderTable.amplitude``; broadband residuals are characterised by their RMS.

Nothing here has a hidden gain (ROADMAP design principle 8): every gain, the
normalisation factor and the achieved energy ratio are returned.
"""

import copy
from collections.abc import Mapping

import numpy as np

from ._components import OrderComponents
from ._ordertable import OrderTable, _check_fs, _resolve_phase, synthesize_orders
from .speed import rpm_to_phase

__all__ = ["modify_orders", "remap_rpm", "resynthesize"]

_TWO_PI = 2.0 * np.pi
_ORDER_TOL = 1e-9  # orders are matched to within this (relative to max(1, order))
_MODES = ("time", "rpm")
_GRID_POINTS = 100  # default size of the speed grid of remap_rpm(mode="rpm")


# --------------------------------------------------------------------------
# modify_orders
# --------------------------------------------------------------------------


def _fmt(order):
    """Dictionary key for an order in the modification record."""
    return format(float(order), ".12g")


def _rows(orders, key, what):
    """Indices of the rows whose order equals ``key``; raises for an unknown order."""
    try:
        o = float(key)
    except (TypeError, ValueError):
        raise TypeError(f"{what}: orders must be numbers, got {key!r}") from None
    if not np.isfinite(o):
        raise ValueError(f"{what}: order must be finite, got {key!r}")
    rows = np.flatnonzero(np.abs(orders - o) <= _ORDER_TOL * max(1.0, abs(o)))
    if rows.size == 0:
        have = ", ".join(f"{v:g}" for v in orders)
        raise ValueError(f"{what}: order {o:g} is not in the object (orders: {have})")
    return rows


def _order_list(spec, what):
    """Orders given as a scalar or a sequence, as a list of floats."""
    if spec is None:
        return []
    if isinstance(spec, Mapping):
        raise TypeError(f"{what} must be an order or a sequence of orders, not a dict")
    if isinstance(spec, (set, frozenset)):
        spec = sorted(spec)
    a = np.atleast_1d(np.asarray(spec, dtype=float))
    if a.ndim != 1:
        raise ValueError(f"{what} must be an order or a 1-D sequence of orders")
    return a.tolist()


def _items(spec, what):
    """Items of a ``{order: value}`` dict (``None`` gives none)."""
    if spec is None:
        return []
    if not isinstance(spec, Mapping):
        raise TypeError(f"{what} must be a dict {{order: value}}, got {type(spec).__name__}")
    return list(spec.items())


def _number(value, what, *, allow_neg_inf=False):
    v = float(value)
    if np.isnan(v) or v == np.inf or (v == -np.inf and not allow_neg_inf):
        raise ValueError(f"{what}: value must be finite, got {value!r}")
    return v


def modify_orders(obj, *, gain_db=None, mute=(), solo=(), order_map=None, freq_offset=None):
    """Per-order gain, mute, solo, relabelling and frequency offset.

    Works on an :class:`~rota_order.resynth.OrderTable` or on
    :class:`~rota_order.extract.OrderComponents` and returns a *new* object of the
    same type; the input is not changed. All dictionaries and sequences name
    orders as they are labelled in the *input* (before ``order_map``); an order
    that is not present raises ``ValueError``. If several rows share an order
    (for example the same order on two shafts) the modification applies to all
    of them. Orders are matched to a relative tolerance of ``1e-9``.

    The operations are applied in this sense: ``gain_db`` scales the row;
    ``mute`` sets it to zero; ``solo`` (if not empty) sets every row that is
    *not* listed to zero, so rows are never dropped and the row order and
    shape of the object are kept; ``order_map`` relabels a row so that it is
    remodulated at a different carrier; ``freq_offset`` shifts the carrier by a
    fixed number of Hz. A silenced row is exactly zero (also where the input
    had NaN).

    Parameters
    ----------
    obj : OrderTable or OrderComponents
        Table, or extractor result (``envelopes`` are complex *peak* envelopes
        and ``waveforms = Re{envelopes exp(i 2 pi order rev)}``).
    gain_db : dict, optional
        ``{order: dB}``. The amplitude of the row is multiplied by
        ``10 ** (dB / 20)`` (``-inf`` silences it). The gain is relative, so it
        applies in the same way to a peak or an RMS table. Repeated matches add.
    mute : float or sequence of float, optional
        Orders to set to zero.
    solo : float or sequence of float, optional
        If given, keep only these orders (all others are set to zero).
    order_map : dict, optional
        ``{old_order: new_order}``. The row keeps its amplitude and phase law
        but is carried by the new order: ``waveform = Re{envelope exp(i 2 pi
        new_order rev)}``. Several orders can be moved at once (a swap
        ``{2: 4, 4: 2}`` is allowed because all keys refer to the input
        labels); the same row cannot be mapped twice.
    freq_offset : dict, optional
        ``{order: Hz}``. Adds a fixed offset to the carrier frequency, as for a
        sideband of a switching frequency. For a table the value is added to
        ``freq_offset`` (the phase convention of the table is unchanged). For
        components the offset is carried in ``params["freq_offset"]`` (one
        value per row, Hz, accumulated over calls) and applied to the
        waveforms now and by :func:`remap_rpm` later:
        ``waveform = Re{envelope exp(i (2 pi order rev + 2 pi offset t))}``
        with ``t = arange(n) / fs`` and ``fs = params["fs"]``.

    Returns
    -------
    OrderTable or OrderComponents
        The modified copy. For components the ``waveforms`` of every row whose
        carrier changed (``order_map`` or ``freq_offset``) are recomputed from
        its envelope on the new carrier, and the envelope and waveform are set
        to zero at the samples where the new carrier frequency is not inside
        ``(0, fs / 2)`` (muted, never clipped to the band edge: ROADMAP
        section 6, item 16). Rows that only change in gain keep their waveform
        scaled by the same factor as the envelope, so ``waveform = Re{envelope
        carrier}`` holds for every row. The record of the call is appended to
        ``params["modify"]`` (components) or ``meta["modify"]`` (table), which
        keeps the result JSON-serialisable.

    Raises
    ------
    TypeError
        If ``obj`` is neither of the two types or a dictionary has the wrong type.
    ValueError
        For an order that is not in ``obj``, a non-finite gain, order or offset,
        an order mapped twice, or components without ``params["fs"]`` when the
        carrier of a row changes.

    Notes
    -----
    Amplitude convention: unchanged. Gains are ratios; a table stays in its own
    convention (peak or RMS), components stay peak.

    References
    ----------
    McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis based on
    a sinusoidal representation. IEEE Transactions on Acoustics, Speech, and
    Signal Processing, 34(4), 744-754.
    Serra, X., & Smith, J. O. (1990). Spectral modeling synthesis: a sound
    analysis/synthesis system based on a deterministic plus stochastic
    decomposition. Computer Music Journal, 14(4), 12-24.
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    """
    if isinstance(obj, OrderTable):
        is_table = True
    elif isinstance(obj, OrderComponents):
        is_table = False
    else:
        raise TypeError(f"obj must be an OrderTable or OrderComponents, got {type(obj).__name__}")
    orders = np.array(obj.orders, dtype=float)
    k = orders.size

    gain = np.zeros(k)
    rec_gain = {}
    for key, g in _items(gain_db, "gain_db"):
        g = _number(g, "gain_db", allow_neg_inf=True)
        gain[_rows(orders, key, "gain_db")] += g
        rec_gain[_fmt(key)] = g

    silent = np.zeros(k, dtype=bool)
    mute_list = _order_list(mute, "mute")
    for o in mute_list:
        silent[_rows(orders, o, "mute")] = True
    solo_list = _order_list(solo, "solo")
    if solo_list:
        keep = np.zeros(k, dtype=bool)
        for o in solo_list:
            keep[_rows(orders, o, "solo")] = True
        silent |= ~keep

    new_orders = orders.copy()
    mapped = np.zeros(k, dtype=bool)
    rec_map = {}
    for old, new in _items(order_map, "order_map"):
        rows = _rows(orders, old, "order_map")
        new = _number(new, "order_map")
        if mapped[rows].any():
            raise ValueError(f"order_map: order {float(old):g} is mapped more than once")
        new_orders[rows] = new
        mapped[rows] = True
        rec_map[_fmt(old)] = new

    offset = np.zeros(k)
    rec_off = {}
    for key, v in _items(freq_offset, "freq_offset"):
        v = _number(v, "freq_offset")
        offset[_rows(orders, key, "freq_offset")] += v
        rec_off[_fmt(key)] = v

    with np.errstate(over="ignore"):
        lin = 10.0 ** (gain / 20.0)
    silent |= lin == 0.0

    record = {}
    if rec_gain:
        record["gain_db"] = rec_gain
    if mute_list:
        record["mute"] = mute_list
    if solo_list:
        record["solo"] = solo_list
    if rec_map:
        record["order_map"] = rec_map
    if rec_off:
        record["freq_offset"] = rec_off

    if is_table:
        return _modify_table(obj, lin, silent, new_orders, offset, record)
    return _modify_components(obj, orders, lin, silent, new_orders, offset, record)


def _history(container, record):
    """``container["modify"]`` extended by ``record`` (a new list; never in place)."""
    old = container.get("modify")
    return (list(old) if isinstance(old, list) else []) + [record]


def _modify_table(table, lin, silent, new_orders, offset, record):
    bshape = (lin.size,) + (1,) * (table.values.ndim - 1)
    values = table.values * lin.reshape(bshape)
    values[silent] = 0.0
    meta = copy.deepcopy(table.meta)
    meta["modify"] = _history(meta, record)
    return OrderTable(
        orders=new_orders,
        rpm=table.rpm,
        values=values,
        amplitude=table.amplitude,
        freq_offset=table.freq_offset + offset,
        unit=table.unit,
        meta=meta,
    )


def _component_offsets(comp, k):
    """Per-row frequency offsets in Hz carried by a components object (zeros if none)."""
    off = comp.params.get("freq_offset")
    if off is None:
        return np.zeros(k)
    off = np.asarray(off, dtype=float)
    if off.shape != (k,) or not np.all(np.isfinite(off)):
        raise ValueError(f"params['freq_offset'] must hold {k} finite values in Hz")
    return off


def _modify_components(comp, orders, lin, silent, new_orders, offset, record):
    env0 = np.asarray(comp.envelopes, dtype=complex)
    wav0 = np.asarray(comp.waveforms, dtype=float)
    k = orders.size
    if env0.ndim not in (2, 3) or env0.shape[1] != k or wav0.shape != env0.shape:
        raise ValueError("envelopes and waveforms must have shape (n, k) or (n, k, ch)")
    scale = lin.reshape((1, k) + (1,) * (env0.ndim - 2))
    env = env0 * scale
    wav = wav0 * scale
    env[:, silent] = 0.0
    wav[:, silent] = 0.0

    params = copy.deepcopy(dict(comp.params))
    total = _component_offsets(comp, k) + offset
    changed = ((new_orders != orders) | (offset != 0.0)) & ~silent
    if changed.any():
        fs = params.get("fs")
        if fs is None:
            raise ValueError(
                "the components carry no sampling rate (params['fs']); it is needed to "
                "remodulate rows whose order or frequency offset changes"
            )
        fs = float(fs)
        rev = np.asarray(comp.rev, dtype=float)
        rev2 = rev[:, None] if rev.ndim == 1 else rev
        shafts = np.asarray(comp.shafts, dtype=int)
        t = np.arange(env.shape[0]) / fs
        nyq = 0.5 * fs
        shaft_hz = {}
        for j in np.flatnonzero(changed):
            s = int(shafts[j])
            if s not in shaft_hz:
                shaft_hz[s] = np.gradient(rev2[:, s], 1.0 / fs)  # rev/s = rpm / 60
            f = new_orders[j] * shaft_hz[s] + total[j]
            live = ((f > 0.0) & (f < nyq)).reshape((-1,) + (1,) * (env.ndim - 2))
            e = np.where(live, env[:, j], 0.0)
            theta = _TWO_PI * (new_orders[j] * rev2[:, s] + total[j] * t)
            carrier = np.exp(1j * theta).reshape((-1,) + (1,) * (env.ndim - 2))
            env[:, j] = e
            wav[:, j] = (e * carrier).real
    if total.any() or "freq_offset" in params:
        params["freq_offset"] = total.tolist()
    params["modify"] = _history(params, record)
    return OrderComponents(
        orders=new_orders,
        shafts=np.array(comp.shafts, dtype=int),
        waveforms=wav,
        envelopes=env,
        rev=np.array(comp.rev, dtype=float),
        method=comp.method,
        params=params,
    )


# --------------------------------------------------------------------------
# remap_rpm
# --------------------------------------------------------------------------


def _speed_columns(rpm, shafts, name):
    """Speed as a float array of shape (n, columns) with a column for every shaft used."""
    r = np.array(rpm, dtype=float)
    if r.ndim == 1:
        if np.any(shafts != 0):
            raise ValueError(
                f"the components refer to several shafts; pass {name} with one column per shaft"
            )
        r = r[:, None]
    elif r.ndim != 2:
        raise ValueError(f"{name} must have shape (n,) or (n, n_shafts)")
    if r.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(r)):
        raise ValueError(f"{name} contains non-finite values; regularise it first")
    if shafts.max() >= r.shape[1]:
        raise ValueError(f"components refer to shaft {int(shafts.max())}, {name} has {r.shape[1]} column(s)")
    return r


def remap_rpm(obj, rpm_new, fs, *, rpm_old=None, mode="time", phase="random", seed=None, rpm_grid=None):
    """Render orders along a new speed curve by remodulating them.

    This is how the apparent speed of a recording is changed (ROADMAP section
    6, item 26): the orders are separated from the residual, carried by new
    carriers ``2 pi order rev_new(t)`` and summed. The audio is never resampled
    or pitch shifted, so every order sits exactly at ``order * rpm_new / 60`` Hz
    and the amplitude of an order follows the law you choose (time course or
    speed law, below), not a stretched copy of the old one.

    *Precision.* A change of 1 km/h is only about 0.25 % in frequency, so the
    speed and the frequency must be exact to far better than that. The carrier
    is the float64 trapezoidal integral of ``rpm_new``
    (:func:`rota_order.speed.rpm_to_phase`, the same phase the analysis uses), so the
    frequency of each rendered order equals ``order * rpm_new / 60`` to rounding
    error: the phase slope of a 12 s segment of the output agrees with it to
    about 1e-9 relative (the tests demand 1e-5, a 0.004 % share of a 1 km/h
    step). The accuracy of the result is the accuracy of ``rpm_new``.

    Input and mode:

    * ``OrderTable``: ``synthesize_orders(obj, rpm_new, fs, phase=phase,
      seed=seed)``. ``mode`` is ignored (a table is always looked up by speed).
    * ``OrderComponents`` with ``mode="time"``: each envelope keeps its time
      course (same recording, apparent speed changed) and modulates the new
      carrier. ``rpm_new`` needs as many samples as the components. With
      ``phase="table"`` the phase of the envelope is kept too, so
      ``rpm_new == rpm_old`` returns the original orders; with the other phase
      modes only ``|envelope|`` is used and the start phases of
      :func:`rota_order.resynth.synthesize_orders` are added.
    * ``OrderComponents`` with ``mode="rpm"``: the envelope is looked up by
      speed (a virtual run-up or run-down from a measured one):
      ``OrderTable.from_components(obj, rpm_old, rpm_grid)`` is built and
      rendered along ``rpm_new``, which can have any length. Speeds outside the
      grid hold the end values (``info["outside_grid_fraction"]``).

    A fixed frequency offset carried by the components
    (``params["freq_offset"]``, see :func:`modify_orders`) is kept in Hz and does
    not scale with speed. Components above Nyquist (or at non-positive
    frequency) are muted, never clipped. The residual is not touched here;
    combine the result with :func:`resynthesize` or
    :func:`rota_order.resynth.synthesize_broadband`.

    Parameters
    ----------
    obj : OrderTable or OrderComponents
        What to render.
    rpm_new : array_like, shape (n_new,) or (n_new, n_shafts)
        Speed in rpm at each output sample (finite). One column per shaft for
        components that refer to several shafts.
    fs : float
        Output sampling rate in Hz. For ``mode="time"`` it must equal the rate
        of the components (``params["fs"]``).
    rpm_old : array_like, shape (n,) or (n, n_shafts), optional
        Speed of the recording the components were extracted from. Required for
        ``mode="rpm"``; for ``mode="time"`` it is only used to report the speed
        ratio in ``info``. Not for tables.
    mode : {"time", "rpm"}, default "time"
        See above (components only).
    phase : {"random", "zero", "table"} or array_like, shape (k,)
        Start phases, as in :func:`rota_order.resynth.synthesize_orders`; ``"table"``
        keeps the phase stored with the data (the envelope phase for
        components).
    seed : int, numpy.random.Generator or None
        Source of the random start phases; the same integer seed gives
        bit-identical output.
    rpm_grid : array_like, shape (m,), optional
        Speed grid in rpm for ``mode="rpm"`` (strictly increasing, at least 2
        points). Default: ``100`` points between the smallest and largest
        ``rpm_old``.

    Returns
    -------
    y : ndarray, shape (n_new,) or (n_new, ch)
        Sum of the rendered orders, *peak*-amplitude sinusoids in the unit of
        the analysed signal (the unit of the table for a table).
    info : dict
        ``source`` (``"table"`` or ``"components"``), ``mode`` (``"time"`` or
        ``"rpm"``), ``fs``, ``n``, ``orders``, ``freq_offset``, ``phase_mode``,
        ``phases`` (rad), ``seed``, ``output_amplitude`` (``"peak"``),
        ``muted_fraction`` per row, ``outside_grid_fraction`` (speed lookups
        only) and, for components, ``shafts``; with ``rpm_old`` in time mode
        ``speed_ratio`` (``min``, ``median``, ``max`` of ``rpm_new / rpm_old``).
        Tables add the entries of :func:`rota_order.resynth.synthesize_orders`.

    Raises
    ------
    ValueError
        For an unknown ``mode``, a missing ``rpm_old`` in ``mode="rpm"``, a
        ``rpm_new`` of the wrong length in ``mode="time"``, non-finite speeds, a
        sampling rate that differs from the components', or ``rpm_old`` /
        ``rpm_grid`` given with a table.

    References
    ----------
    McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis based on
    a sinusoidal representation. IEEE Transactions on Acoustics, Speech, and
    Signal Processing, 34(4), 744-754.
    Serra, X., & Smith, J. O. (1990). Spectral modeling synthesis: a sound
    analysis/synthesis system based on a deterministic plus stochastic
    decomposition. Computer Music Journal, 14(4), 12-24.
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    """
    fs = _check_fs(fs)
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if isinstance(obj, OrderTable):
        if rpm_old is not None or rpm_grid is not None:
            raise ValueError("rpm_old and rpm_grid are for components; a table is looked up by speed")
        y, info = synthesize_orders(obj, rpm_new, fs, phase=phase, seed=seed)
        r = np.asarray(rpm_new, dtype=float)
        info = dict(info)
        info.update(
            source="table",
            mode="rpm",
            outside_grid_fraction=float(np.mean((r < obj.rpm[0]) | (r > obj.rpm[-1]))),
        )
        return y, info
    if not isinstance(obj, OrderComponents):
        raise TypeError(f"obj must be an OrderTable or OrderComponents, got {type(obj).__name__}")
    if mode == "time":
        return _remap_time(obj, rpm_new, fs, rpm_old, phase, seed)
    return _remap_by_speed(obj, rpm_new, fs, rpm_old, rpm_grid, phase, seed)


def _remap_time(comp, rpm_new, fs, rpm_old, phase, seed):
    env = np.asarray(comp.envelopes, dtype=complex)
    if env.ndim not in (2, 3):
        raise ValueError("envelopes must have shape (n, k) or (n, k, ch)")
    mono = env.ndim == 2
    E = env[..., None] if mono else env
    n, k, ch = E.shape
    orders = np.asarray(comp.orders, dtype=float)
    shafts = np.asarray(comp.shafts, dtype=int)
    if orders.shape != (k,) or shafts.shape != (k,):
        raise ValueError("comp.orders and comp.shafts must match the envelopes")
    fs_rec = comp.params.get("fs")
    if fs_rec is not None and not np.isclose(float(fs_rec), fs, rtol=1e-12, atol=0.0):
        raise ValueError(
            f"fs = {fs:g} Hz differs from the components' rate {float(fs_rec):g} Hz; "
            "mode='time' keeps the time axis of the envelopes"
        )
    R = _speed_columns(rpm_new, shafts, "rpm_new")
    if R.shape[0] != n:
        raise ValueError(
            f"mode='time' keeps the time course of the envelopes: rpm_new needs {n} samples "
            f"like the components, got {R.shape[0]} (use mode='rpm' for another length)"
        )
    off = _component_offsets(comp, k)
    ph_mode, start, used = _resolve_phase(phase, k, seed)
    rev = rpm_to_phase(R, fs)
    t = np.arange(n) / fs
    nyq = 0.5 * fs
    keep_phase = ph_mode == "table"
    y = np.zeros((n, ch))
    muted = np.zeros(k, dtype=np.int64)
    for j in range(k):
        s = int(shafts[j])
        f = orders[j] * R[:, s] / 60.0 + off[j]
        live = (f > 0.0) & (f < nyq)  # muted, never clipped (ROADMAP section 6, item 16)
        muted[j] = n - int(live.sum())
        if not live.any():
            continue
        theta = _TWO_PI * (orders[j] * rev[:, s] + off[j] * t) + start[j]
        e = E[:, j, :] if keep_phase else np.abs(E[:, j, :])
        y += (e * np.exp(1j * theta)[:, None]).real * live[:, None]
    info = {
        "source": "components",
        "mode": "time",
        "fs": fs,
        "n": int(n),
        "orders": orders.copy(),
        "shafts": shafts.copy(),
        "freq_offset": off.copy(),
        "phase_mode": ph_mode,
        "phases": start.copy(),
        "seed": used,
        "output_amplitude": "peak",
        "muted_fraction": muted / n,
    }
    if rpm_old is not None:
        old = _speed_columns(rpm_old, shafts, "rpm_old")
        if old.shape[0] != n:
            raise ValueError(f"rpm_old needs {n} samples like the components, got {old.shape[0]}")
        cols = np.unique(shafts)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = R[:, cols] / old[:, cols]
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size:
            info["speed_ratio"] = {
                "min": float(ratio.min()),
                "median": float(np.median(ratio)),
                "max": float(ratio.max()),
            }
    return (y[:, 0] if mono else y), info


def _remap_by_speed(comp, rpm_new, fs, rpm_old, rpm_grid, phase, seed):
    if rpm_old is None:
        raise ValueError("mode='rpm' needs rpm_old, the speed of the recording")
    orders = np.asarray(comp.orders, dtype=float)
    shafts = np.asarray(comp.shafts, dtype=int)
    k = orders.size
    R = _speed_columns(rpm_new, shafts, "rpm_new")
    old = np.asarray(rpm_old, dtype=float)
    if rpm_grid is None:
        finite = old[np.isfinite(old)]
        if finite.size == 0 or not finite.max() > finite.min():
            raise ValueError("rpm_old is constant or empty; pass rpm_grid")
        grid = np.linspace(finite.min(), finite.max(), _GRID_POINTS)
    else:
        grid = np.asarray(rpm_grid, dtype=float)
    table = OrderTable.from_components(comp, old, grid)  # validates shapes and the grid
    off = _component_offsets(comp, k)
    ph_mode, start, used = _resolve_phase(phase, k, seed)

    y = None
    muted = np.zeros(k)
    empty = np.zeros(k, dtype=bool)
    outside = []
    for s in np.unique(shafts):
        rows = np.flatnonzero(shafts == s)
        sub = OrderTable(
            orders=table.orders[rows],
            rpm=table.rpm,
            values=table.values[rows],
            amplitude="peak",
            freq_offset=off[rows],
            unit=table.unit,
            meta=table.meta,
        )
        ys, inf = synthesize_orders(
            sub, R[:, s], fs, phase="table" if ph_mode == "table" else start[rows]
        )
        y = ys if y is None else y + ys
        muted[rows] = inf["muted_fraction"]
        empty[rows] = inf["empty_rows"]
        outside.append(np.mean((R[:, s] < grid[0]) | (R[:, s] > grid[-1])))
    info = {
        "source": "components",
        "mode": "rpm",
        "fs": fs,
        "n": int(R.shape[0]),
        "orders": orders.copy(),
        "shafts": shafts.copy(),
        "freq_offset": off.copy(),
        "rpm_grid": np.array(table.rpm),
        "phase_mode": ph_mode,
        "phases": start.copy(),
        "seed": used,
        "table_amplitude": "peak",
        "output_amplitude": "peak",
        "muted_fraction": muted,
        "empty_rows": empty,
        "outside_grid_fraction": float(np.mean(outside)),
    }
    return y, info


# --------------------------------------------------------------------------
# resynthesize
# --------------------------------------------------------------------------


def _gain_value(value, name):
    v = float(value)
    if np.isnan(v) or v == np.inf:
        raise ValueError(f"{name} must be a finite gain in dB (or -inf), got {value!r}")
    return v


def _energy(a):
    return float(np.sum(a * a))


def resynthesize(
    components, residual, *, order_gain_db=0.0, residual_gain_db=0.0, ratio_db=None, reference_rms=None
):
    """Recombine the orders and the residual with explicit gains.

    The orders are the sum of the order waveforms (``components.total``, or an
    array of that shape), the residual is the broadband remainder of the same
    signal (for example from :func:`rota_order.extract.residual`). With

    ``orders = total * 10 ** (order_gain_db / 20)``,
    ``resid = residual * 10 ** (residual_gain_db / 20)``,

    the result is ``mix = orders + resid``, exactly. The three signals are
    returned separately, so a listening test can present orders only, residual
    only and the mix at the same gain.

    ``ratio_db`` fixes the energy ratio of the two parts instead of the order
    gain: the order gain is chosen such that ``10 log10(sum(orders**2) /
    sum(resid**2))`` equals ``ratio_db`` (energies summed over all samples and
    channels, a single gain for all channels so that the spatial relation
    between channels is kept). It overrides ``order_gain_db``; the residual
    gain still applies first.

    ``reference_rms`` sets the overall level. A single scale factor is computed
    from the *gain-0* mix, ``components.total + residual``, so that this mix has
    RMS ``reference_rms``, and the same factor multiplies the orders and the
    residual. Relative gains therefore survive. (Normalising the already gained
    mix by its own RMS would cancel the gain: ROADMAP section 6, item 12.) The
    RMS is ``sqrt(mean(x ** 2))`` over all samples and channels, without
    removing the mean, in the unit of the inputs.

    Parameters
    ----------
    components : OrderComponents or array_like, shape (n,) or (n, ch)
        Extracted orders, or their summed waveform.
    residual : array_like, same shape
        Broadband residual, same unit.
    order_gain_db, residual_gain_db : float, default 0
        Gains in dB applied to the orders and to the residual (``-inf`` mutes).
    ratio_db : float, optional
        Target energy ratio orders / residual in dB (after the residual gain).
    reference_rms : float, optional
        RMS in the unit of the inputs that the gain-0 mix is scaled to. Without
        it no normalisation is applied (``scale = 1``).

    Returns
    -------
    dict
        ``orders``, ``residual``, ``mix`` (``orders + residual`` exactly): the
        scaled signals in the unit of the inputs; ``scale``: the normalisation
        factor (1 without ``reference_rms``); ``ratio_db``: the achieved energy
        ratio of orders to residual in dB (``inf`` / ``-inf`` if the residual /
        orders have no energy); ``gains_db``: ``{"orders", "residual",
        "normalisation"}``, the gains actually applied in dB (the
        normalisation is ``20 log10(scale)``); ``rms``: ``{"orders",
        "residual", "mix"}`` of the outputs.

    Raises
    ------
    ValueError
        For mismatching shapes, non-finite data, a non-finite ``ratio_db``,
        a non-positive ``reference_rms``, a ``ratio_db`` when the orders or the
        residual have no energy, or ``reference_rms`` when the gain-0 mix has
        none.

    Notes
    -----
    Amplitude convention: RMS and energy are used for the ratio and the
    normalisation, which makes the result independent of whether the orders are
    thought of as peak or RMS amplitudes. The order waveforms of
    :class:`~rota_order.extract.OrderComponents` are peak-amplitude sinusoids; their
    time-domain sum is what is scaled.

    References
    ----------
    Serra, X., & Smith, J. O. (1990). Spectral modeling synthesis: a sound
    analysis/synthesis system based on a deterministic plus stochastic
    decomposition. Computer Music Journal, 14(4), 12-24.
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 2. Wiley.
    """
    total = components.total if isinstance(components, OrderComponents) else components
    o = np.asarray(total, dtype=float)
    r = np.asarray(residual, dtype=float)
    if o.ndim not in (1, 2) or o.size == 0:
        raise ValueError("the orders must have shape (n,) or (n, ch)")
    if r.shape != o.shape:
        raise ValueError(f"residual has shape {r.shape}, the orders have shape {o.shape}")
    if not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
        raise ValueError("orders and residual must be finite")
    g_res = _gain_value(residual_gain_db, "residual_gain_db")
    g_ord = _gain_value(order_gain_db, "order_gain_db")
    with np.errstate(over="ignore"):
        lin_res = 10.0 ** (g_res / 20.0)
    if ratio_db is not None:
        ratio_db = float(ratio_db)
        if not np.isfinite(ratio_db):
            raise ValueError("ratio_db must be finite")
        e_ord, e_res = _energy(o), _energy(r * lin_res)
        if not (e_ord > 0.0 and e_res > 0.0):
            raise ValueError("ratio_db needs orders and residual that both have energy")
        lin_ord = np.sqrt(10.0 ** (ratio_db / 10.0) * e_res / e_ord)
        g_ord = float(20.0 * np.log10(lin_ord))
    else:
        with np.errstate(over="ignore"):
            lin_ord = 10.0 ** (g_ord / 20.0)

    scale = 1.0
    if reference_rms is not None:
        reference_rms = float(reference_rms)
        if not (np.isfinite(reference_rms) and reference_rms > 0.0):
            raise ValueError("reference_rms must be positive and finite")
        rms0 = float(np.sqrt(np.mean((o + r) ** 2)))
        if not rms0 > 0.0:
            raise ValueError("the gain-0 mix has no energy; reference_rms cannot be applied")
        scale = reference_rms / rms0

    orders_out = o * (lin_ord * scale)
    resid_out = r * (lin_res * scale)
    mix = orders_out + resid_out
    e_o, e_r = _energy(orders_out), _energy(resid_out)
    if e_r > 0.0 and e_o > 0.0:
        achieved = float(10.0 * np.log10(e_o / e_r))
    elif e_o > 0.0:
        achieved = float("inf")
    elif e_r > 0.0:
        achieved = float("-inf")
    else:
        achieved = float("nan")
    return {
        "orders": orders_out,
        "residual": resid_out,
        "mix": mix,
        "scale": float(scale),
        "ratio_db": achieved,
        "gains_db": {
            "orders": g_ord,
            "residual": g_res,
            "normalisation": float(20.0 * np.log10(scale)),
        },
        "rms": {
            "orders": float(np.sqrt(np.mean(orders_out**2))),
            "residual": float(np.sqrt(np.mean(resid_out**2))),
            "mix": float(np.sqrt(np.mean(mix**2))),
        },
    }
