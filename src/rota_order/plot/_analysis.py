"""Figures of speed tracks, extracted orders, order tables and tonal analyses.

Units: time in s, speed in rpm, order in cycles per revolution, frequency in
Hz, levels in dB. Amplitude levels are ``20 log10(|a| / ref)`` of values that
keep their amplitude convention (peak for the envelopes of the order extractors,
``OrderTable.amplitude`` for a table); results that already hold levels or
ratios in dB (tone-to-noise and prominence ratios, band levels) are drawn as
they are. NaN stays a gap: nothing is interpolated or smoothed.
"""

from contextlib import contextmanager

import numpy as np

from ..extract import OrderComponents
from ..resynth import OrderTable
from ..speed import RidgeResult
from ..tonal import BandRpmLevels, OrderProminence
from ._core import (
    X_LABELS,
    check_finite,
    check_ref,
    colour_limits,
    draw_reference,
    edges_from_centres,
    get_ax,
    legend_without_repeats,
    level_label,
    level_reference,
    pyplot,
    select_channel,
    to_db,
)

__all__ = [
    "band_levels_vs_rpm",
    "order_components",
    "order_prominence",
    "order_table",
    "rpm_from_ridge",
    "rpm_vs_time",
]

_SCALES = ("db", "linear")
_KINDS = {"tnr": "Tone-to-noise ratio (dB)", "pr": "Prominence ratio (dB)"}
_CRITERION_LABEL = "ECMA-418-1 criterion"
# Markers of the doubtful frames of a ridge: fixed colours outside the default colour cycle (the
# reference takes a neutral grey), a large open circle and a small cross on top of it.
_BOUND_COLOUR = "#b2182b"
_LOW_COLOUR = "black"
_CONVENTION_TEXT = {"peak": "peak", "rms": "RMS"}
# Opacity of the shaded prominent ranges: light enough that two overlapping orders stay distinguishable.
_SHADE_ALPHA = 0.12


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


@contextmanager
def _drawing(ax):
    """Yield ``(fig, ax)`` and close a figure created here if drawing fails.

    Call it after the arguments have been validated; Matplotlib is imported
    only when ``ax`` is None.
    """
    fig, ax_ = get_ax(ax)
    try:
        yield fig, ax_
    except Exception:
        if ax is None:
            pyplot().close(fig)  # do not leave a half-drawn figure behind
        raise


def _vector(a, name, *, allow_nan=False):
    """``a`` as a non-empty 1-D float array; ValueError otherwise (and for infinity)."""
    try:
        v = np.asarray(a, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a 1-D array of numbers") from None
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {v.shape}")
    bad = np.isinf(v) if allow_nan else ~np.isfinite(v)
    if np.any(bad):
        raise ValueError(f"{name} must be finite" + (" or NaN (a gap)" if allow_nan else ""))
    return v


def _reference_pair(reference):
    """``(t_ref, rpm_ref)`` from the ``reference`` option, or None."""
    if reference is None:
        return None
    try:
        t_ref, rpm_ref = reference
    except (TypeError, ValueError):
        raise ValueError("reference must be a pair (t_ref, rpm_ref) of arrays") from None
    t_ref = _vector(t_ref, "reference time")
    rpm_ref = _vector(rpm_ref, "reference rpm", allow_nan=True)
    if t_ref.shape != rpm_ref.shape:
        raise ValueError(
            f"reference time and rpm must have the same length, got {t_ref.size} and {rpm_ref.size}"
        )
    return t_ref, rpm_ref


def _check_choice(value, choices, name):
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")
    return value


def _pick_orders(available, orders):
    """Indices of the rows of ``available`` that match the requested ``orders``.

    None selects every row. Otherwise each requested order selects all rows with
    that order (a table may hold the same order with different offsets), in the
    order of the request; ValueError names the orders that are not present.
    """
    available = np.asarray(available, dtype=float)
    if orders is None:
        return list(range(available.size))
    try:
        wanted = np.atleast_1d(np.asarray(orders, dtype=float))
    except (TypeError, ValueError):
        raise ValueError("orders must be a number or a sequence of numbers") from None
    if wanted.ndim != 1 or wanted.size == 0 or not np.all(np.isfinite(wanted)):
        raise ValueError("orders must be a non-empty sequence of finite numbers (cycles per revolution)")
    index, missing = [], []
    for value in wanted:
        hit = np.flatnonzero(np.isclose(available, value, rtol=1e-9, atol=1e-12))
        if hit.size == 0:
            missing.append(f"{value:g}")
        index.extend(int(h) for h in hit if int(h) not in index)
    if missing:
        have = ", ".join(f"{v:g}" for v in available)
        raise ValueError(f"orders {', '.join(missing)} are not in the result (it holds {have})")
    return index


def _convention(value):
    """Amplitude convention as ``level_label`` expects it (lower case), ValueError if unknown."""
    key = value.lower() if isinstance(value, str) else value
    if key not in _CONVENTION_TEXT:
        raise ValueError(f"amplitude convention must be 'peak' or 'rms', got {value!r}")
    return key


def _linear_label(quantity, unit, convention):
    """Axis label of a linear amplitude, e.g. ``"Order amplitude (Pa, RMS)"``."""
    parts = [str(unit)] if unit else []
    parts.append(_CONVENTION_TEXT[convention])
    return f"{quantity} ({', '.join(parts)})"


def _order_name(order, offset=0.0, shaft=None):
    name = f"Order {order:g}"
    if offset:
        name += f" {offset:+g} Hz"
    if shaft is not None:
        name += f" (shaft {shaft})"
    return name


def _amplitude_rows(values, scale, ref):
    """Magnitude of complex rows, as a level in dB re ``ref`` or linear (a copy; NaN stays NaN)."""
    mag = np.abs(np.ascontiguousarray(values))  # contiguous: np.abs of a strided complex view is not reproducible
    return to_db(mag, ref) if scale == "db" else mag


# --------------------------------------------------------------------------
# Speed
# --------------------------------------------------------------------------


def rpm_vs_time(t, rpm, *, ax=None, reference=None, label=None):
    """Draw a speed curve against time, e.g. the output of ``tacho_to_rpm``.

    The figure of :func:`rota_order.speed.tacho_to_rpm` (or of any speed given as
    arrays). Nothing is smoothed; NaN values are gaps. Units: time in s, speed
    in rpm (no amplitude is involved).

    Parameters
    ----------
    t : array_like, shape (n,)
        Times in s, finite.
    rpm : array_like, shape (n,)
        Speed in rpm at ``t``; NaN is drawn as a gap.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created only when None.
    reference : (array_like, array_like), optional
        ``(t_ref, rpm_ref)``: a reference speed (for example the analytic truth
        of a synthetic run-up), labelled "reference" and drawn in the reference
        style shared with ``truth=`` of the slice figures: dashed, neutral dark
        grey, thinner than the curve and above it.
    label : str, optional
        Legend text of the curve. A legend is added if there is a label or a
        reference.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, with the x label "Time (s)" and the y label "Speed
        (rpm)". The figure is not shown, and the backend, style and ``rcParams``
        are not touched.

    Raises
    ------
    ImportError
        If Matplotlib is not installed (``pip install 'rota-order[plot]'``).
    TypeError
        If ``ax`` is not an Axes.
    ValueError
        If ``t`` and ``rpm`` are not 1-D arrays of the same length, ``t`` is not
        finite, or ``reference`` is not a pair of such arrays.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, sec. 12.2 (speed from
    tachometer signals). Wiley.
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95.
    """
    t = _vector(t, "t")
    rpm = _vector(rpm, "rpm", allow_nan=True)
    if t.shape != rpm.shape:
        raise ValueError(f"t and rpm must have the same length, got {t.size} and {rpm.size}")
    ref_pair = _reference_pair(reference)
    with _drawing(ax) as (_, ax_):
        ax_.plot(t, rpm, label=label)
        if ref_pair is not None:
            draw_reference(ax_, ref_pair[0], ref_pair[1], label="reference")
        ax_.set_xlabel(X_LABELS["time"])
        ax_.set_ylabel("Speed (rpm)")
        legend_without_repeats(ax_)  # a legend if there is a label or a reference
    return ax_


def rpm_from_ridge(
    r, *, ax=None, reference=None, show_unreliable=True, min_confidence=0.1, label="ridge track", marker=None
):
    """Draw the speed track of :func:`rota_order.speed.rpm_from_ridge` with its doubtful frames.

    The track ``r.rpm`` is drawn against the frame times. With
    ``show_unreliable=True`` the frames that must not be trusted are made visible
    on the same axes: frames flagged ``at_bound`` or ``beyond`` (the estimate sits
    at a bound of the search range, or the data prefer a speed outside it; the
    legend entry "flagged at_bound" covers both flags, not only the frames close
    to a bound) get large open dark-red circles, and frames whose ``confidence`` is below
    ``min_confidence`` get small black crosses drawn on top, so that a frame with
    both flags shows a cross inside a circle; the colours of the markers are fixed
    and not taken from the colour cycle. When frames sit at a bound, the search
    range ``r.rpm_range`` is drawn as dotted lines. The track itself is not
    altered. Units: time in s, speed in rpm (no amplitude convention applies).

    Parameters
    ----------
    r : rota_order.speed.RidgeResult
        Result of :func:`rota_order.speed.rpm_from_ridge`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created only when None.
    reference : (array_like, array_like), optional
        ``(t_ref, rpm_ref)``, for example the analytic or tachometer speed,
        labelled "reference" and drawn in the common reference style (dashed,
        neutral dark grey, thinner than the track and above it), as in
        :func:`rota_order.plot.rpm_vs_time`.
    show_unreliable : bool, default True
        Mark the frames flagged ``at_bound`` or ``beyond`` and the frames of low
        confidence.
    min_confidence : float, default 0.1
        Confidence below which a frame is marked, in [0, 1]. The result defines
        no per-frame threshold; 0.1 is the "low confidence" of
        ``r.quality["low_confidence_fraction"]`` (frames below 0.1). A frame
        whose confidence is NaN counts as low.
    label : str, default "ridge track"
        Legend text of the track, as in :func:`rota_order.plot.rpm_vs_time`; None leaves
        the track out of the legend.
    marker : str, optional
        Matplotlib marker of the track (``"."``, ``"o"``, ...). With a marker the
        track is drawn as markers only (``linestyle="none"``), one per frame, so
        that a dense reference curve (a tachometer speed) stays visible under it;
        None (default) draws a line.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, with the labels "Time (s)" and "Speed (rpm)" and a
        legend. The figure is not shown, and the backend, style and ``rcParams``
        are not touched.

    Raises
    ------
    ImportError
        If Matplotlib is not installed (``pip install 'rota-order[plot]'``).
    TypeError
        If ``r`` is not a ``RidgeResult`` or ``ax`` is not an Axes.
    ValueError
        For an invalid ``reference`` or a ``min_confidence`` outside [0, 1].

    Notes
    -----
    ``at_bound`` and ``confidence`` are different tests: a track that follows a
    wrong speed inside the range (an octave, a rival comb) can have a high
    confidence and only ``at_bound`` (through ``beyond``) tells; conversely a
    low confidence does not mean a wrong value. Read the two markers together.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, sec. 12.2. Wiley.
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95.
    """
    if not isinstance(r, RidgeResult):
        raise TypeError(f"rpm_from_ridge expects a rota_order.speed.RidgeResult, got {type(r).__name__}")
    t = np.asarray(r.t, dtype=float)
    rpm = np.asarray(r.rpm, dtype=float)
    confidence = np.asarray(r.confidence, dtype=float)
    bound = np.asarray(r.at_bound, dtype=bool) | np.asarray(r.beyond, dtype=bool)
    if t.ndim != 1 or t.size == 0 or not (rpm.shape == confidence.shape == bound.shape == t.shape):
        raise ValueError("t, rpm, confidence, at_bound and beyond of the result must be 1-D and of one length")
    threshold = check_finite(min_confidence, "min_confidence")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"min_confidence must lie in [0, 1], got {min_confidence!r}")
    ref_pair = _reference_pair(reference)
    low = ~(confidence >= threshold)
    with _drawing(ax) as (_, ax_):
        if marker is None:
            ax_.plot(t, rpm, label=label)
        else:
            ax_.plot(t, rpm, linestyle="none", marker=marker, markersize=4, label=label)
        if show_unreliable:
            if np.any(bound):
                ax_.plot(t[bound], rpm[bound], ls="none", marker="o", mfc="none", ms=9, mew=1.5,
                         color=_BOUND_COLOUR, zorder=3, label="flagged at_bound")
                for k, edge in enumerate(r.rpm_range):
                    ax_.axhline(float(edge), color="0.5", lw=0.8, ls=":", label="search range" if k == 0 else None)
            if np.any(low):
                ax_.plot(t[low], rpm[low], ls="none", marker="x", ms=5, mew=1.3, color=_LOW_COLOUR,
                         zorder=3.5, label=f"confidence < {threshold:g}")
        if ref_pair is not None:
            draw_reference(ax_, ref_pair[0], ref_pair[1], label="reference")
        ax_.set_xlabel(X_LABELS["time"])
        ax_.set_ylabel("Speed (rpm)")
        legend_without_repeats(ax_)
    return ax_


# --------------------------------------------------------------------------
# Extracted orders and order tables
# --------------------------------------------------------------------------


def _shaft_speed_columns(rpm, t_rpm, n, fs):
    """Speed of every sample as an ``(n, S)`` array, whether ``rpm`` was 1-D, and the samples it covers.

    ``rpm`` has one value per sample (shape ``(n,)`` or ``(n, S)``), or, with
    ``t_rpm``, one value per time of ``t_rpm`` (for example ``RidgeResult.t``),
    linearly interpolated to the sample times ``arange(n) / fs``. The third item
    is a boolean mask of the samples with a known speed: all of them, or with
    ``t_rpm`` those within its span (outside it the speed would be a held value,
    not data).
    """
    try:
        r = np.asarray(rpm, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("rpm must be an array of numbers") from None
    if r.ndim not in (1, 2) or r.shape[0] == 0:
        raise ValueError(f"rpm must have shape (n,) or (n, n_shafts), got {r.shape}")
    one_d = r.ndim == 1
    r = r[:, None] if one_d else r
    if t_rpm is None:
        if r.shape[0] != n:
            raise ValueError(
                f"rpm has {r.shape[0]} values but the components have {n} samples; give one value per "
                "sample, or the times of rpm in t_rpm"
            )
        return r, one_d, np.ones(n, dtype=bool)
    tq = _vector(t_rpm, "t_rpm")
    if tq.size != r.shape[0]:
        raise ValueError(f"t_rpm has {tq.size} values but rpm has {r.shape[0]}")
    if tq.size > 1 and not np.all(np.diff(tq) > 0):
        raise ValueError("t_rpm must be strictly increasing")
    if fs is None:
        raise ValueError("the sampling rate of the components (params['fs']) is needed to place t_rpm")
    t = np.arange(n) / float(fs)
    columns = np.stack([np.interp(t, tq, r[:, j]) for j in range(r.shape[1])], axis=1)
    return columns, one_d, (t >= tq[0]) & (t <= tq[-1])


def order_components(
    c,
    *,
    ax=None,
    orders=None,
    ref=1.0,
    unit="",
    x="rpm",
    rpm=None,
    t_rpm=None,
    scale="db",
    channel=0,
):
    """Draw the envelopes of extracted orders, against speed or time.

    The figure of the result of :func:`rota_order.extract.order_waveform` or
    :func:`rota_order.extract.tracked_harmonics`. ``c.envelopes`` are complex *peak*
    envelopes in the signal unit (``params["amplitude"]``, "peak"); one line per
    order shows ``20 log10(|envelope| / ref)`` (``scale="db"``) or ``|envelope|``
    (``scale="linear"``). The components carry the sample index and the shaft
    angle, not the speed: the speed is needed to put them on an rpm axis and is
    given in ``rpm``. Nothing is smoothed; a zero amplitude is a gap in dB.

    Parameters
    ----------
    c : rota_order.extract.OrderComponents
        Result of an order extractor; envelopes of shape ``(n, k)`` or
        ``(n, k, ch)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created only when None.
    orders : float or sequence of float, optional
        Orders to draw (all rows of ``c.orders`` that match); default all.
    ref : float, default 1.0
        Reference of the dB scale in the unit of the signal (``2e-5`` for sound
        pressure in Pa), explicit and never assumed.
    unit : str, default ""
        Unit of the signal, used in the label only; ``rota_order.plot.SOUND_PRESSURE``
        gives ``ref`` and ``unit`` together.
    x : {"rpm", "time"}, default "rpm"
        Quantity on the x axis. ``"rpm"`` needs ``rpm``; the lines are drawn at
        the true speeds, which may double back (a measured speed has small
        reversals, a gear shift a large drop), and that is the data.
        ``"time"`` is ``arange(n) / fs`` with ``fs = c.params["fs"]``.
    rpm : array_like, shape (n,) or (n, n_shafts), optional
        Speed in rpm at every sample, as passed to the extractor (a 1-D array
        is used for the orders of one shaft; with several shafts give one
        column per shaft, and each order is drawn against its own shaft). With
        ``t_rpm``, the speed at those times instead.
    t_rpm : array_like, shape (m,), optional
        Strictly increasing times in s of the values of ``rpm`` (for example
        ``RidgeResult.t`` with ``RidgeResult.rpm``); the speed is interpolated
        linearly to the sample times. Samples outside the span of ``t_rpm`` have
        no speed and are not drawn.
    scale : {"db", "linear"}, default "db"
        Level in dB re ``ref`` or the amplitude in the signal unit.
    channel : int, default 0
        Channel of multichannel components.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, labelled "Order level (dB re <ref>, peak)" (or
        "Order amplitude (<unit>, peak)") against "Speed (rpm)" or "Time (s)",
        with a legend entry per order. The figure is not shown, and the backend,
        style and ``rcParams`` are not touched.

    Raises
    ------
    ImportError
        If Matplotlib is not installed (``pip install 'rota-order[plot]'``).
    TypeError
        If ``c`` is not an ``OrderComponents``, ``channel`` is not an integer or
        ``ax`` is not an Axes.
    ValueError
        If ``x="rpm"`` is requested without ``rpm`` (the message says what to
        pass), if ``rpm`` does not fit the samples or the shafts, if ``orders`` are not in the result, if
        the sampling rate is needed but missing from ``c.params``, or for an
        invalid ``x``, ``scale`` or ``ref``.
    IndexError
        If ``channel`` is out of range.

    Notes
    -----
    The extractors are least reliable within a window or a filter settling time
    of the ends of the record (``params["edge_revs"]`` for the tracked
    harmonics); the envelopes are drawn as they are.

    With ``x="rpm"`` the lines follow the speed as given: a measured speed (a
    tachometer, a ridge) has small reversals, and a line over a speed that goes
    up and down (a gear shift, a run-up followed by a coast-down) doubles back
    on itself. Use ``x="time"`` for those.

    References
    ----------
    Vold, H., & Leuridan, J. (1993). High resolution order tracking at extreme
    slew rates, using Kalman tracking filters. SAE Technical Paper 931288.
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95.
    """
    if not isinstance(c, OrderComponents):
        raise TypeError(f"order_components expects a rota_order.extract.OrderComponents, got {type(c).__name__}")
    _check_choice(x, ("rpm", "time"), "x")
    _check_choice(scale, _SCALES, "scale")
    ref = check_ref(ref)
    params = c.params if isinstance(c.params, dict) else {}
    convention = _convention(params.get("amplitude", "peak"))
    env = np.asarray(c.envelopes)
    if env.ndim not in (2, 3):
        raise ValueError("envelopes must have shape (n, k) or (n, k, ch)")
    env = select_channel(env, channel, 2)
    n, k = env.shape
    all_orders = np.asarray(c.orders, dtype=float)
    shafts = np.asarray(c.shafts, dtype=int)
    if all_orders.shape != (k,) or shafts.shape != (k,):
        raise ValueError("orders and shafts of the components do not match their envelopes")
    index = _pick_orders(all_orders, orders)
    fs = params.get("fs")

    xs = {}  # order index -> x values
    if x == "time":
        if fs is None:
            raise ValueError("x='time' needs the sampling rate, but c.params has no 'fs'")
        t = np.arange(n) / float(fs)
        keep = np.ones(n, dtype=bool)
        xs = dict.fromkeys(index, t)
        x_label = X_LABELS["time"]
    else:
        if rpm is None:
            raise ValueError(
                "x='rpm' needs the speed, which an OrderComponents does not carry: pass rpm= (the speed "
                "in rpm at every sample, as given to the extractor) or rpm= with t_rpm= (its times, "
                "e.g. a RidgeResult), or use x='time'"
            )
        columns, one_d, keep = _shaft_speed_columns(rpm, t_rpm, n, fs)
        used = sorted({int(shafts[j]) for j in index})
        if one_d and len(used) > 1:
            raise ValueError(
                f"the orders belong to shafts {used}: give rpm with one column per shaft, shape (n, n_shafts)"
            )
        for j in index:
            col = 0 if one_d else int(shafts[j])
            if col >= columns.shape[1]:
                raise ValueError(f"order {all_orders[j]:g} belongs to shaft {col}, but rpm has {columns.shape[1]} column(s)")
            xs[j] = columns[keep, col]
        x_label = X_LABELS["rpm"]
    y = _amplitude_rows(env, scale, ref)[keep]
    multi_shaft = len({int(shafts[j]) for j in index}) > 1
    with _drawing(ax) as (_, ax_):
        for j in index:
            ax_.plot(xs[j], y[:, j], label=_order_name(all_orders[j], shaft=int(shafts[j]) if multi_shaft else None))
        ax_.set_xlabel(x_label)
        ax_.set_ylabel(
            level_label("Order level", ref, unit, convention)
            if scale == "db"
            else _linear_label("Order amplitude", unit, convention)
        )
        ax_.legend()
    return ax_


def order_table(tbl, *, ax=None, orders=None, ref=1.0, unit="", scale="db", channel=0):
    """Draw the rows of an order table against speed.

    The figure of an :class:`rota_order.resynth.OrderTable` (from
    :meth:`~rota_order.resynth.OrderTable.from_slices` or built by hand, the input of
    :func:`rota_order.resynth.synthesize_orders`): one line per row, the magnitude
    ``|values|`` in the table's amplitude convention (``tbl.amplitude``, peak or
    RMS), as ``20 log10(|values| / ref)`` (``scale="db"``) or linear. NaN (a grid
    point without data) is a gap. A row with a fixed frequency offset is named
    "Order 2 +300 Hz".

    Parameters
    ----------
    tbl : rota_order.resynth.OrderTable
        Table with values of shape ``(k, m)`` or ``(k, m, ch)`` on the speed grid
        ``tbl.rpm``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created only when None.
    orders : float or sequence of float, optional
        Orders to draw (all rows that match, whatever their offset); default all.
    ref : float, default 1.0
        Reference of the dB scale in the unit of the table, explicit and never
        assumed.
    unit : str, default ""
        Unit of the table in the label. If empty, ``tbl.unit`` is used, so that
        ``ref`` is read in the unit of the table (``ref=2e-5`` of a table in Pa
        reads "dB re 20 µPa").
    scale : {"db", "linear"}, default "db"
        Level in dB re ``ref`` or the amplitude in the unit of the table.
    channel : int, default 0
        Channel of a multichannel table.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, labelled "Order level (dB re <ref>, <peak or RMS>)" (or
        "Order amplitude (<unit>, <peak or RMS>)") against "Speed (rpm)", with a
        legend entry per row. The figure is not shown, and the backend, style and
        ``rcParams`` are not touched.

    Raises
    ------
    ImportError
        If Matplotlib is not installed (``pip install 'rota-order[plot]'``).
    TypeError
        If ``tbl`` is not an ``OrderTable``, ``channel`` is not an integer or
        ``ax`` is not an Axes.
    ValueError
        If ``orders`` are not in the table, or for an invalid ``scale`` or
        ``ref``.
    IndexError
        If ``channel`` is out of range.

    References
    ----------
    McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis based on
    a sinusoidal representation. IEEE Transactions on Acoustics, Speech, and
    Signal Processing, 34(4), 744-754.
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95.
    """
    if not isinstance(tbl, OrderTable):
        raise TypeError(f"order_table expects a rota_order.resynth.OrderTable, got {type(tbl).__name__}")
    _check_choice(scale, _SCALES, "scale")
    ref = check_ref(ref)
    convention = _convention(tbl.amplitude)
    values = select_channel(tbl.values, channel, 2)
    index = _pick_orders(tbl.orders, orders)
    y = _amplitude_rows(values, scale, ref)
    unit = unit if unit else tbl.unit
    marker = "o" if tbl.rpm.size == 1 else None
    with _drawing(ax) as (_, ax_):
        for j in index:
            ax_.plot(tbl.rpm, y[j], marker=marker, label=_order_name(tbl.orders[j], float(tbl.freq_offset[j])))
        ax_.set_xlabel(X_LABELS["rpm"])
        ax_.set_ylabel(
            level_label("Order level", ref, unit, convention)
            if scale == "db"
            else _linear_label("Order amplitude", unit, convention)
        )
        ax_.legend()
    return ax_


# --------------------------------------------------------------------------
# Tonal analysis
# --------------------------------------------------------------------------


def order_prominence(p, *, ax=None, kind="tnr", orders=None, criterion=True, channel=0):
    """Draw tone-to-noise or prominence ratio of orders against speed, with the criterion.

    The figure of the result of :func:`rota_order.tonal.order_prominence`: for each
    order the ratio in dB (``p.tnr`` or ``p.pr``, the values the result holds,
    not recomputed) against the speed ``p.rpm`` of the analysis points (frames, or
    speed bins when ``rpm_bins`` was given). With ``criterion=True`` the ECMA-418-1
    criterion of the order's tone frequency ``order * rpm / 60`` is drawn dashed
    in the colour of the order (it varies with speed below 1 kHz; the standard
    gives none outside 89.1 Hz to 11.2 kHz, where the curve has a gap), and the
    speed ranges where the ratio is at least the criterion are shaded in the
    colour of the order (lightly, with one legend entry "<order> prominent" per
    order that has such a range, so that the shading of two orders that overlap
    can be told apart). Prominence is the objective criterion only: the audibility
    check of the standard (an aural examination) is not assessed.

    Parameters
    ----------
    p : rota_order.tonal.OrderProminence
        Result of :func:`rota_order.tonal.order_prominence`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created only when None.
    kind : {"tnr", "pr"}, default "tnr"
        Tone-to-noise ratio (ECMA-418-1, 11) or prominence ratio (12). Both are
        held by the result, whichever ``p.method`` it was computed with.
    orders : float or sequence of float, optional
        Orders to draw; default all.
    criterion : bool, default True
        Draw the criterion curves and shade the ranges where the order is
        prominent (ratio >= criterion; a gap in either is not prominent).
    channel : int, default 0
        Channel of a multichannel result.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, labelled "Tone-to-noise ratio (dB)" or "Prominence
        ratio (dB)" against "Speed (rpm)", with a legend entry per order, one per
        order with a shaded range ("<order> prominent") and one for the
        criterion. The lines are drawn at the true speeds of the points
        (a line may double back where the speed does). The figure is not shown, and the backend, style and
        ``rcParams`` are not touched.

    Raises
    ------
    ImportError
        If Matplotlib is not installed (``pip install 'rota-order[plot]'``).
    TypeError
        If ``p`` is not an ``OrderProminence``, ``channel`` is not an integer or
        ``ax`` is not an Axes.
    ValueError
        For an invalid ``kind``, or if ``orders`` are not in the result.
    IndexError
        If ``channel`` is out of range.

    Notes
    -----
    Ratios and criteria are in dB and need no reference; the powers behind them are
    mean-square values (RMS convention).

    The speed of unbinned points (frames) need not be monotonic: the shaded ranges
    are the cells between the midpoints of neighbouring speeds in sorted order, so
    that they never overlap, and points whose speed is not finite are not shaded.
    For a result binned along speed (``rpm_bins=`` in
    :func:`rota_order.tonal.order_prominence`, the usual choice for a run-up) a point
    stands for its bin. A speed bin without
    frames has no point; when it lies between two shaded bins (one empty bin
    exactly) the shading is continuous across it, while a gap of several bins, or a
    bin with a point below the criterion, stays open.

    References
    ----------
    Ecma International (2024). ECMA-418-1, Psychoacoustic metrics for ITT
    equipment - Part 1 (prominent discrete tones), 3rd edition. Clauses 11
    (tone-to-noise ratio) and 12 (prominence ratio).
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95.
    """
    if not isinstance(p, OrderProminence):
        raise TypeError(f"order_prominence expects a rota_order.tonal.OrderProminence, got {type(p).__name__}")
    _check_choice(kind, tuple(_KINDS), "kind")
    ratio = select_channel(getattr(p, kind), channel, 2)
    crit = np.asarray(getattr(p, kind + "_criterion"), dtype=float)
    rpm = np.asarray(p.rpm, dtype=float)
    n, k = ratio.shape
    if rpm.shape != (n,) or crit.shape != (n, k) or np.asarray(p.orders).shape != (k,):
        raise ValueError("rpm, orders, ratio and criterion of the result do not match")
    index = _pick_orders(p.orders, orders)
    cells = _point_cells(p.rpm_bins, rpm) if criterion else None
    with _drawing(ax) as (_, ax_):
        marker = "o" if n == 1 else None
        for j in index:
            (line,) = ax_.plot(rpm, ratio[:, j], marker=marker, label=_order_name(p.orders[j]))
            if not criterion:
                continue
            colour = line.get_color()
            ax_.plot(rpm, crit[:, j], ls="--", lw=1.0, color=colour, marker=marker)
            with np.errstate(invalid="ignore"):
                prominent = ratio[:, j] >= crit[:, j]  # NaN compares False
            runs = _merge_cells(cells[prominent & np.isfinite(cells[:, 0])])
            if p.rpm_bins is not None:
                runs = _bridge_empty_bins(runs, p.rpm_bins, cells)
            for i, (lo, hi) in enumerate(runs):
                ax_.axvspan(
                    lo, hi, color=colour, alpha=_SHADE_ALPHA, lw=0,
                    label=f"{_order_name(p.orders[j])} prominent" if i == 0 else None,
                )
        ax_.set_xlabel(X_LABELS["rpm"])
        ax_.set_ylabel(_KINDS[kind])
        handles, labels = ax_.get_legend_handles_labels()
        # lines first, then the shaded ranges (patches), then the criterion
        order_ = sorted(range(len(handles)), key=lambda i: not hasattr(handles[i], "get_xdata"))
        handles, labels = [handles[i] for i in order_], [labels[i] for i in order_]
        if criterion and index:
            from matplotlib.lines import Line2D

            handles.append(Line2D([], [], color="0.3", ls="--", lw=1.0))
            labels.append(_CRITERION_LABEL)
        ax_.legend(handles, labels)
    return ax_


def _point_cells(rpm_bins, rpm):
    """Speed range ``(lo, hi)`` that each analysis point stands for, shape ``(n, 2)``.

    A point of a result binned along speed stands for its bin, between the
    ``rpm_bins`` edges that contain its mean speed (bins without frames have no
    point); an unbinned frame stands for the cell between the midpoints to its
    neighbours in sorted speed order (so that the cells never overlap, whatever
    the order of the frames). Points with a non-finite speed get NaN cells,
    which are never shaded.
    """
    rpm = np.asarray(rpm, dtype=float)
    if rpm_bins is not None:
        bins = np.asarray(rpm_bins, dtype=float)
        if bins.ndim == 1 and bins.size >= 2 and np.all(np.diff(bins) > 0):
            i = np.clip(np.searchsorted(bins, rpm, side="right") - 1, 0, bins.size - 2)
            return np.stack([bins[i], bins[i + 1]], axis=1)
    cells = np.full((rpm.size, 2), np.nan)
    ok = np.flatnonzero(np.isfinite(rpm))
    if ok.size:
        order = ok[np.argsort(rpm[ok], kind="stable")]
        edges = edges_from_centres(rpm[order])
        cells[order, 0] = edges[:-1]
        cells[order, 1] = edges[1:]
    return cells


def _bridge_empty_bins(runs, bins, cells):
    """Join two shaded runs that are separated by exactly one speed bin without any point.

    A speed bin without frames has no point, so it cannot be prominent or not; when both of
    its neighbours are shaded the range is continuous and is drawn as one span. A bin that
    holds a point (not prominent), or a gap of more than one bin, is left open.
    """
    bins = np.asarray(bins, dtype=float)
    if bins.ndim != 1 or bins.size < 3 or not np.all(np.diff(bins) > 0):
        return runs
    occupied = {(float(lo), float(hi)) for lo, hi in cells if np.isfinite(lo)}
    out = [list(runs[0])] if runs else []
    for lo, hi in runs[1:]:
        a, b = out[-1][1], lo
        i = np.flatnonzero(np.isclose(bins, a, rtol=1e-12, atol=0.0))
        single = i.size == 1 and np.isclose(bins[i[0] + 1], b, rtol=1e-12, atol=0.0)
        if single and (float(a), float(b)) not in occupied:
            out[-1][1] = hi
        else:
            out.append([lo, hi])
    return [tuple(r) for r in out]


def _merge_cells(cells):
    """Merge touching speed ranges ``(lo, hi)`` into the runs they form (sorted by speed)."""
    merged = []
    for lo, hi in sorted((float(a), float(b)) for a, b in cells):
        if merged and lo <= merged[-1][1] * (1.0 + 1e-12):
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [tuple(m) for m in merged]


def _band_edges(b):
    """Band edges in Hz from the bands of the result (``n_bands + 1`` values, increasing).

    The lower and upper band limits are used when the bands are adjacent (the
    case of the fractional-octave banks); otherwise the cells are bounded by the
    geometric means of neighbouring mid-band frequencies.
    """
    lower = np.asarray(b.frames.lower, dtype=float)
    upper = np.asarray(b.frames.upper, dtype=float)
    if lower.ndim == 1 and lower.shape == upper.shape and lower.size and np.allclose(lower[1:], upper[:-1], rtol=1e-6):
        return np.concatenate((lower, upper[-1:]))
    centre = np.asarray(b.exact, dtype=float)
    return np.exp(edges_from_centres(np.log(centre)))


def _label_stride(exact):
    """``(stride, start)``: every ``stride``-th band from ``start`` is labelled.

    Up to 12 bands are all labelled. A longer bank is labelled every whole number
    of octaves (every third band of a third-octave bank) so that at most about 12
    labels remain, counted from the band nearest to 1 kHz.
    """
    n = exact.size
    if n <= 12:
        return 1, 0
    per_octave = max(1, round(np.log(2.0) / np.log(exact[1] / exact[0])))
    stride = per_octave * -(-n // (12 * per_octave))
    anchor = int(np.argmin(np.abs(np.log(exact / 1000.0))))
    return stride, anchor % stride


def _nominal_text(f):
    """Nominal mid-band frequency as a tick label: 31.5, 100, 1k, 12.5k."""
    f = float(f)
    return f"{f / 1000.0:g}k" if f >= 1000.0 else f"{f:g}"


def band_levels_vs_rpm(
    b,
    *,
    ax=None,
    ref=None,
    unit="",
    channel=0,
    dynamic_range=40.0,
    vmin=None,
    vmax=None,
    cmap="viridis",
    colorbar=True,
):
    """Draw band levels against speed and band as an image.

    The figure of the result of :func:`rota_order.tonal.band_levels_vs_rpm`. One cell
    per speed bin (on its edges ``b.rpm_edges``) and per octave or
    fractional-octave band (on the band limits, logarithmic frequency axis). The
    colour is the level as the result stores it, the energy-mean RMS band level
    in dB re ``b.ref``, unless ``ref`` is given, which re-references the levels
    exactly (``L + 20 log10(b.ref / ref)``). Nothing is resampled or smoothed; a
    bin with too few frames (NaN) is a gap. The y axis is labelled with the
    nominal mid-band frequencies (all bands up to 12; for a longer bank every
    octave, counted from 1 kHz, so that about 12 labels remain).

    Parameters
    ----------
    b : rota_order.tonal.BandRpmLevels
        Result of :func:`rota_order.tonal.band_levels_vs_rpm`; levels of shape
        ``(n_bins, n_bands)`` or ``(n_bins, n_bands, ch)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created only when None.
    ref : float, optional
        Reference of the dB scale in the unit of the signal. None (default)
        plots the levels as stored, re ``b.ref``. A given ``ref`` re-references
        them exactly, ``L + 20 log10(b.ref / ref)``, and labels the colour bar
        with it (``ref=2e-5`` for sound pressure in Pa). Colour limits
        (``vmin``, ``vmax``) are in the plotted levels.
    unit : str, default ""
        Unit of the analysed signal, used in the label only (the result does not
        store its unit): ``unit`` never changes a number.
        ``**rota_order.plot.SOUND_PRESSURE`` gives ``ref`` and ``unit`` together, and
        reads "dB re 20 µPa" for a result whose ``b.ref`` is 2e-5.
    channel : int, default 0
        Channel of a multichannel result.
    dynamic_range : float, default 40.0
        Span of the colour scale in dB below the maximum of the shown levels
        (``vmin = vmax - dynamic_range``), unless ``vmin`` is given.
    vmin, vmax : float, optional
        Colour limits in dB; ``vmax`` defaults to the largest finite level.
    cmap : str or Colormap, default "viridis"
        A perceptually uniform colour map.
    colorbar : bool, default True
        Add a colour bar labelled "Band level (dB re <ref>, RMS)".

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on, labelled "Speed (rpm)" and "Frequency (Hz)". The figure
        is not shown, and the backend, style and ``rcParams`` are not touched.

    Raises
    ------
    ImportError
        If Matplotlib is not installed (``pip install 'rota-order[plot]'``).
    TypeError
        If ``b`` is not a ``BandRpmLevels``, ``channel`` is not an integer or
        ``ax`` is not an Axes.
    ValueError
        If no level is finite (give ``vmax`` to draw anyway), if the speed bin
        edges or bands do not fit the levels, or for an invalid ``dynamic_range``
        or colour limit, or for an invalid ``ref``.
    IndexError
        If ``channel`` is out of range.

    Notes
    -----
    Band levels are RMS levels (energy means of the frames of a bin). The colour is
    the level of the whole band, not a spectral density: bands of different width
    are not directly comparable as spectra.

    References
    ----------
    IEC 61260-1:2014. Electroacoustics - Octave-band and fractional-octave-band
    filters - Part 1: Specifications.
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95.
    Crameri, F., Shephard, G. E., & Heron, P. J. (2020). The misuse of colour in
    science communication. Nature Communications, 11, 5444.
    """
    if not isinstance(b, BandRpmLevels):
        raise TypeError(f"band_levels_vs_rpm expects a rota_order.tonal.BandRpmLevels, got {type(b).__name__}")
    levels = np.asarray(b.levels, dtype=float)
    if levels.ndim not in (2, 3):
        raise ValueError("levels must have shape (n_bins, n_bands) or (n_bins, n_bands, ch)")
    z = select_channel(levels, channel, 2)
    n_bins, n_bands = z.shape
    rpm_edges = np.asarray(b.rpm_edges, dtype=float)
    nominal = np.asarray(b.nominal, dtype=float)
    exact = np.asarray(b.exact, dtype=float)
    if (
        rpm_edges.shape != (n_bins + 1,)
        or not np.all(np.isfinite(rpm_edges))
        or not np.all(np.diff(rpm_edges) > 0)
    ):
        raise ValueError("rpm_edges of the result must be finite, strictly increasing, one more than the speed bins")
    if nominal.shape != (n_bands,) or exact.shape != (n_bands,):
        raise ValueError("the bands of the result do not match its levels")
    ref_used, shift = level_reference(b.ref, ref)
    z = z + shift  # exact: 20 log10(a / ref_stored) + 20 log10(ref_stored / ref); 0.0 when ref is None
    f_edges = _band_edges(b)
    lo, hi = colour_limits(z, dynamic_range, vmin, vmax)
    label = level_label("Band level", ref_used, unit, "rms")
    stride, start = _label_stride(exact)
    with _drawing(ax) as (fig, ax_):
        mesh = ax_.pcolormesh(
            rpm_edges, f_edges, z.T, shading="flat", cmap=cmap, vmin=lo, vmax=hi, rasterized=True
        )
        ax_.set_yscale("log")
        ax_.set_xlim(rpm_edges[0], rpm_edges[-1])
        ax_.set_ylim(f_edges[0], f_edges[-1])
        from matplotlib.ticker import NullLocator

        ax_.yaxis.set_minor_locator(NullLocator())
        ax_.set_yticks(exact[start::stride])
        ax_.set_yticklabels([_nominal_text(f) for f in nominal[start::stride]])
        ax_.set_xlabel(X_LABELS["rpm"])
        ax_.set_ylabel("Frequency (Hz)")
        if colorbar:
            fig.colorbar(mesh, ax=ax_).set_label(label)
    return ax_
