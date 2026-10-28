"""Figures of rpm-frequency (Campbell) and rpm-order maps, and their comparison.

A map is drawn as the image it is: one cell per frame (or per speed bin of a
:class:`rota_order.maps.BinnedMap`) and per frequency or order bin, coloured by the
level ``20 log10(|value| / ref)`` in dB. The cells are placed on the true axes
of the map with ``pcolormesh``; nothing is resampled or smoothed, and a bin
without a level (a zero amplitude) stays a gap. The map values keep their
amplitude convention (peak or RMS), which the colour bar names.

Units: frequency in Hz, order in cycles per revolution, speed in rpm, time in
s, levels in dB re ``ref`` (``ref`` in the unit of the analysed signal).
"""

import dataclasses
import warnings
from dataclasses import dataclass

import numpy as np

from ..maps import BinnedMap, Map
from ._core import (
    MONOTONIC_TOL,
    X_LABELS,
    as_items,
    check_positive,
    check_tol,
    choose_x,
    colour_limits,
    edges_from_centres,
    get_ax,
    level_label,
    pyplot,
    select_channel,
    speed_envelope,
    to_db,
)

__all__ = ["compare_maps", "rpm_frequency_map", "rpm_order_map"]

_FUNCTION = {"frequency": "rpm_frequency_map", "order": "rpm_order_map"}
_Y_LABEL = {"frequency": "Frequency (Hz)", "order": "Order"}
_LIMIT_NAME = {"frequency": "fmax", "order": "max_order"}

# The order label is anchored inside the axes: this fraction of the axes width at
# the right and of the height at the top is kept free for it.
_INSET_X, _INSET_Y = 0.01, 0.06

# Level differences below this many dB are rounding noise, not a difference between maps.
_NOISE_DB = 1e-6
_CHANGE_DB = 0.01  # a level change below this is no change (dB)
_HIDDEN_ROW_FRACTION = 0.25  # share of a row that a hidden change must cover to count (scattered noise does not)
_HIDDEN_MARGIN_DB = 1.0  # hidden changes must exceed the difference scale by this much to warn (dB)

# Percentile of the shown absolute level differences that sets the default difference scale.
_DIFFERENCE_PERCENTILE = 99.5


# --------------------------------------------------------------------------
# Preparing the image of one map
# --------------------------------------------------------------------------


@dataclass
class _Image:
    """Cells and levels of one map, ready for ``pcolormesh``.

    ``z[i, j]`` is the level in dB of the cell between ``y_edges[i:i+2]`` and
    ``x_edges[j:j+2]``; NaN is a gap. ``x_values`` and ``rpm_frames`` (frame
    position on the x axis and frame speed, as measured) are None for a binned
    map, which is always drawn against speed. With x = rpm the cell edges come
    from the monotone envelope of the frame speed (see :func:`_image`), so that
    ``x_values`` can differ from the cell positions by the reversals of the speed.
    """

    x_edges: np.ndarray
    y_edges: np.ndarray
    z: np.ndarray
    domain: str
    amplitude: str
    x_kind: str
    x_label: str
    y_label: str
    y_top: float
    x_values: object = None
    rpm_frames: object = None


def _binned_columns(m, levels):
    """Contiguous rpm cell edges of a binned map and its levels with gap columns.

    ``bin_map`` drops the bins without frames, so the populated bins need not be
    adjacent. The cells run over the union of all bin edges; a stretch without a
    populated bin becomes one NaN column.
    """
    lo, hi = m.rpm_lo, m.rpm_hi
    if lo is None or hi is None:
        raise ValueError("a BinnedMap needs rpm_lo and rpm_hi (the speed bin edges) to be drawn")
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    n = levels.shape[1]
    if lo.shape != (n,) or hi.shape != (n,) or not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("rpm_lo and rpm_hi of the BinnedMap must be finite, one value per column")
    edges = np.unique(np.concatenate([lo, hi]))
    col = np.searchsorted(edges, lo)
    last = np.minimum(col + 1, edges.size - 1)
    if np.any(col + 1 >= edges.size) or np.any(edges[last] != hi) or np.unique(col).size != n:
        raise ValueError("the rpm bins of the BinnedMap overlap or have no width")
    z = np.full((levels.shape[0], edges.size - 1), np.nan)
    z[:, col] = levels
    return edges, z


def _image(m, domain, *, func, ref, channel, x, limit, tol=MONOTONIC_TOL):
    """Validate a map and build its :class:`_Image` (no Matplotlib involved).

    ``limit`` is ``fmax`` or ``max_order``: bins above it are not shown. ``tol`` is
    the largest reversal of the frame speed, as a fraction of its range, that
    still counts as a run-up or a coast-down (``x="auto"`` and ``x="rpm"``).

    With x = rpm the cell edges are computed from the monotone envelope of the
    frame speed (running maximum for a run-up, running minimum for a coast-down),
    so that the cells tile without overlap. A frame is therefore placed at most
    the reversal (at most ``tol`` times the speed range) away from its measured
    speed, and frames that stand at the same envelope value get cells of zero
    width.
    """
    tol = check_tol(tol)
    if not isinstance(m, Map):
        hint = "; use compare_maps for several maps" if isinstance(m, (list, tuple, dict)) else ""
        raise TypeError(f"{func} expects a rota_order.maps.Map or BinnedMap, got {type(m).__name__}{hint}")
    if m.domain not in _FUNCTION:
        raise ValueError(f"unknown map domain {m.domain!r}")
    if m.domain != domain:
        raise ValueError(
            f"{func} draws a map with domain {domain!r}, but this map has domain {m.domain!r}; "
            f"use rota_order.plot.{_FUNCTION[m.domain]}"
        )
    values = np.asarray(m.values)
    if values.ndim not in (2, 3):
        raise ValueError("map values must have shape (n_bins, n_frames[, ch])")
    axis = np.asarray(m.axis, dtype=float)
    n_bins, n_cols = values.shape[:2]
    if axis.shape != (n_bins,):
        raise ValueError("the axis of the map does not match the rows of its values")
    if n_bins < 2 or not np.all(np.diff(axis) > 0):
        raise ValueError("the map axis must be strictly increasing with at least two bins")
    if n_cols == 0:
        raise ValueError("the map has no frames to draw")
    values = select_channel(values, channel, 2)

    # Frequency or order axis: cells around the bin centres, cropped at the limit.
    y_full = edges_from_centres(axis)
    if limit is None:
        n_keep = n_bins
    else:
        limit = check_positive(limit, _LIMIT_NAME[domain])
        n_keep = int(np.count_nonzero(axis <= limit + 1e-6 * (axis[1] - axis[0])))
        if n_keep == 0:
            raise ValueError(
                f"{_LIMIT_NAME[domain]}={limit:g} is below the lowest bin of the map ({axis[0]:g})"
            )
    y_edges = y_full[: n_keep + 1].copy()
    if y_edges[0] < 0.0 <= axis[0]:
        y_edges[0] = 0.0  # the lowest bin is the DC bin: there is nothing below 0
    y_top = float(y_edges[-1])  # the limit selects bins: the axis ends at the top edge of the last one kept
    levels = to_db(values[:n_keep], ref)

    # Speed or time axis.
    if isinstance(m, BinnedMap):
        if x not in ("auto", "rpm", "time"):
            raise ValueError(f"x must be 'auto', 'rpm' or 'time', got {x!r}")
        if x == "time":
            raise ValueError(
                "a BinnedMap is drawn against speed (its columns are speed bins, not time frames); "
                "use x='auto' or x='rpm'"
            )
        x_edges, levels = _binned_columns(m, levels)
        kind, label, x_values, rpm_frames = "rpm", X_LABELS["rpm"], None, None
    else:
        t = np.asarray(m.t, dtype=float)
        rpm = np.asarray(m.rpm, dtype=float)
        if t.shape != (n_cols,) or rpm.shape != (n_cols,):
            raise ValueError("t and rpm of the map do not match the columns of its values")
        if n_cols < 2:
            raise ValueError("a map needs at least two frames to be drawn as an image")
        kind, x_values, label = choose_x(t, rpm, x, tol=tol)
        x_edges = edges_from_centres(speed_envelope(rpm, tol) if kind == "rpm" else x_values)
        rpm_frames = rpm
    return _Image(
        x_edges=x_edges,
        y_edges=y_edges,
        z=levels,
        domain=domain,
        amplitude=m.amplitude,
        x_kind=kind,
        x_label=label,
        y_label=_Y_LABEL[domain],
        y_top=y_top,
        x_values=x_values,
        rpm_frames=rpm_frames,
    )


def _orders_array(orders, name="mark_orders"):
    """Orders to mark as a 1-D float array (empty for None)."""
    if orders is None:
        return np.empty(0)
    try:
        o = np.atleast_1d(np.asarray(orders, dtype=float))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number or a sequence of numbers") from None
    if o.ndim != 1 or not np.all(np.isfinite(o)) or np.any(o <= 0):
        raise ValueError(f"{name} must be positive, finite numbers (cycles per revolution)")
    return o


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def _lerp(a, b, t):
    """Point at parameter ``t`` between ``a`` and ``b``, exact at both ends."""
    if t == 0.0:
        return a
    if t == 1.0:
        return b
    return a + t * (b - a)


def _clip_segment(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
    """Parameter interval of the segment inside the rectangle (Liang-Barsky), or None."""
    if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
        return None
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    return t0, t1


def _clip_polyline(x, y, xmin, xmax, ymin, ymax):
    """Clip the polyline ``(x, y)`` to a rectangle.

    Returns the points that remain, with the pieces of a polyline that leaves
    and re-enters separated by NaN; points that are inside keep their exact
    coordinates, new points lie on the rectangle. Empty arrays if nothing is
    inside.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xs, ys = [], []
    joined = False  # the previous piece ended at the end of its segment
    for i in range(x.size - 1):
        t = _clip_segment(x[i], y[i], x[i + 1], y[i + 1], xmin, xmax, ymin, ymax)
        if t is None:
            joined = False
            continue
        if not (joined and t[0] == 0.0):
            if xs:
                xs.append(np.nan)
                ys.append(np.nan)
            xs.append(_lerp(x[i], x[i + 1], t[0]))
            ys.append(_lerp(y[i], y[i + 1], t[0]))
        xs.append(_lerp(x[i], x[i + 1], t[1]))
        ys.append(_lerp(y[i], y[i + 1], t[1]))
        joined = t[1] == 1.0
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def _draw_orders(ax, img, orders, box, *, ink="white", edge="black"):
    """Thin light lines at the orders, with the order written at the right end.

    Frequency map: ``f = order * rpm / 60``, a straight line against speed or the
    frame speeds against time. Order map: a horizontal line at the order. The
    lines are clipped to the axes (``box``); the label sits above the line,
    right-aligned at the point where the line leaves the box shrunk by a small
    margin at the right and top, so that it stays inside the axes (a line in
    that margin at the top is labelled below it). ``ink`` is the
    colour of lines and labels and ``edge`` that of the thin outline that keeps
    them readable on light and dark cells (dark ink on white edge for a diverging
    map, whose centre is white).
    """
    from matplotlib import patheffects  # Matplotlib is present: the Axes exists

    x_lo, x_hi, y_lo, y_hi = box
    inner = (x_lo, x_hi - _INSET_X * (x_hi - x_lo), y_lo, y_hi - _INSET_Y * (y_hi - y_lo))
    line_edge = [patheffects.withStroke(linewidth=1.8, foreground=edge, alpha=0.3)]
    text_edge = [patheffects.withStroke(linewidth=1.6, foreground=edge, alpha=0.5)]
    for order in orders:
        if img.domain == "frequency":
            if img.x_kind == "rpm":
                xs = np.array([x_lo, x_hi])
                ys = order * xs / 60.0
            else:
                xs = img.x_values
                ys = order * img.rpm_frames / 60.0
        else:
            xs = np.array([x_lo, x_hi])
            ys = np.full(2, float(order))
        cx, cy = _clip_polyline(xs, ys, x_lo, x_hi, y_lo, y_hi)
        if cx.size == 0:
            continue  # above the displayed range
        ax.plot(cx, cy, color=ink, lw=0.7, alpha=0.8, ls=(0, (5, 3)), zorder=2.5, path_effects=line_edge)
        ix, iy = _clip_polyline(xs, ys, *inner)
        above = True
        if ix.size == 0:
            # a line in the strip kept free at the top (an order at the top bin of the axis): the label
            # sits below it instead of above, so that it is still drawn and stays inside the axes
            ix, iy = _clip_polyline(xs, ys, x_lo, inner[1], y_lo, y_hi)
            above = False
        if ix.size == 0:
            continue
        ax.annotate(
            f"{order:g}",
            xy=(ix[-1], iy[-1]),
            xytext=(-2, 2 if above else -2),
            textcoords="offset points",
            ha="right",
            va="bottom" if above else "top",
            color=ink,
            fontsize="x-small",
            path_effects=text_edge,
            zorder=3,
        )


def _draw_image(ax, img, *, vmin, vmax, cmap, orders, label_y=True, diverging=False):
    """Draw one image on ``ax``; returns the QuadMesh."""
    mesh = ax.pcolormesh(
        img.x_edges, img.y_edges, img.z, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True
    )
    box = (float(img.x_edges.min()), float(img.x_edges.max()), float(img.y_edges[0]), img.y_top)
    if orders.size:
        if diverging:
            _draw_orders(ax, img, orders, box, ink="black", edge="white")
        else:
            _draw_orders(ax, img, orders, box)
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_xlabel(img.x_label)
    if label_y:
        ax.set_ylabel(img.y_label)
    return mesh


def _plot_map(
    m, domain, *, ax, ref, unit, channel, x, monotonic_tol, dynamic_range, vmin, vmax, mark_orders, limit, cmap, colorbar
):
    """Shared body of :func:`rpm_frequency_map` and :func:`rpm_order_map`."""
    func = _FUNCTION[domain]
    img = _image(m, domain, func=func, ref=ref, channel=channel, x=x, limit=limit, tol=monotonic_tol)
    lines = _orders_array(mark_orders)
    vmin, vmax = colour_limits(img.z, dynamic_range, vmin, vmax)
    label = level_label("Level", ref, unit, img.amplitude)
    fig, ax_ = get_ax(ax)
    try:
        mesh = _draw_image(ax_, img, vmin=vmin, vmax=vmax, cmap=cmap, orders=lines)
        if colorbar:
            fig.colorbar(mesh, ax=ax_).set_label(label)
    except Exception:
        if ax is None:
            pyplot().close(fig)  # do not leave a half-drawn figure behind
        raise
    return ax_


# --------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------


def rpm_frequency_map(
    m,
    *,
    ax=None,
    ref=1.0,
    unit="",
    channel=0,
    x="auto",
    monotonic_tol=0.02,
    dynamic_range=40.0,
    vmin=None,
    vmax=None,
    mark_orders=None,
    fmax=None,
    cmap="viridis",
    colorbar=True,
):
    """Draw a frequency map as a Campbell diagram (level against speed and frequency).

    The figure of the result of :func:`rota_order.maps.rpm_frequency_map`. Each frame
    (or speed bin) is a column and each frequency bin a row of cells on the
    true axes of the map; the colour is the level
    ``20 log10(|value| / ref)`` in dB. The map values keep the amplitude
    convention of the map (peak or RMS, ``m.amplitude``), so a stationary line
    on a bin reads its peak or RMS level; the colour bar names the convention.
    Nothing is interpolated or smoothed, and a zero amplitude is a gap.

    The x axis is the speed in rpm when the frame speed is essentially monotonic
    and the time otherwise (``x="auto"``). Essentially monotonic means a run-up
    or a coast-down (the direction is the sign of last minus first speed) whose
    reversals, the drops below the running maximum of a run-up or the rises above
    the running minimum of a coast-down, are at most ``monotonic_tol`` of the
    speed range (2 % by default): a tachometer or ridge speed is rarely
    monotonic to the last digit. A speed that goes up and down (a run-up
    followed by a coast-down, a gear shift) would fold frames onto each other
    and is drawn against time. With rpm on the x axis the cells are placed on the
    monotone envelope of the frame speed (the running maximum or minimum), so
    that they tile without overlap; a frame is displaced by at most its reversal,
    at most ``monotonic_tol`` times the speed range, and frames on a plateau of
    the envelope get cells of zero width: frames at a constant speed (an idle
    hold before a run-up) collapse onto one rpm and are not visible, so use
    ``x="time"`` to see them. A :class:`rota_order.maps.BinnedMap` is always drawn on
    its speed bin edges.

    Parameters
    ----------
    m : rota_order.maps.Map or rota_order.maps.BinnedMap
        Map with ``domain="frequency"``; complex amplitude spectra in the unit of
        the analysed signal, shape ``(n_bins, n_frames)`` or
        ``(n_bins, n_frames, ch)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created only when None.
    ref : float, default 1.0
        Reference of the dB scale in the unit of the signal (``2e-5`` for sound
        pressure in Pa), explicit and never assumed.
    unit : str, default ""
        Unit of the signal (``"Pa"``, ``"m/s²"``), used in the label only: with
        ``ref=2e-5, unit="Pa"`` the colour bar reads "dB re 20 µPa". Both are
        given together by ``rota_order.plot.SOUND_PRESSURE``.
    channel : int, default 0
        Channel of a multichannel map.
    x : {"auto", "rpm", "time"}, default "auto"
        Quantity on the x axis. ``"rpm"`` with a speed that is not essentially
        monotonic raises ValueError (use ``rota_order.maps.bin_map`` or
        ``x="time"``); a binned map accepts ``"auto"`` and ``"rpm"``.
    monotonic_tol : float, default 0.02
        Largest reversal of the frame speed, as a fraction of its range, that
        still counts as monotonic (``x="auto"``, ``x="rpm"``); 0 accepts only a
        non-decreasing or non-increasing speed.
    dynamic_range : float, default 40.0
        Span of the colour scale in dB below the maximum of the displayed
        levels (``vmin = vmax - dynamic_range``), unless ``vmin`` is given.
    vmin, vmax : float, optional
        Colour limits in dB; ``vmax`` defaults to the largest finite level
        shown (after ``fmax``).
    mark_orders : float or sequence of float, optional
        Orders to mark: thin lines at ``f = order * rpm / 60`` (against speed,
        or against time with the speed of each frame), labelled with the order
        at the right end, inside the axes. Parts above ``fmax`` are clipped.
    fmax : float, optional
        Highest frequency shown, in Hz; bins whose centre is above it are left
        out of the figure and of the colour scale. The bin at ``fmax`` is shown
        whole: the axis ends at its upper edge, half a bin above ``fmax`` when
        ``fmax`` is a bin centre.
    cmap : str or Colormap, default "viridis"
        A perceptually uniform colour map.
    colorbar : bool, default True
        Add a colour bar labelled "Level (dB re <ref>, <peak or RMS>)".

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on. The figure is not shown (no ``plt.show()``), and the
        backend, style and ``rcParams`` are not touched.

    Raises
    ------
    ImportError
        If Matplotlib is not installed (``pip install 'rota-order[plot]'``).
    TypeError
        If ``m`` is not a map, or ``ax`` is not an Axes.
    ValueError
        If the map is an order map (the message names
        :func:`rota_order.plot.rpm_order_map`), for ``x="rpm"`` with a speed that is not
        essentially monotonic (or ``x="time"`` with a binned map), for a map with
        a single frame or no bin to draw, or for invalid ``ref``,
        ``monotonic_tol``, ``dynamic_range``, limits or ``mark_orders``.
    IndexError
        If ``channel`` is out of range.

    Notes
    -----
    The colour is the level per bin, not a density: a line between two bins reads
    lower than its true level (up to 1.4 dB for a Hann window). Read the level of
    an order with :func:`rota_order.slices.order_slice`, which sums the power over a
    band.

    The mesh is rasterised, so that PDF and SVG output stay small; axes, labels
    and order lines remain vector graphics.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12 (Campbell diagram,
    order maps). Wiley.
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95.
    Crameri, F., Shephard, G. E., & Heron, P. J. (2020). The misuse of colour in
    science communication. Nature Communications, 11, 5444.
    """
    return _plot_map(
        m, "frequency", ax=ax, ref=ref, unit=unit, channel=channel, x=x, monotonic_tol=monotonic_tol,
        dynamic_range=dynamic_range,
        vmin=vmin, vmax=vmax, mark_orders=mark_orders, limit=fmax, cmap=cmap, colorbar=colorbar,
    )


def rpm_order_map(
    m,
    *,
    ax=None,
    ref=1.0,
    unit="",
    channel=0,
    x="auto",
    monotonic_tol=0.02,
    dynamic_range=40.0,
    vmin=None,
    vmax=None,
    mark_orders=None,
    max_order=None,
    cmap="viridis",
    colorbar=True,
):
    """Draw an order map (level against speed and order).

    The figure of the result of :func:`rota_order.maps.rpm_order_map`. Each frame (or
    speed bin) is a column and each order bin a row of cells on the true axes of
    the map; the colour is the level ``20 log10(|value| / ref)`` in dB. The map
    values keep the amplitude convention of the map (peak or RMS,
    ``m.amplitude``), so an order that falls on a bin reads its peak or RMS
    level; the colour bar names the convention. Nothing is interpolated or
    smoothed, and a zero amplitude is a gap.

    The x axis is the speed in rpm when the frame speed is essentially monotonic
    (a run-up or a coast-down with reversals of at most ``monotonic_tol`` of the
    speed range) and the time otherwise (``x="auto"``); with rpm the cells are
    placed on the monotone envelope of the frame speed, so that they never
    overlap, and a frame is displaced by at most its reversal. The rule is
    spelled out in :func:`rpm_frequency_map`; frames at a constant speed (an idle
    hold) collapse onto one rpm and are not visible, so use ``x="time"`` to see
    them. A :class:`rota_order.maps.BinnedMap` is always drawn on its speed bin edges.

    Parameters
    ----------
    m : rota_order.maps.Map or rota_order.maps.BinnedMap
        Map with ``domain="order"``; complex amplitude spectra in the unit of the
        analysed signal, shape ``(n_bins, n_frames)`` or
        ``(n_bins, n_frames, ch)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created only when None.
    ref : float, default 1.0
        Reference of the dB scale in the unit of the signal (``2e-5`` for sound
        pressure in Pa), explicit and never assumed.
    unit : str, default ""
        Unit of the signal (``"Pa"``, ``"m/s²"``), used in the label only: with
        ``ref=2e-5, unit="Pa"`` the colour bar reads "dB re 20 µPa". Both are
        given together by ``rota_order.plot.SOUND_PRESSURE``.
    channel : int, default 0
        Channel of a multichannel map.
    x : {"auto", "rpm", "time"}, default "auto"
        Quantity on the x axis, as in :func:`rpm_frequency_map`.
    monotonic_tol : float, default 0.02
        Largest reversal of the frame speed as a fraction of its range, as in
        :func:`rpm_frequency_map`.
    dynamic_range : float, default 40.0
        Span of the colour scale in dB below the maximum of the displayed
        levels (``vmin = vmax - dynamic_range``), unless ``vmin`` is given.
    vmin, vmax : float, optional
        Colour limits in dB; ``vmax`` defaults to the largest finite level
        shown (after ``max_order``).
    mark_orders : float or sequence of float, optional
        Orders to mark: thin horizontal lines, labelled with the order at the
        right end, inside the axes. Orders above ``max_order`` are left out.
    max_order : float, optional
        Highest order shown; bins whose centre is above it are left out of the
        figure and of the colour scale. The bin at ``max_order`` is shown whole:
        the axis ends at its upper edge, half a bin above ``max_order`` when it
        is a bin centre (an order marked there stays inside the axes).
    cmap : str or Colormap, default "viridis"
        A perceptually uniform colour map.
    colorbar : bool, default True
        Add a colour bar labelled "Level (dB re <ref>, <peak or RMS>)".

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on. The figure is not shown (no ``plt.show()``), and the
        backend, style and ``rcParams`` are not touched.

    Raises
    ------
    ImportError
        If Matplotlib is not installed (``pip install 'rota-order[plot]'``).
    TypeError
        If ``m`` is not a map, or ``ax`` is not an Axes.
    ValueError
        If the map is a frequency map (the message names
        :func:`rota_order.plot.rpm_frequency_map`), for ``x="rpm"`` with a speed that is
        not essentially monotonic (or ``x="time"`` with a binned map), for a map
        with a single frame or no bin to draw, or for invalid ``ref``,
        ``monotonic_tol``, ``dynamic_range``, limits or ``mark_orders``.
    IndexError
        If ``channel`` is out of range.

    Notes
    -----
    The colour is the level per bin, not a density: an order between two bins
    reads lower than its true level (up to 1.4 dB for a Hann window). Read the
    level of an order with :func:`rota_order.slices.order_slice`, which sums the power
    over a band.

    The mesh is rasterised, so that PDF and SVG output stay small; axes, labels
    and order lines remain vector graphics.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12 (Campbell diagram,
    order maps). Wiley.
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95.
    Crameri, F., Shephard, G. E., & Heron, P. J. (2020). The misuse of colour in
    science communication. Nature Communications, 11, 5444.
    """
    return _plot_map(
        m, "order", ax=ax, ref=ref, unit=unit, channel=channel, x=x, monotonic_tol=monotonic_tol,
        dynamic_range=dynamic_range,
        vmin=vmin, vmax=vmax, mark_orders=mark_orders, limit=max_order, cmap=cmap, colorbar=colorbar,
    )


def _same_cells(a, b):
    """True if two images have the same domain and the same cells on the same axes."""
    return (
        a.domain == b.domain
        and a.x_kind == b.x_kind
        and a.z.shape == b.z.shape
        and a.x_edges.shape == b.x_edges.shape
        and a.y_edges.shape == b.y_edges.shape
        and bool(np.allclose(a.x_edges, b.x_edges, rtol=1e-9, atol=1e-12))
        and bool(np.allclose(a.y_edges, b.y_edges, rtol=1e-9, atol=1e-12))
    )


def _robust_difference_range(arrays):
    """Half range of the difference scale in dB: a high percentile of the finite ``|difference|``.

    ``arrays`` are the differences of the cells that are shown (see :func:`_shown_differences`).
    The largest difference is set by a few noise or edge cells and would leave a real change pale
    (a 10 dB change on a scale of +-28 dB), so the extreme 0.5 % of the cells is allowed to saturate.
    Differences below ``_NOISE_DB`` are rounding noise: if nothing is larger than that, the scale is
    +-1 dB. If more than 99.5 % of the cells are identical, the percentile is zero and the largest
    difference is used, so that the few cells that do differ are not all saturated.
    """
    parts = [np.abs(a[np.isfinite(a)]) for a in arrays]
    a = np.concatenate(parts) if parts else np.empty(0)
    peak = float(a.max()) if a.size else 0.0
    if not peak > _NOISE_DB:
        return 1.0
    lim = float(np.percentile(a, _DIFFERENCE_PERCENTILE))
    return lim if lim > _NOISE_DB else peak


def _row_near_maximum(z, dynamic_range):
    """Cells of ``z`` within ``dynamic_range`` dB of the maximum of their own row (frequency or order bin)."""
    with np.errstate(invalid="ignore"):
        top = np.where(np.isfinite(z), z, -np.inf).max(axis=1, initial=-np.inf)
        return z >= (top - dynamic_range)[:, None]


def _scale_differences(base, images, diffs, vmin, dynamic_range):
    """Differences that set the difference scale, and the changes that the level panels hide.

    Returns ``(counted, hidden, fallback)``; the first two are lists with one array of finite differences in dB per
    difference panel, ``fallback`` a list with one bool per difference panel. ``counted`` are the cells
    that are visible in a map panel (at least one level
    above ``vmin``). If none of them changes by more than ``_CHANGE_DB`` while other cells do (the
    change lies below the colour floor: a weak order next to a dominant rumble), the scale is set by
    the changed cells that are within ``dynamic_range`` dB of the maximum of their own row (either map;
    however few cells of the row change, so a weak order that changes over a short speed range counts),
    or by all changed cells if there are none. A cell in which both maps are far below the floor and
    below their row maxima is resynthesis or estimation noise and does not set the scale otherwise.
    ``hidden`` are the changed cells (``> _CHANGE_DB``) that no map panel shows and that lie within
    ``dynamic_range`` of their own row maximum and that cover at least a quarter of their row (a coherent
    change of an order or a line, not scattered noise cells); that coherence filter only decides about the
    warning, not about the scale. ``fallback[i]`` is True if the scale of panel ``i`` was set by cells below the
    colour floor.
    """
    counted, hidden, fallback = [], [], []
    for im, d in zip(images, diffs):
        ok = np.isfinite(d.z)
        with np.errstate(invalid="ignore"):
            visible = ((base.z > vmin) | (im.z > vmin)) & ok
            changed = (np.abs(d.z) > _CHANGE_DB) & ok
            near = (_row_near_maximum(base.z, dynamic_range) | _row_near_maximum(im.z, dynamic_range)) & ok
        hid = changed & near & ~visible
        # a change that matters is coherent along a row (an order or a line), not scattered cells
        n_ok = np.maximum(ok.sum(axis=1, keepdims=True), 1)
        hid &= hid.sum(axis=1, keepdims=True) >= _HIDDEN_ROW_FRACTION * n_ok
        use = visible
        below = bool(not (visible & changed).any() and changed.any())
        if below:
            near_changed = changed & near
            use = near_changed if near_changed.any() else changed
        fallback.append(below)
        counted.append(d.z[use])
        hidden.append(d.z[hid])
    return counted, hidden, fallback


def compare_maps(
    maps,
    *,
    ref=1.0,
    unit="",
    channel=0,
    x="auto",
    monotonic_tol=0.02,
    dynamic_range=40.0,
    difference=False,
    cmap="viridis",
    vmin=None,
    vmax=None,
    mark_orders=None,
    fmax=None,
    max_order=None,
    difference_range=None,
    colorbar=True,
    figsize=None,
):
    """Draw maps side by side on one colour scale, optionally with their difference.

    One row of panels, one per map, with the keys of a dict as titles. All map
    panels share one colour scale, ``vmax`` being the largest finite level shown
    in any of them, so equal colours mean equal levels, and one colour bar. This
    is the before/after view of, for example, an order removal or a resynthesis.
    The maps can be of different domains (a Campbell diagram next to an order
    map of the same run-up): each panel is drawn with its own frequency or order
    axis and keeps its own y label, and the colour scale is still shared.

    With ``difference=True`` a panel "<label> − <first label> (dB)" is added for
    every map after the first: the level difference
    ``20 log10(|value| / |value_first|)`` per cell on a diverging colour map
    centred on 0 dB with symmetric limits, with its own colour bar (arrows at
    the ends of the bar mark limits that some shown cells exceed). A cell where
    either level is a gap is a gap. The difference needs maps of one domain with
    the same axes, frames and speeds.

    Parameters
    ----------
    maps : dict, list or tuple of rota_order.maps.Map or rota_order.maps.BinnedMap
        The maps to draw: frequency maps, order maps or a mixture of both, with
        the same amplitude convention (peak or RMS). A dict ``{label: map}`` gives the
        panel titles; an unlabelled list is titled "Map 1", "Map 2", ...
    ref : float, default 1.0
        Reference of the dB scale in the unit of the signal (``2e-5`` for sound
        pressure in Pa), explicit and never assumed.
    unit : str, default ""
        Unit of the signal, used in the label only (see
        :func:`rpm_frequency_map`).
    channel : int, default 0
        Channel of multichannel maps.
    x : {"auto", "rpm", "time"}, default "auto"
        Quantity on the x axis of each panel, as in :func:`rpm_frequency_map`
        (essentially monotonic speed: cells on the monotone envelope; frames at a
        constant speed collapse onto one rpm, use ``x="time"`` to see them).
    monotonic_tol : float, default 0.02
        Largest reversal of the frame speed as a fraction of its range, as in
        :func:`rpm_frequency_map`.
    dynamic_range : float, default 40.0
        Span of the shared colour scale in dB below its maximum.
    difference : bool, default False
        Add the difference panels. The maps must then have the same domain and
        identical axes, frames and speeds (ValueError otherwise).
    cmap : str or Colormap, default "viridis"
        Colour map of the map panels; the difference panels use ``"RdBu_r"``.
    vmin, vmax : float, optional
        Limits of the shared scale in dB.
    mark_orders : float or sequence of float, optional
        Orders to mark in every panel, as in :func:`rpm_frequency_map` (lines of
        ``f = order * rpm / 60`` in the frequency panels, horizontal lines at the
        order in the order panels).
    fmax : float, optional
        Highest frequency shown, applied to the frequency maps (ValueError if
        there is none); the bin at ``fmax`` is shown whole, as in
        :func:`rpm_frequency_map`.
    max_order : float, optional
        Highest order shown, applied to the order maps (ValueError if there is
        none).
    difference_range : float, optional
        Half range in dB of the difference scale (limits ``±difference_range``).
        Default: the 99.5th percentile of the finite absolute differences of the
        cells that are visible (at least one of the two levels above the lower limit
        of the shared colour scale), over all difference panels, so that a few noise
        or edge cells with a large difference do not stretch the scale and wash out
        the real change. Cells in which both maps are below the colour scale do not
        count, because there the difference is mostly resynthesis or estimation
        noise. If no visible cell changes by more than 0.01 dB while others do (the
        change is below the colour floor, e.g. a weak order next to a dominant
        low-order rumble), the changed cells within ``dynamic_range`` dB of the
        maximum of their own row (frequency or order bin) set the scale, or all
        changed cells if there are none. If the maps differ by less than 1e-6 dB
        (rounding noise) the scale is ±1 dB; if more than 99.5 % of the counted
        cells are identical, the largest absolute difference is used. Cells beyond
        the limits (a given ``difference_range`` included) saturate, and the
        difference colour bar then has an arrow at that end (or both): a saturated
        band is never read as the limit. A ``UserWarning`` is issued when
        cells that change by more than 0.01 dB lie below the colour floor of the
        level panels but within ``dynamic_range`` of their own row maximum, and the
        change exceeds the difference scale by more than 1 dB: the level panels do
        not show it; raise ``dynamic_range`` (or lower ``vmin``).
    colorbar : bool, default True
        Add the colour bars ("Level (dB re <ref>, <peak or RMS>)" and "Level
        difference (dB)").
    figsize : (float, float), optional
        Figure size in inches; default: the default height and a width that
        grows with the number of panels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        A new figure with constrained layout (for this figure only).
    axes : ndarray of matplotlib.axes.Axes, shape (n_panels,)
        The map panels in the order of ``maps``, then the difference panels. When
        all panels have the same frequency or order axis they share the y axis.

    Raises
    ------
    ImportError
        If Matplotlib is not installed (``pip install 'rota-order[plot]'``).
    TypeError
        If an entry is not a map.
    ValueError
        For maps of different amplitude conventions (a shared scale would
        mislead), for ``difference=True`` with fewer than two maps or with maps
        of different domains, axes or frames, for ``fmax`` with order maps only
        or ``max_order`` with frequency maps only, or for the invalid arguments
        listed for :func:`rpm_frequency_map`.
    IndexError
        If ``channel`` is out of range.

    Notes
    -----
    Levels are ``20 log10(|value| / ref)`` of the map values, which keep the
    amplitude convention of the maps (peak or RMS). The difference does not
    depend on ``ref``.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12 (Campbell diagram,
    order maps). Wiley.
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95.
    Crameri, F., Shephard, G. E., & Heron, P. J. (2020). The misuse of colour in
    science communication. Nature Communications, 11, 5444.
    """
    items = as_items(maps)
    labels = [lab if lab is not None else f"Map {i + 1}" for i, (lab, _) in enumerate(items)]
    maps_list = [m for _, m in items]
    for m in maps_list:
        if not isinstance(m, Map):
            raise TypeError(f"compare_maps expects rota_order.maps.Map or BinnedMap objects, got {type(m).__name__}")
    for m in maps_list:
        if m.domain not in _FUNCTION:
            raise ValueError(f"unknown map domain {m.domain!r}")
    domains = {m.domain for m in maps_list}
    conventions = sorted({m.amplitude for m in maps_list})
    if len(conventions) > 1:
        raise ValueError(
            f"the maps have different amplitude conventions {conventions}; one colour scale and one "
            "colour bar label would mislead (peak and RMS differ by 3 dB)"
        )
    # fmax belongs to the frequency maps and max_order to the order maps; a mixed set takes both.
    for domain, name, value in (("frequency", "max_order", max_order), ("order", "fmax", fmax)):
        if domain in domains and len(domains) == 1 and value is not None:
            raise ValueError(f"{name} does not apply to {domain} maps; use {_LIMIT_NAME[domain]}")
    limits = {"frequency": fmax, "order": max_order}
    n = len(maps_list)
    if difference and n < 2:
        raise ValueError("difference=True needs at least two maps")
    if difference and len(domains) > 1:
        raise ValueError(
            "difference=True needs maps of one domain (a frequency map and an order map have no common "
            "cells), but got " + str(sorted(domains)) + "; draw them side by side without difference"
        )

    images = [
        _image(
            m, m.domain, func="compare_maps", ref=ref, channel=channel, x=x, limit=limits[m.domain],
            tol=monotonic_tol,
        )
        for m in maps_list
    ]
    lines = _orders_array(mark_orders)
    vmin, vmax = colour_limits([im.z for im in images], dynamic_range, vmin, vmax)
    label = level_label("Level", ref, unit, images[0].amplitude)

    diffs = []
    lim = None
    extend = "neither"
    if difference:
        base = images[0]
        for lab, im in zip(labels[1:], images[1:]):
            if not _same_cells(base, im):
                raise ValueError(
                    f"cannot take the difference {lab!r} - {labels[0]!r}: the maps differ in "
                    "axis, frames or speed"
                )
            with np.errstate(invalid="ignore"):  # inf - inf
                diffs.append(dataclasses.replace(im, z=im.z - base.z))
        shown, hidden, fallback = _scale_differences(base, images[1:], diffs, vmin, dynamic_range)
        if difference_range is not None:
            lim = check_positive(difference_range, "difference_range")
        else:
            lim = _robust_difference_range(shown)
        warn = [np.abs(h) for h, fb in zip(hidden, fallback) if h.size and (fb or np.abs(h).max() > lim + _HIDDEN_MARGIN_DB)]
        if warn:
            hid_max = max(float(h.max()) for h in warn)
            warnings.warn(
                f"compare_maps: levels change by up to {hid_max:.2g} dB in cells that lie below the colour "
                f"floor of the level panels ({vmin:.1f} dB), where no panel shows them; raise "
                "dynamic_range (or lower vmin) to show them",
                UserWarning,
                stacklevel=2,
            )
        # cells beyond the limits saturate: the colour bar says so with arrows, never silently
        values = np.concatenate(shown) if shown else np.empty(0)
        over, under = bool(values.size and values.max() > lim), bool(values.size and values.min() < -lim)
        extend = "both" if over and under else "max" if over else "min" if under else "neither"

    panels = images + diffs
    plt = pyplot()
    if figsize is None:
        width, height = plt.rcParams["figure.figsize"]
        figsize = (width * (0.5 * len(panels) + 0.25), 0.9 * height)
    share_y = all(
        im.domain == images[0].domain
        and np.array_equal(im.y_edges, images[0].y_edges)
        and im.y_top == images[0].y_top
        for im in panels
    )
    fig, axs = plt.subplots(
        1, len(panels), figsize=figsize, layout="constrained", squeeze=False, sharey=share_y
    )
    axes = axs[0]
    try:
        meshes = []
        for i, (ax, im) in enumerate(zip(axes, images)):
            meshes.append(
                _draw_image(ax, im, vmin=vmin, vmax=vmax, cmap=cmap, orders=lines, label_y=i == 0 or not share_y)
            )
            ax.set_title(labels[i])
        diff_meshes = []
        for k, (ax, im) in enumerate(zip(axes[n:], diffs)):
            diff_meshes.append(
                _draw_image(
                    ax, im, vmin=-lim, vmax=lim, cmap="RdBu_r", orders=lines, label_y=not share_y, diverging=True
                )
            )
            ax.set_title(f"{labels[k + 1]} \u2212 {labels[0]} (dB)")  # U+2212 minus sign
        if colorbar:
            fig.colorbar(meshes[0], ax=list(axes[:n])).set_label(label)
            if diff_meshes:
                fig.colorbar(diff_meshes[0], ax=list(axes[n:]), extend=extend).set_label("Level difference (dB)")
    except Exception:
        plt.close(fig)
        raise
    return fig, axes
