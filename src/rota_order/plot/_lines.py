"""Line figures of slices, spectra, overall levels and metrics.

Order and frequency slices (level against speed or time), averaged spectra
(level against order or frequency), overall levels (per frame or per speed bin)
and psychoacoustic metrics (per speed bin with its spread, or against time) are
drawn as lines on a Matplotlib Axes. Each function takes one result, a list of
results or a dict ``{label: result}`` (the keys become the legend), and
``relative_to=<label>`` plots every entry minus that one: the before/after view of
an order removal or a resynthesis.

Levels are ``20 log10(|a| / ref)`` in dB with ``ref`` in the unit of the plotted
signal; the amplitudes keep their convention (peak or RMS), which the axis label
names. A zero amplitude has no level and, like NaN, stays a gap in the line (a
silent frame of an overall level, ``-inf``, is a gap as well). Nothing is
interpolated or smoothed. Units: speed in rpm, time in s, frequency in Hz, order in
cycles per revolution.
"""

import re
from collections.abc import Mapping

import numpy as np

from .._application import FrequencySlice, MetricVsRpm, OverallLevel
from ..bridges import TimeMetric
from ..slices import OrderSlice, Spectrum
from ._core import (
    X_LABELS,
    as_items,
    check_ref,
    choose_x,
    darken,
    draw_reference,
    get_ax,
    legend_without_repeats,
    level_label,
    level_reference,
    pyplot,
    relative,
    select_channel,
    to_db,
)
from ._maps import _orders_array

__all__ = [
    "frequency_slice",
    "metric_vs_rpm",
    "order_slice",
    "order_spectrum",
    "overall_level",
    "time_metric",
]

_SCALES = ("db", "linear")
_CONVENTION_WORD = {"peak": "peak", "rms": "RMS"}


# --------------------------------------------------------------------------
# Checks and labels
# --------------------------------------------------------------------------


def _results(obj, types, func, what):
    """``[(label, result), ...]`` of one result, a list or a dict; TypeError for a wrong type."""
    items = as_items(obj)
    for label, r in items:
        if not isinstance(r, types):
            where = f" (entry {label!r})" if label is not None else ""
            raise TypeError(f"{func} expects {what}{where}, got {type(r).__name__}")
    return items


def _check_scale(scale):
    if scale not in _SCALES:
        raise ValueError(f"scale must be 'db' or 'linear', got {scale!r}")
    return scale


def _check_x(x):
    if x not in ("rpm", "time"):
        raise ValueError(f"x must be 'rpm' or 'time', got {x!r}")
    return x


def _same_convention(items):
    """The amplitude convention shared by all entries (ValueError if they differ)."""
    conventions = sorted({str(r.amplitude_convention).lower() for _, r in items})
    if len(conventions) > 1:
        raise ValueError(
            f"the results have different amplitude conventions {conventions}; one axis would mislead "
            "(peak and RMS differ by 3 dB)"
        )
    return conventions[0]


def _real_float(values, what):
    a = np.asarray(values)
    if np.iscomplexobj(a):
        raise ValueError(f"{what} must be real")
    return a.astype(float, copy=False)


def _amplitude_label(quantity_db, quantity_lin, scale, ref, unit, convention, relative_to):
    """Axis label of an amplitude: level in dB re ref, or linear in the signal unit."""
    word = _CONVENTION_WORD.get(convention)
    if scale == "db":
        if relative_to is None:
            return level_label(quantity_db, ref, unit, convention)
        return f"{quantity_db} relative to {relative_to} (dB)"
    parts = [p for p in (str(unit) if unit else "", word) if p]
    tail = f" ({', '.join(parts)})" if parts else ""
    if relative_to is None:
        return f"{quantity_lin}{tail}"
    return f"{quantity_lin} minus {relative_to}{tail}"


def _on_axes(ax, draw):
    """Run ``draw(ax)`` on the given Axes or on a new figure, which is closed if drawing fails."""
    fig, ax_ = get_ax(ax)
    try:
        draw(ax_)
    except Exception:
        if ax is None:
            pyplot().close(fig)  # do not leave a half-drawn figure behind
        raise
    return ax_


def _gap(y):
    """Non-finite values (NaN, infinite) as NaN: they stay a gap in the line."""
    y = np.asarray(y, dtype=float)
    return np.where(np.isfinite(y), y, np.nan)


def _legend_labels(items):
    return [lab for lab, _ in items]


def _plot_lines(ax, items, pairs, *, marker=None, **kw):
    """Draw one line per entry; returns the lines. Unlabelled entries stay out of the legend."""
    lines = []
    for (label, _), (xv, yv) in zip(items, pairs):
        opts = dict(kw)
        if label is not None:
            opts["label"] = label
        if marker is not None:
            opts["marker"] = marker
            opts["markersize"] = 3
        (line,) = ax.plot(xv, yv, **opts)
        lines.append(line)
    return lines


def _finish_legend(ax):
    legend_without_repeats(ax)  # a reference label that is drawn several times is listed once


# --------------------------------------------------------------------------
# Slices: level of one order or frequency against speed or time
# --------------------------------------------------------------------------


def _truth_pair(truth, scale, ref, x_name, what="truth"):
    """``(x, y)`` of one truth curve in the plotted scale."""
    try:
        tx, ta = truth
        tx = np.asarray(tx, dtype=float)
        ta = _real_float(ta, f"the amplitude of {what}")
    except (TypeError, ValueError) as err:
        raise ValueError(f"{what} must be a pair (x, amplitude) of real arrays: {err}") from None
    if tx.ndim != 1 or ta.shape != tx.shape or tx.size == 0:
        raise ValueError(
            f"{what} must be a pair (x, amplitude) of 1-D arrays of equal, non-zero length "
            f"(x in {x_name}), got shapes {tx.shape} and {ta.shape}"
        )
    return tx, to_db(ta, ref) if scale == "db" else ta


def _check_truth(truth, labels, scale, ref, x_name):
    """``[(entry index or None, label, (x, y)), ...]`` of the truth curves in the plotted scale.

    One ``(x, amplitude)`` pair gives one curve (entry index None, label "truth"). A
    mapping ``{label: (x, amplitude)}`` gives one curve per key, which must be a label
    of the results (a dict was passed): the entry index tells which line it belongs to
    and the label is "<label> truth".
    """
    if truth is None:
        return []
    if not isinstance(truth, Mapping):
        return [(None, "truth", _truth_pair(truth, scale, ref, x_name))]
    if not truth:
        raise ValueError("truth is an empty dict: give one (x, amplitude) pair per label, or a single pair")
    known = [lab for lab in labels if lab is not None]
    if not known:
        raise ValueError("truth as a dict needs labelled results: pass the slices as a dict {label: slice}")
    out = []
    for key, pair in truth.items():
        if str(key) not in known:
            raise ValueError(f"truth has the key {key!r}, which is not one of the labels {known}")
        out.append((labels.index(str(key)), f"{key} truth", _truth_pair(pair, scale, ref, x_name, f"truth[{key!r}]")))
    return out


def _plot_slices(
    obj, types, what, func, quantity, *, ax, ref, unit, channel, x, scale, relative_to, truth
):
    """Shared body of :func:`order_slice` and :func:`frequency_slice`."""
    items = _results(obj, types, func, what)
    ref = check_ref(ref)
    scale = _check_scale(scale)
    x = _check_x(x)
    convention = _same_convention(items)
    pairs = []
    x_label = X_LABELS[x]
    for _, s in items:
        t = np.asarray(s.t, dtype=float)
        rpm = np.asarray(s.rpm, dtype=float)
        amp = _real_float(s.amplitude, "the amplitude of a slice")
        if t.ndim != 1 or rpm.shape != t.shape or amp.ndim not in (1, 2) or amp.shape[0] != t.size:
            raise ValueError("t, rpm and amplitude of the slice must have one value per frame")
        _, xv, x_label = choose_x(t, rpm, x, check=False)  # a line may double back: it is the data
        a = select_channel(amp, channel, 1)
        pairs.append((xv, to_db(a, ref) if scale == "db" else a))
    labels = _legend_labels(items)
    if relative_to is not None and truth is not None:
        raise ValueError("truth cannot be combined with relative_to (the truth is an absolute curve)")
    pairs = relative(pairs, labels, relative_to)
    truths = _check_truth(truth, labels, scale, ref, "rpm" if x == "rpm" else "s")
    y_label = _amplitude_label(quantity, quantity.replace("level", "amplitude"), scale, ref, unit, convention, relative_to)

    def draw(ax_):
        data_lines = _plot_lines(ax_, items, pairs)
        for k, label, (tx, ty) in truths:
            # thin and dashed above the data: the estimate shows through the gaps, the truth never vanishes
            colour = None if k is None else darken(data_lines[k].get_color())
            draw_reference(ax_, tx, ty, label=label, colour=colour)
        ax_.set_xlabel(x_label)
        ax_.set_ylabel(y_label)
        _finish_legend(ax_)

    return _on_axes(ax, draw)


def order_slice(
    s,
    *,
    ax=None,
    ref=1.0,
    unit="",
    channel=0,
    x="rpm",
    scale="db",
    relative_to=None,
    truth=None,
):
    """Draw the level of one order against speed (or time).

    The figure of the result of :func:`rota_order.slices.order_slice`: one value per
    map frame, ``20 log10(amplitude / ref)`` in dB, or the amplitude itself with
    ``scale="linear"``. The amplitude keeps the convention of the slice (peak or
    RMS, ``s.amplitude_convention``), which the y label names ("Order level (dB
    re 20 µPa, RMS)"; linear: "Order amplitude (Pa, RMS)"). NaN frames (the
    band leaves the map) and zero amplitudes are gaps.

    Parameters
    ----------
    s : rota_order.slices.OrderSlice, or list or dict of them
        One slice, a list, or a dict ``{label: slice}`` whose keys become the
        legend. All slices must have the same amplitude convention.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created only when None.
    ref : float, default 1.0
        Reference of the dB scale in the unit of the signal (``2e-5`` for sound
        pressure in Pa), explicit and never assumed; ignored for
        ``scale="linear"`` and for differences.
    unit : str, default ""
        Unit of the signal (``"Pa"``), used in the label only. Both are given
        together by ``rota_order.plot.SOUND_PRESSURE``.
    channel : int, default 0
        Channel of a multichannel slice.
    x : {"rpm", "time"}, default "rpm"
        Frame speed in rpm or frame time in s on the x axis. ``"rpm"`` is always
        allowed: the line is drawn at the true frame speeds and may double back
        where the speed does (a measured speed has small reversals, a gear shift
        a large drop); that is the data.
    scale : {"db", "linear"}, default "db"
        Level in dB re ``ref``, or the amplitude in the signal unit.
    relative_to : str, optional
        Label (key of the dict) of the entry to subtract from every entry: the
        difference of the levels in dB (``scale="db"``) or of the amplitudes in
        the signal unit. The slices must have the same frames (a difference
        between different grids would be a resampling).
    truth : (array_like, array_like) or dict, optional
        ``(x, amplitude)``: an analytic or reference curve, in the same scale and
        with the same ``ref``, labelled "truth". ``x`` is in rpm or s as selected
        by ``x``; the amplitude is in the signal unit and in the convention of the
        slices (the caller's responsibility). It is drawn in the reference style
        shared with ``reference=`` of :func:`rota_order.plot.rpm_vs_time`: dashed, in a
        neutral dark grey, thinner than the data and *above* it (the
        estimate shows through the gaps of the dashes, and the truth does not
        disappear under an estimate that agrees with it). When ``s`` is a
        dict, ``truth`` can be a dict with keys among those of ``s`` and one
        ``(x, amplitude)`` pair each: one curve per key, in a darker shade of the
        colour of that entry (the colour mixed 35 % towards black, so that it stays visible over the line) and labelled "<key> truth". Several calls on one
        Axes that each draw "truth" list it once in the legend. Not combined with
        ``relative_to``.

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
        If an entry is not an ``OrderSlice``, or ``ax`` is not an Axes.
    ValueError
        For an empty list or dict, mixed amplitude conventions, invalid ``ref``,
        ``scale``, ``x`` or ``truth`` (also a key of a ``truth`` dict that is not
        a label), a ``relative_to`` that is not a label (or
        results without labels), or slices on different frames.
    IndexError
        If ``channel`` is out of range.

    Notes
    -----
    The slice sums the power over a band of the map, so it reads the level of an
    order without the scalloping loss of the window; the level of a map cell
    can be up to 1.4 dB lower (Hann window).

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 9 and 12. Wiley.
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95.
    """
    return _plot_slices(
        s, OrderSlice, "a rota_order.slices.OrderSlice", "order_slice", "Order level", ax=ax, ref=ref, unit=unit,
        channel=channel, x=x, scale=scale, relative_to=relative_to, truth=truth,
    )


def frequency_slice(
    s,
    *,
    ax=None,
    ref=1.0,
    unit="",
    channel=0,
    x="rpm",
    scale="db",
    relative_to=None,
    truth=None,
):
    """Draw the level of one fixed frequency against speed (or time).

    The figure of the result of :func:`rota_order.slices.frequency_slice`, for example
    the response of a structural resonance or a fixed tone over a run-up: the band
    amplitude per map frame as ``20 log10(amplitude / ref)`` in dB, or the
    amplitude itself with ``scale="linear"``. The amplitude keeps the convention
    of the slice (peak or RMS), which the y label names ("Band level (dB re 20
    µPa, RMS)"; linear: "Band amplitude (Pa, RMS)"). NaN frames and zero
    amplitudes are gaps.

    Parameters
    ----------
    s : rota_order.slices.FrequencySlice, or list or dict of them
        One slice, a list, or a dict ``{label: slice}`` whose keys become the
        legend. All slices must have the same amplitude convention.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created only when None.
    ref : float, default 1.0
        Reference of the dB scale in the unit of the signal (``2e-5`` for sound
        pressure in Pa), explicit and never assumed; ignored for
        ``scale="linear"`` and for differences.
    unit : str, default ""
        Unit of the signal (``"Pa"``), used in the label only.
    channel : int, default 0
        Channel of a multichannel slice.
    x : {"rpm", "time"}, default "rpm"
        Frame speed in rpm or frame time in s on the x axis. ``"rpm"`` is always
        allowed (the line is drawn at the true frame speeds), as in
        :func:`rota_order.plot.order_slice`.
    scale : {"db", "linear"}, default "db"
        Level in dB re ``ref``, or the amplitude in the signal unit.
    relative_to : str, optional
        Label (key of the dict) of the entry to subtract from every entry (dB or
        signal unit); the slices must have the same frames.
    truth : (array_like, array_like) or dict, optional
        ``(x, amplitude)`` labelled "truth", or, when ``s`` is a dict, a dict
        ``{label: (x, amplitude)}`` with one truth per entry ("<label> truth"),
        in the same scale and with the same ``ref``; ``x`` in rpm or s as selected
        by ``x``, the amplitude in the signal unit and the convention of the
        slices. Drawn above the data in the common reference style, as in
        :func:`rota_order.plot.order_slice`. Not combined with ``relative_to``.

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
        If an entry is not a ``FrequencySlice``, or ``ax`` is not an Axes.
    ValueError
        As for :func:`rota_order.plot.order_slice`.
    IndexError
        If ``channel`` is out of range.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 9 and 12. Wiley.
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95.
    """
    return _plot_slices(
        s, FrequencySlice, "a rota_order.slices.FrequencySlice", "frequency_slice", "Band level", ax=ax, ref=ref,
        unit=unit, channel=channel, x=x, scale=scale, relative_to=relative_to, truth=truth,
    )


# --------------------------------------------------------------------------
# Averaged spectra
# --------------------------------------------------------------------------


def _draw_orders(ax, orders, lo, hi):
    """Thin dashed vertical lines at the orders within ``[lo, hi]``, labelled at the top."""
    for o in orders:
        if not lo <= o <= hi:
            continue  # outside the spectrum: it would stretch the axis
        ax.axvline(o, color="0.45", linewidth=0.7, linestyle=(0, (4, 3)), zorder=0.5)
        ax.annotate(
            f"{o:g}",
            xy=(o, 1.0),
            xycoords=ax.get_xaxis_transform(),
            xytext=(2, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            color="0.3",
            fontsize="x-small",
        )


def order_spectrum(
    spec,
    *,
    ax=None,
    ref=1.0,
    unit="",
    channel=0,
    scale="db",
    relative_to=None,
    mark_orders=None,
):
    """Draw an energy-averaged spectrum: level against order (or frequency).

    The figure of the result of :func:`rota_order.slices.order_spectrum`: per bin the
    level ``20 log10(amplitude / ref)`` in dB (``amplitude`` is the RMS over the
    averaged frames of the map's amplitude), or the amplitude with
    ``scale="linear"``. The amplitude keeps the convention of the map (peak or
    RMS), which the y label names ("Spectrum level (dB re 20 µPa, RMS)"). The x
    axis is "Order" or "Frequency (Hz)" as the spectrum's domain. Zero
    amplitudes and NaN are gaps.

    Parameters
    ----------
    spec : rota_order.slices.Spectrum, or list or dict of them
        One spectrum, a list, or a dict ``{label: spectrum}`` whose keys become
        the legend. All spectra must have the same domain and amplitude
        convention.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created only when None.
    ref : float, default 1.0
        Reference of the dB scale in the unit of the signal (``2e-5`` for sound
        pressure in Pa), explicit and never assumed.
    unit : str, default ""
        Unit of the signal (``"Pa"``), used in the label only.
    channel : int, default 0
        Channel of a multichannel spectrum.
    scale : {"db", "linear"}, default "db"
        Level in dB re ``ref``, or the amplitude in the signal unit.
    relative_to : str, optional
        Label (key of the dict) of the entry to subtract from every entry (dB or
        signal unit): the before/after view of an order removal. The spectra must
        have the same axis.
    mark_orders : float or sequence of float, optional
        Orders to mark (not to select: a spectrum is drawn whole): thin dashed vertical lines labelled with the order.
        Orders outside the spectrum are left out. Only for spectra of an order
        map (``domain="order"``).

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
        If an entry is not a ``Spectrum``, or ``ax`` is not an Axes.
    ValueError
        For an empty list or dict, mixed domains or amplitude conventions,
        invalid ``ref``, ``scale`` or ``mark_orders``, ``mark_orders`` with a frequency
        spectrum, a ``relative_to`` that is not a label, or spectra on different
        axes.
    IndexError
        If ``channel`` is out of range.

    Notes
    -----
    Each bin shows the level of the bin, not of a line: a line spread over
    several bins reads lower in each (see ``Spectrum.enbw_bins``).

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 9 and 12. Wiley.
    Welch, P. D. (1967). The use of fast Fourier transform for the estimation of
    power spectra. IEEE Transactions on Audio and Electroacoustics, 15(2), 70-73.
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95.
    """
    items = _results(spec, Spectrum, "order_spectrum", "a rota_order.slices.Spectrum")
    ref = check_ref(ref)
    scale = _check_scale(scale)
    convention = _same_convention(items)
    domains = sorted({r.domain for _, r in items})
    if len(domains) > 1:
        raise ValueError(f"the spectra have different domains {domains}; draw them separately")
    domain = domains[0]
    if domain not in ("order", "frequency"):
        raise ValueError(f"unknown spectrum domain {domain!r}")
    marks = _orders_array(mark_orders)
    if marks.size and domain != "order":
        raise ValueError("mark_orders can only be marked on a spectrum of an order map (domain='order')")
    pairs = []
    for _, r in items:
        axis = np.asarray(r.axis, dtype=float)
        amp = _real_float(r.amplitude, "the amplitude of a spectrum")
        if axis.ndim != 1 or amp.ndim not in (1, 2) or amp.shape[0] != axis.size:
            raise ValueError("the axis and the amplitude of the spectrum must have one value per bin")
        a = select_channel(amp, channel, 1)
        pairs.append((axis, to_db(a, ref) if scale == "db" else a))
    pairs = relative(pairs, _legend_labels(items), relative_to)
    y_label = _amplitude_label(
        "Spectrum level", "Spectrum amplitude", scale, ref, unit, convention, relative_to
    )
    x_label = "Order" if domain == "order" else "Frequency (Hz)"
    lo = min(float(p[0].min()) for p in pairs if p[0].size) if any(p[0].size for p in pairs) else 0.0
    hi = max(float(p[0].max()) for p in pairs if p[0].size) if any(p[0].size for p in pairs) else 0.0

    def draw(ax_):
        _plot_lines(ax_, items, pairs)
        if marks.size:
            _draw_orders(ax_, marks, lo, hi)
        ax_.set_xlabel(x_label)
        ax_.set_ylabel(y_label)
        _finish_legend(ax_)

    return _on_axes(ax, draw)


# --------------------------------------------------------------------------
# Overall level
# --------------------------------------------------------------------------


def overall_level(
    ol,
    *,
    ax=None,
    ref=None,
    unit="",
    channel=0,
    binned=True,
    relative_to=None,
):
    """Draw an overall level against speed (per speed bin) or against time (per frame).

    The figure of the result of :func:`rota_order.slices.overall_level`. The levels
    are RMS levels in dB, already weighted if ``ol.weighting`` is "A" or "C", and
    are plotted as stored (``level``, or ``binned_level``) re the reference the
    result carries (``ol.ref``) unless ``ref`` is given, which re-references
    them exactly (see below). The y label names the weighting and the reference
    ("A-weighted level (dB re 20 µPa, RMS)" with ``**rota_order.plot.SOUND_PRESSURE``).
    With ``binned=True`` the energy-mean level of each speed bin is drawn against
    the bin centres in rpm (empty bins are gaps); with ``binned=False`` the level
    of each frame against time in s (a silent frame, ``-inf``, is a gap).

    Parameters
    ----------
    ol : rota_order.slices.OverallLevel, or list or dict of them
        One result, a list, or a dict ``{label: result}`` whose keys become the
        legend. They may differ in weighting (the label then says "Level"). With
        ``ref=None`` they must have the same ``ol.ref``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created only when None.
    ref : float, optional
        Reference of the dB scale in the unit of the signal. None (default)
        plots the levels as stored, re ``ol.ref``. A given ``ref`` re-references
        them exactly, ``L + 20 log10(ol.ref / ref)``, and labels the axis with it
        (``ref=2e-5`` for sound pressure in Pa); results with different stored
        references can then be drawn together. Without effect on differences
        (``relative_to``), which do not depend on the reference.
    unit : str, default ""
        Unit of the signal (``"Pa"``), used in the label only: ``unit`` never
        changes a number. ``**rota_order.plot.SOUND_PRESSURE`` gives ``ref`` and
        ``unit`` together.
    channel : int, default 0
        Channel of a multichannel result.
    binned : bool, default True
        Speed bins (needs ``rpm_bins`` in :func:`rota_order.slices.overall_level`) or
        frames.
    relative_to : str, optional
        Label (key of the dict) of the entry to subtract from every entry: the
        difference of the levels in dB. The bins (or frames) must be the same.

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
        If an entry is not an ``OverallLevel``, or ``ax`` is not an Axes.
    ValueError
        For an empty list or dict, an invalid ``ref``, results with different
        stored references and ``ref=None``, ``binned`` results without speed
        bins, a ``relative_to`` that is not a label, or results on different bins
        or frames.
    IndexError
        If ``channel`` is out of range.

    Notes
    -----
    The levels are RMS levels (``20 log10(x_rms / ref)``) per frame; the level of
    a bin is the energy mean of its frames, ``10 log10(mean(10**(L / 10)))``.
    Re-referencing adds a constant to every level, so it commutes with the energy
    mean.

    References
    ----------
    IEC 61672-1:2013. Electroacoustics - Sound level meters - Part 1:
    Specifications.
    ISO 1683:2015. Acoustics - Preferred reference values for acoustical and
    vibratory levels.
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95.
    """
    items = _results(ol, OverallLevel, "overall_level", "a rota_order.slices.OverallLevel")
    binned = bool(binned)
    stored = [check_ref(r.ref) for _, r in items]
    if ref is None and not np.allclose(stored, stored[0], rtol=1e-12, atol=0.0):
        raise ValueError(
            f"the results have different references {sorted(set(stored))}; give ref= to express "
            "all levels re one reference"
        )
    ref_used, _ = level_reference(stored[0], ref)
    pairs = []
    for (label, r), r_stored in zip(items, stored):
        if binned:
            if r.centres is None or r.binned_level is None:
                who = f"entry {label!r}" if label is not None else "the result"
                raise ValueError(
                    f"{who} has no speed bins: pass rpm_bins to rota_order.slices.overall_level, "
                    "or draw the frames with binned=False"
                )
            xv = np.asarray(r.centres, dtype=float)
            yv = _real_float(r.binned_level, "the binned level")
        else:
            xv = np.asarray(r.t, dtype=float)
            yv = _real_float(r.level, "the level")
        if xv.ndim != 1 or yv.ndim not in (1, 2) or yv.shape[0] != xv.size:
            raise ValueError("the level must have one value per bin or frame")
        shift = level_reference(r_stored, ref)[1]
        pairs.append((xv, _gap(select_channel(yv, channel, 1) + shift)))
    pairs = relative(pairs, _legend_labels(items), relative_to)
    weightings = {r.weighting for _, r in items}
    quantity = "Level"
    if len(weightings) == 1 and None not in weightings:
        quantity = f"{next(iter(weightings))}-weighted level"
    if relative_to is None:
        y_label = level_label(quantity, ref_used, unit, "rms")  # the stored levels are RMS levels
    else:
        y_label = f"{quantity} relative to {relative_to} (dB)"
    x_label = X_LABELS["rpm" if binned else "time"]

    def draw(ax_):
        _plot_lines(ax_, items, pairs, marker="o" if binned else None)
        ax_.set_xlabel(x_label)
        ax_.set_ylabel(y_label)
        _finish_legend(ax_)

    return _on_axes(ax, draw)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _metric_label(info, relative_to):
    """Y label ``"<name> (<unit>)"`` from the ``(name, unit)`` of the entries."""
    units = {u for _, u in info}
    if len(units) > 1:
        raise ValueError(f"the metrics have different units {sorted(units)} and cannot share an axis")
    unit = next(iter(units))
    names = {n for n, _ in info if n}
    name = next(iter(names)) if len(names) == 1 else "Metric value"
    # a name that already ends with its unit in parentheses ("Loudness (sone)") is not given it twice
    tail = f" ({unit})" if unit and not re.search(r"\([^()]*\)\s*$", name) else ""
    if relative_to is None:
        return f"{name}{tail}"
    return f"{name} relative to {relative_to}{tail}"


def _entry_label(label, r):
    """Legend text of a metric: the dict key, else the ``source`` of the result, else None."""
    if label is not None:
        return label
    return r.source or None


def metric_vs_rpm(
    r,
    *,
    ax=None,
    spread=True,
    relative_to=None,
    control_unit="rpm",
):
    """Draw a metric aggregated along speed, with its spread band.

    The figure of the result of :func:`rota_order.slices.metric_vs_rpm` (or of
    ``TimeMetric.vs_rpm``): the aggregated value of each speed bin against the bin
    centre, as a line with markers. Bins withheld because they hold fewer than
    ``min_count`` samples (``valid`` False) are gaps. With ``spread=True`` a band
    shows the spread of the samples in each bin: the value plus and minus one
    standard deviation (``spread_kind="std"``), or, for the interquartile range
    (``"iqr"``), a band centred on the value with the IQR as its total width (the
    quartiles themselves are not stored). The y label is "<name> (<unit>)" from
    the result when it carries them; the legend text is the dict key, else the
    ``source`` of the result. The values are a perceptual or physical quantity in
    ``unit``, not an amplitude: no peak or RMS convention applies.

    Parameters
    ----------
    r : rota_order.slices.MetricVsRpm, or list or dict of them
        One result, a list, or a dict ``{label: result}`` whose keys become the
        legend. All results must have the same ``unit``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created only when None.
    spread : bool, default True
        Draw the band (labelled "± std" or "IQR" in the legend).
    relative_to : str, optional
        Label (key of the dict) of the entry to subtract from every entry: the
        difference in the unit of the metric. The bins must be the same; the
        bands are left out (the spread of a difference is not known).
    control_unit : str, default "rpm"
        Unit of the control the metric was binned along, for the x label
        ("Speed (rpm)"); ``metric_vs_rpm`` accepts any control, for example a
        vehicle speed in km/h.

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
        If an entry is not a ``MetricVsRpm``, or ``ax`` is not an Axes.
    ValueError
        For an empty list or dict, metrics in different units, an unknown
        ``spread_kind``, a ``relative_to`` that is not a label, or results on
        different bins.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95.
    """
    items = _results(r, MetricVsRpm, "metric_vs_rpm", "a rota_order.slices.MetricVsRpm")
    pairs, bands = [], []
    for _, m in items:
        c = np.asarray(m.centres, dtype=float)
        v = _real_float(m.value, "the metric value")
        ok = np.asarray(m.valid, dtype=bool)
        if c.ndim != 1 or v.shape != c.shape or ok.shape != c.shape:
            raise ValueError("centres, value and valid of the result must have one value per bin")
        if m.spread_kind not in ("std", "iqr"):
            raise ValueError(f"unknown spread_kind {m.spread_kind!r} (expected 'std' or 'iqr')")
        sp = _real_float(m.spread, "the spread")
        if sp.shape != c.shape:
            raise ValueError("spread of the result must have one value per bin")
        y = _gap(np.where(ok, v, np.nan))
        half = _gap(np.where(ok, sp, np.nan)) * (0.5 if m.spread_kind == "iqr" else 1.0)
        pairs.append((c, y))
        bands.append(half)
    labels = [lab for lab, _ in items]
    pairs_plot = relative(pairs, labels, relative_to)
    draw_bands = bool(spread) and relative_to is None
    y_label = _metric_label([(m.name, m.unit) for _, m in items], relative_to)
    x_label = f"Speed ({control_unit})" if control_unit else "Speed"
    plot_items = [(_entry_label(lab, m), m) for lab, m in items]

    def draw(ax_):
        lines = _plot_lines(ax_, plot_items, pairs_plot, marker="o")
        if draw_bands:
            for i, ((lab, m), (c, y), half, line) in enumerate(zip(plot_items, pairs, bands, lines)):
                text = "± std" if m.spread_kind == "std" else "IQR"
                if lab is not None:
                    band_label = f"{lab} ({text})"
                else:
                    band_label = text if i == 0 else "_nolegend_"
                ax_.fill_between(c, y - half, y + half, color=line.get_color(), alpha=0.2, linewidth=0, label=band_label)
        ax_.set_xlabel(x_label)
        ax_.set_ylabel(y_label)
        _finish_legend(ax_)

    return _on_axes(ax, draw)


def time_metric(
    m,
    *,
    ax=None,
    relative_to=None,
):
    """Draw a time-varying metric against time, with its start-up exclusion shaded.

    The figure of a :class:`rota_order.bridges.TimeMetric` (a loudness, sharpness,
    roughness or tonality track from a psychoacoustic toolbox): ``values`` against
    ``t`` in s as computed, NaN being a gap. The start-up exclusion
    ``skip_start`` (counted from ``t[0]``), whose samples are not valid because
    of filter and integrator transients, is shaded and labelled "start-up
    excluded". The y label is "<name> (<unit>)"; the legend text is the dict
    key, else the ``source`` of the metric. The values are a perceptual
    quantity in ``unit``, not an amplitude: no peak or RMS convention applies.

    Parameters
    ----------
    m : rota_order.bridges.TimeMetric, or list or dict of them
        One metric, a list, or a dict ``{label: metric}`` whose keys become the
        legend. All metrics must have the same ``unit``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created only when None.
    relative_to : str, optional
        Label (key of the dict) of the entry to subtract from every entry: the
        difference in the unit of the metric. The metrics must have the same
        time axis.

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
        If an entry is not a ``TimeMetric``, or ``ax`` is not an Axes.
    ValueError
        For an empty list or dict, metrics in different units, a ``relative_to``
        that is not a label, or metrics on different time axes.

    References
    ----------
    ISO 532-1:2017. Acoustics - Methods for calculating loudness - Part 1:
    Zwicker method.
    ECMA-418-2:2025. Psychoacoustic metrics for ITT equipment - Part 2: Models
    based on human perception.
    Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95.
    """
    items = _results(m, TimeMetric, "time_metric", "a rota_order.bridges.TimeMetric")
    pairs, spans = [], []
    for _, tm in items:
        t = np.asarray(tm.t, dtype=float)
        v = _real_float(tm.values, "the metric values")
        if t.ndim != 1 or v.shape != t.shape:
            raise ValueError("t and values of the metric must have one value per time")
        pairs.append((t, _gap(v)))
        skip = float(tm.skip_start) if tm.skip_start is not None else 0.0
        if np.isfinite(skip) and skip > 0 and t.size:
            spans.append((float(t[0]), min(float(t[0]) + skip, float(t[-1]))))
    labels = [lab for lab, _ in items]
    pairs_plot = relative(pairs, labels, relative_to)
    y_label = _metric_label([(tm.name, tm.unit) for _, tm in items], relative_to)
    plot_items = [(_entry_label(lab, tm), tm) for lab, tm in items]
    distinct = list(dict.fromkeys(spans))

    def draw(ax_):
        _plot_lines(ax_, plot_items, pairs_plot)
        for k, (a, b) in enumerate(distinct):
            ax_.axvspan(a, b, color="0.5", alpha=0.2, linewidth=0, label="start-up excluded" if k == 0 else "_nolegend_")
        ax_.set_xlabel(X_LABELS["time"])
        ax_.set_ylabel(y_label)
        _finish_legend(ax_)

    return _on_axes(ax, draw)
