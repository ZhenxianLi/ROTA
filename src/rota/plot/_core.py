"""Private helpers shared by the modules of :mod:`rota.plot`.

Everything here is NumPy except :func:`pyplot` and :func:`get_ax`, which import
Matplotlib on first use (never when ``rota.plot`` is imported). Nothing here
selects a backend, changes ``rcParams`` or styles, or calls ``show``.

Conventions of the helpers. Levels are in dB, ``20 log10(|a| / ref)``, with
``ref`` in the unit of the plotted signal; a zero amplitude has no level and
becomes NaN, which the figures draw as a gap. Speed is in rpm, time in s,
frequency in Hz, order in cycles per revolution. Labels are built from the
result that is plotted, so the quantity, the dB reference and the amplitude
convention (peak or RMS) are always named.
"""

import operator
from collections.abc import Mapping

import numpy as np

__all__ = [
    "MONOTONIC_TOL",
    "REFERENCE_COLOUR",
    "REFERENCE_ZORDER",
    "SOUND_PRESSURE",
    "X_LABELS",
    "as_items",
    "check_finite",
    "check_positive",
    "check_ref",
    "check_tol",
    "choose_x",
    "colour_limits",
    "darken",
    "draw_reference",
    "edges_from_centres",
    "essentially_monotonic",
    "get_ax",
    "label_index",
    "legend_without_repeats",
    "level_label",
    "level_reference",
    "pyplot",
    "ref_text",
    "relative",
    "select_channel",
    "speed_envelope",
    "to_db",
]

# Reference for sound pressure in Pa (ISO 1683), the explicit choice for acoustic
# signals: ``rota.plot.order_slice(s, **rota.plot.SOUND_PRESSURE)``.
SOUND_PRESSURE = {"ref": 2e-5, "unit": "Pa"}

# The one style of every reference curve of rota.plot (``truth=`` of the slices, ``reference=`` of
# the speed figures): a neutral dark grey, dashed, thinner than the data and drawn above it
# (Matplotlib draws lines at zorder 2), so that the reference never disappears under an identical
# estimate, while the estimate shows through the gaps of the dashes. See :func:`draw_reference`.
REFERENCE_COLOUR = "0.25"
REFERENCE_ZORDER = 2.5
_REFERENCE_WIDTH = 0.7  # fraction of the default line width
_REFERENCE_GID = "rota-reference"

# Largest reversal of a speed that still counts as a run-up or a coast-down, as a
# fraction of the speed range (see essentially_monotonic).
MONOTONIC_TOL = 0.02

# Axis labels of the two ways to put a frame on the x axis (see choose_x).
X_LABELS = {"rpm": "Speed (rpm)", "time": "Time (s)"}

_INSTALL_HINT = (
    "rota.plot needs Matplotlib: pip install 'rota-order[plot]' (ROTA itself does not depend on it)"
)

# SI prefixes by power of 1000 (p to G); the micro sign is U+00B5 (not the Greek mu U+03BC).
_PREFIXES = {-4: "p", -3: "n", -2: "\u00b5", -1: "m", 0: "", 1: "k", 2: "M", 3: "G"}

_CONVENTIONS = {None: "", "peak": ", peak", "rms": ", RMS"}


# --------------------------------------------------------------------------
# Matplotlib access
# --------------------------------------------------------------------------


def pyplot():
    """Return ``matplotlib.pyplot``, imported on first use.

    The backend is never selected or changed here; the active backend and style
    are the user's.

    Raises
    ------
    ImportError
        "rota.plot needs Matplotlib: pip install 'rota-order[plot]' (ROTA itself
        does not depend on it)", chained to the original import error, if
        Matplotlib cannot be imported.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(_INSTALL_HINT) from err
    return plt


def darken(colour, amount=0.35):
    """RGB tuple of ``colour`` mixed with black (``amount`` 0: unchanged, 1: black)."""
    from matplotlib.colors import to_rgb

    rgb = np.asarray(to_rgb(colour), dtype=float)
    return tuple(float(c) for c in (1.0 - amount) * rgb)


def draw_reference(ax, x, y, *, label="reference", colour=None):
    """Draw a reference curve (a truth) in the common reference style; returns the Line2D.

    Dashed, 0.7 times the default line width and at ``REFERENCE_ZORDER`` (above the
    data lines), in the neutral dark grey ``REFERENCE_COLOUR`` unless ``colour`` is
    given (a darker shade of the colour of the data it belongs to). The artist is
    tagged so that :func:`legend_without_repeats` keeps one legend entry for the same
    label drawn several times on one Axes. NaN in ``y`` stays a gap.
    """
    from matplotlib import rcParams

    (line,) = ax.plot(
        x,
        y,
        color=REFERENCE_COLOUR if colour is None else colour,
        linestyle="--",
        linewidth=_REFERENCE_WIDTH * float(rcParams["lines.linewidth"]),
        zorder=REFERENCE_ZORDER,
        label=label,
        gid=_REFERENCE_GID,
    )
    return line


def legend_without_repeats(ax):
    """Add the legend of ``ax`` if anything is labelled; a reference label is listed once.

    Several calls on one Axes can each draw a reference with the same label (a
    "truth" per slice); the legend keeps the first of them, and lists the references
    after the data. Data lines are never merged, even if they share a label,
    because their colours differ.
    """
    handles, labels = ax.get_legend_handles_labels()
    seen, data, refs = set(), [], []
    for handle, label in zip(handles, labels):
        if getattr(handle, "get_gid", lambda: None)() == _REFERENCE_GID:
            if label not in seen:
                seen.add(label)
                refs.append((handle, label))
        else:
            data.append((handle, label))
    entries = data + refs  # the references after the data they belong to
    if entries:
        ax.legend([h for h, _ in entries], [lab for _, lab in entries])


def get_ax(ax=None):
    """Return ``(fig, ax)`` for a figure function.

    ``ax=None``: a new figure and Axes from ``pyplot().subplots()`` (active
    style and default size), so Matplotlib is imported here and the ImportError
    of :func:`pyplot` can be raised. Otherwise ``(ax.figure, ax)``: nothing is
    created and Matplotlib is not imported. Call it after the arguments have
    been validated, so that a failed call does not leave a figure behind.

    Raises
    ------
    TypeError
        If ``ax`` is given but is not a Matplotlib Axes (an array of Axes from
        ``plt.subplots(1, 2)`` is the usual mistake).
    """
    if ax is None:
        fig, ax = pyplot().subplots()
        return fig, ax
    if not (hasattr(ax, "plot") and hasattr(ax, "set_xlabel") and hasattr(ax, "figure")):
        raise TypeError(f"ax must be a Matplotlib Axes, got {type(ax).__name__}")
    return ax.figure, ax


# --------------------------------------------------------------------------
# Levels and labels
# --------------------------------------------------------------------------


def check_ref(ref):
    """Return ``ref`` as a float; ValueError unless it is finite and > 0."""
    try:
        r = float(ref)
    except (TypeError, ValueError):
        raise ValueError(f"ref must be a positive number, got {ref!r}") from None
    if not (np.isfinite(r) and r > 0):
        raise ValueError(f"ref must be a positive, finite number, got {ref!r}")
    return r


def check_finite(value, name):
    """Return ``value`` as a float; ValueError (naming ``name``) unless it is finite."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number, got {value!r}") from None
    if not np.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return v


def check_positive(value, name):
    """Return ``value`` as a float; ValueError (naming ``name``) unless it is finite and > 0."""
    v = check_finite(value, name)
    if not v > 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return v


def to_db(amplitude, ref):
    """Level ``20 log10(|amplitude| / ref)`` in dB.

    Parameters
    ----------
    amplitude : array_like, real or complex
        Amplitude in the unit of ``ref``; the magnitude is taken, so complex
        map values can be passed as they are. The amplitude convention (peak or
        RMS) is that of the input and carries over to the level.
    ref : float
        Reference in the unit of ``amplitude``, finite and > 0.

    Returns
    -------
    ndarray of float, same shape as ``amplitude`` (a NumPy float for a scalar)
        Zero amplitudes give NaN, not ``-inf``; NaN stays NaN and an infinite
        amplitude gives ``+inf``; no warnings are raised for zeros, NaN, infinite
        values or overflow. Equal inputs give bit-equal levels, also when they
        are views of a multichannel map (see the note in the code).

    Raises
    ------
    ValueError
        If ``ref`` is not a finite number > 0.
    """
    r = check_ref(ref)
    a = np.asarray(amplitude)
    if np.iscomplexobj(a) and a.ndim > 0:
        # NumPy 1.x computes np.abs of a strided complex array with SIMD loops whose
        # result can differ by one ulp between calls on the same data (it depends on
        # where the temporary lands in memory). A contiguous array is reproducible and
        # gives the same values as np.abs of a contiguous map, so equal maps give equal
        # levels and the difference of identical maps is exactly zero.
        a = np.ascontiguousarray(a)
    a = np.abs(a).astype(float, copy=False)
    with np.errstate(all="ignore"):
        out = np.where(a > 0, 20.0 * np.log10(a / r), np.nan)
    return out[()] if out.ndim == 0 else out


def ref_text(ref, unit=""):
    """Text of a dB reference, with an SI prefix for a unit.

    ``ref_text(2e-5, "Pa")`` is ``"20 µPa"``, ``ref_text(1e-6, "m/s²")`` is
    ``"1 µm/s²"``, ``ref_text(1, "Pa")`` is ``"1 Pa"`` and
    ``ref_text(1.0, "")`` is ``"1"``. With a unit the prefix (p, n, µ, m, none,
    k, M, G; the micro sign is U+00B5) is chosen so that the number lies in
    [1, 1000); outside p to G the number is printed with its exponent and no
    prefix. Without a unit the number is printed as it is (``"1e-06"``): a
    prefix on a pure number would be meaningless. Numbers carry at most ten
    significant digits and no trailing zeros.

    Raises
    ------
    ValueError
        If ``ref`` is not a finite number > 0.
    """
    r = check_ref(ref)
    unit = "" if unit is None else str(unit)
    if not unit:
        return f"{r:g}"
    mantissa, exponent = f"{r:.9e}".split("e")  # rounding fixes the exponent of 0.999999999e-3
    e = int(exponent)
    k = e // 3
    if k not in _PREFIXES:
        return f"{r:g} {unit}"
    value = float(mantissa) * 10.0 ** (e - 3 * k)
    return f"{value:.10g} {_PREFIXES[k]}{unit}"


def level_label(quantity, ref, unit="", convention=None):
    """Label of a level axis or colour bar, e.g. ``"Order level (dB re 20 µPa, RMS)"``.

    Parameters
    ----------
    quantity : str
        What the level is of, e.g. ``"Order level"`` or ``"Level"``.
    ref, unit
        Reference and its unit, see :func:`ref_text`.
    convention : {"peak", "rms", None}, optional
        Amplitude convention of the plotted values (case-insensitive): adds
        ``", peak"`` or ``", RMS"``; None adds nothing (for levels that are not
        amplitudes, or whose convention is stated elsewhere).

    Raises
    ------
    ValueError
        For an invalid ``ref`` or ``convention``.
    """
    key = convention.lower() if isinstance(convention, str) else convention
    if key not in _CONVENTIONS:
        raise ValueError(f"convention must be 'peak', 'rms' or None, got {convention!r}")
    return f"{quantity} (dB re {ref_text(ref, unit)}{_CONVENTIONS[key]})"


def level_reference(stored_ref, ref=None):
    """Reference to label with and the shift in dB for levels that are stored in dB.

    Results such as :class:`rota.slices.OverallLevel` and
    :class:`rota.tonal.BandRpmLevels` hold levels in dB re a reference they
    carry (``stored_ref``). ``ref=None`` plots them as stored. A given ``ref``
    re-references them exactly: ``L_new = L + 20 log10(stored_ref / ref)``, so
    that ``20 log10(a / stored_ref) + 20 log10(stored_ref / ref) = 20 log10(a / ref)``.

    Returns
    -------
    ref_used, shift : float, float
        The reference of the plotted levels and the dB to add to the stored
        levels (exactly 0 when ``ref`` is None or equals ``stored_ref``).

    Raises
    ------
    ValueError
        If ``stored_ref`` or ``ref`` is not a finite number > 0.
    """
    stored = check_ref(stored_ref)
    if ref is None:
        return stored, 0.0
    ref = check_ref(ref)
    return ref, float(20.0 * np.log10(stored / ref))


def colour_limits(arrays, dynamic_range=40.0, vmin=None, vmax=None):
    """Colour limits ``(vmin, vmax)`` of a level image, floats in dB.

    ``vmax`` is the largest finite value in ``arrays`` unless given, and
    ``vmin = vmax - dynamic_range`` unless given, so the scale spans
    ``dynamic_range`` dB below the maximum of what is shown. ``arrays`` is one
    array or a sequence of arrays (a shared scale over several images); NaN and
    infinite values are ignored.

    Raises
    ------
    ValueError
        If ``dynamic_range`` is not a finite number > 0, if ``vmin``/``vmax``
        are not finite, if ``vmin >= vmax``, or if ``vmax`` is not given and
        ``arrays`` holds no finite value.
    """
    dr = check_positive(dynamic_range, "dynamic_range")
    if vmax is None:
        if isinstance(arrays, np.ndarray):
            arrays = [arrays]
        top = -np.inf
        for a in arrays:
            a = np.asarray(a, dtype=float)
            finite = a[np.isfinite(a)]
            if finite.size:
                top = max(top, float(finite.max()))
        if not np.isfinite(top):
            raise ValueError(
                "no finite level to show (all values are zero or not finite); give vmax to draw anyway"
            )
        vmax = top
    else:
        vmax = check_finite(vmax, "vmax")
    vmin = vmax - dr if vmin is None else check_finite(vmin, "vmin")
    if not vmin < vmax:
        raise ValueError(f"vmin ({vmin:g}) must be below vmax ({vmax:g})")
    return vmin, vmax


# --------------------------------------------------------------------------
# One result, a list, or a dict of results
# --------------------------------------------------------------------------


def as_items(obj):
    """Normalise one result, a list/tuple of results or a dict to ``[(label, result), ...]``.

    A single result (anything that is not a list, tuple or mapping) gives
    ``[(None, result)]``; a list or tuple gives labels None; a dict gives its
    keys converted with ``str()`` as labels, in insertion order. The legend of
    a figure uses the labels that are not None.

    Raises
    ------
    ValueError
        For an empty list, tuple or dict.
    """
    if isinstance(obj, Mapping):
        items = [(str(key), value) for key, value in obj.items()]
    elif isinstance(obj, (list, tuple)):
        items = [(None, value) for value in obj]
    else:
        return [(None, obj)]
    if not items:
        raise ValueError("nothing to plot: the list or dict of results is empty")
    return items


def label_index(labels, name):
    """Index of the entry labelled ``name`` in ``labels`` (compared as ``str``).

    Raises
    ------
    ValueError
        If the entries are unlabelled (a list was passed instead of a dict) or
        no label equals ``name``; the message lists the labels.
    """
    known = [lab for lab in labels if lab is not None]
    if not known:
        raise ValueError(
            f"relative_to={name!r} needs labelled results: pass a dict {{label: result}}"
        )
    target = str(name)
    for i, lab in enumerate(labels):
        if lab is not None and lab == target:
            return i
    raise ValueError(f"relative_to={name!r} is not one of the labels {known}")


def relative(values_list, labels, relative_to):
    """Differences to one entry, for before/after figures.

    Parameters
    ----------
    values_list : list of (x, y)
        One pair per entry: the x grid (1-D) and the values on it (same length;
        levels in dB, amplitudes or a metric in its unit).
    labels : list of (str or None)
        Label of each entry, from :func:`as_items`.
    relative_to : str or None
        Label of the reference entry. None returns the entries unchanged.

    Returns
    -------
    list of (x, y - y_ref)
        Same length and order as ``values_list``; ``x`` as float arrays. The
        reference entry itself becomes zero (NaN where it was NaN); the unit is
        that of ``y`` (dB for levels). Use :func:`label_index` to find it.

    Raises
    ------
    ValueError
        If ``relative_to`` is not a label (also if the entries are unlabelled),
        or if the x grid or the shape of the values of an entry differs from
        the reference (a difference between different grids would be a
        resampling; the grids must agree to a relative 1e-9).
    """
    if relative_to is None:
        return list(values_list)
    if len(values_list) != len(labels):
        raise ValueError("values_list and labels must have the same length")
    j = label_index(labels, relative_to)
    x_ref = np.asarray(values_list[j][0], dtype=float)
    y_ref = np.asarray(values_list[j][1], dtype=float)
    out = []
    for i, (x, y) in enumerate(values_list):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != x_ref.shape or not np.allclose(x, x_ref, rtol=1e-9, atol=1e-12, equal_nan=True):
            raise ValueError(
                f"cannot subtract {labels[j]!r} from {labels[i]!r}: their x grids differ"
            )
        if y.shape != y_ref.shape:
            raise ValueError(
                f"cannot subtract {labels[j]!r} from {labels[i]!r}: their values differ in shape "
                f"({y.shape} and {y_ref.shape})"
            )
        with np.errstate(invalid="ignore"):  # inf - inf
            out.append((x, y - y_ref))
    return out


# --------------------------------------------------------------------------
# Channels and axes
# --------------------------------------------------------------------------


def select_channel(values, channel, channel_axis):
    """Select one channel of an array with an optional channel axis.

    Parameters
    ----------
    values : array_like
        Result values whose channel axis, if any, is ``channel_axis`` (2 for map
        values ``(n_bins, n_frames, ch)``, 1 for ``(n_frames, ch)`` or
        ``(n_bins, ch)``). An array with fewer dimensions is a single channel.
    channel : int
        Channel to select; negative counts from the last (as in NumPy).
    channel_axis : int
        Position of the channel axis.

    Returns
    -------
    ndarray
        A view without the channel axis; the array itself for a single-channel
        result with ``channel`` 0 (or -1).

    Raises
    ------
    IndexError
        If ``channel`` is out of range; the message names the number of
        channels, e.g. "the result has 2 channels".
    TypeError
        If ``channel`` is not an integer.
    """
    a = np.asarray(values)
    try:
        channel = operator.index(channel)
    except TypeError:
        raise TypeError(f"channel must be an integer, got {channel!r}") from None
    has_axis = a.ndim > channel_axis
    n = a.shape[channel_axis] if has_axis else 1
    if not -n <= channel < n:
        raise IndexError(
            f"channel {channel} is out of range: the result has {n} channel{'' if n == 1 else 's'} "
            f"(valid: 0 to {n - 1})"
        )
    if not has_axis:
        return a
    index = [slice(None)] * a.ndim
    index[channel_axis] = channel
    return a[tuple(index)]


def check_tol(tol):
    """Return ``tol`` as a float; ValueError unless it is a finite number >= 0."""
    try:
        v = float(tol)
    except (TypeError, ValueError):
        raise ValueError(f"monotonic_tol must be a number >= 0, got {tol!r}") from None
    if not (np.isfinite(v) and v >= 0):
        raise ValueError(f"monotonic_tol must be a finite number >= 0, got {tol!r}")
    return v


def speed_envelope(rpm, tol=MONOTONIC_TOL):
    """Monotone envelope of an essentially monotonic speed, or None.

    A measured speed (a tachometer, a ridge) is a run-up or a coast-down with
    small reversals. It is *essentially monotonic* when it is finite, has at
    least two frames, and, taking the overall direction from the sign of
    ``rpm[-1] - rpm[0]``, no frame falls below the running maximum (run-up) or
    rises above the running minimum (coast-down) by more than ``tol`` times the
    speed range (``max - min``). A speed that ends where it started, a constant
    speed and an up-and-down profile (a run-up followed by a coast-down, the
    drop of a gear shift, about 30 %) are not.

    Parameters
    ----------
    rpm : array_like, shape (n,)
        Frame speed in rpm.
    tol : float, default 0.02
        Largest reversal, as a fraction of the speed range (0.02 is 2 %); 0
        accepts only a non-decreasing or non-increasing speed.

    Returns
    -------
    ndarray of float, shape (n,) or None
        The running maximum (run-up) or running minimum (coast-down) of ``rpm``:
        a monotone sequence, each value differing from ``rpm`` by the reversal at
        that frame, at most ``tol * (max - min)``. None if the speed is not
        essentially monotonic.

    Raises
    ------
    ValueError
        If ``tol`` is not a finite number >= 0.
    """
    tol = check_tol(tol)
    v = np.asarray(rpm, dtype=float)
    if v.ndim != 1 or v.size < 2 or not np.all(np.isfinite(v)):
        return None
    direction = np.sign(v[-1] - v[0])
    span = float(v.max() - v.min())
    if direction == 0 or not span > 0:
        return None
    if direction > 0:
        env = np.maximum.accumulate(v)
        reversal = float(np.max(env - v))
    else:
        env = np.minimum.accumulate(v)
        reversal = float(np.max(v - env))
    if reversal > tol * span * (1.0 + 1e-12):
        return None
    return env


def essentially_monotonic(rpm, tol=MONOTONIC_TOL):
    """True if the speed is a run-up or a coast-down up to reversals of ``tol`` of its range.

    See :func:`speed_envelope` for the rule.
    """
    return speed_envelope(rpm, tol) is not None


def choose_x(t, rpm, x="auto", *, tol=MONOTONIC_TOL, check=True):
    """Choose what the frames of a map or slice are plotted against.

    Parameters
    ----------
    t : array_like, shape (n_frames,)
        Frame times in s.
    rpm : array_like, shape (n_frames,)
        Frame speed in rpm.
    x : {"auto", "rpm", "time"}, default "auto"
        ``"auto"`` gives rpm when the speed is essentially monotonic (see
        :func:`speed_envelope`: a run-up or a coast-down whose reversals are at
        most ``tol`` of the speed range) and time otherwise. A figure of frames
        against a speed that goes up and down would fold them onto each other.
    tol : float, default 0.02
        Largest reversal for ``"auto"`` and, if ``check``, for ``"rpm"``.
    check : bool, default True
        Whether ``x="rpm"`` requires an essentially monotonic speed. Images
        need it (their cells must not overlap); line plots pass False, because
        a line may double back and then shows the data as they are.

    Returns
    -------
    kind, values, label : str, ndarray, str
        ``("rpm", rpm, "Speed (rpm)")`` or ``("time", t, "Time (s)")``; the
        values are the true frame values (not the envelope).

    Raises
    ------
    ValueError
        For an invalid ``x`` or ``tol``, or ``x="rpm"`` with ``check`` and a
        speed that is not essentially monotonic (or not finite, or with fewer
        than two frames); the message suggests ``rota.maps.bin_map`` (or
        ``rpm_bins=`` of the map functions) and ``x="time"``.
    """
    if x not in ("auto", "rpm", "time"):
        raise ValueError(f"x must be 'auto', 'rpm' or 'time', got {x!r}")
    tol = check_tol(tol)
    t = np.asarray(t, dtype=float)
    rpm = np.asarray(rpm, dtype=float)
    if x == "rpm" and check and not essentially_monotonic(rpm, tol):
        raise ValueError(
            f"x='rpm' needs an essentially monotonic speed (a run-up or a coast-down whose reversals are "
            f"at most {100.0 * tol:g} % of the speed range), but this speed is not (or has fewer than two "
            "finite frames); bin the frames along speed with rota.maps.bin_map (rpm_bins=... in the map "
            "functions) or use x='time'"
        )
    kind = "rpm" if x == "rpm" or (x == "auto" and essentially_monotonic(rpm, tol)) else "time"
    return kind, (rpm if kind == "rpm" else t), X_LABELS[kind]


def edges_from_centres(c):
    """Cell edges for ``pcolormesh(shading="flat")`` from the cell centres.

    The inner edges are the midpoints between neighbouring centres and the end
    edges extrapolate by half the first or last step, so the cells tile without
    gaps and each centre lies inside its cell. Works for increasing, decreasing
    and unevenly spaced centres; a single centre gets a cell of width 1.

    Parameters
    ----------
    c : array_like, shape (n,)
        Cell centres (finite, n >= 1).

    Returns
    -------
    ndarray, shape (n + 1,)
        Edges in the order of ``c``.

    Raises
    ------
    ValueError
        If ``c`` is not a non-empty, finite 1-D array.
    """
    c = np.asarray(c, dtype=float)
    if c.ndim != 1 or c.size == 0 or not np.all(np.isfinite(c)):
        raise ValueError("cell centres must be a non-empty, finite 1-D array")
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate(([c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]))
