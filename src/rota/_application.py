"""Application-layer helpers: fixed-frequency slices, overall levels, metrics versus speed, order attribution.

Units: frequency in Hz, order in cycles per revolution, speed in rpm (or any
monotonic control quantity such as vehicle speed in km/h for
:func:`metric_vs_rpm`), time in s. Levels are in dB. Amplitude conventions
are stated per function.
"""

from dataclasses import dataclass

import numpy as np

from ._utils import as_2d, broadcast_rpm

__all__ = [
    "FrequencyOrder",
    "FrequencySlice",
    "MetricVsRpm",
    "OrderShare",
    "OverallLevel",
    "frequency_slice",
    "frequency_to_order",
    "metric_vs_rpm",
    "order_share",
    "overall_level",
]

_WEIGHTINGS = ("A", "C")
_SEGMENTS = ("auto", "runup", "rundown", "all")


# --------------------------------------------------------------------------
# frequency_slice
# --------------------------------------------------------------------------


@dataclass
class FrequencySlice:
    """Level of one fixed frequency versus speed, one value per map frame.

    Attributes
    ----------
    freq : float
        Frequency of the slice in Hz.
    t : ndarray, shape (n_frames,)
        Frame centre times in s.
    rpm : ndarray, shape (n_frames,)
        Mean speed over each frame in rpm.
    amplitude : ndarray, shape (n_frames,) or (n_frames, ch)
        Band amplitude in the signal unit and in the convention given by
        ``amplitude_convention``. NaN where the band leaves the map's axis or
        the speed is not positive (order maps).
    amplitude_convention : str
        ``"peak"`` or ``"rms"``, copied from the map.
    bandwidth : ndarray, shape (n_frames,)
        Total band width used in Hz, excluding the allowance for the motion of
        the line during the frame (order maps).
    domain : str
        ``"order"`` or ``"frequency"``, domain of the map the slice came from.
    """

    freq: float
    t: np.ndarray
    rpm: np.ndarray
    amplitude: np.ndarray
    amplitude_convention: str
    bandwidth: np.ndarray
    domain: str


def frequency_slice(m, freq, *, bandwidth=None):
    """Level of a fixed frequency versus speed, read from an RPM map.

    A structural resonance or a fixed tone stays at the same frequency while
    the speed changes, so its level versus speed is the slice of the map along
    a horizontal line. For a frequency map the band is centred on ``freq`` with
    total width ``bandwidth``. For an order map each frame's frequency is
    converted to order with the frame's speed, ``order = freq / (rpm / 60)``;
    the line moves in order during the frame (from ``freq * 60 / rpm_max`` to
    ``freq * 60 / rpm_min``), so the band is centred on the middle of that
    range and widened by it, so that a line that sweeps through the frame
    stays inside.

    Power is summed over the bins of the band, as in
    :func:`rota.slices.order_slice`:
    ``amplitude = sqrt(sum(|values|**2) / enbw_bins)``. This reading does not
    depend on where the line falls between bins (no scalloping loss) and is
    also right for a line smeared over bins. Broadband content inside the band
    adds to the result. Frames whose band extends beyond the lowest or highest
    bin of the axis, or whose speed is not positive in an order map, give NaN.

    Parameters
    ----------
    m : rota.maps.Map
        Frequency or order map; ``values`` of shape ``(n_bins, n_frames)`` or
        ``(n_bins, n_frames, ch)``.
    freq : float
        Frequency to extract in Hz (> 0).
    bandwidth : float, optional
        Total band width in Hz. Default: ``4 * m.resolution`` (a frequency
        map) or four order-bin spacings at each frame's speed,
        ``4 * m.resolution * rpm / 60`` Hz (an order map).

    Returns
    -------
    FrequencySlice
        ``amplitude`` is in the amplitude convention of the map (peak or RMS)
        and in the unit of the analysed signal.

    Notes
    -----
    In an order map the line is resolved only if its order is below
    ``max_order``; at low speed the order of a fixed frequency is high and
    the frame is NaN. Use a frequency map for resonances at low speed.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 9 and 12. Wiley.
    Harris, F. J. (1978). On the use of windows for harmonic analysis with
    the discrete Fourier transform. Proceedings of the IEEE, 66(1), 51-83.
    """
    if m.domain not in ("order", "frequency"):
        raise ValueError(f"unknown map domain {m.domain!r}")
    freq = float(freq)
    if not (np.isfinite(freq) and freq > 0):
        raise ValueError("freq must be positive")
    res = m.resolution
    if bandwidth is not None:
        bandwidth = float(bandwidth)
        if not (np.isfinite(bandwidth) and bandwidth > 0):
            raise ValueError("bandwidth must be positive")

    V = np.asarray(m.values)
    if V.ndim not in (2, 3):
        raise ValueError("map values must have shape (n_bins, n_frames[, ch])")
    one_channel = V.ndim == 2
    if one_channel:
        V = V[..., None]
    axis = np.asarray(m.axis, dtype=float)
    n_frames = V.shape[1]
    rpm = np.asarray(m.rpm, dtype=float)
    rpm_lo = np.asarray(m.rpm_min, dtype=float)
    rpm_hi = np.asarray(m.rpm_max, dtype=float)

    bad = np.zeros(n_frames, dtype=bool)
    if m.domain == "frequency":
        centre = np.full(n_frames, freq)
        bw_hz = np.full(n_frames, 4.0 * res if bandwidth is None else bandwidth)
        half = 0.5 * bw_hz
    else:
        bad = ~((rpm_lo > 0) & np.isfinite(rpm_lo) & np.isfinite(rpm_hi) & (rpm > 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            o_lo = freq * 60.0 / rpm_hi
            o_hi = freq * 60.0 / rpm_lo
            centre = 0.5 * (o_lo + o_hi)
            smear = o_hi - o_lo
            if bandwidth is None:
                bw_ord = np.full(n_frames, 4.0 * res)
                bw_hz = bw_ord * rpm / 60.0
            else:
                bw_hz = np.full(n_frames, bandwidth)
                bw_ord = bw_hz * 60.0 / rpm
        half = 0.5 * bw_ord + 0.5 * smear
        centre = np.where(bad, 0.0, centre)
        half = np.where(bad, 0.0, half)

    tol = 1e-6 * res
    dist = np.abs(axis[:, None] - centre[None, :])
    nearest = np.argmin(dist, axis=0)
    frames = np.arange(n_frames)
    mask = dist <= half[None, :] + tol
    mask[nearest, frames] = True
    outside = (centre + half > axis[-1] + tol) | (centre - half < axis[0] - tol) | bad

    band = np.einsum("bf,bfc->fc", mask.astype(float), np.abs(V) ** 2)
    amp = np.sqrt(band / m.enbw_bins)
    amp[outside] = np.nan
    if one_channel:
        amp = amp[:, 0]
    return FrequencySlice(
        freq=freq,
        t=np.array(m.t, dtype=float),
        rpm=np.array(m.rpm, dtype=float),
        amplitude=amp,
        amplitude_convention=m.amplitude,
        bandwidth=np.where(bad, np.nan, bw_hz),
        domain=m.domain,
    )


# --------------------------------------------------------------------------
# overall_level
# --------------------------------------------------------------------------


@dataclass
class OverallLevel:
    """Overall level per frame, optionally binned along speed.

    Attributes
    ----------
    t : ndarray, shape (n_frames,)
        Frame centre times in s.
    rpm : ndarray, shape (n_frames,)
        Mean speed over each frame in rpm.
    level : ndarray, shape (n_frames,) or (n_frames, ch)
        RMS level in dB re ``ref`` (``-inf`` for a silent frame).
    ref : float
        Reference value in the signal unit (20e-6 for sound pressure in Pa).
    weighting : str or None
        ``"A"``, ``"C"`` or None.
    frame : float
        Frame length in s (frames do not overlap).
    edges : ndarray or None
        Speed bin edges in rpm (``rpm_bins`` given).
    centres : ndarray or None, shape (n_bins,)
        Speed bin centres in rpm.
    binned_level : ndarray or None, shape (n_bins,) or (n_bins, ch)
        Energy mean of the frame levels in each bin, in dB re ``ref``
        (``10 log10(mean(10**(level / 10)))``); NaN in empty bins.
    counts : ndarray or None, shape (n_bins,)
        Number of frames in each bin.
    """

    t: np.ndarray
    rpm: np.ndarray
    level: np.ndarray
    ref: float
    weighting: object
    frame: float
    edges: object = None
    centres: object = None
    binned_level: object = None
    counts: object = None


def _bin_edges(values, bins, what):
    """Bin edges from an edge array or a bin width (edges on multiples of the width)."""
    if np.ndim(bins) == 0:
        w = float(bins)
        if not (np.isfinite(w) and w > 0):
            raise ValueError(f"{what} width must be positive")
        lo = np.floor(np.min(values) / w) * w
        hi = np.ceil(np.max(values) / w) * w
        if hi <= lo:
            hi = lo + w
        n = round((hi - lo) / w)
        return lo + w * np.arange(n + 1)
    e = np.asarray(bins, dtype=float)
    if (
        e.ndim != 1
        or e.size < 2
        or not np.all(np.isfinite(e))
        or np.any(np.diff(e) <= 0)
    ):
        raise ValueError(
            f"{what} edges must be a strictly increasing 1-D array of at least two values"
        )
    return e


def _assign(values, edges):
    """Bin index of each value (right edge of the last bin inclusive); -1 outside."""
    idx = np.searchsorted(edges, values, side="right") - 1
    idx[values == edges[-1]] = edges.size - 2
    idx[(idx < 0) | (idx > edges.size - 2) | ~np.isfinite(values)] = -1
    return idx


def overall_level(x, fs, rpm, *, frame=0.1, weighting=None, ref=20e-6, rpm_bins=None):
    """Overall RMS level versus time and, optionally, versus speed.

    The signal is cut into consecutive non-overlapping frames of ``frame`` s
    (a trailing incomplete frame is dropped). The level of each frame is
    ``L = 20 log10(x_rms / ref)`` with ``x_rms = sqrt(mean(x**2))`` over the
    frame (RMS definition, so a sine of peak amplitude ``A`` reads
    ``A / sqrt(2)``), in dB re ``ref`` (20 uPa for sound pressure in Pa).

    With ``weighting`` the frequency weighting of IEC 61672-1 (``"A"`` or
    ``"C"``) is applied once to the whole signal, in a single pass, before
    framing (see :func:`rota.tonal.weight`). Binning along speed averages
    energy, not decibels: the level of a bin is
    ``10 log10(mean(10**(L / 10)))`` over the frames whose mean speed falls
    in the bin, which equals the RMS level over all samples of those frames
    (frames have equal length). This is a time-averaged level in the sense of
    IEC 61672-1, evaluated per speed bin.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal, e.g. sound pressure in Pa.
    fs : float
        Sampling rate in Hz.
    rpm : float or array_like, shape (n,)
        Speed in rpm at the signal samples.
    frame : float, default 0.1
        Frame length in s.
    weighting : {None, "A", "C"}, default None
        Frequency weighting; None for the unweighted (linear) level.
    ref : float, default 20e-6
        Reference value in the signal unit.
    rpm_bins : float or array_like, optional
        Speed bin width in rpm (bins on multiples of the width) or an array of
        bin edges in rpm. Frames whose mean speed is outside the edges are
        not binned.

    Returns
    -------
    OverallLevel
        Per-frame ``level`` (RMS, dB re ``ref``) and, if ``rpm_bins`` is given,
        the binned energy-mean level and the frame counts.

    Raises
    ------
    ValueError
        For invalid arguments, or if the signal is shorter than one frame.

    References
    ----------
    IEC 61672-1:2013. Electroacoustics - Sound level meters - Part 1:
    Specifications.
    ISO 1683:2015. Acoustics - Preferred reference values for acoustical and
    vibratory levels (reference sound pressure 20 uPa).
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    """
    fs = float(fs)
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a positive sampling rate in Hz")
    frame = float(frame)
    ref = float(ref)
    if not (np.isfinite(frame) and frame > 0):
        raise ValueError("frame must be positive")
    if not (np.isfinite(ref) and ref > 0):
        raise ValueError("ref must be positive")
    if weighting is not None and weighting not in _WEIGHTINGS:
        raise ValueError(
            f"weighting must be None or one of {_WEIGHTINGS}, got {weighting!r}"
        )
    X, was_1d = as_2d(x)
    n = X.shape[0]
    r = broadcast_rpm(rpm, n)
    nf = round(frame * fs)
    if nf < 1:
        raise ValueError("frame is shorter than one sample")
    n_frames = n // nf
    if n_frames < 1:
        raise ValueError("signal is shorter than one frame")

    if weighting is not None:
        from .tonal import weight  # lazy: optional module, imported only when used

        X = np.stack(
            [
                np.asarray(weight(X[:, c], fs, weighting), float)
                for c in range(X.shape[1])
            ],
            axis=1,
        )

    used = n_frames * nf
    ms = np.mean(X[:used].reshape(n_frames, nf, -1) ** 2, axis=1)  # (frames, ch)
    with np.errstate(divide="ignore"):
        level = 10.0 * np.log10(ms / ref**2)
    t = (np.arange(n_frames) + 0.5) * nf / fs
    rpm_frame = r[:used].reshape(n_frames, nf).mean(axis=1)

    out = OverallLevel(
        t=t,
        rpm=rpm_frame,
        level=level[:, 0] if was_1d else level,
        ref=ref,
        weighting=weighting,
        frame=nf / fs,
    )
    if rpm_bins is not None:
        edges = _bin_edges(rpm_frame, rpm_bins, "rpm_bins")
        idx = _assign(rpm_frame, edges)
        nb = edges.size - 1
        counts = np.zeros(nb, dtype=int)
        binned = np.full((nb, level.shape[1]), np.nan)
        for b in range(nb):
            sel = idx == b
            counts[b] = int(sel.sum())
            if counts[b]:
                with np.errstate(divide="ignore"):
                    binned[b] = 10.0 * np.log10(np.mean(ms[sel], axis=0) / ref**2)
        out.edges = edges
        out.centres = 0.5 * (edges[:-1] + edges[1:])
        out.binned_level = binned[:, 0] if was_1d else binned
        out.counts = counts
    return out


# --------------------------------------------------------------------------
# metric_vs_rpm
# --------------------------------------------------------------------------


@dataclass
class MetricVsRpm:
    """A time-varying quantity aggregated along speed.

    Attributes
    ----------
    centres : ndarray, shape (n_bins,)
        Bin centres in the unit of the control (rpm or km/h).
    edges : ndarray, shape (n_bins + 1,)
        Bin edges.
    value : ndarray, shape (n_bins,)
        Aggregated metric per bin; NaN where ``count < min_count``.
    count : ndarray of int, shape (n_bins,)
        Number of metric samples in each bin (reported also where the value
        is withheld).
    std : ndarray, shape (n_bins,)
        Sample standard deviation (``ddof=1``) of the metric values in each
        bin, in the unit of the metric (dB for a level, also for
        ``aggregate="power"``); NaN where ``count < max(min_count, 2)``.
    iqr : ndarray, shape (n_bins,)
        Interquartile range (75th minus 25th percentile) of the metric values
        in each bin; NaN where ``count < min_count``.
    spread : ndarray, shape (n_bins,)
        ``iqr`` for ``aggregate="median"`` or a percentile, ``std`` otherwise.
    spread_kind : str
        ``"iqr"`` or ``"std"``.
    valid : ndarray of bool, shape (n_bins,)
        ``count >= min_count``.
    used : ndarray of bool, shape (n_metric,)
        True for the metric samples that entered a bin.
    control : ndarray, shape (n_metric,)
        Control (speed) interpolated onto the metric's time axis (NaN outside
        the time range of the control).
    segment : str
        Segment actually used: ``"runup"``, ``"rundown"`` or ``"all"``.
    t_segment : (float, float)
        Start and end time in s of the selected segment (before the removal
        of non-monotonic samples).
    aggregate : str
        Aggregation rule, ``"mean"``, ``"power"``, ``"median"`` or
        ``"percentile"``.
    q : float or None
        Percentile when ``aggregate="percentile"``.
    skip_start : float
        Start-up time skipped in s.
    min_count : int
        Minimum samples per bin.
    """

    centres: np.ndarray
    edges: np.ndarray
    value: np.ndarray
    count: np.ndarray
    std: np.ndarray
    iqr: np.ndarray
    spread: np.ndarray
    spread_kind: str
    valid: np.ndarray
    used: np.ndarray
    control: np.ndarray
    segment: str
    t_segment: tuple
    aggregate: str
    q: object
    skip_start: float
    min_count: int


def _parse_aggregate(aggregate):
    if isinstance(aggregate, str):
        if aggregate in ("mean", "power", "median"):
            return aggregate, None
    elif (
        isinstance(aggregate, (tuple, list))
        and len(aggregate) == 2
        and aggregate[0] == "percentile"
    ):
        q = float(aggregate[1])
        if not 0.0 <= q <= 100.0:
            raise ValueError("percentile q must be in [0, 100]")
        return "percentile", q
    raise ValueError(
        f"aggregate must be 'mean', 'power', 'median' or ('percentile', q), got {aggregate!r}"
    )


def _runup_span(c):
    """(i0, i1) of the run-up: global maximum, preceded by the lowest value before it."""
    i1 = int(np.argmax(c))
    i0 = int(np.argmin(c[: i1 + 1]))
    return i0, i1


def _rundown_span(c):
    """(i0, i1) of the run-down: global maximum, followed by the lowest value after it."""
    i0 = int(np.argmax(c))
    i1 = i0 + int(np.argmin(c[i0:]))
    return i0, i1


def metric_vs_rpm(
    metric,
    t_metric,
    rpm,
    t_rpm,
    *,
    segment="auto",
    bin_width=100.0,
    bins=None,
    min_count=3,
    skip_start=0.0,
    aggregate="mean",
    tol=None,
):
    """Aggregate a time-varying quantity along speed.

    Intended for quantities computed by another tool on its own time grid,
    for instance a loudness, tonality, roughness or fluctuation-strength
    track from a psychoacoustic toolbox, or a sound level. The control (rpm,
    or vehicle speed in km/h; the unit is arbitrary but must be the same for
    ``bin_width`` / ``bins``) is interpolated linearly onto the metric's time
    axis, the start-up transient of the metric is skipped, one monotonic
    segment is selected, and the samples are aggregated in speed bins.

    Order of operations.

    1. Metric samples that are non-finite, or lie outside the time range of
       the control (no extrapolation), or earlier than
       ``t_metric[0] + skip_start``, are discarded.
    2. ``segment`` selects the part of the remaining samples to use.
       ``"runup"``: from the lowest control value before the global maximum
       up to the global maximum; ``"rundown"``: from the global maximum to the
       lowest control value after it; ``"auto"``: whichever of the two is
       longer in time (ties go to the run-up); ``"all"``: every remaining
       sample, whatever the direction of the speed.
    3. For ``"runup"`` / ``"rundown"`` (hence also ``"auto"``), samples where
       the control has moved back against the direction of the sweep by more
       than ``tol`` relative to its running extreme in the segment are
       dropped (see the gear-shift caveat). What remains is monotonic up to
       ``tol``.
    4. The samples are assigned to bins and aggregated. Bins with fewer than
       ``min_count`` samples return NaN but their ``count`` is reported.

    Gear-shift caveat. A shift makes the speed non-monotonic and changes the
    relation between speed and excitation (tooth-pass and order frequencies
    jump), so a metric binned along rpm is meaningful only for one fixed gear
    or one monotonic sweep. The monotonic filter of step 3 drops the dip of a
    shift and the re-climb back to the previous maximum, but it cannot tell
    that the samples after the shift belong to another gear. For a
    gear-by-gear analysis, crop the inputs to one gear (or use
    ``segment="all"`` on a cropped signal) and inspect ``count``.

    Start-up transient. Most psychoacoustic models have filters or
    integrators whose output is not valid at the start of a signal. Typical
    values for ``skip_start`` are about 0.6 s for loudness and tonality,
    about 1.5 s for roughness and 4 to 5 s for fluctuation strength; check
    the documentation of the tool that produced the metric.

    Parameters
    ----------
    metric : array_like, shape (n_metric,)
        Metric values (any unit; for a level, dB).
    t_metric : array_like, shape (n_metric,)
        Time of each metric value in s, strictly increasing.
    rpm : array_like, shape (n_rpm,)
        Control (speed in rpm, or e.g. km/h) at ``t_rpm``. Non-finite values
        are ignored.
    t_rpm : array_like, shape (n_rpm,)
        Times of the control samples in s, strictly increasing.
    segment : {"auto", "runup", "rundown", "all"}, default "auto"
        Segment selection (see above).
    bin_width : float, default 100.0
        Bin width in the unit of the control; edges on multiples of the
        width. Ignored if ``bins`` is given.
    bins : array_like, optional
        Explicit bin edges, strictly increasing. The last bin includes its
        right edge.
    min_count : int, default 3
        Minimum number of samples for a bin to return a value.
    skip_start : float, default 0.0
        Time in s, counted from ``t_metric[0]``, to discard.
    aggregate : {"mean", "power", "median"} or ("percentile", q), default "mean"
        ``"mean"``: arithmetic mean of the values. ``"power"``: energy mean of
        levels in dB, ``10 log10(mean(10**(L / 10)))``; use it for sound
        levels, never for metrics that are not levels. ``"median"``: median.
        ``("percentile", q)``: the q-th percentile (0 to 100, linear
        interpolation) of the values in the bin, e.g. ``("percentile", 95)``
        for the level exceeded in 5 % of the samples of a bin.
    tol : float, optional
        Tolerance of the monotonic filter in the unit of the control.
        Default: half the bin width (or half the smallest bin of ``bins``).
        Use ``float("inf")`` to switch the filter off.

    Returns
    -------
    MetricVsRpm
        Bin centres, aggregated values, counts, spread (standard deviation or
        interquartile range) and the mask of the samples used.

    Raises
    ------
    ValueError
        For invalid arguments, or if no sample remains after the exclusions.

    The metric is binned as given (any unit); no amplitude convention applies except for ``aggregate="power"``, which treats the values as levels in dB of power (RMS) quantities.

    References
    ----------
    Fastl, H., & Zwicker, E. (2007). Psychoacoustics: Facts and Models, 3rd
    ed. Springer (statistical percentile values of time-varying loudness).
    ECMA-418-2:2022. Psychoacoustic metrics for ITT equipment - Part 2
    (models based on human perception).
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12 (run-ups and
    speed bins). Wiley.
    """
    y = np.asarray(metric, dtype=float)
    tm = np.asarray(t_metric, dtype=float)
    c_in = np.asarray(rpm, dtype=float)
    tr = np.asarray(t_rpm, dtype=float)
    if y.ndim != 1 or tm.shape != y.shape:
        raise ValueError("metric and t_metric must be 1-D arrays of the same length")
    if c_in.ndim != 1 or tr.shape != c_in.shape:
        raise ValueError("rpm and t_rpm must be 1-D arrays of the same length")
    if np.any(~np.isfinite(tm)) or np.any(np.diff(tm) <= 0):
        raise ValueError("t_metric must be finite and strictly increasing")
    if np.any(~np.isfinite(tr)) or np.any(np.diff(tr) <= 0):
        raise ValueError("t_rpm must be finite and strictly increasing")
    if segment not in _SEGMENTS:
        raise ValueError(f"segment must be one of {_SEGMENTS}, got {segment!r}")
    agg, q = _parse_aggregate(aggregate)
    min_count = int(min_count)
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    skip_start = float(skip_start)
    if not (np.isfinite(skip_start) and skip_start >= 0):
        raise ValueError("skip_start must be non-negative")
    good = np.isfinite(c_in)
    if good.sum() < 2:
        raise ValueError("rpm needs at least two finite samples")
    tr, c_in = tr[good], c_in[good]

    control = np.interp(tm, tr, c_in)
    control[(tm < tr[0]) | (tm > tr[-1])] = np.nan
    cand = np.isfinite(y) & np.isfinite(control) & (tm >= tm[0] + skip_start)
    idx = np.flatnonzero(cand)
    if idx.size == 0:
        raise ValueError(
            "no metric sample remains (check skip_start and the time ranges)"
        )

    if bins is None:
        bw = float(bin_width)
        if not (np.isfinite(bw) and bw > 0):
            raise ValueError("bin_width must be positive")
        half_tol = 0.5 * bw
        edges = None
    else:
        edges = _bin_edges(None, bins, "bins")
        half_tol = 0.5 * float(np.min(np.diff(edges)))
    tol = half_tol if tol is None else float(tol)
    if np.isnan(tol) or tol < 0:
        raise ValueError("tol must be non-negative")

    c = control[idx]
    chosen = segment
    if segment == "all":
        keep = np.ones(idx.size, dtype=bool)
        t_seg = (float(tm[idx[0]]), float(tm[idx[-1]]))
    else:
        spans = {}
        if segment in ("auto", "runup"):
            spans["runup"] = _runup_span(c)
        if segment in ("auto", "rundown"):
            spans["rundown"] = _rundown_span(c)
        dur = {k: tm[idx[i1]] - tm[idx[i0]] for k, (i0, i1) in spans.items()}
        chosen = max(dur, key=lambda k: (dur[k], k == "runup"))
        i0, i1 = spans[chosen]
        if i1 <= i0:
            raise ValueError(f"no {chosen} segment found in the control")
        keep = np.zeros(idx.size, dtype=bool)
        keep[i0 : i1 + 1] = True
        seg = c[i0 : i1 + 1]
        if chosen == "runup":
            ok = seg >= np.maximum.accumulate(seg) - tol
        else:
            ok = seg <= np.minimum.accumulate(seg) + tol
        keep[i0 : i1 + 1] = ok
        t_seg = (float(tm[idx[i0]]), float(tm[idx[i1]]))

    sel = idx[keep]
    cs = control[sel]
    ys = y[sel]
    if edges is None:
        edges = _bin_edges(cs, bin_width, "bin_width")
    ib = _assign(cs, edges)
    inside = ib >= 0
    sel, cs, ys, ib = sel[inside], cs[inside], ys[inside], ib[inside]

    nb = edges.size - 1
    count = np.zeros(nb, dtype=int)
    value = np.full(nb, np.nan)
    std = np.full(nb, np.nan)
    iqr = np.full(nb, np.nan)
    for b in range(nb):
        v = ys[ib == b]
        count[b] = v.size
        if v.size < min_count:
            continue
        if agg == "mean":
            value[b] = v.mean()
        elif agg == "power":
            value[b] = 10.0 * np.log10(np.mean(10.0 ** (v / 10.0)))
        elif agg == "median":
            value[b] = np.median(v)
        else:
            value[b] = np.percentile(v, q)
        if v.size >= 2:
            std[b] = v.std(ddof=1)
        iqr[b] = np.subtract(*np.percentile(v, [75, 25]))
    used = np.zeros(y.size, dtype=bool)
    used[sel] = True
    spread_kind = "iqr" if agg in ("median", "percentile") else "std"
    return MetricVsRpm(
        centres=0.5 * (edges[:-1] + edges[1:]),
        edges=edges,
        value=value,
        count=count,
        std=std,
        iqr=iqr,
        spread=iqr if spread_kind == "iqr" else std,
        spread_kind=spread_kind,
        valid=count >= min_count,
        used=used,
        control=control,
        segment=chosen,
        t_segment=t_seg,
        aggregate=agg,
        q=q,
        skip_start=skip_start,
        min_count=min_count,
    )


# --------------------------------------------------------------------------
# frequency_to_order, order_share
# --------------------------------------------------------------------------


@dataclass
class FrequencyOrder:
    """Order attribution of a frequency track.

    Attributes
    ----------
    order : ndarray
        ``freq / (rpm / 60)`` in cycles per revolution; NaN where the
        frequency is not finite or the speed is not positive.
    nearest : ndarray
        Nearest candidate order (relative distance); NaN where ``order`` is.
    deviation : ndarray
        ``(order - nearest) / nearest``, relative.
    matched : ndarray of bool
        ``abs(deviation) <= tolerance``.
    tolerance : float
        Relative tolerance used.
    """

    order: np.ndarray
    nearest: np.ndarray
    deviation: np.ndarray
    matched: np.ndarray
    tolerance: float


def _check_tolerance(tolerance):
    tolerance = float(tolerance)
    if not (np.isfinite(tolerance) and 0 <= tolerance < 1):
        raise ValueError("tolerance must be a relative value in [0, 1)")
    return tolerance


def _orders_of(freq, rpm):
    f = np.asarray(freq, dtype=float)
    r = np.asarray(rpm, dtype=float)
    try:
        f, r = np.broadcast_arrays(f, r)
    except ValueError:
        raise ValueError(
            "freq and rpm must have the same shape (or rpm be a scalar)"
        ) from None
    ok = np.isfinite(f) & np.isfinite(r) & (r > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        o = np.where(ok, f / np.where(ok, r, 1.0) * 60.0, np.nan)
    return o, ok


def frequency_to_order(freq, rpm, *, tolerance=0.04, orders=None):
    """Convert a frequency track to order and attribute it to candidate orders.

    The order of a frequency ``f`` (Hz) at shaft speed ``rpm`` is
    ``f / (rpm / 60)``. It is attributed to the candidate order ``o`` closest
    in relative terms, ``abs(order - o) / o``, and counts as a match if that
    relative deviation is at most ``tolerance``. Typical use: the dominant
    tone frequency from a tonality tool, to decide which engine or motor order
    carries it.

    Parameters
    ----------
    freq : array_like
        Frequency track in Hz; NaN where there is no tone.
    rpm : float or array_like
        Shaft speed in rpm, broadcastable to ``freq`` (same time axis).
    tolerance : float, default 0.04
        Relative tolerance in [0, 1). Choose it from the accuracy of the
        frequency estimate and of the speed. Where the tolerance windows of
        two candidates overlap, the nearest candidate wins.
    orders : array_like, optional
        Candidate orders (positive, cycles per revolution; may be
        fractional, e.g. 0.5 or 1.5). Default: the positive integers up to
        just above the largest order in the track.

    Returns
    -------
    FrequencyOrder
        Order, nearest candidate, relative deviation and match flag, each
        with the broadcast shape of ``freq`` and ``rpm``.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    """
    tolerance = _check_tolerance(tolerance)
    o, ok = _orders_of(freq, rpm)
    if orders is None:
        top = np.nanmax(o) if ok.any() else 1.0
        cand = np.arange(1.0, np.floor(top) + 2.0)
    else:
        cand = np.unique(np.asarray(orders, dtype=float).ravel())
        if cand.size == 0 or not np.all(np.isfinite(cand)) or np.any(cand <= 0):
            raise ValueError("orders must be a non-empty set of positive finite values")
    nearest = np.full(o.shape, np.nan)
    if ok.any():
        ov = o[ok]
        j = np.clip(np.searchsorted(cand, ov), 1, max(cand.size - 1, 1))
        lo = cand[np.maximum(j - 1, 0)]
        hi = cand[np.minimum(j, cand.size - 1)]
        pick_hi = np.abs(ov - hi) / hi < np.abs(ov - lo) / lo
        nearest[ok] = np.where(pick_hi, hi, lo)
    with np.errstate(invalid="ignore"):
        dev = (o - nearest) / nearest
        matched = ok & (np.abs(dev) <= tolerance)
    return FrequencyOrder(
        order=o, nearest=nearest, deviation=dev, matched=matched, tolerance=tolerance
    )


@dataclass
class OrderShare:
    """Share of a frequency track on an order and its harmonics.

    Attributes
    ----------
    share : float
        Fraction of the (weighted) valid samples on the base order or any of
        its harmonics; NaN if the total weight is zero.
    share_per_harmonic : ndarray, shape (harmonics,)
        Fraction on each harmonic ``k = 1..harmonics``.
    counts : ndarray of int, shape (harmonics,)
        Number of samples on each harmonic (unweighted).
    weight_per_harmonic : ndarray, shape (harmonics,)
        Total weight on each harmonic.
    harmonics : ndarray of int, shape (harmonics,)
        The multipliers ``k``.
    n_valid, n_total : int
        Samples with a finite frequency, a positive speed and a usable
        weight, and all samples.
    base_order : float
    tolerance : float
    """

    share: float
    share_per_harmonic: np.ndarray
    counts: np.ndarray
    weight_per_harmonic: np.ndarray
    harmonics: np.ndarray
    n_valid: int
    n_total: int
    base_order: float
    tolerance: float


def order_share(freq, rpm, base_order, *, harmonics=4, tolerance=0.04, weights=None):
    """Fraction of a frequency track that lies on an order and its harmonics.

    A sample is on harmonic ``k`` (``k = 1..harmonics``) if its order
    ``freq / (rpm / 60)`` is within a relative ``tolerance`` of
    ``k * base_order``. If windows of neighbouring harmonics overlap (k above
    about ``(1 - tolerance) / (2 * tolerance)``), the sample goes to the
    harmonic with the smaller relative deviation, so no sample is counted
    twice. The share is the (weighted) number of such samples over the
    (weighted) number of valid samples: those with a finite frequency, a
    positive speed and a finite weight. Samples without a detected tone
    (NaN frequency) are therefore not in the denominator; give them a
    finite frequency and zero weight, or count ``n_total`` yourself, if the
    denominator should include them.

    Parameters
    ----------
    freq : array_like
        Frequency track in Hz.
    rpm : float or array_like
        Shaft speed in rpm, broadcastable to ``freq``.
    base_order : float
        Base order (> 0), e.g. 2 for the firing order of a four-cylinder
        four-stroke engine.
    harmonics : int, default 4
        Number of multiples ``k * base_order`` considered (k = 1..harmonics).
    tolerance : float, default 0.04
        Relative tolerance in [0, 1).
    weights : array_like, optional
        Non-negative sample weights, same shape as ``freq``; e.g. a tonality
        value, so that prominent tones count more. Default: equal weights.

    Returns
    -------
    OrderShare
        Total and per-harmonic shares with the sample counts.

    No amplitudes are involved (frequencies in Hz, speed in rpm).

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    """
    tolerance = _check_tolerance(tolerance)
    base_order = float(base_order)
    if not (np.isfinite(base_order) and base_order > 0):
        raise ValueError("base_order must be positive")
    harmonics = int(harmonics)
    if harmonics < 1:
        raise ValueError("harmonics must be at least 1")
    o, ok = _orders_of(freq, rpm)
    n_total = o.size
    if weights is None:
        w = np.ones(o.shape)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != o.shape:
            raise ValueError("weights must have the same shape as freq")
        if np.any(w[np.isfinite(w)] < 0):
            raise ValueError("weights must be non-negative")
        ok = ok & np.isfinite(w)
    ks = np.arange(1, harmonics + 1)
    targets = ks * base_order
    ov, wv = o[ok], w[ok]
    dev = np.abs(ov[:, None] - targets[None, :]) / targets[None, :]  # (n, K)
    kbest = np.argmin(dev, axis=1)
    hit = dev[np.arange(ov.size), kbest] <= tolerance
    counts = np.bincount(kbest[hit], minlength=harmonics).astype(int)
    wk = np.bincount(kbest[hit], weights=wv[hit], minlength=harmonics)
    total = float(wv.sum())
    if total > 0:
        per = wk / total
        share = float(wk.sum() / total)
    else:
        per = np.full(harmonics, np.nan)
        share = float("nan")
    return OrderShare(
        share=share,
        share_per_harmonic=per,
        counts=counts,
        weight_per_harmonic=wk,
        harmonics=ks,
        n_valid=int(ok.sum()),
        n_total=int(n_total),
        base_order=base_order,
        tolerance=tolerance,
    )
