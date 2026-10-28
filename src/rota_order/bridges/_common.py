"""Shared pieces of the bridges: the result container, input checks and binning along rpm.

Nothing here imports a psychoacoustic toolbox.
"""

import dataclasses
import importlib.metadata
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..slices import metric_vs_rpm
from ..tonal import to_calibrated_48k

__all__ = ["TimeMetric"]

FS_MODEL = 48000.0  # rate at which the toolboxes are run, Hz


@dataclass
class TimeMetric:
    """A time-varying psychoacoustic metric computed by an external toolbox.

    The metric is not an amplitude: it is a perceptual quantity in the unit
    stated by ``unit``, computed from the instantaneous sound pressure in Pa
    (calibrated; no peak or RMS convention applies to the result). Bin it along
    rpm with :meth:`vs_rpm`.

    Attributes
    ----------
    name : str
        Metric name, e.g. ``"loudness"``.
    t : ndarray, shape (n_t,)
        Time of each value in s, from sample 0 of the input at t = 0 (the
        toolbox's own time axis, strictly increasing). The last frame of a
        toolbox may lie after the last sample of the input (see
        :meth:`vs_rpm`).
    values : ndarray, shape (n_t,)
        Metric value at each time, in ``unit``.
    unit : str
        ``"sone"``, ``"acum"``, ``"asper"`` (MoSQITo), ``"sone_HMS"``,
        ``"tu_HMS"``, ``"asper_HMS"`` (Sottek hearing model).
    single_value : float
        The toolbox's own single-number value where it provides one (for the
        Sottek hearing model: the time average of the tonality, the power
        average of the loudness, the 90th percentile of the roughness);
        NaN otherwise.
    skip_start : float
        Recommended start-up exclusion in s, counted from ``t[0]``: the
        samples before ``t[0] + skip_start`` are not valid (filter and
        integrator transients). Taken from the examples of rota-order (0.6 s for
        MoSQITo loudness and sharpness, 1.5 s for its roughness; 0.30 s for
        tonality and loudness and 0.31 s for roughness of the Sottek hearing
        model).
    aggregate : str or tuple
        Recommended aggregation of :meth:`vs_rpm`, in the notation of
        :func:`rota_order.slices.metric_vs_rpm` (from the examples of rota-order).
    threshold : float or None
        Frames with ``values <= threshold`` are set to NaN by
        :meth:`vs_rpm` before binning; NaN (or None) means no threshold. Only the
        tonality of the Sottek hearing model (and of SQAT) uses it (0.02 tu_HMS,
        the rule of ECMA-418-2 for its single value).
    extra : dict
        Further outputs of the toolbox (specific spectra, frequency or Bark
        axes, ...), keyed by descriptive names; the keys are documented per
        function.
    source : str
        Toolbox name and version, standard and edition.

    References
    ----------
    LISQM catalogue of psychoacoustic metrics and tools:
    https://zhenxianli.github.io/LISQM/
    """

    name: str
    t: np.ndarray
    values: np.ndarray
    unit: str
    single_value: float = np.nan
    skip_start: float = 0.0
    aggregate: object = "mean"
    threshold: float = np.nan
    extra: dict = field(default_factory=dict)
    source: str = ""

    def vs_rpm(self, rpm, t_rpm, **kwargs):
        """Bin this metric along rpm with :func:`rota_order.slices.metric_vs_rpm`.

        ``bridges.mosqito.loudness(x, fs).vs_rpm(rpm, t_rpm)`` shows the quantity
        and the toolbox in one line; the name follows the ``<quantity>_vs_<axis>``
        family of rota-order (:func:`rota_order.slices.metric_vs_rpm`,
        :func:`rota_order.tonal.band_levels_vs_rpm`, :func:`rota_order.tonal.level_vs_time`).

        The method calls ``metric_vs_rpm(self.values, self.t, rpm, t_rpm, ...)``
        with the metric's recommended ``skip_start`` and ``aggregate``; any keyword
        argument of ``metric_vs_rpm`` given here overrides them. One more keyword,
        ``threshold``, overrides ``self.threshold`` (frames not above it are
        discarded; None or NaN for no threshold). For full control, call
        ``rota_order.slices.metric_vs_rpm(m.values, m.t, rpm, t_rpm, ...)`` yourself
        (``m`` this metric): then none of the recommendations of the metric, the
        threshold or the hold of the speed described here is applied.

        Units. The result has the unit of the metric (``self.unit``) per bin;
        ``rpm`` in rpm (or any monotonic control in a single unit), ``t_rpm`` in s.
        The metric is a perceptual quantity, not an amplitude; no peak or RMS
        convention applies.

        Provenance. The result carries the ``name``, ``unit`` and ``source`` of
        this metric, so that a table or a plot made from it can say what it shows
        and which toolbox computed it (they are empty when
        :func:`rota_order.slices.metric_vs_rpm` is called directly).

        Frames beyond the speed record. ``metric_vs_rpm`` never extrapolates
        speed, but a toolbox may put its last frame after the last sample of the
        signal. The Sottek hearing model computes tonality and loudness in frames
        256 samples apart (5.33 ms at 48 kHz) and puts the last one at the length
        of the signal rounded up to a multiple of 256 samples: one sample after the
        last sample of the signal when its length at 48 kHz is a multiple of 256,
        up to one frame step after it otherwise. Frames that lie after
        ``t_rpm[-1]`` by no more than the sum of the last speed sample interval
        (the last speed sample can lie that far before the end of the signal) and
        the frame step of the metric (the median interval of ``t``) therefore get
        the last speed held (the speed record is extended to the last such frame).
        Frames further beyond the record, frames before ``t_rpm[0]``, and
        frames before the first or after the last finite speed sample (a NaN speed
        at an end of the record is not held; NaN inside it is bridged by
        interpolation) cannot be assigned a speed and are left out. A
        :class:`UserWarning` reports how many frames are left out for each reason;
        it counts only the frames that would otherwise be used, that is the finite
        frames above the threshold and not in the start-up exclusion
        (``skip_start``).

        Parameters
        ----------
        rpm : array_like, shape (n_rpm,)
            Speed in rpm (or km/h) at ``t_rpm``.
        t_rpm : array_like, shape (n_rpm,)
            Times of ``rpm`` in s, strictly increasing, on the time axis of the
            signal passed to the toolbox (sample 0 at t = 0).
        **kwargs
            Passed to :func:`rota_order.slices.metric_vs_rpm` (``segment``, ``bin_width``,
            ``bins``, ``min_count``, ``skip_start``, ``aggregate``, ``tol``), and
            ``threshold`` (unit of the metric; None or NaN for none) in place of
            ``self.threshold``.

        Returns
        -------
        rota_order.slices.MetricVsRpm
            Bin centres, aggregated values (in ``self.unit``), counts and spread,
            with ``name``, ``unit`` and ``source`` taken from this metric.

        Warns
        -----
        UserWarning
            If frames that would otherwise be used cannot be assigned a speed:
            they lie outside the speed record, or where the speed is NaN before
            its first or after its last finite sample. The message gives their
            numbers.

        Raises
        ------
        ValueError
            As :func:`rota_order.slices.metric_vs_rpm`: invalid arguments, or no frame
            remains after the exclusions. If the cause is the threshold (every
            frame after the start-up exclusion is at or below it), the message
            says so and names the threshold and the largest value.

        References
        ----------
        Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
        ISO 532-1:2017; DIN 45692:2009; Daniel, P., & Weber, R. (1997). Acustica
        83, 113-123; ECMA-418-2:2025 (4th ed.): the metrics being binned.
        LISQM catalogue: https://zhenxianli.github.io/LISQM/
        """
        opts = {"skip_start": self.skip_start, "aggregate": self.aggregate}
        threshold = _as_threshold(kwargs.pop("threshold", self.threshold))
        opts.update(kwargs)

        values = np.asarray(self.values, dtype=float)
        t = np.asarray(self.t, dtype=float)
        tr = np.asarray(t_rpm, dtype=float)
        c = np.asarray(rpm, dtype=float)

        # The frames that would enter the binning if each had a speed. None when the time axis or
        # skip_start is invalid: the checks and the warning below are then left out, and
        # metric_vs_rpm raises its own error.
        skip = _as_skip(opts["skip_start"])
        usable = _usable_frames(values, t, skip)
        if np.isfinite(threshold):
            above = values > threshold
            if usable is not None:
                if usable.any() and not np.any(usable & above):
                    scope = f"after the start-up exclusion (skip_start = {skip:g} s) " if skip > 0 else ""
                    limit = f"{threshold:g} {self.unit}".strip()
                    raise ValueError(
                        f"every {self.name} frame {scope}is at or below the threshold of {limit} "
                        f"(the largest of {np.count_nonzero(usable)} frames is {np.max(values[usable]):.4g}), "
                        "so none is left to bin; lower the threshold, or pass threshold=None for no threshold"
                    )
                usable &= above
            values = np.where(above, values, np.nan)

        if usable is not None and _is_speed_record(tr, c):
            tr, c = _hold_last_speed(t, tr, c)
            message = _frames_without_speed(self.name, t, usable, tr, c)
            if message:
                warnings.warn(message, UserWarning, stacklevel=2)
        result = metric_vs_rpm(values, t, c, tr, **opts)
        return dataclasses.replace(result, name=self.name, unit=self.unit, source=self.source)


def _as_threshold(threshold):
    """The threshold as a float; None means no threshold (NaN)."""
    if threshold is None:
        return np.nan
    try:
        return float(threshold)
    except (TypeError, ValueError):
        raise ValueError(
            f"threshold must be a number in the unit of the metric, or None or NaN for none, got {threshold!r}"
        ) from None


def _as_skip(skip_start):
    """``skip_start`` as a float in s, or None if it is not a finite non-negative number."""
    try:
        skip = float(skip_start)
    except (TypeError, ValueError):
        return None
    return skip if np.isfinite(skip) and skip >= 0 else None


def _usable_frames(values, t, skip):
    """Mask of the finite frames from ``t[0] + skip`` on, or None if the arguments are invalid.

    These are the frames that ``metric_vs_rpm`` uses if each of them has a speed (the threshold
    is not applied). Arguments that make ``metric_vs_rpm`` raise give None.
    """
    if skip is None or t.ndim != 1 or values.shape != t.shape or t.size == 0:
        return None
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        return None
    return np.isfinite(values) & (t >= t[0] + skip)


def _is_speed_record(tr, c):
    """True for two or more speed samples at finite, strictly increasing times."""
    return bool(
        tr.ndim == 1 and tr.size >= 2 and tr.shape == c.shape
        and np.all(np.isfinite(tr)) and np.all(np.diff(tr) > 0)
    )


def _hold_last_speed(t, tr, c):
    """Extend the speed record to the last frame that lies shortly after its end, holding the last speed.

    "Shortly" is the sum of the last speed sample interval and the frame step of the metric (the
    median interval of ``t``), with a relative margin of 1e-6 for the rounding of the times: the
    last speed sample can lie one speed interval before the end of the signal, and the last frame
    one frame step after it. A NaN last speed is not held. Returns ``(tr, c)``, with one sample
    appended if there is such a frame.
    """
    if not np.isfinite(c[-1]):
        return tr, c
    reach = (tr[-1] - tr[-2]) + (float(np.median(np.diff(t))) if t.size > 1 else 0.0)
    near = (t > tr[-1]) & (t <= tr[-1] + reach * (1.0 + 1e-6))
    if not np.any(near):
        return tr, c
    return np.append(tr, t[near][-1]), np.append(c, c[-1])


def _frames_without_speed(name, t, usable, tr, c):
    """Warning text for the usable frames that cannot be assigned a speed; empty if there are none.

    ``tr`` and ``c`` are the speed record after :func:`_hold_last_speed`. ``metric_vs_rpm`` drops the
    non-finite speeds, interpolates across them and does not extrapolate: a frame has a speed from
    the first to the last finite sample. A frame before ``tr[0]`` or after ``tr[-1]`` lies outside
    the record; one inside the record but before the first or after the last finite sample lies
    where the speed is NaN.
    """
    finite = np.isfinite(c)
    if np.count_nonzero(finite) < 2:
        return ""  # metric_vs_rpm raises its own error
    first, last = tr[finite][0], tr[finite][-1]
    before = int(np.count_nonzero(usable & (t < tr[0])))
    after = int(np.count_nonzero(usable & (t > tr[-1])))
    nan_head = int(np.count_nonzero(usable & (t >= tr[0]) & (t < first)))
    nan_tail = int(np.count_nonzero(usable & (t > last) & (t <= tr[-1])))
    parts = []
    if before or after:
        parts.append(
            f"{before + after} {name} frame(s) lie outside the speed record "
            f"({before} before {tr[0]:.6g} s, {after} after {tr[-1]:.6g} s) and are left out"
        )
    if nan_head or nan_tail:
        parts.append(
            f"{nan_head + nan_tail} {name} frame(s) lie where the speed is NaN "
            f"({nan_head} before its first finite sample at {first:.6g} s, "
            f"{nan_tail} after its last at {last:.6g} s) and are left out"
        )
    return "; ".join(parts)


def package_version(dist):
    """Version string of an installed distribution, ``"unknown"`` if it cannot be read."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def pressure_48k(x, fs, calibration):
    """Check a mono signal and return ``calibration * x`` in Pa at 48 kHz."""
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(
            f"x must be a mono signal of shape (n,), got shape {a.shape}; pass one channel, "
            "for example x[:, k]"
        )
    if not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a):
        raise ValueError("x must be a real numeric array")
    a = a.astype(float)
    if a.size < 2 or not np.all(np.isfinite(a)):
        raise ValueError("x must contain at least two finite samples")
    fs = float(fs)
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a positive sampling rate in Hz")
    calibration = float(calibration)
    if not (np.isfinite(calibration) and calibration > 0):
        raise ValueError("calibration must be a positive factor from the unit of x to Pa")
    return to_calibrated_48k(a, fs, calibration=calibration, fs_out=FS_MODEL)
