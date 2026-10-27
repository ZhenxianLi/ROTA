"""Shared pieces of the bridges: the result container, input checks and binning along rpm.

Nothing here imports a psychoacoustic toolbox.
"""

import importlib.metadata
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..slices import metric_vs_rpm
from ..tonal import to_calibrated_48k

__all__ = ["TimeMetric", "along_rpm"]

FS_MODEL = 48000.0  # rate at which both toolboxes are run, Hz


@dataclass
class TimeMetric:
    """A time-varying psychoacoustic metric computed by an external toolbox.

    The metric is not an amplitude: it is a perceptual quantity in the unit
    stated by ``unit``, computed from the instantaneous sound pressure in Pa
    (calibrated; no peak or RMS convention applies to the result).

    Attributes
    ----------
    name : str
        Metric name, e.g. ``"loudness"``.
    t : ndarray, shape (n_t,)
        Time of each value in s, from sample 0 of the input at t = 0 (the
        toolbox's own time axis, strictly increasing).
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
        integrator transients). Taken from the examples of ROTA (0.6 s for
        MoSQITo loudness and sharpness, 1.5 s for its roughness; 0.30 s for
        tonality and loudness and 0.31 s for roughness of the Sottek hearing
        model).
    aggregate : str or tuple
        Recommended aggregation of :func:`along_rpm`, in the notation of
        :func:`rota.slices.metric_vs_rpm` (from the examples of ROTA).
    threshold : float
        Frames with ``values <= threshold`` are set to NaN by
        :func:`along_rpm` before binning; NaN means no threshold. Only the
        Sottek tonality uses it (0.02 tu_HMS, the rule of ECMA-418-2 for its
        single value).
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


def along_rpm(metric, rpm, t_rpm, **kwargs):
    """Bin a :class:`TimeMetric` along rpm with :func:`rota.slices.metric_vs_rpm`.

    Calls ``metric_vs_rpm(metric.values, metric.t, rpm, t_rpm, ...)`` with the
    metric's recommended ``skip_start`` and ``aggregate``; any keyword argument
    of ``metric_vs_rpm`` given here overrides them. One more keyword,
    ``threshold``, overrides ``metric.threshold`` (frames not above it are
    discarded).

    Units. The result has the unit of the metric (``metric.unit``) per bin;
    ``rpm`` in rpm (or any monotonic control in a single unit), ``t_rpm`` in s.
    The metric is a perceptual quantity, not an amplitude; no peak or RMS
    convention applies.

    Frames beyond the speed record. The Sottek hearing model puts its last
    frame at the duration of the record, one sample after the last speed
    sample, and ``metric_vs_rpm`` never extrapolates speed. Frames that lie
    after ``t_rpm[-1]`` by no more than one speed sample interval therefore get
    the last speed held (the speed record is extended to the last such frame).
    Frames further beyond the record, and frames before ``t_rpm[0]``, cannot be
    assigned a speed and are left out; a :class:`UserWarning` reports how many.

    Parameters
    ----------
    metric : TimeMetric
        Result of a function of :mod:`rota.bridges.mosqito` or
        :mod:`rota.bridges.shm`.
    rpm : array_like, shape (n_rpm,)
        Speed in rpm (or km/h) at ``t_rpm``.
    t_rpm : array_like, shape (n_rpm,)
        Times of ``rpm`` in s, strictly increasing, on the time axis of the
        signal passed to the toolbox (sample 0 at t = 0).
    **kwargs
        Passed to :func:`rota.slices.metric_vs_rpm` (``segment``, ``bin_width``,
        ``bins``, ``min_count``, ``skip_start``, ``aggregate``, ``tol``).

    Returns
    -------
    rota.slices.MetricVsRpm
        Bin centres, aggregated values, counts and spread.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    ISO 532-1:2017; DIN 45692:2009; Daniel, P., & Weber, R. (1997). Acustica
    83, 113-123; ECMA-418-2:2025 (4th ed.): the metrics being binned.
    LISQM catalogue: https://zhenxianli.github.io/LISQM/
    """
    opts = {"skip_start": metric.skip_start, "aggregate": metric.aggregate}
    threshold = kwargs.pop("threshold", metric.threshold)
    opts.update(kwargs)

    values = np.asarray(metric.values, dtype=float)
    t = np.asarray(metric.t, dtype=float)
    if np.isfinite(threshold):
        values = np.where(values > threshold, values, np.nan)

    tr = np.asarray(t_rpm, dtype=float)
    c = np.asarray(rpm, dtype=float)
    if tr.ndim == 1 and tr.size >= 2 and tr.shape == c.shape and np.all(np.isfinite(tr)):
        t_end, step = tr[-1], tr[-1] - tr[-2]
        near = (t > t_end) & (t <= t_end + step * (1.0 + 1e-6))
        if np.any(near):
            tr = np.append(tr, t[near][-1])
            c = np.append(c, c[-1])
        n_late = int(np.count_nonzero(t > tr[-1]))
        n_early = int(np.count_nonzero(t < tr[0]))
        if n_late or n_early:
            warnings.warn(
                f"{n_late + n_early} {metric.name} frame(s) lie outside the speed record "
                f"({n_early} before {tr[0]:.6g} s, {n_late} after {tr[-1]:.6g} s) and are left out",
                UserWarning,
                stacklevel=2,
            )
    return metric_vs_rpm(values, t, c, tr, **opts)


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
