"""Thin wrappers for time-varying metrics of MoSQITo (optional external toolbox).

MoSQITo (Green Forge Coop, https://github.com/Eomys/MoSQITo, Apache-2.0) is not a
dependency of rota-order; install it yourself with ``pip install mosqito`` (``import mosqito``
also needs ``matplotlib``, which MoSQITo does not declare). It is imported inside the
functions that need it, never at import of this module.

Licence: MoSQITo is distributed under the Apache License 2.0 by its authors. rota-order does
not include, copy, modify or redistribute any of its code; this module calls its public
functions at run time in an installation the user makes. Cite MoSQITo as its authors ask
(CITATION.cff of the MoSQITo repository; see References).

Wrapped (MoSQITo 1.2.1, public functions only):

* :func:`loudness`: ISO 532-1:2017 time-varying loudness (Zwicker), sone, one value per 2 ms.
* :func:`sharpness`: DIN 45692:2009 sharpness, acum, from the specific loudness.
* :func:`roughness`: Daniel and Weber (1997) roughness, asper, one value per 0.1 s
  (200 ms windows); the time axis starts at 0.1 s.

Not wrapped: the ECMA-418-2 functions of MoSQITo (2nd edition only; see
:mod:`rota_order.bridges.sottek_hearing_model` for the 4th edition) and ``tnr_ecma_st`` (rota-order has
its own ECMA-418-1:2024 tone-to-noise ratio in :mod:`rota_order.tonal`).

Input convention for every function: ``x`` is the instantaneous sound pressure in Pa
(``calibration`` multiplies ``x`` into Pa), mono, sampled at ``fs``; it is resampled to
48 kHz with :func:`rota_order.tonal.to_calibrated_48k`. The returned metrics are perceptual
quantities, not amplitudes: no peak or RMS convention applies.

References
----------
ISO 532-1:2017; DIN 45692:2009; Daniel, P., & Weber, R. (1997). Psychoacoustical
roughness: implementation of an optimized model. Acustica 83, 113-123.
LISQM catalogue: https://zhenxianli.github.io/LISQM/
"""

import contextlib
import importlib
import io
import warnings

import numpy as np

from ._common import FS_MODEL, TimeMetric, package_version, pressure_48k

__all__ = ["loudness", "roughness", "sharpness"]

_FIELDS = ("free", "diffuse")
_WEIGHTINGS = ("din", "aures", "bismarck", "fastl")
_SKIP_LOUDNESS = 0.6  # s
_SKIP_ROUGHNESS = 1.5  # s


def _sq_metrics():
    """Import ``mosqito.sq_metrics`` lazily; absolute import, so this module's name does not shadow it."""
    try:
        importlib.import_module("mosqito")  # first the package itself, so a hidden package is seen
        return importlib.import_module("mosqito.sq_metrics")
    except ImportError as err:
        raise ImportError(
            "rota_order.bridges.mosqito needs the optional package MoSQITo: pip install mosqito "
            "(its import also needs matplotlib: pip install matplotlib). rota-order itself does not "
            f"depend on it. Original error: {err}"
        ) from err


def _quiet():
    """MoSQITo prints progress and warns about start-up points; both are silenced."""
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    stack.enter_context(warnings.catch_warnings())
    warnings.simplefilter("ignore")
    return stack


def _source(what):
    return f"MoSQITo {package_version('mosqito')}; {what}"


def loudness(x, fs, *, calibration=1.0, field_type="free"):
    """Time-varying loudness after ISO 532-1:2017 (MoSQITo ``loudness_zwtv``).

    Amplitude and units. ``x`` is the instantaneous sound pressure in Pa (not an
    amplitude spectrum; no peak or RMS convention applies); ``calibration``
    multiplies ``x`` into Pa. The loudness is a perceptual quantity in sone, one
    value per 2 ms. It is not an amplitude either.

    Parameters
    ----------
    x : array_like, shape (n,)
        Mono sound pressure in file units; multichannel input raises ValueError.
    fs : float
        Sampling rate of ``x`` in Hz; resampled to 48 kHz when different.
    calibration : float, default 1.0
        Factor from file units to Pa (1.0: ``x`` is already in Pa).
    field_type : {"free", "diffuse"}, default "free"
        Sound field of ISO 532-1.

    Returns
    -------
    TimeMetric
        ``values`` in sone on ``t`` in s; ``skip_start`` 0.6 s; recommended
        aggregation the 95th percentile (N5). ``single_value`` is NaN (the
        toolbox gives none). ``extra``: ``"specific_loudness"`` (sone/Bark,
        shape (n_bark, n_t)), ``"bark"`` (Bark axis), ``"field_type"``.

    Raises
    ------
    ImportError
        If MoSQITo is not installed.
    ValueError
        For multichannel or non-finite input, or an unknown ``field_type``.

    References
    ----------
    ISO 532-1:2017. Acoustics - Methods for calculating loudness - Part 1:
    Zwicker method.
    MoSQITo: Green Forge Coop (2024). MOSQITO (version 1.2.1) [software]. Zenodo.
    https://doi.org/10.5281/zenodo.11026796. https://github.com/Eomys/MoSQITo
    (Apache-2.0).
    LISQM catalogue: https://zhenxianli.github.io/LISQM/
    """
    if field_type not in _FIELDS:
        raise ValueError(f"field_type must be one of {_FIELDS}, got {field_type!r}")
    p = pressure_48k(x, fs, calibration)
    sq = _sq_metrics()
    with _quiet():
        n, n_spec, bark, t = sq.loudness_zwtv(p, FS_MODEL, field_type=field_type)
    return TimeMetric(
        name="loudness",
        t=np.asarray(t, dtype=float),
        values=np.asarray(n, dtype=float),
        unit="sone",
        skip_start=_SKIP_LOUDNESS,
        aggregate=("percentile", 95),
        extra={
            "specific_loudness": np.asarray(n_spec, dtype=float),
            "bark": np.asarray(bark, dtype=float),
            "field_type": field_type,
        },
        source=_source(f"ISO 532-1:2017 time-varying loudness, {field_type} field"),
    )


_loudness = loudness  # the parameter ``loudness`` of ``sharpness`` shadows the function


def sharpness(x, fs, *, calibration=1.0, field_type="free", weighting="din", loudness=None):
    """Time-varying sharpness after DIN 45692:2009 (MoSQITo ``sharpness_din_from_loudness``).

    Computed from the specific loudness of :func:`loudness`. To avoid computing the
    loudness twice, pass the result of :func:`loudness` as ``loudness`` (then ``x``,
    ``fs``, ``calibration`` and ``field_type`` are not used and ``x`` may be None).

    Amplitude and units. ``x`` is the instantaneous sound pressure in Pa (no peak
    or RMS convention applies); the sharpness is a perceptual quantity in acum,
    not an amplitude.

    Parameters
    ----------
    x : array_like, shape (n,) or None
        Mono sound pressure in file units; multichannel input raises ValueError.
        May be None if ``loudness`` is given.
    fs : float or None
        Sampling rate of ``x`` in Hz.
    calibration : float, default 1.0
        Factor from file units to Pa.
    field_type : {"free", "diffuse"}, default "free"
        Sound field of the loudness (ISO 532-1).
    weighting : {"din", "aures", "bismarck", "fastl"}, default "din"
        Weighting function of the sharpness; only ``"din"`` is the standard
        DIN 45692 method.
    loudness : TimeMetric, optional
        Result of :func:`loudness`; its ``extra["specific_loudness"]`` is used.

    Returns
    -------
    TimeMetric
        ``values`` in acum on the time axis of the loudness; ``skip_start`` 0.6 s
        (MoSQITo warns about the first points; they are silenced here and fall
        within ``skip_start``); aggregation median. ``single_value`` NaN.
        ``extra``: ``"weighting"``, ``"field_type"``.

    Raises
    ------
    ImportError
        If MoSQITo is not installed.
    ValueError
        For invalid input, an unknown ``weighting`` or ``field_type``, or a
        ``loudness`` that is not a loudness result of this module.

    References
    ----------
    DIN 45692:2009. Measurement technique for the simulation of the auditory
    sensation of sharpness.
    ISO 532-1:2017 (the loudness it is computed from).
    MoSQITo: Green Forge Coop (2024). MOSQITO (version 1.2.1) [software]. Zenodo.
    https://doi.org/10.5281/zenodo.11026796. https://github.com/Eomys/MoSQITo
    (Apache-2.0).
    LISQM catalogue: https://zhenxianli.github.io/LISQM/
    """
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"weighting must be one of {_WEIGHTINGS}, got {weighting!r}")
    if loudness is None:
        if x is None:
            raise ValueError("give the signal x or a loudness result")
        loudness = _loudness(x, fs, calibration=calibration, field_type=field_type)
    elif not (
        isinstance(loudness, TimeMetric)
        and loudness.name == "loudness"
        and "specific_loudness" in loudness.extra
        and loudness.source.startswith("MoSQITo")
    ):
        raise ValueError("loudness must be the TimeMetric returned by rota_order.bridges.mosqito.loudness")
    sq = _sq_metrics()
    with _quiet():
        s = sq.sharpness_din_from_loudness(
            loudness.values, loudness.extra["specific_loudness"], weighting=weighting
        )
    field = loudness.extra["field_type"]
    return TimeMetric(
        name="sharpness",
        t=loudness.t.copy(),
        values=np.asarray(s, dtype=float).reshape(-1),
        unit="acum",
        skip_start=_SKIP_LOUDNESS,
        aggregate="median",
        extra={"weighting": weighting, "field_type": field},
        source=_source(f"DIN 45692:2009 sharpness ({weighting} weighting) from ISO 532-1:2017 loudness, {field} field"),
    )


def roughness(x, fs, *, calibration=1.0, overlap=0.5):
    """Roughness after Daniel and Weber (1997) (MoSQITo ``roughness_dw``).

    Amplitude and units. ``x`` is the instantaneous sound pressure in Pa (no peak
    or RMS convention applies); the roughness is a perceptual quantity in asper,
    not an amplitude. One value per window hop: 200 ms windows, hop
    ``0.2 * (1 - overlap)`` s (0.1 s at the default); the time axis starts at 0.1 s.

    Parameters
    ----------
    x : array_like, shape (n,)
        Mono sound pressure in file units; multichannel input raises ValueError.
    fs : float
        Sampling rate of ``x`` in Hz; resampled to 48 kHz when different.
    calibration : float, default 1.0
        Factor from file units to Pa.
    overlap : float, default 0.5
        Overlap of the 200 ms windows, 0 to below 1.

    Returns
    -------
    TimeMetric
        ``values`` in asper; ``skip_start`` 1.5 s counted from ``t[0]`` (that is
        0.1 s + 1.5 s from the start of the signal, as in example 07); aggregation
        90th percentile. ``single_value`` NaN. ``extra``: ``"specific_roughness"``
        (asper/Bark, shape (n_bark, n_t)), ``"bark"``, ``"overlap"``.

    Raises
    ------
    ImportError
        If MoSQITo is not installed.
    ValueError
        For multichannel or non-finite input, or ``overlap`` outside [0, 1).

    References
    ----------
    Daniel, P., & Weber, R. (1997). Psychoacoustical roughness: implementation of
    an optimized model. Acustica 83, 113-123.
    MoSQITo: Green Forge Coop (2024). MOSQITO (version 1.2.1) [software]. Zenodo.
    https://doi.org/10.5281/zenodo.11026796. https://github.com/Eomys/MoSQITo
    (Apache-2.0).
    LISQM catalogue: https://zhenxianli.github.io/LISQM/
    """
    overlap = float(overlap)
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")
    p = pressure_48k(x, fs, calibration)
    sq = _sq_metrics()
    with _quiet():
        r, r_spec, bark, t = sq.roughness_dw(p, FS_MODEL, overlap=overlap)
    return TimeMetric(
        name="roughness",
        t=np.asarray(t, dtype=float),
        values=np.asarray(r, dtype=float).reshape(-1),
        unit="asper",
        skip_start=_SKIP_ROUGHNESS,
        aggregate=("percentile", 90),
        extra={
            "specific_roughness": np.asarray(r_spec, dtype=float),
            "bark": np.asarray(bark, dtype=float),
            "overlap": overlap,
        },
        source=_source("Daniel and Weber (1997) roughness"),
    )
