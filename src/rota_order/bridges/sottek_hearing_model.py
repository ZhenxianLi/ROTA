"""Thin wrappers for the Sottek hearing model, ECMA-418-2:2025 (optional external toolbox).

``sottek-hearing-model`` (Lotinga, https://github.com/mlotinga/sottek-hearing-model,
AGPL-3.0-only, Python 3.11 or later, import name ``sottek_hearing_model``) is not a
dependency of rota-order; install it yourself if you accept its licence:
``pip install sottek-hearing-model``. It is imported inside the functions that need it,
never at import of this module. Importing it changes Matplotlib's global ``rcParams``;
they are restored right after the import, so these wrappers have no such side effect.

The module is named exactly like the package it wraps (as ``mosqito`` and ``sqat`` are).
Inside it the package is imported by its absolute name,
``importlib.import_module("sottek_hearing_model")``, so the name of this module
(``rota_order.bridges.sottek_hearing_model``) does not shadow it. A short alias is up to you:
``from rota_order.bridges import sottek_hearing_model as shm``.

Licence: sottek-hearing-model is distributed under the GNU Affero General Public License
v3.0 only (AGPL-3.0-only) by its author. rota-order (GPL-3.0-or-later) does not include, copy,
modify or redistribute any of its code; this module calls its public functions at run
time in an installation the user makes. GPL-3.0 and AGPL-3.0 code may be combined
(section 13 of both licences); whoever distributes such a combination, or lets others use
it over a network, must meet the AGPL-3.0 terms for the model's part. Cite the package as
its author asks (its README; see References).

Wrapped (version 0.1.14, public functions only), ECMA-418-2:2025 (4th edition):

* :func:`tonality`: time-dependent tonality, tu_HMS, 187.5 Hz frame rate.
* :func:`loudness`: time-dependent loudness, sone_HMS, derived from the tonal and noise
  loudness of the tonality run (so it can reuse a :func:`tonality` result).
* :func:`roughness`: time-dependent roughness, asper_HMS, 50 Hz frame rate.

Not wrapped: the ECMA-418-2 functions of MoSQITo (2nd edition only) and ``tnr_ecma_st``
(rota-order has its own ECMA-418-1:2024 tone-to-noise ratio in :mod:`rota_order.tonal`).

Input convention for every function: ``x`` is the instantaneous sound pressure in Pa
(``calibration`` multiplies ``x`` into Pa), mono, sampled at ``fs``; it is resampled to
48 kHz with :func:`rota_order.tonal.to_calibrated_48k`. The returned metrics are perceptual
quantities, not amplitudes: no peak or RMS convention applies. The model is slow (seconds
per second of signal on one core); ``parallel_cores`` controls its multiprocessing.

References
----------
ECMA-418-2:2025 (4th ed.). Psychoacoustic metrics for ITT equipment, Part 2 (models based
on human perception).
Lotinga, M. J. B., Torjussen, M., & Felix Greco, G. (2025). Verified
implementations of the Sottek psychoacoustic Hearing Model standardised sound
quality metrics (ECMA-418-2 loudness, tonality and roughness). Proceedings of
Forum Acusticum / Euronoise 2025, Malaga, Spain.
https://doi.org/10.61782/fa.2025.0951
sottek-hearing-model (version 0.1.14) [software], M. J. B. Lotinga,
https://github.com/mlotinga/sottek-hearing-model (AGPL-3.0-only).
LISQM catalogue: https://zhenxianli.github.io/LISQM/
"""

import contextlib
import importlib
import sys

import numpy as np

from ._common import FS_MODEL, TimeMetric, package_version, pressure_48k

__all__ = ["loudness", "roughness", "tonality"]

_SOUNDFIELDS = ("free_frontal", "diffuse", "no_outer", "no_ear")
_SKIP_TONALITY = 0.30  # s: the 57 start-up frames end at 0.304 s
_SKIP_ROUGHNESS = 0.31  # s: the first 16 frames of 50 Hz end at 0.32 s
_TONALITY_THRESHOLD = 0.02  # tu_HMS


def _model():
    """Import the package ``sottek_hearing_model`` lazily; undo its change of ``matplotlib.rcParams``.

    The import is absolute (``importlib.import_module``), so this module's own name does not
    shadow the package: what is returned is the external package, never this module.
    """
    try:
        mpl = importlib.import_module("matplotlib")
    except ImportError:
        mpl = None
    saved = mpl.rcParams.copy() if mpl is not None else None
    try:
        return importlib.import_module("sottek_hearing_model")
    except ImportError as err:
        raise ImportError(
            "rota_order.bridges.sottek_hearing_model needs the optional package sottek-hearing-model "
            "(ECMA-418-2 Sottek hearing model, AGPL-3.0-only, Python >= 3.11): pip install "
            "sottek-hearing-model. rota-order itself does not depend on it. "
            f"Original error: {err}"
        ) from err
    finally:
        if saved is not None:
            mpl = sys.modules["matplotlib"]
            for key in saved:
                # a key that cannot be compared or set is left as it is
                # (read with dict.__getitem__: reading "backend" through rcParams would resolve it)
                with contextlib.suppress(KeyError, TypeError, ValueError):
                    if dict.__getitem__(mpl.rcParams, key) != dict.__getitem__(saved, key):
                        mpl.rcParams[key] = dict.__getitem__(saved, key)


def _check(soundfield):
    if soundfield not in _SOUNDFIELDS:
        raise ValueError(f"soundfield must be one of {_SOUNDFIELDS}, got {soundfield!r}")


def _source(what):
    return f"sottek-hearing-model {package_version('sottek-hearing-model')}; {what}"


def _one(a):
    return np.asarray(a, dtype=float).reshape(-1)


def tonality(x, fs, *, calibration=1.0, soundfield="free_frontal", wait_bar=False, parallel_cores=None):
    """Time-dependent tonality after ECMA-418-2:2025 (``shm_tonality_ecma``).

    Amplitude and units. ``x`` is the instantaneous sound pressure in Pa (no peak
    or RMS convention applies); the tonality is a perceptual quantity in tu_HMS,
    not an amplitude. Frame rate 187.5 Hz.

    Parameters
    ----------
    x : array_like, shape (n,)
        Mono sound pressure in file units; multichannel input raises ValueError.
    fs : float
        Sampling rate of ``x`` in Hz; resampled to 48 kHz when different.
    calibration : float, default 1.0
        Factor from file units to Pa.
    soundfield : {"free_frontal", "diffuse", "no_outer", "no_ear"}, default "free_frontal"
        Outer and middle ear stages of the model; the last two are beyond the
        scope of the standard.
    wait_bar : bool, default False
        Show the package's progress bar.
    parallel_cores : int or None, default None
        Cores used by the model (None: all but one; 1: no multiprocessing).

    Returns
    -------
    TimeMetric
        ``values`` in tu_HMS on ``t`` (the model's ``time_out``, in s: frames 256
        samples (5.33 ms) apart, the last one at the length of the signal rounded
        up to a multiple of 256 samples, so one sample after the last sample of
        the signal when its length at 48 kHz is a multiple of 256 and up to one
        frame step after it otherwise; :meth:`rota_order.bridges.TimeMetric.vs_rpm`
        handles it by holding the last speed); ``skip_start`` 0.30 s;
        ``threshold`` 0.02 tu_HMS and mean aggregation (the rule of ECMA-418-2
        for the single value); ``single_value`` the package's ``tonality_avg``.
        ``extra``: ``"spec_tonality"`` (shape (n_t, n_bands)),
        ``"spec_tonality_freqs"`` (Hz, same shape), ``"tonality_freqs"`` (Hz,
        frequency of the dominant tonal component per frame), ``"band_centre_freqs"``
        (Hz), ``"spec_tonal_loudness"`` and ``"spec_noise_loudness"`` (for
        :func:`loudness`), ``"soundfield"``.

    Raises
    ------
    ImportError
        If sottek-hearing-model is not installed.
    ValueError
        For multichannel or non-finite input, or an unknown ``soundfield``.

    References
    ----------
    ECMA-418-2:2025 (4th ed.). Psychoacoustic metrics for ITT equipment, Part 2.
    Lotinga, M. J. B., Torjussen, M., & Felix Greco, G. (2025). Verified
    implementations of the Sottek psychoacoustic Hearing Model standardised sound
    quality metrics (ECMA-418-2 loudness, tonality and roughness). Proceedings of
    Forum Acusticum / Euronoise 2025, Malaga, Spain.
    https://doi.org/10.61782/fa.2025.0951
    sottek-hearing-model (version 0.1.14) [software], M. J. B. Lotinga,
    https://github.com/mlotinga/sottek-hearing-model (AGPL-3.0-only).
    LISQM catalogue: https://zhenxianli.github.io/LISQM/
    """
    _check(soundfield)
    p = pressure_48k(x, fs, calibration)
    model = _model()
    r = model.shm_tonality_ecma(
        p, FS_MODEL, soundfield=soundfield, wait_bar=wait_bar, parallel_cores=parallel_cores
    )
    return TimeMetric(
        name="tonality",
        t=_one(r["time_out"]),
        values=_one(r["tonality_t"]),
        unit="tu_HMS",
        single_value=float(np.ravel(r["tonality_avg"])[0]),
        skip_start=_SKIP_TONALITY,
        aggregate="mean",
        threshold=_TONALITY_THRESHOLD,
        extra={
            "spec_tonality": np.asarray(r["spec_tonality"], dtype=float),
            "spec_tonality_freqs": np.asarray(r["spec_tonality_freqs"], dtype=float),
            "tonality_freqs": _one(r["tonality_t_freqs"]),
            "band_centre_freqs": _one(r["band_centre_freqs"]),
            "spec_tonal_loudness": np.asarray(r["spec_tonal_loudness"], dtype=float),
            "spec_noise_loudness": np.asarray(r["spec_noise_loudness"], dtype=float),
            "soundfield": soundfield,
        },
        source=_source(f"ECMA-418-2:2025 (4th ed.) tonality, {soundfield} sound field"),
    )


_tonality = tonality  # the parameter ``tonality`` of ``loudness`` shadows the function


def loudness(
    x, fs, *, calibration=1.0, soundfield="free_frontal", wait_bar=False, parallel_cores=None, tonality=None
):
    """Time-dependent loudness after ECMA-418-2:2025 (``shm_loudness_ecma_from_comp``).

    The package derives the loudness from the tonal and noise loudness of its
    tonality run, so this function runs the tonality model (the slow part). If you
    need both, run :func:`tonality` once and pass it as ``tonality``: then ``x``
    and the other arguments are not used (``x`` may be None).

    Amplitude and units. ``x`` is the instantaneous sound pressure in Pa (no peak
    or RMS convention applies); the loudness is a perceptual quantity in sone_HMS,
    not an amplitude.

    Parameters
    ----------
    x : array_like, shape (n,) or None
        Mono sound pressure in file units; multichannel input raises ValueError.
    fs : float or None
        Sampling rate of ``x`` in Hz; resampled to 48 kHz when different.
    calibration : float, default 1.0
        Factor from file units to Pa.
    soundfield : {"free_frontal", "diffuse", "no_outer", "no_ear"}, default "free_frontal"
        Outer and middle ear stages of the model.
    wait_bar : bool, default False
        Show the package's progress bar.
    parallel_cores : int or None, default None
        Cores used by the model (None: all but one; 1: no multiprocessing).
    tonality : TimeMetric, optional
        Result of :func:`tonality` to derive the loudness from.

    Returns
    -------
    TimeMetric
        ``values`` in sone_HMS on ``t`` in s (the frames of the tonality run, the
        last one included, see :func:`tonality`); ``skip_start`` 0.30 s;
        aggregation mean (the arithmetic mean of the time-dependent loudness, as
        in example 08). ``single_value`` is the package's ``loudness_powavg``, a power
        average, which differs from the arithmetic mean. ``extra``:
        ``"spec_loudness"`` (shape (n_t, n_bands)), ``"spec_tonal_loudness"``,
        ``"spec_noise_loudness"``, ``"band_centre_freqs"`` (Hz), ``"soundfield"``.

    Raises
    ------
    ImportError
        If sottek-hearing-model is not installed.
    ValueError
        For invalid input, an unknown ``soundfield``, or a ``tonality`` that is
        not a result of :func:`tonality`.

    References
    ----------
    ECMA-418-2:2025 (4th ed.). Psychoacoustic metrics for ITT equipment, Part 2.
    Lotinga, M. J. B., Torjussen, M., & Felix Greco, G. (2025). Verified
    implementations of the Sottek psychoacoustic Hearing Model standardised sound
    quality metrics (ECMA-418-2 loudness, tonality and roughness). Proceedings of
    Forum Acusticum / Euronoise 2025, Malaga, Spain.
    https://doi.org/10.61782/fa.2025.0951
    sottek-hearing-model (version 0.1.14) [software], M. J. B. Lotinga,
    https://github.com/mlotinga/sottek-hearing-model (AGPL-3.0-only).
    LISQM catalogue: https://zhenxianli.github.io/LISQM/
    """
    _check(soundfield)
    if tonality is None:
        if x is None:
            raise ValueError("give the signal x or a tonality result")
        ton = _tonality(
            x, fs, calibration=calibration, soundfield=soundfield, wait_bar=wait_bar,
            parallel_cores=parallel_cores,
        )
    else:
        ton = tonality
        if not (
            isinstance(ton, TimeMetric)
            and ton.name == "tonality"
            and "spec_tonal_loudness" in ton.extra
            and ton.source.startswith("sottek-hearing-model")
        ):
            raise ValueError(
                "tonality must be the TimeMetric returned by rota_order.bridges.sottek_hearing_model.tonality"
            )
    model = _model()
    r = model.shm_loudness_ecma_from_comp(
        ton.extra["spec_tonal_loudness"], ton.extra["spec_noise_loudness"]
    )
    field = ton.extra["soundfield"]
    return TimeMetric(
        name="loudness",
        t=_one(r["time_out"]) if "time_out" in r else ton.t.copy(),
        values=_one(r["loudness_t"]),
        unit="sone_HMS",
        single_value=float(np.ravel(r["loudness_powavg"])[0]),
        skip_start=_SKIP_TONALITY,
        aggregate="mean",
        extra={
            "spec_loudness": np.asarray(r["spec_loudness"], dtype=float),
            "spec_tonal_loudness": ton.extra["spec_tonal_loudness"],
            "spec_noise_loudness": ton.extra["spec_noise_loudness"],
            "band_centre_freqs": _one(r["band_centre_freqs"]),
            "soundfield": field,
        },
        source=_source(f"ECMA-418-2:2025 (4th ed.) loudness, {field} sound field"),
    )


def roughness(x, fs, *, calibration=1.0, soundfield="free_frontal", wait_bar=False, parallel_cores=None):
    """Time-dependent roughness after ECMA-418-2:2025 (``shm_roughness_ecma``).

    Amplitude and units. ``x`` is the instantaneous sound pressure in Pa (no peak
    or RMS convention applies); the roughness is a perceptual quantity in
    asper_HMS, not an amplitude. Frame rate 50 Hz.

    Parameters
    ----------
    x : array_like, shape (n,)
        Mono sound pressure in file units; multichannel input raises ValueError.
    fs : float
        Sampling rate of ``x`` in Hz; resampled to 48 kHz when different.
    calibration : float, default 1.0
        Factor from file units to Pa.
    soundfield : {"free_frontal", "diffuse", "no_outer", "no_ear"}, default "free_frontal"
        Outer and middle ear stages of the model.
    wait_bar : bool, default False
        Show the package's progress bar.
    parallel_cores : int or None, default None
        Cores used by the model (None: all but one; 1: no multiprocessing).

    Returns
    -------
    TimeMetric
        ``values`` in asper_HMS on ``t`` in s; ``skip_start`` 0.31 s;
        aggregation 90th percentile. ``single_value`` is the package's
        ``roughness90pc``. ``extra``: ``"spec_roughness"`` (shape (n_t, n_bands)),
        ``"band_centre_freqs"`` (Hz), ``"soundfield"``.

    Raises
    ------
    ImportError
        If sottek-hearing-model is not installed.
    ValueError
        For multichannel or non-finite input, or an unknown ``soundfield``.

    References
    ----------
    ECMA-418-2:2025 (4th ed.). Psychoacoustic metrics for ITT equipment, Part 2.
    Lotinga, M. J. B., Torjussen, M., & Felix Greco, G. (2025). Verified
    implementations of the Sottek psychoacoustic Hearing Model standardised sound
    quality metrics (ECMA-418-2 loudness, tonality and roughness). Proceedings of
    Forum Acusticum / Euronoise 2025, Malaga, Spain.
    https://doi.org/10.61782/fa.2025.0951
    sottek-hearing-model (version 0.1.14) [software], M. J. B. Lotinga,
    https://github.com/mlotinga/sottek-hearing-model (AGPL-3.0-only).
    LISQM catalogue: https://zhenxianli.github.io/LISQM/
    """
    _check(soundfield)
    p = pressure_48k(x, fs, calibration)
    model = _model()
    r = model.shm_roughness_ecma(
        p, FS_MODEL, soundfield=soundfield, wait_bar=wait_bar, parallel_cores=parallel_cores
    )
    return TimeMetric(
        name="roughness",
        t=_one(r["time_out"]),
        values=_one(r["roughness_t"]),
        unit="asper_HMS",
        single_value=float(np.ravel(r["roughness90pc"])[0]),
        skip_start=_SKIP_ROUGHNESS,
        aggregate=("percentile", 90),
        extra={
            "spec_roughness": np.asarray(r["spec_roughness"], dtype=float),
            "band_centre_freqs": _one(r["band_centre_freqs"]),
            "soundfield": soundfield,
        },
        source=_source(f"ECMA-418-2:2025 (4th ed.) roughness, {soundfield} sound field"),
    )
