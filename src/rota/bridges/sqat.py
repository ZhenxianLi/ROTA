"""Thin wrappers for SQAT (MATLAB) through the MATLAB Engine API for Python (optional).

SQAT, the Sound Quality Analysis Toolbox (Felix Greco, Merino-Martinez, Osses and
Lotinga; https://github.com/ggrecow/SQAT), is a MATLAB toolbox. It is not a dependency of
ROTA. To use this module you need, on your own machine:

* a licensed local MATLAB installation, with the Signal Processing Toolbox for the
  ECMA-418-2 functions (:func:`tonality`);
* the MATLAB Engine API for Python, ``pip install matlabengine==<version>``, where the
  version must match your MATLAB release (for example 25.1.x for R2025a, 25.2.x for
  R2025b, 26.1.x for R2026a; see the release table on PyPI and the MathWorks
  documentation);
* a checkout of SQAT, release v2.0 (commit 94f489c053d1366da05d5e62838c3f54ac7715c3,
  the release these wrappers were written and tested against), whose folder you pass as
  ``sqat_path`` or set in the environment variable ``ROTA_SQAT_PATH``.

``matlab.engine`` is imported inside the functions that need it, never at import of this
module; without it they raise ImportError with these instructions.

Licence: SQAT is distributed by its authors under the GPL-3.0 as a whole; some of its files
carry their own licence, stated in their headers. At the pinned release v2.0 the headers of
the wrapped functions state: ``Loudness_ISO532_1`` BSD-3-Clause; ``Roughness_Daniel1997``
MIT; ``Sharpness_DIN45692_from_loudness`` GPL-3.0; ``Tonality_ECMA418_2`` GPL-3.0 (work of
the RefMap project). ROTA does not include, copy, modify or redistribute any SQAT code; this
module calls SQAT's public functions at run time, through the MATLAB Engine, in an
installation the user makes. MATLAB and the Signal Processing Toolbox are commercial
products of The MathWorks, Inc. that the user must license themselves. Cite SQAT as its
authors ask (README of the SQAT repository, "How to cite this repository" and
"Publications about SQAT"; see References), preferably with the release used.

Wrapped (SQAT v2.0, documented public functions only, called as described in the SQAT
README and the help header of each function, every parameter given explicitly; defaults
as in ``examples/09_sqat_bridge``):

* :func:`loudness`: ``Loudness_ISO532_1``, ISO 532-1:2017 time-varying loudness
  (Zwicker, method 2), sone.
* :func:`sharpness`: ``Sharpness_DIN45692_from_loudness``, DIN 45692:2009 sharpness, acum,
  from the specific loudness of :func:`loudness`.
* :func:`roughness`: ``Roughness_Daniel1997``, Daniel and Weber (1997) roughness, asper.
* :func:`tonality`: ``Tonality_ECMA418_2``, ECMA-418-2:2025 (4th edition, Sottek Hearing
  Model) tonality, tu_HMS (the edition stated in the help header of the function).

Sessions. Every function takes ``session=``: a :class:`Session` (from :func:`start`), or
None for the module's default session, which is started on first use from
``ROTA_SQAT_PATH``, reused by later calls and closed at interpreter exit (or by
:func:`close_default`). Starting MATLAB takes several seconds; reuse a session.

Input convention for every function: ``x`` is the instantaneous sound pressure in Pa
(``calibration`` multiplies ``x`` into Pa), mono, sampled at ``fs``; it is resampled to
48 kHz with :func:`rota.tonal.to_calibrated_48k` and passed to MATLAB as a ``matlab.double``
column vector. The returned metrics are perceptual quantities, not amplitudes: no peak or
RMS convention applies. The SQAT output is not part of ROTA (ROADMAP section 5.5).

Available since ROTA 0.6.0.

References
----------
ISO 532-1:2017. Acoustics - Methods for calculating loudness - Part 1: Zwicker method.
DIN 45692:2009. Measurement technique for the simulation of the auditory sensation of
sharpness.
Daniel, P., & Weber, R. (1997). Psychoacoustical roughness: implementation of an optimized
model. Acustica, 83(1), 113-123.
ECMA-418-2, 4th edition (2025). Psychoacoustic metrics for ITT equipment - Part 2 (methods
for describing human perception based on the Sottek Hearing Model).
Felix Greco, G., Merino-Martínez, R., Osses, A., & Lotinga, M. J. B. (2026). SQAT: a sound
quality analysis toolbox for MATLAB (release v2.0). Zenodo.
https://doi.org/10.5281/zenodo.7934709 (GPL-3.0).
Felix Greco, G., Merino-Martínez, R., Osses, A., & Langer, S. C. (2023). SQAT: a
MATLAB-based toolbox for quantitative sound quality analysis. INTER-NOISE 2023, Chiba,
Japan. https://doi.org/10.3397/IN_2023_1075
MathWorks. MATLAB Engine API for Python (documentation "Call MATLAB from Python" and
"Pass Data Between MATLAB and Python");
https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
LISQM catalogue: https://zhenxianli.github.io/LISQM/
"""

import atexit
import contextlib
import importlib
import io
import os
import subprocess

import numpy as np

from ._common import FS_MODEL, TimeMetric, pressure_48k

__all__ = [
    "Session",
    "close_default",
    "default_session",
    "loudness",
    "roughness",
    "sharpness",
    "start",
    "tonality",
]

ENV_PATH = "ROTA_SQAT_PATH"
SQAT_RELEASE = "v2.0"
SQAT_COMMIT = "94f489c053d1366da05d5e62838c3f54ac7715c3"

_FIELDS_ISO532 = {"free": 0.0, "diffuse": 1.0}
_FIELDS_ECMA = ("free-frontal", "diffuse")
_WEIGHTS = ("DIN45692", "bismarck", "aures")
# Start-up exclusions of examples/09_sqat_bridge (import_results.py), s.
_SKIP_LOUDNESS = 0.6
_SKIP_ROUGHNESS = 1.5
_SKIP_TONALITY = 0.6
_TONALITY_THRESHOLD = 0.02  # tu_HMS, ECMA-418-2 rule for the single value (as rota.bridges.shm)
_ECMA_MIN_SKIP = 0.304  # s, ECMA-418-2:2025 section 6.2.9 as stated in the SQAT help header

_INSTALL_HINT = (
    "rota.bridges.sqat needs the MATLAB Engine API for Python and a local, licensed MATLAB "
    "(with the Signal Processing Toolbox for the ECMA-418-2 functions). Install the engine "
    "with `pip install matlabengine==<version matching your MATLAB release>` (for example "
    "25.1.x for R2025a; see https://pypi.org/project/matlabengine/), and get SQAT release "
    f"{SQAT_RELEASE} (https://github.com/ggrecow/SQAT, commit {SQAT_COMMIT[:7]}); pass its "
    f"folder as sqat_path or set {ENV_PATH}. ROTA itself does not depend on any of these."
)


def _matlab():
    """Import ``matlab`` and ``matlab.engine`` lazily; ImportError with the install hint."""
    try:
        engine = importlib.import_module("matlab.engine")
        mat = importlib.import_module("matlab")
    except ImportError as err:
        raise ImportError(f"{_INSTALL_HINT} Original error: {err}") from err
    return mat, engine


def _resolve_path(sqat_path):
    path = sqat_path if sqat_path is not None else os.environ.get(ENV_PATH)
    if not path:
        raise ValueError(
            f"No SQAT folder given: pass sqat_path or set the environment variable {ENV_PATH} "
            f"to a checkout of SQAT release {SQAT_RELEASE} (https://github.com/ggrecow/SQAT)."
        )
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(os.path.join(path, "startup_SQAT.m")):
        raise FileNotFoundError(
            f"{path} is not a SQAT folder (startup_SQAT.m not found); pass sqat_path or set "
            f"{ENV_PATH} to a checkout of SQAT release {SQAT_RELEASE}."
        )
    return path


def _sqat_version(path):
    """Release and commit of a SQAT checkout from its git metadata, if there is any."""
    try:
        head = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "(version unknown: no git metadata)"
    if head == SQAT_COMMIT:
        return f"{SQAT_RELEASE} ({head[:7]})"
    return f"commit {head[:7]} (not the tested release {SQAT_RELEASE})"


class Session:
    """A MATLAB engine with SQAT on its path.

    Starts a MATLAB engine with ``matlab.engine.start_matlab()`` (or uses the one
    given as ``engine``), adds ``sqat_path`` to the MATLAB path and runs SQAT's
    documented initialiser ``startup_SQAT``. Use it as a context manager, or call
    :meth:`close`: an engine started here is quit, an engine passed in is left
    running (it belongs to the caller).

    Parameters
    ----------
    sqat_path : str or path-like, optional
        Folder of a SQAT checkout (the one containing ``startup_SQAT.m``); release
        v2.0 is the one tested. Default: the environment variable ``ROTA_SQAT_PATH``.
    engine : matlab.engine.MatlabEngine, optional
        A running engine to use, for example from ``matlab.engine.connect_matlab()``.

    Attributes
    ----------
    engine : matlab.engine.MatlabEngine
        The engine.
    sqat_path : str
        Absolute SQAT folder.
    sqat_version : str
        SQAT release and commit when the folder is a git checkout.
    matlab_release : str
        MATLAB release, e.g. ``"R2025a"``.
    source : str
        ``"SQAT <version> via MATLAB <release>"``, used in ``TimeMetric.source``.

    Raises
    ------
    ImportError
        If ``matlab.engine`` (the ``matlabengine`` package) is not installed.
    ValueError
        If neither ``sqat_path`` nor ``ROTA_SQAT_PATH`` is given.
    FileNotFoundError
        If the folder does not contain ``startup_SQAT.m``.

    References
    ----------
    MathWorks, MATLAB Engine API for Python ("Start and Stop MATLAB Engine for
    Python"); https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
    SQAT README (https://github.com/ggrecow/SQAT), section on ``startup_SQAT``.
    Felix Greco, G., Merino-Martínez, R., Osses, A., & Lotinga, M. J. B. (2026).
    SQAT: a sound quality analysis toolbox for MATLAB (release v2.0). Zenodo.
    https://doi.org/10.5281/zenodo.7934709 (GPL-3.0).
    Felix Greco, G., Merino-Martínez, R., Osses, A., & Langer, S. C. (2023). SQAT:
    a MATLAB-based toolbox for quantitative sound quality analysis. INTER-NOISE
    2023, Chiba, Japan. https://doi.org/10.3397/IN_2023_1075
    LISQM catalogue: https://zhenxianli.github.io/LISQM/
    """

    def __init__(self, sqat_path=None, *, engine=None):
        mat, eng_mod = _matlab()
        self._double = mat.double
        self.sqat_path = _resolve_path(sqat_path)
        self._owns = engine is None
        self.engine = eng_mod.start_matlab() if engine is None else engine
        try:
            self.call("addpath", self.sqat_path, nargout=0)
            self.call("startup_SQAT", self.sqat_path + os.sep, nargout=0)
            release = str(self.call("version", "-release"))
        except BaseException:
            self.close()
            raise
        self.matlab_release = release if release.startswith("R") else f"R{release}"
        self.sqat_version = _sqat_version(self.sqat_path)
        self.source = f"SQAT {self.sqat_version} via MATLAB {self.matlab_release}"

    def call(self, fn, *args, nargout=1):
        """Call the MATLAB function ``fn`` with ``args``; MATLAB's console output is captured.

        Raises RuntimeError (chained to the engine's exception) with the function
        name and MATLAB's output when the call fails.
        """
        out = io.StringIO()
        try:
            return getattr(self.engine, fn)(*args, nargout=nargout, stdout=out, stderr=out)
        except Exception as err:
            log = out.getvalue().strip()
            raise RuntimeError(
                f"MATLAB call {fn} failed: {err}" + (f"\nMATLAB output:\n{log}" if log else "")
            ) from err

    def to_matlab(self, a):
        """A NumPy array as ``matlab.double``: shape (n,) becomes an (n, 1) column."""
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if a.ndim != 2:
            raise ValueError("only vectors and matrices are passed to MATLAB")
        a = np.ascontiguousarray(a)
        try:
            m = self._double(a)  # NumPy input is accepted by the engine since R2022a
            if tuple(getattr(m, "size", a.shape)) != a.shape:
                raise ValueError("shape not kept")
        except (TypeError, ValueError):
            m = self._double(a.tolist())  # nested lists: rows of the matrix
        return m

    def close(self):
        """Quit the engine if this session started it; otherwise leave it running."""
        if self._owns and self.engine is not None:
            with contextlib.suppress(Exception):  # engine already gone
                self.engine.quit()
        self.engine = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def start(sqat_path=None, *, engine=None):
    """Start a :class:`Session` (MATLAB engine with SQAT initialised).

    Parameters
    ----------
    sqat_path : str or path-like, optional
        SQAT folder; default the environment variable ``ROTA_SQAT_PATH``.
    engine : matlab.engine.MatlabEngine, optional
        A running engine to use instead of starting one (not quit on close).

    Returns
    -------
    Session

    References
    ----------
    MathWorks, MATLAB Engine API for Python;
    https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
    SQAT README (https://github.com/ggrecow/SQAT).
    Felix Greco, G., Merino-Martínez, R., Osses, A., & Lotinga, M. J. B. (2026).
    SQAT: a sound quality analysis toolbox for MATLAB (release v2.0). Zenodo.
    https://doi.org/10.5281/zenodo.7934709 (GPL-3.0).
    Felix Greco, G., Merino-Martínez, R., Osses, A., & Langer, S. C. (2023). SQAT:
    a MATLAB-based toolbox for quantitative sound quality analysis. INTER-NOISE
    2023, Chiba, Japan. https://doi.org/10.3397/IN_2023_1075
    LISQM catalogue: https://zhenxianli.github.io/LISQM/
    """
    return Session(sqat_path, engine=engine)


_DEFAULT = None


def default_session():
    """The module's default :class:`Session`, started from ``ROTA_SQAT_PATH`` on first use.

    It is reused by every function called with ``session=None`` and closed at
    interpreter exit or by :func:`close_default`.

    References
    ----------
    MathWorks, MATLAB Engine API for Python;
    https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
    Felix Greco, G., Merino-Martínez, R., Osses, A., & Lotinga, M. J. B. (2026).
    SQAT: a sound quality analysis toolbox for MATLAB (release v2.0). Zenodo.
    https://doi.org/10.5281/zenodo.7934709 (GPL-3.0).
    Felix Greco, G., Merino-Martínez, R., Osses, A., & Langer, S. C. (2023). SQAT:
    a MATLAB-based toolbox for quantitative sound quality analysis. INTER-NOISE
    2023, Chiba, Japan. https://doi.org/10.3397/IN_2023_1075
    LISQM catalogue: https://zhenxianli.github.io/LISQM/
    """
    global _DEFAULT
    if _DEFAULT is None or _DEFAULT.engine is None:
        _DEFAULT = Session()
    return _DEFAULT


def close_default():
    """Quit the default session's engine, if one was started.

    References
    ----------
    MathWorks, MATLAB Engine API for Python ("Start and Stop MATLAB Engine for
    Python"); https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
    """
    global _DEFAULT
    if _DEFAULT is not None:
        _DEFAULT.close()
        _DEFAULT = None


atexit.register(close_default)


# ---- conversion of the outputs -------------------------------------------------------------


def _arr(v):
    return np.asarray(v, dtype=float)


def _series(out, t_key, v_key):
    """Time axis and values from a SQAT output struct.

    Both are cut to the shorter length, and samples whose time is not finite or
    does not increase are dropped (``metric_vs_rpm`` needs a strictly increasing
    time), as in ``examples/09_sqat_bridge/import_results.py``. Returns
    ``(t, v, idx)`` with ``idx`` the indices kept.
    """
    t = _arr(out[t_key]).reshape(-1)
    v = _arr(out[v_key]).reshape(-1)
    n = min(t.size, v.size)
    t, v = t[:n], v[:n]
    keep = np.isfinite(t)
    keep[1:] &= np.diff(t) > 0
    idx = np.flatnonzero(keep)
    return t[idx], v[idx], idx


def _stat(out, key):
    """A documented scalar statistic of a SQAT output, NaN if absent."""
    if key not in out:
        return np.nan
    a = _arr(out[key]).reshape(-1)
    return float(a[0]) if a.size else np.nan


def _statistics(out):
    """Every scalar numeric field of a SQAT output struct (its statistics), as floats."""
    stats = {}
    for k, v in out.items():
        if isinstance(v, (bool, str)):
            continue
        try:
            a = _arr(v).reshape(-1)
        except (TypeError, ValueError):
            continue
        if a.size == 1:
            stats[k] = float(a[0])
    return stats


def _session(session):
    return default_session() if session is None else session


# ---- metrics --------------------------------------------------------------------------------


def loudness(x, fs, *, calibration=1.0, field_type="free", time_skip=0.5, session=None):
    """Time-varying loudness after ISO 532-1:2017 (SQAT ``Loudness_ISO532_1``, method 2).

    Calls ``Loudness_ISO532_1(x, 48000, field, 2, time_skip, false)`` with ``x`` the
    calibrated 48 kHz column, field 0 (free) or 1 (diffuse) and method 2
    (time-varying, from the audio signal), as in ``examples/09_sqat_bridge``.

    Amplitude and units. ``x`` is the instantaneous sound pressure in Pa (not an
    amplitude spectrum; no peak or RMS convention applies); ``calibration``
    multiplies ``x`` into Pa. The loudness is a perceptual quantity in sone, on
    SQAT's time axis in s (sample 0 of ``x`` at t = 0). It is not an amplitude.

    Parameters
    ----------
    x : array_like, shape (n,)
        Mono sound pressure in file units; multichannel input raises ValueError.
    fs : float
        Sampling rate of ``x`` in Hz; resampled to 48 kHz when different.
    calibration : float, default 1.0
        Factor from file units to Pa (1.0: ``x`` is already in Pa).
    field_type : {"free", "diffuse"}, default "free"
        Sound field of ISO 532-1 (SQAT ``field`` 0 or 1).
    time_skip : float, default 0.5
        s skipped by SQAT in its own statistics (N5, mean, ...); the time series
        is returned complete.
    session : Session, optional
        Default: the module's default session (see :func:`default_session`).

    Returns
    -------
    TimeMetric
        ``values`` (``InstantaneousLoudness``) in sone on ``t`` (``time``);
        ``skip_start`` 0.6 s; aggregation 95th percentile; ``single_value`` SQAT's
        ``N5`` (loudness exceeded 5 % of the time after ``time_skip``; NaN if SQAT
        does not return it). ``extra``: ``"loudness_level"`` (phon, on ``t``),
        ``"specific_loudness"`` (sone/Bark, shape (n_bark, n_full) on
        ``"time_full"``, SQAT's complete time axis in s), ``"bark"``,
        ``"statistics"`` (every scalar SQAT returns), ``"field_type"``,
        ``"time_skip"``, ``"call"``.

    Raises
    ------
    ImportError
        If the MATLAB Engine API for Python is not installed.
    ValueError
        For invalid input or an unknown ``field_type``, or no SQAT folder.
    RuntimeError
        If the MATLAB call fails.

    References
    ----------
    ISO 532-1:2017. Acoustics - Methods for calculating loudness - Part 1: Zwicker
    method.
    Felix Greco, G., Merino-Martínez, R., Osses, A., & Lotinga, M. J. B. (2026).
    SQAT: a sound quality analysis toolbox for MATLAB (release v2.0). Zenodo.
    https://doi.org/10.5281/zenodo.7934709 (GPL-3.0).
    Felix Greco, G., Merino-Martínez, R., Osses, A., & Langer, S. C. (2023). SQAT:
    a MATLAB-based toolbox for quantitative sound quality analysis. INTER-NOISE
    2023, Chiba, Japan. https://doi.org/10.3397/IN_2023_1075
    MathWorks, MATLAB Engine API for Python;
    https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
    LISQM catalogue: https://zhenxianli.github.io/LISQM/
    """
    return _loudness(x, fs, calibration, field_type, time_skip, session)


def _loudness(x, fs, calibration, field_type, time_skip, session):
    if field_type not in _FIELDS_ISO532:
        raise ValueError(f"field_type must be one of {tuple(_FIELDS_ISO532)}, got {field_type!r}")
    s = _session(session)
    p = pressure_48k(x, fs, calibration)
    out = s.call(
        "Loudness_ISO532_1",
        s.to_matlab(p), FS_MODEL, _FIELDS_ISO532[field_type], 2.0, float(time_skip), False,
    )
    t_raw = _arr(out["time"]).reshape(-1)
    t, v, idx = _series(out, "time", "InstantaneousLoudness")
    spec = _arr(out["InstantaneousSpecificLoudness"])
    if spec.ndim == 2 and spec.shape[0] == t_raw.size and spec.shape[1] != t_raw.size:
        spec = spec.T  # SQAT: [time, Bark]; stored as (n_bark, n_t) like rota.bridges.mosqito
    level = _arr(out["InstantaneousLoudnessLevel"]).reshape(-1)
    if idx.size and level.size > idx[-1]:
        level = level[idx]
    return TimeMetric(
        name="loudness",
        t=t,
        values=v,
        unit="sone",
        single_value=_stat(out, "N5"),
        skip_start=_SKIP_LOUDNESS,
        aggregate=("percentile", 95),
        extra={
            "loudness_level": level,
            "specific_loudness": spec,
            "time_full": t_raw,
            "bark": _arr(out["barkAxis"]).reshape(-1) if "barkAxis" in out else np.array([]),
            "statistics": _statistics(out),
            "field_type": field_type,
            "time_skip": float(time_skip),
            "call": f"Loudness_ISO532_1(x, 48000, {_FIELDS_ISO532[field_type]:g}, 2, {time_skip:g}, false)",
        },
        source=f"{s.source}; ISO 532-1:2017 time-varying loudness, {field_type} field",
    )


def sharpness(
    x, fs, *, calibration=1.0, field_type="free", weight_type="DIN45692", time_skip=0.5,
    loudness=None, session=None,
):
    """Time-varying sharpness after DIN 45692:2009 (SQAT ``Sharpness_DIN45692_from_loudness``).

    Calls ``Sharpness_DIN45692_from_loudness(N', weight_type, time, time_skip, false)``
    on the specific loudness ``N'`` of :func:`loudness` (``Loudness_ISO532_1``), as in
    ``examples/09_sqat_bridge``. To avoid computing the loudness twice, pass the
    result of :func:`loudness` as ``loudness`` (then ``x``, ``fs``, ``calibration``
    and ``field_type`` are not used and ``x`` may be None).

    Amplitude and units. ``x`` is the instantaneous sound pressure in Pa (no peak or
    RMS convention applies); the sharpness is a perceptual quantity in acum, not an
    amplitude.

    Parameters
    ----------
    x : array_like, shape (n,) or None
        Mono sound pressure in file units. May be None if ``loudness`` is given.
    fs : float or None
        Sampling rate of ``x`` in Hz.
    calibration : float, default 1.0
        Factor from file units to Pa.
    field_type : {"free", "diffuse"}, default "free"
        Sound field of the loudness (ISO 532-1).
    weight_type : {"DIN45692", "bismarck", "aures"}, default "DIN45692"
        Weighting function; only ``"DIN45692"`` is the standard method.
    time_skip : float, default 0.5
        s skipped by SQAT in its statistics (series returned complete).
    loudness : TimeMetric, optional
        Result of :func:`loudness` of this module.
    session : Session, optional
        Default: the module's default session.

    Returns
    -------
    TimeMetric
        ``values`` (``InstantaneousSharpness``) in acum on ``t`` (``time``);
        ``skip_start`` 0.6 s; aggregation median; ``single_value`` SQAT's ``S50``
        (sharpness exceeded 50 % of the time after ``time_skip``; NaN if absent).
        ``extra``: ``"statistics"``, ``"weight_type"``, ``"field_type"``,
        ``"time_skip"``, ``"call"``.

    Raises
    ------
    ImportError
        If the MATLAB Engine API for Python is not installed.
    ValueError
        For invalid input, an unknown ``weight_type``, or a ``loudness`` that is not
        a result of :func:`loudness` of this module.
    RuntimeError
        If the MATLAB call fails.

    References
    ----------
    DIN 45692:2009. Measurement technique for the simulation of the auditory
    sensation of sharpness.
    ISO 532-1:2017 (the loudness it is computed from).
    Felix Greco, G., Merino-Martínez, R., Osses, A., & Lotinga, M. J. B. (2026).
    SQAT: a sound quality analysis toolbox for MATLAB (release v2.0). Zenodo.
    https://doi.org/10.5281/zenodo.7934709 (GPL-3.0).
    Felix Greco, G., Merino-Martínez, R., Osses, A., & Langer, S. C. (2023). SQAT:
    a MATLAB-based toolbox for quantitative sound quality analysis. INTER-NOISE
    2023, Chiba, Japan. https://doi.org/10.3397/IN_2023_1075
    MathWorks, MATLAB Engine API for Python;
    https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
    LISQM catalogue: https://zhenxianli.github.io/LISQM/
    """
    if weight_type not in _WEIGHTS:
        raise ValueError(f"weight_type must be one of {_WEIGHTS}, got {weight_type!r}")
    s = _session(session)
    if loudness is None:
        if x is None:
            raise ValueError("give the signal x or a loudness result")
        loudness = _loudness(x, fs, calibration, field_type, time_skip, s)
    elif not (
        isinstance(loudness, TimeMetric)
        and loudness.name == "loudness"
        and "specific_loudness" in loudness.extra
        and "time_full" in loudness.extra
        and loudness.source.startswith("SQAT")
    ):
        raise ValueError("loudness must be the TimeMetric returned by rota.bridges.sqat.loudness")
    spec = np.asarray(loudness.extra["specific_loudness"], dtype=float)
    t_full = np.asarray(loudness.extra["time_full"], dtype=float)
    if spec.ndim == 2 and spec.shape[1] == t_full.size:
        spec = spec.T  # back to SQAT's [time, Bark]
    out = s.call(
        "Sharpness_DIN45692_from_loudness",
        s.to_matlab(spec), weight_type, s.to_matlab(t_full), float(time_skip), False,
    )
    t, v, _ = _series(out, "time", "InstantaneousSharpness")
    field = loudness.extra["field_type"]
    return TimeMetric(
        name="sharpness",
        t=t,
        values=v,
        unit="acum",
        single_value=_stat(out, "S50"),
        skip_start=_SKIP_LOUDNESS,
        aggregate="median",
        extra={
            "statistics": _statistics(out),
            "weight_type": weight_type,
            "field_type": field,
            "time_skip": float(time_skip),
            "call": f"Sharpness_DIN45692_from_loudness(N', '{weight_type}', time, {time_skip:g}, false)",
        },
        source=(
            f"{s.source}; DIN 45692:2009 sharpness ({weight_type} weighting) from "
            f"ISO 532-1:2017 loudness, {field} field"
        ),
    )


def roughness(x, fs, *, calibration=1.0, time_skip=1.5, session=None):
    """Roughness after Daniel and Weber (1997) (SQAT ``Roughness_Daniel1997``).

    Calls ``Roughness_Daniel1997(x, 48000, time_skip, false)``, as in
    ``examples/09_sqat_bridge``. SQAT v2.0 uses 200 ms windows with 50 % overlap
    (help header of the function).

    Amplitude and units. ``x`` is the instantaneous sound pressure in Pa (no peak or
    RMS convention applies); the roughness is a perceptual quantity in asper, not an
    amplitude.

    Parameters
    ----------
    x : array_like, shape (n,)
        Mono sound pressure in file units; multichannel input raises ValueError.
    fs : float
        Sampling rate of ``x`` in Hz; resampled to 48 kHz when different.
    calibration : float, default 1.0
        Factor from file units to Pa.
    time_skip : float, default 1.5
        s skipped by SQAT in its statistics (series returned complete).
    session : Session, optional
        Default: the module's default session.

    Returns
    -------
    TimeMetric
        ``values`` (``InstantaneousRoughness``) in asper on ``t`` (``time``);
        ``skip_start`` 1.5 s; aggregation 90th percentile; ``single_value`` SQAT's
        ``R10`` (roughness exceeded 10 % of the time after ``time_skip``; NaN if
        absent). ``extra``: ``"specific_roughness"`` (asper/Bark as SQAT returns
        it), ``"bark"``, ``"statistics"``, ``"time_skip"``, ``"call"``.

    Raises
    ------
    ImportError
        If the MATLAB Engine API for Python is not installed.
    ValueError
        For invalid input or no SQAT folder.
    RuntimeError
        If the MATLAB call fails.

    References
    ----------
    Daniel, P., & Weber, R. (1997). Psychoacoustical roughness: implementation of an
    optimized model. Acustica, 83(1), 113-123.
    Felix Greco, G., Merino-Martínez, R., Osses, A., & Lotinga, M. J. B. (2026).
    SQAT: a sound quality analysis toolbox for MATLAB (release v2.0). Zenodo.
    https://doi.org/10.5281/zenodo.7934709 (GPL-3.0).
    Felix Greco, G., Merino-Martínez, R., Osses, A., & Langer, S. C. (2023). SQAT:
    a MATLAB-based toolbox for quantitative sound quality analysis. INTER-NOISE
    2023, Chiba, Japan. https://doi.org/10.3397/IN_2023_1075
    MathWorks, MATLAB Engine API for Python;
    https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
    LISQM catalogue: https://zhenxianli.github.io/LISQM/
    """
    s = _session(session)
    p = pressure_48k(x, fs, calibration)
    out = s.call("Roughness_Daniel1997", s.to_matlab(p), FS_MODEL, float(time_skip), False)
    t, v, _ = _series(out, "time", "InstantaneousRoughness")
    extra = {
        "statistics": _statistics(out),
        "time_skip": float(time_skip),
        "call": f"Roughness_Daniel1997(x, 48000, {time_skip:g}, false)",
    }
    if "InstantaneousSpecificRoughness" in out:
        extra["specific_roughness"] = _arr(out["InstantaneousSpecificRoughness"])
    if "barkAxis" in out:
        extra["bark"] = _arr(out["barkAxis"]).reshape(-1)
    return TimeMetric(
        name="roughness",
        t=t,
        values=v,
        unit="asper",
        single_value=_stat(out, "R10"),
        skip_start=_SKIP_ROUGHNESS,
        aggregate=("percentile", 90),
        extra=extra,
        source=f"{s.source}; Daniel and Weber (1997) roughness",
    )


def tonality(x, fs, *, calibration=1.0, field_type="free-frontal", time_skip=0.304, session=None):
    """Tonality after ECMA-418-2:2025, 4th edition (SQAT ``Tonality_ECMA418_2``).

    Calls ``Tonality_ECMA418_2(x, 48000, field_type, time_skip, false)`` on the mono
    signal, as in ``examples/09_sqat_bridge``. The edition is the one stated in the
    help header of the function in SQAT v2.0 (ECMA-418-2:2025, Sottek Hearing Model).
    Requires the Signal Processing Toolbox in MATLAB.

    Amplitude and units. ``x`` is the instantaneous sound pressure in Pa (no peak or
    RMS convention applies); the tonality is a perceptual quantity in tu_HMS, not an
    amplitude.

    Parameters
    ----------
    x : array_like, shape (n,)
        Mono sound pressure in file units; multichannel input raises ValueError.
    fs : float
        Sampling rate of ``x`` in Hz; resampled to 48 kHz when different.
    calibration : float, default 1.0
        Factor from file units to Pa.
    field_type : {"free-frontal", "diffuse"}, default "free-frontal"
        Outer and middle ear filter of ECMA-418-2.
    time_skip : float, default 0.304
        s skipped by SQAT in ``tonalityAvg`` and its statistics; at least 0.304 s
        (ECMA-418-2:2025 section 6.2.9). The series is returned complete.
    session : Session, optional
        Default: the module's default session.

    Returns
    -------
    TimeMetric
        ``values`` (``tonalityTDep``) in tu_HMS on ``t`` (``timeOut``);
        ``skip_start`` 0.6 s (examples/09); ``threshold`` 0.02 tu_HMS and mean
        aggregation (the rule of ECMA-418-2 for the single value, as in
        :mod:`rota.bridges.shm`); ``single_value`` SQAT's ``tonalityAvg``.
        ``extra``: ``"frequency"`` (Hz, ``tonalityTDepFreqs``, on ``t``),
        ``"specific_tonality"`` (``specTonality``, [time, bands]),
        ``"band_centre_freqs"`` (Hz), ``"statistics"``, ``"field_type"``,
        ``"time_skip"``, ``"call"``.

    Raises
    ------
    ImportError
        If the MATLAB Engine API for Python is not installed.
    ValueError
        For invalid input, an unknown ``field_type`` or ``time_skip`` below 0.304 s.
    RuntimeError
        If the MATLAB call fails (for example without the Signal Processing Toolbox).

    References
    ----------
    ECMA-418-2, 4th edition (2025). Psychoacoustic metrics for ITT equipment -
    Part 2 (methods for describing human perception based on the Sottek Hearing
    Model).
    Lotinga, M. J. B., Torjussen, M., & Felix Greco, G. (2025). Verified
    implementations of the Sottek psychoacoustic Hearing Model standardised sound
    quality metrics. Forum Acusticum 2025. doi:10.61782/fa.2025.0951.
    Felix Greco, G., Merino-Martínez, R., Osses, A., & Lotinga, M. J. B. (2026).
    SQAT: a sound quality analysis toolbox for MATLAB (release v2.0). Zenodo.
    https://doi.org/10.5281/zenodo.7934709 (GPL-3.0).
    Felix Greco, G., Merino-Martínez, R., Osses, A., & Langer, S. C. (2023). SQAT:
    a MATLAB-based toolbox for quantitative sound quality analysis. INTER-NOISE
    2023, Chiba, Japan. https://doi.org/10.3397/IN_2023_1075
    MathWorks, MATLAB Engine API for Python;
    https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
    LISQM catalogue: https://zhenxianli.github.io/LISQM/
    """
    if field_type not in _FIELDS_ECMA:
        raise ValueError(f"field_type must be one of {_FIELDS_ECMA}, got {field_type!r}")
    if not float(time_skip) >= _ECMA_MIN_SKIP:
        raise ValueError(f"time_skip must be at least {_ECMA_MIN_SKIP} s (ECMA-418-2:2025 6.2.9)")
    s = _session(session)
    p = pressure_48k(x, fs, calibration)
    out = s.call(
        "Tonality_ECMA418_2", s.to_matlab(p), FS_MODEL, field_type, float(time_skip), False
    )
    t, v, idx = _series(out, "timeOut", "tonalityTDep")
    freq = _arr(out["tonalityTDepFreqs"]).reshape(-1)
    extra = {
        "frequency": freq[idx] if idx.size and freq.size > idx[-1] else freq,
        "statistics": _statistics(out),
        "field_type": field_type,
        "time_skip": float(time_skip),
        "call": f"Tonality_ECMA418_2(x, 48000, '{field_type}', {time_skip:g}, false)",
    }
    if "specTonality" in out:
        extra["specific_tonality"] = _arr(out["specTonality"])
    if "bandCentreFreqs" in out:
        extra["band_centre_freqs"] = _arr(out["bandCentreFreqs"]).reshape(-1)
    return TimeMetric(
        name="tonality",
        t=t,
        values=v,
        unit="tu_HMS",
        single_value=_stat(out, "tonalityAvg"),
        skip_start=_SKIP_TONALITY,
        aggregate="mean",
        threshold=_TONALITY_THRESHOLD,
        extra=extra,
        source=f"{s.source}; ECMA-418-2:2025 (4th edition) tonality, {field_type}",
    )
