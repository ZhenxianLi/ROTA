"""Bridge to SQAT (MATLAB) through the MATLAB Engine API for Python (optional).

Two groups of tests:

* Always run, without MATLAB: the module imports without ``matlabengine``; calling it then
  raises ImportError with the install hint; a missing SQAT folder gives a clear error; and a
  stub engine (a small class that records the calls and returns fixed arrays, with a stub
  ``matlab`` module providing ``matlab.double``) checks the argument marshalling (calibrated
  Pa at 48 kHz as a column vector, explicit parameters), the conversion of SQAT's output
  struct into :class:`rota_order.bridges.TimeMetric`, and the ownership of the engine.
* With MATLAB: run only when ``matlab.engine`` is importable and ``ROTA_ORDER_SQAT_PATH`` points
  to a SQAT checkout (release v2.0). The wrappers must return exactly what the same SQAT
  call made directly through the engine returns on the same signal, and
  ``rota_order.bridges.TimeMetric.vs_rpm`` must bin the results.

Data come from ``rota_order.synth`` only. No MATLAB output is stored.
"""

import os
import subprocess
import sys
import types

import numpy as np
import pytest

import rota_order as ro
from rota_order.bridges import _common, sqat
from rota_order.bridges._common import TimeMetric

FS = 16000.0  # Hz, rate of the synthetic signal (resampled to 48 kHz by the bridge)
DURATION = 0.5  # s, stub tests
CAL = 2.0  # Pa per unit of the test signal
N_T, N_BARK = 50, 240


def _signal(duration=DURATION, fs=FS, seed=3):
    t = np.arange(round(duration * fs)) / fs
    rpm = ro.synth.rpm_profiles(t, "linear", start=2000.0, stop=4000.0)
    sig = ro.synth.synth_runup(
        fs, rpm, orders=[{"order": 6.0, "amplitude": 0.2, "phase": 0.0}], noise_rms=0.02, seed=seed
    )
    return sig


# ---- stub matlab and engine ------------------------------------------------------------------


class FakeDouble:
    """Stand-in for ``matlab.double`` (same constructor signature)."""

    def __init__(self, initializer=None, size=None, is_complex=False):
        self.data = np.array(initializer, dtype=float)
        if self.data.ndim == 1:
            self.data = self.data.reshape(1, -1)  # a flat sequence is a 1-by-n row in MATLAB
        self.size = self.data.shape

    def __array__(self, dtype=None, copy=None):
        return self.data if dtype is None else self.data.astype(dtype)


def _struct(name):
    """Fixed outputs of the SQAT functions, with the documented field names."""
    t = np.arange(N_T) * 0.002
    t_col = FakeDouble(t.reshape(-1, 1))
    if name == "Loudness_ISO532_1":
        n = 1.0 + 0.01 * np.arange(N_T)
        spec = np.outer(n, np.linspace(0.0, 1.0, N_BARK)) / 120.0
        return {"time": t_col, "InstantaneousLoudness": FakeDouble(n.reshape(-1, 1)),
                "InstantaneousSpecificLoudness": FakeDouble(spec),
                "InstantaneousLoudnessLevel": FakeDouble((40.0 + 10 * np.log2(n)).reshape(-1, 1)),
                "barkAxis": FakeDouble(np.arange(1, N_BARK + 1) * 0.1), "N5": 1.47, "Nmean": 1.25,
                "time_insig": FakeDouble(np.zeros((3, 1)))}
    if name == "Sharpness_DIN45692_from_loudness":
        return {"time": t_col, "InstantaneousSharpness": FakeDouble((1.2 + 0 * t).reshape(-1, 1)),
                "Smean": 1.2, "S50": 1.21}
    if name == "Roughness_Daniel1997":
        tr = np.arange(10) * 0.1
        return {"time": FakeDouble(tr.reshape(-1, 1)),
                "InstantaneousRoughness": FakeDouble((0.1 * np.arange(10)).reshape(-1, 1)),
                "InstantaneousSpecificRoughness": FakeDouble(np.ones((10, 47))),
                "barkAxis": FakeDouble(np.arange(47) * 0.5), "Rmean": 0.45, "R10": 0.8}
    if name == "Tonality_ECMA418_2":
        # SQAT returns the time column one longer than needed here to check the cut to the
        # shorter length; a repeated time stamp checks the strict-increase filter.
        tt = np.r_[t, t[-1] + 0.002]
        tt[10] = tt[9]
        return {"timeOut": FakeDouble(tt.reshape(-1, 1)),
                "tonalityTDep": FakeDouble((0.01 * np.arange(N_T)).reshape(-1, 1)),
                "tonalityTDepFreqs": FakeDouble((600.0 + np.arange(N_T)).reshape(-1, 1)),
                "specTonality": FakeDouble(np.zeros((N_T, 53))),
                "bandCentreFreqs": FakeDouble(np.linspace(50, 15000, 53)),
                "tonalityAvg": 0.33, "soundField": "free-frontal", "Tmean": 0.3}
    raise AssertionError(f"unexpected SQAT call {name}")


class FakeEngine:
    """Records every call; returns the documented outputs with fixed values."""

    def __init__(self, release="2025a", fail=None):
        self.calls = []
        self.quit_count = 0
        self.release = release
        self.fail = fail

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def fn(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            assert "nargout" in kwargs
            if name == self.fail:
                kwargs["stdout"].write("Error using " + name)
                raise ValueError("stub failure")
            if name in ("addpath", "startup_SQAT"):
                assert kwargs["nargout"] == 0
                return None
            if name == "version":
                return self.release
            return _struct(name)

        return fn

    def quit(self):
        self.quit_count += 1

    def named(self, name):
        return [c for c in self.calls if c[0] == name]


@pytest.fixture
def fake_matlab(monkeypatch):
    """Install stub ``matlab`` and ``matlab.engine`` modules; return the started engines."""
    started = []
    mat = types.ModuleType("matlab")
    eng = types.ModuleType("matlab.engine")
    mat.double = FakeDouble
    mat.engine = eng

    def start_matlab(*args, **kwargs):
        e = FakeEngine()
        started.append(e)
        return e

    eng.start_matlab = start_matlab
    monkeypatch.setitem(sys.modules, "matlab", mat)
    monkeypatch.setitem(sys.modules, "matlab.engine", eng)
    monkeypatch.setattr(sqat, "_DEFAULT", None)
    return started


@pytest.fixture
def sqat_dir(tmp_path):
    d = tmp_path / "SQAT"
    d.mkdir()
    (d / "startup_SQAT.m").write_text("% stub for the tests\n")
    return d


@pytest.fixture
def stub(fake_matlab, sqat_dir):
    engine = FakeEngine()
    return sqat.start(sqat_dir, engine=engine), engine


# ---- without matlabengine ----------------------------------------------------------------------


def test_import_without_matlabengine():
    code = (
        "import sys; sys.modules['matlab'] = None; sys.modules['matlab.engine'] = None; "
        "import rota_order, rota_order.bridges.sqat as s; "
        "assert sys.modules['matlab'] is None; print(s.SQAT_RELEASE)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "v2.0"


def test_call_without_matlabengine_raises_with_hint(monkeypatch, sqat_dir):
    monkeypatch.setitem(sys.modules, "matlab", None)
    monkeypatch.setitem(sys.modules, "matlab.engine", None)
    monkeypatch.setenv(sqat.ENV_PATH, str(sqat_dir))
    monkeypatch.setattr(sqat, "_DEFAULT", None)
    x = _signal().x
    for fn in (sqat.loudness, sqat.sharpness, sqat.roughness, sqat.tonality):
        with pytest.raises(ImportError) as exc:
            fn(x, FS)
        msg = str(exc.value)
        for part in ("licensed MATLAB", "Signal Processing Toolbox", "pip install matlabengine==",
                     "v2.0", "does not depend"):
            assert part in msg
    with pytest.raises(ImportError, match="matlabengine"):
        sqat.start(sqat_dir)


def test_missing_sqat_path(fake_matlab, monkeypatch, tmp_path):
    monkeypatch.delenv(sqat.ENV_PATH, raising=False)
    with pytest.raises(ValueError, match="ROTA_ORDER_SQAT_PATH"):
        sqat.start()
    with pytest.raises(ValueError, match="ROTA_ORDER_SQAT_PATH"):
        sqat.loudness(_signal().x, FS)
    with pytest.raises(FileNotFoundError, match="startup_SQAT.m"):
        sqat.start(tmp_path)
    monkeypatch.setenv(sqat.ENV_PATH, str(tmp_path))
    with pytest.raises(FileNotFoundError, match="ROTA_ORDER_SQAT_PATH"):
        sqat.start()
    assert fake_matlab == []  # no engine started before the folder is known


# ---- stub engine: session ----------------------------------------------------------------------


def test_session_initialises_sqat(stub, sqat_dir):
    s, engine = stub
    names = [c[0] for c in engine.calls]
    assert names == ["addpath", "startup_SQAT", "version"]
    assert engine.calls[0][1] == (str(sqat_dir),)
    assert engine.calls[1][1] == (str(sqat_dir) + os.sep,)  # as run_sqat.m: folder with separator
    assert engine.calls[2][1] == ("-release",)
    assert s.matlab_release == "R2025a"
    assert s.source.startswith("SQAT ") and s.source.endswith("via MATLAB R2025a")
    assert "no git metadata" in s.sqat_version


def test_engine_ownership(fake_matlab, sqat_dir):
    given = FakeEngine()
    with sqat.start(sqat_dir, engine=given) as s:
        assert s.engine is given
    assert given.quit_count == 0  # not ours to quit
    with sqat.start(sqat_dir) as s:
        assert len(fake_matlab) == 1 and s.engine is fake_matlab[0]
    assert fake_matlab[0].quit_count == 1
    assert s.engine is None


def test_session_closes_engine_when_startup_fails(fake_matlab, sqat_dir, monkeypatch):
    mod = sys.modules["matlab.engine"]
    monkeypatch.setattr(mod, "start_matlab", lambda: FakeEngine(fail="startup_SQAT"))
    with pytest.raises(RuntimeError, match="startup_SQAT failed"):
        sqat.Session(sqat_dir)


def test_default_session_is_lazy_and_reused(fake_matlab, sqat_dir, monkeypatch):
    monkeypatch.setenv(sqat.ENV_PATH, str(sqat_dir))
    assert fake_matlab == []
    x = _signal().x
    sqat.roughness(x, FS)
    sqat.roughness(x, FS)
    assert len(fake_matlab) == 1
    assert len(fake_matlab[0].named("Roughness_Daniel1997")) == 2
    sqat.close_default()
    assert fake_matlab[0].quit_count == 1 and sqat._DEFAULT is None


def test_failed_call_raises_runtime_error(fake_matlab, sqat_dir):
    s = sqat.start(sqat_dir, engine=FakeEngine(fail="Roughness_Daniel1997"))
    with pytest.raises(RuntimeError, match=r"Roughness_Daniel1997 failed: stub failure[\s\S]*Error using"):
        sqat.roughness(_signal().x, FS, session=s)


# ---- stub engine: marshalling and conversion ---------------------------------------------------


def _check_signal_arg(arg, x):
    """The signal reaches MATLAB as an (n, 1) double column: calibrated, Pa, at 48 kHz."""
    assert isinstance(arg, FakeDouble)
    expected = ro.tonal.to_calibrated_48k(x, FS, calibration=CAL, fs_out=48000.0)
    assert arg.size == (expected.size, 1)
    assert expected.size == round(x.size * 48000.0 / FS)
    np.testing.assert_array_equal(arg.data[:, 0], expected)


def test_loudness_marshalling_and_result(stub):
    s, engine = stub
    x = _signal().x
    m = sqat.loudness(x, FS, calibration=CAL, session=s)
    (_, args, kwargs), = engine.named("Loudness_ISO532_1")
    _check_signal_arg(args[0], x)
    assert args[1:] == (48000.0, 0.0, 2.0, 0.5, False)
    assert all(type(a) is float for a in args[1:5])  # no MATLAB integers
    assert kwargs["nargout"] == 1

    ref = _struct("Loudness_ISO532_1")
    assert isinstance(m, TimeMetric)
    assert (m.name, m.unit, m.skip_start, m.aggregate) == ("loudness", "sone", 0.6, ("percentile", 95))
    np.testing.assert_array_equal(m.t, ref["time"].data[:, 0])
    np.testing.assert_array_equal(m.values, ref["InstantaneousLoudness"].data[:, 0])
    assert m.values.shape == m.t.shape == (N_T,)
    assert m.single_value == 1.47  # N5
    assert m.extra["specific_loudness"].shape == (N_BARK, N_T)
    assert m.extra["statistics"] == {"N5": 1.47, "Nmean": 1.25}
    assert m.extra["loudness_level"].shape == (N_T,)
    assert "ISO 532-1:2017" in m.source and "R2025a" in m.source and m.source.startswith("SQAT")

    sqat.loudness(x, FS, calibration=CAL, field_type="diffuse", time_skip=0.2, session=s)
    assert engine.named("Loudness_ISO532_1")[1][1][1:] == (48000.0, 1.0, 2.0, 0.2, False)
    with pytest.raises(ValueError, match="field_type"):
        sqat.loudness(x, FS, field_type="free-frontal", session=s)


def test_sharpness_from_loudness(stub):
    s, engine = stub
    x = _signal().x
    n = sqat.loudness(x, FS, calibration=CAL, session=s)
    sh = sqat.sharpness(None, None, loudness=n, session=s)
    assert len(engine.named("Loudness_ISO532_1")) == 1  # loudness reused, not recomputed
    (_, args, _), = engine.named("Sharpness_DIN45692_from_loudness")
    ref = _struct("Loudness_ISO532_1")
    assert args[0].size == (N_T, N_BARK)  # back in SQAT's [time, Bark] orientation
    np.testing.assert_array_equal(args[0].data, ref["InstantaneousSpecificLoudness"].data)
    assert args[1] == "DIN45692"
    assert args[2].size == (N_T, 1)
    np.testing.assert_array_equal(args[2].data, ref["time"].data)
    assert args[3:] == (0.5, False)
    assert (sh.name, sh.unit, sh.aggregate, sh.single_value) == ("sharpness", "acum", "median", 1.21)
    assert "DIN 45692:2009" in sh.source

    sqat.sharpness(x, FS, calibration=CAL, weight_type="aures", session=s)
    (_, args2, _) = engine.named("Loudness_ISO532_1")[1]
    _check_signal_arg(args2[0], x)
    assert engine.named("Sharpness_DIN45692_from_loudness")[1][1][1] == "aures"
    with pytest.raises(ValueError, match="weight_type"):
        sqat.sharpness(x, FS, weight_type="din", session=s)
    foreign = TimeMetric("loudness", n.t, n.values, "sone", extra=dict(n.extra), source="MoSQITo 1.2.1")
    with pytest.raises(ValueError, match="rota_order.bridges.sqat.loudness"):
        sqat.sharpness(None, None, loudness=foreign, session=s)


def test_roughness_marshalling_and_result(stub):
    s, engine = stub
    x = _signal().x
    r = sqat.roughness(x, FS, calibration=CAL, session=s)
    (_, args, _), = engine.named("Roughness_Daniel1997")
    _check_signal_arg(args[0], x)
    assert args[1:] == (48000.0, 1.5, False)
    assert (r.unit, r.skip_start, r.aggregate, r.single_value) == ("asper", 1.5, ("percentile", 90), 0.8)
    np.testing.assert_array_equal(r.values, 0.1 * np.arange(10))
    assert r.extra["specific_roughness"].shape == (10, 47)


def test_tonality_marshalling_and_result(stub):
    s, engine = stub
    x = _signal().x
    to = sqat.tonality(x, FS, calibration=CAL, session=s)
    (_, args, _), = engine.named("Tonality_ECMA418_2")
    _check_signal_arg(args[0], x)
    assert args[1:] == (48000.0, "free-frontal", 0.304, False)
    assert (to.unit, to.skip_start, to.aggregate, to.threshold) == ("tu_HMS", 0.6, "mean", 0.02)
    assert to.single_value == 0.33
    # cut to the shorter length (N_T) and the repeated time stamp (index 10) dropped
    keep = np.r_[0:10, 11:N_T]
    np.testing.assert_array_equal(to.values, 0.01 * keep)
    np.testing.assert_array_equal(to.extra["frequency"], 600.0 + keep)
    assert np.all(np.diff(to.t) > 0)
    assert "soundField" not in to.extra["statistics"]
    assert "ECMA-418-2:2025 (4th edition)" in to.source
    with pytest.raises(ValueError, match="0.304"):
        sqat.tonality(x, FS, time_skip=0.1, session=s)
    with pytest.raises(ValueError, match="field_type"):
        sqat.tonality(x, FS, field_type="free", session=s)


def test_input_checks(stub):
    s, engine = stub
    x = _signal().x
    with pytest.raises(ValueError, match="mono"):
        sqat.loudness(np.c_[x, x], FS, session=s)
    with pytest.raises(ValueError, match="calibration"):
        sqat.roughness(x, FS, calibration=0.0, session=s)
    assert engine.named("Loudness_ISO532_1") == engine.named("Roughness_Daniel1997") == []


def test_matrix_fallback_to_nested_lists(stub, monkeypatch):
    # An engine whose matlab.double does not take NumPy arrays (releases before R2022a)
    # gets nested lists of rows instead.
    s, _ = stub

    def list_only(initializer=None, size=None, is_complex=False):
        if isinstance(initializer, np.ndarray):
            raise TypeError("initializer must be a rectangular sequence")
        return FakeDouble(initializer)

    monkeypatch.setattr(s, "_double", list_only)
    m = s.to_matlab(np.arange(4.0))
    assert m.size == (4, 1)
    np.testing.assert_array_equal(m.data[:, 0], np.arange(4.0))


def test_vs_rpm_on_stub_result(stub):
    s, _ = stub
    t_rpm = np.linspace(0.0, 0.2, 201)
    rpm = np.linspace(2000.0, 4000.0, 201)
    to = sqat.tonality(_signal().x, FS, session=s)
    to.skip_start = 0.0
    b = to.vs_rpm(rpm, t_rpm, bin_width=500.0, min_count=1)
    assert np.any(np.isfinite(b.value))
    # the binned result carries the name, unit and source of the SQAT metric
    assert (b.name, b.unit, b.source) == ("tonality", "tu_HMS", to.source)
    assert b.source.startswith("SQAT ") and "ECMA-418-2:2025 (4th edition)" in b.source
    # the recommended threshold of the tonality (0.02 tu_HMS) is applied: stub values are 0.01 * k
    assert b.count.sum() == np.count_nonzero((to.t <= t_rpm[-1]) & (to.values > 0.02))
    loud = sqat.loudness(_signal().x, FS, session=s)
    r = loud.vs_rpm(np.linspace(1000.0, 2000.0, 201), t_rpm, skip_start=0.0, bin_width=250.0, min_count=1)
    assert (r.name, r.unit, r.source) == ("loudness", "sone", loud.source)


def test_docstrings_cite_and_state_licence():
    doc = sqat.__doc__
    for part in ("Licence:", "GPL-3.0", "BSD-3-Clause", "MIT", "does not include, copy, modify",
                 "commercial", "10.5281/zenodo.7934709", "10.3397/IN_2023_1075",
                 "matlab-engine-for-python", "zhenxianli.github.io/LISQM"):
        assert part in doc, part
    standards = {"loudness": "ISO 532-1:2017", "sharpness": "DIN 45692:2009",
                 "roughness": "Daniel, P., & Weber, R. (1997)", "tonality": "ECMA-418-2, 4th edition"}
    for name, std in standards.items():
        d = getattr(sqat, name).__doc__
        for part in ("References", std, "10.5281/zenodo.7934709", "10.3397/IN_2023_1075",
                     "matlab-engine-for-python", "LISQM", "RMS", "Pa"):
            assert part in d, (name, part)
    for fn in (sqat.Session, sqat.start, sqat.default_session, sqat.close_default):
        assert "References" in fn.__doc__ and "Engine API" in fn.__doc__


def test_bridges_common_is_shared():
    assert sqat.TimeMetric is _common.TimeMetric


# ---- real MATLAB and SQAT ----------------------------------------------------------------------


def _real_available():
    if not os.environ.get(sqat.ENV_PATH):
        return False
    try:
        import matlab.engine  # noqa: F401
    except ImportError:
        return False
    return True


real = pytest.mark.skipif(
    not _real_available(), reason=f"needs matlab.engine (matlabengine) and {sqat.ENV_PATH}"
)

FS_REAL = 32000.0
DURATION_REAL = 3.0  # s: longer than the 1.5 s start-up of roughness


@pytest.fixture(scope="module")
def real_run():
    import matlab
    import matlab.engine

    sig = _signal(DURATION_REAL, FS_REAL, seed=5)
    # ROTA_ORDER_MATLAB_ENGINE_NAME: connect to a shared engine (matlab.engine.shareEngine) instead
    # of starting one, as the CI workflow does when only its run-command session is licensed.
    name = os.environ.get("ROTA_ORDER_MATLAB_ENGINE_NAME")
    engine = matlab.engine.connect_matlab(name) if name else None
    with sqat.start(engine=engine) as s:
        p = ro.tonal.to_calibrated_48k(sig.x, FS_REAL, calibration=1.0, fs_out=48000.0)
        col = matlab.double(p.reshape(-1, 1).tolist())  # independent route: nested lists
        eng = s.engine
        direct = {
            "loudness": eng.Loudness_ISO532_1(col, 48000.0, 0.0, 2.0, 0.5, False, nargout=1),
            "roughness": eng.Roughness_Daniel1997(col, 48000.0, 1.5, False, nargout=1),
            "tonality": eng.Tonality_ECMA418_2(col, 48000.0, "free-frontal", 0.304, False, nargout=1),
        }
        L = direct["loudness"]
        direct["sharpness"] = eng.Sharpness_DIN45692_from_loudness(
            L["InstantaneousSpecificLoudness"], "DIN45692", L["time"], 0.5, False, nargout=1
        )
        n = sqat.loudness(sig.x, FS_REAL, session=s)
        wrapped = {
            "loudness": n,
            "sharpness": sqat.sharpness(None, None, loudness=n, session=s),
            "roughness": sqat.roughness(sig.x, FS_REAL, session=s),
            "tonality": sqat.tonality(sig.x, FS_REAL, session=s),
        }
        source = s.source
    return {"sig": sig, "direct": direct, "wrapped": wrapped, "source": source}


KEYS = {
    "loudness": ("time", "InstantaneousLoudness", "N5"),
    "sharpness": ("time", "InstantaneousSharpness", "S50"),
    "roughness": ("time", "InstantaneousRoughness", "R10"),
    "tonality": ("timeOut", "tonalityTDep", "tonalityAvg"),
}


@real
@pytest.mark.parametrize("metric", list(KEYS))
def test_real_equals_direct_call(real_run, metric):
    t_key, v_key, single = KEYS[metric]
    d, w = real_run["direct"][metric], real_run["wrapped"][metric]
    t = np.asarray(d[t_key], dtype=float).reshape(-1)
    v = np.asarray(d[v_key], dtype=float).reshape(-1)
    n = min(t.size, v.size)
    assert w.t.size == w.values.size and w.t.size >= n - 2
    np.testing.assert_array_equal(w.t, t[: w.t.size])
    np.testing.assert_array_equal(w.values, v[: w.values.size])
    if single in d:
        assert w.single_value == float(np.ravel(np.asarray(d[single], dtype=float))[0])
    assert w.source.startswith(real_run["source"])
    print(metric, w.source, "n =", w.t.size, "single:", w.single_value)


@real
def test_real_values_plausible(real_run):
    w = real_run["wrapped"]
    tail = w["loudness"].t >= 0.6
    assert np.all(w["loudness"].values[tail] > 0)
    assert np.all(np.isfinite(w["tonality"].extra["frequency"]))
    assert w["tonality"].t[-1] <= DURATION_REAL + 0.01


@real
@pytest.mark.parametrize("metric", list(KEYS))
def test_real_vs_rpm(real_run, metric):
    sig = real_run["sig"]
    m = real_run["wrapped"][metric]
    b = m.vs_rpm(sig.rpm, sig.t, bin_width=250.0, min_count=1)
    assert np.count_nonzero(np.isfinite(b.value)) >= 2
    assert (b.name, b.unit, b.source) == (m.name, m.unit, m.source)
