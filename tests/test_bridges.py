"""rota_order.bridges: lazy optional wrappers for MoSQITo and the Sottek hearing model.

Tests without the toolboxes always run (they simulate the missing packages). Tests with a toolbox
are skipped when it is not installed (every one of them takes the ``mq`` or the ``sh`` fixture,
directly or through a fixture built on it). Data come from rota_order.synth only. The models are slow, so
the signals are short and the results are computed once per module.
"""

import dataclasses
import importlib
import inspect
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

import rota_order as ro
from rota_order import bridges
from rota_order.bridges import TimeMetric, mosqito
from rota_order.bridges import sottek_hearing_model as shm
from rota_order.slices import MetricVsRpm, metric_vs_rpm

FS = 48000.0
RMS_PA = 0.1  # 71 dB re 20 uPa
FRAME = 256 / FS  # frame step of the Sottek hearing model tonality and loudness, s (5.33 ms)

# Modules that importing rota_order and rota_order.bridges must not load: the bridges import the toolboxes, and
# Matplotlib that MoSQITo and the Sottek hearing model need, lazily inside their functions.
LAZY_MODULES = ("matplotlib", "mosqito", "sottek_hearing_model", "matlab")

# Names that 0.6.0 never released. They are written in parts so that a search of the repository
# for the old spelling finds only what still needs to be changed.
REMOVED_FUNCTION = "along" + "_rpm"  # became the method TimeMetric.vs_rpm
REMOVED_MODULE = "s" + "hm"  # became rota_order.bridges.sottek_hearing_model


def _signal(fs, duration, seed=1):
    s = ro.synth.synth_tone_noise(fs, duration, RMS_PA, 6.0, tone_freq=500.0, seed=seed)
    # amplitude-modulated so that roughness and loudness are not constant
    return s.x * (1.0 + 0.5 * np.sin(2 * np.pi * 70.0 * s.t))


def _fake(name="loudness", t_end=1.0, dt=0.1, unit="sone", **kw):
    t = np.arange(0.0, t_end + 1e-9, dt)
    return TimeMetric(name=name, t=t, values=1.0 + t, unit=unit, **kw)


def _is_this_file(filename):
    return pathlib.Path(filename).resolve() == pathlib.Path(__file__).resolve()


def _assert_same_binning(a, b):
    """Every field of two MetricVsRpm except the provenance (name, unit, source) is equal."""
    for f in dataclasses.fields(MetricVsRpm):
        if f.name in ("name", "unit", "source"):
            continue
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, np.ndarray):
            np.testing.assert_array_equal(va, vb, err_msg=f.name)
        else:
            assert va == vb, f.name


# ---- without the toolboxes (always run) ---------------------------------------------------------------


def test_import_without_toolboxes_in_subprocess():
    code = (
        "import sys\n"
        "for m in ('mosqito', 'sottek_hearing_model'):\n"
        "    sys.modules[m] = None\n"  # None blocks the import; it is not a loaded module
        "import rota_order, rota_order.bridges\n"
        f"loaded = [m for m in {LAZY_MODULES!r} if sys.modules.get(m) is not None]\n"
        "assert not loaded, f'imported by rota_order.bridges: {loaded}'\n"
        "for f in (rota_order.bridges.mosqito.loudness, rota_order.bridges.sottek_hearing_model.tonality):\n"
        "    try:\n"
        "        f([0.0, 1.0, 0.0, 1.0], 48000.0)\n"
        "    except ImportError as e:\n"
        "        assert 'pip install' in str(e)\n"
        "    else:\n"
        "        raise SystemExit('no ImportError')\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_import_does_not_import_matplotlib_or_toolboxes_in_subprocess():
    # a clean interpreter, with or without the toolboxes installed: nothing is loaded before a call
    code = (
        "import sys, rota_order, rota_order.bridges\n"
        f"loaded = [m for m in {LAZY_MODULES!r} if m in sys.modules]\n"
        "assert not loaded, f'imported by rota_order.bridges: {loaded}'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


def test_bridge_modules_are_named_like_the_toolboxes():
    assert bridges.__all__ == ["TimeMetric", "mosqito", "sottek_hearing_model", "sqat"]
    for name in ("mosqito", "sottek_hearing_model", "sqat"):
        assert getattr(bridges, name).__name__ == f"rota_order.bridges.{name}"
    assert bridges.TimeMetric is TimeMetric


def test_removed_names_are_gone_without_aliases():
    # 0.6.0 is not released yet: the names left out of the API are not kept as aliases
    assert not hasattr(bridges, REMOVED_FUNCTION)
    assert not hasattr(bridges, REMOVED_MODULE)
    assert not hasattr(bridges._common, REMOVED_FUNCTION)
    assert REMOVED_FUNCTION not in bridges._common.__all__
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"rota_order.bridges.{REMOVED_MODULE}")
    assert callable(TimeMetric.vs_rpm)


def test_sources_of_the_bridges_use_the_final_names():
    root = pathlib.Path(bridges.__file__).parent
    paths = sorted(root.glob("*.py"))
    assert {p.name for p in paths} >= {"__init__.py", "_common.py", "mosqito.py", "sottek_hearing_model.py", "sqat.py"}
    assert not (root / f"{REMOVED_MODULE}.py").exists()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert REMOVED_FUNCTION not in text, path.name
        assert f"bridges.{REMOVED_MODULE}" not in text, path.name


MOSQITO_CALLS = [
    lambda x: mosqito.loudness(x, FS),
    lambda x: mosqito.sharpness(x, FS),
    lambda x: mosqito.roughness(x, FS),
]
SHM_CALLS = [
    lambda x: shm.tonality(x, FS),
    lambda x: shm.loudness(x, FS),
    lambda x: shm.roughness(x, FS),
]


@pytest.mark.parametrize("call", MOSQITO_CALLS)
def test_mosqito_missing_raises_import_error(call, monkeypatch):
    monkeypatch.setitem(sys.modules, "mosqito", None)
    with pytest.raises(ImportError, match="pip install mosqito") as e:
        call(np.zeros(4800))
    assert "does not depend" in str(e.value)
    assert "rota_order.bridges.mosqito" in str(e.value)


@pytest.mark.parametrize("call", SHM_CALLS)
def test_shm_missing_raises_import_error(call, monkeypatch):
    monkeypatch.setitem(sys.modules, "sottek_hearing_model", None)
    with pytest.raises(ImportError, match="pip install sottek-hearing-model") as e:
        call(np.zeros(4800))
    assert "does not depend" in str(e.value) and "3.11" in str(e.value)
    assert "rota_order.bridges.sottek_hearing_model" in str(e.value)


def test_shm_bridge_imports_the_top_level_package_not_itself(tmp_path):
    # Always runs, with or without the real package: a stub package named sottek_hearing_model on
    # sys.path is what the bridge of the same name must find, never the bridge module itself.
    pkg = tmp_path / "sottek_hearing_model"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("MARKER = 'external stub'\n", encoding="utf-8")
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "from rota_order.bridges import sottek_hearing_model as bridge\n"
        "assert bridge.__name__ == 'rota_order.bridges.sottek_hearing_model'\n"
        "model = bridge._model()\n"
        "assert model is not bridge and model.__name__ == 'sottek_hearing_model', model\n"
        "assert model.MARKER == 'external stub' and not hasattr(bridge, 'MARKER')\n"
        "assert sys.modules['sottek_hearing_model'] is model\n"
        "assert sys.modules['rota_order.bridges.sottek_hearing_model'] is bridge\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


@pytest.mark.parametrize("call", MOSQITO_CALLS + SHM_CALLS)
def test_multichannel_rejected(call):
    with pytest.raises(ValueError, match="mono"):
        call(np.zeros((4800, 2)))


def test_invalid_options():
    x = np.zeros(4800)
    with pytest.raises(ValueError, match="field_type"):
        mosqito.loudness(x, FS, field_type="room")
    with pytest.raises(ValueError, match="weighting"):
        mosqito.sharpness(x, FS, weighting="x")
    with pytest.raises(ValueError, match="soundfield"):
        shm.tonality(x, FS, soundfield="x")
    with pytest.raises(ValueError, match="calibration"):
        mosqito.loudness(x, FS, calibration=0.0)
    with pytest.raises(ValueError, match="loudness result"):
        mosqito.sharpness(None, None)
    with pytest.raises(ValueError, match="TimeMetric"):
        mosqito.sharpness(x, FS, loudness=_fake())
    with pytest.raises(ValueError, match=r"TimeMetric returned by rota_order\.bridges\.sottek_hearing_model\.tonality"):
        shm.loudness(x, FS, tonality=_fake())


def test_vs_rpm_equals_metric_vs_rpm_and_uses_recommendations():
    m = _fake(skip_start=0.3, aggregate=("percentile", 90))
    t_rpm = np.arange(0.0, 1.0 + 1e-9, 0.01)
    rpm = 1000.0 + 1000.0 * t_rpm
    a = m.vs_rpm(rpm, t_rpm, bin_width=250.0, min_count=1)
    b = metric_vs_rpm(m.values, m.t, rpm, t_rpm, bin_width=250.0, min_count=1, skip_start=0.3,
                      aggregate=("percentile", 90))
    np.testing.assert_array_equal(a.value, b.value)
    np.testing.assert_array_equal(a.count, b.count)
    _assert_same_binning(a, b)
    c = m.vs_rpm(rpm, t_rpm, bin_width=250.0, min_count=1, skip_start=0.0, aggregate="mean")
    assert c.count.sum() > a.count.sum()  # kwargs override


def test_vs_rpm_result_carries_name_unit_and_source_of_the_metric():
    m = _fake(name="tonality", unit="tu_HMS", source="Toolbox 1.2.3; standard, edition")
    t_rpm = np.arange(0.0, 1.0 + 1e-9, 0.01)
    rpm = 1000.0 + 1000.0 * t_rpm
    r = m.vs_rpm(rpm, t_rpm, bin_width=250.0, min_count=1)
    assert isinstance(r, MetricVsRpm)
    assert (r.name, r.unit, r.source) == ("tonality", "tu_HMS", "Toolbox 1.2.3; standard, edition")
    assert (r.name, r.unit, r.source) == (m.name, m.unit, m.source)
    # the provenance is the only difference from the bare function ...
    d = metric_vs_rpm(m.values, m.t, rpm, t_rpm, bin_width=250.0, min_count=1)
    _assert_same_binning(r, d)
    # ... which leaves it empty, since it receives arrays and no metric
    assert (d.name, d.unit, d.source) == ("", "", "")
    # a metric without a source gives an empty source, not an error
    assert _fake().vs_rpm(rpm, t_rpm, bin_width=250.0, min_count=1).source == ""


def test_metric_vs_rpm_result_gained_optional_fields_at_the_end():
    fields = dataclasses.fields(MetricVsRpm)
    assert [f.name for f in fields[-3:]] == ["name", "unit", "source"]
    assert all(f.default == "" for f in fields[-3:])
    for f in fields[:-3]:  # the released fields have no default, as before
        assert f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING, f.name
    # constructing the released way (all earlier fields by keyword) still works
    kw = {f.name: None for f in fields[:-3]}
    r = MetricVsRpm(**kw)
    assert (r.name, r.unit, r.source) == ("", "", "")
    r = MetricVsRpm(**kw, name="loudness", unit="sone", source="x")
    assert (r.name, r.unit, r.source) == ("loudness", "sone", "x")


def test_vs_rpm_threshold():
    m = _fake(threshold=1.5)  # values 1 + t: frames with t <= 0.5 are discarded
    t_rpm = np.array([0.0, 1.0])
    rpm = np.array([1000.0, 2000.0])
    one = {"segment": "all", "bins": [0.0, 1e6], "min_count": 1}
    r = m.vs_rpm(rpm, t_rpm, **one)
    assert r.count[0] == 5 and np.isclose(r.value[0], np.mean(1.0 + np.arange(0.6, 1.0 + 1e-9, 0.1)))
    # the keyword replaces the threshold of the metric; NaN means none
    assert m.vs_rpm(rpm, t_rpm, threshold=np.nan, **one).count[0] == m.t.size
    r_high = m.vs_rpm(rpm, t_rpm, threshold=1.95, **one)
    assert r_high.count[0] == 1 and np.isclose(r_high.value[0], 2.0)
    # the metric itself is not modified
    assert m.threshold == 1.5 and np.array_equal(m.values, 1.0 + m.t)


def test_vs_rpm_threshold_none_means_no_threshold():
    t_rpm = np.array([0.0, 1.0])
    rpm = np.array([1000.0, 2000.0])
    one = {"segment": "all", "bins": [0.0, 1e6], "min_count": 1}
    m = _fake(threshold=1.5)
    assert m.vs_rpm(rpm, t_rpm, **one).count[0] == 5  # the threshold of the metric
    r = m.vs_rpm(rpm, t_rpm, threshold=None, **one)  # the keyword replaces it: none
    assert r.count[0] == m.t.size and np.isclose(r.value[0], m.values.mean())
    # the same result as NaN, the other way to say none
    np.testing.assert_array_equal(r.value, m.vs_rpm(rpm, t_rpm, threshold=np.nan, **one).value)
    # a metric made with threshold=None has none, too
    n = _fake(threshold=None)
    assert n.vs_rpm(rpm, t_rpm, **one).count[0] == n.t.size
    assert n.vs_rpm(rpm, t_rpm, threshold=1.5, **one).count[0] == 5
    # something that is no number is an error that names the argument
    with pytest.raises(ValueError, match="threshold must be a number"):
        m.vs_rpm(rpm, t_rpm, threshold="high", **one)


def test_vs_rpm_says_when_every_frame_is_at_or_below_the_threshold():
    t_rpm = np.array([0.0, 1.0])
    rpm = np.array([1000.0, 2000.0])
    one = {"segment": "all", "bins": [0.0, 1e6], "min_count": 1}
    m = _fake(name="tonality", unit="tu_HMS", threshold=5.0)  # the values 1 + t reach 2
    with pytest.raises(ValueError) as e:
        m.vs_rpm(rpm, t_rpm, **one)
    msg = str(e.value)
    assert msg.startswith("every tonality frame is at or below the threshold of 5 tu_HMS")
    assert "largest of 11 frames is 2)" in msg and "threshold=None" in msg
    assert "skip_start" not in msg  # not a problem of the start-up exclusion or the time ranges
    assert m.vs_rpm(rpm, t_rpm, threshold=None, **one).count[0] == m.t.size  # what the message suggests
    assert m.vs_rpm(rpm, t_rpm, threshold=1.5, **one).count[0] == 5
    with pytest.raises(ValueError, match="at or below the threshold of 2 "):  # at the largest value: not above
        m.vs_rpm(rpm, t_rpm, threshold=2.0, **one)


def test_vs_rpm_threshold_error_names_the_start_up_exclusion():
    # a start-up transient above the threshold; everything after the start-up exclusion is below it
    t = np.arange(0.0, 1.0 + 1e-9, 0.1)
    values = np.where(t < 0.25, 0.5, 0.01)
    m = TimeMetric("tonality", t, values, "tu_HMS", skip_start=0.25, threshold=0.02)
    t_rpm = np.array([0.0, 1.0])
    rpm = np.array([1000.0, 2000.0])
    one = {"segment": "all", "bins": [0.0, 1e6], "min_count": 1}
    with pytest.raises(ValueError) as e:
        m.vs_rpm(rpm, t_rpm, **one)
    msg = str(e.value)
    assert msg.startswith("every tonality frame after the start-up exclusion (skip_start = 0.25 s) is at or below")
    assert "the threshold of 0.02 tu_HMS" in msg and "largest of 8 frames is 0.01)" in msg
    # the start-up frames (0.0, 0.1 and 0.2 s) are the only ones above the threshold: they are used
    # when the exclusion is switched off; without the threshold the 8 frames after the exclusion are
    assert m.vs_rpm(rpm, t_rpm, skip_start=0.0, **one).count[0] == 3
    assert m.vs_rpm(rpm, t_rpm, threshold=None, **one).count[0] == 8


def test_vs_rpm_other_empty_results_keep_the_general_message():
    m = _fake(threshold=1.5)
    one = {"segment": "all", "bins": [0.0, 1e6], "min_count": 1}
    t_rpm = np.array([0.0, 1.0])
    rpm = np.array([1000.0, 2000.0])
    with pytest.raises(ValueError, match="no metric sample remains"):  # start-up longer than the record
        m.vs_rpm(rpm, t_rpm, skip_start=5.0, **one)
    # no frame lies in the speed record: the warning says why before the error
    with (
        pytest.warns(UserWarning, match="outside the speed record"),
        pytest.raises(ValueError, match="no metric sample remains"),
    ):
        m.vs_rpm(rpm, t_rpm + 10.0, **one)
    with pytest.raises(ValueError, match="skip_start must be non-negative"):  # invalid arguments: as metric_vs_rpm
        m.vs_rpm(rpm, t_rpm, skip_start=-1.0, **one)
    with pytest.raises(ValueError, match="t_rpm must be finite and strictly increasing"):
        m.vs_rpm(rpm, t_rpm[::-1], **one)
    with pytest.raises(ValueError, match="same length"):
        m.vs_rpm(rpm[:1], t_rpm, **one)


def test_vs_rpm_holds_speed_for_last_frame_one_sample_past_end():
    fs = 1000.0
    t_rpm = np.arange(1000) / fs  # last speed sample at 0.999 s
    rpm = 1000.0 + 1000.0 * t_rpm
    m = _fake(t_end=1.0, dt=0.25)  # last frame at 1.0 s: one sample past the record
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = m.vs_rpm(rpm, t_rpm, segment="all", bins=[0.0, 1e6], min_count=1)
    assert r.count[0] == m.t.size
    assert np.isclose(r.value[0], m.values.mean())


def _sottek_like(n, skip_start=0.3):
    """Speed record of n samples at 48 kHz and a tonality-like metric on it.

    The frames are 256 samples apart, from t = 0 to the length rounded up to a multiple of 256
    samples, as the Sottek hearing model puts them; the values are 1 + t, above the threshold.
    """
    t_rpm = np.arange(n) / FS
    rpm = 1000.0 + 1000.0 * t_rpm
    t = np.arange(-(-n // 256) + 1) * FRAME  # frames 0 ... ceil(n / 256)
    m = TimeMetric("tonality", t, 1.0 + t, "tu_HMS", skip_start=skip_start, threshold=0.02)
    return m, rpm, t_rpm


@pytest.mark.parametrize(
    "n",
    [
        187 * 256,  # a multiple of 256: the last frame is one sample after the last sample
        187 * 256 + 1,  # the last frame is one frame step (5.33 ms) after the last sample
        187 * 256 + 2,  # 255 samples after
        188 * 256 - 1,  # 2 samples after
        48000,  # 1 s: the last frame is 129 samples (2.7 ms) after
        120000,  # 2.5 s
        144000,  # 3 s
        240000,  # 5 s
        350400,  # 7.3 s
    ],
    ids=["256k", "256k+1", "256k+2", "256k-1", "1s", "2.5s", "3s", "5s", "7.3s"],
)
def test_vs_rpm_holds_speed_for_the_last_frame_of_sottek_like_frames(n):
    m, rpm, t_rpm = _sottek_like(n)
    behind = m.t[-1] - t_rpm[-1]
    assert 0.0 < behind <= FRAME * (1.0 + 1e-9)  # past the speed record, by at most one frame step
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = m.vs_rpm(rpm, t_rpm, segment="all", bins=[0.0, 1e6], min_count=1)
    used = m.t >= m.skip_start
    assert r.count[0] == np.count_nonzero(used) and r.used[used].all()  # all frames used, the last included
    assert r.control[-1] == rpm[-1]  # the last speed is held for it
    assert t_rpm.size == rpm.size == n  # the arrays passed in are not extended
    assert np.isclose(r.value[0], m.values[used].mean(), rtol=1e-12, atol=0.0)


def test_vs_rpm_reach_after_the_record_is_speed_interval_plus_frame_step():
    one = {"segment": "all", "bins": [0.0, 1e6], "min_count": 1}
    # frames 0.1 s apart, speed every 1 ms: the reach is 0.101 s. The last frame lies 51 ms after the
    # last speed sample, more than the speed interval, and still gets the last speed.
    t_rpm = np.arange(1000) / 1000.0  # last speed sample at 0.999 s
    rpm = 1000.0 + 1000.0 * t_rpm
    t = 0.05 + 0.1 * np.arange(11)  # 0.05 ... 1.05 s
    m = TimeMetric("roughness", t, 1.0 + t, "asper")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = m.vs_rpm(rpm, t_rpm, **one)
    assert r.count[0] == t.size and r.control[-1] == rpm[-1]
    # a frame 151 ms after the last speed sample is beyond the reach: left out
    t = np.append(t, 1.15)
    with pytest.warns(UserWarning, match=r"^1 roughness frame\(s\) lie outside the speed record"):
        r = TimeMetric("roughness", t, 1.0 + t, "asper").vs_rpm(rpm, t_rpm, **one)
    assert r.count[0] == t.size - 1
    # frames 5 ms apart, speed every 0.1 s: the reach is 0.105 s, so frames up to 1.005 s are held
    t_rpm = np.arange(0.0, 1.0, 0.1)  # last speed sample at 0.9 s
    t = np.arange(0.0, 1.1 + 1e-9, 0.005)
    with pytest.warns(UserWarning, match=r"^19 roughness frame\(s\) lie outside the speed record") as rec:
        r = TimeMetric("roughness", t, 1.0 + t, "asper").vs_rpm(1000.0 + t_rpm, t_rpm, **one)
    assert "(0 before 0 s, 19 after 1.005 s)" in str(rec[0].message)
    assert r.count[0] == np.count_nonzero(t <= 1.005 + 1e-9)


def test_vs_rpm_holds_speed_for_a_coarse_speed_record():
    # the speed every 10 ms (np.arange(0, T, 0.01)) ends one speed interval before the end of the
    # signal; the Sottek-like last frame is up to one frame step after that end
    one = {"segment": "all", "bins": [0.0, 1e6], "min_count": 1}
    for extra in (0, 1, 7, 100, 333):  # signals of 1.5 s plus a few samples
        n = 72000 + extra
        t_rpm = np.arange(0.0, n / FS, 0.01)
        rpm = 1000.0 + 1000.0 * t_rpm
        t = np.arange(-(-n // 256) + 1) * FRAME
        m = TimeMetric("tonality", t, 1.0 + t, "tu_HMS", skip_start=0.3, threshold=0.02)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = m.vs_rpm(rpm, t_rpm, **one)
        assert r.count[0] == np.count_nonzero(t >= 0.3) and r.control[-1] == rpm[-1], extra


def test_vs_rpm_leaves_out_frames_more_than_a_frame_step_after_the_record():
    m, rpm, t_rpm = _sottek_like(187 * 256 + 1)  # the last frame is one frame step after the record
    t = np.append(m.t, m.t[-1] + FRAME)  # one more frame: two frame steps after
    late = TimeMetric("tonality", t, 1.0 + t, "tu_HMS", skip_start=0.3, threshold=0.02)
    with pytest.warns(UserWarning, match="outside the speed record") as rec:
        r = late.vs_rpm(rpm, t_rpm, segment="all", bins=[0.0, 1e6], min_count=1)
    assert len(rec) == 1 and str(rec[0].message).startswith("1 tonality frame(s) lie outside the speed record")
    assert "(0 before 0 s, 1 after 1.00267 s)" in str(rec[0].message)  # the held frame ends the record
    assert r.count[0] == np.count_nonzero(m.t >= 0.3) and not r.used[-1]


def test_vs_rpm_warns_for_frames_far_beyond_record():
    t_rpm = np.arange(0.0, 0.36, 0.01)  # ends at 0.35 s; the reach is 0.01 + 0.1 s, to 0.46 s
    m = _fake(t_end=1.0, dt=0.1)
    with pytest.warns(UserWarning, match="outside the speed record") as rec:
        r = m.vs_rpm(1000.0 + 0.0 * t_rpm + t_rpm, t_rpm, segment="all", bins=[0.0, 1e6], min_count=1)
    assert r.count[0] == np.count_nonzero(m.t <= 0.4)  # 0.4 s is held (0.05 s past 0.35 s), 0.5 s is not
    assert len(rec) == 1
    assert str(rec[0].message).startswith("6 loudness frame(s) lie outside the speed record")
    assert "(0 before" in str(rec[0].message) and "6 after" in str(rec[0].message)
    assert _is_this_file(rec[0].filename)  # attributed to the caller of the method


def test_vs_rpm_warns_for_frames_before_the_record():
    t_rpm = np.linspace(0.25, 1.0, 76)
    m = _fake(t_end=1.0, dt=0.1)  # frames at 0.0, 0.1 and 0.2 s precede the speed record
    with pytest.warns(UserWarning, match=r"^3 loudness frame\(s\) lie outside the speed record") as rec:
        r = m.vs_rpm(1000.0 + t_rpm, t_rpm, segment="all", bins=[0.0, 1e6], min_count=1)
    assert "3 before 0.25 s" in str(rec[0].message) and "0 after" in str(rec[0].message)
    assert r.count[0] == 8 and _is_this_file(rec[0].filename)


def test_vs_rpm_warning_counts_only_frames_that_would_otherwise_be_used():
    one = {"segment": "all", "bins": [0.0, 1e6], "min_count": 1}
    # the threshold: the values 1 - t are above 0.35 up to 0.6 s; the speed record ends at 0.45 s and
    # the frame at 0.5 s is held, so 0.6 s is the only frame after it that would be used
    t = np.arange(0.0, 1.0 + 1e-9, 0.1)
    t_rpm = np.arange(0.0, 0.46, 0.01)
    rpm = 1000.0 + t_rpm
    m = TimeMetric("tonality", t, 1.0 - t, "tu_HMS", threshold=0.35)
    with pytest.warns(UserWarning) as rec:
        r = m.vs_rpm(rpm, t_rpm, **one)
    assert len(rec) == 1 and _is_this_file(rec[0].filename)
    assert str(rec[0].message) == (
        "1 tonality frame(s) lie outside the speed record (0 before 0 s, 1 after 0.5 s) and are left out"
    )
    assert r.count[0] == 6  # 0.0 ... 0.5 s
    with pytest.warns(UserWarning, match=r"^5 tonality frame\(s\) lie outside the speed record"):
        m.vs_rpm(rpm, t_rpm, threshold=None, **one)  # without the threshold, 0.6 ... 1.0 s count
    # the start-up exclusion: the frames at 0.0, 0.1 and 0.2 s precede the speed record that starts at
    # 0.25 s; they are counted only if they would be used
    t_rpm = np.linspace(0.25, 1.0, 76)
    m = _fake(t_end=1.0, dt=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = m.vs_rpm(1000.0 + t_rpm, t_rpm, skip_start=0.3, **one)
    assert r.count[0] == 8  # 0.3 ... 1.0 s
    with pytest.warns(UserWarning) as rec:
        r = m.vs_rpm(1000.0 + t_rpm, t_rpm, skip_start=0.15, **one)  # 0.0 and 0.1 s are excluded, 0.2 s is not
    assert str(rec[0].message).startswith(
        "1 loudness frame(s) lie outside the speed record (1 before 0.25 s, 0 after 1 s)"
    )
    assert r.count[0] == 8
    # frames without a value are no loss either
    values = 1.0 + m.t
    values[:3] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = TimeMetric("loudness", m.t, values, "sone").vs_rpm(1000.0 + t_rpm, t_rpm, **one)
    assert r.count[0] == 8


def test_vs_rpm_warning_counts_frames_lost_to_nan_speeds():
    one = {"segment": "all", "bins": [0.0, 1e6], "min_count": 1}
    t_rpm = np.arange(0.0, 1.0 + 1e-9, 0.01)  # 0 ... 1 s
    rpm = 1000.0 + 1000.0 * t_rpm
    m = _fake()  # frames every 0.1 s, 0 ... 1 s
    # NaN inside the record is bridged by interpolation: nothing is lost
    inside = rpm.copy()
    inside[30:50] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert m.vs_rpm(inside, t_rpm, **one).count[0] == m.t.size
    # NaN at the end: the last finite sample is at 0.65 s, so the frames at 0.7 ... 1.0 s have no speed
    # (they lie inside the record, which ends at 1 s; the NaN last sample is not held)
    tail = rpm.copy()
    tail[66:] = np.nan
    with pytest.warns(UserWarning) as rec:
        r = m.vs_rpm(tail, t_rpm, **one)
    assert len(rec) == 1 and _is_this_file(rec[0].filename)
    assert str(rec[0].message) == (
        "4 loudness frame(s) lie where the speed is NaN (0 before its first finite sample at 0 s, "
        "4 after its last at 0.65 s) and are left out"
    )
    assert r.count[0] == 7  # 0.0 ... 0.6 s
    # NaN at the start as well: the frame at 0 s has no speed either
    both_ends = tail.copy()
    both_ends[:5] = np.nan
    with pytest.warns(UserWarning) as rec:
        r = m.vs_rpm(both_ends, t_rpm, **one)
    assert str(rec[0].message) == (
        "5 loudness frame(s) lie where the speed is NaN (1 before its first finite sample at 0.05 s, "
        "4 after its last at 0.65 s) and are left out"
    )
    assert r.count[0] == 6  # 0.1 ... 0.6 s


def test_vs_rpm_warning_names_both_reasons_when_frames_are_lost_twice():
    one = {"segment": "all", "bins": [0.0, 1e6], "min_count": 1}
    t_rpm = np.arange(0.0, 0.5, 0.01)  # 0 ... 0.49 s
    rpm = 1000.0 + t_rpm
    rpm[40:] = np.nan  # finite up to 0.39 s; the last sample is NaN, so no speed is held after it
    m = _fake()  # frames every 0.1 s, 0 ... 1 s
    with pytest.warns(UserWarning) as rec:
        r = m.vs_rpm(rpm, t_rpm, **one)
    assert len(rec) == 1
    assert str(rec[0].message) == (
        "6 loudness frame(s) lie outside the speed record (0 before 0 s, 6 after 0.49 s) and are left out; "
        "1 loudness frame(s) lie where the speed is NaN (0 before its first finite sample at 0 s, "
        "1 after its last at 0.39 s) and are left out"
    )
    assert r.count[0] == 4  # 0.0 ... 0.3 s


def test_public_docstrings_state_references_units_and_amplitude_convention():
    units = {
        mosqito.loudness: "sone", mosqito.sharpness: "acum", mosqito.roughness: "asper",
        shm.tonality: "tu_HMS", shm.loudness: "sone_HMS", shm.roughness: "asper_HMS",
    }
    assert set(mosqito.__all__) | set(shm.__all__) == {"loudness", "sharpness", "roughness", "tonality"}

    def text(obj):  # the docstring with its line breaks and indentation folded to single spaces
        return " ".join(inspect.getdoc(obj).split())

    for fn, unit in units.items():
        doc = text(fn)
        for part in ("References", "LISQM", "Pa", unit, "RMS", "Parameters", "Returns", "Raises"):
            assert part in doc, (fn.__module__, fn.__name__, part)
    doc = text(TimeMetric)
    for part in ("References", "LISQM", "no peak or RMS convention applies", "not an amplitude", "vs_rpm"):
        assert part in doc, part
    doc = text(TimeMetric.vs_rpm)
    for part in ("References", "LISQM", "Brandt", "Units.", "not an amplitude", "no peak or RMS convention applies",
                 "rpm", "metric_vs_rpm(m.values, m.t, rpm, t_rpm", "Provenance.", "name", "unit", "source",
                 "UserWarning", "Parameters", "Returns", "Warns", "Raises",
                 # the hold of the last speed: how far, and for which frames
                 "Frames beyond the speed record.", "256 samples", "rounded up to a multiple of 256",
                 "sum of the last speed sample interval", "and the frame step of the metric",
                 "would otherwise be used", "None or NaN"):
        assert part in doc, part
    # the first wording of the hold was wrong for most lengths of the signal
    for gone in ("one speed sample interval", "at the duration of the record", "one sample after the last speed"):
        assert gone not in doc, gone
    assert REMOVED_FUNCTION not in doc
    doc = text(shm.tonality)
    assert "256 samples" in doc and "multiple of 256" in doc and "holding the last speed" in doc
    assert "one sample past the end of the signal" not in doc
    assert "NaN (or None) means no threshold" in text(TimeMetric)
    doc = text(MetricVsRpm)
    for part in ("name : str", "unit : str", "source : str", "TimeMetric.vs_rpm", "Empty"):
        assert part in doc, part


# ---- MoSQITo ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mq():
    pytest.importorskip("matplotlib")  # imported by mosqito, not declared by it
    return pytest.importorskip("mosqito.sq_metrics")


@pytest.fixture(scope="module")
def x48():
    return _signal(FS, 2.0)


@pytest.fixture(scope="module")
def mq_results(mq, x48):
    loud = mosqito.loudness(x48, FS)
    return {"loud": loud, "sharp": mosqito.sharpness(None, None, loudness=loud),
            "rough": mosqito.roughness(x48, FS)}


def test_mosqito_wrappers_equal_direct_calls(mq, x48, mq_results):
    n, n_spec, _bark, t = mq.loudness_zwtv(x48, FS, field_type="free")
    s = mq.sharpness_din_from_loudness(n, n_spec, weighting="din")
    r, _r_spec, _bark_r, t_r = mq.roughness_dw(x48, FS, overlap=0.5)
    loud, sharp, rough = mq_results["loud"], mq_results["sharp"], mq_results["rough"]
    np.testing.assert_array_equal(loud.values, n)
    np.testing.assert_array_equal(loud.t, t)
    np.testing.assert_array_equal(loud.extra["specific_loudness"], n_spec)
    np.testing.assert_array_equal(sharp.values, s)
    np.testing.assert_array_equal(rough.values, r)
    np.testing.assert_array_equal(rough.t, t_r)
    assert (loud.unit, sharp.unit, rough.unit) == ("sone", "acum", "asper")
    assert np.isnan(loud.single_value) and np.isnan(rough.single_value)
    assert loud.source.startswith("MoSQITo ") and "ISO 532-1:2017" in loud.source
    assert "DIN 45692:2009" in sharp.source
    assert np.isclose(rough.t[0], 0.1, atol=1e-3)  # the roughness time axis starts at 0.1 s
    assert (loud.skip_start, sharp.skip_start, rough.skip_start) == (0.6, 0.6, 1.5)


def test_mosqito_sharpness_from_signal_equals_from_loudness(mq_results, x48):
    s = mosqito.sharpness(x48, FS)
    np.testing.assert_array_equal(s.values, mq_results["sharp"].values)


def test_mosqito_resampling_path(mq):
    fs = 32000.0
    x = _signal(fs, 1.5, seed=2)
    x48 = ro.tonal.to_calibrated_48k(x, fs)
    n, *_ = mq.loudness_zwtv(x48, FS, field_type="free")
    loud = mosqito.loudness(x, fs)
    np.testing.assert_array_equal(loud.values, n)
    assert loud.values.size > 0 and np.all(np.isfinite(loud.values))


def test_mosqito_calibration_scales_loudness(mq, x48):  # mq: skip without MoSQITo
    a = mosqito.loudness(0.5 * x48, FS)
    b = mosqito.loudness(x48, FS)
    c = mosqito.loudness(x48, FS, calibration=0.5)
    d = mosqito.loudness(x48, FS, calibration=2.0)
    np.testing.assert_allclose(c.values, a.values, rtol=1e-12)
    keep = b.t > 0.6
    assert np.all(c.values[keep] < b.values[keep]) and np.all(d.values[keep] > b.values[keep])


def test_mosqito_vs_rpm_matches_hand_call(mq_results, x48):
    t = np.arange(x48.size) / FS
    rpm = 1000.0 + 500.0 * t
    loud = mq_results["loud"]
    a = loud.vs_rpm(rpm, t, bins=np.arange(1000.0, 2001.0, 250.0), min_count=3)
    b = metric_vs_rpm(loud.values, loud.t, rpm, t, bins=np.arange(1000.0, 2001.0, 250.0), min_count=3,
                      skip_start=0.6, aggregate=("percentile", 95))
    np.testing.assert_array_equal(a.value, b.value)
    assert a.valid.any()
    assert (a.name, a.unit, a.source) == ("loudness", "sone", loud.source)
    assert a.source.startswith("MoSQITo ") and "ISO 532-1:2017" in a.source
    assert (b.name, b.unit, b.source) == ("", "", "")


# ---- Sottek hearing model ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sh():
    pytest.importorskip("matplotlib")
    return pytest.importorskip("sottek_hearing_model")


@pytest.fixture(scope="module")
def x_shm():
    return _signal(FS, 1.2, seed=3)


@pytest.fixture(scope="module")
def shm_results(sh, x_shm):
    ton = shm.tonality(x_shm, FS)
    return {"ton": ton, "lou": shm.loudness(None, None, tonality=ton), "rou": shm.roughness(x_shm, FS)}


def test_shm_wrappers_equal_direct_calls(sh, x_shm, shm_results):
    kw = {"soundfield": "free_frontal", "wait_bar": False, "parallel_cores": None}
    ton = sh.shm_tonality_ecma(x_shm, FS, **kw)
    lou = sh.shm_loudness_ecma_from_comp(ton["spec_tonal_loudness"], ton["spec_noise_loudness"])
    rou = sh.shm_roughness_ecma(x_shm, FS, **kw)
    r = shm_results
    np.testing.assert_array_equal(r["ton"].values, ton["tonality_t"])
    np.testing.assert_array_equal(r["ton"].t, ton["time_out"])
    np.testing.assert_array_equal(r["lou"].values, lou["loudness_t"])
    np.testing.assert_array_equal(r["rou"].values, rou["roughness_t"])
    np.testing.assert_array_equal(r["rou"].t, rou["time_out"])
    assert r["ton"].single_value == float(ton["tonality_avg"][0])
    assert r["lou"].single_value == float(lou["loudness_powavg"][0])
    assert r["rou"].single_value == float(rou["roughness90pc"][0])
    assert [r[k].unit for k in ("ton", "lou", "rou")] == ["tu_HMS", "sone_HMS", "asper_HMS"]
    assert "ECMA-418-2:2025" in r["ton"].source and r["ton"].source.startswith("sottek-hearing-model ")
    assert (r["ton"].skip_start, r["lou"].skip_start, r["rou"].skip_start) == (0.30, 0.30, 0.31)


def test_shm_bridge_imports_the_external_package_not_itself(sh):
    # the bridge module is named like the package it wraps; the absolute import must still
    # reach the package, never the bridge
    model = shm._model()
    assert model is sh and model is not shm
    assert model.__name__ == "sottek_hearing_model" and shm.__name__ == "rota_order.bridges.sottek_hearing_model"
    assert model.__file__ != shm.__file__
    assert hasattr(model, "shm_tonality_ecma") and not hasattr(shm, "shm_tonality_ecma")
    assert sys.modules["sottek_hearing_model"] is model
    assert sys.modules["rota_order.bridges.sottek_hearing_model"] is shm
    # the same in a fresh interpreter, in which the package is not imported before the bridge asks for it
    code = (
        "import sys\n"
        "from rota_order.bridges import sottek_hearing_model as bridge\n"
        "assert 'sottek_hearing_model' not in sys.modules\n"
        "model = bridge._model()\n"
        "assert model is not bridge and model.__name__ == 'sottek_hearing_model'\n"
        "assert model.__file__ != bridge.__file__\n"
        "assert hasattr(model, 'shm_tonality_ecma') and not hasattr(bridge, 'shm_tonality_ecma')\n"
        "assert sys.modules['sottek_hearing_model'] is model\n"
        "assert sys.modules['rota_order.bridges.sottek_hearing_model'] is bridge\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_shm_vs_rpm_single_bin_reproduces_package_values(shm_results, x_shm):
    # x_shm has 57600 samples, a multiple of 256: the last model frame lies one sample past the last
    # sample of the record, and vs_rpm holds the speed for it
    t = np.arange(x_shm.size) / FS
    rpm = 1000.0 + 1000.0 * t
    assert x_shm.size % 256 == 0
    assert 0.0 < shm_results["ton"].t[-1] - t[-1] < 1.5 / FS
    one = {"segment": "all", "bins": [0.0, 1.0e6], "min_count": 1}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ton = shm_results["ton"].vs_rpm(rpm, t, **one)
        rou = shm_results["rou"].vs_rpm(rpm, t, **one)
    assert np.isclose(ton.value[0], shm_results["ton"].single_value, rtol=1e-6)
    assert np.isclose(rou.value[0], shm_results["rou"].single_value, rtol=1e-6)
    for res, key, name, unit in ((ton, "ton", "tonality", "tu_HMS"), (rou, "rou", "roughness", "asper_HMS")):
        assert (res.name, res.unit, res.source) == (name, unit, shm_results[key].source)
        assert res.source.startswith("sottek-hearing-model ") and "ECMA-418-2:2025" in res.source


@pytest.fixture(scope="module")
def shm_one_second(sh):
    x = _signal(FS, 1.0, seed=5)  # 48000 samples, not a multiple of 256
    ton = shm.tonality(x, FS)
    return x, ton, shm.loudness(None, None, tonality=ton)


def test_shm_vs_rpm_keeps_the_last_frame_when_the_length_is_not_a_multiple_of_256(shm_one_second):
    # The model puts the last frame at the length rounded up to a multiple of 256 samples: here
    # 48128 samples, 129 samples (2.7 ms) after the last sample of the record, not one.
    x, ton, lou = shm_one_second
    t = np.arange(x.size) / FS
    rpm = 1000.0 + 1000.0 * t
    assert x.size == 48000 and x.size % 256 == 128
    assert np.isclose(ton.t[-1], 188 * FRAME, rtol=0.0, atol=1e-12)
    assert np.isclose(ton.t[-1] - t[-1], 129 / FS, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(np.diff(ton.t), FRAME, rtol=1e-9)
    np.testing.assert_array_equal(lou.t, ton.t)  # the loudness is derived on the frames of the tonality
    one = {"segment": "all", "bins": [0.0, 1.0e6], "min_count": 1}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no frame is reported as outside the record
        res = ton.vs_rpm(rpm, t, **one)
        res_lou = lou.vs_rpm(rpm, t, **one)
    # the single bin is the package's own time average of the tonality: all the frames above the
    # threshold after the start-up, the last one included
    assert np.isclose(res.value[0], ton.single_value, rtol=1e-6)
    keep = ton.t >= 0.30
    assert res.count[0] == np.count_nonzero(keep & (ton.values > 0.02))
    assert res_lou.count[0] == np.count_nonzero(keep)
    assert np.isclose(res_lou.value[0], lou.values[keep].mean(), rtol=1e-12)
    for r in (res, res_lou):
        assert r.control[-1] == rpm[-1]  # the last frame was given the last speed


def test_shm_resampling_and_calibration(sh):
    fs = 32000.0
    x = _signal(fs, 1.0, seed=4)
    a = shm.roughness(x, fs, calibration=0.5)
    b = shm.roughness(0.5 * x, fs)
    np.testing.assert_allclose(a.values, b.values, rtol=1e-12)
    c = shm.roughness(x, fs)
    keep = c.t >= 0.31
    assert c.values[keep].mean() > a.values[keep].mean() >= 0.0


def test_shm_import_restores_matplotlib_rcparams(sh):  # sh: skip without the package (and matplotlib)
    code = (
        "import matplotlib\n"
        "before = matplotlib.rcParams.copy()\n"
        "from rota_order.bridges import sottek_hearing_model as shm\n"
        "shm._model()\n"
        "diff = [k for k in before if dict.__getitem__(before, k) != dict.__getitem__(matplotlib.rcParams, k)]\n"
        "assert not diff, diff\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
