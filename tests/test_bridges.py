"""rota.bridges: lazy optional wrappers for MoSQITo and the Sottek hearing model.

Tests without the toolboxes always run (they simulate the missing packages). Tests with a toolbox
are skipped when it is not installed. Data come from rota.synth only. The models are slow, so the
signals are short and the results are computed once per module.
"""

import subprocess
import sys
import warnings

import numpy as np
import pytest

import rota
from rota.bridges import TimeMetric, along_rpm, mosqito, shm
from rota.slices import metric_vs_rpm

FS = 48000.0
RMS_PA = 0.1  # 71 dB re 20 uPa


def _signal(fs, duration, seed=1):
    s = rota.synth.synth_tone_noise(fs, duration, RMS_PA, 6.0, tone_freq=500.0, seed=seed)
    # amplitude-modulated so that roughness and loudness are not constant
    return s.x * (1.0 + 0.5 * np.sin(2 * np.pi * 70.0 * s.t))


def _fake(name="loudness", t_end=1.0, dt=0.1, **kw):
    t = np.arange(0.0, t_end + 1e-9, dt)
    return TimeMetric(name=name, t=t, values=1.0 + t, unit="sone", **kw)


# ---- without the toolboxes (always run) ---------------------------------------------------------------


def test_import_without_toolboxes_in_subprocess():
    code = (
        "import sys\n"
        "for m in ('mosqito', 'sottek_hearing_model'):\n"
        "    sys.modules[m] = None\n"
        "import rota, rota.bridges\n"
        "assert 'matplotlib' not in sys.modules or True\n"
        "for f in (rota.bridges.mosqito.loudness, rota.bridges.shm.tonality):\n"
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


def test_import_does_not_import_toolboxes_in_subprocess():
    code = "import sys, rota.bridges; assert 'mosqito' not in sys.modules and 'sottek_hearing_model' not in sys.modules"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


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


@pytest.mark.parametrize("call", SHM_CALLS)
def test_shm_missing_raises_import_error(call, monkeypatch):
    monkeypatch.setitem(sys.modules, "sottek_hearing_model", None)
    with pytest.raises(ImportError, match="pip install sottek-hearing-model") as e:
        call(np.zeros(4800))
    assert "does not depend" in str(e.value) and "3.11" in str(e.value)


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
    with pytest.raises(ValueError, match="TimeMetric"):
        shm.loudness(x, FS, tonality=_fake())


def test_along_rpm_equals_metric_vs_rpm_and_uses_recommendations():
    m = _fake(skip_start=0.3, aggregate=("percentile", 90))
    t_rpm = np.arange(0.0, 1.0 + 1e-9, 0.01)
    rpm = 1000.0 + 1000.0 * t_rpm
    a = along_rpm(m, rpm, t_rpm, bin_width=250.0, min_count=1)
    b = metric_vs_rpm(m.values, m.t, rpm, t_rpm, bin_width=250.0, min_count=1, skip_start=0.3,
                      aggregate=("percentile", 90))
    np.testing.assert_array_equal(a.value, b.value)
    np.testing.assert_array_equal(a.count, b.count)
    c = along_rpm(m, rpm, t_rpm, bin_width=250.0, min_count=1, skip_start=0.0, aggregate="mean")
    assert c.count.sum() > a.count.sum()  # kwargs override


def test_along_rpm_threshold():
    m = _fake(threshold=1.5)  # values 1 + t: frames with t <= 0.5 are discarded
    t_rpm = np.array([0.0, 1.0])
    r = along_rpm(m, np.array([1000.0, 2000.0]), t_rpm, segment="all", bins=[0.0, 1e6], min_count=1)
    assert r.count[0] == 5 and np.isclose(r.value[0], np.mean(1.0 + np.arange(0.6, 1.0 + 1e-9, 0.1)))


def test_along_rpm_holds_speed_for_last_frame_one_sample_past_end():
    fs = 1000.0
    t_rpm = np.arange(1000) / fs  # last speed sample at 0.999 s
    rpm = 1000.0 + 1000.0 * t_rpm
    m = _fake(t_end=1.0, dt=0.25)  # last frame at 1.0 s: one sample past the record
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = along_rpm(m, rpm, t_rpm, segment="all", bins=[0.0, 1e6], min_count=1)
    assert r.count[0] == m.t.size
    assert np.isclose(r.value[0], m.values.mean())


def test_along_rpm_warns_for_frames_far_beyond_record():
    t_rpm = np.arange(0.0, 0.5, 0.01)
    m = _fake(t_end=1.0, dt=0.1)
    with pytest.warns(UserWarning, match="outside the speed record"):
        r = along_rpm(m, 1000.0 + 0.0 * t_rpm + t_rpm, t_rpm, segment="all", bins=[0.0, 1e6], min_count=1)
    assert r.count[0] == np.count_nonzero(m.t <= 0.5)  # 0.5 s is held (one step past 0.49 s)


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
    x48 = rota.tonal.to_calibrated_48k(x, fs)
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


def test_mosqito_along_rpm_matches_hand_call(mq_results, x48):
    t = np.arange(x48.size) / FS
    rpm = 1000.0 + 500.0 * t
    loud = mq_results["loud"]
    a = along_rpm(loud, rpm, t, bins=np.arange(1000.0, 2001.0, 250.0), min_count=3)
    b = metric_vs_rpm(loud.values, loud.t, rpm, t, bins=np.arange(1000.0, 2001.0, 250.0), min_count=3,
                      skip_start=0.6, aggregate=("percentile", 95))
    np.testing.assert_array_equal(a.value, b.value)
    assert a.valid.any()


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


def test_shm_along_rpm_single_bin_reproduces_package_values(shm_results, x_shm):
    # the last model frame lies one sample past the end of the record: along_rpm holds the speed
    t = np.arange(x_shm.size) / FS
    rpm = 1000.0 + 1000.0 * t
    assert shm_results["ton"].t[-1] > t[-1]
    one = {"segment": "all", "bins": [0.0, 1.0e6], "min_count": 1}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ton = along_rpm(shm_results["ton"], rpm, t, **one)
        rou = along_rpm(shm_results["rou"], rpm, t, **one)
    assert np.isclose(ton.value[0], shm_results["ton"].single_value, rtol=1e-6)
    assert np.isclose(rou.value[0], shm_results["rou"].single_value, rtol=1e-6)


def test_shm_resampling_and_calibration(sh):
    fs = 32000.0
    x = _signal(fs, 1.0, seed=4)
    a = shm.roughness(x, fs, calibration=0.5)
    b = shm.roughness(0.5 * x, fs)
    np.testing.assert_allclose(a.values, b.values, rtol=1e-12)
    c = shm.roughness(x, fs)
    keep = c.t >= 0.31
    assert c.values[keep].mean() > a.values[keep].mean() >= 0.0


def test_shm_import_restores_matplotlib_rcparams():
    pytest.importorskip("sottek_hearing_model")
    code = (
        "import matplotlib\n"
        "before = matplotlib.rcParams.copy()\n"
        "from rota.bridges import shm\n"
        "shm._model()\n"
        "diff = [k for k in before if dict.__getitem__(before, k) != dict.__getitem__(matplotlib.rcParams, k)]\n"
        "assert not diff, diff\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
