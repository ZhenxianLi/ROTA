"""Bridge to the Sottek hearing model of ECMA-418-2 (optional external dependency).

The package ``sottek-hearing-model`` (import name ``sottek_hearing_model``, AGPL-3.0-only,
Python 3.11 or later) is not a dependency of ROTA; without it this module is skipped. The test
signal is a tone at a known order in white noise from ``rota.synth``: the dominant tonal
frequency that the model reports must map to that order through
``rota.slices.frequency_to_order`` / ``order_share``, and ``rota.slices.metric_vs_rpm`` must
run on the model's tonality, loudness and roughness and reproduce the model's single values.
"""

import numpy as np
import pytest

shm = pytest.importorskip("sottek_hearing_model")

import rota  # noqa: E402

FS = 16000.0  # sampling rate of the synthetic signal, Hz
FS_MODEL = 48000.0
DURATION = 2.5  # s
ORDER = 10.0
RPM0, RPM1 = 3000.0, 3600.0  # tone sweeps from 500 to 600 Hz
TONE_PEAK = 0.5  # Pa, peak
NOISE_RMS = 0.05  # Pa, white
THRESHOLD = 0.02  # tu_HMS, ECMA-418-2 averaging threshold
SKIP = 0.30  # s, start-up frames of tonality and loudness
SKIP_ROUGH = 0.31  # s, start-up frames of roughness


@pytest.fixture(scope="module")
def run():
    t = np.arange(int(DURATION * FS)) / FS
    rpm = rota.synth.rpm_profiles(t, "linear", start=RPM0, stop=RPM1)
    sig = rota.synth.synth_runup(
        FS, rpm, orders=[{"order": ORDER, "amplitude": TONE_PEAK, "phase": 0.0}], noise_rms=NOISE_RMS, seed=1
    )
    p = rota.tonal.to_calibrated_48k(sig.x, FS, calibration=1.0, fs_out=FS_MODEL)  # Pa at 48 kHz
    kw = {"wait_bar": False}
    ton = shm.shm_tonality_ecma(p, FS_MODEL, **kw)
    lou = shm.shm_loudness_ecma_from_comp(ton["spec_tonal_loudness"], ton["spec_noise_loudness"])
    rou = shm.shm_roughness_ecma(p, FS_MODEL, **kw)
    # Speed axis held for 10 ms beyond the last sample: the model's last frame lies at the
    # duration, one sample after it, and metric_vs_rpm does not extrapolate the speed.
    t_rpm, rpm_h = np.append(sig.t, sig.t[-1] + 0.01), np.append(sig.rpm, sig.rpm[-1])
    t_m = np.asarray(ton["time_out"], float)
    return {"sig": sig, "p": p, "ton": ton, "lou": lou, "rou": rou, "t_m": t_m, "t_rpm": t_rpm, "rpm": rpm_h,
            "rpm_m": np.interp(t_m, t_rpm, rpm_h)}


def _tonal(run):
    """Mask of the frames that enter the ECMA-418-2 tonality average."""
    return (run["t_m"] >= SKIP) & (run["ton"]["tonality_t"] > THRESHOLD)


def test_calibrated_48k_keeps_the_level(run):
    # The bridge step: Pa at 48 kHz with the pressure level of the synthetic signal preserved.
    sig, p = run["sig"], run["p"]
    assert p.shape[0] == int(DURATION * FS_MODEL)
    rms_in, rms_out = np.sqrt(np.mean(sig.x**2)), np.sqrt(np.mean(p**2))
    assert abs(20 * np.log10(rms_out / rms_in)) < 0.05


def test_model_output_shapes(run):
    ton, rou = run["ton"], run["rou"]
    n = run["t_m"].size
    assert ton["tonality_t"].shape == (n,) and ton["tonality_t_freqs"].shape == (n,)
    assert np.all(np.diff(run["t_m"]) > 0)
    assert run["lou"]["loudness_t"].shape == (n,)
    assert rou["roughness_t"].shape == np.asarray(rou["time_out"]).shape


def test_dominant_tonal_frequency_maps_to_the_tone_order(run):
    tonal = _tonal(run)
    assert tonal.sum() > 0.8 * (run["t_m"] >= SKIP).sum()  # the tone is heard as tonal almost all the time
    f = np.where(tonal, run["ton"]["tonality_t_freqs"], np.nan)
    fo = rota.slices.frequency_to_order(f, run["rpm_m"], tolerance=0.04, orders=[ORDER])
    assert fo.matched[tonal].mean() >= 0.95
    assert np.all(fo.nearest[tonal] == ORDER)
    # Truth: the model's frequency reading is within 2 % of the tone's true frequency.
    f_true = ORDER * run["rpm_m"] / 60.0
    assert np.median(np.abs(f[tonal] / f_true[tonal] - 1.0)) < 0.02
    share = rota.slices.order_share(f, run["rpm_m"], ORDER, harmonics=1, tolerance=0.04)
    assert share.n_valid == tonal.sum()
    assert share.share >= 0.95
    # A base order without a source collects (almost) nothing.
    other = rota.slices.order_share(f, run["rpm_m"], 7.0, harmonics=1, tolerance=0.04)
    assert other.share < 0.05


def test_metric_vs_rpm_on_the_model_output(run):
    tn = run["ton"]["tonality_t"]
    res = rota.slices.metric_vs_rpm(
        np.where(tn > THRESHOLD, tn, np.nan), run["t_m"], run["rpm"], run["t_rpm"],
        bin_width=100.0, skip_start=SKIP, aggregate="mean", min_count=5,
    )
    assert res.segment == "runup"
    assert res.valid.sum() >= 4
    assert np.all(res.value[res.valid] > THRESHOLD)
    assert res.centres[res.valid].min() >= RPM0 - 100.0 and res.centres[res.valid].max() <= RPM1 + 100.0
    # Loudness and roughness run through the same function.
    lo = rota.slices.metric_vs_rpm(
        run["lou"]["loudness_t"], run["t_m"], run["rpm"], run["t_rpm"], bin_width=100.0, skip_start=SKIP,
        aggregate="mean", min_count=5,
    )
    assert lo.valid.sum() >= 4 and np.all(lo.value[lo.valid] > 0)
    ro = rota.slices.metric_vs_rpm(
        run["rou"]["roughness_t"], np.asarray(run["rou"]["time_out"], float), run["rpm"], run["t_rpm"],
        bin_width=200.0, skip_start=SKIP_ROUGH, aggregate=("percentile", 90), min_count=5,
    )
    assert ro.valid.sum() >= 2 and np.all(ro.value[ro.valid] >= 0)


def test_one_bin_reproduces_the_single_values_of_the_model(run):
    # metric_vs_rpm with one bin over the whole run equals the package's own single values.
    tn = run["ton"]["tonality_t"]
    one = {"segment": "all", "bins": [0.0, 1.0e6], "min_count": 1}
    t_avg = rota.slices.metric_vs_rpm(
        np.where(tn > THRESHOLD, tn, np.nan), run["t_m"], run["rpm"], run["t_rpm"], skip_start=SKIP,
        aggregate="mean", **one,
    )
    assert np.isclose(t_avg.value[0], run["ton"]["tonality_avg"][0], rtol=1e-6)
    r90 = rota.slices.metric_vs_rpm(
        run["rou"]["roughness_t"], np.asarray(run["rou"]["time_out"], float), run["rpm"], run["t_rpm"],
        skip_start=SKIP_ROUGH, aggregate=("percentile", 90), **one,
    )
    assert np.isclose(r90.value[0], run["rou"]["roughness90pc"][0], rtol=1e-6)
