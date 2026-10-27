"""Tests of rota._align.align_signals against known delays, drifts and rates.

Truth comes from rota.synth: ``delay_signal`` (fractional delay) and
``measurement_chain`` (clock drift), applied to a synthetic sound field made of
band-limited ambient noise and two hand-clap-like clicks.
"""

import warnings

import numpy as np
import pytest
from scipy import signal

from rota._align import AlignResult, align_signals, detect_onsets
from rota.synth import delay_signal, measurement_chain, transient_events

FS = 16000.0
DUR = 20.0
CLAPS = (1.5, 18.5)
FS48 = 48000.0
CLAPS48 = (1.0, 11.0)


def _scene(fs, dur, claps=CLAPS, *, seed=0, hi=3500.0, ambient=0.05, clap=2.0):
    """Ambient noise (rms ``ambient``) plus clicks of peak ``clap``, both low-passed at ``hi``."""
    rng = np.random.default_rng(seed)
    n = int(round(dur * fs))
    sos = signal.butter(4, hi, "low", fs=fs, output="sos")
    amb = signal.sosfiltfilt(sos, rng.standard_normal(n))
    amb *= ambient / amb.std()
    ev = [{"time": t, "kind": "click", "amplitude": clap, "decay": 0.01} for t in claps]
    stem, _ = transient_events(fs, n, ev, seed=seed + 1)
    return amb + signal.sosfiltfilt(sos, stem)


@pytest.fixture(scope="module")
def scene():
    return _scene(FS, DUR)


@pytest.fixture(scope="module")
def scene48():
    return _scene(FS48, 12.0, claps=CLAPS48)


def _decimate(x, q):
    return signal.resample_poly(x, 1, q)


# --------------------------------------------------------------------------
# conventions and the time map
# --------------------------------------------------------------------------


def test_time_map_round_trip_and_uncertainty_propagation():
    r = AlignResult(
        ok=True, message="", offset=0.25, drift_ppm=100.0, drift_estimated=True,
        uncertainty=1e-6, offset_uncertainty=1e-6, drift_uncertainty_ppm=0.5, fs=1.0,
        _cov=np.array([[1e-12, 0.0], [0.0, 2.5e-13]]),
    )
    t = np.array([0.0, 1.0, 100.0])
    assert np.allclose(r.to_other(t), 0.25 + 1.0001 * t)
    assert np.allclose(r.to_ref(r.to_other(t)), t, rtol=0, atol=1e-12)
    assert np.allclose(r.uncertainty_at(t), np.sqrt(1e-12 + t**2 * 2.5e-13))


# --------------------------------------------------------------------------
# known fractional delays
# --------------------------------------------------------------------------


@pytest.mark.parametrize("delay_samples", [0.0, 0.25, 0.5, 7.37, -12.62, 400.3, -1234.9])
@pytest.mark.parametrize("markers", [None, CLAPS])
def test_fractional_delay_recovered_below_0p05_sample(scene, delay_samples, markers):
    d = delay_samples / FS
    other = delay_signal(scene, FS, d)
    r = align_signals(scene, other, FS, markers=markers)
    assert r.ok and r.fs == FS
    assert abs(r.offset - d) * FS < 0.05
    assert abs(r.drift_ppm) < 0.2  # nothing drifts (two markers) or it is not estimated (global)
    assert r.drift_estimated == (markers is not None)
    assert np.all(r.polarity == 1)
    # the offset is within a few of its own standard uncertainties
    assert abs(r.offset - d) < 5.0 * r.offset_uncertainty + 0.02 / FS


def test_integer_lag_without_refinement(scene):
    d = 37.4 / FS
    other = delay_signal(scene, FS, d)
    r = align_signals(scene, other, FS, markers=[CLAPS[0]], refine=False)
    assert r.ok
    assert abs(r.offset * FS - 37.0) < 1e-6  # nearest sample
    r2 = align_signals(scene, other, FS, markers=[CLAPS[0]])
    assert abs(r2.offset - d) * FS < 0.05


def test_offset_sign_matches_delay_signal(scene):
    other = delay_signal(scene, FS, 0.2)  # other lags
    assert align_signals(scene, other, FS, markers=CLAPS).offset == pytest.approx(0.2, abs=1e-6)
    early = delay_signal(scene, FS, -0.2)  # other leads
    assert align_signals(scene, early, FS, markers=CLAPS).offset == pytest.approx(-0.2, abs=1e-6)


def test_gain_and_spectral_colouring_do_not_matter(scene):
    """PHAT: a different gain and a different filter on one channel leave the lag alone."""
    d = 0.0321
    other = 7.0 * delay_signal(scene, FS, d)
    sos = signal.butter(2, 900.0, "low", fs=FS, output="sos")
    coloured = signal.sosfilt(sos, other)  # adds a smooth group delay, filter is minimum phase
    r = align_signals(scene, 3.0 * other, FS, markers=CLAPS)
    assert abs(r.offset - d) * FS < 0.05
    r = align_signals(scene, coloured, FS, markers=CLAPS, band=(300.0, 3000.0))
    # the low-pass delays the band-limited content by its group delay (a fraction of a ms)
    assert abs(r.offset - d) < 5e-4


def test_inverted_polarity_is_reported(scene):
    d = 0.01
    r = align_signals(scene, -delay_signal(scene, FS, d), FS, markers=CLAPS)
    assert r.ok and abs(r.offset - d) * FS < 0.05
    assert np.all(r.polarity == -1) and "polarity" in r.message


# --------------------------------------------------------------------------
# clock drift
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ppm", [-150.0, -20.0, 0.0, 35.0, 200.0])
def test_clock_drift_recovered_below_2_ppm_with_two_markers(scene, ppm):
    d = 0.3731
    chain = measurement_chain(scene, FS, clock_drift_ppm=ppm, noise_floor_rms=0.01, seed=3)
    other = delay_signal(chain["x"], FS, d)  # t_other = d + (1 + eps) t_ref
    r = align_signals(scene, other, FS, markers=CLAPS)
    assert r.ok and r.drift_estimated
    assert abs(r.drift_ppm - ppm) < 2.0
    assert abs(r.offset - d) < 50e-6
    # the map sends each clap to its stamp in `other` within a sample
    assert np.allclose(r.to_other(np.array(CLAPS)), d + (1 + ppm * 1e-6) * np.array(CLAPS), atol=1.0 / FS)
    assert np.allclose(r.to_ref(r.to_other(np.array(CLAPS))), CLAPS)
    # estimated uncertainty covers the error of the drift
    assert abs(r.drift_ppm - ppm) < 5.0 * r.drift_uncertainty_ppm + 0.05
    assert r.uncertainty < 20e-6 and r.uncertainty_at(CLAPS[0]) == pytest.approx(r.marker_uncertainty[0], rel=0.2)


def test_one_marker_gives_offset_only(scene):
    chain = measurement_chain(scene, FS, clock_drift_ppm=100.0, seed=0)
    r = align_signals(scene, chain["x"], FS, markers=[CLAPS[0]])
    assert r.ok and not r.drift_estimated and r.drift_ppm == 0.0
    assert "offset only" in r.message
    # the offset at the marker is right (to the drift accumulated over 2 s: 200 us)
    assert abs(r.to_other(CLAPS[0]) - CLAPS[0] * (1 + 100e-6)) < 1.0 / FS


def test_three_markers_are_fitted(scene):
    x = _scene(FS, DUR, claps=(1.5, 10.0, 18.5))
    chain = measurement_chain(x, FS, clock_drift_ppm=60.0, noise_floor_rms=0.01, seed=1)
    other = delay_signal(chain["x"], FS, 0.05)
    r = align_signals(x, other, FS, markers=[1.5, 10.0, 18.5])
    assert r.ok and r.marker_lags.size == 3
    assert abs(r.drift_ppm - 60.0) < 2.0 and abs(r.offset - 0.05) < 50e-6


def test_auto_markers_find_the_claps_and_the_drift():
    x = _scene(FS, DUR, ambient=0.01, clap=5.0)
    chain = measurement_chain(x, FS, clock_drift_ppm=-45.0, seed=0)
    other = delay_signal(chain["x"], FS, 1.2)
    r = align_signals(x, other, FS, markers="auto")
    assert r.ok and r.drift_estimated
    assert r.marker_times.size == 2
    assert np.allclose(r.marker_times, CLAPS, atol=0.01)
    assert abs(r.drift_ppm + 45.0) < 2.0 and abs(r.offset - 1.2) < 100e-6


def test_auto_without_a_transient_falls_back_to_a_global_correlation():
    rng = np.random.default_rng(0)
    sos = signal.butter(4, 3000.0, "low", fs=FS, output="sos")
    x = signal.sosfiltfilt(sos, rng.standard_normal(int(5 * FS)))
    r = align_signals(x, delay_signal(x, FS, 0.01), FS, markers="auto")
    assert r.ok and not r.drift_estimated and "global" in r.message
    assert abs(r.offset - 0.01) * FS < 0.05


# --------------------------------------------------------------------------
# different sampling rates
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ppm", [0.0, 80.0])
def test_integer_rate_ratio_48k_against_9p6k(scene48, ppm):
    d = 0.4321567
    chain = measurement_chain(scene48, FS48, clock_drift_ppm=ppm, noise_floor_rms=0.01, seed=2)
    other = _decimate(delay_signal(chain["x"], FS48, d), 5)
    r = align_signals(scene48, other, FS48, 9600.0, markers=CLAPS48)
    assert r.ok and r.fs == pytest.approx(9600.0)
    assert abs(r.offset - d) * 9600.0 < 0.05
    assert abs(r.drift_ppm - ppm) < 2.0
    # the other way round: the faster recording is the second one
    r2 = align_signals(_decimate(scene48, 5), delay_signal(chain["x"], FS48, d), 9600.0, FS48, markers=CLAPS48)
    assert r2.ok and abs(r2.offset - d) * 9600.0 < 0.05 and abs(r2.drift_ppm - ppm) < 2.0


def test_non_integer_rate_ratio_48k_against_44p1k(scene48):
    d = -0.0817
    other = signal.resample_poly(delay_signal(scene48, FS48, d), 147, 160)  # 44.1 kHz
    r = align_signals(scene48, other, FS48, 44100.0, markers=CLAPS48)
    assert r.ok and r.fs == pytest.approx(44100.0)
    assert abs(r.offset - d) * 44100.0 < 0.05
    assert abs(r.drift_ppm) < 2.0
    g = align_signals(scene48, other, FS48, 44100.0)
    assert abs(g.offset - d) * 44100.0 < 0.05


def test_global_correlation_at_unrelated_rates_and_fs_common(scene48):
    d = 0.21
    other = _decimate(delay_signal(scene48, FS48, d), 5)
    r = align_signals(scene48, other, FS48, 9600.0, fs_common=8000.0)
    assert r.ok and r.fs == pytest.approx(8000.0)
    assert abs(r.offset - d) * 8000.0 < 0.05
    with pytest.raises(ValueError, match="fs_common"):
        align_signals(scene48, other, FS48, 9600.0, fs_common=48000.0)


def test_pair_markers_search_around_the_given_lag():
    fs = 8000.0
    x = _scene(fs, 22.0, claps=(2.0, 17.0), hi=3000.0)
    d = 3.2  # larger than max_lag
    other = delay_signal(x, fs, d)
    ok = align_signals(x, other, fs, markers=[[2.0, 17.0], [2.0 + d + 0.02, 17.0 + d - 0.03]], max_lag=0.2)
    assert ok.ok and abs(ok.offset - d) * fs < 0.05
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bad = align_signals(x, other, fs, markers=[2.0, 17.0], max_lag=1.0)
    assert not bad.ok


# --------------------------------------------------------------------------
# noise
# --------------------------------------------------------------------------


def test_robust_to_independent_noise_and_uncertainty_is_calibrated():
    fs = 8000.0
    zs, errs = [], []
    for seed in range(30):
        x = _scene(fs, 10.0, claps=(1.0, 9.0), seed=seed, hi=3200.0, ambient=0.05, clap=2.0)
        d = (0.0123 + 0.37e-4) + seed * 1e-4
        o = delay_signal(x, fs, d)
        rng = np.random.default_rng(100 + seed)
        a = x + 0.1 * rng.standard_normal(x.size)  # noise 0.1 rms against 0.05 ambient
        b = o + 0.1 * rng.standard_normal(x.size)
        r = align_signals(a, b, fs, markers=[1.0])
        assert r.ok, (seed, r.message)
        errs.append(r.offset - d)
        zs.append((r.offset - d) / r.uncertainty)
    errs, zs = np.array(errs), np.array(zs)
    assert np.sqrt(np.mean(errs**2)) * fs < 0.05  # high-SNR claps: still below 0.05 sample
    assert np.mean(np.abs(zs) < 3.0) >= 0.95
    assert 0.4 < np.sqrt(np.mean(zs**2)) < 2.0


def test_two_markers_in_strong_noise_keep_drift_error_small():
    fs = 8000.0
    x = _scene(fs, 40.0, claps=(2.0, 37.0), seed=4, hi=3200.0, ambient=0.05, clap=2.0)
    chain = measurement_chain(x, fs, clock_drift_ppm=40.0, noise_floor_rms=0.12, seed=5)
    other = delay_signal(chain["x"], fs, 0.05)
    ref = x + 0.12 * np.random.default_rng(6).standard_normal(x.size)
    r = align_signals(ref, other, fs, markers=[2.0, 37.0])
    assert r.ok
    assert abs(r.drift_ppm - 40.0) < 2.0
    assert abs(r.drift_ppm - 40.0) < 5.0 * r.drift_uncertainty_ppm


# --------------------------------------------------------------------------
# failures
# --------------------------------------------------------------------------


@pytest.mark.parametrize("markers", [None, [1.0, 9.0]])
def test_no_common_content_is_flagged(markers):
    fs = 8000.0
    n = int(10 * fs)
    a = np.random.default_rng(1).standard_normal(n)
    b = np.random.default_rng(2).standard_normal(n)
    with pytest.warns(RuntimeWarning, match="align_signals failed"):
        r = align_signals(a, b, fs, markers=markers)
    assert not r.ok and r.message.startswith("failed")
    assert np.isnan(r.offset) and np.isnan(r.drift_ppm) and np.isinf(r.uncertainty)
    assert np.all(np.isnan(r.to_other([0.0, 1.0]))) and np.all(np.isnan(r.to_ref(0.5)))
    assert np.all(r.psr < r.threshold)
    with pytest.raises(ValueError, match="failed"):
        r.resample_other(b, fs, fs)


@pytest.mark.parametrize("fs, dur, seed", [(FS, DUR, 0), (48000.0, 30.0, 0), (48000.0, 30.0, 1)])
def test_lag_beyond_max_lag_is_flagged(fs, dur, seed):
    x = _scene(fs, dur, claps=(2.0, dur - 3.0), seed=seed)
    other = delay_signal(x, fs, 3.0)
    with pytest.warns(RuntimeWarning):
        r = align_signals(x, other, fs, max_lag=1.0)
    assert not r.ok
    ok = align_signals(x, other, fs, max_lag=4.0)
    assert ok.ok and abs(ok.offset - 3.0) * fs < 0.05


def test_steady_tone_alone_is_ambiguous_and_flagged():
    fs = 8000.0
    t = np.arange(int(10 * fs)) / fs
    rng = np.random.default_rng(0)
    a = np.sin(2 * np.pi * 100.0 * t) + 0.1 * rng.standard_normal(t.size)
    b = np.sin(2 * np.pi * 100.0 * t + 1.0) + 0.1 * rng.standard_normal(t.size)
    with pytest.warns(RuntimeWarning, match="ambiguous"):
        r = align_signals(a, b, fs)
    assert not r.ok and r.ambiguity[0] > 0.8


def test_marker_outside_the_record_is_flagged(scene):
    other = delay_signal(scene, FS, 0.01)
    with pytest.warns(RuntimeWarning):
        r = align_signals(scene, other, FS, markers=[2.0, 500.0])
    assert not r.ok and "outside" in r.message


# --------------------------------------------------------------------------
# onset detection, resampling onto the reference clock, validation
# --------------------------------------------------------------------------


def test_detect_onsets_locates_claps_and_ignores_stationary_noise():
    x = _scene(FS, DUR, claps=(1.5, 1.8, 18.5), ambient=0.01, clap=5.0)
    t = detect_onsets(x, FS)
    assert t.size == 2  # 1.5 and 1.8 are one event (closer than min_separation)
    assert np.allclose(t, [1.5, 18.5], atol=0.01)
    both = detect_onsets(x, FS, min_separation=0.1)
    assert both.size == 3 and abs(both[1] - 1.8) < 0.01
    noise = np.random.default_rng(0).standard_normal(int(10 * FS))
    assert detect_onsets(noise, FS).size == 0
    with pytest.raises(ValueError, match="too short"):
        detect_onsets(np.zeros(10), FS)


def test_resample_other_puts_other_on_the_reference_clock():
    fs_ref, fs_o = 2000.0, 1000.0
    n = int(40 * fs_ref)
    t = np.arange(n) / fs_ref
    f = lambda tt: np.sin(2 * np.pi * 7.0 * tt) + 0.5 * np.sin(2 * np.pi * 31.0 * tt + 0.4)
    d, eps = 0.2371, 90e-6
    t_o = np.arange(int(39 * fs_o)) / fs_o
    other = f((t_o - d) / (1 + eps))  # t_other = d + (1 + eps) t_ref
    ref = f(t)
    r = AlignResult(
        ok=True, message="", offset=d, drift_ppm=eps * 1e6, drift_estimated=True,
        uncertainty=0.0, offset_uncertainty=0.0, drift_uncertainty_ppm=0.0, fs=fs_o,
    )
    y = r.resample_other(other, fs_o, fs_ref)
    assert np.array_equal(r.resample_other(other, fs_o, fs_ref, n_out=y.size + 5)[: y.size], y, equal_nan=True)
    assert np.all(np.isnan(r.resample_other(other, fs_o, fs_ref, n_out=y.size + 5)[-1:]))  # past the record
    k = np.arange(y.size)
    sel = np.isfinite(y) & (k > 1000) & (k < y.size - 1000)  # leave the ends of the record out
    assert sel.sum() > 0.9 * y.size
    # in the interior the resampled signal equals the reference (cubic-spline error < 2e-3)
    assert np.max(np.abs(y[sel] - ref[: y.size][sel])) < 2e-3
