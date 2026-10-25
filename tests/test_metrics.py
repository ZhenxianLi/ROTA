"""Tests of rota.metrics against hand-computed and rota.synth ground truth.

track_error: hand-computed cases. waveform_similarity and order_residual_energy:
analytic values from components rendered by rota.synth (orthogonal order stems
at constant speed, scaled copies of a run-up). window_checks: closed-form
sums of the Hann window; modulation_peak and envelope_stats: tones, two-tone
beats and white Gaussian noise from rota.synth with analytic envelope laws.
"""

import numpy as np
import pytest
from scipy import signal

from rota.metrics import (
    envelope_stats,
    modulation_peak,
    order_residual_energy,
    track_error,
    waveform_similarity,
    window_checks,
)
from rota.synth import synth_runup


def test_hand_computed_case():
    est = [100.0, 102.0, 95.0, 110.0]
    truth = [100.0, 100.0, 100.0, 100.0]
    r = track_error(est, truth)
    # relative errors: 0, 0.02, 0.05, 0.10
    assert r["median_rel_error"] == pytest.approx(0.035)
    assert r["mean_rel_error"] == pytest.approx(0.0425)
    assert r["max_rel_error"] == pytest.approx(0.10)
    assert list(r["within"]) == [0.03, 0.05]
    assert r["within"][0.03] == pytest.approx(0.5)
    assert r["within"][0.05] == pytest.approx(0.75)  # an error of exactly 5 % counts
    assert r["n"] == 4
    assert r["n_invalid"] == 0
    assert "bound_fraction" not in r
    assert all(isinstance(r[k], float) for k in ("median_rel_error", "mean_rel_error", "max_rel_error"))
    assert isinstance(r["n"], int) and isinstance(r["n_invalid"], int)


def test_relative_to_truth_not_estimate():
    r = track_error([50.0], [100.0])
    assert r["max_rel_error"] == pytest.approx(0.5)
    r = track_error([200.0], [100.0])
    assert r["max_rel_error"] == pytest.approx(1.0)


def test_exact_estimate():
    truth = np.linspace(1000.0, 3000.0, 50)
    r = track_error(truth, truth)
    assert r["median_rel_error"] == 0.0
    assert r["max_rel_error"] == 0.0
    assert r["within"] == {0.03: 1.0, 0.05: 1.0}


def test_estimate_scaled_by_three_percent_is_within_three_percent():
    truth = np.array([800.0, 1500.0, 2200.0, 3100.0])
    r = track_error(truth * 1.03, truth)
    assert r["within"][0.03] == 1.0
    assert r["median_rel_error"] == pytest.approx(0.03)


def test_nan_in_either_input_is_excluded_and_counted():
    est = [100.0, np.nan, 102.0, 98.0, 100.0]
    truth = [100.0, 100.0, np.nan, 100.0, 100.0]
    r = track_error(est, truth)
    assert r["n"] == 3 and r["n_invalid"] == 2
    # valid errors: 0, 0.02, 0
    assert r["mean_rel_error"] == pytest.approx(0.02 / 3)
    assert r["median_rel_error"] == 0.0
    assert r["max_rel_error"] == pytest.approx(0.02)
    assert r["within"][0.03] == 1.0


def test_non_finite_and_zero_truth_are_invalid():
    est = [1.0, np.inf, 2.0, 3.0]
    truth = [1.0, 1.0, 0.0, np.inf]
    r = track_error(est, truth)
    assert r["n"] == 1 and r["n_invalid"] == 3
    assert r["max_rel_error"] == 0.0


def test_all_invalid_gives_nan_not_error():
    r = track_error([np.nan, np.nan], [1.0, 2.0], at_bound=[True, False])
    assert r["n"] == 0 and r["n_invalid"] == 2
    assert np.isnan(r["median_rel_error"]) and np.isnan(r["max_rel_error"])
    assert np.isnan(r["within"][0.03]) and np.isnan(r["bound_fraction"])


def test_custom_thresholds():
    est = [100.0, 101.0, 103.0, 120.0]
    truth = [100.0] * 4
    r = track_error(est, truth, thresholds=(0.005, 0.015, 0.5))
    assert list(r["within"]) == [0.005, 0.015, 0.5]
    assert r["within"][0.005] == pytest.approx(0.25)
    assert r["within"][0.015] == pytest.approx(0.5)
    assert r["within"][0.5] == pytest.approx(1.0)
    r1 = track_error(est, truth, thresholds=[0.03])
    assert list(r1["within"]) == [0.03]


def test_at_bound_fraction_counts_valid_samples_only():
    est = [100.0, 50.0, np.nan, 100.0, 100.0]
    truth = [100.0, 100.0, 100.0, 100.0, 100.0]
    flag = [False, True, True, False, True]  # the flagged NaN sample is not counted
    r = track_error(est, truth, at_bound=flag)
    assert r["n"] == 4 and r["n_invalid"] == 1
    assert r["bound_fraction"] == pytest.approx(2 / 4)
    r0 = track_error(est, truth, at_bound=np.zeros(5, bool))
    assert r0["bound_fraction"] == 0.0


def test_scalar_truth_and_multidimensional_input():
    est = np.array([[99.0, 101.0], [100.0, 104.0]])
    r = track_error(est, 100.0)
    assert r["n"] == 4
    assert r["max_rel_error"] == pytest.approx(0.04)
    assert r["within"][0.03] == pytest.approx(0.75)
    flags = np.array([[True, False], [False, False]])
    assert track_error(est, 100.0, at_bound=flags)["bound_fraction"] == pytest.approx(0.25)


def test_unit_invariance():
    rng = np.random.default_rng(1)
    truth = np.linspace(900.0, 3000.0, 200)
    est = truth * (1.0 + 0.04 * rng.standard_normal(truth.size))
    a = track_error(est, truth)
    b = track_error(est / 60.0, truth / 60.0)  # rpm -> Hz
    for k in ("median_rel_error", "mean_rel_error", "max_rel_error"):
        assert a[k] == pytest.approx(b[k])
    assert a["within"] == b["within"]


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        track_error([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        track_error([1.0, 2.0], [1.0, 2.0], at_bound=[True, False, True])
    with pytest.raises(ValueError):
        track_error([1.0], [1.0], thresholds=(0.0,))


# ==========================================================================
# waveform_similarity
# ==========================================================================


def _orthogonal_stems(amplitudes=(1.0, 1.0, 1.0)):
    """Order stems 1, 2, 3 at constant 3000 rpm: 50, 100, 150 Hz over 1 s at 8 kHz.

    The lines have whole numbers of cycles in the record, so the stems are
    exactly orthogonal and have zero mean; each has RMS ``amplitude / sqrt(2)``.
    """
    fs = 8000.0
    orders = [{"order": float(k + 1), "amplitude": a} for k, a in enumerate(amplitudes)]
    sig = synth_runup(fs, np.full(8000, 3000.0), orders, seed=5)
    return sig.order_stems.T  # rows are the stems


def test_identical_waveforms():
    s1, _, _ = _orthogonal_stems()
    r = waveform_similarity(s1, s1)
    assert r["corr"] == pytest.approx(1.0, abs=1e-12)
    assert r["nrmse"] == 0.0
    assert r["corr_per_channel"].shape == (1,) and r["nrmse_per_channel"].shape == (1,)
    assert r["n"] == s1.size
    assert r["weights"] == pytest.approx([1.0])
    assert isinstance(r["corr"], float) and isinstance(r["nrmse"], float)
    for key in ("baseline_corr", "cross_corr", "baseline_nrmse"):
        assert key not in r


@pytest.mark.parametrize("gain, corr, nrmse", [(0.7, 1.0, 0.3), (2.0, 1.0, 1.0), (-1.0, -1.0, 2.0)])
def test_gain_changes_nrmse_but_not_correlation(gain, corr, nrmse):
    s1, _, _ = _orthogonal_stems()
    r = waveform_similarity(gain * s1, s1)
    assert r["corr"] == pytest.approx(corr, abs=1e-12)
    assert r["nrmse"] == pytest.approx(nrmse, rel=1e-12)


def test_offset_is_removed_for_correlation_not_for_nrmse():
    s1, _, _ = _orthogonal_stems(amplitudes=(2.0, 1.0, 1.0))
    r = waveform_similarity(s1 + 0.5, s1)
    rms = 2.0 / np.sqrt(2.0)
    assert r["corr"] == pytest.approx(1.0, abs=1e-12)
    assert r["nrmse"] == pytest.approx(0.5 / rms, rel=1e-12)


def test_orthogonal_error_has_analytic_correlation_and_nrmse():
    # truth: order 1 (peak 1); estimate adds order 2 with peak 0.25 -> energy ratio 1/16.
    s1, s2, _ = _orthogonal_stems(amplitudes=(1.0, 0.25, 1.0))
    r = waveform_similarity(s1 + s2, s1)
    assert r["corr"] == pytest.approx(1.0 / np.sqrt(1.0 + 0.0625), rel=1e-12)
    assert r["nrmse"] == pytest.approx(0.25, rel=1e-12)


def test_trim_drops_edge_samples_of_every_input():
    s1, s2, _ = _orthogonal_stems(amplitudes=(1.0, 0.5, 1.0))
    k = 80  # 49 whole periods of 50 Hz and 98 of 100 Hz remain: orthogonality is kept
    est = s1.copy()
    est[:k] += 5.0 * np.arange(k)  # filter start-up garbage
    est[-k:] -= 5.0
    bad = waveform_similarity(est, s1)
    assert bad["corr"] < 0.9 and bad["nrmse"] > 1.0
    good = waveform_similarity(est, s1, trim=k)
    assert good["corr"] == pytest.approx(1.0, abs=1e-12) and good["nrmse"] == pytest.approx(0.0, abs=1e-12)
    assert good["n"] == s1.size - 2 * k
    # mixture and other are trimmed in the same way
    r = waveform_similarity(est, s1, mixture=est, other=s2, trim=k)
    assert r["baseline_corr"] == pytest.approx(1.0, abs=1e-12)
    assert r["baseline_nrmse"] == pytest.approx(0.0, abs=1e-12)
    assert r["cross_corr"] == pytest.approx(0.0, abs=1e-12)


def test_energy_weighting_is_not_fooled_by_quiet_channels():
    # ROADMAP section 6, item 8: 4 channels, channel 0 holds 99 % of the truth energy.
    s1, s2, _ = _orthogonal_stems()
    a0 = 10.0
    a1 = a0 / np.sqrt(297.0)  # three channels of 1/297 each: 99 % in channel 0
    truth = np.column_stack([a0 * s1, a1 * s1, a1 * s1, a1 * s1])
    # Channel 0 is estimated perfectly; the quiet channels get an unrelated line of equal energy.
    est = np.column_stack([a0 * s1, a1 * s2, a1 * s2, a1 * s2])
    r = waveform_similarity(est, truth)
    assert r["weights"] == pytest.approx([0.99, 0.01 / 3, 0.01 / 3, 0.01 / 3], rel=1e-12)
    assert r["corr_per_channel"] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-12)
    assert r["nrmse_per_channel"] == pytest.approx([0.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)], abs=1e-12)
    assert r["corr"] == pytest.approx(0.99, abs=1e-12)
    assert r["nrmse"] == pytest.approx(0.01 * np.sqrt(2.0), rel=1e-12)
    u = waveform_similarity(est, truth, weighting="uniform")
    assert u["corr"] == pytest.approx(0.25, abs=1e-12)  # the uniform mean underestimates by 4x
    assert u["nrmse"] == pytest.approx(0.75 * np.sqrt(2.0), rel=1e-12)
    assert u["weights"] == pytest.approx([0.25] * 4)
    # per-channel values do not depend on the weighting
    assert u["corr_per_channel"] == pytest.approx(r["corr_per_channel"])


def test_single_channel_2d_input_matches_1d():
    s1, s2, _ = _orthogonal_stems(amplitudes=(1.0, 0.3, 1.0))
    a = waveform_similarity(s1 + s2, s1)
    b = waveform_similarity((s1 + s2)[:, None], s1[:, None])
    assert b["corr"] == pytest.approx(a["corr"]) and b["nrmse"] == pytest.approx(a["nrmse"])


def test_zero_baseline_is_the_mixture_against_the_truth():
    s1, s2, s3 = _orthogonal_stems(amplitudes=(1.0, 0.5, 0.5))
    truth = s1
    mixture = s1 + s2 + s3  # source 1 plus two interferers, each at 0.25 of its energy
    est = s1 + 0.1 * s2  # a good separation
    r = waveform_similarity(est, truth, mixture=mixture)
    assert r["baseline_corr"] == pytest.approx(1.0 / np.sqrt(1.0 + 0.5), rel=1e-12)
    assert r["baseline_nrmse"] == pytest.approx(np.sqrt(0.5), rel=1e-12)
    assert r["corr"] == pytest.approx(1.0 / np.sqrt(1.0 + 0.01 * 0.25), rel=1e-12)
    assert r["corr"] > r["baseline_corr"] and r["nrmse"] < r["baseline_nrmse"]
    assert r["baseline_corr_per_channel"].shape == (1,)
    assert r["baseline_nrmse_per_channel"].shape == (1,)


def test_cross_capture_correlation_with_another_source():
    s1, s2, _ = _orthogonal_stems(amplitudes=(1.0, 1.0, 1.0))
    est = s1 + 0.3 * s2  # source 1 estimate that also captured 30 % of source 2
    r = waveform_similarity(est, s1, other=s2)
    assert r["cross_corr"] == pytest.approx(0.3 / np.sqrt(1.0 + 0.09), rel=1e-12)
    assert r["cross_corr_per_channel"].shape == (1,)
    clean = waveform_similarity(s1, s1, other=s2)
    assert clean["cross_corr"] == pytest.approx(0.0, abs=1e-12)
    assert "baseline_corr" not in r


def test_cross_capture_aggregate_is_weighted_by_the_other_sources_energy():
    s1, s2, s3 = _orthogonal_stems()
    # The other source is only present in channel 1; the estimate captured it there only.
    truth = np.column_stack([s1, s1])
    other = np.column_stack([0.0 * s2, s2])
    est = np.column_stack([s1, s1 + 0.5 * s2])
    r = waveform_similarity(est, truth, other=other)
    assert r["cross_corr_per_channel"][1] == pytest.approx(0.5 / np.sqrt(1.25), rel=1e-12)
    assert np.isnan(r["cross_corr_per_channel"][0])  # other is silent there
    assert r["cross_corr"] == pytest.approx(0.5 / np.sqrt(1.25), rel=1e-12)
    u = waveform_similarity(est, truth, other=other, weighting="uniform")
    assert np.isnan(u["cross_corr"])  # a NaN channel with positive weight propagates


def test_degenerate_channels():
    s1, s2, _ = _orthogonal_stems()
    truth = np.column_stack([s1, np.zeros_like(s1)])  # channel 1 has no truth energy
    est = np.column_stack([s1, np.ones_like(s1)])  # ... and a constant estimate
    r = waveform_similarity(est, truth)
    assert np.isnan(r["corr_per_channel"][1]) and np.isnan(r["nrmse_per_channel"][1])
    assert r["corr"] == pytest.approx(1.0) and r["nrmse"] == pytest.approx(0.0, abs=1e-12)
    assert r["weights"] == pytest.approx([1.0, 0.0])
    u = waveform_similarity(est, truth, weighting="uniform")
    assert np.isnan(u["corr"]) and np.isnan(u["nrmse"])
    # constant estimate against a live truth: correlation undefined
    c = waveform_similarity(np.ones_like(s1), s1)
    assert np.isnan(c["corr"]) and c["nrmse"] == pytest.approx(np.sqrt(3.0))  # rms(1 - s1) / rms(s1)
    z = waveform_similarity(s1, np.zeros_like(s1))
    assert np.isnan(z["corr"]) and np.isnan(z["nrmse"])


def test_waveform_similarity_on_a_run_up_with_noise():
    # truth = order stems of a run-up; estimate = the full mix (orders + tone + noise).
    fs = 8000.0
    t = np.arange(int(4 * fs)) / fs
    sig = synth_runup(
        fs, 600.0 + 600.0 * t, [{"order": 2.0, "amplitude": 1.0}, {"order": 4.0, "amplitude": 0.5}],
        tones=[{"freq": 333.0, "amplitude": 0.3}], noise_rms=0.1, seed=2,
    )
    r = waveform_similarity(sig.x, sig.stems["orders"])
    err = sig.x - sig.stems["orders"]
    assert r["nrmse"] == pytest.approx(np.sqrt(np.mean(err**2) / np.mean(sig.stems["orders"] ** 2)), rel=1e-12)
    assert 0.0 < r["corr"] < 1.0
    expected = np.corrcoef(sig.x, sig.stems["orders"])[0, 1]
    assert r["corr"] == pytest.approx(expected, rel=1e-12)


def test_waveform_similarity_errors():
    a = np.arange(10.0)
    with pytest.raises(ValueError, match="same shape"):
        waveform_similarity(a, a[:-1])
    with pytest.raises(ValueError, match="same shape"):
        waveform_similarity(a, a[:, None])
    with pytest.raises(ValueError, match="mixture"):
        waveform_similarity(a, a, mixture=a[:-1])
    with pytest.raises(ValueError, match="other"):
        waveform_similarity(a, a, other=np.ones((10, 2)))
    with pytest.raises(ValueError, match="weighting"):
        waveform_similarity(a, a, weighting="median")
    with pytest.raises(ValueError, match="trim"):
        waveform_similarity(a, a, trim=-1)
    with pytest.raises(ValueError, match="trim"):
        waveform_similarity(a, a, trim=1.5)
    with pytest.raises(ValueError, match="fewer than 2"):
        waveform_similarity(a, a, trim=5)
    bad = a.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="estimate contains non-finite"):
        waveform_similarity(bad, a)
    with pytest.raises(ValueError, match="truth contains non-finite"):
        waveform_similarity(a, bad)
    with pytest.raises(ValueError, match="mixture contains non-finite"):
        waveform_similarity(a, a, mixture=bad)
    with pytest.raises(ValueError, match="shape"):
        waveform_similarity(np.ones((4, 3, 2)), np.ones((4, 3, 2)))


# ==========================================================================
# order_residual_energy
# ==========================================================================

FS_R = 8000.0
_T_R = np.arange(int(4 * FS_R)) / FS_R
RPM_R = 600.0 + 600.0 * _T_R  # 600 -> 3000 rpm in 4 s
_ORDERS_R = [
    {"order": 1.0, "amplitude": 1.0},
    # order 2 grows with speed (0.5 -> 2), so its energy differs between frames
    {"order": 2.0, "amplitude": 1.0, "gain_rpm": [600.0, 3000.0], "gain": [0.5, 2.0]},
    {"order": 3.5, "amplitude": 0.8},
    {"order": 5.0, "amplitude": 0.5},
]


@pytest.fixture(scope="module")
def runup():
    return synth_runup(FS_R, RPM_R, _ORDERS_R, seed=1)


@pytest.mark.parametrize("order", [1.0, 2.0, 3.5, 5.0])
@pytest.mark.parametrize("scale, db", [(0.1, -20.0), (0.01, -40.0), (1.0, 0.0)])
def test_scaled_residual_gives_the_scale_in_db(runup, order, scale, db):
    r = order_residual_energy(scale * runup.x, runup.x, FS_R, RPM_R, order)
    assert r == pytest.approx(db, abs=0.05)


def test_residual_with_the_order_at_minus_30_db_and_other_orders(runup):
    S = runup.order_stems
    target = 10.0 ** (-30.0 / 20.0) * S[:, 1]
    others = S[:, [0, 2, 3]].sum(axis=1)  # untouched orders 1, 3.5 and 5 at full level
    alone = order_residual_energy(target, runup.x, FS_R, RPM_R, 2.0)
    with_others = order_residual_energy(target + others, runup.x, FS_R, RPM_R, 2.0)
    assert alone == pytest.approx(-30.0, abs=0.05)
    assert with_others == pytest.approx(-30.0, abs=0.05)
    assert with_others == pytest.approx(alone, abs=0.01)  # unaffected by the other orders
    # same with the true order stem as reference
    vs_stem = order_residual_energy(target + others, S[:, 1], FS_R, RPM_R, 2.0)
    assert vs_stem == pytest.approx(-30.0, abs=0.05)


def test_partial_removal_is_measured_by_the_remaining_amplitude(runup):
    S = runup.order_stems
    r = order_residual_energy(runup.x - 0.9 * S[:, 1], runup.x, FS_R, RPM_R, 2.0)
    assert r == pytest.approx(-20.0, abs=0.05)  # 10 % amplitude left = 1 % energy


def test_perfect_subtraction_leaves_nothing(runup):
    S = runup.order_stems
    clean = order_residual_energy(runup.x - S[:, 1], runup.x, FS_R, RPM_R, 2.0)
    assert clean < -60.0  # ROADMAP T12 asks for <= -20 dB
    for k, order in enumerate([1.0, 3.5, 5.0]):
        col = [0, 2, 3][k]
        assert order_residual_energy(runup.x - S[:, col], runup.x, FS_R, RPM_R, order) < -60.0
    # Broadband noise in the analysis band sets the floor.
    noisy = synth_runup(FS_R, RPM_R, _ORDERS_R, noise_rms=0.01, seed=1)
    floor = order_residual_energy(noisy.x - noisy.order_stems[:, 1], noisy.x, FS_R, RPM_R, 2.0)
    assert -90.0 < floor < -50.0
    # no order removed: the residual still holds everything
    assert order_residual_energy(runup.x, runup.x, FS_R, RPM_R, 2.0) == pytest.approx(0.0, abs=1e-9)


def test_a_different_order_is_not_counted(runup):
    S = runup.order_stems
    # The residual holds only order 2; at order 3.5 (and 1) there is nothing left.
    assert order_residual_energy(S[:, 1], runup.x, FS_R, RPM_R, 3.5) < -60.0
    assert order_residual_energy(S[:, 1], runup.x, FS_R, RPM_R, 1.0) < -60.0


def test_independent_of_unit_and_of_a_common_gain(runup):
    a = order_residual_energy(0.1 * runup.x, runup.x, FS_R, RPM_R, 2.0)
    b = order_residual_energy(1000.0 * 0.1 * runup.x, 1000.0 * runup.x, FS_R, RPM_R, 2.0)
    assert a == pytest.approx(b, abs=1e-9)


def test_rpm_range_selects_frames(runup):
    t_step = 2.0  # rpm 1800
    scale = np.where(_T_R < t_step, 0.1, 0.01)
    res = scale * runup.x
    low = order_residual_energy(res, runup.x, FS_R, RPM_R, 2.0, rpm_range=(None, 1400.0))
    high = order_residual_energy(res, runup.x, FS_R, RPM_R, 2.0, rpm_range=(2200.0, None))
    assert low == pytest.approx(-20.0, abs=0.05)
    assert high == pytest.approx(-40.0, abs=0.05)
    both = order_residual_energy(res, runup.x, FS_R, RPM_R, 2.0)
    assert -40.0 < both < -20.0
    full = order_residual_energy(res, runup.x, FS_R, RPM_R, 2.0, rpm_range=(0.0, 1e9))
    assert full == pytest.approx(both, abs=1e-12)
    open_ = order_residual_energy(res, runup.x, FS_R, RPM_R, 2.0, rpm_range=(None, None))
    assert open_ == pytest.approx(both, abs=1e-12)


def test_energy_is_summed_over_channels(runup):
    ref = np.column_stack([runup.x, 0.5 * runup.x])  # channel 1 holds 1/4 of the energy
    res = np.column_stack([0.1 * ref[:, 0], 0.5 * ref[:, 1]])
    expected = 10.0 * np.log10((0.01 * 1.0 + 0.25 * 0.25) / 1.25)
    assert order_residual_energy(res, ref, FS_R, RPM_R, 2.0) == pytest.approx(expected, abs=0.05)


def test_constant_speed_and_scalar_rpm():
    sig = synth_runup(FS_R, np.full(int(2 * FS_R), 2400.0), _ORDERS_R, seed=3)
    r = order_residual_energy(0.1 * sig.x, sig.x, FS_R, 2400.0, 2.0)  # scalar rpm
    assert r == pytest.approx(-20.0, abs=0.05)


def test_residual_without_energy_is_minus_infinity(runup):
    assert order_residual_energy(np.zeros_like(runup.x), runup.x, FS_R, RPM_R, 2.0) == -np.inf


def test_order_residual_energy_errors(runup):
    x = runup.x
    with pytest.raises(ValueError, match="same shape"):
        order_residual_energy(x[:-1], x, FS_R, RPM_R, 2.0)
    with pytest.raises(ValueError, match="order must be positive"):
        order_residual_energy(x, x, FS_R, RPM_R, 0.0)
    with pytest.raises(ValueError, match="order must be positive"):
        order_residual_energy(x, x, FS_R, RPM_R, -2.0)
    with pytest.raises(ValueError, match="outside the order axis"):
        order_residual_energy(x, x, FS_R, RPM_R, 500.0)  # far above fs / (2 * rpm / 60)
    with pytest.raises(ValueError, match="no energy"):
        order_residual_energy(x, np.zeros_like(x), FS_R, RPM_R, 2.0)
    with pytest.raises(ValueError, match="no frame selected"):
        order_residual_energy(x, x, FS_R, RPM_R, 2.0, rpm_range=(5000.0, 6000.0))
    with pytest.raises(ValueError, match="lo <= hi"):
        order_residual_energy(x, x, FS_R, RPM_R, 2.0, rpm_range=(2000.0, 1000.0))
    with pytest.raises(ValueError, match="pair"):
        order_residual_energy(x, x, FS_R, RPM_R, 2.0, rpm_range=1000.0)
    bad = x.copy()
    bad[10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        order_residual_energy(bad, x, FS_R, RPM_R, 2.0)


# ==========================================================================
# window_checks
# ==========================================================================

L_WIN = 800


def _hann(length=L_WIN):
    return signal.get_window("hann", length)  # periodic, as used for overlap-add


def test_hann_at_50_percent_is_cola_but_not_power_complementary():
    r = window_checks(_hann(), L_WIN // 2)
    assert r["cola"] is True and r["power_complementary"] is False
    assert r["cola_ripple_db"] == pytest.approx(0.0, abs=1e-9)
    # sin^4 + cos^4 = 1 - sin^2(2 theta) / 2 swings between 1 and 1/2: 10 log10(2) dB
    assert r["power_ripple_db"] == pytest.approx(10 * np.log10(2.0), abs=1e-9)
    assert r["cola_constant"] == pytest.approx(1.0, abs=1e-12)
    assert r["power_constant"] == pytest.approx(0.75, abs=1e-12)  # mean of sin^4 + cos^4


def test_sqrt_hann_at_50_percent_is_power_complementary_but_not_cola():
    r = window_checks(np.sqrt(_hann()), L_WIN // 2)
    assert r["cola"] is False and r["power_complementary"] is True
    assert r["power_ripple_db"] == pytest.approx(0.0, abs=1e-9)
    assert r["power_constant"] == pytest.approx(1.0, abs=1e-12)  # sin^2 + cos^2
    # sin + cos swings between 1 and sqrt(2): 20 log10(sqrt(2)) dB, mean 4 / pi
    assert r["cola_ripple_db"] == pytest.approx(20 * np.log10(np.sqrt(2.0)), abs=1e-9)
    assert r["cola_constant"] == pytest.approx(4.0 / np.pi, abs=1e-4)


def test_hann_at_75_and_66_percent_overlap_is_both():
    r75 = window_checks(_hann(), L_WIN // 4)
    assert r75["cola"] and r75["power_complementary"]
    assert r75["cola_constant"] == pytest.approx(2.0, abs=1e-12)
    assert r75["power_constant"] == pytest.approx(1.5, abs=1e-12)
    r67 = window_checks(_hann(900), 300)  # hop = L / 3
    assert r67["cola"] and r67["power_complementary"]
    assert r67["cola_constant"] == pytest.approx(1.5, abs=1e-12)
    assert r67["power_constant"] == pytest.approx(1.125, abs=1e-12)  # 3 * 3/8


def test_sqrt_hann_at_75_percent_overlap_is_not_cola():
    r = window_checks(np.sqrt(_hann()), L_WIN // 4)
    assert r["power_complementary"] and r["power_constant"] == pytest.approx(2.0, abs=1e-12)
    assert not r["cola"] and 0.5 < r["cola_ripple_db"] < 1.0


def test_rectangular_window_and_a_poor_hop():
    r = window_checks(np.ones(L_WIN), L_WIN // 2)
    assert r["cola"] and r["power_complementary"]
    assert r["cola_constant"] == pytest.approx(2.0) and r["power_constant"] == pytest.approx(2.0)
    bad = window_checks(_hann(), 480)  # 40 % overlap: neither
    assert not bad["cola"] and not bad["power_complementary"]
    assert bad["cola_ripple_db"] > 1.0 and bad["power_ripple_db"] > 1.0
    gap = window_checks(_hann(), L_WIN + 100)  # frames do not touch: the sum reaches zero
    assert gap["cola_ripple_db"] == np.inf and gap["power_ripple_db"] == np.inf
    assert not gap["cola"] and not gap["power_complementary"]


def test_symmetric_hann_is_not_exactly_cola():
    # numpy.hanning repeats the zero at both ends: a 0.017 dB ripple, above the 0.01 dB tolerance
    r = window_checks(np.hanning(L_WIN), L_WIN // 2)
    assert r["cola"] is False
    assert 0.01 < r["cola_ripple_db"] < 0.03


def test_window_checks_equals_the_closed_form_phase_sums():
    # In the steady region the sum at phase p of the hop is sum_j w[p + j hop].
    rng = np.random.default_rng(3)
    w = rng.uniform(0.2, 1.0, 37)
    hop = 5
    s1 = np.array([w[p::hop].sum() for p in range(hop)])
    s2 = np.array([(w[p::hop] ** 2).sum() for p in range(hop)])
    r = window_checks(w, hop)
    assert r["cola_ripple_db"] == pytest.approx(20 * np.log10(s1.max() / s1.min()), abs=1e-12)
    assert r["power_ripple_db"] == pytest.approx(10 * np.log10(s2.max() / s2.min()), abs=1e-12)
    assert r["cola_constant"] == pytest.approx(s1.mean(), rel=1e-12)
    assert r["power_constant"] == pytest.approx(s2.mean(), rel=1e-12)


def test_window_checks_is_independent_of_scale_and_of_n_frames():
    a = window_checks(_hann(), 400)
    b = window_checks(3.0 * _hann(), 400, n_frames=7)
    assert b["cola_ripple_db"] == pytest.approx(a["cola_ripple_db"], abs=1e-12)
    assert b["power_ripple_db"] == pytest.approx(a["power_ripple_db"], abs=1e-12)
    assert b["cola_constant"] == pytest.approx(3.0 * a["cola_constant"])
    assert b["power_constant"] == pytest.approx(9.0 * a["power_constant"])
    # a small n_frames is raised until a steady region exists (L / hop = 80)
    fine = window_checks(_hann(), 10, n_frames=2)
    assert fine["cola"] and fine["power_complementary"]
    assert fine["cola_constant"] == pytest.approx(L_WIN / 2 / 10)
    assert isinstance(a["cola"], bool) and isinstance(a["cola_ripple_db"], float)


def test_window_checks_errors():
    w = _hann(64)
    with pytest.raises(ValueError, match="1-D"):
        window_checks(np.ones((4, 4)), 2)
    with pytest.raises(ValueError, match="at least 2"):
        window_checks([1.0], 1)
    with pytest.raises(ValueError, match="finite"):
        window_checks([1.0, np.nan, 1.0], 1)
    with pytest.raises(ValueError, match="all zero"):
        window_checks(np.zeros(8), 2)
    with pytest.raises(ValueError, match="hop"):
        window_checks(w, 0)
    with pytest.raises(ValueError, match="hop"):
        window_checks(w, 2.5)
    with pytest.raises(ValueError, match="n_frames"):
        window_checks(w, 32, n_frames=1)


# ==========================================================================
# modulation_peak
# ==========================================================================

FS_M = 16000.0
N_M = 160000
T_M = np.arange(N_M) / FS_M


def _white(rms, seed, n=N_M, fs=FS_M):
    return synth_runup(fs, np.full(n, 1000.0), [], noise_rms=rms, seed=seed).x


def _am_tone_plus_noise(depth, rms, rate=40.0, seed=3):
    """cos(2 pi 2000 t) (1 + depth cos(2 pi rate t)) + white noise of the given RMS."""
    tone = synth_runup(FS_M, np.full(N_M, 1000.0), [], tones=[{"freq": 2000.0, "amplitude": 1.0}], seed=0).x
    return tone * (1.0 + depth * np.cos(2 * np.pi * rate * T_M)) + _white(rms, seed)


@pytest.mark.parametrize("half_width", [1.0, 2.0])
@pytest.mark.parametrize("depth", [0.1, 0.2, 0.3, 0.5])
def test_modulation_peak_of_an_am_tone_in_noise_matches_the_analytic_level(depth, half_width):
    # Carrier A = 1 (peak), white noise of one-sided PSD N0 = rms^2 / (fs / 2). The power
    # envelope is (1 + a cos)^2 A^2 + 2 A (1 + a cos) n_I + ..., n_I the in-phase noise:
    #   line at the rate: 2 a A^2 peak, PSD peak (2 a A^2)^2 / 2 / (1.5 df)  (Hann, df = bin spacing)
    #   floor: 8 A^2 N0 (1 + a^2 / 2)   (flat, one-sided, noise-noise terms neglected)
    # reading = 10 log10(1 + peak / floor) = 10 log10(1 + a^2 A^2 / (6 df N0 (1 + a^2 / 2))).
    rms = 0.3
    n0 = rms**2 / (FS_M / 2)
    expected = 10 * np.log10(1 + depth**2 / (6 * half_width * n0 * (1 + depth**2 / 2)))
    got = modulation_peak(_am_tone_plus_noise(depth, rms), FS_M, 40.0, half_width=half_width)
    assert got == pytest.approx(expected, abs=0.5)
    assert got > 15.0


def test_modulation_peak_is_localised_at_the_rate_and_independent_of_scale():
    x = _am_tone_plus_noise(0.3, 0.3)
    at = modulation_peak(x, FS_M, 40.0)
    assert at > 20.0
    for other in (25.0, 55.0, 70.0):  # no modulation there: a noise-level reading
        assert abs(modulation_peak(x, FS_M, other)) < 4.0
    assert modulation_peak(1e3 * x, FS_M, 40.0) == pytest.approx(at, abs=1e-9)
    # a rate that falls between bins of the 1 Hz spectrum is read as well
    y = _am_tone_plus_noise(0.3, 0.3, rate=37.3)
    assert modulation_peak(y, FS_M, 37.3) > 20.0


def test_modulation_peak_reads_zero_decibels_for_stationary_noise():
    # statistic with a spread of about 1 dB (10 s, half_width 1 Hz): zero mean, 4 sigma bound
    values = [modulation_peak(_white(1.0, s), FS_M, 40.0) for s in range(8)]
    assert abs(np.mean(values)) < 1.0
    assert max(abs(v) for v in values) < 4.0


def test_modulation_peak_other_rates_bands_and_sampling_rates():
    x = _am_tone_plus_noise(0.5, 0.3)
    assert modulation_peak(x, FS_M, 40.0, band=(1500.0, 2500.0)) > 15.0
    assert modulation_peak(x, FS_M, 40.0, env_fs=400.0, half_width=2.0) > 15.0
    # a sampling rate that is not an integer multiple of env_fs (44100 / 200 = 220.5)
    fs = 44100.0
    n = 441000
    t = np.arange(n) / fs
    tone = synth_runup(fs, np.full(n, 1000.0), [], tones=[{"freq": 2000.0, "amplitude": 1.0}], seed=0).x
    y = tone * (1 + 0.5 * np.cos(2 * np.pi * 40.0 * t)) + _white(0.3, 4, n=n, fs=fs)
    assert modulation_peak(y, fs, 40.0) > 15.0
    assert abs(modulation_peak(y, fs, 61.0)) < 4.0


def test_modulation_peak_without_envelope_fluctuation():
    assert np.isnan(modulation_peak(np.zeros(N_M), FS_M, 40.0))  # nothing to compare: NaN, no warning
    # a pure tone has a constant envelope: the spectrum holds numerical noise only, no line
    tone = synth_runup(FS_M, np.full(N_M, 1000.0), [], tones=[{"freq": 2000.0, "amplitude": 1.0}], seed=0).x
    assert abs(modulation_peak(tone, FS_M, 40.0)) < 4.0


def test_modulation_peak_errors():
    x = _white(1.0, 0)
    with pytest.raises(ValueError, match="1-D"):
        modulation_peak(np.ones((100, 2)), FS_M, 40.0)
    with pytest.raises(ValueError, match="finite"):
        modulation_peak(np.r_[x[:-1], np.nan], FS_M, 40.0)
    with pytest.raises(ValueError, match="rate must be positive"):
        modulation_peak(x, FS_M, 0.0)
    with pytest.raises(ValueError, match="rate must be below"):
        modulation_peak(x, FS_M, 90.0)  # env_fs 200 Hz: limit 80 Hz
    with pytest.raises(ValueError, match="above 0 Hz"):
        modulation_peak(x, FS_M, 2.5)  # within 3 bins of DC
    with pytest.raises(ValueError, match="band"):
        modulation_peak(x, FS_M, 40.0, band=(4000.0, 1000.0))
    with pytest.raises(ValueError, match="band"):
        modulation_peak(x, FS_M, 40.0, band=(1000.0, 9000.0))
    with pytest.raises(ValueError, match="env_fs must be below fs"):
        modulation_peak(x, 150.0, 40.0, band=(10.0, 60.0), env_fs=200.0)
    with pytest.raises(ValueError, match="too short"):
        modulation_peak(x[:8000], FS_M, 40.0, half_width=0.2)  # needs 2 / 0.2 = 10 s
    with pytest.raises(ValueError, match="half_width"):
        modulation_peak(x, FS_M, 40.0, half_width=-1.0)


# ==========================================================================
# envelope_stats
# ==========================================================================

FS_E = 8000.0


def _rayleigh_db(p):
    """Envelope percentile of unit-RMS Gaussian noise (Rayleigh, sigma = 1) in dB re RMS."""
    return 20 * np.log10(np.sqrt(-2 * np.log(1 - p / 100.0)))


def test_pure_sine_has_no_swing_and_a_crest_factor_of_3_db():
    n = int(2 * FS_E)
    x = synth_runup(FS_E, np.full(n, 1000.0), [], tones=[{"freq": 1000.0, "amplitude": 2.0, "phase": 0.0}]).x
    r = envelope_stats(x, FS_E)
    assert r["swing_db"] == pytest.approx(0.0, abs=1e-6)
    assert r["crest_factor_db"] == pytest.approx(20 * np.log10(np.sqrt(2.0)), abs=1e-3)  # 3.01 dB
    # the envelope is the constant peak amplitude A: A / (A / sqrt(2)) = +3.01 dB re RMS
    assert list(r["percentiles_db"]) == [5.0, 95.0]
    assert r["percentiles_db"][5.0] == pytest.approx(3.0103, abs=1e-3)
    assert r["percentiles_db"][95.0] == pytest.approx(3.0103, abs=1e-3)
    assert "spectral_error_db" not in r
    assert envelope_stats(1e4 * x, FS_E)["crest_factor_db"] == pytest.approx(r["crest_factor_db"], abs=1e-9)


def test_two_equal_tones_beat_with_an_analytic_swing():
    # envelope 2 A |cos(pi df t)|: 20 log10 of the 95th over the 5th percentile of |cos|
    # is 20 log10(sin(0.95 pi / 2) / sin(0.05 pi / 2)) = 22.07 dB; peak 2A, RMS A: 6.02 dB crest
    n = int(2 * FS_E)  # 20 whole beats of 10 Hz, whole cycles of both tones
    x = synth_runup(FS_E, np.full(n, 1000.0), [],
                    tones=[{"freq": 1000.0, "amplitude": 1.0, "phase": 0.0},
                           {"freq": 1010.0, "amplitude": 1.0, "phase": 0.0}]).x  # in phase at t = 0
    r = envelope_stats(x, FS_E)
    expected = 20 * np.log10(np.sin(0.95 * np.pi / 2) / np.sin(0.05 * np.pi / 2))
    assert r["swing_db"] == pytest.approx(expected, abs=0.1)
    assert r["crest_factor_db"] == pytest.approx(20 * np.log10(2.0), abs=1e-6)  # peak 2 A sampled at t = 0
    assert r["percentiles_db"][95.0] == pytest.approx(20 * np.log10(2 * np.sin(0.95 * np.pi / 2)), abs=0.05)
    # a tone plus a weak second tone: small swing, growing with the second tone
    weak = synth_runup(FS_E, np.full(n, 1000.0), [],
                       tones=[{"freq": 1000.0, "amplitude": 1.0}, {"freq": 1010.0, "amplitude": 0.1}], seed=2).x
    assert 0.0 < envelope_stats(weak, FS_E)["swing_db"] < r["swing_db"] / 2


def test_gaussian_noise_has_a_rayleigh_envelope():
    x = synth_runup(FS_E, np.full(400000, 1000.0), [], noise_rms=3.0, seed=4).x
    r = envelope_stats(x, FS_E)
    assert r["percentiles_db"][5.0] == pytest.approx(_rayleigh_db(5), abs=0.3)  # -9.89 dB
    assert r["percentiles_db"][95.0] == pytest.approx(_rayleigh_db(95), abs=0.3)  # +7.77 dB
    assert r["swing_db"] == pytest.approx(_rayleigh_db(95) - _rayleigh_db(5), abs=0.3)  # 17.66 dB
    assert 12.0 < r["crest_factor_db"] < 15.5  # peak of 400k Gaussian samples: 4.4 - 5.3 sigma
    # band-limited Gaussian noise is Gaussian: same envelope law inside the band
    rb = envelope_stats(x, FS_E, band=(500.0, 2500.0))
    assert rb["swing_db"] == pytest.approx(_rayleigh_db(95) - _rayleigh_db(5), abs=0.4)
    assert rb["percentiles_db"][5.0] == pytest.approx(_rayleigh_db(5), abs=0.4)


def test_custom_percentiles_and_band_applied_to_a_tone_in_noise():
    x = synth_runup(FS_E, np.full(400000, 1000.0), [], noise_rms=1.0, seed=5).x
    r = envelope_stats(x, FS_E, percentiles=(10, 50, 90))
    assert list(r["percentiles_db"]) == [10.0, 50.0, 90.0]
    for p in (10, 50, 90):
        assert r["percentiles_db"][float(p)] == pytest.approx(_rayleigh_db(p), abs=0.3)
    assert r["swing_db"] == pytest.approx(_rayleigh_db(90) - _rayleigh_db(10), abs=0.3)
    # percentiles given in any order: the swing spans the extremes
    assert envelope_stats(x, FS_E, percentiles=(95, 5))["swing_db"] == pytest.approx(
        envelope_stats(x, FS_E)["swing_db"], abs=1e-12
    )
    # a strong tone in the band, noise outside it: the band-passed signal is the tone
    n = x.size
    tone = synth_runup(FS_E, np.full(n, 1000.0), [], tones=[{"freq": 1000.0, "amplitude": 1.0}], seed=1).x
    r_band = envelope_stats(tone + 0.3 * x, FS_E, band=(900.0, 1100.0))
    assert r_band["swing_db"] < 2.0 < envelope_stats(tone + 0.3 * x, FS_E)["swing_db"]


def test_spectral_error_against_a_reference():
    x = synth_runup(FS_E, np.full(80000, 1000.0), [], noise_rms=1.0, seed=6).x
    other = synth_runup(FS_E, np.full(80000, 1000.0), [], noise_rms=1.0, seed=7).x
    assert envelope_stats(x, FS_E, reference=x)["spectral_error_db"] == pytest.approx(0.0, abs=1e-12)
    # a gain of 2 is a PSD ratio of 4: 6.02 dB at every frequency
    assert envelope_stats(2.0 * x, FS_E, reference=x)["spectral_error_db"] == pytest.approx(
        20 * np.log10(2.0), abs=1e-9
    )
    assert envelope_stats(x, FS_E, reference=0.5 * x)["spectral_error_db"] == pytest.approx(
        20 * np.log10(2.0), abs=1e-9
    )
    # two independent realisations of the same noise: only the estimator scatter, about
    # 6.1 dB / sqrt(K) with K = 99 segments of the default 0.2 s
    e = envelope_stats(x, FS_E, reference=other)["spectral_error_db"]
    assert 0.3 < e < 1.0
    # shorter reference, explicit nperseg
    e2 = envelope_stats(x, FS_E, reference=other[:40000], nperseg=800)["spectral_error_db"]
    assert 0.3 < e2 < 1.3
    # the error is restricted to the band: a tone outside it does not count
    tone = synth_runup(FS_E, np.full(80000, 1000.0), [], tones=[{"freq": 3000.0, "amplitude": 10.0}], seed=1).x
    assert envelope_stats(x + tone, FS_E, reference=x)["spectral_error_db"] > 1.5
    assert envelope_stats(x + tone, FS_E, reference=x, band=(500.0, 1500.0))["spectral_error_db"] < 0.05


def test_spectral_error_with_nothing_to_compare_is_nan():
    x = synth_runup(FS_E, np.full(16000, 1000.0), [], noise_rms=1.0, seed=8).x
    zeros = np.zeros(16000)
    assert np.isnan(envelope_stats(x, FS_E, reference=zeros)["spectral_error_db"])


def test_envelope_stats_errors():
    x = synth_runup(FS_E, np.full(16000, 1000.0), [], noise_rms=1.0, seed=9).x
    with pytest.raises(ValueError, match="1-D"):
        envelope_stats(np.ones((100, 2)), FS_E)
    with pytest.raises(ValueError, match="finite"):
        envelope_stats(np.r_[x[:-1], np.inf], FS_E)
    with pytest.raises(ValueError, match="fs must be positive"):
        envelope_stats(x, 0.0)
    with pytest.raises(ValueError, match="no energy"):
        envelope_stats(np.zeros(100), FS_E)
    with pytest.raises(ValueError, match="no energy"):
        envelope_stats(np.full(100, 3.0), FS_E)  # a constant has no AC part
    with pytest.raises(ValueError, match="percentiles"):
        envelope_stats(x, FS_E, percentiles=(50,))
    with pytest.raises(ValueError, match="percentiles"):
        envelope_stats(x, FS_E, percentiles=(5, 105))
    with pytest.raises(ValueError, match="distinct"):
        envelope_stats(x, FS_E, percentiles=(5, 5))
    with pytest.raises(ValueError, match="band"):
        envelope_stats(x, FS_E, band=(1000.0, 5000.0))
    with pytest.raises(ValueError, match="band"):
        envelope_stats(x, FS_E, band=1000.0)
    with pytest.raises(ValueError, match="reference"):
        envelope_stats(x, FS_E, reference=np.ones((10, 2)))
    with pytest.raises(ValueError, match="nperseg"):
        envelope_stats(x, FS_E, reference=x, nperseg=4)
