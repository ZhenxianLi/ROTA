"""Tests of rota.metrics against hand-computed and rota.synth ground truth.

track_error: hand-computed cases. waveform_similarity and order_residual_energy:
analytic values from components rendered by rota.synth (orthogonal order stems
at constant speed, scaled copies of a run-up).
"""

import numpy as np
import pytest

from rota.metrics import order_residual_energy, track_error, waveform_similarity
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
