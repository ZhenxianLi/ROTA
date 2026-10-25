"""Tests of rota.metrics.track_error with hand-computed cases."""

import numpy as np
import pytest

from rota.metrics import track_error


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
