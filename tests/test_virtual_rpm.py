"""Tests of rota.speed.virtual_rpm against hand values and the closed-form step response."""

import numpy as np
import pytest

from rota.speed import speed_to_rpm, virtual_rpm

K, IDLE, REDLINE = 30.0, 800.0, 6000.0  # rpm per km/h, rpm, rpm


def _v(speed):
    return virtual_rpm(speed, K, idle=IDLE, redline=REDLINE)


# --------------------------------------------------------------------------
# clip(k v, idle, redline)
# --------------------------------------------------------------------------


def test_hand_values():
    #          standstill, below idle, linear, linear, linear, above redline
    v = [0.0, 10.0, 30.0, 50.0, 100.0, 250.0]
    assert np.array_equal(_v(v), [800.0, 800.0, 900.0, 1500.0, 3000.0, 6000.0])


def test_saturation_speeds_are_idle_over_k_and_redline_over_k():
    v_idle, v_red = IDLE / K, REDLINE / K  # 26.667 and 200 km/h
    eps = 1e-6
    r = _v([v_idle - eps, v_idle, v_idle + eps, v_red - eps, v_red, v_red + eps])
    assert r[0] == IDLE and r[1] == IDLE
    assert r[2] == pytest.approx(IDLE + K * eps)  # leaves idle linearly with slope k
    assert r[3] == pytest.approx(REDLINE - K * eps)
    assert r[4] == pytest.approx(REDLINE) and r[5] == REDLINE
    v = np.linspace(0.0, 300.0, 3001)
    rpm = _v(v)
    assert np.all(np.diff(rpm) >= 0)
    assert rpm.min() == IDLE and rpm.max() == REDLINE
    lin = (v > v_idle) & (v < v_red)
    assert np.allclose(rpm[lin], K * v[lin])


def test_negative_speed_maps_to_idle_and_units_are_kmh():
    assert _v(-20.0) == IDLE
    # 100 km/h is 27.78 m/s: feeding m/s would give 27.78 * 30 = 833 rpm, not 3000 rpm
    assert _v(100.0) == pytest.approx(3000.0)
    assert _v(100.0 / 3.6) == pytest.approx(833.333, abs=1e-3)


def test_same_map_as_the_lumped_model_of_speed_to_rpm():
    v = np.linspace(-5.0, 260.0, 500)
    expected = speed_to_rpm(v, rpm_per_kmh=K, idle=IDLE, redline=REDLINE)
    assert np.array_equal(_v(v), expected)


def test_scalar_array_and_multichannel_inputs_and_inputs_are_left_alone():
    assert np.shape(_v(80.0)) == () and float(_v(80.0)) == pytest.approx(2400.0)
    v = np.array([[0.0, 50.0], [100.0, 300.0]])
    before = v.copy()
    out = _v(v)
    assert out.shape == v.shape
    assert np.array_equal(out, [[800.0, 1500.0], [3000.0, 6000.0]])
    assert np.array_equal(v, before)
    assert virtual_rpm([10, 50], 30, idle=800, redline=6000).dtype == float  # integers accepted


def test_no_upper_limit_with_an_infinite_redline_and_equal_limits_hold_a_constant_rpm():
    assert virtual_rpm(500.0, K, idle=IDLE, redline=np.inf) == pytest.approx(15000.0)
    assert np.all(virtual_rpm([0.0, 50.0, 500.0], K, idle=1200.0, redline=1200.0) == 1200.0)


def test_nan_speed_stays_nan_without_smoothing():
    r = _v([0.0, np.nan, 100.0])
    assert r[0] == IDLE and np.isnan(r[1]) and r[2] == 3000.0


# --------------------------------------------------------------------------
# causal first-order smoothing
# --------------------------------------------------------------------------

FS, TAU = 1000.0, 0.2  # Hz, s


def _step(n_before=1000, n_after=2000, high=100.0):
    return np.concatenate([np.zeros(n_before), np.full(n_after, high)])


def test_step_response_is_the_exponential_with_the_given_time_constant():
    v = _step()
    r = virtual_rpm(v, K, idle=IDLE, redline=REDLINE, smooth=TAU, fs=FS)
    lo, hi = IDLE, K * 100.0  # 800 -> 3000 rpm
    alpha = np.exp(-1.0 / (TAU * FS))
    j = np.arange(2000)
    # the discrete response equals 1 - exp(-t / tau) at the samples (t counted from the step sample)
    assert np.allclose(r[1000:], hi - (hi - lo) * alpha ** (j + 1), rtol=0, atol=1e-9)
    assert np.allclose(r[1000:], hi - (hi - lo) * np.exp(-(j + 1) / FS / TAU), rtol=0, atol=1e-9)
    # 63.2 % of the step is reached after one time constant, to within a sample (the threshold
    # is hit exactly at t = tau, so rounding may put the first crossing one sample later)
    t63 = (np.argmax(r >= lo + (1 - np.exp(-1)) * (hi - lo)) - 1000 + 1) / FS
    assert t63 == pytest.approx(TAU, abs=1.5 / FS)
    # 95 % after three
    t95 = (np.argmax(r >= lo + 0.95 * (hi - lo)) - 1000 + 1) / FS
    assert t95 == pytest.approx(3.0 * TAU, abs=0.01)


def test_smoothing_is_causal_and_starts_without_a_transient():
    v = _step()
    r = virtual_rpm(v, K, idle=IDLE, redline=REDLINE, smooth=TAU, fs=FS)
    assert np.all(r[:1000] == IDLE)  # nothing happens before the step
    assert r[1000] > IDLE  # and the first sample after it already moves
    # a constant start value passes unchanged from the first sample (initial state = r[0])
    c = virtual_rpm(np.full(50, 60.0), K, idle=IDLE, redline=REDLINE, smooth=TAU, fs=FS)
    assert np.array_equal(c, np.full(50, 1800.0))
    s = virtual_rpm(np.full(50, 400.0), K, idle=IDLE, redline=REDLINE, smooth=TAU, fs=FS)
    assert np.array_equal(s, np.full(50, REDLINE))


def test_smoothing_lags_a_ramp_by_the_time_constant():
    # in steady state a first-order lag follows a ramp rpm = a t with a delay of tau
    t = np.arange(5000) / FS
    v = 40.0 + 10.0 * t  # 1200 -> 2200 rpm at slope 300 rpm/s, inside the limits
    r = virtual_rpm(v, K, idle=IDLE, redline=REDLINE, smooth=TAU, fs=FS)
    target = K * v
    lag = (target - r)[-1] / (K * 10.0)  # rpm error over slope in rpm/s
    alpha = np.exp(-1.0 / (TAU * FS))
    assert lag == pytest.approx(alpha / (1.0 - alpha) / FS, rel=1e-6)  # exact for the recursion
    assert lag == pytest.approx(TAU - 0.5 / FS, abs=1e-5)  # a delay of tau, less half a sample
    # the lag is a delay, not an attenuation of slow motion: the slope is unchanged
    assert np.allclose(np.diff(r)[-500:], K * 10.0 / FS, rtol=1e-3)


def test_smoothed_speed_stays_between_idle_and_redline_and_clips_before_smoothing():
    rng = np.random.default_rng(3)
    v = np.abs(np.cumsum(rng.normal(0.0, 5.0, 5000))) * 3.0  # wanders far beyond both limits
    r = virtual_rpm(v, K, idle=IDLE, redline=REDLINE, smooth=0.05, fs=FS)
    assert r.min() >= IDLE and r.max() <= REDLINE
    assert (v * K > REDLINE).any() and (v * K < IDLE).any()
    # the limits are respected through the lag: a long stay above redline settles at redline
    hold = virtual_rpm(np.r_[np.zeros(100), np.full(3000, 500.0)], K, idle=IDLE, redline=REDLINE,
                       smooth=TAU, fs=FS)
    assert hold[-1] == pytest.approx(REDLINE, abs=1e-2) and hold.max() <= REDLINE


def test_smoothing_runs_along_the_first_axis_of_each_channel():
    v = np.column_stack([_step(), _step(high=50.0)])
    r = virtual_rpm(v, K, idle=IDLE, redline=REDLINE, smooth=TAU, fs=FS)
    assert r.shape == v.shape
    for j in range(2):
        single = virtual_rpm(v[:, j], K, idle=IDLE, redline=REDLINE, smooth=TAU, fs=FS)
        assert np.array_equal(r[:, j], single)


def test_gaps_are_bridged_for_the_smoothing_only():
    v = _step()
    v[1500:1503] = np.nan
    r = virtual_rpm(v, K, idle=IDLE, redline=REDLINE, smooth=TAU, fs=FS)
    assert np.isnan(r[1500:1503]).all() and np.isfinite(np.delete(r, np.s_[1500:1503])).all()
    clean = virtual_rpm(_step(), K, idle=IDLE, redline=REDLINE, smooth=TAU, fs=FS)
    assert np.allclose(np.delete(r, np.s_[1500:1503]), np.delete(clean, np.s_[1500:1503]), atol=1e-9)
    # a gap at the start is filled with the first valid value; an all-NaN channel stays NaN
    w = np.r_[np.nan, np.nan, np.full(10, 60.0)]
    assert np.array_equal(virtual_rpm(w, K, idle=IDLE, redline=REDLINE, smooth=TAU, fs=FS)[2:], np.full(10, 1800.0))
    assert np.isnan(virtual_rpm(np.full(5, np.nan), K, idle=IDLE, redline=REDLINE, smooth=TAU, fs=FS)).all()


def test_zero_or_missing_smoothing_is_a_plain_clip_and_fs_is_then_unused():
    v = _step()
    plain = _v(v)
    assert np.array_equal(virtual_rpm(v, K, idle=IDLE, redline=REDLINE, smooth=0), plain)
    assert np.array_equal(virtual_rpm(v, K, idle=IDLE, redline=REDLINE, smooth=None, fs=FS), plain)
    assert np.array_equal(virtual_rpm(v, K, idle=IDLE, redline=REDLINE, smooth=0.0, fs=FS), plain)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"rpm_per_kmh": 0.0}, "rpm_per_kmh"),
        ({"rpm_per_kmh": -3.0}, "rpm_per_kmh"),
        ({"rpm_per_kmh": np.nan}, "rpm_per_kmh"),
        ({"rpm_per_kmh": np.inf}, "rpm_per_kmh"),
        ({"idle": -1.0}, "idle"),
        ({"idle": np.nan}, "idle"),
        ({"idle": np.inf}, "idle"),
        ({"redline": 700.0}, "redline"),  # below idle
        ({"redline": np.nan}, "redline"),
        ({"smooth": 0.2}, "fs"),  # smoothing needs the sampling rate
        ({"smooth": 0.2, "fs": 0.0}, "fs"),
        ({"smooth": 0.2, "fs": -10.0}, "fs"),
        ({"smooth": 0.2, "fs": np.nan}, "fs"),
        ({"smooth": -0.2, "fs": FS}, "smooth"),
        ({"smooth": np.nan, "fs": FS}, "smooth"),
        ({"smooth": np.inf, "fs": FS}, "smooth"),
    ],
)
def test_invalid_arguments_raise(kw, match):
    args = {"speed_kmh": np.arange(10.0), "rpm_per_kmh": K, "idle": IDLE, "redline": REDLINE, **kw}
    with pytest.raises(ValueError, match=match):
        virtual_rpm(args.pop("speed_kmh"), args.pop("rpm_per_kmh"), **args)


def test_smoothing_needs_a_series_and_the_input_is_at_most_two_dimensional():
    with pytest.raises(ValueError, match="scalar"):
        virtual_rpm(50.0, K, idle=IDLE, redline=REDLINE, smooth=TAU, fs=FS)
    with pytest.raises(ValueError, match="shape"):
        virtual_rpm(np.zeros((4, 3, 2)), K, idle=IDLE, redline=REDLINE)
