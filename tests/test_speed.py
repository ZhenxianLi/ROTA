"""Tests of rota_order.speed against analytic truth (rota_order.synth supplies the signals)."""

import numpy as np
import pytest

from rota_order.speed import (
    Segment,
    fit_speed_to_rpm,
    regularize_control,
    rpm_to_phase,
    speed_to_rpm,
    steady_segments,
    tacho_to_rpm,
)
from rota_order.synth import rpm_profiles, synth_runup

# Linear speed curve used for closed-form checks: rpm(t) = R0 + ACC * t.
R0, ACC = 600.0, 400.0  # rpm, rpm/s


def _ramp(fs=1000.0, dur=3.0):
    t = np.arange(round(dur * fs) + 1) / fs
    return t, R0 + ACC * t


def _rev_exact(t):
    """Integral of the linear speed curve in revolutions."""
    return (R0 * t + 0.5 * ACC * t**2) / 60.0


# --------------------------------------------------------------------------
# rpm_to_phase
# --------------------------------------------------------------------------


def test_rpm_to_phase_exact_for_linear_speed():
    t, rpm = _ramp()
    rev = rpm_to_phase(rpm, 1000.0)
    assert rev.shape == rpm.shape
    assert rev[0] == 0.0
    assert np.max(np.abs(rev - _rev_exact(t))) < 1e-11


def test_rpm_to_phase_constant_speed_and_hand_values():
    rev = rpm_to_phase(np.full(2001, 3000.0), 1000.0)  # 50 rev/s for 2 s
    assert rev[-1] == pytest.approx(100.0, rel=1e-13)
    # knots 600 -> 1200 over 1 s, hold 2 s, 1200 -> 0 over 1 s: 15, 40, 10 rev
    t = np.arange(5) * 1.0
    rpm = np.array([600.0, 1200.0, 1200.0, 1200.0, 0.0])
    assert np.allclose(rpm_to_phase(rpm, 1.0), [0.0, 15.0, 35.0, 55.0, 65.0])
    assert np.allclose(rpm_to_phase(rpm, t=t), rpm_to_phase(rpm, 1.0))


def test_rpm_to_phase_piecewise_profile_from_synth_profiles():
    fs = 100.0
    t = np.arange(401) / fs
    rpm = rpm_profiles(t, "piecewise", knots=([0, 1, 3, 4], [600, 1200, 1200, 0]))
    rev = rpm_to_phase(rpm, fs)
    # knots lie on samples, so the trapezoidal integral is exact
    assert rev[100] == pytest.approx(15.0, abs=1e-12)
    assert rev[300] == pytest.approx(55.0, abs=1e-12)
    assert rev[400] == pytest.approx(65.0, abs=1e-12)


def test_rpm_to_phase_is_strictly_increasing_for_positive_speed():
    t = np.arange(4000) / 8000.0
    rpm = rpm_profiles(t, "s_curve", start=300.0, stop=6000.0)
    assert np.all(np.diff(rpm_to_phase(rpm, 8000.0)) > 0)


def test_rpm_to_phase_two_dimensional_input_is_per_column():
    t, rpm = _ramp()
    R = np.column_stack([rpm, 1.37 * rpm, np.full(rpm.size, 300.0)])
    rev = rpm_to_phase(R, 1000.0)
    assert rev.shape == R.shape
    for j in range(3):
        assert np.array_equal(rev[:, j], rpm_to_phase(R[:, j], 1000.0))
    assert np.allclose(rev[:, 1], 1.37 * rev[:, 0], rtol=1e-13)
    assert np.allclose(rev[:, 2], 5.0 * t, atol=1e-12)  # 300 rpm = 5 rev/s


def test_rpm_to_phase_non_uniform_time_axis():
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0.0, 3.0, 300))
    t[0] = 0.0
    rev = rpm_to_phase(R0 + ACC * t, t=t)
    assert np.max(np.abs(rev - _rev_exact(t))) < 1e-11


def test_rpm_to_phase_rev0_offsets_every_sample():
    _, rpm = _ramp()
    base = rpm_to_phase(rpm, 1000.0)
    shifted = rpm_to_phase(rpm, 1000.0, rev0=2.5)
    assert shifted[0] == 2.5
    assert np.allclose(shifted - base, 2.5, atol=1e-12)
    r2 = rpm_to_phase(np.column_stack([rpm, rpm]), 1000.0, rev0=-1.0)
    assert np.all(r2[0] == -1.0)


def test_rpm_to_phase_delay_convention_late_reference():
    # The reference lags the signal by d: ref(t) = true(t - d). Passing delay=d
    # evaluates ref at t + d, which recovers the true angle (end values held
    # over the last d seconds).
    fs, d = 1000.0, 0.05
    t, _true = _ramp(fs)
    ref = R0 + ACC * (t - d)  # the lagging reference (extended below t = 0)
    k = round(d * fs)
    rev = rpm_to_phase(ref, fs, delay=d)
    assert np.max(np.abs(rev[:-k] - _rev_exact(t)[:-k])) < 1e-11
    # without compensation the angle drifts by ACC * d * t / 60 revolutions
    raw = rpm_to_phase(ref, fs)
    assert (_rev_exact(t) - raw)[-1] == pytest.approx(ACC * d * t[-1] / 60.0, abs=1e-9)  # 1 rev


def test_rpm_to_phase_delay_equals_integral_of_shifted_speed():
    fs = 1000.0
    t, rpm = _ramp(fs)
    d = 0.037  # not a multiple of the sample step
    tq = t + d
    expected = rpm_to_phase(np.interp(tq, t, rpm), fs)
    assert np.allclose(rpm_to_phase(rpm, fs, delay=d), expected, atol=1e-12)
    # the same on a non-uniform axis, and per column of a 2-D array
    rng = np.random.default_rng(1)
    tn = np.sort(rng.uniform(0.0, 3.0, 200))
    tn[0] = 0.0
    rn = R0 + ACC * tn
    en = rpm_to_phase(np.interp(tn + d, tn, rn), t=tn)
    assert np.allclose(rpm_to_phase(rn, t=tn, delay=d), en, atol=1e-12)
    R = np.column_stack([rpm, 2.0 * rpm])
    r2 = rpm_to_phase(R, fs, delay=d)
    assert np.allclose(r2[:, 0], expected, atol=1e-12)
    assert np.allclose(r2[:, 1], 2.0 * expected, atol=1e-12)


def test_rpm_to_phase_negative_delay_holds_first_value():
    # reference leads by 50 ms: ref(t) = true(t + 0.05). Compensation needs
    # ref(t - 0.05), which does not exist for t < 0.05, so ref[0] is held. The
    # angle then carries a constant offset a*d^2/2/60 from the held interval.
    fs, d = 1000.0, 0.05
    t, _ = _ramp(fs)
    ref = R0 + ACC * (t + d)
    rev = rpm_to_phase(ref, fs, delay=-d)
    err = rev - _rev_exact(t)
    held = round(d * fs)
    assert np.allclose(err[held:], ACC * d**2 / 2.0 / 60.0, atol=1e-9)


def test_rpm_to_phase_min_rpm_floor():
    # zero speed with a 60 rpm floor is 1 rev/s
    assert np.allclose(rpm_to_phase(np.zeros(4), 1.0, min_rpm=60.0), [0, 1, 2, 3])
    # negative and zero samples are raised to the floor before integration
    rev = rpm_to_phase(np.array([-60.0, 120.0, 0.0]), 1.0, min_rpm=60.0)
    assert np.allclose(rev, [0.0, 1.5, 3.0])
    # without a floor, a stopped reference gives a flat angle
    assert np.all(np.diff(rpm_to_phase(np.zeros(4), 1.0)) == 0)
    # the floor is inactive when the speed is above it
    _t, rpm = _ramp()
    assert np.array_equal(rpm_to_phase(rpm, 1000.0, min_rpm=100.0), rpm_to_phase(rpm, 1000.0))
    # run-up from standstill: floor makes the angle strictly increasing
    s = np.linspace(0.0, 3000.0, 3001)
    assert np.all(np.diff(rpm_to_phase(s, 1000.0, min_rpm=30.0)) > 0)
    # 2-D input
    r2 = rpm_to_phase(np.zeros((4, 2)), 1.0, min_rpm=120.0)
    assert np.allclose(r2, np.outer([0, 2, 4, 6], [1, 1]))


def test_rpm_to_phase_does_not_modify_its_input():
    rpm = np.array([60.0, 0.0, 60.0, 90.0])
    before = rpm.copy()
    rpm_to_phase(rpm, 1.0, min_rpm=30.0, delay=0.5, rev0=1.0)
    assert np.array_equal(rpm, before)


def test_rpm_to_phase_single_sample_returns_rev0():
    assert np.array_equal(rpm_to_phase(np.array([600.0]), 1.0, rev0=3.0), [3.0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rpm_to_phase_rejects_non_finite(bad):
    rpm = np.array([60.0, bad, 60.0])
    with pytest.raises(ValueError, match="non-finite"):
        rpm_to_phase(rpm, 1.0)
    with pytest.raises(ValueError, match="non-finite"):
        rpm_to_phase(np.column_stack([rpm, rpm]), 1.0)


def test_rpm_to_phase_input_validation():
    with pytest.raises(ValueError):
        rpm_to_phase(np.ones((3, 2, 2)), 1.0)
    with pytest.raises(ValueError):
        rpm_to_phase(np.ones(3))  # neither fs nor t
    with pytest.raises(ValueError):
        rpm_to_phase(np.ones(3), 0.0)
    with pytest.raises(ValueError):
        rpm_to_phase(np.ones(3), t=[0.0, 1.0, 1.0])  # not strictly increasing
    with pytest.raises(ValueError):
        rpm_to_phase(np.ones(3), t=[0.0, 1.0])  # wrong length


# --------------------------------------------------------------------------
# tacho_to_rpm
# --------------------------------------------------------------------------

_TACHO_CASES = [(1, 8000.0), (4, 16000.0), (60, 24000.0)]  # (ppr, fs)


def _tacho(ppr, fs, *, lo=600.0, hi=3000.0, dur=3.0, **kw):
    n = int(dur * fs)
    t = np.arange(n) / fs
    rpm = rpm_profiles(t, "linear", start=lo, stop=hi)
    return synth_runup(fs, rpm, [{"order": 1, "phase": 0.0}], tacho_ppr=ppr, seed=3, **kw)


def _rel_err(rpm_est, sig):
    return np.abs(rpm_est / sig.rpm - 1.0)


def _internal_missing(sig, ppr):
    """Dropped pulses lying between the first and last surviving pulse."""
    k = np.rint(sig.pulse_revs * ppr).astype(int)
    return int(k[-1] - k[0] + 1 - k.size)


@pytest.mark.parametrize("ppr, fs", _TACHO_CASES)
@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"tacho_jitter": 1e-5},
        {"tacho_drop": 0.05},
        {"tacho_jitter": 1e-5, "tacho_drop": 0.05},
    ],
    ids=["clean", "jitter", "drops", "jitter+drops"],
)
def test_tacho_median_speed_error_within_roadmap_tolerance(ppr, fs, kw):
    # ROADMAP 5.4: median relative speed error <= 0.1 % against analytic truth
    sig = _tacho(ppr, fs, **kw)
    rpm = tacho_to_rpm(sig.tacho, fs, ppr)
    assert rpm.shape == sig.tacho.shape
    assert np.median(_rel_err(rpm, sig)) <= 1e-3


@pytest.mark.parametrize(
    "kind, lo, hi",
    [("linear", 3000.0, 600.0), ("exponential", 3000.0, 800.0), ("s_curve", 600.0, 3000.0)],
)
def test_tacho_run_down_and_nonlinear_profiles(kind, lo, hi):
    ppr, fs = 4, 16000.0
    t = np.arange(int(3 * fs)) / fs
    rpm = rpm_profiles(t, kind, start=lo, stop=hi)
    sig = synth_runup(
        fs, rpm, [{"order": 1, "phase": 0.0}], tacho_ppr=ppr, tacho_jitter=5e-6, seed=7
    )
    est = tacho_to_rpm(sig.tacho, fs, ppr)
    assert np.median(_rel_err(est, sig)) <= 1e-3


def test_tacho_does_not_modify_its_input():
    ppr, fs = 4, 16000.0
    sig = _tacho(ppr, fs)
    y = sig.tacho.copy()
    tacho_to_rpm(y, fs, ppr, edge="falling")
    assert np.array_equal(y, sig.tacho)


@pytest.mark.parametrize("ppr, fs", _TACHO_CASES)
def test_tacho_pulse_times_and_revs_recover_synth_truth(ppr, fs):
    sig = _tacho(ppr, fs)
    _rpm, info = tacho_to_rpm(sig.tacho, fs, ppr, return_info=True)
    assert set(info) >= {"pulse_times", "pulse_revs", "n_missing", "n_spurious", "extrapolated"}
    assert info["n_missing"] == 0 and info["n_spurious"] == 0
    assert info["pulse_times"].shape == sig.pulse_times.shape
    # sub-sample edge interpolation of the two-sample ramps is exact
    assert np.max(np.abs(info["pulse_times"] - sig.pulse_times)) < 1e-9
    # the first detected pulse defines angle 0
    assert np.allclose(info["pulse_revs"], sig.pulse_revs - sig.pulse_revs[0], atol=1e-12)


def test_tacho_return_type_without_info_is_array():
    sig = _tacho(4, 16000.0)
    out = tacho_to_rpm(sig.tacho, 16000.0, 4)
    assert isinstance(out, np.ndarray)


@pytest.mark.parametrize("ppr, fs", [(1, 8000.0), (60, 24000.0)])
def test_tacho_dropped_pulses_are_counted_and_filled(ppr, fs):
    sig = _tacho(ppr, fs, tacho_drop=0.08, tacho_jitter=5e-6)
    n_dropped = sig.config["tacho_n_dropped"]
    assert n_dropped > 3  # the seed produces a meaningful number of drops
    rpm, info = tacho_to_rpm(sig.tacho, fs, ppr, return_info=True)
    assert info["n_spurious"] == 0
    assert info["n_missing"] == _internal_missing(sig, ppr)
    # drops before the first or after the last surviving pulse are invisible
    k = np.rint(sig.pulse_revs * ppr).astype(int)
    m = int(np.floor(sig.rev[-1] * ppr + 1e-9))
    leading, trailing = k[0] - 1, m - k[-1]
    assert info["n_missing"] + leading + trailing == n_dropped
    # for this seed the first and last pulses survive, so the counts agree exactly
    assert leading == trailing == 0
    assert info["n_missing"] == n_dropped
    # filled pulses keep their true angle labels
    assert np.allclose(np.diff(info["pulse_revs"]) * ppr, np.diff(k), atol=1e-12)
    assert np.median(_rel_err(rpm, sig)) <= 1e-3


def test_tacho_long_dropout_of_consecutive_pulses_is_bridged():
    ppr, fs = 60, 24000.0
    sig = _tacho(ppr, fs)
    pt = sig.pulse_times
    # blank the pulses 200..209: from after the fall of 199 to before the rise of 210
    a = int(np.ceil((pt[199] + 0.75 * (pt[200] - pt[199])) * fs))
    b = int(np.floor((pt[210] - 0.25 * (pt[210] - pt[209])) * fs))
    y = sig.tacho.copy()
    y[a:b] = 0.0
    rpm, info = tacho_to_rpm(y, fs, ppr, return_info=True)
    assert info["n_missing"] == 10
    assert info["n_spurious"] == 0
    assert info["pulse_times"].size == pt.size - 10
    assert np.median(_rel_err(rpm, sig)) <= 1e-3
    far = (sig.t < pt[150]) | (sig.t > pt[260])
    assert np.max(_rel_err(rpm, sig)[far & ~info["extrapolated"]]) < 5e-3


def _inject_pulse(tacho, fs, t_start, width):
    """A spurious pulse of `width` s starting at `t_start` s."""
    y = tacho.copy()
    i0 = round(t_start * fs)
    y[i0 : i0 + max(1, round(width * fs))] = tacho.max()
    return y


@pytest.mark.parametrize("frac, width_frac", [(0.6, 0.2), (0.55, 0.05), (0.65, 0.1)])
def test_tacho_spurious_pulse_is_discarded(frac, width_frac):
    ppr, fs = 4, 16000.0
    sig = _tacho(ppr, fs)
    clean, _ = tacho_to_rpm(sig.tacho, fs, ppr, return_info=True)
    pt = sig.pulse_times
    k = 100
    T = pt[k + 1] - pt[k]
    # the pulse is high for the first half of T, so frac > 0.5 lands in the low part
    y = _inject_pulse(sig.tacho, fs, pt[k] + frac * T, width_frac * T)
    rpm, info = tacho_to_rpm(y, fs, ppr, return_info=True)
    assert info["n_spurious"] == 1
    assert info["n_missing"] == 0
    assert info["pulse_times"].size == pt.size
    assert np.max(np.abs(info["pulse_times"] - pt)) < 1e-9
    assert np.max(np.abs(rpm - clean)) < 1e-6  # identical to the clean result


@pytest.mark.parametrize("kind", ["one-sample glitch", "notch in a pulse"])
def test_tacho_glitches_are_counted_as_spurious(kind):
    ppr, fs = 4, 16000.0
    sig = _tacho(ppr, fs)
    clean = tacho_to_rpm(sig.tacho, fs, ppr)
    pt = sig.pulse_times
    y = sig.tacho.copy()
    if kind == "one-sample glitch":
        T = pt[51] - pt[50]
        y[int((pt[50] + 0.6 * T) * fs)] = 5.0
    else:
        T = pt[61] - pt[60]
        i = int((pt[60] + 0.25 * T) * fs)
        y[i : i + 3] = 0.0
    rpm, info = tacho_to_rpm(y, fs, ppr, return_info=True)
    assert info["n_spurious"] == 1
    assert np.max(np.abs(rpm - clean)) < 1e-6


def test_tacho_two_spurious_pulses_and_drops_together():
    ppr, fs = 4, 16000.0
    sig = _tacho(ppr, fs, tacho_drop=0.05, tacho_jitter=1e-5)
    pt = sig.pulse_times
    # pulses k and k + 1 both survived, so the interval between them is clean
    adjacent = np.flatnonzero(np.diff(np.rint(sig.pulse_revs * ppr)) == 1)
    y = sig.tacho
    for k in (adjacent[100], adjacent[200]):
        T = pt[k + 1] - pt[k]
        y = _inject_pulse(y, fs, pt[k] + 0.6 * T, 0.2 * T)
    rpm, info = tacho_to_rpm(y, fs, ppr, return_info=True)
    assert info["n_spurious"] == 2
    assert info["n_missing"] == _internal_missing(sig, ppr)
    assert np.median(_rel_err(rpm, sig)) <= 1e-3


def test_tacho_spurious_pulse_just_before_a_genuine_pulse_keeps_speed_accurate():
    ppr, fs = 4, 16000.0
    sig = _tacho(ppr, fs)
    clean = tacho_to_rpm(sig.tacho, fs, ppr)
    pt = sig.pulse_times
    T = pt[101] - pt[100]
    y = _inject_pulse(sig.tacho, fs, pt[100] + 0.8 * T, 0.1 * T)
    rpm, info = tacho_to_rpm(y, fs, ppr, return_info=True)
    assert info["n_spurious"] == 1  # counted correctly ...
    assert np.max(np.abs(rpm - clean)) < 1.0  # ... but the wrong pulse is removed (88 rpm)


@pytest.mark.parametrize(
    "a, b, edge",
    [
        (1.0, 0.0, "rising"),
        (2.0, -4.0, "rising"),  # bipolar signal
        (0.01, 100.0, "rising"),  # small swing on a large offset
        (-3.0, 7.0, "falling"),  # inverted polarity: synth rise is now a fall
    ],
)
def test_tacho_signal_scale_offset_and_polarity_do_not_change_result(a, b, edge):
    ppr, fs = 4, 16000.0
    sig = _tacho(ppr, fs, tacho_jitter=1e-5)
    ref, ref_info = tacho_to_rpm(sig.tacho, fs, ppr, return_info=True)
    out, info = tacho_to_rpm(a * sig.tacho + b, fs, ppr, edge=edge, return_info=True)
    assert np.allclose(out, ref, rtol=1e-9)
    assert np.max(np.abs(info["pulse_times"] - ref_info["pulse_times"])) < 1e-9


def test_tacho_falling_edge_gives_same_speed_curve():
    ppr, fs = 4, 16000.0
    sig = _tacho(ppr, fs)
    _rise, ir = tacho_to_rpm(sig.tacho, fs, ppr, return_info=True)
    fall, i_f = tacho_to_rpm(sig.tacho, fs, ppr, edge="falling", return_info=True)
    n = min(ir["pulse_times"].size, i_f["pulse_times"].size)
    lag = i_f["pulse_times"][:n] - ir["pulse_times"][:n]
    interval = np.diff(ir["pulse_times"])[: n - 1]
    # synth pulses are high for half the local interval
    assert np.all((lag[:-1] / interval > 0.45) & (lag[:-1] / interval < 0.55))
    assert np.median(_rel_err(fall, sig)) <= 1e-3
    # a rising-edge read of the inverted signal is a falling-edge read of the original
    inv = tacho_to_rpm(5.0 - sig.tacho, fs, ppr, edge="rising")
    assert np.allclose(inv, fall, rtol=1e-9)


def test_tacho_explicit_level_and_hysteresis_match_the_defaults():
    ppr, fs = 4, 16000.0
    sig = _tacho(ppr, fs)  # swing 0..5 V
    ref = tacho_to_rpm(sig.tacho, fs, ppr)
    out = tacho_to_rpm(sig.tacho, fs, ppr, level=2.5, hysteresis=1.0)
    assert np.allclose(out, ref, rtol=1e-12)
    with pytest.raises(ValueError, match="fewer than 4"):
        tacho_to_rpm(sig.tacho, fs, ppr, level=10.0)  # above every sample


def test_tacho_additive_noise_does_not_create_false_pulses():
    ppr, fs = 4, 16000.0
    sig = _tacho(ppr, fs)
    rng = np.random.default_rng(0)
    noisy = sig.tacho + 0.3 * rng.standard_normal(sig.tacho.size)  # 6 % of the swing
    rpm, info = tacho_to_rpm(noisy, fs, ppr, return_info=True)
    assert info["n_spurious"] == 0 and info["n_missing"] == 0
    assert info["pulse_times"].size == sig.pulse_times.size
    assert np.median(_rel_err(rpm, sig)) <= 1e-3


def test_tacho_smooth_zero_is_a_plain_central_difference():
    ppr, fs = 60, 24000.0
    sig = _tacho(ppr, fs)
    rpm, info = tacho_to_rpm(sig.tacho, fs, ppr, smooth=0, return_info=True)
    assert np.median(_rel_err(rpm, sig)) <= 1e-3
    # at a pulse in the interior it is 60 / central difference of t(rev)
    te = info["pulse_times"]
    j = 2000
    expected = 60.0 / ((te[j + 1] - te[j - 1]) * ppr / 2.0)
    at_pulse = tacho_to_rpm(sig.tacho, fs, ppr, smooth=0, t_out=[te[j]])
    assert at_pulse[0] == pytest.approx(expected, rel=1e-12)
    # one pulse per revolution is a much coarser difference but still accurate
    sig1 = _tacho(1, 8000.0)
    r1 = tacho_to_rpm(sig1.tacho, 8000.0, 1, smooth=0)
    assert np.median(_rel_err(r1, sig1)) <= 2e-3


@pytest.mark.parametrize(
    "ppr, fs, smooth, window, jitter",
    [
        (1, 8000.0, 1.0, 5, 2e-4),  # the window holds at least 5 pulses
        (4, 16000.0, 2.0, 9, 1e-4),  # 8 pulses made odd
        (60, 24000.0, 0.5, 31, 1e-5),  # half a revolution
        (60, 24000.0, 1.0, 61, 1e-5),  # a full revolution, made odd
    ],
)
def test_tacho_jitter_averaging_follows_savitzky_golay_theory(ppr, fs, smooth, window, jitter):
    # A local quadratic fit over N pulses estimates dt/drev with variance
    # sigma^2 / (drev^2 sum m^2), m = -(N-1)/2 ... (N-1)/2. At constant speed the
    # relative speed error has the standard deviation sigma / (sqrt(sum m^2) * T).
    rpm0 = 1200.0
    n = int(30 * fs)
    sig = synth_runup(
        fs,
        np.full(n, rpm0),
        [{"order": 1, "phase": 0.0}],
        tacho_ppr=ppr,
        tacho_jitter=jitter,
        seed=5,
    )
    _, info = tacho_to_rpm(sig.tacho, fs, ppr, smooth=smooth, return_info=True)
    pulse_times = info["pulse_times"]
    rpm = tacho_to_rpm(sig.tacho, fs, ppr, smooth=smooth, t_out=pulse_times)
    err = (rpm / rpm0 - 1.0)[window:-window]  # leave out the edge fits
    m = np.arange(window) - 0.5 * (window - 1)
    period = 60.0 / (rpm0 * ppr)  # time between pulses
    expected = jitter / (np.sqrt(np.sum(m**2)) * period)
    assert np.std(err) == pytest.approx(expected, rel=0.1)
    assert abs(np.mean(err)) < 0.1 * expected  # unbiased at constant speed


def test_tacho_default_hysteresis_rejects_ripple_that_crosses_the_level():
    ppr, fs = 4, 16000.0
    sig = _tacho(ppr, fs)
    # 2.7 V of ripple dips below the 2.5 V mid-level on a high pulse and rises above it on a
    # low one, but never leaves the default hysteresis band (20 % of the range)
    ripple = 2.7 * np.sin(2 * np.pi * 2000.0 * sig.t)
    y = sig.tacho + ripple
    rpm, info = tacho_to_rpm(y, fs, ppr, return_info=True)
    assert info["pulse_times"].size == sig.pulse_times.size
    assert info["n_spurious"] == 0 and info["n_missing"] == 0
    assert np.median(_rel_err(rpm, sig)) < 3e-3  # edge timing carries the ripple
    # a hysteresis band narrower than the ripple chatters
    _, chatter = tacho_to_rpm(y, fs, ppr, hysteresis=0.05, return_info=True)
    assert chatter["pulse_times"].size > 5 * sig.pulse_times.size


def test_tacho_smoothing_suppresses_pulse_jitter():
    ppr, fs = 60, 24000.0
    sig = _tacho(ppr, fs, tacho_jitter=1e-5)
    raw = tacho_to_rpm(sig.tacho, fs, ppr, smooth=0)
    smoothed = tacho_to_rpm(sig.tacho, fs, ppr, smooth=1.0)
    e_raw = np.median(_rel_err(raw, sig))
    e_smooth = np.median(_rel_err(smoothed, sig))
    assert e_smooth < 0.2 * e_raw
    assert e_smooth <= 1e-3
    # an excessive window is clipped to the data instead of failing
    big = tacho_to_rpm(sig.tacho, fs, ppr, smooth=1e4)
    assert np.all(np.isfinite(big)) and big.shape == sig.tacho.shape


def test_tacho_explicit_output_times_and_extrapolation_mask():
    ppr, fs = 4, 16000.0
    sig = _tacho(ppr, fs)
    tq = np.arange(0.5, 2.5, 0.01)
    rpm = tacho_to_rpm(sig.tacho, fs, ppr, t_out=tq)
    assert rpm.shape == tq.shape
    truth = 600.0 + 800.0 * tq  # 600 -> 3000 rpm in 3 s
    assert np.max(np.abs(rpm / truth - 1.0)) < 1e-3

    pt = sig.pulse_times
    tq = np.array([0.0, pt[0] - 1e-3, pt[0], pt[0] + 1e-3, 1.5, pt[-1], pt[-1] + 1e-3, 3.0])
    rpm, info = tacho_to_rpm(sig.tacho, fs, ppr, t_out=tq, return_info=True)
    # the first and last pulse times themselves are inside the span of pulses
    assert list(info["extrapolated"]) == [True, True, False, False, False, False, True, True]
    # outside the pulses the end estimates are held
    assert rpm[0] == rpm[1] == tacho_to_rpm(sig.tacho, fs, ppr, t_out=[pt[0]])[0]
    assert rpm[-1] == rpm[-2] == tacho_to_rpm(sig.tacho, fs, ppr, t_out=[pt[-1]])[0]


def test_tacho_default_output_grid_and_extrapolated_mask():
    ppr, fs = 4, 16000.0
    sig = _tacho(ppr, fs)
    _, info = tacho_to_rpm(sig.tacho, fs, ppr, return_info=True)
    expected = (sig.t < sig.pulse_times[0]) | (sig.t > sig.pulse_times[-1])
    assert info["extrapolated"].shape == sig.tacho.shape
    assert np.array_equal(info["extrapolated"], expected)
    assert 0 < expected.sum() < 0.1 * expected.size


def test_tacho_too_few_pulses_raises():
    ppr, fs = 4, 16000.0
    sig = _tacho(ppr, fs)
    pt = sig.pulse_times
    cut3 = int((pt[2] + 0.75 * (pt[3] - pt[2])) * fs)  # three complete pulses
    with pytest.raises(ValueError, match="fewer than 4"):
        tacho_to_rpm(sig.tacho[:cut3], fs, ppr)
    for const in (np.zeros(1000), np.full(1000, 3.0)):
        with pytest.raises(ValueError, match="fewer than 4"):
            tacho_to_rpm(const, fs, ppr)


def test_tacho_four_pulses_is_enough():
    ppr, fs = 4, 16000.0
    sig = _tacho(ppr, fs)
    pt = sig.pulse_times
    cut4 = int((pt[3] + 0.75 * (pt[4] - pt[3])) * fs)
    rpm, info = tacho_to_rpm(sig.tacho[:cut4], fs, ppr, return_info=True)
    assert info["pulse_times"].size == 4
    assert np.all(np.isfinite(rpm)) and np.all(rpm > 0)
    assert np.median(np.abs(rpm / sig.rpm[:cut4] - 1.0)) < 0.05


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_tacho_rejects_invalid_ppr(bad):
    sig = _tacho(4, 16000.0)
    with pytest.raises(ValueError, match="ppr"):
        tacho_to_rpm(sig.tacho, 16000.0, bad)


def test_tacho_rejects_invalid_edge():
    sig = _tacho(4, 16000.0)
    with pytest.raises(ValueError, match="edge"):
        tacho_to_rpm(sig.tacho, 16000.0, 4, edge="both")


def test_tacho_second_shaft_ratio_via_ppr():
    # tacho on a shaft that turns 1.37 times faster than the reference shaft
    fs, ppr = 16000.0, 2
    t = np.arange(int(3 * fs)) / fs
    base = rpm_profiles(t, "linear", start=500.0, stop=2000.0)
    R = np.column_stack([base, 1.37 * base])
    sig = synth_runup(fs, R, [{"order": 1, "phase": 0.0}], tacho_ppr=ppr, tacho_shaft=1, seed=2)
    rpm = tacho_to_rpm(sig.tacho, fs, ppr)
    assert np.median(np.abs(rpm / (1.37 * base) - 1.0)) <= 1e-3


# --------------------------------------------------------------------------
# regularize_control
# --------------------------------------------------------------------------


def _scurve(t):
    return rpm_profiles(t, "s_curve", start=800.0, stop=3000.0, t_start=1.0, t_stop=7.0)


def test_regularize_decimates_1khz_speed_to_100hz_grid():
    fs = 1000.0
    t = np.arange(int(10 * fs)) / fs
    t_out, y = regularize_control(_scurve(t), fs=fs)
    assert t_out.size == y.size == 1000
    assert np.allclose(np.diff(t_out), 0.01)
    assert t_out[0] == 0.0 and t_out[-1] == pytest.approx(9.99)
    assert np.max(np.abs(y / _scurve(t_out) - 1.0)) < 1e-5
    assert not np.any(np.isnan(y))


def test_regularize_zero_phase_lowpass_reproduces_a_ramp_in_the_interior():
    fs = 1000.0
    t = np.arange(10000) / fs
    t_out, y = regularize_control(500.0 + 200.0 * t, fs=fs)
    # filter start-up costs a few hundredths of a percent in the first and last
    # 0.1 s only
    err = np.abs(y - (500.0 + 200.0 * t_out))
    assert err[:5].max() < 0.1 and err[-5:].max() < 0.1
    assert err[20:-20].max() < 1e-6


def test_regularize_output_grid_is_aligned_to_multiples_of_the_output_period():
    ts = 0.0137 + np.arange(1000) / 100.0  # a time axis that does not start on the grid
    t_out, y = regularize_control(500.0 + 200.0 * ts, t=ts)
    assert t_out[0] == pytest.approx(0.02)  # first multiple of 1/fs_out inside the span
    assert np.max(np.abs(t_out * 100.0 - np.round(t_out * 100.0))) < 1e-9
    assert t_out[-1] <= ts[-1] and t_out[-1] + 0.01 > ts[-1]
    assert np.allclose(y, 500.0 + 200.0 * t_out, atol=1e-9)
    # a different output rate
    t_out, _ = regularize_control(np.ones(2000), fs=1000.0, fs_out=40.0)
    assert np.allclose(t_out * 40.0, np.round(t_out * 40.0), atol=1e-9)
    assert np.allclose(np.diff(t_out), 0.025)


@pytest.mark.parametrize("fs_slow", [200.0, 50.0, 125.0])
def test_regularize_slow_channel_at_other_rates(fs_slow):
    # 200 Hz is 1/5 of a 1 kHz acoustic axis; 50 Hz is below the output rate
    ts = np.arange(int(10 * fs_slow)) / fs_slow
    t_out, y = regularize_control(_scurve(ts), fs=fs_slow)
    assert not np.any(np.isnan(y))
    assert np.allclose(t_out * 1000.0, np.round(t_out * 1000.0), atol=1e-6)  # on the 1 kHz axis
    inner = (t_out > 0.3) & (t_out < 9.6)
    assert np.max(np.abs(y[inner] / _scurve(t_out[inner]) - 1.0)) < 2e-3


def test_regularize_upsampling_interpolates_linearly_without_filtering():
    ts = np.arange(500) / 50.0
    t_out, y = regularize_control(500.0 + 200.0 * ts, fs=50.0, fs_out=100.0)
    assert np.allclose(y, 500.0 + 200.0 * t_out, atol=1e-9)
    assert t_out.size == 2 * ts.size - 1


def test_regularize_input_rate_equal_to_output_rate_is_an_identity():
    ts = np.arange(1000) / 100.0
    x = _scurve(ts)
    t_out, y = regularize_control(x, fs=100.0)  # fs_out defaults to 100 Hz
    assert np.allclose(t_out, ts, atol=1e-12)
    assert np.allclose(y, x, rtol=1e-12)


def test_regularize_offset_shifts_the_time_axis():
    fs = 1000.0
    t = np.arange(10000) / fs
    t_out, y = regularize_control(_scurve(t), fs=fs, offset=0.25)
    assert t_out[0] == pytest.approx(0.25)
    assert t_out[-1] == pytest.approx(10.24)
    inner = (t_out > 1.0) & (t_out < 9.5)
    assert np.max(np.abs(y[inner] / _scurve(t_out[inner] - 0.25) - 1.0)) < 1e-5
    # a negative offset (measured delay of the other system)
    t_out, y = regularize_control(500.0 + 200.0 * t, fs=fs, offset=-1.0)
    assert t_out[0] == pytest.approx(-1.0) and t_out[-1] == pytest.approx(8.99)
    assert np.allclose(y[20:-20], 500.0 + 200.0 * (t_out[20:-20] + 1.0), atol=1e-6)


def test_regularize_long_gap_is_nan_beyond_max_gap_and_short_gap_is_bridged():
    fs = 1000.0
    t = np.arange(10000) / fs

    def truth(tt):
        return 500.0 + 200.0 * tt

    x = truth(t)
    x[3000:3500] = np.nan  # valid samples end at 2.999 and resume at 3.500
    t_out, y = regularize_control(x, fs=fs)
    max_gap = 0.02  # default: twice the larger of 1/fs_in and 1/fs_out
    dist = np.minimum(np.abs(t_out - 2.999), np.abs(t_out - 3.5))
    in_gap = (t_out > 2.999) & (t_out < 3.5)
    must_be_nan = in_gap & (dist > max_gap + 1e-6)
    must_be_valid = ~in_gap | (dist < max_gap - 1e-6)
    assert must_be_nan.sum() > 40
    assert np.all(np.isnan(y[must_be_nan]))
    assert not np.any(np.isnan(y[must_be_valid]))
    ok = ~np.isnan(y)
    assert np.max(np.abs(y[ok] - truth(t_out[ok]))[20:-20]) < 1e-6  # bridged ramp is exact

    x = truth(t)
    x[3000:3010] = np.nan  # 10 ms, shorter than the max gap
    t_out, y = regularize_control(x, fs=fs)
    assert not np.any(np.isnan(y))
    assert np.max(np.abs(y - truth(t_out))[20:-20]) < 1e-6


def test_regularize_leading_and_trailing_gaps_leave_nan_outside_the_valid_span():
    fs = 1000.0
    t = np.arange(10000) / fs
    x = 500.0 + 200.0 * t
    x[:500] = np.nan  # valid from 0.500 s
    x[-300:] = np.nan  # valid until 9.699 s
    t_out, y = regularize_control(x, fs=fs)
    valid = np.flatnonzero(~np.isnan(y))
    assert t_out[valid[0]] == pytest.approx(0.5) and t_out[valid[-1]] == pytest.approx(9.69)
    assert np.all(np.isnan(y[: valid[0]])) and np.all(np.isnan(y[valid[-1] + 1 :]))
    assert not np.any(np.isnan(y[valid[0] : valid[-1] + 1]))


def test_regularize_max_gap_argument_controls_bridging():
    fs = 1000.0
    t = np.arange(10000) / fs
    x = 500.0 + 200.0 * t
    x[3000:3500] = np.nan
    _, y = regularize_control(x, fs=fs, max_gap=1.0)
    assert not np.any(np.isnan(y))
    _, y = regularize_control(x, fs=fs, max_gap=0.05)
    n_nan = np.isnan(y).sum()
    assert 30 < n_nan < 45  # gap of 0.5 s minus 2 * 0.05 s, in 10 ms steps


def test_regularize_irregular_time_axis_with_a_logging_dropout():
    fs = 1000.0
    t = np.arange(10000) / fs
    keep = (t < 3.0) | (t >= 3.5)  # the logger stopped for half a second
    ts = t[keep]
    t_out, y = regularize_control(500.0 + 200.0 * ts, t=ts)
    assert np.sum(np.isnan(y)) >= 45
    assert np.all(np.isnan(y[(t_out > 3.03) & (t_out < 3.47)]))
    ok = ~np.isnan(y)
    assert np.max(np.abs(y[ok] - (500.0 + 200.0 * t_out[ok]))[20:-20]) < 1e-6


def test_regularize_despike_removes_injected_spikes():
    fs = 1000.0
    t = np.arange(10000) / fs
    clean = _scurve(t)
    x = clean.copy()
    x[[2000, 5000, 5001, 8000]] *= 1.3  # +30 % spikes, on the 100 Hz grid
    t_out, y0 = regularize_control(x.copy(), fs=fs)
    t_out, y1 = regularize_control(x.copy(), fs=fs, despike_tol=0.07)
    truth = _scurve(t_out)
    assert np.max(np.abs(y1 / truth - 1.0)) < 1e-5
    assert np.max(np.abs(y0 / truth - 1.0)) > 1e-2  # the spikes are visible without despiking


def test_regularize_despike_leaves_clean_data_and_genuine_steps_alone():
    fs = 1000.0
    t = np.arange(10000) / fs
    clean = _scurve(t)
    _, a = regularize_control(clean.copy(), fs=fs)
    _, b = regularize_control(clean.copy(), fs=fs, despike_tol=0.07)
    assert np.array_equal(a, b)
    # a gear-shift style step is not a spike: the running median follows it
    step = np.where(t < 5.0, 3000.0, 2000.0)
    _, a = regularize_control(step.copy(), fs=fs)
    _, b = regularize_control(step.copy(), fs=fs, despike_tol=0.07)
    assert not np.any(np.isnan(b))
    assert np.array_equal(a, b)


def test_regularize_despike_window_is_in_seconds():
    # a 1.5 s plateau that is 20 % high is a spike for a 5 s median window
    # (removed, and a gap this long stays NaN) but is followed by a 0.3 s one
    fs = 100.0
    t = np.arange(2000) / fs
    x = np.full(t.size, 1000.0)
    x[900:1050] = 1200.0  # 9.00 - 10.50 s
    kw = {"fs": fs, "fs_out": 50.0, "despike_tol": 0.1}
    t_out, long_w = regularize_control(x.copy(), despike_window=5.0, **kw)
    _, short_w = regularize_control(x.copy(), despike_window=0.3, **kw)
    gone = np.isnan(long_w)
    assert gone.sum() > 60
    assert np.all((t_out[gone] > 8.95) & (t_out[gone] < 10.55))
    assert np.nanmax(long_w) == pytest.approx(1000.0, abs=1.0)
    assert np.nanmin(long_w) == pytest.approx(1000.0, abs=1.0)
    assert not np.any(np.isnan(short_w))
    assert np.max(short_w) > 1190.0


def test_regularize_does_not_modify_a_1d_input_array():
    x = np.full(1000, 100.0)
    x[500] = 200.0
    before = x.copy()
    regularize_control(x, fs=1000.0, despike_tol=0.07)
    assert np.array_equal(x, before)


def test_regularize_does_not_modify_a_2d_input_array():
    x = np.full((1000, 2), 100.0)
    x[500, 0] = 200.0
    before = x.copy()
    regularize_control(x, fs=1000.0, despike_tol=0.07)
    assert np.array_equal(x, before)


def test_regularize_two_dimensional_input_processes_channels_independently():
    fs = 1000.0
    t = np.arange(10000) / fs
    a = _scurve(t)
    b = 0.3 * a + 5.0
    b[3000:3500] = np.nan  # a gap in one channel only
    t1, ya = regularize_control(a, fs=fs)
    t2, yb = regularize_control(b, fs=fs)
    t3, Y = regularize_control(np.column_stack([a, b]), fs=fs)
    assert Y.shape == (t1.size, 2)
    assert np.array_equal(t3, t1) and np.array_equal(t2, t1)
    assert np.array_equal(Y[:, 0], ya)
    assert np.array_equal(Y[:, 1], yb, equal_nan=True)
    assert not np.any(np.isnan(Y[:, 0])) and np.any(np.isnan(Y[:, 1]))


def test_regularize_antialias_rejects_a_tone_above_the_output_nyquist():
    fs = 1000.0
    t = np.arange(10000) / fs
    tone = np.sin(2 * np.pi * 90.0 * t)  # aliases to 10 Hz at a 100 Hz output rate
    _, y = regularize_control(tone, fs=fs)
    assert np.max(np.abs(y[20:-20])) < 0.005  # > 46 dB
    _, y_raw = regularize_control(tone, fs=fs, antialias=False)
    assert np.max(np.abs(y_raw)) > 0.9  # plain interpolation lets the alias through


def test_regularize_antialias_starts_as_soon_as_the_input_rate_exceeds_the_output_rate():
    fs_in = 200.0  # twice the output rate: a 70 Hz tone folds to 30 Hz at 100 Hz
    t = np.arange(4000) / fs_in
    tone = np.sin(2 * np.pi * 70.0 * t)
    _, y = regularize_control(tone, fs=fs_in)
    assert np.max(np.abs(y[20:-20])) < 0.02  # |H|^2 = 1 / (1 + (70/40)^8) = 0.011
    _, y_raw = regularize_control(tone, fs=fs_in, antialias=False)
    assert np.max(np.abs(y_raw)) > 0.9


def test_regularize_passband_gain_matches_the_zero_phase_butterworth():
    fs = 1000.0
    t = np.arange(10000) / fs
    f = 20.0  # 0.2 * fs_out; the cut-off is 40 Hz and the filter is applied twice
    t_out, y = regularize_control(np.sin(2 * np.pi * f * t), fs=fs)
    gain = 1.0 / (1.0 + (f / 40.0) ** 8)  # |H|^2 of a 4th-order Butterworth
    assert np.max(np.abs(y - gain * np.sin(2 * np.pi * f * t_out))[20:-20]) < 2e-3


def test_regularize_explicit_output_times_and_out_of_span_nan():
    fs = 1000.0
    t = np.arange(10000) / fs
    t_out, y = regularize_control(500.0 + 200.0 * t, fs=fs, t_out=[-1.0, 0.5, 5.0, 11.0])
    assert np.array_equal(t_out, [-1.0, 0.5, 5.0, 11.0])
    assert np.isnan(y[0]) and np.isnan(y[3])
    assert y[1] == pytest.approx(600.0, abs=1e-3) and y[2] == pytest.approx(1500.0, abs=1e-6)


def test_regularize_channel_without_valid_samples_is_all_nan():
    t_out, y = regularize_control(np.full(1000, np.nan), fs=1000.0)
    assert y.shape == t_out.shape and np.all(np.isnan(y))
    both = np.column_stack([np.full(1000, np.nan), np.arange(1000.0)])
    t_out, Y = regularize_control(both, fs=1000.0)
    assert np.all(np.isnan(Y[:, 0])) and np.all(np.isfinite(Y[5:-5, 1]))


def test_regularize_input_validation():
    x = np.arange(100.0)
    with pytest.raises(ValueError):
        regularize_control(x)  # neither fs nor t
    with pytest.raises(ValueError):
        regularize_control(x, fs=-1.0)
    with pytest.raises(ValueError):
        regularize_control(x, t=np.arange(99.0))
    with pytest.raises(ValueError):
        regularize_control(x, t=np.r_[np.arange(50.0), np.arange(49.0, 99.0)])  # not increasing


# --------------------------------------------------------------------------
# speed_to_rpm and fit_speed_to_rpm
# --------------------------------------------------------------------------

GEARS = [3.5, 2.0, 1.4, 1.0]
SHIFTS = [30.0, 60.0, 90.0]


def test_speed_to_rpm_lumped_slope():
    v = np.array([0.0, 10.0, 100.0])
    assert np.allclose(speed_to_rpm(v, rpm_per_kmh=31.5), [0.0, 315.0, 3150.0])
    assert speed_to_rpm(5.0, rpm_per_kmh=30.0) == 150.0
    out = speed_to_rpm([[1.0, 2.0], [3.0, 4.0]], rpm_per_kmh=10.0)
    assert np.allclose(out, [[10, 20], [30, 40]])


def test_speed_to_rpm_driveline_chain_hand_value():
    # 100 km/h = 27.78 m/s; C = 2 m -> 13.89 wheel rev/s = 833.3 rpm; x1.0 x4.0 = 3333.3 rpm
    rpm = speed_to_rpm(100.0, wheel_circumference=2.0, gear_ratios=[1.0], final_drive=4.0)
    assert rpm == pytest.approx(100.0 / 3.6 / 2.0 * 60.0 * 4.0, rel=1e-14)
    assert rpm == pytest.approx(10000.0 / 3.0, rel=1e-14)
    # engaged gear 3 of a gearbox: 1.4 * final drive 3.0
    rpm = speed_to_rpm(72.0, wheel_circumference=1.8, gear_ratios=GEARS, final_drive=3.0, gear=3)
    assert rpm == pytest.approx(20.0 / 1.8 * 60.0 * 1.4 * 3.0, rel=1e-14)
    # no final-drive argument means 1
    assert speed_to_rpm(36.0, wheel_circumference=2.0, gear_ratios=[2.0], gear=1) == pytest.approx(
        10.0 / 2.0 * 60.0 * 2.0
    )


def test_speed_to_rpm_single_gear_needs_no_selector():
    a = speed_to_rpm([20.0, 40.0], wheel_circumference=2.0, gear_ratios=[9.0])
    b = speed_to_rpm([20.0, 40.0], rpm_per_kmh=60.0 / 3.6 / 2.0 * 9.0)
    assert np.allclose(a, b)


def test_speed_to_rpm_shift_speeds_make_rpm_drop_at_each_upshift():
    chain = {"wheel_circumference": 2.0, "gear_ratios": GEARS, "final_drive": 3.0}
    kw = {**chain, "shift_speeds": SHIFTS}
    v = np.linspace(10.0, 120.0, 11001)  # 0.01 km/h steps
    rpm = speed_to_rpm(v, **kw)
    drops = np.flatnonzero(np.diff(rpm) < 0)
    assert drops.size == 3
    # the shift happens at the listed speeds (the shift speed itself is already in the next gear)
    assert np.allclose(v[drops + 1], SHIFTS, atol=1e-9)
    for i, (g_lo, g_hi) in zip(drops, [(0, 1), (1, 2), (2, 3)]):
        ratio = rpm[i + 1] / rpm[i]
        assert ratio == pytest.approx(GEARS[g_hi] / GEARS[g_lo] * v[i + 1] / v[i], rel=1e-12)
    # each segment equals the single-gear chain
    for g, (lo, hi) in enumerate([(0, 30), (30, 60), (60, 90), (90, 121)], start=1):
        sel = (v >= lo) & (v < hi)
        single = speed_to_rpm(v[sel], gear=g, **chain)
        assert np.allclose(rpm[sel], single, rtol=1e-13)


def test_speed_to_rpm_explicit_gear_overrides_shift_speeds_and_may_vary_per_sample():
    v = np.array([20.0, 50.0, 80.0, 110.0])
    kw = {"wheel_circumference": 2.0, "gear_ratios": GEARS, "final_drive": 3.0}
    per_sample = speed_to_rpm(v, gear=[1, 1, 2, 4], shift_speeds=SHIFTS, **kw)
    expected = v / 3.6 / 2.0 * 60.0 * 3.0 * np.array([3.5, 3.5, 2.0, 1.0])
    assert np.allclose(per_sample, expected)
    scalar = speed_to_rpm(v, gear=2, **kw)
    assert np.allclose(scalar, v / 3.6 / 2.0 * 60.0 * 3.0 * 2.0)


def test_speed_to_rpm_neutral_gear_gives_idle():
    v = np.array([0.0, 20.0, 50.0, 80.0])
    kw = {"wheel_circumference": 2.0, "gear_ratios": GEARS, "final_drive": 3.0}
    assert np.all(speed_to_rpm(v, gear=0, idle=800.0, **kw) == 800.0)
    out = speed_to_rpm(v, gear=[0, 1, 0, 2], idle=800.0, redline=6000.0, **kw)
    assert out[0] == 800.0 and out[2] == 800.0
    assert out[1] == pytest.approx(20.0 / 3.6 / 2.0 * 60.0 * 3.5 * 3.0)
    # without an idle value neutral means zero rotation
    assert np.all(speed_to_rpm(v, gear=0, **kw) == 0.0)


def test_speed_to_rpm_clips_to_idle_and_redline():
    v = np.array([0.0, 10.0, 100.0, 300.0])
    both = speed_to_rpm(v, rpm_per_kmh=40.0, idle=800.0, redline=6000.0)
    assert np.array_equal(both, [800, 800, 4000, 6000])
    assert np.array_equal(speed_to_rpm(v, rpm_per_kmh=40.0, redline=6000.0), [0, 400, 4000, 6000])
    assert np.array_equal(speed_to_rpm(v, rpm_per_kmh=40.0, idle=800.0), [800, 800, 4000, 12000])
    kw = {"wheel_circumference": 2.0, "gear_ratios": GEARS, "final_drive": 3.0}
    out = speed_to_rpm(
        np.array([1.0, 29.0, 200.0]), idle=1000.0, redline=3500.0, shift_speeds=SHIFTS, **kw
    )
    assert out[0] == 1000.0  # launch: clutch slip holds idle
    assert out[2] == 3500.0  # limiter
    assert 1000.0 < out[1] < 3500.0


def test_speed_to_rpm_nan_speed_stays_nan():
    out = speed_to_rpm([np.nan, 50.0], rpm_per_kmh=30.0, idle=800.0, redline=6000.0)
    assert np.isnan(out[0]) and out[1] == 1500.0
    out = speed_to_rpm(
        [np.nan, 50.0],
        wheel_circumference=2.0,
        gear_ratios=GEARS,
        shift_speeds=SHIFTS,
        idle=800.0,
        redline=6000.0,
    )
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_speed_to_rpm_argument_errors():
    v = np.array([10.0, 50.0])
    with pytest.raises(ValueError):
        speed_to_rpm(v)
    with pytest.raises(ValueError):
        speed_to_rpm(v, wheel_circumference=2.0)
    with pytest.raises(ValueError):
        speed_to_rpm(v, gear_ratios=GEARS)
    with pytest.raises(ValueError, match="gear or shift_speeds"):
        speed_to_rpm(v, wheel_circumference=2.0, gear_ratios=GEARS)
    with pytest.raises(ValueError, match="shift_speeds"):
        speed_to_rpm(v, wheel_circumference=2.0, gear_ratios=GEARS, shift_speeds=[30.0, 60.0])
    for bad in (5, -1, [1, 7]):
        with pytest.raises(ValueError):
            speed_to_rpm(v, wheel_circumference=2.0, gear_ratios=GEARS, gear=bad)


def test_fit_speed_to_rpm_recovers_slope_exactly_from_noiseless_pairs():
    k = 31.7
    v = np.array([20.0, 40.0, 60.0, 100.0, 130.0])
    fit = fit_speed_to_rpm(v, k * v)
    assert fit["rpm_per_kmh"] == pytest.approx(k, rel=1e-13)
    assert fit["residual_rms"] < 1e-9
    assert fit["max_rel_residual"] < 1e-13
    assert fit["n"] == 5
    assert isinstance(fit["n"], int) and isinstance(fit["rpm_per_kmh"], float)


def test_fit_speed_to_rpm_hand_computed_residuals():
    v, r = np.array([10.0, 20.0, 30.0]), np.array([105.0, 190.0, 320.0])
    fit = fit_speed_to_rpm(v, r)
    k = (10 * 105 + 20 * 190 + 30 * 320) / (10**2 + 20**2 + 30**2)
    assert k == pytest.approx(14450.0 / 1400.0)
    assert fit["rpm_per_kmh"] == pytest.approx(k, rel=1e-14)
    res = r - k * v
    assert fit["residual_rms"] == pytest.approx(np.sqrt(np.mean(res**2)), rel=1e-12)
    assert fit["residual_rms"] == pytest.approx(11.2599, abs=1e-4)
    assert fit["max_rel_residual"] == pytest.approx(np.max(np.abs(res) / r), rel=1e-12)
    # the fit through the origin minimises the squared residual: perturbing k cannot improve it
    for dk in (-0.01, 0.01):
        assert np.sum((r - (k + dk) * v) ** 2) > np.sum(res**2)
    # the least-squares residual is orthogonal to v
    assert abs(np.dot(res, v)) < 1e-9


def test_fit_speed_to_rpm_ignores_nan_pairs_and_honours_mask():
    v = np.array([10.0, np.nan, 30.0, 40.0, 50.0])
    r = np.array([100.0, 200.0, np.nan, 400.0, 900.0])
    fit = fit_speed_to_rpm(v, r)
    assert fit["n"] == 3
    expected = (10 * 100 + 40 * 400 + 50 * 900) / (10**2 + 40**2 + 50**2)
    assert fit["rpm_per_kmh"] == pytest.approx(expected)
    masked = fit_speed_to_rpm(v, r, mask=[True, True, True, True, False])
    assert masked["n"] == 2
    assert masked["rpm_per_kmh"] == pytest.approx(10.0, rel=1e-13)  # exact pairs only
    assert masked["residual_rms"] < 1e-9


def test_fit_speed_to_rpm_rejects_unusable_data():
    with pytest.raises(ValueError):
        fit_speed_to_rpm([np.nan, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError):
        fit_speed_to_rpm([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        fit_speed_to_rpm([1.0, 2.0], [1.0, 2.0], mask=[False, False])


def test_calibration_chain_steady_segments_to_fit():
    # a 40-100-40 km/h schedule from synth, mapped to rpm with a known slope,
    # then recovered from the medians of the steady segments
    t = np.arange(1001) / 100.0
    v = rpm_profiles(t, "piecewise", knots=([0, 2, 4, 7, 9, 10], [40, 40, 100, 100, 40, 40]))
    k = 28.4
    rpm = speed_to_rpm(v, rpm_per_kmh=k)
    segs = steady_segments(v, [40.0, 100.0], 0.1, fs=100.0)
    assert all(isinstance(s, Segment) for s in segs)
    sv = [np.median(v[s.i_start : s.i_stop]) for s in segs]
    sr = [np.median(rpm[s.i_start : s.i_stop]) for s in segs]
    fit = fit_speed_to_rpm(sv, sr)
    assert fit["rpm_per_kmh"] == pytest.approx(k, rel=1e-12)
    assert fit["n"] == 2 and fit["residual_rms"] < 1e-9


# --------------------------------------------------------------------------
# steady_segments
# --------------------------------------------------------------------------

FS_CTRL = 100.0


def _schedule():
    """40 km/h hold (0-2 s), ramp, 100 km/h hold (4-7 s), ramp, 40 km/h hold (9-10 s)."""
    t = np.arange(1001) / FS_CTRL
    v = rpm_profiles(t, "piecewise", knots=([0, 2, 4, 7, 9, 10], [40, 40, 100, 100, 40, 40]))
    return t, v


def test_steady_segments_finds_the_hold_exactly():
    _, v = _schedule()
    (seg,) = steady_segments(v, 100.0, 0.1, fs=FS_CTRL)
    assert isinstance(seg, Segment)
    assert (seg.i_start, seg.i_stop) == (400, 701)  # samples at 4.00 ... 7.00 s
    assert seg.start == pytest.approx(4.0) and seg.stop == pytest.approx(7.01)
    assert seg.duration == pytest.approx(3.01)
    assert seg.target == 100.0
    assert seg.mean == pytest.approx(100.0) and seg.std == pytest.approx(0.0, abs=1e-12)
    assert isinstance(seg.i_start, int) and isinstance(seg.start, float)


def test_steady_segments_returns_one_entry_per_target_in_order():
    _, v = _schedule()
    segs = steady_segments(v, [100.0, 40.0, 70.0, 250.0], 0.1, fs=FS_CTRL, min_duration=0.5)
    assert len(segs) == 4
    assert segs[0].target == 100.0 and segs[1].target == 40.0
    assert segs[2] is None  # only a single sample passes through 70 km/h on the ramp
    assert segs[3] is None  # never reached
    assert steady_segments(v, np.array([100.0]), 0.1, fs=FS_CTRL)[0] == segs[0]


def test_steady_segments_takes_the_longest_run():
    _, v = _schedule()
    (seg,) = steady_segments(v, 40.0, 0.1, fs=FS_CTRL)
    assert (seg.i_start, seg.i_stop) == (0, 201)  # 2.01 s beats the 1.01 s hold at the end
    assert seg.duration == pytest.approx(2.01)


def test_steady_segments_min_duration_rejects_short_runs():
    _, v = _schedule()
    assert steady_segments(v, 40.0, 0.1, fs=FS_CTRL, min_duration=2.5) == [None]
    assert steady_segments(v, 40.0, 0.1, fs=FS_CTRL, min_duration=2.01)[0] is not None
    assert steady_segments(v, 40.0, 0.1, fs=FS_CTRL, min_duration=2.02) == [None]
    assert steady_segments(v, 100.0, 0.1, fs=FS_CTRL, min_duration=3.0)[0] is not None
    assert steady_segments(v, 100.0, 0.1, fs=FS_CTRL, min_duration=3.1) == [None]


def test_steady_segments_unmatched_target_is_none():
    _, v = _schedule()
    assert steady_segments(v, [250.0, -5.0], 1.0, fs=FS_CTRL) == [None, None]
    # the ramp samples are 0.3 km/h apart: 70.1 +- 0.05 falls between two of them
    assert steady_segments(v, 70.1, 0.05, fs=FS_CTRL) == [None]


def test_steady_segments_length_gives_a_centred_excerpt():
    _, v = _schedule()
    # run: 301 samples (4.00-7.00 s); a 101-sample excerpt centres exactly
    (seg,) = steady_segments(v, 100.0, 0.1, fs=FS_CTRL, length=1.01)
    assert (seg.i_start, seg.i_stop) == (500, 601)
    assert 0.5 * (seg.start + seg.stop) == pytest.approx(0.5 * (4.0 + 7.01))
    assert seg.duration == pytest.approx(1.01)
    # an even excerpt cannot centre exactly: within half a sample of the run centre
    (seg,) = steady_segments(v, 100.0, 0.1, fs=FS_CTRL, length=1.0)
    assert seg.i_stop - seg.i_start == 100
    assert abs(0.5 * (seg.start + seg.stop) - 0.5 * (4.0 + 7.01)) <= 0.5 / FS_CTRL + 1e-9
    assert 400 <= seg.i_start and seg.i_stop <= 701
    # the excerpt statistics describe the excerpt
    assert seg.mean == pytest.approx(100.0) and seg.std == pytest.approx(0.0, abs=1e-12)
    # an excerpt as long as the run is the run; longer is rejected
    (seg,) = steady_segments(v, 100.0, 0.1, fs=FS_CTRL, length=3.01)
    assert (seg.i_start, seg.i_stop) == (400, 701)
    assert steady_segments(v, 100.0, 0.1, fs=FS_CTRL, length=3.5) == [None]


@pytest.mark.parametrize("n_samples", [1, 29, 57, 58, 99, 115, 230, 301])
def test_steady_segments_excerpt_has_exactly_the_requested_whole_number_of_samples(n_samples):
    _, v = _schedule()  # the 100 km/h run holds 301 samples
    (seg,) = steady_segments(v, 100.0, 0.1, fs=FS_CTRL, length=n_samples / FS_CTRL)
    assert seg.i_stop - seg.i_start == n_samples
    assert 400 <= seg.i_start and seg.i_stop <= 701
    # centred on the run within half a sample
    centre = 0.5 * (seg.start + seg.stop)
    assert abs(centre - 0.5 * (4.0 + 7.01)) <= 0.5 / FS_CTRL + 1e-9


def test_steady_segments_excerpt_length_is_rounded_to_the_nearest_sample():
    _, v = _schedule()
    (seg,) = steady_segments(v, 100.0, 0.1, fs=FS_CTRL, length=0.586)  # 58.6 samples
    assert seg.i_stop - seg.i_start == 59


def test_steady_segments_relative_tolerance_scales_with_the_target():
    _, v = _schedule()
    # relative 0.001 of 100 km/h is 0.1 km/h, relative 0.0025 of 40 is 0.1
    a = steady_segments(v, [40.0, 100.0], 0.1, fs=FS_CTRL)
    assert steady_segments(v, 100.0, 0.001, fs=FS_CTRL, relative=True)[0] == a[1]
    assert steady_segments(v, 40.0, 0.0025, fs=FS_CTRL, relative=True)[0] == a[0]
    # +-5 % tolerance: +-5 km/h at 100 (ramp slope 30 km/h/s) and +-2 km/h at 40
    s100, s40 = steady_segments(v, [100.0, 40.0], 0.05, fs=FS_CTRL, relative=True)
    assert s100.duration == pytest.approx(3.0 + 2 * 5.0 / 30.0, abs=0.025)
    assert s40.duration == pytest.approx(2.0 + 2.0 / 30.0, abs=0.025)
    # the same tolerance in absolute terms is the same for both targets
    abs100, abs40 = steady_segments(v, [100.0, 40.0], 5.0, fs=FS_CTRL)
    assert abs100.duration == pytest.approx(3.0 + 2 * 5.0 / 30.0, abs=0.025)
    assert abs40.duration == pytest.approx(2.0 + 5.0 / 30.0, abs=0.025)


def test_steady_segments_nan_never_matches_and_splits_runs():
    _, v = _schedule()
    v = v.copy()
    v[500] = np.nan
    (seg,) = steady_segments(v, 100.0, 0.1, fs=FS_CTRL)
    assert (seg.i_start, seg.i_stop) == (501, 701)  # the longer half
    v[500], v[550] = 100.0, np.nan  # two halves of 150 samples each: the first wins the tie
    (seg,) = steady_segments(v, 100.0, 0.1, fs=FS_CTRL)
    assert (seg.i_start, seg.i_stop) == (400, 550)


def test_steady_segments_with_noise_and_explicit_time_axis():
    t, v = _schedule()
    rng = np.random.default_rng(0)
    noisy = v + 0.05 * rng.standard_normal(v.size)
    seg = steady_segments(noisy, 100.0, 0.5, fs=FS_CTRL)[0]
    # tolerance 0.5 km/h reaches 1.7 samples up each 30 km/h/s ramp
    assert 4.0 - 0.04 <= seg.start <= 4.0 and 7.0 <= seg.stop - 0.01 <= 7.04
    assert seg.mean == pytest.approx(100.0, abs=0.05)
    assert 0.03 < seg.std < 0.12
    shifted = steady_segments(v, 100.0, 0.1, t=t + 10.0)[0]
    assert shifted.start == pytest.approx(14.0) and shifted.stop == pytest.approx(17.01)
    assert (shifted.i_start, shifted.i_stop) == (400, 701)


def test_steady_segments_needs_a_time_base():
    _, v = _schedule()
    with pytest.raises(ValueError):
        steady_segments(v, 100.0, 0.1)
