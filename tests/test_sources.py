"""Tests for rota_order._sources: driveline speed, angle-periodic ripple, angle-locked
impulses, modulation sidebands and carrier components, all against analytic truth."""

import inspect

import numpy as np
import pytest
from scipy import signal, special
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline

from rota_order import _sources
from rota_order.angle import angular_resample
from rota_order.maps import rpm_order_map
from rota_order.speed import rpm_to_phase, speed_to_rpm
from rota_order.synth import rpm_profiles, synth_runup
from rota_order._sources import (
    angle_impulses,
    angular_speed_ripple,
    carrier_components,
    drive_rpm,
    sideband_orders,
)

TWO_PI = 2 * np.pi


def _w(u):
    """Raised-cosine ramp, written out independently of the module."""
    return 0.5 * (1.0 - np.cos(np.pi * np.asarray(u)))


def test_public_functions_are_documented_with_references():
    assert set(_sources.__all__) == {
        "drive_rpm", "angular_speed_ripple", "angle_impulses", "sideband_orders",
        "carrier_components",
    }
    for name in _sources.__all__:
        doc = inspect.getdoc(getattr(_sources, name))
        assert doc and "References" in doc and "Parameters" in doc and "Returns" in doc, name


# --------------------------------------------------------------------------
# drive_rpm
# --------------------------------------------------------------------------

DRIVE = dict(
    gear_ratios=[3.6, 2.0, 1.4, 1.0, 0.8],
    final_drive=3.7,
    wheel_circumference=1.95,
    shift_speeds=[25.0, 45.0, 70.0, 100.0],
    idle=800.0,
    redline=6500.0,
)
CHAIN = dict(
    wheel_circumference=DRIVE["wheel_circumference"],
    gear_ratios=DRIVE["gear_ratios"],
    final_drive=DRIVE["final_drive"],
)
SLOPE = 6.0  # km/h per s of the test acceleration
T_STAND = 2.0  # s of standstill before it


def _k(gear):
    """rpm per km/h in a gear, from the static driveline function."""
    return float(speed_to_rpm(1.0, gear=gear, **CHAIN))


def _accel(fs=2000):
    t = np.arange(int(24 * fs)) / fs
    v = rpm_profiles(t, "piecewise", knots=([0, 2, 22, 24], [0, 0, 120, 120]))
    return t, v


def _t_cross(s):
    return T_STAND + s / SLOPE


def test_drive_rpm_equals_static_driveline_between_transients():
    t, v = _accel()
    d = drive_rpm(t, v, **DRIVE)
    static = speed_to_rpm(v, shift_speeds=DRIVE["shift_speeds"], idle=800.0, redline=6500.0, **CHAIN)
    free = t >= _t_cross(2 * 800.0 / _k(1)) + 1e-6  # after the launch lock-up at 2 * idle
    for s in DRIVE["shift_speeds"]:
        free &= ~((t >= _t_cross(s) - 1e-6) & (t <= _t_cross(s) + 0.3 + 1e-6))
    assert free.sum() > 0.7 * t.size
    assert np.max(np.abs(d["rpm"][free] - static[free])) < 1e-9
    assert np.array_equal(d["gear"], np.searchsorted(DRIVE["shift_speeds"], v, side="right") + 1)
    # shift times are the analytic crossings of the (piecewise-linear) speed
    assert np.allclose(d["shift_times"], [_t_cross(s) for s in DRIVE["shift_speeds"]], atol=1e-9)
    assert list(d["shift_from"]) == [1, 2, 3, 4] and list(d["shift_to"]) == [2, 3, 4, 5]
    in_shift = np.zeros(t.size, bool)
    for ts in d["shift_times"]:
        in_shift |= (t >= ts) & (t < ts + 0.3)
    assert np.array_equal(d["in_shift"], in_shift)


def test_upshift_transition_is_raised_cosine_and_monotone():
    t, v = _accel()
    shift_time = 0.3
    d = drive_rpm(t, v, shift_time=shift_time, **DRIVE)
    rpm = d["rpm"]
    for i, s in enumerate(DRIVE["shift_speeds"]):
        ts = _t_cross(s)
        k_old, k_new = _k(i + 1), _k(i + 2)
        r_start = k_old * s  # old gear at the shift time
        r_end = k_new * SLOPE * (ts + shift_time - T_STAND)  # new gear at the end of the shift
        idx = np.flatnonzero((t >= ts) & (t < ts + shift_time))
        u = (t[idx] - ts) / shift_time
        assert np.allclose(rpm[idx], r_start + _w(u) * (r_end - r_start), atol=1e-8)
        assert np.all(np.diff(rpm[idx]) < 0)  # strictly falling within shift_time
        assert rpm[idx[0]] > r_start - 1.0 and rpm[idx[-1]] < r_end + 1.0
        # joins the static curve on both sides without a jump
        assert abs(rpm[idx[0] - 1] - k_old * v[idx[0] - 1]) < 1e-9
        assert abs(rpm[idx[-1] + 1] - k_new * v[idx[-1] + 1]) < 1e-9
        assert abs(rpm[idx[0]] - rpm[idx[0] - 1]) < 2.0 and abs(rpm[idx[-1] + 1] - rpm[idx[-1]]) < 2.0
        # half way through, half of the fall has happened
        half = np.argmin(np.abs(u - 0.5))
        assert rpm[idx[half]] == pytest.approx(0.5 * (r_start + r_end), abs=3.0)  # u is within 1/1200 of 0.5


def test_shift_time_zero_is_a_step_and_longer_shift_takes_longer():
    t, v = _accel()
    d0 = drive_rpm(t, v, shift_time=0.0, **DRIVE)
    assert not d0["in_shift"].any()
    j = np.searchsorted(t, _t_cross(25.0))
    assert d0["rpm"][j - 1] - d0["rpm"][j] > 1000.0  # one-sample drop
    d1 = drive_rpm(t, v, shift_time=0.8, **DRIVE)
    assert d1["in_shift"].sum() == pytest.approx(4 * 0.8 * 2000, abs=8)


def test_launch_starts_at_idle_and_meets_first_gear():
    t, v = _accel()
    fs = 2000
    d = drive_rpm(t, v, **DRIVE)
    rpm = d["rpm"]
    k1, idle, r_lock = _k(1), 800.0, 1600.0  # default lock-up is 2 * idle
    assert rpm[0] == idle
    assert np.all(rpm[t <= T_STAND + 0.5 / SLOPE] == idle)  # engine idles while stationary
    assert d["launch_times"].shape == (1, 2)
    t_on, t_lock = d["launch_times"][0]
    assert T_STAND < t_on <= T_STAND + 0.5 / SLOPE and t_on > T_STAND + 0.5 / SLOPE - 1 / fs
    assert t_lock == pytest.approx(_t_cross(r_lock / k1), abs=1e-9)
    seg = (t >= t_on) & (t < t_lock)
    assert np.array_equal(d["slip"], seg)
    expected = idle + _w((t[seg] - t_on) / (t_lock - t_on)) * (r_lock - idle)
    assert np.allclose(rpm[seg], expected, atol=1e-8)
    assert np.all(rpm[seg] >= k1 * v[seg] - 1e-9)  # slip: never below the locked value
    assert np.all(np.diff(rpm[seg]) >= 0)
    mean_step = (r_lock - idle) / ((t_lock - t_on) * fs)
    assert rpm[seg][1] - rpm[seg][0] < 1e-2 * mean_step  # leaves idle with zero slope
    # meets the gear-1 curve at lock-up without a jump, and follows it afterwards
    j = np.flatnonzero(seg)[-1] + 1
    assert rpm[j] == pytest.approx(k1 * v[j], abs=1e-9)
    assert abs(rpm[j] - rpm[j - 1]) < 2 * mean_step + 0.5

    off = drive_rpm(t, v, launch_slip=False, **DRIVE)
    assert not off["slip"].any() and off["launch_times"].shape == (0, 2)
    static = speed_to_rpm(v, shift_speeds=DRIVE["shift_speeds"], idle=800.0, redline=6500.0, **CHAIN)
    assert np.allclose(off["rpm"][~off["in_shift"]], static[~off["in_shift"]], atol=1e-9)
    assert np.all(rpm >= off["rpm"] - 1e-9)  # slip only raises the engine speed


def test_explicit_launch_rpm():
    t, v = _accel()
    d = drive_rpm(t, v, launch_rpm=1200.0, **DRIVE)
    assert d["launch_times"][0, 1] == pytest.approx(_t_cross(1200.0 / _k(1)), abs=1e-9)
    for bad in (700.0, 800.0, 7000.0, 0.9 * 25.0 * _k(1) + 0.2 * 25.0 * _k(1)):
        with pytest.raises(ValueError, match="launch_rpm"):
            drive_rpm(t, v, launch_rpm=bad, **DRIVE)


def test_drive_cycle_with_downshifts_second_launch_and_limits():
    fs = 1000
    t = np.arange(int(34 * fs)) / fs
    v = rpm_profiles(t, "piecewise", knots=([0, 2, 8, 14, 22, 26, 28, 34], [0, 0, 50, 110, 50, 0, 0, 30]))
    d = drive_rpm(t, v, **DRIVE)
    rpm = d["rpm"]
    assert rpm.min() == 800.0 and 800.0 <= rpm.max() <= 6500.0
    # up through the gears, back down while braking to a stop, then up once more after the restart
    assert list(d["shift_from"]) == [1, 2, 3, 4, 5, 4, 3, 2, 1]
    assert list(d["shift_to"]) == [2, 3, 4, 5, 4, 3, 2, 1, 2]
    assert np.all(np.abs(np.diff(d["gear"])) <= 1)
    # a downshift raises the engine speed, monotonically, from the old to the new gear value
    for ts, a, b in zip(d["shift_times"], d["shift_from"], d["shift_to"]):
        idx = np.flatnonzero((t >= ts) & (t < ts + 0.3))
        dr = np.diff(rpm[idx])
        assert np.all(dr < 0) if b > a else np.all(dr > 0)
    # two launches (the vehicle stops and sets off again), each from idle
    assert d["launch_times"].shape == (2, 2)
    assert d["launch_times"][1, 0] == pytest.approx(28.0, abs=0.2)
    assert rpm[np.searchsorted(t, 27.0)] == 800.0
    assert np.max(np.abs(np.diff(rpm))) < 15.0  # no jumps anywhere (fs = 1 kHz)
    # rev limiter
    lim = drive_rpm(t, v, **{**DRIVE, "redline": 3000.0})
    assert lim["rpm"].max() == 3000.0 and lim["rpm"].min() == 800.0
    assert np.all(lim["rpm"] <= 3000.0)


def test_overlapping_shifts_start_from_the_value_reached():
    t, v = _accel()
    ds = dict(DRIVE, shift_speeds=[25.0, 26.0, 70.0, 100.0])
    d = drive_rpm(t, v, shift_time=0.5, **ds)
    rpm = d["rpm"]
    t1, t2 = d["shift_times"][:2]
    assert 0 < t2 - t1 < 0.5
    seg = (t >= t1) & (t < t2 + 0.5)
    assert np.all(np.diff(rpm[seg]) < 0)  # one continuous fall through both shifts
    assert np.max(np.abs(np.diff(rpm))) < 3.0
    j = np.searchsorted(t, t2 + 0.5)
    assert abs(rpm[j] - speed_to_rpm(v[j], gear=3, **CHAIN)) < 1e-9


def test_gear_skip_in_one_sample():
    t, _ = _accel()
    v = np.where(t < 5.0, 10.0, 80.0)
    d = drive_rpm(t, v, **DRIVE)
    assert list(d["shift_from"]) == [1] and list(d["shift_to"]) == [4]
    assert 4.9995 <= d["shift_times"][0] <= 5.0
    assert np.all((d["rpm"] >= 800.0) & (d["rpm"] <= 6500.0))


def test_drive_rpm_non_uniform_time_and_single_gear():
    t = np.concatenate((np.linspace(0, 5, 300), np.linspace(5.01, 10, 120)))
    v = np.linspace(20, 90, t.size)
    d = drive_rpm(
        t, v, gear_ratios=[2.0], final_drive=3.0, wheel_circumference=2.0, shift_speeds=(),
        idle=700.0, redline=6000.0,
    )
    assert not d["shift_times"].size and not d["in_shift"].any() and np.all(d["gear"] == 1)
    assert np.allclose(d["rpm"], np.clip(v / 3.6 / 2.0 * 60 * 2.0 * 3.0, 700.0, 6000.0))


@pytest.mark.parametrize(
    "change",
    [
        {"shift_speeds": [25.0, 20.0, 70.0, 100.0]},
        {"shift_speeds": [25.0, 45.0]},
        {"gear_ratios": [3.6, -2.0, 1.4, 1.0, 0.8]},
        {"idle": 7000.0},
        {"redline": 800.0},
        {"shift_time": -0.1},
        {"wheel_circumference": 0.0},
    ],
)
def test_drive_rpm_validation(change):
    t, v = _accel(200)
    with pytest.raises(ValueError):
        drive_rpm(t, v, **{**DRIVE, **change})
    with pytest.raises(ValueError):
        drive_rpm(t[::-1], v, **DRIVE)
    with pytest.raises(ValueError):
        drive_rpm(t, v[:-1], **DRIVE)


# --------------------------------------------------------------------------
# angular_speed_ripple
# --------------------------------------------------------------------------


def _ramp(fs, dur, r0, r1):
    n = int(fs * dur)
    return np.linspace(r0, r1, n)


def test_ripple_single_order_matches_closed_form():
    # d theta / d psi = 1 + a cos(2 pi theta + phi) has the closed-form solution
    # tan(x / 2) = sqrt((1+a)/(1-a)) tan(theta0 + s pi psi), s = sqrt(1-a^2).
    fs, a, phi, rev0 = 8000, 0.2, 0.7, 0.3
    rpm = _ramp(fs, 4.0, 1500.0, 4500.0)
    rpm_inst, rev = angular_speed_ripple(rpm, fs, ripple=[(1.0, a, phi)], rev0=rev0)
    psi = rpm_to_phase(rpm, fs)
    s, b = np.sqrt(1 - a * a), np.sqrt((1 - a) / (1 + a))
    theta0 = np.arctan(b * np.tan((TWO_PI * rev0 + phi) / 2))
    vartheta = theta0 + s * np.pi * psi
    x = 2 * np.unwrap(np.arctan2(np.sin(vartheta), b * np.cos(vartheta)))
    assert np.max(np.abs(rev - (x - phi) / TWO_PI)) < 1e-9
    assert abs(rev[0] - rev0) < 1e-10
    assert np.all(np.diff(rev) > 0)
    # over many cycles the shaft advances at sqrt(1 - a^2) times the nominal rate
    assert rev[-1] / psi[-1] == pytest.approx(s, abs=2e-3)
    # rpm_inst is rpm * g(rev), by definition
    assert np.allclose(rpm_inst, rpm * (1 + a * np.cos(TWO_PI * rev + phi)), rtol=0, atol=1e-9)


def test_ripple_multi_term_matches_independent_ode_solution():
    fs = 2000
    rpm = _ramp(fs, 0.5, 3000.0, 6000.0)
    terms = [(2.0, 0.004, 0.3), (4.0, 0.002, 1.1), (1.5, 0.003, 2.0), (0.5, 0.002, -0.4)]
    rev0 = -0.35  # negative start angle, half orders: period of two revolutions
    rpm_inst, rev = angular_speed_ripple(rpm, fs, ripple=terms, rev0=rev0)
    t = np.arange(rpm.size) / fs

    def rhs(tq, y):
        g = 1 + sum(a * np.cos(TWO_PI * k * y[0] + p) for k, a, p in terms)
        return [np.interp(tq, t, rpm) / 60.0 * g]

    sol = solve_ivp(rhs, (0, t[-1]), [rev0], t_eval=t, method="DOP853", rtol=1e-13, atol=1e-14,
                    max_step=1 / fs)
    assert np.max(np.abs(rev - sol.y[0])) < 1e-9
    assert np.all(np.diff(rev) > 0)


def test_ripple_rpm_inst_is_the_derivative_of_rev():
    fs = 16000
    rpm = _ramp(fs, 3.0, 1500.0, 4500.0)
    rev0 = 12.25
    rpm_inst, rev = angular_speed_ripple(rpm, fs, ripple=[(2.0, 0.004, 0.2), (4.0, 0.002, -1.0)],
                                         rev0=rev0)
    # trapezoid integral of rpm_inst/60 reproduces the exact angle up to the sampling error
    # of a 150-300 Hz ripple at 16 kHz (second order in ripple frequency / fs)
    assert np.max(np.abs(rpm_to_phase(rpm_inst, fs, rev0=rev0) - rev)) < 3e-7
    # central difference of rev reproduces rpm_inst/60
    d = np.gradient(rev) * fs * 60.0
    assert np.max(np.abs(d[2:-2] - rpm_inst[2:-2])) < 5e-3 * np.max(rpm) * 0.004
    # mean speed preserved to the second order in the ripple: the shaft dwells longer where it is
    # slow, so the time average of rpm_inst is below the nominal speed by sum(a^2) / 2
    assert rpm_inst.mean() / rpm.mean() - 1 == pytest.approx(-(0.004**2 + 0.002**2) / 2, abs=1.5e-6)


def test_ripple_without_terms_is_the_nominal_speed():
    fs = 4000
    rpm = _ramp(fs, 1.0, 800.0, 2000.0)
    rpm_inst, rev = angular_speed_ripple(rpm, fs, ripple=(), rev0=0.4)
    assert np.array_equal(rpm_inst, rpm) and rpm_inst is not rpm
    assert np.allclose(rev, rpm_to_phase(rpm, fs, rev0=0.4), atol=1e-12)


def test_ripple_order_and_amplitude_measured_in_the_angle_domain():
    fs = 8000
    rpm = _ramp(fs, 6.0, 2400.0, 3600.0)
    terms = [(2.0, 0.004, 0.7), (4.0, 0.0015, -1.2)]
    rpm_inst, rev = angular_speed_ripple(rpm, fs, ripple=terms, rev0=0.0)
    # what a perfect tachometer sees, resampled to angle with its own phase
    a = angular_resample(rpm_inst, fs, rpm_inst, samples_per_rev=128, max_order=40.0)
    trend = np.polynomial.Polynomial.fit(a.rev, a.x, 6)
    rel = a.x / trend(a.rev) - 1.0
    whole = int(np.floor((a.rev[-1] - 20.0) / 2.0)) * 2  # whole periods away from the edges
    sel = (a.rev >= 10.0) & (a.rev < 10.0 + whole)
    ang, r = a.rev[sel], rel[sel]

    def line(order):
        return np.mean(r * np.exp(-2j * np.pi * order * ang))

    for order, amp, phase in terms:
        c = line(order)
        assert 2 * abs(c) == pytest.approx(amp, rel=2e-3)
        assert np.angle(c * np.exp(-1j * phase)) == pytest.approx(0.0, abs=2e-3)
    for absent in (1.0, 3.0, 5.0, 6.0, 1.5):
        assert 2 * abs(line(absent)) < 1e-6


def test_ripple_validation_and_edge_cases():
    rpm = np.full(200, 3000.0)
    with pytest.raises(ValueError, match="periodic"):
        angular_speed_ripple(rpm, 1000.0, ripple=[(1.0, 0.01, 0.0), (np.sqrt(2.0), 0.01, 0.0)])
    with pytest.raises(ValueError, match="below 1"):
        angular_speed_ripple(rpm, 1000.0, ripple=[(2.0, 0.6, 0.0), (4.0, 0.5, 0.0)])
    with pytest.raises(ValueError):
        angular_speed_ripple(-rpm, 1000.0, ripple=[(2.0, 0.01, 0.0)])
    with pytest.raises(ValueError):
        angular_speed_ripple(rpm, 1000.0, ripple=[(0.0, 0.01, 0.0)])
    with pytest.raises(ValueError):
        angular_speed_ripple(rpm, 1000.0, ripple=[(2.0, 0.01)])
    # a stopped shaft keeps its angle; rev never decreases
    rpm2 = np.concatenate((np.full(100, 600.0), np.zeros(100), np.full(100, 900.0)))
    _, rev = angular_speed_ripple(rpm2, 1000.0, ripple=[(2.0, 0.05, 1.0)])
    assert np.all(np.diff(rev) >= 0) and rev[150] == rev[100] == rev[199]
    # many seconds at audio rate are cheap (the angle is not stepped sample by sample)
    r, v = angular_speed_ripple(np.full(16000 * 20, 4000.0), 16000.0, ripple=[(2.0, 0.004, 0.0)])
    assert np.all(np.diff(v) > 0) and r.shape == v.shape


# --------------------------------------------------------------------------
# angle_impulses
# --------------------------------------------------------------------------


def _burst(tau, amp, f0, zeta):
    """Analytic burst, written out independently: peak-normalised damped sine."""
    w0 = TWO_PI * f0
    wd = w0 * np.sqrt(1 - zeta**2)
    tp = np.arccos(zeta) / wd  # time of the first maximum
    norm = np.exp(-zeta * w0 * tp) * np.sin(wd * tp)
    return np.where(tau >= 0, amp / norm * np.exp(-zeta * w0 * np.maximum(tau, 0)) * np.sin(wd * tau), 0.0)


def _linear_ramp(fs, dur, r0, r1):
    t = np.arange(int(fs * dur)) / fs
    slope = (r1 - r0) / dur
    rpm = r0 + slope * t  # exactly linear, so the trapezoidal angle is exact
    return t, rpm, rpm_to_phase(rpm, fs), slope


def _ramp_time_at_angle(theta, r0, slope):
    return (-r0 + np.sqrt(r0 * r0 + 120.0 * slope * theta)) / slope


def test_impulse_times_are_the_analytic_angle_crossings():
    fs = 8000
    t, rpm, rev, slope = _linear_ramp(fs, 4.0, 1500.0, 3500.0)
    out = angle_impulses(rev, fs, per_rev=2.0, amplitude=1.0, resonance_hz=1200.0, damping=0.05)
    k = np.arange(int(np.floor(rev[-1] * 2.0)) + 1)
    assert np.allclose(out["impulse_rev_nominal"], k / 2.0)
    assert np.array_equal(out["impulse_rev"], out["impulse_rev_nominal"])
    assert np.max(np.abs(out["impulse_times"] - _ramp_time_at_angle(k / 2.0, 1500.0, slope))) < 1e-12
    # same with the speed given explicitly, and with fractional events per revolution
    out2 = angle_impulses(rev, fs, per_rev=1.5, amplitude=1.0, resonance_hz=1200.0, damping=0.05, rpm=rpm)
    k2 = np.arange(int(np.floor(rev[-1] * 1.5)) + 1)
    assert np.max(np.abs(out2["impulse_times"] - _ramp_time_at_angle(k2 / 1.5, 1500.0, slope))) < 1e-12
    # a record that does not start at angle 0 keeps only the events inside it
    out3 = angle_impulses(rev + 0.7, fs, per_rev=1.0, amplitude=1.0, resonance_hz=1200.0, damping=0.05)
    assert out3["impulse_rev_nominal"][0] == 1.0 and out3["impulse_rev_nominal"][-1] <= rev[-1] + 0.7
    assert np.allclose(out3["impulse_times"], _ramp_time_at_angle(out3["impulse_rev_nominal"] - 0.7, 1500.0, slope),
                       atol=1e-12)


def test_burst_equals_the_analytic_damped_resonance_at_sub_sample_times():
    fs, f0, zeta, amp = 16000, 1500.0, 0.05, 0.8
    n = int(0.1 * fs)
    rev0 = 0.0137  # events at angles 1..5, between samples
    rpm = np.full(n, 3000.0)
    rev = rpm_to_phase(rpm, fs, rev0=rev0)
    out = angle_impulses(rev, fs, per_rev=1.0, amplitude=amp, resonance_hz=f0, damping=zeta)
    te = (np.arange(1, 6) - rev0) / 50.0
    assert np.allclose(out["impulse_times"], te, atol=1e-12)
    assert np.any(np.abs(te * fs - np.round(te * fs)) > 0.1)  # genuinely sub-sample
    t = np.arange(n) / fs
    expected = sum(_burst(t - e, amp, f0, zeta) for e in te)
    assert np.max(np.abs(out["x"] - expected)) < 2e-6 * amp  # the cut tail is -120 dB
    assert np.all(out["x"][: int(np.ceil(te[0] * fs))] == 0.0)  # causal: nothing before the first event
    # peak: the continuous-time maximum of one burst is the amplitude
    fine = np.arange(0, 0.02, 1e-7)
    assert np.max(np.abs(_burst(fine, amp, f0, zeta))) == pytest.approx(amp, rel=1e-5)
    first = out["x"][int(np.ceil(te[0] * fs)) : int(np.ceil(te[0] * fs)) + 40]
    assert 0.85 * amp < np.max(np.abs(first)) <= amp * (1 + 1e-9)
    assert np.all(out["impulse_amplitude"] == amp)


def test_burst_decay_rate_is_zeta_omega():
    fs, f0, zeta = 16000, 1000.0, 0.05
    n = int(0.45 * fs)
    rev = rpm_to_phase(np.full(n, 600.0), fs, rev0=0.0042)  # 10 rev/s: events 100 ms apart
    out = angle_impulses(rev, fs, per_rev=1.0, amplitude=1.0, resonance_hz=f0, damping=zeta)
    env = np.abs(signal.hilbert(out["x"]))
    t = np.arange(n) / fs
    te = out["impulse_times"][1]
    sel = (t > te + 0.002) & (t < te + 0.015)
    slope = np.polyfit(t[sel] - te, np.log(env[sel]), 1)[0]
    assert -slope == pytest.approx(zeta * TWO_PI * f0, rel=0.01)


def test_stem_is_the_superposition_of_overlapping_bursts(monkeypatch):
    fs, f0, zeta = 8000, 900.0, 0.004  # long ring-down: many overlapping events
    n = int(0.6 * fs)
    rpm = np.linspace(2400.0, 4200.0, n)
    rev = rpm_to_phase(rpm, fs)
    kw = dict(per_rev=3.0, amplitude=1.3, resonance_hz=f0, damping=zeta, rpm=rpm)
    out = angle_impulses(rev, fs, **kw)
    t = np.arange(n) / fs
    expected = sum(_burst(t - e, 1.3, f0, zeta) for e in out["impulse_times"])
    assert out["impulse_times"].size > 80 and out["burst_samples"] == n
    assert np.max(np.abs(out["x"] - expected)) < 1e-9  # the burst covers the whole record: no cut tail
    monkeypatch.setattr(_sources, "_CHUNK_ELEMENTS", 5000)  # force several blocks
    chunked = angle_impulses(rev, fs, **kw)
    assert np.allclose(chunked["x"], out["x"], rtol=0, atol=1e-12)


def test_amplitude_follows_the_rpm_law():
    fs = 8000
    t, rpm, rev, slope = _linear_ramp(fs, 4.0, 1000.0, 3000.0)
    law_rpm, law_gain = [1000.0, 2000.0, 3000.0], [1.0, 0.5, 2.0]
    out = angle_impulses(rev, fs, per_rev=2.0, amplitude=0.6, resonance_hz=1100.0, damping=0.1,
                         amplitude_rpm=law_rpm, amplitude_gain=law_gain, rpm=rpm)
    r_event = 1000.0 + slope * out["impulse_times"]
    assert np.allclose(out["impulse_amplitude"], 0.6 * np.interp(r_event, law_rpm, law_gain), atol=1e-9)
    assert out["impulse_amplitude"].min() < 0.31 and out["impulse_amplitude"].max() > 1.18
    # each isolated burst peaks at its own amplitude
    for e, a in list(zip(out["impulse_times"], out["impulse_amplitude"]))[5:60:11]:
        i = int(np.ceil(e * fs))
        peak = np.max(np.abs(out["x"][i : i + 12]))
        assert 0.8 * a < peak < 1.05 * a
    with pytest.raises(ValueError, match="together"):
        angle_impulses(rev, fs, per_rev=2.0, amplitude=1.0, resonance_hz=1100.0, damping=0.1,
                       amplitude_rpm=law_rpm)


def test_orders_at_multiples_of_per_rev_and_fixed_frequency_hump():
    fs, dur, per = 8000, 8.0, 2.0
    f0, zeta, amp = 1200.0, 0.1, 1.0
    t, rpm, rev, slope = _linear_ramp(fs, dur, 1800.0, 3600.0)
    out = angle_impulses(rev, fs, per_rev=per, amplitude=amp, resonance_hz=f0, damping=zeta, rpm=rpm)
    m = rpm_order_map(out["x"], fs, rpm, order_resolution=0.25, max_order=40.0, amplitude="peak")
    mag = np.abs(m.values)[:, 4:-4]
    rpm_m = m.rpm[4:-4]
    even = [i for i, o in enumerate(m.axis) if 0 < o <= 20 and abs(o / per - round(o / per)) < 1e-9]
    odd = [i for i, o in enumerate(m.axis) if 0 < o <= 20 and abs(o - round(o)) < 1e-9 and round(o) % 2 == 1]
    ratio = 10 * np.log10(np.mean(mag[even] ** 2, axis=0) / np.mean(mag[odd] ** 2, axis=0))
    assert ratio.min() > 25.0  # the energy sits on the multiples of per_rev
    # line amplitude = 2 A |H(f)| per_rev rpm / 60 / N, with H the transfer function of the burst:
    # the resonance stays at f0 while the orders sweep through it
    w0 = TWO_PI * f0
    wd = w0 * np.sqrt(1 - zeta**2)
    tp = np.arccos(zeta) / wd
    norm = np.exp(-zeta * w0 * tp) * np.sin(wd * tp)
    for order in (4, 8, 12, 16, 20, 24, 30):
        i = int(np.argmin(np.abs(m.axis - order)))
        w = TWO_PI * order * rpm_m / 60.0
        h = np.abs(wd / (w0**2 - w**2 + 2j * zeta * w0 * w))
        pred = 2 * (amp / norm) * h * per * rpm_m / 60.0
        err_db = 20 * np.log10(mag[i] / pred)
        assert np.max(np.abs(err_db)) < 1.0, order
    # the hump: along one order the line is strongest where order * rpm / 60 is near f0
    i = int(np.argmin(np.abs(m.axis - 28)))
    f_peak = 28 * rpm_m[np.argmax(mag[i])] / 60.0
    assert f_peak == pytest.approx(f0, rel=0.1)


def test_angle_jitter_is_seeded_and_located_exactly():
    fs = 8000
    t, rpm, rev, slope = _linear_ramp(fs, 6.0, 2000.0, 3000.0)
    kw = dict(per_rev=2.0, amplitude=1.0, resonance_hz=1000.0, damping=0.08, jitter_rev=0.002)
    a = angle_impulses(rev, fs, seed=5, **kw)
    b = angle_impulses(rev, fs, seed=5, **kw)
    c = angle_impulses(rev, fs, seed=6, **kw)
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["impulse_times"], b["impulse_times"])
    assert not np.array_equal(a["impulse_times"], c["impulse_times"])
    err = a["impulse_rev"] - a["impulse_rev_nominal"]
    assert np.std(err) == pytest.approx(0.002, rel=0.15) and abs(np.mean(err)) < 3e-4
    assert np.all((a["impulse_rev"] >= rev[0]) & (a["impulse_rev"] <= rev[-1]))
    assert np.allclose(a["impulse_times"], _ramp_time_at_angle(a["impulse_rev"], 2000.0, slope), atol=1e-12)
    # a Generator is accepted, and no jitter needs no random numbers
    angle_impulses(rev, fs, seed=np.random.default_rng(1), **kw)
    z = angle_impulses(rev, fs, per_rev=2.0, amplitude=1.0, resonance_hz=1000.0, damping=0.08)
    assert np.array_equal(z["impulse_rev"], z["impulse_rev_nominal"])


def test_impulses_are_locked_to_the_exact_angle_of_a_rippled_speed():
    # exact angle on an 8 times finer grid (the same linear nominal speed) as independent truth
    fs, fine = 8000, 8
    kw = dict(ripple=[(2.0, 0.01, 0.4)])
    rpm = _ramp(fs, 1.0, 3000.0, 3600.0)
    rpm_inst, rev = angular_speed_ripple(rpm, fs, **kw)
    n_fine = (rpm.size - 1) * fine + 1
    rpm_f = np.linspace(rpm[0], rpm[-1], n_fine)
    rpm_f_inst, rev_f = angular_speed_ripple(rpm_f, fs * fine, **kw)
    out = angle_impulses(rev, fs, per_rev=2.0, amplitude=1.0, resonance_hz=2000.0, damping=0.1, rpm=rpm_inst)
    te = out["impulse_times"]
    t_f = np.arange(n_fine) / (fs * fine)
    truth_angle = CubicHermiteSpline(t_f, rev_f, rpm_f_inst / 60.0)(te)
    assert te.size > 100
    assert np.max(np.abs(truth_angle - out["impulse_rev_nominal"])) < 1e-8
    # without ripple knowledge (no rpm given) the finite-difference slopes are less exact but close
    out2 = angle_impulses(rev, fs, per_rev=2.0, amplitude=1.0, resonance_hz=2000.0, damping=0.1)
    assert np.max(np.abs(out2["impulse_times"] - te)) < 2e-6


def test_angle_impulses_validation():
    fs = 8000
    t, rpm, rev, _ = _linear_ramp(fs, 1.0, 1000.0, 2000.0)
    ok = dict(per_rev=2.0, amplitude=1.0, resonance_hz=1000.0, damping=0.1)
    for change in ({"resonance_hz": 4000.0}, {"damping": 1.0}, {"damping": 0.0}, {"per_rev": 0.0},
                   {"jitter_rev": -1.0}, {"amplitude": np.nan}):
        with pytest.raises(ValueError):
            angle_impulses(rev, fs, **{**ok, **change})
    with pytest.raises(ValueError, match="non-decreasing"):
        angle_impulses(rev[::-1], fs, **ok)
    with pytest.raises(ValueError):
        angle_impulses(rev, fs, rpm=rpm[:-1], **ok)
    with pytest.raises(ValueError):
        angle_impulses(rev, fs, amplitude_rpm=[1000.0, 900.0], amplitude_gain=[1.0, 1.0], **{**ok})
    # a record shorter than one event interval has no events and an empty stem
    short = angle_impulses(rev[:20] - rev[0] + 0.1, fs, **ok)
    assert short["impulse_times"].size == 0 and not short["x"].any()


# --------------------------------------------------------------------------
# sideband_orders
# --------------------------------------------------------------------------


def _truth(orders, rpm_value=3000.0, fs=4000, dur=0.5):
    rpm = np.full(int(fs * dur), rpm_value)
    return synth_runup(fs, rpm, orders, seed=0)


def test_sideband_orders_levels_and_frequencies_are_exact_in_the_truth():
    levels = [-6.0, -12.0, -20.0]
    specs = sideband_orders(10.0, 1.0, levels, carrier_amplitude=2.5, shaft=0)
    assert [s["order"] for s in specs] == [10, 9, 11, 8, 12, 7, 13]
    assert all(set(s) == {"order", "amplitude", "shaft"} for s in specs)  # accepted by synth_runup
    assert specs[0]["amplitude"] == 2.5
    sig = _truth(specs)
    amp = sig.order_amplitude[100]
    db = 20 * np.log10(amp[1:] / amp[0])
    assert np.allclose(db, np.repeat(levels, 2), atol=1e-9)
    assert amp[0] == pytest.approx(2.5)
    assert np.allclose(sig.order_frequency[0], np.array([10, 9, 11, 8, 12, 7, 13]) * 50.0)


def test_sideband_levels_per_side_and_absent_sidebands():
    specs = sideband_orders(8.0, 0.5, [[-10.0, -20.0], [-np.inf, -30.0]], carrier_amplitude=1.0)
    assert [s["order"] for s in specs] == [8.0, 7.5, 8.5, 7.0, 9.0]
    amp = np.array([s["amplitude"] for s in specs])
    assert np.allclose(20 * np.log10(amp[[1, 2, 4]]), [-10.0, -20.0, -30.0])
    assert amp[3] == 0.0
    sig = _truth(specs)
    assert sig.order_amplitude[50, 3] == 0.0


def test_amplitude_modulation_identity():
    m, phi_c, phi_m = 0.4, 0.3, 1.1
    specs = sideband_orders(12.0, 2.0, [20 * np.log10(m / 2)], carrier_amplitude=1.7,
                            modulation="am", carrier_phase=phi_c, mod_phase=phi_m)
    sig = _truth(specs, fs=8000)
    rev = sig.rev
    expected = 1.7 * (1 + m * np.cos(TWO_PI * 2.0 * rev + phi_m)) * np.cos(TWO_PI * 12.0 * rev + phi_c)
    assert np.max(np.abs(sig.stems["orders"] - expected)) < 1e-12


def test_phase_modulation_first_order_and_bessel_identities():
    beta, phi_c, phi_m = 0.05, 0.2, -0.9
    specs = sideband_orders(12.0, 2.0, [20 * np.log10(beta / 2)], modulation="pm",
                            carrier_phase=phi_c, mod_phase=phi_m)
    sig = _truth(specs, fs=8000)
    pc, pm = TWO_PI * 12.0 * sig.rev + phi_c, TWO_PI * 2.0 * sig.rev + phi_m
    assert np.max(np.abs(sig.stems["orders"] - (np.cos(pc) - beta * np.sin(pc) * np.sin(pm)))) < 1e-12
    # a larger index with Bessel levels reproduces cos(Phi_c + beta sin(Phi_m)) up to the 4th sideband
    beta = 0.8
    levels = [20 * np.log10(special.jv(k, beta) / special.jv(0, beta)) for k in (1, 2, 3)]
    specs = sideband_orders(12.0, 2.0, levels, carrier_amplitude=special.jv(0, beta), modulation="pm",
                            carrier_phase=phi_c, mod_phase=phi_m)
    sig = _truth(specs, fs=8000)
    pc, pm = TWO_PI * 12.0 * sig.rev + phi_c, TWO_PI * 2.0 * sig.rev + phi_m
    assert np.max(np.abs(sig.stems["orders"] - np.cos(pc + beta * np.sin(pm)))) < 2 * abs(special.jv(4, beta))


def test_sidebands_below_zero_order_are_folded_and_dc_is_dropped():
    # 2 - 2*1.5 = -1: folded to order 1 with the phase negated; 2 - 1.5 = 0.5 stays
    specs = sideband_orders(2.0, 1.5, [-10.0, -10.0], modulation="am", carrier_phase=0.4, mod_phase=0.9)
    assert [s["order"] for s in specs] == [2.0, 0.5, 3.5, 1.0, 5.0]
    sig = _truth(specs, fs=8000)
    a1 = 10 ** (-10 / 20)
    expected = (
        np.cos(TWO_PI * 2.0 * sig.rev + 0.4)
        + a1 * np.cos(TWO_PI * (2.0 - 1.5) * sig.rev + 0.4 - 0.9)
        + a1 * np.cos(TWO_PI * (2.0 + 1.5) * sig.rev + 0.4 + 0.9)
        + a1 * np.cos(TWO_PI * (2.0 - 3.0) * sig.rev + 0.4 - 1.8)
        + a1 * np.cos(TWO_PI * (2.0 + 3.0) * sig.rev + 0.4 + 1.8)
    )
    assert np.max(np.abs(sig.stems["orders"] - expected)) < 1e-12
    dc = sideband_orders(2.0, 2.0, [-10.0])  # lower sideband at order 0
    assert [s["order"] for s in dc] == [2.0, 4.0]


def test_sideband_orders_without_modulation_leave_phases_to_the_generator():
    specs = sideband_orders(5.0, 1.0, [-6.0])
    assert all("phase" not in s for s in specs)
    a, b = _truth(specs), _truth(specs)
    assert np.array_equal(a.x, b.x)


def test_sideband_orders_validation():
    for kw in (
        dict(carrier_order=0.0, mod_order=1.0, levels_db=[-6.0]),
        dict(carrier_order=5.0, mod_order=-1.0, levels_db=[-6.0]),
        dict(carrier_order=5.0, mod_order=1.0, levels_db=[]),
        dict(carrier_order=5.0, mod_order=1.0, levels_db=[np.nan]),
        dict(carrier_order=5.0, mod_order=1.0, levels_db=[np.inf]),
        dict(carrier_order=5.0, mod_order=1.0, levels_db=np.zeros((2, 3))),
        dict(carrier_order=5.0, mod_order=1.0, levels_db=[-6.0], modulation="fm"),
        dict(carrier_order=5.0, mod_order=1.0, levels_db=[-6.0], carrier_amplitude=-1.0),
        dict(carrier_order=5.0, mod_order=1.0, levels_db=[-6.0], shaft=-1),
    ):
        with pytest.raises(ValueError):
            sideband_orders(**kw)


# --------------------------------------------------------------------------
# carrier_components
# --------------------------------------------------------------------------


def test_carrier_frequency_truth_is_the_derivative_of_the_phase():
    fs = 16000
    rpm = _ramp(fs, 2.0, 1000.0, 3000.0)
    rev = rpm_to_phase(rpm, fs)
    terms = [(0.0, 1.0), (6.0, 0.3), (-6.0, 0.25), (12.0, 0.1), (-12.0, 0.05)]
    out = carrier_components(rev, fs, rpm, carrier_hz=4000.0, terms=terms, seed=3)
    q = np.array([x[0] for x in terms])
    assert np.allclose(out["frequency"], 4000.0 + np.outer(rpm / 60.0, q), atol=1e-9)
    assert np.array_equal(out["q"], q)
    # derivative of phase / 2 pi = frequency: the trapezoid rule makes this exact between samples
    mid = np.diff(out["phase"], axis=0) * fs / TWO_PI
    assert np.max(np.abs(mid - 0.5 * (out["frequency"][1:] + out["frequency"][:-1]))) < 1e-6
    # components are A cos(phase), the carrier term is a plain tone, x is their sum
    amp = np.array([x[1] for x in terms])
    assert np.allclose(out["amplitude"], amp)
    assert np.allclose(out["stems"], out["amplitude"] * np.cos(out["phase"]), atol=1e-12)
    assert np.allclose(out["x"], out["stems"].sum(axis=1), atol=1e-12)
    t = np.arange(rpm.size) / fs
    assert np.allclose(out["stems"][:, 0], np.cos(TWO_PI * 4000.0 * t + out["phase0"][0]), atol=1e-9)
    assert np.all((out["phase0"] >= 0) & (out["phase0"] < TWO_PI))


def test_carrier_frequency_truth_with_speed_ripple():
    fs = 16000
    rpm = _ramp(fs, 2.0, 2400.0, 3000.0)
    rpm_inst, rev = angular_speed_ripple(rpm, fs, ripple=[(4.0, 0.004, 0.3)])
    out = carrier_components(rev, fs, rpm_inst, carrier_hz=3000.0, terms=[(6.0, 1.0), (-6.0, 1.0)], seed=0)
    inst = np.gradient(out["phase"], axis=0) * fs / TWO_PI  # central difference of the exact phase
    assert np.max(np.abs(inst[2:-2] - out["frequency"][2:-2])) < 5e-3  # Hz, vs 1.2 Hz ripple depth
    ripple_depth_hz = 6 * (rpm_inst - rpm).max() / 60.0
    assert ripple_depth_hz > 1.0


def test_carrier_components_are_muted_at_nyquist_and_at_non_positive_frequency():
    fs = 16000
    rpm = _ramp(fs, 2.0, 2000.0, 6000.0)
    rev = rpm_to_phase(rpm, fs)
    out = carrier_components(rev, fs, rpm, carrier_hz=7500.0, terms=[(6.0, 0.9), (-6.0, 0.4)], seed=1)
    f = out["frequency"]
    above = f[:, 0] >= 0.5 * fs
    assert above.any() and (~above).any()
    # f = 7500 + 6 rpm / 60 reaches fs / 2 at 5000 rpm: amplitude drops to exactly zero, no clipping
    assert np.all(out["amplitude"][above, 0] == 0.0) and np.all(out["stems"][above, 0] == 0.0)
    assert np.all(out["amplitude"][~above, 0] == 0.9)
    assert np.all(out["amplitude"][:, 1] == 0.4)  # the lower sideband stays in band
    assert f[:, 0].max() > 0.5 * fs  # the truth keeps the unmuted frequency
    assert np.allclose(out["x"][above], out["stems"][above, 1])
    # a lower sideband that reaches zero frequency is muted, not folded
    rpm2 = _ramp(fs, 2.0, 200.0, 3000.0)
    out2 = carrier_components(rpm_to_phase(rpm2, fs), fs, rpm2, carrier_hz=100.0, terms=[(-6.0, 0.5)], seed=1)
    f2 = out2["frequency"][:, 0]
    assert f2[0] > 0 and f2[-1] < 0
    assert np.all(out2["amplitude"][f2 <= 0, 0] == 0.0) and np.all(out2["amplitude"][f2 > 0, 0] == 0.5)


def test_carrier_components_mute_exactly_at_the_band_edges():
    fs = 8000
    rpm = np.full(400, 6000.0)  # 100 rev/s
    rev = rpm_to_phase(rpm, fs)
    # 3000 + 10 * 100 = 4000 Hz = fs / 2 and 600 - 6 * 100 = 0 Hz exactly: both are muted
    edge = carrier_components(rev, fs, rpm, carrier_hz=3000.0, terms=[(10.0, 1.0), (9.99, 1.0)], seed=0)
    assert np.all(edge["frequency"][:, 0] == 4000.0)
    assert np.all(edge["amplitude"][:, 0] == 0.0) and np.all(edge["amplitude"][:, 1] == 1.0)
    zero = carrier_components(rev, fs, rpm, carrier_hz=600.0, terms=[(-6.0, 1.0), (-5.99, 1.0)], seed=0)
    assert np.all(zero["frequency"][:, 0] == 0.0)
    assert np.all(zero["amplitude"][:, 0] == 0.0) and np.all(zero["amplitude"][:, 1] == 1.0)


def test_carrier_components_peak_amplitude_and_determinism():
    fs, n = 8000, 8000
    rpm = np.full(n, 1200.0)
    rev = rpm_to_phase(rpm, fs)
    kw = dict(carrier_hz=100.0, terms=[(0.0, 0.7)])  # 80 samples per cycle, 100 whole cycles
    a = carrier_components(rev, fs, rpm, seed=11, **kw)
    assert np.max(np.abs(a["x"])) == pytest.approx(0.7, abs=2e-3)  # peak, not RMS
    assert np.sqrt(np.mean(a["x"] ** 2)) == pytest.approx(0.7 / np.sqrt(2), rel=1e-6)  # whole cycles
    b = carrier_components(rev, fs, rpm, seed=11, **kw)
    c = carrier_components(rev, fs, rpm, seed=12, **kw)
    g1 = carrier_components(rev, fs, rpm, seed=np.random.default_rng(4), **kw)
    g2 = carrier_components(rev, fs, rpm, seed=np.random.default_rng(4), **kw)
    for key in ("x", "phase", "phase0", "frequency", "amplitude"):
        assert np.array_equal(a[key], b[key]) and np.array_equal(g1[key], g2[key])
    assert not np.array_equal(a["phase0"], c["phase0"])
    # sideband stems stay at fixed phase offsets from the seeded phases, so adding terms does not
    # change the phase of the first term
    d = carrier_components(rev, fs, rpm, carrier_hz=100.0, terms=[(0.0, 0.7), (4.0, 0.2)], seed=11)
    assert d["phase0"][0] == a["phase0"][0]


def test_carrier_components_validation():
    fs = 4000
    rpm = np.full(100, 1000.0)
    rev = rpm_to_phase(rpm, fs)
    ok = dict(carrier_hz=500.0, terms=[(2.0, 1.0)])
    with pytest.raises(ValueError):
        carrier_components(rev, fs, rpm[:-1], **ok)
    for change in ({"carrier_hz": -1.0}, {"terms": [(2.0, -1.0)]}, {"terms": [2.0, 1.0]}, {"terms": []},
                   {"terms": [(np.nan, 1.0)]}):
        with pytest.raises(ValueError):
            carrier_components(rev, fs, rpm, **{**ok, **change})
    with pytest.raises(ValueError):
        carrier_components(rev, 0.0, rpm, **ok)


# --------------------------------------------------------------------------
# the sources together
# --------------------------------------------------------------------------


def test_sources_compose_with_synth_runup_on_one_speed_truth():
    fs = 8000
    t = np.arange(int(8 * fs)) / fs
    v = rpm_profiles(t, "piecewise", knots=([0, 1, 7, 8], [0, 0, 90, 90]))
    d = drive_rpm(t, v, **DRIVE)
    rpm_inst, rev = angular_speed_ripple(d["rpm"], fs, ripple=[(2.0, 0.004, 0.3)])
    orders = [{"order": 2.0, "amplitude": 1.0, "phase": 0.0}] + sideband_orders(
        9.0, 1.0, [-12.0], modulation="am", carrier_phase=0.5, mod_phase=0.2
    )
    sig = synth_runup(fs, rpm_inst, orders, seed=1)
    # synth_runup integrates the speed it is given with the trapezoid rule; the exact angle
    # differs by the sampling error of the ripple only (about 1e-6 rev here)
    assert np.max(np.abs(sig.rev - rev)) < 1e-5
    imp = angle_impulses(rev, fs, per_rev=2.0, amplitude=0.3, resonance_hz=1800.0, damping=0.08,
                         rpm=rpm_inst, jitter_rev=0.001, seed=2)
    car = carrier_components(rev, fs, rpm_inst, carrier_hz=2500.0, terms=[(0.0, 0.1), (6.0, 0.05)], seed=3)
    total = sig.x + imp["x"] + car["x"]
    assert total.shape == t.shape and np.all(np.isfinite(total))
    assert np.all(d["rpm"] >= 800.0) and d["shift_times"].size == 3  # 90 km/h stays below the last shift
