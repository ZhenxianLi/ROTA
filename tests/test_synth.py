"""Tests for rota.synth: profiles, order truth, resonances, noise and tachometer."""

import json

import numpy as np
import pytest
from scipy import signal

import rota
from rota import speed, synth
from rota.synth import SynthSignal, rpm_profiles, synth_runup

TWO_PI = 2 * np.pi


def _ramp_rpm(fs, dur, lo, hi):
    t = np.arange(int(fs * dur)) / fs
    return rpm_profiles(t, "linear", start=lo, stop=hi)


# --------------------------------------------------------------------------
# rpm_profiles
# --------------------------------------------------------------------------


def test_profile_constant():
    t = np.linspace(0, 5, 51)
    y = rpm_profiles(t, "constant", start=900.0)
    assert y.shape == t.shape
    assert np.all(y == 900.0)


def test_profile_linear_endpoints_and_holds():
    t = np.linspace(0, 10, 1001)
    y = rpm_profiles(t, "linear", start=800, stop=3000, t_start=2, t_stop=8)
    assert np.all(y[t <= 2] == 800)
    assert np.all(y[t >= 8] == 3000)
    assert y[500] == pytest.approx(1900.0)  # t = 5
    mid = (t > 2) & (t < 8)
    assert np.allclose(np.diff(y[mid]), 2200 / 6 * 0.01)


def test_profile_default_span_is_whole_axis():
    t = np.linspace(1, 3, 21)
    y = rpm_profiles(t, "linear", start=10, stop=30)
    assert y[0] == 10 and y[-1] == 30


def test_profile_exponential_constant_relative_rate():
    t = np.linspace(0, 4, 401)
    y = rpm_profiles(t, "exponential", start=500, stop=4000)
    assert y[0] == pytest.approx(500) and y[-1] == pytest.approx(4000)
    assert y[200] == pytest.approx(np.sqrt(500 * 4000))
    assert np.allclose(np.diff(np.log(y)), np.log(8) / 400)
    d = rpm_profiles(t, "exponential", start=4000, stop=500)  # run-down
    assert np.all(np.diff(d) < 0)
    z = rpm_profiles(t, "exponential", start=500, stop=4000, t_start=1, t_stop=3)
    assert np.all(z[t <= 1] == 500) and np.all(z[t >= 3] == 4000)


def test_profile_power():
    t = np.linspace(0, 1, 101)
    y = rpm_profiles(t, "power", start=0.0, stop=100.0, exponent=2.0)
    assert y[50] == pytest.approx(25.0)
    assert y[0] == 0 and y[-1] == 100
    y = rpm_profiles(t, "power", start=0.0, stop=100.0, exponent=0.5)
    assert y[25] == pytest.approx(50.0)
    assert np.all(np.diff(y) > 0)


def test_profile_s_curve_zero_end_slopes():
    t = np.linspace(0, 10, 2001)
    y = rpm_profiles(t, "s_curve", start=1000, stop=5000, t_start=1, t_stop=9)
    assert y[0] == 1000 and y[-1] == 5000
    assert np.all(y[t <= 1] == 1000) and np.all(y[t >= 9] == 5000)
    d = np.gradient(y, t)
    mean_slope = 4000 / 8
    i1, i2 = np.searchsorted(t, 1.0), np.searchsorted(t, 9.0)
    assert abs(d[i1]) < 1e-3 * mean_slope  # zero slope leaving idle
    assert abs(d[i2]) < 1e-3 * mean_slope  # and arriving at the top
    assert d.max() == pytest.approx(0.5 * np.pi * mean_slope, rel=1e-3)
    mid = np.searchsorted(t, 5.0)
    assert y[mid] == pytest.approx(3000.0)
    assert np.allclose(y + y[::-1], 6000.0)  # point symmetry


@pytest.mark.parametrize("kind", ["linear", "exponential", "power", "s_curve"])
@pytest.mark.parametrize("up", [True, False])
def test_profile_monotonic(kind, up):
    t = np.linspace(0, 6, 601)
    lo, hi = 700.0, 5500.0
    start, stop = (lo, hi) if up else (hi, lo)
    y = rpm_profiles(t, kind, start=start, stop=stop, t_start=0.5, t_stop=5.5)
    d = np.diff(y)
    assert np.all(d >= 0) if up else np.all(d <= 0)
    assert y.min() >= lo - 1e-9 and y.max() <= hi + 1e-9


def test_profile_piecewise_holds_outside_knots():
    t = np.linspace(0, 20, 2001)
    knots = ([2.0, 8.0, 14.0], [40.0, 100.0, 40.0])  # km/h schedule
    y = rpm_profiles(t, "piecewise", knots=knots)
    assert np.all(y[t <= 2] == 40) and np.all(y[t >= 14] == 40)
    assert y[np.searchsorted(t, 8.0)] == pytest.approx(100.0)
    assert np.allclose(y, np.interp(t, *knots))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kind="bogus", start=1, stop=2),
        dict(kind="linear", start=1),
        dict(kind="linear", stop=1),
        dict(kind="constant"),
        dict(kind="exponential", start=0, stop=10),
        dict(kind="exponential", start=10, stop=-1),
        dict(kind="linear", start=1, stop=2, t_start=5, t_stop=5),
        dict(kind="power", start=1, stop=2, exponent=0),
        dict(kind="piecewise"),
        dict(kind="piecewise", knots=([0, 1], [1])),
        dict(kind="piecewise", knots=([1, 1], [1, 2])),
        dict(kind="linear", start=1, stop=2, knots=([0, 1], [1, 2])),
    ],
)
def test_profile_validation(kwargs):
    with pytest.raises(ValueError):
        rpm_profiles(np.linspace(0, 1, 11), **kwargs)


def test_profile_rejects_bad_time_axis():
    with pytest.raises(ValueError):
        rpm_profiles(np.zeros((3, 2)), "constant", start=1)
    with pytest.raises(ValueError):
        rpm_profiles([], "constant", start=1)


# --------------------------------------------------------------------------
# synth_runup: structure, determinism, identities
# --------------------------------------------------------------------------


def _basic(seed=5, **kw):
    fs = 4000.0
    rpm = _ramp_rpm(fs, 3.0, 900.0, 3300.0)
    orders = [
        {"order": 2.0, "amplitude": 1.0},
        {"order": 4.5, "amplitude": 0.4, "am_depth": 0.3, "am_freq": 2.0},
        {"order": 1.0, "amplitude": 0.7, "fm_depth": 0.02, "fm_freq": 5.0},
    ]
    kw.setdefault("tones", [{"freq": 123.0, "amplitude": 0.2}])
    kw.setdefault("noise_rms", 0.05)
    return fs, rpm, synth_runup(fs, rpm, orders, seed=seed, **kw)


def test_returns_synth_signal_with_documented_shapes():
    fs, rpm, s = _basic(tacho_ppr=12)
    n, K = rpm.size, 3
    assert isinstance(s, SynthSignal)
    assert s.fs == fs
    for a in (s.t, s.x, s.rpm, s.rev, s.tacho, s.stems["orders"], s.stems["tones"], s.stems["noise"]):
        assert a.shape == (n,)
    for a in (s.order_stems, s.order_amplitude, s.order_phase, s.order_frequency):
        assert a.shape == (n, K) and a.dtype == np.float64
    assert np.array_equal(s.orders, [2.0, 4.5, 1.0])
    assert set(s.stems) == {"orders", "tones", "noise"}
    assert np.array_equal(s.t, np.arange(n) / fs)
    assert np.array_equal(s.rev, speed.rpm_to_phase(rpm, fs))
    assert s.rev[0] == 0


def test_rpm_input_is_not_aliased():
    rpm = np.full(100, 1000.0)
    s = synth_runup(1000.0, rpm, [{"order": 1.0}], seed=0)
    rpm[:] = 0.0
    assert np.all(s.rpm == 1000.0)


def test_same_seed_is_bit_identical():
    _, _, a = _basic(seed=11, tacho_ppr=6, tacho_jitter=1e-5, tacho_drop=0.05)
    _, _, b = _basic(seed=11, tacho_ppr=6, tacho_jitter=1e-5, tacho_drop=0.05)
    for name in ("x", "order_stems", "order_amplitude", "order_phase", "order_frequency", "tacho", "pulse_times"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.config == b.config
    _, _, c = _basic(seed=12, tacho_ppr=6, tacho_jitter=1e-5, tacho_drop=0.05)
    assert not np.array_equal(a.x, c.x)


def test_generator_seed_and_unseeded_runs():
    fs, rpm, a = _basic(seed=np.random.default_rng(3))
    _, _, b = _basic(seed=np.random.default_rng(3))
    assert np.array_equal(a.x, b.x)
    assert a.config["seed"] is None
    _, _, c = _basic(seed=None)
    _, _, d = _basic(seed=None)
    assert not np.array_equal(c.x, d.x)


def test_random_phases_resolved_and_independent_of_noise_and_tacho():
    _, _, a = _basic(seed=7, noise_rms=0.0)
    _, _, b = _basic(seed=7, noise_rms=1.0, tacho_ppr=10, tacho_jitter=1e-5)
    pa = [o["phase"] for o in a.config["orders"]] + [t["phase"] for t in a.config["tones"]]
    pb = [o["phase"] for o in b.config["orders"]] + [t["phase"] for t in b.config["tones"]]
    assert pa == pb
    assert all(0.0 <= p < TWO_PI for p in pa)
    assert len(set(pa)) == len(pa)
    assert np.array_equal(a.order_stems, b.order_stems)
    assert np.array_equal(a.stems["tones"], b.stems["tones"])


def test_explicit_phase_is_used():
    rpm = np.full(2000, 1200.0)
    s = synth_runup(2000.0, rpm, [{"order": 3.0, "phase": 0.75}], seed=0)
    assert s.config["orders"][0]["phase"] == 0.75
    assert s.order_phase[0, 0] == pytest.approx(0.75)
    assert s.order_stems[0, 0] == pytest.approx(np.cos(0.75))


def test_mix_is_exact_sum_of_stems():
    _, _, s = _basic()
    assert np.array_equal(s.x, s.stems["orders"] + s.stems["tones"] + s.stems["noise"])
    assert np.array_equal(s.stems["orders"], s.order_stems.sum(axis=1))
    assert np.allclose(s.stems["orders"], s.order_stems.sum(axis=1), rtol=0, atol=1e-14)


def test_order_stems_equal_amplitude_times_cos_phase():
    _, _, s = _basic()
    assert np.max(np.abs(s.order_stems - s.order_amplitude * np.cos(s.order_phase))) < 1e-12


def test_no_orders_gives_empty_truth():
    s = synth_runup(1000.0, np.full(500, 600.0), [], tones=[{"freq": 50.0}], seed=1)
    assert s.order_stems.shape == (500, 0) and s.orders.shape == (0,)
    assert np.array_equal(s.x, s.stems["tones"])
    assert np.all(s.stems["orders"] == 0)


# --------------------------------------------------------------------------
# synth_runup: order truth
# --------------------------------------------------------------------------


def test_constant_rpm_truth_phase():
    fs, r = 2000.0, 2400.0
    rpm = np.full(4000, r)
    order, phi0 = 2.5, 1.1
    s = synth_runup(fs, rpm, [{"order": order, "amplitude": 0.8, "phase": phi0}])
    expect = TWO_PI * order * r / 60.0 * s.t + phi0
    assert np.max(np.abs(s.order_phase[:, 0] - expect)) < 1e-9
    assert np.allclose(s.order_frequency[:, 0], order * r / 60.0, rtol=0, atol=1e-12)
    assert np.allclose(s.order_amplitude[:, 0], 0.8)
    assert np.allclose(s.x, 0.8 * np.cos(expect), atol=1e-9)


def test_fractional_and_second_shaft_orders():
    fs = 2000.0
    t = np.arange(4000) / fs
    rpm = np.column_stack([np.full(t.size, 1200.0), rpm_profiles(t, "linear", start=600, stop=1500)])
    s = synth_runup(
        fs, rpm, [{"order": 0.5, "shaft": 0, "phase": 0.0}, {"order": 1.5, "shaft": 1, "phase": 0.0}]
    )
    assert s.rpm.shape == s.rev.shape == (t.size, 2)
    assert np.allclose(s.order_frequency[:, 0], 0.5 * 1200 / 60)
    assert np.allclose(s.order_frequency[:, 1], 1.5 * rpm[:, 1] / 60)
    assert np.allclose(s.order_phase[:, 1], TWO_PI * 1.5 * s.rev[:, 1])


def test_gain_law_follows_shaft_speed_with_hold():
    fs = 2000.0
    rpm = _ramp_rpm(fs, 4.0, 600.0, 3000.0)
    gr, g = [1000.0, 2000.0, 2500.0], [0.2, 1.0, 0.5]
    s = synth_runup(fs, rpm, [{"order": 2.0, "amplitude": 2.0, "gain_rpm": gr, "gain": g}], seed=0)
    expect = 2.0 * np.interp(rpm, gr, g)
    assert np.allclose(s.order_amplitude[:, 0], expect, rtol=0, atol=1e-12)
    assert s.order_amplitude[0, 0] == pytest.approx(0.4)  # held below the first knot
    assert s.order_amplitude[-1, 0] == pytest.approx(1.0)  # held above the last


def test_gain_law_uses_the_orders_own_shaft():
    fs = 1000.0
    rpm = np.column_stack([np.full(1000, 600.0), np.full(1000, 1800.0)])
    s = synth_runup(
        fs, rpm, [{"order": 1.0, "shaft": 1, "gain_rpm": [600.0, 1800.0], "gain": [0.0, 3.0]}], seed=0
    )
    assert np.allclose(s.order_amplitude, 3.0)


def test_min_freq_mask_and_taper():
    fs = 2000.0
    rpm = _ramp_rpm(fs, 4.0, 300.0, 3000.0)  # order 2: 10 -> 100 Hz
    s = synth_runup(fs, rpm, [{"order": 2.0, "min_freq": 40.0}], seed=0)
    f, a = s.order_frequency[:, 0], s.order_amplitude[:, 0]
    assert np.all(a[f < 40.0] == 0) and np.all(a[f >= 40.0] == 1)
    assert np.all(s.order_stems[f < 40.0, 0] == 0)

    s = synth_runup(fs, rpm, [{"order": 2.0, "min_freq": 40.0, "min_freq_taper": 20.0}], seed=0)
    f, a = s.order_frequency[:, 0], s.order_amplitude[:, 0]
    assert np.all(a[f <= 40.0] == 0) and np.all(a[f >= 60.0] == 1)
    assert a[np.searchsorted(f, 50.0)] == pytest.approx(0.5, abs=1e-3)
    ramp = (f > 40.0) & (f < 60.0)
    assert np.all(np.diff(a[ramp]) >= 0)
    assert np.allclose(a[ramp], 0.5 * (1 - np.cos(np.pi * (f[ramp] - 40.0) / 20.0)))


def test_nyquist_components_are_muted_not_clipped():
    fs = 1000.0
    rpm = _ramp_rpm(fs, 4.0, 300.0, 900.0)  # order 40: 200 -> 600 Hz, crosses fs/2
    s = synth_runup(fs, rpm, [{"order": 40.0, "amplitude": 1.5}], seed=2)
    f, a = s.order_frequency[:, 0], s.order_amplitude[:, 0]
    assert f.max() > 500.0
    assert np.all(a[f >= 500.0] == 0)
    assert np.all(a[f < 500.0] == 1.5)
    assert np.all(s.order_stems[f >= 500.0, 0] == 0)
    assert np.all(s.x[f >= 500.0] == 0)
    assert np.any(s.x[f < 500.0] != 0)


def test_nyquist_mutes_tones_and_rejects_nothing_silently():
    s = synth_runup(1000.0, np.full(300, 600.0), [], tones=[{"freq": 500.0}, {"freq": 700.0}], seed=0)
    assert np.all(s.stems["tones"] == 0)


def test_am_truth():
    fs = 2000.0
    rpm = np.full(4000, 1800.0)
    s = synth_runup(
        fs, rpm, [{"order": 2.0, "amplitude": 2.0, "am_depth": 0.5, "am_freq": 3.0}], seed=0
    )
    expect = 2.0 * (1 + 0.5 * np.sin(TWO_PI * 3.0 * s.t))
    assert np.allclose(s.order_amplitude[:, 0], expect, rtol=0, atol=1e-12)
    assert s.order_amplitude.min() >= 0


def test_fm_truth_frequency_is_phase_derivative():
    fs = 8000.0
    rpm = _ramp_rpm(fs, 4.0, 1000.0, 2000.0)
    d, fm_f, fm_ph, order = 0.05, 7.0, 0.3, 3.0
    s = synth_runup(
        fs, rpm, [{"order": order, "fm_depth": d, "fm_freq": fm_f, "fm_phase": fm_ph}], seed=0
    )
    f = s.order_frequency[:, 0]
    expect = order * rpm / 60.0 * (1 + d * np.sin(TWO_PI * fm_f * s.t + fm_ph))
    assert np.allclose(f, expect, rtol=0, atol=1e-12)
    # Trapezoidal integration makes the sample-to-sample phase step exact.
    step = np.diff(s.order_phase[:, 0]) * fs / TWO_PI
    assert np.allclose(step, 0.5 * (f[1:] + f[:-1]), rtol=0, atol=1e-8)
    # Without modulation the same relation holds for the plain order.
    s0 = synth_runup(fs, rpm, [{"order": order}], seed=0)
    assert np.allclose(np.diff(s0.order_phase[:, 0]) * fs / TWO_PI, 0.5 * (
        s0.order_frequency[1:, 0] + s0.order_frequency[:-1, 0]), atol=1e-8)


def test_fm_phase_matches_closed_form_at_constant_speed():
    fs, r, order, d, fm_f, fm_ph = 8000.0, 1200.0, 2.0, 0.05, 7.0, 0.3
    rpm = np.full(16000, r)
    s = synth_runup(
        fs, rpm, [{"order": order, "phase": 0.0, "fm_depth": d, "fm_freq": fm_f, "fm_phase": fm_ph}]
    )
    f0 = order * r / 60.0
    closed = TWO_PI * f0 * s.t - (f0 * d / fm_f) * (np.cos(TWO_PI * fm_f * s.t + fm_ph) - np.cos(fm_ph))
    assert np.max(np.abs(s.order_phase[:, 0] - closed)) < 1e-5
    assert np.max(np.abs(s.order_frequency[:, 0] - f0)) == pytest.approx(f0 * d, rel=1e-3)


def test_spec_validation_errors():
    rpm = np.full(100, 1000.0)
    bad_orders = [
        [{"order": 1.0, "ampl": 1.0}],
        [{"amplitude": 1.0}],
        [{"order": -1.0}],
        [{"order": 1.0, "shaft": 1}],
        [{"order": 1.0, "gain_rpm": [1.0, 2.0]}],
        [{"order": 1.0, "gain_rpm": [2.0, 1.0], "gain": [1.0, 1.0]}],
        [{"order": 1.0, "am_depth": 0.2}],
        [{"order": 1.0, "fm_depth": 0.2}],
        [{"order": 1.0, "am_depth": 1.5, "am_freq": 1.0}],
        [{"order": 1.0, "min_freq_taper": 5.0}],
        ["not a dict"],
    ]
    for spec in bad_orders:
        with pytest.raises((ValueError, TypeError)):
            synth_runup(1000.0, rpm, spec)
    with pytest.raises(ValueError, match="unknown key"):
        synth_runup(1000.0, rpm, [{"order": 1.0, "typo": 1}])
    with pytest.raises(ValueError, match="unknown key"):
        synth_runup(1000.0, rpm, [], tones=[{"freq": 10.0, "gain": 1}])
    with pytest.raises(ValueError, match="unknown key"):
        synth_runup(1000.0, rpm, [], resonances=[{"freq": 10.0, "q": 5, "gain": 2, "bw": 1}])
    with pytest.raises(ValueError):
        synth_runup(1000.0, rpm, [], resonances=[{"freq": 10.0, "q": 5}])
    with pytest.raises(ValueError):
        synth_runup(1000.0, rpm, [], resonances=[{"freq": 600.0, "q": 5, "gain": 2}])
    with pytest.raises(ValueError):
        synth_runup(0.0, rpm, [])
    with pytest.raises(ValueError):
        synth_runup(1000.0, -rpm, [])
    with pytest.raises(ValueError):
        synth_runup(1000.0, rpm[:1], [])
    with pytest.raises(ValueError):
        synth_runup(1000.0, rpm, [], noise_envelope=np.ones(5), noise_rms=1.0)


# --------------------------------------------------------------------------
# synth_runup: resonances
# --------------------------------------------------------------------------


def _res_case(resonances, **kw):
    fs = 4000.0
    rpm = _ramp_rpm(fs, 20.0, 600.0, 2400.0)  # order 10: 100 -> 400 Hz
    s = synth_runup(fs, rpm, [{"order": 10.0, "phase": 0.4, "amplitude": 1.0}], resonances=resonances, **kw)
    return fs, rpm, s


def test_resonance_peak_gain_equals_gain_at_f0():
    res = [{"freq": 250.0, "q": 10.0, "gain": 3.0}]
    fs, rpm, s = _res_case(res)
    a, f = s.order_amplitude[:, 0], s.order_frequency[:, 0]
    assert a.max() == pytest.approx(3.0, rel=1e-6)
    assert abs(f[np.argmax(a)] - 250.0) < 0.5
    assert a[0] == pytest.approx(1.0, abs=0.05)  # far below: unity
    assert a[-1] == pytest.approx(1.0, abs=0.05)  # far above: unity
    i = np.argmin(np.abs(f - 250.0))
    assert a[i] == pytest.approx(3.0, rel=1e-6)
    # Peak gain below one is a notch.
    _, _, n = _res_case([{"freq": 250.0, "q": 10.0, "gain": 0.25}])
    assert n.order_amplitude[:, 0].min() == pytest.approx(0.25, rel=1e-6)


def test_resonance_matches_independent_transfer_function():
    res = [{"freq": 250.0, "q": 6.0, "gain": 2.5}, {"freq": 320.0, "q": 12.0, "gain": 0.5}]
    fs, rpm, s = _res_case(res)
    b, a = np.array([1.0]), np.array([1.0])
    for r in res:
        w0 = TWO_PI * r["freq"]
        b = np.convolve(b, [1.0, r["gain"] * w0 / r["q"], w0**2])
        a = np.convolve(a, [1.0, w0 / r["q"], w0**2])
    f = s.order_frequency[:, 0]
    _, h = signal.freqs(b, a, worN=TWO_PI * f)
    assert np.allclose(s.order_amplitude[:, 0], np.abs(h), rtol=1e-9, atol=1e-12)
    carrier = TWO_PI * 10.0 * s.rev + 0.4
    dphi = s.order_phase[:, 0] - carrier
    assert np.max(np.abs(np.exp(1j * dphi) - np.exp(1j * np.angle(h)))) < 1e-9
    assert np.all(np.abs(np.diff(dphi)) < 0.5)  # continuous, no 2 pi jumps
    assert np.max(np.abs(s.order_stems[:, 0] - s.order_amplitude[:, 0] * np.cos(s.order_phase[:, 0]))) < 1e-12


def test_resonance_phase_is_zero_at_f0_for_peak_and_applies_to_tones():
    res = [{"freq": 300.0, "q": 8.0, "gain": 4.0}]
    fs = 4000.0
    t = np.arange(4000) / fs
    s = synth_runup(
        fs,
        np.full(t.size, 600.0),
        [],
        tones=[{"freq": 300.0, "amplitude": 0.5, "phase": 0.2}, {"freq": 1000.0, "amplitude": 0.5, "phase": 0.0}],
        resonances=res,
    )
    w0 = TWO_PI * 300.0
    _, h = signal.freqs([1.0, 4.0 * w0 / 8.0, w0**2], [1.0, w0 / 8.0, w0**2], worN=TWO_PI * np.array([300.0, 1000.0]))
    expect = 0.5 * abs(h[0]) * np.cos(TWO_PI * 300.0 * t + 0.2 + np.angle(h[0]))
    expect += 0.5 * abs(h[1]) * np.cos(TWO_PI * 1000.0 * t + np.angle(h[1]))
    assert abs(h[0]) == pytest.approx(4.0)
    assert np.angle(h[0]) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(s.stems["tones"], expect, atol=1e-12)


def test_digital_resonance_has_gain_at_f0():
    fs = 8000.0
    res = [{"freq": 1000.0, "q": 8.0, "gain": 4.0}, {"freq": 3000.0, "q": 3.0, "gain": 0.3}]
    sos = synth._resonance_sos(res, fs)
    assert sos.shape == (2, 6)
    for r, row in zip(res, sos):
        _, h = signal.sosfreqz(row[None, :], worN=[TWO_PI * r["freq"] / fs])
        assert abs(h[0]) == pytest.approx(r["gain"], rel=1e-9)


def test_resonance_on_noise_filters_noise():
    fs, n = 8000.0, 2**17
    rpm = np.full(n, 600.0)
    res = [{"freq": 1000.0, "q": 8.0, "gain": 4.0}]
    flat = synth_runup(fs, rpm, [], noise_rms=1.0, resonances=res, resonance_on_noise=False, seed=4)
    filt = synth_runup(fs, rpm, [], noise_rms=1.0, resonances=res, resonance_on_noise=True, seed=4)
    assert np.sqrt(np.mean(flat.stems["noise"] ** 2)) == pytest.approx(1.0, rel=1e-12)
    f, p0 = signal.welch(flat.stems["noise"], fs, nperseg=2048)
    _, p1 = signal.welch(filt.stems["noise"], fs, nperseg=2048)
    _, h = signal.sosfreqz(synth._resonance_sos(res, fs), worN=TWO_PI * f / fs)
    band = (f > 900) & (f < 1100)
    ratio = np.median(p1[band] / p0[band] / np.abs(h[band]) ** 2)
    assert ratio == pytest.approx(1.0, rel=0.1)
    assert (p1[band] / p0[band]).max() > 10.0  # clear resonant boost near 16 at f0
    far = f > 2500
    assert np.median(p1[far] / p0[far]) == pytest.approx(1.0, rel=0.3)


# --------------------------------------------------------------------------
# synth_runup: noise
# --------------------------------------------------------------------------


def test_noise_rms_is_exact():
    fs, n = 8000.0, 40000
    rpm = np.full(n, 900.0)
    for rms in (0.01, 1.0, 37.5):
        s = synth_runup(fs, rpm, [], noise_rms=rms, seed=9)
        assert np.sqrt(np.mean(s.stems["noise"] ** 2)) == pytest.approx(rms, rel=1e-12)
        assert np.array_equal(s.x, s.stems["noise"])
    s = synth_runup(fs, rpm, [], noise_rms=0.0, seed=9)
    assert np.all(s.stems["noise"] == 0)
    s = synth_runup(fs, rpm, [], noise_rms=2.0, noise_band=(500.0, 1500.0), seed=9)
    assert np.sqrt(np.mean(s.stems["noise"] ** 2)) == pytest.approx(2.0, rel=1e-12)


def test_noise_band_limits_spectrum():
    fs, n = 16000.0, 2**16
    rpm = np.full(n, 900.0)
    s = synth_runup(fs, rpm, [], noise_rms=1.0, noise_band=(1000.0, 2000.0), seed=1)
    f, p = signal.welch(s.stems["noise"], fs, nperseg=2048)
    inband = p[(f > 1200) & (f < 1800)].mean()
    assert p[f < 400].mean() < inband * 1e-4  # 40 dB down
    assert p[f > 4000].mean() < inband * 1e-4
    lo = synth_runup(fs, rpm, [], noise_rms=1.0, noise_band=(None, 1000.0), seed=1)
    _, pl = signal.welch(lo.stems["noise"], fs, nperseg=2048)
    assert pl[f > 3000].mean() < pl[f < 600].mean() * 1e-4
    hi = synth_runup(fs, rpm, [], noise_rms=1.0, noise_band=(4000.0, None), seed=1)
    _, ph = signal.welch(hi.stems["noise"], fs, nperseg=2048)
    assert ph[f < 1500].mean() < ph[(f > 5000) & (f < 7000)].mean() * 1e-4
    for bad in ((2000.0, 1000.0), (0.0, None), (100.0, 9000.0)):
        with pytest.raises(ValueError):
            synth_runup(fs, rpm, [], noise_rms=1.0, noise_band=bad)


def test_noise_envelope_multiplies_after_scaling():
    fs, n = 4000.0, 20000
    rpm = np.full(n, 900.0)
    env = np.linspace(0.0, 2.0, n)
    flat = synth_runup(fs, rpm, [], noise_rms=0.5, seed=6)
    shaped = synth_runup(fs, rpm, [], noise_rms=0.5, noise_envelope=env, seed=6)
    assert np.allclose(shaped.stems["noise"], flat.stems["noise"] * env, rtol=0, atol=1e-15)
    assert shaped.config["noise_envelope"] is True


# --------------------------------------------------------------------------
# synth_runup: tachometer
# --------------------------------------------------------------------------


def _tacho_case(**kw):
    fs = 48000.0
    t = np.arange(int(fs * 6)) / fs
    rpm = rpm_profiles(t, "s_curve", start=700, stop=4500, t_start=0.3, t_stop=5.5)
    kw.setdefault("tacho_ppr", 60)
    s = synth_runup(fs, rpm, [{"order": 2.0}], seed=21, **kw)
    return fs, rpm, s


def test_tacho_round_trip_with_jitter_and_drops():
    fs, rpm, s = _tacho_case(tacho_jitter=5e-6, tacho_drop=0.002)
    assert s.config["tacho_n_dropped"] >= 1
    est, info = speed.tacho_to_rpm(s.tacho, fs, 60, return_info=True)
    n = rpm.size
    inner = slice(int(0.1 * n), int(0.9 * n))
    rel = np.abs(est - rpm) / rpm
    assert np.median(rel[inner]) < 1e-3
    assert info["n_missing"] == s.config["tacho_n_dropped"]
    assert info["n_spurious"] == 0


def test_tacho_pulse_times_recovered_exactly_without_jitter():
    fs, rpm, s = _tacho_case()
    _, info = speed.tacho_to_rpm(s.tacho, fs, 60, return_info=True)
    pt = info["pulse_times"]
    assert pt.size == s.pulse_times.size
    assert np.max(np.abs(pt - s.pulse_times)) < 1e-9  # sub-nanosecond, far below 1/fs
    # The first detected pulse is labelled rev 0; compare angles relative to it.
    assert np.max(np.abs(np.diff(info["pulse_revs"]) - np.diff(s.pulse_revs))) < 1e-12


def test_tacho_pulse_angles_follow_the_truth_phase():
    fs, rpm, s = _tacho_case()
    assert s.pulse_revs[0] == pytest.approx(1 / 60)
    assert np.allclose(np.diff(s.pulse_revs), 1 / 60)
    # Evaluate the exact shaft angle (quadratic arc between samples) at the pulse times.
    i = np.floor(s.pulse_times * fs).astype(int)
    tau = s.pulse_times - i / fs
    a = (rpm[i + 1] - rpm[i]) * fs
    rev_at = s.rev[i] + (rpm[i] * tau + 0.5 * a * tau**2) / 60.0
    assert np.max(np.abs(rev_at - s.pulse_revs)) < 1e-9


def test_tacho_pulse_times_match_analytic_ramp_from_standstill():
    fs = 8000.0
    t = np.arange(int(fs * 3)) / fs
    k = 3000.0 / t[-1]  # rpm per s; rpm = k t, so rev = k t^2 / 120
    s = synth_runup(fs, rpm_profiles(t, "linear", start=0.0, stop=3000.0), [], tacho_ppr=4, seed=0)
    assert s.pulse_times.size > 100
    assert np.max(np.abs(s.pulse_times - np.sqrt(120.0 * s.pulse_revs / k))) < 1e-12


def test_tacho_levels_not_in_mix_and_stream_independent():
    fs, rpm, s = _tacho_case(tacho_high=3.3)
    assert s.tacho.max() == pytest.approx(3.3) and s.tacho.min() == 0.0
    no_tacho = synth_runup(fs, rpm, [{"order": 2.0}], seed=21)
    assert np.array_equal(s.x, no_tacho.x)
    assert no_tacho.tacho is None
    assert no_tacho.pulse_times.size == 0 and no_tacho.pulse_revs.size == 0
    # Roughly half of each interval is high.
    duty = np.mean(s.tacho > 1.65)
    assert duty == pytest.approx(0.5, abs=0.01)


def test_tacho_drop_removes_pulses():
    _, _, full = _tacho_case()
    _, _, dropped = _tacho_case(tacho_drop=0.1)
    assert dropped.pulse_times.size == full.pulse_times.size - dropped.config["tacho_n_dropped"]
    assert dropped.config["tacho_n_dropped"] == pytest.approx(0.1 * full.pulse_times.size, rel=0.2)
    assert set(np.round(dropped.pulse_revs * 60)) <= set(np.round(full.pulse_revs * 60))


def test_tacho_on_second_shaft():
    fs = 24000.0
    t = np.arange(int(fs * 3)) / fs
    rpm = np.column_stack([np.full(t.size, 1500.0), rpm_profiles(t, "linear", start=600, stop=2400)])
    s = synth_runup(fs, rpm, [{"order": 1.0}], tacho_ppr=24, tacho_shaft=1, seed=2)
    est = speed.tacho_to_rpm(s.tacho, fs, 24)
    inner = slice(int(0.1 * t.size), int(0.9 * t.size))
    assert np.median(np.abs(est - rpm[:, 1])[inner] / rpm[inner, 1]) < 1e-3
    with pytest.raises(ValueError):
        synth_runup(fs, rpm, [], tacho_ppr=24, tacho_shaft=2)


def test_tacho_validation():
    rpm = np.full(2000, 3000.0)
    for kw in (
        dict(tacho_ppr=0),
        dict(tacho_ppr=2.5),
        dict(tacho_ppr=4, tacho_drop=1.0),
        dict(tacho_ppr=4, tacho_jitter=-1e-6),
        dict(tacho_ppr=4, tacho_high=0),
        dict(tacho_ppr=100),  # 5 kHz pulses at fs = 8 kHz: under 2 samples high
    ):
        with pytest.raises(ValueError):
            synth_runup(8000.0, rpm, [], **kw)
    with pytest.raises(ValueError):
        synth_runup(8000.0, np.zeros(2000), [], tacho_ppr=4)


def test_pulse_train_matches_sum_of_ramps():
    fs = 1000.0
    t = np.arange(2000) / fs
    rng = np.random.default_rng(0)
    for spacing in (0.02, 0.0025):  # well separated, and edges closer than a ramp
        rise = np.cumsum(rng.uniform(0.9, 1.1, 60)) * spacing
        fall = rise + 0.4 * spacing

        def ramp(u):
            return np.clip(0.5 + u * fs / 2.0, 0.0, 1.0)

        brute = np.zeros(t.size)
        for r, f in zip(rise, fall):
            brute += ramp(t - r) - ramp(t - f)
        assert np.max(np.abs(synth._pulse_train(t, fs, rise, fall) - brute)) < 1e-12


# --------------------------------------------------------------------------
# config and persistence
# --------------------------------------------------------------------------


def test_config_is_json_serialisable_and_complete():
    _, _, s = _basic(
        seed=42,
        resonances=[{"freq": 300.0, "q": 5.0, "gain": 2.0}],
        noise_band=(100.0, 900.0),
        noise_envelope=np.ones(12000),
        tacho_ppr=8,
        tacho_jitter=1e-5,
    )
    text = json.dumps(s.config)
    cfg = json.loads(text)
    assert cfg == s.config
    assert cfg["rota_version"] == rota.__version__
    assert cfg["seed"] == 42
    assert cfg["fs"] == 4000.0 and cfg["n"] == 12000
    assert cfg["noise_rms"] == 0.05 and cfg["noise_band"] == [100.0, 900.0]
    assert cfg["tacho_ppr"] == 8 and cfg["tacho_jitter"] == 1e-5
    assert [o["order"] for o in cfg["orders"]] == [2.0, 4.5, 1.0]
    assert cfg["orders"][1]["am_depth"] == 0.3
    assert cfg["resonances"] == [{"freq": 300.0, "q": 5.0, "gain": 2.0}]
    assert cfg["tones"][0]["freq"] == 123.0 and isinstance(cfg["tones"][0]["phase"], float)
    # Gain-law arrays and numpy seeds serialise too.
    fs = 1000.0
    s2 = synth_runup(
        fs, np.full(500, 900.0),
        [{"order": 1.0, "gain_rpm": np.array([500.0, 1000.0]), "gain": np.array([1.0, 2.0])}],
        seed=np.int64(5),
    )
    cfg2 = json.loads(json.dumps(s2.config))
    assert cfg2["orders"][0]["gain_rpm"] == [500.0, 1000.0] and cfg2["seed"] == 5


def test_config_round_trip_reproduces_the_signal():
    fs, rpm, a = _basic(seed=8, tacho_ppr=6, tacho_jitter=1e-5)
    cfg = json.loads(json.dumps(a.config))
    b = synth_runup(
        fs, rpm, cfg["orders"], tones=cfg["tones"], resonances=cfg["resonances"],
        noise_rms=cfg["noise_rms"], tacho_ppr=cfg["tacho_ppr"], tacho_jitter=cfg["tacho_jitter"],
        seed=cfg["seed"],
    )
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.tacho, b.tacho)


def test_save_writes_npz_and_json_sidecar(tmp_path):
    _, _, s = _basic(seed=3, tacho_ppr=10)
    npz, sidecar = s.save(tmp_path / "run1")
    assert npz == tmp_path / "run1.npz" and sidecar == tmp_path / "run1.json"
    assert npz.is_file() and sidecar.is_file()
    assert json.loads(sidecar.read_text(encoding="utf-8")) == s.config
    with np.load(npz) as d:
        for key in ("t", "x", "rpm", "rev", "orders", "order_stems", "order_amplitude",
                    "order_phase", "order_frequency", "tacho", "pulse_times", "pulse_revs"):
            assert np.array_equal(d[key], getattr(s, key)), key
        assert float(d["fs"]) == s.fs
        assert np.array_equal(d["stem_orders"], s.stems["orders"])
        assert np.array_equal(d["stem_tones"], s.stems["tones"])
        assert np.array_equal(d["stem_noise"], s.stems["noise"])
        assert json.loads(str(d["config_json"])) == s.config


def test_save_accepts_suffixes_and_omits_missing_tacho(tmp_path):
    _, _, s = _basic(seed=3)
    npz, sidecar = s.save(str(tmp_path / "run2.npz"))
    assert npz.name == "run2.npz" and sidecar.name == "run2.json"
    npz, sidecar = s.save(tmp_path / "run3.json")
    assert npz.name == "run3.npz" and sidecar.name == "run3.json"
    with np.load(npz) as d:
        assert "tacho" not in d.files
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "run2.json", "run2.npz", "run3.json", "run3.npz",
    ]
