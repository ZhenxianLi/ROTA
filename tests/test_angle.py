"""Tests of rota.angle against analytic truth (rota.synth supplies the signals)."""

import numpy as np
import pytest

from rota.angle import AngularSignal, angular_resample
from rota.speed import rpm_to_phase
from rota.synth import rpm_profiles, synth_runup

FS = 16000.0
R0, ACC = 1000.0, 1000.0  # rpm and rpm/s: 1000 -> 4000 rpm in 3 s


def _runup(dur=3.0, fs=FS):
    """Time axis and a linear 1000 -> 4000 rpm run-up, rpm = R0 + ACC * t."""
    t = np.arange(round(dur * fs)) / fs
    return t, R0 + ACC * t


def _line(a, order, first_rev, n_rev):
    """Complex peak amplitude at `order` over n_rev whole revolutions of the angle signal."""
    spr = a.samples_per_rev
    sl = slice(first_rev * spr, (first_rev + n_rev) * spr)
    return 2.0 / (n_rev * spr) * np.sum(a.x[sl] * np.exp(-2j * np.pi * order * a.rev[sl]))


def _blocks(a, n_rev=8, trim=3):
    """Whole-revolution blocks of the angle signal, skipping filter edge revolutions."""
    spr = a.samples_per_rev
    total = a.x.shape[0] // spr
    for first in range(trim, total - trim - n_rev + 1, n_rev):
        yield first, a.x[first * spr : (first + n_rev) * spr]


def _db(x):
    return 20.0 * np.log10(x)


def _synth(rpm, orders, **kw):
    return synth_runup(FS, rpm, orders, seed=1, **kw)


# --------------------------------------------------------------------------
# constant orders are pure tones in angle
# --------------------------------------------------------------------------

_ORDERS = [
    {"order": 2, "amplitude": 1.0, "phase": 0.3},
    {"order": 5.5, "amplitude": 0.5, "phase": -1.0},
    {"order": 10, "amplitude": 0.25, "phase": 2.0},
]


@pytest.mark.parametrize("direction", ["run-up", "run-down"])
def test_constant_orders_are_single_lines_with_exact_amplitude_and_phase(direction):
    _t, rpm = _runup()
    if direction == "run-down":
        rpm = 5000.0 - rpm  # 4000 -> 1000 rpm
    sig = _synth(rpm, _ORDERS)
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    assert isinstance(a, AngularSignal)
    R = 16  # whole revolutions: orders 2, 5.5, 10 all fall on DFT bins
    for first in (4, 40, 100):
        Y = np.fft.rfft(a.x[first * a.samples_per_rev : (first + R) * a.samples_per_rev])
        Y = 2.0 * Y / (R * a.samples_per_rev)
        orders_axis = np.arange(Y.size) / R
        is_line = np.zeros(Y.size, bool)
        for k, o in enumerate(sig.orders):
            c = _line(a, o, first, R)
            amp, phase = sig.order_amplitude[0, k], sig.order_phase[0, k]
            assert abs(_db(abs(c) / amp)) < 0.01  # dB
            assert abs(np.angle(c * np.exp(-1j * phase))) < 1e-3  # rad
            is_line |= np.abs(orders_axis - o) < 0.01
        assert np.max(np.abs(Y[~is_line])) < 1e-5  # everything else is below -100 dB re 1 V


def test_angle_domain_samples_follow_the_analytic_signal():
    _t, rpm = _runup()
    sig = _synth(rpm, _ORDERS)
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    truth = sum(
        sig.order_amplitude[0, k] * np.cos(2 * np.pi * o * a.rev + sig.order_phase[0, k])
        for k, o in enumerate(sig.orders)
    )
    spr = a.samples_per_rev
    err = np.abs(a.x - truth)
    assert err[2 * spr : -3 * spr].max() < 1e-4


def test_amplitude_law_of_rpm_is_followed_in_angle():
    _t, rpm = _runup()
    order = {
        "order": 4,
        "amplitude": 1.0,
        "phase": 0.2,
        "gain_rpm": [1000, 4000],
        "gain": [1.0, 3.0],
    }
    sig = _synth(rpm, [order])
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    envelope = np.interp(a.t, sig.t, sig.order_amplitude[:, 0])
    truth = envelope * np.cos(2 * np.pi * 4 * a.rev + 0.2)
    spr = a.samples_per_rev
    assert np.max(np.abs(a.x - truth)[3 * spr : -3 * spr]) < 1e-3
    assert envelope.max() > 2.9  # the law really spans a factor of three


def test_non_monotonic_speed_curve_with_a_drop_still_gives_constant_orders():
    t = np.arange(int(4 * FS)) / FS
    rpm = rpm_profiles(t, "piecewise", knots=([0, 1, 1.5, 2.5, 4], [1500, 3500, 2200, 2200, 4000]))
    orders = [
        {"order": 3, "amplitude": 1.0, "phase": 0.3},
        {"order": 7.5, "amplitude": 0.5, "phase": 1.1},
    ]
    sig = _synth(rpm, orders)
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    truth = sum(
        o["amplitude"] * np.cos(2 * np.pi * o["order"] * a.rev + o["phase"]) for o in orders
    )
    spr = a.samples_per_rev
    assert np.max(np.abs(a.x - truth)[3 * spr : -3 * spr]) < 1e-4
    assert np.all(np.diff(a.t) > 0)


def test_amplitude_is_preserved_up_to_the_pass_band_edge():
    _t, rpm = _runup()
    sig = _synth(
        rpm,
        [
            {"order": 18, "amplitude": 1.0, "phase": 0.4},
            {"order": 20, "amplitude": 1.0, "phase": -0.4},
        ],
    )
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    for first in (5, 50, 100):
        for o, phi in ((18, 0.4), (20, -0.4)):
            c = _line(a, o, first, 8)
            assert abs(_db(abs(c))) < 0.01
            assert abs(np.angle(c * np.exp(-1j * phi))) < 1e-3


@pytest.mark.parametrize("order", [40, 80, 115, 140])
def test_default_pass_band_amplitude_is_flat_within_0p2_db_at_constant_speed(order):
    # ROADMAP 5.4: stationary order amplitude within 0.2 dB of the analytic value.
    # 3000 rpm = 50 Hz: default max_order 144 (7.2 kHz), 360 samples/rev; order 115 is 5.75 kHz
    rpm = np.full(int(3 * FS), 3000.0)
    sig = _synth(rpm, [{"order": order, "amplitude": 1.0, "phase": 0.3}])
    a = angular_resample(sig.x, FS, rpm)
    assert a.max_order == pytest.approx(144.0) and a.samples_per_rev == 360
    c = _line(a, order, 10, 8)
    assert abs(_db(abs(c))) < 0.2
    assert abs(np.angle(c * np.exp(-0.3j))) < 1e-3


@pytest.mark.parametrize("order", [54, 80, 100])
def test_default_pass_band_amplitude_is_flat_within_0p2_db_at_the_top_of_a_run_up(order):
    _t, rpm = _runup()  # 1000 -> 4000 rpm: default max_order 108 at the top speed
    sig = _synth(rpm, [{"order": order, "amplitude": 1.0, "phase": 0.3}])
    a = angular_resample(sig.x, FS, rpm)
    c = _line(a, order, 118, 4)  # about 3900 rpm
    assert abs(_db(abs(c))) < 0.2
    assert abs(np.angle(c * np.exp(-0.3j))) < 1e-3


def test_interior_revolutions_are_free_of_filter_edge_effects():
    _t, rpm = _runup()
    sig = _synth(rpm, [{"order": 4, "amplitude": 1.0, "phase": 0.0}])
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    truth = np.cos(2 * np.pi * 4 * a.rev)
    err = np.abs(a.x - truth)
    spr = a.samples_per_rev
    revs = err[: (err.size // spr) * spr].reshape(-1, spr).max(axis=1)
    assert np.all(revs[2:-2] < 1e-4)


# --------------------------------------------------------------------------
# anti-aliasing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alias_order", [26, 30, 40])
def test_orders_above_the_angle_nyquist_are_rejected_at_every_speed(alias_order):
    # max_order 20 gives 50 samples/rev: Nyquist 25. An unfiltered resampler would fold
    # order 30 down to order 20, 26 to 24 and 40 to 10.
    _t, rpm = _runup()
    sig = _synth(
        rpm,
        [
            {"order": 5, "amplitude": 1.0, "phase": 0.2},
            {"order": alias_order, "amplitude": 1.0, "phase": 0.7},
        ],
    )
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    assert a.samples_per_rev == 50
    folded = 50 - alias_order
    n_blocks = 0
    for first, seg in _blocks(a, n_rev=8):
        Y = 2.0 * np.abs(np.fft.rfft(seg)) / seg.size
        orders_axis = np.arange(Y.size) / 8.0
        ref = Y[np.argmin(np.abs(orders_axis - 5))]
        assert abs(_db(ref)) < 0.01  # the pass-band order is intact ...
        other = Y.copy()
        other[np.abs(orders_axis - 5) < 0.01] = 0.0
        assert other.max() < 1e-3  # ... and nothing else exceeds -60 dB re the unit input
        assert Y[np.argmin(np.abs(orders_axis - folded))] < 1e-3  # the alias bin in particular
        n_blocks += 1
    assert n_blocks >= 10  # the whole 1000 -> 4000 rpm range is covered
    # the rejected component was really there in the input
    assert np.max(np.abs(sig.order_stems[:, 1])) == pytest.approx(1.0)


@pytest.mark.parametrize("freq", [2000.0, 3000.0])
def test_fixed_frequency_tone_does_not_alias_into_low_orders(freq):
    # the tone sweeps from order 180 (1000 rpm) to order 45 (4000 rpm) at 3 kHz,
    # always above the angle Nyquist order of 25
    _t, rpm = _runup()
    sig = _synth(
        rpm,
        [{"order": 3, "amplitude": 0.0}],
        tones=[{"freq": freq, "amplitude": 1.0, "phase": 0.0}],
    )
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    spr = a.samples_per_rev
    assert np.max(np.abs(a.x[3 * spr : -3 * spr])) < 1e-3  # > 60 dB


@pytest.mark.parametrize("freq", [5000.0, 7000.0])
def test_fixed_frequency_tone_near_nyquist_is_rejected_by_60_db(freq):
    _t, rpm = _runup()
    sig = _synth(
        rpm,
        [{"order": 3, "amplitude": 0.0}],
        tones=[{"freq": freq, "amplitude": 1.0, "phase": 0.0}],
    )
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    spr = a.samples_per_rev
    assert np.max(np.abs(a.x[3 * spr : -3 * spr])) < 1e-3


def test_slow_fixed_tone_below_max_order_is_preserved():
    # 100 Hz is order 6 at 1000 rpm and order 1.5 at 4000 rpm: inside the pass band throughout
    _t, rpm = _runup()
    sig = _synth(
        rpm,
        [{"order": 3, "amplitude": 0.0}],
        tones=[{"freq": 100.0, "amplitude": 1.0, "phase": 0.0}],
    )
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    mid = a.x[3 * a.samples_per_rev : -3 * a.samples_per_rev]
    assert np.max(np.abs(mid)) == pytest.approx(1.0, abs=1e-3)
    assert np.sqrt(np.mean(mid**2)) == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-3)


def test_attenuation_parameter_sets_the_stop_band_depth():
    _t, rpm = _runup()
    sig = _synth(rpm, [{"order": 26, "amplitude": 1.0, "phase": 0.0}])
    peak = {}
    for att in (40.0, 80.0):
        a = angular_resample(sig.x, FS, rpm, max_order=20, attenuation=att)
        spr = a.samples_per_rev
        peak[att] = np.max(np.abs(a.x[3 * spr : -3 * spr]))
    assert peak[40.0] < 0.02  # about -40 dB
    assert peak[80.0] < 1e-3
    assert peak[40.0] > 10.0 * peak[80.0]


# --------------------------------------------------------------------------
# t, rpm and rev outputs
# --------------------------------------------------------------------------


def test_time_rpm_and_angle_outputs_are_consistent_with_the_shaft_angle():
    _t, rpm = _runup()
    sig = _synth(rpm, [{"order": 4, "amplitude": 1.0, "phase": 0.0}])
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    spr = a.samples_per_rev
    m = a.x.shape[0]
    assert a.rev.shape == a.t.shape == a.rpm.shape == (m,)
    assert np.allclose(a.rev, np.arange(m) / spr, rtol=0, atol=1e-12)
    # inverse of rev(t) = (R0 t + ACC t^2 / 2) / 60 in closed form
    t_exact = (-R0 + np.sqrt(R0**2 + 120.0 * ACC * a.rev)) / ACC
    assert np.max(np.abs(a.t - t_exact)) < 1e-8
    assert np.max(np.abs(a.rpm - (R0 + ACC * a.t))) < 1e-4
    assert a.t[0] == 0.0 and np.all(np.diff(a.t) > 0)
    # the last angle sample is within one sample of the total angle of the record
    assert sig.rev[-1] - 2.0 / spr <= a.rev[-1] <= sig.rev[-1]
    assert a.samples_per_rev == 50 and isinstance(a.samples_per_rev, int)
    assert a.max_order == 20.0 and isinstance(a.max_order, float)


def test_sampling_in_angle_is_uniform_in_angle_not_in_time():
    _t, rpm = _runup()
    sig = _synth(rpm, [{"order": 4, "amplitude": 1.0, "phase": 0.0}])
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    dt = np.diff(a.t)
    # one revolution is 60/rpm seconds: 50 samples/rev -> dt shrinks as the speed rises
    assert dt[0] == pytest.approx(60.0 / 1000.0 / 50.0, rel=1e-3)
    assert dt[-1] == pytest.approx(60.0 / 4000.0 / 50.0, rel=1e-3)
    # sample count: total revolutions times samples per revolution
    assert abs(a.x.shape[0] - sig.rev[-1] * 50) <= 2


def test_default_samples_per_rev_and_max_order_follow_the_documented_rules():
    _t, rpm = _runup()
    x = np.cos(2 * np.pi * 4 * rpm_to_phase(rpm, FS))
    rpm_max = rpm.max()
    a = angular_resample(x, FS, rpm)
    expected_max = 0.45 * FS / (rpm_max / 60.0)
    assert a.max_order == pytest.approx(expected_max)
    assert a.samples_per_rev >= 2.5 * expected_max and a.samples_per_rev % 2 == 0
    assert a.samples_per_rev < 2.5 * expected_max + 2
    # explicit max_order: ceil(2.5 * max_order) rounded up to even
    assert angular_resample(x, FS, rpm, max_order=21).samples_per_rev == 54
    assert angular_resample(x, FS, rpm, max_order=20).samples_per_rev == 50
    # explicit samples_per_rev alone: max_order = 0.4 * samples_per_rev
    b = angular_resample(x, FS, rpm, samples_per_rev=64)
    assert b.samples_per_rev == 64 and b.max_order == pytest.approx(25.6)
    # both given are kept as they are
    c = angular_resample(x, FS, rpm, samples_per_rev=64, max_order=10)
    assert c.samples_per_rev == 64 and c.max_order == 10.0


# --------------------------------------------------------------------------
# multichannel, scalar speed, constant speed
# --------------------------------------------------------------------------


def test_channels_are_resampled_identically():
    _t, rpm = _runup()
    chans = [
        _synth(rpm, [{"order": o, "amplitude": amp, "phase": ph}]).x
        for o, amp, ph in ((2, 1.0, 0.1), (6.5, 2.0, 1.4), (11, 0.5, -2.0))
    ]
    X = np.column_stack(chans)
    a = angular_resample(X, FS, rpm, max_order=20)
    assert a.x.shape == (a.rev.size, 3)
    assert a.rev.shape == a.t.shape == a.rpm.shape
    for j, c in enumerate(chans):
        single = angular_resample(c, FS, rpm, max_order=20)
        assert single.x.shape == (a.rev.size,)
        assert np.allclose(a.x[:, j], single.x, rtol=0, atol=1e-12)
    # linearity: a scaled copy of a channel is the scaled result
    b = angular_resample(np.column_stack([chans[0], 3.0 * chans[0]]), FS, rpm, max_order=20)
    assert np.allclose(b.x[:, 1], 3.0 * b.x[:, 0], atol=1e-12)


def test_resampling_is_linear_and_leaves_its_input_alone():
    _t, rpm = _runup()
    x1 = _synth(rpm, [{"order": 3, "amplitude": 1.0, "phase": 0.1}]).x
    x2 = _synth(rpm, [{"order": 9.5, "amplitude": 2.0, "phase": 1.1}]).x
    a1 = angular_resample(x1, FS, rpm, max_order=20)
    a2 = angular_resample(x2, FS, rpm, max_order=20)
    x_in, r_in = (2.0 * x1 - 0.5 * x2).copy(), rpm.copy()
    x_before, r_before = x_in.copy(), r_in.copy()
    mix = angular_resample(x_in, FS, r_in, max_order=20)
    assert np.allclose(mix.x, 2.0 * a1.x - 0.5 * a2.x, rtol=0, atol=1e-12)
    assert np.array_equal(x_in, x_before) and np.array_equal(r_in, r_before)
    # a DC offset passes the angle-domain filter with unit gain
    dc = angular_resample(x1 + 0.75, FS, rpm, max_order=20)
    spr = dc.samples_per_rev
    assert np.allclose((dc.x - a1.x)[2 * spr : -3 * spr], 0.75, atol=1e-6)


def test_constant_speed_needs_no_filter_and_accepts_a_scalar():
    n = int(3 * FS)
    rpm = np.full(n, 3000.0)
    sig = _synth(rpm, [{"order": 5, "amplitude": 1.0, "phase": 0.4}])
    a = angular_resample(sig.x, FS, rpm)  # defaults: 144 orders, 360 samples/rev, no decimation
    assert a.samples_per_rev == 360 and a.max_order == pytest.approx(144.0)
    c = _line(a, 5, 5, 10)
    assert abs(_db(abs(c))) < 1e-3
    assert abs(np.angle(c * np.exp(-0.4j))) < 1e-6
    # scalar speed is broadcast: 3000 rpm = 50 rev/s for 3 s
    b = angular_resample(sig.x, FS, 3000.0, samples_per_rev=128)
    assert b.rev[-1] == pytest.approx(150.0, abs=2.0 / 128)
    assert np.allclose(b.rpm, 3000.0)


def test_speed_reference_delay_is_compensated():
    # the speed reference lags the signal by 50 ms: ref(t) = true(t - d)
    t, rpm = _runup()
    d = 0.05
    ref = R0 + ACC * (t - d)
    sig = _synth(rpm, [{"order": 5, "amplitude": 1.0, "phase": 0.3}])
    ok = angular_resample(sig.x, FS, ref, max_order=20, delay=d)
    for first in (10, 60):
        c = _line(ok, 5, first, 8)
        assert abs(_db(abs(c))) < 0.01
        assert abs(np.angle(c * np.exp(-0.3j))) < 1e-3
    # uncompensated, the angle drifts and the line smears
    bad = angular_resample(sig.x, FS, ref, max_order=20)
    assert abs(_db(abs(_line(bad, 5, 60, 8)))) > 3.0
    # the compensated output speed is the true speed at the signal time (the last
    # d seconds hold the final reference value)
    inner = ok.t < t[-1] - d - 0.01
    assert np.allclose(ok.rpm[inner], R0 + ACC * ok.t[inner], atol=1e-3)


# --------------------------------------------------------------------------
# run-up from standstill and the min_rpm floor
# --------------------------------------------------------------------------


def test_runup_from_zero_rpm_floors_the_speed_and_keeps_orders_constant():
    t = np.arange(int(3 * FS)) / FS
    rpm = rpm_profiles(t, "linear", start=0.0, stop=4000.0)
    orders = [
        {"order": 4, "amplitude": 1.0, "phase": 0.5},
        {"order": 9, "amplitude": 0.5, "phase": -0.7},
    ]
    sig = _synth(rpm, orders)
    a = angular_resample(sig.x, FS, rpm, max_order=20)  # default floor: 2 % of 4000 = 80 rpm
    assert np.all(np.isfinite(a.x)) and np.all(np.diff(a.t) > 0)
    assert a.rpm.min() == pytest.approx(80.0)  # the output speed is the floored speed
    assert a.rpm[0] == pytest.approx(80.0)
    # the floored interval adds an angle offset delta to the integrated angle; the
    # signal phase is 2 pi o (rev_floored - delta), so each line carries -2 pi o delta
    delta = rpm_to_phase(np.maximum(rpm, 80.0), FS)[-1] - sig.rev[-1]
    assert 0.0 < delta < 0.06
    for o in orders:
        c = _line(a, o["order"], 10, 8)
        assert abs(_db(abs(c) / o["amplitude"])) < 0.01
        expected = o["phase"] - 2 * np.pi * o["order"] * delta
        assert abs(np.angle(c * np.exp(-1j * expected))) < 1e-4


def test_explicit_min_rpm_overrides_the_default_floor():
    t = np.arange(int(3 * FS)) / FS
    rpm = rpm_profiles(t, "linear", start=0.0, stop=4000.0)
    sig = _synth(rpm, [{"order": 4, "amplitude": 1.0, "phase": 0.0}])
    a = angular_resample(sig.x, FS, rpm, max_order=20, min_rpm=400.0)
    assert a.rpm.min() == pytest.approx(400.0)
    # a higher floor adds angle during the start: the total angle is larger
    b = angular_resample(sig.x, FS, rpm, max_order=20)
    assert a.rev[-1] > b.rev[-1]


def test_intermediate_grid_size_guard_raises_memory_error_before_allocating():
    t = np.arange(int(3 * FS)) / FS
    rpm = rpm_profiles(t, "linear", start=0.0, stop=4000.0)
    with pytest.raises(MemoryError, match="min_rpm"):
        angular_resample(np.zeros(t.size), FS, rpm, max_order=20, min_rpm=0.1)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"samples_per_rev": 40, "max_order": 20},  # max_order must be below Nyquist
        {"samples_per_rev": 40, "max_order": 25},
        {"samples_per_rev": 0},
        {"samples_per_rev": -10},
        {"max_order": 0.0},
        {"max_order": -5.0},
    ],
)
def test_invalid_order_or_sampling_arguments_raise(kw):
    t, rpm = _runup(dur=0.5)
    with pytest.raises(ValueError):
        angular_resample(np.zeros(t.size), FS, rpm, **kw)


@pytest.mark.parametrize("min_rpm", [0.0, -5.0])
def test_min_rpm_must_be_positive(min_rpm):
    t, rpm = _runup(dur=0.5)
    with pytest.raises(ValueError, match="min_rpm"):
        angular_resample(np.zeros(t.size), FS, rpm, min_rpm=min_rpm)


def test_invalid_speed_or_signal_arguments_raise():
    n = 8000
    x = np.zeros(n)
    with pytest.raises(ValueError, match="positive"):
        angular_resample(x, FS, np.zeros(n))
    with pytest.raises(ValueError):
        angular_resample(x, FS, np.full(n, -100.0))
    bad = np.full(n, 1000.0)
    bad[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        angular_resample(x, FS, bad)
    with pytest.raises(ValueError):
        angular_resample(x, FS, np.full(n - 1, 1000.0))
    with pytest.raises(ValueError):
        angular_resample(np.zeros((n, 2, 2)), FS, np.full(n, 1000.0))
