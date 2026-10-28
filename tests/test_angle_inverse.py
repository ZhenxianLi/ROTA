"""Tests of angular_to_time, stationarize and destationarize against analytic truth.

rota_order.synth supplies the signals: every order is a cosine of the exact shaft
angle, so the forward transform, its inverse and the time warp can be checked
against closed forms.
"""

import numpy as np
import pytest
from scipy import signal

from rota_order.angle import (
    AngularSignal,
    StationarizedSignal,
    angular_resample,
    angular_to_time,
    destationarize,
    stationarize,
)
from rota_order.speed import rpm_to_phase
from rota_order.synth import rpm_profiles, synth_runup

FS = 16000.0
R0, ACC = 1000.0, 1000.0  # rpm and rpm/s: 1000 -> 4000 rpm in 3 s

_ORDERS = [
    {"order": 2, "amplitude": 1.0, "phase": 0.3},
    {"order": 5.5, "amplitude": 0.5, "phase": -1.0},
    {"order": 10, "amplitude": 0.25, "phase": 2.0},
    {"order": 14.3, "amplitude": 0.3, "phase": 0.5},
]
_TONES = [  # 1000 rpm is the lowest speed: 100 Hz is order 6 and 250 Hz order 15 there
    {"freq": 100.0, "amplitude": 0.5, "phase": 0.2},
    {"freq": 250.0, "amplitude": 0.3, "phase": -0.7},
]


def _t(dur=3.0):
    return np.arange(round(dur * FS)) / FS


def _runup(dur=3.0):
    t = _t(dur)
    return R0 + ACC * t


def _profile(name):
    t = _t(4.0)
    if name == "run-up":
        return R0 + ACC * t[: round(3 * FS)]
    if name == "run-down":
        return 4000.0 - ACC * t[: round(3 * FS)]
    return rpm_profiles(t, "piecewise", knots=([0, 1, 1.5, 2.5, 4], [1500, 3500, 2200, 2200, 4000]))


def _synth(rpm, orders=_ORDERS, tones=_TONES, **kw):
    return synth_runup(FS, rpm, orders, tones=tones, seed=1, **kw)


def _nrmse_db(y, x):
    """Normalised RMS error of y against x, in dB."""
    return 20.0 * np.log10(np.sqrt(np.mean((y - x) ** 2) / np.mean(x**2)))


def _interior(n, head=0.1, tail=0.1):
    """Slice that skips `head` and `tail` seconds (about half a revolution is affected)."""
    return slice(round(head * FS), n - round(tail * FS))


# --------------------------------------------------------------------------
# angular_to_time: round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("use_rpm", [True, False], ids=["angle-from-rpm", "angle-from-a.t/a.rev"])
@pytest.mark.parametrize("profile", ["run-up", "run-down", "drop"])
def test_round_trip_reproduces_band_limited_content_within_minus_60_db(profile, use_rpm):
    rpm = _profile(profile)
    sig = _synth(rpm)
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    y = angular_to_time(a, FS, rpm if use_rpm else None)
    n = min(y.size, sig.x.size)
    assert y.shape == ((rpm.size,) if use_rpm else (n,))
    sl = _interior(n)
    err = _nrmse_db(y[sl], sig.x[sl])
    assert err < -60.0  # the specification
    assert err < -80.0  # what the implementation reaches (about -90 dB)


def test_round_trip_keeps_high_orders_when_the_signal_stays_below_the_nyquist_rate():
    # default max_order is 108 at the top speed (4000 rpm): order 90 is 6 kHz there,
    # still below 0.45 fs, so no anti-aliasing filter is applied and nothing is lost
    rpm = _runup()
    sig = _synth(
        rpm,
        [{"order": 20, "amplitude": 1.0, "phase": 0.3}, {"order": 90, "amplitude": 0.5, "phase": -1.0}],
        tones=[],
    )
    a = angular_resample(sig.x, FS, rpm)
    assert a.max_order == pytest.approx(108.0, rel=1e-3)
    y = angular_to_time(a, FS, rpm)
    sl = _interior(y.size)
    assert _nrmse_db(y[sl], sig.x[sl]) < -60.0


def test_edge_effects_are_confined_to_the_first_and_last_revolution():
    rpm = _runup()
    sig = _synth(rpm)
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    y = angular_to_time(a, FS, rpm)
    err = np.abs(y - sig.x)
    assert err[round(0.1 * FS) : -round(0.05 * FS)].max() < 1e-3
    assert err[: round(0.05 * FS)].max() > 0.05  # and the edge really is imperfect


def test_output_times_and_length():
    rpm = _runup()
    sig = _synth(rpm)
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    # with an rpm array the output has one sample per speed sample
    assert angular_to_time(a, FS, rpm).shape == (rpm.size,)
    # without rpm, the grid spans a.t at fs
    y = angular_to_time(a, FS)
    assert y.size == int(np.floor((a.t[-1] - a.t[0]) * FS + 1e-9)) + 1
    assert y.size <= rpm.size
    # an explicit uniform grid selects the samples of the default result
    t = a.t[0] + np.arange(2000, 9000) / FS
    assert np.allclose(angular_to_time(a, FS, t=t), y[2000:9000], rtol=0, atol=1e-12)


def test_a_new_speed_curve_retimes_the_angle_domain_signal():
    # an angle-domain signal read on a slower speed curve: the order frequencies follow the new
    # speed (this is the angle -> time map, the basis of re-synthesis at another speed)
    rpm = _runup()
    orders = [{"order": 4, "amplitude": 1.0, "phase": 0.2}, {"order": 7.5, "amplitude": 0.5, "phase": 1.1}]
    sig = _synth(rpm, orders, tones=[])
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    rpm_new = 0.8 * rpm
    y = angular_to_time(a, FS, rpm_new)
    theta = rpm_to_phase(rpm_new, FS)
    truth = sum(
        o["amplitude"] * np.cos(2 * np.pi * o["order"] * theta + o["phase"]) for o in orders
    )
    sl = _interior(y.size)
    assert _nrmse_db(y[sl], truth[sl]) < -80.0


def test_constant_speed_with_a_scalar_rpm_and_an_explicit_grid():
    n = round(2 * FS)
    rpm = np.full(n, 3000.0)
    sig = _synth(rpm, [{"order": 5, "amplitude": 1.0, "phase": 0.4}], tones=[])
    a = angular_resample(sig.x, FS, 3000.0, max_order=40)
    y = angular_to_time(a, FS, 3000.0, t=np.arange(n) / FS)
    sl = _interior(n)
    assert _nrmse_db(y[sl], sig.x[sl]) < -80.0
    # a scalar speed alone does not define the length of the output
    with pytest.raises(ValueError, match="rpm array"):
        angular_to_time(a.x, FS, 3000.0, samples_per_rev=a.samples_per_rev)


def test_run_up_from_standstill_uses_the_same_speed_floor_as_the_forward_transform():
    t = _t()
    rpm = rpm_profiles(t, "linear", start=0.0, stop=4000.0)  # default floor: 80 rpm
    sig = _synth(rpm, [{"order": 4, "amplitude": 1.0, "phase": 0.5}, {"order": 9, "amplitude": 0.5}], tones=[])
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    y = angular_to_time(a, FS, rpm)
    sl = slice(round(0.5 * FS), y.size - round(0.1 * FS))
    assert _nrmse_db(y[sl], sig.x[sl]) < -80.0
    # a different floor shifts the angle and spoils the round trip
    other = angular_to_time(a, FS, rpm, min_rpm=10.0)
    assert _nrmse_db(other[sl], sig.x[sl]) > -20.0


def test_speed_reference_delay_is_passed_through_like_in_the_forward_transform():
    t = _t()
    d = 0.05
    ref = R0 + ACC * (t - d)  # the speed reference lags by 50 ms
    sig = _synth(R0 + ACC * t)
    a = angular_resample(sig.x, FS, ref, max_order=20, delay=d)
    y = angular_to_time(a, FS, ref, delay=d)
    sl = _interior(y.size, head=0.2)
    assert _nrmse_db(y[sl], sig.x[sl]) < -80.0
    assert _nrmse_db(angular_to_time(a, FS, ref)[sl], sig.x[sl]) > -30.0


def test_plain_array_input_matches_the_angular_signal_input():
    rpm = _runup()
    sig = _synth(rpm)
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    y1 = angular_to_time(a, FS, rpm)
    y2 = angular_to_time(a.x, FS, rpm, samples_per_rev=a.samples_per_rev, max_order=a.max_order)
    assert np.array_equal(y1, y2)
    y3 = angular_to_time(a.x, FS, rpm, samples_per_rev=50)  # max_order defaults to 0.4 * 50 = 20
    assert np.array_equal(y1, y3)


def test_channels_are_independent_and_the_map_is_linear():
    rpm = _runup()
    chans = [_synth(rpm, [{"order": o, "amplitude": amp, "phase": ph}], tones=[]).x
             for o, amp, ph in ((2, 1.0, 0.1), (6.5, 2.0, 1.4), (11, 0.5, -2.0))]
    X = np.column_stack(chans)
    a = angular_resample(X, FS, rpm, max_order=20)
    y = angular_to_time(a, FS, rpm)
    assert y.shape == X.shape
    for j in range(3):
        single = angular_to_time(angular_resample(chans[j], FS, rpm, max_order=20), FS, rpm)
        assert single.shape == (rpm.size,)
        assert np.allclose(y[:, j], single, rtol=0, atol=1e-12)
    a2 = AngularSignal(x=3.0 * a.x - 0.5, rev=a.rev, t=a.t, rpm=a.rpm,
                       samples_per_rev=a.samples_per_rev, max_order=a.max_order)
    y2 = angular_to_time(a2, FS, rpm)
    sl = _interior(rpm.size, head=0.3)
    assert np.allclose(y2[sl], 3.0 * y[sl] - 0.5, rtol=0, atol=1e-5)  # a DC offset passes with unit gain


def test_inputs_are_left_alone():
    rpm = _runup()
    sig = _synth(rpm)
    a = angular_resample(sig.x, FS, rpm, max_order=20)
    x_before, rpm_before = a.x.copy(), rpm.copy()
    angular_to_time(a, FS, rpm)
    assert np.array_equal(a.x, x_before) and np.array_equal(rpm, rpm_before)


# --------------------------------------------------------------------------
# angular_to_time: anti-aliasing
# --------------------------------------------------------------------------


def test_anti_aliasing_follows_the_speed_when_the_output_rate_is_lower():
    # analysed at 16 kHz with the default max_order 108; read out at 8 kHz. Order 90 is
    # 1.5 kHz at 1000 rpm and 6 kHz at 4000 rpm: it must be kept while below 0.45 * 8 kHz
    # and removed (not folded) once above 0.55 * 8 kHz, at the same instant-by-instant
    # criterion for the whole run-up. Order 20 never exceeds 1.4 kHz and is always kept.
    rpm = _runup()
    sig = _synth(
        rpm,
        [{"order": 20, "amplitude": 1.0, "phase": 0.3}, {"order": 90, "amplitude": 1.0, "phase": -1.0}],
        tones=[],
    )
    a = angular_resample(sig.x, FS, rpm)
    fs2 = FS / 2
    y = angular_to_time(a, fs2, rpm[::2])
    assert y.shape == (rpm.size // 2,)
    stem20, stem90 = sig.order_stems[::2, 0], sig.order_stems[::2, 1]
    f90 = 90.0 * rpm[::2] / 60.0
    inner = np.zeros(y.size, bool)
    inner[round(0.1 * fs2) : -round(0.05 * fs2)] = True
    keep = inner & (f90 < 0.45 * fs2 - 100.0)
    drop = inner & (f90 > 0.55 * fs2)
    assert keep.sum() > 4000 and drop.sum() > 4000
    assert np.abs((y - stem20 - stem90)[keep]).max() < 1e-3  # pass band intact, order 90 included
    assert np.abs((y - stem20)[drop]).max() < 1e-3  # order 90 removed, below -60 dB, not folded


def test_content_above_the_new_nyquist_rate_is_not_folded_at_constant_speed():
    # constant 3000 rpm = 50 Hz: order 100 is 5 kHz; at 8 kHz output (Nyquist 4 kHz) it
    # would fold to 3 kHz. It must vanish, while order 30 (1.5 kHz) stays.
    n = round(3 * FS)
    rpm = np.full(n, 3000.0)
    sig = _synth(
        rpm,
        [{"order": 30, "amplitude": 1.0, "phase": 0.1}, {"order": 100, "amplitude": 1.0, "phase": 0.2}],
        tones=[],
    )
    a = angular_resample(sig.x, FS, rpm)
    assert a.max_order > 100
    y = angular_to_time(a, FS / 2, rpm[::2])
    sl = slice(round(0.2 * FS / 2), y.size - round(0.1 * FS / 2))
    assert _nrmse_db(y[sl], sig.order_stems[::2, 0][sl]) < -60.0


def test_attenuation_sets_the_depth_of_the_anti_aliasing_filter():
    n = round(3 * FS)
    rpm = np.full(n, 3000.0)
    sig = _synth(rpm, [{"order": 100, "amplitude": 1.0, "phase": 0.2}], tones=[])
    a = angular_resample(sig.x, FS, rpm)
    sl = slice(round(0.2 * FS / 2), n // 2 - round(0.1 * FS / 2))
    peak = {att: np.abs(angular_to_time(a, FS / 2, rpm[::2], attenuation=att)[sl]).max()
            for att in (40.0, 80.0)}
    assert peak[40.0] < 0.02 and peak[80.0] < 1e-3
    assert peak[40.0] > 10.0 * peak[80.0]


# --------------------------------------------------------------------------
# angular_to_time: validation
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _a():
    rpm = _runup(1.0)
    sig = _synth(rpm)
    return angular_resample(sig.x, FS, rpm, max_order=20), rpm


def test_the_output_must_stay_inside_the_angle_domain_data(_a):
    a, rpm = _a
    with pytest.raises(ValueError, match="beyond"):
        angular_to_time(a, FS, 1.3 * rpm)  # needs 30 % more angle than the signal holds
    with pytest.raises(ValueError, match="beyond"):  # the stored (t, rev) map ends with the data
        angular_to_time(a, FS, t=a.t[0] + 0.5 + np.arange(rpm.size) / FS)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"fs": 0.0}, "fs"),
        ({"fs": np.nan}, "fs"),
        ({"samples_per_rev": 50}, "samples_per_rev"),  # an AngularSignal has its own
        ({"max_order": 0.0}, "max_order"),
        ({"max_order": 25.0}, "max_order"),  # Nyquist order of 50 samples/rev
        ({"upsample": 0}, "upsample"),
        ({"min_rpm": 0.0}, "min_rpm"),
        ({"min_rpm": -3.0}, "min_rpm"),
    ],
)
def test_invalid_arguments_raise(_a, kw, match):
    a, rpm = _a
    args = {"fs": FS, **kw}
    with pytest.raises(ValueError, match=match):
        angular_to_time(a, args.pop("fs"), rpm, **args)


def test_invalid_speed_and_grid_arguments_raise(_a):
    a, rpm = _a
    with pytest.raises(ValueError):
        angular_to_time(a, FS, rpm[:-1], t=a.t[0] + np.arange(rpm.size) / FS)  # length mismatch
    with pytest.raises(ValueError, match="positive"):
        angular_to_time(a, FS, np.zeros(rpm.size))
    bad = rpm.copy()
    bad[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        angular_to_time(a, FS, bad)
    with pytest.raises(ValueError, match="uniformly"):
        angular_to_time(a, FS, t=np.sort(np.random.default_rng(0).uniform(0.1, 0.5, 100)))
    with pytest.raises(ValueError, match="uniformly"):
        angular_to_time(a, FS, t=0.1 + np.arange(100) / (2 * FS))  # spacing 1/(2 fs)
    with pytest.raises(ValueError):
        angular_to_time(a, FS, t=[0.1])


def test_a_huge_oversampled_time_grid_raises_memory_error_before_allocating(_a):
    # fs = 1 Hz with a 4000 rpm speed needs an oversampling factor U of about 3000 on 1e5 samples
    a, _rpm = _a
    with pytest.raises(MemoryError, match="intermediate"):
        angular_to_time(a, 1.0, np.full(100_001, 4000.0))


def test_plain_array_needs_its_angular_rate_and_a_speed(_a):
    a, rpm = _a
    with pytest.raises(ValueError, match="samples_per_rev"):
        angular_to_time(a.x, FS, rpm)
    with pytest.raises(ValueError, match="rpm"):
        angular_to_time(a.x, FS, samples_per_rev=50)
    with pytest.raises(ValueError):
        angular_to_time(a.x, FS, rpm, samples_per_rev=-5)


def test_bad_angle_domain_signals_raise(_a):
    a, rpm = _a
    with pytest.raises(ValueError, match="at least 4"):
        angular_to_time(a.x[:3], FS, rpm, samples_per_rev=50)
    x = a.x.copy()
    x[10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        angular_to_time(x, FS, rpm, samples_per_rev=50)
    with pytest.raises(ValueError):
        angular_to_time(np.zeros((10, 2, 2)), FS, rpm, samples_per_rev=50)


# --------------------------------------------------------------------------
# stationarize
# --------------------------------------------------------------------------

_RU_ORDERS = [
    {"order": 4, "amplitude": 1.0, "phase": 0.3},
    {"order": 9.5, "amplitude": 0.5, "phase": -1.0},
    {"order": 15, "amplitude": 0.25, "phase": 2.0},
]


def _band_fraction(x, freqs, nperseg=2048, bins=2):
    """Share of the Welch power within +-`bins` bins of any of `freqs` (Hz)."""
    f, p = signal.welch(x, FS, nperseg=nperseg, window="hann")
    sel = np.zeros(f.size, bool)
    for fc in freqs:
        i = int(np.argmin(np.abs(f - fc)))
        sel[max(i - bins, 0) : i + bins + 1] = True
    return p[sel].sum() / p.sum()


def _line(y, order):
    """Complex peak amplitude of `order` in a stationarized signal, Hann-windowed whole record."""
    k = np.arange(y.x.size)
    w = np.hanning(y.x.size)
    f = order * y.rpm_ref / 60.0
    return 2.0 * np.sum(w * y.x * np.exp(-2j * np.pi * f * k / y.fs)) / w.sum()


def test_a_run_up_order_becomes_a_single_sharp_welch_peak_at_order_times_rpm_ref_over_60():
    rpm = _runup()
    sig = _synth(rpm, _RU_ORDERS, tones=[])
    y = stationarize(sig.x, FS, rpm)
    assert isinstance(y, StationarizedSignal) and isinstance(y, AngularSignal)
    assert y.rpm_ref == pytest.approx(np.median(rpm))
    freqs = [o["order"] * y.rpm_ref / 60.0 for o in _RU_ORDERS]
    spr = y.samples_per_rev
    xs = y.x[round(2 * spr) : -round(2 * spr)]  # without the edge revolutions
    assert _band_fraction(xs, freqs) > 0.99
    # the original sweeps over 3 x in frequency: no such concentration
    xo = sig.x[round(0.3 * FS) : -round(0.1 * FS)]
    assert _band_fraction(xo, freqs) < 0.3
    f, p = signal.welch(xs, FS, nperseg=2048, window="hann")
    _f0, po = signal.welch(xo, FS, nperseg=2048, window="hann")
    assert p.max() > 8.0 * po.max()  # the strongest line is much sharper
    assert abs(f[np.argmax(p)] - freqs[0]) <= f[1] - f[0]


@pytest.mark.parametrize("rpm_ref", [None, 2437.3, 1000.0, 4000.0])
def test_warped_orders_have_exact_frequency_amplitude_and_phase(rpm_ref):
    rpm = _runup()
    sig = _synth(rpm, _RU_ORDERS, tones=[])
    y = stationarize(sig.x, FS, rpm, rpm_ref=rpm_ref)
    ref = np.median(rpm) if rpm_ref is None else rpm_ref
    assert y.rpm_ref == pytest.approx(ref)
    k = slice(round(2 * y.samples_per_rev), -round(2 * y.samples_per_rev))
    yy = StationarizedSignal(
        x=y.x[k], rev=y.rev[k], t=y.t[k], rpm=y.rpm[k], samples_per_rev=y.samples_per_rev,
        max_order=y.max_order, fs=y.fs, rpm_ref=y.rpm_ref,
    )
    # the phase reference is sample k0 of the full record
    k0 = k.start
    for o in _RU_ORDERS:
        c = _line(yy, o["order"]) * np.exp(-2j * np.pi * o["order"] * y.rpm_ref / 60.0 * k0 / FS)
        assert abs(abs(c) / o["amplitude"] - 1.0) < 1e-3  # within 0.01 dB
        assert abs(np.angle(c * np.exp(-1j * o["phase"]))) < 1e-3


def test_samples_per_rev_is_the_exact_float_not_a_rounded_integer():
    rpm = _runup()
    sig = _synth(rpm, _RU_ORDERS, tones=[])
    y = stationarize(sig.x, FS, rpm, rpm_ref=2437.3)
    assert isinstance(y.samples_per_rev, float)
    assert y.samples_per_rev == pytest.approx(FS * 60.0 / 2437.3, rel=1e-14)
    assert abs(y.samples_per_rev - round(y.samples_per_rev)) > 0.1  # genuinely non-integer
    assert np.allclose(y.rev, np.arange(y.x.size) / y.samples_per_rev, rtol=0, atol=1e-9)
    assert y.max_order == pytest.approx(0.4 * y.samples_per_rev)
    # whole record: spr * (total revolutions) samples
    assert abs(y.x.size - y.samples_per_rev * sig.rev[-1]) <= 2.0
    assert y.t_warped[1] == pytest.approx(1.0 / FS) and y.t_warped.shape == y.rev.shape
    # a speed curve and the integer path are unaffected: angular_resample still truncates
    z = angular_resample(sig.x, FS, rpm, samples_per_rev=64.9, max_order=10)
    assert z.samples_per_rev == 64 and isinstance(z.samples_per_rev, int)


def test_constant_speed_with_rpm_ref_equal_to_the_speed_is_the_identity_warp():
    n = round(2 * FS)
    rpm = np.full(n, 3000.0)
    sig = _synth(rpm, [{"order": 5, "amplitude": 1.0, "phase": 0.4}, {"order": 40, "amplitude": 0.5}], tones=[])
    y = stationarize(sig.x, FS, rpm, rpm_ref=3000.0)
    assert y.samples_per_rev == pytest.approx(320.0)
    m = min(y.x.size, n)
    sl = slice(round(0.2 * FS), m - round(0.1 * FS))
    assert _nrmse_db(y.x[sl], sig.x[sl]) < -60.0
    assert y.x.size in (n, n - 1)


def test_content_above_the_scaled_cutoff_is_removed_not_aliased():
    # 1000 rpm for 1.5 s, then a ramp to 4000 rpm. With rpm_ref = 2500 the angle-domain filter
    # (pass band 0.4 spr, stop band from 0.5 spr) corresponds to 0.4 and 0.5 fs * rpm / rpm_ref:
    # 2560 and 3200 Hz at 1000 rpm, 10 and 12.8 kHz at 4000 rpm. A fixed 4 kHz tone (order 240
    # at 1000 rpm) is removed at the start and kept later (order 85 at 2800 rpm).
    t = _t(4.0)
    rpm = rpm_profiles(t, "piecewise", knots=([0, 1.5, 4], [1000, 1000, 4000]))
    sig = _synth(rpm, [{"order": 3, "amplitude": 0.0}], tones=[{"freq": 4000.0, "amplitude": 1.0, "phase": 0.0}])
    y = stationarize(sig.x, FS, rpm, rpm_ref=2500.0)
    early = (y.t > 0.3) & (y.t < 1.3)
    late = (y.t > 3.0) & (y.t < 3.8)
    assert early.sum() > 1000 and late.sum() > 1000
    assert np.abs(y.x[early]).max() < 1e-3
    assert np.abs(y.x[late]).max() == pytest.approx(1.0, abs=2e-3)
    # lower rpm_ref keeps the bandwidth (at the price of a longer warped signal)
    y2 = stationarize(sig.x, FS, rpm, rpm_ref=1000.0)
    assert y2.x.size > y.x.size
    early2 = (y2.t > 0.3) & (y2.t < 1.3)
    assert np.abs(y2.x[early2]).max() == pytest.approx(1.0, abs=2e-3)


def test_a_fixed_frequency_tone_is_swept_not_stationary_after_warping():
    rpm = _runup()
    sig = _synth(rpm, [{"order": 3, "amplitude": 0.0}], tones=[{"freq": 100.0, "amplitude": 1.0, "phase": 0.0}])
    y = stationarize(sig.x, FS, rpm)
    xs = y.x[round(2 * y.samples_per_rev) : -round(2 * y.samples_per_rev)]
    assert _band_fraction(xs, [100.0]) < 0.5  # the warped tone spreads over its sweep (a stationary tone: > 0.99)
    assert _band_fraction(xs, [100.0], nperseg=512) < _band_fraction(sig.x[1000:-1000], [100.0], nperseg=512) + 0.2


def test_stationarize_channels_are_independent():
    rpm = _runup()
    chans = [_synth(rpm, [{"order": o, "amplitude": 1.0, "phase": 0.1}], tones=[]).x for o in (3, 8)]
    y = stationarize(np.column_stack(chans), FS, rpm)
    assert y.x.shape == (y.rev.size, 2)
    for j, c in enumerate(chans):
        single = stationarize(c, FS, rpm)
        assert np.allclose(y.x[:, j], single.x, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# destationarize
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rpm_ref", "variant"),
    [(r, "object") for r in (None, 2437.3, 1000.0, 4000.0)]
    + [(None, "array-default-ref")]
    + [(r, "array-explicit-ref") for r in (None, 2437.3, 1000.0, 4000.0)],
)
def test_destationarize_inverts_stationarize_within_minus_60_db(rpm_ref, variant):
    rpm = _runup()
    sig = _synth(rpm)
    y = stationarize(sig.x, FS, rpm, rpm_ref=rpm_ref)
    if variant == "object":
        x = destationarize(y, FS, rpm)
    else:
        x = destationarize(y.x, FS, rpm, rpm_ref=None if variant == "array-default-ref" else y.rpm_ref)
    assert x.shape == sig.x.shape
    sl = _interior(x.size)
    err = _nrmse_db(x[sl], sig.x[sl])
    assert err < -60.0
    assert err < -80.0  # measured: about -88 dB


@pytest.mark.parametrize("profile", ["run-down", "drop"])
def test_destationarize_round_trip_on_other_speed_curves(profile):
    rpm = _profile(profile)
    sig = _synth(rpm)
    y = stationarize(sig.x, FS, rpm)
    x = destationarize(y, FS, rpm)
    sl = _interior(x.size)
    assert _nrmse_db(x[sl], sig.x[sl]) < -60.0


def test_destationarize_returns_orders_to_their_true_frequencies():
    rpm = _runup()
    sig = _synth(rpm, _RU_ORDERS, tones=[])
    x = destationarize(stationarize(sig.x, FS, rpm), FS, rpm)
    sl = _interior(x.size)
    truth = sig.stems["orders"]
    assert np.max(np.abs(x - truth)[sl]) < 1e-3
    # re-timing: the same warped signal read on a faster curve moves the orders up in frequency
    y = stationarize(sig.x, FS, rpm)
    x_slow = destationarize(y, FS, 0.8 * rpm)
    theta = rpm_to_phase(0.8 * rpm, FS)
    new_truth = sum(o["amplitude"] * np.cos(2 * np.pi * o["order"] * theta + o["phase"]) for o in _RU_ORDERS)
    assert _nrmse_db(x_slow[sl], new_truth[sl]) < -60.0


def test_destationarize_multichannel_and_output_length():
    rpm = _runup()
    chans = [_synth(rpm, [{"order": o, "amplitude": 1.0, "phase": 0.1}], tones=[]).x for o in (3, 8)]
    X = np.column_stack(chans)
    x = destationarize(stationarize(X, FS, rpm), FS, rpm)
    assert x.shape == X.shape
    sl = _interior(x.shape[0])
    assert _nrmse_db(x[sl], X[sl]) < -60.0


# --------------------------------------------------------------------------
# stationarize / destationarize: validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "exc", "match"),
    [
        ({"rpm_ref": 0.0}, ValueError, "rpm_ref"),
        ({"rpm_ref": -100.0}, ValueError, "rpm_ref"),
        ({"rpm_ref": np.nan}, ValueError, "rpm_ref"),
        ({"samples_per_rev": 64}, TypeError, "rpm_ref"),
        ({"bogus": 1}, TypeError, "bogus"),
        ({"min_rpm": 0.0}, ValueError, "min_rpm"),
        ({"upsample": 0}, ValueError, "upsample"),
        # above half of the 384 samples per revolution at 2500 rpm
        ({"rpm_ref": 2500.0, "max_order": 200.0}, ValueError, "max_order"),
    ],
)
def test_stationarize_invalid_arguments_raise(kw, exc, match):
    rpm = _runup(1.0)
    with pytest.raises(exc, match=match):
        stationarize(np.zeros(rpm.size), FS, rpm, **kw)


def test_stationarize_invalid_speed_signal_and_rate_raise():
    rpm = _runup(1.0)
    x = np.zeros(rpm.size)
    with pytest.raises(ValueError, match="positive"):
        stationarize(x, FS, np.zeros(rpm.size))
    with pytest.raises(ValueError):
        stationarize(x, FS, rpm[:-1])
    with pytest.raises(ValueError):
        stationarize(x, 0.0, rpm)
    with pytest.raises(ValueError):
        stationarize(np.zeros((rpm.size, 2, 2)), FS, rpm)
    bad = rpm.copy()
    bad[3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        stationarize(x, FS, bad)


def test_destationarize_invalid_arguments_raise():
    rpm = _runup(1.0)
    y = stationarize(np.zeros(rpm.size), FS, rpm)
    with pytest.raises(ValueError, match="rpm_ref"):
        destationarize(y, FS, rpm, rpm_ref=2500.0)  # the object carries its own rate
    with pytest.raises(ValueError, match="rpm_ref"):
        destationarize(y.x, FS, rpm, rpm_ref=-1.0)
    with pytest.raises(ValueError, match="array"):
        destationarize(y.x, FS, 2500.0)
    with pytest.raises(ValueError, match="non-finite"):
        destationarize(y.x, FS, np.where(np.arange(rpm.size) == 4, np.nan, rpm))
    with pytest.raises(TypeError, match="rpm_ref"):
        destationarize(y.x, FS, rpm, samples_per_rev=64)
    with pytest.raises(TypeError):
        destationarize(y, FS, rpm, bogus=1)
    with pytest.raises(ValueError):
        destationarize(y.x, 0.0, rpm)
