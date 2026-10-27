"""Tests of the tacholess speed estimator against analytic truth.

All signals come from ``rota.synth.synth_runup``; the truth is the speed curve
that was put into the signal, evaluated at the frame times of the estimate.
"""

import time
import warnings

import numpy as np
import pytest

from rota._ridge import (
    RidgeQualityWarning,
    RidgeResult,
    WhiteningWarning,
    _auto_slope,
    _l1_max_sum,
    _path_rate,
    _running_median_warped,
    _viterbi,
    rpm_from_ridge,
)
from rota.metrics import track_error
from rota.synth import rpm_profiles, synth_runup

FS = 8000.0  # Hz
AMPS = (1.0, 0.7, 0.5, 0.35)  # peak amplitudes of orders 1..4


def _runup(
    r0,
    r1,
    dur=10.0,
    *,
    kind="linear",
    noise=3.0,
    orders=(1, 2, 3, 4),
    amps=AMPS,
    seed=4,
    fs=FS,
    phase=None,
    **kwargs,
):
    """Run-up from ``r0`` to ``r1`` rpm with the given orders and white noise."""
    t = np.arange(round(dur * fs)) / fs
    rpm = rpm_profiles(t, kind, start=r0, stop=r1)
    specs = [dict(order=o, amplitude=a, phase=phase) for o, a in zip(orders, amps)]
    return synth_runup(fs, rpm, specs, noise_rms=noise, seed=seed, **kwargs)


def _truth(res, s):
    """True speed at the frame times of ``res``."""
    return np.interp(res.t, s.t, s.rpm)


def _err(res, s, mask=None):
    truth = _truth(res, s)
    if mask is None:
        return track_error(res.rpm, truth, at_bound=res.at_bound)
    return track_error(res.rpm[mask], truth[mask])


def _quiet(*args, **kwargs):
    """Call the estimator and fail if it warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return rpm_from_ridge(*args, **kwargs)


# --------------------------------------------------------------------------
# accuracy on in-range signals (ROADMAP 5.4: >= 90 % of frames within 3 %)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("noise", [1.0, 4.0])
@pytest.mark.parametrize("r0, r1", [(3000.0, 3000.0), (1500.0, 3000.0)])
def test_steady_and_slow_runup_within_three_percent(r0, r1, noise):
    s = _runup(r0, r1, noise=noise)
    res = _quiet(s.x, FS, (1000.0, 6000.0))  # no bound warning on valid data
    e = _err(res, s)
    assert e["within"][0.03] >= 0.9  # ROADMAP 5.4
    assert e["within"][0.03] >= 0.98  # measured: 1.0
    assert e["median_rel_error"] < 0.005
    assert res.bound_fraction == 0.0
    assert not res.beyond.any()


@pytest.mark.parametrize("r0, r1", [(3000.0, 3000.0), (1500.0, 3000.0)])
def test_noisy_signal_still_within_three_percent(r0, r1):
    # Noise at the level of the 4th order (about 3 dB per-bin SNR) and below.
    # The out-of-range guard may now fire by chance on a few frames, which then
    # carry very low confidence; accuracy is what is checked here.
    s = _runup(r0, r1, noise=8.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = rpm_from_ridge(s.x, FS, (1000.0, 6000.0))
    assert _err(res, s)["within"][0.03] >= 0.9  # measured: 0.97-1.0
    assert res.confidence[res.beyond].mean() < 0.1 if res.beyond.any() else True


def test_accuracy_at_high_snr_and_time_alignment():
    # Low noise: the error is the grid/estimator error, so a shifted time axis
    # (decimation delay) would show as 150 rpm/s * shift / 2250 rpm.
    s = _runup(1500.0, 3000.0, noise=0.3)
    res = _quiet(s.x, FS, (1000.0, 6000.0))
    e = _err(res, s)
    assert e["median_rel_error"] < 0.001  # measured: ~0.03 %
    # Interior frames are within 0.5 % (measured: 0.25 %). The first and last
    # frames can be pulled by up to ~1 % towards their only neighbour: the L1
    # path penalty is not balanced there (the score peak of a 25 Hz order
    # chain is several % wide).
    inner = slice(3, -3)
    truth = _truth(res, s)
    assert np.max(np.abs(res.rpm[inner] - truth[inner]) / truth[inner]) < 0.005
    assert e["max_rel_error"] < 0.015  # measured: 1.04 % at frame 0
    assert np.corrcoef(res.rpm, truth)[0, 1] > 0.9999


def test_resonance_and_fixed_tone_t2_like():
    # ROADMAP T2: orders + structural resonance + fixed line + noise.
    s = _runup(
        1500.0, 3000.0,
        resonances=[dict(freq=120.0, q=8.0, gain=4.0)],
        tones=[dict(freq=60.0, amplitude=2.0)],
    )
    for moving in (False, True):
        res = _quiet(s.x, FS, (1000.0, 6000.0), moving=moving)
        assert _err(res, s)["within"][0.03] >= 0.98  # measured: 1.0


def test_exponential_rundown_t3_like():
    # ROADMAP T3: non-linear (constant relative rate) deceleration.
    s = _runup(4000.0, 1500.0, kind="exponential")
    res = _quiet(s.x, FS, (1000.0, 6000.0))
    assert _err(res, s)["within"][0.03] >= 0.98  # measured: 1.0
    assert res.rpm[0] > res.rpm[-1]


def test_fast_runup_follows_with_short_window():
    # 1000 rpm/s: order 4 sweeps 17 Hz in 0.25 s, so whiten must be wider.
    s = _runup(1200.0, 5200.0, dur=4.0)
    res = _quiet(s.x, FS, (1000.0, 6000.0), window=0.25, whiten=30.0)
    assert _err(res, s)["within"][0.03] >= 0.95  # measured: 0.99


def test_low_speed_needs_long_window():
    # Harmonic spacing 15 Hz at 900 rpm: a 1 s window keeps the lines narrow
    # enough for the whitening median to leave them in place.
    s = _runup(900.0, 1800.0)
    res = _quiet(s.x, FS, (600.0, 3000.0), window=1.0, hop=0.1, whiten=20.0)
    assert _err(res, s)["within"][0.03] >= 0.95  # measured: 1.0


@pytest.mark.parametrize("fs, orders", [(300.0, (1, 2)), (48000.0, (1, 2, 3, 4))])
def test_sampling_rates_with_and_without_decimation(fs, orders):
    # fs=300 needs no decimation (comb at 100 Hz); fs=48 kHz decimates by 24+.
    s = _runup(1200.0, 2400.0, dur=6.0, noise=1.0 if fs < 1000 else 3.0,
               orders=orders, amps=AMPS[: len(orders)], fs=fs)
    res = _quiet(s.x, fs, (600.0, 3000.0), orders=orders)
    assert _err(res, s)["within"][0.03] >= 0.98


def test_scale_invariance_and_input_untouched():
    s = _runup(1500.0, 3000.0, dur=6.0)
    x = s.x.copy()
    r1 = _quiet(x, FS, (1000.0, 6000.0))
    r2 = _quiet(1e4 * x, FS, (1000.0, 6000.0))
    assert np.array_equal(x, s.x)
    assert np.allclose(r1.rpm, r2.rpm, rtol=1e-9)


def test_whitening_removes_the_bias_of_a_noise_hump():
    # A band-limited noise hump (70-130 Hz) that carries no order makes every
    # candidate with a comb tooth in the band look good unless the spectrum is
    # whitened. (A resonance would not do: it also amplifies the orders.)
    from scipy import signal

    sos = signal.butter(4, [70.0, 130.0], btype="bandpass", fs=FS, output="sos")
    w_on, w_off = [], []
    for seed in (1, 2, 3):
        s = _runup(1500.0, 3000.0, noise=4.0, seed=seed)
        hump = signal.sosfiltfilt(sos, np.random.default_rng(100 + seed).standard_normal(s.x.size))
        x = s.x + 40.0 * hump / hump.std()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w_on.append(_err(rpm_from_ridge(x, FS, (1000.0, 6000.0), whiten=15.0), s)["within"][0.03])
            w_off.append(_err(rpm_from_ridge(x, FS, (1000.0, 6000.0), whiten=None), s)["within"][0.03])
    assert np.mean(w_on) >= 0.8  # measured: 0.88
    assert np.mean(w_on) > np.mean(w_off) + 0.15  # measured: 0.88 vs 0.61


def test_whiten_none_runs():
    s = _runup(3000.0, 3000.0, dur=6.0, noise=1.0)
    res = _quiet(s.x, FS, (1000.0, 6000.0), whiten=None)
    assert _err(res, s)["within"][0.03] >= 0.98


# --------------------------------------------------------------------------
# fixed lines: moving=True
# --------------------------------------------------------------------------


def test_moving_suppresses_strong_fixed_tone():
    # A 100 Hz line 5000x the order amplitude puts a stripe on the score map
    # at the speeds whose comb tooth m sits at 100 Hz (6000, 3000, 2000, 1500
    # rpm for m = 1..4), which out-scores the true comb at this SNR. Checked
    # over 8 seeds: without `moving` within-3 % is 0.0 each time, with it
    # at least 0.99. These numbers are those of a constant 15 Hz median, the
    # default before whiten="auto": it is given explicitly in the first two
    # runs (this test encoded the old default); the default is checked below.
    s = _runup(1200.0, 2800.0, noise=6.0, tones=[dict(freq=100.0, amplitude=5000.0, phase=0.3)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = rpm_from_ridge(s.x, FS, (1000.0, 6000.0), moving=False, whiten=15.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # noise this high: a few chance flags
        moved = rpm_from_ridge(s.x, FS, (1000.0, 6000.0), moving=True, whiten=15.0)
    w_plain = _err(plain, s)["within"][0.03]
    w_moved = _err(moved, s)["within"][0.03]
    assert w_moved >= 0.95  # measured: 1.0
    assert moved.bound_fraction < 0.1  # measured: ~0.01
    assert w_plain < 0.5  # measured: 0.0 (locks on the stripes)
    # The stripe's own score is what moving removes: score at 3000 rpm
    # (tooth m=2 on the tone) while the true speed is far away. (Whitening
    # already lowers a tone this strong, whose sidelobes raise the local
    # median; measured 4.9 vs 1.1.)
    far = _truth(moved, s) < 2000.0
    g = int(np.argmin(np.abs(moved.rpm_grid - 3000.0)))
    assert plain.score[far, g].mean() > 3.0 * moved.score[far, g].mean()

    # Default whitening (whiten="auto"): moving=True still removes the tone
    # (over 8 seeds 0.98 to 1.0 within 3 %; stripe score 2.6 times lower than
    # without moving).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto_moved = rpm_from_ridge(s.x, FS, (1000.0, 6000.0), moving=True)
        auto_plain = rpm_from_ridge(s.x, FS, (1000.0, 6000.0), moving=False)
    assert _err(auto_moved, s)["within"][0.03] >= 0.95  # measured: 0.99
    assert auto_moved.bound_fraction < 0.1
    assert not auto_moved.whiten_violated.any()
    # (window="auto" may choose different windows, hence frames, for the two runs)
    far_m, far_p = _truth(auto_moved, s) < 2000.0, _truth(auto_plain, s) < 2000.0
    g_m = int(np.argmin(np.abs(auto_moved.rpm_grid - 3000.0)))
    g_p = int(np.argmin(np.abs(auto_plain.rpm_grid - 3000.0)))
    assert auto_plain.score[far_p, g_p].mean() > 2.0 * auto_moved.score[far_m, g_m].mean()


# --------------------------------------------------------------------------
# T10: speed outside the search range must be reported, never returned silently
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rng", [(1000.0, 2000.0), (2200.0, 6000.0), (2900.0, 6000.0)])
def test_truth_outside_range_is_flagged_and_warns(rng):
    # The true speed runs 1500 -> 3000 rpm; the range excludes its upper part
    # (1000-2000), its lower part (2200-6000) or nearly all of it
    # (2900-6000). The truth stays within a factor 2 of the range, which is
    # what the guard bands cover (see the Notes of rpm_from_ridge).
    s = _runup(1500.0, 3000.0)
    with pytest.warns(RuntimeWarning, match="search range"):
        res = rpm_from_ridge(s.x, FS, rng)
    truth = _truth(res, s)
    out = (truth < rng[0] * 0.99) | (truth > rng[1] * 1.01)
    inside = (truth > rng[0] * 1.05) & (truth < rng[1] * 0.95)
    assert out.sum() > 50
    assert res.at_bound[out].mean() >= 0.9  # measured: 0.98-0.99
    assert res.bound_fraction > 0
    # No frame that is wrong by more than 3 % goes unflagged (apart from a
    # handful at the transition).
    wrong = np.abs(res.rpm - truth) / truth > 0.03
    assert np.mean(wrong & ~res.at_bound) < 0.03
    if inside.sum() > 20:
        assert res.at_bound[inside].mean() <= 0.1
        assert _err(res, s, inside)["within"][0.03] >= 0.9


def test_at_bound_is_end_test_or_beyond():
    s = _runup(1500.0, 3000.0)
    with pytest.warns(RuntimeWarning):
        res = rpm_from_ridge(s.x, FS, (1000.0, 2000.0))
    step = np.log(res.rpm_grid[1] / res.rpm_grid[0])
    near_end = (res.rpm <= 1000.0 * np.exp(2 * step)) | (res.rpm >= 2000.0 * np.exp(-2 * step))
    # every flag is explained by the end test or by `beyond`, and `beyond`
    # implies the flag
    assert not np.any(res.at_bound & ~res.beyond & ~near_end)
    assert np.all(res.at_bound[res.beyond])
    assert res.rpm.min() >= 1000.0 - 1e-9 and res.rpm.max() <= 2000.0 + 1e-9


def test_end_test_flags_when_no_guard_band_fits():
    # fs = 300 Hz leaves no room above the range for a guard band (the comb
    # would reach the Nyquist frequency), so only "path at the end of the
    # range" can flag; at 4100 rpm the best in-range candidate is the end.
    fs = 300.0
    s = _runup(4100.0, 4100.0, dur=8.0, noise=0.5, orders=(1, 2), amps=(1.0, 0.7), fs=fs)
    with pytest.warns(RuntimeWarning, match="search range"):
        res = rpm_from_ridge(s.x, fs, (600.0, 4000.0), orders=(1, 2))
    assert res.guard_range[1] == pytest.approx(4000.0)  # no upper guard
    assert res.guard_range[0] == pytest.approx(300.0, rel=0.01)  # lower guard: half
    assert not res.beyond.any()
    assert res.at_bound.all()
    assert res.rpm.max() <= 4000.0


def test_guard_range_is_reported():
    s = _runup(1500.0, 1800.0, dur=4.0)
    res = _quiet(s.x, FS, (1000.0, 2000.0))
    assert res.guard_range[0] == pytest.approx(500.0, rel=0.01)
    assert res.guard_range[1] == pytest.approx(4000.0, rel=0.01)


def test_integer_comb_outside_range_flags_the_wrong_ratio():
    # See the half-order test: the integer comb wants 1.5 x speed. Where that
    # leaves the range the lock is flagged rather than returned silently.
    s = _runup(1200.0, 2400.0, orders=(1.5, 3, 4.5), amps=(1, 0.7, 0.5))
    with pytest.warns(RuntimeWarning):
        res = rpm_from_ridge(s.x, FS, (900.0, 3000.0))
    truth = _truth(res, s)
    locked = np.abs(res.rpm / truth - 1.5) < 0.05
    assert locked.mean() > 0.5  # measured: 0.73
    # where 1.5 x speed is more than 3 % above the range, the frame is flagged
    wanted_beyond = 1.5 * truth > 3000.0 * 1.03
    assert wanted_beyond.sum() > 20
    assert res.at_bound[wanted_beyond].mean() >= 0.9


# --------------------------------------------------------------------------
# comb configuration (ROADMAP 6, item 3)
# --------------------------------------------------------------------------


def test_half_order_comb_tracks_integer_comb_locks_to_ratio():
    # Three-cylinder-like: firing orders 1.5, 3, 4.5 carry the energy.
    s = _runup(1200.0, 2400.0, orders=(1.5, 3, 4.5), amps=(1, 0.7, 0.5))
    good = _quiet(s.x, FS, (900.0, 4000.0), orders=(1.5, 3, 4.5))
    assert _err(good, s)["within"][0.03] >= 0.98  # measured: 1.0
    assert np.median(good.rpm / _truth(good, s)) == pytest.approx(1.0, abs=0.01)

    bad = _quiet(s.x, FS, (900.0, 4000.0))  # default integer comb 1..4
    e = _err(bad, s)
    assert e["within"][0.03] < 0.1  # measured: 0.0
    # exactly the ratio 3/2: integer teeth 1..4 land on 1.5, 3, 4.5, 6
    assert np.median(bad.rpm / _truth(bad, s)) == pytest.approx(1.5, abs=0.02)


def test_half_order_signal_with_weak_integer_orders():
    s = _runup(1200.0, 2400.0, orders=(1.5, 3, 4.5, 1, 2), amps=(1, 0.7, 0.5, 0.15, 0.12))
    res = _quiet(s.x, FS, (900.0, 4000.0), orders=(1.5, 3, 4.5, 6))
    assert _err(res, s)["within"][0.03] >= 0.98


def test_even_order_machine_needs_even_comb():
    # Only orders 2, 4, 6 are strong: the default 1..4 comb scores twice the
    # speed as well as the truth (documented in the Notes); the matching comb
    # fixes it.
    s = _runup(1500.0, 3000.0, orders=(1, 2, 3, 4, 6), amps=(0.1, 1.0, 0.1, 0.6, 0.3))
    default = _quiet(s.x, FS, (1000.0, 6000.0))
    assert np.median(default.rpm / _truth(default, s)) == pytest.approx(2.0, abs=0.02)
    even = _quiet(s.x, FS, (1000.0, 6000.0), orders=(2, 4, 6))
    assert _err(even, s)["within"][0.03] >= 0.98  # measured: 1.0


def test_default_weights_break_the_subharmonic_tie():
    # One visible order, comb 1..4, independent frames (penalty 0): speed/2,
    # /3, /4 put comb teeth 2, 3, 4 on the order. Equal weights tie with the
    # truth; the decreasing default prefers it.
    s = _runup(1500.0, 3000.0, orders=(1,), amps=(1.0,))

    def sub_fraction(res):
        ratio = res.rpm / _truth(res, s)
        return np.mean(
            (np.abs(ratio - 0.5) < 0.05) | (np.abs(ratio - 1 / 3) < 0.04) | (np.abs(ratio - 0.25) < 0.03)
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = rpm_from_ridge(s.x, FS, (600.0, 6000.0), penalty=0.0)
        equal = rpm_from_ridge(s.x, FS, (600.0, 6000.0), penalty=0.0, weights=(1, 1, 1, 1))
    assert sub_fraction(equal) > 0.2  # measured: 0.38
    assert sub_fraction(default) < 0.5 * sub_fraction(equal)  # measured: 0.10
    assert _err(default, s)["within"][0.03] > _err(equal, s)["within"][0.03] + 0.15
    assert default.weights[0] > default.weights[-1]
    assert default.weights.sum() == pytest.approx(1.0)


def test_weights_are_normalised_and_used():
    s = _runup(3000.0, 3000.0, dur=4.0, noise=1.0)
    a = _quiet(s.x, FS, (1000.0, 6000.0), weights=(4, 3, 2, 1))
    b = _quiet(s.x, FS, (1000.0, 6000.0), weights=(0.4, 0.3, 0.2, 0.1))
    assert np.allclose(a.weights, [0.4, 0.3, 0.2, 0.1])
    assert np.allclose(a.rpm, b.rpm)
    assert np.array_equal(a.orders, [1, 2, 3, 4])


# --------------------------------------------------------------------------
# confidence
# --------------------------------------------------------------------------


def test_confidence_is_low_in_a_noise_only_section():
    # Orders switched off for 2100-2400 rpm through the gain law: the signal
    # holds noise only there, in the middle of the run-up.
    t = np.arange(round(10.0 * FS)) / FS
    rpm = rpm_profiles(t, "linear", start=1500.0, stop=3000.0)
    specs = [
        dict(order=o, amplitude=a, gain_rpm=[1500, 2090, 2100, 2400, 2410, 3000],
             gain=[1, 1, 0, 0, 1, 1])
        for o, a in zip((1, 2, 3, 4), AMPS)
    ]
    s = synth_runup(FS, rpm, specs, noise_rms=3.0, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the empty section may flag by chance
        res = rpm_from_ridge(s.x, FS, (1000.0, 6000.0))
    truth = _truth(res, s)
    gap = (truth > 2150.0) & (truth < 2350.0)
    clean = (truth < 2000.0) | (truth > 2500.0)
    assert gap.sum() > 20
    assert not res.at_bound[clean].any()  # the signal sections are not flagged
    assert res.confidence[clean].mean() > 0.3  # measured: 0.40
    assert res.confidence[gap].mean() < 0.1  # measured: 0.00
    assert np.all((res.confidence >= 0) & (res.confidence <= 1))
    # the tracked part outside the gap is right
    assert _err(res, s, clean)["within"][0.03] >= 0.98


def test_confidence_falls_with_noise():
    lo = _quiet(_runup(1500.0, 3000.0, noise=1.0).x, FS, (1000.0, 6000.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hi = rpm_from_ridge(_runup(1500.0, 3000.0, noise=10.0).x, FS, (1000.0, 6000.0))
    assert lo.confidence.mean() > hi.confidence.mean() + 0.15  # measured: 0.39 vs 0.14


# --------------------------------------------------------------------------
# result object: t_out, rpm_at, axes, refinement, path
# --------------------------------------------------------------------------


def test_t_out_interpolates_onto_signal_samples():
    s = _runup(1500.0, 3000.0, dur=6.0)
    res = _quiet(s.x, FS, (1000.0, 6000.0), t_out=s.t)
    assert res.rpm_out.shape == s.t.shape and res.t_out.shape == s.t.shape
    assert np.allclose(res.rpm_out, np.interp(s.t, res.t, res.rpm))
    assert np.allclose(res.rpm_at(res.t), res.rpm)
    # the ends are held, not extrapolated
    assert res.rpm_out[0] == res.rpm[0] and res.rpm_out[-1] == res.rpm[-1]
    # between the first and last frame centre the interpolated speed is right
    inner = (s.t >= res.t[0]) & (s.t <= res.t[-1])
    e = track_error(res.rpm_out[inner], s.rpm[inner])
    assert e["median_rel_error"] < 0.005
    assert e["within"][0.03] >= 0.98
    # without t_out nothing is attached
    assert _quiet(s.x, FS, (1000.0, 6000.0)).rpm_out is None


def test_multichannel_averages_the_power_spectra():
    # The channels' power spectra are averaged, not the signals: identical
    # channels give the single-channel result, and so do channels in opposite
    # phase (whose signal mean is zero).
    s = _runup(1500.0, 3000.0, dur=6.0)
    one = _quiet(s.x, FS, (1000.0, 6000.0))
    two = _quiet(np.column_stack([s.x, s.x]), FS, (1000.0, 6000.0))
    anti = _quiet(np.column_stack([s.x, -s.x]), FS, (1000.0, 6000.0))
    assert np.allclose(two.rpm, one.rpm, rtol=1e-9) and np.allclose(anti.rpm, one.rpm, rtol=1e-9)
    # Channels whose orders have unrelated phases (two microphones some way
    # apart) and independent noise: power averaging beats one channel and the
    # signal mean, in which the orders partly cancel.
    chans = [_runup(1500.0, 3000.0, noise=10.0, seed=k) for k in (1, 2, 3)]
    X = np.column_stack([c.x for c in chans])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multi = rpm_from_ridge(X, FS, (1000.0, 6000.0))
        mean = rpm_from_ridge(X.mean(axis=1), FS, (1000.0, 6000.0))
        single = rpm_from_ridge(chans[0].x, FS, (1000.0, 6000.0))
    w_multi = _err(multi, chans[0])["within"][0.03]
    w_mean = _err(mean, chans[0])["within"][0.03]
    w_single = _err(single, chans[0])["within"][0.03]
    assert w_multi >= 0.97 and w_multi >= w_single and w_multi >= w_mean


def test_axes_grid_and_result_shapes():
    s = _runup(1500.0, 3000.0)
    res = _quiet(s.x, FS, (1000.0, 6000.0))
    assert isinstance(res, RidgeResult)
    T, G = res.score.shape
    assert res.t.shape == res.rpm.shape == res.confidence.shape == res.at_bound.shape == (T,)
    assert res.beyond.shape == (T,) and res.rpm_grid.shape == (G,)
    assert res.at_bound.dtype == bool and res.beyond.dtype == bool
    # frames: first centre half a window in, spacing = hop
    assert res.t[0] == pytest.approx(0.2, abs=2e-3)
    assert np.allclose(np.diff(res.t), 0.05, atol=1e-3)
    assert res.t[-1] <= s.t[-1] - 0.2 + 2e-3
    # grid: logarithmic from rpm_range[0] to rpm_range[1], fine by default
    assert res.rpm_grid[0] == pytest.approx(1000.0) and res.rpm_grid[-1] == pytest.approx(6000.0)
    rel = np.diff(np.log(res.rpm_grid))
    assert np.ptp(rel) < 1e-9
    assert 5e-4 < np.expm1(rel[0]) < 2e-3  # about 0.1 %
    assert res.rpm_range == (1000.0, 6000.0)
    assert res.penalty > 0 and np.isfinite(res.penalty)
    assert res.bound_fraction == 0.0
    # scores are weighted means of dB levels floored at 0
    assert res.score.min() >= 0.0


def test_custom_grid_and_refinement():
    # Steady speed halfway between two points of a 1 % grid, the worst case
    # for an unrefined estimate (error = half a step = 0.5 %).
    probe = _runup(3000.0, 3000.0, dur=3.0, noise=1.0)
    g0 = rpm_from_ridge(probe.x, FS, (1000.0, 6000.0), grid=0.01).rpm_grid
    k = int(np.argmin(np.abs(g0 - 3000.0)))
    mid = float(np.sqrt(g0[k] * g0[k + 1]))
    s = _runup(mid, mid, dur=6.0, noise=1.0)
    coarse = _quiet(s.x, FS, (1000.0, 6000.0), grid=0.01, refine=False)
    refined = _quiet(s.x, FS, (1000.0, 6000.0), grid=0.01, refine=True)
    step = np.diff(np.log(coarse.rpm_grid))[0]
    assert np.expm1(step) <= 0.0101
    # without refinement the estimate sits on the grid
    assert np.all(np.isin(coarse.rpm, coarse.rpm_grid))
    # refinement moves it by at most half a step and improves the error
    assert np.all(np.abs(np.log(refined.rpm / coarse.rpm)) <= 0.5 * step + 1e-12)
    e_c, e_r = _err(coarse, s)["median_rel_error"], _err(refined, s)["median_rel_error"]
    assert e_c > 0.003  # about half a step
    assert e_r < 0.3 * e_c
    assert e_r < 0.002


def test_penalty_zero_and_large():
    s = _runup(1500.0, 3000.0, dur=6.0, noise=6.0)

    def run(**kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # unconstrained paths flag by chance
            return rpm_from_ridge(s.x, FS, (1000.0, 6000.0), refine=False, **kw)

    free = run(penalty=0.0)
    assert np.array_equal(free.rpm, free.rpm_grid[np.argmax(free.score, axis=1)])
    stiff = run(penalty=1e6)
    assert np.ptp(stiff.rpm) == 0.0
    assert stiff.rpm[0] == stiff.rpm_grid[np.argmax(stiff.score.sum(axis=0))]
    # the default penalty is in between: smoother than independent frames
    default = run()
    assert 0.0 < default.penalty < 100.0

    def jump(r):
        return np.mean(np.abs(np.diff(np.log(r.rpm))))

    assert jump(default) < jump(free)
    assert free.penalty == 0.0 and stiff.penalty == 1e6


# --------------------------------------------------------------------------
# the dynamic programme
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 17, 200])
@pytest.mark.parametrize("c", [0.0, 0.1, 1.0, 5.0])
def test_l1_max_sum_matches_brute_force(n, c):
    rng = np.random.default_rng(n * 10 + int(c * 10))
    a = 3.0 * rng.standard_normal(n)
    m, src = _l1_max_sum(a, c)
    g = np.arange(n)
    full = a[None, :] - c * np.abs(g[:, None] - g[None, :])
    assert np.allclose(m, full.max(axis=1))
    assert np.allclose(a[src] - c * np.abs(g - src), m)  # src realises the maximum


def test_l1_max_sum_is_linear_time():
    # O(G^2) would need 4e10 operations here.
    n = 200_000
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n)
    t0 = time.perf_counter()
    m, _ = _l1_max_sum(a, 0.01)
    assert time.perf_counter() - t0 < 2.0
    for g in rng.integers(0, n, size=20):
        assert m[g] == pytest.approx(np.max(a - 0.01 * np.abs(np.arange(n) - g)))


@pytest.mark.parametrize("T, G, c", [(5, 7, 0.3), (12, 30, 1.0), (30, 40, 0.0), (40, 25, 4.0)])
def test_viterbi_matches_quadratic_dynamic_programme(T, G, c):
    rng = np.random.default_rng(T * G)
    S = rng.standard_normal((T, G))
    path = _viterbi(S, c)
    g = np.arange(G)
    cost = c * np.abs(g[:, None] - g[None, :])
    d = S[0].copy()
    back = np.zeros((T, G), int)
    for t in range(1, T):
        M = d[None, :] - cost
        back[t] = M.argmax(axis=1)
        d = S[t] + M.max(axis=1)
    ref = np.empty(T, int)
    ref[-1] = d.argmax()
    for t in range(T - 1, 0, -1):
        ref[t - 1] = back[t, ref[t]]

    def total(p):
        return S[np.arange(T), p].sum() - c * np.abs(np.diff(p)).sum()

    assert total(path) == pytest.approx(total(ref))


# --------------------------------------------------------------------------
# degenerate input and validation
# --------------------------------------------------------------------------


def test_silence_is_reported_not_invented():
    with pytest.warns(RuntimeWarning):
        res = rpm_from_ridge(np.zeros(int(3 * FS)), FS, (1000.0, 6000.0))
    assert np.all(res.confidence == 0.0)
    assert res.bound_fraction == 1.0


def test_signal_of_exactly_one_window():
    x = np.random.default_rng(0).standard_normal(int(0.4 * FS))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = rpm_from_ridge(x, FS, (1000.0, 6000.0), t_out=[0.0, 1.0])
    assert res.t.size == 1 and res.rpm_out.shape == (2,)
    assert res.rpm_out[0] == res.rpm_out[1] == res.rpm[0]


_GOOD = dict(fs=FS, rpm_range=(1000.0, 6000.0))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(fs=0.0), "fs"),
        (dict(fs=-8000.0), "fs"),
        (dict(fs=np.nan), "fs"),
        (dict(rpm_range=(2000.0, 1000.0)), "rpm_range"),
        (dict(rpm_range=(0.0, 1000.0)), "rpm_range"),
        (dict(rpm_range=(-5.0, 1000.0)), "rpm_range"),
        (dict(rpm_range=(1000.0, np.inf)), "rpm_range"),
        (dict(rpm_range=1000.0), "rpm_range"),
        (dict(rpm_range=(1000.0, 2000.0, 3000.0)), "rpm_range"),
        (dict(orders=()), "orders"),
        (dict(orders=(1, 0)), "orders"),
        (dict(orders=(1, -2)), "orders"),
        (dict(orders=(1, np.nan)), "orders"),
        (dict(weights=(1, 1, 1)), "weights"),
        (dict(weights=(1, 1, 1, -1)), "weights"),
        (dict(weights=(0, 0, 0, 0)), "weights"),
        (dict(weights=(1, 1, 1, np.nan)), "weights"),
        (dict(window=0.0), "window"),
        (dict(window=-1.0), "window"),
        (dict(window=0.0005), "window"),
        (dict(window=60.0), "shorter than one window"),
        (dict(hop=0.0), "hop"),
        (dict(whiten=0.0), "whiten"),
        (dict(whiten=-3.0), "whiten"),
        (dict(penalty=-1.0), "penalty"),
        (dict(penalty=np.nan), "penalty"),
        (dict(grid=0.0), "grid"),
        (dict(grid=0.6), "grid"),
        (dict(grid=np.nan), "grid"),
        (dict(grid=1e-7), "too large"),
        (dict(t_out=np.zeros((2, 2))), "t_out"),
        (dict(t_out=[0.0, np.nan]), "t_out"),
        (dict(rpm_range=(1000.0, 70000.0)), "fs/2"),
    ],
)
def test_validation_errors(kwargs, match):
    x = np.random.default_rng(1).standard_normal(int(2 * FS))
    args = {**_GOOD, **kwargs}
    with pytest.raises(ValueError, match=match):
        rpm_from_ridge(x, args.pop("fs"), args.pop("rpm_range"), **args)


@pytest.mark.parametrize(
    "x",
    [np.zeros((4, 4, 4)), np.array([1.0]), np.array([np.nan] * 5000), np.array(3.0)],
    ids=["3-D", "one sample", "nan", "scalar"],
)
def test_bad_signal_raises(x):
    with pytest.raises(ValueError):
        rpm_from_ridge(x, FS, (1000.0, 6000.0))


# --------------------------------------------------------------------------
# whitening width: whiten="auto" (default), explicit widths, diagnostics
# --------------------------------------------------------------------------

# Three-cylinder-like order set with weak sub-orders between the comb lines: the
# lines are dense, which is what makes a too-narrow running median erase them.
HALF_ORDERS = ((0.5, 0.15), (1.0, 0.35), (1.5, 1.0), (3.0, 0.5), (4.5, 0.3), (6.0, 0.2))
HALF_COMB = (1.5, 3.0, 4.5, 6.0)
HALF_RANGE = (600.0, 6000.0)


def _half_runup(dur, *, down=False, lo=900.0, hi=4500.0, hold=1.2, seed=7):
    """Hold at one end for ``hold`` s, then a linear ramp to the other until ``dur``.

    The slew is ``(hi - lo) / (dur - hold)``: 750 rpm/s for 6 s, 333 for 12 s,
    158 for 24 s, 1500 for 3.6 s.
    """
    t = np.arange(round(dur * FS)) / FS
    a, b = (hi, lo) if down else (lo, hi)
    rpm = rpm_profiles(t, "linear", start=a, stop=b, t_start=hold, t_stop=dur)
    specs = [{"order": o, "amplitude": amp} for o, amp in HALF_ORDERS]
    return synth_runup(FS, rpm, specs, noise_rms=0.05, seed=seed)


@pytest.mark.parametrize(
    "dur", [6.0, 12.0, 24.0, 3.6], ids=["750rpm/s", "333rpm/s", "158rpm/s", "1500rpm/s"]
)
def test_default_whitening_follows_ordinary_runups(dur):
    # The defect: with the former fixed 15 Hz median the top comb order (6 x speed)
    # sweeps 30 Hz (750 rpm/s) or 13 Hz (333 rpm/s) within a window, wider than
    # the median, and the lines were erased: 66 % median error, only a misleading
    # search-range warning. All defaults now; no warning of any kind.
    s = _half_runup(dur)
    res = _quiet(s.x, FS, HALF_RANGE, orders=HALF_COMB)
    e = _err(res, s)
    assert e["within"][0.03] >= 0.9  # ROADMAP 5.4
    assert e["within"][0.03] >= 0.95  # measured: 0.97 (1500 rpm/s), 0.99 (750), 1.0 (333, 158)
    assert e["median_rel_error"] < 0.005  # measured: 0.00014 (158 rpm/s) to 0.0034 (1500 rpm/s)
    assert res.bound_fraction == 0.0 and not res.beyond.any()
    assert res.whiten_violated.mean() < 0.05  # at most a frame or two where a ramp ends


@pytest.mark.parametrize("dur", [6.0, 12.0], ids=["750rpm/s", "333rpm/s"])
def test_default_whitening_follows_rundowns(dur):
    s = _half_runup(dur, down=True)
    res = _quiet(s.x, FS, HALF_RANGE, orders=HALF_COMB)
    e = _err(res, s)
    assert e["within"][0.03] >= 0.95  # measured: 0.99 (750 rpm/s), 1.0 (333)
    assert res.rpm[0] > 4000.0 and res.rpm[-1] < 1200.0  # it followed the decay
    assert res.bound_fraction == 0.0


def test_default_whitening_at_steady_speed_and_comb_to_order_ten():
    t = np.arange(round(5.0 * FS)) / FS
    specs = [{"order": o, "amplitude": a} for o, a in HALF_ORDERS]
    s = synth_runup(FS, np.full(t.size, 2500.0), specs, noise_rms=0.05, seed=7)
    res = _quiet(s.x, FS, HALF_RANGE, orders=HALF_COMB)
    assert _err(res, s)["within"][0.03] >= 0.98  # measured: 1.0
    # A comb of ten integer orders at 1500 rpm/s (order 10 sweeps 100 Hz per window).
    comb = tuple(range(1, 11))
    t = np.arange(round(2.5 * FS)) / FS
    t_stop = 1.2 + 1900.0 / 1500.0
    rpm = rpm_profiles(t, "linear", start=900.0, stop=2800.0, t_start=1.2, t_stop=t_stop)
    specs = [{"order": k, "amplitude": k**-0.5} for k in comb]
    s = synth_runup(FS, rpm, specs, noise_rms=0.05, seed=7)
    res = _quiet(s.x, FS, (600.0, 3000.0), orders=comb)
    assert _err(res, s)["within"][0.03] >= 0.95  # measured: 1.0


def test_auto_width_follows_the_documented_rule():
    # At the former default window of 0.4 s (window="auto" would shorten it).
    s = _half_runup(6.0)
    res = _quiet(s.x, FS, HALF_RANGE, orders=HALF_COMB, window=0.4)
    # idle, no sweep: the static 6 / window = 15 Hz (the sweep estimate reaches
    # about 0.5 s ahead of the start of the ramp at 1.2 s)
    hold = res.t < 0.7
    assert hold.sum() > 8 and np.allclose(res.whiten_width[hold], 6.0 / 0.4)
    # in the ramp the median is at least the line extent 4 / window + sweep of
    # order 6 (30 Hz sweep at 750 rpm/s and 900 rpm; 4 / window = 10 Hz)
    ramp = (res.t > 2.0) & (res.t < 5.0)
    assert np.all(res.whiten_width[ramp] >= res.line_width[ramp])
    truth_extent = 4.0 / 0.4 + 6.0 * 750.0 / 60.0 * 0.4  # 40 Hz, the true value
    extent = np.median(res.line_width[ramp])
    assert extent == pytest.approx(truth_extent, rel=0.03)  # measured: 40.0
    assert np.median(res.whiten_width[ramp]) > truth_extent
    # an explicit number is reported as given, None as infinite
    assert np.all(
        _quiet(s.x, FS, HALF_RANGE, orders=HALF_COMB, whiten=60.0, window=0.4).whiten_width == 60.0
    )
    none = _quiet(s.x, FS, HALF_RANGE, orders=HALF_COMB, whiten=None, window=0.4)
    assert np.all(np.isinf(none.whiten_width)) and not none.whiten_violated.any()
    assert _err(none, s)["within"][0.03] >= 0.98  # the frame median is robust against any sweep


def test_default_whitening_keeps_removing_a_noise_hump():
    # As test_whitening_removes_the_bias_of_a_noise_hump, with the default width.
    from scipy import signal

    sos = signal.butter(4, [70.0, 130.0], btype="bandpass", fs=FS, output="sos")
    w_auto, w_off = [], []
    for seed in (1, 2, 3):
        s = _runup(1500.0, 3000.0, noise=4.0, seed=seed)
        hump = signal.sosfiltfilt(sos, np.random.default_rng(100 + seed).standard_normal(s.x.size))
        x = s.x + 40.0 * hump / hump.std()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto = rpm_from_ridge(x, FS, (1000.0, 6000.0))
            off = rpm_from_ridge(x, FS, (1000.0, 6000.0), whiten=None)
        w_auto.append(_err(auto, s)["within"][0.03])
        w_off.append(_err(off, s)["within"][0.03])
    assert np.mean(w_auto) >= 0.8  # measured: 0.88
    assert np.mean(w_auto) > np.mean(w_off) + 0.15  # measured: 0.88 vs 0.61


def _record(*args, **kwargs):
    """Call the estimator; return its result and the list of warnings it issued."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = rpm_from_ridge(*args, **kwargs)
    return res, rec


def test_too_narrow_explicit_whiten_warns_and_lowers_confidence():
    # The former default on the 6 s run: the median (15 Hz) is narrower than the
    # 40 Hz a line occupies. The result is wrong; it must not be silent.
    s = _half_runup(6.0)
    good = _quiet(s.x, FS, HALF_RANGE, orders=HALF_COMB, window=0.4)
    bad, rec = _record(s.x, FS, HALF_RANGE, orders=HALF_COMB, whiten=15.0, window=0.4)
    assert _err(bad, s)["within"][0.03] < 0.5  # measured: 0.19; the warning is right
    assert issubclass(WhiteningWarning, RuntimeWarning)
    specific = [str(w.message) for w in rec if issubclass(w.category, WhiteningWarning)]
    assert len(specific) == 1
    # the specific message names the width, the cause and the remedy
    assert "whitening width (15 Hz)" in specific[0] and "whiten='auto'" in specific[0]
    # frames are marked, and their confidence is derated by (width / (0.8 line))^2
    v = bad.whiten_violated
    assert v.mean() > 0.5  # measured: 0.84
    assert np.all(bad.whiten_width[v] < 0.8 * bad.line_width[v])
    assert np.all(bad.confidence[v] <= (bad.whiten_width[v] / (0.8 * bad.line_width[v])) ** 2 + 1e-12)
    assert np.median(bad.confidence) < 0.5 * np.median(good.confidence)  # measured: 0.075 vs 0.24
    assert np.all((bad.confidence >= 0) & (bad.confidence <= 1))
    # the search-range warning, if it also fires (19.5 % of the frames here),
    # says that the whitening can cause it
    bound = [str(w.message) for w in rec if "search range" in str(w.message)]
    assert bound and all("whitening width is too small" in m for m in bound)


def test_a_width_slightly_too_narrow_is_not_silent():
    # 30 Hz against 40 Hz (0.75): the final path is wrong here, and so is its own
    # slope (it shows 19 Hz); the sweep taken from the first pass, which does not
    # depend on the width, is what catches it.
    s = _half_runup(6.0)
    res, rec = _record(s.x, FS, HALF_RANGE, orders=HALF_COMB, whiten=30.0, window=0.4)
    assert _err(res, s)["within"][0.03] < 0.5  # measured: 0.19
    assert any(issubclass(w.category, WhiteningWarning) for w in rec)
    assert np.median(res.line_width[(res.t > 2.0) & (res.t < 5.0)]) > 35.0  # measured: 40


@pytest.mark.parametrize("whiten, dur", [(45.0, 6.0), (60.0, 6.0), (30.0, 12.0)])
def test_a_sufficient_explicit_whiten_is_quiet_and_right(whiten, dur):
    # 45 and 60 Hz against 40 Hz; 30 Hz against 23 Hz: at or above the line extent.
    s = _half_runup(dur)
    res = _quiet(s.x, FS, HALF_RANGE, orders=HALF_COMB, whiten=whiten)
    assert _err(res, s)["within"][0.03] >= 0.95  # measured: 1.0
    assert not res.whiten_violated.any()


def test_a_run_beyond_the_sweep_limit_is_not_silent():
    # 2000 rpm/s at a 0.4 s window: order 6 sweeps 80 Hz per window and the
    # lines blur into each other; no width helps and the track falls to a third
    # of the speed in the ramp (0.40 within 3 %). It must not do so silently.
    s = _half_runup(3.0)
    res, rec = _record(s.x, FS, HALF_RANGE, orders=HALF_COMB, window=0.4)
    assert _err(res, s)["within"][0.03] < 0.9  # measured: 0.40 (all frames)
    assert any(issubclass(w.category, RidgeQualityWarning) for w in rec)
    assert not res.quality["reliable"]
    assert res.quality["low_confidence_fraction"] > 0.4  # measured: 0.47
    # A shorter window cures it (0.25 s: 1.0), and window="auto" finds one.
    short, rec = _record(s.x, FS, HALF_RANGE, orders=HALF_COMB, window=0.25)
    assert _err(short, s)["within"][0.03] >= 0.95  # measured: 1.0
    auto = _quiet(s.x, FS, HALF_RANGE, orders=HALF_COMB)
    assert _err(auto, s)["within"][0.03] >= 0.95  # measured: 1.0
    assert 0.15 < auto.window < 0.3 and auto.quality["reliable"]  # measured: 0.22 s


def test_warnings_are_separate_when_the_speed_really_leaves_the_range():
    # Truth outside the range with an adequate width: only the search-range warning.
    s = _runup(1500.0, 3000.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = rpm_from_ridge(s.x, FS, (1000.0, 2000.0))
    assert any("search range" in str(w.message) for w in rec)
    assert not any(issubclass(w.category, WhiteningWarning) for w in rec)
    assert not res.whiten_violated.any()


def test_whiten_argument_forms():
    s = _half_runup(6.0)
    default = _quiet(s.x, FS, HALF_RANGE, orders=HALF_COMB)
    for form in ("auto", "Auto", " AUTO "):
        same = _quiet(s.x, FS, HALF_RANGE, orders=HALF_COMB, whiten=form)
        assert np.array_equal(same.rpm, default.rpm)
    for bad in ("wide", "", "15"):
        with pytest.raises(ValueError, match="whiten"):
            rpm_from_ridge(s.x, FS, HALF_RANGE, orders=HALF_COMB, whiten=bad)


def test_path_rate_is_the_relative_slope():
    hop = 0.05
    t = np.arange(60) * hop
    rpm = 1000.0 + 200.0 * t  # a ramp of 200 rpm/s: exact for a linear fit, ends included
    assert np.allclose(_path_rate(rpm, hop, 0.4), 200.0 / rpm, rtol=1e-9)
    assert np.allclose(_path_rate(np.full(30, 2000.0), hop, 0.4), 0.0, atol=1e-12)
    # one wrong frame (an outlier of the path) does not make a sweep
    spiked = np.full(60, 2000.0)
    spiked[30] = 3500.0
    assert _path_rate(spiked, hop, 0.4).max() < 1e-9
    assert np.array_equal(_path_rate(np.array([1000.0, 1100.0]), hop, 0.4), [0.0, 0.0])
    # the spread envelope is never below the plain rate and widens a step
    step = np.where(np.arange(60) < 30, 1000.0, 1500.0)
    assert np.all(_path_rate(step, hop, 0.4, spread=True) >= _path_rate(step, hop, 0.4))


def test_auto_slope_rounds_the_relative_rate_up_on_a_ladder():
    hop, window = 0.05, 0.4
    t = np.arange(80) * hop
    rpm = 1000.0 * np.exp(0.5 * t)  # constant relative rate 0.5 /s
    q = _auto_slope(rpm, window, hop)
    want = 0.5 * window  # _AUTO_FACTOR = 1: q = rate * window
    assert np.all(q >= want * (1 - 1e-2)) and np.all(q <= want * 2 ** (1 / 3) * (1 + 1e-2))
    assert np.all(_auto_slope(np.full(80, 2000.0), window, hop) == 0.0)  # steady: static width


def test_warped_median_widens_with_frequency():
    # A flat-topped line of 14 Hz in a flat floor of 1, at 30 Hz and at 300 Hz.
    # The median erases a line wider than half of its width (it ignores up to half
    # of the samples): width(30 Hz) = 16 Hz erases it, width(300 Hz) = 70 Hz does not.
    df, nb = 0.625, 700
    w0, w1, q = 15.0, 10.0, 0.2  # width(f) = max(15, 10 + 0.2 f)
    P = np.ones((1, nb))
    for f0 in (30.0, 300.0):
        k = round(f0 / df)
        P[0, k - 11 : k + 12] = 100.0  # 23 bins, 14.4 Hz
    floor = _running_median_warped(P, df, w0, w1, q)[0]
    assert floor[round(30.0 / df)] > 50.0  # erased: 16 Hz wide median
    assert floor[round(300.0 / df)] < 2.0  # kept: 70 Hz wide median
    # a flat floor stays flat, and q -> 0 gives a constant width (the line at
    # 300 Hz is then erased as well)
    flat = _running_median_warped(np.full((2, nb), 3.0), df, w0, w1, q)
    assert np.allclose(flat, 3.0)
    narrow = _running_median_warped(P, df, w0, w1, 1e-9)[0]
    assert narrow[round(300.0 / df)] > 50.0


# --------------------------------------------------------------------------
# interior-noise-like signals: four-stroke four-cylinder-like orders (2, 4, 6,
# 8; no distinct odd or half orders) through a low-frequency boom, plus
# speed-dependent road noise falling steeply with frequency
# --------------------------------------------------------------------------

FOUR_CYL = ((2.0, 0.0), (4.0, -15.0), (6.0, -24.0), (8.0, -30.0))  # order, dB re order 2
BOOM = dict(freq=45.0, q=3.0, gain=4.0)  # modal boom, linear peak gain
EVEN_COMB = (2, 4, 6)  # the comb for a four-stroke four-cylinder engine
CAR_RANGE = (600.0, 7000.0)


def _interior(rpm, *, onr_db=5.0, tilt=-8.5, seed=0):
    """Four-cylinder-like orders and road noise; truth in the returned SynthSignal.

    The orders (peak amplitudes from FOUR_CYL) pass the 45 Hz boom (Q = 3). The
    noise has a PSD tilt of ``tilt`` dB/octave, grows by 6 dB per doubling of a
    road speed of rpm / 40 km/h, passes the same boom and is scaled to an
    orders-to-noise RMS ratio of ``onr_db``.
    """
    from scipy import signal as sps

    from rota._channel import speed_noise
    from rota.synth import _resonance_sos

    specs = [dict(order=o, amplitude=10 ** (a / 20)) for o, a in FOUR_CYL]
    s = synth_runup(FS, rpm, specs, resonances=[BOOM], seed=seed)
    noise, _ = speed_noise(
        FS, rpm.size, rpm / 40.0, level_db_at_ref=0.0, ref_speed_kmh=60.0,
        level_slope_db_per_doubling=6.0, tilt_db_per_octave=tilt, highpass=4.0, grain=0.25,
        seed=seed + 100,
    )
    noise = sps.sosfilt(_resonance_sos([BOOM], FS), noise)
    noise *= np.sqrt(np.mean(s.x**2) / np.mean(noise**2)) / 10 ** (onr_db / 20)
    return s, s.x + noise


def _ramp(r0, r1, rate, hold=0.5, tail=0.3):
    """Hold ``r0`` for ``hold`` s, ramp to ``r1`` at ``rate`` rpm/s, hold ``tail`` s."""
    t_ramp = abs(r1 - r0) / rate
    t = np.arange(round((hold + t_ramp + tail) * FS)) / FS
    return rpm_profiles(t, "linear", start=r0, stop=r1, t_start=hold, t_stop=hold + t_ramp)


def test_interior_fast_full_load_runup_needs_and_gets_a_short_window():
    # 850 rpm/s from 1200 rpm under strong low-frequency road noise. The order 6
    # sweeps 85 Hz/s, so a 0.4 s window smears it over 34 Hz: the former default
    # tracked it with a confidence near 0 and false search-range and whitening
    # warnings. window="auto" picks the window of the least line extent,
    # 2 / sqrt(85 Hz/s) = 0.22 s.
    s, x = _interior(_ramp(1200.0, 6000.0, 850.0), onr_db=-10.0)
    res = _quiet(x, FS, CAR_RANGE, orders=EVEN_COMB)
    e = _err(res, s)
    assert e["within"][0.03] >= 0.9  # measured: 1.0
    assert 0.18 < res.window < 0.26  # measured: 0.217 s
    assert res.quality["reliable"] and res.quality["median_confidence"] > 0.5  # measured: 1.0
    old, rec = _record(x, FS, CAR_RANGE, orders=EVEN_COMB, window=0.4)
    assert old.quality["median_confidence"] < 0.2  # measured: 0.03 (99 % within 3 %)
    assert any(issubclass(w.category, RidgeQualityWarning) for w in rec)


@pytest.mark.parametrize(
    "r0, r1, rate",
    [(1200.0, 6000.0, 500.0), (1500.0, 3000.0, 130.0), (3500.0, 1700.0, 180.0), (2200.0, 2200.0, 1.0)],
    ids=["runup500", "partload130", "coast180", "cruise"],
)
def test_interior_manoeuvres_are_tracked_quietly_with_the_even_comb(r0, r1, rate):
    # Correct tracks of a clean signal must not warn and must read a high
    # confidence (the former confidence read 0.1 to 0.45 here).
    rpm = _ramp(r0, r1, rate) if r0 != r1 else np.full(round(6.0 * FS), r0)
    s, x = _interior(rpm, seed=3)
    res = _quiet(x, FS, CAR_RANGE, orders=EVEN_COMB)
    assert _err(res, s)["within"][0.03] >= 0.9  # measured: 1.0
    q = res.quality
    assert q["reliable"] and q["median_confidence"] > 0.7  # measured: 0.85 to 1.0
    assert q["bound_fraction"] == 0.0 and q["jump_fraction"] < 0.05
    assert np.isfinite(q["octave_ratio"]).all() and not q["octave_ambiguous"]


def test_interior_default_comb_reads_twice_the_speed_of_a_four_cylinder():
    # Orders 2, 4, 6, 8 at speed r are the orders 1, 2, 3, 4 at speed 2r: the
    # spectra are identical and no test on the data can tell them apart (the
    # octave ratios read alike). The comb must come from the machine (Notes:
    # orders=(2, 4, 6) for four cylinders). Where twice the speed leaves the
    # range, the guard bands report it.
    s, x = _interior(np.full(round(6.0 * FS), 2200.0), seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = rpm_from_ridge(x, FS, CAR_RANGE)
    assert np.median(default.rpm / _truth(default, s)) == pytest.approx(2.0, abs=0.02)
    s, x = _interior(_ramp(1200.0, 6000.0, 500.0), seed=3)
    with pytest.warns(RuntimeWarning, match="search range"):
        fast = rpm_from_ridge(x, FS, CAR_RANGE)
    over = 2.0 * _truth(fast, s) > CAR_RANGE[1] * 1.03
    assert over.sum() > 20 and fast.at_bound[over].mean() >= 0.9


def test_interior_weak_cruise_and_noise_only_are_flagged():
    # Weak orders under strong low-frequency noise cannot be tracked; that must
    # read clearly: median confidence near 0 and a RidgeQualityWarning.
    s, x = _interior(np.full(round(6.0 * FS), 2000.0), onr_db=-30.0, tilt=-9.0)
    res, rec = _record(x, FS, CAR_RANGE, orders=EVEN_COMB)
    assert any(issubclass(w.category, RidgeQualityWarning) for w in rec)
    assert not res.quality["reliable"] and res.quality["median_confidence"] < 0.05  # measured: 0
    noise, rec = _record(x - s.x, FS, CAR_RANGE, orders=EVEN_COMB)
    assert any(issubclass(w.category, RidgeQualityWarning) for w in rec)
    assert noise.quality["median_confidence"] < 0.05  # measured: 0


def test_single_line_comb_locked_on_a_rational_neighbour_has_low_confidence():
    # A line of order 6 about 8 dB below a noise floor falling at 9 dB/octave
    # (RMS), a line at 7/6 of it (order 7, -2 dB) and a fixed tone 10 dB above
    # order 6 that it crosses. With the comb (6,) and moving=True (which removes
    # the tone) the track falls onto the order 7 line, 7/6 of the speed; the
    # rival at the true speed is counted, so the confidence of those frames is
    # low and the track is reported as unreliable.
    from rota._channel import speed_noise

    t = np.arange(round(8.0 * FS)) / FS
    rpm = rpm_profiles(t, "linear", start=2000.0, stop=8000.0, t_start=0.5, t_stop=7.5)
    s = synth_runup(FS, rpm, [dict(order=6.0, amplitude=1.0), dict(order=7.0, amplitude=0.8)],
                    tones=[dict(freq=500.0, amplitude=3.16)], seed=0)
    noise, _ = speed_noise(FS, t.size, 50.0, level_db_at_ref=0.0, ref_speed_kmh=50.0,
                           level_slope_db_per_doubling=0.0, tilt_db_per_octave=-9.0,
                           highpass=10.0, grain=0.1, seed=7)
    noise *= np.sqrt(0.5 / np.mean(noise**2)) * 10 ** (8.0 / 20)
    res, rec = _record(s.x + noise, FS, (1500.0, 10000.0), orders=(6,), moving=True)
    ratio = res.rpm / _truth(res, s)
    wrong = np.abs(ratio - 1.0) > 0.03
    assert np.mean(np.abs(ratio - 7 / 6) < 0.03) > 0.5  # measured: 0.92
    assert np.median(res.confidence[wrong]) < 0.2  # measured: 0.08
    assert not res.quality["reliable"]
    assert any(issubclass(w.category, RidgeQualityWarning) for w in rec)
