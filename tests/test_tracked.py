"""Tests of the phase-locked harmonic extractor against analytic truth (rota.synth)."""

import json
import warnings

import numpy as np
import pytest

from rota import _tracked
from rota._components import OrderComponents
from rota._tracked import tracked_harmonics
from rota.speed import rpm_to_phase
from rota.synth import rpm_profiles, synth_runup

FS = 16000.0


def _truth(sig, shafts=None):
    """Complex peak envelope of every order relative to its carrier, from the synth truth."""
    rev = sig.rev if sig.rev.ndim == 2 else sig.rev[:, None]
    sh = np.zeros(sig.orders.size, dtype=int) if shafts is None else np.asarray(shafts)
    carrier = 2.0 * np.pi * sig.orders * rev[:, sh]
    return sig.order_amplitude * np.exp(1j * (sig.order_phase - carrier))


def _db(est, ref):
    return 20.0 * np.log10(np.abs(est) / np.abs(ref))


def _rms(z):
    return np.sqrt(np.mean(np.abs(z) ** 2))


def _inside(rev, margin):
    """Samples at least `margin` revolutions from both ends of the record."""
    return (rev >= margin) & (rev <= rev[-1] - margin)


def _runup(dur=4.0, start=800.0, stop=4000.0, fs=FS):
    t = np.arange(round(dur * fs)) / fs
    return rpm_profiles(t, "linear", start=start, stop=stop)


def _pair(rpm_value, revs, order_a=2.0, order_b=2.5):
    """Constant speed, two equal orders, `revs` revolutions: signal, rpm, truth."""
    n = round(revs * 60.0 / rpm_value * FS)
    rpm = np.full(n, float(rpm_value))
    sig = synth_runup(
        FS,
        rpm,
        [
            {"order": order_a, "amplitude": 1.0, "phase": 0.3},
            {"order": order_b, "amplitude": 1.0, "phase": 1.0},
        ],
        seed=3,
    )
    return sig, rpm, _truth(sig)


# --------------------------------------------------------------------------
# accuracy on a run-up
# --------------------------------------------------------------------------


def test_constant_amplitude_orders_in_a_noisy_run_up_match_the_truth_inside_the_edges():
    rpm = _runup()
    sig = synth_runup(
        FS,
        rpm,
        [
            {"order": 2, "amplitude": 1.0, "phase": 0.3},
            {"order": 3.5, "amplitude": 0.5, "phase": -1.2},
        ],
        noise_rms=0.05,
        seed=3,
    )
    res = tracked_harmonics(sig.x, FS, rpm, sig.orders)
    inside = _inside(sig.rev, res.params["edge_revs"])
    assert inside.sum() > 0.8 * inside.size  # the edge region is a small part of the record
    truth = _truth(sig)
    for k in range(2):
        assert np.max(np.abs(_db(res.envelopes[inside, k], truth[inside, k]))) < 0.2  # dB
        corr = np.corrcoef(res.waveforms[inside, k], sig.order_stems[inside, k])[0, 1]
        assert corr >= 0.99
    corr = np.corrcoef(res.total[inside], sig.stems["orders"][inside])[0, 1]
    assert corr >= 0.99


def test_noiseless_envelope_amplitude_and_phase_equal_the_truth():
    rpm = _runup()
    sig = synth_runup(
        FS,
        rpm,
        [
            {"order": 2, "amplitude": 1.0, "phase": 0.3},
            {"order": 6.5, "amplitude": 0.4, "phase": 2.5},
        ],
        seed=1,
    )
    res = tracked_harmonics(sig.x, FS, rpm, sig.orders)
    inside = _inside(sig.rev, res.params["edge_revs"])
    truth = _truth(sig)
    for k in range(2):
        assert np.max(np.abs(_db(res.envelopes[inside, k], truth[inside, k]))) < 1e-3  # dB
        phase = np.angle(res.envelopes[inside, k] * np.conj(truth[inside, k]))
        assert np.max(np.abs(phase)) < 1e-3  # rad
    # the documented convention: waveform = Re{envelope * exp(1j * 2 pi order rev)}
    carrier = np.exp(2j * np.pi * sig.orders * sig.rev[:, None])
    assert np.allclose(res.waveforms, (res.envelopes * carrier).real, rtol=0, atol=1e-12)
    # and it is the peak envelope: a unit-amplitude cosine gives |envelope| = 1
    assert np.max(np.abs(res.envelopes[inside, 0])) == pytest.approx(1.0, abs=1e-3)


def test_amplitude_law_of_rpm_is_tracked():
    rpm = _runup()
    order = {
        "order": 3,
        "amplitude": 1.0,
        "phase": 0.7,
        "gain_rpm": [800, 4000],
        "gain": [1.0, 4.0],
    }
    sig = synth_runup(FS, rpm, [order], noise_rms=0.05, seed=2)
    res = tracked_harmonics(sig.x, FS, rpm, sig.orders)
    inside = _inside(sig.rev, res.params["edge_revs"])
    truth = _truth(sig)
    amp = np.abs(truth[inside, 0])
    assert _db(amp.max(), amp.min()) > 9.0  # the law spans a factor of about 3.5 (11 dB) inside
    assert np.max(np.abs(_db(res.envelopes[inside, 0], truth[inside, 0]))) < 0.5  # dB


def test_a_stationary_order_keeps_its_amplitude_even_at_the_edges():
    rpm = np.full(int(4 * FS), 1500.0)
    sig = synth_runup(FS, rpm, [{"order": 3, "amplitude": 1.0, "phase": 0.3}], seed=3)
    res = tracked_harmonics(sig.x, FS, rpm, [3])
    err = _db(res.envelopes[:, 0], _truth(sig)[:, 0])
    assert np.all(np.isfinite(res.envelopes))
    assert np.max(np.abs(err)) < 0.2  # dB, including the first and last sample
    assert np.max(np.abs(err[_inside(sig.rev, res.params["edge_revs"])])) < 1e-6


def test_run_up_from_standstill_and_a_stop_in_the_middle_stay_finite_and_accurate():
    t = np.arange(int(6 * FS)) / FS
    rpm = rpm_profiles(
        t, "piecewise", knots=([0, 2, 2.2, 3.0, 3.2, 6], [0, 3000, 0, 0, 3000, 3000])
    )
    sig = synth_runup(FS, rpm, [{"order": 3, "amplitude": 1.0, "phase": 0.3}], seed=3)
    res = tracked_harmonics(sig.x, FS, rpm, [3])
    assert np.all(np.isfinite(res.envelopes)) and np.all(np.isfinite(res.waveforms))
    err = _db(res.envelopes[:, 0], _truth(sig)[:, 0])
    assert np.max(np.abs(err[sig.rev > res.params["edge_revs"]])) < 0.1  # dB
    still = rpm == 0.0
    assert still.sum() > 0.5 * FS
    assert np.max(np.abs(err[still])) < 0.1  # the angle does not advance, nor does the envelope


# --------------------------------------------------------------------------
# resolution in angle (ROADMAP section 6, items 20 and 21)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["idle", "run-up"])
def test_orders_half_an_order_apart_are_separated_with_an_eight_revolution_window(profile):
    if profile == "idle":
        sig, rpm, truth = _pair(800.0, 80)
    else:
        rpm = _runup(dur=5.0)
        sig = synth_runup(
            FS,
            rpm,
            [
                {"order": 2, "amplitude": 1.0, "phase": 0.3},
                {"order": 2.5, "amplitude": 1.0, "phase": 1.0},
            ],
            seed=3,
        )
        truth = _truth(sig)
    res = tracked_harmonics(sig.x, FS, rpm, sig.orders, window_revs=8.0)
    inside = _inside(sig.rev, res.params["edge_revs"])
    for k in range(2):
        assert np.max(np.abs(_db(res.envelopes[inside, k], truth[inside, k]))) < 0.5  # dB


def test_a_two_revolution_window_cannot_separate_orders_half_an_order_apart():
    sig, rpm, truth = _pair(800.0, 80)
    errors = {}
    for w in (8.0, 2.0):
        res = tracked_harmonics(sig.x, FS, rpm, sig.orders, window_revs=w)
        inside = _inside(sig.rev, w)
        errors[w] = max(np.max(np.abs(_db(res.envelopes[inside, k], truth[inside, k]))) for k in (0, 1))
    assert errors[8.0] < 0.5  # dB
    assert errors[2.0] > 1.0  # the main lobe (+-1 order) covers the neighbour: beats of +-2.5 dB
    assert errors[2.0] > 100.0 * errors[8.0]


def test_order_resolution_is_the_same_at_idle_and_at_high_speed():
    # 1.5-bin spacing (window 3 revolutions): the leakage is visible, and being a function of
    # the angle only it must not depend on the speed (a fixed-duration window would differ 5x)
    err = {}
    for rpm_value in (800.0, 4000.0):
        sig, rpm, truth = _pair(rpm_value, 60)
        res = tracked_harmonics(sig.x, FS, rpm, sig.orders, window_revs=3.0)
        inside = _inside(sig.rev, 3.0)
        err[rpm_value] = np.max(np.abs(_db(res.envelopes[inside, 0], truth[inside, 0])))
    assert err[800.0] > 0.1  # dB: the neighbour does leak at this window length
    assert err[4000.0] == pytest.approx(err[800.0], rel=0.05)


def test_the_envelope_of_a_pair_of_orders_is_separated_at_every_speed_with_the_default_window():
    for rpm_value in (800.0, 4000.0):
        sig, rpm, truth = _pair(rpm_value, 60)
        res = tracked_harmonics(sig.x, FS, rpm, sig.orders)
        inside = _inside(sig.rev, res.params["edge_revs"])
        for k in range(2):
            assert np.max(np.abs(_db(res.envelopes[inside, k], truth[inside, k]))) < 0.01  # dB


# --------------------------------------------------------------------------
# windows and overlaps
# --------------------------------------------------------------------------


@pytest.mark.parametrize("window", ["hann", "hamming", "blackman", "blackmanharris", "bartlett", "boxcar"])
@pytest.mark.parametrize("overlap", [0.0, 0.5, 0.6, 0.75, 0.9])
def test_every_window_and_overlap_keeps_the_amplitude(window, overlap):
    # the normalised overlap-add keeps a stationary order exact whatever the frame layout
    rpm = _runup(dur=2.5, start=1000.0, stop=3000.0)
    sig = synth_runup(FS, rpm, [{"order": 3, "amplitude": 1.0, "phase": 0.3}], seed=3)
    res = tracked_harmonics(sig.x, FS, rpm, [3], window=window, overlap=overlap)
    inside = _inside(sig.rev, res.params["edge_revs"])
    assert np.max(np.abs(_db(res.envelopes[inside, 0], _truth(sig)[inside, 0]))) < 0.01  # dB


def test_the_blocking_of_the_samples_does_not_change_the_result(monkeypatch):
    rpm = _runup(dur=2.0)
    sig = synth_runup(
        FS,
        rpm,
        [{"order": 2, "amplitude": 1.0, "phase": 0.3}, {"order": 5.5, "amplitude": 0.4, "phase": 1.0}],
        noise_rms=0.05,
        seed=3,
    )
    ref = tracked_harmonics(sig.x, FS, rpm, sig.orders)
    monkeypatch.setattr(_tracked, "_CHUNK", 777)
    small = tracked_harmonics(sig.x, FS, rpm, sig.orders)
    assert np.allclose(small.envelopes, ref.envelopes, rtol=0, atol=1e-12)
    assert np.allclose(small.waveforms, ref.waveforms, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# Nyquist
# --------------------------------------------------------------------------


def test_orders_above_nyquist_are_muted_and_the_others_are_untouched():
    fs = 4000.0
    t = np.arange(int(6 * fs)) / fs
    rpm = rpm_profiles(t, "linear", start=1000.0, stop=6000.0)  # order 24: 400 -> 2400 Hz
    sig = synth_runup(
        fs,
        rpm,
        [{"order": 24, "amplitude": 1.0, "phase": 0.3}, {"order": 4, "amplitude": 0.5, "phase": 1.0}],
        seed=1,
    )
    res = tracked_harmonics(sig.x, fs, rpm, sig.orders)
    muted = sig.order_frequency[:, 0] >= 0.5 * fs
    assert 0.1 * muted.size < muted.sum() < 0.3 * muted.size  # it does cross, near 5000 rpm
    assert np.all(res.envelopes[muted, 0] == 0.0) and np.all(res.waveforms[muted, 0] == 0.0)
    assert np.all(np.abs(res.envelopes[~muted, 0]) > 0.0)
    inside = _inside(sig.rev, res.params["edge_revs"])
    truth = _truth(sig)
    # up to 2 % below Nyquist (the aliased image of the line is outside the main lobe from there)
    ok = inside & (sig.order_frequency[:, 0] < 0.98 * 0.5 * fs)
    assert np.max(np.abs(_db(res.envelopes[ok, 0], truth[ok, 0]))) < 0.2  # dB
    # order 4 sees the step where the truth switches order 24 off, as a broadband click
    assert np.max(np.abs(_db(res.envelopes[inside, 1], truth[inside, 1]))) < 0.05  # dB


def test_samples_above_nyquist_are_left_out_of_the_frame_averages():
    # white-box: a clean tone everywhere, but the speed reference says that the order is
    # above Nyquist in the second half; the frames that straddle the boundary must average
    # the first half only (zero filling would give the fraction of the window that is left)
    n = int(4 * FS)
    rev = rpm_to_phase(np.full(n, 1500.0), FS)
    speed = np.where(np.arange(n) < n // 2, 1500.0, 1.0e6)
    x = np.cos(2.0 * np.pi * 3.0 * rev)
    frames = _tracked._Frames(rev, 8.0, 2.0, "hann")
    c, cov = _tracked._frame_coefficients(x[:, None], rev, speed, FS, frames, np.array([3.0]))
    centre = np.arange(frames.n_frames) * 2.0
    straddle = np.abs(centre - rev[n // 2]) < 3.0
    assert straddle.sum() >= 3
    assert np.all((cov[straddle, 0] > 0.05) & (cov[straddle, 0] < 0.95))
    live = cov[:, 0] > 0.05
    assert np.max(np.abs(np.abs(c[live, 0, 0]) - 1.0)) < 0.02
    assert np.all(c[cov[:, 0] == 0.0] == 0.0)


def test_an_order_that_is_always_above_nyquist_is_zero():
    fs = 4000.0
    rpm = np.linspace(1500.0, 3000.0, int(3 * fs))
    sig = synth_runup(fs, rpm, [{"order": 4, "amplitude": 1.0, "phase": 0.3}], seed=1)
    res = tracked_harmonics(sig.x, fs, rpm, [4, 80])  # 80 * 25 Hz = 2000 Hz at the slowest speed
    assert np.all(np.isfinite(res.envelopes))
    assert np.all(res.envelopes[:, 1] == 0.0) and np.all(res.waveforms[:, 1] == 0.0)


# --------------------------------------------------------------------------
# several shafts, several channels, delay
# --------------------------------------------------------------------------


def test_each_order_follows_the_speed_of_its_own_shaft():
    rpm = _runup(dur=4.0, start=800.0, stop=3000.0)
    both = np.column_stack([rpm, 1.37 * rpm])
    orders = [
        {"order": 2, "amplitude": 1.0, "phase": 0.3, "shaft": 0},
        {"order": 3, "amplitude": 0.7, "phase": 1.0, "shaft": 1},
        {"order": 5.5, "amplitude": 0.4, "phase": 2.0, "shaft": 0},
    ]
    sig = synth_runup(FS, both, orders, noise_rms=0.02, seed=3)
    res = tracked_harmonics(sig.x, FS, both, sig.orders, shafts=[0, 1, 0])
    assert list(res.shafts) == [0, 1, 0] and res.rev.shape == both.shape
    assert np.allclose(res.rev, rpm_to_phase(both, FS))
    assert sorted(res.params["n_frames"]) == ["0", "1"]
    truth = _truth(sig, [0, 1, 0])
    for k, shaft in enumerate([0, 1, 0]):
        inside = _inside(sig.rev[:, shaft], res.params["edge_revs"])
        assert np.max(np.abs(_db(res.envelopes[inside, k], truth[inside, k]))) < 0.2  # dB
    # a single speed column with the default shafts is shaft 0
    one = tracked_harmonics(sig.x, FS, rpm, [2])
    assert one.shafts.tolist() == [0]
    assert one.rev.shape == rpm.shape


def test_channels_are_processed_identically_and_linearly():
    rpm = _runup(dur=2.5)
    sa = synth_runup(FS, rpm, [{"order": 2, "amplitude": 1.0, "phase": 0.3}], noise_rms=0.05, seed=1)
    sb = synth_runup(FS, rpm, [{"order": 2, "amplitude": 0.5, "phase": -1.0}], noise_rms=0.05, seed=2)
    orders = [2, 3.5]
    X = np.column_stack([sa.x, 3.0 * sb.x])
    res = tracked_harmonics(X, FS, rpm, orders)
    assert res.envelopes.shape == (rpm.size, 2, 2) and res.waveforms.shape == (rpm.size, 2, 2)
    assert res.total.shape == (rpm.size, 2)
    for ch in (0, 1):
        mono = tracked_harmonics(X[:, ch], FS, rpm, orders)
        assert mono.envelopes.shape == (rpm.size, 2)
        assert np.allclose(res.envelopes[:, :, ch], mono.envelopes, rtol=0, atol=1e-12)
        assert np.allclose(res.waveforms[:, :, ch], mono.waveforms, rtol=0, atol=1e-12)
    ref = tracked_harmonics(sb.x, FS, rpm, orders)
    assert np.allclose(res.envelopes[:, :, 1], 3.0 * ref.envelopes, rtol=0, atol=1e-12)
    # the channels keep their relative amplitude and phase
    inside = _inside(sa.rev, res.params["edge_revs"])
    ratio = res.envelopes[inside, 0, 1] / res.envelopes[inside, 0, 0]
    expected = 3.0 * 0.5 * np.exp(-1j * 1.0) / np.exp(0.3j)
    assert np.max(np.abs(ratio - expected)) < 0.15


def test_speed_reference_delay_is_compensated():
    t = np.arange(int(4 * FS)) / FS
    rpm = rpm_profiles(t, "linear", start=800.0, stop=3500.0)
    d = 0.05
    ref = np.interp(t - d, t, rpm)  # the reference lags the signal by d
    sig = synth_runup(FS, rpm, [{"order": 4, "amplitude": 1.0, "phase": 0.3}], seed=3)
    truth = _truth(sig)
    ok = (sig.rev > 8.0) & (t < t[-1] - d - 0.2)
    good = tracked_harmonics(sig.x, FS, ref, [4], delay=d)
    assert good.params["delay"] == d
    assert np.max(np.abs(_db(good.envelopes[ok, 0], truth[ok, 0]))) < 0.01  # dB
    assert np.max(np.abs(np.angle(good.envelopes[ok, 0] * np.conj(truth[ok, 0])))) < 1e-3  # rad
    bad = tracked_harmonics(sig.x, FS, ref, [4])  # uncompensated: the angle drifts
    assert np.max(np.abs(_db(bad.envelopes[ok, 0], truth[ok, 0]))) > 3.0  # dB


# --------------------------------------------------------------------------
# significance gate
# --------------------------------------------------------------------------


def _noisy_run_up():
    t = np.arange(int(8 * FS)) / FS
    rpm = rpm_profiles(t, "linear", start=800.0, stop=3000.0)
    sig = synth_runup(
        FS, rpm, [{"order": 2, "amplitude": 0.5, "phase": 0.3}], noise_rms=0.3, seed=2
    )
    return sig, rpm


def test_significance_gate_zeroes_a_noise_only_order_and_keeps_a_real_one():
    sig, rpm = _noisy_run_up()
    orders = [2, 7.3, 11.7]  # only order 2 is in the signal
    plain = tracked_harmonics(sig.x, FS, rpm, orders)
    gated = tracked_harmonics(sig.x, FS, rpm, orders, significance=0.99)
    assert gated.params["significance"] == 0.99
    kept = gated.params["kept_fraction"]
    assert kept[0] == 1.0 and kept[1] < 0.1 and kept[2] < 0.1
    assert np.mean(gated.envelopes[:, 1] != 0) < 0.25  # a few false alarms at most
    for k in (1, 2):
        assert _rms(gated.envelopes[:, k]) < 0.5 * _rms(plain.envelopes[:, k])
    # the real order passes unchanged
    assert np.array_equal(gated.envelopes[:, 0], plain.envelopes[:, 0])
    assert np.array_equal(gated.waveforms[:, 0], plain.waveforms[:, 0])
    assert "kept_fraction" not in plain.params


@pytest.mark.parametrize("level", [0.9, 0.99])
def test_significance_false_alarm_rate_is_the_requested_one(level):
    fs = 8000.0
    rpm = np.linspace(1000.0, 3000.0, int(60 * fs))
    sig = synth_runup(fs, rpm, [{"order": 3, "amplitude": 0.0}], noise_rms=1.0, seed=5)
    res = tracked_harmonics(
        sig.x, fs, rpm, [7.3, 20.3], window_revs=4.0, significance=level
    )
    q = 1.0 - level  # noise only: every kept frame is a false alarm
    # frames overlap by 75 %, so about a quarter of them are independent
    sigma = np.sqrt(q * (1.0 - q) / (res.params["n_frames"]["0"] / 4.0))
    rate = np.array(res.params["kept_fraction"])
    assert np.all(np.abs(rate - q) < 4.0 * sigma) and np.all(rate > 0.2 * q)


def test_orders_without_probe_orders_clear_of_the_other_orders_are_left_ungated():
    # a comb of orders every 0.5: inside the comb, every probe of an order (3 to 7.5 bins of
    # an 8-revolution window, i.e. 0.375 to 0.94 orders away) is in the main lobe of another one
    sig, rpm = _noisy_run_up()
    comb = np.arange(4.0, 12.01, 0.5)
    plain = tracked_harmonics(sig.x, FS, rpm, comb)
    with pytest.warns(UserWarning, match="ungated"):
        gated = tracked_harmonics(sig.x, FS, rpm, comb, significance=0.99)
    mid = comb.size // 2
    assert np.array_equal(gated.envelopes[:, mid], plain.envelopes[:, mid])
    assert gated.params["kept_fraction"][mid] == 1.0
    # the lowest order has free probes on its low side and is tested: it is noise only here
    assert gated.params["kept_fraction"][0] < 0.1
    assert _rms(gated.envelopes[:, 0]) < 0.5 * _rms(plain.envelopes[:, 0])


def test_significance_acts_on_each_channel_separately():
    sig, rpm = _noisy_run_up()
    quiet = np.random.default_rng(0).standard_normal(sig.x.size) * 0.3
    X = np.column_stack([sig.x, quiet])
    res = tracked_harmonics(X, FS, rpm, [2, 7.3], significance=0.99)
    assert np.mean(res.envelopes[:, 0, 0] != 0) == 1.0  # order 2 present in channel 0
    assert np.mean(res.envelopes[:, 0, 1] != 0) < 0.25  # and absent from channel 1


# --------------------------------------------------------------------------
# result container and bookkeeping
# --------------------------------------------------------------------------


def test_result_is_an_order_components_with_a_json_serialisable_record():
    rpm = _runup(dur=2.0)
    sig = synth_runup(FS, rpm, [{"order": 2, "amplitude": 1.0, "phase": 0.3}], seed=1)
    res = tracked_harmonics(sig.x, FS, rpm, 2, window_revs=6.0, overlap=0.5, window="hamming")
    assert isinstance(res, OrderComponents)
    assert res.method == "tracked-harmonics"
    assert res.orders.tolist() == [2.0] and res.shafts.tolist() == [0]
    assert res.waveforms.shape == res.envelopes.shape == (rpm.size, 1)
    assert np.iscomplexobj(res.envelopes) and not np.iscomplexobj(res.waveforms)
    assert np.array_equal(res.rev, rpm_to_phase(rpm, FS))
    assert res.params["window_revs"] == 6.0 and res.params["overlap"] == 0.5
    assert res.params["window"] == "hamming" and res.params["hop_revs"] == 3.0
    assert res.params["amplitude"] == "peak"
    assert json.loads(json.dumps(res.params)) == res.params
    assert res.params["n_frames"] == {"0": int(np.floor(sig.rev[-1] / 3.0)) + 1}


def test_inputs_are_not_modified():
    rpm = _runup(dur=1.5)
    sig = synth_runup(FS, rpm, [{"order": 2, "amplitude": 1.0, "phase": 0.3}], seed=1)
    x, r = sig.x.copy(), rpm.copy()
    orders = np.array([2.0, 4.0])
    tracked_harmonics(x, FS, r, orders)
    assert np.array_equal(x, sig.x) and np.array_equal(r, rpm)
    assert orders.tolist() == [2.0, 4.0]


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

_N = 4000
_X = np.zeros(_N)
_RPM = np.full(_N, 1500.0)


@pytest.mark.parametrize(
    "kw",
    [
        {"x": np.zeros((_N, 2, 2))},
        {"x": np.zeros(1), "rpm": np.full(1, 1500.0)},
        {"x": np.where(np.arange(_N) == 5, np.nan, 0.0)},
        {"fs": 0.0},
        {"fs": -16000.0},
        {"fs": np.nan},
        {"rpm": np.full(_N - 1, 1500.0)},
        {"rpm": np.full((_N, 2, 2), 1500.0)},
        {"rpm": np.where(np.arange(_N) == 5, np.nan, 1500.0)},
        {"rpm": np.where(np.arange(_N) == 5, -10.0, 1500.0)},
        {"rpm": np.zeros(_N)},  # the shaft does not turn
        {"orders": []},
        {"orders": [0.0]},
        {"orders": [-2.0]},
        {"orders": [np.nan]},
        {"orders": [[2.0, 3.0]]},
        {"shafts": [1]},
        {"shafts": [-1]},
        {"shafts": [0, 0]},
        {"shafts": [0.5]},
        {"window_revs": 0.0},
        {"window_revs": -4.0},
        {"window_revs": np.inf},
        {"overlap": 1.0},
        {"overlap": -0.1},
        {"window": "kaiser"},
        {"delay": np.nan},
        {"significance": 0.0},
        {"significance": 1.0},
        {"significance": 1.5},
    ],
)
def test_invalid_arguments_raise_value_error(kw):
    args = {"x": _X, "fs": FS, "rpm": _RPM, "orders": [2.0], **kw}
    with pytest.raises(ValueError):
        tracked_harmonics(**args)


def test_a_record_shorter_than_the_window_warns_that_it_is_all_edge():
    rpm = np.full(int(0.5 * FS), 1500.0)  # 12.5 revolutions
    x = np.cos(2 * np.pi * 3 * rpm_to_phase(rpm, FS))
    with pytest.warns(UserWarning, match="edge"):
        res = tracked_harmonics(x, FS, rpm, [3], window_revs=20.0)
    assert np.all(np.isfinite(res.envelopes))
    # still the right amplitude for a stationary component
    assert np.max(np.abs(_db(res.envelopes[:, 0], 1.0))) < 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tracked_harmonics(x, FS, rpm, [3], window_revs=6.0)
