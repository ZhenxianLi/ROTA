"""Tests of the Vold-Kalman order filter against analytic truth (signals from rota.synth)."""

import json

import numpy as np
import pytest

import rota._vk as vk
from rota._components import OrderComponents
from rota._vk import _bandwidth_to_weight, _weight_to_bandwidth, order_waveform
from rota.speed import rpm_to_phase
from rota.synth import rpm_profiles, synth_runup


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def _truth(sig, k, shaft=0):
    """Complex peak envelope of order k as put into the signal."""
    rev = sig.rev if sig.rev.ndim == 1 else sig.rev[:, shaft]
    phase = sig.order_phase[:, k] - 2 * np.pi * sig.orders[k] * rev
    return sig.order_amplitude[:, k] * np.exp(1j * phase)


def _trim(n, fs, seconds):
    m = round(seconds * fs)
    return slice(m, n - m)


# --------------------------------------------------------------------------
# bandwidth <-> weight, and the measured -3 dB point (ROADMAP 5.4: within 5 %)
# --------------------------------------------------------------------------


def test_weight_matches_closed_form_and_inverts():
    fs, bw = 4000.0, 5.0
    r = _bandwidth_to_weight(bw, fs)
    assert r == pytest.approx(np.sqrt((np.sqrt(2) - 1) / 2) / (2 * np.sin(np.pi * bw / fs)) ** 2)
    # low-frequency limit 0.455 (fs / (2 pi B))^2
    big = _bandwidth_to_weight(0.4, 4000.0)
    assert big == pytest.approx(0.4551 * (4000.0 / (2 * np.pi * 0.4)) ** 2, rel=1e-3)
    b = np.array([0.1, 1.0, 20.0, 300.0])
    assert _weight_to_bandwidth(_bandwidth_to_weight(b, fs), fs) == pytest.approx(b, rel=1e-12)


@pytest.mark.parametrize("bw", [2.0, 5.0, 20.0])
def test_minus_3db_point_equals_bandwidth(bw):
    fs, f_car = 4000.0, 400.0
    n = int(3 * fs)
    rpm = np.full(n, 6000.0)  # 100 Hz per order: order 4 is the 400 Hz carrier
    offs = bw * np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
    gain = np.empty(offs.size)
    for i, df in enumerate(offs):
        tone = {"order": 4 + df / 100.0, "amplitude": 1.0, "phase": 0.4}
        sig = synth_runup(fs, rpm, [tone], seed=1)
        env = order_waveform(sig.x, fs, rpm, [4], bandwidth=bw).envelopes[:, 0]
        mid = _trim(n, fs, 0.75)
        gain[i] = np.median(np.abs(env[mid]))
    # closed-form single-order response, H = 1 / (1 + 2 r^2 (2 sin(pi df / fs))^4)
    r = _bandwidth_to_weight(bw, fs)
    h = 1.0 / (1.0 + 2.0 * r**2 * (2.0 * np.sin(np.pi * offs / fs)) ** 4)
    assert gain == pytest.approx(h, abs=2e-3)
    # measured half-power (-3 dB) offset, interpolated on log(offset)
    assert np.all(np.diff(gain) < 0)
    f3 = np.exp(np.interp(-np.sqrt(0.5), -gain, np.log(offs)))
    assert abs(f3 / bw - 1.0) < 0.05
    assert f_car > 10 * bw  # image term negligible


# --------------------------------------------------------------------------
# single order
# --------------------------------------------------------------------------


def test_single_order_in_noise_amplitude_and_waveform():
    fs, dur = 4000.0, 6.0
    n = int(dur * fs)
    rpm = rpm_profiles(np.arange(n) / fs, "linear", start=1500, stop=3500)
    orders = [{"order": 2, "amplitude": 1.0, "phase": 0.7}]
    sig = synth_runup(fs, rpm, orders, noise_rms=0.05, seed=5)
    res = order_waveform(sig.x, fs, rpm, [2], bandwidth=5.0)
    tr = _trim(n, fs, 0.5)
    level_db = 20 * np.log10(np.abs(res.envelopes[tr, 0]) / sig.order_amplitude[tr, 0])
    assert np.abs(level_db).max() < 0.2  # ROADMAP 5.4: 0.2 dB (measured 0.1)
    assert abs(level_db.mean()) < 0.01
    assert _corr(res.waveforms[tr, 0], sig.order_stems[tr, 0]) > 0.999
    # the order is peak amplitude 1: the waveform's peak is 1 too
    assert np.abs(res.waveforms[tr, 0]).max() == pytest.approx(1.0, abs=0.1)
    # the phase of the envelope is the truth phase relative to the carrier
    d = np.angle(res.envelopes[tr, 0] * np.conj(_truth(sig, 0)[tr]))
    assert np.abs(d).max() < 0.1


@pytest.mark.parametrize("fs, bw, n", [(8000.0, 0.3, 3000), (48000.0, 1.0, 2000)])
def test_noise_free_constant_envelope_is_exact_for_narrow_bands(fs, bw, n):
    # r^2 ~ 7e13 and 7e14: the plain Cholesky solution is wrong in the 2nd-4th digit
    rpm = np.full(n, 6000.0)
    sig = synth_runup(fs, rpm, [{"order": 4, "amplitude": 1.0, "phase": 0.3}], seed=1)
    res = order_waveform(sig.x, fs, rpm, [4], bandwidth=bw)
    assert res.params["refinement_iterations"] > 2
    assert np.abs(res.envelopes[:, 0] - np.exp(0.3j)).max() < 1e-9
    assert np.abs(res.waveforms[:, 0] - sig.x).max() < 1e-9


def test_amplitude_ramp_and_phase_tracking():
    # gain law (piece-wise linear amplitude) and a resonance (amplitude and phase change)
    fs, dur = 4000.0, 5.0
    n = int(dur * fs)
    rpm = rpm_profiles(np.arange(n) / fs, "linear", start=1500, stop=4500)  # order 4: 100 -> 300 Hz
    gain = {"gain_rpm": [1500, 3000, 4500], "gain": [0.5, 1.5, 1.0]}
    orders = [{"order": 4, "amplitude": 1.0, "phase": 0.9, **gain}]
    resonance = [{"freq": 200.0, "q": 8.0, "gain": 3.0}]
    sig = synth_runup(fs, rpm, orders, resonances=resonance, noise_rms=0.05, seed=4)
    res = order_waveform(sig.x, fs, rpm, [4], bandwidth=5.0)
    tr = _trim(n, fs, 0.5)
    amp = sig.order_amplitude[tr, 0]
    assert amp.max() / amp.min() > 5  # a real amplitude excursion
    level_db = 20 * np.log10(np.abs(res.envelopes[tr, 0]) / amp)
    assert np.abs(level_db).max() < 0.2
    truth = _truth(sig, 0)[tr]
    assert (np.abs(res.envelopes[tr, 0] - truth) / amp).max() < 0.03
    # 2 pi-unwrapped phase against the truth phase relative to the carrier
    ph_true = sig.order_phase[:, 0] - 2 * np.pi * 4 * sig.rev
    assert np.ptp(ph_true[tr]) > 0.3  # the resonance moves the phase
    d = np.unwrap(np.angle(res.envelopes[tr, 0])) - ph_true[tr]
    d -= 2 * np.pi * np.round(np.median(d) / (2 * np.pi))
    assert np.abs(d).max() < 0.03


# --------------------------------------------------------------------------
# close carriers (two shafts, 200 and 201.2 Hz) and crossing orders
# --------------------------------------------------------------------------


def _two_shaft_pair(am=None, noise=0.05):
    fs, dur = 2000.0, 8.0
    n = int(dur * fs)
    rpm = np.column_stack([np.full(n, 200.0 * 60), np.full(n, 201.2 * 60)])
    o0 = {"order": 1, "shaft": 0, "amplitude": 1.0, "phase": 0.3}
    if am is not None:
        o0.update(am_depth=0.5, am_freq=am)
    o1 = {"order": 1, "shaft": 1, "amplitude": 0.5, "phase": 1.1}
    sig = synth_runup(fs, rpm, [o0, o1], noise_rms=noise, seed=2)
    return fs, n, rpm, sig


def test_close_carriers_separated_by_narrow_joint_filter():
    fs, n, rpm, sig = _two_shaft_pair()
    res = order_waveform(sig.x, fs, rpm, [1, 1], shafts=[0, 1], bandwidth=0.3)
    tr = _trim(n, fs, 1.0)
    for k, amp in enumerate((1.0, 0.5)):
        assert _corr(res.waveforms[tr, k], sig.order_stems[tr, k]) > 0.95  # measured > 0.999
        assert np.abs(res.envelopes[tr, k]).mean() == pytest.approx(amp, rel=0.02)


def test_close_carriers_wide_band_cannot_be_separated_by_independent_filters():
    # a pass band (2 x 5 Hz) much wider than the 1.2 Hz spacing: each filter sees both orders
    fs, n, rpm, sig = _two_shaft_pair()
    tr = _trim(n, fs, 1.0)
    wide = order_waveform(sig.x, fs, rpm, [1, 1], shafts=[0, 1], bandwidth=5.0, joint=False)
    assert _corr(wide.waveforms[tr, 1], sig.order_stems[tr, 1]) < 0.6  # measured 0.43
    assert _corr(wide.waveforms[tr, 0], sig.order_stems[tr, 0]) < 0.95  # measured 0.89
    # narrow independent filters reject the neighbour
    narrow = order_waveform(sig.x, fs, rpm, [1, 1], shafts=[0, 1], bandwidth=0.3, joint=False)
    assert _corr(narrow.waveforms[tr, 1], sig.order_stems[tr, 1]) > 0.99


def test_close_carriers_wide_band_joint_works_only_for_stationary_envelopes():
    # The coupled least-squares has a single low-penalty explanation of two stationary
    # tones, so it separates them even with a pass band wider than the spacing ...
    fs, n, rpm, sig = _two_shaft_pair()
    tr = _trim(n, fs, 1.5)
    res = order_waveform(sig.x, fs, rpm, [1, 1], shafts=[0, 1], bandwidth=5.0)
    assert _corr(res.waveforms[tr, 1], sig.order_stems[tr, 1]) > 0.95
    # ... but when the strong order is amplitude modulated at the carrier spacing, its sideband
    # coincides with the weak carrier: not identifiable, whatever the bandwidth (limit of the data).
    fs, n, rpm, sig = _two_shaft_pair(am=1.2)
    tr = _trim(n, fs, 1.5)
    for bw in (0.3, 5.0):
        res = order_waveform(sig.x, fs, rpm, [1, 1], shafts=[0, 1], bandwidth=bw)
        dev = np.abs(res.envelopes[tr, 1]) - sig.order_amplitude[tr, 1]
        err = np.sqrt(np.mean(dev**2)) / 0.5
        assert err > 0.15  # measured 0.27


def test_crossing_orders_on_two_shafts_joint_beats_independent():
    # ROADMAP T4: order 2 of a run-up shaft (100 -> 200 Hz) crosses order 2 of a shaft at 150 Hz
    fs, dur = 2000.0, 6.0
    n = int(dur * fs)
    t = np.arange(n) / fs
    rpm = np.column_stack([rpm_profiles(t, "linear", start=3000, stop=6000), np.full(n, 4500.0)])
    orders = [
        {"order": 2, "shaft": 0, "amplitude": 1.0, "phase": 0.4},
        {"order": 2, "shaft": 1, "amplitude": 0.6, "phase": 2.0},
    ]
    sig = synth_runup(fs, rpm, orders, noise_rms=0.05, seed=7)
    assert abs(t[np.argmin(np.abs(2 * rpm[:, 0] / 60 - 2 * rpm[:, 1] / 60))] - 3.0) < 0.01
    win = np.abs(t - 3.0) < 1.0
    truth = np.column_stack([_truth(sig, 0, 0), _truth(sig, 1, 1)])
    err = {}
    for joint in (True, False):
        res = order_waveform(sig.x, fs, rpm, [2, 2], shafts=[0, 1], bandwidth=3.0, joint=joint)
        rms = np.sqrt(np.mean(np.abs(res.envelopes[win] - truth[win]) ** 2, axis=0))
        err[joint] = rms / np.abs(truth).mean(axis=0)
    assert np.all(err[True] < 0.03)  # measured 0.006, 0.009
    assert np.all(err[False] > 0.1)  # measured 0.26, 0.72
    assert np.all(err[False] > 10 * err[True])


def test_joint_false_agrees_with_joint_for_separated_orders():
    fs, dur = 2000.0, 4.0
    n = int(dur * fs)
    rpm = rpm_profiles(np.arange(n) / fs, "linear", start=2000, stop=3000)
    orders = [
        {"order": 1, "amplitude": 1.0, "phase": 0.2},
        {"order": 3, "amplitude": 0.5, "phase": 1.0},
    ]
    sig = synth_runup(fs, rpm, orders, noise_rms=0.05, seed=3)
    a = order_waveform(sig.x, fs, rpm, [1, 3], bandwidth=4.0, joint=True)
    b = order_waveform(sig.x, fs, rpm, [1, 3], bandwidth=4.0, joint=False)
    tr = _trim(n, fs, 0.75)
    assert np.abs(a.envelopes[tr] - b.envelopes[tr]).max() < 1e-3 * np.abs(a.envelopes).max()
    # at the record ends an independent filter has a transient from the other order (joint has none)
    assert np.abs(a.envelopes - b.envelopes)[:200].max() > 0.02


# --------------------------------------------------------------------------
# blocks, memory, multichannel, exactness
# --------------------------------------------------------------------------


def _three_orders():
    fs, dur = 2000.0, 12.0
    n = int(dur * fs)
    rpm = rpm_profiles(np.arange(n) / fs, "linear", start=2000, stop=5000)
    orders = [
        {"order": 2, "amplitude": 1.0, "phase": 0.3, "am_depth": 0.3, "am_freq": 0.7},
        {"order": 3.5, "amplitude": 0.4, "phase": 1.0},
        {"order": 5, "amplitude": 0.2, "phase": 2.0, "fm_depth": 0.01, "fm_freq": 0.5},
    ]
    return fs, rpm, synth_runup(fs, rpm, orders, noise_rms=0.1, seed=11)


@pytest.mark.parametrize("bw, block", [(20.0, 3000), (10.0, 4000)])
def test_block_processing_equals_single_solve(bw, block):
    fs, rpm, sig = _three_orders()
    full = order_waveform(sig.x, fs, rpm, [2, 3.5, 5], bandwidth=bw)
    blk = order_waveform(sig.x, fs, rpm, [2, 3.5, 5], bandwidth=bw, block=block)
    assert full.params["n_blocks"] == 1
    assert blk.params["n_blocks"] >= 5
    assert blk.params["block"] == block
    scale = np.abs(full.envelopes).max()
    assert np.abs(blk.envelopes - full.envelopes).max() < 1e-6 * scale  # measured 1e-11
    assert np.abs(blk.waveforms - full.waveforms).max() < 1e-6 * scale


def test_block_processing_joint_false_and_multichannel():
    fs, rpm, sig = _three_orders()
    x = np.column_stack([sig.x, 0.5 * sig.x[::-1]])
    kw = {"bandwidth": 20.0, "joint": False}
    full = order_waveform(x, fs, rpm, [2, 3.5], **kw)
    blk = order_waveform(x, fs, rpm, [2, 3.5], block=3000, **kw)
    assert blk.params["n_blocks"] > 1
    assert np.abs(blk.envelopes - full.envelopes).max() < 1e-6 * np.abs(full.envelopes).max()


def test_default_block_follows_memory_budget(monkeypatch):
    fs, rpm, sig = _three_orders()
    full = order_waveform(sig.x, fs, rpm, [2, 3.5], bandwidth=20.0)
    monkeypatch.setattr(vk, "_BUDGET", 600_000)  # a few thousand samples
    small = order_waveform(sig.x, fs, rpm, [2, 3.5], bandwidth=20.0)
    assert small.params["n_blocks"] > 1
    assert small.params["block"] < sig.x.size
    assert small.params["block"] >= 4 * small.params["overlap"]
    assert np.abs(small.envelopes - full.envelopes).max() < 1e-6 * np.abs(full.envelopes).max()
    monkeypatch.setattr(vk, "_BUDGET", 1_000)
    with pytest.raises(MemoryError):
        order_waveform(sig.x, fs, rpm, [2, 3.5], bandwidth=20.0)
    # an explicit block overrides the memory guard
    ok = order_waveform(sig.x, fs, rpm, [2, 3.5], bandwidth=20.0, block=3000)
    assert ok.params["n_blocks"] > 1


def test_multichannel_equals_per_channel_and_is_linear():
    fs, rpm, sig = _three_orders()
    n = sig.x.size
    rng = np.random.default_rng(0)
    x = np.column_stack([sig.x, 3.0 * sig.x + 0.2 * rng.standard_normal(n), rng.standard_normal(n)])
    multi = order_waveform(x, fs, rpm, [2, 3.5], bandwidth=8.0)
    assert multi.envelopes.shape == (n, 2, 3)
    assert multi.waveforms.shape == (n, 2, 3)
    for c in range(3):
        one = order_waveform(x[:, c], fs, rpm, [2, 3.5], bandwidth=8.0)
        assert one.envelopes.shape == (n, 2)
        pairs = ((multi.envelopes, one.envelopes), (multi.waveforms, one.waveforms))
        for m, o in pairs:
            assert np.abs(m[:, :, c] - o).max() < 1e-9 * np.abs(o).max()
    # a single column keeps its channel axis
    col = order_waveform(x[:, :1], fs, rpm, [2, 3.5], bandwidth=8.0)
    assert col.envelopes.shape == (n, 2, 1)
    # linearity: scaling a channel by 3 scales its envelopes by 3
    one0 = order_waveform(x[:, 0], fs, rpm, [2, 3.5], bandwidth=8.0)
    scaled = order_waveform(3.0 * x[:, 0], fs, rpm, [2, 3.5], bandwidth=8.0)
    err = np.abs(scaled.envelopes - 3.0 * one0.envelopes).max()
    assert err < 1e-9 * np.abs(scaled.envelopes).max()


def test_matches_dense_least_squares_with_per_order_bandwidth():
    # independent dense assembly of the cost function (two shafts, fractional order, K = 3)
    fs, n = 400.0, 80
    t = np.arange(n) / fs
    rpm = np.column_stack([2400.0 + 600.0 * t / t[-1], np.full(n, 1800.0)])
    orders = np.array([0.4, 1.0, 1.0])
    shafts = np.array([0, 0, 1])
    bw = np.array([20.0, 40.0, 30.0])
    rng = np.random.default_rng(1)
    x = rng.standard_normal((n, 2))
    res = order_waveform(x, fs, rpm, orders, shafts=shafts, bandwidth=bw)
    r = np.array(res.params["weight"])
    assert r[0] > r[2] > r[1]  # narrower band, larger weight
    rev = rpm_to_phase(rpm, fs)
    theta = 2 * np.pi * orders[None, :] * rev[:, shafts]
    K = 3
    phi = np.zeros((n, 2 * K * n))
    for k in range(K):
        phi[np.arange(n), 2 * K * np.arange(n) + 2 * k] = np.cos(theta[:, k])
        phi[np.arange(n), 2 * K * np.arange(n) + 2 * k + 1] = -np.sin(theta[:, k])
    d2 = np.diff(np.eye(n), n=2, axis=0)
    rows = []
    for k in range(K):
        for part in range(2):
            blk = np.zeros((n - 2, 2 * K * n))
            blk[:, 2 * K * np.arange(n) + 2 * k + part] = d2
            rows.append(r[k] * blk)
    a = np.vstack([phi] + rows)
    for c in range(2):
        rhs = np.concatenate([x[:, c], np.zeros(a.shape[0] - n)])
        u = np.linalg.lstsq(a, rhs, rcond=None)[0].reshape(n, K, 2)
        ref = u[..., 0] + 1j * u[..., 1]
        assert np.abs(res.envelopes[:, :, c] - ref).max() < 1e-8 * np.abs(ref).max()


def test_zero_signal_gives_zero_components():
    n = 500
    res = order_waveform(np.zeros(n), 1000.0, np.full(n, 3000.0), [1, 2], bandwidth=5.0)
    assert np.all(res.envelopes == 0) and np.all(res.waveforms == 0)


# --------------------------------------------------------------------------
# result contract
# --------------------------------------------------------------------------


def test_returns_order_components_with_the_documented_convention():
    fs, n = 2000.0, 3000
    t = np.arange(n) / fs
    rpm = np.column_stack([rpm_profiles(t, "linear", start=2000, stop=3000), np.full(n, 2400.0)])
    orders = [
        {"order": 2, "shaft": 0, "amplitude": 1.0, "phase": 0.5},
        {"order": 3, "shaft": 1, "amplitude": 0.5, "phase": 1.5},
    ]
    sig = synth_runup(fs, rpm, orders, noise_rms=0.05, seed=3)
    res = order_waveform(sig.x, fs, rpm, [2, 3], shafts=[0, 1], bandwidth=[4.0, 6.0])
    assert isinstance(res, OrderComponents)
    assert res.method == "vold-kalman"
    assert res.orders.tolist() == [2.0, 3.0]
    assert res.shafts.tolist() == [0, 1]
    assert res.waveforms.shape == res.envelopes.shape == (n, 2)
    assert np.iscomplexobj(res.envelopes) and not np.iscomplexobj(res.waveforms)
    assert res.rev.shape == (n, 2)
    assert np.array_equal(res.rev, rpm_to_phase(rpm, fs))
    # waveforms = Re{envelope exp(i 2 pi order rev)}
    theta = 2 * np.pi * res.orders[None, :] * res.rev[:, res.shafts]
    assert np.allclose(res.waveforms, (res.envelopes * np.exp(1j * theta)).real, atol=1e-12)
    assert np.allclose(res.total, res.waveforms.sum(axis=1))
    # peak amplitude: |envelope| is the order's peak amplitude
    tr = _trim(n, fs, 0.4)
    for k, amp in enumerate((1.0, 0.5)):
        assert np.abs(res.envelopes[tr, k]).mean() == pytest.approx(amp, rel=0.05)
    # parameters are recorded and JSON-serialisable
    p = res.params
    json.dumps(p)
    assert p["bandwidth"] == [4.0, 6.0] and p["joint"] is True and p["amplitude"] == "peak"
    assert p["fs"] == fs and len(p["weight"]) == 2
    assert p["weight"] == pytest.approx([float(_bandwidth_to_weight(b, fs)) for b in (4.0, 6.0)])
    # bare-array return
    kw = {"shafts": [0, 1], "bandwidth": [4.0, 6.0], "return_components": False}
    wave = order_waveform(sig.x, fs, rpm, [2, 3], **kw)
    assert isinstance(wave, np.ndarray) and np.array_equal(wave, res.waveforms)
    # a 1-D speed gives a 1-D angle
    r1 = order_waveform(sig.x, fs, rpm[:, 0], 2.0, bandwidth=4.0)
    assert r1.rev.shape == (n,) and r1.waveforms.shape == (n, 1) and r1.shafts.tolist() == [0]


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------

_N = 600
_X = np.sin(np.arange(_N) / 7.0)
_RPM = np.full(_N, 3000.0)  # 50 Hz per order


def _call(**over):
    kw = {"x": _X, "fs": 1000.0, "rpm": _RPM, "orders": [1, 2], "bandwidth": 5.0}
    kw.update(over)
    return order_waveform(**kw)


@pytest.mark.parametrize(
    "over, match",
    [
        ({"fs": 0.0}, "fs"),
        ({"fs": float("nan")}, "fs"),
        ({"x": np.zeros((_N, 2, 2))}, "shape"),
        ({"x": np.full(_N, np.nan)}, "non-finite"),
        ({"x": np.zeros(2), "rpm": np.zeros(2)}, "at least 3"),
        ({"rpm": _RPM[:-1]}, "rpm"),
        ({"rpm": np.full(_N, np.nan)}, "non-finite"),
        ({"rpm": np.zeros((_N + 1, 2))}, "rpm"),
        ({"rpm": np.zeros((_N, 2, 2))}, "rpm"),
        ({"orders": []}, "orders"),
        ({"orders": [[1, 2]]}, "orders"),
        ({"orders": [0.0, 1.0]}, "positive"),
        ({"orders": [-1.0, 1.0]}, "positive"),
        ({"orders": [float("nan"), 1.0]}, "positive"),
        ({"shafts": [0]}, "shafts"),
        ({"shafts": [0, 1]}, "columns"),
        ({"shafts": [0, -1]}, "columns"),
        ({"shafts": [0, 0.5]}, "integer"),
        ({"bandwidth": 0.0}, "bandwidth"),
        ({"bandwidth": -1.0}, "bandwidth"),
        ({"bandwidth": 500.0}, "bandwidth"),
        ({"bandwidth": float("nan")}, "bandwidth"),
        ({"bandwidth": [5.0, 5.0, 5.0]}, "bandwidth"),
        ({"block": 0}, "block"),
        ({"block": 2.5}, "block"),
        ({"block": 100}, "at least"),  # under two overlaps, signal longer than a block
        ({"orders": [1, 10]}, "Nyquist|fs / 2"),  # 10 x 50 Hz = 500 Hz = fs / 2
        ({"orders": [2, 2]}, "distinct"),
    ],
)
def test_input_validation(over, match):
    with pytest.raises(ValueError, match=match):
        _call(**over)


def test_duplicate_orders_are_fine_when_not_joint():
    res = _call(orders=[2, 2], joint=False)
    assert np.allclose(res.envelopes[:, 0], res.envelopes[:, 1])


def test_bandwidth_too_narrow_for_the_sampling_rate_raises():
    fs, n = 48000.0, 3000
    x = np.cos(2 * np.pi * 200.0 * np.arange(n) / fs)
    with pytest.raises(ValueError, match="narrow|singular"):
        order_waveform(x, fs, np.full(n, 6000.0), [2], bandwidth=0.01)
