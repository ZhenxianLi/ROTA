"""Tests of rota_order.extract.remove_orders and rota_order.extract.residual, both extractors.

The signals are rendered by rota_order.synth.synth_runup, so the truth of what was
put in is stored with them: ``order_stems`` (each order as rendered),
``stems["orders"]``, ``stems["tones"]`` and ``stems["noise"]``. After removing
some orders the residual must equal "everything else", that is the signal minus
the truth stems of the removed orders, which is noise plus tones plus the
orders that were kept.

Amplitudes are peak (rota_order.synth convention); the checks below are ratios or
correlations and do not depend on the convention.
"""

import numpy as np
import pytest

from rota_order.extract import OrderComponents, remove_orders, residual
from rota_order.metrics import order_residual_energy, waveform_similarity
from rota_order.synth import rpm_profiles, synth_runup

FS = 4000.0
DURATION = 6.0
ORDERS = [1.0, 2.0, 3.0, 4.5]
AMPLITUDES = [0.5, 1.0, 0.7, 0.4]
REMOVED = [2.0, 4.5]  # indices 1 and 3 of ORDERS
KEPT = [1.0, 3.0]
NOISE_RMS = 0.05
TONE = {"freq": 450.0, "amplitude": 0.2}  # above the highest order frequency (337 Hz)

# One configuration per method: the keyword arguments of its extractor.
METHODS = {
    "vold-kalman": {"bandwidth": 2.0},
    "tracked-harmonics": {"window_revs": 8.0},
}
EDGE = round(1.0 * FS)  # samples dropped at each end (VK: 0.75 s, tracked: 8 revs <= 0.32 s)


def _runup(seed, gain=1.0, noise_rms=NOISE_RMS, rpm=None):
    t = np.arange(round(DURATION * FS)) / FS
    if rpm is None:
        rpm = rpm_profiles(t, "linear", start=1500.0, stop=4500.0)
    orders = [{"order": o, "amplitude": gain * a} for o, a in zip(ORDERS, AMPLITUDES)]
    return synth_runup(
        FS, rpm, orders, tones=[{**TONE, "amplitude": gain * TONE["amplitude"]}],
        noise_rms=noise_rms, seed=seed,
    )


def _stems(sig, orders):
    """Sum of the true stems of ``orders`` (shape (n,))."""
    idx = [list(sig.orders).index(o) for o in orders]
    return sig.order_stems[:, idx].sum(axis=1)


@pytest.fixture(scope="module")
def sig():
    return _runup(seed=11)


@pytest.fixture(scope="module", params=list(METHODS))
def method(request):
    return request.param


@pytest.fixture(scope="module")
def removed(sig, method):
    """``remove_orders`` of ``REMOVED`` with the method's default test settings."""
    return remove_orders(sig.x, FS, sig.rpm, REMOVED, method=method, **METHODS[method])


# --------------------------------------------------------------------------
# remove_orders: residual = noise + tones + the orders that were kept
# --------------------------------------------------------------------------


def test_residual_equals_the_signal_without_the_removed_orders(sig, removed):
    y, comp = removed
    others = sig.x - _stems(sig, REMOVED)  # noise + tone + kept orders (truth)
    removed_truth = _stems(sig, REMOVED)
    r = waveform_similarity(y, others, mixture=sig.x, other=removed_truth, trim=EDGE)
    assert r["corr"] >= 0.99
    assert r["nrmse"] <= 0.05
    # Nothing was done by accident: the unprocessed mixture is a poor stand-in for the
    # remaining signal, and the residual contains none of the removed orders.
    assert r["baseline_corr"] < 0.8
    assert abs(r["cross_corr"]) < 0.05
    # The extracted components are the removed orders.
    assert isinstance(comp, OrderComponents)
    assert list(comp.orders) == REMOVED
    ws = waveform_similarity(comp.total, removed_truth, trim=EDGE)
    assert ws["corr"] >= 0.99 and ws["nrmse"] <= 0.05


def test_energy_at_removed_orders_drops_and_kept_orders_are_untouched(sig, removed):
    y, _ = removed
    for o in REMOVED:  # ROADMAP T12 criterion, here on a short signal
        assert order_residual_energy(y, sig.x, FS, sig.rpm, o) <= -20.0
    for o in KEPT:  # 0 dB: the energy of the kept orders is the original energy
        assert order_residual_energy(y, sig.x, FS, sig.rpm, o) == pytest.approx(0.0, abs=0.3)


def test_components_plus_residual_is_the_input(sig, removed):
    y, comp = removed
    # y is computed as x - total, so this holds bit for bit ...
    assert np.array_equal(y, sig.x - comp.total)
    # ... and the sum returns x to rounding: (x - T) + T can differ from x by one ulp.
    tol = 4.0 * np.finfo(float).eps * np.abs(sig.x).max()
    assert np.abs(comp.total + y - sig.x).max() <= tol
    assert y.shape == sig.x.shape == (round(DURATION * FS),)
    assert comp.waveforms.shape == (sig.x.size, len(REMOVED))
    assert np.allclose(comp.total, comp.waveforms.sum(axis=1), atol=0, rtol=0)


def test_lists_and_integer_arrays_are_accepted(sig, method):
    kw = METHODS[method]
    x, rpm = sig.x[:6000], sig.rpm[:6000]
    ref, _ = remove_orders(x, FS, rpm, [2.0], method=method, **kw)
    y, _ = remove_orders(x.tolist(), FS, rpm.tolist(), [2.0], method=method, **kw)
    assert np.array_equal(y, ref)
    counts = np.round(1000.0 * x).astype(int)  # integer samples (e.g. raw ADC counts)
    yi, _ = remove_orders(counts, FS, rpm, [2.0], method=method, **kw)
    yf, _ = remove_orders(counts.astype(float), FS, rpm, [2.0], method=method, **kw)
    assert yi.dtype == float and np.array_equal(yi, yf)


# --------------------------------------------------------------------------
# residual: all orders removed leaves noise (and tones)
# --------------------------------------------------------------------------


def test_residual_of_every_order_is_noise_and_tone(sig, method):
    y, comp = residual(sig.x, FS, sig.rpm, ORDERS, method=method, **METHODS[method])
    background = sig.stems["tones"] + sig.stems["noise"]
    r = waveform_similarity(y, background, mixture=sig.x, trim=EDGE)
    # Removing the orders also removes the noise inside the extraction bands
    # (a few percent of the noise power for four orders), so the agreement is
    # not perfect; the unprocessed mixture has a correlation of about 0.15.
    assert r["corr"] >= 0.99
    assert r["baseline_corr"] < 0.3
    assert np.array_equal(y, sig.x - comp.total)
    assert list(comp.orders) == ORDERS
    for o in ORDERS:
        assert order_residual_energy(y, sig.x, FS, sig.rpm, o) <= -20.0


def test_residual_is_remove_orders_with_the_same_arguments(sig, method):
    kw = METHODS[method]
    y1, c1 = residual(sig.x, FS, sig.rpm, REMOVED, method=method, **kw)
    y2, c2 = remove_orders(sig.x, FS, sig.rpm, REMOVED, method=method, **kw)
    assert np.array_equal(y1, y2)
    assert np.array_equal(c1.envelopes, c2.envelopes)


# --------------------------------------------------------------------------
# Method validation and keyword pass-through
# --------------------------------------------------------------------------


@pytest.mark.parametrize("func", [remove_orders, residual])
@pytest.mark.parametrize("bad", ["vk", "vold_kalman", "tracked", "", "Vold-Kalman", None, 1])
def test_unknown_method_is_rejected(sig, func, bad):
    with pytest.raises(ValueError, match="method must be one of"):
        func(sig.x[:4000], FS, sig.rpm[:4000], [2.0], method=bad)


def test_method_error_lists_the_valid_methods(sig):
    with pytest.raises(ValueError) as err:
        remove_orders(sig.x[:4000], FS, sig.rpm[:4000], [2.0], method="nope")
    assert "vold-kalman" in str(err.value) and "tracked-harmonics" in str(err.value)
    assert "'nope'" in str(err.value)


def test_default_method_is_vold_kalman(sig):
    _, comp = remove_orders(sig.x, FS, sig.rpm, [2.0], bandwidth=2.0)
    assert comp.method == "vold-kalman"
    assert comp.params["bandwidth"] == [2.0]
    _, comp_th = remove_orders(sig.x, FS, sig.rpm, [2.0], method="tracked-harmonics")
    assert comp_th.method == "tracked-harmonics"
    assert comp_th.params["window_revs"] == 8.0  # the extractor's own default


def test_keyword_arguments_reach_the_vold_kalman_filter(sig):
    base = remove_orders(sig.x, FS, sig.rpm, REMOVED, bandwidth=2.0)
    wide = remove_orders(sig.x, FS, sig.rpm, REMOVED, bandwidth=6.0)
    alone = remove_orders(sig.x, FS, sig.rpm, REMOVED, bandwidth=2.0, joint=False)
    assert base[1].params["bandwidth"] == [2.0, 2.0]
    assert wide[1].params["bandwidth"] == [6.0, 6.0]
    assert base[1].params["joint"] is True and alone[1].params["joint"] is False
    # the settings change the result, and by the expected amount: a wider band takes
    # more noise out, so the residual of the wide filter has less energy
    assert not np.allclose(base[0], wide[0], atol=1e-6)
    assert np.mean(wide[0][EDGE:-EDGE] ** 2) < np.mean(base[0][EDGE:-EDGE] ** 2)
    # per-order bandwidths are passed on unchanged
    each = remove_orders(sig.x, FS, sig.rpm, REMOVED, bandwidth=[1.0, 3.0])
    assert each[1].params["bandwidth"] == [1.0, 3.0]


def test_keyword_arguments_reach_the_tracked_harmonics_extractor(sig):
    kw = {"method": "tracked-harmonics"}
    short = remove_orders(sig.x, FS, sig.rpm, REMOVED, window_revs=4.0, **kw)
    long = remove_orders(sig.x, FS, sig.rpm, REMOVED, window_revs=16.0, **kw)
    assert short[1].params["window_revs"] == 4.0 and long[1].params["window_revs"] == 16.0
    assert not np.allclose(short[0], long[0], atol=1e-6)
    hann = remove_orders(sig.x, FS, sig.rpm, REMOVED, window="blackman", overlap=0.5, **kw)
    assert hann[1].params["window"] == "blackman" and hann[1].params["overlap"] == 0.5
    gated = remove_orders(sig.x, FS, sig.rpm, REMOVED, significance=0.99, **kw)
    assert gated[1].params["significance"] == 0.99 and "kept_fraction" in gated[1].params


def test_a_keyword_of_the_other_extractor_is_an_error(sig):
    with pytest.raises(TypeError):
        remove_orders(sig.x[:4000], FS, sig.rpm[:4000], [2.0], window_revs=8.0)  # vold-kalman
    with pytest.raises(TypeError):
        remove_orders(
            sig.x[:4000], FS, sig.rpm[:4000], [2.0], method="tracked-harmonics", bandwidth=2.0
        )
    # an invalid value of a valid keyword is reported by the extractor
    with pytest.raises(ValueError, match="bandwidth"):
        remove_orders(sig.x[:4000], FS, sig.rpm[:4000], [2.0], bandwidth=-1.0)
    with pytest.raises(ValueError, match="window_revs"):
        remove_orders(
            sig.x[:4000], FS, sig.rpm[:4000], [2.0], method="tracked-harmonics", window_revs=0.0
        )


def test_shafts_select_the_speed_column(method):
    # Order 2 of shaft 0 and order 3 of shaft 1 at 1.37 times the speed, noise only
    # besides: removing both needs shafts=[0, 1]; the default (shaft 0 for all) removes
    # the first only.
    t = np.arange(round(DURATION * FS)) / FS
    base = rpm_profiles(t, "linear", start=1500.0, stop=3000.0)
    rpm = np.column_stack([base, 1.37 * base])
    orders = [
        {"order": 2.0, "shaft": 0, "amplitude": 1.0},
        {"order": 3.0, "shaft": 1, "amplitude": 0.8},
    ]
    s2 = synth_runup(FS, rpm, orders, noise_rms=0.02, seed=5)
    kw = METHODS[method]
    y, comp = remove_orders(s2.x, FS, rpm, [2.0, 3.0], method=method, shafts=[0, 1], **kw)
    assert list(comp.shafts) == [0, 1]
    assert order_residual_energy(y, s2.x, FS, rpm[:, 0], 2.0) <= -20.0
    assert order_residual_energy(y, s2.x, FS, rpm[:, 1], 3.0) <= -20.0
    r = waveform_similarity(y, s2.stems["noise"], trim=EDGE)
    assert r["corr"] >= 0.99
    # Without shafts, order 3 is looked for at shaft 0's frequency and is not removed.
    y0, c0 = remove_orders(s2.x, FS, rpm, [2.0, 3.0], method=method, **kw)
    assert list(c0.shafts) == [0, 0]
    assert order_residual_energy(y0, s2.x, FS, rpm[:, 1], 3.0) > -3.0
    with pytest.raises(ValueError, match="shafts"):
        remove_orders(s2.x, FS, rpm, [2.0, 3.0], method=method, shafts=[0, 2], **kw)


# --------------------------------------------------------------------------
# Multichannel
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def three_channels():
    """Three channels on the same speed: different phases, noise and level (1, 0.5, 0.1)."""
    gains = (1.0, 0.5, 0.1)
    sigs = [_runup(seed=20 + i, gain=g, noise_rms=NOISE_RMS * g) for i, g in enumerate(gains)]
    x = np.column_stack([s.x for s in sigs])
    truth_removed = np.column_stack([_stems(s, REMOVED) for s in sigs])
    return sigs, x, truth_removed


def test_multichannel_residual_has_the_input_shape_and_matches_the_truth(three_channels, method):
    sigs, x, truth_removed = three_channels
    y, comp = remove_orders(x, FS, sigs[0].rpm, REMOVED, method=method, **METHODS[method])
    assert y.shape == x.shape
    assert comp.waveforms.shape == (x.shape[0], len(REMOVED), 3)
    assert comp.envelopes.shape == comp.waveforms.shape
    assert comp.total.shape == x.shape
    assert np.array_equal(y, x - comp.total)
    r = waveform_similarity(y, x - truth_removed, mixture=x, trim=EDGE)
    # the aggregate weights each channel by its energy; the quiet channel does not drag it
    assert r["weights"][0] > r["weights"][1] > r["weights"][2]
    assert r["corr"] >= 0.99
    assert np.all(r["corr_per_channel"] >= 0.99)
    assert r["baseline_corr"] < 0.8


def test_channels_are_processed_independently(three_channels, method):
    sigs, x, _ = three_channels
    kw = METHODS[method]
    rpm = sigs[0].rpm
    y, comp = remove_orders(x, FS, rpm, REMOVED, method=method, **kw)
    for c in range(3):
        y1, comp1 = remove_orders(x[:, c], FS, rpm, REMOVED, method=method, **kw)
        scale = np.abs(x[:, c]).max()
        assert np.allclose(y[:, c], y1, rtol=0, atol=1e-9 * scale)
        assert np.allclose(comp.envelopes[:, :, c], comp1.envelopes, rtol=0, atol=1e-9 * scale)


def test_inter_channel_amplitude_and_phase_relations_are_kept(sig, method):
    # channel B is an exact copy of channel A scaled by -0.5 (gain and polarity):
    # a linear per-channel operator must return the same relation for the residual
    x = np.column_stack([sig.x, -0.5 * sig.x])
    y, comp = remove_orders(x, FS, sig.rpm, REMOVED, method=method, **METHODS[method])
    scale = np.abs(sig.x).max()
    assert np.allclose(y[:, 1], -0.5 * y[:, 0], rtol=0, atol=1e-9 * scale)
    assert np.allclose(comp.envelopes[:, :, 1], -0.5 * comp.envelopes[:, :, 0], rtol=0, atol=1e-9 * scale)
