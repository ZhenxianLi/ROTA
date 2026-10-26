"""v0.2 validation cases T3, T4, T6, T8, T10, T12 and T13 (ROADMAP section 5.3) against analytic truth.

The signals come from ``validation/make_signals.py`` (built here in memory,
nothing is read from disk). Every expected amplitude and waveform is read from
the truth stored in the :class:`rota.synth.SynthSignal` (``order_amplitude``,
``order_stems``, ``stems``, ``rpm``), never from the parameter lists that built
the signal (ROADMAP section 6, item 19).

Tolerances (ROADMAP section 5.4, against analytic truth):

* order amplitude, stationary segments / interior: 0.2 dB
* order waveform correlation, ends removed: 0.99
* share of frames within 3 % of the speed (tacholess, steady or slow run-up): 90 %
* energy left at a removed order (T12): -20 dB

Where the extractors are far better than these tolerances the margins are
reported in ``validation/README.md``; the assertions keep the ROADMAP values.
Ends are removed where the method says its output is not to be trusted: the
Vold-Kalman filter after ``1.5 / bandwidth`` seconds (its free-end transient
has decayed to 3e-4), the tracked-harmonics extractor after ``window_revs``
revolutions (``params["edge_revs"]``), the order map by ignoring its first and
last three frames.
"""

import importlib.util
import sys
import warnings
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from rota._ridge import rpm_from_ridge
from rota.extract import (
    ResolutionWarning,
    order_waveform,
    remove_orders,
    resolution_check,
    tracked_harmonics,
)
from rota.maps import rpm_order_map
from rota.metrics import order_residual_energy, track_error, waveform_similarity
from rota.slices import order_slice

# --------------------------------------------------------------------------
# Signals: validation/make_signals.py is a script, not part of the package.
# --------------------------------------------------------------------------

_SCRIPT = Path(__file__).resolve().parents[1] / "validation" / "make_signals.py"
_spec = importlib.util.spec_from_file_location("make_signals_v02", _SCRIPT)
make_signals = importlib.util.module_from_spec(_spec)
sys.modules["make_signals_v02"] = make_signals
_spec.loader.exec_module(make_signals)

# ROADMAP section 5.4, "against analytic truth".
TOL_AMPLITUDE_DB = 0.2  # order amplitude
TOL_CORRELATION = 0.99  # order waveform
TOL_WITHIN_3PCT = 0.90  # tacholess speed, share of frames within 3 %
TOL_RESIDUAL_DB = -20.0  # energy left at a removed order (T12)

EDGE_FRAMES = 3  # frames dropped at each end of an order map
VK_EDGE_BANDWIDTHS = 1.5  # Vold-Kalman: edge = this / bandwidth seconds


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _db(a, ref):
    return 20.0 * np.log10(np.asarray(a, dtype=float) / np.asarray(ref, dtype=float))


def _frame_truth(sig, m, k):
    """Truth of order ``k`` (peak amplitude) in every frame of the order map ``m``.

    A short-time transform reads a window-weighted mean of the amplitude
    envelope: the Hann-weighted mean of ``order_amplitude[:, k]`` over the
    frame, with time samples weighted by the Hann value at their shaft angle
    and by the speed (time average -> angle average).
    """
    n = m.nperseg
    spr = n * m.resolution  # angle samples per revolution
    centre = np.interp(m.t, sig.t, sig.rev)
    lo = centre - 0.5 * (n - 1) / spr
    span = n / spr
    amp = sig.order_amplitude[:, k]
    out = np.empty(lo.size)
    for i, start in enumerate(lo):
        i0, i1 = np.searchsorted(sig.rev, [start, start + span])
        u = (sig.rev[i0:i1] - start) / span
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * u)) * sig.rpm[i0:i1]
        out[i] = np.sum(w * amp[i0:i1]) / np.sum(w)
    return out


def _vk_edge(fs, bandwidth):
    """Samples to drop at each end of a Vold-Kalman result."""
    return int(np.ceil(VK_EDGE_BANDWIDTHS * fs / bandwidth))


def _revs_slice(sig, revs):
    """Samples that are at least ``revs`` revolutions from both ends (of every shaft)."""
    rev = sig.rev if sig.rev.ndim == 2 else sig.rev[:, None]
    first = max(int(np.searchsorted(rev[:, s], revs)) for s in range(rev.shape[1]))
    last = min(int(np.searchsorted(rev[:, s], rev[-1, s] - revs)) for s in range(rev.shape[1]))
    return slice(first, last)


def _envelope_error_db(comp, sig, k, sl):
    """|dB| error of the envelope magnitude of component ``k`` against the truth amplitude."""
    return np.abs(_db(np.abs(comp.envelopes[sl, k]), sig.order_amplitude[sl, k]))


def _correlation(comp, sig, k, sl):
    return waveform_similarity(comp.waveforms[sl, k], sig.order_stems[sl, k])["corr"]


def _check_quiet(*args, **kwargs):
    """resolution_check with its warning silenced; returns the report."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResolutionWarning)
        return resolution_check(*args, **kwargs)


def _warns(func, *args, category=ResolutionWarning, **kwargs):
    """Call ``func``; return ``(result, True if it issued a warning of the category)``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    return result, any(issubclass(w.category, category) for w in caught)


def test_all_cases_are_reproducible_and_registered():
    for name in ("T3", "T4", "T6", "T8_r1005", "T8_r1020", "T8_r1040", "T10", "T12", "T13"):
        assert name in make_signals.CASES
    for build in (make_signals.t3, make_signals.t4, make_signals.t6, make_signals.t10):
        a, b = build(), build()
        assert np.array_equal(a.x, b.x) and np.array_equal(a.order_phase, b.order_phase)
        assert a.config["seed"] is not None
    a, b = make_signals.t8(1.005), make_signals.t8(1.005)
    assert np.array_equal(a.x, b.x)
    a, b = make_signals.t13(), make_signals.t13()
    ra, rb = make_signals.t13_references(a), make_signals.t13_references(b)
    assert np.array_equal(a.x, b.x)
    assert all(np.array_equal(ra[k], rb[k]) for k in ra)
    # T12 is the T2 signal
    assert np.array_equal(make_signals.t12().x, make_signals.t2().x)


def test_main_writes_the_v02_cases_and_the_t13_references(tmp_path):
    written = make_signals.main(tmp_path, ["T6", "T13"])
    assert set(written) == {"T6", "T13"}
    for npz, sidecar in written.values():
        assert npz.exists() and sidecar.exists()
    ref = make_signals.t13()
    with np.load(tmp_path / "T13_references.npz") as data:
        expected = make_signals.t13_references(ref)
        assert set(data.files) == set(expected)
        for k in expected:
            assert np.array_equal(data[k], expected[k])
    assert not (tmp_path / "T6_references.npz").exists()


# --------------------------------------------------------------------------
# T3: exponential run-down
# --------------------------------------------------------------------------

T3_EXTRACTORS = {
    "vold-kalman": {"bandwidth": 2.0},
    "tracked-harmonics": {"window_revs": 8.0},
}


@pytest.fixture(scope="module")
def t3():
    return make_signals.t3()


def _extract(sig, method, shafts=None, **kw):
    func = order_waveform if method == "vold-kalman" else tracked_harmonics
    return func(sig.x, sig.fs, sig.rpm, sig.orders, shafts=shafts, **kw)


def _interior_slice(sig, method, kw):
    if method == "vold-kalman":
        e = _vk_edge(sig.fs, kw["bandwidth"])
        return slice(e, sig.x.size - e)
    return _revs_slice(sig, kw["window_revs"])


def test_t3_speed_is_an_exponential_run_down(t3):
    r = t3.rpm
    assert r[0] == pytest.approx(4000.0) and r[-1] == pytest.approx(1000.0)
    assert np.all(np.diff(r) < 0)
    # constant relative rate (exponential), so the absolute slope changes by 4 times
    log_step = np.diff(np.log(r))
    assert np.allclose(log_step, log_step.mean(), rtol=1e-9)
    slope = np.diff(r) * t3.fs
    assert slope[0] / slope[-1] == pytest.approx(4.0, rel=1e-3)
    # the truth envelope of order 3 follows the gain law, not the parameter list
    k3 = list(t3.orders).index(3.0)
    assert t3.order_amplitude[:, k3].min() < 0.55 * t3.order_amplitude[:, k3].max()


def test_t3_order_map_slices_follow_the_run_down(t3):
    m = rpm_order_map(t3.x, t3.fs, t3.rpm, order_resolution=0.25, max_order=16)
    ok = np.zeros(m.t.size, dtype=bool)
    ok[EDGE_FRAMES:-EDGE_FRAMES] = True
    assert m.rpm[0] > 3500.0 and m.rpm[-1] < 1500.0  # the map covers the run-down
    assert ok.sum() >= 100
    for k, order in enumerate(t3.orders):
        s = order_slice(m, order)
        truth = _frame_truth(t3, m, k) / np.sqrt(2.0)  # peak -> RMS, the map's convention
        assert np.all(np.isfinite(s.amplitude[ok]))
        err = np.abs(_db(s.amplitude, truth))[ok]
        assert err.max() <= TOL_AMPLITUDE_DB, f"order {order}: {err.max():.3f} dB"


@pytest.mark.parametrize("method", list(T3_EXTRACTORS))
def test_t3_extractors_follow_the_run_down(t3, method):
    kw = T3_EXTRACTORS[method]
    comp = _extract(t3, method, **kw)
    sl = _interior_slice(t3, method, kw)
    assert sl.stop - sl.start > 0.5 * t3.x.size
    for k, order in enumerate(t3.orders):
        env = _envelope_error_db(comp, t3, k, sl)
        assert env.max() <= TOL_AMPLITUDE_DB, f"order {order}: envelope {env.max():.3f} dB"
        assert _correlation(comp, t3, k, sl) >= TOL_CORRELATION, f"order {order}"


def test_t3_ridge_tracking_of_the_run_down(t3):
    # The speed from the signal alone, over a range that holds the whole run-down.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no bound warning: the truth is inside the range
        res = rpm_from_ridge(t3.x, t3.fs, (800.0, 4500.0))
    truth = np.interp(res.t, t3.t, t3.rpm)
    stats = track_error(res.rpm, truth, at_bound=res.at_bound)
    assert stats["within"][0.03] >= TOL_WITHIN_3PCT
    assert stats["median_rel_error"] <= 0.01
    assert stats["bound_fraction"] == 0.0
    assert res.rpm[0] > 3500.0 and res.rpm[-1] < 1500.0  # it followed the decay


# --------------------------------------------------------------------------
# T4: two shafts with crossing orders
# --------------------------------------------------------------------------

T4_BANDWIDTH = 2.0
T4_WINDOW = 0.5  # s, half width of the window around a crossing


@pytest.fixture(scope="module")
def t4():
    return make_signals.t4()


@pytest.fixture(scope="module")
def t4_extractions(t4):
    shafts = [c["shaft"] for c in t4.config["orders"]]
    joint = order_waveform(t4.x, t4.fs, t4.rpm, t4.orders, shafts=shafts, bandwidth=T4_BANDWIDTH)
    alone = order_waveform(
        t4.x, t4.fs, t4.rpm, t4.orders, shafts=shafts, bandwidth=T4_BANDWIDTH, joint=False
    )
    return joint, alone


def _crossings(sig):
    """Truth crossings: ``(i, j, t)`` for every pair of order frequencies that change order."""
    out = []
    f = sig.order_frequency
    for i in range(f.shape[1]):
        for j in range(i + 1, f.shape[1]):
            d = f[:, i] - f[:, j]
            for k in np.flatnonzero(d[:-1] * d[1:] < 0):
                frac = abs(d[k]) / (abs(d[k]) + abs(d[k + 1]))
                out.append((i, j, float(sig.t[k] + frac / sig.fs)))
    return out


def test_t4_design_has_three_crossing_events(t4):
    assert t4.rpm.shape == (t4.x.size, 2)
    cross = _crossings(t4)
    times = sorted(c[2] for c in cross)
    assert len(cross) == 4  # 2.4 s, 4.0 s (two pairs), 5.6 s
    assert times == pytest.approx([2.4, 4.0, 4.0, 5.6], abs=1e-3)
    # the pairs are shaft 0 x shaft 1 (never two orders of one shaft)
    shafts = [c["shaft"] for c in t4.config["orders"]]
    assert all(shafts[i] != shafts[j] for i, j, _ in cross)
    # the single-order criterion calls these unresolved, at the crossing times
    rep = _check_quiet(
        t4.rpm, t4.fs, [c["order"] for c in t4.config["orders"]], shafts=shafts,
        bandwidth=T4_BANDWIDTH,
    )
    assert not rep.ok and rep.recommended_bandwidth is None
    reported = sorted(t for p in rep.pairs for t in p["crossings"])
    assert reported == pytest.approx(times, abs=1e-3)


def test_t4_joint_filter_recovers_both_crossing_orders(t4, t4_extractions):
    joint, _ = t4_extractions
    sl = slice(_vk_edge(t4.fs, T4_BANDWIDTH), t4.x.size - _vk_edge(t4.fs, T4_BANDWIDTH))
    for k, cfg in enumerate(t4.config["orders"]):
        label = f"order {cfg['order']:g} of shaft {cfg['shaft']}"
        assert _correlation(joint, t4, k, sl) >= TOL_CORRELATION, label
        assert _envelope_error_db(joint, t4, k, sl).max() <= TOL_AMPLITUDE_DB, label


def test_t4_joint_filter_is_clearly_better_through_the_crossings(t4, t4_extractions):
    joint, alone = t4_extractions
    for i, j, tc in _crossings(t4):
        a, b = int((tc - T4_WINDOW) * t4.fs), int((tc + T4_WINDOW) * t4.fs)
        for k in (i, j):
            ref = t4.order_stems[a:b, k]
            sj = waveform_similarity(joint.waveforms[a:b, k], ref)
            sa = waveform_similarity(alone.waveforms[a:b, k], ref)
            label = f"component {k} at the crossing t = {tc:.1f} s"
            assert sj["corr"] >= TOL_CORRELATION, label
            assert sa["corr"] < TOL_CORRELATION, label  # one filter per order is not enough
            assert sj["nrmse"] < 0.05 and sa["nrmse"] > 0.1, label
            assert sa["nrmse"] > 10.0 * sj["nrmse"], label


# --------------------------------------------------------------------------
# T6: fast run-up
# --------------------------------------------------------------------------

T6_BANDWIDTH = 16.0  # Hz: pass band 32 Hz, just below the 33 Hz between orders 2 and 4 at 1000 rpm
T6_WINDOW_REVS = 4.0
T6_SLEWS = {10.0: 500.0, 2.5: 2000.0, 1.0: 5000.0, 0.5: 10000.0}  # duration (s) -> rpm/s


@pytest.fixture(scope="module")
def t6():
    return make_signals.t6()


def _t6_vk(sig):
    comp = order_waveform(sig.x, sig.fs, sig.rpm, sig.orders, bandwidth=T6_BANDWIDTH)
    e = _vk_edge(sig.fs, T6_BANDWIDTH)
    return comp, slice(e, sig.x.size - e)


def _t6_map_error(sig):
    """Largest |dB| error of the order slices of a 0.25-order-resolution map, interior frames."""
    m = rpm_order_map(sig.x, sig.fs, sig.rpm, order_resolution=0.25, max_order=16)
    worst = 0.0
    for k, order in enumerate(sig.orders):
        s = order_slice(m, order)
        truth = _frame_truth(sig, m, k) / np.sqrt(2.0)
        err = np.abs(_db(s.amplitude, truth))[EDGE_FRAMES:-EDGE_FRAMES]
        assert np.all(np.isfinite(err)) and err.size >= 20
        worst = max(worst, float(err.max()))
    return worst


def test_t6_design_is_a_fast_sweep(t6):
    slew = np.diff(t6.rpm) * t6.fs
    assert slew.mean() == pytest.approx(5000.0, rel=1e-3)
    assert t6.rpm[0] == pytest.approx(1000.0) and t6.rpm[-1] == pytest.approx(6000.0)
    # the first order-map frame (4 revolutions) already spans a large speed range
    m = rpm_order_map(t6.x, t6.fs, t6.rpm, order_resolution=0.25, max_order=16)
    assert (m.rpm_max[0] - m.rpm_min[0]) / m.rpm_min[0] > 0.8
    # the closest orders are resolved by the bandwidth chosen
    assert _check_quiet(t6.rpm, t6.fs, t6.orders, bandwidth=T6_BANDWIDTH).ok


def test_t6_vold_kalman_at_the_high_slew_rate(t6):
    comp, sl = _t6_vk(t6)
    for k, order in enumerate(t6.orders):
        assert _correlation(comp, t6, k, sl) >= TOL_CORRELATION, f"order {order}"
        assert _envelope_error_db(comp, t6, k, sl).max() <= TOL_AMPLITUDE_DB, f"order {order}"


def test_t6_tracked_harmonics_at_the_high_slew_rate(t6):
    comp = tracked_harmonics(t6.x, t6.fs, t6.rpm, t6.orders, window_revs=T6_WINDOW_REVS)
    sl = _revs_slice(t6, T6_WINDOW_REVS)
    for k, order in enumerate(t6.orders):
        assert _correlation(comp, t6, k, sl) >= TOL_CORRELATION, f"order {order}"
        assert _envelope_error_db(comp, t6, k, sl).max() <= TOL_AMPLITUDE_DB, f"order {order}"


def test_t6_order_map_at_the_high_slew_rate(t6):
    assert _t6_map_error(t6) <= TOL_AMPLITUDE_DB


@pytest.mark.parametrize("duration", list(T6_SLEWS))
def test_t6_slew_sweep_of_the_cot_map_and_the_vold_kalman_filter(duration):
    # The same speed range and amplitude laws at 500 ... 10000 rpm/s. Computed order
    # tracking and the Vold-Kalman filter hold the tolerance over the whole sweep; the
    # tracked-harmonics window limit (4 revolutions) is reported in validation/README.md.
    sig = make_signals.t6(duration=duration)
    assert np.diff(sig.rpm).mean() * sig.fs == pytest.approx(T6_SLEWS[duration], rel=1e-3)
    comp, sl = _t6_vk(sig)
    for k in range(sig.orders.size):
        assert _correlation(comp, sig, k, sl) >= TOL_CORRELATION
        assert _envelope_error_db(comp, sig, k, sl).max() <= TOL_AMPLITUDE_DB
    assert _t6_map_error(sig) <= TOL_AMPLITUDE_DB


# --------------------------------------------------------------------------
# T8: twin shafts, speed ratio 1.005 / 1.02 / 1.04
# --------------------------------------------------------------------------

T8_RATIOS = (1.005, 1.02, 1.04)
T8_BANDWIDTH = 1.0  # Hz: pass band 2 Hz
T8_WINDOW_REVS = 32.0  # about 0.8 s at the 2400 rpm of the start
T8_ORDERS = [4.0, 4.0]
T8_SHAFTS = [0, 1]


@pytest.fixture(scope="module")
def t8():
    return {r: make_signals.t8(r) for r in T8_RATIOS}


def _t8_check(sig, **setting):
    return resolution_check(sig.rpm, sig.fs, T8_ORDERS, shafts=T8_SHAFTS, **setting)


def _t8_corr(sig, comp, sl):
    return [_correlation(comp, sig, k, sl) for k in range(2)]


def test_t8_design_beat_frequency(t8):
    for ratio, sig in t8.items():
        df = np.abs(sig.order_frequency[:, 0] - sig.order_frequency[:, 1])
        assert df[0] == pytest.approx(4.0 * (ratio - 1.0) * 2400.0 / 60.0, rel=1e-9)
        assert df.min() == df[0]
        assert np.allclose(sig.rpm[:, 1], ratio * sig.rpm[:, 0], rtol=1e-12)
        # amplitude-modulated lines: the truth envelope is not constant
        assert sig.order_amplitude[:, 0].std() > 0.1 and sig.order_amplitude[:, 1].std() > 0.1


@pytest.mark.parametrize(
    "setting, warns_at",
    [
        ({"bandwidth": T8_BANDWIDTH}, {1.005: True, 1.02: False, 1.04: False}),
        ({"window_revs": T8_WINDOW_REVS}, {1.005: True, 1.02: False, 1.04: False}),
    ],
    ids=["vold-kalman", "tracked-harmonics"],
)
def test_t8_resolution_check_warns_exactly_when_the_twins_are_unresolved(t8, setting, warns_at):
    for ratio, sig in t8.items():
        report, warned = _warns(_t8_check, sig, **setting)
        assert warned is warns_at[ratio], f"ratio {ratio}"
        assert report.ok is (not warns_at[ratio])
        if warns_at[ratio]:
            assert any("shaft 1" in w for w in report.warnings)
            # the pair's smallest separation is the beat frequency at the start
            assert report.pairs[0]["min_separation_hz"] == pytest.approx(
                np.abs(sig.order_frequency[:, 0] - sig.order_frequency[:, 1]).min(), rel=1e-9
            )


def test_t8_a_resolved_setting_separates_the_twins(t8):
    # Vold-Kalman, joint: both shafts at once; tracked harmonics with a 32-revolution window.
    for ratio in (1.02, 1.04):
        sig = t8[ratio]
        assert _t8_check(sig, bandwidth=T8_BANDWIDTH).ok
        comp = order_waveform(
            sig.x, sig.fs, sig.rpm, T8_ORDERS, shafts=T8_SHAFTS, bandwidth=T8_BANDWIDTH
        )
        e = _vk_edge(sig.fs, T8_BANDWIDTH)
        sl = slice(e, sig.x.size - e)
        assert min(_t8_corr(sig, comp, sl)) >= TOL_CORRELATION, f"vold-kalman, ratio {ratio}"
        for k in range(2):
            assert _envelope_error_db(comp, sig, k, sl).max() <= TOL_AMPLITUDE_DB
        assert _t8_check(sig, window_revs=T8_WINDOW_REVS).ok
        comp = tracked_harmonics(
            sig.x, sig.fs, sig.rpm, T8_ORDERS, shafts=T8_SHAFTS, window_revs=T8_WINDOW_REVS
        )
        sl = _revs_slice(sig, T8_WINDOW_REVS)
        assert min(_t8_corr(sig, comp, sl)) >= TOL_CORRELATION, f"tracked, ratio {ratio}"


def test_t8_half_percent_twins_the_warning_and_what_each_extractor_achieves(t8):
    sig = t8[1.005]
    _, warned = _warns(_t8_check, sig, bandwidth=T8_BANDWIDTH)
    assert warned  # the ROADMAP criterion for T8 (section 6, item 7)
    _, warned = _warns(_t8_check, sig, window_revs=T8_WINDOW_REVS)
    assert warned
    e = _vk_edge(sig.fs, T8_BANDWIDTH)
    sl = slice(e, sig.x.size - e)

    # What a per-order filter (joint=False) and a 32-revolution window achieve: the
    # neighbour leaks in, the warning was right. Measured: Vold-Kalman 0.84 and 0.71,
    # tracked harmonics 0.91 and 0.80.
    alone = order_waveform(
        sig.x, sig.fs, sig.rpm, T8_ORDERS, shafts=T8_SHAFTS, bandwidth=T8_BANDWIDTH, joint=False
    )
    assert max(_t8_corr(sig, alone, sl)) < 0.9
    comp = tracked_harmonics(
        sig.x, sig.fs, sig.rpm, T8_ORDERS, shafts=T8_SHAFTS, window_revs=T8_WINDOW_REVS
    )
    assert max(_t8_corr(sig, comp, _revs_slice(sig, T8_WINDOW_REVS))) < 0.95

    # The joint least-squares solution still separates lines 0.8 to 1 Hz apart when the
    # envelopes vary slowly compared with the beat (0.4 and 0.25 Hz here; measured 0.9988
    # and 0.9982): the check is a criterion for single-order filters and windows.
    joint = order_waveform(sig.x, sig.fs, sig.rpm, T8_ORDERS, shafts=T8_SHAFTS, bandwidth=T8_BANDWIDTH)
    corr = _t8_corr(sig, joint, sl)
    assert min(corr) >= TOL_CORRELATION
    assert min(corr) > max(_t8_corr(sig, alone, sl)) + 0.1


def test_t8_envelope_wall_when_the_envelope_varies_at_the_beat_rate():
    # Amplitude modulation at 0.8 and 1.2 Hz is as fast as the beat of the 0.5 % twins
    # (0.8 to 1 Hz). The check names it a wall (no setting both resolves the pair and
    # follows the envelope) and no bandwidth of the joint filter reaches the tolerance;
    # 4 % twins, 6.4 Hz apart, are separated by a bandwidth wide enough for the envelope.
    am = (0.8, 1.2)
    twin = make_signals.t8(1.005, am_freq=am)
    far = make_signals.t8(1.04, am_freq=am)
    report, warned = _warns(_t8_check, twin, bandwidth=0.4, envelope_rate=max(am))
    assert warned and not report.ok and report.pairs[0]["resolved"]  # resolved, but a wall
    assert any("bias-variance wall" in w for w in report.warnings)
    assert report.recommended_bandwidth is None
    assert _t8_check(far, bandwidth=3.0, envelope_rate=max(am)).ok

    def best_correlation(sig, bandwidths):
        best = 0.0
        for b in bandwidths:
            comp = order_waveform(sig.x, sig.fs, sig.rpm, T8_ORDERS, shafts=T8_SHAFTS, bandwidth=b)
            e = _vk_edge(sig.fs, b)
            best = max(best, min(_t8_corr(sig, comp, slice(e, sig.x.size - e))))
        return best

    assert 0.9 < best_correlation(twin, (0.4, 1.0, 3.0)) < TOL_CORRELATION
    assert best_correlation(far, (3.0,)) >= TOL_CORRELATION


# --------------------------------------------------------------------------
# T10: true speed outside the search range
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def t10():
    return make_signals.t10()


@pytest.mark.parametrize("rpm_range", [make_signals.T10_RANGE, (1500.0, 2500.0)])
def test_t10_out_of_range_frames_are_flagged_and_in_range_frames_are_right(t10, rpm_range):
    lo, hi = rpm_range
    with pytest.warns(RuntimeWarning, match="bound of the search range"):
        res = rpm_from_ridge(t10.x, t10.fs, rpm_range)
    truth = np.interp(res.t, t10.t, t10.rpm)
    inside = (truth >= lo) & (truth <= hi)
    outside = ~inside
    far = (truth < lo * 0.98) | (truth > hi * 1.02)  # clearly outside, not a boundary frame
    assert inside.sum() >= 80 and far.sum() >= 40  # both populations are large

    # The estimator never returns a boundary value silently: every frame that is clearly
    # outside the range is flagged, and the flagged share is the outside share.
    assert res.at_bound[far].all()
    assert res.at_bound[outside].mean() >= 0.95
    assert res.bound_fraction == pytest.approx(outside.mean(), abs=0.03)
    # The path cannot leave the range, and far outside it need not sit at a boundary (it
    # may lock to an octave inside the range); the guard bands beyond the range are what
    # report that the data prefer a speed outside.
    assert res.rpm.min() >= lo * (1.0 - 1e-9) and res.rpm.max() <= hi * (1.0 + 1e-9)
    assert res.beyond[far].mean() >= 0.8

    # Frames whose speed is inside the range meet the tacholess tolerance.
    stats = track_error(res.rpm[inside], truth[inside])
    assert stats["within"][0.03] >= TOL_WITHIN_3PCT
    assert res.at_bound[inside].mean() <= 0.05  # a frame next to the bound may be flagged


# --------------------------------------------------------------------------
# T12: order removal on the T2 signal
# --------------------------------------------------------------------------

T12_SETTINGS = {
    "vold-kalman": {"bandwidth": 2.0},
    "tracked-harmonics": {"window_revs": 8.0},
}


@pytest.fixture(scope="module")
def t12():
    return make_signals.t12()


@pytest.fixture(scope="module", params=list(T12_SETTINGS))
def t12_removed(request, t12):
    method = request.param
    orders = [float(o) for o in t12.orders]
    y, comp = remove_orders(
        t12.x, t12.fs, t12.rpm, orders, method=method, **T12_SETTINGS[method]
    )
    return method, y, comp


def test_t12_is_the_t2_signal(t12):
    assert list(t12.orders) == [1.0, 2.0, 3.0, 4.5]
    assert t12.config["resonances"][0]["freq"] == 200.0


@pytest.mark.parametrize("order", [1.0, 2.0, 3.0, 4.5])
def test_t12_removed_orders_leave_less_than_minus_20_db(t12, t12_removed, order):
    method, y, _ = t12_removed
    left = order_residual_energy(y, t12.x, t12.fs, t12.rpm, order)
    assert left <= TOL_RESIDUAL_DB, f"{method}, order {order}: {left:.1f} dB"


def test_t12_the_residual_is_the_tone_and_noise(t12, t12_removed):
    _, y, comp = t12_removed
    background = t12.stems["tones"] + t12.stems["noise"]
    edge = round(1.0 * t12.fs)
    r = waveform_similarity(y, background, mixture=t12.x, trim=edge)
    assert r["corr"] >= TOL_CORRELATION and r["nrmse"] < 0.05
    assert r["baseline_corr"] < 0.5  # the unprocessed signal is dominated by the orders
    assert np.array_equal(y, t12.x - comp.total)


# --------------------------------------------------------------------------
# T13: errors in the speed reference
# --------------------------------------------------------------------------

T13_SETTINGS = {
    "vold-kalman": {"bandwidth": 2.0},
    "tracked-harmonics": {"window_revs": 8.0},
}
T13_BIAS_SWEEP = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02)
T13_EDGE = 1.0  # s dropped at each end (VK: 0.75 s, tracked harmonics: 8 revolutions < 0.5 s)


@pytest.fixture(scope="module")
def t13():
    return make_signals.t13()


@pytest.fixture(scope="module")
def t13_refs(t13):
    return make_signals.t13_references(t13)


def _t13_extract(sig, rpm_ref, method):
    func = order_waveform if method == "vold-kalman" else tracked_harmonics
    return func(sig.x, sig.fs, rpm_ref, sig.orders, **T13_SETTINGS[method])


def _t13_metrics(sig, comp):
    """Worst order: median and 95th percentile of the |dB| envelope error, lowest correlation."""
    e = round(T13_EDGE * sig.fs)
    sl = slice(e, sig.x.size - e)
    err = np.abs(_db(np.abs(comp.envelopes[sl]), sig.order_amplitude[sl]))  # (n, K)
    corr = np.array([_correlation(comp, sig, k, sl) for k in range(sig.orders.size)])
    nrmse = np.array(
        [waveform_similarity(comp.waveforms[sl, k], sig.order_stems[sl, k])["nrmse"]
         for k in range(sig.orders.size)]
    )
    return {
        "env_median": float(np.median(err, axis=0).max()),
        "env_p95": float(np.percentile(err, 95, axis=0).max()),
        "per_order_median": np.median(err, axis=0),
        "corr_min": float(corr.min()),
        "nrmse_max": float(nrmse.max()),
        "finite": bool(np.all(np.isfinite(comp.envelopes)) and np.all(np.isfinite(comp.waveforms))),
    }


@pytest.fixture(scope="module")
def t13_results(t13, t13_refs):
    out = {}
    for method in T13_SETTINGS:
        for name, ref in t13_refs.items():
            out[method, name] = _t13_metrics(t13, _t13_extract(t13, ref, method))
    return out


@pytest.fixture(scope="module")
def t13_bias_sweep(t13):
    return {
        method: [
            _t13_metrics(t13, _t13_extract(t13, t13.rpm * (1.0 + b), method))
            for b in T13_BIAS_SWEEP
        ]
        for method in T13_SETTINGS
    }


def test_t13_references_are_as_documented(t13, t13_refs):
    rpm = t13.rpm
    assert set(t13_refs) == {"truth", "bias", "jitter", "step"}
    assert np.array_equal(t13_refs["truth"], rpm)
    assert np.allclose(t13_refs["bias"] / rpm, 1.0 + make_signals.T13_BIAS, rtol=0, atol=1e-12)
    rel = t13_refs["jitter"] / rpm - 1.0
    assert np.sqrt(np.mean(rel**2)) == pytest.approx(make_signals.T13_JITTER_RMS, rel=1e-9)
    assert abs(rel.mean()) < 1e-9 and np.abs(rel).max() < 6.0 * make_signals.T13_JITTER_RMS
    f, psd = signal.welch(rel, t13.fs, nperseg=8192)  # low-passed: little power above the band edge
    assert psd[f > 3.0 * make_signals.T13_JITTER_BAND].sum() < 1e-3 * psd.sum()
    step = t13_refs["step"] / rpm - 1.0
    after = t13.t >= make_signals.T13_STEP_TIME
    assert np.allclose(step[after], make_signals.T13_STEP) and np.all(step[~after] == 0.0)
    for ref in t13_refs.values():
        assert ref.shape == rpm.shape and np.all(ref > 0)


@pytest.mark.parametrize("method", list(T13_SETTINGS))
def test_t13_exact_speed_meets_the_tolerances(t13_results, method):
    r = t13_results[method, "truth"]
    assert r["env_median"] <= TOL_AMPLITUDE_DB and r["env_p95"] <= TOL_AMPLITUDE_DB
    assert r["corr_min"] >= TOL_CORRELATION


@pytest.mark.parametrize("name", ["bias", "jitter", "step"])
@pytest.mark.parametrize("method", list(T13_SETTINGS))
def test_t13_speed_errors_degrade_the_extraction_gracefully(t13_results, method, name):
    r = t13_results[method, name]
    exact = t13_results[method, "truth"]
    assert r["finite"]  # no NaN or infinity whatever the error
    # worse than the exact speed (or equal), never absurd: the order is still there
    assert r["env_median"] >= exact["env_median"] - 0.005
    assert r["corr_min"] >= 0.9
    assert r["nrmse_max"] < 0.5
    # The errors of ROADMAP T13 at their default size leave a usable extraction.
    assert r["env_median"] < 2.0 and r["env_p95"] < 20.0
    if name in ("bias", "jitter"):  # 0.5 % bias, 0.5 % rms jitter
        assert r["corr_min"] >= 0.98 and r["env_median"] < 0.7


@pytest.mark.parametrize("method", list(T13_SETTINGS))
def test_t13_error_grows_monotonically_with_the_bias(t13_bias_sweep, method):
    rows = t13_bias_sweep[method]
    med = [r["env_median"] for r in rows]
    corr = [r["corr_min"] for r in rows]
    nrmse = [r["nrmse_max"] for r in rows]
    slack = 0.005  # dB: below this the error is the noise floor of the exact-speed extraction
    assert all(r["finite"] for r in rows)
    assert all(b - a >= -slack for a, b in pairwise(med)), med
    assert all(b - a <= 1e-4 for a, b in pairwise(corr)), corr
    assert all(b - a >= -1e-3 for a, b in pairwise(nrmse)), nrmse
    assert med[-1] > 10.0 * med[T13_BIAS_SWEEP.index(0.0025)]  # a steep, not a flat, dependence
    assert med[0] <= 0.02 and med[-1] > 1.0


def _t13_bias_error(sig, method, bias):
    return _t13_metrics(sig, _t13_extract(sig, sig.rpm * (1.0 + bias), method))["env_median"]


@pytest.mark.parametrize("bandwidth", [1.0, 2.0, 4.0])
def test_t13_guidance_for_the_vold_kalman_filter(t13, bandwidth):
    # A bias d moves the order off its carrier by d f; the filter attenuates a tone at
    # that offset (|H| = 1 / (1 + 0.414 (offset / B)^4)), so the envelope is within 0.2 dB
    # while d * f_top <= 0.5 B, f_top the highest order frequency (375 Hz).
    f_top = float(np.max(t13.order_frequency))
    d = 0.5 * bandwidth / f_top
    kw = {"bandwidth": bandwidth}
    ok = _t13_metrics(t13, order_waveform(t13.x, t13.fs, t13.rpm * (1.0 + d), t13.orders, **kw))
    bad = _t13_metrics(t13, order_waveform(t13.x, t13.fs, t13.rpm * (1.0 + 4.0 * d), t13.orders, **kw))
    assert ok["env_median"] <= TOL_AMPLITUDE_DB
    assert bad["env_median"] > TOL_AMPLITUDE_DB


@pytest.mark.parametrize("window_revs", [4.0, 8.0, 16.0])
def test_t13_guidance_for_tracked_harmonics(t13, window_revs):
    # The offset in orders is d * order; a Hann window of W revolutions loses amplitude
    # as (d * order * W)^2, about 10 dB x (d k W)^2, and the median envelope error stays
    # within 0.2 dB while d * k_max * W <= 0.12.
    d = 0.12 / (float(np.max(t13.orders)) * window_revs)
    kw = {"window_revs": window_revs}
    ok = _t13_metrics(t13, tracked_harmonics(t13.x, t13.fs, t13.rpm * (1.0 + d), t13.orders, **kw))
    bad = _t13_metrics(t13, tracked_harmonics(t13.x, t13.fs, t13.rpm * (1.0 + 4.0 * d), t13.orders, **kw))
    assert ok["env_median"] <= TOL_AMPLITUDE_DB
    assert bad["env_median"] > TOL_AMPLITUDE_DB


def test_t13_degradation_table(t13_results, t13_bias_sweep):
    """Print the table that validation/README.md quotes (``pytest -s -k t13_degradation``)."""
    rows = [
        "reference | VK med/p95 dB, min corr | TH med/p95 dB, min corr",
    ]
    for name in ("truth", "bias", "jitter", "step"):
        cells = []
        for method in T13_SETTINGS:
            r = t13_results[method, name]
            cells.append(f"{r['env_median']:.3f}/{r['env_p95']:.3f}, {r['corr_min']:.4f}")
        rows.append(f"{name} | " + " | ".join(cells))
    rows.append("bias sweep (worst-order median envelope error, dB / min correlation)")
    for i, b in enumerate(T13_BIAS_SWEEP):
        cells = []
        for method in T13_SETTINGS:
            r = t13_bias_sweep[method][i]
            cells.append(f"{r['env_median']:.3f}, {r['corr_min']:.4f}")
        rows.append(f"{100 * b:.2f} % | " + " | ".join(cells))
    print("\n" + "\n".join(rows))
    assert len(rows) == 4 + 2 + len(T13_BIAS_SWEEP)
