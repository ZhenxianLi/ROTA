"""Tests of rota._application against analytic ground truth from rota.synth."""

import numpy as np
import pytest

from rota._application import (
    frequency_slice,
    frequency_to_order,
    metric_vs_rpm,
    order_share,
    overall_level,
)
from rota.maps import rpm_frequency_map, rpm_order_map
from rota.synth import rpm_profiles, synth_runup

FS = 8000.0


def _db(a, ref):
    return 20.0 * np.log10(np.asarray(a) / ref)


# --------------------------------------------------------------------------
# frequency_slice
# --------------------------------------------------------------------------


def _tone_runup(f0=437.3, amp=2.0, with_order=True):
    n = int(4 * FS)
    t = np.arange(n) / FS
    rpm = rpm_profiles(t, "linear", start=1200.0, stop=3600.0)
    orders = [{"order": 2.0, "amplitude": 1.0}] if with_order else []
    s = synth_runup(FS, rpm, orders, tones=[{"freq": f0, "amplitude": amp}], seed=3)
    return s.x, rpm


@pytest.mark.parametrize("kind", ["frequency", "order"])
@pytest.mark.parametrize("amplitude, scale", [("rms", 1 / np.sqrt(2)), ("peak", 1.0)])
def test_frequency_slice_of_fixed_tone_is_constant_vs_rpm(kind, amplitude, scale):
    f0, amp = 437.3, 2.0
    x, rpm = _tone_runup(f0, amp)
    if kind == "frequency":
        m = rpm_frequency_map(x, FS, rpm, resolution=5.0, amplitude=amplitude)
    else:
        m = rpm_order_map(
            x, FS, rpm, max_order=40, order_resolution=0.25, amplitude=amplitude
        )
    s = frequency_slice(m, f0)
    ok = np.isfinite(s.amplitude)
    assert ok.sum() > 50
    assert s.domain == kind and s.amplitude_convention == amplitude
    err = _db(s.amplitude[ok], amp * scale)
    assert np.all(np.abs(err) < 0.1)
    assert np.ptp(err) < 0.1  # constant versus rpm
    assert s.amplitude.shape == m.rpm.shape and s.freq == f0


def test_frequency_slice_empty_frequency_and_axis_edges():
    x, rpm = _tone_runup(437.3, 2.0, with_order=False)
    m = rpm_frequency_map(x, FS, rpm, resolution=5.0, amplitude="rms")
    # band reaching beyond Nyquist: NaN everywhere
    assert np.all(np.isnan(frequency_slice(m, 3995.0).amplitude))
    # an empty frequency has no tone: far below the tone level
    s = frequency_slice(m, 1500.0)
    assert np.all(s.amplitude < 1e-3)
    # order map: low speed pushes the order of a fixed frequency out of the axis
    mo = rpm_order_map(x, FS, rpm, max_order=10, order_resolution=0.25)
    so = frequency_slice(mo, 437.3)
    assert np.isnan(so.amplitude[0]) and np.isfinite(so.amplitude[-1])


def test_frequency_slice_finds_structural_resonance_position():
    n = int(4 * FS)
    t = np.arange(n) / FS
    rpm = rpm_profiles(t, "linear", start=1200.0, stop=3600.0)
    f0 = 200.0
    s = synth_runup(
        FS,
        rpm,
        [{"order": 5.0, "amplitude": 1.0}],
        resonances=[{"freq": f0, "q": 20.0, "gain": 4.0}],
        seed=1,
    )
    m = rpm_frequency_map(s.x, FS, rpm, resolution=5.0, amplitude="peak")
    # a fixed-frequency slice of a sweeping order peaks where the order crosses it
    sl = frequency_slice(m, f0, bandwidth=30.0)
    k = int(np.nanargmax(sl.amplitude))
    assert abs(sl.rpm[k] - 60.0 * f0 / 5.0) < 100.0
    assert np.all(sl.bandwidth == 30.0)


def test_frequency_slice_validation():
    x, rpm = _tone_runup()
    m = rpm_frequency_map(x, FS, rpm, resolution=5.0)
    with pytest.raises(ValueError):
        frequency_slice(m, -1.0)
    with pytest.raises(ValueError):
        frequency_slice(m, 100.0, bandwidth=0.0)


# --------------------------------------------------------------------------
# overall_level
# --------------------------------------------------------------------------


def _sine_runup(amp=1.5, f=1000.0, duration=2.0):
    n = int(duration * FS)
    t = np.arange(n) / FS
    rpm = np.linspace(1000.0, 3000.0, n)
    return amp * np.sin(2 * np.pi * f * t), rpm


def test_overall_level_of_known_rms():
    amp = 1.5
    x, rpm = _sine_runup(amp)
    r = overall_level(x, FS, rpm)  # 0.1 s frames hold 100 whole cycles
    expect = 20 * np.log10(amp / np.sqrt(2) / 20e-6)
    assert r.level.shape == (20,) and r.weighting is None
    assert np.allclose(r.level, expect, atol=1e-9)
    assert np.allclose(r.t, 0.05 + 0.1 * np.arange(20))
    assert np.all(np.diff(r.rpm) > 0) and r.binned_level is None
    # reference and white-noise RMS
    rng = np.random.default_rng(0)
    w = rng.standard_normal(int(4 * FS))
    w *= 0.2 / np.sqrt(np.mean(w**2))
    r2 = overall_level(w, FS, 2000.0, frame=4.0, ref=1.0)
    assert r2.level.shape == (1,) and abs(r2.level[0] - 20 * np.log10(0.2)) < 1e-9


def test_overall_level_multichannel_and_silence():
    x, rpm = _sine_runup(1.0)
    X = np.stack([x, 2 * x, 0 * x], axis=1)
    r = overall_level(X, FS, rpm)
    assert r.level.shape == (20, 3)
    assert np.allclose(r.level[:, 1] - r.level[:, 0], 20 * np.log10(2))
    assert np.all(np.isneginf(r.level[:, 2]))


def test_overall_level_binning_is_energy_mean_with_counts():
    # amplitude alternates between frames: 1 and 3 -> mean square (0.5 + 4.5) / 2
    n_frame = int(0.1 * FS)
    n_fr = 20
    t = np.arange(n_frame * n_fr) / FS
    amp = np.repeat(np.where(np.arange(n_fr) % 2 == 0, 1.0, 3.0), n_frame)
    x = amp * np.sin(2 * np.pi * 1000.0 * t)
    rpm = np.repeat(np.where(np.arange(n_fr) < 10, 1500.0, 2500.0), n_frame)
    r = overall_level(x, FS, rpm, rpm_bins=[1000.0, 2000.0, 3000.0])
    assert list(r.counts) == [10, 10]
    expect = 10 * np.log10(2.5 / 20e-6**2)
    assert np.allclose(r.binned_level, expect, atol=1e-9)
    assert np.allclose(r.centres, [1500.0, 2500.0])
    # an energy mean exceeds the arithmetic mean of the decibel values
    assert expect > np.mean(r.level[:10]) + 0.5
    # bin width form, an empty bin and frames outside the edges
    r2 = overall_level(x, FS, rpm, rpm_bins=500.0)
    assert r2.edges[0] == 1500.0 and r2.counts.sum() == n_fr
    r3 = overall_level(x, FS, rpm, rpm_bins=[0.0, 1000.0, 2000.0, 2200.0])
    assert list(r3.counts) == [0, 10, 0] and np.isnan(r3.binned_level[0])


def test_overall_level_validation():
    x, rpm = _sine_runup()
    with pytest.raises(ValueError):
        overall_level(x, FS, rpm, weighting="B")
    with pytest.raises(ValueError):
        overall_level(x, FS, rpm, frame=10.0)
    with pytest.raises(ValueError):
        overall_level(x, FS, rpm, rpm_bins=[2.0, 1.0])
    with pytest.raises(ValueError):
        overall_level(x, FS, rpm[:-1])


@pytest.mark.parametrize("weighting", ["A", "C"])
def test_overall_level_weighting(weighting):
    tonal = pytest.importorskip("rota.tonal")
    # the digital weighting filters are specified for audio sampling rates (>= 44.1 kHz)
    fs = 48000.0
    amp = 1.0
    n = int(4 * fs)
    t = np.arange(n) / fs
    rpm = np.full(n, 2000.0)
    ref_level = 20 * np.log10(amp / np.sqrt(2) / 20e-6)
    for f in (1000.0, 100.0):
        expected = float(tonal.weighting_db(np.array([f]), weighting)[0])
        x = amp * np.sin(2 * np.pi * f * t)
        r = overall_level(x, fs, rpm, frame=0.5, weighting=weighting)
        assert r.weighting == weighting
        # skip the filter's start-up frames
        assert abs(np.mean(r.level[2:]) - (ref_level + expected)) < 0.05


# --------------------------------------------------------------------------
# metric_vs_rpm
# --------------------------------------------------------------------------

A0, SLOPE = 40.0, 0.005  # analytic metric: L = A0 + SLOPE * rpm (+ gear offset)


def _gear_shift_case():
    """Run-up in gear 1 (to 4000 rpm), a shift down to 2500 rpm, gear 2 up to 4500 rpm."""
    t_rpm = np.arange(0.0, 10.0 + 1e-9, 0.05)
    rpm = np.interp(t_rpm, [0.0, 6.0, 6.5, 10.0], [1000.0, 4000.0, 2500.0, 4500.0])
    t_m = np.arange(501) * 0.02  # the metric has its own time grid
    r_m = np.interp(t_m, t_rpm, rpm)
    offset = np.where(t_m >= 6.5, 10.0, 0.0)
    offset[(t_m > 6.0) & (t_m < 6.5)] = 5.0  # during the shift
    return t_m, A0 + SLOPE * r_m + offset, t_rpm, rpm


def test_metric_vs_rpm_runup_with_gear_shift_selects_monotonic_samples():
    t_m, y, t_r, rpm = _gear_shift_case()
    r = metric_vs_rpm(y, t_m, rpm, t_r, segment="auto", bin_width=100.0)
    assert r.segment == "runup" and r.t_segment == (0.0, 10.0)
    assert r.used.shape == y.shape
    # the shift dip and the re-climb to the previous maximum are excluded
    assert not r.used[(t_m > 6.05) & (t_m < 9.0)].any()
    assert r.used[t_m <= 6.0].all() and r.used[t_m >= 9.1].all()
    # analytic values in gear 1 (bins fully inside) and gear 2 (+10 dB)
    g1 = (r.centres < 3900.0) & r.valid
    assert g1.sum() >= 25
    assert np.allclose(r.value[g1], A0 + SLOPE * r.centres[g1], atol=0.06)
    g2 = (r.centres > 4150.0) & r.valid
    assert g2.sum() >= 2
    assert np.allclose(r.value[g2], A0 + 10.0 + SLOPE * r.centres[g2], atol=0.06)
    # counts: 0.02 s * 500 rpm/s = 10 rpm per sample -> 10 samples per 100 rpm
    assert np.all(np.abs(r.count[g1] - 10) <= 1)
    assert r.count.sum() == r.used.sum()
    # inside a bin the metric is almost constant; the spread is the std
    assert r.spread_kind == "std" and np.all(
        r.std[g1] < 0.2
    )  # 0.05 dB per sample across 10 samples
    assert np.allclose(r.control[~np.isnan(r.control)], np.interp(t_m, t_r, rpm))


def test_metric_vs_rpm_all_segment_mixes_gears():
    t_m, y, t_r, rpm = _gear_shift_case()
    r = metric_vs_rpm(y, t_m, rpm, t_r, segment="all")
    assert r.segment == "all" and r.used.all()
    k = int(np.argmin(np.abs(r.centres - 3050.0)))
    assert r.count[k] >= 17  # gear 1 and gear 2 both pass 3000-3100 rpm
    assert r.value[k] - (A0 + SLOPE * r.centres[k]) > 2.0
    # the monotonic filter can be switched off for a segment
    r2 = metric_vs_rpm(y, t_m, rpm, t_r, segment="runup", tol=np.inf)
    assert r2.used.all()
    assert r2.count.sum() == len(y)


def test_metric_vs_rpm_skip_start_discards_startup_transient():
    t_m, y, t_r, rpm = _gear_shift_case()
    y = y.copy()
    y[t_m < 1.0] = -999.0  # the metric's invalid start-up values
    r0 = metric_vs_rpm(y, t_m, rpm, t_r)
    assert np.nanmin(r0.value) < -100.0  # contaminated without the skip
    r = metric_vs_rpm(y, t_m, rpm, t_r, skip_start=0.99)
    assert not r.used[t_m < 0.99].any()
    assert np.nanmin(r.value) > A0
    assert r.skip_start == 0.99
    assert np.all(r.count[r.centres < 1500.0] == 0)
    assert np.all(np.isnan(r.value[r.centres < 1500.0]))
    with pytest.raises(ValueError):
        metric_vs_rpm(y, t_m, rpm, t_r, skip_start=11.0)


def test_metric_vs_rpm_min_count_withholds_sparse_bins_but_reports_counts():
    t_m, y, t_r, rpm = _gear_shift_case()
    r = metric_vs_rpm(y, t_m, rpm, t_r, min_count=15)
    assert np.all(r.count[r.centres < 3900.0][1:-1] >= 9)
    assert np.all(np.isnan(r.value[r.count < 15]))
    assert not r.valid[r.count < 15].any()
    assert r.valid.sum() < metric_vs_rpm(y, t_m, rpm, t_r, min_count=3).valid.sum()


def _simple_runup(duration=6.0):
    t_r = np.arange(0.0, duration + 1e-9, 0.1)
    rpm = 1000.0 + 500.0 * t_r
    t_m = np.arange(int(duration / 0.02) + 1) * 0.02
    return t_m, t_r, rpm, np.interp(t_m, t_r, rpm)


def test_metric_vs_rpm_power_versus_arithmetic_mean_of_levels():
    t_m, t_r, rpm, r_m = _simple_runup()
    d = 3.0
    sign = np.where(np.arange(t_m.size) % 2 == 0, 1.0, -1.0)
    y = A0 + SLOPE * r_m + d * sign
    mean = metric_vs_rpm(y, t_m, rpm, t_r, aggregate="mean")
    power = metric_vs_rpm(y, t_m, rpm, t_r, aggregate="power")
    gap = 10 * np.log10(np.cosh(d * np.log(10) / 10))  # exact for equal halves
    ok = mean.valid
    assert power.aggregate == "power" and ok.sum() > 25
    assert np.allclose((power.value - mean.value)[ok], gap, atol=0.1)
    assert np.all(power.value[ok] > mean.value[ok])
    hi = metric_vs_rpm(y, t_m, rpm, t_r, aggregate=("percentile", 100))
    lo = metric_vs_rpm(y, t_m, rpm, t_r, aggregate=("percentile", 0))
    # range of 2 d plus the drift of L over the ~9 sample steps of a bin (0.05 dB each)
    assert np.allclose(hi.value[ok] - lo.value[ok], 2 * d + 9 * 0.05, atol=0.1)
    assert hi.q == 100 and hi.spread_kind == "iqr"
    # a constant level has the same power and arithmetic mean
    c = metric_vs_rpm(np.full(t_m.size, 60.0), t_m, rpm, t_r, aggregate="power")
    assert np.allclose(c.value[c.valid], 60.0)


def test_metric_vs_rpm_median_resists_outliers_and_reports_iqr():
    t_m, t_r, rpm, r_m = _simple_runup()
    base = A0 + SLOPE * r_m
    y = base.copy()
    y[::7] += 30.0
    med = metric_vs_rpm(y, t_m, rpm, t_r, aggregate="median")
    avg = metric_vs_rpm(y, t_m, rpm, t_r, aggregate="mean")
    ok = med.valid
    ref = A0 + SLOPE * med.centres
    assert np.all(np.abs(med.value[ok] - ref[ok]) < 0.2)
    assert np.mean(avg.value[ok] - ref[ok]) > 3.0
    assert med.spread_kind == "iqr" and np.array_equal(
        med.spread, med.iqr, equal_nan=True
    )
    assert avg.spread_kind == "std" and np.array_equal(
        avg.spread, avg.std, equal_nan=True
    )
    assert np.all(np.isfinite(med.iqr[ok]))


def test_metric_vs_rpm_rundown_and_auto_choice():
    # up in 3 s, down in 6 s: the run-down is the longer part
    t_r = np.arange(0.0, 9.0 + 1e-9, 0.05)
    rpm = np.interp(t_r, [0.0, 3.0, 9.0], [1000.0, 4000.0, 1000.0])
    t_m = np.arange(451) * 0.02
    r_m = np.interp(t_m, t_r, rpm)
    y = (
        A0 + SLOPE * r_m + np.where(t_m > 3.0, 5.0, 0.0)
    )  # hysteresis of 5 dB on the way down
    auto = metric_vs_rpm(y, t_m, rpm, t_r)
    assert auto.segment == "rundown"
    assert auto.t_segment == (3.0, 9.0)
    down = metric_vs_rpm(y, t_m, rpm, t_r, segment="rundown")
    assert np.array_equal(auto.used, down.used)
    assert not down.used[t_m < 3.0].any()
    ok = down.valid & (down.centres > 1200.0) & (down.centres < 3800.0)
    assert ok.sum() > 20
    assert np.allclose(down.value[ok], A0 + 5.0 + SLOPE * down.centres[ok], atol=0.06)
    # bins ascend in rpm whatever the direction; counts: 0.02 s * 500 rpm/s = 10 rpm per sample
    assert np.all(np.diff(down.centres) > 0)
    assert np.all(np.abs(down.count[ok] - 10) <= 1)
    up = metric_vs_rpm(y, t_m, rpm, t_r, segment="runup")
    assert up.t_segment == (0.0, 3.0)
    assert np.allclose(
        up.value[up.valid & (up.centres < 3800)],
        A0 + SLOPE * up.centres[up.valid & (up.centres < 3800)],
        atol=0.06,
    )


def test_metric_vs_rpm_handles_gaps_explicit_bins_and_other_controls():
    t_m, t_r, rpm, r_m = _simple_runup()
    rpm = rpm.copy()
    rpm[10:15] = np.nan  # gap in the control: interpolated across (linear -> exact)
    y = A0 + SLOPE * r_m
    y[100:110] = np.nan  # gap in the metric
    kmh = rpm / 20.0  # any monotonic control works: vehicle speed in km/h
    edges = np.arange(50.0, 350.0 + 1e-9, 25.0)
    r = metric_vs_rpm(y, t_m, kmh, t_r, bins=edges)
    assert np.array_equal(r.edges, edges)
    assert not r.used[100:110].any()
    assert r.count.sum() == np.isfinite(y).sum()
    ok = r.valid
    assert ok.sum() >= 6
    # the metric is linear in the control, so each bin reads the mean control of its samples
    # (not the bin centre: the gap removes samples)
    for b in np.flatnonzero(ok):
        sel = r.used & (r.control >= edges[b]) & (r.control < edges[b + 1])
        assert r.value[b] == pytest.approx(
            A0 + SLOPE * 20.0 * r.control[sel].mean(), abs=1e-9
        )
    # samples outside the control's time range are not extrapolated
    r2 = metric_vs_rpm(y, t_m + 1.0, rpm[np.isfinite(rpm)], t_r[np.isfinite(rpm)])
    assert not r2.used[-10:].any() and np.isnan(r2.control[-1])


def test_metric_vs_rpm_validation():
    t_m, t_r, rpm, r_m = _simple_runup()
    y = r_m * 0.0
    with pytest.raises(ValueError):
        metric_vs_rpm(y, t_m, rpm, t_r, segment="sideways")
    with pytest.raises(ValueError):
        metric_vs_rpm(y, t_m, rpm, t_r, aggregate="mode")
    with pytest.raises(ValueError):
        metric_vs_rpm(y, t_m, rpm, t_r, aggregate=("percentile", 120))
    with pytest.raises(ValueError):
        metric_vs_rpm(y, t_m, rpm, t_r, bin_width=0.0)
    with pytest.raises(ValueError):
        metric_vs_rpm(y, t_m, rpm, t_r, min_count=0)
    with pytest.raises(ValueError):
        metric_vs_rpm(y[:-1], t_m, rpm, t_r)
    with pytest.raises(ValueError):
        metric_vs_rpm(y, t_m[::-1], rpm, t_r)
    with pytest.raises(ValueError):
        metric_vs_rpm(
            y, t_m, np.full_like(rpm, 1000.0), t_r, segment="runup"
        )  # no sweep


# --------------------------------------------------------------------------
# frequency_to_order, order_share
# --------------------------------------------------------------------------


def test_frequency_to_order_integer_candidates_and_tolerance():
    rpm = 3000.0  # 50 Hz shaft
    f = np.array([100.0, 205.0, 125.0, 210.0, np.nan])
    r = frequency_to_order(f, rpm)
    assert np.allclose(r.order[:4], [2.0, 4.1, 2.5, 4.2]) and np.isnan(r.order[4])
    # 2.5 is nearer to 3 in relative terms (0.167 against 0.25)
    assert np.allclose(r.nearest[:4], [2.0, 4.0, 3.0, 4.0])
    assert np.allclose(r.deviation[:4], [0.0, 0.025, -1 / 6, 0.05])
    assert list(r.matched) == [True, True, False, False, False]
    assert r.tolerance == 0.04
    wide = frequency_to_order(f, rpm, tolerance=0.06)
    assert list(wide.matched) == [True, True, False, True, False]


def test_frequency_to_order_tracks_with_speed_and_given_candidates():
    t = np.linspace(0.0, 1.0, 11)
    rpm = 1500.0 + 1500.0 * t
    f = 3.0 * rpm / 60.0 * (1.0 + 0.02)  # order 3.06
    r = frequency_to_order(f, rpm, orders=[1.5, 3.0, 4.5])
    assert np.allclose(r.order, 3.06) and np.all(r.nearest == 3.0) and np.all(r.matched)
    # fractional (half) orders as candidates
    h = frequency_to_order(1.5 * rpm / 60.0, rpm, orders=[1.0, 1.5, 2.0])
    assert np.allclose(h.order, 1.5) and np.all(h.nearest == 1.5)
    # non-positive speed gives no order
    z = frequency_to_order([100.0, 100.0], [0.0, 3000.0])
    assert np.isnan(z.order[0]) and not z.matched[0] and z.matched[1]
    with pytest.raises(ValueError):
        frequency_to_order([1.0], 3000.0, orders=[0.0, 1.0])
    with pytest.raises(ValueError):
        frequency_to_order([1.0], 3000.0, tolerance=1.5)
    with pytest.raises(ValueError):
        frequency_to_order([1.0, 2.0, 3.0], [1000.0, 2000.0])


def _order_track():
    orders = np.array(
        [2.0] * 6 + [2.07] * 2 + [4.0] * 4 + [6.0] * 2 + [3.0] * 3 + [2.5] * 3
    )
    rpm = np.linspace(1000.0, 4000.0, orders.size)
    return orders * rpm / 60.0, rpm


def test_order_share_exact_fractions_and_counts():
    f, rpm = _order_track()  # 20 samples: 8 on order 2, 4 on 4, 2 on 6, 6 elsewhere
    r = order_share(f, rpm, 2.0)
    assert r.n_total == 20 and r.n_valid == 20
    assert list(r.counts) == [8, 4, 2, 0]
    assert list(r.harmonics) == [1, 2, 3, 4]
    assert np.allclose(r.share_per_harmonic, [0.4, 0.2, 0.1, 0.0])
    assert r.share == pytest.approx(0.7)
    # fewer harmonics, tighter tolerance (2.07 is 3.5 % off)
    r2 = order_share(f, rpm, 2.0, harmonics=2)
    assert list(r2.counts) == [8, 4] and r2.share == pytest.approx(0.6)
    r3 = order_share(f, rpm, 2.0, tolerance=0.03)
    assert list(r3.counts) == [6, 4, 2, 0] and r3.share == pytest.approx(0.6)
    # a base order with no tone on it
    assert order_share(f, rpm, 5.0).share == 0.0


def test_order_share_weights_gaps_and_overlapping_windows():
    f, rpm = _order_track()
    w = np.where(
        np.isin(np.arange(20), np.arange(8, 12)), 3.0, 1.0
    )  # weight the order-4 samples
    r = order_share(f, rpm, 2.0, weights=w)
    total = w.sum()
    assert total == 28.0
    assert np.allclose(r.weight_per_harmonic, [8.0, 12.0, 2.0, 0.0])
    assert r.share == pytest.approx(22.0 / 28.0)
    assert list(r.counts) == [8, 4, 2, 0]
    # NaN frequency, zero speed and NaN weight are not valid samples
    f2, rpm2, w2 = f.copy(), rpm.copy(), w.copy()
    f2[0] = np.nan
    rpm2[1] = 0.0
    w2[2] = np.nan
    r2 = order_share(f2, rpm2, 2.0, weights=w2)
    assert r2.n_valid == 17 and r2.n_total == 20
    assert list(r2.counts) == [5, 4, 2, 0]
    # zero total weight
    assert np.isnan(order_share(f, rpm, 2.0, weights=np.zeros(20)).share)
    # tolerance windows of neighbouring harmonics overlap: no double counting
    ks = np.arange(1, 21)
    r3 = order_share(ks * 1.0 * 1.0, 60.0, 1.0, harmonics=20, tolerance=0.3)
    assert r3.counts.sum() == 20 and list(r3.counts) == [1] * 20


def test_order_share_validation():
    f, rpm = _order_track()
    with pytest.raises(ValueError):
        order_share(f, rpm, 0.0)
    with pytest.raises(ValueError):
        order_share(f, rpm, 2.0, harmonics=0)
    with pytest.raises(ValueError):
        order_share(f, rpm, 2.0, weights=np.ones(3))
    with pytest.raises(ValueError):
        order_share(f, rpm, 2.0, weights=-np.ones(20))
