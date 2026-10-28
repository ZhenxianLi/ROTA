"""Tests of the rpm-bin aggregation of rota_order.maps against analytic truth."""

import numpy as np
import pytest

from rota_order.maps import BinnedMap, Map, bin_map, rpm_frequency_map, rpm_order_map
from rota_order.synth import rpm_profiles, synth_runup

FS = 8000.0
ORDER = 2.0
GAIN_RPM = [0.0, 6000.0]
GAIN = [0.5, 2.0]  # linear gain law of the order: amplitude(rpm), peak
FALL_FACTOR = 5.0  # extra level on the falling stretch, to tell the segments apart


def _gain(r):
    return np.interp(r, GAIN_RPM, GAIN)


def _db(a, ref):
    return 20.0 * np.log10(np.asarray(a) / np.asarray(ref))


def _shifted_record():
    """Run-up with a lift-off in the middle: rise, fall (x5 level), rise again.

    Speed 1000 -> 4000 rpm in 3 s, 4000 -> 2500 in 1.5 s, 2500 -> 5000 in 2.5 s.
    Returns the signal, the speed, the time and the sample masks of the rises
    and the fall.
    """
    knots = ([0.0, 3.0, 4.5, 7.0], [1000.0, 4000.0, 2500.0, 5000.0])
    n = int(7.0 * FS) + 1
    t = np.arange(n) / FS
    rpm = rpm_profiles(t, "piecewise", knots=knots)
    s = synth_runup(
        FS, rpm,
        [{"order": ORDER, "amplitude": 1.0, "phase": 0.3, "gain_rpm": GAIN_RPM, "gain": GAIN}],
        seed=1,
    )
    fall = (t > 3.0) & (t < 4.5)
    # level step on the fall, smoothed by a 0.1 s moving average
    k = int(0.1 * FS)
    step = np.pad(np.where(fall, FALL_FACTOR, 1.0), k, mode="edge")
    env = np.convolve(step, np.ones(k) / k, mode="same")[k:-k]
    return s.x * env, rpm, t, ~fall, fall


@pytest.fixture(scope="module")
def record():
    x, rpm, t, _rise, _fall = _shifted_record()
    return x, rpm, t


@pytest.fixture(scope="module")
def omap(record):
    x, rpm, _ = record
    return rpm_order_map(x, FS, rpm, order_resolution=0.25, max_order=6.0, amplitude="rms")


def _near_turning(centres):
    """True for bins whose frames may straddle a turning point (4000 and 2500 rpm)."""
    return (np.abs(centres - 4000.0) < 300.0) | (np.abs(centres - 2500.0) < 300.0)


# --------------------------------------------------------------------------
# default behaviour unchanged
# --------------------------------------------------------------------------


def test_default_returns_plain_per_frame_map(record):
    x, rpm, _ = record
    a = rpm_order_map(x, FS, rpm, order_resolution=0.25, max_order=6.0)
    b = rpm_order_map(x, FS, rpm, order_resolution=0.25, max_order=6.0, rpm_bins=None)
    assert type(a) is Map and type(b) is Map
    assert np.array_equal(a.values, b.values) and np.array_equal(a.rpm, b.rpm)
    f1 = rpm_frequency_map(x, FS, rpm, resolution=10.0)
    f2 = rpm_frequency_map(x, FS, rpm, resolution=10.0, rpm_bins=None, segment=None)
    assert type(f1) is Map and np.array_equal(f1.values, f2.values)


def test_segment_without_bins_is_an_error(record):
    x, rpm, _ = record
    with pytest.raises(ValueError, match="rpm_bins"):
        rpm_order_map(x, FS, rpm, segment="runup")
    with pytest.raises(ValueError, match="rpm_bins"):
        rpm_frequency_map(x, FS, rpm, segment="runup")


# --------------------------------------------------------------------------
# segment selection with a gear shift / lift-off (non-monotonic speed)
# --------------------------------------------------------------------------


def test_runup_selects_rising_stretches_and_amplitude_matches_truth(omap):
    b = bin_map(omap, 100.0, segment="runup")
    assert isinstance(b, BinnedMap) and b.segment == "runup"
    k = int(np.argmin(np.abs(omap.axis - ORDER)))
    # every kept frame is a rising one, none comes from the fall
    assert np.all(b.counts >= 1)
    far = ~_near_turning(b.rpm)
    amp = np.abs(b.values[k])
    truth = _gain(b.rpm) / np.sqrt(2.0)  # peak gain law -> rms convention
    assert np.max(np.abs(_db(amp[far], truth[far]))) < 0.2
    # the x5 level of the fall never leaks in: all far bins read the rise level
    assert np.max(amp[far] / truth[far]) < 1.05


def test_rundown_selects_the_falling_stretch(omap):
    b = bin_map(omap, 100.0, segment="rundown")
    assert b.segment == "rundown"
    k = int(np.argmin(np.abs(omap.axis - ORDER)))
    far = ~_near_turning(b.rpm)
    amp = np.abs(b.values[k])
    truth = FALL_FACTOR * _gain(b.rpm) / np.sqrt(2.0)
    assert far.sum() >= 3
    assert np.max(np.abs(_db(amp[far], truth[far]))) < 0.2
    # the falling stretch spans 2500..4000 rpm only
    assert b.rpm_lo.min() >= 2400.0 - 1e-9 and b.rpm_hi.max() <= 4100.0 + 1e-9


def test_auto_picks_runup_for_a_net_rise_and_rundown_for_a_net_fall(omap):
    assert bin_map(omap, 100.0, segment="auto").segment == "runup"
    rev = Map(
        values=omap.values[:, ::-1], axis=omap.axis, domain=omap.domain, t=omap.t,
        rpm=omap.rpm[::-1], rpm_min=omap.rpm_min[::-1], rpm_max=omap.rpm_max[::-1],
        amplitude=omap.amplitude, enbw_bins=omap.enbw_bins, window=omap.window,
        nperseg=omap.nperseg, hop=omap.hop,
    )
    assert bin_map(rev, 100.0, segment="auto").segment == "rundown"


def test_counts_equal_frames_on_the_rising_stretches(omap):
    b = bin_map(omap, 100.0, segment="runup")
    # truth: the speed is known to rise before 3 s and after 4.5 s
    rising = (omap.t < 3.0 - 0.3) | (omap.t > 4.5 + 0.3)
    far = ~_near_turning(b.rpm)
    for cnt, lo, hi in zip(b.counts[far], b.rpm_lo[far], b.rpm_hi[far]):
        inside = (omap.rpm >= lo) & (omap.rpm < hi) & rising
        assert cnt == inside.sum()
    # the bins 2500..4000 rpm are passed on both rises: more frames than either alone
    # (3000..3600 rpm): first rise at 1000 rpm/s, second at 1000 rpm/s -> twice the frames of a single pass
    both = (b.rpm_centre > 3000.0) & (b.rpm_centre < 3700.0)
    single = (b.rpm_centre > 1200.0) & (b.rpm_centre < 2200.0)
    assert np.all(b.counts[both] >= 1.8 * np.median(b.counts[single]))


def test_unselected_binning_mixes_the_fall_in(omap):
    """Without a segment the x5 fall contaminates the overlapped bins (control)."""
    b = bin_map(omap, 100.0)
    k = int(np.argmin(np.abs(omap.axis - ORDER)))
    amp = np.abs(b.values[k])
    truth_up = _gain(b.rpm) / np.sqrt(2.0)
    mixed = (b.rpm > 3000.0) & (b.rpm < 3800.0)
    assert np.all(amp[mixed] / truth_up[mixed] > 1.5)
    assert b.segment is None
    assert int(b.counts.sum()) == omap.rpm.size


# --------------------------------------------------------------------------
# a steady-state-style check on a clean run-up and a run-down, frequency map
# --------------------------------------------------------------------------


def _simple(start, stop, dur=4.0):
    n = int(dur * FS)
    t = np.arange(n) / FS
    rpm = rpm_profiles(t, "linear", start=start, stop=stop)
    s = synth_runup(FS, rpm, [{"order": ORDER, "amplitude": 2.0, "phase": 0.0}], seed=0)
    return s, rpm


@pytest.mark.parametrize("start, stop, seg", [(800.0, 3200.0, "runup"), (3200.0, 800.0, "rundown")])
def test_monotonic_run_amplitude_and_counts(start, stop, seg):
    s, rpm = _simple(start, stop)
    m = rpm_order_map(s.x, FS, rpm, order_resolution=0.25, max_order=6.0, amplitude="peak")
    b = rpm_order_map(
        s.x, FS, rpm, order_resolution=0.25, max_order=6.0, amplitude="peak",
        rpm_bins=200.0, segment=seg,
    )
    assert isinstance(b, BinnedMap) and b.segment == seg
    k = int(np.argmin(np.abs(m.axis - ORDER)))
    assert np.max(np.abs(_db(np.abs(b.values[k]), 2.0))) < 0.2
    # a monotonic run keeps every frame, except possibly the end frames (zero gradient never occurs)
    assert int(b.counts.sum()) == m.rpm.size
    assert np.all(np.diff(b.rpm) > 0)  # bins are always returned in increasing speed
    # bin geometry: edges are multiples of the width and contain the mean speed
    assert np.allclose(b.rpm_hi - b.rpm_lo, 200.0)
    assert np.allclose(b.rpm_lo / 200.0, np.round(b.rpm_lo / 200.0))
    assert np.all((b.rpm >= b.rpm_lo) & (b.rpm < b.rpm_hi))
    # the opposite direction is empty
    other = "rundown" if seg == "runup" else "runup"
    assert bin_map(m, 200.0, segment=other).counts.size == 0


def test_frequency_map_binning_matches_truth_and_keeps_convention():
    t = np.arange(int(4.0 * FS)) / FS
    rpm = rpm_profiles(t, "linear", start=900.0, stop=2100.0)
    s = synth_runup(FS, rpm, [{"order": 1.0, "amplitude": 1.5, "phase": 0.0}], seed=0)
    b = rpm_frequency_map(
        s.x, FS, rpm, resolution=5.0, window="flattop", amplitude="peak",
        rpm_bins=[800.0, 1200.0, 1600.0, 2200.0], segment="auto",
    )
    assert b.domain == "frequency" and b.segment == "runup" and b.amplitude == "peak"
    assert b.rpm_lo.tolist() == [800.0, 1200.0, 1600.0] and b.rpm_hi.tolist() == [1200.0, 1600.0, 2200.0]
    # the line is spread over bins by the sweep: sqrt(sum |V|^2 / enbw) is its peak amplitude
    amp = np.sqrt(np.sum(np.abs(b.values) ** 2, axis=0) / b.enbw_bins)
    assert np.max(np.abs(_db(amp, 1.5))) < 0.2
    # and its frequency is that of the mean speed of the bin (the sweep is symmetric in each bin)
    peak_f = b.axis[np.argmax(np.abs(b.values), axis=0)]
    assert np.all(np.abs(peak_f - b.rpm / 60.0) < 5.0)


# --------------------------------------------------------------------------
# bin_map on hand-made maps: exact arithmetic
# --------------------------------------------------------------------------


def _hand_map(values, rpm, rpm_min=None, rpm_max=None):
    rpm = np.asarray(rpm, float)
    nf = rpm.size
    return Map(
        values=np.asarray(values, complex),
        axis=np.arange(np.asarray(values).shape[0], dtype=float),
        domain="order",
        t=np.arange(nf, dtype=float),
        rpm=rpm,
        rpm_min=rpm - 1.0 if rpm_min is None else rpm_min,
        rpm_max=rpm + 1.0 if rpm_max is None else rpm_max,
        amplitude="rms", enbw_bins=1.5, window="hann", nperseg=8, hop=2,
    )


def test_energy_mean_counts_and_geometry_exact():
    # bins [0,100) holds frames 1 and 3 (|V| = 1, 3), [100,200) frame 5 only
    v = np.array([[1.0, 3.0, 5.0], [0.0, 4.0, 2.0]])  # (2 bins, 3 frames)
    m = _hand_map(v, [10.0, 50.0, 150.0])
    b = bin_map(m, 100.0)
    assert np.array_equal(b.counts, [2, 1])
    assert np.allclose(np.abs(b.values[:, 0]), [np.sqrt((1 + 9) / 2), np.sqrt(16 / 2)])
    assert np.allclose(np.abs(b.values[:, 1]), [5.0, 2.0])
    assert np.allclose(b.rpm, [30.0, 150.0]) and np.allclose(b.t, [0.5, 2.0])
    assert np.allclose(b.rpm_min, [9.0, 149.0]) and np.allclose(b.rpm_max, [51.0, 151.0])
    assert np.allclose(b.rpm_centre, [50.0, 150.0])
    assert np.all(b.values.imag == 0)
    assert (b.amplitude, b.enbw_bins, b.window, b.nperseg, b.hop) == ("rms", 1.5, "hann", 8, 2)
    assert np.array_equal(b.axis, m.axis) and b.domain == "order"


def test_edge_array_discards_outside_and_closes_last_edge():
    v = np.ones((1, 5))
    m = _hand_map(v, [50.0, 100.0, 150.0, 200.0, 250.0])
    assert np.array_equal(bin_map(m, [100.0, 200.0]).counts, [3])  # 100, 150 and 200 (closed last edge)
    b2 = bin_map(m, [100.0, 150.0, 200.0])
    assert np.array_equal(b2.counts, [1, 2])  # 100 | 150, 200 (last edge closed)
    assert b2.rpm_lo.tolist() == [100.0, 150.0] and b2.rpm_hi.tolist() == [150.0, 200.0]


def test_min_count_drops_sparse_bins_and_empty_result_is_well_formed():
    v = np.ones((3, 4))
    m = _hand_map(v, [10.0, 20.0, 30.0, 150.0])
    b = bin_map(m, 100.0, min_count=2)
    assert b.counts.tolist() == [3] and b.values.shape == (3, 1)
    e = bin_map(m, 100.0, min_count=10)
    assert e.values.shape == (3, 0) and e.counts.size == 0 and e.rpm.size == 0


def test_circular_phase_and_drop():
    ph = np.array([0.2, 0.4, 0.6])
    v = (np.array([1.0, 2.0, 3.0]) * np.exp(1j * ph))[None, :]
    m = _hand_map(v, [10.0, 20.0, 30.0])
    c = bin_map(m, 100.0, phase="circular")
    assert c.phase == "circular"
    assert np.angle(c.values[0, 0]) == pytest.approx(0.4, abs=1e-12)  # symmetric phases
    assert np.abs(c.values[0, 0]) == pytest.approx(np.sqrt((1 + 4 + 9) / 3))
    d = bin_map(m, 100.0)
    assert d.phase == "drop" and np.all(d.values.imag == 0)
    # opposite phasors cancel: phase 0 by convention, amplitude kept
    o = bin_map(_hand_map(np.array([[1.0, -1.0]]), [10.0, 20.0]), 100.0, phase="circular")
    assert np.abs(o.values[0, 0]) == pytest.approx(1.0) and np.isfinite(np.angle(o.values[0, 0]))


def test_multichannel_values_keep_channel_axis():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((4, 6, 2)) + 1j * rng.standard_normal((4, 6, 2))
    m = _hand_map(v, [10.0, 20.0, 30.0, 110.0, 120.0, 130.0])
    b = bin_map(m, 100.0)
    assert b.values.shape == (4, 2, 2) and b.counts.tolist() == [3, 3]
    assert np.allclose(np.abs(b.values[:, 0, 1]), np.sqrt(np.mean(np.abs(v[:, :3, 1]) ** 2, axis=1)))


def test_segment_rule_is_memoryless_and_excludes_flat_frames():
    rpm = [100.0, 200.0, 300.0, 300.0, 300.0, 250.0, 200.0]
    m = _hand_map(np.ones((1, 7)), rpm)
    up = bin_map(m, 1000.0, segment="runup")
    dn = bin_map(m, 1000.0, segment="rundown")
    assert up.counts.tolist() == [3]  # frames 0, 1, 2 (rising or leading to the plateau)
    assert dn.counts.tolist() == [3]  # frames 4 (one-sided gradient is negative: -25), 5, 6
    assert int(up.counts[0] + dn.counts[0]) <= 7


def test_validation_errors():
    m = _hand_map(np.ones((1, 3)), [10.0, 20.0, 30.0])
    for bad in (0.0, -5.0, [100.0], [200.0, 100.0], [[0.0, 1.0]], float("nan")):
        with pytest.raises(ValueError):
            bin_map(m, bad)
    with pytest.raises(ValueError):
        bin_map(m, 100.0, segment="up")
    with pytest.raises(ValueError):
        bin_map(m, 100.0, phase="mean")
    with pytest.raises(ValueError):
        bin_map(m, 100.0, min_count=0)
    with pytest.raises(ValueError, match="already binned"):
        bin_map(bin_map(m, 100.0), 100.0)
