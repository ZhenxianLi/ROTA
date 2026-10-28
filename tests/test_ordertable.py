"""Tests for OrderTable, synthesize_orders and OrderSynth against analytic truth."""

import json

import numpy as np
import pytest
from scipy import signal

from rota_order import _ordertable
from rota_order._components import OrderComponents
from rota_order._ordertable import OrderSynth, OrderTable, synthesize_orders
from rota_order.maps import rpm_order_map
from rota_order.slices import OrderSlice, order_slice
from rota_order.speed import rpm_to_phase
from rota_order.synth import synth_runup

TWO_PI = 2 * np.pi
FS = 8000.0


def _db(a, ref):
    return 20.0 * np.log10(np.asarray(a) / np.asarray(ref))


def _wrap(a):
    """Wrap angles to (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(a)))


def _flat_table(orders, mag=1.0, grid=(500.0, 6000.0), phase=0.0, **kw):
    orders = np.atleast_1d(orders).astype(float)
    grid = np.asarray(grid, dtype=float)
    vals = np.full((orders.size, grid.size), mag * np.exp(1j * phase), dtype=complex)
    kw.setdefault("amplitude", "peak")
    return OrderTable(orders, grid, vals, **kw)


def _inst_freq(x, fs):
    """Instantaneous frequency in Hz from the unwrapped phase of the analytic signal."""
    ph = np.unwrap(np.angle(signal.hilbert(x)))
    return np.gradient(ph, 1.0 / fs) / TWO_PI


def _slice(order, rpm, amp, cpx, conv="peak"):
    rpm = np.asarray(rpm, dtype=float)
    return OrderSlice(
        order=order,
        t=np.arange(rpm.size) * 0.1,
        rpm=rpm,
        amplitude=np.asarray(amp, dtype=float),
        complex=np.asarray(cpx, dtype=complex),
        amplitude_convention=conv,
        bandwidth=1.0,
        domain="order",
    )


# --------------------------------------------------------------------------
# OrderTable: construction and conversion
# --------------------------------------------------------------------------


def test_construction_defaults_and_copies():
    orders = np.array([1.0, 2.5])
    grid = np.array([1000.0, 2000.0, 3000.0])
    vals = np.ones((2, 3)) * 2.0  # real input is accepted as zero phase
    t = OrderTable(orders, grid, vals, "rms", unit="Pa", meta={"a": [1, 2]})
    assert np.iscomplexobj(t.values) and t.values.shape == (2, 3)
    assert np.all(t.freq_offset == 0.0) and t.freq_offset.shape == (2,)
    assert t.amplitude == "rms" and t.unit == "Pa" and t.n_channels is None
    orders[0] = 99.0
    vals[:] = 0.0
    t.meta["a"].append(3)
    assert t.orders[0] == 1.0 and np.all(t.values == 2.0)
    assert "k=2" in repr(t) and "rms" in repr(t)
    t3 = OrderTable(orders, grid, np.ones((2, 3, 4)), "peak", freq_offset=7.0)
    assert t3.n_channels == 4 and np.all(t3.freq_offset == 7.0)


def test_amplitude_convention_is_required_and_checked():
    with pytest.raises(TypeError):
        OrderTable([1.0], [1.0, 2.0], np.ones((1, 2)))  # no default (ROADMAP item 14)
    with pytest.raises(ValueError, match="amplitude"):
        OrderTable([1.0], [1.0, 2.0], np.ones((1, 2)), "power")


@pytest.mark.parametrize(
    "kw",
    [
        dict(orders=[[1.0]]),  # not 1-D
        dict(orders=[np.nan]),
        dict(rpm=[2000.0, 1000.0, 3000.0]),  # not increasing
        dict(rpm=[1000.0, 1000.0, 3000.0]),  # not strictly
        dict(rpm=[1000.0, 2000.0]),  # wrong length for values
        dict(values=np.ones((2, 3))),  # wrong number of rows
        dict(values=np.ones((1, 3, 2, 2))),
        dict(values=np.array([[1.0, np.inf, 1.0]])),
        dict(freq_offset=[1.0, 2.0]),
        dict(freq_offset=[np.nan]),
        dict(unit=3),
        dict(meta=[1]),
    ],
)
def test_construction_rejects_bad_input(kw):
    args = dict(
        orders=[1.0], rpm=[1000.0, 2000.0, 3000.0], values=np.ones((1, 3)), amplitude="peak"
    )
    args.update(kw)
    with pytest.raises((ValueError, TypeError)):
        OrderTable(**args)


def test_peak_rms_conversion():
    t = _flat_table([1.0, 2.0], mag=3.0, phase=0.7, amplitude="peak", unit="Pa", freq_offset=[0, 5])
    r = t.to_rms()
    assert r.amplitude == "rms" and r is not t
    assert np.allclose(np.abs(r.values), 3.0 / np.sqrt(2))
    assert np.allclose(np.angle(r.values), 0.7)
    back = r.to_peak()
    assert back.amplitude == "peak" and np.allclose(back.values, t.values, rtol=1e-14)
    same = t.to_peak()
    assert same is not t and np.array_equal(same.values, t.values)
    same.values[0, 0] = 0.0
    assert t.values[0, 0] != 0.0
    assert r.unit == "Pa" and np.array_equal(r.freq_offset, t.freq_offset)
    assert np.array_equal(r.orders, t.orders) and np.array_equal(r.rpm, t.rpm)


# --------------------------------------------------------------------------
# OrderTable: magnitude_at and phase_at
# --------------------------------------------------------------------------


def test_magnitude_at_linear_hold_and_shapes():
    grid = np.array([1000.0, 2000.0, 4000.0])
    vals = np.array([[1.0, 3.0, 2.0], [4.0, 4.0, 4.0]]) * np.exp(1j * 0.3)
    t = OrderTable([1.0, 2.0], grid, vals, "peak")
    q = np.array([500.0, 1000.0, 1500.0, 3000.0, 5000.0])
    m = t.magnitude_at(q)
    assert m.shape == (2, 5)
    assert np.allclose(m[0], [1.0, 1.0, 2.0, 2.5, 2.0])
    assert np.allclose(m[1], 4.0)
    s = t.magnitude_at(1500.0)
    assert s.shape == (2,) and np.allclose(s, [2.0, 4.0])
    with pytest.raises(ValueError):
        t.magnitude_at(np.ones((2, 2)))


def test_magnitude_and_phase_skip_nan_and_all_nan_rows():
    grid = np.array([1000.0, 2000.0, 3000.0])
    vals = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]], dtype=complex)
    vals[0, 0] *= np.exp(1j * 0.2)
    vals[0, 2] *= np.exp(1j * 0.6)
    t = OrderTable([1.0, 2.0], grid, vals, "peak")
    assert t.magnitude_at(2000.0)[0] == pytest.approx(2.0)
    assert t.phase_at(2000.0)[0] == pytest.approx(0.4)
    assert t.magnitude_at(100.0)[0] == pytest.approx(1.0)  # end value held
    assert t.magnitude_at(9000.0)[0] == pytest.approx(3.0)
    assert np.all(np.isnan(t.magnitude_at([1000.0, 2500.0])[1]))
    assert np.all(np.isnan(t.phase_at([1000.0, 2500.0])[1]))


def test_phase_at_is_unwrapped_and_skips_zero_magnitude():
    grid = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    true_phase = np.array([3.0, 3.5, 4.0, 4.5])  # crosses +pi
    t = OrderTable([1.0], grid, np.exp(1j * true_phase)[None], "peak")
    q = np.array([1500.0, 2500.0, 3500.0])
    assert np.allclose(t.phase_at(q)[0], [3.25, 3.75, 4.25])
    assert t.phase_at(0.0)[0] == pytest.approx(3.0)
    # A zero value has no phase: it must not enter the unwrapping.
    z = np.exp(1j * np.array([0.1, 0.2, 0.3, 0.4]))
    z[1] = 0.0
    t0 = OrderTable([1.0], grid, z[None], "peak")
    assert t0.phase_at(2000.0)[0] == pytest.approx(0.2)  # interpolated between 0.1 and 0.3
    assert t0.magnitude_at(2000.0)[0] == 0.0


def test_multichannel_evaluation_is_per_channel():
    grid = np.array([1000.0, 2000.0])
    vals = np.zeros((2, 2, 2), dtype=complex)
    vals[:, :, 0] = [[1.0, 2.0], [3.0, 3.0]]
    vals[:, :, 1] = [[10.0, np.nan], [0.5, 1.5]]
    vals[0, 1, 1] = np.nan
    t = OrderTable([1.0, 2.0], grid, vals, "peak")
    m = t.magnitude_at([1000.0, 1500.0, 2000.0])
    assert m.shape == (2, 3, 2) and t.n_channels == 2
    assert np.allclose(m[0, :, 0], [1.0, 1.5, 2.0])
    assert np.allclose(m[0, :, 1], 10.0)  # NaN point skipped in channel 1 only
    assert np.allclose(m[1, :, 1], [0.5, 1.0, 1.5])
    assert t.phase_at(1500.0).shape == (2, 2)
    assert t.magnitude_at(1500.0).shape == (2, 2)


def test_smooth_rpm_moving_average_nan_aware():
    rng = np.random.default_rng(3)
    grid = np.arange(1000.0, 3001.0, 100.0)
    mag = 2.0 + rng.standard_normal((2, grid.size)) * 0.5
    mag[0, 7] = np.nan
    t = OrderTable([1.0, 2.0], grid, mag.astype(complex), "peak")
    width = 300.0
    expected = np.full_like(mag, np.nan)
    for j in range(2):
        for i in range(grid.size):
            sel = np.abs(grid - grid[i]) <= width / 2 + 1e-6
            vals = mag[j, sel]
            if np.isfinite(mag[j, i]):
                expected[j, i] = np.nanmean(vals)
    got = t.magnitude_at(grid, smooth_rpm=width)
    finite = np.isfinite(expected)
    assert np.allclose(got[finite], expected[finite], rtol=1e-12)
    assert np.std(got[1]) < np.std(mag[1])
    # The NaN grid point stays skipped: interpolated from its smoothed neighbours.
    assert got[0, 7] == pytest.approx(0.5 * (expected[0, 6] + expected[0, 8]))
    with pytest.raises(ValueError):
        t.magnitude_at(grid, smooth_rpm=0.0)


# --------------------------------------------------------------------------
# ROADMAP section 6, item 10: interpolate magnitude, not Re/Im
# --------------------------------------------------------------------------


def test_item10_magnitude_survives_alternating_phase_where_re_im_collapses():
    grid = np.arange(1000.0, 4001.0, 500.0)
    phases = np.where(np.arange(grid.size) % 2 == 0, 0.0, TWO_PI / 3)  # 120 degrees apart
    t = OrderTable([3.0], grid, np.exp(1j * phases)[None], "peak")
    mid = 0.5 * (grid[1:] + grid[:-1])

    assert np.allclose(t.magnitude_at(mid)[0], 1.0, atol=1e-12)
    assert np.allclose(t.phase_at(mid)[0], 0.5 * (phases[1:] + phases[:-1]))

    v = t.values[0]
    naive = np.interp(mid, grid, v.real) + 1j * np.interp(mid, grid, v.imag)
    assert np.allclose(np.abs(naive), 0.5, atol=1e-12)  # -6 dB collapse at the midpoints
    dense = np.linspace(grid[0], grid[-1], 2001)
    naive_dense = np.interp(dense, grid, v.real) + 1j * np.interp(dense, grid, v.imag)
    assert np.abs(naive_dense).min() < 0.55
    assert np.allclose(t.magnitude_at(dense)[0], 1.0, atol=1e-12)

    # Rendered: the envelope of the synthesised order stays at the table magnitude.
    fs = 4000.0
    rpm = np.linspace(1000.0, 4000.0, int(8 * fs))
    x, _ = synthesize_orders(t, rpm, fs, phase="table")
    env = np.abs(signal.hilbert(x))[int(0.5 * fs) : -int(0.5 * fs)]
    assert env.min() > 0.97 and env.max() < 1.03


# --------------------------------------------------------------------------
# from_slices
# --------------------------------------------------------------------------


def test_from_slices_binning_energy_average_and_phasor_phase():
    grid = np.array([1000.0, 2000.0, 3000.0, 4000.0])  # bins: 500-1500-2500-3500-4500
    rpm = [900, 1100, 1400, 1500, 1600, 2400, 2600, 3400, 3500, 3600, 400, 4600]
    amp = [3.0, 4.0, np.nan, 1.0, 2.0, 2.0, 5.0, 5.0, 5.0, 100.0, 100.0, 100.0]
    ph = [np.pi - 0.1, -np.pi + 0.1, np.nan, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    cpx = np.asarray(amp) * np.exp(1j * np.asarray(ph))
    s = _slice(2.0, rpm, amp, cpx)
    t = OrderTable.from_slices([s], grid, unit="Pa")
    assert t.values.shape == (1, 4) and t.amplitude == "peak" and t.unit == "Pa"
    assert t.orders[0] == 2.0 and t.meta["source"] == "from_slices"
    v = t.values[0]
    # Bin 0: frames 900, 1100 (the NaN frame is ignored); phase across the +-pi wrap.
    assert abs(v[0]) == pytest.approx(np.sqrt((9 + 16) / 2))
    assert abs(_wrap(np.angle(v[0]) - np.pi)) < 1e-9
    # Bin 1: frames 1500 (a bin edge belongs to the upper bin), 1600, 2400.
    assert abs(v[1]) == pytest.approx(np.sqrt((1 + 4 + 4) / 3))
    assert np.angle(v[1]) == pytest.approx(0.5)
    # Bin 2: frames 2600 and 3400, and 3500 (inside the last-but-one edge). 3600 is bin 3.
    assert abs(v[2]) == pytest.approx(5.0) and np.angle(v[2]) == pytest.approx(1.0)
    # Bin 3 (3500-4500): frames 3500 (edge -> upper bin) and 3600, and the outliers
    # at 400 and 4600 lie outside every bin and must not count.
    assert abs(v[3]) == pytest.approx(np.sqrt((25 + 10000) / 2))


def test_from_slices_outer_edge_empty_bins_and_outside_frames_dropped():
    grid = np.array([1000.0, 2000.0, 3000.0])  # outer edges 500 and 3500
    s = _slice(1.0, [3500.0, 499.0, 3501.0, 1000.0], [5.0, 50.0, 50.0, 2.0], [5, 50, 50, 2])
    t = OrderTable.from_slices([s], grid)
    assert abs(t.values[0, 2]) == pytest.approx(5.0)  # the outer edge belongs to the last bin
    assert abs(t.values[0, 0]) == pytest.approx(2.0)
    assert np.isnan(t.values[0, 1])  # empty bin
    assert t.magnitude_at(2000.0)[0] == pytest.approx(3.5)  # skipped, interpolated


def test_from_slices_zero_phasor_keeps_magnitude_and_multichannel():
    grid = np.array([1000.0, 2000.0])
    amp = np.array([[2.0, 1.0], [2.0, np.nan], [2.0, np.nan]])
    cpx = np.array([[2.0, 1j], [2.0, np.nan], [2.0, np.nan]], dtype=complex)
    cpx[:, 0] = 0.0  # no phase information in channel 0, but a magnitude
    s0 = _slice(1.0, [1000.0, 1010.0, 990.0], amp, cpx, "rms")
    s1 = _slice(2.0, [1000.0, 2000.0, 2010.0], amp, cpx, "rms")
    t = OrderTable.from_slices([s0, s1], grid, freq_offset=[0.0, 12.0])
    assert t.values.shape == (2, 2, 2) and t.amplitude == "rms"
    assert np.array_equal(t.freq_offset, [0.0, 12.0])
    assert t.values[0, 0, 0] == pytest.approx(2.0 + 0j)  # phase 0 by definition
    assert t.values[0, 0, 1] == pytest.approx(1j)  # amplitude 1, phasor 1j (NaN frames ignored)
    assert np.isnan(t.values[0, 1, 0]) and np.isnan(t.values[0, 1, 1])  # no frames at 2000
    assert abs(t.values[1, 0, 0]) == pytest.approx(2.0)
    assert abs(t.values[1, 1, 0]) == pytest.approx(2.0) and np.isnan(t.values[1, 1, 1])


def test_from_slices_rejects_inconsistent_input():
    grid = np.array([1000.0, 2000.0])
    a = _slice(1.0, [1000.0], [1.0], [1.0], "peak")
    b = _slice(2.0, [1000.0], [1.0], [1.0], "rms")
    with pytest.raises(ValueError, match="convention"):
        OrderTable.from_slices([a, b], grid)
    with pytest.raises(ValueError):
        OrderTable.from_slices([], grid)
    with pytest.raises(ValueError):
        OrderTable.from_slices([a], [1000.0])  # one point: no spacing to bin with
    c = _slice(3.0, [1000.0, 1000.0], [[1.0, 2.0]], [[1.0, 2.0]], "peak")
    with pytest.raises(ValueError):
        OrderTable.from_slices([a, c], grid)


# --------------------------------------------------------------------------
# from_components, with the truth of rota_order.synth as input
# --------------------------------------------------------------------------


def _components_from_truth(sig, shafts):
    """Peak envelopes of the synth truth: order_amplitude * exp(i (phase - 2 pi order rev))."""
    rev = sig.rev if sig.rev.ndim == 2 else sig.rev[:, None]
    carrier = TWO_PI * sig.orders[None, :] * rev[:, shafts]
    env = sig.order_amplitude * np.exp(1j * (sig.order_phase - carrier))
    return OrderComponents(
        orders=sig.orders,
        shafts=np.asarray(shafts, dtype=int),
        waveforms=sig.order_stems,
        envelopes=env,
        rev=sig.rev,
        method="truth",
    )


def test_from_components_recovers_gain_law_and_phase():
    fs, dur = 4000.0, 4.0
    rpm = np.linspace(1000.0, 3000.0, int(fs * dur))
    sig = synth_runup(
        fs,
        rpm,
        [
            dict(order=2.0, amplitude=1.0, phase=0.7, gain_rpm=[1000, 3000], gain=[0.5, 2.0]),
            dict(order=4.5, amplitude=0.8, phase=-1.2),
        ],
    )
    comp = _components_from_truth(sig, [0, 0])
    grid = np.arange(1000.0, 3001.0, 250.0)
    t = OrderTable.from_components(comp, rpm, grid, unit="Pa", meta={"run": 1})
    assert t.amplitude == "peak" and t.unit == "Pa" and t.values.shape == (2, grid.size)
    assert np.array_equal(t.orders, [2.0, 4.5])
    assert t.meta["source"] == "from_components" and t.meta["method"] == "truth"
    assert t.meta["run"] == 1 and "shafts" not in t.meta
    # Order 2: gain law linear in rpm; the energy average over a full bin is
    # sqrt(A_c^2 + (A' w)^2 / 12) -- interior bins are fully populated.
    centre = 0.5 + 0.75 * (grid - 1000.0) / 1000.0
    expected = np.sqrt(centre**2 + (0.75 / 1000.0 * 250.0) ** 2 / 12.0)
    assert np.allclose(np.abs(t.values[0, 1:-1]), expected[1:-1], rtol=2e-3)
    assert np.max(np.abs(_db(np.abs(t.values[0, 1:-1]), centre[1:-1]))) < 0.06
    assert np.allclose(np.abs(t.values[1]), 0.8, rtol=1e-9)
    assert np.allclose(np.angle(t.values[0]), 0.7, atol=1e-9)
    assert np.allclose(np.angle(t.values[1]), -1.2, atol=1e-9)


def test_from_components_two_shafts_multichannel_and_errors():
    fs, dur = 2000.0, 3.0
    n = int(fs * dur)
    r0 = np.linspace(1000.0, 3000.0, n)
    rpm = np.column_stack([r0, 0.8 * r0])  # shaft 1 only reaches 2400 rpm
    sig = synth_runup(
        fs,
        rpm,
        [dict(order=2.0, amplitude=1.0, phase=0.3), dict(order=1.0, shaft=1, amplitude=0.6, phase=-0.4)],
    )
    comp = _components_from_truth(sig, [0, 1])
    grid = np.arange(1000.0, 3001.0, 250.0)
    t = OrderTable.from_components(comp, rpm, grid)
    assert t.meta["shafts"] == [0, 1]
    assert np.allclose(np.abs(t.values[0]), 1.0) and np.allclose(np.angle(t.values[0]), 0.3)
    # Row 1 is binned against shaft 1: bins above its top speed (2400 + half a bin) are empty.
    assert np.array_equal(np.isnan(t.values[1]), grid > 2525.0)
    ok = ~np.isnan(t.values[1])
    assert np.allclose(np.abs(t.values[1, ok]), 0.6) and np.allclose(np.angle(t.values[1, ok]), -0.4)
    with pytest.raises(ValueError, match="shaft"):
        OrderTable.from_components(comp, r0, grid)  # several shafts, one speed
    with pytest.raises(ValueError):
        OrderTable.from_components(comp, rpm[:-1], grid)

    # Channel axis: the second channel is half the first.
    env = np.stack([comp.envelopes, 0.5 * comp.envelopes], axis=-1)
    comp2 = OrderComponents(sig.orders, np.array([0, 1]), env, env, comp.rev, "truth")
    t2 = OrderTable.from_components(comp2, rpm, grid)
    assert t2.values.shape == (2, grid.size, 2)
    assert np.allclose(np.abs(t2.values[..., 1]), 0.5 * np.abs(t2.values[..., 0]), equal_nan=True)


# --------------------------------------------------------------------------
# T9: table -> synthesize_orders -> rpm_order_map -> order_slice -> table
# --------------------------------------------------------------------------

T9_ORDERS = np.array([2.0, 3.5, 5.0, 8.0])


def _t9_table(amplitude):
    grid = np.arange(1000.0, 4001.0, 250.0)
    law = np.array(
        [
            (0.5 + 0.1 * o) * (1.0 + 0.4 * np.sin(TWO_PI * (grid - 1000.0) / 6000.0 + o))
            for o in T9_ORDERS
        ]
    )
    vals = law * np.exp(1j * 0.3 * T9_ORDERS)[:, None]
    if amplitude == "rms":
        vals = vals / np.sqrt(2.0)
    return OrderTable(T9_ORDERS, grid, vals, amplitude, unit="Pa")


@pytest.mark.parametrize("amplitude", ["peak", "rms"])
def test_t9_round_trip_within_0p2_db(amplitude):
    table = _t9_table(amplitude)
    n = int(12.0 * FS)
    rpm = np.linspace(1000.0, 4000.0, n)
    x, info = synthesize_orders(table, rpm, FS, seed=11)
    assert info["output_amplitude"] == "peak" and info["table_amplitude"] == amplitude
    m = rpm_order_map(x, FS, rpm, order_resolution=0.25, amplitude=amplitude)
    slices = [order_slice(m, o) for o in table.orders]
    rebuilt = OrderTable.from_slices(slices, table.rpm, unit="Pa")
    assert rebuilt.amplitude == amplitude
    assert np.all(np.isfinite(rebuilt.values))
    err = _db(np.abs(rebuilt.values), np.abs(table.values))
    interior = err[:, 2:-2]  # 1500 ... 3500 rpm: away from the start-up and end of the run
    assert np.max(np.abs(interior)) < 0.2
    assert np.max(np.abs(err)) < 0.5  # edge bins hold half a bin of frames and filter edges


def test_rms_table_renders_peak_amplitude():
    rms = _t9_table("rms")
    rpm = np.linspace(1000.0, 4000.0, 4000)
    xr, _ = synthesize_orders(rms, rpm, FS, phase="zero")
    xp, _ = synthesize_orders(rms.to_peak(), rpm, FS, phase="zero")
    assert np.allclose(xr, xp, rtol=0, atol=1e-12)
    flat = _flat_table([1.0], mag=2.0, amplitude="rms")
    x0, _ = synthesize_orders(flat, np.full(10, 1200.0), FS, phase="zero")
    assert x0[0] == pytest.approx(2.0 * np.sqrt(2.0))  # RMS 2 -> peak 2.83


def test_t9_resynthesis_reproduces_the_waveform_of_a_synth_runup():
    """Truth from rota_order.synth -> order map -> table -> phase='table' synthesis -> same waveform."""
    n = int(10 * FS)
    rpm = np.linspace(1000.0, 4000.0, n)
    spec = [
        {"order": 2.0, "amplitude": 1.0, "phase": 0.7, "gain_rpm": [1000, 4000], "gain": [0.6, 1.8]},
        {"order": 4.0, "amplitude": 0.5, "phase": -1.2},
        {"order": 6.5, "amplitude": 0.3, "phase": 2.5},
    ]
    sig = synth_runup(FS, rpm, spec, seed=1)
    m = rpm_order_map(sig.x, FS, rpm, order_resolution=0.25, amplitude="peak")  # orders on bins
    grid = np.arange(1000.0, 4001.0, 125.0)
    table = OrderTable.from_slices([order_slice(m, o) for o in sig.orders], grid, unit="Pa")
    inner = slice(int(0.15 * n), int(0.85 * n))
    for k in range(3):
        one = OrderTable(table.orders[k : k + 1], grid, table.values[k : k + 1], "peak")
        est, _ = synthesize_orders(one, rpm, FS, phase="table")
        ref = sig.order_stems[:, k]
        nrmse = np.sqrt(np.mean((est[inner] - ref[inner]) ** 2) / np.mean(ref[inner] ** 2))
        assert nrmse < 0.005, (sig.orders[k], nrmse)
    # The table carries the truth phases (relative to the carrier) in the interior.
    truth = np.array([0.7, -1.2, 2.5])
    assert np.max(np.abs(_wrap(np.angle(table.values[:, 4:-4]) - truth[:, None]))) < 0.01
    x, _ = synthesize_orders(table, rpm, FS, phase="table")
    assert np.sqrt(np.mean((x[inner] - sig.x[inner]) ** 2) / np.mean(sig.x[inner] ** 2)) < 0.005


# --------------------------------------------------------------------------
# synthesize_orders
# --------------------------------------------------------------------------


def test_nyquist_components_are_muted_not_clipped():
    n = int(3 * FS)
    rpm = np.linspace(3000.0, 6000.0, n)
    t = _flat_table([2.0, 50.0], mag=1.0)
    x, stems, info = synthesize_orders(t, rpm, FS, phase="zero", return_stems=True)
    f = 50.0 * rpm / 60.0
    live = f < 0.5 * FS
    assert 0.1 < live.mean() < 0.9  # the run does cross fs / 2 (at 4800 rpm, 40 % of the way)
    assert np.all(stems[~live, 1] == 0.0)
    assert np.abs(stems[live, 1]).max() > 0.99 and np.mean(stems[live, 1] != 0.0) > 0.99
    assert info["muted_fraction"][1] == pytest.approx(np.mean(~live))
    assert info["muted_fraction"][0] == 0.0
    assert np.allclose(x, stems.sum(axis=1), atol=1e-12)
    # Exactly at fs / 2 or at 0 Hz: muted too.
    edge = _flat_table([80.0, -2.0], mag=1.0, freq_offset=[0.0, 100.0])
    xe, ie = synthesize_orders(edge, np.full(50, 3000.0), FS, phase="zero")
    assert np.all(xe == 0.0) and np.allclose(ie["muted_fraction"], 1.0)


def test_zero_and_negative_frequency_muted_and_reported():
    n = int(2 * FS)
    rpm = np.linspace(1500.0, 4500.0, n)
    # f = 100 - rpm / 30 Hz: positive below 3000 rpm, then <= 0.
    sideband = _flat_table([-2.0], mag=1.0, freq_offset=[100.0])
    _, stems, info = synthesize_orders(sideband, rpm, FS, phase="zero", return_stems=True)
    f = 100.0 - rpm / 30.0
    assert np.all(stems[f <= 0.0, 0] == 0.0) and np.abs(stems[f > 0.0, 0]).max() > 0.99
    assert info["muted_fraction"][0] == pytest.approx(np.mean(f <= 0.0))
    # A negative order without carrier offset has negative frequency throughout.
    pure = _flat_table([-1.0], mag=1.0)
    xp, ip = synthesize_orders(pure, rpm, FS, phase="zero")
    assert np.all(xp == 0.0) and ip["muted_fraction"][0] == 1.0


def test_negative_fractional_orders_and_freq_offset_follow_the_instantaneous_frequency():
    fs = 4000.0
    n = int(4 * fs)
    rpm = np.linspace(1200.0, 2400.0, n)
    orders = np.array([-2.0, 0.5, 2.5, 7.25])
    offset = np.array([600.0, 100.0, 0.0, -100.0])
    t = _flat_table(orders, mag=1.0, freq_offset=offset)
    x, stems, info = synthesize_orders(t, rpm, fs, seed=2, return_stems=True)
    expected = orders[None, :] * rpm[:, None] / 60.0 + offset[None, :]
    assert np.all(expected > 0.0) and np.all(expected < 0.5 * fs)
    assert np.all(info["muted_fraction"] == 0.0)
    trim = slice(int(0.4 * fs), -int(0.4 * fs))
    for j in range(orders.size):
        f = _inst_freq(stems[:, j], fs)
        assert np.max(np.abs(f[trim] - expected[trim, j])) < 0.3, j
    assert np.max(np.abs(stems[:, 0])) == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(x, stems.sum(axis=1))
    # Every row is exactly cos(2 pi (order rev + offset t) + phi) with rev = rpm_to_phase(rpm).
    rev = rpm_to_phase(rpm, fs)
    tt = np.arange(n) / fs
    for j in range(orders.size):
        theta = TWO_PI * (orders[j] * rev + offset[j] * tt) + info["phases"][j]
        assert np.allclose(stems[:, j], np.cos(theta), rtol=0, atol=1e-12), j


def test_phase_modes_zero_array_and_validation():
    fs = 4000.0
    t = _flat_table([3.0, 5.0], mag=2.0, phase=1.1)  # the table's own phase is not used
    rpm = np.full(100, 1200.0)
    _, stems, info = synthesize_orders(t, rpm, fs, phase="zero", return_stems=True)
    assert np.allclose(stems[0], 2.0) and info["phase_mode"] == "zero"
    assert np.all(info["phases"] == 0.0) and info["seed"] is None
    ph = np.array([0.4, -1.3])
    _, stems, info = synthesize_orders(t, rpm, fs, phase=ph, return_stems=True)
    assert np.allclose(stems[0], 2.0 * np.cos(ph)) and info["phase_mode"] == "array"
    assert np.array_equal(info["phases"], ph)
    for bad in ("bogus", np.zeros(3), [np.nan, 0.0]):
        with pytest.raises(ValueError):
            synthesize_orders(t, rpm, fs, phase=bad)


def test_random_phase_is_uniform_seeded_and_recorded():
    fs = 4000.0
    t = _flat_table(np.arange(1.0, 9.0), mag=1.0)
    rpm = np.linspace(1000.0, 2000.0, 2000)
    x1, i1 = synthesize_orders(t, rpm, fs, seed=5)
    x2, _ = synthesize_orders(t, rpm, fs, seed=5)
    x3, _ = synthesize_orders(t, rpm, fs, seed=6)
    assert np.array_equal(x1, x2) and not np.array_equal(x1, x3)
    assert i1["seed"] == 5 and i1["phase_mode"] == "random"
    assert np.all((i1["phases"] >= 0.0) & (i1["phases"] < TWO_PI))
    assert np.array_equal(i1["phases"], np.random.default_rng(5).uniform(0.0, TWO_PI, 8))
    # seed=None: a fresh seed is drawn and reported, and reproduces the run.
    xa, ia = synthesize_orders(t, rpm, fs)
    assert isinstance(ia["seed"], int)
    xb, _ = synthesize_orders(t, rpm, fs, seed=ia["seed"])
    assert np.array_equal(xa, xb)
    # A Generator is accepted; the phases it produced are in the info.
    xg, ig = synthesize_orders(t, rpm, fs, seed=np.random.default_rng(5))
    assert ig["seed"] is None and np.array_equal(xg, x1)
    xc, _ = synthesize_orders(t, rpm, fs, phase=ig["phases"])
    assert np.array_equal(xc, x1)
    with pytest.raises(TypeError):
        synthesize_orders(t, rpm, fs, seed=1.5)


def test_table_phase_mode_reproduces_the_table_phase_on_a_bin_order():
    orders = np.array([2.0, 4.0])
    grid = np.arange(1000.0, 4001.0, 250.0)
    base = np.array([2.8, 0.2])  # the first row runs through +pi
    phase = base[:, None] + 0.0004 * (grid[None, :] - 1000.0)
    mag = np.array([1.0, 0.6])[:, None] * np.ones((2, grid.size))
    table = OrderTable(orders, grid, mag * np.exp(1j * phase), "peak")
    rpm = np.linspace(1000.0, 4000.0, int(8 * FS))
    x, info = synthesize_orders(table, rpm, FS, phase="table")
    assert info["phase_mode"] == "table" and np.all(info["phases"] == 0.0)
    m = rpm_order_map(x, FS, rpm, order_resolution=0.25, amplitude="peak")
    slices = [order_slice(m, o) for o in orders]
    inner = slice(5, -5)
    for j, s in enumerate(slices):
        expected = base[j] + 0.0004 * (s.rpm - 1000.0)
        assert np.max(np.abs(_wrap(np.angle(s.complex) - expected)[inner])) < 0.03, j
        assert np.max(np.abs(_db(s.amplitude[inner], mag[j, 0]))) < 0.05, j
    rebuilt = OrderTable.from_slices(slices, grid)
    d = _wrap(np.angle(rebuilt.values) - np.angle(table.values))
    assert np.max(np.abs(d[:, 2:-2])) < 0.03
    # phase_at agrees with the interpolation of the unwrapped table phase.
    assert np.allclose(table.phase_at(grid)[0], phase[0])


def test_smooth_rpm_is_applied_before_interpolation():
    rng = np.random.default_rng(1)
    grid = np.arange(1000.0, 3001.0, 100.0)
    mag = 2.0 + rng.standard_normal((1, grid.size)) * 0.4
    t = OrderTable([2.0], grid, mag.astype(complex), "peak")
    rpm = np.full(20, 2000.0)  # constant speed: x[0] = magnitude at 2000 rpm (zero phase)
    x_raw, i_raw = synthesize_orders(t, rpm, FS, phase="zero")
    x_sm, i_sm = synthesize_orders(t, rpm, FS, phase="zero", smooth_rpm=400.0)
    assert x_raw[0] == pytest.approx(t.magnitude_at(2000.0)[0])
    assert x_sm[0] == pytest.approx(t.magnitude_at(2000.0, smooth_rpm=400.0)[0])
    assert x_sm[0] != pytest.approx(x_raw[0]) and i_sm["smooth_rpm"] == 400.0
    assert i_raw["smooth_rpm"] is None
    with pytest.raises(ValueError):
        synthesize_orders(t, rpm, FS, smooth_rpm=-1.0)


def test_multichannel_table_gives_multichannel_output():
    fs = 4000.0
    grid = np.array([1000.0, 2000.0, 3000.0])
    rng = np.random.default_rng(0)
    vals = (1.0 + rng.random((3, 3, 2))) * np.exp(1j * rng.uniform(-3, 3, (3, 3, 2)))
    vals[1, 1, 1] = np.nan
    table = OrderTable([2.0, 3.0, 4.5], grid, vals, "peak", freq_offset=[0, 50, 0])
    rpm = np.linspace(1000.0, 3000.0, 3000)
    for mode in ("random", "table"):
        x, stems, _ = synthesize_orders(table, rpm, fs, phase=mode, seed=4, return_stems=True)
        assert x.shape == (3000, 2) and stems.shape == (3000, 3, 2)
        assert np.allclose(stems.sum(axis=1), x)
        for c in range(2):
            mono = OrderTable([2.0, 3.0, 4.5], grid, vals[..., c], "peak", freq_offset=[0, 50, 0])
            xc, _ = synthesize_orders(mono, rpm, fs, phase=mode, seed=4)
            assert xc.shape == (3000,) and np.allclose(x[:, c], xc, atol=1e-12)
    # A single channel axis is kept.
    t1 = OrderTable([2.0], grid, np.ones((1, 3, 1)), "peak")
    assert synthesize_orders(t1, rpm, fs)[0].shape == (3000, 1)


def test_empty_rows_stay_silent_and_the_result_does_not_depend_on_chunking(monkeypatch):
    fs = 4000.0
    grid = np.array([1000.0, 3000.0])
    vals = np.array([[1.0, 2.0], [np.nan, np.nan], [1.5, 1.5]], dtype=complex)
    table = OrderTable([2.0, 3.0, 40.0], grid, vals, "peak", freq_offset=[0, 0, 0])
    rpm = np.linspace(1000.0, 3000.0, 5000)
    x, stems, info = synthesize_orders(table, rpm, fs, seed=3, return_stems=True)
    assert np.array_equal(info["empty_rows"], [False, True, False])
    assert np.all(stems[:, 1] == 0.0)
    assert info["muted_fraction"][2] > 0.0  # 40 * 3000 / 60 = 2000 Hz = fs / 2
    monkeypatch.setattr(_ordertable, "_CHUNK", 777)
    x2, stems2, info2 = synthesize_orders(table, rpm, fs, seed=3, return_stems=True)
    assert np.array_equal(x, x2) and np.array_equal(stems, stems2)
    assert np.array_equal(info["muted_fraction"], info2["muted_fraction"])


def test_synthesize_orders_rejects_bad_input():
    t = _flat_table([1.0])
    rpm = np.full(10, 1000.0)
    with pytest.raises(TypeError):
        synthesize_orders("table", rpm, FS)
    with pytest.raises(ValueError):
        synthesize_orders(t, rpm, 0.0)
    with pytest.raises(ValueError):
        synthesize_orders(t, np.ones((5, 2)), FS)
    with pytest.raises(ValueError):
        synthesize_orders(t, np.array([1000.0, np.nan]), FS)
    with pytest.raises(ValueError):
        synthesize_orders(t, np.array([]), FS)
    x, _ = synthesize_orders(t, np.array([1200.0]), FS, phase="zero")  # a single sample
    assert x.shape == (1,) and x[0] == pytest.approx(1.0)


def test_info_is_json_convertible_with_provenance(tmp_path):
    from rota_order._provenance import provenance

    t = _flat_table([1.0, 2.0], freq_offset=[0, 10], unit="Pa")
    _, info = synthesize_orders(t, np.full(100, 1500.0), FS, seed=8)
    rec = provenance(tmp_path / "stimulus", params=info, seed=8)
    assert rec["params"]["phases"] == pytest.approx(list(info["phases"]))
    assert rec["params"]["output_amplitude"] == "peak" and rec["params"]["seed"] == 8


# --------------------------------------------------------------------------
# save / load
# --------------------------------------------------------------------------


def test_save_load_round_trip_is_exact(tmp_path):
    rng = np.random.default_rng(2)
    vals = rng.standard_normal((3, 5, 2)) + 1j * rng.standard_normal((3, 5, 2))
    vals[1, 2, 0] = np.nan
    meta = {"source": "unit test", "n": 3, "nested": {"a": [1, 2.5, "x"], "b": None}}
    t = OrderTable(
        [-2.0, 0.5, 6.0],
        np.array([800.0, 1500.0, 2250.0, 3000.0, 4100.0]),
        vals,
        "rms",
        freq_offset=[600.0, 0.0, -50.0],
        unit="Pa",
        meta=meta,
    )
    npz, side = t.save(tmp_path / "table")
    assert npz.name == "table.npz" and side.name == "table.json" and npz.exists() and side.exists()
    for path in (tmp_path / "table", tmp_path / "table.npz", tmp_path / "table.json", npz):
        back = OrderTable.load(path)
        assert np.array_equal(back.values, t.values, equal_nan=True)
        assert back.values.dtype == complex and back.values.shape == t.values.shape
        assert np.array_equal(back.orders, t.orders) and np.array_equal(back.rpm, t.rpm)
        assert np.array_equal(back.freq_offset, t.freq_offset)
        assert back.amplitude == "rms" and back.unit == "Pa" and back.meta == meta
    # save accepts a path that already carries a suffix; same files.
    npz2, side2 = t.save(tmp_path / "again.npz")
    assert npz2.name == "again.npz" and side2.name == "again.json"
    # The JSON sidecar describes the table (and is strict JSON).
    d = json.loads(side.read_text(encoding="utf-8"))
    assert d["format"] == "rota_order.OrderTable" and d["amplitude"] == "rms" and d["unit"] == "Pa"
    assert d["shape"] == [3, 5, 2] and d["orders"] == [-2.0, 0.5, 6.0]
    assert d["freq_offset"] == [600.0, 0.0, -50.0] and d["meta"] == meta
    assert isinstance(d["rota_order_version"], str) and d["rpm"][0] == 800.0
    # Load needs only the NPZ.
    side.unlink()
    assert OrderTable.load(tmp_path / "table").amplitude == "rms"


def test_save_load_mono_table_and_numpy_meta(tmp_path):
    t = _flat_table(
        [1.0, 2.0],
        mag=2.0,
        phase=0.3,
        meta={"gain": np.float32(1.5), "orders": np.array([1, 2]), "pair": (1, 2)},
    )
    t.save(tmp_path / "mono")
    back = OrderTable.load(tmp_path / "mono")
    assert back.values.shape == (2, 2) and back.n_channels is None
    assert np.array_equal(back.values, t.values)
    assert back.meta == {"gain": 1.5, "orders": [1, 2], "pair": [1, 2]}
    assert back.freq_offset.shape == (2,) and back.unit == ""


# --------------------------------------------------------------------------
# OrderSynth (streaming)
# --------------------------------------------------------------------------


def _stream_table():
    grid = np.array([400.0, 1200.0, 2000.0, 3000.0])
    rng = np.random.default_rng(7)
    vals = (0.5 + rng.random((5, 4))) * np.exp(1j * rng.uniform(-3, 3, (5, 4)))
    # order 100 crosses fs / 2 = 4000 Hz at 2400 rpm; order -2 + 600 Hz is a sideband.
    return OrderTable(
        [1.0, 2.5, -2.0, 30.0, 100.0], grid, vals, "peak", freq_offset=[0.0, 0.0, 600.0, 0.0, 0.0]
    )


def _stream_speed(n, fs=FS):
    t = np.arange(n) / fs
    return 1500.0 + 1000.0 * np.sin(TWO_PI * 0.7 * t)  # 500 ... 2500 rpm


def _random_blocks(n, rng, hi=1500):
    sizes = []
    while sum(sizes) < n:
        sizes.append(int(rng.integers(1, hi)))
    sizes[-1] -= sum(sizes) - n
    return [s for s in sizes if s > 0]


@pytest.mark.parametrize("mode", ["random", "zero", "table"])
def test_orderSynth_matches_synthesize_orders_for_random_block_sizes(mode):
    table = _stream_table()
    n = 20000
    rpm = _stream_speed(n)
    whole, info = synthesize_orders(table, rpm, FS, phase=mode, seed=9)
    assert info["muted_fraction"][4] > 0.0 and info["muted_fraction"][4] < 1.0
    synth = OrderSynth(table, FS, phase=mode, seed=9)
    assert synth.fs == FS and np.array_equal(synth.phases, info["phases"])
    pos, parts = 0, []
    for size in _random_blocks(n, np.random.default_rng(1)):
        parts.append(synth.process(rpm[pos : pos + size]))
        pos += size
    out = np.concatenate(parts)
    assert out.shape == whole.shape and synth.n_samples == n
    assert np.max(np.abs(out - whole)) < 1e-12
    assert np.max(np.abs(whole)) > 1.0
    assert np.allclose(synth.info()["muted_fraction"], info["muted_fraction"])
    # One sample at a time agrees as well.
    one = OrderSynth(table, FS, phase=mode, seed=9)
    first = np.concatenate([one.process(rpm[i : i + 1]) for i in range(300)])
    assert np.max(np.abs(first - whole[:300])) < 1e-12


def test_orderSynth_ramps_and_scalars_match_the_equivalent_speed_curve():
    table = _stream_table()
    knots = [1000.0, 1800.0, 1500.0, 2400.0, 2600.0, 900.0]
    sizes = [700, 1500, 333, 2048, 1]
    synth = OrderSynth(table, FS, seed=4)
    blocks = [synth.process((a, b), n) for a, b, n in zip(knots[:-1], knots[1:], sizes)]
    rpm = np.concatenate(
        [np.linspace(a, b, n, endpoint=False) for a, b, n in zip(knots[:-1], knots[1:], sizes)]
    )
    whole, _ = synthesize_orders(table, rpm, FS, seed=4)
    assert np.max(np.abs(np.concatenate(blocks) - whole)) < 1e-12
    # A chained ramp is a continuous schedule: the block's end value starts the next block.
    assert rpm[sizes[0]] == knots[1]
    # Scalars (constant speed per block) and lists of length two with n != 2.
    consts, lens = [1200.0, 1200.0, 1700.0, 800.0], [400, 1, 999, 250]
    synth.reset()
    got = np.concatenate([synth.process(c, n) for c, n in zip(consts, lens)])
    ref, _ = synthesize_orders(
        table, np.concatenate([np.full(n, c) for c, n in zip(consts, lens)]), FS, seed=4
    )
    assert np.max(np.abs(got - ref)) < 1e-12
    synth.reset()
    as_list = synth.process([1000.0, 1500.0], 100)
    synth.reset()
    assert np.array_equal(as_list, synth.process((1000.0, 1500.0), 100))
    synth.reset()
    two = synth.process(np.array([1000.0, 1500.0]), 2)  # n == 2: per-sample speeds
    assert two.shape == (2,)
    synth.reset()
    assert np.array_equal(two, synth.process([1000.0, 1500.0]))


def test_orderSynth_multichannel_and_smoothing_match_offline():
    grid = np.array([1000.0, 1500.0, 2000.0, 2500.0, 3000.0])
    rng = np.random.default_rng(5)
    vals = (1 + rng.random((2, 5, 2))) * np.exp(1j * rng.uniform(-3, 3, (2, 5, 2)))
    table = OrderTable([2.0, 3.5], grid, vals, "rms")
    rpm = np.linspace(1000.0, 3000.0, 6000)
    whole, _ = synthesize_orders(table, rpm, 4000.0, phase="table", smooth_rpm=700.0)
    synth = OrderSynth(table, 4000.0, phase="table", smooth_rpm=700.0)
    out = np.concatenate([synth.process(rpm[i : i + 1000]) for i in range(0, 6000, 1000)])
    assert out.shape == (6000, 2) and np.max(np.abs(out - whole)) < 1e-12
    assert synth.process(rpm[:0]).shape == (0, 2)  # an empty block changes nothing
    assert synth.n_samples == 6000


def test_orderSynth_reset_repeats_the_run_and_keeps_the_phases():
    table = _stream_table()
    rpm = _stream_speed(5000)
    synth = OrderSynth(table, FS, seed=None)
    phases = synth.phases
    first = np.concatenate([synth.process(rpm[:1234]), synth.process(rpm[1234:])])
    assert synth.n_samples == 5000
    synth.reset()
    assert synth.n_samples == 0 and synth.info()["n"] == 0
    assert np.all(synth.info()["muted_fraction"] == 0.0)
    again = np.concatenate([synth.process(rpm[:777]), synth.process(rpm[777:])])
    assert np.max(np.abs(first - again)) < 1e-12
    assert np.array_equal(synth.phases, phases)
    # Reset in the middle of a block sequence, then start from the first sample again.
    synth.reset()
    synth.process(rpm[:50])
    synth.reset()
    assert np.array_equal(synth.process(rpm[:1000]), first[:1000])


def test_orderSynth_reads_the_table_once_and_validates_input():
    table = _flat_table([2.0], mag=1.0)
    synth = OrderSynth(table, 4000.0, phase="zero")
    table.values[:] = 5.0  # later changes to the table object do not reach the synthesiser
    assert synth.process(1200.0, 4)[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="n is required"):
        synth.process(1200.0)
    with pytest.raises(ValueError, match="n is required"):
        synth.process((1000.0, 2000.0))
    with pytest.raises(ValueError):
        synth.process(np.ones(10), 11)
    with pytest.raises(ValueError):
        synth.process(np.array([1000.0, np.nan, 1000.0]))
    with pytest.raises(ValueError):
        synth.process((1000.0, 2000.0, 3000.0), 10)
    with pytest.raises(ValueError):
        synth.process(np.ones((3, 3)))
    with pytest.raises(ValueError):
        synth.process(1200.0, -1)
    assert synth.process(1200.0, 0).shape == (0,)
    with pytest.raises(ValueError):
        OrderSynth(table, -1.0)
    with pytest.raises(TypeError):
        OrderSynth("table", 4000.0)
    # A failed call leaves the state alone.
    assert synth.n_samples == 4


def test_orderSynth_has_no_drift_over_a_long_run_at_high_order():
    # 5 s at 48 kHz with orders 20 and 60: the shaft angle reaches about 300 revolutions,
    # so a block-wise integration that rounded differently from the whole-curve one would
    # show up as a phase error of 2 pi * 60 * (rounding of ~1e-13 rev) per block.
    fs = 48000.0
    n = int(5 * fs)
    rpm = np.linspace(800.0, 6000.0, n)
    table = OrderTable([20.0, 60.0], [500.0, 6500.0], np.ones((2, 2)), "peak")
    whole, _ = synthesize_orders(table, rpm, fs, seed=2)
    synth = OrderSynth(table, fs, seed=2)
    pos, parts = 0, []
    for size in _random_blocks(n, np.random.default_rng(8), hi=6000):
        parts.append(synth.process(rpm[pos : pos + size]))
        pos += size
    assert np.max(np.abs(np.concatenate(parts) - whole)) < 1e-12
    assert np.max(np.abs(whole)) > 1.5
