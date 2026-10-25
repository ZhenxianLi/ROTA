"""Tests of rota.slices against analytic signals (no rota.synth dependency)."""

import dataclasses

import numpy as np
import pytest

from rota.maps import rpm_frequency_map, rpm_order_map
from rota.slices import OrderSlice, Spectrum, order_slice, order_spectrum
from rota.speed import rpm_to_phase

FS = 8000.0
TRIM = 3  # frames dropped at each end (filter and window edge effects)


def _runup(duration=3.0, lo=1200.0, hi=3000.0):
    """Linear run-up: speed in rpm and shaft angle in revolutions."""
    n = int(duration * FS)
    rpm = np.linspace(lo, hi, n)
    return rpm, rpm_to_phase(rpm, FS)


def _order_map(x, rpm, **kw):
    kw.setdefault("max_order", 16)
    return rpm_order_map(x, FS, rpm, **kw)


def _db(a, ref):
    return 20.0 * np.log10(np.asarray(a) / ref)


def _mid(a):
    return a[TRIM:-TRIM]


# --------------------------------------------------------------------------
# order_slice, order map
# --------------------------------------------------------------------------


@pytest.mark.parametrize("amplitude, scale", [("peak", 1.0), ("rms", 1 / np.sqrt(2))])
@pytest.mark.parametrize("order", [4.0, 2.125])  # on bin and half a bin off
def test_order_map_level_without_scalloping(amplitude, scale, order):
    A = 1.5
    rpm, rev = _runup()
    x = A * np.cos(2 * np.pi * order * rev + 0.7)
    m = _order_map(x, rpm, order_resolution=0.25, amplitude=amplitude)
    s = order_slice(m, order)
    assert isinstance(s, OrderSlice)
    assert s.amplitude.shape == (m.values.shape[1],)
    assert np.all(np.abs(_db(_mid(s.amplitude), A * scale)) < 0.02)


def test_peak_method_shows_scalloping_off_bin_only():
    A = 1.5
    rpm, rev = _runup()
    on = _order_map(A * np.cos(2 * np.pi * 4.0 * rev), rpm, amplitude="peak")
    off = _order_map(A * np.cos(2 * np.pi * 2.125 * rev), rpm, amplitude="peak")
    p_on = order_slice(on, 4.0, method="peak").amplitude
    p_off = order_slice(off, 2.125, method="peak").amplitude
    assert np.all(np.abs(_db(_mid(p_on), A)) < 0.02)
    # Hann window, half a bin off: 1.42 dB below the true amplitude.
    assert np.all(np.abs(_db(_mid(p_off), A) + 1.42) < 0.1)
    b_off = order_slice(off, 2.125, method="band").amplitude
    assert np.all(_mid(b_off) > _mid(p_off) * 1.1)


def test_complex_phase_on_bin_order_map():
    A, phi = 1.5, 0.7
    rpm, rev = _runup()
    x = A * np.cos(2 * np.pi * 4.0 * rev + phi)
    m = _order_map(x, rpm, amplitude="peak")
    s = order_slice(m, 4.0)
    assert s.complex.shape == s.amplitude.shape
    assert np.iscomplexobj(s.complex)
    c = _mid(s.complex)
    assert np.all(np.abs(np.angle(c * np.exp(-1j * phi))) < 0.02)
    assert np.all(np.abs(_db(np.abs(c), A)) < 0.02)
    # RMS convention: complex value carries A / sqrt(2)
    m_rms = _order_map(x, rpm, amplitude="rms")
    c_rms = _mid(order_slice(m_rms, 4.0).complex)
    assert np.allclose(np.abs(c_rms), A / np.sqrt(2), rtol=3e-3)


def test_band_result_independent_of_bandwidth_for_a_pure_line():
    A = 1.0
    rpm, rev = _runup()
    x = A * np.cos(2 * np.pi * 6.1 * rev)
    m = _order_map(x, rpm, amplitude="peak")
    s_def = order_slice(m, 6.1)
    s_wide = order_slice(m, 6.1, bandwidth=2.0)
    assert s_def.bandwidth == pytest.approx(4 * m.resolution)
    assert s_wide.bandwidth == 2.0
    assert np.all(np.abs(_db(_mid(s_wide.amplitude), A)) < 0.02)
    assert np.allclose(_mid(s_def.amplitude), _mid(s_wide.amplitude), rtol=5e-3)


def test_slice_metadata():
    rpm, rev = _runup()
    x = np.cos(2 * np.pi * 4.0 * rev)
    m = _order_map(x, rpm, amplitude="rms")
    s = order_slice(m, 4.0)
    assert s.order == 4.0
    assert s.domain == "order"
    assert s.amplitude_convention == "rms"
    assert np.array_equal(s.t, m.t) and np.array_equal(s.rpm, m.rpm)
    s.rpm[0] = -1.0  # a copy: the map is untouched
    assert m.rpm[0] != -1.0


def test_adjacent_orders_do_not_contaminate():
    A1, A2 = 1.0, 3.0  # the neighbour is 9.5 dB stronger
    rpm, rev = _runup()
    for o1, o2 in [(4.0, 4.5), (4.0625, 4.5625)]:  # on bin and half a bin off
        x = A1 * np.cos(2 * np.pi * o1 * rev) + A2 * np.cos(2 * np.pi * o2 * rev + 1.0)
        m = _order_map(x, rpm, order_resolution=0.125, amplitude="peak")
        s1 = order_slice(m, o1)
        s2 = order_slice(m, o2)
        assert np.all(np.abs(_db(_mid(s1.amplitude), A1)) < 0.1)
        assert np.all(np.abs(_db(_mid(s2.amplitude), A2)) < 0.1)


def test_multichannel_order_map():
    amps = (1.0, 2.5)
    rpm, rev = _runup()
    x = np.column_stack(
        [a * np.cos(2 * np.pi * 4.0 * rev + p) for a, p in zip(amps, (0.0, 1.0))]
    )
    m = _order_map(x, rpm, amplitude="peak")
    assert m.values.ndim == 3
    s = order_slice(m, 4.0)
    n_frames = m.values.shape[1]
    assert s.amplitude.shape == (n_frames, 2)
    assert s.complex.shape == (n_frames, 2)
    for ch, a in enumerate(amps):
        assert np.all(np.abs(_db(_mid(s.amplitude[:, ch]), a)) < 0.02)
    # per-channel result equals the single-channel analysis
    m0 = _order_map(x[:, 0], rpm, amplitude="peak")
    assert np.allclose(order_slice(m0, 4.0).amplitude, s.amplitude[:, 0])
    p = order_slice(m, 4.0, method="peak")
    assert p.amplitude.shape == (n_frames, 2)
    assert np.all(np.abs(_db(_mid(p.amplitude[:, 1]), 2.5)) < 0.02)


def test_order_map_nan_where_band_leaves_axis():
    rpm, rev = _runup()
    m = _order_map(np.cos(2 * np.pi * 4.0 * rev), rpm)  # axis up to order 16
    assert m.axis[-1] == pytest.approx(16.0)
    assert np.all(np.isnan(order_slice(m, 16.0).amplitude))  # band reaches 16.5
    assert np.all(np.isnan(order_slice(m, 16.0).complex))
    assert np.all(np.isfinite(order_slice(m, 15.5).amplitude))  # band ends at 16
    assert np.all(np.isnan(order_slice(m, 0.25).amplitude))  # band reaches below DC
    assert np.all(np.isfinite(order_slice(m, 0.5).amplitude))


# --------------------------------------------------------------------------
# order_slice, frequency map
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", [2.0, 4.37])
def test_frequency_map_runup_level(order):
    A = 1.5
    rpm, rev = _runup(duration=4.0)
    x = A * np.cos(2 * np.pi * order * rev + 0.3)
    m = rpm_frequency_map(x, FS, rpm, resolution=5.0, amplitude="peak")
    s = order_slice(m, order)
    assert s.domain == "frequency"
    assert s.bandwidth == pytest.approx(4 * 5.0)
    assert np.all(np.abs(_db(s.amplitude[2:-2], A)) < 0.2)
    # scalloping loss in the peak reading is larger than the band error
    p = order_slice(m, order, method="peak").amplitude
    assert np.min(_db(p[2:-2], A)) < -0.5


def test_frequency_map_smear_allowance_follows_speed_change():
    # A fast sweep of a high order moves the line over ~13 bins during one
    # frame. The smear allowance keeps it inside the band; a fixed band of
    # four bins (speed range removed from the map) loses power.
    A, order = 1.0, 16.0
    rpm, rev = _runup(duration=0.5, lo=600.0, hi=3000.0)
    x = A * np.cos(2 * np.pi * order * rev)
    m = rpm_frequency_map(x, FS, rpm, resolution=10.0, amplitude="peak")
    assert order * np.max(m.rpm_max - m.rpm_min) / 60.0 > 10 * m.resolution
    s = order_slice(m, order)
    assert np.all(np.abs(_db(s.amplitude[1:-1], A)) < 0.2)
    still = dataclasses.replace(m, rpm_min=m.rpm, rpm_max=m.rpm)
    s0 = order_slice(still, order)
    assert np.max(np.abs(_db(s0.amplitude[1:-1], A))) > 0.5
    assert s.bandwidth == s0.bandwidth == pytest.approx(4 * m.resolution)


def test_frequency_map_stationary_line_scalloping_and_phase():
    A, phi = 1.5, 0.7
    n = int(2.0 * FS)
    rpm = np.full(n, 3000.0)  # 50 Hz shaft
    t = np.arange(n) / FS
    # 100 Hz is on a bin of the 5 Hz grid, 102.5 Hz is half a bin off
    on = rpm_frequency_map(A * np.cos(2 * np.pi * 100.0 * t + phi), FS, rpm,
                           resolution=5.0, amplitude="peak")
    off = rpm_frequency_map(A * np.cos(2 * np.pi * 102.5 * t + phi), FS, rpm,
                            resolution=5.0, amplitude="peak")
    s_on = order_slice(on, 2.0)
    assert np.allclose(s_on.amplitude, A, rtol=1e-6)
    assert np.allclose(np.angle(s_on.complex * np.exp(-1j * phi)), 0.0, atol=1e-6)
    assert np.all(np.abs(_db(order_slice(off, 2.05).amplitude, A)) < 0.02)
    assert np.allclose(_db(order_slice(off, 2.05, method="peak").amplitude, A), -1.42, atol=0.05)


def test_frequency_map_nan_beyond_nyquist_and_max_freq():
    rpm, rev = _runup(duration=4.0)
    x = np.cos(2 * np.pi * 20.0 * rev)
    m = rpm_frequency_map(x, FS, rpm, resolution=5.0)  # axis to 4000 Hz
    order = 100.0
    s = order_slice(m, order)
    half = 0.5 * s.bandwidth + 0.5 * order * (m.rpm_max - m.rpm_min) / 60.0
    expect_nan = order * m.rpm / 60.0 + half > m.axis[-1]
    assert expect_nan.any() and not expect_nan.all()
    assert np.array_equal(np.isnan(s.amplitude), expect_nan)
    assert np.array_equal(np.isnan(s.complex), expect_nan)
    # the same through a truncated axis
    m2 = rpm_frequency_map(x, FS, rpm, resolution=5.0, max_freq=300.0)
    s2 = order_slice(m2, 6.0)
    half2 = 0.5 * s2.bandwidth + 0.5 * 6.0 * (m2.rpm_max - m2.rpm_min) / 60.0
    exp2 = 6.0 * m2.rpm / 60.0 + half2 > 300.0 + 1e-6
    assert exp2.any() and not exp2.all()
    assert np.array_equal(np.isnan(s2.amplitude), exp2)


def test_frequency_map_multichannel():
    amps = (1.0, 2.0)
    rpm, rev = _runup(duration=4.0)
    x = np.column_stack([a * np.cos(2 * np.pi * 3.0 * rev) for a in amps])
    m = rpm_frequency_map(x, FS, rpm, resolution=5.0, amplitude="peak")
    s = order_slice(m, 3.0)
    assert s.amplitude.shape == (m.values.shape[1], 2)
    for ch, a in enumerate(amps):
        assert np.all(np.abs(_db(s.amplitude[2:-2, ch], a)) < 0.2)


# --------------------------------------------------------------------------
# argument checks
# --------------------------------------------------------------------------


def test_order_slice_argument_errors():
    rpm, rev = _runup(duration=1.5)
    m = _order_map(np.cos(2 * np.pi * 4.0 * rev), rpm)
    with pytest.raises(ValueError):
        order_slice(m, 4.0, method="max")
    with pytest.raises(ValueError):
        order_slice(m, 0.0)
    with pytest.raises(ValueError):
        order_slice(m, -2.0)
    with pytest.raises(ValueError):
        order_slice(m, 4.0, bandwidth=0.0)


def test_narrow_bandwidth_still_reads_the_nearest_bin():
    A = 1.0
    rpm, rev = _runup()
    m = _order_map(A * np.cos(2 * np.pi * 4.0 * rev), rpm, amplitude="peak")
    s = order_slice(m, 4.0, bandwidth=0.01)
    assert np.all(np.isfinite(s.amplitude))
    assert np.all(np.abs(_db(_mid(s.amplitude), A / np.sqrt(1.5))) < 0.02)  # one bin / enbw


# --------------------------------------------------------------------------
# order_spectrum
# --------------------------------------------------------------------------


def test_order_spectrum_constant_order():
    A = 2.0
    rpm, rev = _runup()
    x = A * np.cos(2 * np.pi * 3.0 * rev)
    m = _order_map(x, rpm, amplitude="peak")
    sp = order_spectrum(m)
    assert isinstance(sp, Spectrum)
    assert sp.domain == "order" and sp.amplitude_convention == "peak"
    assert sp.n_frames == m.values.shape[1]
    assert sp.amplitude.shape == m.axis.shape
    assert np.array_equal(sp.axis, m.axis)
    k = int(np.argmax(sp.amplitude))
    assert sp.axis[k] == pytest.approx(3.0)
    # edge frames carry small filter transients: compare on the inner frames
    sp_in = order_spectrum(m, t_range=(m.t[TRIM], m.t[-TRIM - 1]))
    assert abs(_db(sp_in.amplitude[k], A)) < 0.02
    # elsewhere the spectrum is far below the line
    others = np.delete(sp_in.amplitude, [k - 1, k, k + 1])
    assert np.all(others < 1e-3 * A)
    # power summation over the spectrum recovers the line amplitude
    tot = np.sqrt(np.sum(sp_in.amplitude**2) / sp_in.enbw_bins)
    assert abs(_db(tot, A)) < 0.02


def test_order_spectrum_energy_average_with_rpm_range():
    # amplitude proportional to speed: the average is over energy, not amplitude
    rpm, rev = _runup(duration=4.0)
    x = (rpm / 1000.0) * np.cos(2 * np.pi * 3.0 * rev)
    m = _order_map(x, rpm, amplitude="peak")
    k = int(np.argmin(np.abs(m.axis - 3.0)))
    for lo, hi in [(1800.0, 2200.0), (1500.0, 2700.0), (None, 2000.0), (2500.0, None)]:
        sp = order_spectrum(m, rpm_range=(lo, hi))
        sel = np.ones(m.rpm.size, bool)
        sel &= m.rpm >= (-np.inf if lo is None else lo)
        sel &= m.rpm <= (np.inf if hi is None else hi)
        assert sp.n_frames == sel.sum() > 3
        expected = np.sqrt(np.mean((m.rpm[sel] / 1000.0) ** 2))
        assert sp.amplitude[k] == pytest.approx(expected, rel=0.015)
    sp = order_spectrum(m, rpm_range=(1500.0, 2700.0))
    arithmetic = np.mean(m.rpm[(m.rpm >= 1500) & (m.rpm <= 2700)] / 1000.0)
    assert sp.amplitude[k] > arithmetic * 1.01


def test_order_spectrum_energy_average_of_amplitude_step():
    A1, A2 = 1.0, 3.0
    n = int(2.0 * FS)
    t = np.arange(n) / FS
    rpm = np.full(n, 3000.0)
    a = np.where(t < 1.0, A1, A2)
    x = a * np.cos(2 * np.pi * 100.0 * t)
    m = rpm_frequency_map(x, FS, rpm, resolution=5.0, amplitude="peak")
    sp = order_spectrum(m)
    assert sp.domain == "frequency"
    k = int(np.argmin(np.abs(m.axis - 100.0)))
    energy = np.sqrt(0.5 * (A1**2 + A2**2))
    assert sp.amplitude[k] == pytest.approx(energy, rel=0.03)
    assert abs(sp.amplitude[k] - 0.5 * (A1 + A2)) > 0.1  # not the arithmetic mean
    early = order_spectrum(m, t_range=(0.0, 0.7))
    late = order_spectrum(m, t_range=(1.3, 2.0))
    assert early.amplitude[k] == pytest.approx(A1, rel=1e-6)
    assert late.amplitude[k] == pytest.approx(A2, rel=1e-6)
    assert early.n_frames == np.sum((m.t >= 0.0) & (m.t <= 0.7))


def test_order_spectrum_time_and_speed_ranges_combine():
    rpm, rev = _runup(duration=4.0)
    m = _order_map(np.cos(2 * np.pi * 3.0 * rev), rpm)
    both = order_spectrum(m, rpm_range=(1500.0, 2500.0), t_range=(1.0, 3.0))
    sel = (m.rpm >= 1500) & (m.rpm <= 2500) & (m.t >= 1.0) & (m.t <= 3.0)
    assert both.n_frames == sel.sum()
    assert both.n_frames < order_spectrum(m, rpm_range=(1500.0, 2500.0)).n_frames
    ref = np.sqrt(np.mean(np.abs(m.values[:, sel]) ** 2, axis=1))
    assert np.allclose(both.amplitude, ref)


def test_order_spectrum_multichannel():
    amps = (1.0, 2.5)
    rpm, rev = _runup()
    x = np.column_stack([a * np.cos(2 * np.pi * 3.0 * rev) for a in amps])
    m = _order_map(x, rpm, amplitude="peak")
    sp = order_spectrum(m, t_range=(m.t[TRIM], m.t[-TRIM - 1]))
    assert sp.amplitude.shape == (m.axis.size, 2)
    k = int(np.argmin(np.abs(m.axis - 3.0)))
    for ch, a in enumerate(amps):
        assert abs(_db(sp.amplitude[k, ch], a)) < 0.02


def test_order_spectrum_empty_selection_raises():
    rpm, rev = _runup(duration=1.5)
    m = _order_map(np.cos(2 * np.pi * 3.0 * rev), rpm)
    with pytest.raises(ValueError, match="no frames"):
        order_spectrum(m, rpm_range=(10000.0, 20000.0))
    with pytest.raises(ValueError, match="no frames"):
        order_spectrum(m, t_range=(100.0, 200.0))
    with pytest.raises(ValueError, match="no frames"):
        order_spectrum(m, rpm_range=(1200.0, 1300.0), t_range=(1.4, 1.5))
