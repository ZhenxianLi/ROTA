"""Tests of rota_order.maps against analytic truth (rota_order.synth supplies the signals)."""

import numpy as np
import pytest

from rota_order.angle import angular_resample
from rota_order.maps import Map, rpm_frequency_map, rpm_order_map
from rota_order.synth import rpm_profiles, synth_runup

FS = 8000.0
R0, ACC = 600.0, 600.0  # rpm and rpm/s: 600 -> 3000 rpm in 4 s

_SCALE = {"peak": 1.0, "rms": 1.0 / np.sqrt(2.0)}


def _runup(dur=4.0):
    """Time axis and linear speed curve rpm = R0 + ACC * t."""
    t = np.arange(round(dur * FS)) / FS
    return t, R0 + ACC * t


def _tone(f, amp, phi, n=int(2 * FS)):
    t = np.arange(n) / FS
    return amp * np.cos(2 * np.pi * f * t + phi)


def _db(a, ref):
    return 20.0 * np.log10(np.asarray(a) / ref)


# --------------------------------------------------------------------------
# rpm_frequency_map: scaling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("amplitude", ["peak", "rms"])
@pytest.mark.parametrize("window, rtol", [("hann", 1e-12), ("boxcar", 1e-12), ("flattop", 1e-5)])
@pytest.mark.parametrize("nperseg", [800, 801])  # even and odd frame length
def test_frequency_map_on_bin_tone_reads_peak_or_rms_amplitude(nperseg, window, rtol, amplitude):
    A, phi, k = 1.5, 0.7, 100
    f = k * FS / nperseg  # exactly on bin k
    x = _tone(f, A, phi)
    m = rpm_frequency_map(x, FS, 3000.0, nperseg=nperseg, window=window, amplitude=amplitude)
    assert m.values.shape[0] == nperseg // 2 + 1
    assert m.axis[k] == pytest.approx(f)
    v = m.values[k]
    assert np.allclose(np.abs(v), A * _SCALE[amplitude], rtol=rtol, atol=0)
    # the bin reads the complex amplitude A exp(i phi), in every frame
    assert np.max(np.abs(np.angle(v * np.exp(-1j * phi)))) < 1e-4
    # a component spread over bins: sqrt(sum |V|^2 / enbw) is the amplitude of the convention
    spread = np.sqrt(np.sum(np.abs(m.values) ** 2, axis=0) / m.enbw_bins)
    assert np.allclose(spread, A * _SCALE[amplitude], rtol=max(rtol, 1e-5))
    assert m.amplitude == amplitude and m.window == window and m.domain == "frequency"


def test_frequency_map_peak_and_rms_differ_by_sqrt2_except_at_dc_and_nyquist():
    x = _tone(1000.0, 2.0, 0.3) + 0.7
    pk = rpm_frequency_map(x, FS, 3000.0, resolution=10.0, amplitude="peak")
    rm = rpm_frequency_map(x, FS, 3000.0, resolution=10.0, amplitude="rms")
    assert np.allclose(rm.values[1:-1], pk.values[1:-1] / np.sqrt(2.0), rtol=1e-12, atol=1e-14)
    assert np.allclose(rm.values[0], pk.values[0], rtol=1e-12)  # DC: peak == rms
    assert np.allclose(rm.values[-1], pk.values[-1], rtol=1e-12)  # Nyquist: peak == rms


@pytest.mark.parametrize("amplitude", ["peak", "rms"])
def test_frequency_map_dc_and_nyquist_bins(amplitude):
    n = int(2 * FS)
    m = rpm_frequency_map(np.full(n, 2.0), FS, 3000.0, resolution=10.0, amplitude=amplitude)
    assert np.allclose(m.values[0], 2.0, rtol=1e-12)  # a DC level reads itself, not 2x or 1/sqrt(2)
    assert m.axis[0] == 0.0
    # alternating sequence: the Nyquist component, amplitude A, frequency fs / 2
    A = 1.3
    m = rpm_frequency_map(
        A * (-1.0) ** np.arange(n), FS, 3000.0, resolution=10.0, amplitude=amplitude
    )
    assert m.axis[-1] == pytest.approx(FS / 2)
    assert np.allclose(m.values[-1], A, rtol=1e-12)
    # with a phase, the Nyquist and DC bins read A cos(phi) (real)
    phi = 0.5
    xn = A * np.cos(np.pi * np.arange(n) + phi)
    m = rpm_frequency_map(xn, FS, 3000.0, resolution=10.0, amplitude=amplitude)
    assert np.allclose(m.values[-1], A * np.cos(phi), rtol=1e-12, atol=1e-9)
    assert np.max(np.abs(m.values[-1].imag)) < 1e-9
    m = rpm_frequency_map(
        np.full(n, A * np.cos(phi)),
        FS,
        3000.0,
        resolution=10.0,
        window="boxcar",
        amplitude=amplitude,
    )
    assert np.allclose(m.values[0], A * np.cos(phi), rtol=1e-12)
    assert np.max(np.abs(m.values[1:])) < 1e-12  # a rectangular window has no leakage


def test_frequency_map_dc_offset_does_not_disturb_the_tone_reading():
    A, c = 1.5, 3.0
    m = rpm_frequency_map(_tone(1000.0, A, 0.2) + c, FS, 3000.0, resolution=10.0, amplitude="peak")
    assert np.allclose(m.values[0], c, rtol=1e-12)
    assert np.allclose(np.abs(m.values[100]), A, rtol=1e-12)


@pytest.mark.parametrize("phi", [0.0, 0.7, -2.5, 3.0])
def test_frequency_map_phase_is_referenced_to_t_zero_in_every_frame(phi):
    A = 1.5
    m = rpm_frequency_map(_tone(1000.0, A, phi), FS, 3000.0, resolution=10.0, amplitude="peak")
    assert m.values.shape[1] > 50
    v = m.values[100]
    assert np.allclose(
        v, A * np.exp(1j * phi), atol=1e-10
    )  # also the 77th frame: not just the first
    # a delayed copy shifts the phase by -2 pi f tau, the same in all frames
    tau = 1.25e-4
    x2 = A * np.cos(2 * np.pi * 1000.0 * (np.arange(int(2 * FS)) / FS - tau) + phi)
    v2 = rpm_frequency_map(x2, FS, 3000.0, resolution=10.0, amplitude="peak").values[100]
    assert np.allclose(np.angle(v2 / v), -2 * np.pi * 1000.0 * tau, atol=1e-9)


def test_frequency_map_enbw_of_known_windows():
    x = _tone(1000.0, 1.0, 0.0)
    enbw = {
        w: rpm_frequency_map(x, FS, 3000.0, resolution=10.0, window=w).enbw_bins
        for w in ("boxcar", "hann", "flattop")
    }
    assert enbw["boxcar"] == pytest.approx(1.0, rel=1e-12)
    assert enbw["hann"] == pytest.approx(1.5, rel=1e-12)
    assert enbw["flattop"] == pytest.approx(3.77, abs=0.01)


def test_frequency_map_scalloping_hann_versus_flattop():
    A = 1.5
    x = _tone(1005.0, A, 0.3)  # half a bin off the 10 Hz grid
    hann = rpm_frequency_map(x, FS, 3000.0, resolution=10.0, amplitude="peak", window="hann")
    flat = rpm_frequency_map(x, FS, 3000.0, resolution=10.0, amplitude="peak", window="flattop")
    assert np.all(np.abs(_db(np.abs(hann.values).max(axis=0), A) + 1.42) < 0.05)  # Hann: -1.42 dB
    assert np.all(np.abs(_db(np.abs(flat.values).max(axis=0), A)) < 0.02)  # flat top: <0.02 dB


# --------------------------------------------------------------------------
# rpm_frequency_map: frames, speed tags, axes
# --------------------------------------------------------------------------


def test_frequency_map_frame_times_are_frame_centres():
    t, rpm = _runup()
    x = np.zeros(t.size)
    m = rpm_frequency_map(x, FS, rpm, nperseg=800, overlap=0.75)
    assert m.nperseg == 800 and m.hop == 200
    n_frames = (t.size - 800) // 200 + 1
    assert m.t.shape == (n_frames,) and m.values.shape == (401, n_frames)
    for i in (0, 1, 17, n_frames - 1):
        centre = t[i * 200 : i * 200 + 800].mean()  # mean sample time of the frame
        assert m.t[i] == pytest.approx(centre, abs=1e-12)
    # the final partial frame is dropped, never zero-padded
    assert m.t[-1] + 400 / FS <= t[-1] + 1 / FS


def test_frequency_map_rpm_tags_of_a_run_up_and_a_run_down():
    t, rpm = _runup()
    for r in (rpm, rpm[::-1].copy()):
        m = rpm_frequency_map(np.zeros(t.size), FS, r, nperseg=800)
        assert np.all(m.rpm_min <= m.rpm) and np.all(m.rpm <= m.rpm_max)
        starts = np.arange(m.t.size) * m.hop
        assert np.allclose(m.rpm_min, np.minimum(r[starts], r[starts + 799]))
        assert np.allclose(m.rpm_max, np.maximum(r[starts], r[starts + 799]))
    m = rpm_frequency_map(np.zeros(t.size), FS, rpm, nperseg=800)
    # mean of a linear curve over a frame is its value at the frame centre
    assert np.allclose(m.rpm, R0 + ACC * m.t, atol=1e-9)
    assert np.all(np.diff(m.rpm) > 0)
    # constant speed has no spread
    c = rpm_frequency_map(np.zeros(t.size), FS, 1800.0, nperseg=800)
    assert np.all(c.rpm == 1800.0) and np.all(c.rpm_min == 1800.0) and np.all(c.rpm_max == 1800.0)


def test_frequency_map_follows_a_synth_order_ridge_in_frequency_and_level():
    # slow 1500 -> 2100 rpm sweep, orders 4 and 7 with rising amplitude laws
    t = np.arange(int(4 * FS)) / FS
    rpm = 1500.0 + 150.0 * t
    orders = [
        {"order": 4, "amplitude": 1.0, "phase": 0.4, "gain_rpm": [1500, 2100], "gain": [1.0, 2.0]},
        {"order": 7, "amplitude": 0.5, "phase": -1.0},
    ]
    sig = synth_runup(FS, rpm, orders, seed=1)
    for amplitude in ("peak", "rms"):
        m = rpm_frequency_map(sig.x, FS, rpm, resolution=10.0, amplitude=amplitude)
        for k, o in enumerate(sig.orders):
            f_ridge = o * m.rpm / 60.0
            amp_truth = np.interp(m.t, sig.t, sig.order_amplitude[:, k]) * _SCALE[amplitude]
            for i in range(m.t.size):
                k0 = round(f_ridge[i] / m.resolution)
                band = m.values[k0 - 3 : k0 + 4, i]
                level = np.sqrt(np.sum(np.abs(band) ** 2) / m.enbw_bins)
                assert abs(_db(level, amp_truth[i])) < 0.15
                assert abs(np.argmax(np.abs(band)) - 3) <= 1  # the ridge is where it should be
                assert abs(m.axis[k0 - 3 + np.argmax(np.abs(band))] - f_ridge[i]) <= m.resolution


def test_frequency_map_max_freq_truncates_the_axis():
    x = _tone(1000.0, 1.0, 0.0)
    full = rpm_frequency_map(x, FS, 3000.0, resolution=10.0)
    m = rpm_frequency_map(x, FS, 3000.0, resolution=10.0, max_freq=1500.0)
    assert m.axis[-1] == 1500.0 and m.values.shape[0] == m.axis.size == 151
    assert np.array_equal(m.values, full.values[:151])
    m = rpm_frequency_map(x, FS, 3000.0, resolution=10.0, max_freq=1505.0)
    assert m.axis[-1] == 1500.0  # bins above max_freq are discarded, none is added
    assert (
        rpm_frequency_map(x, FS, 3000.0, resolution=10.0, max_freq=1e6).axis.size == full.axis.size
    )


def test_frequency_map_resolution_nperseg_and_defaults():
    x = _tone(1000.0, 1.0, 0.0)
    m = rpm_frequency_map(x, FS, 3000.0)
    assert m.nperseg == 800 and m.resolution == pytest.approx(10.0)  # 0.1 s frames
    m = rpm_frequency_map(x, FS, 3000.0, resolution=5.0)
    assert m.nperseg == 1600 and m.resolution == pytest.approx(5.0) and m.axis[-1] == FS / 2
    m = rpm_frequency_map(x, FS, 3000.0, resolution=7.0)  # not a divisor of fs
    assert m.nperseg == round(FS / 7.0)
    assert m.resolution == pytest.approx(FS / m.nperseg)
    m = rpm_frequency_map(x, FS, 3000.0, nperseg=400)
    assert m.resolution == pytest.approx(20.0) and m.values.shape[0] == 201
    assert isinstance(m, Map)
    assert np.array_equal(m.magnitude, np.abs(m.values))


@pytest.mark.parametrize("overlap", [0.0, 0.5, 0.75, 0.9])
def test_frequency_map_overlap_sets_hop_and_frame_count(overlap):
    n = int(2 * FS)
    m = rpm_frequency_map(np.zeros(n), FS, 3000.0, nperseg=800, overlap=overlap)
    hop = round(800 * (1 - overlap))
    assert m.hop == hop
    assert m.values.shape[1] == (n - 800) // hop + 1
    assert np.allclose(np.diff(m.t), hop / FS)


def test_frequency_map_multichannel_shape_and_independence():
    A = 1.5
    X = np.column_stack([_tone(1000.0, A, 0.1), _tone(1000.0, 2 * A, 0.9), _tone(1500.0, A, 0.0)])
    m = rpm_frequency_map(X, FS, 3000.0, resolution=10.0, amplitude="peak")
    one = rpm_frequency_map(X[:, 1], FS, 3000.0, resolution=10.0, amplitude="peak")
    assert m.values.shape == (401, one.values.shape[1], 3)
    assert one.values.ndim == 2
    assert np.array_equal(m.values[:, :, 1], one.values)
    assert np.allclose(np.abs(m.values[100, :, 0]), A) and np.allclose(
        np.abs(m.values[100, :, 1]), 2 * A
    )
    assert np.max(np.abs(m.values[100, :, 2])) < 1e-10  # channel 3 has nothing at 1 kHz
    assert np.allclose(np.abs(m.values[150, :, 2]), A)
    assert m.t.shape == m.rpm.shape == (m.values.shape[1],)  # tags are shared by the channels


def test_frequency_map_validation():
    x = _tone(1000.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        rpm_frequency_map(x, FS, 3000.0, window="not-a-window")
    for bad in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="overlap"):
            rpm_frequency_map(x, FS, 3000.0, overlap=bad)
    for bad in ("power", "RMS", None):
        with pytest.raises(ValueError, match="amplitude"):
            rpm_frequency_map(x, FS, 3000.0, amplitude=bad)
    with pytest.raises(ValueError, match="shorter than one frame"):
        rpm_frequency_map(x[:700], FS, 3000.0, nperseg=800)
    with pytest.raises(ValueError):
        rpm_frequency_map(x, FS, np.full(x.size - 1, 3000.0))  # wrong length
    r = np.full(x.size, 3000.0)
    r[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        rpm_frequency_map(x, FS, r)
    with pytest.raises(ValueError):
        rpm_frequency_map(np.zeros((x.size, 2, 2)), FS, 3000.0)


# --------------------------------------------------------------------------
# rpm_order_map
# --------------------------------------------------------------------------

_MAP_ORDERS = [
    {"order": 2, "amplitude": 1.0, "phase": 0.4},
    {"order": 2.5, "amplitude": 0.5, "phase": -1.2},
    {"order": 3, "amplitude": 1.0, "phase": 2.0},
    {"order": 8, "amplitude": 0.3, "phase": 1.0},
]


def _order_run():
    _t, rpm = _runup()
    return rpm, synth_runup(FS, rpm, _MAP_ORDERS, seed=1)


def test_order_map_axis_resolution_and_truncation():
    rpm, sig = _order_run()
    m = rpm_order_map(sig.x, FS, rpm, order_resolution=0.25, max_order=16)
    assert m.domain == "order"
    assert m.axis[0] == 0.0 and m.axis[-1] == pytest.approx(16.0)
    assert m.resolution == pytest.approx(0.25)
    assert np.allclose(np.diff(m.axis), 0.25)
    assert m.values.shape[0] == m.axis.size == 65
    assert m.nperseg == 160  # 1 / 0.25 = 4 revolutions at 40 samples/rev
    m2 = rpm_order_map(sig.x, FS, rpm, order_resolution=0.125, max_order=16)
    assert m2.resolution == pytest.approx(0.125) and m2.nperseg == 320
    # a resolution that does not divide the angular rate: the frame length is rounded
    m5 = rpm_order_map(sig.x, FS, rpm, order_resolution=0.26, max_order=16)
    assert m5.nperseg == round(40 / 0.26) == 154
    assert m5.resolution == pytest.approx(40 / 154)
    # explicit samples_per_rev is passed on: 64 samples/rev, bins up to max_order 25.6
    m3 = rpm_order_map(sig.x, FS, rpm, samples_per_rev=64)
    assert m3.nperseg == 256 and m3.axis[-1] == pytest.approx(25.5)  # last bin <= 0.4 * 64
    assert m3.axis[-1] + 0.25 > 0.4 * 64
    # default max_order is 0.45 * fs / (rpm_max / 60) = 72 (rpm_max just below 3000)
    m4 = rpm_order_map(sig.x, FS, rpm)
    assert 71.5 < m4.axis[-1] <= 72.1


@pytest.mark.parametrize("amplitude", ["peak", "rms"])
def test_order_map_reads_amplitude_and_phase_of_every_order_at_every_speed(amplitude):
    rpm, sig = _order_run()
    m = rpm_order_map(sig.x, FS, rpm, order_resolution=0.25, max_order=16, amplitude=amplitude)
    nf = m.values.shape[1]
    inner = slice(3, nf - 3)  # edge frames hold the resampling filter's start-up
    for k, o in enumerate(sig.orders):
        v = m.values[round(o / 0.25), inner]
        amp, phase = sig.order_amplitude[0, k] * _SCALE[amplitude], sig.order_phase[0, k]
        assert np.max(np.abs(_db(np.abs(v), amp))) < 0.01
        assert np.max(np.abs(np.angle(v * np.exp(-1j * phase)))) < 1e-3
    # elsewhere the map is empty (no leakage between bins of unrelated orders)
    far = m.values[round(5.0 / 0.25), inner]
    assert np.max(np.abs(far)) < 1e-4


def test_order_map_min_rpm_leaves_out_frames_below_it_instead_of_misplacing_them():
    # Run-up from 600 rpm; min_rpm = 1000 floors the speed of the angle, so a frame below it would
    # carry a wrong angle (and the floor speed). Those frames must be absent; the others exact.
    rpm, sig = _order_run()
    full = rpm_order_map(sig.x, FS, rpm, order_resolution=0.25, max_order=16)
    m = rpm_order_map(sig.x, FS, rpm, order_resolution=0.25, max_order=16, min_rpm=1000.0)
    assert m.values.shape[1] < full.values.shape[1]
    assert m.rpm_min.min() >= 1000.0 * (1 - 1e-9)
    assert m.t.size == m.rpm.size == m.values.shape[1]
    inner = slice(3, m.values.shape[1] - 3)
    for k, o in enumerate(sig.orders):
        v = m.values[round(o / 0.25), inner]
        amp = sig.order_amplitude[0, k] / np.sqrt(2.0)  # default amplitude is rms
        assert np.max(np.abs(_db(np.abs(v), amp))) < 0.01
    # the frame speed is the true speed at the frame centre (linear run-up: rpm = R0 + ACC t)
    assert np.allclose(m.rpm[inner], R0 + ACC * m.t[inner], atol=1.0)
    with pytest.raises(ValueError, match="min_rpm"):
        rpm_order_map(sig.x, FS, rpm, order_resolution=0.25, max_order=16, min_rpm=5000.0)


@pytest.mark.parametrize(
    "kind, lo, hi, dur",
    [
        ("linear", 800.0, 6000.0, 1.5),  # fast run-up, ROADMAP T6
        ("exponential", 6000.0, 800.0, 3.0),  # exponential run-down, T3
        ("s_curve", 700.0, 5000.0, 2.0),
    ],
)
def test_order_map_is_exact_for_fast_and_non_linear_sweeps(kind, lo, hi, dur):
    fs = 12000.0
    t = np.arange(int(dur * fs)) / fs
    rpm = rpm_profiles(t, kind, start=lo, stop=hi)
    orders = [
        {"order": 2, "amplitude": 1.0, "phase": 0.5},
        {"order": 6, "amplitude": 0.5, "phase": -0.5},
        {"order": 12, "amplitude": 0.25, "phase": 1.0},
    ]
    sig = synth_runup(fs, rpm, orders, seed=2)
    m = rpm_order_map(sig.x, fs, rpm, order_resolution=0.25, max_order=24, amplitude="peak")
    inner = slice(3, m.t.size - 3)
    for k, o in enumerate(sig.orders):
        v = m.values[round(o / 0.25), inner]
        assert np.max(np.abs(_db(np.abs(v), sig.order_amplitude[0, k]))) < 0.01
        assert np.max(np.abs(np.angle(v * np.exp(-1j * sig.order_phase[0, k])))) < 1e-3
    # the speed tags follow the sweep, whatever its direction
    assert np.all(np.diff(m.rpm) * np.sign(hi - lo) > 0)


def test_order_map_follows_an_amplitude_law():
    _t, rpm = _runup()
    order = {
        "order": 4,
        "amplitude": 1.0,
        "phase": 0.2,
        "gain_rpm": [600, 3000],
        "gain": [1.0, 3.0],
    }
    sig = synth_runup(FS, rpm, [order], seed=1)
    m = rpm_order_map(sig.x, FS, rpm, order_resolution=0.25, max_order=16, amplitude="peak")
    truth = np.interp(m.t, sig.t, sig.order_amplitude[:, 0])
    v = np.abs(m.values[16])
    assert np.max(np.abs(_db(v[3:-3], truth[3:-3]))) < 0.1
    assert truth.max() / truth.min() > 2.5


def test_order_map_resolves_half_orders_at_idle_where_a_frequency_map_cannot():
    # orders 2, 2.5 and 3 start at 600 rpm (10 Hz): half an order is 5 Hz
    rpm, sig = _order_run()
    m = rpm_order_map(sig.x, FS, rpm, order_resolution=0.25, max_order=16, amplitude="peak")
    assert m.rpm[0] < 800.0  # the first frames are the idle ones
    for k, o in enumerate(sig.orders[:3]):
        v = np.abs(m.values[round(o / 0.25), 3:8])
        assert np.max(np.abs(_db(v, sig.order_amplitude[0, k]))) < 0.1
    # 10 Hz bins of the frequency map smear the three together at the same speed
    f = rpm_frequency_map(sig.x, FS, rpm, resolution=10.0, amplitude="peak")
    f_half = 2.5 * f.rpm[0] / 60.0
    kb = int(np.argmin(np.abs(f.axis - f_half)))
    assert (
        abs(_db(np.abs(f.values[kb, 0]), 0.5)) > 3.0
    )  # reading is more than 3 dB off the true 0.5


def test_order_map_main_lobe_has_the_same_width_in_orders_at_every_speed():
    _t, rpm = _runup()
    A, o = 1.0, 4.125  # half a bin off the 0.25-order grid
    sig = synth_runup(FS, rpm, [{"order": o, "amplitude": A, "phase": 0.0}], seed=1)

    def width_in_orders(values, bin_orders):
        mag = np.abs(values)
        return np.sum(mag >= 0.1 * mag.max()) * bin_orders

    om = rpm_order_map(sig.x, FS, rpm, order_resolution=0.25, max_order=16, amplitude="peak")
    w = [
        width_in_orders(om.values[:, i], om.resolution) for i in (3, om.t.size // 2, om.t.size - 4)
    ]
    assert w[0] == w[1] == w[2]
    fm = rpm_frequency_map(sig.x, FS, rpm, resolution=10.0, amplitude="peak")
    first, last = 3, fm.t.size - 4
    wf = [
        width_in_orders(fm.values[:, i], fm.resolution / (fm.rpm[i] / 60.0)) for i in (first, last)
    ]
    assert (
        wf[0] > 3.5 * wf[1]
    )  # a fixed bandwidth in Hz is 5x wider in orders at 600 than at 3000 rpm


def test_order_map_frame_times_and_speed_tags_follow_the_angle_frames():
    _t, rpm = _runup()
    sig = synth_runup(FS, rpm, [{"order": 4, "amplitude": 1.0, "phase": 0.0}], seed=1)
    m = rpm_order_map(sig.x, FS, rpm, order_resolution=0.25, max_order=16)
    spr = 40
    centre_rev = (np.arange(m.t.size) * m.hop + 0.5 * (m.nperseg - 1)) / spr
    t_exact = (-R0 + np.sqrt(R0**2 + 120.0 * ACC * centre_rev)) / ACC
    assert np.max(np.abs(m.t - t_exact)) < 1e-4
    assert np.all(np.diff(m.t) > 0)
    assert np.all(np.diff(m.rpm) > 0)
    assert np.all(m.rpm_min <= m.rpm) and np.all(m.rpm <= m.rpm_max)
    # the tags are the mean, minimum and maximum speed of the samples of each frame
    a = angular_resample(sig.x, FS, rpm, max_order=16)
    for i in (0, 1, 50, m.t.size - 1):
        seg = a.rpm[i * m.hop : i * m.hop + m.nperseg]
        assert m.rpm[i] == pytest.approx(seg.mean(), rel=1e-12)
        assert m.rpm_min[i] == seg.min() and m.rpm_max[i] == seg.max()
    # frames are uniform in angle, so the tagged speed follows the run-up curve at the frame time
    assert np.max(np.abs(m.rpm / (R0 + ACC * m.t) - 1.0)) < 5e-3
    # a frame spans 4 revolutions, during which the speed changes
    assert np.all(m.rpm_max - m.rpm_min > 0)


@pytest.mark.parametrize("overlap", [0.0, 0.5, 0.75])
def test_order_map_overlap_sets_hop_in_angle_samples(overlap):
    rpm, sig = _order_run()
    m = rpm_order_map(sig.x, FS, rpm, max_order=16, overlap=overlap)
    a = angular_resample(sig.x, FS, rpm, max_order=16)
    hop = round(m.nperseg * (1 - overlap))
    assert m.hop == hop
    assert m.values.shape[1] == (a.x.shape[0] - m.nperseg) // hop + 1
    assert m.t.shape == m.rpm.shape == m.rpm_min.shape == m.rpm_max.shape == (m.values.shape[1],)


def test_maps_are_linear_and_leave_their_inputs_alone():
    rpm, sig = _order_run()
    x1, x2 = sig.order_stems[:, 0].copy(), sig.order_stems[:, 3].copy()  # orders 2 and 8
    mix = (1.5 * x1 + 0.5 * x2).copy()
    rpm_in = rpm.copy()
    om = rpm_order_map(mix, FS, rpm_in, max_order=16, amplitude="peak")
    o1 = rpm_order_map(x1, FS, rpm, max_order=16, amplitude="peak")
    o2 = rpm_order_map(x2, FS, rpm, max_order=16, amplitude="peak")
    assert np.allclose(om.values, 1.5 * o1.values + 0.5 * o2.values, rtol=0, atol=1e-12)
    fm = rpm_frequency_map(mix, FS, rpm_in, resolution=10.0, amplitude="peak")
    f1 = rpm_frequency_map(x1, FS, rpm, resolution=10.0, amplitude="peak")
    f2 = rpm_frequency_map(x2, FS, rpm, resolution=10.0, amplitude="peak")
    assert np.allclose(fm.values, 1.5 * f1.values + 0.5 * f2.values, rtol=0, atol=1e-12)
    assert np.array_equal(mix, 1.5 * x1 + 0.5 * x2) and np.array_equal(rpm_in, rpm)


def test_order_map_dc_offset_lands_in_the_dc_bin_only():
    rpm, sig = _order_run()
    m = rpm_order_map(sig.x + 2.0, FS, rpm, order_resolution=0.25, max_order=16, amplitude="peak")
    inner = slice(3, m.t.size - 3)
    assert np.allclose(m.values[0, inner], 2.0, atol=1e-4)  # DC reads its level
    k = round(2.0 / 0.25)
    assert np.max(np.abs(_db(np.abs(m.values[k, inner]), 1.0))) < 0.01  # order 2 unaffected


def test_order_map_multichannel_shape_and_identical_processing():
    rpm, sig = _order_run()
    X = np.column_stack([sig.x, 0.5 * sig.x, -sig.x])
    m = rpm_order_map(X, FS, rpm, order_resolution=0.25, max_order=16, amplitude="peak")
    one = rpm_order_map(sig.x, FS, rpm, order_resolution=0.25, max_order=16, amplitude="peak")
    assert one.values.ndim == 2
    assert m.values.shape == (65, one.values.shape[1], 3)
    assert np.allclose(m.values[:, :, 0], one.values, atol=1e-12)
    assert np.allclose(m.values[:, :, 1], 0.5 * one.values, atol=1e-12)
    assert np.allclose(m.values[:, :, 2], -one.values, atol=1e-12)
    assert np.array_equal(m.t, one.t) and np.array_equal(m.rpm, one.rpm)


def test_order_map_peak_and_rms_conventions_differ_by_sqrt2():
    rpm, sig = _order_run()
    pk = rpm_order_map(sig.x, FS, rpm, max_order=16, amplitude="peak")
    rm = rpm_order_map(sig.x, FS, rpm, max_order=16, amplitude="rms")
    assert np.allclose(rm.values[1:-1], pk.values[1:-1] / np.sqrt(2.0), atol=1e-12)
    assert np.allclose(rm.values[0], pk.values[0], atol=1e-12)
    assert pk.amplitude == "peak" and rm.amplitude == "rms"


def test_order_map_passes_speed_floor_and_delay_to_the_resampler():
    t = np.arange(int(3 * FS)) / FS
    rpm = 4000.0 * t / 3.0  # standstill start
    sig = synth_runup(FS, rpm, [{"order": 4, "amplitude": 1.0, "phase": 0.0}], seed=1)
    m = rpm_order_map(sig.x, FS, rpm, max_order=16, min_rpm=200.0)
    a = angular_resample(sig.x, FS, rpm, max_order=16, min_rpm=200.0)
    assert m.rpm.min() >= 200.0 - 1e-9 and a.rpm.min() == pytest.approx(200.0)
    assert np.all(np.isfinite(m.values))
    # the reference lags by 50 ms; delay compensates the angle, so the phase becomes constant
    d = 0.05
    t2 = np.arange(int(3 * FS)) / FS
    true = 1000.0 + 1000.0 * t2
    sig2 = synth_runup(FS, true, [{"order": 4, "amplitude": 1.0, "phase": 0.3}], seed=1)
    ref = 1000.0 + 1000.0 * (t2 - d)
    good = rpm_order_map(sig2.x, FS, ref, max_order=16, amplitude="peak", delay=d)
    v = good.values[16, 4:-6]  # order 4 is on bin 16 of the 0.25 grid
    assert np.max(np.abs(np.angle(v * np.exp(-0.3j)))) < 1e-3
    assert np.max(np.abs(_db(np.abs(v), 1.0))) < 0.02
    bad = rpm_order_map(sig2.x, FS, ref, max_order=16, amplitude="peak")
    assert np.max(np.abs(_db(np.abs(bad.values[16, 4:-6]), 1.0))) > 2.0


def test_order_map_validation():
    rpm, sig = _order_run()
    for bad in ("power", "RMS", None):
        with pytest.raises(ValueError, match="amplitude"):
            rpm_order_map(sig.x, FS, rpm, amplitude=bad)
    for bad in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="overlap"):
            rpm_order_map(sig.x, FS, rpm, max_order=16, overlap=bad)
    with pytest.raises(ValueError, match="shorter than one frame"):
        rpm_order_map(sig.x[:2000], FS, rpm[:2000], max_order=16)  # < 4 revolutions
    with pytest.raises(ValueError):
        rpm_order_map(sig.x, FS, rpm[:-1], max_order=16)
