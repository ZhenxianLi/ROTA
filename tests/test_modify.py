"""Tests for modify_orders, remap_rpm and resynthesize against analytic truth (rota_order.synth)."""

import json

import numpy as np
import pytest
from scipy import signal

import rota_order.resynth as rs
from rota_order._components import OrderComponents
from rota_order._provenance import jsonable
from rota_order.extract import residual as extract_residual
from rota_order.extract import tracked_harmonics
from rota_order.maps import rpm_order_map
from rota_order.resynth import OrderTable, modify_orders, remap_rpm, resynthesize, synthesize_orders
from rota_order.slices import order_slice
from rota_order.synth import rpm_profiles, synth_runup

TWO_PI = 2 * np.pi
FS = 8000.0
PHASE_TOL = 1e-10  # rounding of 2 pi order rev (thousands of rad) in a recomputed carrier
ORDERS = np.array([2.0, 4.0, 6.0])
PHASES = np.array([0.3, 1.1, 2.0])
SPECS = [
    {"order": 2.0, "amplitude": 1.0, "phase": 0.3, "gain_rpm": [1000, 4000], "gain": [1.0, 2.0]},
    {"order": 4.0, "amplitude": 0.5, "phase": 1.1},
    {"order": 6.0, "amplitude": 0.25, "phase": 2.0, "gain_rpm": [1000, 4000], "gain": [2.0, 0.5]},
]


def _db(a, ref):
    return 20.0 * np.log10(np.asarray(a) / np.asarray(ref))


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def _truth_components(sig, shafts=None):
    """OrderComponents holding the exact synth truth (envelope, waveform, shaft angle)."""
    rev = sig.rev if sig.rev.ndim == 2 else sig.rev[:, None]
    sh = np.zeros(sig.orders.size, dtype=int) if shafts is None else np.asarray(shafts, dtype=int)
    env = sig.order_amplitude * np.exp(1j * (sig.order_phase - TWO_PI * sig.orders * rev[:, sh]))
    return OrderComponents(
        orders=sig.orders.copy(),
        shafts=sh,
        waveforms=sig.order_stems.copy(),
        envelopes=env,
        rev=sig.rev.copy(),
        method="truth",
        params={"fs": FS},
    )


def _truth_table(grid=None, amplitude="peak"):
    """Table with the amplitude laws of SPECS and the start phases as table phase."""
    grid = np.arange(1000.0, 4001.0, 250.0) if grid is None else grid
    rows = []
    for sp in SPECS:
        a = sp["amplitude"] * np.interp(grid, sp.get("gain_rpm", [0, 1]), sp.get("gain", [1, 1]))
        rows.append(a * np.exp(1j * sp["phase"]))
    values = np.array(rows)
    if amplitude == "rms":
        values = values / np.sqrt(2.0)
    return OrderTable(ORDERS, grid, values, amplitude, unit="Pa")


@pytest.fixture(scope="module")
def run():
    t = np.arange(round(6 * FS)) / FS
    rpm = rpm_profiles(t, "linear", start=1500, stop=3000)
    sig = synth_runup(FS, rpm, SPECS, seed=1)
    return sig, rpm, _truth_components(sig)


# --------------------------------------------------------------------------
# facade
# --------------------------------------------------------------------------


def test_public_names_are_exported():
    for name in ("modify_orders", "remap_rpm", "resynthesize", "stationary_excerpt", "ExcerptResult"):
        assert name in rs.__all__ and hasattr(rs, name)


def test_truth_components_follow_the_documented_convention(run):
    sig, _, comp = run
    carrier = np.exp(1j * TWO_PI * sig.orders * sig.rev[:, None])
    assert np.max(np.abs((comp.envelopes * carrier).real - comp.waveforms)) < 1e-12


# --------------------------------------------------------------------------
# modify_orders: components
# --------------------------------------------------------------------------


def test_gain_and_mute_on_components_are_exact(run):
    sig, _, comp = run
    before = (comp.envelopes.copy(), comp.waveforms.copy(), dict(comp.params), comp.orders.copy())
    m = modify_orders(comp, gain_db={2: 6.0, 6.0: -12.0}, mute=[4])
    g2, g6 = 10 ** (6 / 20), 10 ** (-12 / 20)
    assert isinstance(m, OrderComponents) and m is not comp
    assert np.max(np.abs(m.waveforms[:, 0] - g2 * sig.order_stems[:, 0])) < 1e-12
    assert np.max(np.abs(m.waveforms[:, 2] - g6 * sig.order_stems[:, 2])) < 1e-12
    assert np.all(m.waveforms[:, 1] == 0.0) and np.all(m.envelopes[:, 1] == 0.0)
    assert np.max(np.abs(np.abs(m.envelopes[:, 0]) - g2 * sig.order_amplitude[:, 0])) < 1e-12
    assert np.allclose(m.total, g2 * sig.order_stems[:, 0] + g6 * sig.order_stems[:, 2], atol=1e-12)
    # the input is untouched and nothing is shared
    assert np.array_equal(comp.envelopes, before[0]) and np.array_equal(comp.waveforms, before[1])
    assert comp.params == before[2] and np.array_equal(comp.orders, before[3])
    assert not np.shares_memory(m.envelopes, comp.envelopes)
    assert not np.shares_memory(m.rev, comp.rev) and m.method == comp.method


def test_solo_keeps_only_the_listed_orders(run):
    sig, _, comp = run
    s = modify_orders(comp, solo=[4.0])
    assert np.max(np.abs(s.waveforms[:, 1] - sig.order_stems[:, 1])) < 1e-15
    assert np.all(s.waveforms[:, [0, 2]] == 0.0) and s.waveforms.shape == comp.waveforms.shape
    s = modify_orders(comp, solo=(2.0, 4.0), mute=4.0, gain_db={2.0: -6.0})
    assert np.allclose(s.total, 10 ** (-6 / 20) * sig.order_stems[:, 0], atol=1e-12)
    assert np.array_equal(modify_orders(comp, solo=[2.0, 4.0, 6.0]).waveforms, comp.waveforms)
    # -inf dB silences a row
    z = modify_orders(comp, gain_db={6.0: -np.inf})
    assert np.all(z.waveforms[:, 2] == 0.0) and np.all(np.isfinite(z.envelopes))


def test_no_modification_is_a_copy(run):
    _, _, comp = run
    m = modify_orders(comp)
    assert np.array_equal(m.waveforms, comp.waveforms) and np.array_equal(m.envelopes, comp.envelopes)
    assert m.params["modify"] == [{}] and "modify" not in comp.params


def test_order_map_moves_energy_to_the_new_order(run):
    """Order-map slices: the energy of order 4 is found at order 8 and nowhere else."""
    sig, rpm, comp = run
    m = modify_orders(comp, order_map={4.0: 8.0})
    assert list(m.orders) == [2.0, 8.0, 6.0]
    omap = rpm_order_map(m.total, FS, rpm, order_resolution=0.25, amplitude="peak")
    a8 = order_slice(omap, 8.0).amplitude
    assert np.nanmax(np.abs(_db(a8, 0.5))) < 0.05
    assert np.nanmax(order_slice(omap, 4.0).amplitude) < 1e-4  # more than 70 dB below 0.5
    a2 = order_slice(omap, 2.0)
    assert np.nanmax(np.abs(_db(a2.amplitude, np.interp(a2.rpm, [1000, 4000], [1.0, 2.0])))) < 0.05
    # the same input without the map has its energy at order 4
    base = rpm_order_map(comp.total, FS, rpm, order_resolution=0.25, amplitude="peak")
    assert np.nanmax(np.abs(_db(order_slice(base, 4.0).amplitude, 0.5))) < 0.05
    assert np.nanmax(order_slice(base, 8.0).amplitude) < 1e-4


def test_order_map_swap_and_waveform_definition(run):
    sig, rpm, comp = run
    m = modify_orders(comp, order_map={2.0: 4.0, 4.0: 2.0})
    assert list(m.orders) == [4.0, 2.0, 6.0]
    # waveform = Re{envelope exp(i 2 pi new_order rev)}: order 2's law now rides on order 4
    expect = sig.order_amplitude[:, 0] * np.cos(
        sig.order_phase[:, 0] - TWO_PI * 2 * sig.rev + TWO_PI * 4 * sig.rev
    )
    assert np.max(np.abs(m.waveforms[:, 0] - expect)) < PHASE_TOL
    omap = rpm_order_map(m.total, FS, rpm, order_resolution=0.25, amplitude="peak")
    s4, s2 = order_slice(omap, 4.0), order_slice(omap, 2.0)
    assert np.nanmax(np.abs(_db(s4.amplitude, np.interp(s4.rpm, [1000, 4000], [1.0, 2.0])))) < 0.05
    assert np.nanmax(np.abs(_db(s2.amplitude, 0.5))) < 0.05


def test_freq_offset_on_components(run):
    sig, rpm, comp = run
    m = modify_orders(comp, freq_offset={6.0: 25.0})
    assert m.params["freq_offset"] == [0.0, 0.0, 25.0]
    expect = sig.order_amplitude[:, 2] * np.cos(sig.order_phase[:, 2] + TWO_PI * 25.0 * sig.t)
    assert np.max(np.abs(m.waveforms[:, 2] - expect)) < PHASE_TOL
    assert np.array_equal(m.waveforms[:, :2], comp.waveforms[:, :2])  # unchanged rows keep their data
    # independent check: the frequency of the row is order * rpm / 60 + 25 Hz
    f_inst = np.gradient(np.unwrap(np.angle(signal.hilbert(m.waveforms[:, 2]))), 1 / FS) / TWO_PI
    mid = slice(int(1 * FS), int(5 * FS))
    assert np.max(np.abs(f_inst[mid] - (6 * rpm[mid] / 60 + 25.0))) < 0.2
    # offsets accumulate over calls and the record is kept
    m2 = modify_orders(m, freq_offset={6.0: 5.0}, gain_db={6.0: 3.0})
    assert m2.params["freq_offset"] == [0.0, 0.0, 30.0] and len(m2.params["modify"]) == 2
    expect2 = 10 ** (3 / 20) * sig.order_amplitude[:, 2] * np.cos(
        sig.order_phase[:, 2] + TWO_PI * 30.0 * sig.t
    )
    assert np.max(np.abs(m2.waveforms[:, 2] - expect2)) < PHASE_TOL
    json.dumps(jsonable(m2.params), allow_nan=False)


def test_remodulated_row_is_muted_above_nyquist(run):
    _, rpm, comp = run
    m = modify_orders(comp, order_map={6.0: 100.0})  # 100 * rpm / 60 reaches fs / 2 at 2400 rpm
    above, below = rpm > 2410.0, rpm < 2390.0
    assert above.any() and below.any()
    assert np.all(m.waveforms[above, 2] == 0.0) and np.all(m.envelopes[above, 2] == 0.0)
    assert np.all(m.envelopes[below, 2] != 0.0)
    assert np.max(np.abs(m.waveforms[below, 2])) > 0.2


def test_multichannel_components():
    t = np.arange(2000) / FS
    rpm = np.full(t.size, 3000.0)
    sig = synth_runup(FS, rpm, [{"order": 2.0, "amplitude": 1.0, "phase": 0.4}], seed=2)
    c1 = _truth_components(sig)
    env = np.stack([c1.envelopes, 0.5 * c1.envelopes], axis=2)
    wav = np.stack([c1.waveforms, 0.5 * c1.waveforms], axis=2)
    comp = OrderComponents(c1.orders, c1.shafts, wav, env, c1.rev, "truth", {"fs": FS})
    m = modify_orders(comp, gain_db={2.0: 6.0}, order_map={2.0: 3.0})
    assert m.waveforms.shape == (t.size, 1, 2)
    assert np.allclose(m.waveforms[:, :, 1], 0.5 * m.waveforms[:, :, 0])
    expect = 10 ** (6 / 20) * sig.order_amplitude[:, 0] * np.cos(
        sig.order_phase[:, 0] - TWO_PI * 2 * sig.rev + TWO_PI * 3 * sig.rev
    )
    assert np.max(np.abs(m.waveforms[:, 0, 0] - expect)) < PHASE_TOL


# --------------------------------------------------------------------------
# modify_orders: tables
# --------------------------------------------------------------------------


def test_truth_table_reproduces_the_synth_stems(run):
    sig, rpm, _ = run
    x, stems, _ = synthesize_orders(_truth_table(), rpm, FS, phase=PHASES, return_stems=True)
    assert np.max(np.abs(stems - sig.order_stems)) < 1e-12


def test_gain_mute_solo_on_tables_are_exact(run):
    sig, rpm, _ = run
    table = _truth_table()
    before = table.values.copy()
    m = modify_orders(table, gain_db={2.0: 6.0, 6.0: -12.0}, mute=[4.0])
    assert isinstance(m, OrderTable) and m.amplitude == "peak" and m.unit == "Pa"
    assert np.array_equal(table.values, before) and m.meta["modify"][0]["mute"] == [4.0]
    _, stems, _ = synthesize_orders(m, rpm, FS, phase=PHASES, return_stems=True)
    assert np.max(np.abs(stems[:, 0] - 10 ** (6 / 20) * sig.order_stems[:, 0])) < 1e-12
    assert np.max(np.abs(stems[:, 2] - 10 ** (-12 / 20) * sig.order_stems[:, 2])) < 1e-12
    assert np.all(stems[:, 1] == 0.0) and np.all(m.values[1] == 0.0)
    s = modify_orders(table, solo=[6.0])
    _, stems, _ = synthesize_orders(s, rpm, FS, phase=PHASES, return_stems=True)
    assert np.max(np.abs(stems[:, 2] - sig.order_stems[:, 2])) < 1e-12
    assert np.all(stems[:, :2] == 0.0)


def test_table_gain_is_relative_for_rms_tables_and_mute_clears_nan():
    table = _truth_table(amplitude="rms")
    table.values[1, 3] = np.nan  # missing data in a row that is then muted
    table.values[0, 2] = np.nan  # missing data in a row that keeps its gain
    m = modify_orders(table, gain_db={2.0: 6.0}, mute=4.0)
    assert m.amplitude == "rms"
    assert np.all(m.values[1] == 0.0)  # muted rows are exactly zero, also where the input was NaN
    assert np.isnan(m.values[0, 2])
    ok = np.isfinite(table.values[0])
    assert np.allclose(np.abs(m.values[0, ok]), 10 ** (6 / 20) * np.abs(table.values[0, ok]))
    assert np.allclose(np.angle(m.values[0, ok]), np.angle(table.values[0, ok]))


def test_order_map_on_table_moves_energy(run):
    sig, rpm, _ = run
    m = modify_orders(_truth_table(), order_map={4.0: 8.0})
    assert list(m.orders) == [2.0, 8.0, 6.0]
    x, _ = synthesize_orders(m, rpm, FS, phase=PHASES)
    omap = rpm_order_map(x, FS, rpm, order_resolution=0.25, amplitude="peak")
    assert np.nanmax(np.abs(_db(order_slice(omap, 8.0).amplitude, 0.5))) < 0.05
    assert np.nanmax(order_slice(omap, 4.0).amplitude) < 1e-4


def test_freq_offset_on_table(run):
    sig, rpm, _ = run
    table = _truth_table()
    m = modify_orders(table, freq_offset={6.0: 25.0})
    assert np.array_equal(m.freq_offset, [0.0, 0.0, 25.0])
    _, stems, _ = synthesize_orders(m, rpm, FS, phase=PHASES, return_stems=True)
    expect = sig.order_amplitude[:, 2] * np.cos(sig.order_phase[:, 2] + TWO_PI * 25.0 * sig.t)
    assert np.max(np.abs(stems[:, 2] - expect)) < PHASE_TOL
    again = modify_orders(m, freq_offset={6.0: -5.0})
    assert np.array_equal(again.freq_offset, [0.0, 0.0, 20.0]) and len(again.meta["modify"]) == 2


def test_modified_table_survives_save_and_load(tmp_path):
    m = modify_orders(_truth_table(), gain_db={2.0: -np.inf, 4.0: 3.0}, order_map={6.0: 7.0})
    npz, _ = m.save(tmp_path / "t")
    back = OrderTable.load(npz)
    assert np.array_equal(back.orders, m.orders) and np.array_equal(back.values, m.values)
    assert back.meta["modify"][0]["gain_db"]["2"] == "-inf"


# --------------------------------------------------------------------------
# modify_orders: errors
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"gain_db": {3.0: 1.0}},
        {"mute": [5.0]},
        {"solo": [2.0, 9.0]},
        {"order_map": {3.0: 4.0}},
        {"freq_offset": {1.0: 5.0}},
    ],
)
def test_unknown_orders_raise(run, kw):
    _, _, comp = run
    with pytest.raises(ValueError, match="not in the object"):
        modify_orders(comp, **kw)
    with pytest.raises(ValueError, match="not in the object"):
        modify_orders(_truth_table(), **kw)


def test_invalid_arguments_raise(run):
    _, _, comp = run
    table = _truth_table()
    with pytest.raises(TypeError):
        modify_orders(np.zeros(4))
    with pytest.raises(TypeError):
        modify_orders(table, gain_db=[2.0, 3.0])
    with pytest.raises(TypeError):
        modify_orders(table, mute={2.0: 1})
    assert np.all(modify_orders(table, mute={2.0, 4.0}).values[:2] == 0.0)  # a set of orders
    with pytest.raises(ValueError):
        modify_orders(table, gain_db={2.0: np.nan})
    with pytest.raises(ValueError):
        modify_orders(table, gain_db={2.0: np.inf})
    with pytest.raises(ValueError):
        modify_orders(table, order_map={2.0: np.nan})
    with pytest.raises(ValueError):
        modify_orders(table, freq_offset={2.0: np.inf})
    with pytest.raises(ValueError, match="more than once"):
        modify_orders(table, order_map={2.0: 3.0, 2.0000000001: 5.0})
    nofs = OrderComponents(comp.orders, comp.shafts, comp.waveforms, comp.envelopes, comp.rev, "x", {})
    with pytest.raises(ValueError, match="sampling rate"):
        modify_orders(nofs, order_map={2.0: 3.0})
    assert modify_orders(nofs, gain_db={2.0: 1.0}).orders.size == 3  # gains need no rate


# --------------------------------------------------------------------------
# remap_rpm
# --------------------------------------------------------------------------


def test_remap_of_a_table_is_synthesize_orders(run):
    _, rpm, _ = run
    table = _truth_table()
    new = rpm * 1.1
    y, info = remap_rpm(table, new, FS, seed=5)
    ref, ref_info = synthesize_orders(table, new, FS, seed=5)
    assert np.array_equal(y, ref) and info["seed"] == 5 and info["source"] == "table"
    assert info["output_amplitude"] == "peak" and info["outside_grid_fraction"] == 0.0
    with pytest.raises(ValueError, match="components"):
        remap_rpm(table, new, FS, rpm_old=rpm)
    with pytest.raises(ValueError, match="components"):
        remap_rpm(table, new, FS, rpm_grid=np.arange(5.0))


def test_remap_time_with_unchanged_speed_returns_the_orders(run):
    sig, rpm, comp = run
    y, info = remap_rpm(comp, rpm, FS, phase="table")
    assert np.max(np.abs(y - sig.stems["orders"])) < 1e-12
    assert info["mode"] == "time" and info["output_amplitude"] == "peak"
    assert np.all(info["muted_fraction"] == 0.0)
    y2, info2 = remap_rpm(comp, rpm, FS, phase="table", rpm_old=rpm)
    assert np.array_equal(y, y2) and info2["speed_ratio"] == {"min": 1.0, "median": 1.0, "max": 1.0}
    json.dumps(jsonable(info), allow_nan=False)


def test_remap_time_random_phase_is_seeded_and_keeps_the_amplitude_course(run):
    sig, rpm, comp = run
    new = rpm * 1.1
    y1, i1 = remap_rpm(comp, new, FS, seed=7)
    y2, _ = remap_rpm(comp, new, FS, seed=7)
    y3, _ = remap_rpm(comp, new, FS, seed=8)
    # same seed, same result up to rounding: on NumPy 1.x the magnitude of a strided complex
    # array may differ by one ulp between calls, so bit equality is not required
    assert np.allclose(y1, y2, rtol=0.0, atol=1e-12 * np.max(np.abs(y1)))
    assert not np.allclose(y1, y3)
    assert i1["phase_mode"] == "random" and i1["seed"] == 7 and i1["phases"].shape == (3,)
    # the envelope magnitude of every order keeps its time course (same recording, other speed)
    c = tracked_harmonics(y1, FS, new, ORDERS)
    inner = slice(int(1 * FS), int(5 * FS))
    assert np.max(np.abs(_db(np.abs(c.envelopes[inner]), sig.order_amplitude[inner]))) < 0.05


@pytest.mark.parametrize("ratio", [1.0025, 0.9975, 1.05])
def test_remap_frequency_precision(ratio):
    """1 km/h is about 0.25 % in frequency (ROADMAP 6, item 26); demand 1e-5 relative or better."""
    n = round(12 * FS)
    t = np.arange(n) / FS
    rpm = np.full(n, 3000.0)  # order 2 -> 100 Hz
    sig = synth_runup(FS, rpm, [{"order": 2.0, "amplitude": 1.0, "phase": 0.3}], noise_rms=0.01, seed=1)
    comp = tracked_harmonics(sig.x, FS, rpm, [2.0])
    y, _ = remap_rpm(comp, rpm * ratio, FS, seed=3)

    def freq(v):  # slope of the unwrapped analytic phase over the central 8 s
        sel = slice(int(2 * FS), int(10 * FS))
        ph = np.unwrap(np.angle(signal.hilbert(v)))
        return np.polyfit(t[sel], ph[sel], 1)[0] / TWO_PI

    f_old, f_new = freq(sig.x), freq(y)
    assert abs(f_old / 100.0 - 1.0) < 1e-5
    assert abs(f_new / f_old / ratio - 1.0) < 1e-5
    assert abs(f_new / (100.0 * ratio) - 1.0) < 1e-5


def test_remap_frequency_precision_on_a_run_up(run):
    """Accumulated phase of the output order versus the speed ratio, on a run-up."""
    sig, rpm, comp = run
    ratio = 1.0025
    y, _ = remap_rpm(modify_orders(comp, solo=[4.0]), rpm * ratio, FS, seed=1)
    old = modify_orders(comp, solo=[4.0]).total
    sel = (int(1 * FS), int(5 * FS))
    ph_old = np.unwrap(np.angle(signal.hilbert(old)))
    ph_new = np.unwrap(np.angle(signal.hilbert(y)))
    got = (ph_new[sel[1]] - ph_new[sel[0]]) / (ph_old[sel[1]] - ph_old[sel[0]])
    assert abs(got / ratio - 1.0) < 1e-5


def test_remap_time_mutes_orders_above_nyquist(run):
    _, rpm, comp = run
    only2 = modify_orders(comp, solo=[2.0])
    y, info = remap_rpm(only2, rpm * 45.0, FS, seed=0)  # order 2: 2250 Hz ... 4500 Hz
    f = 2 * rpm * 45.0 / 60.0
    assert np.all(y[f >= FS / 2] == 0.0) and np.max(np.abs(y[f < 0.45 * FS])) > 0.5
    assert info["muted_fraction"][0] == pytest.approx(np.mean(f >= FS / 2))
    assert np.all(info["muted_fraction"][1:] == 1.0)  # orders 4 and 6 are above Nyquist throughout


def test_remap_time_carries_the_frequency_offset(run):
    sig, rpm, comp = run
    m = modify_orders(comp, freq_offset={6.0: 25.0}, solo=[6.0])
    y, _ = remap_rpm(m, rpm, FS, phase="table")
    assert np.max(np.abs(y - m.total)) < 1e-12  # offset kept in Hz at an unchanged speed
    y2, _ = remap_rpm(m, rpm * 1.2, FS, phase="table")
    f_inst = np.gradient(np.unwrap(np.angle(signal.hilbert(y2))), 1 / FS) / TWO_PI
    mid = slice(int(1 * FS), int(5 * FS))
    assert np.max(np.abs(f_inst[mid] - (6 * rpm[mid] * 1.2 / 60 + 25.0))) < 0.3  # not scaled


def test_remap_by_speed_makes_a_run_down_from_a_run_up():
    """mode="rpm": amplitude follows the measured law along another speed curve (0.3 dB)."""
    fs = 8000.0
    t = np.arange(round(10 * fs)) / fs
    rpm = rpm_profiles(t, "linear", start=1500, stop=3500)
    specs = [
        {"order": 2.0, "amplitude": 1.0, "phase": 0.3, "gain_rpm": [1000, 4000], "gain": [1.0, 2.0]},
        {"order": 4.0, "amplitude": 0.5, "phase": 1.0, "gain_rpm": [1000, 4000], "gain": [2.0, 0.5]},
    ]
    sig = synth_runup(fs, rpm, specs, noise_rms=0.02, seed=1)
    comp = tracked_harmonics(sig.x, fs, rpm, [2.0, 4.0])
    t2 = np.arange(round(8 * fs)) / fs
    rpm_down = rpm_profiles(t2, "linear", start=3400, stop=1600)
    y, info = remap_rpm(comp, rpm_down, fs, rpm_old=rpm, mode="rpm", seed=2)
    assert y.shape == rpm_down.shape and info["mode"] == "rpm" and info["outside_grid_fraction"] == 0.0
    assert info["rpm_grid"].size == 100 and info["seed"] == 2
    again = tracked_harmonics(y, fs, rpm_down, [2.0, 4.0])
    truth = np.stack(
        [np.interp(rpm_down, [1000, 4000], [1.0, 2.0]), 0.5 * np.interp(rpm_down, [1000, 4000], [2.0, 0.5])],
        axis=1,
    )
    inner = slice(int(0.5 * fs), -int(0.5 * fs))
    assert np.max(np.abs(_db(np.abs(again.envelopes[inner]), truth[inner]))) < 0.3
    # the orders sit at the frequencies of the new speed: demodulating with the new carrier
    # leaves the (constant) amplitude, so the output is not a stretched copy of the input
    assert abs(_rms(y) - _rms(sig.stems["orders"])) < 0.5 * _rms(sig.stems["orders"])


def test_remap_by_speed_holds_the_ends_and_reports_it():
    fs = 8000.0
    t = np.arange(round(6 * fs)) / fs
    rpm = rpm_profiles(t, "linear", start=1500, stop=3000)
    sig = synth_runup(
        fs, rpm, [{"order": 2.0, "amplitude": 1.0, "phase": 0.3, "gain_rpm": [1000, 4000], "gain": [1, 2]}],
        seed=1,
    )
    comp = tracked_harmonics(sig.x, fs, rpm, [2.0])
    new = np.full(round(2 * fs), 3300.0)  # above the measured range
    y, info = remap_rpm(comp, new, fs, rpm_old=rpm, mode="rpm", rpm_grid=np.arange(1500.0, 3001.0, 25.0))
    assert info["outside_grid_fraction"] == 1.0
    held = np.interp(3000.0, [1000, 4000], [1, 2])
    assert abs(_db(np.max(np.abs(y)), held)) < 0.3  # the amplitude at the last grid point is held


def test_remap_by_speed_identity_with_constant_amplitudes():
    fs = 8000.0
    n = round(4 * fs)
    rpm = rpm_profiles(np.arange(n) / fs, "linear", start=1500, stop=3000)
    sig = synth_runup(fs, rpm, [{"order": 2.0, "amplitude": 1.0, "phase": 0.3}], seed=1)
    comp = _truth_components(sig)
    y, _ = remap_rpm(comp, rpm, fs, rpm_old=rpm, mode="rpm", phase="table")
    assert np.max(np.abs(y - sig.stems["orders"])) < 1e-9  # the table phase is the constant start phase


def test_remap_two_shafts():
    fs = 8000.0
    n = round(4 * fs)
    r0 = rpm_profiles(np.arange(n) / fs, "linear", start=1500, stop=3000)
    rpm = np.column_stack([r0, 1.37 * r0])
    specs = [
        {"order": 2.0, "shaft": 0, "amplitude": 1.0, "phase": 0.3},
        {"order": 3.0, "shaft": 1, "amplitude": 0.5, "phase": 1.0},
    ]
    sig = synth_runup(fs, rpm, specs, seed=1)
    comp = _truth_components(sig, shafts=[0, 1])
    for mode in ("time", "rpm"):
        y, info = remap_rpm(comp, rpm, fs, rpm_old=rpm, mode=mode, phase="table")
        assert np.max(np.abs(y - sig.stems["orders"])) < 1e-9, mode
    with pytest.raises(ValueError, match="several shafts"):
        remap_rpm(comp, r0, fs, mode="time")
    with pytest.raises(ValueError, match="several shafts"):
        remap_rpm(comp, r0, fs, rpm_old=rpm, mode="rpm")
    # each shaft is carried by its own new speed
    y, _ = remap_rpm(comp, np.column_stack([1.1 * r0, 0.8 * r0]), fs, phase="table")
    ref = remap_rpm(modify_orders(comp, solo=[2.0]), np.column_stack([1.1 * r0, 0.8 * r0]), fs, phase="table")[0]
    ref2 = remap_rpm(modify_orders(comp, solo=[3.0]), np.column_stack([1.1 * r0, 0.8 * r0]), fs, phase="table")[0]
    assert np.allclose(y, ref + ref2, atol=1e-12)


def test_remap_errors(run):
    _, rpm, comp = run
    with pytest.raises(ValueError, match="mode"):
        remap_rpm(comp, rpm, FS, mode="speed")
    with pytest.raises(ValueError, match="same|samples"):
        remap_rpm(comp, rpm[:-10], FS)  # mode="time" needs the length of the components
    with pytest.raises(ValueError, match="rpm_old"):
        remap_rpm(comp, rpm, FS, mode="rpm")
    with pytest.raises(ValueError, match="fs"):
        remap_rpm(comp, rpm, 2 * FS)
    with pytest.raises(ValueError, match="non-finite"):
        remap_rpm(comp, np.where(np.arange(rpm.size) == 5, np.nan, rpm), FS)
    with pytest.raises(ValueError):
        remap_rpm(comp, rpm, -1.0)
    with pytest.raises(TypeError):
        remap_rpm(np.zeros(3), rpm, FS)
    with pytest.raises(ValueError, match="constant"):
        remap_rpm(comp, rpm, FS, rpm_old=np.full(rpm.size, 2000.0), mode="rpm")
    with pytest.raises(ValueError):
        remap_rpm(comp, rpm, FS, rpm_old=rpm[:-1], mode="rpm")


# --------------------------------------------------------------------------
# resynthesize
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parts():
    """A noisy run-up split into orders and residual by the extractor."""
    t = np.arange(round(4 * FS)) / FS
    rpm = rpm_profiles(t, "linear", start=1500, stop=3000)
    sig = synth_runup(FS, rpm, SPECS, noise_rms=0.1, seed=3)
    resid, comp = extract_residual(sig.x, FS, rpm, ORDERS, method="tracked-harmonics")
    return sig, comp, resid


def test_defaults_add_nothing(parts):
    sig, comp, resid = parts
    out = resynthesize(comp, resid)
    assert np.array_equal(out["orders"], comp.total) and np.array_equal(out["residual"], resid)
    assert np.array_equal(out["mix"], comp.total + resid) and np.allclose(out["mix"], sig.x, atol=1e-12)
    assert out["scale"] == 1.0 and out["gains_db"] == {"orders": 0.0, "residual": 0.0, "normalisation": 0.0}
    assert out["ratio_db"] == pytest.approx(10 * np.log10(np.sum(comp.total**2) / np.sum(resid**2)))
    assert set(out["rms"]) == {"orders", "residual", "mix"}


def test_mix_is_exactly_orders_plus_residual(parts):
    _, comp, resid = parts
    for kw in (
        {"order_gain_db": 4.0, "residual_gain_db": -7.5},
        {"ratio_db": 3.0, "residual_gain_db": 2.0},
        {"order_gain_db": -3.0, "reference_rms": 0.37},
        {"ratio_db": -20.0, "reference_rms": 1.0},
    ):
        out = resynthesize(comp, resid, **kw)
        assert np.array_equal(out["mix"], out["orders"] + out["residual"]), kw


def test_gains_are_applied_as_documented(parts):
    _, comp, resid = parts
    out = resynthesize(comp, resid, order_gain_db=6.0, residual_gain_db=-10.0)
    assert np.allclose(out["orders"], comp.total * 10 ** (6 / 20), rtol=1e-14, atol=0)
    assert np.allclose(out["residual"], resid * 10 ** (-10 / 20), rtol=1e-14, atol=0)
    assert out["gains_db"] == {"orders": 6.0, "residual": -10.0, "normalisation": 0.0}
    assert _rms(out["orders"]) == pytest.approx(out["rms"]["orders"])
    assert resynthesize(comp.total, resid, order_gain_db=6.0)["orders"] == pytest.approx(out["orders"])


@pytest.mark.parametrize("target", [-10.0, 0.0, 12.5, 30.0])
def test_ratio_db_is_achieved_exactly(parts, target):
    _, comp, resid = parts
    out = resynthesize(comp, resid, ratio_db=target, order_gain_db=99.0, residual_gain_db=-3.0)
    measured = 10 * np.log10(np.sum(out["orders"] ** 2) / np.sum(out["residual"] ** 2))
    assert out["ratio_db"] == pytest.approx(target, abs=1e-9)
    assert measured == pytest.approx(target, abs=1e-9)
    assert out["gains_db"]["residual"] == -3.0  # the residual gain still applies; order_gain_db is overridden
    assert np.allclose(out["residual"], resid * 10 ** (-3 / 20), rtol=1e-14, atol=0)
    assert out["gains_db"]["orders"] == pytest.approx(
        20 * np.log10(np.linalg.norm(out["orders"]) / np.linalg.norm(comp.total)), abs=1e-9
    )


def test_reference_rms_keeps_the_relative_gain(parts):
    _, comp, resid = parts
    ref = 0.2
    base = resynthesize(comp, resid, reference_rms=ref)
    assert _rms(base["mix"]) == pytest.approx(ref, rel=1e-12)
    assert base["scale"] == pytest.approx(ref / _rms(comp.total + resid), rel=1e-12)
    up = resynthesize(comp, resid, order_gain_db=6.0, reference_rms=ref)
    assert up["scale"] == base["scale"]  # computed from the gain-0 mix, so the gain does not change it
    assert _rms(up["orders"]) / _rms(base["orders"]) == pytest.approx(10 ** (6 / 20), rel=1e-12)
    assert _rms(up["residual"]) == pytest.approx(_rms(base["residual"]), rel=1e-12)
    assert up["gains_db"]["orders"] == 6.0
    assert up["gains_db"]["normalisation"] == pytest.approx(20 * np.log10(base["scale"]))
    assert _rms(up["mix"]) > ref  # the louder orders are not normalised away (ROADMAP 6, item 12)
    # normalising the gained mix by its own RMS would have cancelled the gain
    naive = (comp.total * 10 ** (6 / 20) + resid) * (ref / _rms(comp.total * 10 ** (6 / 20) + resid))
    assert _rms(naive) == pytest.approx(ref)
    # the residual gain behaves the same way; the scale still comes from the gain-0 mix
    down = resynthesize(comp, resid, residual_gain_db=-6.0, reference_rms=ref)
    assert down["scale"] == base["scale"]
    assert _rms(down["residual"]) / _rms(base["residual"]) == pytest.approx(10 ** (-6 / 20), rel=1e-12)


def test_multichannel_keeps_the_channel_relations(parts):
    _, comp, resid = parts
    o2 = np.column_stack([comp.total, 0.25 * comp.total])
    r2 = np.column_stack([resid, -0.5 * resid])
    out = resynthesize(o2, r2, ratio_db=5.0, reference_rms=0.1)
    assert out["orders"].shape == o2.shape
    assert np.allclose(out["orders"][:, 1], 0.25 * out["orders"][:, 0], atol=1e-14)
    assert np.allclose(out["residual"][:, 1], -0.5 * out["residual"][:, 0], atol=1e-14)
    measured = 10 * np.log10(np.sum(out["orders"] ** 2) / np.sum(out["residual"] ** 2))
    assert measured == pytest.approx(5.0, abs=1e-9)


def test_recovering_the_original_ratio_needs_no_gain(parts):
    _, comp, resid = parts
    original = 10 * np.log10(np.sum(comp.total**2) / np.sum(resid**2))
    out = resynthesize(comp, resid, ratio_db=original)
    assert out["gains_db"]["orders"] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(out["mix"], comp.total + resid, atol=1e-12)


def test_resynthesize_errors(parts):
    _, comp, resid = parts
    with pytest.raises(ValueError, match="shape"):
        resynthesize(comp, resid[:-1])
    with pytest.raises(ValueError, match="shape"):
        resynthesize(comp.total, resid[:, None])
    bad = resid.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        resynthesize(comp, bad)
    with pytest.raises(ValueError, match="energy"):
        resynthesize(comp, np.zeros_like(resid), ratio_db=3.0)
    with pytest.raises(ValueError, match="energy"):
        resynthesize(np.zeros_like(resid), resid, ratio_db=3.0)
    with pytest.raises(ValueError, match="ratio_db"):
        resynthesize(comp, resid, ratio_db=np.nan)
    with pytest.raises(ValueError, match="ratio_db"):
        resynthesize(comp, resid, ratio_db=np.inf)
    with pytest.raises(ValueError, match="reference_rms"):
        resynthesize(comp, resid, reference_rms=0.0)
    with pytest.raises(ValueError, match="reference_rms"):
        resynthesize(comp, resid, reference_rms=-1.0)
    with pytest.raises(ValueError, match="gain"):
        resynthesize(comp, resid, order_gain_db=np.nan)
    with pytest.raises(ValueError, match="gain"):
        resynthesize(comp, resid, residual_gain_db=np.inf)
    with pytest.raises(ValueError, match="gain-0 mix"):
        resynthesize(np.zeros(10), np.zeros(10), reference_rms=1.0)
    out = resynthesize(comp, resid, order_gain_db=-np.inf)  # muting the orders is allowed
    assert np.all(out["orders"] == 0.0) and out["ratio_db"] == -np.inf
