"""Tests of rota.synth.synth_scenario: complete drive-train recordings with exact truth.

The scenarios chain the building blocks of rota._sources and rota._channel (each
tested on its own). What is checked here is the assembly:

* the stems add up to the recording (bit for bit with an ideal recorder), the
  stems after the path are what the path does to the source stems, and the truth
  fields are consistent with each other (rev against rpm, the speed ripple, the
  order truth against the rendered orders, shift drops, tachometer pulses at the
  true shaft angles plus the stated delay);
* the documented default parameters: relative order levels and speed slopes, the
  overall level and the order-to-residual ratio, the spectral shapes of the
  broadband noise, the diffuse-field coherence between two ears, the EV tonal-to-
  broadband ratio, the dithered inverter carrier, the steady cruise;
* determinism for a seed, JSON configuration, multichannel shape and the
  inter-channel delay of the cabin impulse response, the measurement chain
  (clipping flag etc.) and the error paths;
* every kind runs at its default settings in reasonable time.
"""

import json
import time

import numpy as np
import pytest
from scipy import signal

from rota import synth
from rota.speed import rpm_to_phase, speed_to_rpm, tacho_to_rpm
from rota.synth import (
    SCENARIO_KINDS,
    ScenarioSignal,
    scenario_defaults,
    synth_scenario,
)

TWO_PI = 2.0 * np.pi
P_REF = 20e-6

# Short, fast variants of the kinds (low sampling rates, brisk acceleration and early
# shifts so that three upshifts fit into ten seconds; tachometers sized for the rate).
ICE_KW = dict(tacho={"ppr": 20}, speed={"accel_ms2": 3.5}, drive={"shift_rpm": 4500.0})
EV_KW = dict(
    tacho={"ppr": 24},
    inverter={"carrier_hz": 7000.0, "dither_hz": 600.0},
    tones=[
        {"freq": 5300.0, "tnr_db": 10.0, "bandwidth_hz": 20.0},
        {"freq": 9500.0, "tnr_db": 6.0, "bandwidth_hz": 20.0},
    ],
)
CRUISE_KW = dict(tacho={"ppr": 20})
ONE_MODE = {"modes": [{"freq": 2200.0, "zeta": 0.4, "gain": 1.0}]}  # a path that fits low rates


def level_db(x):
    """RMS level in dB re 20 uPa."""
    return 20.0 * np.log10(np.sqrt(np.mean(np.asarray(x) ** 2)) / P_REF)


def a_weighted_level_db(x, fs):
    """A-weighted RMS level in dB re 20 uPa (IEC 61672-1 analogue poles, bilinear transform,
    normalised to 0 dB at 1 kHz; applied once)."""
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    poles = -TWO_PI * np.array([f1, f1, f4, f4, f2, f3])
    zd, pd, kd = signal.bilinear_zpk([0, 0, 0, 0], poles, (TWO_PI * f4) ** 2, fs)
    sos = signal.zpk2sos(zd, pd, kd)
    gain = 1.0 / abs(signal.sosfreqz(sos, worN=[1000.0], fs=fs)[1][0])
    return level_db(gain * signal.sosfilt(sos, x))


@pytest.fixture(scope="module")
def ice():
    return synth_scenario("ice_runup", 16000.0, 11.0, seed=11, **ICE_KW)


@pytest.fixture(scope="module")
def ice_mid():
    """Default full-load run-up through the mid-speed condition (3600 rpm in second gear)."""
    return synth_scenario("ice_runup", 16000.0, 6.0, seed=7, tacho={"ppr": 20})


@pytest.fixture(scope="module")
def ev():
    return synth_scenario("ev_runup", 24000.0, 8.0, seed=12, **EV_KW)


@pytest.fixture(scope="module")
def cruise():
    return synth_scenario("steady_cruise", 16000.0, 6.0, seed=13, **CRUISE_KW)


@pytest.fixture(scope="module")
def cruise_ev():
    return synth_scenario("steady_cruise", 24000.0, 6.0, seed=14, powertrain="ev", **EV_KW)


# --------------------------------------------------------------------------
# API surface
# --------------------------------------------------------------------------


def test_building_blocks_are_reexported():
    from rota import _channel, _sources

    names = [
        "drive_rpm", "angular_speed_ripple", "angle_impulses", "sideband_orders",
        "carrier_components", "modal_path", "modal_response", "room_impulse_response",
        "apply_ir", "speed_noise", "transient_events", "measurement_chain", "delay_signal",
    ]
    for name in names:
        assert name in synth.__all__, name
        owner = _sources if name in _sources.__all__ else _channel
        assert getattr(synth, name) is getattr(owner, name), name
    for name in ("synth_scenario", "ScenarioSignal", "scenario_defaults", "SCENARIO_KINDS"):
        assert name in synth.__all__
    for name in ("rpm_profiles", "synth_runup", "SynthSignal", "synth_tone_noise", "ToneNoiseSignal"):
        assert name in synth.__all__  # the earlier public names are still there


def test_public_scenario_api_is_documented():
    import inspect

    for obj in (synth_scenario, scenario_defaults, ScenarioSignal.through_path,
                ScenarioSignal.path_response, ScenarioSignal.quasi_static_amplitude,
                ScenarioSignal.stem_sum, ScenarioSignal.save, ScenarioSignal.at_recording):
        doc = inspect.getdoc(obj)
        assert doc and "References" in doc and "Returns" in doc, obj
    doc = " ".join(inspect.getdoc(synth_scenario).split())
    assert "peak" in doc and "Parameters" in doc and "Notes" in doc
    assert "illustrative values typical of passenger cars, rounded" in doc
    assert "measurements of any specific" in doc


def test_defaults_are_json_fresh_copies():
    assert SCENARIO_KINDS == ("ice_runup", "ev_runup", "steady_cruise")
    cases = [("ice_runup", "ice"), ("ev_runup", "ice"), ("steady_cruise", "ice"),
             ("steady_cruise", "ev")]
    for kind, pt in cases:
        d = scenario_defaults(kind, pt)
        assert json.loads(json.dumps(d, allow_nan=False)) == d
        d["tacho"]["ppr"] = 1
        assert scenario_defaults(kind, pt)["tacho"]["ppr"] != 1  # a fresh dict every call
    assert scenario_defaults("steady_cruise")["powertrain"] == "ice"
    assert scenario_defaults("steady_cruise", "ev")["powertrain"] == "ev"
    with pytest.raises(ValueError, match="kind"):
        scenario_defaults("diesel")
    with pytest.raises(ValueError, match="powertrain"):
        scenario_defaults("steady_cruise", "diesel")


# --------------------------------------------------------------------------
# stems and the recording
# --------------------------------------------------------------------------


def test_stem_names_per_kind(ice, ev, cruise, cruise_ev):
    assert list(ice.stems) == [
        "orders", "combustion", "transients", "engine_broadband", "road", "rumble"
    ]
    assert list(cruise.stems) == ["orders", "combustion", "engine_broadband", "road", "rumble"]
    assert list(ev.stems) == ["motor", "gear", "inverter", "tones", "transients", "road", "rumble"]
    assert list(cruise_ev.stems) == ["motor", "gear", "inverter", "tones", "road", "rumble"]
    receiver = ("engine_broadband", "road", "rumble")
    for sc in (ice, ev, cruise, cruise_ev):
        assert list(sc.source_stems) == [k for k in sc.stems if k not in receiver]
        assert sc.config["resolved"]["stems"] == list(sc.stems)
    assert ice.config["resolved"]["powertrain"] == "ice"
    assert cruise_ev.config["resolved"]["powertrain"] == "ev"


def test_stems_sum_exactly_to_x_for_an_ideal_recorder(ice, ev, cruise, cruise_ev):
    for sc in (ice, ev, cruise, cruise_ev):
        assert isinstance(sc, ScenarioSignal)
        acc = None
        for s in sc.stems.values():
            assert s.shape == sc.x.shape
            acc = s.copy() if acc is None else acc + s
        assert np.array_equal(sc.x, acc)  # bit for bit, accumulated in dict order
        assert np.array_equal(sc.x_clean, sc.x)
        assert np.array_equal(sc.stem_sum(), sc.x)
        assert not sc.clipped.any()
        assert np.array_equal(sc.t_true, sc.t)
        assert sc.config["measurement"] is None
        assert np.allclose(sc.x, np.sum(list(sc.stems.values()), axis=0), rtol=0, atol=1e-12)
    part = ice.stem_sum(["road", "rumble"])
    assert np.array_equal(part, ice.stems["road"] + ice.stems["rumble"])
    with pytest.raises(ValueError, match="unknown stem"):
        ice.stem_sum(["nonexistent"])


def test_stems_are_the_source_stems_through_the_path(ice, ev):
    for sc in (ice, ev):
        for name, src in sc.source_stems.items():
            assert src.shape == (sc.n,)
            assert np.array_equal(sc.through_path(src), sc.stems[name]), name
    # Linearity: the stem of all orders is the sum of the orders each through the path.
    parts = sum(ice.through_path(ice.order_stems[:, k]) for k in range(ice.orders.size))
    ref = ice.stems["orders"]
    assert np.sqrt(np.mean((parts - ref) ** 2)) < 1e-9 * np.sqrt(np.mean(ref**2)) + 1e-15
    assert np.allclose(ice.source_stems["orders"], ice.order_stems.sum(axis=1), atol=1e-12)
    # The EV's motor and gear stems are the sums of their orders at the source.
    names = ev.order_names
    motor = [i for i, n in enumerate(names) if n.startswith("motor_")]
    gear = [i for i, n in enumerate(names) if n.startswith("gear_mesh_")]
    assert np.allclose(ev.source_stems["motor"], ev.order_stems[:, motor].sum(axis=1), atol=1e-12)
    assert np.allclose(ev.source_stems["gear"], ev.order_stems[:, gear].sum(axis=1), atol=1e-12)
    for k in ("engine_broadband", "road", "rumble"):  # generated at the receiver
        assert k not in ice.source_stems


def test_off_components_are_absent_and_the_sum_still_holds():
    sc = synth_scenario(
        "ice_runup", 8000.0, 4.0, seed=1, tacho=None, combustion=None, engine_broadband=None,
        road=None, rumble=None, transients=[], path=ONE_MODE,
    )
    assert list(sc.stems) == ["orders"]
    assert np.array_equal(sc.x, sc.stems["orders"])
    assert sc.tacho is None and sc.pulse_times.size == 0 and sc.pulse_revs.size == 0
    sc2 = synth_scenario(
        "ev_runup", 24000.0, 3.0, seed=1, tacho=None, gear_mesh=None, inverter=None, tones=[],
        road=None, rumble=None, transients=[],
    )
    assert list(sc2.stems) == ["motor"]
    assert sc2.carrier is None
    assert np.array_equal(sc2.x, sc2.stems["motor"])
    with pytest.raises(ValueError, match="keep road"):
        synth_scenario("ev_runup", 24000.0, 2.0, road=None, **EV_KW)


# --------------------------------------------------------------------------
# truth: speed, angle, ripple, shifts
# --------------------------------------------------------------------------


def test_rev_is_consistent_with_rpm(ice, ev, cruise):
    for sc in (ice, ev, cruise):
        assert np.all(np.diff(sc.rev) > 0)
        dev = np.max(np.abs(rpm_to_phase(sc.rpm, sc.fs) - sc.rev))
        assert dev < 1e-5  # revolutions: the angle the orders are rendered with
        assert sc.config["resolved"]["rev_deviation_max"] == pytest.approx(dev, abs=1e-12)
        d = 60.0 * sc.fs * (sc.rev[2:] - sc.rev[:-2]) / 2.0  # central differences
        assert np.max(np.abs(d - sc.rpm[1:-1]) / sc.rpm[1:-1]) < 2e-4
        for a in (sc.rpm, sc.rpm_mean, sc.rev, sc.speed_kmh, sc.gear, sc.t):
            assert a.shape == (sc.n,)
        assert np.all(sc.rpm > 0)


def test_speed_ripple_is_the_stated_angle_periodic_fluctuation(ice, ev, cruise):
    """rpm = rpm_mean * g(rev) with the config's ripple terms, recovered exactly by least squares."""
    for sc in (ice, ev, cruise):
        terms = np.array(sc.config["resolved"]["ripple"])
        g = sc.rpm / sc.rpm_mean - 1.0
        cols = []
        for order, _, _ in terms:
            cols += [np.cos(TWO_PI * order * sc.rev), np.sin(TWO_PI * order * sc.rev)]
        coef, *_ = np.linalg.lstsq(np.column_stack(cols), g, rcond=None)
        for j, (order, amp, phase) in enumerate(terms):
            a, b = coef[2 * j], coef[2 * j + 1]
            assert np.hypot(a, b) == pytest.approx(amp, rel=1e-6)
            assert np.arctan2(-b, a) == pytest.approx(np.angle(np.exp(1j * phase)), abs=1e-6)
    amps = {o: a for o, a, _ in np.array(ice.config["resolved"]["ripple"])}
    assert 0.001 <= amps[2.0] <= 0.005  # a few 0.1 % at the firing order
    assert np.std(ice.rpm / ice.rpm_mean - 1.0) > 0.001


def test_ice_runup_follows_500_rpm_per_second_and_shifts_drop_about_30_percent():
    """Default run-up: second gear from 1200 rpm at about 500 rpm/s to the 6000 rpm shift point."""
    sc = synth_scenario("ice_runup", 8000.0, 16.0, seed=2, tacho=None, path=ONE_MODE)
    assert sc.rpm_mean[0] == pytest.approx(1200.0, abs=1.0) and sc.gear[0] == 2
    first = sc.events["shift_times"][0]
    sel = sc.t < first
    slope = np.polyfit(sc.t[sel], sc.rpm_mean[sel], 1)[0]
    assert 430.0 < slope < 570.0  # rpm/s
    assert sc.rpm_mean[np.searchsorted(sc.t, first) - 1] == pytest.approx(6000.0, abs=5.0)
    assert sc.events["shift_times"].size == 2
    assert np.array_equal(sc.events["shift_from"], [2, 3])
    assert np.array_equal(sc.events["shift_to"], [3, 4])
    shift_time = sc.config["parameters"]["drive"]["shift_time"]
    assert 0.3 <= shift_time <= 0.5
    for ts in sc.events["shift_times"]:
        i0 = np.searchsorted(sc.t, ts) - 1
        j1 = np.searchsorted(sc.t, ts + shift_time)
        drop = 1.0 - sc.rpm_mean[j1] / sc.rpm_mean[i0]  # within the transient
        assert 0.25 < drop < 0.36
    assert sc.rpm_mean.max() <= 6500.0
    assert sc.events["launch_times"].shape == (0, 2) and not sc.events["slip"].any()


def test_gear_shifts_drop_the_engine_speed(ice):
    ev_ = ice.events
    assert ev_["shift_times"].size == 3
    assert np.array_equal(ev_["shift_from"], [2, 3, 4])
    assert np.array_equal(ev_["shift_to"], [3, 4, 5])
    drive = ice.config["parameters"]["drive"]
    chain = dict(
        wheel_circumference=drive["wheel_circumference"], gear_ratios=drive["gear_ratios"],
        final_drive=drive["final_drive"],
    )
    shift_time = drive["shift_time"]
    for ts, a, b in zip(ev_["shift_times"], ev_["shift_from"], ev_["shift_to"]):
        i0 = np.searchsorted(ice.t, ts) - 1
        i1 = np.searchsorted(ice.t, ts + shift_time + 0.02)
        before, after = ice.rpm_mean[i0], ice.rpm_mean[i1]
        assert after < 0.85 * before  # an upshift drops the speed by a gear-ratio step
        assert ice.gear[i0] == a and ice.gear[i1] == b
        static = float(speed_to_rpm(ice.speed_kmh[i1], gear=int(b), **chain))
        assert after == pytest.approx(static, rel=1e-9)  # the static relation holds afterwards
        j1 = np.searchsorted(ice.t, ts + shift_time) - 1
        assert np.all(np.diff(ice.rpm_mean[i0 + 1 : j1]) < 0)  # the transient falls monotonically
        assert ice.events["in_shift"][i0 + 2 : j1].all() and not ice.events["in_shift"][i1]
    assert ice.rpm_mean.max() <= drive["redline"]


def test_launch_variant_has_idle_hold_and_clutch_slip():
    sc = synth_scenario(
        "ice_runup", 8000.0, 8.0, seed=3, tacho=None, path=ONE_MODE,
        speed={"hold": 1.0, "start_kmh": 0.0, "accel_ms2": 3.0, "jerk_time": 0.6},
        drive={"start_gear": 1, "shift_rpm": 4500.0},
    )
    idle = sc.config["parameters"]["drive"]["idle"]
    assert np.all(sc.rpm_mean[sc.t < 0.9] == idle) and np.all(sc.speed_kmh[sc.t < 0.9] == 0.0)
    lt = sc.events["launch_times"]
    assert lt.shape == (1, 2) and lt[0, 0] >= 1.0 and lt[0, 1] > lt[0, 0]
    assert sc.events["slip"].any() and sc.rpm_mean[np.searchsorted(sc.t, lt[0, 1]) + 5] > idle
    assert sc.gear[0] == 1 and sc.events["shift_times"].size >= 1


def test_ev_speed_law_and_the_coast_variant(ev, cruise, cruise_ev):
    assert ev.events["shift_times"].size == 0 and np.all(ev.gear == 1)
    assert ev.speed_kmh[0] == pytest.approx(2.0, abs=1e-9) and np.all(np.diff(ev.speed_kmh) > 0)
    assert np.allclose(ev.rpm_mean, 100.0 * ev.speed_kmh, rtol=1e-12)  # shaft rpm = 100 v
    # full-load 0 to 130 km/h in about 10 s: the default speed law, evaluated directly
    d = scenario_defaults("ev_runup")
    v = synth._speed_profile(np.array([0.0, 5.0, 10.0]), d["speed"])
    assert v[0] == pytest.approx(2.0) and 120.0 < v[2] < 135.0 and 50.0 < v[1] < 80.0
    # coast-down variant 120 to 10 km/h in 20 s: a linear ramp
    sc = synth_scenario(
        "ev_runup", 24000.0, 5.0, seed=4, tacho=None, inverter=None, tones=[],
        speed={"mode": "ramp", "start_kmh": 120.0, "end_kmh": 10.0, "ramp_time": 20.0},
    )
    assert sc.speed_kmh[0] == 120.0
    assert sc.speed_kmh[-1] == pytest.approx(120.0 - 110.0 * 5.0 / 20.0, abs=0.01)
    assert np.all(np.diff(sc.speed_kmh) < 0) and np.allclose(sc.rpm_mean, 100.0 * sc.speed_kmh)
    # steady cruise: constant speed, fixed gear, the ripple kept
    assert np.all(cruise.speed_kmh == 90.0) and np.all(cruise.gear == 5)
    assert np.allclose(cruise.rpm_mean, cruise.rpm_mean[0], rtol=0, atol=1e-9)
    assert cruise.rpm_mean[0] == pytest.approx(90.0 * 0.84 * 3.7 * 60 / 3.6 / 1.95, rel=1e-9)
    assert np.std(cruise.rpm / cruise.rpm_mean) > 1e-3
    assert cruise.rev[-1] != pytest.approx(cruise.rpm_mean[0] / 60 * cruise.t[-1], abs=1e-9)
    assert np.all(cruise_ev.speed_kmh == 90.0) and np.allclose(cruise_ev.rpm_mean, 9000.0)


# --------------------------------------------------------------------------
# source orders: truth against what was rendered, and the documented levels
# --------------------------------------------------------------------------


def test_order_truth_matches_rendered_orders(ice, ev):
    for sc in (ice, ev):
        K = sc.orders.size
        assert len(sc.order_names) == K and sc.order_amplitude.shape == (sc.n, K)
        assert np.allclose(sc.order_stems, sc.order_amplitude * np.cos(sc.order_phase), atol=1e-14)
        assert np.allclose(sc.order_frequency, np.outer(sc.rpm / 60.0, sc.orders), rtol=1e-12)
        rev = rpm_to_phase(sc.rpm, sc.fs)
        resid = sc.order_phase - TWO_PI * np.outer(rev, sc.orders)
        assert np.max(np.ptp(resid, axis=0)) < 1e-9  # a constant phase offset per order
    assert dict(zip(ice.order_names, ice.orders)) == {
        "firing": 2.0, "order_4": 4.0, "order_6": 6.0, "order_8": 8.0, "order_3": 3.0,
        "order_5": 5.0, "order_7": 7.0, "order_1": 1.0, "order_0.5": 0.5, "order_1.5": 1.5,
        "order_2.5": 2.5,
    }
    assert dict(zip(ev.order_names, ev.orders)) == {
        "motor_2": 2.0, "motor_4": 4.0, "motor_8": 8.0, "motor_10": 10.0, "motor_12": 12.0,
        "motor_14": 14.0, "gear_mesh_40": 40.0, "gear_mesh_6.7": 6.7,
    }


def test_ice_orders_have_the_documented_levels_and_speed_slopes(ice_mid):
    """Levels re order 2 at 3600 rpm, and +3 dB per 1000 rpm for order 2, +1 dB for the rest."""
    sc = ice_mid
    amp = dict(zip(sc.order_names, sc.order_amplitude.T))
    i = int(np.argmin(np.abs(sc.rpm_mean - 3600.0)))
    expected = {
        "order_4": -15.0, "order_6": -24.0, "order_8": -30.0, "order_3": -25.0, "order_5": -28.0,
        "order_7": -32.0, "order_1": -32.0, "order_0.5": -38.0, "order_1.5": -34.0,
        "order_2.5": -36.0,
    }
    for name, lev in expected.items():
        assert 20 * np.log10(amp[name][i] / amp["firing"][i]) == pytest.approx(lev, abs=0.4), name
    weak = ("order_1", "order_0.5", "order_1.5", "order_2.5")
    assert max(expected[k] for k in weak) <= -30.0  # order 1 and the half orders are weak
    i1 = int(np.argmin(np.abs(sc.rpm_mean - 1500.0)))
    i2 = int(np.argmin(np.abs(sc.rpm_mean - 3900.0)))
    for name, slope in [("firing", 3.0), ("order_4", 1.0), ("order_3", 1.0), ("order_8", 1.0)]:
        gain_db = 20 * np.log10(amp[name][i2] / amp[name][i1])
        assert gain_db == pytest.approx(slope * 2.4, abs=0.4), name
    ref = scenario_defaults("ice_runup")["orders"]["ref_amplitude"]
    assert amp["firing"][i] == pytest.approx(ref, rel=0.03)  # the calibration, before the path


def test_ev_orders_have_the_documented_levels_and_speed_slopes(ev):
    """Levels re the strongest order, dB per doubling of speed (the law of each entry)."""
    amp = dict(zip(ev.order_names, ev.order_amplitude.T))
    spec = {
        "motor_2": (0.0, 5.0), "motor_4": (-6.0, 7.0), "motor_8": (-9.0, 6.0),
        "motor_10": (-12.0, 6.0), "motor_12": (-10.0, 2.0), "motor_14": (-15.0, 5.0),
        "gear_mesh_6.7": (-8.0, 5.0),
    }
    ref_rpm = scenario_defaults("ev_runup")["motor"]["ref_rpm"]
    a_ref = amp["motor_2"]
    for rpm in (3000.0, 6000.0):  # inside the record, away from the reference speed
        i = int(np.argmin(np.abs(ev.rpm_mean - rpm)))
        for name, (lev, slope) in spec.items():
            want = lev + (slope - spec["motor_2"][1]) * np.log2(ev.rpm_mean[i] / ref_rpm)
            got = 20 * np.log10(amp[name][i] / a_ref[i])
            assert got == pytest.approx(want, abs=0.4), (name, rpm)
    # the whine: -14 dB, +3 dB per doubling, and a +-6 dB (depth 0.6) modulation at 0.5 Hz
    gm = ev.config["parameters"]["gear_mesh"]
    assert gm["teeth"] == 40 and gm["am_depth"] == 0.6 and gm["am_freq"] == 0.5
    assert gm["sidebands"] is None  # off by default
    w = amp["gear_mesh_40"] / (1.0 + 0.6 * np.sin(TWO_PI * 0.5 * ev.t))
    i1 = int(np.argmin(np.abs(ev.rpm_mean - 3000.0)))
    i2 = int(np.argmin(np.abs(ev.rpm_mean - 6000.0)))
    assert 20 * np.log10(w[i2] / w[i1]) == pytest.approx(3.0, abs=0.3)
    want = -14.0 + (3.0 - 5.0) * np.log2(ev.rpm_mean[i1] / ref_rpm)
    assert 20 * np.log10(w[i1] / a_ref[i1]) == pytest.approx(want, abs=0.4)
    mod = amp["gear_mesh_40"] / w
    assert 20 * np.log10(mod.max() / mod.min()) == pytest.approx(12.0, abs=0.7)  # +-6 dB
    assert ev.config["parameters"]["pole_pairs"] == 4  # electrical frequency at order 4


def test_whine_and_firing_order_sidebands_can_be_switched_on():
    sc = synth_scenario(
        "ev_runup", 24000.0, 4.0, seed=5, gear_mesh={"sidebands": [-10.0, -20.0]}, **EV_KW
    )
    amp = dict(zip(sc.order_names, sc.order_amplitude.T))
    assert {38.0, 39.0, 41.0, 42.0} <= set(sc.orders.tolist())
    i = 100
    ref = amp["gear_mesh_40"][i]
    assert amp["gear_mesh_40_sb_39"][i] / ref == pytest.approx(10 ** (-10 / 20), rel=1e-6)
    assert amp["gear_mesh_40_sb_38"][i] / ref == pytest.approx(10 ** (-20 / 20), rel=1e-6)
    ice_sb = synth_scenario(
        "ice_runup", 8000.0, 3.0, seed=5, tacho=None, path=ONE_MODE,
        orders={"sidebands": {"carrier": "firing", "mod_order": 1.0, "levels_db": [-12.0],
                              "modulation": "am"}},
    )
    assert {1.0, 3.0} <= set(ice_sb.orders.tolist())
    assert "firing_sb_3" in ice_sb.order_names


def test_pole_pairs_set_the_inverter_sidebands(ev):
    p = ev.config["parameters"]["pole_pairs"]
    assert sorted(ev.carrier["q"]) == sorted([0.0, -2.0 * p, 2.0 * p, -4.0 * p, 4.0 * p])
    other = synth_scenario("ev_runup", 24000.0, 3.0, seed=1, pole_pairs=3, **EV_KW)
    assert sorted(other.carrier["q"]) == [-12.0, -6.0, 0.0, 6.0, 12.0]


def test_inverter_carrier_hops_and_the_truth_is_exact(ev):
    c = ev.carrier
    track = c["carrier_hz"]
    fc, dith = 7000.0, 600.0
    assert track.shape == (ev.n,) and np.all(np.abs(track - fc) <= dith + 1e-9)
    hop_t, hop_f = ev.events["carrier_hop_times"], ev.events["carrier_hop_hz"]
    assert hop_t[0] == 0.0 and hop_t.size >= 8
    dt = np.diff(hop_t)  # hops every 0.2 to 0.5 s onto a 50 Hz grid
    assert dt.min() >= 0.2 - 1.0 / ev.fs and dt.max() <= 0.5 + 1.0 / ev.fs
    assert np.allclose((hop_f - fc) / 50.0, np.round((hop_f - fc) / 50.0))
    idx = np.round(hop_t * ev.fs).astype(int)
    assert np.array_equal(track[idx], hop_f)  # the track holds each hop frequency
    segments = np.split(track, idx[1:])
    assert all(np.all(s == s[0]) for s in segments)  # piecewise constant
    # frequency truth: carrier track plus q * instantaneous speed
    assert np.allclose(c["frequency"], track[:, None] + np.outer(ev.rpm / 60.0, c["q"]), rtol=1e-12)
    # phase truth: the carrier column advances by 2 pi fc(t) / fs per sample, continuously
    j0 = int(np.flatnonzero(c["q"] == 0.0)[0])
    dphi = np.diff(c["phase"][:, j0]) * ev.fs / TWO_PI
    assert np.allclose(dphi, track[:-1], rtol=1e-9)
    src = (c["amplitude"] * np.cos(c["phase"])).sum(axis=1)  # the source stem is their sum
    assert np.allclose(ev.source_stems["inverter"], src, atol=1e-12)
    assert np.all(c["amplitude"][c["frequency"] >= ev.fs / 2] == 0.0)  # muted, never clipped
    assert np.all(c["frequency"][c["amplitude"] > 0] < ev.fs / 2)
    with pytest.raises(ValueError, match="carrier"):
        synth_scenario("ev_runup", 24000.0, 2.0, **{**EV_KW, "inverter": {"carrier_hz": 13000.0}})


def test_ev_tones_are_fixed_and_stand_above_the_local_floor(ev):
    """Fixed tones at tnr_db above the broadband PSD at their frequency (20 Hz band, 90 km/h)."""
    src = ev.source_stems["tones"]
    modes = ev.config["parameters"]["path"]["modes"]
    m = (ev.speed_kmh > 85) & (ev.speed_kmh < 95)
    f, pxx = signal.welch(ev.stems["road"][m], ev.fs, nperseg=4096)
    tt = ev.t
    for freq, tnr in ((5300.0, 10.0), (9500.0, 6.0)):
        a_src = 2.0 * abs(np.mean(src * np.exp(-2j * np.pi * freq * tt)))  # fixed frequency
        assert a_src > 0
        # no other component at this frequency: the projection leaves a pure tone
        a_out = a_src * abs(synth.modal_response(freq, modes, ev.fs))  # after the modal path
        floor = np.mean(pxx[(f > freq - 300) & (f < freq + 300)]) * 20.0  # power in 20 Hz
        got = 10.0 * np.log10((a_out**2 / 2.0) / floor)
        assert got == pytest.approx(tnr, abs=1.5), freq
    assert ev.stems["tones"].shape == ev.x.shape


# --------------------------------------------------------------------------
# documented overall levels, spectral shapes and the two ears
# --------------------------------------------------------------------------


def _ir_delays(ir, fs):
    """Delay of each IR channel against channel 0 from the phase slope of H_c / H_0 (30-120 Hz)."""
    h = np.fft.rfft(ir, 1 << 16, axis=0)
    f = np.fft.rfftfreq(1 << 16, 1.0 / fs)
    sel = (f >= 30) & (f <= 120)
    out = [0.0]
    for c in range(1, ir.shape[1]):
        phase = np.unwrap(np.angle(h[sel, c] / h[sel, 0]))
        out.append(-np.polyfit(TWO_PI * f[sel], phase, 1)[0])
    return np.array(out)


def _window(sc, lo_rpm, hi_rpm):
    return (sc.rpm_mean > lo_rpm) & (sc.rpm_mean < hi_rpm) & (sc.t > 1.0) & ~sc.events["in_shift"]


def test_ice_overall_level_and_order_to_residual_ratio(ice_mid):
    """About 88 dB re 20 uPa unweighted at full load, mid speed; orders 5 dB above the rest."""
    sc = ice_mid
    m = _window(sc, 3300.0, 3900.0)
    assert m.sum() > 8000
    x = sc.x[m]
    orders = sc.stems["orders"][m]
    resid = x - orders
    assert level_db(x) == pytest.approx(88.0, abs=1.5)
    assert level_db(orders) - level_db(resid) == pytest.approx(5.0, abs=1.5)
    rest = sc.stem_sum([k for k in sc.stems if k != "orders"])[m]
    assert np.allclose(rest, resid, atol=1e-12)  # the residual: all other stems


def test_broadband_shapes_levels_and_speed_laws(ice, ice_mid):
    fs = ice_mid.fs
    m = _window(ice_mid, 3300.0, 3900.0)
    inside = lambda f, lo, hi: (f >= lo) & (f <= hi)
    for name in ("road", "engine_broadband"):
        f, p = signal.welch(ice_mid.stems[name][m], fs, nperseg=1024)
        k = inside(f, 400.0, 4000.0)
        slope = np.polyfit(np.log2(f[k]), 10 * np.log10(p[k]), 1)[0]
        assert slope == pytest.approx(-8.5, abs=0.8), name  # PSD tilt in dB per octave
    f, p = signal.welch(ice_mid.stems["rumble"][m], fs, nperseg=4096)
    k = f < 300.0
    assert 20.0 < f[k][np.argmax(p[k])] < 60.0  # the rumble hump
    assert np.sum(p[inside(f, 15.0, 80.0)]) > 0.8 * np.sum(p)
    par = ice.config["parameters"]
    # reference levels (RMS at the reference vehicle speed), in the first gear
    for name, tol in (("road", 1.5), ("rumble", 2.5)):
        ref = par[name]["ref"]
        w = (np.abs(ice.speed_kmh - ref) < 5.0) & (ice.t < ice.events["shift_times"][0] - 0.3)
        assert w.sum() > 3000
        assert level_db(ice.stems[name][w]) == pytest.approx(par[name]["level_db_spl"], abs=tol), name

    def law(name, control, v1, v2, rel=0.12):
        x = ice.speed_kmh if control == "speed" else ice.rpm_mean
        first = ice.t < ice.events["shift_times"][0] - 0.3
        w1 = (np.abs(x - v1) < rel * v1) & first
        w2 = (np.abs(x - v2) < rel * v2) & first
        assert w1.sum() > 1500 and w2.sum() > 1500, (name, w1.sum(), w2.sum())
        return level_db(ice.stems[name][w2]) - level_db(ice.stems[name][w1])

    # +4.5 dB per doubling of vehicle speed (road), +3 dB per doubling of rpm (engine broadband)
    assert law("road", "speed", 20.0, 40.0) == pytest.approx(4.5, abs=1.2)
    assert law("engine_broadband", "rpm", 1600.0, 3200.0) == pytest.approx(3.0, abs=1.2)


def test_two_ears_are_coherent_at_low_frequency_and_independent_above_1_khz():
    fs = 16000.0
    sc = synth_scenario("ice_runup", fs, 12.0, seed=3, n_channels=2, **ICE_KW)
    m = sc.t > 1.0

    def coherence(a):
        f, pxy = signal.csd(a[:, 0], a[:, 1], fs, nperseg=4096)
        _, p11 = signal.welch(a[:, 0], fs, nperseg=4096)
        _, p22 = signal.welch(a[:, 1], fs, nperseg=4096)
        return f, pxy / np.sqrt(p11 * p22)

    for name in ("road", "rumble", "engine_broadband"):  # diffuse-field coherence, sinc(2 pi f d/c)
        f, rho = coherence(sc.stems[name][m])
        assert np.mean(rho.real[(f >= 100) & (f < 250)]) == pytest.approx(0.95, abs=0.04), name
        assert np.mean(np.abs(rho[(f >= 1500) & (f < 4000)])) < 0.25, name
        a = sc.stems[name][m]
        assert abs(level_db(a[:, 0]) - level_db(a[:, 1])) < 0.5, name
    # sources reach the ears through the cabin response: smooth direct arrival below the
    # Schroeder frequency (the channels differ by the delay only), diffuse tail above it
    f, rho = coherence(sc.stems["orders"][m])
    assert np.mean(rho.real[(f >= 100) & (f < 250)]) == pytest.approx(0.95, abs=0.04)
    f, rho = coherence(sc.stems["combustion"][m])  # bursts at 1.9 kHz
    assert abs(np.mean(rho.real[(f >= 1500) & (f < 4000)])) < 0.3
    x = sc.x[m]
    assert abs(level_db(x[:, 0]) - level_db(x[:, 1])) < 1.0  # the ears hear the same level
    o = sc.stems["orders"][m]
    assert abs(level_db(o[:, 0]) - level_db(o[:, 1])) < 1.0


def test_ev_levels_and_weak_tones(ev):
    """~90 dB re 20 uPa at 90 km/h; >= 95 % of the broadband energy below 200 Hz; tonal-to-
    broadband about -12 dB A-weighted at full load, and much lower unweighted."""
    m = (ev.speed_kmh > 85) & (ev.speed_kmh < 95)
    assert m.sum() > 8000
    tonal = sum(ev.stems[k] for k in ("motor", "gear", "inverter", "tones"))[m]
    broad = (ev.stems["road"] + ev.stems["rumble"])[m]
    assert level_db(ev.x[m]) == pytest.approx(90.0, abs=3.0)
    f, p = signal.welch(broad, ev.fs, nperseg=4096)
    assert np.sum(p[f < 200.0]) / np.sum(p) >= 0.95
    ratio_a = a_weighted_level_db(tonal, ev.fs) - a_weighted_level_db(broad, ev.fs)
    assert ratio_a == pytest.approx(-12.0, abs=4.5)  # the cabin fading of the lines is random
    assert level_db(tonal) - level_db(broad) < -20.0  # much lower unweighted
    f, p = signal.welch(ev.stems["road"][m], ev.fs, nperseg=2048)
    k = (f >= 400) & (f <= 4000)
    assert np.polyfit(np.log2(f[k]), 10 * np.log10(p[k]), 1)[0] == pytest.approx(-9.0, abs=0.9)
    f, p = signal.welch(ev.stems["rumble"][m], ev.fs, nperseg=16384)
    kk = f < 300.0
    assert 24.0 < f[kk][np.argmax(p[kk])] < 36.0  # the 25 to 35 Hz hump
    w1 = (ev.speed_kmh > 36) & (ev.speed_kmh < 44) & (ev.t > 0.5)
    w2 = (ev.speed_kmh > 72) & (ev.speed_kmh < 88)
    d = level_db(ev.stems["rumble"][w2]) - level_db(ev.stems["rumble"][w1])
    assert d == pytest.approx(6.0, abs=2.5)  # +6 dB per doubling of speed (40 to 80 km/h)


def test_steady_cruise_has_weak_tones(cruise, cruise_ev):
    """Tonal-to-broadband about -20 dB A-weighted at 90 km/h, for either powertrain."""
    m = cruise.t > 1.0
    tonal = cruise.stems["orders"][m]
    broad = (cruise.x_clean - cruise.stems["orders"])[m]
    r = a_weighted_level_db(tonal, cruise.fs) - a_weighted_level_db(broad, cruise.fs)
    assert r == pytest.approx(-20.0, abs=4.5)
    m = cruise_ev.t > 1.0
    keys = ("motor", "gear", "inverter", "tones")
    tonal = sum(cruise_ev.stems[k] for k in keys)[m]
    broad = (cruise_ev.stems["road"] + cruise_ev.stems["rumble"])[m]
    r = a_weighted_level_db(tonal, cruise_ev.fs) - a_weighted_level_db(broad, cruise_ev.fs)
    assert r == pytest.approx(-20.0, abs=4.5)
    assert cruise.config["parameters"]["load"] < 1.0 and cruise_ev.config["parameters"]["load"] < 1.0


# --------------------------------------------------------------------------
# tachometer
# --------------------------------------------------------------------------


def test_tacho_pulses_sit_at_the_true_angles_plus_delay():
    delay = 0.0023  # 36.8 samples at 16 kHz: a fractional delay
    sc = synth_scenario(
        "ice_runup", 16000.0, 6.0, seed=3, tacho={"ppr": 20, "jitter": 0.0, "delay": delay},
        speed={"accel_ms2": 3.5},
    )
    assert sc.config["resolved"]["tacho_delay"] == delay
    rpm_est, info = tacho_to_rpm(sc.tacho, sc.fs, 20, return_info=True)
    assert info["n_missing"] == 0 and info["n_spurious"] == 0
    detected = info["pulse_times"]
    assert detected.size == sc.pulse_times.size
    assert np.max(np.abs(detected - sc.pulse_times)) < 0.02 / sc.fs  # exact edge times
    angle = np.interp(detected - delay, sc.t, sc.rev)
    assert np.max(np.abs(angle - sc.pulse_revs)) < 2e-5  # k / ppr at the undelayed time
    assert np.allclose(np.diff(sc.pulse_revs), 1.0 / 20, atol=1e-12)
    ref = synth_scenario(
        "ice_runup", 16000.0, 6.0, seed=3, tacho={"ppr": 20, "jitter": 0.0, "delay": 0.0},
        speed={"accel_ms2": 3.5},
    )
    k = min(ref.pulse_times.size, sc.pulse_times.size)
    assert np.allclose(sc.pulse_times[:k] - ref.pulse_times[:k], delay, atol=1e-12)
    sl = slice(4000, -4000)
    lag_err = np.abs(rpm_est[sl] - np.interp(sc.t[sl] - delay, sc.t, sc.rpm_mean))
    plain_err = np.abs(rpm_est[sl] - sc.rpm_mean[sl])
    assert np.median(lag_err) < np.median(plain_err) + 1e-9


def test_tacho_jitter_and_defaults(ice, ev):
    for sc, ppr, jit in ((ice, 20, 3e-6), (ev, 24, 1e-6)):
        tc = sc.config["parameters"]["tacho"]
        assert tc["ppr"] == ppr and tc["jitter"] == jit
        assert sc.tacho.shape == (sc.n,) and sc.pulse_times.size == sc.pulse_revs.size
        assert sc.pulse_times.size > 0 and sc.pulse_times[-1] < sc.t[-1]
        assert sc.tacho.max() > 4.5 and sc.tacho.min() < 0.5
        delay = tc["delay"]
        err = np.interp(sc.pulse_times - delay, sc.t, sc.rev) - sc.pulse_revs
        rps = np.interp(sc.pulse_times - delay, sc.t, sc.rpm) / 60.0
        z = err / (jit * rps)
        assert 0.7 < np.std(z) < 1.3
    d = scenario_defaults("ice_runup")["tacho"]
    assert d["ppr"] == 60 and d["jitter"] == 3e-6 and d["delay"] == 0.0015
    d = scenario_defaults("ev_runup")["tacho"]
    assert d["ppr"] == 48 and d["jitter"] == 1e-6 and d["delay"] == 0.0005


# --------------------------------------------------------------------------
# determinism and configuration
# --------------------------------------------------------------------------


def test_same_seed_is_bit_identical_and_other_seeds_differ():
    kw = dict(ICE_KW)
    a = synth_scenario("ice_runup", 8000.0, 4.0, seed=21, **kw)
    b = synth_scenario("ice_runup", 8000.0, 4.0, seed=21, **kw)
    c = synth_scenario("ice_runup", 8000.0, 4.0, seed=22, **kw)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.tacho, b.tacho)
    for k in a.stems:
        assert np.array_equal(a.stems[k], b.stems[k]), k
    assert a.config == b.config
    assert not np.array_equal(a.x, c.x)
    assert not np.array_equal(a.stems["road"], c.stems["road"])
    assert not np.array_equal(a.order_phase, c.order_phase)
    d = synth_scenario("ice_runup", 8000.0, 4.0, seed=None, **kw)
    assert isinstance(d.config["seed"], int)
    e = synth_scenario("ice_runup", 8000.0, 4.0, seed=d.config["seed"], **kw)
    assert np.array_equal(d.x, e.x)
    g = synth_scenario("ice_runup", 8000.0, 4.0, seed=np.random.default_rng(5), **kw)
    assert g.config["seed"] is None
    h1 = synth_scenario("ev_runup", 24000.0, 3.0, seed=9, **EV_KW)  # the hopping carrier too
    h2 = synth_scenario("ev_runup", 24000.0, 3.0, seed=9, **EV_KW)
    assert np.array_equal(h1.x, h2.x)
    assert np.array_equal(h1.carrier["carrier_hz"], h2.carrier["carrier_hz"])


def test_adding_a_component_does_not_change_the_others():
    full = synth_scenario("ice_runup", 8000.0, 4.0, seed=31, **ICE_KW)
    less = synth_scenario("ice_runup", 8000.0, 4.0, seed=31, road=None, **ICE_KW)
    for k in less.stems:
        assert np.array_equal(full.stems[k], less.stems[k]), k  # independent random streams


def test_config_is_strict_json_and_complete(ice, ev, cruise, cruise_ev):
    for sc in (ice, ev, cruise, cruise_ev):
        text = json.dumps(sc.config, allow_nan=False)
        assert json.loads(text) == sc.config
        assert sc.config["kind"] == sc.kind and sc.config["fs"] == sc.fs
        assert isinstance(sc.config["seed"], int) and sc.config["rota_version"]
        assert sc.config["parameters"]["duration"] == sc.t.size / sc.fs
        assert sc.config["n"] == sc.n and sc.config["n_channels"] == 1
        assert "blocks" in sc.config and "resolved" in sc.config
    assert ice.config["parameters"]["tacho"]["ppr"] == 20
    assert ice.config["parameters"]["drive"]["shift_rpm"] == 4500.0
    assert ice.config["parameters"]["drive"]["gear_ratios"] == [3.5, 2.45, 1.71, 1.2, 0.84]
    rebuild = lambda sc: synth_scenario(
        sc.kind, sc.fs, sc.t.size / sc.fs, seed=sc.config["seed"],
        **{k: v for k, v in sc.config["parameters"].items() if k != "duration"},
    )
    for sc in (ice, ev, cruise_ev):
        assert np.array_equal(rebuild(sc).x, sc.x)  # the recorded parameters rebuild the signal


def test_save_writes_npz_and_json_sidecar(tmp_path, ice, ev):
    for sc in (ice, ev):
        npz, side = sc.save(tmp_path / f"{sc.kind}.npz")
        assert npz.name == f"{sc.kind}.npz" and side.name == f"{sc.kind}.json"
        with np.load(npz) as z:
            assert np.array_equal(z["x"], sc.x) and np.array_equal(z["rev"], sc.rev)
            assert np.array_equal(z["tacho"], sc.tacho)
            for name, s in sc.stems.items():
                assert np.array_equal(z[f"stem_{name}"], s)
            for name, s in sc.source_stems.items():
                assert np.array_equal(z[f"source_{name}"], s)
            assert np.array_equal(z["ir"], sc.ir) and float(z["fs"]) == sc.fs
            assert "event_shift_times" in z.files
            assert json.loads(str(z["config_json"])) == sc.config
        assert json.loads(side.read_text(encoding="utf-8")) == sc.config
    with np.load(tmp_path / "ev_runup.npz") as z:
        assert "carrier_q" in z.files and "carrier_carrier_hz" in z.files


# --------------------------------------------------------------------------
# channels
# --------------------------------------------------------------------------


def test_two_channels_have_shapes_and_the_interchannel_delay():
    fs = 16000.0
    two = synth_scenario("ice_runup", fs, 6.0, seed=5, n_channels=2, **ICE_KW)
    one = synth_scenario("ice_runup", fs, 6.0, seed=5, n_channels=1, **ICE_KW)
    assert two.x.shape == (two.n, 2) and one.x.shape == (one.n,)
    assert two.n_channels == 2 and one.n_channels == 1
    for k, s in two.stems.items():
        assert s.shape == (two.n, 2), k
    for k, s in two.source_stems.items():
        assert s.shape == (two.n,), k  # one source, two receivers
        assert np.array_equal(s, one.source_stems[k]), k  # the source does not depend on channels
    assert two.ir.shape[1] == 2 and one.ir.ndim == 1
    # unit gain below the Schroeder frequency (338 Hz), mean-square response 1 above it
    for c in range(2):
        for fk in (50.0, 100.0, 200.0):
            h = np.sum(two.ir[:, c] * np.exp(-2j * np.pi * fk / fs * np.arange(two.ir.shape[0])))
            assert abs(h) == pytest.approx(1.0, abs=0.03), (c, fk)
        hf = np.fft.rfft(two.ir[:, c], 1 << 15)
        ff = np.fft.rfftfreq(1 << 15, 1 / fs)
        assert np.mean(np.abs(hf[(ff > 1000) & (ff < 6000)]) ** 2) == pytest.approx(1.0, rel=0.2)
    cab = two.config["parameters"]["cabin"]
    taus = _ir_delays(two.ir, fs)  # the smooth direct response below the crossover
    assert taus[1] == pytest.approx(cab["interchannel_delay"], abs=1.5e-5)
    assert taus[0] == pytest.approx(0.0, abs=1e-12)
    assert np.array_equal(two.x, two.stem_sum())
    assert level_db(two.stems["road"][:, 0]) == pytest.approx(level_db(one.stems["road"]), abs=0.3)


def test_three_channels_have_increasing_delays():
    sc = synth_scenario("steady_cruise", 16000.0, 3.0, seed=2, n_channels=3, **CRUISE_KW)
    assert sc.x.shape == (sc.n, 3)
    taus = _ir_delays(sc.ir, sc.fs)
    d = sc.config["parameters"]["cabin"]["interchannel_delay"]
    assert np.allclose(taus, d * np.arange(3), atol=3e-5)


# --------------------------------------------------------------------------
# path and quasi-static truth
# --------------------------------------------------------------------------


def test_path_response_is_the_steady_state_gain_of_the_path():
    sc = synth_scenario("steady_cruise", 16000.0, 4.0, seed=4, **CRUISE_KW)
    f = np.array([40.0, 112.0, 300.0, 850.0])
    h = sc.path_response(f)
    assert h.shape == f.shape and np.iscomplexobj(h)
    for fk, hk in zip(f, h):
        tone = np.cos(TWO_PI * fk * sc.t)
        y = sc.through_path(tone)[6000:]  # past the start-up of path and tail
        tt = sc.t[6000:]
        a = 2 * np.mean(y * np.cos(TWO_PI * fk * tt))
        b = 2 * np.mean(y * np.sin(TWO_PI * fk * tt))
        assert abs(a - 1j * b - hk) < 0.03 * abs(hk) + 0.01  # y = Re{H e^{jwt}}
    assert sc.path_response(np.ones((2, 3)) * 200.0).shape == (2, 3)
    two = synth_scenario("steady_cruise", 16000.0, 2.0, seed=4, n_channels=2, **CRUISE_KW)
    assert abs(two.path_response(300.0, channel=0)) != abs(two.path_response(300.0, channel=1))


def test_ice_path_has_the_two_hump_modes():
    """Hump at 45 Hz (+8 dB) and 175 Hz (+6 dB) over the mid-frequency level (modal gain only)."""
    modes = scenario_defaults("ice_runup")["path"]["modes"]
    fs = 48000.0
    mid = np.mean(20 * np.log10(np.abs(synth.modal_response(np.geomspace(300, 1500, 40), modes, fs))))
    h = lambda f: 20 * np.log10(abs(synth.modal_response(f, modes, fs))) - mid
    assert h(45.0) == pytest.approx(8.0, abs=1.5)
    assert h(175.0) == pytest.approx(6.0, abs=1.5)
    assert min(h(f) for f in np.arange(70.0, 95.0)) < h(45.0) - 5.0  # a dip near 80 Hz


def test_quasi_static_amplitude_is_exact_for_a_stationary_order(cruise):
    k = list(cruise.orders).index(2.0)
    qs = cruise.quasi_static_amplitude(k)
    assert qs.shape == (cruise.n,)
    stem = cruise.through_path(cruise.order_stems[:, k])
    sl = slice(int(1.0 * cruise.fs), None)
    z = np.exp(-1j * cruise.order_phase[sl, k]) * stem[sl]
    amp = 2 * np.abs(np.mean(z))
    assert amp == pytest.approx(np.mean(qs[sl]), rel=0.03)


# --------------------------------------------------------------------------
# recorder
# --------------------------------------------------------------------------


def test_measurement_chain_is_applied_and_flags_clipping():
    kw = dict(ICE_KW)
    ideal = synth_scenario("ice_runup", 8000.0, 5.0, seed=7, n_channels=2, **kw)
    peak = np.max(np.abs(ideal.x), axis=0)
    clip = 0.6 * peak
    rec = synth_scenario(
        "ice_runup", 8000.0, 5.0, seed=7, n_channels=2,
        measurement={"gain": 2.0, "clip": 2.0 * clip, "dc": 0.01, "noise_floor_rms": 1e-3}, **kw
    )
    assert rec.x.shape == ideal.x.shape and rec.clipped.shape == rec.x.shape
    assert rec.clipped.any() and np.all(np.abs(rec.x) <= 2.0 * clip + 1e-12)
    exact = synth_scenario(
        "ice_runup", 8000.0, 5.0, seed=7, n_channels=2,
        measurement={"gain": 2.0, "clip": 2.0 * clip, "dc": 0.01}, **kw
    )
    assert np.array_equal(exact.clipped, np.abs(2.0 * ideal.x + 0.01) > 2.0 * clip)
    assert np.array_equal(exact.x, np.clip(2.0 * ideal.x + 0.01, -2.0 * clip, 2.0 * clip))
    assert np.abs(rec.clipped.mean() - exact.clipped.mean()) < 0.02
    for k in ideal.stems:
        assert np.array_equal(ideal.stems[k], rec.stems[k])  # the truth before the recorder
    assert np.array_equal(rec.x_clean, ideal.x)
    assert rec.config["measurement"]["clip"] is not None
    assert rec.config["measurement"]["n_out"] == rec.x.shape[0]
    lin = synth_scenario("ice_runup", 8000.0, 5.0, seed=7, measurement={"gain": 2.0, "dc": 0.5}, **kw)
    mono = synth_scenario("ice_runup", 8000.0, 5.0, seed=7, **kw)
    assert np.allclose(lin.x, 2.0 * mono.x + 0.5, atol=1e-12) and not lin.clipped.any()
    noisy = synth_scenario("ice_runup", 8000.0, 5.0, seed=7, measurement={"noise_floor_rms": 2e-3}, **kw)
    assert np.std(noisy.x - mono.x) == pytest.approx(2e-3, rel=0.02)


def test_quantisation_puts_the_recording_on_a_grid():
    sc = synth_scenario(
        "ice_runup", 8000.0, 3.0, seed=8, measurement={"bits": 8, "full_scale": 4.0}, **ICE_KW
    )
    step = 8.0 / 2**8
    q = sc.x / step
    assert np.allclose(q, np.round(q), atol=1e-9)
    err = sc.x - sc.x_clean
    assert np.max(np.abs(err[~sc.clipped])) <= step / 2 + 1e-12
    assert np.std(err) == pytest.approx(step / np.sqrt(12), rel=0.1)


def test_clock_drift_changes_the_length_and_the_true_times():
    ppm = 400.0
    sc = synth_scenario("ice_runup", 8000.0, 5.0, seed=9, measurement={"clock_drift_ppm": ppm}, **ICE_KW)
    ideal = synth_scenario("ice_runup", 8000.0, 5.0, seed=9, **ICE_KW)
    eps = ppm * 1e-6
    n = ideal.n
    assert sc.x.shape[0] == int(np.floor((n - 1) * (1 + eps))) + 1 and sc.x.shape[0] > n
    assert sc.t.size == n and sc.t_true.size == sc.x.size
    assert np.allclose(sc.t_true, np.arange(sc.x.size) / (sc.fs * (1 + eps)))
    r = sc.at_recording(sc.rpm)
    assert r.shape == sc.x.shape and np.allclose(r, np.interp(sc.t_true, sc.t, sc.rpm))
    assert sc.at_recording(sc.rev).shape == sc.x.shape
    assert np.array_equal(ideal.at_recording(ideal.rpm), ideal.rpm)
    with pytest.raises(ValueError):
        sc.at_recording(np.zeros(5))
    i = np.arange(2000, 30000)
    ref = np.interp(sc.t_true[i], ideal.t, ideal.x)
    assert np.sqrt(np.mean((sc.x[i] - ref) ** 2)) < 0.05 * np.sqrt(np.mean(ref**2))
    assert sc.tacho.shape == (sc.n,)  # the tachometer is on its own clock


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="kind"):
        synth_scenario("diesel")
    with pytest.raises(ValueError, match="unknown parameter"):
        synth_scenario("ice_runup", 8000.0, 2.0, nonsense=1)
    with pytest.raises(ValueError, match="unknown parameter"):
        synth_scenario("ice_runup", 8000.0, 2.0, tacho={"pulses": 5})
    with pytest.raises(ValueError, match="unknown parameter"):
        synth_scenario("ev_runup", 24000.0, 2.0, combustion=None)  # not a key of the EV
    with pytest.raises(ValueError, match="unknown parameter"):
        synth_scenario("ice_runup", 8000.0, 2.0, powertrain="ev")  # only for steady_cruise
    with pytest.raises(ValueError, match="powertrain"):
        synth_scenario("steady_cruise", 8000.0, 2.0, powertrain="diesel")
    with pytest.raises(ValueError, match="n_channels"):
        synth_scenario("ice_runup", 8000.0, 2.0, n_channels=0)
    with pytest.raises(ValueError, match="fs"):
        synth_scenario("ice_runup", -1.0, 2.0)
    with pytest.raises(ValueError, match="duration"):
        synth_scenario("ice_runup", 8000.0, -1.0)
    with pytest.raises(ValueError, match="measurement key"):
        synth_scenario("ice_runup", 8000.0, 2.0, measurement={"volume": 1.0})
    with pytest.raises(TypeError):
        synth_scenario("ice_runup", 8000.0, 2.0, measurement=[1])
    with pytest.raises(TypeError):
        synth_scenario("ice_runup", 8000.0, 2.0, seed="a")
    with pytest.raises(ValueError, match="shorter than 2 samples"):  # 60 ppr at 16 kHz
        synth_scenario("ice_runup", 16000.0, 6.0)
    with pytest.raises(ValueError, match="speed.mode"):
        synth_scenario("ice_runup", 8000.0, 2.0, speed={"mode": "spiral"})
    with pytest.raises(ValueError, match="cruise_gear"):
        synth_scenario("steady_cruise", 8000.0, 2.0, tacho=None, speed={"cruise_gear": 9})
    with pytest.raises(ValueError):  # a path mode above Nyquist
        synth_scenario("ice_runup", 4000.0, 2.0, tacho=None)
    with pytest.raises(ValueError, match="end_kmh"):
        synth_scenario("ev_runup", 24000.0, 2.0, speed={"mode": "ramp"}, **EV_KW)
    with pytest.raises(ValueError, match="at least 64"):
        synth_scenario("ice_runup", 8000.0, 0.001)
    with pytest.raises(ValueError, match="below fs/2"):
        bad = [{"freq": 13000.0, "tnr_db": 6.0, "bandwidth_hz": 20.0}]
        synth_scenario("ev_runup", 24000.0, 2.0, **{**EV_KW, "tones": bad})


# --------------------------------------------------------------------------
# defaults run in reasonable time
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind,extra", [
    ("ice_runup", {}), ("ev_runup", {}), ("steady_cruise", {}),
    ("steady_cruise", {"powertrain": "ev"}),
])
def test_default_settings_run_in_reasonable_time(kind, extra):
    t0 = time.perf_counter()
    sc = synth_scenario(kind, seed=1, **extra)
    elapsed = time.perf_counter() - t0
    assert elapsed < 40.0  # about 1 to 5 s on a laptop; generous for slow CI
    d = scenario_defaults(kind, extra.get("powertrain", "ice"))
    assert sc.fs == 48000.0 and sc.n == round(d["duration"] * 48000.0)
    assert sc.x.shape == (sc.n,) and np.all(np.isfinite(sc.x))
    assert np.array_equal(sc.x, sc.stem_sum())
    assert 0.05 < np.max(np.abs(sc.x)) < 50.0  # Pa: a plausible interior pressure
    rpm, info = tacho_to_rpm(sc.tacho, sc.fs, d["tacho"]["ppr"], return_info=True)
    assert info["n_missing"] == 0 and info["n_spurious"] == 0
    assert np.nanmedian(np.abs(rpm[5000:-5000] / sc.rpm_mean[5000:-5000] - 1)) < 0.01
    if kind == "ice_runup":
        assert sc.events["shift_times"].size == 2 and sc.rpm_mean.max() <= 6500.0
    if sc.config["resolved"]["powertrain"] == "ev":
        assert sc.carrier is not None and sc.rpm_mean.max() < 15000.0
        assert np.max(sc.carrier["frequency"][sc.carrier["amplitude"] > 0]) < sc.fs / 2
