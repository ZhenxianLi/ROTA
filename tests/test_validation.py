"""v0.1 validation cases T1, T2, T5 and T14 (ROADMAP section 5.3) against analytic truth.

The signals come from ``validation/make_signals.py`` (built here in memory,
nothing is read from disk). Every expected amplitude is read from the truth
stored in the :class:`rota.synth.SynthSignal` (``order_amplitude``, ``stems``,
``rpm``, ``pulse_times``), never from the parameter lists that built the
signal (ROADMAP section 6, item 19).

Tolerances (ROADMAP section 5.4, against analytic truth):

* speed, median relative error: 0.1 %
* order amplitude, stationary segments: 0.2 dB
* order amplitude near a resonance: 1 dB

Truth of a map frame. A short-time Fourier transform reads a window-weighted
mean of an amplitude envelope. The expected reading of a frame is therefore
the Hann-weighted mean of ``order_amplitude`` over the samples the frame
spans (time samples for a frequency map; for an order map the window is Hann
in shaft angle, so time samples are weighted by the Hann value at their angle
and by the speed, which converts a time average into an angle average).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from rota.maps import rpm_frequency_map, rpm_order_map
from rota.metrics import track_error
from rota.slices import order_slice, order_spectrum
from rota.speed import tacho_to_rpm

# --------------------------------------------------------------------------
# Signals: validation/make_signals.py is a script, not part of the package.
# --------------------------------------------------------------------------

_SCRIPT = Path(__file__).resolve().parents[1] / "validation" / "make_signals.py"
_spec = importlib.util.spec_from_file_location("make_signals", _SCRIPT)
make_signals = importlib.util.module_from_spec(_spec)
sys.modules["make_signals"] = make_signals
_spec.loader.exec_module(make_signals)

# ROADMAP section 5.4, "against analytic truth".
TOL_SPEED = 1e-3  # median relative speed error
TOL_STATIONARY_DB = 0.2  # order amplitude, stationary segments
TOL_RESONANCE_DB = 1.0  # order amplitude near a resonance

EDGE = 3  # frames dropped at each end of a map (filter and window edge effects)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _db(a, ref):
    return 20.0 * np.log10(np.asarray(a, dtype=float) / np.asarray(ref, dtype=float))


def _frame_truth(sig, m, k):
    """Truth of order ``k`` in every frame of map ``m``.

    Returns ``(amplitude, f_min, f_max)``: the Hann-weighted mean peak
    amplitude of ``sig.order_amplitude[:, k]`` over the frame (Pa, peak) and
    the lowest and highest instantaneous frequency of the order within it (Hz).
    """
    n = m.nperseg
    if m.domain == "order":
        spr = n * m.resolution  # angle samples per revolution
        centre = np.interp(m.t, sig.t, sig.rev)  # frame centre, rev
        coord, density = sig.rev, sig.rpm  # time samples -> angle average
    else:
        spr = sig.fs
        centre = m.t
        coord, density = sig.t, np.ones(sig.t.size)
    lo = centre - 0.5 * (n - 1) / spr  # first sample of each frame
    span = n / spr
    amp = sig.order_amplitude[:, k]
    freq = sig.order_frequency[:, k]
    a, f_min, f_max = (np.empty(lo.size) for _ in range(3))
    for i, start in enumerate(lo):
        i0, i1 = np.searchsorted(coord, [start, start + span])
        u = (coord[i0:i1] - start) / span
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * u)) * density[i0:i1]
        a[i] = np.sum(w * amp[i0:i1]) / np.sum(w)
        f_min[i], f_max[i] = freq[i0:i1].min(), freq[i0:i1].max()
    return a, f_min, f_max


def _rms_truth(sig, m, k):
    """Truth reading of an RMS-scaled map: peak amplitude over sqrt(2)."""
    assert m.amplitude == "rms"
    return _frame_truth(sig, m, k)[0] / np.sqrt(2.0)


def _interior(n):
    mask = np.zeros(n, dtype=bool)
    mask[EDGE:-EDGE] = True
    return mask


# --------------------------------------------------------------------------
# Reproducibility of the signals themselves
# --------------------------------------------------------------------------


def test_signals_are_reproducible():
    a, b = make_signals.t1(), make_signals.t1()
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.order_phase, b.order_phase)
    assert a.config["seed"] == 1


def test_main_writes_signals_and_sidecars(tmp_path):
    written = make_signals.main(tmp_path, ["T1"])
    npz, sidecar = written["T1"]
    assert npz.parent == tmp_path and sidecar.parent == tmp_path
    ref = make_signals.t1()
    with np.load(npz) as data:
        assert np.array_equal(data["x"], ref.x)
        assert np.array_equal(data["order_amplitude"], ref.order_amplitude)
    assert '"seed": 1' in sidecar.read_text(encoding="utf-8")
    with pytest.raises(KeyError):
        make_signals.main(tmp_path, ["T99"])


# --------------------------------------------------------------------------
# T1: one order, constant amplitude, linear run-up
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def t1():
    return make_signals.t1()


@pytest.fixture(scope="module")
def t1_order_map(t1):
    return rpm_order_map(t1.x, t1.fs, t1.rpm, order_resolution=0.25, max_order=16)


def test_t1_truth_is_constant_unit_peak(t1):
    assert t1.orders.tolist() == [2.0]
    assert np.allclose(t1.order_amplitude, 1.0, atol=1e-12)
    assert t1.rpm[0] == pytest.approx(1000.0) and t1.rpm[-1] == pytest.approx(4000.0)


def test_t1_order_slice_from_order_map(t1, t1_order_map):
    m = t1_order_map
    assert m.amplitude == "rms"
    s = order_slice(m, 2.0)
    truth = _rms_truth(t1, m, 0)
    ok = _interior(m.t.size)
    assert np.all(np.isfinite(s.amplitude[ok]))
    err = np.abs(_db(s.amplitude, truth))[ok]
    assert err.max() <= TOL_STATIONARY_DB
    # The map covers the whole run-up.
    assert m.rpm.min() < 1100.0 and m.rpm.max() > 3900.0


def test_t1_order_spectrum_peaks_at_order_2(t1, t1_order_map):
    m = t1_order_map
    sp = order_spectrum(m)
    i = int(np.argmax(sp.amplitude))
    assert sp.axis[i] == pytest.approx(2.0)
    # Constant line: the energy average equals the RMS level of the line.
    assert abs(_db(sp.amplitude[i], 1.0 / np.sqrt(2.0))) <= TOL_STATIONARY_DB
    # Nothing else in the spectrum: every bin more than 1 order away is >= 60 dB down.
    far = np.abs(sp.axis - 2.0) > 1.0
    assert _db(sp.amplitude[far].max(), sp.amplitude[i]) < -60.0


@pytest.mark.parametrize("resolution", [1.0, 2.0, 4.0])
def test_t1_order_slice_from_frequency_map(t1, resolution):
    m = rpm_frequency_map(t1.x, t1.fs, t1.rpm, resolution=resolution)
    s = order_slice(m, 2.0)
    truth = _rms_truth(t1, m, 0)
    ok = _interior(m.t.size)
    assert np.all(np.isfinite(s.amplitude[ok]))
    err = np.abs(_db(s.amplitude, truth))[ok]
    assert err.max() <= TOL_STATIONARY_DB


# --------------------------------------------------------------------------
# T2: four orders, a resonance, a fixed tone and noise
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def t2():
    return make_signals.t2()


@pytest.fixture(scope="module")
def t2_order_map(t2):
    return rpm_order_map(t2.x, t2.fs, t2.rpm, order_resolution=0.25, max_order=16)


def test_t2_truth_is_not_the_parameter_list(t2):
    # Order 4.5 passes through the resonance (peak gain 4): its true envelope
    # is far from the 0.40 Pa of the parameter list, so a test written against
    # the parameter list would be wrong by more than 6 dB there.
    k = list(t2.orders).index(4.5)
    nominal = t2.config["orders"][k]["amplitude"]
    assert _db(t2.order_amplitude[:, k].max(), nominal) > 6.0
    assert t2.stems["noise"].std() >= 0.02  # noise is present (resonance only raises it)
    assert np.array_equal(t2.x, t2.stems["orders"] + t2.stems["tones"] + t2.stems["noise"])


@pytest.mark.parametrize("order, has_near_frames", [(1.0, False), (2.0, True), (3.0, True), (4.5, True)])
def test_t2_order_amplitudes(t2, t2_order_map, order, has_near_frames):
    m = t2_order_map
    k = list(t2.orders).index(order)
    reading = order_slice(m, order).amplitude
    truth, f_min, f_max = _frame_truth(t2, m, k)
    truth = truth / np.sqrt(2.0)  # peak -> RMS, the map's convention
    res = t2.config["resonances"][0]
    half_width = 2.0 * res["freq"] / res["q"]
    near = (f_max >= res["freq"] - half_width) & (f_min <= res["freq"] + half_width)
    ok = _interior(m.t.size)
    assert np.all(np.isfinite(reading[ok]))
    err = np.abs(_db(reading, truth))
    away = ok & ~near
    assert away.sum() >= 100
    assert err[away].max() <= TOL_STATIONARY_DB
    if has_near_frames:
        assert (ok & near).sum() >= 50  # the resonance zone is actually probed
        assert err[ok & near].max() <= TOL_RESONANCE_DB
    else:
        assert not np.any(ok & near)


def test_t2_fixed_tone_in_frequency_map(t2):
    tone = t2.config["tones"][0]["freq"]
    # Truth amplitude of the tone as rendered (resonance included), by a
    # least-squares fit of the tone stem.
    basis = np.column_stack([np.cos(2 * np.pi * tone * t2.t), np.sin(2 * np.pi * tone * t2.t)])
    coef = np.linalg.lstsq(basis, t2.stems["tones"], rcond=None)[0]
    truth = float(np.hypot(*coef))
    assert truth != pytest.approx(t2.config["tones"][0]["amplitude"], rel=0.02)  # resonance acts on it

    res = 5.0  # Hz; 450 Hz falls on a bin, so there is no scalloping loss
    m = rpm_frequency_map(t2.x, t2.fs, t2.rpm, resolution=res, amplitude="peak")
    j = int(np.argmin(np.abs(m.axis - tone)))
    assert m.axis[j] == pytest.approx(tone)
    reading = np.abs(m.values[j])
    assert np.abs(_db(reading, truth)).max() <= TOL_STATIONARY_DB
    # In every frame the tone is the largest line within +-30 Hz and stands at
    # least 20 dB above everything else in that band (outside its own lobe).
    band = np.abs(m.axis - tone) <= 30.0
    lobe = np.abs(m.axis - tone) <= 2.5 * res
    mag = np.abs(m.values)
    assert np.all(np.flatnonzero(band)[np.argmax(mag[band], axis=0)] == j)
    others = np.max(mag[band & ~lobe], axis=0)
    assert np.min(_db(reading, others)) > 20.0
    # It sits at a fixed frequency while the speed changes by a factor of 5.
    assert m.rpm.max() / m.rpm.min() > 4.0


# --------------------------------------------------------------------------
# T5: tachometer with jitter and dropped pulses
# --------------------------------------------------------------------------


@pytest.fixture(scope="module", params=[60, 1], ids=["ppr60", "ppr1"])
def t5(request):
    ppr = request.param
    sig = make_signals.t5(ppr=ppr)
    est, info = tacho_to_rpm(sig.tacho, sig.fs, ppr, return_info=True)
    return ppr, sig, est, info


def test_t5_median_speed_error(t5):
    ppr, sig, est, info = t5
    assert sig.config["tacho_ppr"] == ppr
    assert sig.config["tacho_jitter"] == pytest.approx(5e-6)
    stats = track_error(est, sig.rpm)
    assert stats["n_invalid"] == 0
    assert stats["median_rel_error"] <= TOL_SPEED


def test_t5_counts_missing_pulses(t5):
    ppr, sig, est, info = t5
    n_dropped = sig.config["tacho_n_dropped"]
    assert n_dropped > 0
    # A dropped first or last pulse leaves no gap and cannot be counted; the
    # case is built so that neither end pulse is dropped.
    assert sig.pulse_revs[0] == pytest.approx(1.0 / ppr)
    interior = int(round((sig.pulse_revs[-1] - sig.pulse_revs[0]) * ppr)) + 1 - sig.pulse_revs.size
    assert interior == n_dropped
    assert info["n_missing"] == n_dropped
    assert info["n_spurious"] == 0


def test_t5_detected_edges_match_rendered_pulses(t5):
    ppr, sig, est, info = t5
    assert info["pulse_times"].size == sig.pulse_times.size
    assert np.abs(info["pulse_times"] - sig.pulse_times).max() < 1e-9  # s, jitter included


# --------------------------------------------------------------------------
# T14: start from idle, orders 2 and 2.5 half an order apart
# --------------------------------------------------------------------------

ORDER_RESOLUTION = 0.125  # orders; window of 8 revolutions


def _idle_frames(sig, m):
    """Frames whose speed range lies entirely at the idle speed (not the final hold)."""
    idle = sig.rpm[0]
    return m.rpm_max <= idle * (1.0 + 1e-9)


def _td_map(sig, **kw):
    """Equal-bandwidth time-domain map: the angle-domain window duration at top speed."""
    window_s = (1.0 / ORDER_RESOLUTION) / (sig.rpm.max() / 60.0)  # 8 rev at top speed
    return rpm_frequency_map(sig.x, sig.fs, sig.rpm, resolution=1.0 / window_s, **kw)


@pytest.fixture(scope="module")
def t14():
    return make_signals.t14()


@pytest.fixture(scope="module")
def t14_order_map(t14):
    return rpm_order_map(t14.x, t14.fs, t14.rpm, order_resolution=ORDER_RESOLUTION, max_order=16)


def test_t14_design(t14):
    assert t14.orders.tolist() == [1.0, 2.0, 2.5]
    assert t14.rpm[0] == pytest.approx(800.0)
    assert np.all(t14.rpm[: int(2.0 * t14.fs)] == t14.rpm[0])  # idle for the first 2 s
    k2, k25 = list(t14.orders).index(2.0), list(t14.orders).index(2.5)
    assert np.allclose(t14.order_amplitude[:, k2], t14.order_amplitude[:, k25])


def test_t14_angle_domain_map_separates_half_orders(t14, t14_order_map):
    m = t14_order_map
    idle = _idle_frames(t14, m)
    assert idle.sum() >= 8
    ok = _interior(m.t.size)
    for order in (1.0, 2.0, 2.5):
        k = list(t14.orders).index(order)
        s = order_slice(m, order)
        err = np.abs(_db(s.amplitude, _rms_truth(t14, m, k)))
        assert err[idle].max() <= TOL_STATIONARY_DB, f"order {order} at idle"
        assert err[ok].max() <= TOL_STATIONARY_DB, f"order {order} over the run-up"
    # The two lines are four bins apart, outside each other's default band
    # (+-2 bins), at every speed.
    assert 0.5 / m.resolution == pytest.approx(4.0)


@pytest.mark.parametrize("seed", [14, 0, 4, 7])  # four relative phases of orders 2 and 2.5
def test_t14_equal_bandwidth_time_map_cannot_separate_at_idle(seed):
    sig = make_signals.t14(seed=seed)
    m = _td_map(sig)  # amplitude "rms"
    idle = _idle_frames(sig, m)
    assert idle.sum() >= 30

    # At idle the lines are 6.7 Hz apart, about one bin of the time-domain map:
    # inside each other's default band of +-2 bins (the Hann main lobe).
    spacing_bins = 0.5 * (sig.rpm[0] / 60.0) / m.resolution
    assert spacing_bins < 2.0

    for order in (2.0, 2.5):
        k = list(sig.orders).index(order)
        reading = order_slice(m, order).amplitude[idle]
        truth = _rms_truth(sig, m, k)[idle]
        err = _db(reading, truth)
        # Frame by frame the two lines beat against each other, so the error
        # changes sign with their relative phase; its size does not.
        assert np.median(np.abs(err)) > 1.0
        # Averaged over the idle frames, the lines add in power: the mean
        # reading is too high by more than 1 dB whatever the relative phase.
        assert 10.0 * np.log10(np.mean(reading**2) / np.mean(truth**2)) > 1.0
