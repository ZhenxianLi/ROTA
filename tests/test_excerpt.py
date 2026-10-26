"""Tests for stationary_excerpt against analytic truth (rota.synth)."""

import json

import numpy as np
import pytest
from scipy import signal

from rota.resynth import ExcerptResult, stationary_excerpt
from rota.synth import rpm_profiles, synth_runup

FS = 16000.0
RPM_REF = 2500.0
NOISE_RMS = 0.05
NOISE_PSD = NOISE_RMS**2 / (FS / 2)  # one-sided PSD of white noise of that RMS, unit**2/Hz
# peak amplitude of every order at RPM_REF, from the specification below (not from the signal)
SPECS = [
    {"order": 1.0, "amplitude": 1.0, "phase": 0.3},
    {"order": 2.0, "amplitude": 0.5, "phase": 1.0, "gain_rpm": [1000, 4000], "gain": [1.0, 2.0]},
    {"order": 3.5, "amplitude": 0.25, "phase": 2.0, "gain_rpm": [1000, 4000], "gain": [2.0, 0.5]},
]
ORDERS = [1.0, 2.0, 3.5]
TRUTH = {
    sp["order"]: sp["amplitude"] * float(np.interp(RPM_REF, sp.get("gain_rpm", [0, 1]), sp.get("gain", [1, 1])))
    for sp in SPECS
}
OCTAVES = [(250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]


def _db(a, ref):
    return 20.0 * np.log10(np.asarray(a) / np.asarray(ref))


def _octave_levels(freqs, psd, ref):
    """Power average of ``psd`` in each octave, in dB re ``ref`` (a PSD value)."""
    return np.array(
        [10 * np.log10(psd[(freqs >= lo) & (freqs < hi)].mean() / ref) for lo, hi in OCTAVES]
    )


@pytest.fixture(scope="module")
def runup():
    """14 s run-up 1500 -> 3500 rpm; RPM_REF is passed at t = 7 s; white noise, known orders."""
    t = np.arange(round(14 * FS)) / FS
    rpm = rpm_profiles(t, "linear", start=1500, stop=3500)
    sig = synth_runup(FS, rpm, SPECS, noise_rms=NOISE_RMS, seed=4)
    return sig, rpm


@pytest.fixture(scope="module")
def sinusoid(runup):
    sig, rpm = runup
    return stationary_excerpt(sig.x, FS, rpm, RPM_REF, 4.0, orders=ORDERS, seed=1)


@pytest.fixture(scope="module")
def cluster(runup):
    sig, rpm = runup
    return stationary_excerpt(sig.x, FS, rpm, RPM_REF, 4.0, orders=ORDERS, model="cluster", seed=1)


# --------------------------------------------------------------------------
# levels against the truth
# --------------------------------------------------------------------------


def test_amplitudes_match_the_truth_at_the_reference_speed(sinusoid):
    assert isinstance(sinusoid, ExcerptResult) and list(sinusoid.amplitudes) == ORDERS
    for order, a in sinusoid.amplitudes.items():
        assert abs(_db(a, TRUTH[order])) < 0.3, order
    assert sinusoid.info["amplitude"] == "peak" and sinusoid.rpm_ref == RPM_REF


def test_orders_stem_has_the_modelled_frequencies_and_peak_amplitudes(sinusoid):
    stem = sinusoid.orders[320:-320]  # between the fades
    win = signal.windows.flattop(stem.size)  # flat top: less than 0.02 dB scalloping loss
    spec = np.abs(np.fft.rfft(stem * win)) * 2 / win.sum()
    f = np.fft.rfftfreq(stem.size, 1 / FS)
    for order, a in sinusoid.amplitudes.items():
        f0 = order * RPM_REF / 60.0
        near = np.abs(f - f0) < 2.0
        assert abs(f[near][np.argmax(spec[near])] - f0) < 0.25  # on the constant frequency
        assert abs(_db(spec[near].max(), a)) < 0.3  # peak amplitude A_k


def test_residual_psd_matches_the_noise_psd(sinusoid):
    f, p = sinusoid.psd
    assert f.shape == p.shape and f[0] == 0.0 and np.all(p >= 0)
    assert sinusoid.info["psd"]["stationarized"] is True  # the speed varies by +-3 % in the window
    assert sinusoid.info["psd"]["resolution_hz"] <= 5.0 + 1e-9
    levels = _octave_levels(f, p, NOISE_PSD)
    assert np.max(np.abs(levels)) < 1.0, levels
    # the rendered residual has the same level and shape (grain synthesis of that PSD)
    n_f = round(0.02 * FS)
    fr, pr = signal.welch(sinusoid.residual[n_f:-n_f], FS, nperseg=4096)
    assert np.max(np.abs(_octave_levels(fr, pr, NOISE_PSD))) < 1.0


def test_residual_psd_follows_a_band_limited_noise():
    t = np.arange(round(14 * FS)) / FS
    rpm = rpm_profiles(t, "linear", start=1500, stop=3500)
    sig = synth_runup(FS, rpm, SPECS, noise_rms=0.1, noise_band=(None, 2000.0), seed=6)
    ex = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 3.0, orders=ORDERS, seed=2)
    a, b = ex.info["window"]["start"], ex.info["window"]["stop"]
    nperseg = ex.info["psd"]["nperseg"]
    f, p = ex.psd
    ft, pt = signal.welch(sig.stems["noise"][a:b], FS, nperseg=nperseg)  # truth in the same window
    assert np.array_equal(f, ft)
    for lo, hi in [(250, 500), (500, 1000), (1000, 2000)]:
        sel = (f >= lo) & (f < hi)
        assert abs(10 * np.log10(p[sel].mean() / pt[sel].mean())) < 1.0, (lo, hi)
    top = (f >= 4000) & (f < 8000)
    assert 10 * np.log10(p[(f >= 1000) & (f < 2000)].mean() / p[top].mean()) > 30.0


# --------------------------------------------------------------------------
# the two models: envelope swing (ROADMAP section 6, item 13)
# --------------------------------------------------------------------------


def test_sinusoid_model_has_steady_orders(sinusoid, runup):
    sig, rpm = runup
    for key, swing in sinusoid.info["per_order_swing_db"].items():
        assert swing < 0.5, key
    ex = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 3.0, orders=[2.0], seed=3)
    assert ex.info["order_stem"]["swing_db"] < 0.5
    assert ex.info["order_stem"]["crest_factor_db"] == pytest.approx(3.01, abs=0.05)  # a sine
    assert ex.info["mix"]["swing_db"] > 10.0  # the noise floor is there: mix is not a tone
    assert ex.info["model"] == "sinusoid"


def test_cluster_model_swings_clearly_more(cluster, sinusoid, runup):
    sig, rpm = runup
    ex = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 3.0, orders=[2.0], model="cluster", seed=3)
    assert ex.info["order_stem"]["swing_db"] > 6.0  # narrow-band noise: ~18 dB (Rayleigh), not a tone
    assert ex.info["order_stem"]["crest_factor_db"] > 8.0
    for key, swing in cluster.info["per_order_swing_db"].items():
        assert swing > 6.0, key
    assert max(sinusoid.info["per_order_swing_db"].values()) < 0.5


@pytest.mark.parametrize("model", ["sinusoid", "cluster"])
def test_both_models_carry_the_same_power_per_order(runup, model):
    sig, rpm = runup
    ex = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 4.0, orders=[2.0], model=model, fade=0.0, seed=5)
    a = ex.amplitudes[2.0]
    tol = 1e-9 if model == "cluster" else 5e-3
    assert np.mean(ex.orders**2) == pytest.approx(a**2 / 2, rel=tol)


def test_cluster_lines_stay_inside_the_cluster_width(runup):
    sig, rpm = runup
    width = 6.25
    ex = stationary_excerpt(
        sig.x, FS, rpm, RPM_REF, 4.0, orders=[2.0], model="cluster", cluster_width_hz=width, fade=0.0, seed=5
    )
    spec = np.abs(np.fft.rfft(ex.orders)) ** 2
    f = np.fft.rfftfreq(ex.orders.size, 1 / FS)
    f0 = 2.0 * RPM_REF / 60.0
    inside = np.abs(f - f0) <= width / 2 + 1e-9
    assert spec[~inside].sum() < 1e-20 * spec.sum()
    # one line on every 1 / duration bin of the cluster: width * duration lines, give or take one
    assert np.count_nonzero(spec[inside] > 0) == inside.sum() in (25, 26)


# --------------------------------------------------------------------------
# length, fades, determinism
# --------------------------------------------------------------------------


def test_length_stems_and_psd_shapes(sinusoid):
    assert sinusoid.x.shape == (round(4.0 * FS),) and sinusoid.fs == FS
    assert sinusoid.orders.shape == sinusoid.residual.shape == sinusoid.x.shape
    assert np.array_equal(sinusoid.x, sinusoid.orders + sinusoid.residual)  # no hidden gain
    assert np.all(np.isfinite(sinusoid.x)) and sinusoid.x.dtype == np.float64
    assert sinusoid.info["n"] == sinusoid.x.size and "ExcerptResult" in repr(sinusoid)
    w = sinusoid.info["window"]
    assert 0 <= w["start"] < w["stop"] <= 14 * FS and w["revolutions"] > 1


def test_odd_duration_rounds_to_samples(runup):
    sig, rpm = runup
    ex = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.00003, orders=[2.0], seed=1)
    assert ex.x.size == round(1.00003 * FS)


def test_raised_cosine_fades(runup):
    sig, rpm = runup
    kw = dict(orders=[1.0, 2.0], seed=11)
    plain = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 2.0, fade=0.0, **kw)
    faded = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 2.0, fade=0.05, **kw)
    n_f = round(0.05 * FS)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(n_f) / n_f))
    assert faded.x[0] == 0.0 and faded.x[-1] == 0.0
    assert np.allclose(faded.x[:n_f], ramp * plain.x[:n_f], atol=1e-15)
    assert np.allclose(faded.x[-n_f:], ramp[::-1] * plain.x[-n_f:], atol=1e-15)
    assert np.array_equal(faded.x[n_f:-n_f], plain.x[n_f:-n_f])  # nothing else changes
    assert np.allclose(faded.orders[:n_f], ramp * plain.orders[:n_f], atol=1e-15)
    assert np.allclose(faded.residual[:n_f], ramp * plain.residual[:n_f], atol=1e-15)
    # the fade is not part of the steadiness statistics: they are taken between the fades
    assert faded.info["order_stem"]["swing_db"] == pytest.approx(plain.info["order_stem"]["swing_db"], abs=0.5)
    assert faded.info["n_fade"] == n_f


def test_determinism_and_seeds(runup):
    sig, rpm = runup
    kw = dict(orders=[1.0, 2.0])
    a = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, seed=7, **kw)
    b = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, seed=7, **kw)
    c = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, seed=8, **kw)
    assert np.array_equal(a.x, b.x) and a.info["phases"] == b.info["phases"]
    assert not np.allclose(a.x, c.x) and a.info["phases"] != c.info["phases"]
    assert a.info["seed"] == 7
    # amplitudes and PSD are data, not random
    assert a.amplitudes == c.amplitudes and np.array_equal(a.psd[1], c.psd[1])
    # seed=None draws a seed that is recorded and reproduces the excerpt
    d = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, seed=None, **kw)
    assert isinstance(d.info["seed"], int)
    e = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, seed=d.info["seed"], **kw)
    assert np.array_equal(d.x, e.x)
    # a generator is accepted; its state cannot be recorded
    g = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, seed=np.random.default_rng(3), **kw)
    assert g.info["seed"] is None and g.x.size == a.x.size
    # the phases differ between orders but are shared by the two models for one seed
    m = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, seed=7, model="cluster", **kw)
    assert m.info["phases"] == a.info["phases"]


def test_info_is_json_serialisable_and_complete(sinusoid):
    text = json.dumps(sinusoid.info, allow_nan=False)
    info = json.loads(text)
    for key in ("model", "extractor", "rpm_ref", "duration", "orders", "frequencies_hz", "tolerance",
                "window_revs", "cluster_width_hz", "fade", "seed", "phases", "window", "psd",
                "order_stem", "mix", "per_order_swing_db"):
        assert key in info, key
    assert info["frequencies_hz"] == pytest.approx([o * RPM_REF / 60 for o in ORDERS])
    assert {"swing_db", "crest_factor_db"} <= set(info["mix"])


# --------------------------------------------------------------------------
# operating window and options
# --------------------------------------------------------------------------


def test_window_is_within_the_tolerance(sinusoid, runup):
    _, rpm = runup
    w = sinusoid.info["window"]
    inside = rpm[w["start"] : w["stop"]]
    assert np.all(np.abs(inside / RPM_REF - 1) <= 0.03 + 1e-12)
    assert abs(rpm[w["start"] - 1] / RPM_REF - 1) > 0.03 and abs(rpm[w["stop"]] / RPM_REF - 1) > 0.03


def test_the_longest_run_near_the_reference_speed_is_used():
    t = np.arange(round(18 * FS)) / FS
    rpm = rpm_profiles(t, "piecewise", knots=([0, 6, 18], [1500, 3500, 1500]))  # fast up, slow down
    sig = synth_runup(FS, rpm, SPECS, noise_rms=NOISE_RMS, seed=4)
    ex = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, orders=[2.0], seed=1)
    w = ex.info["window"]
    assert w["t_start"] > 6.0  # the slow pass, which stays in the band longer
    fast_pass = 2 * 0.03 * RPM_REF / (2000 / 6) * FS  # samples of the fast pass: 150 rpm at 333 rpm/s
    assert w["n_samples"] == pytest.approx(2 * fast_pass, rel=0.01)  # the slow pass (167 rpm/s) takes twice as long


def test_error_when_the_reference_speed_is_never_reached(runup):
    sig, rpm = runup
    with pytest.raises(ValueError, match="never comes within"):
        stationary_excerpt(sig.x, FS, rpm, 5000.0, 1.0, orders=[2.0])
    with pytest.raises(ValueError, match="never comes within"):
        stationary_excerpt(sig.x, FS, rpm, 1000.0, 1.0, orders=[2.0], tolerance=0.1)
    with pytest.raises(ValueError, match="increase tolerance"):  # reached, but only for an instant
        stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, orders=[2.0], tolerance=1e-5)


def test_steady_recording_is_not_warped():
    n = round(6 * FS)
    rpm = np.full(n, RPM_REF)
    sig = synth_runup(FS, rpm, SPECS, noise_rms=NOISE_RMS, seed=2)
    ex = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 2.0, orders=ORDERS, seed=1)
    assert ex.info["psd"]["stationarized"] is False
    for order, a in ex.amplitudes.items():
        assert abs(_db(a, TRUTH[order])) < 0.1
    assert np.max(np.abs(_octave_levels(*ex.psd, NOISE_PSD))) < 1.0


def test_vold_kalman_extractor():
    fs = 4000.0
    t = np.arange(round(6 * fs)) / fs
    rpm = rpm_profiles(t, "linear", start=1500, stop=3500)
    sig = synth_runup(fs, rpm, SPECS[:2], noise_rms=NOISE_RMS, seed=4)
    ex = stationary_excerpt(
        sig.x, fs, rpm, RPM_REF, 1.0, orders=[1.0, 2.0], extractor="vold-kalman", bandwidth=4.0, seed=1
    )
    for order, a in ex.amplitudes.items():
        assert abs(_db(a, TRUTH[order])) < 0.3, order
    assert ex.info["extractor"] == "vold-kalman" and ex.info["extractor_options"] == {"bandwidth": 4.0}


def test_orders_above_nyquist_are_muted(runup):
    sig, rpm = runup
    ex = stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, orders=[2.0, 200.0], seed=1)  # 8333 Hz
    assert ex.info["muted_orders"] == [200.0] and ex.amplitudes[200.0] == 0.0
    assert ex.info["per_order_swing_db"]["200"] is None
    assert np.max(np.abs(ex.orders)) <= ex.amplitudes[2.0] + 1e-12  # a single sinusoid: no alias tone
    assert abs(_db(ex.amplitudes[2.0], TRUTH[2.0])) < 0.3


# --------------------------------------------------------------------------
# argument checks
# --------------------------------------------------------------------------


def test_invalid_arguments_raise(runup):
    sig, rpm = runup
    ok = dict(orders=[2.0])
    with pytest.raises(ValueError, match="mono"):
        stationary_excerpt(np.column_stack([sig.x, sig.x]), FS, rpm, RPM_REF, 1.0, **ok)
    with pytest.raises(ValueError, match="shape"):
        stationary_excerpt(sig.x, FS, rpm[:-1], RPM_REF, 1.0, **ok)
    with pytest.raises(ValueError, match="model"):
        stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, model="gauss", **ok)
    with pytest.raises(ValueError, match="extractor"):
        stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, extractor="fft", **ok)
    with pytest.raises(ValueError, match="distinct"):
        stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, orders=[2.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, orders=[2.0, -1.0])
    with pytest.raises(ValueError, match="tolerance"):
        stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, tolerance=1.5, **ok)
    with pytest.raises(ValueError, match="duration"):
        stationary_excerpt(sig.x, FS, rpm, RPM_REF, 0.0, **ok)
    with pytest.raises(ValueError, match="rpm_ref"):
        stationary_excerpt(sig.x, FS, rpm, -1.0, 1.0, **ok)
    with pytest.raises(ValueError, match="fade"):
        stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, fade=-0.1, **ok)
    with pytest.raises(ValueError, match="fades"):
        stationary_excerpt(sig.x, FS, rpm, RPM_REF, 0.05, fade=0.03, **ok)
    with pytest.raises(ValueError, match="non-finite"):
        stationary_excerpt(np.where(np.arange(sig.x.size) == 3, np.nan, sig.x), FS, rpm, RPM_REF, 1.0, **ok)
    with pytest.raises(ValueError, match="non-finite"):
        stationary_excerpt(sig.x, FS, np.where(np.arange(rpm.size) == 3, np.nan, rpm), RPM_REF, 1.0, **ok)
    with pytest.raises(TypeError):
        stationary_excerpt(sig.x, FS, rpm, RPM_REF, 1.0, seed="a", **ok)
