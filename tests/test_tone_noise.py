"""Tests of rota.synth.synth_tone_noise: levels, ratios, spectra, determinism, validation."""

import json

import numpy as np
import pytest
from scipy import signal

from rota.synth import ToneNoiseSignal, synth_tone_noise

FS = 48000.0
TWO_PI = 2 * np.pi


def _rms(v):
    return float(np.sqrt(np.mean(np.asarray(v) ** 2)))


def _power(v):
    return float(np.mean(np.asarray(v) ** 2))


def _spec(v):
    """Complex rfft and frequency axis (Hz) of a stem."""
    n = v.size
    return np.fft.rfft(v), np.arange(n // 2 + 1) * (FS / n)


# --------------------------------------------------------------------------
# level and ratio
# --------------------------------------------------------------------------

_CASES = [
    # (tone kwargs, noise kwargs)
    ({"tone_freq": 1000.0}, {}),
    ({"tone_freq": 1234.5}, {}),  # not on a DFT bin
    ({"tone_freq": 500.0, "harmonics": (1.0, 0.5, 0.25)}, {"noise": "hump", "hump_fc": 800.0}),
    ({"tone_freq": 1000.0, "tone_bandwidth_oct": 1 / 3}, {}),
    ({"tone_freq": 700.0, "harmonics": (1.0, 0.0, 0.6), "tone_bandwidth_oct": 0.25},
     {"noise": "hump", "hump_fc": 300.0, "hump_sigma_oct": 1.2}),
]


@pytest.mark.parametrize("level_rms", [0.02, 1.0, 37.5])
@pytest.mark.parametrize("tnr_db", [-20.0, 0.0, 6.0, 25.0])
@pytest.mark.parametrize(("tone_kw", "noise_kw"), _CASES)
def test_total_rms_and_stem_power_ratio_are_exact(tone_kw, noise_kw, tnr_db, level_rms):
    s = synth_tone_noise(FS, 1.0, level_rms, tnr_db, seed=5, **tone_kw, **noise_kw)
    r = 10.0 ** (tnr_db / 10.0)
    assert _rms(s.x) == pytest.approx(level_rms, rel=1e-12)
    assert _power(s.tone) == pytest.approx(level_rms**2 * r / (1 + r), rel=1e-12)
    assert _power(s.noise) == pytest.approx(level_rms**2 / (1 + r), rel=1e-12)
    assert 10 * np.log10(_power(s.tone) / _power(s.noise)) == pytest.approx(tnr_db, abs=1e-9)
    # the stems are exactly orthogonal, so the cross term of x vanishes
    assert abs(s.tone @ s.noise) <= 1e-12 * np.linalg.norm(s.tone) * np.linalg.norm(s.noise)


def test_x_is_the_sum_of_the_stems_bit_for_bit_and_the_container_is_consistent():
    s = synth_tone_noise(FS, 0.5, 0.02, 3.0, tone_freq=1000.0, seed=1)
    assert isinstance(s, ToneNoiseSignal)
    assert np.array_equal(s.x, s.tone + s.noise)
    assert s.fs == FS and s.x.shape == s.tone.shape == s.noise.shape == s.t.shape == (24000,)
    assert np.array_equal(s.t, np.arange(24000) / FS)
    assert synth_tone_noise(44100.0, 0.33, 1.0, 0.0, tone_freq=500.0).x.size == round(0.33 * 44100.0)


def test_60_db_spl_example_and_the_anchor_stimuli():
    # 60 dB SPL re 20 uPa is 0.02 Pa RMS
    s = synth_tone_noise(FS, 2.0, 0.02, 0.0, tone_freq=1000.0, seed=1)
    assert 20 * np.log10(_rms(s.x) / 20e-6) == pytest.approx(60.0, abs=1e-9)
    # equal tone and noise power: each 3.01 dB below the total
    assert 20 * np.log10(_rms(s.tone) / 20e-6) == pytest.approx(60.0 - 10 * np.log10(2), abs=1e-9)
    tone = synth_tone_noise(FS, 2.0, 0.02, np.inf, tone_freq=1000.0, seed=1)
    assert np.all(tone.noise == 0.0) and _rms(tone.x) == pytest.approx(0.02, rel=1e-12)
    white = synth_tone_noise(FS, 2.0, 0.02, -np.inf, seed=1)
    assert np.all(white.tone == 0.0) and _rms(white.x) == pytest.approx(0.02, rel=1e-12)
    mute = synth_tone_noise(FS, 0.1, 0.0, 0.0, tone_freq=1000.0, seed=1)
    assert np.all(mute.x == 0.0)
    # tone_freq given but -inf: the tone is switched off
    off = synth_tone_noise(FS, 0.5, 0.02, -np.inf, tone_freq=1000.0, seed=1)
    assert np.all(off.tone == 0.0) and _rms(off.noise) == pytest.approx(0.02, rel=1e-12)


def test_the_ratio_is_broadband_not_the_critical_band_tnr():
    # white noise of power Pn spreads evenly over 0 .. fs/2, so the noise in a band of B Hz
    # round the tone is Pn * B / (fs/2); the critical-band ratio is tnr_db + 10 log10((fs/2) / B)
    tnr_db, f0, band = 0.0, 1000.0, 100.0
    s = synth_tone_noise(FS, 4.0, 0.02, tnr_db, tone_freq=f0, seed=2)
    X, f = _spec(s.noise)
    in_band = (f >= f0 - band / 2) & (f < f0 + band / 2)
    p_band = _power(s.noise) * np.sum(np.abs(X[in_band]) ** 2) / np.sum(np.abs(X[1:-1]) ** 2)
    tnr_band = 10 * np.log10(_power(s.tone) / p_band)
    assert tnr_band == pytest.approx(tnr_db + 10 * np.log10((FS / 2) / band), abs=0.02)
    assert tnr_band - tnr_db > 15.0  # about 24 dB: far from the broadband figure
    assert "not ECMA-74" in s.config["tnr_definition"]


# --------------------------------------------------------------------------
# tone: pure, harmonic complex, narrow-band
# --------------------------------------------------------------------------


def test_pure_tone_has_one_line_at_the_requested_frequency_with_the_right_amplitude():
    s = synth_tone_noise(FS, 1.0, 0.02, np.inf, tone_freq=1000.0, seed=3)  # whole periods, bin-centred
    X, f = _spec(s.tone)
    amp = 2.0 * np.abs(X) / s.tone.size
    assert f[np.argmax(amp)] == pytest.approx(1000.0)
    assert amp.max() == pytest.approx(np.sqrt(2.0) * 0.02, rel=1e-12)  # peak = sqrt(2) * RMS
    other = np.delete(amp, np.argmax(amp))
    assert other.max() < 1e-12


def test_harmonic_weights_set_the_linear_amplitude_ratios_of_the_partials():
    w = (1.0, 0.5, 0.0, 0.25)
    s = synth_tone_noise(FS, 1.0, 1.0, np.inf, tone_freq=500.0, harmonics=w, seed=4)
    X, _f = _spec(s.tone)
    amp = 2.0 * np.abs(X) / s.tone.size
    partial = np.array([amp[k * 500] for k in (1, 2, 3, 4, 5)])
    expected = np.sqrt(2.0) * np.array([*w, 0.0]) / np.linalg.norm(w)  # peak amplitudes at RMS 1
    assert np.allclose(partial, expected, rtol=0, atol=1e-12)
    # nothing between the partials
    mask = np.ones(amp.size, bool)
    for k in (1, 2, 4):
        mask[k * 500] = False
    assert amp[mask].max() < 1e-12
    # phases are random: not all partials start in phase
    ph = np.angle(X[[500, 1000, 2000]])
    assert np.ptp(np.cos(ph)) > 0.05


def test_config_holds_enough_to_rebuild_the_tone():
    w = (1.0, 0.5, 0.25)
    s = synth_tone_noise(FS, 1.0, 0.02, 4.0, tone_freq=500.0, harmonics=w, seed=9)
    cfg = s.config
    amp = np.sqrt(2.0 * cfg["tone_power"]) * np.array(w) / np.linalg.norm(w)
    rebuilt = sum(
        a * np.cos(TWO_PI * (k + 1) * 500.0 * s.t + ph)
        for k, (a, ph) in enumerate(zip(amp, cfg["tone_phases"], strict=True))
    )
    assert np.allclose(rebuilt, s.tone, rtol=0, atol=1e-12)
    assert cfg["tone_power"] == pytest.approx(_power(s.tone), rel=1e-12)
    assert cfg["noise_power"] == pytest.approx(_power(s.noise), rel=1e-12)


def test_narrow_band_noise_tone_fills_its_band_flat_and_nothing_else():
    f0, bw = 1000.0, 1.0 / 3.0
    s = synth_tone_noise(FS, 10.0, 0.02, np.inf, tone_freq=f0, tone_bandwidth_oct=bw, seed=1)
    X, f = _spec(s.tone)
    p = np.abs(X) ** 2
    lo, hi = f0 * 2 ** (-bw / 2), f0 * 2 ** (bw / 2)
    band = (f >= lo) & (f <= hi)
    assert band.sum() > 2000
    assert p[~band].sum() < 1e-20 * p.sum()  # no energy outside the band
    assert p[band].std() / p[band].mean() < 1e-9  # flat inside
    assert _rms(s.tone) == pytest.approx(0.02, rel=1e-12)
    # it is a noise, not a sinusoid: Rayleigh-like envelope and a crest factor near 4-5, not 1.41
    assert np.abs(s.tone).max() / _rms(s.tone) > 3.0
    # band powers follow the squared weights; partial 2 is centred on 2 f0
    w = (1.0, 0.5)
    t2 = synth_tone_noise(FS, 10.0, 1.0, np.inf, tone_freq=f0, harmonics=w,
                          tone_bandwidth_oct=bw, seed=1)
    X2, f2 = _spec(t2.tone)
    p2 = np.abs(X2) ** 2
    b1 = (f2 >= lo) & (f2 <= hi)
    b2 = (f2 >= 2 * lo) & (f2 <= 2 * hi)
    assert p2[b2].sum() / p2[b1].sum() == pytest.approx(0.25, rel=1e-9)
    assert (p2[b1].sum() + p2[b2].sum()) / p2.sum() == pytest.approx(1.0, abs=1e-12)


def test_overlapping_bands_add_their_power():
    # a very wide band makes the partials' bands overlap; the powers still follow the weights
    s = synth_tone_noise(FS, 4.0, 1.0, np.inf, tone_freq=1000.0, harmonics=(1.0, 1.0),
                         tone_bandwidth_oct=1.5, seed=2)
    assert _rms(s.tone) == pytest.approx(1.0, rel=1e-12)
    X, f = _spec(s.tone)
    assert (np.abs(X[(f < 1000 * 2**-0.75) | (f > 2000 * 2**0.75)]) ** 2).sum() < 1e-20


# --------------------------------------------------------------------------
# noise: white and hump
# --------------------------------------------------------------------------


def test_white_noise_spectrum_is_flat_and_has_no_dc_or_nyquist():
    s = synth_tone_noise(FS, 2.0, 0.02, -np.inf, seed=1)
    X, _f = _spec(s.noise)
    p = np.abs(X) ** 2
    assert p[0] < 1e-20 and p[-1] < 1e-20
    assert p[1:-1].std() / p[1:-1].mean() < 1e-9
    assert abs(np.mean(s.noise)) < 1e-12
    # near-Gaussian amplitudes from the random phases: kurtosis 3, no heavy tails
    z = s.noise / _rms(s.noise)
    assert np.mean(z**4) == pytest.approx(3.0, abs=0.1)
    # Welch of a longer record agrees: flat within a fraction of a dB
    _f, pw = signal.welch(synth_tone_noise(FS, 10.0, 0.02, -np.inf, seed=1).noise, FS, nperseg=2048)
    assert 10 * np.log10(pw[10:-10].max() / pw[10:-10].min()) < 2.0


def test_hump_periodogram_follows_the_gaussian_in_log_frequency_exactly():
    fc, sigma = 600.0, 0.7
    s = synth_tone_noise(FS, 4.0, 0.02, -np.inf, noise="hump", hump_fc=fc, hump_sigma_oct=sigma, seed=1)
    X, f = _spec(s.noise)
    p = np.abs(X) ** 2
    target = np.exp(-0.5 * (np.log2(np.maximum(f, 1e-12) / fc) / sigma) ** 2)
    ok = target > 1e-8
    ok[0] = False
    ratio = p[ok] / target[ok]
    assert ratio.std() / ratio.mean() < 1e-9  # no chi-squared scatter: the phases alone are random
    assert p[0] < 1e-20 and p[-1] < 1e-20
    assert _rms(s.noise) == pytest.approx(0.02, rel=1e-12)
    # the density peaks (per Hz) at fc to within a bin
    assert f[np.argmax(p)] == pytest.approx(fc, abs=f[1])


@pytest.mark.parametrize("sigma", [0.5, 0.7, 1.0])
def test_hump_welch_peak_and_minus_3_db_width_follow_sigma(sigma):
    fc = 500.0
    s = synth_tone_noise(16000.0, 20.0, 0.02, -np.inf, noise="hump", hump_fc=fc,
                         hump_sigma_oct=sigma, seed=3)
    f, p = signal.welch(s.noise, 16000.0, nperseg=2048, window="hann")
    lf, pn = np.log2(f[1:]), p[1:]
    # smooth over +-5 % of the frequency to suppress the estimate's scatter, then locate the
    # half-power points on both sides of the peak in log2 frequency
    ps = np.array([pn[np.abs(lf - c) < 0.07].mean() for c in lf])
    j = int(np.argmax(ps))
    pk = ps[j]
    lo = np.interp(0.5 * pk, ps[: j + 1], lf[: j + 1])
    hi = np.interp(0.5 * pk, ps[j:][::-1], lf[j:][::-1])
    width = hi - lo
    assert width == pytest.approx(2 * np.sqrt(2 * np.log(2)) * sigma, rel=0.06)  # FWHM = 2.355 sigma
    assert 2 ** (0.5 * (lo + hi)) == pytest.approx(fc, rel=0.07)  # centred on fc (Welch peak)
    # the wings of the Gaussian: -8.7 dB (exp(-2)) at fc * 2**(+-2 sigma), with the same smoothing
    for sign in (-1, 1):
        at = np.interp(np.log2(fc) + sign * 2 * sigma, lf, ps)
        assert 10 * np.log10(at / pk) == pytest.approx(10 * np.log10(np.exp(-2)), abs=1.0)


def test_hump_power_is_all_in_the_hump_when_the_tone_is_off():
    s = synth_tone_noise(FS, 2.0, 1.0, -np.inf, noise="hump", hump_fc=1000.0, hump_sigma_oct=0.5, seed=1)
    X, f = _spec(s.noise)
    p = np.abs(X) ** 2
    near = (f > 1000 / 2**3) & (f < 1000 * 2**3)  # +-6 sigma
    assert p[near].sum() / p.sum() > 1 - 1e-8


# --------------------------------------------------------------------------
# determinism, streams, configuration
# --------------------------------------------------------------------------


def test_same_seed_gives_identical_output_and_other_seeds_differ():
    kw = {"tone_freq": 800.0, "harmonics": (1.0, 0.4), "noise": "hump", "hump_fc": 400.0}
    a = synth_tone_noise(FS, 0.5, 0.02, 2.0, seed=11, **kw)
    b = synth_tone_noise(FS, 0.5, 0.02, 2.0, seed=11, **kw)
    c = synth_tone_noise(FS, 0.5, 0.02, 2.0, seed=12, **kw)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.tone, b.tone) and np.array_equal(a.noise, b.noise)
    assert a.config == b.config
    assert not np.array_equal(a.tone, c.tone) and not np.array_equal(a.noise, c.noise)
    # a Generator is accepted; the same Generator state gives the same stimulus
    g1 = synth_tone_noise(FS, 0.5, 0.02, 2.0, seed=np.random.default_rng(5), **kw)
    g2 = synth_tone_noise(FS, 0.5, 0.02, 2.0, seed=np.random.default_rng(5), **kw)
    assert np.array_equal(g1.x, g2.x)
    assert g1.config["seed"] is None and a.config["seed"] == 11
    # no seed: not reproducible
    n1 = synth_tone_noise(FS, 0.5, 0.02, 2.0, **kw)
    n2 = synth_tone_noise(FS, 0.5, 0.02, 2.0, **kw)
    assert not np.array_equal(n1.noise, n2.noise)


def test_changing_tnr_or_level_only_rescales_the_stems_for_a_given_seed():
    kw = {"tone_freq": 800.0, "harmonics": (1.0, 0.4), "noise": "hump", "hump_fc": 400.0, "seed": 7}
    a = synth_tone_noise(FS, 0.5, 0.02, 0.0, **kw)
    b = synth_tone_noise(FS, 0.5, 0.05, 10.0, **kw)
    ra, rb = 1.0, 10.0
    k_tone = (0.05 / 0.02) * np.sqrt((rb / (1 + rb)) / (ra / (1 + ra)))
    k_noise = (0.05 / 0.02) * np.sqrt((1 / (1 + rb)) / (1 / (1 + ra)))
    assert np.allclose(b.tone, k_tone * a.tone, rtol=1e-10, atol=1e-14)
    assert np.allclose(b.noise, k_noise * a.noise, rtol=1e-10, atol=1e-14)


def test_tone_and_noise_use_independent_random_streams():
    # switching the tone off or changing its shape leaves the noise phases untouched
    a = synth_tone_noise(FS, 0.5, 1.0, -np.inf, seed=3)
    b = synth_tone_noise(FS, 0.5, 1.0, -np.inf, tone_freq=900.0, seed=3)
    assert np.array_equal(a.noise, b.noise)


def test_config_is_json_serialisable_and_records_the_inputs():
    for tnr in (3.5, np.inf, -np.inf):
        kw = {} if tnr == -np.inf else {"tone_freq": 1000.0}
        s = synth_tone_noise(FS, 0.25, 0.02, tnr, noise="hump", hump_fc=500.0, seed=np.int64(4), **kw)
        text = json.dumps(s.config, allow_nan=False)  # strict JSON: no Infinity or NaN
        back = json.loads(text)
        assert back == s.config
        assert back["seed"] == 4 and back["fs"] == FS and back["n"] == 12000
        assert back["level_rms"] == 0.02 and back["noise"] == "hump" and back["hump_fc"] == 500.0
        assert back["tnr_db"] == (tnr if np.isfinite(tnr) else ("inf" if tnr > 0 else "-inf"))
        assert isinstance(back["rota_version"], str)
    s = synth_tone_noise(FS, 0.25, 0.02, 0.0, tone_freq=1000.0, seed=1)
    assert s.config["hump_fc"] is None and s.config["hump_sigma_oct"] is None
    assert s.config["tone_freq"] == 1000.0 and s.config["harmonics"] == [1.0]


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"fs": 0.0}, "fs"),
        ({"fs": -1.0}, "fs"),
        ({"fs": np.nan}, "fs"),
        ({"duration": 0.0}, "duration"),
        ({"duration": -1.0}, "duration"),
        ({"duration": np.inf}, "duration"),
        ({"duration": 1e-4}, "16 samples"),
        ({"level_rms": -0.1}, "level_rms"),
        ({"level_rms": np.nan}, "level_rms"),
        ({"level_rms": np.inf}, "level_rms"),
        ({"tnr_db": np.nan}, "tnr_db"),
        ({"tone_freq": None}, "tone_freq"),  # needed unless the tone is off
        ({"tone_freq": 0.0}, "tone_freq"),
        ({"tone_freq": -5.0}, "tone_freq"),
        ({"tone_freq": FS / 2}, "tone_freq"),
        ({"tone_freq": np.nan}, "tone_freq"),
        ({"harmonics": ()}, "harmonics"),
        ({"harmonics": (0.0, 0.0)}, "harmonics"),
        ({"harmonics": (1.0, -0.5)}, "harmonics"),
        ({"harmonics": (1.0, np.nan)}, "harmonics"),
        ({"harmonics": [[1.0, 0.5]]}, "harmonics"),
        ({"tone_freq": 9000.0, "harmonics": (1.0, 0.5, 0.2, 0.1, 0.1)}, "fs/2"),  # partial 3 at 27 kHz
        ({"tone_bandwidth_oct": -0.5}, "tone_bandwidth_oct"),
        ({"tone_bandwidth_oct": np.nan}, "tone_bandwidth_oct"),
        ({"tone_freq": 23000.0, "tone_bandwidth_oct": 0.2}, "fs/2"),  # upper band edge above Nyquist
        ({"noise": "pink"}, "noise"),
        ({"noise": "hump"}, "hump_fc"),
        ({"noise": "hump", "hump_fc": 0.0}, "hump_fc"),
        ({"noise": "hump", "hump_fc": FS / 2}, "hump_fc"),
        ({"noise": "hump", "hump_fc": 500.0, "hump_sigma_oct": 0.0}, "hump_sigma_oct"),
        ({"noise": "hump", "hump_fc": 500.0, "hump_sigma_oct": -1.0}, "hump_sigma_oct"),
        # 0.25 s gives 4 Hz bins: 501 Hz is between bins and a 1e-9 octave hump holds none
        ({"noise": "hump", "hump_fc": 501.0, "hump_sigma_oct": 1e-9}, "resolution"),
        ({"hump_fc": 500.0}, "hump_fc"),  # only meaningful for noise='hump'
    ],
)
def test_invalid_arguments_raise(kw, match):
    args = {"fs": FS, "duration": 0.25, "level_rms": 0.02, "tnr_db": 0.0, "tone_freq": 1000.0, **kw}
    with pytest.raises(ValueError, match=match):
        synth_tone_noise(args.pop("fs"), args.pop("duration"), args.pop("level_rms"),
                         args.pop("tnr_db"), **args)


def test_a_zero_weight_partial_above_nyquist_is_allowed_but_a_weighted_one_is_not():
    ok = synth_tone_noise(16000.0, 0.25, 1.0, 0.0, tone_freq=3000.0, harmonics=(1.0, 0.5, 0.0, 0.0), seed=1)
    assert _rms(ok.x) == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(ValueError, match="fs/2"):
        synth_tone_noise(16000.0, 0.25, 1.0, 0.0, tone_freq=3000.0, harmonics=(1.0, 0.5, 0.0, 0.1))


def test_a_band_with_fewer_than_two_bins_is_rejected():
    # 0.05 s at 48 kHz: 20 Hz resolution; a 1/50-octave band at 1 kHz is 14 Hz wide
    with pytest.raises(ValueError, match="fewer than 2"):
        synth_tone_noise(FS, 0.05, 1.0, 0.0, tone_freq=1000.0, tone_bandwidth_oct=0.02)
