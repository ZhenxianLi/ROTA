"""Tests of rota._broadband.synthesize_broadband (ROADMAP case T11).

Ground truth is the target PSD itself (Welch estimate of the output against
it), the analytic power sum of the overlap-add window (a power-complementary
window gives a constant long-term power, an amplitude-COLA window pumps by
3 dB), and white Gaussian noise from rota.synth as the reference for the
statistical spread of block levels and of the modulation detector.
"""

import numpy as np
import pytest
from scipy import signal, stats

from rota import _broadband
from rota._broadband import _synthesize, synthesize_broadband
from rota.metrics import envelope_stats, modulation_peak
from rota.synth import synth_runup

FS = 16000.0
N = 160000  # 10 s
GRAIN = 0.05
HOP = 400  # samples; grain 800 -> grain rate FS / HOP = 40 Hz
RATE = FS / HOP
F = np.linspace(0.0, FS / 2.0, 2049)


def _shape(kind):
    if kind == "flat":
        return np.full_like(F, 1e-3)
    if kind == "sloped":  # -3 dB per octave, 20 Hz and up
        return 1e-3 * 1000.0 / np.maximum(F, 20.0)
    if kind == "lowpass":  # 48 dB of dynamic range
        return 1e-2 / (1.0 + (F / 500.0) ** 4)
    if kind == "bump":
        return 1e-5 + 4e-3 * np.exp(-0.5 * ((F - 2000.0) / 300.0) ** 2)
    raise KeyError(kind)


def _octave_errors_db(y, psd):
    """Welch PSD of y over the target, averaged in octave bands from 125 Hz up, in dB."""
    f, p = signal.welch(y, FS, nperseg=4096)
    target = np.interp(f, F, psd)
    out = []
    for lo in (125, 250, 500, 1000, 2000, 4000):
        m = (f >= lo) & (f < 2 * lo)
        out.append(10.0 * np.log10(p[m].mean() / target[m].mean()))
    return np.array(out)


def _block_levels_db(y, seconds):
    m = int(seconds * FS)
    blocks = y[: (y.size // m) * m].reshape(-1, m)
    return 10.0 * np.log10(np.mean(blocks**2, axis=1))


def _fold_ripple_db(y, hop=HOP, n_phase=20):
    """Max/min (dB) of the mean power at n_phase positions within the hop period."""
    k = y.size // hop
    prof = (y[: k * hop] ** 2).reshape(k, n_phase, hop // n_phase).mean(axis=(0, 2))
    return 10.0 * np.log10(prof.max() / prof.min())


def _trapz(y, x):
    """Trapezoid rule (np.trapz was renamed np.trapezoid in NumPy 2)."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _white_gaussian(rms, n=N, seed=0):
    return synth_runup(FS, np.full(n, 1000.0), [], noise_rms=rms, seed=seed).x


# ==========================================================================
# Spectrum and level (T11)
# ==========================================================================


@pytest.mark.parametrize("kind", ["flat", "sloped", "lowpass", "bump"])
def test_welch_psd_of_the_output_matches_the_target(kind):
    psd = _shape(kind)
    for seed in (1, 2):
        y = synthesize_broadband(psd, F, FS, N, seed=seed)
        errors = _octave_errors_db(y, psd)
        assert np.max(np.abs(errors)) < 0.5, errors  # measured below 0.16 dB


def test_total_power_is_the_integral_of_the_psd():
    # flat: integral = psd * fs / 2 exactly (DC and Nyquist bins count half)
    y = synthesize_broadband(_shape("flat"), F, FS, N, highpass=0.0, seed=3)
    assert 10 * np.log10(np.mean(y**2) / (1e-3 * FS / 2)) == pytest.approx(0.0, abs=0.1)
    # sloped: the sum of psd(f_k) df over the retained grain bins
    psd = _shape("sloped")
    length = 2 * round(GRAIN * FS / 2)
    fk = np.arange(length // 2 + 1) * FS / length
    w = np.where(fk >= 20.0, np.interp(fk, F, psd), 0.0)
    w[-1] *= 0.5
    expected = np.sum(w) * FS / length
    y = synthesize_broadband(psd, F, FS, N, seed=3)
    assert 10 * np.log10(np.mean(y**2) / expected) == pytest.approx(0.0, abs=0.3)


def test_output_is_linear_in_the_psd_and_has_the_unit_of_its_square_root():
    psd = _shape("sloped")
    a = synthesize_broadband(psd, F, FS, 20000, seed=4)
    b = synthesize_broadband(9.0 * psd, F, FS, 20000, seed=4)
    assert b == pytest.approx(3.0 * a, rel=1e-9, abs=1e-11)  # exact up to round-off


def test_block_power_varies_only_statistically_like_gaussian_noise():
    y = synthesize_broadband(_shape("flat"), F, FS, N, seed=5)
    gauss = _white_gaussian(np.std(y))
    s_out = np.std(_block_levels_db(y, 0.5))
    s_ref = np.std(_block_levels_db(gauss, 0.5))
    assert s_out < 1.5 * s_ref
    assert s_out < 0.15  # dB; measured about 0.05
    # the long-term level of 0.5 s blocks does not drift
    assert np.ptp(_block_levels_db(y, 0.5)) < 0.4


def test_no_power_pumping_at_the_grain_rate_in_the_time_domain():
    # Mean power at 20 positions within the hop period: flat for the
    # power-complementary window, 3 dB peak-to-peak for the amplitude-COLA Hann.
    for kind in ("flat", "sloped"):
        psd = _shape(kind)
        good = synthesize_broadband(psd, F, FS, N, seed=6)
        assert _fold_ripple_db(good) < 1.2  # measured 0.2 - 0.85 over 30 seeds (statistical)
        hann = _synthesize(psd, F, FS, N, control=None, control_axis=None,
                           grain=GRAIN, highpass=20.0, seed=6, window="hann")
        # sin^4 + cos^4 = 1 - sin^2(2 theta) / 2: 10 log10(2) = 3.01 dB
        assert 2.5 < _fold_ripple_db(hann) < 4.0


def test_output_is_close_to_gaussian():
    y = synthesize_broadband(_shape("flat"), F, FS, N, seed=7)
    assert abs(stats.kurtosis(y)) < 0.15  # excess kurtosis of a Gaussian is 0
    assert abs(stats.skew(y)) < 0.05
    env = envelope_stats(y, FS, band=(1000.0, 4000.0))
    # Rayleigh envelope: 20 log10 of the ratio of the 95th to the 5th percentile
    expected = 20 * np.log10(np.sqrt(-2 * np.log(0.05)) / np.sqrt(-2 * np.log(0.95)))
    assert env["swing_db"] == pytest.approx(expected, abs=0.5)
    assert 12.0 < env["crest_factor_db"] < 16.0  # peak of 160k Gaussian samples: 4.3 - 5.5 sigma


# ==========================================================================
# Pumping detector on the synthesis (T11, ROADMAP section 6 item 11)
# ==========================================================================


@pytest.mark.parametrize("kind", ["flat", "sloped"])
def test_modulation_peak_at_the_grain_rate_is_noise_level(kind):
    psd = _shape(kind)
    peaks = [
        modulation_peak(synthesize_broadband(psd, F, FS, N, seed=s), FS, RATE, half_width=2.0)
        for s in range(8)
    ]
    # The detector is a statistic: about 0 dB +- 0.8 dB (1 sigma) for 10 s and
    # half_width 2 Hz, as for white Gaussian noise. The ROADMAP tolerance of 1 dB
    # holds for its mean; a single realisation is within 3 sigma.
    assert np.mean(peaks) < 1.0, peaks
    assert max(peaks) < 3.0, peaks


def test_modulation_peak_meets_one_db_on_a_long_record():
    # 40 s and half_width 2 Hz: the statistical spread is about 0.4 dB, so a
    # single realisation meets the ROADMAP T11 tolerance of 1 dB.
    n = 40 * int(FS)
    for kind, seed in (("flat", 11), ("flat", 12), ("sloped", 11)):
        y = synthesize_broadband(_shape(kind), F, FS, n, seed=seed)
        assert modulation_peak(y, FS, RATE, half_width=2.0) < 1.0


@pytest.mark.parametrize("kind", ["flat", "sloped"])
def test_amplitude_cola_window_shows_a_clear_pumping_peak(kind):
    psd = _shape(kind)
    peaks = [
        modulation_peak(
            _synthesize(psd, F, FS, N, control=None, control_axis=None, grain=GRAIN,
                        highpass=20.0, seed=s, window="hann"),
            FS, RATE, half_width=2.0,
        )
        for s in range(4)
    ]
    assert min(peaks) > 6.0, peaks  # measured 13 - 15 dB
    good = modulation_peak(synthesize_broadband(psd, F, FS, N, seed=0), FS, RATE, half_width=2.0)
    assert min(peaks) > good + 6.0


# ==========================================================================
# High-pass
# ==========================================================================


def test_highpass_removes_sub_audio_energy():
    f = np.linspace(0.0, FS / 2.0, 16001)
    psd = np.full_like(f, 1e-3)
    psd[f < 18.0] = 10.0  # 40 dB more than the rest, below the cut-off
    kw = dict(grain=0.2, seed=1)  # 5 Hz bin spacing: resolves the cut-off
    raw = synthesize_broadband(psd, f, FS, N, highpass=0.0, **kw)
    cut = synthesize_broadband(psd, f, FS, N, highpass=20.0, **kw)

    def low_power(y):
        ff, p = signal.welch(y, FS, nperseg=32768)
        return _trapz(p[ff < 12.0], ff[ff < 12.0])

    assert 10 * np.log10(low_power(raw) / low_power(cut)) > 50.0  # measured 69 dB
    assert abs(np.mean(cut)) < 0.01 * np.std(cut)  # grains have no offset
    # the pass band is untouched
    ff, p = signal.welch(cut, FS, nperseg=4096)
    band = (ff > 100) & (ff < 1000)
    assert 10 * np.log10(p[band].mean() / 1e-3) == pytest.approx(0.0, abs=0.3)


def test_psd_outside_the_given_freqs_is_held_and_zero_end_points_band_limit():
    # held beyond both ends: the 1-2 kHz definition reaches Nyquist
    held = synthesize_broadband([1e-3, 1e-3], [1000.0, 2000.0], FS, N, seed=1)
    f, p = signal.welch(held, FS, nperseg=4096)
    assert 10 * np.log10(p[(f > 4000) & (f < 7500)].mean() / 1e-3) == pytest.approx(0.0, abs=0.3)
    # zero end points confine the noise to the band
    freqs = [0.0, 900.0, 1000.0, 2000.0, 2100.0, FS / 2]
    band = synthesize_broadband([0, 0, 1e-3, 1e-3, 0, 0], freqs, FS, N, seed=1)
    f, p = signal.welch(band, FS, nperseg=4096)
    inside = p[(f > 1100) & (f < 1900)].mean()
    assert 10 * np.log10(inside / 1e-3) == pytest.approx(0.0, abs=0.3)
    assert 10 * np.log10(p[(f > 3000) & (f < 7500)].mean() / inside) < -60.0
    assert 10 * np.log10(p[(f > 100) & (f < 700)].mean() / inside) < -60.0


# ==========================================================================
# Control-dependent envelope
# ==========================================================================


def _two_points(p0, p1):
    return np.vstack([p0, p1]), np.array([0.0, 1.0])


def test_level_follows_a_control_ramp():
    table, axis = _two_points(np.full_like(F, 1e-4), np.full_like(F, 4e-4))  # +6.02 dB
    control = np.linspace(0.0, 1.0, N)
    y = synthesize_broadband(table, F, FS, N, control=control, control_axis=axis, seed=2)
    levels = _block_levels_db(y, 1.0)
    centre = (np.arange(levels.size) + 0.5) / levels.size  # control at the block centre
    expected = 10 * np.log10(((1 - centre) * 1e-4 + centre * 4e-4) * FS / 2)
    assert np.max(np.abs(levels - expected)) < 0.3
    assert np.all(np.diff(levels) > 0)  # monotone crossfade
    assert levels[-1] - levels[0] == pytest.approx(10 * np.log10(4.0) * 0.9, abs=0.3)


def test_no_level_dip_when_crossfading_two_different_spectra_of_equal_power():
    low = ((F >= 50.0) & (F <= 1000.0)).astype(float)
    high = ((F >= 3000.0) & (F <= 6000.0)).astype(float)
    table, axis = _two_points(low / _trapz(low, F), high / _trapz(high, F))  # 1 unit^2 each
    control = np.linspace(0.0, 1.0, N)
    y = synthesize_broadband(table, F, FS, N, control=control, control_axis=axis, seed=3)
    levels = _block_levels_db(y, 0.5)
    ends = 0.5 * (levels[0] + levels[-1])
    assert abs(levels[0] - levels[-1]) < 0.4
    assert np.min(levels) > ends - 0.5  # a linear amplitude crossfade of independent noises dips 3 dB
    assert np.max(levels) < ends + 0.5
    assert levels[levels.size // 2 - 1 : levels.size // 2 + 1].mean() == pytest.approx(ends, abs=0.5)


def test_constant_control_between_two_points_gives_the_power_average_spectrum():
    low = ((F >= 100.0) & (F <= 900.0)).astype(float) * 1e-3
    high = ((F >= 3000.0) & (F <= 6000.0)).astype(float) * 4e-3
    table, axis = _two_points(low, high)

    def band_levels(a):
        y = synthesize_broadband(table, F, FS, N, control=np.full(N, a), control_axis=axis, seed=4)
        f, p = signal.welch(y, FS, nperseg=4096)
        lo = p[(f > 200) & (f < 800)].mean()
        hi = p[(f > 3200) & (f < 5800)].mean()
        return 10 * np.log10(lo / 1e-3), 10 * np.log10(hi / 4e-3)

    lo0, hi0 = band_levels(0.0)
    lo5, hi5 = band_levels(0.5)
    lo1, hi1 = band_levels(1.0)
    assert (lo0, hi1) == pytest.approx((0.0, 0.0), abs=0.3)
    assert hi0 < -50.0 and lo1 < -50.0
    # half of the power of each point: (1 - a) P0 + a P1 with a = 0.5
    assert lo5 == pytest.approx(-3.01, abs=0.3)
    assert hi5 == pytest.approx(-3.01, abs=0.3)
    # a quarter of the way: 75 % of the low band, 25 % of the high band
    lo25, hi25 = band_levels(0.25)
    assert lo25 == pytest.approx(10 * np.log10(0.75), abs=0.3)
    assert hi25 == pytest.approx(10 * np.log10(0.25), abs=0.3)


def test_control_outside_the_axis_holds_the_end_rows_and_the_unit_is_arbitrary():
    table = np.vstack([np.full_like(F, 1e-4), np.full_like(F, 4e-4), np.full_like(F, 16e-4)])
    axis = np.array([30.0, 60.0, 120.0])  # e.g. km/h, not evenly spaced
    levels = []
    for c in (0.0, 30.0, 60.0, 120.0, 400.0):
        y = synthesize_broadband(table, F, FS, 40000, control=np.full(40000, c), control_axis=axis, seed=5)
        levels.append(10 * np.log10(np.mean(y**2) / (FS / 2)))
    expected = 10 * np.log10([1e-4, 1e-4, 4e-4, 16e-4, 16e-4])
    assert levels == pytest.approx(expected, abs=0.2)


def test_stationary_psd_equals_a_constant_control_table_with_identical_rows():
    psd = _shape("sloped")
    table = np.vstack([psd, psd])
    a = synthesize_broadband(psd, F, FS, 20000, seed=8)
    b = synthesize_broadband(table, F, FS, 20000, control=np.full(20000, 0.3),
                             control_axis=[0.0, 1.0], seed=8)
    assert b == pytest.approx(a, rel=1e-10, abs=1e-14)


# ==========================================================================
# Determinism, structure
# ==========================================================================


def test_seed_determinism():
    psd = _shape("bump")
    a = synthesize_broadband(psd, F, FS, 30000, seed=11)
    assert np.array_equal(a, synthesize_broadband(psd, F, FS, 30000, seed=11))
    assert np.array_equal(
        synthesize_broadband(psd, F, FS, 30000, seed=np.random.default_rng(5)),
        synthesize_broadband(psd, F, FS, 30000, seed=np.random.default_rng(5)),
    )
    b = synthesize_broadband(psd, F, FS, 30000, seed=12)
    assert not np.array_equal(a, b)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.03
    c = synthesize_broadband(psd, F, FS, 30000)  # seed=None: fresh entropy
    d = synthesize_broadband(psd, F, FS, 30000)
    assert c.shape == (30000,) and np.all(np.isfinite(c)) and not np.array_equal(c, d)


def test_output_for_a_longer_record_starts_with_the_shorter_one():
    psd = _shape("sloped")
    short = synthesize_broadband(psd, F, FS, 5000, seed=2)
    long = synthesize_broadband(psd, F, FS, 9000, seed=2)
    assert np.array_equal(short, long[:5000])
    table, axis = _two_points(psd, 2 * psd)
    ctl = np.linspace(0, 1, 9000)
    short = synthesize_broadband(table, F, FS, 5000, control=ctl[:5000], control_axis=axis, seed=2)
    long = synthesize_broadband(table, F, FS, 9000, control=ctl, control_axis=axis, seed=2)
    # the last grain of the short record sits beyond its end and holds the last control value
    assert np.array_equal(short[:4800], long[:4800])


def test_chunked_overlap_add_gives_the_same_signal(monkeypatch):
    psd = _shape("lowpass")
    ref = synthesize_broadband(psd, F, FS, 30000, seed=9)
    monkeypatch.setattr(_broadband, "_CHUNK_ELEMENTS", 1000)  # a few grains per block
    assert np.array_equal(synthesize_broadband(psd, F, FS, 30000, seed=9), ref)


@pytest.mark.parametrize("n", [1, 2, 399, 400, 401, 800, 1234])
def test_short_and_odd_lengths(n):
    y = synthesize_broadband(_shape("flat"), F, FS, n, seed=1)
    assert y.shape == (n,) and y.dtype == np.float64 and np.all(np.isfinite(y))


def test_silent_psd_gives_silence():
    assert not np.any(synthesize_broadband(np.zeros_like(F), F, FS, 5000, seed=1))


def test_grain_length_is_rounded_to_an_even_number_of_samples():
    # grain 0.0501 s -> 801.6 samples -> 802; the output still has the right level
    y = synthesize_broadband(_shape("flat"), F, FS, N, grain=0.0501, seed=2)
    assert 10 * np.log10(np.mean(y**2) / (1e-3 * FS / 2)) == pytest.approx(0.0, abs=0.1)
    assert np.max(np.abs(_octave_errors_db(y, _shape("flat")))) < 0.5


# ==========================================================================
# Validation
# ==========================================================================

_BASE = dict(psd=np.full(5, 1e-3), freqs=np.linspace(0.0, 8000.0, 5), fs=16000.0, n=2000)
_TABLE = np.full((3, 5), 1e-3)


def _call(**over):
    kw = {**_BASE, **over}
    return synthesize_broadband(kw.pop("psd"), kw.pop("freqs"), kw.pop("fs"), kw.pop("n"), **kw)


@pytest.mark.parametrize(
    "over, match",
    [
        (dict(fs=0.0), "fs must be positive"),
        (dict(fs=np.nan), "fs must be positive"),
        (dict(n=0), "n must be a positive integer"),
        (dict(n=100.5), "n must be a positive integer"),
        (dict(freqs=[100.0]), "at least two"),
        (dict(freqs=np.zeros((2, 2))), "at least two"),
        (dict(freqs=[0.0, 2000.0, 2000.0, 6000.0, 8000.0]), "strictly increasing"),
        (dict(freqs=[-1.0, 2000.0, 4000.0, 6000.0, 8000.0]), "non-negative"),
        (dict(psd=np.full(4, 1e-3)), "one column per entry"),
        (dict(psd=np.array([1e-3, -1e-3, 1e-3, 1e-3, 1e-3])), "non-negative"),
        (dict(psd=np.array([1e-3, np.nan, 1e-3, 1e-3, 1e-3])), "finite"),
        (dict(psd=np.ones((2, 2, 5))), "shape"),
        (dict(control=np.zeros(2000)), "only to a 2-D psd"),
        (dict(control_axis=[0.0, 1.0]), "only to a 2-D psd"),
        (dict(psd=_TABLE), "needs control_axis and control"),
        (dict(psd=_TABLE, control=np.zeros(2000)), "needs control_axis and control"),
        (dict(psd=_TABLE, control_axis=[0.0, 1.0, 2.0]), "needs control_axis and control"),
        (dict(psd=_TABLE, control_axis=[0.0, 1.0], control=np.zeros(2000)), "control_axis must have one entry"),
        (dict(psd=_TABLE, control_axis=[0.0, 1.0, 1.0], control=np.zeros(2000)), "strictly increasing"),
        (dict(psd=_TABLE, control_axis=[0.0, 1.0, 2.0], control=np.zeros(1999)), "control must have shape"),
        (dict(psd=_TABLE, control_axis=[0.0, 1.0, 2.0], control=np.full(2000, np.nan)), "non-finite"),
        (dict(psd=_TABLE[:1], control_axis=[0.0], control=np.zeros(2000)), "at least two rows"),
        (dict(grain=0.0), "grain must be positive"),
        (dict(grain=0.0002), "shorter than 8 samples"),
        (dict(highpass=-1.0), "highpass"),
        (dict(highpass=8000.0), "highpass"),
    ],
)
def test_validation_errors(over, match):
    with pytest.raises(ValueError, match=match):
        _call(**over)


def test_unknown_window_is_rejected():
    with pytest.raises(ValueError, match="window"):
        _synthesize(_BASE["psd"], _BASE["freqs"], FS, 100, control=None, control_axis=None,
                    grain=GRAIN, highpass=20.0, seed=0, window="blackman")
