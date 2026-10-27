"""Tests of rota.tonal against analytic truth and the limits of IEC 61672-1 / IEC 61260-1.

Signals are analytic sines, periodic flat-spectrum multisines and seeded
Gaussian noise; expected values are closed-form (weighting expressions,
tone levels, effective bandwidths, exponential time weighting).
"""

import numpy as np
import pytest
from scipy import signal

from rota import tonal
from rota.tonal import (
    band_levels_vs_rpm,
    combine_channels,
    level_db,
    level_vs_time,
    octave_band_centres,
    octave_filterbank,
    octave_levels,
    to_calibrated_48k,
    weight,
    weighting_db,
    weighting_filter,
)

P0 = 20e-6
G10 = 10.0**0.3


def _sine_fit(y, f0, fs):
    """Least-squares amplitude and phase of a cosine at f0 (phase at sample 0)."""
    t = np.arange(len(y)) / fs
    M = np.c_[np.cos(2 * np.pi * f0 * t), np.sin(2 * np.pi * f0 * t)]
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    return np.hypot(*c), np.arctan2(-c[1], c[0])


# --------------------------------------------------------------------------
# IEC 61672-1 frequency weightings
# --------------------------------------------------------------------------

# Nominal frequency, A and C weighting (dB), class-1 acceptance limits (dB),
# transcribed from the IEC 61672-1:2013 table of frequency weightings and
# acceptance limits, 10 Hz - 20 kHz.
IEC61672_TABLE = [
    # f_nom,   A,     C,    lo,      hi
    (10,     -70.4, -14.3, -np.inf, 3.5),
    (12.5,   -63.4, -11.2, -np.inf, 3.0),
    (16,     -56.7,  -8.5, -4.5, 2.5),
    (20,     -50.5,  -6.2, -2.5, 2.5),
    (25,     -44.7,  -4.4, -2.0, 2.5),
    (31.5,   -39.4,  -3.0, -1.5, 2.0),
    (40,     -34.6,  -2.0, -1.0, 1.5),
    (50,     -30.2,  -1.3, -1.0, 1.0),
    (63,     -26.2,  -0.8, -1.0, 1.0),
    (80,     -22.5,  -0.5, -1.0, 1.0),
    (100,    -19.1,  -0.3, -1.0, 1.0),
    (125,    -16.1,  -0.2, -1.0, 1.0),
    (160,    -13.4,  -0.1, -1.0, 1.0),
    (200,    -10.9,   0.0, -1.0, 1.0),
    (250,     -8.6,   0.0, -1.0, 1.0),
    (315,     -6.6,   0.0, -1.0, 1.0),
    (400,     -4.8,   0.0, -1.0, 1.0),
    (500,     -3.2,   0.0, -1.0, 1.0),
    (630,     -1.9,   0.0, -1.0, 1.0),
    (800,     -0.8,   0.0, -1.0, 1.0),
    (1000,     0.0,   0.0, -0.7, 0.7),
    (1250,     0.6,   0.0, -1.0, 1.0),
    (1600,     1.0,  -0.1, -1.0, 1.0),
    (2000,     1.2,  -0.2, -1.0, 1.0),
    (2500,     1.3,  -0.3, -1.0, 1.0),
    (3150,     1.2,  -0.5, -1.0, 1.0),
    (4000,     1.0,  -0.8, -1.0, 1.0),
    (5000,     0.5,  -1.3, -1.5, 1.5),
    (6300,    -0.1,  -2.0, -2.0, 1.5),
    (8000,    -1.1,  -3.0, -2.5, 1.5),
    (10000,   -2.5,  -4.4, -3.0, 2.0),
    (12500,   -4.3,  -6.2, -5.0, 2.0),
    (16000,   -6.6,  -8.5, -16.0, 2.5),
    (20000,   -9.3, -11.2, -np.inf, 3.0),
]
_T = np.array(IEC61672_TABLE, dtype=float)
F_EXACT = 1000.0 * 10.0 ** (np.arange(-20, 14) / 10.0)  # exact base-10 frequencies


def test_table_frequencies_are_the_exact_base10_series():
    assert np.all(np.abs(F_EXACT / _T[:, 0] - 1) < 0.03)


@pytest.mark.parametrize("kind, col", [("A", 1), ("C", 2)])
def test_analytic_weighting_matches_iec_table(kind, col):
    w = weighting_db(F_EXACT, kind)
    assert np.max(np.abs(w - _T[:, col])) <= 0.05 + 1e-9  # table rounded to 0.1 dB
    assert weighting_db(1000.0, kind) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("fs", [48000.0, 44100.0])
@pytest.mark.parametrize("kind", ["A", "C"])
def test_digital_weighting_within_class1_limits(kind, fs):
    sos = weighting_filter(kind, fs)
    _, h = signal.sosfreqz(sos, worN=F_EXACT, fs=fs)
    dev = 20 * np.log10(np.abs(h)) - weighting_db(F_EXACT, kind)
    lo, hi = _T[:, 3], _T[:, 4]
    margin = np.minimum(dev - lo, hi - dev)
    assert np.all(margin > 0), F_EXACT[margin <= 0]
    # mid band essentially exact; worst margins (1 kHz +-0.7 dB, 10-12.5 kHz
    # bilinear deviation) stay well inside the limits
    assert np.min(margin) >= 0.69
    assert np.max(np.abs(dev[F_EXACT < 1100])) < 0.01
    _, h1k = signal.sosfreqz(sos, worN=[1000.0], fs=fs)
    assert abs(h1k[0]) == pytest.approx(1.0, abs=1e-12)


def test_bilinear_deviation_grows_towards_nyquist():
    sos = weighting_filter("A", 48000.0)
    f = np.array([6300.0, 10000.0, 16000.0, 20000.0])
    _, h = signal.sosfreqz(sos, worN=f, fs=48000.0)
    dev = 20 * np.log10(np.abs(h)) - weighting_db(f, "A")
    assert np.all(np.diff(dev) < 0) and dev[0] < 0
    assert -2.0 < dev[1] < -0.5   # documented ~ -1.2 dB at 10 kHz


@pytest.mark.parametrize("kind", ["A", "C"])
def test_weight_applies_the_weighting_once(kind):
    fs = 48000.0
    t = np.arange(int(1.0 * fs)) / fs
    freqs = np.array([63.0, 250.0, 1000.0, 4000.0, 8000.0])
    amps = np.array([1.0, 0.5])
    for f0 in freqs:
        x = amps * np.cos(2 * np.pi * f0 * t)[:, None]
        y = weight(x, fs, kind)
        assert y.shape == x.shape and y.dtype == np.float64
        sel = slice(int(0.5 * fs), None)  # after the start-up transient
        got = level_db(y[sel], ref=1.0) - level_db(x[sel], ref=1.0)
        _, h = signal.sosfreqz(weighting_filter(kind, fs), worN=[f0], fs=fs)
        expected = 20 * np.log10(abs(h[0]))
        assert np.all(np.abs(got - expected) < 0.01)          # once ...
        if abs(expected) > 1:
            assert np.all(np.abs(got - 2 * expected) > 1)     # ... not twice
    # causal (not zero-phase): equals a single sosfilt pass
    y = weight(x[:, 0], fs, kind)
    assert np.allclose(y, signal.sosfilt(weighting_filter(kind, fs), x[:, 0]))


def test_weight_axis_and_z():
    fs = 48000.0
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 4800))
    assert np.allclose(weight(x, fs, "A", axis=1), weight(x.T, fs, "A").T)
    assert np.array_equal(weight(x, fs, "Z"), x)
    assert np.array_equal(weighting_filter("Z", fs), [[1, 0, 0, 1, 0, 0]])
    assert np.all(weighting_db([10.0, 1e4], "Z") == 0)
    with pytest.raises(ValueError):
        weighting_filter("B", fs)


# --------------------------------------------------------------------------
# Levels
# --------------------------------------------------------------------------


def test_level_db_sine_and_reference():
    fs = 48000.0
    t = np.arange(48000) / fs
    A = np.array([1.0, 0.1, 0.0])
    x = A * np.sin(2 * np.pi * 1000.0 * t)[:, None]
    L = level_db(x)
    assert L[0] == pytest.approx(20 * np.log10(1 / np.sqrt(2) / P0), abs=1e-9)
    assert L[1] == pytest.approx(L[0] - 20.0, abs=1e-9)
    assert L[2] == -np.inf
    assert level_db(x[:, 0], ref=1.0) == pytest.approx(-10 * np.log10(2), abs=1e-9)
    assert np.allclose(level_db(x.T, axis=1), L)
    with pytest.raises(ValueError):
        level_db(x, ref=0.0)


@pytest.mark.parametrize("tw, tau", [("F", 0.125), ("S", 1.0)])
def test_exponential_time_weighting_steady_and_decay(tw, tau):
    fs = 8000.0
    on = 6.0 * tau
    t = np.arange(int((on + 2 * tau) * fs)) / fs
    A = 2.0
    x = np.where(t < on, A * np.sin(2 * np.pi * 1000.0 * t), 0.0)
    tt, L = level_vs_time(x, fs, time_weighting=tw)
    assert tt.shape == L.shape == t.shape
    steady = 20 * np.log10(A / np.sqrt(2) / P0)
    i_on = int(on * fs) - 1
    assert L[i_on] == pytest.approx(steady + 10 * np.log10(1 - np.exp(-on / tau)), abs=0.01)
    # decay after switch-off: 10 log10(e) / tau dB per second
    i1, i2 = int((on + 0.5 * tau) * fs), int((on + 1.5 * tau) * fs)
    rate = (L[i1] - L[i2]) / ((i2 - i1) / fs)
    assert rate == pytest.approx(10 * np.log10(np.e) / tau, rel=1e-3)
    # explicit time constant gives the same as the letter
    _, L2 = level_vs_time(x, fs, time_weighting=tau)
    assert np.allclose(L2, L)


@pytest.mark.parametrize(
    "tw, tau, tb, nominal",
    [("F", 0.125, 0.2, -1.0), ("F", 0.125, 0.002, -18.0), ("F", 0.125, 0.00025, -27.0),
     ("S", 1.0, 0.2, -7.4), ("S", 1.0, 0.002, -27.0)],
)
def test_toneburst_response(tw, tau, tb, nominal):
    """Maximum level of a 4 kHz tone burst re the steady level: 10 log10(1 - exp(-Tb / tau))."""
    fs = 48000.0
    n = int(1.0 * fs)
    t = np.arange(n) / fs
    start = int(0.1 * fs)
    nb = int(round(tb * fs))
    x = np.zeros(n)
    x[start:start + nb] = np.sin(2 * np.pi * 4000.0 * (t[start:start + nb] - t[start]))
    _, L = level_vs_time(x, fs, time_weighting=tw, ref=1.0)
    delta = np.max(L) - 10 * np.log10(0.5)
    expected = 10 * np.log10(1 - np.exp(-tb / tau))
    assert delta == pytest.approx(expected, abs=0.05)
    assert delta == pytest.approx(nominal, abs=0.1)


def test_block_rms_levels_and_frames():
    fs = 8000.0
    t = np.arange(int(3 * fs)) / fs
    A = np.where(t < 1.0, 1.0, np.where(t < 2.0, 0.1, 0.01))
    x = np.stack([A, 2 * A], axis=1) * np.sin(2 * np.pi * 500.0 * t)[:, None]
    tt, L = level_vs_time(x, fs, time_weighting=None, frame=0.5, ref=1.0)
    assert np.allclose(tt, np.arange(0.25, 3.0, 0.5))
    expected = 20 * np.log10(np.repeat([1.0, 0.1, 0.01], 2) / np.sqrt(2))
    assert np.allclose(L[:, 0], expected, atol=1e-9)
    assert np.allclose(L[:, 1] - L[:, 0], 20 * np.log10(2), atol=1e-9)
    # exponential read at frame ends
    te, Le = level_vs_time(x[:, 0], fs, time_weighting="F", frame=0.5)
    assert np.allclose(te, np.arange(0.5, 3.01, 0.5) - 1 / fs)
    assert Le[1] == pytest.approx(20 * np.log10(1 / np.sqrt(2) / P0), abs=0.01)  # 8 tau in
    with pytest.raises(ValueError):
        level_vs_time(x, fs, time_weighting=None)


def test_combine_channels():
    L = np.array([[60.0, 70.0], [50.0, 50.0]])
    e = combine_channels(L)
    assert e[0] == pytest.approx(10 * np.log10((1e6 + 1e7) / 2))
    assert e[1] == pytest.approx(50.0)
    assert combine_channels(L, "sum")[1] == pytest.approx(50 + 10 * np.log10(2))
    assert np.allclose(combine_channels(L, "max"), [70.0, 50.0])
    assert np.allclose(combine_channels(L.T, axis=0), e)
    # energy average of channel levels = level of the pooled channels
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4000, 2)) * [1.0, 3.0]
    assert combine_channels(level_db(x)) == pytest.approx(level_db(x.reshape(-1)))
    with pytest.raises(ValueError):
        combine_channels(L, "mean")


# --------------------------------------------------------------------------
# Calibrated resampling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fs, fs_out", [(44100.0, 48000.0), (96000.0, 48000.0),
                                        (8000.0, 48000.0), (51200.0, 48000.0),
                                        (48000.0, 44100.0)])
def test_resampling_preserves_calibrated_amplitude_and_phase(fs, fs_out):
    cal = 2.5
    t = np.arange(int(0.5 * fs)) / fs
    lo = min(fs, fs_out)
    freqs = [0.01 * lo, 0.2 * lo, 0.449 * lo]   # up to the pass-band edge 0.45 min(fs, fs_out)
    x = np.stack([0.3 * np.cos(2 * np.pi * f * t + 0.4) for f in freqs], axis=1)
    y = to_calibrated_48k(x, fs, calibration=cal, fs_out=fs_out)
    assert y.dtype == np.float64
    assert y.shape == (int(np.ceil(x.shape[0] * fs_out / fs)), 3)
    edge = 300
    for j, f in enumerate(freqs):
        a, ph = _sine_fit(y[edge:-edge, j], f, fs_out)
        assert 20 * np.log10(a / (cal * 0.3)) == pytest.approx(0.0, abs=1e-3)
        ph_true = 2 * np.pi * f * edge / fs_out + 0.4
        assert abs(np.angle(np.exp(1j * (ph - ph_true)))) < 1e-3


def test_resampling_suppresses_aliases():
    fs = 96000.0
    t = np.arange(int(0.5 * fs)) / fs
    for f0 in (24500.0, 30000.0, 40000.0):
        y = to_calibrated_48k(np.cos(2 * np.pi * f0 * t), fs)
        alias = 48000.0 - f0
        a, _ = _sine_fit(y[300:-300], alias, 48000.0)
        assert 20 * np.log10(a) < -99.0


def test_resampling_identity_rate_and_errors():
    x = np.arange(10.0)
    assert np.array_equal(to_calibrated_48k(x, 48000.0, calibration=0.5), 0.5 * x)
    with pytest.raises(ValueError):
        to_calibrated_48k(x, 48000.0 * np.sqrt(2) / 1.4)
    with pytest.raises(ValueError):
        to_calibrated_48k(x, -1.0)


# --------------------------------------------------------------------------
# IEC 61260-1 octave bands
# --------------------------------------------------------------------------

NOMINAL_THIRDS = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
                  630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                  10000, 12500, 16000, 20000]


def test_band_centres():
    exact, nominal = octave_band_centres(3, 20, 20000)
    assert np.array_equal(nominal, NOMINAL_THIRDS)
    assert np.allclose(exact, 1000 * G10 ** (np.arange(-17, 14) / 3), rtol=1e-12)
    exact1, nominal1 = octave_band_centres(1, 31.5, 16000)
    assert np.array_equal(nominal1, [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    assert np.allclose(exact1, 1000 * G10 ** np.arange(-5, 5), rtol=1e-12)
    # even fraction: mid-band frequencies at G**((2x+1)/(2b)), none at 1 kHz
    e6, _ = octave_band_centres(6, 900, 1100)
    assert np.allclose(e6, 1000 * G10 ** (np.array([-1, 1]) / 12))
    # base 2
    e2, n2 = octave_band_centres(3, 500, 2000, base=2)
    assert np.allclose(e2, 1000 * 2.0 ** (np.arange(-3, 4) / 3))
    assert np.array_equal(n2, [500, 630, 800, 1000, 1250, 1600, 2000])
    bank = octave_filterbank(48000.0, 3, 100, 1000)
    assert np.allclose(bank.upper / bank.lower, G10 ** (1 / 3))
    assert np.allclose(np.sqrt(bank.upper * bank.lower), bank.exact)


# IEC 61260-1:2014, class-1 acceptance limits on relative attenuation (dB),
# breakpoints at octave-normalised frequencies Ω = G**e (and 1/Ω), e >= 0.
CLASS1_BREAKPOINTS = [
    # e,    min,  max
    (0.0,   -0.4, 0.4),
    (1 / 8, -0.4, 0.5),
    (1 / 4, -0.4, 0.7),
    (3 / 8, -0.4, 1.4),
    (1 / 2,  1.2, 5.3),
    (1.0,   16.6, np.inf),
    (2.0,   40.5, np.inf),
    (3.0,   60.0, np.inf),
    (4.0,   70.0, np.inf),
]


def _breakpoint(e, G, b):
    """Normalised breakpoint frequency of a 1/b-octave filter (IEC 61260-1 mapping)."""
    return 1 + (G ** (1 / (2 * b)) - 1) / (G ** 0.5 - 1) * (G**e - 1)


def class1_mask(f, fm, G, b):
    """Lower and upper class-1 limits on relative attenuation at frequencies f.

    At a breakpoint the tabulated limits apply; between two breakpoints the
    minimum attenuation of the inner one and the maximum attenuation of the
    outer one (the limit lines are steps through the tabulated values).
    """
    r = np.asarray(f, dtype=float) / fm
    r = np.where(r >= 1, r, 1 / r)
    pts = np.array([_breakpoint(e, G, b) for e, _, _ in CLASS1_BREAKPOINTS])
    lo = np.full(r.shape, -np.inf)
    hi = np.full(r.shape, np.inf)
    for i, (_, mn, mx) in enumerate(CLASS1_BREAKPOINTS):
        nxt = pts[i + 1] if i + 1 < len(pts) else np.inf
        sel = (r >= pts[i] * (1 - 1e-12)) & (r < nxt)
        lo[sel] = mn
        hi[sel] = CLASS1_BREAKPOINTS[i + 1][2] if i + 1 < len(pts) else np.inf
        hi[np.isclose(r, pts[i], rtol=1e-12)] = mx
    return lo, hi


def _mask_margins(bank, fs):
    """Minimum margin (dB) to the class-1 mask per region over a dense grid."""
    G, b = G10, bank.fraction
    bps = np.concatenate([[fm * _breakpoint(e, G, b) ** s for e, _, _ in CLASS1_BREAKPOINTS
                           for s in (1, -1)] for fm in bank.exact])
    f = np.unique(np.r_[np.geomspace(5.0, fs / 2, 4000), bps])
    f = f[(f > 0) & (f <= fs / 2)]
    H = bank.frequency_response(f)
    Hm = np.diag(bank.frequency_response(bank.exact))
    out = {"pass": np.inf, "edge": np.inf, "stop": np.inf}
    for i, fm in enumerate(bank.exact):
        with np.errstate(divide="ignore"):
            att = -20 * np.log10(H[:, i] / Hm[i])
        lo, hi = class1_mask(f, fm, G, b)
        with np.errstate(invalid="ignore"):
            m = np.fmin(att - lo, hi - att)
        assert np.all(m > 0), (bank.nominal[i], f[m <= 0])
        r = np.maximum(f / fm, fm / f)
        e_edge = _breakpoint(0.5, G, b)
        for key, sel in (("pass", r < e_edge * (1 - 1e-9)),
                         ("edge", np.isclose(r, e_edge)),
                         ("stop", r >= _breakpoint(1.0, G, b) * (1 - 1e-9))):
            if np.any(sel):
                out[key] = min(out[key], np.min(m[sel]))
    return out


@pytest.mark.parametrize("fs, fraction, fmax", [(48000.0, 3, 20000), (48000.0, 1, 16000),
                                                (44100.0, 3, 16000), (44100.0, 1, 8000)])
def test_filterbank_within_class1_mask(fs, fraction, fmax):
    bank = octave_filterbank(fs, fraction, 20, fmax)
    m = _mask_margins(bank, fs)
    # measured: band edge 3.01 dB (1.81 dB inside), stop band >= 2.0 dB inside
    assert m["edge"] > 1.7
    assert m["stop"] > 1.5
    assert m["pass"] > 0.3


def test_single_rate_design_fails_near_nyquist():
    """Why the top bands are interpolated: bilinear warping breaks the stop-band limit."""
    bank = octave_filterbank(48000.0, 3, 20000, 20000, multirate=False)
    with pytest.raises(AssertionError):
        _mask_margins(bank, 48000.0)


def test_rate_assignment():
    bank = octave_filterbank(48000.0, 3, 20, 20000)
    ratio = bank.rates / bank.upper
    assert np.all(ratio >= 5.0)
    assert np.all(ratio[bank.stages > 0] < 40.0) and np.all(ratio[bank.stages > 0] >= 20.0)
    assert np.all(ratio[bank.stages < 0] < 10.0)
    assert bank.stages[0] == 6 and bank.stages[-1] == -2
    with pytest.raises(ValueError):
        octave_filterbank(44100.0, 3, 20, 20000)   # 20 kHz band edge above Nyquist


FS = 48000.0
BANK = octave_filterbank(FS, 3, 20, 20000)


@pytest.mark.parametrize("i", [0, 7, 17, 25, 29, 30])
def test_tone_at_band_centre(i):
    fm = BANK.exact[i]
    A = 0.2
    t = np.arange(int(3.0 * FS)) / FS
    x = A * np.cos(2 * np.pi * fm * t + 0.3)
    bl = octave_levels(x, FS, skip=1.5, filterbank=BANK)
    target = 20 * np.log10(A / np.sqrt(2) / P0)
    assert bl.levels[i] == pytest.approx(target, abs=0.1)
    assert np.argmax(bl.levels) == i
    assert abs(bl.levels[i] - target) < 0.02
    # neighbours attenuated at least as the class-1 mask requires
    for j in (i - 1, i + 1, i - 2, i + 2):
        if 0 <= j < BANK.exact.size:
            lo, _ = class1_mask(fm, BANK.exact[j], G10, 3)
            att = target - bl.levels[j]
            assert att >= lo
            assert att > (15.0 if abs(j - i) == 1 else 35.0)


def test_measured_tone_levels_follow_frequency_response():
    """The decimation/interpolation chain behaves as its predicted response, aliases included."""
    t = np.arange(int(2.5 * FS)) / FS
    for f0 in (19.5, 333.0, 2900.0, 11000.0, 21500.0, 23500.0):
        x = np.sqrt(2) * P0 * np.cos(2 * np.pi * f0 * t)   # 0 dB re 20 µPa
        bl = octave_levels(x, FS, skip=1.0, filterbank=BANK)
        pred = 20 * np.log10(BANK.frequency_response([f0])[0])
        sel = pred > -60
        assert np.max(np.abs(bl.levels[sel] - pred[sel])) < 0.05


def _flat_multisine(fs, period, rng):
    """Periodic random-phase multisine, flat on the 1/period grid, unit variance."""
    n = int(period * fs)
    X = np.zeros(n // 2 + 1, complex)
    k = np.arange(1, n // 2)
    X[k] = np.exp(2j * np.pi * rng.random(k.size))
    p = np.fft.irfft(X, n)
    return p / np.std(p)


def test_flat_spectrum_band_levels_follow_bandwidth():
    """Band level = 10 log10(S * enbw), enbw = 1.047 * (f2 - f1): +1 dB per third octave."""
    period = 2.0
    x = np.tile(_flat_multisine(FS, period, np.random.default_rng(3)), 3)
    bl = octave_levels(x, FS, skip=period, filterbank=BANK)
    S = 2.0 / FS                          # one-sided PSD of unit-variance white signal
    x_ = np.pi / 6
    enbw = (bl.upper - bl.lower) * x_ / np.sin(x_)
    expected = 10 * np.log10(S * enbw / P0**2)
    assert np.allclose(BANK.enbw, enbw)
    assert np.max(np.abs(bl.levels - expected)) < 0.1
    assert np.max(np.abs(bl.levels[:-1] - expected[:-1])) < 0.02   # warping only in the top band
    assert np.allclose(np.diff(bl.levels[:20]), 1.0, atol=0.01)
    # energy over all bands = total energy in range, plus the Butterworth overlap 1.047
    total = 10 * np.log10(np.sum(10 ** (bl.levels / 10)) * P0**2)
    in_range = 10 * np.log10(S * (bl.upper[-1] - bl.lower[0]))
    assert total - in_range == pytest.approx(10 * np.log10(x_ / np.sin(x_)), abs=0.05)
    assert abs(total - in_range) < 0.25


def test_white_noise_band_levels_within_statistical_tolerance():
    rng = np.random.default_rng(7)
    T = 8.0
    sigma = 0.05                          # Pa
    x = sigma * rng.standard_normal((int(T * FS), 2))
    bank = octave_filterbank(FS, 3, 50, 20000)
    bl = octave_levels(x, FS, skip=0.5, filterbank=bank)
    expected = 10 * np.log10(sigma**2 * 2 / FS * bank.enbw / P0**2)
    # relative std of a mean-square estimate over T - skip: 1 / sqrt(B T)
    tol = 4 * 10 * np.log10(np.e) / np.sqrt(bank.enbw * (T - 0.5))
    assert np.all(np.abs(bl.levels - expected[:, None]) < np.maximum(tol, 0.05)[:, None])
    p = np.sum(10 ** (bl.levels / 10), axis=0) * P0**2
    in_range = sigma**2 * 2 / FS * (bank.upper[-1] - bank.lower[0])
    assert np.all(np.abs(10 * np.log10(p / in_range) - 0.2) < 0.1)


def test_multirate_matches_single_rate():
    single = octave_filterbank(FS, 3, 20, 8000, multirate=False)
    multi = octave_filterbank(FS, 3, 20, 8000)
    assert np.any(multi.stages > 0) and np.all(single.stages == 0)
    f = np.geomspace(5.0, FS / 2, 6000)
    with np.errstate(divide="ignore"):
        Hs = 20 * np.log10(single.frequency_response(f))
        Hm = 20 * np.log10(multi.frequency_response(f))
    sel = Hs > -20
    assert np.max(np.abs(Hm - Hs)[sel]) < 0.1
    rng = np.random.default_rng(11)
    x = rng.standard_normal(int(4 * FS))
    a = octave_levels(x, FS, skip=1.0, filterbank=single).levels
    b = octave_levels(x, FS, skip=1.0, filterbank=multi).levels
    assert np.max(np.abs(a - b)) < 0.1


def test_octave_levels_multichannel_and_frames():
    t = np.arange(int(3.0 * FS)) / FS
    env = np.where(t < 1.5, 1.0, 0.1)
    tone = np.cos(2 * np.pi * 1000.0 * t)
    x = np.stack([env * tone, 2 * env * tone], axis=1)
    bank = octave_filterbank(FS, 3, 250, 4000)
    bl = octave_levels(x, FS, frame=0.25, filterbank=bank)
    assert bl.levels.shape == (12, bank.exact.size, 2)
    assert np.allclose(bl.t, np.arange(0.125, 3.0, 0.25))
    i = int(np.argmin(np.abs(bank.exact - 1000)))
    expected = 20 * np.log10(np.r_[np.ones(6), 0.1 * np.ones(6)] / np.sqrt(2) / P0)
    keep = np.r_[1:6, 7:12]                # frames away from the start and the step
    assert np.allclose(bl.levels[keep, i, 0], expected[keep], atol=0.01)
    assert np.allclose(bl.levels[:, :, 1] - bl.levels[:, :, 0], 20 * np.log10(2), atol=1e-9)
    mono = octave_levels(x[:, 1], FS, frame=0.25, filterbank=bank)
    assert np.allclose(mono.levels, bl.levels[:, :, 1])
    assert bl.frame == pytest.approx(0.25)


def test_band_levels_vs_rpm():
    """Tone at 1 kHz with amplitude proportional to rpm on a linear run-up."""
    fs = 16000.0
    T = 6.2
    t = np.arange(int(T * fs)) / fs
    rpm = 1000.0 + 500.0 * t
    a0 = 0.1
    x = (a0 * rpm / 1000.0)[:, None] * np.cos(2 * np.pi * 1000.0 * t)[:, None] * [1.0, 0.5]
    edges = np.arange(1500.0, 4001.0, 500.0)
    res = band_levels_vs_rpm(x, fs, rpm, rpm_bins=edges, frame=0.1, fraction=1,
                             fmin=250, fmax=4000, skip=0.5)
    assert np.array_equal(res.counts, [10, 10, 10, 10, 10])
    assert np.allclose(res.rpm_centres, edges[:-1] + 250)
    i = int(np.argmin(np.abs(res.exact - 1000)))
    r1, r2 = edges[:-1], edges[1:]
    mean_a2 = (a0 / 1000.0) ** 2 * (r1**2 + r1 * r2 + r2**2) / 3   # mean of A² over the bin
    expected = 10 * np.log10(mean_a2 / 2 / P0**2)
    assert np.allclose(res.levels[:, i, 0], expected, atol=0.05)
    assert np.allclose(res.levels[:, i, 0] - res.levels[:, i, 1], 20 * np.log10(2), atol=1e-9)
    assert res.levels.shape == (5, res.exact.size, 2)
    # min_count and frames outside the bins
    res2 = band_levels_vs_rpm(x[:, 0], fs, rpm, rpm_bins=[500.0, 1000.0, 1500.0, 2000.0],
                              frame=0.1, fraction=1, fmin=250, fmax=4000, skip=0.5,
                              min_count=5)
    assert np.array_equal(res2.counts, [0, 5, 10])
    assert np.all(np.isnan(res2.levels[0])) and np.all(np.isfinite(res2.levels[1:]))
    assert res2.frame_rpm.size == res2.frames.levels.shape[0] == 57


def test_public_api_docstrings():
    for name in tonal.__all__:
        obj = getattr(tonal, name)
        doc = obj.__doc__ or ""
        assert doc.strip(), name
        if callable(obj) and not isinstance(obj, type):
            assert "References" in doc, name
