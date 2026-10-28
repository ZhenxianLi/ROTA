"""Tests of the ECMA-418-1:2024 (ECMA-74 Annex D) tone prominence functions.

Analytic truth: white noise of known one-sided PSD ``N0`` and sinusoids of
known mean square from rota_order.synth, or deterministic spectra built here line
by line, for which the standard's formulas (3rd edition) give the TNR and PR
exactly.
"""

import numpy as np
import pytest
from scipy import signal

from rota_order._prominence import (
    OrderComponent,
    OrderProminence,
    ToneProminence,
    _proximity_spacing,
    components,
    critical_band_edges,
    critical_bandwidth,
    order_prominence,
    prominence_criterion,
    tone_prominence,
)
from rota_order.maps import rpm_frequency_map
from rota_order.synth import synth_runup, synth_tone_noise


def _db(x):
    return 10.0 * np.log10(x)


# ---------------------------------------------------------------------------
# Critical bands and criteria (worked values of ECMA-418-1)
# ---------------------------------------------------------------------------


def test_critical_bandwidth_worked_values():
    # Clause 10 examples: 162.2 Hz at 1 kHz, 117.3 Hz at 500 Hz.
    assert critical_bandwidth(1000.0) == pytest.approx(162.2, abs=0.05)
    assert critical_bandwidth(500.0) == pytest.approx(117.3, abs=0.05)
    f = np.array([100.0, 1000.0, 5000.0])
    np.testing.assert_allclose(
        critical_bandwidth(f), 25 + 75 * (1 + 1.4 * (f / 1000) ** 2) ** 0.69
    )


def test_band_edges_arithmetic_and_geometric():
    # Example of 12.2: 922.2 Hz and 1084.4 Hz at 1 kHz (geometric centring).
    f1, f2 = critical_band_edges(1000.0)
    assert (f1, f2) == pytest.approx((922.2, 1084.4), abs=0.05)
    assert np.sqrt(f1 * f2) == pytest.approx(1000.0)
    # Arithmetic centring up to 500 Hz.
    f1, f2 = critical_band_edges(300.0)
    dfc = critical_bandwidth(300.0)
    assert (f1, f2) == pytest.approx((300 - dfc / 2, 300 + dfc / 2))
    # Adjacent bands are contiguous with the middle band.
    f1l, a, b, f2u = critical_band_edges(1000.0, adjacent=True)
    assert (a, b) == critical_band_edges(1000.0)
    assert f1l == pytest.approx(-149.5 + 1.001e3 - 6.9e-5 * 1e6)
    assert f2u == pytest.approx(149.5 + 1.035e3 + 7.7e-5 * 1e6)
    # Low-frequency truncation at 20 Hz, and continuity of the fits.
    assert critical_band_edges(150.0, adjacent=True)[0] == 20.0
    assert critical_band_edges(171.4, adjacent=True)[0] == pytest.approx(20.0, abs=0.1)
    lo = critical_band_edges(1600.0, adjacent=True)
    hi = critical_band_edges(1600.0 + 1e-6, adjacent=True)
    np.testing.assert_allclose(lo, hi, atol=0.2)
    with pytest.raises(ValueError):
        critical_band_edges(-1.0)


def test_criteria():
    assert prominence_criterion(1000.0, "tnr") == 8.0
    assert prominence_criterion(5000.0, "pr") == 9.0
    assert prominence_criterion(100.0, "tnr") == pytest.approx(16.33)
    assert prominence_criterion(100.0, "pr") == pytest.approx(19.0)
    assert np.isnan(prominence_criterion(50.0))
    assert np.isnan(prominence_criterion(11200.0, "pr"))
    c = prominence_criterion(np.array([89.1, 500.0, 999.0, 1000.0]))
    assert np.all(np.diff(c) < 0) or c[-1] == 8.0
    with pytest.raises(ValueError):
        prominence_criterion(1000.0, "aures")


def test_proximity_spacing_worked_values():
    # Examples of 11.6 / A.6 (3rd edition): 23 Hz at 150 Hz, 63.8 Hz at 850 Hz.
    assert _proximity_spacing(150.0) == pytest.approx(23.0, abs=0.05)
    assert _proximity_spacing(850.0) == pytest.approx(63.8, abs=0.05)
    # From 1 kHz up every tone of the band is proximate (NOTE of 11.6: small
    # discontinuity, 81.5 Hz just below 1 kHz).
    assert _proximity_spacing(999.9) == pytest.approx(81.5, abs=0.1)
    assert _proximity_spacing(1000.0) == np.inf


def test_band_edges_3rd_edition_corrections():
    # Formula (20) of the 3rd edition: f2,M = f1,M + delta_fM (the 2nd
    # edition printed ft + delta_fM); geometric centring above 500 Hz.
    for ft in (600.0, 1600.0, 8000.0):
        f1, f2 = critical_band_edges(ft)
        assert f2 - f1 == pytest.approx(critical_bandwidth(ft))
        assert f2 < ft + critical_bandwidth(ft)
    # f1,L by Formula (21) / (B.5) with C_L,1 (Formula (21) misprints C_L,0):
    # contiguous with the 20 Hz truncation at 171.4 Hz.
    assert critical_band_edges(171.4, adjacent=True)[0] == pytest.approx(
        -149.5 + 1.001 * 171.4 - 6.9e-5 * 171.4**2)
    # 500 Hz itself: arithmetic centring (12.2, f <= 500 Hz).
    f1, f2 = critical_band_edges(500.0)
    assert (f1 + f2) / 2 == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# tone_prominence on synthetic spectra
# ---------------------------------------------------------------------------


def _deterministic(ft, pt, n0, df=0.5, fmax=6000.0, extra=()):
    """Flat PSD n0 plus Hann-shaped on-line tones of mean square pt.

    A Hann-windowed sinusoid on a line occupies lines -1, 0, +1 with power
    ratios 1/4, 1, 1/4 (ENBW 1.5 lines); in PSD terms its lines sum to the
    tone power. Returns the axis and the PSD.
    """
    f = np.arange(0.0, fmax, df)
    S = np.full(f.size, n0)
    for fr, p in ((ft, pt), *extra):
        k = round(fr / df)
        S[k - 1:k + 2] += p / df * np.array([0.25, 1.0, 0.25]) / 1.5
    return f, S


@pytest.mark.parametrize("ft", [140.0, 1000.0, 3000.0])
def test_tnr_pr_white_noise_analytic(ft):
    fs = 48000.0
    s = synth_tone_noise(fs, 8.0, 1.0, -12.0, tone_freq=ft, seed=1)
    pt, n0 = np.mean(s.tone**2), np.mean(s.noise**2) / (fs / 2)
    f, S = signal.welch(s.x, fs, window="hann", nperseg=2**15)
    r = tone_prominence(f, S, ft)
    assert isinstance(r, ToneProminence)
    tnr_true = _db(pt / (n0 * critical_bandwidth(ft)))
    assert r.tnr == pytest.approx(tnr_true, abs=0.5)
    assert r.tone_level == pytest.approx(_db(pt), abs=0.2)
    assert r.noise_level == pytest.approx(_db(n0 * critical_bandwidth(ft)), abs=0.5)
    f1l, f1, f2, f2u = critical_band_edges(ft, adjacent=True)
    xl = n0 * (f1 - f1l) * (100.0 / (f1 - f1l) if ft <= 171.4 else 1.0)
    pr_true = _db((pt + n0 * (f2 - f1)) / (0.5 * (xl + n0 * (f2u - f2))))
    assert r.pr == pytest.approx(pr_true, abs=0.5)
    assert r.band == (f1, f2) and r.lower_band == (f1l, f1)
    assert not r.tone_band_exceeded


def test_deterministic_exact_tnr_and_pr():
    n0, df = 1e-4, 0.5
    for ft in (150.0, 600.0, 2000.0):
        pt = 0.05
        f, S = _deterministic(ft, pt, n0, df)
        r = tone_prominence(f, S, ft)
        # Tone band: the minimal points bracketing the peak are the first
        # noise lines, +-2 lines (the Hann main lobe), 5 lines in all.
        assert r.tone_bandwidth == pytest.approx(5 * df)
        assert r.tone_band == pytest.approx((ft - 2.5 * df, ft + 2.5 * df))
        assert not r.tone_band_exceeded
        # Power above the line joining the end points: the sinusoid alone;
        # the noise under it stays in X_n = (X_tot - X_t) dfc / df_tot.
        dfc = critical_bandwidth(ft)
        assert r.tone_power == pytest.approx(pt, rel=1e-9)
        assert r.noise_power == pytest.approx(n0 * dfc, rel=1e-9)
        assert r.tnr == pytest.approx(_db(pt / (n0 * dfc)), abs=1e-9)
        f1l, f1, f2, f2u = critical_band_edges(ft, adjacent=True)
        xm = pt + n0 * (f2 - f1)
        xl = n0 * (f1 - f1l) * (100.0 / (f1 - f1l) if ft <= 171.4 else 1.0)
        assert r.pr == pytest.approx(_db(xm / (0.5 * (xl + n0 * (f2u - f2)))), abs=0.03)


def _band_lines(f, ft):
    f1, f2 = critical_band_edges(ft)
    return np.flatnonzero((f >= f1) & (f < f2))


def test_tone_band_end_points_inside_window():
    """ECMA-418-1:2024, 11.2 / A.2 and Formulae (A.1), (A.2), exact.

    Dips of the noise floor at +-3 lines, inside the 15 % window, are the
    minimal points: the tone band has 7 lines and the tone power is the
    power above the line joining the dips.
    """
    n0, df, ft, pt = 1e-4, 0.5, 1000.0, 0.05
    f, S = _deterministic(ft, pt, n0, df)
    k = round(ft / df)
    dfc = critical_bandwidth(ft)
    m = int(np.floor(0.075 * dfc / df))  # 24 lines: 12 Hz on either side
    assert m == 24
    S[[k - 3, k + 3]] = 0.5 * n0
    # Deeper dips outside the window are not end points.
    S[[k - m - 2, k + m + 2]] = 0.2 * n0
    r = tone_prominence(f, S, ft)
    assert r.tone_band == pytest.approx((f[k - 3] - df / 2, f[k + 3] + df / 2))
    assert r.tone_bandwidth == pytest.approx(7 * df)
    assert not r.tone_band_exceeded
    xt_prime = pt + (5 * n0 + 2 * 0.5 * n0) * df
    xt = xt_prime - (0.5 * n0 + 0.5 * n0) * df * 7 / 2
    assert r.tone_power == pytest.approx(xt, rel=1e-12)
    band = _band_lines(f, ft)
    xtot = S[band].sum() * df
    assert r.total_power == pytest.approx(xtot, rel=1e-12)
    xn = (xtot - xt) * dfc / (band.size * df)
    assert r.noise_power == pytest.approx(xn, rel=1e-12)
    assert r.tnr == pytest.approx(_db(xt / xn), abs=1e-9)


def test_tone_band_end_point_beyond_window_edge():
    """A minimal point on the window edge moves to the nearest local minimum.

    The floor falls linearly beyond the main lobe on the upper side down to a
    minimum ``m`` lines past the window edge, then rises: the upper end point
    is that minimum and the tone band is wider than 15 % of the critical
    band. A linear floor under the tone band is removed exactly by the line
    joining the end points, so the tone power is still the sinusoid's.
    """
    n0, df, ft, pt = 1e-4, 0.5, 1000.0, 0.05
    f = np.arange(0.0, 6000.0, df)
    k = round(ft / df)
    m = int(np.floor(0.075 * critical_bandwidth(ft) / df))
    lo_end, hi_end = k - 2, k + 2 * m
    j = np.arange(f.size)
    # Linear from n0 at lo_end to 0.3 n0 at hi_end, then rising again; flat
    # n0 below lo_end (the lower end point is the first line off the lobe).
    slope = (0.3 * n0 - n0) / (hi_end - lo_end)
    S = n0 + slope * (j - lo_end)
    S[j < lo_end] = n0
    S[j > hi_end] = 0.3 * n0 + 0.5 * n0 * (j[j > hi_end] - hi_end) / 10.0
    S = np.minimum(S, 2 * n0)
    S[k - 1:k + 2] += pt / df * np.array([0.25, 1.0, 0.25]) / 1.5
    r = tone_prominence(f, S, ft)
    assert r.tone_band == pytest.approx((f[lo_end] - df / 2, f[hi_end] + df / 2))
    assert r.tone_band_exceeded
    assert r.tone_bandwidth > 0.15 * critical_bandwidth(ft)
    assert r.tone_power == pytest.approx(pt, rel=1e-9)
    band = _band_lines(f, ft)
    xn = (S[band].sum() * df - pt) * critical_bandwidth(ft) / (band.size * df)
    assert r.noise_power == pytest.approx(xn, rel=1e-9)


def test_no_tone_has_no_power_above_the_line():
    # Flat noise only: (almost) no power above the line, not prominent.
    n0 = 1e-4
    f, S = _deterministic(1000.0, 0.0, n0)
    r = tone_prominence(f, S, 1000.0)
    assert r.tone_power <= 1e-12 * n0
    assert r.tnr < -100.0 and not r.prominent
    assert r.noise_power == pytest.approx(n0 * critical_bandwidth(1000.0))


def test_low_frequency_pr_normalises_lower_band():
    # At 120 Hz the lower band [20 Hz, f1) is narrower than a critical band;
    # normalised to 100 Hz, white noise alone gives PR close to the
    # 10 lg(dfc_M / ((100 + dfc_U) / 2)) of the band widths.
    n0 = 1e-4
    f, S = _deterministic(120.0, 0.0, n0)
    r = tone_prominence(f, S, 120.0, method="pr")
    f1l, f1, f2, f2u = critical_band_edges(120.0, adjacent=True)
    assert f1l == 20.0
    assert r.lower_power == pytest.approx(100.0 * n0, rel=0.01)
    assert r.pr == pytest.approx(_db((f2 - f1) / (0.5 * (100.0 + f2u - f2))), abs=0.03)
    assert not r.prominent


@pytest.mark.parametrize("method", ["tnr", "pr"])
@pytest.mark.parametrize("ft", [150.0, 500.0, 2500.0])
def test_threshold_behaviour(method, ft):
    """The flag switches where the ratio crosses the criterion."""
    n0, df = 1e-4, 0.5
    crit = prominence_criterion(ft, method)
    f, S0 = _deterministic(ft, 0.0, n0, df)
    base = tone_prominence(f, S0, ft, method=method)
    for delta, expected in ((-0.2, False), (0.2, True)):
        # Find the tone power that gives crit + delta, by bisection on the
        # monotonic ratio.
        lo, hi = 1e-8, 1e3
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            r = tone_prominence(*_deterministic(ft, mid, n0, df), ft, method=method)
            lo, hi = (mid, hi) if r.ratio < crit + delta else (lo, mid)
        r = tone_prominence(*_deterministic(ft, hi, n0, df), ft, method=method)
        assert r.ratio == pytest.approx(crit + delta, abs=1e-3)
        assert r.criterion == crit
        assert r.prominent is expected
        assert r.margin == pytest.approx(delta, abs=1e-3)
    assert not base.prominent


def test_window_and_scaling_enbw():
    """Flat-top, Hann and PSD or power scaling give the same TNR."""
    fs = 16000.0
    s = synth_tone_noise(fs, 8.0, 1.0, -8.0, tone_freq=700.0, seed=4)
    pt, n0 = np.mean(s.tone**2), np.mean(s.noise**2) / (fs / 2)
    truth = _db(pt / (n0 * critical_bandwidth(700.0)))
    out = []
    for win in ("hann", "flattop", "blackmanharris"):
        for scaling in ("psd", "power"):
            sc = "density" if scaling == "psd" else "spectrum"
            f, S = signal.welch(s.x, fs, window=win, nperseg=2**13, scaling=sc)
            r = tone_prominence(f, S, 700.0, scaling=scaling, window=win)
            out.append(r.tnr)
            assert r.tone_level == pytest.approx(_db(pt), abs=0.25)
    np.testing.assert_allclose(out, truth, atol=0.5)
    # Ignoring the ENBW biases the masking noise by 10 lg(ENBW).
    f, P = signal.welch(s.x, fs, window="flattop", nperseg=2**13, scaling="spectrum")
    wrong = tone_prominence(f, P, 700.0, scaling="power", window="flattop", enbw=1.0)
    assert wrong.noise_level - _db(n0 * critical_bandwidth(700.0)) == pytest.approx(
        _db(3.77), abs=0.5)


def test_multiple_tones_in_band():
    n0 = 1e-4
    # Above 1 kHz a second tone in the band is always combined (11.6):
    # Formulae (15) and (16), X_t = X_t,p + X_t,s (A.7 misprints a minus)
    # and X_n = [X_tot - (X_t,p + X_t,s)] dfc / df_tot.
    f, S = _deterministic(1000.0, 0.05, n0, extra=[(1050.0, 0.02)])
    r = tone_prominence(f, S, 1000.0, other_tones=[1050.0])
    dfc = critical_bandwidth(1000.0)
    assert r.combined_freqs == (1050.0,)
    assert r.tone_power == pytest.approx(0.07, rel=1e-9)
    assert r.tone_bandwidth == pytest.approx(10 * 0.5)
    assert r.noise_power == pytest.approx(n0 * dfc, rel=1e-9)
    assert r.tnr == pytest.approx(_db(0.07 / (n0 * dfc)), abs=1e-9)
    # At 300 Hz, 40 Hz apart exceeds the 23 Hz proximity spacing: the second
    # tone is removed from the masking noise but not added to the tone.
    f, S = _deterministic(300.0, 0.05, n0, extra=[(340.0, 0.02)])
    r = tone_prominence(f, S, 300.0, other_tones=[340.0])
    dfc = critical_bandwidth(300.0)
    assert r.combined_freqs == ()
    assert r.tone_power == pytest.approx(0.05, rel=1e-9)
    assert r.noise_power == pytest.approx(n0 * dfc, rel=1e-9)
    assert r.tnr == pytest.approx(_db(0.05 / (n0 * dfc)), abs=1e-9)
    # Not declared, the second tone counts as masking noise.
    plain = tone_prominence(f, S, 300.0)
    assert plain.tnr < r.tnr - 3.0
    assert plain.noise_power == pytest.approx(n0 * dfc + 0.02 * dfc / (_band_lines(f, 300.0).size * 0.5))
    # A declared tone that is absent (no line above the noise) is ignored.
    f, S = _deterministic(300.0, 0.05, n0)
    r = tone_prominence(f, S, 300.0, other_tones=[340.0])
    assert r.noise_power == pytest.approx(n0 * dfc, rel=1e-9)
    # Two proximate tones 6 Hz apart at 1 kHz, inside one 15 % window: the
    # primary's band stops at the noise line between them, nothing is
    # counted twice.
    f, S = _deterministic(1000.0, 0.05, n0, extra=[(1006.0, 0.02)])
    r = tone_prominence(f, S, 1000.0, other_tones=[1006.0])
    assert r.combined_freqs == (1006.0,)
    assert r.tone_power == pytest.approx(0.07, rel=1e-9)
    assert r.noise_power == pytest.approx(n0 * critical_bandwidth(1000.0), rel=1e-9)


def test_multichannel_spectrum_and_validation():
    n0 = 1e-4
    f, S = _deterministic(800.0, 0.05, n0)
    S2 = np.column_stack([S, 4.0 * S])
    r = tone_prominence(f, S2, 800.0)
    one = tone_prominence(f, S, 800.0)
    assert r.tnr.shape == (2,)
    np.testing.assert_allclose(r.tnr, one.tnr)
    np.testing.assert_allclose(r.tone_level, one.tone_level + np.array([0, _db(4.0)]))
    assert r.prominent.dtype == bool
    with pytest.raises(ValueError):
        tone_prominence(f, -S, 800.0)
    with pytest.raises(ValueError):
        tone_prominence(f[::-1], S, 800.0)
    with pytest.raises(ValueError):
        tone_prominence(f, S, 800.0, method="x")
    with pytest.raises(ValueError):
        tone_prominence(f, S, 800.0, scaling="amplitude")
    # Band outside the spectrum: NaN, not prominent.
    r = tone_prominence(f, S, 5990.0)
    assert np.isnan(r.tnr) and np.isnan(r.pr) and not r.prominent


# ---------------------------------------------------------------------------
# order_prominence on a run-up
# ---------------------------------------------------------------------------

FS = 4000.0
ORDER = 6.0
AMP = np.sqrt(2.0)  # peak; mean square AMP**2 / 2 = 1
SIGMA = 1.0  # white noise RMS; one-sided PSD SIGMA**2 / (FS / 2)
# The scatter of the masking-noise estimate is set by the bandwidth-time
# product: about 4.3 / sqrt(delta_fc * T) dB, i.e. 0.2 dB for a critical band
# of 110 Hz averaged over T = 4 s. The accuracy run therefore lasts 16 s and
# is averaged in 1200 rpm bins.


def _rpm(dur):
    t = np.arange(int(FS * dur)) / FS
    return 1200.0 + 4800.0 * t / dur  # 1200 -> 6000 rpm: order 6 at 120-600 Hz


def _runup(dur, orders, seed):
    rpm = _rpm(dur)
    return synth_runup(FS, rpm, orders, noise_rms=SIGMA, seed=seed), rpm


@pytest.fixture(scope="module")
def runup():
    """Slow run-up, 16 s (order 6 sweeps at 30 Hz/s)."""
    return _runup(16.0, [{"order": ORDER, "amplitude": AMP, "phase": 0.0}], 3)


@pytest.fixture(scope="module")
def fast_runup():
    """Fast run-up, 8 s (order 6 sweeps at 60 Hz/s)."""
    return _runup(8.0, [{"order": ORDER, "amplitude": AMP, "phase": 0.0}], 4)


def _tnr_truth(rpm, order=ORDER, amp=AMP):
    n0 = SIGMA**2 / (FS / 2)
    return _db(amp**2 / 2 / (n0 * critical_bandwidth(order * rpm / 60.0)))


def test_order_prominence_runup_matches_analytic(runup):
    s, rpm = runup
    res = order_prominence(s.x, FS, rpm, [ORDER], rpm_bins=1200.0)
    assert isinstance(res, OrderProminence)
    assert res.tnr.shape == (res.rpm.size, 1) and res.rpm.size == 4
    truth = _tnr_truth(res.rpm)
    err = res.tnr[:, 0] - truth
    assert np.all(np.abs(err) < 1.0), err
    assert abs(np.mean(err)) < 0.4
    np.testing.assert_allclose(res.freq[:, 0], ORDER * res.rpm / 60.0)
    np.testing.assert_allclose(res.tone_level[:, 0], _db(AMP**2 / 2), atol=0.5)
    # Frame rule: the 1 % rule gives 100/6 -> 17 rev; 54 Hann lines across
    # the critical band at 6000 rpm (600 Hz), so that the 15 % window of
    # 11.2 spans +-4 lines, give 54 * 100 / 124.6 -> 44 rev.
    assert res.frame_revs == 44.0
    assert res.params["min_lines"] == 54.0
    assert np.all(res.critical_bandwidth / res.line_spacing[:, None] >= 54.0)
    assert np.all(res.line_spacing <= 0.01 * res.freq[:, 0])
    assert not np.any(res.tone_band_exceeded)
    # The flag follows the frequency-dependent criterion: not prominent near
    # 190 Hz (TNR 13.0 dB < 14.1 dB), prominent near 540 Hz (12.3 > 10.2 dB).
    clear = np.abs(truth - res.criterion[:, 0]) > 0.8
    assert clear.sum() >= 2
    np.testing.assert_array_equal(res.prominent[clear, 0],
                                  (truth > res.criterion[:, 0])[clear])
    assert not res.prominent[0, 0] and res.prominent[-1, 0]
    # Per frame (no averaging): a median error near zero.
    fr = order_prominence(s.x, FS, rpm, [ORDER])
    assert np.all(fr.n_frames == 1) and fr.rpm.size > 30
    assert abs(np.median(fr.tnr[:, 0] - _tnr_truth(fr.rpm))) < 0.5
    np.testing.assert_allclose(fr.t, np.sort(fr.t))
    # PR: middle band holds tone + noise, adjacent bands noise only.
    assert np.all(np.isfinite(res.pr)) and np.all(res.pr[:, 0] > 10.0)


def test_fixed_time_window_is_biased(fast_runup):
    """A fixed window meeting the 1 % resolution near 100 Hz smears the order.

    The order sweeps at 60 Hz/s; a 1 s window (1 Hz lines) sees it spread
    over about 60 Hz, about half the critical band, in every frame. The tone
    band of ECMA-418-1:2024 (11.2) is confined to about 15 % of the critical
    band around the peak, and the line joining its end points lies on the
    smeared plateau: most of the order's power counts as masking noise and
    the TNR is biased by about -19 dB. In the angle domain the order stays a
    line and the estimate is unbiased.
    """
    s, rpm = fast_runup
    m = rpm_frequency_map(s.x, FS, rpm, nperseg=int(FS), overlap=0.5)
    td = []
    for j in range(m.t.size):
        P = np.abs(m.values[:, j]) ** 2
        ft = ORDER * m.rpm[j] / 60.0
        r = tone_prominence(m.axis, P, ft, scaling="power")
        td.append(r.tnr)
        assert r.tone_bandwidth < 0.25 * r.critical_bandwidth
    truth = _tnr_truth(m.rpm)
    td_err = np.array(td) - truth
    assert np.max(td_err) < -10.0
    ang = order_prominence(s.x, FS, rpm, [ORDER], rpm_bins=1200.0)
    ang_err = ang.tnr[:, 0] - _tnr_truth(ang.rpm)
    assert np.all(np.abs(ang_err) < 1.5)
    assert abs(np.mean(ang_err)) < 0.6
    assert np.mean(ang_err) - np.mean(td_err) > 10.0


def test_window_robustness(runup):
    s, rpm = runup
    h = order_prominence(s.x, FS, rpm, [ORDER], window="hann", rpm_bins=1200.0)
    f = order_prominence(s.x, FS, rpm, [ORDER], window="flattop", rpm_bins=1200.0)
    assert f.frame_revs > h.frame_revs  # wider main lobe -> more lines needed
    assert f.enbw_bins == pytest.approx(3.77, abs=0.01)
    for res in (h, f):
        assert np.all(np.abs(res.tnr[:, 0] - _tnr_truth(res.rpm)) < 1.0)
        np.testing.assert_allclose(res.tone_level[:, 0], _db(AMP**2 / 2), atol=0.5)
    assert np.mean(np.abs(f.tnr[:, 0] - h.tnr[:, 0])) < 0.6


def test_multichannel_and_pr_method(runup):
    s, rpm = runup
    x = np.column_stack([s.x, 2.0 * s.x])
    res = order_prominence(x, FS, rpm, [ORDER], method="pr", rpm_bins=1200.0)
    assert res.ratio.shape == (res.rpm.size, 1, 2)
    np.testing.assert_allclose(res.ratio[..., 0], res.ratio[..., 1], atol=1e-9)
    np.testing.assert_allclose(res.ratio, res.pr)
    np.testing.assert_allclose(res.tone_level[..., 1] - res.tone_level[..., 0], _db(4.0))
    np.testing.assert_allclose(res.criterion, prominence_criterion(res.freq, "pr"))
    mono = order_prominence(s.x, FS, rpm, [ORDER], method="pr", rpm_bins=1200.0)
    np.testing.assert_allclose(mono.pr, res.pr[..., 0], atol=1e-9)
    # PR truth: tone + noise in the middle band over the mean of the
    # adjacent bands' noise (all above 171.4 Hz here: no truncation).
    n0 = SIGMA**2 / (FS / 2)
    pr_truth = []
    for f in res.freq[:, 0]:
        f1l, f1, f2, f2u = critical_band_edges(f, adjacent=True)
        pr_truth.append(_db((AMP**2 / 2 + n0 * (f2 - f1)) / (0.5 * n0 * (f2u - f2 + f1 - f1l))))
    np.testing.assert_allclose(mono.pr[:, 0], pr_truth, atol=1.0)


def test_harmonics_components_and_prominent_range():
    # Order 12 only appears above 3000 rpm; order 9 sits between harmonics
    # and stays below its criterion.
    orders = [
        {"order": 6.0, "amplitude": AMP, "phase": 0.0},
        {"order": 12.0, "amplitude": AMP, "phase": 0.0,
         "gain_rpm": [1200.0, 2999.0, 3000.0, 6000.0], "gain": [0.0, 0.0, 1.0, 1.0]},
        {"order": 9.0, "amplitude": 0.3 * AMP, "phase": 0.0},
    ]
    s, rpm = _runup(16.0, orders, 5)
    res = order_prominence(s.x, FS, rpm, [6.0, 12.0, 9.0], rpm_bins=np.arange(1200, 6001, 600))
    # Order 12 at 600 Hz-1.2 kHz: TNR 12.1-10.5 dB, criterion 9.8-8 dB.
    above = res.rpm > 3000.0
    assert np.all(res.prominent[above, 1])
    assert not np.any(res.prominent[~above, 1])
    t12 = _tnr_truth(res.rpm[above], order=12.0)
    assert np.all(np.abs(res.tnr[above, 1] - t12) < 1.0)
    comps = components(res)
    assert [type(c) for c in comps] == [OrderComponent] * 3
    c6, c12, c9 = comps
    assert c6.fundamental == 6.0 and c12.harmonic_number == 2 and c12.harmonicity == 1.0
    assert c9.harmonicity == pytest.approx(0.0)
    assert 3000.0 < c12.rpm_range[0] < 3600.0 and c12.rpm_range[1] > 5400.0
    assert c9.prominent_fraction == 0.0 and np.isnan(c9.rpm_range[0])
    assert c6.level_mean == pytest.approx(0.0, abs=0.3)
    assert c6.level >= c6.level_mean
    # A pure order: tone lines = Hann main lobe (+ leakage), about 5 lines.
    assert 3 * res.order_resolution <= c6.bandwidth <= 12 * res.order_resolution
    assert c6.bandwidth_hz > 0
    assert components(res, fundamental=3.0)[2].harmonic_number == 3


def test_order_prominence_validation(fast_runup):
    s, rpm = fast_runup
    with pytest.raises(ValueError):
        order_prominence(s.x, FS, rpm, [0.0])
    with pytest.raises(ValueError):
        order_prominence(s.x, FS, rpm, [ORDER], method="loudness")
    with pytest.raises(ValueError):
        order_prominence(s.x, FS, rpm, [ORDER], order_resolution=0.1, frame_revs=20)
    r = order_prominence(s.x, FS, rpm, [ORDER], order_resolution=1 / 40)
    assert r.frame_revs == 40.0 and r.order_resolution == pytest.approx(1 / 40)
    # The 0.25 % resolution recommended for the TNR: 400 / 6 -> 67 rev.
    r = order_prominence(s.x, FS, rpm, [ORDER], resolution_fraction=0.0025, rpm_bins=2400.0)
    assert r.frame_revs == 67.0
    with pytest.raises(TypeError):
        components("not a result")
