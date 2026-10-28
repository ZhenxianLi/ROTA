"""Tests of rota_order._resolution.resolution_check against analytic frequencies.

The check works on frequencies f = order * rpm / 60, so the truth is analytic:
separations, beat frequencies and crossing times follow from the speed curves.
Where a signal is involved (beat frequency of twin lines), it is rendered by
rota_order.synth.synth_runup and the check is compared with what was synthesised.
"""

import warnings

import numpy as np
import pytest
from scipy import signal

from rota_order._resolution import (
    HANN_ENVELOPE_CUTOFF,
    HANN_MAINLOBE_HALF_WIDTH,
    VK_SEPARATION_FACTOR,
    ResolutionReport,
    ResolutionWarning,
    resolution_check,
)
from rota_order.extract import order_waveform
from rota_order.synth import rpm_profiles, synth_runup

FS = 2000.0
ORDER = 10.0  # order of the twin lines
W = 8.0  # window in revolutions of the T8 cases


def _check(*args, **kwargs):
    """resolution_check with its warning silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResolutionWarning)
        return resolution_check(*args, **kwargs)


def _twin_rpm(ratio, dur=10.0, lo=2800.0, hi=3200.0, fs=FS):
    """Two shafts, the second ``ratio`` times faster, both ramping lo -> hi."""
    r = np.linspace(lo, hi, round(dur * fs))
    return np.column_stack([r, ratio * r])


def _idle_runup(fs=FS):
    """T14-like speed: 800 rpm idle for 2 s, linear to 3000 rpm at 11 s, held."""
    t = np.arange(round(12.0 * fs)) / fs
    return rpm_profiles(t, "linear", start=800.0, stop=3000.0, t_start=2.0, t_stop=11.0)


def _twin(ratio, **setting):
    return _check(_twin_rpm(ratio), FS, [ORDER, ORDER], shafts=[0, 1], **setting)


def test_constants_are_the_documented_ones():
    assert HANN_MAINLOBE_HALF_WIDTH == 2.0
    assert HANN_ENVELOPE_CUTOFF == 0.72
    assert VK_SEPARATION_FACTOR == 1.0


# --------------------------------------------------------------------------
# T8: twin shafts of nearly equal speed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [1.005, 1.02])
def test_t8_close_twins_are_unresolved_and_warn(ratio):
    rpm = _twin_rpm(ratio)
    with pytest.warns(ResolutionWarning, match="shaft 1") as rec:
        rep = resolution_check(rpm, FS, [ORDER, ORDER], shafts=[0, 1], window_revs=W)
    assert isinstance(rep, ResolutionReport)
    assert not rep.ok
    assert len(rep.warnings) == 1
    assert "order 10 (shaft 0) and order 10 (shaft 1)" in rep.warnings[0]
    assert "t = 0.00 s" in rep.warnings[0]  # worst time named
    assert issubclass(rec[0].category, UserWarning)
    (p,) = rep.pairs
    assert not p["resolved"]
    assert p["fraction_unresolved"] == 1.0  # ratio of separation to resolution is constant
    assert p["orders"] == (ORDER, ORDER) and p["shafts"] == (0, 1)
    # Analytic separation: f1 - f0 = order * (ratio - 1) * rpm / 60, smallest at 2800 rpm.
    df_min = ORDER * (ratio - 1.0) * 2800.0 / 60.0
    assert p["min_separation_hz"] == pytest.approx(df_min, rel=1e-9)
    assert p["beat_frequency_hz"] == pytest.approx(df_min, rel=1e-9)
    assert p["worst_time"] == 0.0
    assert p["unresolved_intervals"][0][0] == 0.0
    # Resolution: c * (faster shaft's rpm) / (60 * W).
    assert p["resolution_hz"] == pytest.approx(2.0 * ratio * 2800.0 / (60.0 * W), rel=1e-9)


def test_t8_half_percent_and_four_percent_with_eight_revolutions():
    # Acceptance case of ROADMAP T8: 0.5 % is unresolved (warning), 4 % resolved.
    with pytest.warns(ResolutionWarning):
        resolution_check(_twin_rpm(1.005), FS, [ORDER] * 2, shafts=[0, 1], window_revs=W)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a resolved case must not warn
        rep = resolution_check(_twin_rpm(1.04), FS, [ORDER] * 2, shafts=[0, 1], window_revs=W)
    assert rep.ok and rep.warnings == []
    assert rep.pairs[0]["fraction_unresolved"] == 0.0
    assert rep.pairs[0]["resolved"]


def test_t8_limit_sits_at_two_bins():
    # Resolved iff order * eps * W / (c * (1 + eps)) >= 1, i.e. eps >= c / (order * W - c).
    eps = HANN_MAINLOBE_HALF_WIDTH / (ORDER * W - HANN_MAINLOBE_HALF_WIDTH)
    assert not _twin(1.0 + 0.995 * eps, window_revs=W).ok
    assert _twin(1.0 + 1.005 * eps, window_revs=W).ok


@pytest.mark.parametrize("ratio", [1.005, 1.02, 1.04])
@pytest.mark.parametrize("kind", ["window_revs", "bandwidth", "time_window"])
def test_recommended_setting_resolves(ratio, kind):
    start = {"window_revs": W, "bandwidth": 50.0, "time_window": 0.05}[kind]
    rep = _twin(ratio, **{kind: start})
    rec = {
        "window_revs": rep.recommended_window_revs,
        "bandwidth": rep.recommended_bandwidth,
        "time_window": rep.recommended_time_window,
    }
    assert all(v is not None for v in rec.values())
    again = _twin(ratio, **{kind: rec[kind]})
    assert again.ok and again.warnings == []
    # and the other two kinds resolve as well
    for other, value in rec.items():
        assert _twin(ratio, **{other: value}).ok
    # The recommendation is the coarsest: slightly coarser does not resolve.
    coarser = {
        "window_revs": rec["window_revs"] * 0.97,
        "bandwidth": rec["bandwidth"] * 1.03,
        "time_window": rec["time_window"] * 0.97,
    }[kind]
    assert not _twin(ratio, **{kind: coarser}).ok


def test_recommendation_matches_analytic_values_for_half_percent_twins():
    eps = 0.005
    rep = _twin(1.0 + eps, window_revs=W)
    # window: c * (1 + eps) / (order * eps) revolutions
    assert rep.recommended_window_revs == pytest.approx(2.0 * (1 + eps) / (ORDER * eps), rel=3e-3)
    # bandwidth B (one-sided -3 dB offset, pass band 2 B): the pass band must fit
    # in the smallest separation (at 2800 rpm), B = d_min / 2; time window: c / d_min
    df_min = ORDER * eps * 2800.0 / 60.0
    # (rounded down to three significant figures: 1.1667 -> 1.16, a 0.6 % step)
    assert rep.recommended_bandwidth == pytest.approx(df_min / 2.0, rel=1e-2)
    assert rep.recommended_time_window == pytest.approx(2.0 / df_min, rel=3e-3)
    # recommended window is rounded up, bandwidth down (never to the unsafe side)
    assert rep.recommended_window_revs >= 2.0 * (1 + eps) / (ORDER * eps) * (1 - 1e-9)
    assert rep.recommended_bandwidth <= df_min / 2.0 * (1 + 1e-9)


def test_vk_bandwidth_limit_is_half_the_separation():
    # bandwidth B is the one-sided -3 dB offset, as in order_waveform: the pass
    # band is 2 B wide and two lines are resolved from 2 B * VK_SEPARATION_FACTOR apart.
    df_min = ORDER * 0.02 * 2800.0 / 60.0  # 9.33 Hz for 2 % twins
    assert _twin(1.02, bandwidth=0.99 * df_min / 2.0).ok
    assert not _twin(1.02, bandwidth=1.01 * df_min / 2.0).ok
    # The resolution is reported in Hz: the full pass-band width 2 B.
    assert _twin(1.02, bandwidth=3.0).pairs[0]["resolution_hz"] == pytest.approx(6.0)
    # ... and the old reading (B as full width) would have resolved B = df_min.
    assert not _twin(1.02, bandwidth=df_min).ok


@pytest.mark.parametrize("delta", [3.0, 8.0])
def test_vk_rule_matches_the_real_filter_on_stationary_twins(delta):
    """The bandwidth rule against order_waveform: leakage of the neighbour line.

    Two stationary lines ``delta`` Hz apart (order 10 on two shafts, strong
    line 1.0, weak line 0.3). The weak line is extracted alone
    (``joint=False``), so whatever the strong neighbour contributes to its
    envelope is leakage; its size relative to the neighbour's amplitude is the
    single-order response ``H(delta)`` of the filter. With ``B = delta / 2``
    (pass band ``2 B = delta``: just resolved according to the check) that is
    -17.6 dB; with ``B = delta`` the neighbour sits at the -3 dB point of the
    pass band (the check calls the pair unresolved) and leaks -3 dB.
    """
    fs, n, r0 = 2000.0, 12000, 3000.0
    rpm = np.column_stack([np.full(n, r0), np.full(n, r0 + 6.0 * delta)])  # 10 * dr / 60 = delta
    orders = [
        {"order": ORDER, "shaft": 0, "amplitude": 0.3, "phase": 0.4},
        {"order": ORDER, "shaft": 1, "amplitude": 1.0, "phase": 2.0},
    ]
    sig = synth_runup(fs, rpm, orders, seed=1)
    assert np.abs(sig.order_frequency[:, 0] - sig.order_frequency[:, 1]).min() == pytest.approx(delta)
    truth = sig.order_amplitude[:, 0] * np.exp(1j * (sig.order_phase[:, 0] - 2 * np.pi * ORDER * sig.rev[:, 0]))
    interior = slice(round(2.0 * fs), -round(2.0 * fs))  # many time constants (0.18 / B) from the ends

    def leakage_db(bandwidth):
        comp = order_waveform(sig.x, fs, rpm, [ORDER], shafts=[0], bandwidth=bandwidth, joint=False)
        err = comp.envelopes[interior, 0] - truth[interior]
        return 20.0 * np.log10(np.sqrt(np.mean(np.abs(err) ** 2)) / 1.0)  # re the neighbour's amplitude

    def response_db(bandwidth):  # H(delta) of the documented single-order response
        r2 = 0.5 * (np.sqrt(2.0) - 1.0) / (2.0 * np.sin(np.pi * bandwidth / fs)) ** 4
        return -20.0 * np.log10(1.0 + 2.0 * r2 * (2.0 * np.sin(np.pi * delta / fs)) ** 4)

    # B = delta: the check says unresolved and recommends B = delta / 2 ...
    too_wide = _check(rpm, fs, [ORDER, ORDER], shafts=[0, 1], bandwidth=delta)
    assert not too_wide.ok
    b_ok = too_wide.recommended_bandwidth
    assert b_ok == pytest.approx(delta / 2.0, rel=1e-2) and b_ok <= delta / 2.0
    assert _check(rpm, fs, [ORDER, ORDER], shafts=[0, 1], bandwidth=b_ok).ok
    # ... with which the neighbour is below -15 dB, against about -3 dB at B = delta.
    leak_ok, leak_wide = leakage_db(b_ok), leakage_db(delta)
    assert leak_ok < -15.0
    assert leak_wide > -6.0 and leak_wide - leak_ok > 10.0
    # Both readings are the analytic filter response, to a fraction of a dB.
    assert leak_ok == pytest.approx(response_db(b_ok), abs=0.1)
    assert leak_wide == pytest.approx(response_db(delta), abs=0.1)


def test_time_window_limit_is_two_over_window():
    df_min = ORDER * 0.02 * 2800.0 / 60.0
    assert _twin(1.02, time_window=2.0 / df_min * 1.01).ok
    assert not _twin(1.02, time_window=2.0 / df_min * 0.99).ok


# --------------------------------------------------------------------------
# Idle: half orders, angle window against fixed time window
# --------------------------------------------------------------------------


def test_time_window_fails_at_idle_where_angle_window_passes():
    rpm = _idle_runup()
    top_speed_window = 4.0 / (3000.0 / 60.0)  # 4 revolutions at 3000 rpm = 0.08 s
    # Angle domain, order resolution 0.25 (T14): orders 2 and 2.5 are exactly 2 bins apart.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok = resolution_check(rpm, FS, [2.0, 2.5], window_revs=4.0)
    assert ok.ok and ok.pairs[0]["fraction_unresolved"] == 0.0
    # A time window of the same duration (equal-bandwidth STFT) merges them at low speed.
    with pytest.warns(ResolutionWarning, match=r"order 2 \(shaft 0\) and order 2.5 \(shaft 0\)"):
        bad = resolution_check(rpm, FS, [2.0, 2.5], time_window=top_speed_window)
    assert not bad.ok
    p = bad.pairs[0]
    # Separation 0.5 * rpm / 60 against 2 / 0.08 = 25 Hz: resolved only at 3000 rpm (t >= 11 s).
    assert p["fraction_unresolved"] == pytest.approx(11.0 / 12.0, abs=2e-3)
    assert p["worst_time"] <= 2.0  # the idle plateau
    assert p["beat_frequency_hz"] == pytest.approx(0.5 * 800.0 / 60.0, rel=1e-9)
    assert p["min_separation_hz"] == pytest.approx(0.5 * 800.0 / 60.0, rel=1e-9)
    assert p["resolution_hz"] == pytest.approx(25.0)
    assert p["unresolved_intervals"][0][0] == 0.0
    assert p["unresolved_intervals"][0][1] == pytest.approx(11.0, abs=0.01)
    # The angle-domain window is the advice: 4 revolutions (2 bins of 0.5 order = 0.5 order).
    assert bad.recommended_window_revs == 4.0
    assert _check(rpm, FS, [2.0, 2.5], window_revs=bad.recommended_window_revs).ok
    # A fixed window would need 2 / 6.67 Hz = 0.3 s.
    assert bad.recommended_time_window == pytest.approx(0.3, rel=1e-6)
    assert _check(rpm, FS, [2.0, 2.5], time_window=bad.recommended_time_window).ok


@pytest.mark.parametrize("profile", ["idle_runup", "constant", "random_walk"])
def test_same_shaft_orders_resolve_independently_of_speed_in_angle_domain(profile):
    n = round(12 * FS)
    if profile == "idle_runup":
        rpm = _idle_runup()
    elif profile == "constant":
        rpm = np.full(n, 1500.0)
    else:
        g = np.random.default_rng(3)
        rpm = 1800.0 + np.cumsum(g.standard_normal(n)) * 2.0
        rpm = np.clip(rpm, 600.0, 3000.0)
    assert _check(rpm, FS, [2.0, 2.5], window_revs=3.0).pairs[0]["fraction_unresolved"] == 1.0
    assert _check(rpm, FS, [2.0, 2.5], window_revs=4.0).pairs[0]["fraction_unresolved"] == 0.0


# --------------------------------------------------------------------------
# Identical and crossing frequencies: no setting helps
# --------------------------------------------------------------------------


def test_identical_frequencies_have_no_recommendation():
    rep = _twin(1.0, window_revs=W)
    assert not rep.ok
    assert rep.recommended_window_revs is None
    assert rep.recommended_bandwidth is None
    assert rep.recommended_time_window is None
    p = rep.pairs[0]
    assert p["min_separation_hz"] == 0.0 and p["fraction_unresolved"] == 1.0
    assert "identical" in rep.warnings[0]
    with pytest.warns(ResolutionWarning, match="No setting resolves"):
        resolution_check(_twin_rpm(1.0), FS, [ORDER] * 2, shafts=[0, 1], window_revs=W)


def test_duplicate_order_on_one_shaft_is_identical():
    rep = _check(np.full(4000, 3000.0), FS, [3.0, 3.0], bandwidth=1.0)
    assert not rep.ok and rep.recommended_bandwidth is None


def test_crossing_orders_report_the_crossing_time():
    n = round(10 * FS)
    t = np.arange(n) / FS
    t_cross = 4.9873  # between two samples
    rpm = np.column_stack([np.full(n, 3000.0), 3000.0 + 100.0 * (t - t_cross)])
    rep = _check(rpm, FS, [ORDER, ORDER], shafts=[0, 1], window_revs=W)
    (p,) = rep.pairs
    assert len(p["crossings"]) == 1
    assert p["crossings"][0] == pytest.approx(t_cross, abs=1e-9)
    assert p["min_separation_hz"] == 0.0
    assert not rep.ok
    assert rep.recommended_window_revs is None and rep.recommended_bandwidth is None
    assert f"{t_cross:.2f}" in rep.warnings[0] and "cross" in rep.warnings[0]
    # Unresolved only around the crossing: |df| = 10 * 100 * |t - t_cross| / 60 < resolution.
    res = 2.0 * 3000.0 / (60.0 * W)  # the faster shaft is within 3 % of 3000 rpm: about
    half = res / (ORDER * 100.0 / 60.0)
    (a, b) = p["unresolved_intervals"][0]
    assert a == pytest.approx(t_cross - half, abs=0.05)
    assert b == pytest.approx(t_cross + half, abs=0.05)
    assert p["fraction_unresolved"] == pytest.approx(2 * half / 10.0, abs=0.01)


# --------------------------------------------------------------------------
# Envelope variation (bias-variance wall)
# --------------------------------------------------------------------------


def test_beat_frequency_matches_the_synthesised_twin_lines():
    fs, dur, r0, ratio = 2000.0, 10.0, 3000.0, 1.006
    n = round(dur * fs)
    rpm = np.column_stack([np.full(n, r0), np.full(n, r0 * ratio)])
    orders = [
        {"order": ORDER, "shaft": 0, "amplitude": 1.0, "phase": 0.0},
        {"order": ORDER, "shaft": 1, "amplitude": 1.0, "phase": 0.0},
    ]
    sig = synth_runup(fs, rpm, orders)
    rep = _check(rpm, fs, orders, bandwidth=1.0)  # orders as dicts, like the synth config
    p = rep.pairs[0]
    truth = np.abs(sig.order_frequency[:, 0] - sig.order_frequency[:, 1])
    assert p["beat_frequency_hz"] == pytest.approx(truth.min(), rel=1e-9)
    assert p["beat_frequency_hz"] == pytest.approx(ORDER * (ratio - 1.0) * r0 / 60.0, rel=1e-9)
    # The envelope of the summed lines really beats at that frequency.
    env = np.abs(signal.hilbert(sig.stems["orders"]))
    spec = np.abs(np.fft.rfft(env - env.mean()))
    f_beat = np.fft.rfftfreq(n, 1.0 / fs)[np.argmax(spec)]
    assert f_beat == pytest.approx(p["beat_frequency_hz"], abs=1.0 / dur)


def test_resolved_near_equal_pair_gets_an_envelope_note_not_a_warning():
    # 4 % twins are at least 18.7 Hz apart; B = 5 Hz has a 10 Hz pass band.
    rep = _twin(1.04, bandwidth=5.0)
    assert rep.ok and rep.warnings == []
    p = rep.pairs[0]
    assert p["near_equal"]
    # Separation / 2 for a Vold-Kalman bandwidth (Hz): the envelope cut-off of a
    # setting that just resolves the pair is B = d_min / 2.
    assert p["max_envelope_rate_hz"] == pytest.approx(ORDER * 0.04 * 2800.0 / 60.0 / 2.0, rel=1e-9)
    assert any("envelope" in s and "order 10" in s for s in rep.notes)
    far = _check(_twin_rpm(1.0), FS, [2.0, 8.0], window_revs=W)
    assert not far.pairs[0]["near_equal"] and far.notes == []


def test_window_envelope_bound_is_0p36_of_the_separation():
    p = _twin(1.04, window_revs=W).pairs[0]
    df_min = ORDER * 0.04 * 2800.0 / 60.0
    expected = HANN_ENVELOPE_CUTOFF / HANN_MAINLOBE_HALF_WIDTH * (1.0 / 1.04) * df_min
    assert p["max_envelope_rate_hz"] == pytest.approx(expected, rel=1e-9)
    q = _twin(1.04, time_window=0.1).pairs[0]
    assert q["max_envelope_rate_hz"] == pytest.approx(0.36 * df_min, rel=1e-9)


def test_envelope_wall_for_half_percent_twins():
    # 0.5 % twins at ~3000 rpm beat at 2.3 Hz or more; a Vold-Kalman filter narrow enough
    # to separate them (B <= 1.17 Hz, pass band 2.3 Hz) follows envelope changes up to
    # its cut-off B = 1.17 Hz only.
    df_min = ORDER * 0.005 * 2800.0 / 60.0
    b_sep = 0.9 * df_min / 2.0  # 90 % of the largest bandwidth that resolves the twins
    ok = _twin(1.005, bandwidth=b_sep, envelope_rate=0.5)
    assert ok.ok and ok.recommended_bandwidth is not None
    with pytest.warns(ResolutionWarning, match="bias-variance wall"):
        wall = resolution_check(
            _twin_rpm(1.005), FS, [ORDER] * 2, shafts=[0, 1],
            bandwidth=b_sep, envelope_rate=2.0,
        )
    assert not wall.ok
    # The pair itself is resolved: the problem is the envelope, not the separation.
    assert wall.pairs[0]["resolved"]
    assert any("wall" in s and "shaft 1" in s for s in wall.warnings)
    assert wall.recommended_bandwidth is None
    assert wall.recommended_window_revs is None and wall.recommended_time_window is None


def test_setting_too_narrow_for_the_envelope_is_reported():
    rep = _twin(1.04, bandwidth=0.5, envelope_rate=1.0)  # envelope cut-off B = 0.5 Hz < 1 Hz
    assert not rep.ok
    assert any("0.5 Hz" in s and "bias" in s for s in rep.warnings)
    assert rep.pairs[0]["resolved"]
    # B = 1 Hz follows an envelope rate of 1 Hz (the cut-off is B itself).
    assert _twin(1.04, bandwidth=1.0, envelope_rate=1.0).ok
    # A wider setting that still resolves the pair is recommended and follows the envelope.
    assert rep.recommended_bandwidth is not None and rep.recommended_bandwidth >= 1.0
    assert _twin(1.04, bandwidth=rep.recommended_bandwidth, envelope_rate=1.0).ok
    # window_revs / time_window variants of the same message
    assert not _twin(1.04, time_window=5.0, envelope_rate=1.0).ok  # 0.72 / 5 = 0.14 Hz
    assert not _twin(1.04, window_revs=300.0, envelope_rate=1.0).ok


# --------------------------------------------------------------------------
# Single lines, images, inactive samples, long records
# --------------------------------------------------------------------------


def test_single_order_is_checked_against_its_negative_frequency_image():
    rpm = np.full(4000, 800.0)
    # Order 0.5 at 800 rpm is 6.67 Hz; the image sits at -6.67 Hz, 13.3 Hz away.
    bad = _check(rpm, FS, [0.5], window_revs=1.0)  # resolution 2 * 800 / 60 = 26.7 Hz
    assert not bad.ok and bad.pairs == []
    assert "negative-frequency image" in bad.warnings[0]
    assert bad.images[0]["min_separation_hz"] == pytest.approx(2 * 0.5 * 800.0 / 60.0)
    assert bad.recommended_window_revs == 2.0  # c / (2 * order)
    assert _check(rpm, FS, [0.5], window_revs=2.0).ok
    assert _check(rpm, FS, [0.5], window_revs=2.0).recommended_window_revs == 2.0
    # With widely spaced orders the image limit also bounds the recommendation.
    wide = _check(rpm, FS, [1.0, 5.0], window_revs=0.1)
    assert wide.recommended_window_revs == 1.0  # pair needs 0.5, image of order 1 needs 1


def test_inactive_samples_are_ignored():
    n = round(12 * FS)
    rpm = np.full(n, 3000.0)
    rpm[: round(2 * FS)] = 0.0  # engine off
    rpm[round(5 * FS) : round(6 * FS)] = np.nan  # gap in the reference
    rep = _check(rpm, FS, [2.0, 2.5], window_revs=4.0)
    assert rep.ok
    assert rep.pairs[0]["n_evaluated"] == n - round(2 * FS) - round(1 * FS)
    # The unresolved interval of a failing setting starts when the engine runs.
    bad = _check(rpm, FS, [2.0, 2.5], window_revs=3.0)
    assert bad.pairs[0]["unresolved_intervals"][0][0] == pytest.approx(2.0)
    assert bad.pairs[0]["fraction_unresolved"] == 1.0  # of the evaluated samples


def test_components_above_nyquist_are_not_checked():
    rpm = np.full(750, 3000.0)  # 50 Hz per order; fs = 150 Hz so Nyquist is 75 Hz
    rep = _check(rpm, 150.0, [1.0, 2.0], window_revs=1.0)
    assert any("never active" in s and "order 2" in s for s in rep.notes)
    assert rep.pairs[0]["n_evaluated"] == 0 and rep.pairs[0]["resolved"]
    with pytest.raises(ValueError, match="no component is active"):
        resolution_check(rpm, 150.0, [2.0], window_revs=1.0)
    with pytest.raises(ValueError, match="no component is active"):
        resolution_check(np.zeros(100), FS, [2.0], window_revs=1.0)


def test_long_record_is_evaluated_on_a_strided_grid():
    fs = 20000.0
    rpm = _twin_rpm(1.04, dur=6.0, fs=fs)  # 120 000 samples: every third one is evaluated
    rep = _check(rpm, fs, [ORDER, ORDER], shafts=[0, 1], window_revs=W)
    assert rep.ok
    assert rep.pairs[0]["n_evaluated"] == 40000
    assert rep.pairs[0]["min_separation_hz"] == pytest.approx(ORDER * 0.04 * 2800.0 / 60.0, rel=1e-9)


def test_many_orders_give_all_pairs():
    sig = synth_runup(
        FS, _twin_rpm(1.37, dur=4.0),
        [{"order": 1.0, "amplitude": 1.0, "phase": 0.0},
         {"order": 2.0, "amplitude": 1.0, "phase": 0.0},
         {"order": 2.0, "amplitude": 1.0, "shaft": 1, "phase": 0.0},
         {"order": 4.0, "amplitude": 1.0, "shaft": 1, "phase": 0.0}],
    )
    rep = _check(sig.rpm, FS, sig.config["orders"], window_revs=20.0)
    assert [(p["i"], p["j"]) for p in rep.pairs] == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert list(rep.shafts) == [0, 0, 1, 1] and list(rep.orders) == [1.0, 2.0, 2.0, 4.0]
    # Frequency separations equal the synthesised instantaneous frequencies.
    F = sig.order_frequency
    for p in rep.pairs:
        truth = np.abs(F[:, p["i"]] - F[:, p["j"]])
        assert p["min_separation_hz"] == pytest.approx(truth.min(), rel=1e-9)
    # Orders 1 and 2 of one shaft are one order apart, 20 revolutions resolve them.
    assert rep.pairs[0]["resolved"]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_argument_errors():
    rpm = np.full(1000, 3000.0)
    kw = {"window_revs": 4.0}
    with pytest.raises(ValueError, match="exactly one"):
        resolution_check(rpm, FS, [2.0, 3.0])
    with pytest.raises(ValueError, match="exactly one"):
        resolution_check(rpm, FS, [2.0, 3.0], window_revs=4.0, bandwidth=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        resolution_check(rpm, FS, [2.0, 3.0], bandwidth=1.0, time_window=0.1)
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="window_revs"):
            resolution_check(rpm, FS, [2.0, 3.0], window_revs=bad)
    with pytest.raises(ValueError, match="bandwidth"):
        resolution_check(rpm, FS, [2.0, 3.0], bandwidth=-1.0)
    with pytest.raises(ValueError, match="time_window"):
        resolution_check(rpm, FS, [2.0, 3.0], time_window=0.0)
    with pytest.raises(ValueError, match="window_revs"):
        resolution_check(rpm, FS, [2.0, 3.0], window_revs="long")
    for fs in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="fs"):
            resolution_check(rpm, fs, [2.0], **kw)
    with pytest.raises(ValueError, match="rpm"):
        resolution_check(np.ones((4, 2, 2)), FS, [2.0], **kw)
    with pytest.raises(ValueError, match="at least 2"):
        resolution_check([3000.0], FS, [2.0], **kw)
    with pytest.raises(ValueError, match="orders"):
        resolution_check(rpm, FS, [], **kw)
    with pytest.raises(ValueError, match="positive"):
        resolution_check(rpm, FS, [2.0, -1.0], **kw)
    with pytest.raises(ValueError, match="positive"):
        resolution_check(rpm, FS, [2.0, 0.0], **kw)
    with pytest.raises(ValueError, match="orders"):
        resolution_check(rpm, FS, ["a", "b"], **kw)
    with pytest.raises(ValueError, match="all numbers or all dicts"):
        resolution_check(rpm, FS, [2.0, {"order": 3.0}], **kw)
    with pytest.raises(ValueError, match="envelope_rate"):
        resolution_check(rpm, FS, [2.0, 3.0], envelope_rate=0.0, **kw)


def test_shaft_errors():
    two = np.full((1000, 2), 3000.0)
    kw = {"window_revs": 4.0}
    with pytest.raises(ValueError, match="columns of rpm"):
        resolution_check(two, FS, [2.0, 3.0], shafts=[0, 2], **kw)
    with pytest.raises(ValueError, match="columns of rpm"):
        resolution_check(two, FS, [2.0, 3.0], shafts=[0, -1], **kw)
    with pytest.raises(ValueError, match="columns of rpm"):
        resolution_check(two[:, 0], FS, [2.0, 3.0], shafts=[0, 1], **kw)
    with pytest.raises(ValueError, match="one entry per order"):
        resolution_check(two, FS, [2.0, 3.0], shafts=[0, 1, 1], **kw)
    with pytest.raises(ValueError, match="integers"):
        resolution_check(two, FS, [2.0, 3.0], shafts=[0, 0.5], **kw)
    with pytest.raises(ValueError, match="not both"):
        resolution_check(two, FS, [{"order": 2.0}, {"order": 3.0}], shafts=[0, 1], **kw)
    with pytest.raises(ValueError, match="columns of rpm"):
        resolution_check(two, FS, [{"order": 2.0, "shaft": 5}], **kw)
    # a single integer applies to all orders
    rep = _check(two, FS, [2.0, 3.0], shafts=1, **kw)
    assert list(rep.shafts) == [1, 1]


def test_scalar_and_dict_orders_are_accepted():
    rpm = np.full(1000, 3000.0)
    assert list(_check(rpm, FS, 2.0, window_revs=4.0).orders) == [2.0]
    assert list(_check(rpm, FS, {"order": 2.0}, window_revs=4.0).orders) == [2.0]
