"""Stress tests of the analysis chain on the realistic scenarios of ``rota.synth.synth_scenario``.

The validation cases (``tests/test_validation*.py``) check each method on a
signal built for it. Here the same methods meet complete synthetic recordings
(``ice_runup``, ``ev_runup``, ``steady_cruise``): a speed ripple locked to the
shaft angle, gear shifts and a clutch-slip launch, half orders, angle-locked
combustion bursts, a transfer path with modes and a reverberant cabin impulse
response, broadband and road noise, transients, a delayed tachometer and an
imperfect recorder. All data come from ``rota.synth`` with fixed seeds and the
scenario defaults (only the sampling rate and the duration are set here).

The orders under test are not hard-coded: they are the strongest orders of each
recording at the receiver (largest RMS of their post-path stems), and the
tachometer resolution and path modes are read from the scenario's
configuration, so that the tests follow the scenario defaults.

The bounds are regression guards set with a margin above what the current code
measures (``validation/stress.md`` lists the measured values); they are not the
tolerances of ROADMAP section 5.4. Where a method degrades for a physical
reason (the path has memory, the speed ripple is faster than a smoothed
tachometer, a clipped recorder distorts), the test asserts that the degradation
stays bounded and the document explains it. Run with ``-s`` to print the
measured numbers.

Conventions: order maps and slices are computed with ``amplitude="peak"`` and
compared with peak truth; levels are compared in dB, speeds as relative
errors.
"""

import warnings

import numpy as np
import pytest

from rota.extract import order_waveform, remove_orders, tracked_harmonics
from rota.maps import rpm_order_map
from rota.metrics import order_residual_energy, track_error, waveform_similarity
from rota.slices import order_slice
from rota.speed import rpm_from_ridge, tacho_to_rpm
from rota.synth import measurement_chain, synth_scenario

# Sampling rates: the default tachometers need at least 2 samples per pulse at top speed
# (60 pulses per revolution up to the ICE limiter at 32 kHz; the EV runs faster, so it
# keeps the default 48 kHz).
FS_ICE = 32000.0
FS_EV = 48000.0
FS_CRUISE = 16000.0


def _db(a, b):
    return 20.0 * np.log10(np.asarray(a, dtype=float) / np.asarray(b, dtype=float))


def _q(e, p):
    e = np.abs(np.asarray(e, dtype=float))
    e = e[np.isfinite(e)]
    return float(np.percentile(e, p))


def _report(label, **values):
    parts = ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in values.items())
    print(f"\n[stress] {label}: {parts}")


@pytest.fixture(scope="module")
def ice():
    return synth_scenario("ice_runup", FS_ICE, 10.0, seed=101)


@pytest.fixture(scope="module")
def ev():
    return synth_scenario("ev_runup", FS_EV, 6.0, seed=102)


@pytest.fixture(scope="module")
def cruise():
    return synth_scenario("steady_cruise", FS_CRUISE, 4.0, seed=103)


def _kind(name, ice, ev, cruise=None):
    return {"ice": ice, "ev": ev, "cruise": cruise}[name]


def _ppr(s):
    return int(s.config["parameters"]["tacho"]["ppr"])


def _delay(s):
    return float(s.config["resolved"]["tacho_delay"])


_SNR = {}
_MAPS = {}


def _xmap(s, res):
    """Order map (peak) of the scenario's recording with the true speed, cached per signal."""
    key = (id(s), res)
    if key not in _MAPS:
        _MAPS[key] = rpm_order_map(s.x, s.fs, s.rpm, order_resolution=res, amplitude="peak")
    return _MAPS[key]


def _band_snr(s, res=0.125):
    """Median in-band SNR in dB of every source order against the rest of the recording.

    Order-map slices (true speed) of the post-path order stems and of the non-order floor
    (everything else in the recording), median over interior frames. Cached per signal.
    """
    key = id(s)
    if key not in _SNR:
        m_o = rpm_order_map(s.through_path(s.order_stems.sum(axis=1)), s.fs, s.rpm, order_resolution=res,
                            amplitude="peak")
        m_f = rpm_order_map(_order_floor(s), s.fs, s.rpm, order_resolution=res, amplitude="peak")
        ok = _interior(s, m_f.t)
        _SNR[key] = np.array([
            float(np.median(_db(order_slice(m_o, o).amplitude[ok], order_slice(m_f, o).amplitude[ok])))
            for o in s.orders
        ])
    return _SNR[key]


def _visible(s, n=3):
    """Columns of the ``n`` source orders with the highest in-band SNR, best first."""
    return [int(k) for k in np.argsort(_band_snr(s))[::-1][:n]]


def _low_modes(s, fmax=500.0):
    return [float(m["freq"]) for m in s.config["parameters"]["path"]["modes"] if float(m["freq"]) < fmax]


def _tacho_rpm(s, *, compensate=True, smooth=1.0):
    """Speed from the scenario's tachometer; ``compensate`` removes the conditioning delay."""
    d = _delay(s) if compensate else 0.0
    return tacho_to_rpm(s.tacho, s.fs, ppr=_ppr(s), smooth=smooth, t_out=s.t + d)


def _interior(s, t, margin=0.3, start=0.5):
    return (np.asarray(t) > start) & (np.asarray(t) < s.t[-1] - margin)


def _away_from_shifts(s, t, half_width=0.6):
    keep = np.ones(np.shape(t), dtype=bool)
    for ts in s.events["shift_times"]:
        keep &= np.abs(np.asarray(t) - ts) > half_width
    return keep


def _order_floor(s):
    """The recording without its source orders: what a perfect removal would leave."""
    return s.x - s.through_path(s.order_stems.sum(axis=1))


# --------------------------------------------------------------------------
# 1. Speed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["ice", "ev", "cruise"])
def test_tacho_speed_against_mean_and_instantaneous_speed(which, ice, ev, cruise):
    s = _kind(which, ice, ev, cruise)
    keep = _interior(s, s.t, margin=0.2, start=0.2)
    rpm = _tacho_rpm(s)
    e_mean = track_error(rpm[keep], s.rpm_mean[keep])
    e_inst = track_error(rpm[keep], s.rpm[keep])
    ripple = float(np.median(np.abs(s.rpm[keep] / s.rpm_mean[keep] - 1.0)))
    fine = _tacho_rpm(s, smooth=0.25)
    e_fine = track_error(fine[keep], s.rpm[keep])
    _report(
        f"tacho {which} (smooth 1 rev, ppr {_ppr(s)})", med_vs_mean_pct=100 * e_mean["median_rel_error"],
        max_vs_mean_pct=100 * e_mean["max_rel_error"], med_vs_inst_pct=100 * e_inst["median_rel_error"],
        ripple_med_pct=100 * ripple, smooth025_med_vs_inst_pct=100 * e_fine["median_rel_error"],
    )
    # One-revolution smoothing averages the angle-periodic ripple away: the result is the mean
    # speed to better than 0.05 % (median), and differs from the instantaneous speed by about
    # the ripple itself.
    assert e_mean["median_rel_error"] < 5e-4
    assert e_inst["median_rel_error"] < ripple + 5e-4
    if ripple > 1e-3:
        # A firing ripple of 0.3 % at order 2: a quarter-revolution fit follows it.
        assert e_inst["median_rel_error"] > 0.5 * ripple
        assert e_fine["median_rel_error"] < 0.5 * e_inst["median_rel_error"]


def test_tacho_delay_compensation_effect_on_order_levels(ice):
    """The 1.5 ms conditioning delay is absorbed by the demodulation; levels barely change."""
    s = ice
    cols = _visible(s, 3)
    orders = s.orders[cols]
    true = tracked_harmonics(s.x, s.fs, s.rpm, orders, window_revs=8)
    rpm_late = _tacho_rpm(s, compensate=False)
    late = tracked_harmonics(s.x, s.fs, rpm_late, orders, window_revs=8)
    comp = tracked_harmonics(s.x, s.fs, rpm_late, orders, window_revs=8, delay=_delay(s))
    keep = _interior(s, s.t, margin=0.5, start=1.0) & _away_from_shifts(s, s.t)

    def p95(a, b):
        return [_q(_db(np.abs(a.envelopes[keep, j]), np.abs(b.envelopes[keep, j])), 95) for j in range(3)]

    lv_late, lv_comp, lv_delay = p95(late, true), p95(comp, true), p95(late, comp)
    stem = s.through_path(s.order_stems[:, cols[0]])
    trim = int(1.0 * s.fs)
    c_late = waveform_similarity(late.waveforms[:, 0], stem, trim=trim)
    c_comp = waveform_similarity(comp.waveforms[:, 0], stem, trim=trim)
    _report(
        f"tacho delay (ice, W 8, orders {' '.join(f'{o:g}' for o in orders)})",
        p95_db_late_vs_true_speed=" ".join(f"{v:.3f}" for v in lv_late),
        p95_db_comp_vs_true_speed=" ".join(f"{v:.3f}" for v in lv_comp),
        p95_db_late_vs_comp=" ".join(f"{v:.3f}" for v in lv_delay),
        corr_dominant_late=c_late["corr"], corr_dominant_comp=c_comp["corr"],
    )
    # The delay shifts the angle by rpm/60 * delay, which varies only as fast as the speed: the
    # demodulation absorbs it into the complex coefficient.
    assert max(lv_delay) < 0.5
    assert lv_comp[0] < 0.3 and max(lv_comp) < 1.5
    assert c_comp["corr"] > 0.99 and c_late["corr"] > 0.99


ENGINE_COMB = (2, 4, 6)  # four-stroke, N = 4 cylinders: (N/2, N, 3N/2), as rpm_from_ridge recommends
COUNTER_COMB = (1, 2, 3, 4, 6)  # not recommended: the odd orders are weak, the comb locks to 2x


def _ridge_settings(s, orders):
    # rpm_from_ridge defaults otherwise; range: the expected speeds with a 30 % margin.
    lo, hi = float(s.rpm_mean.min()), float(s.rpm_mean.max())
    return {"rpm_range": (0.7 * lo, 1.3 * hi), "orders": tuple(orders)}


def _motor_comb(s):
    """The motor order family of the electric drive (from the scenario configuration)."""
    return tuple(float(e["order"]) for e in s.config["parameters"]["motor"]["list"])


def _ridge_run(s, label, orders):
    kw = _ridge_settings(s, orders)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = rpm_from_ridge(s.x, s.fs, **kw)
    truth = np.interp(r.t, s.t, s.rpm_mean)
    e = track_error(r.rpm, truth, at_bound=r.at_bound)
    bad = np.abs(r.rpm / truth - 1.0) > 0.03
    # A frame is flagged when it is in at_bound or its confidence is below 0.1.
    flagged = np.asarray(r.at_bound, dtype=bool) | (np.asarray(r.confidence) < 0.1)
    out = {
        "within3": e["within"][0.03], "median_err_pct": 100 * e["median_rel_error"],
        "at_bound": e["bound_fraction"], "wrong_and_unflagged": float(np.mean(bad & ~flagged)),
        "confidence_median": float(np.median(r.confidence)), "reliable": bool(r.quality["reliable"]),
        "warnings": sorted({type(x.message).__name__ for x in w}),
    }
    _report(
        f"rpm_from_ridge {label} (comb {kw['orders']}, range {kw['rpm_range'][0]:.0f}-{kw['rpm_range'][1]:.0f})",
        within3=out["within3"], median_err_pct=out["median_err_pct"], at_bound=out["at_bound"],
        wrong_and_unflagged=out["wrong_and_unflagged"], confidence_median=out["confidence_median"],
        reliable=out["reliable"], warnings=",".join(out["warnings"]) or "none",
    )
    return out


@pytest.mark.parametrize("which", ["ice", "cruise"])
def test_tacholess_speed_engine_with_the_recommended_comb(which, ice, cruise):
    s = {"ice": ice, "cruise": cruise}[which]
    r = _ridge_run(s, which, ENGINE_COMB)
    assert r["within3"] >= 0.9
    assert r["wrong_and_unflagged"] <= 0.05
    assert r["reliable"]


def test_tacholess_speed_engine_counter_example_comb(ice):
    """Orders 1 to 6 (not recommended): worse, and the estimator says so."""
    r = _ridge_run(ice, "ice counter-example", COUNTER_COMB)
    if r["within3"] < 0.9:
        assert not r["reliable"]
        assert "RidgeQualityWarning" in r["warnings"]


def test_tacholess_speed_electric_drive_flags_its_failure(ev):
    """The weak cabin lines of the electric drive cannot be tracked, but not silently."""
    r = _ridge_run(ev, "ev", _motor_comb(ev))
    assert not r["reliable"]
    assert "RidgeQualityWarning" in r["warnings"]
    assert r["wrong_and_unflagged"] <= 0.05


# --------------------------------------------------------------------------
# 2. Order levels: order map slices against the truth after the path
# --------------------------------------------------------------------------

def _slice_levels(s, k, res):
    o = float(s.orders[k])
    m = _xmap(s, res)
    stem = s.through_path(s.order_stems[:, k])
    ms = rpm_order_map(stem, s.fs, s.rpm, order_resolution=res, amplitude="peak")
    return m, order_slice(m, o).amplitude, order_slice(ms, o).amplitude


RES = {"ice": 0.125, "ev": 0.125}


@pytest.mark.parametrize("which,rank", [("ice", 0), ("ice", 1), ("ice", 2), ("ev", 0), ("ev", 1), ("ev", 2)])
def test_order_slice_against_stem_and_quasi_static_truth(which, rank, ice, ev):
    s = _kind(which, ice, ev)
    k = _visible(s, 3)[rank]
    res = RES[which]
    m, a, b = _slice_levels(s, k, res)
    qs = np.interp(m.t, s.t, s.quasi_static_amplitude(k))
    ok = _interior(s, m.t) & (qs > 0)
    calm = ok & _away_from_shifts(s, m.t)
    e_stem = _db(a[ok], b[ok])  # interference of everything else in the recording
    e_qs = _db(b[calm], qs[calm])  # memory of the path (modes, reverberant tail)
    f = s.orders[k] * m.rpm / 60.0
    near = np.zeros(m.t.shape, dtype=bool)
    for f0 in _low_modes(s):
        near |= np.abs(f - f0) < 15.0
    near &= calm
    e_mode = _db(b[near], qs[near])
    # Everything else in the band adds in power: the slice reads sqrt(stem^2 + rest^2).
    mr = rpm_order_map(s.x - s.through_path(s.order_stems[:, k]), s.fs, s.rpm, order_resolution=res,
                       amplitude="peak")
    rest = order_slice(mr, s.orders[k]).amplitude
    e_add = 10.0 * np.log10(np.mean(a[ok] ** 2) / np.mean(b[ok] ** 2 + rest[ok] ** 2))
    snr = _db(b[ok], rest[ok])
    clear = snr > 10.0
    e_clear = e_stem[clear] if clear.sum() >= 10 else np.array([0.0])
    _report(
        f"order slice {which} order {s.orders[k]:g} (res {res})", band_snr_median_db=float(np.median(snr)),
        vs_stem_median_db=_q(e_stem, 50), vs_stem_p95_db=_q(e_stem, 95),
        snr_above_10db_frames=float(np.mean(clear)), vs_stem_p95_db_snr_above_10db=_q(e_clear, 95),
        mean_power_vs_sum_db=float(e_add),
        stem_vs_qs_median_db=_q(e_qs, 50), stem_vs_qs_p95_db=_q(e_qs, 95),
        near_mode_max_db=_q(e_mode, 100) if near.any() else float("nan"), n_near_mode=int(near.sum()),
        low_modes_hz=" ".join(f"{f0:g}" for f0 in _low_modes(s)),
    )
    # On average the band holds the power of the order plus that of everything else (the cross
    # term averages out over frames; frame by frame it does not when the SNR is near 0 dB).
    assert abs(e_add) < 0.5
    # Where the order stands 10 dB above the rest of its band, the slice reads the order.
    assert _q(e_clear, 50) < 1.0 and _q(e_clear, 95) < 2.5
    # The quasi-static amplitude neglects the memory of modes and cabin: bounded, larger for fast
    # sweeps and high frequencies (stress.md).
    if which == "ice":
        assert _q(e_qs, 50) < 1.0 and _q(e_qs, 95) < 4.0
    else:  # orders up to several kHz sweeping through the fine structure of the cabin response
        assert _q(e_qs, 50) < 3.0 and _q(e_qs, 95) < 12.0


def test_half_orders_need_a_fine_order_map(ice):
    """With 0.25 orders per bin the half orders 1.5 and 2.5 leak into the band of order 2."""
    s = ice
    k2 = int(np.flatnonzero(s.orders == 2.0)[0])
    _, a1, b1 = _slice_levels(s, k2, 0.125)
    _, a2, b2 = _slice_levels(s, k2, 0.25)
    e1, e2 = _db(a1, b1), _db(a2, b2)
    _report("order 2 slice vs stem by resolution", max_db_res0125=_q(e1, 100), max_db_res025=_q(e2, 100),
            p95_res0125=_q(e1, 95), p95_res025=_q(e2, 95))
    assert _q(e1, 95) < 3.0
    assert _q(e2, 100) > _q(e1, 100)


# --------------------------------------------------------------------------
# 3. Extraction and removal
# --------------------------------------------------------------------------

# Vold-Kalman bandwidth (Hz) and tracked-harmonics window (revolutions) per kind.
EXTRACT = {"ice": {"bandwidth": 2.0, "window_revs": 8.0}, "ev": {"bandwidth": 4.0, "window_revs": 8.0}}
CORR_FLOOR = {"ice": 0.98, "ev": 0.25}


@pytest.fixture(scope="module")
def extraction(ice, ev):
    out = {}
    for which, s in (("ice", ice), ("ev", ev)):
        cfg = EXTRACT[which]
        cols = _visible(s, 3)
        orders = s.orders[cols]
        stems = np.column_stack([s.through_path(s.order_stems[:, k]) for k in cols])
        # Envelope reference: the same extractor, true speed, applied to the orders' own stems.
        ref_vk = order_waveform(stems.sum(axis=1), s.fs, s.rpm, orders, bandwidth=cfg["bandwidth"])
        ref_th = tracked_harmonics(stems.sum(axis=1), s.fs, s.rpm, orders, window_revs=cfg["window_revs"])
        for sp, rpm in (("true", s.rpm), ("tacho", _tacho_rpm(s))):
            vk = order_waveform(s.x, s.fs, rpm, orders, bandwidth=cfg["bandwidth"])
            th = tracked_harmonics(s.x, s.fs, rpm, orders, window_revs=cfg["window_revs"])
            out[which, sp] = {"vk": vk, "th": th, "ref_vk": ref_vk, "ref_th": ref_th, "stems": stems,
                              "orders": orders}
    return out


@pytest.mark.parametrize("which", ["ice", "ev"])
@pytest.mark.parametrize("speed", ["true", "tacho"])
@pytest.mark.parametrize("method", ["vk", "th"])
def test_extracted_orders_against_post_path_stems(which, speed, method, ice, ev, extraction):
    s = _kind(which, ice, ev)
    r = extraction[which, speed]
    trim = int(1.0 * s.fs)
    keep = _interior(s, s.t, margin=1.0, start=1.0) & _away_from_shifts(s, s.t, 0.4)
    rows = []
    for j, o in enumerate(r["orders"]):
        sim = waveform_similarity(r[method].waveforms[:, j], r["stems"][:, j], mixture=s.x, trim=trim)
        env = _db(np.abs(r[method].envelopes[keep, j]), np.abs(r["ref_" + method].envelopes[keep, j]))
        rows.append((float(o), sim["corr"], sim["nrmse"], _q(env, 50), _q(env, 95), sim["baseline_corr"]))
    _report(
        f"extract {which} {method} {speed} speed", worst_corr=min(x[1] for x in rows),
        worst_nrmse=max(x[2] for x in rows), worst_env_median_db=max(x[3] for x in rows),
        worst_env_p95_db=max(x[4] for x in rows),
        per_order=" ".join(f"{o:g}:{c:.4f}/{m:.3f}/{p:.3f}(raw {b:.3f})" for o, c, _, m, p, b in rows),
    )
    # The order with the best in-band SNR: close to the ideal case. The others: bounded by the
    # noise in the extractor's band (stress.md), and always far closer to the stem than the raw
    # recording is.
    _, corr0, _, med0, p950, _ = rows[0]
    assert corr0 > CORR_FLOOR[which]
    if which == "ice":  # in-band SNR about 15 dB
        assert med0 < 0.5 and p950 < 2.0
    for o, corr, nrmse, med, p95, base in rows:
        # The extracted order is always closer to its stem than the raw recording is; for the
        # noise-dominated orders (in-band SNR at or below 0 dB) that is the only guard.
        assert corr > base + 0.05 * (1.0 - base), o
        assert med < 10.0, o


@pytest.mark.parametrize("which", ["ice", "ev"])
def test_remove_orders_against_the_non_order_floor(which, ice, ev):
    s = _kind(which, ice, ev)
    floor_signal = _order_floor(s)
    # The tested orders only: the other orders lie at least half an order away, outside the
    # +-0.25-order band of the metric and the main lobe of the 8-revolution window.
    tested = s.orders[_visible(s, 3)]
    y, _ = remove_orders(s.x, s.fs, s.rpm, tested, method="tracked-harmonics", window_revs=8.0)
    res = 0.125
    lo = 0.5 * (float(s.rpm_mean.min()) + float(s.rpm_mean.max())) * 0.4  # leave out the start
    rows = []
    for k in _visible(s, 3):
        o = float(s.orders[k])
        e_res = order_residual_energy(y, s.x, s.fs, s.rpm, o, order_resolution=res, rpm_range=(lo, None))
        e_floor = order_residual_energy(floor_signal, s.x, s.fs, s.rpm, o, order_resolution=res, rpm_range=(lo, None))
        rows.append((o, e_res, e_floor))
    # Controls: the same removal applied to the order stems alone, after the path (every tested
    # order) and at source level (the first one; the extractor itself).
    src = s.order_stems.sum(axis=1)
    post = s.through_path(src)
    y_post, _ = remove_orders(post, s.fs, s.rpm, tested, method="tracked-harmonics", window_revs=8.0)
    y_src, _ = remove_orders(src, s.fs, s.rpm, tested[:1], method="tracked-harmonics", window_revs=8.0)
    e_src = order_residual_energy(y_src, src, s.fs, s.rpm, rows[0][0], order_resolution=res, rpm_range=(lo, None))
    e_post = [order_residual_energy(y_post, post, s.fs, s.rpm, o, order_resolution=res, rpm_range=(lo, None))
              for o, _, _ in rows]
    _report(f"remove_orders {which} (W 8, tested orders; frames above {lo:.0f} rpm)",
            residual_vs_floor=" ".join(f"{o:g}:{a:.1f}/{f:.1f}dB" for o, a, f in rows),
            stems_only_post_path=" ".join(f"{r[0]:g}:{e:.1f}dB" for r, e in zip(rows, e_post)),
            stems_only_source_first_order=e_src)
    # The extractor removes a clean source-level order almost completely ...
    assert e_src < -30.0
    for (o, e_res, e_floor), e_p in zip(rows, e_post):
        # ... and what it leaves of the post-path stems is the path's memory (the reverberant tail
        # of a sweeping order lies off the order line): the residual of the recording is within a
        # few dB of the floor plus that.
        bound = 10.0 * np.log10(10.0 ** (e_floor / 10.0) + 10.0 ** (e_p / 10.0)) + 3.0
        assert e_res < min(bound, 0.0), o


# --------------------------------------------------------------------------
# 4. Measurement impairments
# --------------------------------------------------------------------------

def _slice_error(s, x, rpm, cols, res):
    """Per order, dB difference of the order slice of ``x`` against the ideal recording."""
    out = []
    m = rpm_order_map(x, s.fs, rpm, order_resolution=res, amplitude="peak")
    m0 = _xmap(s, res)
    for k in cols:
        a, b = order_slice(m, s.orders[k]).amplitude, order_slice(m0, s.orders[k]).amplitude
        n = min(a.size, b.size)
        ok = _interior(s, m0.t[:n])
        out.append(_db(a[:n][ok], b[:n][ok]))
    return out


def test_clipping_distorts_the_orders_boundedly(ice):
    s = ice
    cols = _visible(s, 3)
    mag = np.abs(s.x_clean)
    rows = []
    for share in (1e-3, 1e-2):  # clip level set so that 0.1 % and 1 % of the samples clip
        clip = float(np.quantile(mag, 1.0 - share))
        mc = measurement_chain(s.x_clean, s.fs, clip=clip)
        errs = _slice_error(s, mc["x"], s.rpm, cols, 0.125)
        rows.append((share, clip / float(mag.max()), float(np.mean(mc["clipped"])),
                     [(_q(e, 50), _q(e, 95)) for e in errs]))
    _report(f"clipping (ice, orders {' '.join(f'{s.orders[k]:g}' for k in cols)})", rows=" ".join(
        f"{c:.2%} clipped (clip at {lvl:.0%} of peak): median/p95 " + " ".join(f"{m:.3f}/{p:.3f}" for m, p in e)
        for _, lvl, c, e in rows))
    for share, _, clipped, e in rows:
        assert clipped > 0
        if share <= 1e-3:
            assert e[0][1] < 0.2 and max(p for _, p in e) < 1.0
        else:  # bounded distortion, growing with the order
            assert e[0][1] < 1.0 and max(p for _, p in e) < 6.0


def test_quantisation_at_realistic_headroom(ice):
    s = ice
    cols = _visible(s, 3)
    peak = float(np.max(np.abs(s.x_clean)))
    rows = []
    for headroom_db in (20.0, 40.0):
        fsc = peak * 10.0 ** (headroom_db / 20.0)
        mc = measurement_chain(s.x_clean, s.fs, bits=16, full_scale=fsc)
        snr = 20 * np.log10(np.std(s.x_clean) / np.std(mc["x"] - s.x_clean))
        errs = _slice_error(s, mc["x"], s.rpm, cols, 0.125)
        rows.append((headroom_db, snr, max(_q(e, 95) for e in errs), int(mc["clipped"].sum())))
    _report("16-bit quantisation (ice)", rows=" ".join(
        f"headroom {h:.0f} dB: SNR {snr:.1f} dB, worst p95 {p:.4f} dB, clipped {c}" for h, snr, p, c in rows))
    for headroom_db, snr, p95, clipped in rows:
        assert clipped == 0
        assert p95 < 0.1


def test_clock_drift_between_tacho_and_audio(ev):
    """A 50 ppm fast audio clock: error grows with order frequency; t_true corrects it."""
    s = ev
    eps = 50e-6
    mc = measurement_chain(s.x_clean, s.fs, clock_drift_ppm=50.0)
    x, t_true = mc["x"], mc["t_true"]
    top = int(np.argmax(s.orders))
    cols = sorted(set(_visible(s, 3)) | {top}, key=lambda k: s.orders[k])
    res = 0.125
    # Uncorrected: the speed on the tachometer clock is used sample by sample with the audio.
    n = s.n
    raw = _slice_error(s, x[:n], s.rpm, cols, res)
    # Corrected: speed at the true time of every audio sample, in revolutions per nominal second.
    rpm_corr = np.interp(t_true, s.t, s.rpm) * np.gradient(t_true) * s.fs
    m_corr = rpm_order_map(x, s.fs, rpm_corr, order_resolution=res, amplitude="peak")
    m0 = _xmap(s, res)
    t_frames_true = np.interp(m_corr.t * s.fs, np.arange(x.size), t_true)
    ok = _interior(s, m0.t)
    corr = []
    for k in cols:
        a = order_slice(m_corr, s.orders[k]).amplitude
        b = order_slice(m0, s.orders[k]).amplitude
        corr.append(_db(np.interp(m0.t[ok], t_frames_true, a), b[ok]))
    o_top = float(s.orders[top])
    vk_raw = order_waveform(x[:n], s.fs, s.rpm, [o_top], bandwidth=1.0)
    vk_cor = order_waveform(x, s.fs, rpm_corr, [o_top], bandwidth=1.0)
    vk_0 = order_waveform(s.x, s.fs, s.rpm, [o_top], bandwidth=1.0)
    keep = _interior(s, s.t, margin=1.5, start=1.5)
    e_raw = _db(np.abs(vk_raw.envelopes[keep, 0]), np.abs(vk_0.envelopes[keep, 0]))
    env_cor = np.interp(s.t[keep], t_true, np.abs(vk_cor.envelopes[:, 0]))
    e_cor = _db(env_cor, np.abs(vk_0.envelopes[keep, 0]))
    _report(
        "clock drift 50 ppm (ev)", n_extra_samples=int(x.size - n),
        slice_p95_raw=" ".join(f"{s.orders[k]:g}:{_q(e, 95):.3f}" for k, e in zip(cols, raw)),
        slice_p95_corr=" ".join(f"{s.orders[k]:g}:{_q(e, 95):.3f}" for k, e in zip(cols, corr)),
        vk_top_order=o_top, vk_B1_median_raw=_q(e_raw, 50), vk_B1_median_corr=_q(e_cor, 50),
        eps_f_at_top_hz=eps * o_top * s.rpm.max() / 60,
    )
    assert x.size - n == int(np.floor((n - 1) * eps))
    for e in raw:
        assert _q(e, 95) < 1.0
    for e in corr:
        assert _q(e, 95) < 0.05
    assert _q(e_cor, 50) < 0.1
    assert _q(e_raw, 50) > _q(e_cor, 50)


def test_two_channel_recording_with_interchannel_delay():
    s = synth_scenario("ice_runup", FS_ICE, 8.0, seed=104, n_channels=2)
    assert s.x.shape == (s.n, 2)
    k = _visible(s, 1)[0]
    res = 0.125
    m = rpm_order_map(s.x, s.fs, s.rpm, order_resolution=res, amplitude="peak")
    stem = s.through_path(s.order_stems[:, k])
    ms = rpm_order_map(stem, s.fs, s.rpm, order_resolution=res, amplitude="peak")
    a, b = order_slice(m, s.orders[k]).amplitude, order_slice(ms, s.orders[k]).amplitude
    ok = _interior(s, m.t)
    e = [_db(a[ok, c], b[ok, c]) for c in range(2)]
    kw = _ridge_settings(s, ENGINE_COMB)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r_avg = rpm_from_ridge(s.x, s.fs, **kw)
        r_one = rpm_from_ridge(s.x[:, 0], s.fs, **kw)
    w_avg = track_error(r_avg.rpm, np.interp(r_avg.t, s.t, s.rpm_mean))["within"][0.03]
    w_one = track_error(r_one.rpm, np.interp(r_one.t, s.t, s.rpm_mean))["within"][0.03]
    tau = s.config["parameters"]["cabin"]["interchannel_delay"]
    _report(
        f"two channels (ice, order {s.orders[k]:g})", median_db_ch0=_q(e[0], 50), p95_db_ch0=_q(e[0], 95),
        median_db_ch1=_q(e[1], 50), p95_db_ch1=_q(e[1], 95),
        ridge_within3_channel_mean=w_avg, ridge_within3_channel0=w_one, delay_ms=1e3 * tau,
    )
    for ec in e:
        assert _q(ec, 50) < 0.5 and _q(ec, 95) < 4.0
    # Averaging channels is harmless for the engine comb (below 1 / (2 tau)).
    assert abs(w_avg - w_one) < 0.2
