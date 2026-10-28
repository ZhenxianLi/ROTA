"""An electric drive run-up you can listen to: motor orders, gear whine, inverter carrier.

The recording comes from ``rota_order.synth.synth_scenario("ev_runup")``: a synthetic
permanent-magnet motor with four pole pairs behind a single-speed 12:1 reduction, in a
vehicle that accelerates at full load from walking pace to about 110 km/h. The motor
orders are the electromagnetic family 2, 4, 8, 10, 12 and 14 (shaft orders; order 4 is
the electrical frequency ``f_el``), there is a gear-mesh order 6.7, and the gear whine
at order 40 carries a slow amplitude modulation. The inverter carrier sits at 10 kHz,
dithered by up to +-0.8 kHz in hops every 0.2 to 0.5 s, with sidebands at
``carrier +- 2 f_el`` and ``carrier +- 4 f_el``: they move with speed but are not
orders (their frequency is not proportional to the speed), and they stand only 2 to
4 dB above the local noise floor. Two fixed tones (6.3 and 14.2 kHz), a low-frequency
rumble that carries most of the energy, road noise, a road-joint thump, a click, a
modal transfer path and a cabin impulse response complete the recording. The tonal
lines are weak against the broadband, as in a quiet electric vehicle. All of it is
synthetic and comes with exact truth; every scenario parameter is a default of
``synth_scenario`` except the duration and the tachometer pulse count in
``SCENARIO_OVERRIDES``.

Analysis, with the speed from the tachometer channel: ``rpm_frequency_map``
(Campbell diagram) of the full band and, with its own colour scale, of the inverter
region; ``rpm_order_map`` shows the motor order family and the whine, and
``order_slice`` gives the level of every source order against the quasi-static truth
at the receiver; the levels of the carrier and its first sidebands are read from the
Campbell diagram along their true (dithered) frequency tracks, with the local noise
floor subtracted; ``remove_orders`` (tracked harmonics) removes all source orders in
two passes, the whine with short frames and the motor and gear-mesh orders with long
ones. What is left is the inverter, the fixed tones, road noise and rumble: the
inverter sidebands are not orders, so order removal leaves them where they are.

The truth decides what the numbers mean. Most motor orders are weaker than the road
noise inside their own order band (the noise in each band is known from the true
non-order stems), so an order slice there reads the noise, not the order; the slices
are compared with the truth only in frames where the line is ``MIN_SNR_DB`` above that
noise, and the extracted motor orders carry the noise of their bands along. The whine
stands above its band noise in most frames.

Audio: three mono 16-bit WAV files in ``docs/audio/`` (the recording, the residual
after removing the source orders, and the extracted orders: motor and whine),
scaled with the same ``FULL_SCALE_PA`` as ``05_realistic_engine_audio.py`` so that the
levels of all files can be compared directly. The sampling rate is at least 2.2 times
the highest carrier frequency (checked).

Amplitude conventions: maps, slices and levels are RMS in Pa, levels in dB re 20 uPa;
truth amplitudes are peak values (divided by sqrt(2) for comparison). Speed in rpm,
time in s, frequency in Hz.

Run:  python examples/06_electric_drive_audio.py     (needs matplotlib, about 25 s)
Writes docs/images/electric_drive.png and docs/audio/electric_runup*.wav.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import rota_order as ro

P_REF = 20e-6  # reference pressure in Pa (dB re 20 uPa), passed explicitly
FULL_SCALE_PA = 4.0  # sound pressure in Pa that maps to digital full scale in every WAV file
FADE_S = 0.02  # raised-cosine fade-in and fade-out of every WAV file, s
FS = 32000.0  # sampling rate of the scenario and of the WAV files, Hz
DURATION = 8.0  # s (the scenario default is 10 s)
SEED = 3
# At 32 kHz the default 48-pulse tachometer would give pulses shorter than 2 samples
# above 10000 rpm; 24 pulses per revolution keep it valid up to 20000 rpm.
SCENARIO_OVERRIDES = {"tacho": {"ppr": 24}}
# Tracked-harmonics frame lengths in revolutions, one per pass. The whine sweeps through
# the lightly damped 2.1 and 3.3 kHz path modes within a few tenths of a second and is
# removed first with short frames that follow it. The motor and gear-mesh orders are weak
# against the road noise, and the noise inside an order band falls with the band width,
# so they are removed next with long frames (Hann main lobe +-0.125 orders; the orders
# are at least 1.3 orders apart).
WHINE_WINDOW_REVS = 4.0
MOTOR_WINDOW_REVS = 16.0
ORDER_RES = 0.125  # bin spacing of the order map, orders (8 revolutions per frame)
MAX_ORDER_MAP = 44.0  # top of the order map, orders
MIN_SNR_DB = 6.0  # order levels are compared where the line is this far above the band noise
CAMPBELL_HZ = 20.0  # bin spacing of the Campbell diagram, Hz
INVERTER_BAND = (6.0e3, 15.0e3)  # frequency range of the inverter zoom, Hz
MIN_RPM_STATS = 1500.0  # orders and carrier sidebands are compared above this speed
EDGE_S = 0.3  # start of the record left out of the statistics (start-up of the path), s
SHOW_ORDERS = ("motor_12", "gear_mesh_40")  # order slices drawn in panel (e), all are printed
ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = ROOT / "docs" / "images" / "electric_drive.png"
AUDIO_DIR = ROOT / "docs" / "audio"


def to_db(amplitude):
    """Level in dB re P_REF of an amplitude in Pa (RMS in, RMS level out)."""
    return 20.0 * np.log10(np.maximum(amplitude, 1e-12) / P_REF)


def write_wav(name, x, fs):
    """Write ``x`` (Pa, shape (n,) or (n, ch)) as 16-bit PCM, scaled by FULL_SCALE_PA.

    Sample value = round(32767 * p / FULL_SCALE_PA) after a raised-cosine fade of
    FADE_S at both ends; nothing else is applied (no normalisation, no limiter).
    """
    x = np.array(x, dtype=float)
    n_fade = round(FADE_S * fs)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_fade) / n_fade))
    shape = (n_fade,) + (1,) * (x.ndim - 1)
    x[:n_fade] *= ramp.reshape(shape)
    x[-n_fade:] *= ramp[::-1].reshape(shape)
    peak = float(np.max(np.abs(x)))
    if peak >= FULL_SCALE_PA:
        raise ValueError(f"{name}: peak {peak:.3f} Pa would clip; raise FULL_SCALE_PA")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIO_DIR / name
    wavfile.write(path, int(fs), np.round(x / FULL_SCALE_PA * 32767.0).astype(np.int16))
    rms = float(np.sqrt(np.mean(x**2)))
    print(f"  {name:42s} {to_db(rms):5.1f} dB SPL (RMS)   peak {20 * np.log10(peak / FULL_SCALE_PA):6.1f} dBFS"
          f"   {path.stat().st_size / 1e6:.2f} MB")


def track_level(fmap, t, freq):
    """RMS level per frame of a component along its true frequency track, floor removed.

    Power is summed over the bins within two bin spacings of the frequency at the frame
    centre, widened by the distance the component moves within the frame, and divided by
    the window's equivalent noise bandwidth (as ``order_slice`` does for orders). The
    broadband floor in that band is estimated from the median power of the bins 4 to 10
    spacings away on both sides and subtracted (frames where the line is not above it
    give NaN). Returns the level, the centre frequency and the sweep within the frame.

    The floor is that of the recording; a sideband only 2 to 4 dB above it (in a 20 Hz
    band) is read with a few dB of scatter per frame, hence the medians per carrier hop.
    """
    f_c = np.interp(fmap.t, t, freq)
    half_frame = 0.5 * fmap.nperseg / FS
    sweep = np.abs(np.interp(fmap.t + half_frame, t, freq) - np.interp(fmap.t - half_frame, t, freq))
    dist = np.abs(fmap.axis[:, None] - f_c[None, :])
    power = np.abs(fmap.values) ** 2
    band = dist <= 2.0 * fmap.resolution + 0.5 * sweep[None, :]
    ring = (dist >= 4.0 * fmap.resolution + 0.5 * sweep[None, :]) & (
        dist <= 10.0 * fmap.resolution + 0.5 * sweep[None, :])
    floor = np.array([np.median(power[ring[:, i], i]) for i in range(f_c.size)])
    line = np.sum(power * band, axis=0) - floor * band.sum(axis=0)
    return np.sqrt(np.where(line > 0, line, np.nan) / fmap.enbw_bins), f_c, sweep


def main():
    plt.switch_backend("Agg")  # file output only; no display needed
    sc = ro.synth.synth_scenario("ev_runup", fs=FS, duration=DURATION, seed=SEED, **SCENARIO_OVERRIDES)
    par = sc.config["parameters"]
    p = int(par["pole_pairs"])
    fc_track = np.asarray(sc.carrier["carrier_hz"])
    if FS < 2.2 * fc_track.max():
        raise ValueError(f"FS = {FS:g} Hz is below 2.2 x the highest carrier frequency {fc_track.max():g} Hz")
    ppr = int(par["tacho"]["ppr"])
    delay = float(sc.config["resolved"]["tacho_delay"])
    hops = sc.events["carrier_hop_hz"]
    print(f"Scenario ev_runup: {DURATION:g} s at {FS:g} Hz, seed {SEED}; "
          f"{sc.rpm_mean.min():.0f} -> {sc.rpm_mean.max():.0f} rpm, "
          f"{sc.speed_kmh[0]:.0f} -> {sc.speed_kmh[-1]:.0f} km/h; {p} pole pairs")
    print(f"  inverter carrier {par['inverter']['carrier_hz']:g} Hz, {hops.size} hops between "
          f"{hops.min():g} and {hops.max():g} Hz; fixed tones "
          + ", ".join(f"{tone['freq']:g} Hz" for tone in par["tones"]))
    print("  source orders: " + ", ".join(f"{o:g} ({n})" for o, n in zip(sc.orders, sc.order_names)))

    rpm = ro.speed.tacho_to_rpm(sc.tacho, FS, ppr=ppr, t_out=sc.t + delay)
    rel = np.abs(rpm / sc.rpm_mean - 1.0)
    print(f"Speed from {ppr}-pulse tacho (delay {delay * 1e3:g} ms compensated) against the true mean speed: "
          f"median {100 * np.median(rel):.3f} %, max {100 * rel[sc.t > EDGE_S].max():.2f} %")

    # ---- order slices against the truth ------------------------------------------------------
    # The lines are weak: in most order bands the road noise is stronger than the order
    # itself. The noise in each band is known (order map of the true non-order content),
    # so the slices are compared with the truth only where the line stands MIN_SNR_DB above it.
    whine = sc.stem_sum(["motor", "gear"])
    rest = sc.x_clean - whine  # inverter, tones, transients, road and rumble
    okw = {"order_resolution": ORDER_RES, "max_order": MAX_ORDER_MAP, "amplitude": "rms"}
    omap = ro.maps.rpm_order_map(sc.x, FS, rpm, **okw)
    omap_rest = ro.maps.rpm_order_map(rest, FS, rpm, **okw)
    show = {}  # name -> (order, slice, truth RMS, band noise)
    print(f"Order slices ({ORDER_RES:g} order bins) against the quasi-static truth at the receiver (RMS, dB), "
          f"frames after {EDGE_S:g} s above {MIN_RPM_STATS:g} rpm:")
    for k, name in enumerate(sc.order_names):
        s = ro.slices.order_slice(omap, sc.orders[k])
        noise = ro.slices.order_slice(omap_rest, sc.orders[k]).amplitude
        truth = sc.quasi_static_amplitude(k) / np.sqrt(2.0)  # peak -> RMS
        tr = np.interp(s.t, sc.t, truth)
        err, snr = 20.0 * np.log10(s.amplitude / tr), 20.0 * np.log10(tr / noise)
        ok = (s.t > EDGE_S) & (s.rpm > MIN_RPM_STATS) & np.isfinite(err)
        good = ok & (snr > MIN_SNR_DB)
        stat = (f"median |error| {np.median(np.abs(err[good])):.2f} dB" if good.sum() >= 5 else "too few frames")
        print(f"  order {sc.orders[k]:4g} ({name:13s}): line re band noise {np.median(snr[ok]):5.1f} dB (median); "
              f"{100 * good.sum() / ok.sum():3.0f} % of frames {MIN_SNR_DB:g} dB above it, there {stat}")
        show[name] = (sc.orders[k], s, truth, noise)

    # ---- inverter carrier and sidebands from the Campbell diagram ------------------------------
    fmap = ro.maps.rpm_frequency_map(sc.x, FS, rpm, resolution=CAMPBELL_HZ, amplitude="rms")
    tg = np.arange(0.0, DURATION, 0.01)
    q = np.asarray(sc.carrier["q"])
    # Lines that must stay out of the band and the floor ring: orders, tones, other inverter lines.
    others_t = [np.outer(sc.rpm_mean, sc.orders) / 60.0, np.tile([tn["freq"] for tn in par["tones"]], (sc.n, 1)),
                sc.carrier["frequency"]]
    hop_t = np.asarray(sc.events["carrier_hop_times"])
    carrier_rows = {}
    print("Inverter components from the Campbell diagram against the truth (RMS, dB, floor removed), "
          f"frames above {MIN_RPM_STATS:g} rpm without a carrier hop or another line nearby:")
    for j in np.argsort(q):
        freq = sc.carrier["frequency"][:, j]
        gain = np.abs(sc.path_response(np.interp(tg, sc.t, freq)))
        truth = sc.carrier["amplitude"][:, j] * np.interp(sc.t, tg, gain) / np.sqrt(2.0)
        level, f_c, sweep = track_level(fmap, sc.t, freq)
        near = np.zeros(fmap.t.size, dtype=bool)
        for o in others_t:
            o_c = np.column_stack([np.interp(fmap.t, sc.t, o[:, i]) for i in range(o.shape[1])])
            d = np.abs(o_c - f_c[:, None])
            near |= np.any((d > 0.5) & (d < 12.0 * fmap.resolution), axis=1)
        ok = (~near & (sweep < fmap.resolution) & (fmap.rpm > MIN_RPM_STATS) & (fmap.t > EDGE_S)
              & np.isfinite(level))
        err = 20.0 * np.log10(level / np.interp(fmap.t, sc.t, truth))
        # Median per carrier hop (the dither keeps the carrier still between hops).
        seg = np.searchsorted(hop_t, fmap.t, side="right") - 1
        hop_med = []
        for h in range(hop_t.size):
            m = ok & (seg == h)
            if m.sum() >= 3:
                hop_med.append((np.median(fmap.t[m]), np.median(to_db(level[m])),
                                np.median(to_db(np.interp(fmap.t[m], sc.t, truth)))))
        hop_med = np.array(hop_med)
        label = "carrier" if q[j] == 0 else f"carrier {'+' if q[j] > 0 else '-'} {abs(q[j]) / p:g} f_el"
        print(f"  {label:17s} truth {np.median(to_db(truth[sc.rpm_mean > MIN_RPM_STATS])):4.1f} dB (median); "
              f"median |error| per frame {np.nanmedian(np.abs(err[ok])):.2f} dB ({int(ok.sum())} frames), "
              f"per hop {np.median(np.abs(hop_med[:, 1] - hop_med[:, 2])):.2f} dB ({len(hop_med)} hops)")
        carrier_rows[q[j]] = (label, hop_med)

    # ---- removal of the motor-shaft orders ---------------------------------------------------
    k_whine = sc.order_names.index(f"gear_mesh_{par['gear_mesh']['teeth']:g}")
    others = np.delete(sc.orders, k_whine)
    res_1, comp_1 = ro.extract.remove_orders(
        sc.x, FS, rpm, [sc.orders[k_whine]], method="tracked-harmonics", window_revs=WHINE_WINDOW_REVS
    )
    residual, comp_2 = ro.extract.remove_orders(
        res_1, FS, rpm, others, method="tracked-harmonics", window_revs=MOTOR_WINDOW_REVS
    )
    extracted = comp_1.total + comp_2.total
    sel = sc.t > EDGE_S
    sim_w = ro.metrics.waveform_similarity(extracted[sel], whine[sel])
    sim_1 = ro.metrics.waveform_similarity(comp_1.total[sel], sc.through_path(sc.order_stems[:, k_whine])[sel])
    sim_r = ro.metrics.waveform_similarity(residual[sel], rest[sel])
    print(f"remove_orders (tracked harmonics): order {sc.orders[k_whine]:g} with {WHINE_WINDOW_REVS:g} revolutions, "
          f"then the other {others.size} orders with {MOTOR_WINDOW_REVS:g}:")
    print(f"  extracted whine against its true stem (order {sc.orders[k_whine]:g} through the path): "
          f"corr {sim_1['corr']:.4f}, nrmse {sim_1['nrmse']:.3f}")
    print(f"  all extracted orders against the true motor and gear stems: corr {sim_w['corr']:.4f}, "
          f"nrmse {sim_w['nrmse']:.3f} (the weak motor orders take the road noise of their bands along)")
    print(f"  residual against the true inverter, tone, road, rumble and transient stems: "
          f"corr {sim_r['corr']:.4f}, nrmse {sim_r['nrmse']:.3f}")
    for o in (2.0, 12.0, 40.0):
        e = [ro.metrics.order_residual_energy(r[sel], sc.x[sel], FS, rpm[sel], o, order_resolution=ORDER_RES)
             for r in (residual, rest)]
        print(f"  energy left at order {o:g}: {e[0]:.1f} dB re the recording (true non-order content {e[1]:.1f} dB)")

    print(f"WAV files (16-bit PCM, {int(FS)} Hz, mono, full scale = {FULL_SCALE_PA:g} Pa):")
    write_wav("electric_runup.wav", sc.x, FS)
    write_wav("electric_runup_motor_orders_removed.wav", residual, FS)
    write_wav("electric_runup_whine_only.wav", extracted, FS)

    # ---- figure ------------------------------------------------------------------------------
    fmap_res = ro.maps.rpm_frequency_map(residual, FS, rpm, resolution=CAMPBELL_HZ, amplitude="rms")
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig = plt.figure(figsize=(8.0, 10.6), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.8])
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ax_e, ax_f = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    white = {"color": "w", "fontsize": 7}

    # (a), (b): full band. The rumble below 100 Hz is far stronger than any line, so the
    # colour scale is set from the levels above 300 Hz (the lines then stand out).
    db_a, db_b = to_db(fmap.magnitude), to_db(fmap_res.magnitude)
    hi = fmap.axis > 300.0
    vmax = np.ceil(np.percentile(db_a[hi], 99.95) / 5.0) * 5.0
    kw = {"cmap": "viridis", "vmin": vmax - 45.0, "vmax": vmax, "shading": "nearest", "rasterized": True}
    ax_a.pcolormesh(fmap.t, fmap.axis / 1e3, db_a, **kw)
    im = ax_b.pcolormesh(fmap_res.t, fmap_res.axis / 1e3, db_b, **kw)
    cb = fig.colorbar(im, ax=[ax_a, ax_b], location="bottom", fraction=0.05, pad=0.02, extend="both")
    cb.set_label(f"Level per bin, dB re 20 µPa (RMS), {CAMPBELL_HZ:g} Hz bins")
    ax_a.set_title("(a) Campbell diagram of the recording", loc="left")
    ax_b.set_title("(b) After remove_orders (all source orders)", loc="left")
    t_lab = DURATION - 0.15
    r_lab = float(np.interp(t_lab, sc.t, sc.rpm_mean))
    ax_a.text(t_lab, 40.0 * r_lab / 60e3 + 0.25, "whine (order 40)", ha="right", va="bottom", **white)
    ax_a.text(t_lab, 14.0 * r_lab / 60e3 + 0.25, "motor orders", ha="right", va="bottom", **white)
    for ax in (ax_a, ax_b):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_xlim(0.0, DURATION)
        ax.set_ylim(0.0, 0.5 * FS / 1e3)
        ax.text(0.25, 10.0 + 1.0, "inverter carrier (dithered)\nand sidebands, see (d)", va="bottom", **white)
        for tone in par["tones"]:
            ax.text(0.25, tone["freq"] / 1e3 - 0.25, f"fixed tone {tone['freq'] / 1e3:g} kHz", va="top", **white)
    ax_b.text(t_lab, 40.0 * r_lab / 60e3 + 0.25, "orders removed", ha="right", va="bottom", **white)

    # (c): order map of the recording, colour scale from the orders above 1.
    db_c = to_db(omap.magnitude)
    vmax_c = np.ceil(np.percentile(db_c[omap.axis > 1.0], 99.9) / 5.0) * 5.0
    im_c = ax_c.pcolormesh(omap.t, omap.axis, db_c, cmap="viridis", vmin=vmax_c - 35.0, vmax=vmax_c,
                           shading="nearest", rasterized=True)
    cb_c = fig.colorbar(im_c, ax=ax_c, location="bottom", fraction=0.05, pad=0.02, extend="both")
    cb_c.set_label(f"Level per bin, dB re 20 µPa (RMS), {ORDER_RES:g} order bins")
    ax_c.set_title("(c) Order map: motor orders and whine", loc="left")
    ax_c.set_ylabel("Order (cycles per revolution)")
    ax_c.set_ylim(0.0, MAX_ORDER_MAP)
    ax_c.set_facecolor(plt.get_cmap("viridis")(0.0))  # the frames start once the speed allows
    for o in sc.orders:  # labels in the right margin, on the line
        ax_c.text(DURATION + 0.08, o, f"{o:g}", ha="left", va="center", fontsize=6.5, clip_on=False)
    ax_c.text(0.7, 36.0, "order 40: gear whine\n(slow amplitude modulation)", va="top", **white)
    ax_c.text(0.7, 24.0, "2, 4, 8, 10, 12, 14: motor\n6.7: gear mesh", va="top", **white)

    # (d): zoom on the inverter region, colour scale from that band alone.
    zb = (fmap.axis >= INVERTER_BAND[0]) & (fmap.axis <= INVERTER_BAND[1])
    db_d = db_a[zb]
    lo_d, hi_d = np.percentile(db_d, 5.0), np.percentile(db_d, 99.9)
    im_d = ax_d.pcolormesh(fmap.t, fmap.axis[zb] / 1e3, db_d, cmap="viridis", vmin=np.floor(lo_d),
                           vmax=np.ceil(hi_d), shading="nearest", rasterized=True)
    cb_d = fig.colorbar(im_d, ax=ax_d, location="bottom", fraction=0.05, pad=0.02, extend="both")
    cb_d.set_label(f"Level per bin, dB re 20 µPa (RMS), {CAMPBELL_HZ:g} Hz bins")
    ax_d.set_title("(d) Inverter region of (a), own colour scale", loc="left")
    ax_d.set_ylabel("Frequency (kHz)")
    ax_d.set_ylim(INVERTER_BAND[0] / 1e3, INVERTER_BAND[1] / 1e3)
    for qq, txt in ((0.0, "carrier"), (2.0 * p, "+2 $f_{el}$"), (-2.0 * p, "-2 $f_{el}$"),
                    (4.0 * p, "+4 $f_{el}$"), (-4.0 * p, "-4 $f_{el}$")):
        j = int(np.flatnonzero(q == qq)[0])  # labels in the right margin, at the last frequency
        ax_d.text(DURATION + 0.08, sc.carrier["frequency"][-1, j] / 1e3, txt, ha="left", va="center",
                  fontsize=6.5, clip_on=False)
    for tone in par["tones"]:
        ax_d.text(0.25, tone["freq"] / 1e3 + 0.12, f"fixed tone {tone['freq'] / 1e3:g} kHz", va="bottom",
                  **white)
    t_w = DURATION - 0.8
    f_w = float(np.interp(t_w, sc.t, sc.order_frequency[:, k_whine])) / 1e3
    ax_d.annotate("whine (order 40)", (t_w, f_w), xytext=(t_w - 1.0, f_w + 0.9), ha="center", va="bottom",
                  arrowprops={"arrowstyle": "->", "lw": 0.6, "color": "w"}, **white)
    for c_ax in (ax_c, ax_d):
        c_ax.set_xlabel("Time (s)")
        c_ax.set_xlim(0.0, DURATION)

    # (e): order slices against the truth, with the noise in the same order band.
    colours = {"motor_12": "#0072B2", "gear_mesh_40": "#D55E00"}
    for name in SHOW_ORDERS:
        o, s, truth, noise = show[name]
        ax_e.plot(s.t, to_db(s.amplitude), color=colours[name], lw=1.6, label=f"order {o:g} ({name})")
        ax_e.plot(sc.t, to_db(truth), color="k", lw=0.8, ls=(0, (4, 3)),
                  label="truth (quasi-static)" if name == SHOW_ORDERS[0] else None)
        ax_e.plot(s.t, to_db(noise), color=colours[name], lw=0.7, ls=(0, (1, 1.5)),
                  label="band noise (truth, dotted)" if name == SHOW_ORDERS[0] else None)
    ax_e.set_title("(e) Order slices against the truth", loc="left")
    ax_e.set_ylabel("Order level, dB re 20 µPa (RMS)")
    top = max(np.nanmax(to_db(show[n][1].amplitude[show[n][1].t > EDGE_S])) for n in SHOW_ORDERS)
    ax_e.set_ylim(15.0, np.ceil(top / 5.0) * 5.0 + 12.0)  # room for the legend above the curves
    ax_e.legend(loc="upper left", frameon=True, fontsize=6.5, ncol=2)

    # (f): inverter components against the truth, median per carrier hop.
    styles = {0.0: ("k", "o"), -2.0 * p: ("#0072B2", "v"), 2.0 * p: ("#D55E00", "^")}
    for qq, (label, hop_med) in carrier_rows.items():
        if qq not in styles:
            continue
        colour, marker = styles[qq]
        ax_f.plot(hop_med[:, 0], hop_med[:, 1], color=colour, lw=0.0, marker=marker, ms=4.0, label=label)
        ax_f.plot(hop_med[:, 0], hop_med[:, 2], color=colour, lw=0.9, ls=(0, (4, 3)),
                  label="truth (dashed)" if qq == 0 else None)
    ax_f.set_title("(f) Inverter lines (floor removed), median per hop", loc="left")
    ax_f.set_ylabel("Level, dB re 20 µPa (RMS)")
    ax_f.legend(loc="lower right", frameon=True, fontsize=6.5, ncol=2)
    for ax in (ax_e, ax_f):
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0.0, DURATION)
        ax.grid(alpha=0.3)

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Figure written to {FIG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
