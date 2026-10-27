"""A realistic engine run-up you can listen to: analysis, order removal and a faster engine.

The recording comes from ``rota.synth.synth_scenario("ice_runup")``: a synthetic
four-stroke four-cylinder engine (firing order 2) in a vehicle that pulls away from
idle with clutch slip and accelerates at full load through two upshifts. Order 2
dominates; orders 4, 6 and 8 follow 15, 24 and 30 dB below it, the odd and half
orders weaker still. The scenario adds an angle-locked speed ripple, combustion bursts
that ring a structural resonance, engine broadband noise, road noise and a
low-frequency rumble, a road-joint thump and a click, a modal transfer path (humps at
45 and 175 Hz, a dip near 80 Hz), and a cabin impulse response with two receiver
channels. All of it is synthetic and comes with exact truth; no measured data are
used. The default record starts in second gear; the overrides in
``SCENARIO_OVERRIDES`` select the launch from idle, and every other scenario parameter
is a default of ``synth_scenario``.

Analysis, with the speed taken from the tachometer channel as in a measurement:
``tacho_to_rpm`` (conditioning delay compensated), ``rpm_frequency_map`` (Campbell
diagram), ``rpm_order_map``, ``order_slice`` of the firing order against the
quasi-static truth at the receiver, and ``remove_orders`` (tracked harmonics) of the
engine orders: the source orders of the scenario plus the harmonics of the firing
order up to ``MAX_ORDER``, which carry the line part of the combustion bursts. The
extracted orders are then rendered along a speed 15 % higher with ``remap_rpm``
(envelopes keep their time course, carriers follow the new speed) and recombined with
the untouched residual by ``resynthesize``.

The truth decides what the numbers mean. The firing-order slice is compared with the
quasi-static truth at each receiver (``ScenarioSignal.quasi_static_amplitude``); it is
only approximate where the order sweeps quickly through a path mode or a shift. The
extracted orders are compared with the true crank-locked stems (orders plus
combustion bursts) and the residual with the true remainder (engine broadband, road,
rumble, transients). The residual is quieter than that remainder: an order band of a
frame of 8 revolutions also holds broadband noise (0.19 orders per extracted order),
so the low-frequency rumble and road noise inside the 27 bands, and the 32 Hz
road-joint thump where it coincides with order 0.5, leave with the orders. The combustion bursts are not
perfectly periodic (cycle-to-cycle jitter), so part of them stays in the residual.

Audio: four stereo 16-bit WAV files in ``docs/audio/`` (the recording, the residual
after order removal, the extracted orders, and the faster engine). They share one
digital scaling, ``FULL_SCALE_PA``, so their levels can be compared directly; the
script refuses to write a file that would clip.

Amplitude conventions: maps, slices and levels are RMS in Pa, levels in dB re 20 uPa;
the truth ``order_amplitude`` and extracted envelopes are peak values (divided by
sqrt(2) for comparison). Speed in rpm, time in s.

Run:  python examples/05_realistic_engine_audio.py     (needs matplotlib, about 11 s)
Writes docs/images/realistic_runup.png and docs/audio/engine_runup*.wav.
"""

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import rota

P_REF = 20e-6  # reference pressure in Pa (dB re 20 uPa), passed explicitly
FULL_SCALE_PA = 4.0  # sound pressure in Pa that maps to digital full scale in every WAV file
FADE_S = 0.02  # raised-cosine fade-in and fade-out of every WAV file, s
FS = 32000.0  # sampling rate of the scenario and of the WAV files, Hz
DURATION = 8.5  # s
SEED = 1
N_CHANNELS = 2  # two receiver channels (direct sound 0.4 ms apart, independent reverberation)
# The default record starts in second gear at 1200 rpm. These overrides give the launch
# variant instead: 0.8 s of idle, a pull-away in first gear with clutch slip and a brisker
# full-load pull, so that 8.5 s hold the launch and two upshifts. Everything else is the
# scenario default.
SCENARIO_OVERRIDES = {
    "speed": {"hold": 0.8, "start_kmh": 0.0, "accel_ms2": 4.0, "jerk_time": 0.6},
    "drive": {"start_gear": 1},
}
SPEED_FACTOR = 1.15  # apparent engine speed of the "faster" file, relative to the recording
MAX_ORDER = 40.0  # highest harmonic of the firing order that is extracted
WINDOW_REVS = 8.0  # tracked-harmonics frame length, revolutions
EDGE_S = 0.3  # start of the record left out of the statistics (start-up of the path), s
ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = ROOT / "docs" / "images" / "realistic_runup.png"
AUDIO_DIR = ROOT / "docs" / "audio"


def to_db(amplitude):
    """Level in dB re P_REF of an amplitude in Pa (RMS in, RMS level out)."""
    return 20.0 * np.log10(np.maximum(amplitude, 1e-12) / P_REF)


def write_wav(name, x, fs):
    """Write ``x`` (Pa, shape (n,) or (n, ch)) as 16-bit PCM, scaled by FULL_SCALE_PA.

    Sample value = round(32767 * p / FULL_SCALE_PA) after a raised-cosine fade of
    FADE_S at both ends; nothing else is applied (no normalisation, no limiter).
    Returns the RMS level in dB re 20 uPa (all channels) and the peak in dBFS.
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
    print(f"  {name:36s} {to_db(rms):5.1f} dB SPL (RMS)   peak {20 * np.log10(peak / FULL_SCALE_PA):6.1f} dBFS"
          f"   {path.stat().st_size / 1e6:.2f} MB")
    return rms, peak


def masked_stats(err, mask):
    e = np.abs(err[mask & np.isfinite(err)])
    return float(np.median(e)), float(np.percentile(e, 95))


def main():
    plt.switch_backend("Agg")  # file output only; no display needed
    sc = rota.synth.synth_scenario(
        "ice_runup", fs=FS, duration=DURATION, seed=SEED, n_channels=N_CHANNELS, **SCENARIO_OVERRIDES
    )
    par = sc.config["parameters"]
    firing = float(par["combustion"]["per_rev"])  # firing events per revolution = firing order
    k_fire = int(np.flatnonzero(sc.orders == firing)[0])
    ppr = int(par["tacho"]["ppr"])
    delay = float(sc.config["resolved"]["tacho_delay"])
    shifts = sc.events["shift_times"]
    print(f"Scenario ice_runup: {DURATION:g} s at {FS:g} Hz, {sc.n_channels} channels, seed {SEED}; "
          f"{sc.rpm_mean.min():.0f} -> {sc.rpm_mean.max():.0f} rpm, "
          f"{sc.speed_kmh[0]:.0f} -> {sc.speed_kmh[-1]:.0f} km/h")
    print(f"  launch {sc.events['launch_times'].round(2).tolist()} s, upshifts at {shifts.round(2).tolist()} s "
          f"(gears {sc.events['shift_from'].tolist()} -> {sc.events['shift_to'].tolist()})")
    print(f"  source orders {sc.orders.tolist()} ({', '.join(sc.order_names)}); firing order {firing:g}")

    # ---- speed from the tachometer -------------------------------------------------------
    # The tacho lags the audio by the conditioning delay, so the speed at audio time t is
    # read from the tacho at t + delay.
    rpm = rota.speed.tacho_to_rpm(sc.tacho, FS, ppr=ppr, t_out=sc.t + delay)
    rel = np.abs(rpm / sc.rpm_mean - 1.0)
    calm = ~(sc.events["in_shift"] | sc.events["slip"])
    print(f"Speed from {ppr}-pulse tacho (delay {delay * 1e3:g} ms compensated) against the true mean speed: "
          f"median {100 * np.median(rel):.3f} %, max {100 * rel[calm].max():.2f} % outside shifts, "
          f"max {100 * rel.max():.2f} % overall")

    # ---- maps and the firing-order slice --------------------------------------------------
    fmap = rota.maps.rpm_frequency_map(sc.x, FS, rpm, resolution=4.0, amplitude="rms", max_freq=700.0)
    omap = rota.maps.rpm_order_map(sc.x, FS, rpm, order_resolution=0.125, max_order=10.0, amplitude="rms")
    s_fire = rota.slices.order_slice(omap, firing)
    frame_calm = (np.interp(s_fire.t, sc.t, calm.astype(float)) == 1.0) & (s_fire.t > EDGE_S)
    truth_fire = []
    print(f"Order {firing:g} slice against the quasi-static truth at the receiver (RMS, dB), "
          f"{int(frame_calm.sum())} frames outside shifts and launch:")
    for c in range(sc.n_channels):
        truth = sc.quasi_static_amplitude(k_fire, channel=c) / np.sqrt(2.0)  # peak -> RMS
        truth_fire.append(truth)
        err = 20.0 * np.log10(s_fire.amplitude[:, c] / np.interp(s_fire.t, sc.t, truth))
        med, p95 = masked_stats(err, frame_calm)
        print(f"  channel {c + 1}: median |error| {med:.2f} dB, 95th percentile {p95:.2f} dB")

    # ---- order removal ----------------------------------------------------------------------
    orders = np.union1d(sc.orders, firing * np.arange(1, int(MAX_ORDER / firing) + 1))
    residual, comp = rota.extract.remove_orders(
        sc.x, FS, rpm, orders, method="tracked-harmonics", window_revs=WINDOW_REVS
    )
    sel = sc.t > EDGE_S
    engine = sc.stem_sum(["orders", "combustion"])  # everything locked to the crank angle
    rest = sc.x_clean - engine  # broadband, road, rumble, transients: an ideal removal leaves this
    print(f"remove_orders (tracked harmonics, {WINDOW_REVS:g} revolutions): {orders.size} orders, "
          f"{orders.min():g} to {orders.max():g}")
    e_fire = [rota.metrics.order_residual_energy(r[sel, 0], sc.x[sel, 0], FS, rpm[sel], firing)
              for r in (residual, rest)]
    print(f"  energy left at order {firing:g}, dB re the recording (channel 1): {e_fire[0]:.1f} "
          f"(true non-engine content: {e_fire[1]:.1f})")
    sim_o = rota.metrics.waveform_similarity(comp.total[sel], engine[sel])
    sim_r = rota.metrics.waveform_similarity(residual[sel], rest[sel])
    print(f"  extracted orders against the true engine stems (orders + combustion): corr {sim_o['corr']:.4f}, "
          f"nrmse {sim_o['nrmse']:.3f}")
    print(f"  residual against the true non-engine content: corr {sim_r['corr']:.4f}, nrmse {sim_r['nrmse']:.3f}")
    rms = {k: float(np.sqrt(np.mean(v[sel] ** 2))) for k, v in
           (("recording", sc.x), ("residual", residual), ("non-engine truth", rest), ("engine truth", engine))}
    print("  RMS levels: " + ", ".join(f"{k} {to_db(v):.1f} dB" for k, v in rms.items()))
    # The order bands also hold broadband noise: the low-frequency rumble and road noise and
    # the 32 Hz road-joint thump inside them leave with the orders, so the residual is quieter
    # than the true non-engine content (see the module docstring).

    # ---- faster engine ------------------------------------------------------------------------
    fast, info = rota.resynth.remap_rpm(comp, SPEED_FACTOR * rpm, FS, rpm_old=rpm, mode="time", phase="table")
    mix = rota.resynth.resynthesize(fast, residual)
    print(f"remap_rpm: speed ratio {info['speed_ratio']['median']:.3f}, top speed "
          f"{SPEED_FACTOR * rpm.max():.0f} rpm; orders muted above Nyquist: "
          f"{100 * float(np.max(info['muted_fraction'])):.1f} % of samples at most")
    print(f"  resynthesize: orders {to_db(mix['rms']['orders']):.1f} dB, residual {to_db(mix['rms']['residual']):.1f} dB, "
          f"mix {to_db(mix['rms']['mix']):.1f} dB (RMS, re 20 uPa); order/residual ratio {mix['ratio_db']:.1f} dB")

    print(f"WAV files (16-bit PCM, {int(FS)} Hz, {sc.n_channels} channels, full scale = {FULL_SCALE_PA:g} Pa):")
    write_wav("engine_runup.wav", sc.x, FS)
    write_wav("engine_runup_orders_removed.wav", residual, FS)
    write_wav("engine_runup_orders_only.wav", comp.total, FS)
    write_wav("engine_runup_faster.wav", mix["mix"], FS)

    # ---- figure ---------------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig = plt.figure(figsize=(8.0, 8.4), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.55, 0.75])
    ax_f, ax_o = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[1, :])
    ax_s = fig.add_subplot(gs[2, :], sharex=ax_r)

    ch = 0
    db_f, db_o = to_db(fmap.magnitude[:, :, ch]), to_db(omap.magnitude[:, :, ch])
    vmax = np.ceil(max(db_f.max(), db_o.max()) / 5.0) * 5.0
    kw = {"cmap": "viridis", "vmin": vmax - 50.0, "vmax": vmax, "shading": "nearest", "rasterized": True}
    ax_f.pcolormesh(fmap.t, fmap.axis, db_f, **kw)
    im = ax_o.pcolormesh(omap.t, omap.axis, db_o, **kw)
    cb = fig.colorbar(im, ax=[ax_f, ax_o], location="bottom", fraction=0.05, pad=0.02)
    cb.set_label("Level per bin, dB re 20 µPa (RMS), channel 1")
    ax_f.set_title("(a) Campbell diagram, 4 Hz bins", loc="left")
    ax_f.set_ylabel("Frequency (Hz)")
    ax_o.set_title("(b) Order map, 0.125 order bins", loc="left")
    ax_o.set_ylabel("Order (cycles per revolution)")
    ax_o.set_ylim(0.0, 10.0)
    ax_o.set_yticks(np.arange(0, 11, 1))
    for ax in (ax_f, ax_o):
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0.0, DURATION)
    i_end = np.flatnonzero(sc.t < shifts[0] - 0.05)[-1] if shifts.size else sc.n - 1
    f_end = firing * sc.rpm_mean[i_end] / 60.0
    ax_f.annotate(f"firing order {firing:g}", (sc.t[i_end], f_end), xytext=(sc.t[i_end] - 2.4, f_end + 160.0),
                  color="w", fontsize=7, ha="center", arrowprops={"arrowstyle": "->", "lw": 0.6, "color": "w"})
    for ts in shifts:
        ax_f.annotate("shift", (ts, 650.0), color="w", fontsize=7, ha="center", va="center")
    ax_o.annotate(f"firing order {firing:g}", (3.4, firing), xytext=(1.5, 9.3), color="w", fontsize=7,
                  ha="center", arrowprops={"arrowstyle": "->", "lw": 0.6, "color": "w"})
    ax_o.annotate("order 4 (-15 dB)", (5.6, 2 * firing), xytext=(5.6, 5.4), color="w", fontsize=7,
                  ha="center", arrowprops={"arrowstyle": "->", "lw": 0.6, "color": "w"})

    ax_r.plot(sc.t, rpm, color="#0072B2", lw=2.4, label=f"tacho_to_rpm ({ppr} pulses per revolution)")
    ax_r.plot(sc.t, sc.rpm_mean, color="k", lw=0.8, ls=(0, (4, 3)), label="truth (synth, mean speed)")
    ax_r.set_ylabel("Engine speed (rpm)")
    ax_r.set_title("(c) Engine speed (launch with clutch slip, two upshifts) and vehicle speed (dashed, right)",
                   loc="left")
    ax_r.legend(loc="upper left", frameon=True)
    ax_v = ax_r.twinx()
    ax_v.plot(sc.t, sc.speed_kmh, color="#D55E00", lw=1.0, ls=(0, (4, 2)))
    ax_v.set_ylabel("Vehicle speed (km/h)", color="#D55E00")
    ax_v.tick_params(axis="y", colors="#D55E00")
    edges = np.concatenate(([sc.events["launch_times"][0, 1]], shifts, [DURATION]))
    for g, (a, b) in zip(np.unique(sc.gear), pairwise(edges)):
        ax_r.text(0.5 * (a + b), 250.0, f"gear {g}", ha="center", va="bottom", fontsize=7)
    ax_r.set_ylim(0.0, 7000.0)

    colours = ("#0072B2", "#009E73")
    for c in range(sc.n_channels):
        ax_s.plot(s_fire.t, to_db(s_fire.amplitude[:, c]), color=colours[c], lw=2.2 if c == 0 else 1.2,
                  label=f"order_slice, channel {c + 1}")
        ax_s.plot(sc.t, to_db(truth_fire[c]), color="k", lw=0.8, ls=(0, (4, 3)) if c == 0 else (0, (1, 2)),
                  label=f"truth, channel {c + 1} (quasi-static)")
    s_res = rota.slices.order_slice(rota.maps.rpm_order_map(residual[:, 0], FS, rpm, order_resolution=0.125,
                                                            max_order=10.0, amplitude="rms"), firing)
    ax_s.plot(s_res.t, to_db(s_res.amplitude), color="#D55E00", lw=1.0, label="after remove_orders, channel 1")
    # Path modes swept by the firing order in first gear (frequencies from the scenario).
    first = sc.t < (shifts[0] if shifts.size else DURATION)
    for f_mode in (45.0, 175.0):
        i = np.flatnonzero(first & (firing * sc.rpm_mean / 60.0 >= f_mode))[0]
        ax_s.annotate(f"path mode {f_mode:g} Hz", (sc.t[i], to_db(truth_fire[0][i]) + 1.0),
                      xytext=(sc.t[i], to_db(truth_fire[0][i]) + 7.0), fontsize=7, ha="center",
                      arrowprops={"arrowstyle": "->", "lw": 0.6})
    ax_s.set_title(f"(d) Level of the firing order {firing:g} against the truth at the receiver", loc="left")
    ax_s.set_xlabel("Time (s)")
    ax_s.set_ylabel("Order level, dB re 20 µPa (RMS)")
    ax_s.set_xlim(0.0, DURATION)
    # Room below the residual for the legend (the first second, where the residual of the
    # idle is far below the rest, may leave the axes) and above the truth for the labels.
    lo = np.floor(np.percentile(to_db(s_res.amplitude[s_res.t > 1.0]), 1) / 5.0) * 5.0 - 20.0
    ax_s.set_ylim(lo, np.ceil(to_db(truth_fire[0].max()) / 5.0) * 5.0 + 12.0)
    ax_s.legend(loc="lower right", frameon=True, ncol=3, fontsize=7)
    for ax in (ax_r, ax_s):
        ax.grid(alpha=0.3)
        for ts in shifts:
            ax.axvline(ts, color="0.4", lw=0.6, ls=":")

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Figure written to {FIG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
