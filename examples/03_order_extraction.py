"""Order extraction and removal: Vold-Kalman filter and tracked harmonics.

The signal of ``01_order_map.py`` is analysed with ``rota.extract.order_waveform``
(Vold-Kalman, six orders solved jointly, 2 Hz bandwidth) and
``rota.extract.tracked_harmonics`` (frames of 8 revolutions, much cheaper).
``resolution_check`` first confirms that both settings separate the half orders at
idle. The speed reference is the speed that went into the signal (see
``02_tacho_and_tacholess_speed.py`` for estimating it).

Figure: (a) peak envelope of order 3, which sweeps through the resonance, against
the truth, (b) the level error of both extractors, (c, d) the order map before and
after ``remove_orders`` (all six orders, Vold-Kalman), (e) the average order
spectrum of the original, of both residuals and of the true non-order content (tone
plus noise), which is what an ideal removal would leave.

Amplitude conventions: envelopes and waveforms are peak amplitudes in Pa; order maps
and spectra are RMS in Pa, with levels in dB re 20 uPa.

The extractors know nothing about the fixed 250 Hz tone, so orders 4.5 and 6 pick it
up briefly where they cross it (at about 3330 and 2500 rpm). Those frames are left
out of the error statistics, and the true non-order content is the floor that
``order_residual_energy`` of these two orders can reach.

Run:  python examples/03_order_extraction.py     (needs matplotlib, about 8 s)
Writes docs/images/03_order_extraction.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import rota

P_REF = 20e-6  # reference pressure in Pa (dB re 20 uPa), passed explicitly
FS = 8000.0  # sampling rate, Hz
DURATION = 6.0  # s
IDLE_RPM, TOP_RPM = 900.0, 4500.0
T_RUNUP = 1.2  # end of the idle hold, s
TONE_HZ = 250.0  # fixed-frequency line, Hz
VK_BANDWIDTH = 2.0  # Vold-Kalman -3 dB bandwidth, Hz
WINDOW_REVS = 8.0  # tracked-harmonics frame length, revolutions
EDGE = (0.6, 0.2)  # s at the start and end left out of the statistics (filter edge regions)
FIG_PATH = Path(__file__).resolve().parent.parent / "docs" / "images" / "03_order_extraction.png"

ORDERS = [
    {"order": 0.5, "amplitude": 0.15},
    {"order": 1.0, "amplitude": 0.35},
    {"order": 1.5, "amplitude": 1.0, "gain_rpm": [IDLE_RPM, TOP_RPM], "gain": [0.7, 1.3]},
    {"order": 3.0, "amplitude": 0.5},
    {"order": 4.5, "amplitude": 0.3},
    {"order": 6.0, "amplitude": 0.2},
]


def build_signal():
    """Idle hold, then linear run-up; orders, resonance, tone and noise (peak amplitudes in Pa)."""
    t = np.arange(round(DURATION * FS)) / FS
    rpm = rota.synth.rpm_profiles(
        t, "linear", start=IDLE_RPM, stop=TOP_RPM, t_start=T_RUNUP, t_stop=DURATION
    )
    return rota.synth.synth_runup(
        FS,
        rpm,
        ORDERS,
        tones=[{"freq": TONE_HZ, "amplitude": 0.12}],
        resonances=[{"freq": 180.0, "q": 6.0, "gain": 4.0}],
        noise_rms=0.05,
        seed=7,
    )


def to_db(amplitude):
    """Level in dB re P_REF of an amplitude in Pa."""
    return 20.0 * np.log10(np.maximum(amplitude, 1e-12) / P_REF)


def add_rpm_axis(ax, sig, rpm_ticks=(2000.0, 3000.0, 4000.0)):
    """Add a top axis that labels the time axis below with the speed in rpm."""
    run = sig.t >= T_RUNUP
    ticks = [IDLE_RPM, *rpm_ticks, TOP_RPM]
    times = np.interp(ticks, sig.rpm[run], sig.t[run])  # strictly increasing in the run-up
    top = ax.secondary_xaxis("top")
    top.set_xticks(times)
    top.set_xticklabels([f"{r:g}" for r in ticks])
    top.set_xlabel("Speed (rpm)")
    return top


def main():
    plt.switch_backend("Agg")  # file output only; no display needed
    sig = build_signal()
    orders = sig.orders

    # Can these settings separate the orders (half orders at idle)?
    check_vk = rota.extract.resolution_check(sig.rpm, FS, orders, bandwidth=VK_BANDWIDTH)
    check_th = rota.extract.resolution_check(sig.rpm, FS, orders, window_revs=WINDOW_REVS)

    vk = rota.extract.order_waveform(sig.x, FS, sig.rpm, orders, bandwidth=VK_BANDWIDTH)
    th = rota.extract.tracked_harmonics(sig.x, FS, sig.rpm, orders, window_revs=WINDOW_REVS)

    # Statistics over the interior, without the frames where the tone crosses an order.
    sel = (sig.t >= EDGE[0]) & (sig.t <= DURATION - EDGE[1])
    print(f"Signal: {DURATION:g} s at {FS:g} Hz, {IDLE_RPM:g} -> {TOP_RPM:g} rpm; orders {orders.tolist()}")
    print(f"resolution_check ok: Vold-Kalman {VK_BANDWIDTH:g} Hz -> {check_vk.ok}, "
          f"tracked harmonics {WINDOW_REVS:g} revolutions -> {check_th.ok}")
    print("Envelope (peak) error against the synth truth, and waveform agreement (interior samples):")
    print("  order |  Vold-Kalman: median/max |err| dB, corr, nrmse |  tracked harmonics: same")
    for k, order in enumerate(orders):
        use = sel & (np.abs(TONE_HZ * 60.0 / sig.rpm - order) > 0.5)  # tone not in the band
        row = []
        for comp in (vk, th):
            err = np.abs(20.0 * np.log10(np.abs(comp.envelopes[use, k]) / sig.order_amplitude[use, k]))
            sim = rota.metrics.waveform_similarity(comp.waveforms[use, k], sig.order_stems[use, k])
            row.append(f"{np.median(err):6.3f} /{err.max():5.2f}  {sim['corr']:.4f}  {sim['nrmse']:.3f}")
        print(f"  {order:5.1f} |  {row[0]}  |  {row[1]}")

    # Order removal with the Vold-Kalman filter and with tracked harmonics.
    res_vk, _ = rota.extract.remove_orders(sig.x, FS, sig.rpm, orders, bandwidth=VK_BANDWIDTH)
    res_th, _ = rota.extract.remove_orders(
        sig.x, FS, sig.rpm, orders, method="tracked-harmonics", window_revs=WINDOW_REVS
    )
    floor = sig.x - sig.stems["orders"]  # tone plus noise: what an ideal removal leaves
    print("Energy left at each removed order, dB relative to the original signal "
          "(order_residual_energy):")
    print("  order | Vold-Kalman | tracked harmonics | true non-order content")
    for order in orders:
        e = [rota.metrics.order_residual_energy(r[sel], sig.x[sel], FS, sig.rpm[sel], order)
             for r in (res_vk, res_th, floor)]
        print(f"  {order:5.1f} | {e[0]:11.1f} | {e[1]:17.1f} | {e[2]:22.1f}")
    sim = rota.metrics.waveform_similarity(res_vk[sel], floor[sel])
    print(f"Vold-Kalman residual against the true non-order content: corr {sim['corr']:.4f}, "
          f"nrmse {sim['nrmse']:.3f}; RMS {np.sqrt(np.mean(res_vk[sel] ** 2)):.4f} Pa "
          f"(truth {np.sqrt(np.mean(floor[sel] ** 2)):.4f} Pa, original "
          f"{np.sqrt(np.mean(sig.x[sel] ** 2)):.4f} Pa)")

    # ---- figure ----------------------------------------------------------------------------
    maps = {
        name: rota.maps.rpm_order_map(x, FS, sig.rpm, order_resolution=0.125, max_order=8.0,
                                      amplitude="rms")
        for name, x in (("orig", sig.x), ("vk", res_vk), ("th", res_th), ("floor", floor))
    }
    spectra = {name: rota.slices.order_spectrum(m, rpm_range=(1000.0, None)) for name, m in maps.items()}

    blue, orange = "#0072B2", "#D55E00"
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig = plt.figure(figsize=(8.0, 9.0), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.15, 0.9])
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    k3 = int(np.flatnonzero(orders == 3.0)[0])
    truth3 = sig.order_amplitude[:, k3]
    ax_a.plot(sig.t, np.abs(vk.envelopes[:, k3]), color=blue, lw=2.4, label="order_waveform")
    ax_a.plot(sig.t, np.abs(th.envelopes[:, k3]), color=orange, lw=1.4, label="tracked_harmonics")
    ax_a.plot(sig.t, truth3, color="k", lw=0.8, ls=(0, (4, 3)), label="truth (synth)")
    ax_a.set_title("(a) Order 3, peak envelope", loc="left")
    ax_a.set_ylabel("Peak amplitude (Pa)")
    ax_a.legend(loc="upper left")

    for comp, colour, name in ((vk, blue, "order_waveform"), (th, orange, "tracked_harmonics")):
        ax_b.plot(sig.t, 20.0 * np.log10(np.abs(comp.envelopes[:, k3]) / truth3), color=colour,
                  lw=1.0, label=name)
    ax_b.axhline(0.0, color="k", lw=0.6)
    ax_b.axvspan(0.0, EDGE[0], color="0.85", lw=0)
    ax_b.axvspan(DURATION - EDGE[1], DURATION, color="0.85", lw=0)
    ax_b.set_ylim(-0.5, 0.5)
    ax_b.set_title("(b) Order 3, level error (edges shaded)", loc="left")
    ax_b.set_ylabel("Error (dB)")
    ax_b.legend(loc="upper right")
    for ax in (ax_a, ax_b):
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0.0, DURATION)
        ax.grid(alpha=0.3)
        add_rpm_axis(ax, sig)

    db_o, db_r = to_db(maps["orig"].magnitude), to_db(maps["vk"].magnitude)
    vmax = np.ceil(db_o.max() / 5.0) * 5.0
    kw = {"cmap": "viridis", "vmin": vmax - 60.0, "vmax": vmax, "shading": "nearest", "rasterized": True}
    ax_c.pcolormesh(maps["orig"].t, maps["orig"].axis, db_o, **kw)
    im = ax_d.pcolormesh(maps["vk"].t, maps["vk"].axis, db_r, **kw)
    cb = fig.colorbar(im, ax=[ax_c, ax_d], location="bottom", fraction=0.05, pad=0.02)
    cb.set_label("Level per bin, dB re 20 µPa (RMS)")
    ax_c.set_title("(c) Order map of the original", loc="left")
    ax_d.set_title("(d) Order map after remove_orders (all six)", loc="left")
    for ax in (ax_c, ax_d):
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0.0, DURATION)
        ax.set_ylim(0.0, 8.0)
        ax.set_yticks(np.arange(0, 9, 1))
        add_rpm_axis(ax, sig)
    ax_c.set_ylabel("Order (cycles per revolution)")
    ax_d.annotate("250 Hz tone", (3.1, 7.2), color="w", fontsize=7, ha="left", va="center")

    for name, label, colour, lw, ls in (
        ("orig", "original", "k", 1.2, "-"),
        ("th", "residual, tracked_harmonics", orange, 1.0, "-"),
        ("vk", "residual, order_waveform", blue, 1.0, "-"),
        ("floor", "true non-order content (tone + noise)", "0.5", 1.0, (0, (4, 3))),
    ):
        s = spectra[name]
        ax_e.plot(s.axis, to_db(s.amplitude), color=colour, lw=lw, ls=ls, label=label)
    ax_e.set_title("(e) Average order spectrum over the run-up (frames above 1000 rpm)", loc="left")
    ax_e.set_xlabel("Order (cycles per revolution)")
    ax_e.set_ylabel("Level, dB re 20 µPa (RMS)")
    ax_e.set_xlim(0.0, 8.0)
    ax_e.set_ylim(0.0, 100.0)
    ax_e.set_xticks(np.arange(0, 9, 1))
    ax_e.grid(alpha=0.3)
    ax_e.legend(loc="lower right", frameon=True)

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Figure written to {FIG_PATH.relative_to(FIG_PATH.parent.parent.parent)}")


if __name__ == "__main__":
    main()
