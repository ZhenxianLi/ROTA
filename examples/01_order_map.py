"""Campbell diagram, order map and order slices of a synthetic engine run-up.

The 6 s signal from ``rota.synth`` holds 900 rpm for 1.2 s and then runs up linearly
to 4500 rpm. It contains six orders (including the half order 0.5), a 180 Hz
resonance that amplifies whichever order passes through it, a fixed 250 Hz tone and
white noise, and is read as sound pressure in Pa.

Figure: (a) rpm-frequency map (Campbell diagram): orders are rays, the tone is a
horizontal line, and at idle the orders merge because a fixed-length time window
spans several of them; (b) rpm-order map: orders are horizontal lines and stay
separated down to half an order at idle, the tone bends into a hyperbola and each
order brightens where it passes the resonance; (c) order slices of orders 1.5 and 3
against the amplitude put into the signal (analytic truth).

Amplitude convention: RMS, in Pa (a line that falls on a bin reads its peak
amplitude divided by sqrt(2)); levels are in dB re 20 uPa. The script also prints
the slice error against the truth, in dB, for all six orders.

Run:  python examples/01_order_map.py      (needs matplotlib, about 2 s)
Writes docs/images/01_order_map.png.
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
FIG_PATH = Path(__file__).resolve().parent.parent / "docs" / "images" / "01_order_map.png"

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
        resonances=[{"freq": 180.0, "q": 6.0, "gain": 4.0}],  # linear peak gain 4 (12 dB)
        noise_rms=0.05,
        seed=7,
    )


def to_db(amplitude):
    """Level in dB re P_REF of an amplitude in Pa (RMS in, RMS level out)."""
    return 20.0 * np.log10(np.maximum(amplitude, 1e-12) / P_REF)


def add_rpm_axis(ax, sig, rpm_ticks=(2000.0, 3000.0, 4000.0)):
    """Add a top axis that labels the time axis below with the speed in rpm.

    The speed curve is flat during the idle hold, so it cannot be inverted over
    the whole record; the ticks are placed by inverting the run-up part only.
    """
    run = sig.t >= T_RUNUP
    ticks = [IDLE_RPM, *rpm_ticks, TOP_RPM]
    times = np.interp(ticks, sig.rpm[run], sig.t[run])  # strictly increasing in the run-up
    top = ax.secondary_xaxis("top")
    top.set_xticks(times)
    top.set_xticklabels([f"{r:g}" for r in ticks])
    top.set_xlabel("Speed (rpm)")
    return top


def slice_errors(sig, omap):
    """Order-slice level error in dB against the synth truth, for every order of the signal."""
    idle = omap.rpm_max <= IDLE_RPM + 1e-6  # frames entirely inside the idle hold
    runup = omap.rpm_min > IDLE_RPM + 1e-6  # frames entirely inside the run-up
    rows = []
    for k, order in enumerate(sig.orders):
        s = rota.slices.order_slice(omap, order)
        truth = np.interp(s.t, sig.t, sig.order_amplitude[:, k]) / np.sqrt(2.0)  # peak -> RMS
        err = 20.0 * np.log10(s.amplitude / truth)
        # The tone adds to an order slice while it is inside the band (+-0.5 order); leave it out.
        clean = np.abs(TONE_HZ * 60.0 / s.rpm - order) > 0.5
        rows.append(
            {
                "order": order,
                "idle": float(np.median(err[idle])),
                "median": float(np.median(np.abs(err[runup & clean]))),
                "max": float(np.max(np.abs(err[runup & clean]))),
                "n_tone": int(np.sum(runup & ~clean)),
            }
        )
    return rows


def main():
    plt.switch_backend("Agg")  # file output only; no display needed
    sig = build_signal()

    # Campbell diagram: STFT over time, 4 Hz bins (0.25 s frames).
    fmap = rota.maps.rpm_frequency_map(
        sig.x, FS, sig.rpm, resolution=4.0, amplitude="rms", max_freq=600.0
    )
    # Order map: STFT over shaft angle, 0.125 order bins (frames of 8 revolutions).
    omap = rota.maps.rpm_order_map(
        sig.x, FS, sig.rpm, order_resolution=0.125, max_order=8.0, amplitude="rms"
    )

    print(f"Signal: {DURATION:g} s at {FS:g} Hz, idle {IDLE_RPM:g} rpm until {T_RUNUP:g} s, "
          f"then linear to {TOP_RPM:g} rpm; orders {sig.orders.tolist()}")
    print(f"Campbell map: {fmap.values.shape[1]} frames, {fmap.resolution:g} Hz bins; "
          f"order map: {omap.values.shape[1]} frames, {omap.resolution:g} order bins")
    print("Order-slice level error against the synth truth (RMS, dB):")
    print("  order   idle (median)   run-up median |err|   run-up max |err|")
    for r in slice_errors(sig, omap):
        print(f"  {r['order']:5.1f}   {r['idle']:+13.3f}   {r['median']:19.3f}   {r['max']:16.3f}"
              + (f"   ({r['n_tone']} frames with the 250 Hz tone in the band left out)"
                 if r["n_tone"] else ""))

    # ---- figure ----------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig = plt.figure(figsize=(8.0, 7.2), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.8])
    ax_f, ax_o, ax_s = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    db_f, db_o = to_db(fmap.magnitude), to_db(omap.magnitude)
    vmax = np.ceil(max(db_f.max(), db_o.max()) / 5.0) * 5.0
    kw = {"cmap": "viridis", "vmin": vmax - 60.0, "vmax": vmax, "shading": "nearest",
          "rasterized": True}
    ax_f.pcolormesh(fmap.t, fmap.axis, db_f, **kw)
    im = ax_o.pcolormesh(omap.t, omap.axis, db_o, **kw)
    cb = fig.colorbar(im, ax=[ax_f, ax_o], location="bottom", fraction=0.05, pad=0.02)
    cb.set_label("Level per bin, dB re 20 µPa (RMS)")

    ax_f.set_title("(a) Campbell diagram, 4 Hz bins", loc="left")
    ax_f.set_ylabel("Frequency (Hz)")
    ax_o.set_title("(b) Order map, 0.125 order bins", loc="left")
    ax_o.set_ylabel("Order (cycles per revolution)")
    ax_o.set_yticks(sig.orders)
    ax_o.set_yticklabels([f"{o:g}" for o in sig.orders])
    ax_o.set_ylim(0, 8)
    for ax in (ax_f, ax_o):
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0.0, DURATION)
        add_rpm_axis(ax, sig)
    ax_f.annotate("250 Hz tone", (0.1, TONE_HZ + 6.0), color="w", fontsize=7, ha="left", va="bottom")
    for o in (0.5, 1.0, 1.5, 3.0, 4.5, 6.0):  # order labels at the end of the run-up
        ax_f.text(5.8, o * TOP_RPM / 60.0 + 8.0, f"{o:g}", color="w", fontsize=7,
                  ha="right", va="bottom")
    ax_o.annotate("250 Hz tone", (3.1, 7.2), color="w", fontsize=7, ha="left", va="center")

    # Order slices against the truth.
    colours = {1.5: "#0072B2", 3.0: "#D55E00"}
    for order, colour in colours.items():
        k = int(np.flatnonzero(sig.orders == order)[0])
        s = rota.slices.order_slice(omap, order)
        truth = sig.order_amplitude[:, k] / np.sqrt(2.0)  # peak -> RMS
        ax_s.plot(s.t, to_db(s.amplitude), color=colour, lw=2.4, label=f"order {order:g}, order_slice")
        ax_s.plot(sig.t, to_db(truth), color="k", lw=0.8, ls=(0, (4, 3)),
                  label="truth (synth)" if order == 1.5 else None)
    ax_s.set_title("(c) Order slices of orders 1.5 and 3 against the amplitude put into the signal",
                   loc="left")
    ax_s.set_xlabel("Time (s)")
    ax_s.set_ylabel("Order level, dB re 20 \u00b5Pa (RMS)")
    ax_s.set_xlim(0.0, DURATION)
    ax_s.set_ylim(83.0, 104.0)
    ax_s.grid(alpha=0.3)
    ax_s.legend(loc="upper left", frameon=True)
    add_rpm_axis(ax_s, sig)
    t_res = float(np.interp(3600.0, sig.rpm[sig.t >= T_RUNUP], sig.t[sig.t >= T_RUNUP]))
    ax_s.annotate("order 3 passes the\n180 Hz resonance", (t_res, 97.3), xytext=(t_res - 1.4, 101.0),
                  fontsize=7, ha="center", va="center", arrowprops={"arrowstyle": "->", "lw": 0.6})

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Figure written to {FIG_PATH.relative_to(FIG_PATH.parent.parent.parent)}")


if __name__ == "__main__":
    main()
