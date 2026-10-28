"""Order-table resynthesis: analyse a run-up, render a run-down.

A synthetic run-up (900 rpm for 1.2 s, then linear to 4500 rpm; orders 0.5 to 6, the
180 Hz resonance, low-passed noise) is analysed, and a different signal is built from
the analysis alone:

1. Order slices of an order map (RMS, Pa) are binned on a 50 rpm grid with
   ``OrderTable.from_slices``.
2. ``rota_order.resynth.synthesize_orders`` renders the table along another speed profile,
   an S-shaped run-down from 4200 to 1100 rpm (random start phases, seeded).
3. The Welch PSD of the residual left by ``rota_order.extract.remove_orders`` drives
   ``rota_order.resynth.synthesize_broadband``.
4. The sum is analysed again and compared with a reference: ``rota_order.synth`` rendering
   the same orders along the same run-down.

The resonance has a fixed frequency, so the gain of each order is a function of speed
alone, and a table over speed carries it over to a run-down.

The fixed 250 Hz tone of the other examples is left out here. A fixed line is neither
an order nor broadband noise: in the residual PSD it is a spike narrower than the
envelope resolution of the broadband synthesiser (about 1 / grain), and the level of
the synthesised noise would come out wrong. In an order table it belongs in a row of
order 0 with a ``freq_offset``.

Figure: (a) the table with the amplitude put into the run-up, (b) the two speed
profiles, (c) the order map of the resynthesis, (d) the order map of the reference,
(e) two order slices of the resynthesis against the reference, (f) the target and the
synthesised broadband PSD.

Amplitude conventions: the table and the order maps are RMS (Pa, levels in dB re
20 uPa); ``synthesize_orders`` converts to peak amplitudes when it renders; the
broadband noise is a random signal given by its PSD (Pa^2/Hz), RMS convention.

Run:  python examples/04_order_table_resynthesis.py   (needs matplotlib, about 5 s)
Writes docs/images/04_order_table_resynthesis.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.integrate import trapezoid

import rota_order as ro

P_REF = 20e-6  # reference pressure in Pa (dB re 20 uPa), passed explicitly
FS = 8000.0  # sampling rate, Hz
IDLE_RPM, TOP_RPM = 900.0, 4500.0
T_RUNUP, DURATION = 1.2, 6.0  # end of the idle hold and length of the analysed run-up, s
RUNDOWN = {"start": 4200.0, "stop": 1100.0, "duration": 5.0}  # new speed profile
GRID = np.arange(900.0, 4401.0, 50.0)  # speed grid of the table, rpm
NPERSEG = 1024  # Welch segment, samples (7.8 Hz bins)
FIG_PATH = Path(__file__).resolve().parent.parent / "docs" / "images" / "04_order_table_resynthesis.png"

ORDERS = [
    {"order": 0.5, "amplitude": 0.15},
    {"order": 1.0, "amplitude": 0.35},
    {"order": 1.5, "amplitude": 1.0, "gain_rpm": [IDLE_RPM, TOP_RPM], "gain": [0.7, 1.3]},
    {"order": 3.0, "amplitude": 0.5},
    {"order": 4.5, "amplitude": 0.3},
    {"order": 6.0, "amplitude": 0.2},
]
SCENE = {  # everything but the speed: resonance and noise (low-passed at 2.5 kHz)
    "resonances": [{"freq": 180.0, "q": 6.0, "gain": 4.0}],
    "noise_rms": 0.05,
    "noise_band": (None, 2500.0),
}


def to_db(amplitude):
    """Level in dB re P_REF of an amplitude in Pa."""
    return 20.0 * np.log10(np.maximum(amplitude, 1e-12) / P_REF)


def psd_to_db(psd):
    """Level of a PSD in Pa^2/Hz in dB re P_REF^2 per Hz."""
    return 10.0 * np.log10(np.maximum(psd, 1e-30) / P_REF**2)


def add_rpm_axis(ax, t, rpm, rpm_ticks):
    """Add a top axis that labels the time axis below with the speed in rpm.

    ``rpm`` must be monotonic (rising or falling); ticks are placed by interpolation.
    """
    order = slice(None) if rpm[-1] > rpm[0] else slice(None, None, -1)
    times = np.interp(rpm_ticks, rpm[order], t[order])
    top = ax.secondary_xaxis("top")
    top.set_xticks(times)
    top.set_xticklabels([f"{r:g}" for r in rpm_ticks])
    top.set_xlabel("Speed (rpm)")
    return top


def main():
    plt.switch_backend("Agg")  # file output only; no display needed
    # ---- the run-up that is analysed ---------------------------------------------------------
    t_up = np.arange(round(DURATION * FS)) / FS
    rpm_up = ro.synth.rpm_profiles(
        t_up, "linear", start=IDLE_RPM, stop=TOP_RPM, t_start=T_RUNUP, t_stop=DURATION
    )
    up = ro.synth.synth_runup(FS, rpm_up, ORDERS, seed=7, **SCENE)
    orders = up.orders

    # Order table: order slices of an order map, binned on a speed grid. A dense overlap
    # gives enough frames in every 50 rpm bin at low speed, where a frame spans many seconds of angle.
    omap_up = ro.maps.rpm_order_map(
        up.x, FS, up.rpm, order_resolution=0.125, max_order=8.0, overlap=0.9, amplitude="rms"
    )
    slices = [ro.slices.order_slice(omap_up, o) for o in orders]
    table = ro.resynth.OrderTable.from_slices(
        slices, GRID, unit="Pa", meta={"source": "synthetic run-up, seed 7"}
    )

    # Residual of the run-up and its PSD (edges of the extractor left out).
    residual, _ = ro.extract.remove_orders(up.x, FS, up.rpm, orders, bandwidth=2.0)
    inner = slice(round(0.6 * FS), round((DURATION - 0.2) * FS))
    f_psd, psd = signal.welch(residual[inner], FS, nperseg=NPERSEG)  # Pa^2/Hz, one-sided

    # ---- the new signal, rendered from the analysis only --------------------------------------
    t_dn = np.arange(round(RUNDOWN["duration"] * FS)) / FS
    rpm_dn = ro.synth.rpm_profiles(t_dn, "s_curve", start=RUNDOWN["start"], stop=RUNDOWN["stop"])
    x_orders, info = ro.resynth.synthesize_orders(table, rpm_dn, FS, phase="random", seed=11)
    x_noise = ro.resynth.synthesize_broadband(psd, f_psd, FS, t_dn.size, grain=0.05, seed=12)
    x_new = x_orders + x_noise

    # Reference: rota_order.synth on the same run-down (what the resynthesis should contain).
    ref = ro.synth.synth_runup(FS, rpm_dn, ORDERS, seed=8, **SCENE)

    omap_new = ro.maps.rpm_order_map(x_new, FS, rpm_dn, order_resolution=0.125, max_order=8.0,
                                       amplitude="rms")
    omap_ref = ro.maps.rpm_order_map(ref.x, FS, rpm_dn, order_resolution=0.125, max_order=8.0,
                                       amplitude="rms")

    # ---- numbers ----------------------------------------------------------------------------
    print(f"Analysed run-up: {DURATION:g} s at {FS:g} Hz, {IDLE_RPM:g} -> {TOP_RPM:g} rpm; "
          f"{omap_up.values.shape[1]} order-map frames")
    print(f"Table: {table}")
    print(f"synthesize_orders: run-down {RUNDOWN['start']:g} -> {RUNDOWN['stop']:g} rpm in "
          f"{RUNDOWN['duration']:g} s, phases random (seed {info['seed']}), muted fraction "
          f"{info['muted_fraction'].max():g}, output convention {info['output_amplitude']}")
    print("Order levels of the resynthesis (orders + broadband), read back with order_slice, "
          "|error| in dB:")
    print("  order |  against the table (round trip) |  against the reference (truth)")
    errors = {}
    for k, order in enumerate(orders):
        s = ro.slices.order_slice(omap_new, order)
        use = (s.t > 0.3) & (s.t < RUNDOWN["duration"] - 0.3)  # frames clear of the record ends
        in_table = table.magnitude_at(s.rpm)[k]  # RMS, Pa
        truth = np.interp(s.t, ref.t, ref.order_amplitude[:, k]) / np.sqrt(2.0)  # peak -> RMS
        e_tab = np.abs(20.0 * np.log10(s.amplitude[use] / in_table[use]))
        e_ref = np.abs(20.0 * np.log10(s.amplitude[use] / truth[use]))
        errors[order] = (e_tab, e_ref)
        print(f"  {order:5.1f} |  median {np.median(e_tab):5.3f}  max {e_tab.max():5.2f}"
              f"            |  median {np.median(e_ref):5.3f}  max {e_ref.max():5.2f}")
    f_syn, psd_syn = signal.welch(x_noise, FS, nperseg=NPERSEG)
    band = (f_syn >= 50.0) & (f_syn <= 2000.0)
    d_psd = 10.0 * np.log10(psd_syn[band] / psd[band])
    print(f"synthesize_broadband: RMS {x_noise.std():.4f} Pa against {np.sqrt(trapezoid(psd, f_psd)):.4f} Pa "
          f"from the target PSD; PSD difference 50-2000 Hz: median {np.median(d_psd):+.2f} dB, "
          f"largest {np.abs(d_psd).max():.2f} dB (Welch scatter)")
    print(f"Whole signal: RMS {x_new.std():.4f} Pa resynthesised, {ref.x.std():.4f} Pa reference "
          f"({20 * np.log10(x_new.std() / ref.x.std()):+.2f} dB)")

    # ---- figure ----------------------------------------------------------------------------
    colours = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig = plt.figure(figsize=(8.0, 9.6), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.15, 1.0])
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ax_e, ax_f = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

    run = up.rpm > IDLE_RPM
    for k, (order, colour) in enumerate(zip(orders, colours)):
        ax_a.plot(table.rpm, to_db(table.magnitude[k]), color=colour, lw=1.3, marker="o", ms=2.0,
                  label=f"order {order:g}")
        ax_a.plot(up.rpm[run], to_db(up.order_amplitude[run, k] / np.sqrt(2.0)), color="k", lw=0.7,
                  ls=(0, (4, 3)), label="truth (synth)" if k == 0 else None)
    ax_a.set_title("(a) Order table from the run-up", loc="left")
    ax_a.set_xlabel("Speed (rpm)")
    ax_a.set_ylabel("Level, dB re 20 µPa (RMS)")
    ax_a.grid(alpha=0.3)
    ax_a.set_ylim(72.0, 111.0)
    ax_a.legend(loc="upper left", ncol=3, fontsize=6.5)

    ax_b.plot(t_up, rpm_up, color="0.45", lw=1.6, label="analysed run-up")
    ax_b.plot(t_dn, rpm_dn, color="#0072B2", lw=1.6, label="resynthesised run-down")
    ax_b.set_title("(b) Speed profiles", loc="left")
    ax_b.set_xlabel("Time (s)")
    ax_b.set_ylabel("Speed (rpm)")
    ax_b.grid(alpha=0.3)
    ax_b.legend(loc="upper center")
    ax_b.set_ylim(500.0, 5400.0)

    db_new, db_ref = to_db(omap_new.magnitude), to_db(omap_ref.magnitude)
    vmax = np.ceil(max(db_new.max(), db_ref.max()) / 5.0) * 5.0
    kw = {"cmap": "viridis", "vmin": vmax - 60.0, "vmax": vmax, "shading": "nearest", "rasterized": True}
    ax_c.pcolormesh(omap_new.t, omap_new.axis, db_new, **kw)
    im = ax_d.pcolormesh(omap_ref.t, omap_ref.axis, db_ref, **kw)
    cb = fig.colorbar(im, ax=[ax_c, ax_d], location="bottom", fraction=0.05, pad=0.02)
    cb.set_label("Level per bin, dB re 20 µPa (RMS)")
    ax_c.set_title("(c) Order map of the resynthesis", loc="left")
    ax_d.set_title("(d) Order map of the reference (rota_order.synth)", loc="left")
    for ax in (ax_c, ax_d):
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0.0, RUNDOWN["duration"])
        ax.set_ylim(0.0, 8.0)
        ax.set_yticks(np.arange(0, 9, 1))
        add_rpm_axis(ax, t_dn, rpm_dn, (4000.0, 3000.0, 2000.0))
    ax_c.set_ylabel("Order (cycles per revolution)")

    for order in (3.0, 4.5):
        k = int(np.flatnonzero(orders == order)[0])
        colour = colours[k]  # same colour as in (a)
        s = ro.slices.order_slice(omap_new, order)
        ax_e.plot(s.t, to_db(s.amplitude), color=colour, lw=2.2, label=f"order {order:g}, resynthesis")
        ax_e.plot(ref.t, to_db(ref.order_amplitude[:, k] / np.sqrt(2.0)), color="k", lw=0.8,
                  ls=(0, (4, 3)), label="truth (synth)" if order == 3.0 else None)
    ax_e.set_title("(e) Order slices of the resynthesis", loc="left")
    ax_e.set_xlabel("Time (s)")
    ax_e.set_ylabel("Order level, dB re 20 µPa (RMS)")
    ax_e.set_xlim(0.0, RUNDOWN["duration"])
    ax_e.grid(alpha=0.3)
    ax_e.set_ylim(80.0, 99.0)
    ax_e.legend(loc="upper right", fontsize=7)
    add_rpm_axis(ax_e, t_dn, rpm_dn, (4000.0, 3000.0, 2000.0))

    ax_f.plot(f_psd, psd_to_db(psd), color="k", lw=1.2, label="target: Welch PSD of the residual")
    ax_f.plot(f_syn, psd_to_db(psd_syn), color="#0072B2", lw=1.0, label="synthesize_broadband")
    ax_f.set_xscale("log")
    ax_f.set_xlim(20.0, FS / 2.0)
    ax_f.set_ylim(0.0, 50.0)
    ax_f.set_title("(f) Broadband part", loc="left")
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("PSD, dB re (20 µPa)²/Hz")
    ax_f.grid(alpha=0.3, which="both")
    ax_f.legend(loc="lower left", fontsize=7)

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Figure written to {FIG_PATH.relative_to(FIG_PATH.parent.parent.parent)}")


if __name__ == "__main__":
    main()
