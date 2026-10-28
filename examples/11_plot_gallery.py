"""Gallery of rota.plot: one figure per function, from one synthetic run-up.

Every public function of ``rota.plot`` draws one figure here, written to
``docs/images/plot_<function>.png`` and shown with its call in
``docs/plotting.md``. Two more figures show the before/after view of an order
modification: ``compare_maps`` with the level difference, and an order slice
drawn ``relative_to`` the original (``plot_order_slice_relative_to.png``).

The signal comes from ``rota.synth`` (seeded): 6 s at 8 kHz, a linear run-up from
1000 to 4500 rpm with six orders (0.5, 1, 1.5, 3, 4.5, 6), a 180 Hz resonance, a
fixed 250 Hz tone, white noise and a tachometer of 12 pulses per revolution. It is
read as sound pressure in Pa, so the level plots take ``**rp.SOUND_PRESSURE``
(dB re 20 µPa). The modified signal has order 3 lowered by 10 dB
(``order_waveform`` through ``rota.extract.residual``, ``modify_orders``,
``resynthesize``).

No psychoacoustic toolbox is needed. ``metric_vs_rpm`` and ``time_metric`` draw
results of ``rota.bridges.TimeMetric``, normally made by a toolbox (loudness,
sharpness, ...); here the metric is a quantity computed by ROTA itself, the
A-weighted sound level with Fast time weighting (``rota.tonal.level_vs_time``),
wrapped in a ``TimeMetric`` by hand.

Amplitude conventions: maps, slices, spectra and the order table are RMS (the
maps are computed with ``amplitude="rms"``); extracted order envelopes are peak.
Every label says which.

Run:  python examples/11_plot_gallery.py   (needs matplotlib, about 5 s)
Writes docs/images/plot_*.png (16 figures, 6 x 3.6 in at 110 dpi).
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import rota
import rota.plot as rp
from rota.bridges import TimeMetric

FS = 8000.0  # sampling rate, Hz
DURATION = 6.0  # s
START_RPM, TOP_RPM = 1000.0, 4500.0
PPR = 12  # tachometer pulses per revolution
EDGES = np.arange(1000.0, 4501.0, 250.0)  # speed bins, rpm
IMAGES = Path(__file__).resolve().parent.parent / "docs" / "images"
DPI = 110

ORDERS = [
    {"order": 0.5, "amplitude": 0.15},
    {"order": 1.0, "amplitude": 0.35},
    {"order": 1.5, "amplitude": 1.0, "gain_rpm": [START_RPM, TOP_RPM], "gain": [0.7, 1.3]},
    {"order": 3.0, "amplitude": 0.5},
    {"order": 4.5, "amplitude": 0.3},
    {"order": 6.0, "amplitude": 0.2},
]


def build_signal():
    """Linear run-up with orders, resonance, tone, noise and a tachometer (peak amplitudes in Pa)."""
    t = np.arange(round(DURATION * FS)) / FS
    rpm = rota.synth.rpm_profiles(t, "linear", start=START_RPM, stop=TOP_RPM)
    return rota.synth.synth_runup(
        FS,
        rpm,
        ORDERS,
        tones=[{"freq": 250.0, "amplitude": 0.12}],
        resonances=[{"freq": 180.0, "q": 6.0, "gain": 4.0}],
        noise_rms=0.05,
        tacho_ppr=PPR,
        tacho_jitter=30e-6,  # s
        seed=7,
    )


def save(ax_or_fig, name):
    """Write the figure of an Axes (or a Figure) to docs/images/plot_<name>.png and close it."""
    fig = getattr(ax_or_fig, "figure", ax_or_fig)
    path = IMAGES / f"plot_{name}.png"
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote docs/images/{path.name}")


def a_weighted_level(x):
    """A-weighted, Fast time-weighted level every 50 ms, as a TimeMetric (computed by ROTA)."""
    t, level = rota.tonal.level_vs_time(rota.tonal.weight(x, FS, "A"), FS, frame=0.05)
    return TimeMetric(
        name="A-weighted level, Fast", t=t, values=level, unit="dB re 20 µPa",
        skip_start=0.5,  # the meter starts at zero and reads low for a few time constants
        aggregate="power",  # energy mean of levels in a speed bin
        source="rota.tonal.level_vs_time",
    )


def main():
    plt.switch_backend("Agg")  # file output only; no display needed
    plt.rcdefaults()  # Matplotlib's default style, whatever a local matplotlibrc sets
    # The size of the gallery figures. rota.plot creates a figure with the current rcParams
    # when no Axes is given, so the calls below are the plain one-line calls.
    plt.rcParams.update({"figure.figsize": (6.0, 3.6), "figure.constrained_layout.use": True,
                         "font.size": 8.5})
    IMAGES.mkdir(parents=True, exist_ok=True)

    sig = build_signal()
    x, rpm, t = sig.x, sig.rpm, sig.t

    # ---- results of ROTA ------------------------------------------------------------------
    fmap = rota.maps.rpm_frequency_map(x, FS, rpm, resolution=4.0, amplitude="rms", max_freq=600.0)
    omap = rota.maps.rpm_order_map(x, FS, rpm, order_resolution=0.125, max_order=8.0, amplitude="rms")
    s15, s3 = rota.slices.order_slice(omap, 1.5), rota.slices.order_slice(omap, 3.0)
    truth15, truth3 = sig.order_amplitude[:, 2] / np.sqrt(2.0), sig.order_amplitude[:, 3] / np.sqrt(2.0)  # RMS, Pa
    tone = rota.slices.frequency_slice(fmap, 250.0)
    tone_rms = np.full_like(rpm, np.sqrt(np.mean(sig.stems["tones"] ** 2)))  # after the resonance, Pa
    spec = rota.slices.order_spectrum(omap)
    rpm_tacho = rota.speed.tacho_to_rpm(sig.tacho, FS, ppr=PPR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the narrow search range is deliberate (RuntimeWarning)
        ridge = rota.speed.rpm_from_ridge(x, FS, (600.0, 3500.0), orders=(1.5, 3.0, 4.5, 6.0), whiten=None)
    comp = rota.extract.tracked_harmonics(x, FS, rpm, [1.5, 3.0, 6.0], window_revs=8.0)
    table = rota.resynth.OrderTable.from_slices(
        [rota.slices.order_slice(omap, o) for o in (1.5, 3.0, 4.5)], EDGES[:-1] + 125.0, unit="Pa"
    )
    prom = rota.tonal.order_prominence(x, FS, rpm, [3.0, 4.5], rpm_bins=EDGES)
    bands = rota.tonal.band_levels_vs_rpm(x, FS, rpm, rpm_bins=EDGES, frame=0.1, fmin=25.0, fmax=3150.0)
    level = a_weighted_level(x)
    level_vs_rpm = level.vs_rpm(rpm, t, bins=EDGES)

    # The before/after pair: order 3 lowered by 10 dB, everything else unchanged.
    resid, c3 = rota.extract.residual(x, FS, rpm, [3.0], method="vold-kalman")
    y = rota.resynth.resynthesize(rota.resynth.modify_orders(c3, gain_db={3.0: -10.0}), resid)["mix"]
    omap_y = rota.maps.rpm_order_map(y, FS, rpm, order_resolution=0.125, max_order=8.0, amplitude="rms")
    s3_y = rota.slices.order_slice(omap_y, 3.0)
    ol = rota.slices.overall_level(x, FS, rpm, weighting="A", rpm_bins=EDGES)
    ol_y = rota.slices.overall_level(y, FS, rpm, weighting="A", rpm_bins=EDGES)

    # ---- one figure per function ----------------------------------------------------------
    # Maps
    ax = rp.rpm_frequency_map(fmap, **rp.SOUND_PRESSURE, mark_orders=(1.5, 3, 4.5, 6))
    save(ax, "rpm_frequency_map")

    ax = rp.rpm_order_map(omap, **rp.SOUND_PRESSURE, mark_orders=(1.5, 3, 4.5, 6))
    save(ax, "rpm_order_map")

    fig, _ = rp.compare_maps({"original": omap, "order 3 -10 dB": omap_y}, **rp.SOUND_PRESSURE,
                             difference=True, figsize=(9.0, 3.2))
    save(fig, "compare_maps")

    # Slices, spectra, levels, metrics
    ax = rp.order_slice({"order 1.5": s15, "order 3": s3}, **rp.SOUND_PRESSURE,
                        truth={"order 1.5": (rpm, truth15), "order 3": (rpm, truth3)})
    save(ax, "order_slice")

    ax = rp.order_slice({"original": s3, "order 3 -10 dB": s3_y}, relative_to="original")
    save(ax, "order_slice_relative_to")

    ax = rp.frequency_slice({"250 Hz": tone}, **rp.SOUND_PRESSURE, truth=(rpm, tone_rms))
    save(ax, "frequency_slice")

    ax = rp.order_spectrum(spec, **rp.SOUND_PRESSURE, mark_orders=(0.5, 1, 1.5, 3, 4.5, 6))
    save(ax, "order_spectrum")

    ax = rp.overall_level({"original": ol, "order 3 -10 dB": ol_y}, **rp.SOUND_PRESSURE)
    save(ax, "overall_level")

    ax = rp.metric_vs_rpm(level_vs_rpm)
    save(ax, "metric_vs_rpm")

    ax = rp.time_metric(level)
    save(ax, "time_metric")

    # Speed, extraction, tonal
    ax = rp.rpm_vs_time(t, rpm_tacho, label="tacho_to_rpm", reference=(t, rpm))
    save(ax, "rpm_vs_time")

    ax = rp.rpm_from_ridge(ridge, reference=(t, rpm))
    save(ax, "rpm_from_ridge")

    ax = rp.order_components(comp, rpm=rpm, **rp.SOUND_PRESSURE)
    save(ax, "order_components")

    ax = rp.order_table(table, **rp.SOUND_PRESSURE)
    save(ax, "order_table")

    ax = rp.order_prominence(prom)
    save(ax, "order_prominence")

    ax = rp.band_levels_vs_rpm(bands, **rp.SOUND_PRESSURE)
    save(ax, "band_levels_vs_rpm")


if __name__ == "__main__":
    main()
