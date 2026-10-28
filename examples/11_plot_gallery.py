"""Gallery of rota_order.plot: one figure per function, on two realistic synthetic recordings.

Every public function of ``rota_order.plot`` draws one figure here, written to
``docs/images/plot_<function>.png`` and shown with its call in
``docs/plotting.md``. Two more figures show the before/after view of an order
modification: ``compare_maps`` with the level difference, and an order slice
drawn ``relative_to`` the original (``plot_order_slice_relative_to.png``).

The recordings come from ``rota_order.synth.synth_scenario`` (seeded, 32 kHz, sound
pressure in Pa at the receiver, so the level plots take ``**rp.SOUND_PRESSURE``,
dB re 20 µPa). All of it is synthetic and comes with exact truth.

* ``ice``: ``"ice_runup"`` with its defaults, 18 s. A four-stroke four-cylinder
  engine (firing order 2, then 4, 6, 8, odd and half orders) pulls at full load
  from 1200 rpm in second gear to 6000 rpm and shifts up at 9.2 and 15.2 s; combustion bursts
  ring a 1.9 kHz resonance, a modal path has humps at 45 and 175 Hz and a dip at
  80 Hz, a cabin, engine broadband, road noise, rumble, a thump and a click
  complete it. The maps, slices, order table and band levels use the second-gear
  pull before the first upshift (``x2``, ``rpm2``), where the speed rises
  monotonically; the speed plots and the level against time use the whole record.
* ``ev``: ``"ev_runup"``, 8 s, set up as in ``10_worked_analysis.py``: an electric
  drive (motor orders 2 to 14, gear mesh 6.7, gear whine at order 40, inverter
  carrier near 10 kHz, fixed tones at 6.3 and 14.2 kHz) accelerating from 200 to
  10900 rpm, with one structural resonance added to the path (2900 Hz, Q = 20), so
  the whine is amplified around 4350 rpm, and a cabin without diffuse-field fading
  up to 7.9 kHz. The whine is extracted with the Vold-Kalman filter (its envelope
  is the ``order_components`` figure); for the before/after figures it is lowered by
  10 dB with ``modify_orders`` and added back to the residual with ``resynthesize``.

The speed of both comes from the tachometer channel (``tacho_to_rpm``, with the
conditioning delay of the scenario compensated), as in a measurement.

No psychoacoustic toolbox is needed. ``metric_vs_rpm`` and ``time_metric`` draw
results of ``rota_order.bridges.TimeMetric``, normally made by a toolbox (loudness,
sharpness, ...); here the metric is a quantity computed by rota-order itself, the
A-weighted sound level with Fast time weighting (``rota_order.tonal.level_vs_time``),
wrapped in a ``TimeMetric`` by hand.

Amplitude conventions: maps, slices, spectra and the order table are RMS (the
maps are computed with ``amplitude="rms"``); extracted order envelopes are peak;
the truth amplitudes of the scenario are peak and are divided by sqrt(2) where
they are compared with RMS slices. Every label says which.

Run:  python examples/11_plot_gallery.py   (needs matplotlib, about 20 s)
Writes docs/images/plot_*.png (16 figures, 6 x 3.6 in at 110 dpi).
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import rota_order as ro
import rota_order.plot as rp
from rota_order.bridges import TimeMetric

FS = 32000.0  # sampling rate of both scenarios, Hz
IMAGES = Path(__file__).resolve().parent.parent / "docs" / "images"
DPI = 110
EDGES = np.arange(1250.0, 6001.0, 250.0)  # speed bins of the engine, rpm
EV_EDGES = np.arange(1000.0, 11001.0, 250.0)  # speed bins of the electric drive, rpm
WHINE = 40.0  # gear-whine order of the electric drive (gear_mesh.teeth)
TONE = 6300.0  # fixed tone of the electric drive, Hz


def engine():
    """The ice_runup scenario with its defaults (18 s): second-gear pull and two upshifts."""
    return ro.synth.synth_scenario("ice_runup", fs=FS, seed=1)


def electric_drive():
    """The ev_runup scenario of example 10: a 2900 Hz path resonance added, no cabin fading below 7.9 kHz."""
    modes = ro.synth.scenario_defaults("ev_runup")["path"]["modes"] + [{"freq": 2900.0, "q": 20.0, "gain": 0.5}]
    return ro.synth.synth_scenario("ev_runup", fs=FS, duration=8.0, seed=3, tacho={"ppr": 24},
                                     path={"modes": modes}, cabin={"crossover_hz": 7900.0})


def tacho_rpm(sc):
    """Speed from the tachometer channel at the audio sample times (the conditioning delay compensated)."""
    delay = sc.config["resolved"]["tacho_delay"]
    return ro.speed.tacho_to_rpm(sc.tacho, FS, ppr=sc.config["parameters"]["tacho"]["ppr"], t_out=sc.t + delay)


def order_truth(sc, order):
    """True RMS amplitude (Pa) of a source order at the receiver, quasi-statically."""
    return sc.quasi_static_amplitude(int(np.flatnonzero(sc.orders == order)[0])) / np.sqrt(2.0)


def tone_truth(sc, freq):
    """True RMS amplitude (Pa) at the receiver of the fixed tone ``freq`` of the scenario.

    The tone stem before the path is a sum of cosines at fixed frequencies; the amplitude of the
    one at ``freq`` is its projection on that frequency, and the path response gives the receiver.
    """
    a = 2.0 * abs(np.mean(sc.source_stems["tones"] * np.exp(-2j * np.pi * freq * sc.t)))  # peak, Pa
    return a * abs(sc.path_response(freq)) / np.sqrt(2.0)


def a_weighted_level(x):
    """A-weighted, Fast time-weighted level every 50 ms, as a TimeMetric (computed by rota-order)."""
    t, level = ro.tonal.level_vs_time(ro.tonal.weight(x, FS, "A"), FS, frame=0.05)
    return TimeMetric(
        name="A-weighted level, Fast", t=t, values=level, unit="dB re 20 µPa",
        skip_start=0.5,  # the meter starts at zero and reads low for a few time constants
        aggregate="power",  # energy mean of levels in a speed bin
        source="rota-order level_vs_time",
    )


def save(ax_or_fig, name):
    """Write the figure of an Axes (or a Figure) to docs/images/plot_<name>.png and close it."""
    fig = getattr(ax_or_fig, "figure", ax_or_fig)
    path = IMAGES / f"plot_{name}.png"
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote docs/images/{path.name}")


def main():
    plt.switch_backend("Agg")  # file output only; no display needed
    plt.rcdefaults()  # Matplotlib's default style, whatever a local matplotlibrc sets
    # The size of the gallery figures. rota_order.plot creates a figure with the current rcParams
    # when no Axes is given, so the calls below are the plain one-line calls.
    plt.rcParams.update({"figure.figsize": (6.0, 3.6), "figure.constrained_layout.use": True,
                         "font.size": 8.5})
    IMAGES.mkdir(parents=True, exist_ok=True)

    # ---- the engine run-up ------------------------------------------------------------------
    ice = engine()
    t, rpm = ice.t, tacho_rpm(ice)
    second = t < ice.events["shift_times"][0]  # the second-gear pull, before the first upshift
    x2, rpm2 = ice.x[second], rpm[second]

    fmap = ro.maps.rpm_frequency_map(x2, FS, rpm2, resolution=4.0, amplitude="rms", max_freq=1000.0)
    omap = ro.maps.rpm_order_map(x2, FS, rpm2, order_resolution=0.125, max_order=10.0, amplitude="rms")
    s2, s4 = ro.slices.order_slice(omap, 2.0), ro.slices.order_slice(omap, 4.0)
    truth2, truth4 = order_truth(ice, 2.0)[second], order_truth(ice, 4.0)[second]  # RMS, Pa
    spec = ro.slices.order_spectrum(omap)
    table = ro.resynth.OrderTable.from_slices(
        [ro.slices.order_slice(omap, o) for o in (2.0, 4.0, 6.0)], EDGES[:-1] + 125.0, unit="Pa"
    )
    ridge = ro.speed.rpm_from_ridge(ice.x, FS, (700.0, 7000.0), orders=(2, 4, 6))
    level = a_weighted_level(ice.x)
    bands = ro.tonal.band_levels_vs_rpm(x2, FS, rpm2, rpm_bins=EDGES, frame=0.1, fmax=5000.0)

    # ---- the electric drive -----------------------------------------------------------------
    ev = electric_drive()
    rpm_ev = tacho_rpm(ev)
    ev_fmap = ro.maps.rpm_frequency_map(ev.x, FS, rpm_ev, resolution=4.0, amplitude="rms", max_freq=8000.0)
    tone = ro.slices.frequency_slice(ev_fmap, TONE)
    tone_rms = np.full_like(rpm_ev, tone_truth(ev, TONE))
    prom = ro.tonal.order_prominence(ev.x, FS, rpm_ev, [WHINE], min_rpm=1500.0)
    ev_level = a_weighted_level(ev.x).vs_rpm(rpm_ev, ev.t, bins=EV_EDGES)

    # The before/after pair: the whine lowered by 10 dB, everything else unchanged.
    resid, whine = ro.extract.residual(ev.x, FS, rpm_ev, [WHINE], method="vold-kalman", bandwidth=20.0)
    y = ro.resynth.resynthesize(ro.resynth.modify_orders(whine, gain_db={WHINE: -10.0}), resid)["mix"]
    okw = {"order_resolution": 0.125, "max_order": 44.0, "amplitude": "rms"}
    ev_omap, ev_omap_y = (ro.maps.rpm_order_map(s, FS, rpm_ev, **okw) for s in (ev.x, y))
    s40, s40_y = ro.slices.order_slice(ev_omap, WHINE), ro.slices.order_slice(ev_omap_y, WHINE)
    ol = ro.slices.overall_level(ev.x, FS, rpm_ev, weighting="A", rpm_bins=EV_EDGES)
    ol_y = ro.slices.overall_level(y, FS, rpm_ev, weighting="A", rpm_bins=EV_EDGES)

    # ---- one figure per function ----------------------------------------------------------
    # Maps
    ax = rp.rpm_frequency_map(fmap, **rp.SOUND_PRESSURE, mark_orders=(2, 4, 6))
    save(ax, "rpm_frequency_map")

    ax = rp.rpm_order_map(omap, **rp.SOUND_PRESSURE, mark_orders=(2, 4, 6))
    save(ax, "rpm_order_map")

    with warnings.catch_warnings():
        # At the lowest speeds the whine and its neighbours change (the Vold-Kalman band spans more than
        # one order there) below the 70 dB colour range; compare_maps says so, and the figure is meant so.
        warnings.filterwarnings("ignore", message="compare_maps", category=UserWarning)
        fig, _ = rp.compare_maps({"original": ev_omap, "order 40 -10 dB": ev_omap_y}, **rp.SOUND_PRESSURE,
                                 difference=True, dynamic_range=70, figsize=(9.0, 3.2))
    save(fig, "compare_maps")

    # Slices, spectra, levels, metrics
    ax = rp.order_slice({"order 2": s2, "order 4": s4}, **rp.SOUND_PRESSURE,
                        truth={"order 2": (rpm2, truth2), "order 4": (rpm2, truth4)})
    save(ax, "order_slice")

    ax = rp.order_slice({"original": s40, "order 40 -10 dB": s40_y}, relative_to="original")
    save(ax, "order_slice_relative_to")

    ax = rp.frequency_slice({"6.3 kHz tone": tone}, **rp.SOUND_PRESSURE, truth=(rpm_ev, tone_rms))
    save(ax, "frequency_slice")

    ax = rp.order_spectrum(spec, **rp.SOUND_PRESSURE, mark_orders=(2, 4, 6, 8))
    save(ax, "order_spectrum")

    ax = rp.overall_level({"original": ol, "order 40 -10 dB": ol_y}, **rp.SOUND_PRESSURE)
    save(ax, "overall_level")

    ax = rp.metric_vs_rpm(ev_level)
    save(ax, "metric_vs_rpm")

    ax = rp.time_metric(level)
    save(ax, "time_metric")

    # Speed, extraction, tonal
    ax = rp.rpm_vs_time(t, rpm, label="tacho_to_rpm", reference=(t, ice.rpm_mean))
    save(ax, "rpm_vs_time")

    ax = rp.rpm_from_ridge(ridge, reference=(t, ice.rpm_mean))
    save(ax, "rpm_from_ridge")

    ax = rp.order_components(whine, rpm=rpm_ev, **rp.SOUND_PRESSURE)
    save(ax, "order_components")

    ax = rp.order_table(table, **rp.SOUND_PRESSURE)
    save(ax, "order_table")

    ax = rp.order_prominence(prom)
    save(ax, "order_prominence")

    ax = rp.band_levels_vs_rpm(bands, **rp.SOUND_PRESSURE)
    save(ax, "band_levels_vs_rpm")


if __name__ == "__main__":
    main()
