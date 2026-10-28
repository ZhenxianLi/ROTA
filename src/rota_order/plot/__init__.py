"""Standard figures of rota-order results, on Matplotlib (optional dependency, since 0.7.0).

``rota_order.plot`` removes the boilerplate that every analysis repeats by hand: the
conversion to dB, labels such as "Order level (dB re 20 µPa, RMS)", a colour range
below the maximum, order lines on a Campbell diagram, an rpm axis, before/after
overlays. It is not a plotting framework. Each function draws the standard figure
of one rota-order result on a Matplotlib Axes and returns that Axes, so everything
else (titles, limits, annotations, saving) is ordinary Matplotlib::

    import rota_order as ro
    import rota_order.plot as rp

    omap = ro.maps.rpm_order_map(x, fs, rpm, amplitude="rms")
    ax = rp.rpm_order_map(omap, **rp.SOUND_PRESSURE, mark_orders=(2, 4, 6))
    ax.figure.savefig("order_map.png", dpi=150)

Importing. Each function is named like the rota-order function that returned the result
it draws: ``rota_order.plot.rpm_order_map(m)`` draws what ``rota_order.maps.rpm_order_map``
computed. Some names are therefore the same as those of the computing functions.
Use the module prefix (``rp.rpm_order_map``, ``rota_order.plot.rpm_order_map``) and do
not import the bare names (``from rota_order.plot import rpm_order_map``) next to the
computing functions: the second import would silently replace the first.

Optional Matplotlib. rota-order depends on NumPy and SciPy only. Matplotlib is imported
inside the functions of ``rota_order.plot`` when a figure is drawn, never by
``import rota_order`` or ``import rota_order.plot``. Without it a call raises ImportError
with the install command, ``pip install 'rota-order[plot]'``.

Principles.

* One call per result, one function per rota-order function that returns a result;
  options are keyword-only.
* Composable. ``ax=None`` is the first option: a new figure is created only when
  no Axes is given, and the Axes drawn on is returned. Nothing calls
  ``plt.show()``, changes ``rcParams``, the style or the backend; the figures
  use the active style and colour cycle. ``compare_maps`` builds a figure of
  several panels and returns ``(fig, axes)``.
* Correct labels by default, read from the result: the quantity, the unit, the
  dB reference and the amplitude convention (peak or RMS) are in the axis and
  colour bar labels ("Level (dB re 20 µPa, RMS)", "Speed (rpm)", "Order",
  "Frequency (Hz)", "Time (s)").
* The dB reference is explicit. Level plots take ``ref`` (in the unit of the
  signal) and ``unit``, defaulting to ``ref=1.0, unit=""`` ("dB re 1");
  :data:`SOUND_PRESSURE` is the explicit choice for sound pressure in Pa,
  ``rp.rpm_order_map(m, **rp.SOUND_PRESSURE)``. Levels are ``20 log10(|a| / ref)``
  of values that keep their amplitude convention (peak or RMS); no convention is
  converted behind your back.
* Honest data. NaN and zero amplitudes stay gaps. Maps are drawn with
  ``pcolormesh`` on their true axes, without resampling or smoothing.
* The speed axis. A measured speed (a tachometer, a ridge) is a run-up or a
  coast-down with small reversals, so the rule is "essentially monotonic": taking
  the overall direction from the sign of last minus first speed, every reversal
  (a drop below the running maximum of a run-up, a rise above the running minimum
  of a coast-down) is at most ``monotonic_tol`` of the speed range (2 % by
  default). A run-up followed by a coast-down, or the drop of a gear shift, is not
  essentially monotonic. In the maps ``x="auto"`` gives rpm for an essentially
  monotonic speed and time otherwise, and ``x="rpm"`` raises ValueError for any
  other (a map binned along speed with :func:`rota_order.maps.bin_map` is drawn on its
  speed bins). With rpm on the x axis the cells are placed on the monotone
  envelope of the frame speed (running maximum or minimum), so that they never
  overlap; a frame is displaced by at most its reversal. Line plots with
  ``x="rpm"`` always draw the true speeds; a line may double back, which is the
  data.
* Colour. Maps use a perceptually uniform colour map (``viridis``), a colour
  range of ``dynamic_range`` dB (default 40) below the maximum of what is shown,
  and a colour bar by default. Differences use a diverging map centred on 0 dB.
* Results that store levels (:func:`overall_level`, :func:`band_levels_vs_rpm`)
  are plotted as stored with ``ref=None``; a given ``ref`` re-references them
  exactly, ``L + 20 log10(ref_stored / ref)``, and ``unit`` only labels.
* ``orders=`` selects orders, as in the computing functions
  (:func:`order_components`, :func:`order_table`, :func:`order_prominence`);
  ``mark_orders=`` draws order lines on a figure that is shown whole
  (:func:`rpm_frequency_map`, :func:`rpm_order_map`, :func:`compare_maps`,
  :func:`order_spectrum`).
* Multichannel results: ``channel=0`` selects the channel.

Functions
---------
Maps
    rpm_frequency_map   Campbell diagram of a frequency map, with order lines.
    rpm_order_map       Order map, with order lines.
    compare_maps        Several maps side by side on one colour scale, with the
                        dB difference to the first.
Slices, spectra, levels, metrics
    order_slice         Level of one order against speed or time.
    frequency_slice     Level of one fixed frequency against speed or time.
    order_spectrum      Energy-averaged spectrum against order or frequency.
    overall_level       Overall level per speed bin or per frame.
    metric_vs_rpm       Metric binned along speed, with its spread band.
    time_metric         Psychoacoustic metric against time.
Speed, extraction, tonal
    rpm_vs_time         Speed curve against time, e.g. from a tachometer.
    rpm_from_ridge      Ridge speed track with its doubtful frames.
    order_components    Envelopes of extracted orders.
    order_table         Rows of an order table against speed.
    order_prominence    Tone-to-noise or prominence ratio with the ECMA-418-1
                        criterion.
    band_levels_vs_rpm  Band levels against speed and band, as an image.

Constants
---------
SOUND_PRESSURE
    ``{"ref": 2e-5, "unit": "Pa"}``: the reference sound pressure of ISO 1683 (20
    µPa), to be unpacked into the ``ref`` and ``unit`` options.

References
----------
Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
Science & Engineering, 9(3), 90-95.
Crameri, F., Shephard, G. E., & Heron, P. J. (2020). The misuse of colour in
science communication. Nature Communications, 11, 5444.
"""

from ._analysis import (
    band_levels_vs_rpm,
    order_components,
    order_prominence,
    order_table,
    rpm_from_ridge,
    rpm_vs_time,
)
from ._core import SOUND_PRESSURE
from ._lines import (
    frequency_slice,
    metric_vs_rpm,
    order_slice,
    order_spectrum,
    overall_level,
    time_metric,
)
from ._maps import compare_maps, rpm_frequency_map, rpm_order_map

__all__ = [
    "SOUND_PRESSURE",
    "band_levels_vs_rpm",
    "compare_maps",
    "frequency_slice",
    "metric_vs_rpm",
    "order_components",
    "order_prominence",
    "order_slice",
    "order_spectrum",
    "order_table",
    "overall_level",
    "rpm_frequency_map",
    "rpm_from_ridge",
    "rpm_order_map",
    "rpm_vs_time",
    "time_metric",
]
