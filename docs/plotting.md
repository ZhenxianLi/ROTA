# Plotting with rota.plot

`rota.plot` (since 0.7) draws the standard figure of a ROTA result in one call: an order map, a Campbell diagram, an order slice, a speed track with its doubtful frames, a tone prominence against its criterion. It takes care of what every analysis otherwise repeats by hand: the conversion to dB, axis labels with the unit, the dB reference and the amplitude convention, a colour range below the maximum, order lines, before/after overlays. It is not a plotting framework. Every function draws on an ordinary Matplotlib Axes and returns it, so titles, limits, annotations and saving stay plain Matplotlib.

This page shows one figure per function. All of them come from [examples/11_plot_gallery.py](https://github.com/ZhenxianLi/rota-order/blob/main/examples/11_plot_gallery.py), and the code under each figure is the call in that script that drew it.

## Installing and importing

Matplotlib is optional. ROTA itself depends on NumPy and SciPy only, and `import rota` never imports Matplotlib. Install the extra for the figures:

```
pip install "rota-order[plot]"
```

Without Matplotlib, a call to a `rota.plot` function raises `ImportError` with that command.

Import the module under a short name and keep the prefix:

```python
import rota
import rota.plot as rp
```

The functions have the names of the ROTA functions whose results they draw: `rp.rpm_order_map(m)` draws what `rota.maps.rpm_order_map` returned. That is why `from rota.plot import rpm_order_map` is a bad idea: next to the computing function of the same name, one import silently replaces the other.

## What to expect from every function

* **One call per result, named like the function that produced it.** `rota.slices.order_slice` returns a slice and `rp.order_slice` draws it. Where several functions return the same type, the plot is named after the type: `order_components` for the result of `order_waveform` or `tracked_harmonics`, `order_table` for an `OrderTable`, `time_metric` for a `rota.bridges.TimeMetric`. A speed given as arrays is drawn by `rpm_vs_time(t, rpm)`.
* **Composable.** Every option is a keyword. `ax=` draws on an Axes you made; without it a new figure is created, with your current Matplotlib settings. Each function returns the Axes it drew on (`compare_maps` builds a grid and returns `(fig, axes)`). Nothing calls `plt.show()` or changes `rcParams`, the style or the backend, and lines use the colour cycle of the Axes.
* **Labels with units, dB reference and amplitude convention.** The labels are read from the result: "Order level (dB re 20 µPa, RMS)", "Order level (dB re 20 µPa, peak)" for extracted envelopes, "Speed (rpm)", "Frequency (Hz)", "Time (s)". Levels are `20 log10(|a| / ref)` of values that keep their convention; no peak-to-RMS conversion happens behind your back.
* **An explicit reference.** Level plots take `ref` (in the unit of the signal) and `unit`, which only labels. The default is `ref=1.0, unit=""` ("dB re 1"). For sound pressure in Pa write `**rp.SOUND_PRESSURE`, which is `{"ref": 2e-5, "unit": "Pa"}` and gives "dB re 20 µPa". Results that store levels (`overall_level`, `band_levels_vs_rpm`) are drawn as stored with the default `ref=None`; a given `ref` re-references them exactly.
* **Before and after in one call.** Line plots take one result, a list, or a dict `{label: result}` whose keys become the legend. `relative_to=<label>` draws every entry minus that one (in dB for levels, in the unit of a metric). `compare_maps` puts maps side by side on one colour scale, with the dB difference if you ask for it.
* **`orders=` selects, `mark_orders=` marks.** `orders=` picks which orders of a result are drawn, as in the computing functions (`order_components`, `order_table`, `order_prominence`). `mark_orders=` draws thin order lines on a figure that is shown whole (the maps, `compare_maps`, `order_spectrum`).
* **The speed axis of maps, `x="auto"`.** A map is drawn against speed when the frame speed is essentially monotonic and against time otherwise. Essentially monotonic means a run-up or a coast-down whose reversals are at most `monotonic_tol` (2 % by default) of the speed range: a measured speed is rarely monotonic to the last digit. A run-up followed by a coast-down, or the drop of a gear shift, would fold frames onto each other, so it is drawn against time, and `x="rpm"` raises `ValueError` with the two ways out (`rota.maps.bin_map`, or `x="time"`). Frames at a constant speed, such as an idle hold, collapse onto one rpm; draw those with `x="time"` (example 01 does). Line plots with `x="rpm"` draw the true speeds, and a line may double back where the speed does.
* **Honest data.** NaN and zero amplitudes stay gaps; maps are drawn cell by cell on their true axes, without resampling or smoothing.

A figure of several panels is ordinary Matplotlib:

```python
import matplotlib.pyplot as plt

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
rp.rpm_order_map(omap, ax=a, **rp.SOUND_PRESSURE, max_order=6)
rp.order_slice({"order 3": s3}, ax=b, **rp.SOUND_PRESSURE)
b.set_ylim(80, 100)
b.set_title("Order 3")
fig.savefig("run_up.png", dpi=150)
```

Examples [01](https://github.com/ZhenxianLi/rota-order/blob/main/examples/01_order_map.py) and [02](https://github.com/ZhenxianLi/rota-order/blob/main/examples/02_tacho_and_tacholess_speed.py) build their figures this way.

## The signal of the gallery

A seeded signal from `rota.synth`: 6 s at 8 kHz, a linear run-up from 1000 to 4500 rpm with six orders (0.5, 1, 1.5, 3, 4.5 and 6), a structural resonance at 180 Hz that amplifies whichever order passes it, a fixed 250 Hz tone, white noise and a tachometer of 12 pulses per revolution. It is read as sound pressure in Pa. For the before/after figures, order 3 is extracted with the Vold-Kalman filter, lowered by 10 dB with `rota.resynth.modify_orders` and added back to the residual with `resynthesize`:

```python
resid, c3 = rota.extract.residual(x, FS, rpm, [3.0], method="vold-kalman")
y = rota.resynth.resynthesize(rota.resynth.modify_orders(c3, gain_db={3.0: -10.0}), resid)["mix"]
```

The figures are 6 by 3.6 inches: the script sets `figure.figsize` and constrained layout once in Matplotlib's `rcParams`, and `rota.plot` creates its figures with them. The maps are RMS (`amplitude="rms"`).

## Maps

### rpm_frequency_map

The Campbell diagram: level against speed and frequency. Orders are rays, the 250 Hz tone is a horizontal line. The marked orders are thin dashed lines at f = order × rpm / 60, labelled at the right edge. The run-up is monotonic, so `x="auto"` puts speed on the x axis. The colour range is 40 dB below the maximum (`dynamic_range=40`), so the noise floor is dark.

![Campbell diagram](images/plot_rpm_frequency_map.png)

```python
fmap = rota.maps.rpm_frequency_map(x, FS, rpm, resolution=4.0, amplitude="rms", max_freq=600.0)
ax = rp.rpm_frequency_map(fmap, **rp.SOUND_PRESSURE, mark_orders=(1.5, 3, 4.5, 6))
```

### rpm_order_map

The order map: orders are horizontal lines, the fixed tone bends into a hyperbola, and each order brightens where it passes the 180 Hz resonance (order 6 at 1800 rpm, order 4.5 at 2400 rpm, order 3 at 3600 rpm). A map cell reads the level of its bin; read the level of an order with `order_slice`, which sums the power over a band.

![Order map](images/plot_rpm_order_map.png)

```python
omap = rota.maps.rpm_order_map(x, FS, rpm, order_resolution=0.125, max_order=8.0, amplitude="rms")
ax = rp.rpm_order_map(omap, **rp.SOUND_PRESSURE, mark_orders=(1.5, 3, 4.5, 6))
```

### compare_maps

Maps side by side on one colour scale, so equal colours mean equal levels, with the dict keys as titles. `difference=True` adds the level difference to the first map on a diverging scale centred on 0 dB: here order 3 is 10 dB lower and everything else is unchanged. The difference scale spans the 99.5th percentile of the absolute differences over the cells that are shown (at least one map above the bottom of the colour scale), so a few noise cells with a large difference saturate instead of stretching it, and arrows on the colour bar show that some cells lie beyond it (`difference_range=` sets it). The maps may also be of different domains, for example a Campbell diagram next to the order map of the same run, on one colour scale.

![Order maps before and after lowering order 3 by 10 dB, and their difference](images/plot_compare_maps.png)

```python
fig, _ = rp.compare_maps({"original": omap, "order 3 -10 dB": omap_y}, **rp.SOUND_PRESSURE,
                         difference=True, figsize=(9.0, 3.2))
```

## Slices, spectra, levels, metrics

### order_slice

The level of one order against speed, with an analytic curve for comparison (`truth=`, in the same convention as the slice: here the RMS amplitude put into the signal). For a dict of slices, `truth` can be a dict with the same keys: each truth is drawn thin and dashed in a lighter shade of its slice's colour, on top of the slice, and labelled "<key> truth". Where slice and truth agree, the dashes lie on the line; the truth shows alone at the ends of the run, where the slice has no frame centre. Order 3 peaks at 3600 rpm, where it passes the resonance.

![Order slices of orders 1.5 and 3 against the truth](images/plot_order_slice.png)

```python
s15, s3 = rota.slices.order_slice(omap, 1.5), rota.slices.order_slice(omap, 3.0)
ax = rp.order_slice({"order 1.5": s15, "order 3": s3}, **rp.SOUND_PRESSURE,
                    truth={"order 1.5": (rpm, truth15), "order 3": (rpm, truth3)})
```

### order_slice with relative_to

The before/after view: every entry of the dict minus the one named by `relative_to`, in dB. The slice of the modified signal sits at -10.0 dB (median; -10.4 to -9.3 dB over the run). No reference is needed for a difference.

![Order 3 after the modification, relative to the original](images/plot_order_slice_relative_to.png)

```python
ax = rp.order_slice({"original": s3, "order 3 -10 dB": s3_y}, relative_to="original")
```

### frequency_slice

The level of one fixed frequency against speed, from the Campbell diagram: the 250 Hz tone, at 75.3 dB re 20 µPa after the resonance (the truth). The two peaks are orders crossing 250 Hz: order 6 at 2500 rpm and order 4.5 at 3333 rpm.

![Frequency slice at 250 Hz](images/plot_frequency_slice.png)

```python
tone = rota.slices.frequency_slice(fmap, 250.0)
ax = rp.frequency_slice({"250 Hz": tone}, **rp.SOUND_PRESSURE, truth=(rpm, tone_rms))
```

### order_spectrum

The energy average of the order map over the run, against order, with the orders of the signal marked. Each point is the level of one bin.

![Order spectrum](images/plot_order_spectrum.png)

```python
spec = rota.slices.order_spectrum(omap)
ax = rp.order_spectrum(spec, **rp.SOUND_PRESSURE, mark_orders=(0.5, 1, 1.5, 3, 4.5, 6))
```

### overall_level

The A-weighted overall level per speed bin, before and after: lowering order 3 by 10 dB lowers the A-weighted level by up to 5.6 dB around the resonance and by less than 0.4 dB below 2500 rpm. The label names the weighting and the reference.

![A-weighted overall level before and after](images/plot_overall_level.png)

```python
ol = rota.slices.overall_level(x, FS, rpm, weighting="A", rpm_bins=EDGES)
ax = rp.overall_level({"original": ol, "order 3 -10 dB": ol_y}, **rp.SOUND_PRESSURE)
```

`overall_level` stores levels, so the default `ref=None` draws them as stored and the label states the stored reference as a number ("dB re 2e-05"); `**rp.SOUND_PRESSURE` gives the same numbers labelled "dB re 20 µPa".

### metric_vs_rpm

A metric aggregated in speed bins, with its spread (± one standard deviation, or the interquartile range for a median or a percentile) as a band; bins with too few samples are gaps. Normally the metric comes from a psychoacoustic toolbox through `rota.bridges` (`bridges.mosqito.loudness(x, fs).vs_rpm(rpm, t)`). This gallery needs no toolbox: the metric is the A-weighted level with Fast time weighting computed by ROTA (`rota.tonal.level_vs_time`), wrapped in a `TimeMetric` by hand with a 0.5 s start-up exclusion and the energy mean as aggregation. The first bin falls inside the exclusion and is a gap.

![A-weighted Fast level aggregated in 250 rpm bins](images/plot_metric_vs_rpm.png)

```python
level = a_weighted_level(x)
level_vs_rpm = level.vs_rpm(rpm, t, bins=EDGES)
ax = rp.metric_vs_rpm(level_vs_rpm)
```

where the function `a_weighted_level` of the script does, in short:

```python
t, level = rota.tonal.level_vs_time(rota.tonal.weight(x, FS, "A"), FS, frame=0.05)
TimeMetric(name="A-weighted level, Fast", t=t, values=level, unit="dB re 20 µPa",
           skip_start=0.5, aggregate="power", source="rota.tonal.level_vs_time")
```

### time_metric

The same `TimeMetric` against time. The start-up exclusion (`skip_start`, here 0.5 s from the first value) is shaded: a time-weighted meter starts at zero and reads low for a few time constants.

![A-weighted Fast level against time](images/plot_time_metric.png)

```python
ax = rp.time_metric(level)
```

## Speed, extraction, tonal

### rpm_vs_time

A speed given as arrays, here from the tachometer (`tacho_to_rpm`), with the true speed as the reference. A reference (and a single `truth=` curve of the slice figures) is drawn in one neutral style: thin, dark grey, dashed, on top of the data. The scatter at high speed is the 30 µs timing jitter of the pulses.

![Speed from the tachometer](images/plot_rpm_vs_time.png)

```python
rpm_tacho = rota.speed.tacho_to_rpm(sig.tacho, FS, ppr=PPR)
ax = rp.rpm_vs_time(t, rpm_tacho, label="tacho_to_rpm", reference=(t, rpm))
```

### rpm_from_ridge

The tacholess speed track with the frames that must not be trusted. The search range is deliberately too narrow here (600 to 3500 rpm, dotted): from 4.28 s, when the true speed leaves the range, every frame is flagged `at_bound` (open dark-red circles), first on the bound and then at a wrong speed inside the range. Black crosses mark frames whose `confidence` is below 0.1; a frame with both flags shows a cross inside a circle. The marker colours are fixed, not taken from the colour cycle. Read the two markers together: a wrong track can have a high confidence. `label=` sets the legend text of the track.

![Tacholess speed track with a search range that is too narrow](images/plot_rpm_from_ridge.png)

```python
ridge = rota.speed.rpm_from_ridge(x, FS, (600.0, 3500.0), orders=(1.5, 3.0, 4.5, 6.0), whiten=None)
ax = rp.rpm_from_ridge(ridge, reference=(t, rpm))
```

### order_components

The envelopes of extracted orders against speed. Extractors return peak envelopes, and the label says "peak". The components carry samples, not speeds, so the speed is passed as `rpm=`. Order 6 passes the resonance at 1800 rpm; the ripple near 2500 rpm is the 250 Hz tone, which order 6 crosses there.

![Envelopes of orders 1.5, 3 and 6 from tracked harmonics](images/plot_order_components.png)

```python
comp = rota.extract.tracked_harmonics(x, FS, rpm, [1.5, 3.0, 6.0], window_revs=8.0)
ax = rp.order_components(comp, rpm=rpm, **rp.SOUND_PRESSURE)
```

### order_table

The rows of an `OrderTable` against its speed grid, here built from order slices in 250 rpm bins; the input of `rota.resynth.synthesize_orders`. The table keeps the convention of the slices (RMS) and its unit (Pa).

![Order table of orders 1.5, 3 and 4.5](images/plot_order_table.png)

```python
table = rota.resynth.OrderTable.from_slices(
    [rota.slices.order_slice(omap, o) for o in (1.5, 3.0, 4.5)], EDGES[:-1] + 125.0, unit="Pa"
)
ax = rp.order_table(table, **rp.SOUND_PRESSURE)
```

### order_prominence

The tone-to-noise ratio of orders against speed (`kind="pr"` for the prominence ratio), with the ECMA-418-1 criterion of each order dashed in its colour and the speed ranges where the order is prominent shaded. The criterion is defined from 89.1 Hz to 11.2 kHz, so order 3 has none at the lowest point (76 Hz). Below 2000 rpm orders 3 and 4.5 are not prominent: other orders of the signal lie in the same critical band. Prominence here is the objective criterion only; the aural check of the standard is not assessed.

![Tone-to-noise ratio of orders 3 and 4.5 against the ECMA-418-1 criterion](images/plot_order_prominence.png)

```python
prom = rota.tonal.order_prominence(x, FS, rpm, [3.0, 4.5], rpm_bins=EDGES)
ax = rp.order_prominence(prom)
```

### band_levels_vs_rpm

1/3-octave band levels against speed, as an image on the band limits (logarithmic frequency axis, nominal mid-band frequencies as labels). The colour is the RMS level of the whole band, not a spectral density.

![1/3-octave band levels against speed](images/plot_band_levels_vs_rpm.png)

```python
bands = rota.tonal.band_levels_vs_rpm(x, FS, rpm, rpm_bins=EDGES, frame=0.1, fmin=25.0, fmax=3150.0)
ax = rp.band_levels_vs_rpm(bands, **rp.SOUND_PRESSURE)
```

## References

Every function lists its references in its docstring. For the figures as such: Brandt, A. (2011). *Noise and Vibration Analysis*, ch. 12. Wiley. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95. Crameri, F., Shephard, G. E., & Heron, P. J. (2020). The misuse of colour in science communication. *Nature Communications*, 11, 5444.
