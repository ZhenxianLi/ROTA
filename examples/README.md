# Examples

Four runnable scripts that show the main parts of ROTA on synthetic signals. All data come from `rota.synth`, so every result can be compared with the analytic truth that went into the signal, and each script prints those comparisons as numbers before it saves one figure.

## Running

Matplotlib is needed for the figures only (the library itself depends on NumPy and SciPy). From the repository root:

```
pip install -e ".[plot]"
python examples/01_order_map.py
python examples/02_tacho_and_tacholess_speed.py
python examples/03_order_extraction.py
python examples/04_order_table_resynthesis.py
```

The scripts can be started from any directory. Each runs in a few seconds (about 2 s, 2 s, 8 s and 5 s) and writes its figure to `docs/images/`, 8 inches wide at 150 dpi. Random elements use fixed seeds, so the numbers and figures are reproducible.

## The signal

Scripts 1 to 3 use the same 6 s signal at 8 kHz: an idle at 900 rpm for 1.2 s, then a linear run-up to 4500 rpm. It contains six orders (0.5, 1, 1.5, 3, 4.5 and 6, so that half orders are present), a structural resonance at 180 Hz that amplifies whichever order passes through it, a fixed 250 Hz tone and white noise. The signal is read as sound pressure in Pa; levels are in dB re 20 uPa, and the reference pressure is a named constant in each script. Amplitude conventions are stated in each script: order maps, slices and spectra are RMS, extractor envelopes and synthesised sinusoids are peak.

## 01_order_map.py

Builds the Campbell diagram (`rpm_frequency_map`) and the order map (`rpm_order_map`) of the signal and reads two order slices (`order_slice`) from the order map. In the Campbell diagram the orders are rays and the tone is a horizontal line, but at idle the half orders merge, because a fixed-length time window spans several of them. In the order map the orders are horizontal lines and stay separated at idle; the tone bends into a hyperbola and each order brightens where it passes the resonance. The slices of orders 1.5 and 3 are compared with the amplitude put into the signal, including the passage of order 3 through the resonance. The script prints the slice error in dB for all six orders (idle, run-up median and run-up maximum). It uses an order resolution of 0.125, which is what separates orders half an order apart; frames in which the fixed tone lies inside an order's band are left out of the statistics.

![Campbell diagram, order map and order slices](../docs/images/01_order_map.png)

Figure: `docs/images/01_order_map.png`.

## 02_tacho_and_tacholess_speed.py

Compares two speed estimates with the true speed. The signal gets a 12-pulse tachometer with timing jitter and dropped pulses, which `tacho_to_rpm` converts to rpm. `rpm_from_ridge` estimates the speed from the signal alone, with a comb of the half-order family (1.5, 3, 4.5, 6) and without spectral whitening, which the dense, fast-sweeping lines of this signal call for (the comments in the script explain both). The figure shows the speed curves, their relative errors, and a second tacholess run whose search range is deliberately too narrow: above the upper limit the estimate does not sit on the bound but jumps to a wrong value, and only `at_bound` and the collapse of `confidence` reveal it. The script prints the speed errors, the number of pulses filled in, and how many frames of the narrow run were flagged.

![Tacho and tacholess speed](../docs/images/02_tacho_and_tacholess_speed.png)

Figure: `docs/images/02_tacho_and_tacholess_speed.png`.

## 03_order_extraction.py

Extracts the six orders with the Vold-Kalman filter (`order_waveform`) and with phase-locked tracked harmonics (`tracked_harmonics`), after `resolution_check` has confirmed that the settings separate half orders at idle. The figure compares the peak envelope of order 3 with the truth for both extractors and shows the level error. It then removes all six orders (`remove_orders`) and shows the order map before and after, and the average order spectrum of the original, of both residuals and of the true non-order content (tone and noise), which is what an ideal removal would leave. The script prints envelope errors and waveform correlation per order, and the energy left at each removed order (`order_residual_energy`). The extractors do not know about the fixed tone, so orders 4.5 and 6 pick it up briefly where they cross it; those frames are left out of the error statistics, and the true non-order content serves as the floor for the residual energy.

![Order extraction and removal](../docs/images/03_order_extraction.png)

Figure: `docs/images/03_order_extraction.png`.

## 04_order_table_resynthesis.py

Analyses a run-up and builds a different signal from the analysis alone. The orders are read from an order map and binned into an order table (`OrderTable.from_slices`), which `synthesize_orders` renders along another speed profile, an S-shaped run-down from 4200 to 1100 rpm. The broadband part comes from the Welch PSD of the residual left by `remove_orders`, rendered with `synthesize_broadband`. The sum is analysed again and compared with a reference made by `rota.synth` along the same run-down. The figure shows the table, the two speed profiles, the order maps of the resynthesis and of the reference, two order slices, and the target and synthesised broadband PSD. The script prints the order-level errors of the resynthesis against the table (round trip) and against the reference, the level and PSD match of the broadband part, and the total level. This script uses the signal without the fixed tone: a fixed line is neither an order nor broadband noise, and in the residual PSD it would be a spike narrower than the envelope resolution of the broadband synthesiser. In an order table it belongs in its own row, with order 0 and a `freq_offset`.

![Order table resynthesis](../docs/images/04_order_table_resynthesis.png)

Figure: `docs/images/04_order_table_resynthesis.png`.
