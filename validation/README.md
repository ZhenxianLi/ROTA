# Validation cases (v0.1 and v0.2)

Eleven of the validation signals defined in `ROADMAP.md` section 5.3 are implemented: T1, T2, T5 and T14 (v0.1) and T3, T4, T6, T8, T10, T12 and T13 (v0.2). Each is a function in `make_signals.py` that returns a `rota.synth.SynthSignal` built from explicit, documented parameters and a fixed seed, so the signal and its analytic truth are bit-identical on every run. The parameters are documented in each function's docstring.

| Case | Signal | What it examines |
|---|---|---|
| T1 | One order (2), 1 Pa peak, constant; linear run-up 1000 to 4000 rpm in 10 s | Absolute accuracy of order amplitude against speed |
| T2 | Orders 1, 2, 3 (with a speed gain law) and 4.5; a 200 Hz resonance (Q 8, gain 4) that orders 4.5 and 3 sweep through; a fixed 450 Hz tone; white noise (0.02 Pa RMS); run-up 1000 to 5000 rpm in 12 s | Maps, slices, resonance handling, noise |
| T3 | Orders 1, 2, 3 (gain law with a dip) and 5; exponential run-down 4000 to 1000 rpm in 8 s (relative rate -17 %/s); noise 0.01 Pa RMS; 4 kHz | Angle resampling, order maps, both extractors and `rpm_from_ridge` when the speed is not linear in time |
| T4 | Two shafts, 1200 to 3600 rpm up and 3600 to 1200 rpm down; orders 2 and 3 on each; their frequencies cross three times (2.4, 4.0 twice, 5.6 s); noise 0.01 Pa RMS; 4 kHz | Decoupling of crossing orders by the joint Vold-Kalman filter |
| T5 | Speed with idle, an S-curve run-up and a hold (800 to 5000 rpm in 15 s); tachometer with 60 pulses per revolution (`T5`) or one (`T5_ppr1`), 5 us timing jitter, 0.5 % dropped pulses; 48 kHz | `rota.speed.tacho_to_rpm` |
| T6 | Orders 2, 4 (gain hump) and 6; linear 1000 to 6000 rpm in 1 s (5000 rpm/s, 83 rev/s^2; `t6(duration=...)` changes the slew rate); noise 0.01 Pa RMS; 4 kHz | Computed order tracking and both extractors at a high slew rate |
| T8 | Two shafts, the second 0.5 % (`T8_r1005`), 2 % (`T8_r1020`) or 4 % (`T8_r1040`) faster; order 4 on each (1.0 and 0.8 Pa, amplitude-modulated at 0.4 and 0.25 Hz); 2400 to 3000 rpm in 10 s; noise 0.02 Pa RMS; 4 kHz | Resolvability limit, `resolution_check` must warn when the twins are not separated |
| T10 | Orders 1 to 4; linear 600 to 3400 rpm in 12 s; searched in the range 1200 to 2800 rpm (and 1500 to 2500); noise 0.05 Pa RMS; 8 kHz | `rpm_from_ridge` must flag the frames whose true speed is outside the range |
| T12 | The T2 signal (`T12` is a copy of `T2`) | `remove_orders` and `residual` on a signal with resonance, tone and noise |
| T13 | T2 with 4 kHz sampling and 0.01 Pa noise; `t13_references()` gives the speed with a 0.5 % bias, a 0.5 % RMS jitter (low-passed at 20 Hz) and a 1 % step at 6 s | Degradation of both extractors with errors in the speed reference |
| T14 | Idle at 800 rpm for 2 s, then run-up to 3000 rpm; orders 1, 2 and 2.5 (2 and 2.5 equal in amplitude) | Order resolution at low speed: angle-domain map against an equal-bandwidth time-domain map |

T7, T9 and T11 are not part of v0.1 or v0.2 (T7 needs rights-cleared measurements, T9 and T11 belong to the resynthesis of v0.3).

## Regenerating the signals

```
python validation/make_signals.py            # all cases, to validation/signals/
python validation/make_signals.py T1 T14     # selected cases
python validation/make_signals.py --out DIR  # another output directory
```

The case names are `T1 T2 T3 T4 T5 T5_ppr1 T6 T8_r1005 T8_r1020 T8_r1040 T10 T12 T13 T14`. Each case is written as `<name>.npz` (signal, speed, shaft angle, and the truth: amplitude, phase and frequency of every order, the component stems, tachometer pulse times) and `<name>.json` (parameters, resolved random phases, seed, ROTA version); `T13` also writes `T13_references.npz` with the perturbed speed references. About 290 MB in total, mostly the two 48 kHz T5 files and the copy of T2 that is T12. `validation/signals/` is git-ignored. The tests do not need these files: they build the signals in memory from the same functions.

## Tests and tolerances

`tests/test_validation.py` (v0.1) and `tests/test_validation_v02.py` (v0.2; `tests/test_extract.py` and `tests/test_resolution.py` complete it) run the cases with pytest (about 25 s for the v0.2 files) and check them against the truth stored in the signal. Expected amplitudes are always read from `order_amplitude` (or the stems), never from the parameter lists, because a resonance or a gain law changes the amplitude actually present in the signal (ROADMAP section 6, item 19). The expected reading of a map frame is the window-weighted mean of that truth over the frame.

The tolerances are the initial values of ROADMAP section 5.4, column "against analytic truth":

| Quantity | Tolerance | Where |
|---|---|---|
| Speed, median relative error | 0.1 % | T5, both pulse rates |
| Order amplitude, stationary segments (and, in these cases, away from the resonance) | 0.2 dB | T1 (order map and frequency map), T2, T14 |
| Order amplitude near a resonance | 1 dB | T2: frames in which the order is within 2 f0/Q of the resonance frequency f0 |
| Order amplitude, order map slices and extractor envelopes | 0.2 dB | T3 (all three), T4 (joint filter), T6 (all three), T8 at 2 % and 4 % |
| Order waveform correlation, ends removed | 0.99 | T3, T4, T6, T8 at 2 % and 4 %, T13 with the exact speed |
| Tacholess speed, share of frames within 3 % | 90 % | T3 (whole record), T10 (frames inside the range) |
| Energy left at a removed order | -20 dB | T12, each of the four orders, both extractors |

Ends are removed where the method says its output is not to be trusted: `1.5 / bandwidth` seconds for the Vold-Kalman filter (its free-end transient has decayed to 3e-4), `window_revs` revolutions for tracked harmonics, three frames for an order map.

Checks that belong to the v0.1 cases but have no row in section 5.4: the number of missing pulses reported by `tacho_to_rpm` equals the number dropped (T5); the fixed tone is the largest line near its frequency in every frame of the frequency map (T2); the average order spectrum peaks at order 2 (T1); and with the same window duration as the angle-domain map has at top speed, a time-domain map cannot separate orders 2 and 2.5 at idle, reading them too high by more than 1 dB (T14, ROADMAP section 6, item 20).

## Results of the v0.2 cases

Settings: Vold-Kalman `bandwidth` B in Hz (the -3 dB offset from the carrier, pass band 2 B), tracked harmonics `window_revs` W, order maps with `order_resolution = 0.25`. Worst case over all orders of the case, ends removed.

| Case | Quantity | Worst case | Tolerance |
|---|---|---|---|
| T3 | Order slice of the map, amplitude error | 0.079 dB | 0.2 dB |
| T3 | Vold-Kalman (B 2) / tracked harmonics (W 8) envelope error | 0.076 / 0.055 dB | 0.2 dB |
| T3 | Waveform correlation, both extractors | 0.999998 | 0.99 |
| T3 | `rpm_from_ridge`, frames within 3 % (median error) | 100 % (0.22 %) | 90 % |
| T4 | Joint filter (B 2): correlation / envelope error | 0.999999 / 0.022 dB | 0.99 / 0.2 dB |
| T4 | Within 0.5 s of a crossing, joint: NRMSE | 0.0018 | |
| T4 | Within 0.5 s of a crossing, one filter per order: correlation (NRMSE) | 0.82 to 0.98 (0.20 to 0.82) | 0.99 |
| T6 | Vold-Kalman (B 16): correlation / envelope error | 0.99998 / 0.125 dB | 0.99 / 0.2 dB |
| T6 | Tracked harmonics (W 4): correlation / envelope error | 0.99999 / 0.157 dB | 0.99 / 0.2 dB |
| T6 | Order map slice, amplitude error | 0.146 dB | 0.2 dB |
| T10 | Range 1200 to 2800 rpm: frames inside within 3 % | 100 % (137 frames) | 90 % |
| T10 | Range 1200 to 2800 rpm: frames more than 2 % outside flagged `at_bound` | 100 % (89 frames) | all |
| T10 | Range 1500 to 2500 rpm: inside within 3 % / more than 2 % outside flagged | 100 % (87) / 100 % (142) | 90 % / all |
| T12 | Energy left at orders 1, 2, 3 and 4.5 (Vold-Kalman, B 2) | -49.4, -55.0, -48.1, -46.4 dB | -20 dB |
| T12 | The same (tracked harmonics, W 8) | -48.9, -55.7, -48.7, -45.9 dB | -20 dB |
| T12 | Residual of the four orders against tone plus noise: correlation | 0.99997 / 0.99993 | 0.99 |

Reading the numbers. The 0.1 to 0.15 dB errors of T6 are noise (0.01 Pa on a 0.3 Pa order), not slew; the energy left at a removed order in T12 is the broadband noise inside the 0.5-order analysis band of the metric.

### T6: slew rate

The same speed range and amplitude laws at four slew rates (signals `t6(duration=...)`, worst envelope error in dB over the three orders, with noise as in T6; and without noise, which isolates the error caused by the sweep):

| Slew rate (rpm/s) | 500 | 2000 | 5000 (T6) | 10 000 |
|---|---|---|---|---|
| Vold-Kalman, B 16, with noise | 0.142 | 0.101 | 0.125 | 0.108 |
| Tracked harmonics, W 4, with noise | 0.103 | 0.077 | 0.157 | 0.340 |
| Order map slice, with noise | 0.128 | 0.113 | 0.146 | 0.075 |
| Vold-Kalman, B 16, no noise | 0.006 | 0.023 | 0.056 | 0.113 |
| Tracked harmonics, W 4, no noise | 0.017 | 0.068 | 0.171 | 0.344 |
| Order map slice, no noise | 0.002 | 0.010 | 0.025 | 0.049 |

The error due to the sweep grows in proportion to the slew rate: about 0.011 dB per 1000 rpm/s for the Vold-Kalman filter with B 16, 0.034 dB per 1000 rpm/s for tracked harmonics with W 4 (the first four-revolution window lasts 0.17 s, in which the speed rises by 84 % at 5000 rpm/s) and 0.005 dB per 1000 rpm/s for the order map. The 0.2 dB tolerance is therefore reached by the sweep alone at about 6000 rpm/s for tracked harmonics with W 4 (use a shorter window at higher rates), and only at much higher rates for the Vold-Kalman filter and the map (extrapolation of this signal family; the tests cover up to 10 000 rpm/s). The map slices read 0.13 to 0.15 dB with noise at all slew rates; a coarser map (resolution 0.5) reads 0.225 dB with noise at 5000 rpm/s but 0.012 dB without: its band is twice as wide and takes in more noise, which is not an effect of the sweep.

### T8: twin shafts

Min. separation is the beat frequency at the start of the record. The settings are B 1 Hz (pass band 2 Hz) and W 32 revolutions. "Warns" is `resolution_check` with that setting. Correlation of the two orders (shaft 0, shaft 1), ends removed.

| Ratio | Min. separation | Warns (B 1 / W 32) | Joint filter, B 1 | One filter per order, B 1 | Tracked harmonics, W 32 |
|---|---|---|---|---|---|
| 1.005 | 0.8 Hz | yes / yes | 0.9989, 0.9982 | 0.8513, 0.6865 | 0.9025, 0.8139 |
| 1.02 | 3.2 Hz | no / no | 1.0000, 1.0000 | 0.9999, 0.9998 | 0.9998, 1.0000 |
| 1.04 | 6.4 Hz | no / no | 1.0000, 1.0000 | 1.0000, 1.0000 | 0.9998, 1.0000 |

What the 0.5 % twins show. The warning is right for what it describes, a filter or a window that has to separate the lines by its own response: a filter per order and a 32-revolution window both let the neighbour in (correlation 0.69 to 0.90). It is not a statement about every possible estimator. The joint Vold-Kalman solution separates lines 0.8 Hz apart with a 2 Hz pass band (correlation 0.998, envelope error up to 0.8 dB, so not the 0.2 dB of the ROADMAP) as long as the envelopes vary slowly compared with the beat, here at 0.4 and 0.25 Hz. When the envelopes vary as fast as the beat (modulation at 0.8 and 1.2 Hz, `t8(1.005, am_freq=(0.8, 1.2))`) no bandwidth works: the best of B 0.4, 1 and 3 Hz gives a lowest correlation of 0.972, below 0.99, and `resolution_check(..., envelope_rate=1.2)` reports the bias-variance wall (ROADMAP section 6, item 7). The 4 % twins with the same modulation, B 3 Hz, reach 1.0000.

### T10: speed outside the search range

`rpm_from_ridge` never returned a clearly out-of-range frame unflagged with the T10 signal. Frames far outside the range are mostly flagged by the guard bands (`beyond`, 98 % and 99 %) and not by the path: the path cannot leave the range and in these frames sits anywhere in it (it locks to a ratio of the true speed, up to 2115 rpm in a 1200 to 2800 rpm range while the true speed is below 1200). Outside the guard bands (true speed more than a factor 2 from the range) flags can be missed: with the range 3600 to 6000 rpm and the T10 speed (600 to 3400 rpm) 8 of the 233 frames, all more than a factor 2 below the range, are neither flagged nor within 3 % (`rpm_from_ridge` documents this). Not part of the tests.

## Speed reference errors (T13)

Both extractors are run on the T13 signal with the exact speed and with the three perturbed references of `t13_references()` (bias +0.5 %, jitter 0.5 % RMS with a 20 Hz band, step +1 % at 6 s), using Vold-Kalman B 2 Hz and tracked harmonics with W 8. The columns are, for the worst of the four orders (1, 2, 3, 4.5; up to 375 Hz at the end of the run), the median and 95th percentile of the envelope error in dB over the record with 1 s removed at both ends, the lowest waveform correlation and the largest NRMSE. The table is printed by `pytest tests/test_validation_v02.py -s -k t13_degradation`.

| Speed reference | Vold-Kalman: median / 95 % dB | corr | NRMSE | Tracked harmonics: median / 95 % dB | corr | NRMSE |
|---|---|---|---|---|---|---|
| exact | 0.008 / 0.019 | 1.0000 | 0.003 | 0.008 / 0.028 | 1.0000 | 0.004 |
| bias 0.5 % | 0.413 / 1.639 | 0.9990 | 0.066 | 0.350 / 0.486 | 0.9999 | 0.049 |
| jitter 0.5 % rms | 0.092 / 0.602 | 0.9922 | 0.125 | 0.084 / 0.302 | 0.9960 | 0.089 |
| step 1 % | 1.188 / 12.757 | 0.9548 | 0.305 | 0.426 / 1.453 | 0.9976 | 0.078 |

Bias sweep (a constant relative error of the speed), worst order, median envelope error in dB and lowest correlation:

| Bias | 0 | 0.1 % | 0.25 % | 0.5 % | 1 % | 2 % |
|---|---|---|---|---|---|---|
| Vold-Kalman, B 2 | 0.008, 1.0000 | 0.007, 1.0000 | 0.026, 1.0000 | 0.413, 0.9990 | 4.398, 0.9633 | 21.195, 0.7485 |
| Tracked harmonics, W 8 | 0.008, 1.0000 | 0.012, 1.0000 | 0.084, 1.0000 | 0.350, 0.9999 | 1.419, 0.9997 | 5.743, 0.9989 |

By order (median dB, 0.5 % bias): Vold-Kalman 0.005, 0.017, 0.065, 0.413; tracked harmonics 0.017, 0.069, 0.156, 0.350 for orders 1, 2, 3, 4.5. The error grows with the order.

Acceptable speed error. A relative speed error d moves an order of frequency f off its carrier by d f. The extractor treats that offset as an amplitude change at that frequency, so the loss is set by how the filter or window attenuates a tone at d f, and it is larger for high orders and at high speed.

* The waveform correlation is the less sensitive measure: it stays above 0.99 for biases up to 0.5 % for both extractors and (tracked harmonics) above 0.998 even at 2 %, because the extracted waveform still follows the signal in phase while its amplitude is attenuated. What is lost is level: the envelope error and NRMSE grow, and an order removed with a biased speed leaves a residual of that size (NRMSE 0.07 at 0.5 % bias, so about -23 dB left, against the -20 dB of T12).
* For an envelope error of at most 0.2 dB, the speed error must keep **d f_top <= 0.5 B** for the Vold-Kalman filter, with f_top the highest frequency of the highest order (the filter attenuates a tone at the offset d f by 3.6 (d f / B)^4 dB, which is 0.2 dB at d f = 0.49 B at the top speed; the median over the T13 run is lower and reaches 0.2 dB at 0.7 B), and **d k_max W <= 0.12** for tracked harmonics (the median error is about 10 (d k W)^2 dB at order k for a Hann window of W revolutions; it reaches 0.2 dB at 0.14).
* For the orders of T13 (k_max 4.5, f_top 375 Hz) that is a speed error of at most 0.27 % with B 2 Hz (0.13 % with B 1, 0.53 % with B 4, 1.07 % with B 8) and 0.33 % with W 8 (0.67 % with W 4, 0.17 % with W 16, 0.08 % with W 32). A wider band or a shorter window tolerates more, at the price of less noise rejection and resolution. The tests check these two rules (`test_t13_guidance_*`): the error is within 0.2 dB at the rule and above it at four times the rule.
* A jitter of 0.5 % RMS in a 20 Hz band costs less than 0.1 dB at the median and 0.6 dB at the 95th percentile with these settings. A step is a bias that starts mid-record: after the step the extraction behaves as with a bias of the size of the step, so the numbers of the 1 % column of the sweep apply to the second half (the table, over the whole record, is lower).

## Not yet

The black-box reference comparison (ROADMAP section 5.5, steps 2 and 3: running an external implementation on these signals, storing its output in `reference/`, and the `compare.py` report) has not been run for any release so far (0.1 to 0.3). Only the comparison with analytic truth exists, and it is what continuous integration runs.
