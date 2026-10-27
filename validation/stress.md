# Stress tests on realistic synthetic scenarios

## Purpose

The validation cases (`README.md` in this directory) check each method on a signal built for it: one effect at a time, against analytic truth. The stress tests (`tests/test_stress.py`) run the same methods on the complete synthetic recordings of `rota.synth.synth_scenario`. Those recordings combine:

* a speed ripple locked to the shaft angle, a gear shift and a clutch-slip launch;
* half orders and angle-locked combustion bursts;
* a transfer path with modes and a reverberant cabin impulse response (RT60 0.1 s);
* low-frequency rumble, broadband engine and road noise, and transients;
* a delayed tachometer and, where stated, an imperfect recorder.

The tests answer two questions. How far do the results move from the ideal case? And which of those moves come from the method, and which from the physics of the signal?

All signals come from `rota.synth` with fixed seeds and the scenario defaults. Only the sampling rate and the duration are set:

| Scenario | Call | Content |
|---|---|---|
| ICE run-up | `synth_scenario("ice_runup", 32000, 10.0, seed=101)` | 1200 to 6000 rpm, one upshift at 9.2 s, orders 0.5 to 8; tachometer 60 pulses per revolution, 1.5 ms delay |
| EV run-up | `synth_scenario("ev_runup", 48000, 6.0, seed=102)` | 200 to 8600 rpm, motor orders 2 to 14, gear mesh 40 and 6.7, inverter, fixed tones; tachometer 48 pulses per revolution, 0.5 ms delay |
| Steady cruise | `synth_scenario("steady_cruise", 16000, 4.0, seed=103)` | 2390 rpm |
| ICE, two channels | `synth_scenario("ice_runup", 32000, 8.0, seed=104, n_channels=2)` | 0.3 ms inter-channel delay of the direct sound, independent tails and noise |

The sampling rates are the lowest at which the default tachometers have at least 2 samples per pulse at top speed.

**The orders under test are chosen, not hard-coded.** They are the three source orders with the highest in-band SNR: the median over frames of the order-map slice of the order's post-path stem divided by that of everything else in the recording. The tachometer resolution, the delay and the path modes are read from the scenario configuration, so the tests follow the scenario defaults. With the defaults measured here:

* ICE: orders 2, 4 and 6, at +20.6, -5.1 and -11.1 dB.
* EV: orders 40 (gear mesh), 12 and 6.7, at +7.9, +2.1 and -2.3 dB.

The low-frequency rumble and the broadband noise of the defaults bury every order except the ICE firing order and the EV gear mesh. This shapes most of the results below.

The bounds in the tests are regression guards set with a margin above the numbers below. They are not the tolerances of ROADMAP section 5.4, which apply to the validation cases. Some defaults were still being tuned while these tests were written, so the numbers belong to the defaults of the run that produced them. `python -m pytest tests/test_stress.py -q -s` prints all of them (about 85 s of CPU time; 90 to 100 s wall time on a shared 4-core machine).

Definitions used in the tables:

* "Stem": the order's source waveform passed through the scenario's path (`through_path(order_stems[:, k])`), the exact contribution of that order to the recording.
* "Quasi-static truth": `quasi_static_amplitude(k)`, the source amplitude times the steady-state gain |H(f)| of path and cabin. It ignores the memory of the path.
* "Floor": the recording minus all post-path order stems, i.e. what a perfect order removal would leave.
* Errors in dB are absolute values. Interior frames only: 0.5 s at the start and 0.3 s at the end are left out. "Away from shifts" leaves out ±0.6 s around each gear shift.

## 1. Speed

### Tachometer (`tacho_to_rpm`, default `smooth=1` revolution, delay compensated with `t_out = t + delay`)

| Quantity | ICE | EV | Cruise | Ideal-case reference | Note |
|---|---|---|---|---|---|
| Median relative error against `rpm_mean` | 0.015 % | 0.004 % | 0.014 % | tolerance 0.1 % (T5) | Within the ideal-case tolerance |
| Largest relative error against `rpm_mean` | 0.06 % | 1.5 % | 0.04 % | | EV: at the start of the run-up from 200 rpm, where one revolution lasts 0.3 s |
| Median relative error against the true instantaneous `rpm` | 0.22 % | 0.030 % | 0.22 % | | About the size of the ripple itself |
| Median ripple, abs(`rpm`/`rpm_mean` - 1) | 0.20 % | 0.028 % | 0.20 % | none (T5 has no ripple) | 0.3 % peak at order 2 (ICE), 0.04 % (EV) |
| `smooth=0.25`: median error against the instantaneous `rpm` | 0.063 % | 0.030 % | 0.052 % | | A quarter-revolution fit follows the firing ripple |

The firing-ripple limitation: a fit over one revolution averages a ripple at order 2 (period half a revolution) away almost completely. The default output is therefore the mean speed, accurate to better than 0.02 %. It differs from the instantaneous speed by about the ripple (0.2 % median). To follow the ripple, the fit window must be shorter than its period. `smooth=0.25` gets within 0.06 % of the instantaneous speed and carries more timing jitter. For order analysis the ripple barely matters: 0.3 % at order 2 moves the shaft angle by 0.003 / (2π · 2) = 2.4e-4 revolutions, a phase modulation of 0.009 rad at order 6.

### Tachometer delay (ICE, 1.5 ms; tracked harmonics, W 8; 95th percentile of the envelope difference, orders 2 / 4 / 6)

| Quantity | Measured | Note |
|---|---|---|
| Uncompensated against compensated (`delay=` of `rpm_to_phase`, passed through by `tracked_harmonics`) | 0.008 / 0.13 / 0.30 dB | Grows with the order; for the weak orders it is mostly noise reweighted |
| Compensated tacho speed against the true speed | 0.016 / 0.42 / 0.24 dB | |
| Order-2 waveform correlation with its stem, uncompensated / compensated | 0.9986 / 0.9986 | |

A constant delay shifts the angle by rpm/60 · delay. That shift varies only as fast as the speed does, so the demodulating extractors absorb it into the complex coefficient, and levels and waveforms barely change. Compensating is still the right default (the examples do it), because the error grows with order, slew rate and delay.

### Tacholess speed (`rpm_from_ridge`; defaults: window 0.4 s, hop 0.05 s, `whiten="auto"`)

Settings follow the recommendation of `rpm_from_ridge` for the comb by machine type. A four-stroke engine with N cylinders uses (N/2, N, 3N/2), i.e. orders (2, 4, 6) for the four-cylinder engine of `ice_runup` and `steady_cruise`. The electric drive uses its motor order family (2, 4, 8, 10, 12, 14), read from the configuration. The search range is the true speed range with a 30 % margin. A frame counts as flagged when it is in `at_bound` or its `confidence` is below 0.1.

What the tests assert:

* Engine, recommended comb: at least 90 % of the frames within 3 %, at most 5 % wrong and unflagged, and `quality["reliable"]` True.
* Electric drive: the result is flagged unreliable (`reliable` False, `RidgeQualityWarning`), with at most 5 % of the frames wrong and unflagged. This tests for no silent failure, not for accuracy.
* Counter-example: the non-recommended comb (1, 2, 3, 4, 6) on the engine must, when it fails, report itself unreliable.

| Quantity | ICE, comb (2, 4, 6) | Cruise, comb (2, 4, 6) | ICE, comb (1, 2, 3, 4, 6) (counter-example) | EV, motor family | Ideal-case reference |
|---|---|---|---|---|---|
| Frames within 3 % of `rpm_mean` | 99.5 % | 100 % | 47 % | 0 % | 100 % (T3, T10); tolerance 90 % |
| Median relative error | 0.40 % | 0.22 % | 95 % | 109 % | 0.22 % (T3) |
| Frames flagged `at_bound` | 0 % | 0 % | 49 % | 100 % | |
| Frames wrong by more than 3 % and not flagged | 0 % | 0 % | 45 % | 0 % | 0 % (T10, inside the range) |
| Median `confidence` | 0.50 | 0.24 | 0.18 | 0 | |
| `quality["reliable"]` | True | True | False | False | |
| Warnings | none | none | RidgeQualityWarning, RuntimeWarning | RidgeQualityWarning, RuntimeWarning | |

Reading the table:

* **Engine.** With the recommended comb, tacholess estimation works on the realistic engine scenario: 99.5 % of the run-up frames and every cruise frame are within 3 %, and nothing is flagged. The two-channel recording gives 100 % as well (section 4).
* **Counter-example.** The comb (1, 2, 3, 4, 6) adds the weak odd orders. At twice the speed its teeth fall on orders 2, 4, 6, 8 and 12, which are present or buried in noise, so the track locks there on about half of the frames (median error 95 %). The estimator reports this (`reliable` False, a warning, half of the frames in `at_bound`), but 45 % of the frames are wrong without a frame-level flag. The comb has to match the machine.
* **Electric drive.** The cabin lines of the motor are too weak against the rumble below 100 Hz and the road noise, over a run-up spanning a factor of 40 in speed, and no frame is tracked. The failure is not silent: every frame is in `at_bound`, the confidence is 0 and `reliable` is False with a `RidgeQualityWarning`. On such a recording the tacholess estimate is not usable unattended, but it says so.

## 2. Order levels (`rpm_order_map` at 0.125 orders per bin, `amplitude="peak"`, `order_slice` with the default band, true speed)

| Scenario, order (in-band SNR) | Mix vs stem: median / p95 | Mean power vs (stem + rest) | Frames with SNR > 10 dB: share, p95 vs stem | Stem vs quasi-static: median / p95 | Near a path mode below 500 Hz (±15 Hz): largest |
|---|---|---|---|---|---|
| ICE 2 (+20.6 dB) | 0.23 / 1.29 dB | +0.07 dB | 95 %, 1.11 dB | 0.05 / 0.18 dB | 0.59 dB (45, 80, 175 Hz; 120 frames) |
| ICE 4 (-5.1 dB) | 6.4 / 14.3 dB | -0.06 dB | 11 %, 1.33 dB | 0.06 / 0.34 dB | 0.28 dB |
| ICE 6 (-11.1 dB) | 10.9 / 20.3 dB | -0.09 dB | 1 % | 0.15 / 1.48 dB | 0.38 dB |
| EV 40 (+7.9 dB) | 0.98 / 6.0 dB | -0.02 dB | 38 %, 1.39 dB | 2.3 / 9.4 dB | none in range |
| EV 12 (+2.1 dB) | 2.3 / 6.9 dB | +0.16 dB | 4 % | 0.92 / 4.3 dB | none |
| EV 6.7 (-2.3 dB) | 4.3 / 16.2 dB | -0.20 dB | 4 % | 0.43 / 3.3 dB | 1.8 dB (95 Hz) |

Ideal-case reference: 0.079 dB (T3) and 0.146 dB (T6) worst case on the order slice, 0.2 dB tolerance, and 1 dB near a resonance (T2).

Reading the table:

* **The map is exact; the band is not empty.** On average over the frames, the power read in the band equals the power of the order plus that of everything else in it, to within 0.2 dB for every order. Frame by frame it does not, because the cross term between order and noise is as large as either when the SNR is near 0 dB. Where the order stands 10 dB above the rest of its band, the slice reads it within 1.1 to 1.4 dB at the 95th percentile (median well below 1 dB). The ideal cases have no such noise.
* **Weak orders read the noise.** ICE orders 4 and 6 and EV order 6.7 are below the rest of their band at most frames, so their slices read the noise floor: 4 to 11 dB too high at the median. This is a property of the recording, not of the map. A band-power slice cannot tell order energy from noise energy in the band.
* **The half orders need a fine map.** Order 2 against its stem: largest error 3.1 dB at 0.125 orders per bin and 5.0 dB at 0.25 (95th percentile 1.28 and 1.96 dB). At 0.25 the half orders 1.5 and 2.5, two bins away, leak into the band.
* **The quasi-static truth** is a good reference for the ICE orders (median at most 0.15 dB, and at most 0.6 dB near the 45, 80 and 175 Hz modes). It is not a good reference for the EV gear mesh at 1 to 6 kHz, which sweeps at hundreds of Hz per second through the fine structure of a reverberant response with a 0.1 s tail: 2.3 dB median and 9.4 dB at the 95th percentile. For a path with memory, the stem passed through the path is the truth to compare with.

## 3. Extraction and removal

Vold-Kalman `order_waveform` (joint, B 2 Hz for the ICE, 4 Hz for the EV) and `tracked_harmonics` (W 8) of the three tested orders. Waveforms are compared with the stems, with 1 s trimmed at both ends; "raw" is the correlation of the unprocessed recording with the stem. Envelopes are compared with the same extractor applied to the sum of the stems with the true speed: 1 s left out at both ends and ±0.4 s around the shift. "Tacho" is the speed of `tacho_to_rpm` with the delay compensated.

| Scenario, extractor | Order: correlation (raw) / envelope median / p95, true speed | Same with tacho speed | Ideal-case reference |
|---|---|---|---|
| ICE, VK | 2: 0.9996 (0.910) / 0.12 / 1.05 dB; 4: 0.914 / 1.9 / 6.1 dB; 6: 0.613 / 4.5 / 14.5 dB | 2: 0.9996 / 0.12 / 1.04 dB; 4 and 6 within 0.05 dB | correlation 0.999998, envelope 0.076 dB (T3) |
| ICE, TH | 2: 0.9986 / 0.15 / 1.11 dB; 4: 0.821 / 3.1 / 9.1 dB; 6: 0.475 / 6.3 / 14.4 dB | 2: 0.9986 / 0.15 / 1.10 dB | 0.055 dB (T3) |
| EV, VK | 40: 0.503 (0.006) / 3.4 / 17.8 dB; 12: 0.885 / 1.7 / 7.8 dB; 6.7: 0.795 / 3.1 / 14.4 dB | the same to within 0.07 dB | |
| EV, TH | 40: 0.727 / 2.6 / 39.8 dB; 12: 0.903 / 2.1 / 6.8 dB; 6.7: 0.787 / 3.1 / 14.4 dB | the same to within 0.3 dB | |

The firing order of the ICE is extracted almost as in the ideal case: correlation 0.9996 against 0.910 for the raw recording, envelope median 0.12 dB. Using the tachometer speed instead of the exact speed changes nothing that matters (at most 0.07 dB on any median). The noise-dominated orders are bounded by their SNR, as with the map, and the extractors still raise their correlation with the stem from 0.01 to 0.07 (raw) to 0.47 to 0.92. The EV gear mesh (order 40) has an SNR of +7.9 dB but a correlation of only 0.50 (VK) or 0.73 (TH). As the removal controls below show, a large part of its stem is the reverberant tail of a fast sweep, which lies off the order line.

### Removal (`remove_orders`, tracked harmonics W 8, the tested orders; `order_residual_energy` relative to the input; frames above 1440 rpm ICE, 1770 rpm EV)

The other orders lie at least half an order away, outside the ±0.25-order band of the metric and the main lobe of the 8-revolution window, so removing only the tested orders measures the same thing as removing all of them.

| Order | Residual of the recording | Floor | Control: residual of the stems alone, after the path (source level) | Ideal-case reference |
|---|---|---|---|---|
| ICE 2 | -20.8 dB | -18.7 dB | -36.4 dB (-59.7 dB) | -46 to -56 dB (T12); tolerance -20 dB |
| ICE 4 | -4.0 dB | -1.4 dB | -29.5 dB | |
| ICE 6 | -2.4 dB | -0.3 dB | -18.5 dB | |
| EV 40 | -3.5 dB | -11.6 dB | -3.6 dB (-56.6 dB) | |
| EV 12 | -6.1 dB | -4.7 dB | -12.8 dB | |
| EV 6.7 | -5.5 dB | -2.6 dB | -26.5 dB | |

The controls separate the extractor from the signal. The extractor removes a clean source-level order to -57 to -60 dB (the first order of each kind; an earlier run with all three orders gave -57 to -74 dB), so it is not the limit. Applied to the post-path stems alone, it leaves -18 to -36 dB of the ICE orders and only -3.6 dB of the EV gear mesh. The rest is the path's memory: an order sweeping through the 0.1 s reverberant tail leaves energy near, but not on, its line, inside the ±0.25-order analysis band and outside the extractor's window. In the recording, every residual lies within 3 dB of the power sum of floor and that path residual (asserted). The ICE residuals sit at or below the floor, because the extractor also takes the noise inside its band. The EV gear mesh stays 8 dB above its floor because of the path.

## 4. Measurement impairments

All against the ideal recording of the same scenario: order-map slices at 0.125 orders, true speed, interior frames, the tested orders.

| Impairment | Quantity | Measured | Note |
|---|---|---|---|
| Clipping, 0.1 % of the samples (ICE; clip at 85 % of the peak) | p95 slice error, orders 2 / 4 / 6 | 0.010 / 0.060 / 0.10 dB | Only the largest peaks clip |
| Clipping, 1 % of the samples (clip at 71 % of the peak) | same | 0.14 / 0.68 / 0.78 dB | Bounded distortion, growing with the order (harmonics of the clipped firing order); medians 0.000 dB |
| 16 bits, 20 dB headroom (full scale = 10 × peak) | SNR of quantisation; worst p95 slice error | 70.0 dB; 0.002 dB | |
| 16 bits, 40 dB headroom | same | 50.0 dB; 0.022 dB | Even a badly set gain is harmless for the orders |
| Clock drift 50 ppm, audio fast (EV) | extra samples in 6 s | 14 | `n_out = floor((n - 1)(1 + eps)) + 1` |
| Clock drift, uncorrected (tacho clock used as is) | p95 slice error, orders 6.7 / 12 / 40 | 0.086 / 0.063 / 0.11 dB | Maps hardly see it: a relative shift of about 1e-4 is 0.03 bins at order 40 |
| Clock drift, uncorrected | VK B 1 Hz, order 40, median envelope error | 1.07 dB | Offset eps · k (rpm + t · d rpm/dt) / 60: eps · f = 0.29 Hz at the top plus the accumulated time offset eps · t times the sweep rate, comparable to B |
| Clock drift, corrected with `t_true` | p95 slice error, all orders; VK order 40 median | < 0.001 dB; 0.0002 dB | Speed `interp(t_true, t, rpm) · gradient(t_true) · fs` (revolutions per nominal second of the recorder) |
| Two channels, 0.3 ms inter-channel delay (ICE) | order 2 against each channel's own stem: median / p95 | 0.17 / 1.55 dB (ch 0), 0.17 / 1.29 dB (ch 1) | Channels are analysed independently: same quality as mono |
| Two channels | `rpm_from_ridge` (comb 2, 4, 6) within 3 %: mean of channels / channel 0 | 100 % / 100 % | The comb stays below 600 Hz, far below 1 / (2 · 0.3 ms) = 1.7 kHz, where the mean of the channels cancels |

A clock drift between tachometer and audio becomes a frequency offset of eps · f. In a run-up it also adds the accumulated time offset eps · t times the sweep rate of the order. The order maps do not notice either. A narrow Vold-Kalman filter does (1 dB at order 40 with B 1 Hz). If the drift is known (here from `t_true`), expressing the speed on the recorder's time base removes the error completely.

## Conclusions and recommended settings for realistic signals

1. **Tachometer speed.** With the default one-revolution smoothing and the delay compensated (`tacho_to_rpm(..., t_out=t + delay)`, or `delay=` in `rpm_order_map`, `tracked_harmonics` and `rpm_to_phase`), the speed is accurate to better than 0.02 % of the mean speed (median). It does not follow the angle-periodic speed ripple. Use `smooth` of about a quarter of the ripple period when the ripple itself is wanted. Order levels and extracted waveforms are the same as with the exact speed.
2. **Order maps.** For four-stroke engines, use `order_resolution=0.125` or finer, because the half orders leak into a 0.25-order band. Read a slice as order level only where the order stands clear of its band: the slice of a noise-dominated order is the noise level. The in-band SNR against the rest of the recording is a useful quality flag. Above 10 dB the slices are within about 1.4 dB at the 95th percentile, and the average power always adds up (within 0.2 dB).
3. **Truth for levels after a real path.** Compare with the stem passed through the path, not with the steady-state gain. For fast-sweeping orders of several kHz in a reverberant cabin, the quasi-static amplitude is off by several dB.
4. **Extractors.** With a tachometer speed they extract an order that stands clear of the noise almost as in the ideal case (ICE firing order: correlation 0.9996, envelope median 0.12 dB). Their output for a buried order is bounded by its SNR. For orders that sweep fast through a reverberant path, neither the correlation with the stem nor `remove_orders` reaches the noise floor, because the reverberant tail is not part of the order. The control above shows that this is the signal, not the extractor.
5. **Recorder.** 16 bits with 20 to 40 dB of headroom and up to about 0.1 % clipped samples are harmless for order levels. Around 1 % clipped samples costs up to 0.8 dB at the 95th percentile at order 6. Before using a narrow Vold-Kalman bandwidth at high orders, correct any clock drift between tachometer and audio. For an envelope error of at most 0.2 dB, the offset eps · k (rpm + t · d rpm/dt) / 60 must stay below about 0.5 B (the T13 rule d · f ≤ 0.5 B of the validation README).
6. **Tacholess speed.** Choose the comb by machine type, (N/2, N, 3N/2) for a four-stroke N-cylinder engine. With it, `rpm_from_ridge` tracks the realistic engine scenario (99.5 % of the run-up frames within 3 %, 100 % at cruise, reliable, no warning). A comb that does not match the machine locks to twice the speed on half the frames and reports itself unreliable. On the electric drive, whose motor lines are buried in cabin noise, the estimate is not usable unattended, but it flags itself (`reliable` False, `RidgeQualityWarning`, no silently wrong frames). Check `quality["reliable"]` and the warnings before using a tacholess speed, and use a tachometer where one exists.

**Defects.** No crash and no wrong result attributable to a library defect was found. The shortfalls are explained above and asserted as bounded or self-reported behaviour, not marked as expected failures: the residual of `remove_orders` at the EV gear mesh (the path, shown by the controls) and `rpm_from_ridge` on the electric drive (untrackable lines, reported as unreliable).
