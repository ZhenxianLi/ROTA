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

## Black-box reference (MATLAB)

ROADMAP section 5.5 describes a second reference besides the analytic truth: the same signals analysed by an external implementation, used as a black box. This section describes the set-up. The comparison runs in CI; its output stays in the workflow artifact and has not been published with any release, so the statement under "Not yet" above still holds for the releases so far.

**Rules (ROADMAP section 4).** `validation/matlab/run_reference.m` only calls documented public functions of the Signal Processing Toolbox, with every parameter given explicitly, on the signals of `make_signals.py`, and saves what they return. No toolbox code is read or reused, no other data is processed and nothing is evaluated in that script. The comparison is done by `compare.py`. When ROTA and MATLAB disagree, the analytic truth decides (ROADMAP section 5.1): every row of the report also says how far MATLAB is from the truth.

### Running it in CI

Actions, then `matlab-reference`, then Run workflow. The workflow also runs on pushes that change `validation/matlab/**`, `validation/export_mat.py`, `validation/compare.py` or the workflow itself. One job per case, and for T2, T3 and T14 one per part of the calls (`main`, `vkwave`: `orderwaveform`, `vktrack`: `ordertrack` in its Vold-Kalman form; environment variable `ROTA_REF_PART`), all in parallel, installs ROTA, runs `python validation/export_mat.py <case>`, sets up MATLAB with the Signal Processing Toolbox (`matlab-actions/setup-matlab`, which licenses MATLAB itself for a public repository on a GitHub-hosted runner), runs `run('validation/matlab/run_reference.m')` and uploads `validation/reference/` as the artifact `reference-<case>-<part>`. The `report` job then exports all cases, collects the `<case>_ref.mat` and `<case>_ref_<part>.mat` files (`compare.py` merges the parts of a case), concatenates the logs and runs `python validation/compare.py --out validation/reference/report.md`. The report is appended to the job summary, and `validation/reference/` (report, `<case>_ref.mat`, `versions.txt` with the MATLAB release and toolbox versions, and `log.txt` with every call, its duration, warnings and errors) is uploaded as the artifact `matlab-reference`. Every call is wrapped in `try`/`catch`: a failing call is logged and the others continue; the report then shows `MATLAB call failed` for it.

**The MATLAB output is not committed.** It stays in the CI artifact (and `validation/reference/` and `validation/signals/` are git-ignored) until the licence terms of the institution concerned have been confirmed (ROADMAP section 5.5).

### Without MATLAB

```
python validation/export_mat.py
python validation/compare.py --out report.md
```

writes `validation/signals/<case>.mat` and a report in which ROTA is compared with the truth and every MATLAB cell reads `MATLAB output missing`. The same two commands, with `run('validation/matlab/run_reference.m')` in MATLAB between them (started from anywhere; the script finds the repository from its own location), reproduce the CI run on a workstation. `compare.py --cases T1 T5` restricts the cases, `--strict` returns a non-zero exit status if a row fails.

### Cases and parameters

`export_mat.py` writes T1, T2, T3, T5 (60 pulses per revolution), T6, T14 and the probe VKBW as MATLAB v5 files with `x`, `fs`, `t`, `rpm` (the exact speed, column), `tacho` (T5), `orders`, `order_amplitude` (peak truth), `order_stems` (truth waveforms) and the struct `par` of analysis parameters that both sides use; `compare.py` reads the same table, so they cannot drift apart. VKBW is built in `export_mat.py` with `rota.synth.synth_runup`: 1500 rpm constant, order 1 (carrier 25 Hz, 1 Pa peak) and unit tones at 0.5 to 5 Hz above and below the carrier, 200 Hz sampling (a low `fs / bandwidth`, because MATLAB's iterative Vold-Kalman solver did not converge on the first version of the probe at 2 kHz), 30 s, no noise.

| Case | Order resolution | Frequency map | Vold-Kalman bandwidth (ROTA; MATLAB `Bandwidth`) | Tacholess speed (MATLAB order; ROTA comb = all orders) |
|---|---|---|---|---|
| T1 | 0.25 | 8 Hz | 2 Hz; 4 Hz | order 2, 4 Hz; ROTA range 800 to 4500 rpm |
| T2 | 0.25 | none | 2 Hz; 4 Hz | order 2, 4 Hz; 800 to 5500 rpm |
| T3 | 0.25 | none | 2 Hz; 4 Hz | order 2, 6 Hz; 800 to 4500 rpm |
| T5 | none | none | none | tachometer: 60 pulses per revolution, state levels from the signal |
| T6 | 0.25 | none | 16 Hz; 32 Hz | none (1 s record) |
| T14 | 0.125 | none | 2 Hz; 4 Hz | order 1, 3 Hz; 600 to 3500 rpm |
| VKBW | none | none | 2 Hz; 4 Hz and 2 Hz (probe only) | none |

Order maps use a Hann window and 75 % overlap, the frequency map a flat-top window (no scalloping) and 75 % overlap; amplitudes are RMS. The frequency map is run for T1 only, the one case whose single line is further from its neighbours than the flat-top main lobe. The Vold-Kalman filter is joint over several orders (decoupled) with ROTA's second-difference structure (MATLAB `FilterOrder` 1, see "Conventions").

### Conventions

The two sides define some arguments differently. The parameters are mapped so that both estimate the same quantity; the results are never rescaled to agree.

* **Vold-Kalman bandwidth.** ROTA's `bandwidth` (`par.vk_bandwidth`) is the offset from the carrier at which the single-order filter attenuates a tone by 3 dB, so its pass band is twice as wide. MATLAB's `Bandwidth` is the full width of the pass band between the two -3 dB points, so it is given `par.vk_bandwidth_matlab` = 2 x `par.vk_bandwidth`. The probe VKBW measures this (table 10): each filter's -3 dB offset is read off its gains at the known tone offsets, MATLAB with `Bandwidth` = 2 x and = 1 x the ROTA value. Independently, the envelopes of the Vold-Kalman form of `ordertrack` with `Bandwidth` = 2 x the ROTA value follow ROTA's closely (table 9), also on the fast run-up of T6, where the envelope lag depends on the bandwidth and ROTA's own envelope error changes clearly between 8, 16 and 32 Hz.
* **Vold-Kalman structure.** ROTA's second-difference structure has the response `1 / (1 + c d^4)` at an offset `d` from the carrier. On the probe, MATLAB's `FilterOrder` 1 follows the same `d^4` law, with its -3 dB offset at about half of `Bandwidth`, and `FilterOrder` 2 a `d^6` law, so MATLAB is given `FilterOrder` 1 (`par.vk_filter_order`); the probe keeps running both and reports the fitted exponent.
* **Amplitude.** Map readings are RMS on both sides (`Amplitude` `'rms'`; a sine of peak amplitude A reads A / sqrt(2)), Vold-Kalman envelopes peak (`Amplitude` `'peak'`). The truth is peak; RMS rows divide it by sqrt(2). Waveforms are compared by correlation and least-squares gain, which need no amplitude convention.
* **Time axes.** MATLAB's `time`, `t` and `tout` outputs are used as returned (s from the first sample). A MATLAB map reading is compared with the truth at the speed MATLAB returns for that frame, and with ROTA by interpolation in time onto the ROTA frame centres; table 6 checks that the exact speed at MATLAB's `time` matches its `rpm` output, i.e. that `time` holds the frame centres.
* **Resolution.** ROTA's `order_resolution` and `resolution` are the bin spacing (one over the window length). MATLAB's `res` behaves as a resolution bandwidth, the equivalent noise bandwidth (ENBW) of the window times the bin spacing: the bins MATLAB returns are `res` / ENBW apart. MATLAB is therefore given `par.order_resolution_matlab` / `par.freq_resolution_matlab` = ENBW x the ROTA resolution (Hann 1.5, flat top 3.77 bins, computed in `export_mat.py`), so that both sides use the same window length; table 6 reports the spacing each side obtains and MATLAB's `res` input and output. A map is read at the bin nearest the order (MATLAB) or by band power over the main lobe (ROTA `order_slice`).
* **Tachometer.** ROTA's `tacho_to_rpm` fills dropped pulses; MATLAB's `tachorpm` returns the pulses it found (`tp`) and, judging from the intervals of about twice their neighbours in `tp`, computes the speed across the gaps as they are. T5 drops 0.5 % of its pulses, which pulls the MATLAB speed down; the report counts the gaps.

### Solver time and convergence of MATLAB's Vold-Kalman filter

`FilterOrder` 1 is used because its response has the structure of ROTA's filter (see "Conventions"). The Vold-Kalman calls solve their filter iteratively; any solver warning is written to the log and shown in the notes of the affected rows, and rows where ROTA meets the truth and the reference does not read `pass (truth decides)`. These calls can take many minutes on the long multi-order cases, so the workflow runs them for T2, T3 and T14 in jobs of their own (parts `vkwave` and `vktrack`). No solver option is documented, so none is set.

### Calls made (MATLAB)

| Function | Arguments |
|---|---|
| `tachorpm` | `(tacho, fs, 'PulsesPerRev', 60, 'StateLevels', [low high], 'OutputFs', 1000, 'FitType', 'smooth', 'FitPoints', 60)`; and with `'FitType', 'linear'` (no `FitPoints`) |
| `rpmordermap` | `(x, fs, rpm, res, 'Window', 'hann', 'OverlapPercent', 75, 'Amplitude', 'rms', 'Scale', 'linear')`, `res` = 1.5 x the ROTA order resolution (orders) |
| `rpmfreqmap` | `(x, fs, rpm, res, 'Window', 'flattopwin', 'OverlapPercent', 75, 'Amplitude', 'rms', 'Scale', 'linear')`, `res` = 3.77 x the ROTA resolution (Hz) |
| `orderspectrum` | `(map, order, 'Amplitude', 'rms')`, with the map and order axis of `rpmordermap` |
| `ordertrack` (order map) | `(map, order, rpm, time, orders)`, from `rpmordermap`; the map form takes no name-value arguments (amplitude type and scale are those of the map) |
| `ordertrack` (Vold-Kalman) | `(x, fs, [rpm rpm], orders, ones(1, K), 'Amplitude', 'peak', 'Scale', 'linear', 'Bandwidth', bw, 'Decouple', dec)`, `bw` = `par.vk_bandwidth_matlab` (2 x the ROTA bandwidth). `ordertrack` uses the Vold-Kalman filter only when `rpmrefidx` is given (otherwise it refuses `Bandwidth` and `Decouple`), and accepts `rpmrefidx` only with a speed matrix, so the one speed is passed as two identical columns and every order refers to column 1 |
| `orderwaveform` | `(x, fs, rpm, orders, ones(1, K), 'FilterOrder', 1, 'Bandwidth', bw, 'Decouple', dec)`, `bw` as above |
| `orderwaveform` (probe VKBW) | `(x, fs, rpm, 1, 1, 'FilterOrder', f, 'Bandwidth', b, 'Decouple', false)` with `b` = 4 Hz (`par.vk_bandwidth_matlab`) and `b` = 2 Hz (`par.vk_bandwidth`), each with `f` = 1 (the comparison) and `f` = 2 (the other structure) |
| `rpmtrack` | `(x, fs, order, [t0 f0], 'Method', 'stft', 'FrequencyResolution', res, 'PowerPenalty', Inf, 'FrequencyPenalty', 0, 'StartTime', t(1), 'EndTime', t(end))`, with `[t0 f0]` a point on the ridge read off the exact speed at mid-record |

The script keeps a short list of fallbacks for each call (the same call without the optional arguments that a given release might not accept, or with the speed repeated in two columns). The variant that ran is saved with the output, listed under "MATLAB calls recorded" in the report, and marked in the notes of the rows whenever it was not the first.

### What the report compares

One table per comparison, with the columns ROTA to truth, MATLAB to truth, ROTA to MATLAB, the tolerance (ROADMAP section 5.4) and the result. A row fails if ROTA is outside the truth tolerance, or if ROTA and MATLAB differ beyond their tolerance while MATLAB is within the truth tolerance (or no truth applies). When ROTA is within the truth tolerance and MATLAB is not, the truth decides (ROADMAP section 5.1) and the row reads `pass (truth decides)`. Physical quantities are compared on common grids, never map cells:

1. Tachometer speed, `tacho_to_rpm` against `tachorpm` (T5): median relative error, tolerance 0.1 %.
2. Tacholess speed, `rpm_from_ridge` against `rpmtrack`: share of frames within 3 % (90 % against the truth; the two trackers within 5 percentage points), and the boundary-hit flag of ROTA.
3. Amplitude of an order along its ridge in an order map, `rpm_order_map` with `order_slice` against `rpmordermap` (largest dB difference over interior frames where the truth amplitude varies by at most 2 % over a frame; 0.2 dB against the truth, 0.5 dB against MATLAB).
4. The same against `ordertrack` applied to the MATLAB order map.
5. The same for the frequency map (T1).
6. Spacing of the order and frequency axes against the requested resolution (5 %), and the time axis of the MATLAB maps against the exact speed (informational).
7. Average order spectrum, `order_spectrum` against `orderspectrum`: peak order (within half a bin) and, where the truth amplitude is constant, peak amplitude.
8. Vold-Kalman waveforms, `order_waveform` against `orderwaveform`: correlation (0.99) and least-squares gain of the waveform on the true waveform (0.2 dB, 0.5 dB).
9. Vold-Kalman envelope amplitude, `order_waveform` against `ordertrack` in its Vold-Kalman form (0.2 dB, 0.5 dB).
10. Vold-Kalman -3 dB bandwidth on the probe VKBW, `order_waveform` against `orderwaveform`: the offset at which each filter attenuates a tone by 3 dB, above and below the carrier (within 5 % of the set value and of MATLAB, ROADMAP section 5.4), the gain at the carrier, and informationally the offset MATLAB reaches with `Bandwidth` twice and once ROTA's number and the exponent of its response, with `FilterOrder` 1 and 2.

Not covered: the cases T4, T8, T10 and T13. `rpmtrack` has no argument for a frequency range, so the tacholess comparison is anchored by the initial ridge point and the frequency resolution instead. Where the two sides define an argument differently the parameters are mapped (see "Conventions"); a remaining systematic offset in a row is a finding, not noise, and the report is the place to read it.
