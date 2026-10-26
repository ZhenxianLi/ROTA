# ROTA Development Roadmap

**ROTA**: **R**otational **O**rder **T**racking & **A**nalysis

Latin *rota* means "wheel", from the same root as rotate, rotation and rotor.

ROTA is a lightweight, open-source, validated Python library for order analysis of noise and vibration signals from rotating machinery (engines, electric motors, gearboxes, etc.), with tools for order-based sound resynthesis. It depends only on NumPy and SciPy. Each function is implemented independently from the published literature and comes with a validation script.

Naming convention: the project is written ROTA in running text; the package and import name in code is lowercase `rota` (`pip install rota`, `import rota`).

Licence: GPL-3.0-or-later (decided 2026-10-25).

**Revision (2026-10-25)**: on top of the first draft, lessons from the author's earlier research projects were incorporated (section 6). ROTA reuses no earlier code; all functions are implemented from scratch from the published literature. Main changes:

- Three modules added: `rota.resynth` (resynthesis), `rota.metrics` (validation metrics) and `rota.tonal` (psychoacoustic bridge);
- `rota.vkf` extended to `rota.extract`, which adds a phase-locked short-time harmonic extractor alongside Vold-Kalman;
- `rota.speed` gains tacholess comb–Viterbi rpm estimation, rpm → phase, vehicle speed → rpm and control-signal clean-up;
- Validation cases extended from T1–T7 to T1–T14, and the version plan changed to v0.1–v0.4 + v1.0;
- The independent-implementation rules extended from MATLAB to all commercial software, proprietary file formats, third-party code of unclear licence and restricted data.

---

## 1. Design principles

1. **Independent implementation**: algorithms follow the published literature only and do not consult the source code or internal data structures of any commercial software (see section 4).
2. **Verifiable**: every function is tested on synthetic signals with known analytic ground truth. Where commercial software is available for comparison, a separate black-box comparison is made and the results are published.
3. **Transparent**: the docstring states the literature relied on, the amplitude definition (RMS or peak) and the unit conventions. Estimator functions (rpm, order amplitude) return a confidence and report whether the search boundary was reached.
4. **Python style**: functions are named in snake_case, inputs and outputs are NumPy arrays, and computation is separated from plotting.
5. **Interoperable**: results can be passed directly to psychoacoustic tools such as SQAT, MoSQITo and open-source ECMA-418-2 implementations.
6. **Multichannel first**: every operator acts on each channel independently and linearly; the input shape is `(n,)` or `(n, ch)`, and the amplitude and phase relations between channels are not changed. Multichannel recordings such as Ambisonics and microphone arrays can be processed directly, and the spatial information is preserved after orders are extracted or removed.
7. **Reproducible**: every random process accepts `rng` (a `numpy.random.Generator`) or an integer seed; resynthesis output can come with a JSON sidecar file (parameters, seed, version, input digest).
8. **No hidden constants**: gains, calibration factors and reference sound pressures are always passed explicitly; no default multiplier is buried inside a function.

---

## 2. Feature list

Modules: `rota.speed`, `rota.angle`, `rota.maps`, `rota.slices`, `rota.extract`, `rota.resynth`, `rota.synth`, `rota.metrics`, `rota.tonal`, `rota.plot`.

### 2.1 Speed reference (`rota.speed`)

| Function | Purpose | Literature | Version |
|---|---|---|---|
| `tacho_to_rpm` | Tachometer pulses → rpm curve. Supports several pulses per revolution, sub-sample edge interpolation, handling of dropped pulses and jitter, and smoothing | Brandt 2011 | v0.1 |
| `rpm_to_phase` | Rpm curve → accumulated shaft angle (revolutions). Trapezoidal integration; optional tachometer delay compensation; minimum-rpm floor (avoids the f→0 singularity) | — | v0.1 |
| `regularize_control` | Clean-up of control signals such as rpm and vehicle speed: spike removal (moving median + tolerance), resampling to a uniform grid (e.g. 100 Hz), gaps set to NaN, alignment to the audio time axis; supports slow channels at an integer ratio to the acoustic channels (e.g. 5:1) | — | v0.1 |
| `speed_to_rpm` | Vehicle speed → rpm. Two models: lumped slope (least-squares fit through the origin, residual reported); physical chain (wheel circumference, gear ratio, final drive ratio, limited to idle and redline). The proportionality factor can be calibrated automatically from steady segments | — | v0.1 |
| `steady_segments` | Extracts steady segments from accelerating or varying-speed recordings by target vehicle speed or rpm: the centre of the longest continuous segment within tolerance is taken, with a minimum-duration constraint; used for steady-state spectra and listening excerpts | — | v0.1 |
| `rpm_from_ridge` | Estimates rpm from the signal when there is no tachometer: order combs (generated from priors such as number of cylinders, pole pairs, tooth count) are scored on a whitened spectrogram, then a path is selected by Viterbi dynamic programming with a continuity penalty; optional "moving-condition" mode (removes fixed-frequency lines). Returns rpm, confidence and a search-boundary flag | Peeters et al. 2019 (review); comb and dynamic-programming design follows the fundamental-frequency tracking literature (Hermes 1988) | v0.2 |
| `virtual_rpm` | Generates a virtual rpm curve `clip(k·v, idle, redline)` from vehicle speed, for decoupling sound from vehicle dynamics in listening or driving experiments | — | v0.3 |
| `align_signals` | Time alignment of two unsynchronised acquisition systems: onset detection of transients such as hand claps and coarse alignment by generalised cross-correlation (GCC-PHAT), sub-sample refinement on the low-frequency waveform; clock drift estimated from start and end markers; reports the uncertainty | Knapp & Carter 1976 | v0.4 |

### 2.2 Angle domain (`rota.angle`)

| Function | Purpose | Literature | Version |
|---|---|---|---|
| `angular_resample` | Uniform time sampling → uniform shaft-angle sampling (the core of computed order tracking, COT). Anti-aliasing low-pass adapted to the minimum rpm; high-order interpolation (spline or polyphase); returns the angle-domain signal and the samples per revolution | Fyfe & de Munck 1997; Brandt 2011 | v0.1 |
| `angular_to_time` | Maps an angle-domain signal back to uniform time sampling (the inverse of the angle-domain transform, used for resynthesis) | As above | v0.3 |
| `stationarize` | Time warping for a known rpm proportionality (fixed gear ratio, linear run-up) that makes the order frequencies constant, for long-window spectral metrics; provides the inverse transform | — | v0.3 |

### 2.3 Maps (`rota.maps`)

| Function | Purpose | Literature | Version |
|---|---|---|---|
| `rpm_frequency_map` | Frequency–rpm map (Campbell diagram) | Brandt 2011 | v0.1 |
| `rpm_order_map` | Order–rpm map (based on `angular_resample`) | Fyfe & de Munck 1997; Brandt 2011 | v0.1 |
| Binning options | By time frame, or by Δrpm bin (with hysteresis); monotonic segment selection (run-up, run-down, automatic) | Brandt 2011 | v0.4 |

### 2.4 Spectra and slices (`rota.slices`)

| Function | Purpose | Literature | Version |
|---|---|---|---|
| `order_spectrum` | Energy-averages the order map over time to obtain the average order spectrum | Brandt 2011 | v0.1 |
| `order_slice` | Amplitude of a given order as a function of rpm (can return the complex amplitude, for use by `OrderTable`) | Brandt 2011 | v0.1 |
| `frequency_slice` | Fixed-frequency slice, for observing structural resonances | Brandt 2011 | v0.4 |
| `overall_level` | Overall sound level versus rpm, optionally A-weighted | IEC 61672-1 | v0.4 |
| `metric_vs_rpm` | Bins any time-varying quantity (psychoacoustic metric, sound level, output of third-party tools) along rpm or vehicle speed: monotonic segment selection, bin width, minimum samples per bin, start skip, aggregation (mean, percentile, power average); returns the counts and the dispersion | — | v0.4 |
| `frequency_to_order`, `order_share` | Frequency track (e.g. the dominant tone frequency given by a tonality tool) → order attribution (within a given tolerance); computes the fraction falling on a given order and its harmonics | — | v0.4 |

### 2.5 Order extraction (`rota.extract`, formerly `rota.vkf`)

| Function | Purpose | Literature | Version |
|---|---|---|---|
| `order_waveform` | Extracts the time-domain order waveform and complex envelope with the second-generation Vold-Kalman filter. Joint over several orders (banded symmetric positive-definite equations, solved exactly with the SciPy banded solver); −3 dB bandwidth → weight mapping; a separate rpm reference for each of several shafts | Vold & Leuridan 1993, 1995; Vold et al. 1997; Feldbauer & Höldrich 2000; Tůma 2005; Pan & Lin 2006 | v0.2 |
| `tracked_harmonics` | Phase-locked short-time harmonic extraction: each order is demodulated with the phase given by `rpm_to_phase`, and short-time averaging with a normalised window satisfying the COLA condition gives a complex coefficient track, which is then re-modulated and summed back to the time domain. Optional significance threshold (χ² test), tachometer delay. Far cheaper than Vold-Kalman, suited to long recordings and many channels | Antoni 2009; angle–time cyclostationary analysis (Antoni et al., to be verified) | v0.2 |
| `remove_orders` | Subtracts the specified orders from the original signal (using either extractor), for "remove an order and listen" | As above | v0.2 |
| `residual` | Original signal minus all shafts and all orders, giving the broadband residual | As above | v0.2 |
| `resolution_check` | Resolvability diagnostic: for adjacent orders, or two shafts at nearly the same speed, the relation of the beat frequency to the reciprocal of the window length and to the envelope bandwidth; the number of orders spanned by a constant-bandwidth time window at low rpm; whether the demodulation bandwidth exceeds the spacing between adjacent orders. Gives warnings and a recommended window length (in revolutions) or bandwidth | — | v0.2 |
| `kalman_smoother` (optional) | Kalman/RTS smoother for a random-walk envelope model, as a Bayesian reference implementation of Vold-Kalman | — | after v0.4 |

### 2.6 Resynthesis (`rota.resynth`, new)

Goal: turn analysis results (order tables, extracted order waveforms, residuals) back into audible sound, and allow resynthesis after modification. Typical uses: listening-experiment stimuli ("remove an order and listen", "keep only the broadband"), virtual operating conditions (mapping an order table onto a new rpm profile), and the sound engine of a driving simulator (block-wise synthesis).

| Function or class | Purpose | Literature | Version |
|---|---|---|---|
| `OrderTable` | Data structure: order × rpm → complex amplitude (Pa), explicitly marked as peak or RMS; supports non-integer and negative orders and a fixed frequency offset per order (carrier sidebands, e.g. sidebands of an inverter switching frequency); can be built from `order_slice` or `tracked_harmonics`; NPZ/JSON read and write | — | v0.3 |
| `synthesize_orders` | Additive synthesis from an `OrderTable` and `rpm(t)`: the phase is the continuous integral 2π·order·∫rpm/60 dt; between rpm grid points the amplitude is interpolated, not the real and imaginary parts; orders above Nyquist are muted rather than clamped; the initial phase can be random (seedable), shared, or taken from the phases in the table | McAulay & Quatieri 1986 | v0.3 |
| `OrderSynth` (streaming) | Block-wise API: keeps the phase accumulator and the state of the previous block, takes one block of rpm or load at a time and outputs one block of audio, for real-time programs or simulators (ROTA itself does no audio I/O) | — | v0.3 |
| `synthesize_broadband` | Generates broadband noise from a magnitude-spectrum envelope (a lookup table that can vary with rpm or vehicle speed): random-phase IFFT grains + overlap-add with a power-complementary window (√Hann, 50 % overlap), no frame-rate pumping; very low frequencies removed within each grain; cross-fade between adjacent operating conditions with √ weights | Serra & Smith 1990; Allen & Rabiner 1977 | v0.3 |
| `remap_rpm` | Maps an order table or extraction result onto a new rpm profile (virtual acceleration, changed gear ratio, changed apparent vehicle speed of a recording). This is the correct way to change the apparent vehicle speed, not a general pitch shift or speed change; the frequency changes by only about 0.25 % per km/h, so rpm and frequency accuracy must be far better than this | — | v0.3 |
| `modify_orders` | Per-order gain, muting, frequency shift and replacement applied to extracted coefficient tracks or an order table, followed by re-modulation | — | v0.3 |
| `resynthesize` | Recombines orders + residual, with an order-to-residual energy ratio (dB) that can be specified; returns three outputs: orders-only, residual-only and mix | — | v0.3 |
| `stationary_excerpt` | Generates a listening excerpt with constant frequency and steady amplitude from an accelerating recording: angle-domain average spectrum (or a sinusoid + residual model) + random-phase IFFT; tonal components can use either of two models, true sinusoids (no beating) or random-phase clusters; reports the envelope fluctuation and the crest factor | — | v0.3 |
| `provenance` | Writes a JSON sidecar for every resynthesis: all parameters, seed, ROTA version, input file digest | — | v0.3 |

### 2.7 Test signals (`rota.synth`)

| Function | Purpose | Version |
|---|---|---|
| `synth_runup` | Synthesises run-up or run-down signals. Kinematic chain: vehicle speed profile → wheel speed → gear ratio and final drive ratio → rpm; the profile can be linear, power-law, S-shaped with zero slope at the end points, or an arbitrary array; idle, gear-shift events and rpm drop-back. Orders are given as data: order (including half orders), reference amplitude, amplitude law versus rpm (initial ramp, minimum-frequency mask, optional AM/FM), initial phase (seedable); optional resonances (transfer functions) and fixed-frequency components; road noise and wind noise with envelopes versus vehicle speed; several shafts (independent speeds, fixed speed ratio, nearly equal speeds); tachometer pulses (several pulses per revolution, jitter, dropped pulses). Outputs the mix and a stem for each component (same gain), together with the truth: rpm(t), shaft angle, amplitude and phase of each order, noise envelopes, plus the JSON configuration and the seed | v0.1 |
| `rpm_profiles` | Rpm profile generation: linear, exponential, piecewise with holds (e.g. a 40→100→40 km/h mapping), twin shafts (independent, locked ratio, small difference) | v0.1 |
| `synth_tone_noise` | Listening-experiment stimuli: a pure tone or harmonic complex + broadband or hump-shaped noise, with the tone-to-noise ratio (TNR) set by allocating the total power; anchor signals (e.g. a pure tone and white noise at 60 dB SPL) | v0.3 |

### 2.8 Validation metrics (`rota.metrics`, new)

| Function | Purpose | Version |
|---|---|---|
| `track_error` | Rpm track error: median relative error, fraction of frames within 3 % and 5 %, boundary-hit statistics | v0.1 |
| `waveform_similarity` | Correlation coefficient, NRMSE; energy-weighted for multichannel signals (avoids uniform averaging over low-energy channels); cross-capture (correlation of the estimate with the ground truth of another source); zero baseline (correlation of the raw mixture with the truth) | v0.2 |
| `order_residual_energy` | Remaining energy of the target order in the residual after order removal (dB, relative to the original order energy) | v0.2 |
| `window_checks` | COLA and power-complementarity checks; peak of the modulation spectrum at the frame rate (pumping detection) | v0.3 |
| `envelope_stats` | Envelope fluctuation (percentiles), crest factor, error relative to the original spectrum | v0.3 |

### 2.9 Psychoacoustic bridge (`rota.tonal`, thin layer, new)

ROTA does not implement metrics that already have open-source implementations, such as loudness, sharpness, roughness and ECMA-418-2 tonality. It only provides bridges:

| Function | Purpose | Literature | Version |
|---|---|---|---|
| `a_weighting`, `level_db` | IEC 61672-1 A/C weighting (bilinear transform of the analogue poles, normalised at 1 kHz), applied only once (one pass in the time domain, or in the frequency domain), not zero-phase forward-backward filtering; dB re 20 µPa; RMS definition explicit; binaural energy averaging | IEC 61672-1 | v0.4 |
| `to_calibrated_48k` | Calibration-preserving polyphase resampling, outputting a 48 kHz signal in Pa, for use with SQAT, MoSQITo and open-source ECMA-418-2 implementations | — | v0.4 |
| `order_prominence` | Computes the ECMA-74 Annex D tone-to-noise ratio TNR and prominence ratio PR on the order-tracked spectrum (angle domain): per-tone detection, prominence flag, frequency-dependent threshold; the window is long enough to resolve low-frequency tones (e.g. a 140 Hz tone lies within a critical band about 100 Hz wide); yields an "order prominence versus rpm" curve | ECMA-74; ISO 7779; ECMA-418-1 | v0.4 |
| `components` | Description structure for order components (order, rpm range, level, bandwidth, harmonicity index), to ease comparison with the component output of tonality tools | Becker & Sottek 2019 (to be verified) | v0.4 |

### 2.10 Plotting (`rota.plot`)

Plotting helpers, separate from computation: Campbell diagram, order map, slices, rpm track and confidence, before/after comparison of resynthesis. v0.4.

---

## 3. Version plan

**v0.1 Core chain**: complete the analysis of an accelerating run with a tachometer.
`tacho_to_rpm`, `rpm_to_phase`, `regularize_control`, `speed_to_rpm`, `steady_segments`, `angular_resample`, `rpm_frequency_map`, `rpm_order_map`, `order_spectrum`, `order_slice`, `synth_runup`, `rpm_profiles`, `track_error`. Validation cases T1, T2, T5, T14.

**v0.2 Order extraction and tacholess operation**:
`rpm_from_ridge`, `order_waveform`, `tracked_harmonics`, `remove_orders`, `residual`, `resolution_check`, `waveform_similarity`, `order_residual_energy`. Validation cases T3, T4, T6, T8, T10, T12, T13.

**v0.3 Resynthesis**:
All of `rota.resynth`, `angular_to_time`, `stationarize`, `virtual_rpm`, `synth_tone_noise`, `window_checks`, `envelope_stats`. Validation cases T9, T11.

**v0.4 Application layer and bridges**:
`frequency_slice`, `overall_level`, `metric_vs_rpm`, `order_share`, Δrpm binning, `align_signals`, `rota.tonal`, `rota.plot`. Provides examples linking to SQAT, MoSQITo and open-source ECMA-418-2 implementations, for example "how loudness or tonality varies with rpm after an order is removed".

**v1.0 Official release**:
Full validation report, English documentation, PyPI release, Zenodo DOI. Submit a validation paper (JOSS or an acoustics conference).

**Not planned for now**:

- Real-time audio I/O (only a block-wise synthesis API is provided), GUI;
- Analysis specific to rotor dynamics (orbits, Bode plots);
- Spatial audio (DoA, beamforming, Ambisonics decoding and binaural rendering): to be a separate companion package; ROTA only guarantees multichannel linear operators (design principle 6);
- Reading any proprietary or commercial file format (section 4);
- The metrics themselves: loudness, sharpness, roughness, ECMA-418-2 tonality.

---

## 4. Independent implementation rules

ROTA's API is friendly to MATLAB users, but the implementation is fully independent. All contributors follow the rules below.

### Prohibited

1. Reading, copying or translating the source code or internal data structures of any commercial software (MATLAB toolboxes, commercial NVH analysis and simulation software). This includes opening built-in functions with `edit` or `type`, and stepping into built-in functions in a debugger.
2. Copying the text, example code or example data of commercial software documentation. Docstrings, tutorials and examples are all written independently. Example data are generated by `synth_runup`, or come from own measurements with clear rights.
3. Implementing or bundling a parser for any proprietary or unpublished file format, or describing its internal structure in the documentation. Users convert their data to WAV, NPZ, HDF5 or CSV before using ROTA.
4. Porting third-party code whose licence is unclear, including research code shared by colleagues or collaborators. Reimplement it only from the published paper, and cite the paper.
5. Using trademarks such as MATLAB in the project name, the package name or in promotion.
6. Writing "reproduces function X of software Y" in code comments. Write "implemented based on reference Z" instead.
7. Using restricted data (partner vehicle recordings, participant data, unpublished datasets) as test fixtures or examples. The repository contains no vehicle models, partner organisations, participant information or absolute local paths.

### Allowed

1. Implementing algorithms from the published literature, with the source stated in the References section of the docstring.
2. Using MATLAB as a black-box reference: call only public functions, record the inputs and outputs, and use them for numerical comparison.
3. Providing a "MATLAB user correspondence table" in the documentation that states the functional correspondence factually.
4. Reusing open-source code compatible with GPL-3.0-or-later (e.g. BSD, MIT, Apache-2.0, GPL-3.0), with acknowledgement in the NOTICE file.

### Process

- The PR template contains one confirmation: "This submission did not consult the source code of any commercial software and did not port code of unclear licence."
- Before consulting or reusing third-party open-source code, confirm that its licence is compatible with ROTA, and acknowledge it in the NOTICE file.
- Before each release, search the whole git history for sensitive words (commercial software names, partners, vehicle models, personal names, local paths).

### Independence statement (placed in the README)

> ROTA is an independent academic project. It is not affiliated with, endorsed by, or supported by MathWorks or any commercial software vendor. MATLAB is a registered trademark of The MathWorks, Inc.

---

## 5. Validation and comparison

### 5.1 Principles

- **Analytic ground truth is the first standard; MATLAB is the second reference.** The purpose of the comparison is to confirm that the two agree in their estimate of the same physical quantity, not to copy MATLAB's output digit for digit. When the two disagree, the ground truth decides, and the report explains the source of the difference.
- **Specify parameters explicitly.** In MATLAB calls, give all parameters explicitly (window function, resolution, overlap, amplitude type, filter bandwidth, etc.) and do not rely on defaults. This keeps the comparison reproducible, and ROTA does not need to copy MATLAB's default settings.
- **Compare physical quantities, not grids.** The framing details of the two sides may differ. The comparison therefore uses derived quantities such as amplitudes along order ridges, peak positions and rpm curves, and does not compare map cells one by one.
- **Round-trip tests for resynthesis.** Synthesis → analysis → comparison with the input table; broadband synthesis uses self-checks (window power sum, modulation spectrum).

### 5.2 Comparison table

| ROTA | MATLAB (Signal Processing Toolbox) | Comparison method | Comparison metric |
|---|---|---|---|
| `tacho_to_rpm` | `tachorpm` | Direct comparison | Rpm curve error |
| `rpm_from_ridge` | `rpmtrack` | Direct comparison | Rpm curve error; boundary-hit flag |
| `angular_resample` | No public counterpart | Analytic truth; indirect through `rpm_order_map` | Spectral peak width and leakage of a constant order in the angle domain |
| `rpm_frequency_map` | `rpmfreqmap` | Direct comparison | Amplitude of each order along the ridge; frequency axis |
| `rpm_order_map` | `rpmordermap` | Direct comparison | Amplitude of each order; order resolution |
| `order_spectrum` | `orderspectrum` | Direct comparison | Peak order and amplitude |
| `order_slice` | `ordertrack` (input is an order map) | Direct comparison | Amplitude–rpm curve |
| `order_waveform` (waveform) | `orderwaveform` | Direct comparison | Waveform correlation coefficient; −3 dB bandwidth |
| `order_waveform` (envelope) | `ordertrack` (input is a signal, Vold-Kalman method) | Direct comparison | Envelope amplitude |
| `tracked_harmonics` | No public counterpart | Analytic truth; cross-check with `order_waveform` | Waveform correlation coefficient; amplitude of the coefficient track |
| `remove_orders`, `residual` | No public counterpart | Indirect: original signal minus the `orderwaveform` result | Remaining energy of that order in the residual |
| `resolution_check` | None | By definition (T8) | Whether the warning is triggered |
| `frequency_slice` | No public counterpart | Analytic truth | Position and amplitude of the resonance peak |
| `overall_level`, `rota.tonal` | Not compared with MATLAB | Cross-validation against the `sound_level_meter` of SQAT (open source, verified against IEC 61672-1) | Level difference; TNR/PR difference |
| `rota.resynth` | No public counterpart | Round-trip test (T9); self-check (T11) | Table round-trip error; pumping peak |
| `rota.metrics`, `synth_runup` | None | By definition | — |

### 5.3 Test signals

| ID | Content | Main focus |
|---|---|---|
| T1 | Single order, constant amplitude, linear acceleration | Absolute accuracy of amplitude and rpm |
| T2 | Several orders + fixed-frequency resonance + noise | Maps, slices, resonance identification |
| T3 | Exponential deceleration (nonlinear rpm) | Resampling and ridge tracking |
| T4 | Two shafts, crossing orders | Vold-Kalman multi-order decoupling |
| T5 | Tachometer with several pulses per revolution, pulse jitter, dropped pulses | Robustness of `tacho_to_rpm` |
| T6 | Fast run-up with a high rate of rpm change | Limits of COT and Vold-Kalman |
| T7 (optional) | Own measured data | Real operating conditions; published only if the data rights allow |
| T8 | Twin-shaft speed ratio close to 1 (differing by 0.5 %, 2 %, 4 %), same orders | Resolvability limit; `resolution_check` must warn when the orders are unresolvable |
| T9 | Order-table round trip: `OrderTable` → `synthesize_orders` → `rpm_order_map` or `order_slice` → table | Consistency of the resynthesis and analysis chains |
| T10 | True rpm outside the search range of `rpm_from_ridge` | Must report a boundary hit, and must not silently return the boundary value |
| T11 | Output of `synthesize_broadband` for a constant spectral envelope | No peak at the frame rate in the modulation spectrum; constant long-term power |
| T12 | T2 signal passed through `remove_orders` | Remaining energy of the target order in the residual |
| T13 | Reference rpm with errors (0.5 % bias, random jitter, steps): degradation curves of each extractor | Robustness of extraction to rpm error; the documentation gives the acceptable rpm error |
| T14 | Low-rpm start-up including idle, adjacent orders only 0.5 order apart | Variation of order resolution with rpm; angle-domain STFT versus constant-bandwidth time-domain STFT |

### 5.4 Initial tolerances (to be adjusted after measurement)

| Quantity | Against analytic truth | Against MATLAB |
|---|---|---|
| Median relative rpm error | ≤ 0.1 % | ≤ 0.1 % |
| Tacholess rpm: fraction of frames within 3 % (steady or slowly rising) | ≥ 90 % | Differs from `rpmtrack` by ≤ 5 percentage points |
| Order amplitude, steady segments | ≤ 0.2 dB | ≤ 0.5 dB |
| Order amplitude near a resonance | ≤ 1 dB | ≤ 1 dB |
| Order waveform correlation coefficient (edges removed) | ≥ 0.99 | ≥ 0.99 |
| Vold-Kalman −3 dB bandwidth | Differs from the set value by ≤ 5 % | Differs from MATLAB by ≤ 5 % |
| Order-table round-trip amplitude (T9) | ≤ 0.2 dB | — |
| Remaining energy of the target order in the residual after order removal (T12) | ≤ −20 dB | — |
| Frame-rate pumping of broadband synthesis (T11, peak of the modulation spectrum at the frame rate; a single realisation of a 10 s recording has about 1 dB of statistical scatter, so judge by the mean over several realisations or by a recording of ≥ 40 s) | ≤ 1 dB | — |

### 5.5 Directory and workflow

```
validation/
├── make_signals.py      # generates T1–T14 and the analytic truth with synth_runup, saved as .mat / .npz
├── signals/             # test signals and analytic truth
├── matlab/
│   └── run_reference.m  # contains only: load data, call public functions, save output
├── reference/           # MATLAB output, with MATLAB and toolbox version information
└── compare.py           # runs ROTA, compares with the truth and with MATLAB, generates the report
```

1. Run `make_signals.py` to generate the test signals and the analytic truth.
2. Run `run_reference.m` in MATLAB. Save the output to `reference/`, and record the version information given by `ver`.
3. Run `compare.py` to generate the comparison report (tables and figures).
4. Continuous integration runs only pytest (against the analytic truth) and the comparison with the stored reference data; MATLAB does not need to be installed.

Before publishing the MATLAB output in `reference/`, confirm that the MATLAB licence terms of the institution concerned allow it.

---

## 6. Design constraints: lessons from earlier work

The following experience comes from the author's earlier work on order analysis, preparation of listening-experiment stimuli and sound synthesis, and it shaped the design of several functions above. ROTA reuses none of that code; it takes over only the lessons.

**Speed and phase**

1. A speed estimator must report whether it has hit the search boundary. A fundamental-frequency tracker once locked at the lower search limit on an entire batch of files, so the subsequent angle-domain resampling effectively did nothing, and the pipeline raised no error. → the boundary-hit flag of `rpm_from_ridge`, T10.
2. A single fundamental frequency cannot align several unrelated harmonic families (e.g. motor orders and inverter sidebands). → multi-shaft, multi-carrier design: the frequency offset of `OrderTable`; `tracked_harmonics` and `order_waveform` process each shaft separately.
3. In a three-cylinder engine the integer orders and the firing order (order 1.5) have comparable energy, so a comb built only on the firing order locks onto 2/3 or 4/3 of the true speed; in a fast run-up the change of order frequency within the window exceeds the whitening bandwidth, which erases the order itself. → configurable combs, whitening bandwidth adjusted to the operating condition, Viterbi continuity penalty.
4. Gear shifts make rpm non-monotonic, so binning along rpm is meaningful only for fixed-gear conditions. → monotonic segment selection and count output in `metric_vs_rpm`.
5. Unit pitfalls between vehicle speed and rpm (m/s versus km/h, rad/s versus rpm) caused repeated errors. → units explicit in function signatures and stated in the documentation.

**Angle domain and extraction**

6. Angle-domain resampling with linear interpolation has no anti-aliasing; a static Gaussian order mask cannot follow amplitude and phase modulation. → `angular_resample` has an adaptive low-pass and high-order interpolation; extraction uses phase-locked demodulation or Vold-Kalman.
7. When two shafts differ in speed by 0.5 %, any single-channel tracker meets a bias–variance wall (the beat frequency and the rate of envelope change are of the same order). → `resolution_check` gives a resolvability warning, T8.
8. Uniform averaging of multichannel metrics severely underestimates them (the energy is concentrated in a few channels). → energy weighting in `waveform_similarity`.
9. A time-averaged spectrum smears drifting orders; pure-sinusoid resynthesis loses the noise floor and does not sound like the original recording. → `stationarize` or angle-domain averaging + modelling orders and residual separately.

**Resynthesis**

10. When the real and imaginary parts of complex amplitudes are interpolated between rpm grid points, adjacent points with different phases cancel; the amplitude can collapse by tens of dB, and chopping at tens of Hz appears. → interpolate the amplitude, and handle the phase separately.
11. Independent frames with random phases must be overlap-added with a power-complementary window (√Hann); an amplitude-COLA window produces power pumping at the frame rate (once as much as +14 dB). Very low frequencies within a grain produce clicks at the window switch. → window design and grain high-pass in `synthesize_broadband`, T11.
12. If RMS normalisation is applied after a gain is applied to the whole spectrum, the gain is cancelled out. → the gain acts only on the part by which the orders exceed the residual, and normalisation is based on the reference synthesis.
13. Nearby random-phase spectral lines produce beating, whereas merging spectral peaks into Gaussian clusters turns the tone into narrowband noise with a strongly fluctuating envelope. → `stationary_excerpt` offers two models, true sinusoids and random-phase clusters, and reports the envelope fluctuation.
14. An unclear amplitude convention (peak or RMS, amplitude or power per bin) causes errors of 3 dB or even 30 dB. → explicit marking in `OrderTable`, design principle 8.
15. Parameters drifted between notebooks and scripts, and the settings used for the final stimulus could not be reconstructed. → `provenance` sidecar, deterministic seeds derived from the file name.
16. An order above Nyquist clamped to fs/2 produces a spurious tone. → mute it.

**Bridging**

17. Third-party psychoacoustic tools need calibrated Pa, 48 kHz, a regular time axis and rpm(t) with gaps; each has its own start-up transient (about 0.6 s for loudness and tonality, about 1.5 s for roughness, 4–5 s for fluctuation strength), which must be skipped when aggregating. → `to_calibrated_48k`, `regularize_control`, and the start-skip parameter of `metric_vs_rpm`.
18. Long-window spectral metrics (TNR, PR, single-frame Aures) are sensitive to order drift; auditory-model metrics are not. → `order_prominence` is computed in the angle domain (orders constant).

**Test signals and validation**

19. Earlier synthetic test signals started the rpm at 0 with every order at full amplitude, so the amplitude was independent of rpm, and they were validated with "the exact rpm used for synthesis + the known list of orders", which is circular reasoning. → `synth_runup` stores the true amplitude and phase of each order, with idle and amplitude laws; validation adds rpm errors (T13) and low-rpm start-up (T14).
20. A constant-bandwidth time-domain STFT relabelled onto an order axis has a main lobe that spans several orders at low rpm, so half orders cannot be separated. → `rpm_order_map` uses an angle-domain STFT with the window length defined in revolutions; `resolution_check` gives a hint.
21. Synchronous demodulation with a fixed 20 Hz low-pass lets adjacent orders (0.5-order spacing) leak into each other, and at rpm near 0 all orders collapse. → the demodulation bandwidth scales with rpm according to the order spacing; the window length of `tracked_harmonics` is in revolutions.
22. The MATLAB and Python implementations of the same algorithm silently differed in the number of samples, the window function and the integration rule, so the results could not corroborate each other. → a single implementation, a single ground-truth generator, configuration saved to disk.

**Measurement and alignment**

23. Without hardware synchronisation between several acquisition systems, only hand claps are available, with an accuracy of tens of ms; a recording without a marker cannot be aligned. → `align_signals` supports start and end markers and drift estimation, and reports the uncertainty; the documentation recommends one marker at the start and one at the end of each recording.
24. Slow channels (vehicle speed, acceleration) have a sampling rate at an integer ratio to the acoustic channels (e.g. 5:1), and the channel layout and calibration differ from vehicle to vehicle. → `regularize_control` accepts any sampling rate; calibration factors are passed explicitly.
25. Spherical-harmonic conventions (ACN/SN3D, Condon–Shortley phase, N3D scaling) are inconsistent between codes, which can rotate a directional pattern by 180° or weight it per order. This is a lesson for the companion spatial package; the ROTA core only guarantees per-channel linearity and builds in no spherical-harmonic convention.

**Sound manipulation**

26. Changing the apparent vehicle speed of a recording cannot be done by video speed change or by general pitch shifting and speed change; the orders and the residual must be separated, the orders re-modulated at the new rpm, and the residual scaled according to a vehicle-speed law. The frequency changes by only about 0.25 % per km/h, so rpm and frequency accuracy must be far better than this. → `remap_rpm`, `resynthesize`.
27. The gains used when components are rendered separately and then mixed must be recorded explicitly (there was once an undocumented ×2 and a 1/N scaling in a mixing tool); zero-phase forward-backward A-weighting weights twice. → design principle 8; `a_weighting` is applied only once.

---

## 7. References

Entries marked "to be verified" are checked against their bibliographic records before the corresponding function is implemented.

- Brandt, A. (2011). *Noise and Vibration Analysis: Signal Analysis and Experimental Procedures*. Wiley.
- Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order tracking. *Mechanical Systems and Signal Processing*, 11(2), 187–205. doi:10.1006/mssp.1996.0056
- Potter, R. (1990). A new order tracking method for rotating machinery. *Sound and Vibration*, 24(9), 30–34. (to be verified)
- Gade, S., Herlufsen, H., Konstantin-Hansen, H., & Wismer, N. J. (1995). Order tracking analysis. *Brüel & Kjær Technical Review*, No. 2. (to be verified)
- Vold, H., & Leuridan, J. (1993). High resolution order tracking at extreme slew rates, using Kalman tracking filters. SAE Technical Paper 931288.
- Vold, H., & Leuridan, J. (1995). High resolution order tracking using Kalman tracking filters: theory and applications. SAE Technical Paper 951332.
- Vold, H., Mains, M., & Blough, J. (1997). Theoretical foundations for high performance order tracking with the Vold-Kalman tracking filter. SAE Technical Paper 972007.
- Feldbauer, C., & Höldrich, R. (2000). Realization of Vold-Kalman tracking filter: a least squares problem. *Proceedings of the COST G-6 Conference on Digital Audio Effects (DAFx-00)*, Verona.
- Tůma, J. (2005). The passband width of the Vold-Kalman order tracking filter. *Sborník vědeckých prací VŠB-TU Ostrava, Řada strojní*, 51(2), 149–154.
- Pan, M.-C., & Lin, Y.-F. (2006). Further exploration of Vold–Kalman-filtering order tracking with shaft-speed information—I: Theoretical part, numerical implementation and parameter investigations. *Mechanical Systems and Signal Processing*, 20(5), 1134–1154.
- Pan, M.-C., & Wu, C.-X. (2007). Adaptive Vold–Kalman filtering order tracking. *Mechanical Systems and Signal Processing*, 21(8), 2957–2969. (to be verified)
- Peeters, C., Leclère, Q., Antoni, J., Lindahl, P., Donnal, J., Leeb, S., & Helsen, J. (2019). Review and comparison of tacholess instantaneous speed estimation methods on experimental vibration data. *Mechanical Systems and Signal Processing*, 129, 407–436. doi:10.1016/j.ymssp.2019.02.031
- Antoni, J. (2009). Cyclostationarity by examples. *Mechanical Systems and Signal Processing*, 23(4), 987–1036.
- Antoni, J., et al. (2022). Harmonic tracking and extraction in the angle–time cyclostationary framework. *AIAA Journal*, 60. (to be verified)
- Rémond, D., Antoni, J., & Randall, R. B. (2014). Editorial for the special issue on instantaneous angular speed (IAS) processing and angular applications. *Mechanical Systems and Signal Processing*, 44. (to be verified)
- Hermes, D. J. (1988). Measurement of pitch by subharmonic summation. *Journal of the Acoustical Society of America*, 83(1), 257–264.
- McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis based on a sinusoidal representation. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 34(4), 744–754.
- Serra, X., & Smith, J. O. (1990). Spectral modeling synthesis: A sound analysis/synthesis system based on a deterministic plus stochastic decomposition. *Computer Music Journal*, 14(4), 12–24.
- Allen, J. B., & Rabiner, L. R. (1977). A unified approach to short-time Fourier analysis and synthesis. *Proceedings of the IEEE*, 65(11), 1558–1564.
- Welch, P. D. (1967). The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70–73.
- Knapp, C. H., & Carter, G. C. (1976). The generalized correlation method for estimation of time delay. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 24(4), 320–327.
- Griffin, D. W., & Lim, J. S. (1984). Signal estimation from modified short-time Fourier transform. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 32(2), 236–243.
- Becker, J., & Sottek, R. (2019). Psychoacoustic order tonality calculation. SAE Technical Paper. (to be verified)
- Sottek, R., Becker, J., & Lobato, T. (2025). High-resolution option for the Sottek hearing model tonality enables accurate order analysis. SAE Technical Paper. (to be verified)
- Dupré, T., et al. (2023). Analysis-by-synthesis of engine sounds. *Acta Acustica*, 7. (to be verified)
- IEC 61672-1:2013. Electroacoustics – Sound level meters – Part 1: Specifications.
- ECMA-74 (Annex D: prominent discrete tones); ISO 7779 (Annex D); ECMA-418-1; ECMA-418-2:2025.
- ISO 532-1:2017. Acoustics – Methods for calculating loudness – Part 1: Zwicker method.

---

## 8. Author and acknowledgements

ROTA is designed, implemented and maintained by Zhenxian Li. Claude (Anthropic) was used as an assistant during development.
