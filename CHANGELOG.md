# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project intends to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Application layer and bridges to psychoacoustic toolboxes.

### Added

- `rota.tonal`: A and C frequency weighting (IEC 61672-1; `weighting_db`, `weighting_filter`, `weight`), levels and level versus time, channel combination, `to_calibrated_48k` (calibration to Pa and resampling to 48 kHz for psychoacoustic tools), octave and fractional-octave filterbank (IEC 61260-1; `octave_band_centres`, `octave_filterbank`, `octave_levels`) and band levels along rpm (`band_levels_vs_rpm`).
- `rota.tonal`: tone-to-noise ratio and prominence ratio after ECMA-418-1 (3rd edition, 2024): `tone_prominence` for a stationary spectrum, `order_prominence` for orders in the angle domain along a run-up, `prominence_criterion`, `critical_bandwidth`, `critical_band_edges`, `components`.
- `rota.slices`: `frequency_slice`, `overall_level`, `metric_vs_rpm` (bins any time-varying metric, for instance loudness from a psychoacoustic toolbox, along rpm), `frequency_to_order`, `order_share`.
- `rota.maps`: binning of map frames into rpm bins (`rpm_bins`, `segment`, `min_count`, `bin_phase`; `bin_map`, `BinnedMap`).
- `rota.speed`: `align_signals` (GCC-PHAT delay), `detect_onsets`, `AlignResult`.
- Examples 07 (MoSQITo), 08 (sottek-hearing-model) and 09 (SQAT in MATLAB, workflow `sqat-bridge`, results kept in the workflow artifact); the toolboxes are optional, external dependencies and the tests of examples 07 and 08 are skipped without them.
- Worked analysis (`examples/10_worked_analysis.py`, `docs/worked_example.md`, `tests/test_worked_example.py`): an electric-drive run-up with a gear whine that passes a resonance, from speed and overview to the ECMA-418-1 diagnosis, a what-if reduction of the order, its effect on loudness, sharpness and ECMA-418-2 tonality, and the findings checked against the built-in truth; audio before and after in `docs/audio/`.

### Changed

- Roadmap: octave and 1/3-octave band analysis added to v0.5; `rota.plot` moves to v0.6.

## [0.4.0] - 2026-10-27

Realistic test signals, stress tests and the black-box reference comparison.

### Added

- `rota.synth.synth_scenario` and `ScenarioSignal`: complete synthetic drive-train recordings with exact truth. Kinds: `ice_runup` (four-stroke four-cylinder engine: full-load run-up from 1200 rpm at about 500 rpm/s through upshifts that drop the speed by about 30 %, optional clutch-slip launch, angle-periodic speed ripple, orders with level and speed-slope laws, combustion bursts ringing a structure resonance), `ev_runup` (electric drive: shaft rpm 100 per km/h, motor-family orders, gear mesh and whine with slow modulation, optional whine sidebands, a dithered inverter carrier with speed-dependent sidebands, fixed high-frequency tones set against the local noise floor, coast-down variant) and `steady_cruise` (either powertrain at constant speed with weak tones). Each has a tachometer channel with jitter and delay, a modal transfer path with a cabin boom, a cabin impulse response (smooth below the Schroeder frequency, diffuse above), receiver-level broadband noise with a stated spectral tilt, low-frequency hump and speed laws, transient events, one or more receiver channels with diffuse-field coherence, and an optional recorder (`measurement`: clipping, quantisation, self-noise, clock drift). All default parameters are illustrative values typical of passenger cars, rounded, not measurements of any specific vehicle; `scenario_defaults` returns them. The result holds the recording, the true instantaneous speed and angle, every stem before and after the path (the stems add up to the recording bit for bit for an ideal recorder), the source-order truth, `through_path`, `path_response` and `quasi_static_amplitude`, and a JSON configuration with the seed.
- `rota.synth` re-exports the building blocks behind the scenarios: `drive_rpm`, `angular_speed_ripple`, `angle_impulses`, `sideband_orders`, `carrier_components`, `modal_path`, `modal_response`, `room_impulse_response`, `apply_ir`, `speed_noise`, `transient_events`, `measurement_chain` and `delay_signal`.
- Tests of the scenarios against their own truth and the documented defaults (`tests/test_scenario.py`).

- Stress tests on the realistic scenarios (`tests/test_stress.py`, results and recommended settings in `validation/stress.md`).
- Black-box comparison with MATLAB in CI (`validation/export_mat.py`, `validation/matlab/run_reference.m`, `validation/compare.py`, workflow `matlab-reference`); documented public functions only, explicit parameters, output kept in the workflow artifact.
- Examples 05 (engine run-up, two channels) and 06 (electric drive) with synthetic audio in `docs/audio/`; the README shows the engine run-up and links the audio.
- `RidgeResult.quality`, `RidgeQualityWarning`.

### Changed

- `rpm_from_ridge`: automatic window (`window="auto"`) chosen from the sweep rate; channels averaged in power, not as signals; `confidence` recalibrated (detection × distinctness, including rational neighbours); search-bound and whitening warnings only when a failure is likely. Recommended comb by machine type documented (four-stroke N-cylinder: `(N/2, N, 3N/2)`).
- Roadmap: the application layer and psychoacoustic bridges move from v0.4 to v0.5.

## [0.3.0] - 2026-10-26

Order-based resynthesis.

### Added

- `rota.resynth`: `OrderTable` (orders × rpm complex amplitudes with an explicit peak/RMS convention and per-row frequency offsets; built from order slices or extracted components; magnitude and phase interpolated separately), `synthesize_orders` and the streaming `OrderSynth` (identical output for any block split; components outside (0, fs/2) muted), `synthesize_broadband` (random-phase grains with power-complementary windows, no frame-rate pumping), `modify_orders`, `remap_rpm`, `resynthesize`, `stationary_excerpt`, `provenance`, `stable_seed`.
- `rota.angle`: `angular_to_time`, `stationarize`, `destationarize`.
- `rota.speed`: `virtual_rpm`; `WhiteningWarning`.
- `rota.synth`: `synth_tone_noise` (tone-plus-noise stimuli with an exact broadband power split).
- `rota.metrics`: `window_checks`, `modulation_peak`, `envelope_stats`.
- Validation cases T9 (order-table round trip) and T11 (broadband synthesis).
- Four example scripts with figures (`examples/`).
- CI job testing the oldest supported dependencies (Python 3.10, NumPy 1.23, SciPy 1.9).
- Published on PyPI as `rota-order` (`pip install rota-order`; the import name stays `rota`).

### Changed

- `rpm_from_ridge`: whitening now adapts to the sweep of the lines by default (`whiten="auto"`, two passes). With the former fixed 15 Hz default an ordinary run-up at 750 rpm/s was tracked with 66 % median error and no warning; it is now within 3 % on 99 % of frames. Too narrow explicit widths raise `WhiteningWarning` and lower `confidence`.

## [0.2.0] - 2026-10-26

Order extraction and tacholess speed estimation. Development milestone; not released separately (first public release: 0.3.0).

### Added

- `rota.extract`: `order_waveform` (second-generation Vold-Kalman filter: joint least-squares over orders and shafts, banded Cholesky solve with iterative refinement, -3 dB envelope bandwidth mapped analytically to the weight, block processing), `tracked_harmonics` (phase-locked averaging over windows of fixed length in revolutions, optional significance gate), `remove_orders`, `residual`, `resolution_check` (resolvability of order pairs for a window, bandwidth or fixed time window, with recommended settings and the envelope-rate limit), `OrderComponents`.
- `rota.speed.rpm_from_ridge`: tacholess speed from a whitened STFT, a configurable order comb and a linear-time dynamic-programming path, with confidence and search-boundary flags (including true speeds outside the search range).
- Validation cases T3, T4, T6, T8, T10, T12 and T13; the T13 table of extraction error versus speed-reference error is in `validation/README.md`.
- Unit tests for `speed`, `angle` and `maps`.

### Fixed

- `angular_resample`: high orders near `max_order` no longer droop (up to -2.8 dB before) and content near the Nyquist frequency no longer folds into the pass band; the signal is upsampled by a polyphase FIR before spline interpolation.
- `tacho_to_rpm`: a spurious pulse just before a genuine one no longer removes the genuine pulse.
- `regularize_control`: no longer modifies a 1-D input array when despiking.
- `rpm_frequency_map`: the Nyquist bin is scaled like the DC bin in RMS mode.

## [0.1.0] - 2026-10-25

Core chain for the analysis of a run-up with a tachometer. Development milestone; not released separately (first public release: 0.3.0).

### Added

- `rota.speed`: `tacho_to_rpm`, `rpm_to_phase`, `regularize_control`, `speed_to_rpm`, `fit_speed_to_rpm`, `steady_segments`.
- `rota.angle`: `angular_resample` (resampling to uniform shaft angle with polyphase upsampling and an angle-domain anti-aliasing filter).
- `rota.maps`: `rpm_frequency_map`, `rpm_order_map`.
- `rota.slices`: `order_spectrum`, `order_slice`.
- `rota.synth`: `rpm_profiles`, `synth_runup` (test signals with analytic ground truth and a JSON configuration sidecar).
- `rota.metrics`: `track_error`, `waveform_similarity`, `order_residual_energy`.
- Validation cases T1, T2, T5 and T14 (`validation/make_signals.py`, `tests/test_validation.py`) against the analytic ground truth.
- Package metadata, `README.md`, `CONTRIBUTING.md`, `CITATION.cff`, issue and pull request templates, and a GitHub Actions workflow that runs the tests on Python 3.10 to 3.13.
- Licence: GPL-3.0-or-later.

[Unreleased]: https://github.com/ZhenxianLi/ROTA/compare/v0.4.0...main
[0.4.0]: https://github.com/ZhenxianLi/ROTA/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ZhenxianLi/ROTA/releases/tag/v0.3.0
[0.2.0]: https://github.com/ZhenxianLi/ROTA/commit/767093c
[0.1.0]: https://github.com/ZhenxianLi/ROTA/commit/9c1404d
