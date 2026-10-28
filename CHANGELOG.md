# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project intends to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-10-28

Figures of ROTA results.

### Added

- `rota.plot`: standard figures of ROTA results on Matplotlib (optional extra `plot`; `import rota` does not import Matplotlib, and a call without it raises `ImportError` with the install command). Fifteen functions, each named like the function whose result it draws: `rpm_frequency_map`, `rpm_order_map`, `compare_maps`, `order_slice`, `frequency_slice`, `order_spectrum`, `overall_level`, `metric_vs_rpm`, `time_metric`, `rpm_vs_time`, `rpm_from_ridge`, `order_components`, `order_table`, `order_prominence`, `band_levels_vs_rpm`. Each takes `ax=` and returns the Axes it drew on (except `compare_maps`, which builds its own grid and returns `(fig, axes)`), never shows the figure and never changes `rcParams` or the style. Labels carry the unit, the dB reference and the amplitude convention, from an explicit `ref`/`unit` (the constant `SOUND_PRESSURE` for sound pressure in Pa) or the reference a result stores. Line plots take one result, a list or a dict and plot differences with `relative_to=`; `compare_maps` shows maps on one colour scale, with their dB difference. `orders=` selects orders, `mark_orders=` draws order lines. Maps are drawn against rpm when the speed is essentially monotonic (reversals up to `monotonic_tol` of the range, 2 % by default) and against time otherwise; `channel=` selects the channel of multichannel results.
- Gallery `examples/11_plot_gallery.py` and page `docs/plotting.md`, one figure per function with the call that made it.

### Changed

- Examples 01 and 02 draw their figures with `rota.plot`; their printed numbers are unchanged.
- Continuous integration installs the `plot` extra in the main test matrix; the oldest-dependency job runs without Matplotlib.

## [0.6.0] - 2026-10-28

Optional bridges to psychoacoustic toolboxes.

### Added

- `rota.bridges`: optional thin wrappers for open-source psychoacoustic toolboxes, imported lazily inside the functions (never vendored, never required; `ImportError` with the install command when missing). A bridge module is named exactly like the toolbox it wraps. `rota.bridges.mosqito`: ISO 532-1:2017 time-varying loudness, DIN 45692:2009 sharpness (optionally from a loudness result) and Daniel and Weber roughness. `rota.bridges.sottek_hearing_model`: ECMA-418-2:2025 (4th edition) tonality, loudness (optionally from a tonality result) and roughness of the Sottek hearing model, with the Matplotlib `rcParams` it changes on import restored. The functions return a `TimeMetric` result container whose method `vs_rpm(rpm, t_rpm, **kwargs)` bins the result along rpm with `metric_vs_rpm`, so that the quantity and the toolbox are visible in one line (`bridges.mosqito.loudness(x, fs).vs_rpm(rpm, t_rpm)`): it uses the recommended start-up exclusion, aggregation and threshold of the metric (keyword arguments override them), holds the last speed for frames that lie after the last speed sample by no more than the speed-sample interval plus the frame step of the metric (the Sottek hearing model puts its last tonality and loudness frame at the signal length rounded up to a multiple of 256 samples at 48 kHz, up to 5.33 ms after the last sample), warns about the frames it cannot assign a speed (outside the speed record, or where the speed is NaN; only frames that would otherwise be used are counted), says so in the `ValueError` when every frame after the start-up exclusion is at or below the threshold (`threshold=None` means none), and fills `name`, `unit` and `source` of the binned result from the metric. For full control, call `rota.slices.metric_vs_rpm(m.values, m.t, rpm, t_rpm, ...)` yourself.
- `rota.bridges.sqat`: optional wrappers that call SQAT v2.0 (loudness, sharpness, roughness, tonality) in a local, licensed MATLAB through the MATLAB Engine API for Python; tested with a stub engine, the tests against a real MATLAB are skipped unless `matlab.engine` and `ROTA_SQAT_PATH` are available (hosted runners cannot license an engine started from Python; the `sqat-engine` workflow is a manual probe).
- Optional dependency extras `mosqito` and `sottek-hearing-model`, named like the toolbox (no extra for SQAT: `matlabengine` must match the user's MATLAB release).
- Each wrapper states the licence of its toolbox and the citation its authors ask for; the README lists them.

### Changed

- `rota.slices.MetricVsRpm` gains the optional fields `name`, `unit` and `source` (default: empty strings; filled by `TimeMetric.vs_rpm` from the metric, left empty by `metric_vs_rpm`). Backward compatible: construction by keyword and attribute access are unchanged.
- The repository is renamed `rota-order` (github.com/ZhenxianLi/rota-order), the same name as the PyPI package; links are updated and the old address redirects. The import name stays `rota`.
- Examples 07, 08 and 10 use `rota.bridges`: MoSQITo and the Sottek hearing model are called through `rota.bridges.mosqito` and `rota.bridges.sottek_hearing_model`, and the results are binned with `TimeMetric.vs_rpm`, which replaces the resampling to 48 kHz, start-up exclusions, aggregations, tonality threshold, last-frame speed hold, silencing of MoSQITo and restoring of Matplotlib's settings that the examples did by hand; example 07 still calls MoSQITo's `tnr_ecma_st` directly (not wrapped: ROTA has its own tone-to-noise ratio). The printed numbers, figures and audio files of the examples are unchanged. The install notes of the examples use the extras `mosqito` and `sottek-hearing-model`, and `examples/09_sqat_bridge/README.md` shows `rota.bridges.sqat` with a licensed local MATLAB (CI keeps `run_sqat.m`, since a MATLAB engine started from Python is not licensed on hosted runners).
- Roadmap: `rota.plot` moves to v0.7.
- Releases are published by continuous integration: a commit `Release X.Y.Z` on `main` runs the tests, uploads to PyPI, tags `vX.Y.Z` and creates the GitHub release with the CHANGELOG section as notes (`.github/workflows/publish.yml`; see CONTRIBUTING.md).

## [0.5.0] - 2026-10-27

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

[Unreleased]: https://github.com/ZhenxianLi/rota-order/compare/v0.7.0...main
[0.7.0]: https://github.com/ZhenxianLi/rota-order/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/ZhenxianLi/rota-order/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/ZhenxianLi/rota-order/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ZhenxianLi/rota-order/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ZhenxianLi/rota-order/releases/tag/v0.3.0
[0.2.0]: https://github.com/ZhenxianLi/rota-order/commit/767093c
[0.1.0]: https://github.com/ZhenxianLi/rota-order/commit/9c1404d
