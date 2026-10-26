# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project intends to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Version 0.3 (resynthesis) is in development.

## [0.2.0] - 2026-10-26

Order extraction and tacholess speed estimation.

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

Core chain for the analysis of a run-up with a tachometer.

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

[Unreleased]: https://github.com/ZhenxianLi/ROTA/compare/v0.2.0...main
[0.2.0]: https://github.com/ZhenxianLi/ROTA/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ZhenxianLi/ROTA/releases/tag/v0.1.0
