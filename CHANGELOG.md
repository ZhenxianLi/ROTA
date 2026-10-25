# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project intends to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Version 0.2 (order extraction and tacholess speed) and version 0.3 (resynthesis) are in development.

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

[Unreleased]: https://github.com/ZhenxianLi/ROTA/compare/v0.1.0...main
[0.1.0]: https://github.com/ZhenxianLi/ROTA/releases/tag/v0.1.0
