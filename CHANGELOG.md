# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project intends to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once released.

## [Unreleased]

Version 0.1 (core chain for a run-up with a tachometer) is in development.

### Added

- `rota.speed`: `tacho_to_rpm`, `rpm_to_phase`, `regularize_control`, `speed_to_rpm`, `fit_speed_to_rpm`, `steady_segments`.
- `rota.angle`: `angular_resample` (resampling to uniform shaft angle with an angle-domain anti-aliasing filter).
- `rota.maps`: `rpm_frequency_map`, `rpm_order_map`.
- `rota.slices`: `order_spectrum`, `order_slice`.
- `rota.synth`: `rpm_profiles`, `synth_runup` (test signals with analytic ground truth).
- `rota.metrics`: `track_error`.
- Package metadata (`pyproject.toml`) and the design document `ROADMAP.md`.
- `README.md`, `CONTRIBUTING.md`, a pull request template, and a GitHub Actions workflow that runs the tests on Python 3.10 to 3.13.

[Unreleased]: https://github.com/zhenxianli/rota/commits/main
