# ROTA

[![Tests](https://github.com/ZhenxianLi/ROTA/actions/workflows/tests.yml/badge.svg)](https://github.com/ZhenxianLi/ROTA/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/rota-order.svg)](https://pypi.org/project/rota-order/)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](https://github.com/ZhenxianLi/ROTA/blob/main/LICENSE)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://github.com/ZhenxianLi/ROTA/blob/main/pyproject.toml)
[![Dependencies: NumPy + SciPy](https://img.shields.io/badge/dependencies-NumPy%20%2B%20SciPy-informational.svg)](https://github.com/ZhenxianLi/ROTA/blob/main/pyproject.toml)

**R**otational **O**rder **T**racking & **A**nalysis. *Rota* is Latin for "wheel".

ROTA is a lightweight, open-source Python library for order analysis of noise and vibration signals from rotating machinery (engines, electric motors, gearboxes), with tools for order-based sound resynthesis. It depends only on NumPy and SciPy. The project rule is that each function is implemented independently from the published literature and tested on synthetic signals with known analytic ground truth.

The design document is [ROADMAP.md](https://github.com/ZhenxianLi/ROTA/blob/main/ROADMAP.md).

## Status

Pre-alpha. The current release is 0.4.0; functions and signatures may still change before 1.0.

| Module | Contents | Since |
|---|---|---|
| `rota.speed` | Tachometer pulses to rpm, rpm to shaft angle, control-signal cleanup and resampling, vehicle speed to rpm, steady-segment selection; tacholess speed estimation (order comb + dynamic programming, with confidence and search-boundary flags); virtual rpm; alignment of signals by GCC-PHAT delay and onset detection | 0.1–0.5 |
| `rota.angle` | Resampling from uniform time to uniform shaft angle (computed order tracking) with angle-domain anti-aliasing; the inverse mapping; time warping to a reference speed (stationarize) | 0.1, 0.3 |
| `rota.maps` | Rpm-frequency (Campbell) map, rpm-order map; binning of frames into rpm bins (per segment, minimum count, phase-coherent option) | 0.1, 0.5 |
| `rota.slices` | Order spectrum, order slice; frequency slice, overall level, any metric binned along rpm, attribution of frequency tracks to orders, order share | 0.1, 0.5 |
| `rota.extract` | Order extraction (second-generation Vold-Kalman filter, joint over orders and shafts; phase-locked tracked harmonics), order removal, residual, resolvability check | 0.2 |
| `rota.resynth` | Order tables (explicit peak/RMS), additive order synthesis offline and in blocks, broadband synthesis without frame-rate pumping, order editing, speed remapping, order-plus-residual recombination, steady listening excerpts, provenance sidecars | 0.3 |
| `rota.synth` | Speed profiles and run-up signals with analytic ground truth; tone-plus-noise listening stimuli; realistic scenarios (engine run-up, electric drive, cruise) with transfer paths, background noise and recorder impairments | 0.1, 0.3, 0.4 |
| `rota.metrics` | Speed-track error, waveform similarity (energy-weighted), residual order energy, window and pumping checks, envelope statistics | 0.1–0.3 |
| `rota.tonal` | Bridge to open-source psychoacoustic tools: A/C weighting (IEC 61672-1), calibrated levels and resampling to 48 kHz, octave and 1/3-octave bands (IEC 61260-1) and band levels along rpm, tone-to-noise ratio and prominence ratio of orders (ECMA-418-1, 3rd edition) | 0.5 (unreleased) |
| `rota.plot` | Plotting helpers, kept separate from computation | v0.6 (planned) |

Planned for v1.0: full validation report, complete documentation, Zenodo DOI.

Out of scope: real-time audio I/O, GUI, rotor-dynamics analysis (orbits, Bode plots), spatial audio, reading proprietary file formats, and the psychoacoustic metrics themselves (loudness, sharpness, roughness, tonality).

## Installation

Python 3.10 or later, NumPy 1.23 or later, SciPy 1.9 or later. From PyPI:

```
pip install rota-order
```

The distribution is called `rota-order` because the name `rota` is taken on PyPI; the import name is `rota` (`import rota`).

From source:

```
git clone https://github.com/ZhenxianLi/ROTA.git
cd ROTA
pip install -e .
```

To run the tests:

```
pip install -e ".[test]"
pytest
```

Matplotlib is an optional extra for `rota.plot` and the examples: `pip install -e ".[plot]"`.

## Quick start

A synthetic run-up from 1000 to 4000 rpm containing a single order, analysed with an order-rpm map. Because the signal comes from `rota.synth`, the result can be compared with the known amplitude.

```python
import numpy as np
import rota

fs = 16000.0                              # sampling rate, Hz
t = np.arange(int(10 * fs)) / fs          # time, s

# Linear speed ramp in rpm, and a signal with one order (order 2, amplitude 1.0)
rpm = rota.synth.rpm_profiles(t, "linear", start=1000, stop=4000)
sig = rota.synth.synth_runup(fs, rpm, orders=[{"order": 2, "amplitude": 1.0}], seed=0)
# sig.x: signal, sig.rpm: speed in rpm

# Order-rpm map with 0.25 order resolution, amplitudes as RMS
m = rota.maps.rpm_order_map(sig.x, fs, sig.rpm, order_resolution=0.25, amplitude="rms")

# Amplitude of order 2 as a function of speed
s = rota.slices.order_slice(m, 2)
# s.rpm: speed in rpm, s.amplitude: order amplitude (RMS, as selected for the map)
```

The amplitude convention of every function is stated in its docstring.

## Examples

Six scripts in [examples/](https://github.com/ZhenxianLi/ROTA/tree/main/examples) cover order maps and slices, tachometer and tacholess speed, order extraction and removal, order-table resynthesis, and two complete synthetic recordings that you can listen to, an engine run-up and an electric drive. Each builds its signal with `rota.synth` and prints its results against the known truth.

![Campbell diagram, order map, engine speed and firing-order level of a synthetic engine run-up](https://raw.githubusercontent.com/ZhenxianLi/ROTA/main/docs/images/realistic_runup.png)

*A synthetic four-cylinder engine run-up from `rota.synth.synth_scenario("ice_runup")`: launch from idle with clutch slip, two upshifts, combustion bursts, a modal transfer path, engine broadband, road noise and rumble, two receiver channels. Campbell diagram and order map, the speed from the tachometer against the truth, and the level of the firing order against the truth at the receiver, before and after `remove_orders` ([examples/05_realistic_engine_audio.py](https://github.com/ZhenxianLi/ROTA/blob/main/examples/05_realistic_engine_audio.py)).*

### Listen

The recordings below are synthetic (no measured data) and are written by [examples/05_realistic_engine_audio.py](https://github.com/ZhenxianLi/ROTA/blob/main/examples/05_realistic_engine_audio.py) and [examples/06_electric_drive_audio.py](https://github.com/ZhenxianLi/ROTA/blob/main/examples/06_electric_drive_audio.py): 16-bit WAV at 32 kHz, all with the same scaling (4 Pa is digital full scale), so levels can be compared between files.

- [engine_runup.wav](https://github.com/ZhenxianLi/ROTA/blob/main/docs/audio/engine_runup.wav): the engine run-up as recorded (two channels).
- [engine_runup_orders_removed.wav](https://github.com/ZhenxianLi/ROTA/blob/main/docs/audio/engine_runup_orders_removed.wav): the residual after `remove_orders`: engine broadband, road noise, rumble and what is not periodic in the combustion bursts.
- [engine_runup_orders_only.wav](https://github.com/ZhenxianLi/ROTA/blob/main/docs/audio/engine_runup_orders_only.wav): the extracted engine orders.
- [engine_runup_faster.wav](https://github.com/ZhenxianLi/ROTA/blob/main/docs/audio/engine_runup_faster.wav): the same recording with the apparent engine speed raised by 15 % (`remap_rpm`, then `resynthesize` with the residual kept).
- [electric_runup.wav](https://github.com/ZhenxianLi/ROTA/blob/main/docs/audio/electric_runup.wav): an electric drive run-up: motor orders and gear whine, a dithered 10 kHz inverter carrier with sidebands, two fixed tones, road noise and rumble. The tonal lines are weak against the broadband.
- [electric_runup_motor_orders_removed.wav](https://github.com/ZhenxianLi/ROTA/blob/main/docs/audio/electric_runup_motor_orders_removed.wav): after removing the motor and gear orders; the inverter sidebands are not orders and remain.
- [electric_runup_whine_only.wav](https://github.com/ZhenxianLi/ROTA/blob/main/docs/audio/electric_runup_whine_only.wav): the extracted motor and gear orders.

## Tacholess speed

`rota.speed.rpm_from_ridge` estimates the speed from the signal alone with a comb of orders and a dynamic-programming ridge. Give it the orders the machine shows: the comb, not the data, decides between a speed and its octaves. For a four-stroke engine with N cylinders the firing order N/2 and its harmonics dominate, so use `orders=(2, 4, 6)` for four cylinders (the default `(1, 2, 3, 4)` then reads twice the crankshaft speed), `(1.5, 3, 4.5)` for three and `(3, 6, 9)` for six. Set `rpm_range` from a priori knowledge (idle speed to redline). The window adapts to the sweep by default (`window="auto"`: 0.4 s, shorter for fast run-ups), and several channels are combined by averaging their power spectra. Each result carries a per-frame `confidence` (near 1 on a clean correct track, near 0 in noise) and a `quality` summary (median confidence, fraction of frames at a search bound, fraction of jumping frames, octave scores); a `RidgeQualityWarning` is issued when the track as a whole is likely wrong.

## Conventions

- **Units.** Time in s, sampling rate and frequency in Hz, rotational speed in rev/min (rpm), shaft angle in revolutions, order in cycles per revolution, vehicle speed in km/h.
- **Arrays.** Signals have shape `(n,)` or `(n, ch)`. Every operator acts on each channel independently and linearly, so amplitude and phase relations between channels are preserved.
- **Amplitude.** Peak or RMS is always stated in the docstring and, where there is a choice, selected by an explicit `amplitude` argument (`"peak"` or `"rms"`).
- **No hidden constants.** Gains, calibration factors and reference pressures are passed explicitly. They are never built into a function.
- **Randomness.** Every random process takes a `numpy.random.Generator` or an integer seed.
- **Estimators.** Functions that estimate quantities (speed, order amplitude) are meant to return a confidence and to report when a search boundary was reached.

## Design principles

1. Independent implementation: algorithms follow public literature only (see Independence statement).
2. Verifiable: each function is tested on synthetic signals with known analytic ground truth.
3. Transparent: docstrings give the literature, the amplitude definition and the units.
4. Plain Python: snake_case functions, NumPy arrays as input and output, computation separate from plotting.
5. Interoperable: results can be passed on to open-source psychoacoustic tools; examples 07 to 09 link ROTA to [MoSQITo](https://github.com/Eomys/MoSQITo), [sottek-hearing-model](https://github.com/mlotinga/sottek-hearing-model) and [SQAT](https://github.com/ggrecow/SQAT). ROTA does not depend on them. Open-source implementations of psychoacoustic metrics, with the standard and edition each follows, are listed in [LISQM](https://zhenxianli.github.io/LISQM/).
6. Multichannel first: operators are linear per channel (see Conventions).
7. Reproducible: seedable randomness, and planned JSON sidecar files recording the parameters of each resynthesis.
8. No hidden constants.

## Validation

Analytic ground truth from `rota.synth` is the first reference: every test signal comes with its true speed, shaft angle, and amplitude and phase of each order. A black-box comparison with MATLAB's documented public functions, all parameters explicit, is the second reference: it runs in CI (workflow `matlab-reference`, see [validation/README.md](https://github.com/ZhenxianLi/ROTA/blob/main/validation/README.md)) and compares physical quantities (speed curves, order amplitudes along a ridge, Vold-Kalman waveforms and bandwidth), not map cells; when the two disagree, the analytic truth decides. Its output stays in the workflow artifact and is not published in the repository until the licence terms are confirmed. Resynthesis is tested by round trip: synthesise, analyse, compare with the input table. Continuous integration runs pytest only.

The design document defines 14 validation signals (T1 to T14). Version 0.1 covers T1 (single order, constant amplitude, linear run-up), T2 (several orders, a fixed-frequency resonance, noise), T5 (multi-pulse tachometer with jitter and dropped pulses) and T14 (low-speed start from idle with orders half an order apart). Version 0.2 adds T3 (exponential run-down), T4 (crossing orders on two shafts), T6 (fast run-up), T8 (twin shafts 0.5–4 % apart), T10 (true speed outside the search range), T12 (order removal) and T13 (extraction error versus speed-reference error). Version 0.3 adds T9 (order table → synthesis → analysis → table round trip, `tests/test_ordertable.py`) and T11 (broadband synthesis: PSD accuracy and no frame-rate pumping, `tests/test_broadband.py`). Version 0.4 adds stress tests on realistic scenarios ([validation/stress.md](https://github.com/ZhenxianLi/ROTA/blob/main/validation/stress.md)). Cases and measured results: [validation/README.md](https://github.com/ZhenxianLi/ROTA/blob/main/validation/README.md).

Tolerances (the analytic column is enforced by the test suite; the black-box column is the target of the CI comparison):

| Quantity | Against analytic truth | Against black-box reference |
|---|---|---|
| Speed, median relative error | <= 0.1 % | <= 0.1 % |
| Tacholess speed, fraction of frames within 3 % (steady or slowly rising) | >= 90 % | within 5 percentage points |
| Order amplitude, stationary segments | <= 0.2 dB | <= 0.5 dB |
| Order amplitude near a resonance | <= 1 dB | <= 1 dB |
| Order waveform correlation (edges removed) | >= 0.99 | >= 0.99 |
| Vold-Kalman -3 dB bandwidth | within 5 % of the set value | within 5 % |
| Order-table round trip, amplitude (T9) | <= 0.2 dB | not compared |
| Remaining energy of a removed order in the residual (T12) | <= -20 dB | not compared |
| Frame-rate pumping of broadband synthesis (T11) | <= 1 dB | not compared |

All rows of the analytic column are met by the current release; measured values are in [validation/README.md](https://github.com/ZhenxianLi/ROTA/blob/main/validation/README.md).

## Independence statement

ROTA is an independent academic project. It is not affiliated with, endorsed by, or supported by MathWorks or any commercial software vendor. MATLAB is a registered trademark of The MathWorks, Inc.

All algorithms are implemented from the published literature cited in each docstring, and ROTA reads no proprietary file formats.

## License

ROTA is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [LICENSE](https://github.com/ZhenxianLi/ROTA/blob/main/LICENSE).

## Citation

See [CITATION.cff](https://github.com/ZhenxianLi/ROTA/blob/main/CITATION.cff) (GitHub shows it as "Cite this repository"). A Zenodo DOI is planned for v1.0.

## Acknowledgements

ROTA is designed, implemented and maintained by Zhenxian Li. Claude (Anthropic) was used as an assistant during development.
