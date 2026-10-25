# ROTA

**R**otational **O**rder **T**racking & **A**nalysis. *Rota* is Latin for "wheel".

ROTA is a lightweight, open-source Python library for order analysis of noise and vibration signals from rotating machinery (engines, electric motors, gearboxes), with tools for order-based sound resynthesis. It depends only on NumPy and SciPy. The project rule is that each function is implemented independently from the published literature and tested on synthetic signals with known analytic ground truth.

The design document is [ROADMAP.md](ROADMAP.md) (written in Chinese).

## Status

Pre-alpha. Version 0.1 is in development; functions and signatures may change before it is released. Not every function listed below exists yet. The package is not published on PyPI (a PyPI release is planned for v1.0).

| Module | Contents | Status |
|---|---|---|
| `rota.speed` | v0.1: tachometer pulses to rpm, rpm to shaft angle, control-signal cleanup and resampling, vehicle speed to rpm, steady-segment selection. Later: tacholess rpm estimation (v0.2), virtual rpm (v0.3), signal alignment (v0.4) | v0.1 (in progress) |
| `rota.angle` | v0.1: resampling from uniform time to uniform shaft angle (computed order tracking). Later: inverse mapping and time warping (v0.3) | v0.1 (in progress) |
| `rota.maps` | v0.1: rpm-frequency (Campbell) map, rpm-order map. Later: binning options (v0.4) | v0.1 (in progress) |
| `rota.slices` | v0.1: order spectrum, order slice. Later: frequency slice, overall level, metric versus rpm, order share (v0.4) | v0.1 (in progress) |
| `rota.synth` | v0.1: speed profiles and run-up signals with analytic ground truth. Later: tone-plus-noise stimuli (v0.3) | v0.1 (in progress) |
| `rota.metrics` | v0.1: speed-track error. Later: waveform similarity and residual order energy (v0.2), window and envelope checks (v0.3) | v0.1 (in progress) |
| `rota.extract` | Order extraction (Vold-Kalman filter, phase-locked tracked harmonics), order removal, residual, resolvability check | v0.2 (planned) |
| `rota.resynth` | Order tables, order-based synthesis, broadband synthesis, speed remapping, resynthesis | v0.3 (planned) |
| `rota.tonal` | Thin bridge to open-source psychoacoustic tools: calibrated level, resampling, order prominence | v0.4 (planned) |
| `rota.plot` | Plotting helpers, kept separate from computation | v0.4 (planned) |

Planned for v1.0: full validation report, English documentation, PyPI release, Zenodo DOI.

Out of scope: real-time audio I/O, GUI, rotor-dynamics analysis (orbits, Bode plots), spatial audio, reading proprietary file formats, and the psychoacoustic metrics themselves (loudness, sharpness, roughness, tonality).

## Installation

Python 3.10 or later, NumPy 1.23 or later, SciPy 1.9 or later. Install from source:

```
git clone https://github.com/zhenxianli/rota.git
cd rota
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
5. Interoperable: results can be passed on to open-source psychoacoustic tools such as SQAT and MoSQITo. ROTA does not depend on them.
6. Multichannel first: operators are linear per channel (see Conventions).
7. Reproducible: seedable randomness, and planned JSON sidecar files recording the parameters of each resynthesis.
8. No hidden constants.

## Validation

Analytic ground truth from `rota.synth` is the first reference: every test signal comes with its true speed, shaft angle, and amplitude and phase of each order. Where a public implementation elsewhere exists, a black-box comparison with explicit parameters is a second reference, and it compares physical quantities (speed curves, order amplitudes along a ridge), not map cells. When the two disagree, the analytic truth decides. Resynthesis is tested by round trip: synthesise, analyse, compare with the input table. Continuous integration runs pytest only.

The design document defines 14 validation signals (T1 to T14). Version 0.1 covers T1 (single order, constant amplitude, linear run-up), T2 (several orders, a fixed-frequency resonance, noise), T5 (multi-pulse tachometer with jitter and dropped pulses) and T14 (low-speed start from idle with orders half an order apart).

Initial tolerances, to be adjusted after measurement:

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

Only the rows for speed and order amplitude concern the v0.1 modules; the others apply as the corresponding modules are added.

## Independence statement

ROTA is an independent academic project. It is not affiliated with, endorsed by, or supported by MathWorks or any commercial software vendor. MATLAB is a registered trademark of The MathWorks, Inc.

All algorithms are implemented from the published literature cited in each docstring, and ROTA reads no proprietary file formats.

## License

To be decided before the v0.1 release.

## Citation

A citation file will be added with the first release.

## Acknowledgements

ROTA is designed, implemented and maintained by Zhenxian Li. Claude (Anthropic) was used as an assistant for literature review, code review and documentation; all algorithmic choices, implementation and validation are the author's responsibility.
