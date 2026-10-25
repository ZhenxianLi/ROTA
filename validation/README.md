# Validation cases (v0.1)

Four of the validation signals defined in `ROADMAP.md` section 5.3 are implemented for v0.1: T1, T2, T5 and T14. Each is a function in `make_signals.py` that returns a `rota.synth.SynthSignal` built from explicit, documented parameters and a fixed seed, so the signal and its analytic truth are bit-identical on every run. The parameters are documented in each function's docstring.

| Case | Signal | What it examines |
|---|---|---|
| T1 | One order (2), 1 Pa peak, constant; linear run-up 1000 to 4000 rpm in 10 s | Absolute accuracy of order amplitude against speed |
| T2 | Orders 1, 2, 3 (with a speed gain law) and 4.5; a 200 Hz resonance (Q 8, gain 4) that orders 4.5 and 3 sweep through; a fixed 450 Hz tone; white noise (0.02 Pa RMS); run-up 1000 to 5000 rpm in 12 s | Maps, slices, resonance handling, noise |
| T5 | Speed with idle, an S-curve run-up and a hold (800 to 5000 rpm in 15 s); tachometer with 60 pulses per revolution (`T5`) or one (`T5_ppr1`), 5 us timing jitter, 0.5 % dropped pulses; 48 kHz | `rota.speed.tacho_to_rpm` |
| T14 | Idle at 800 rpm for 2 s, then run-up to 3000 rpm; orders 1, 2 and 2.5 (2 and 2.5 equal in amplitude) | Order resolution at low speed: angle-domain map against an equal-bandwidth time-domain map |

## Regenerating the signals

```
python validation/make_signals.py            # all cases, to validation/signals/
python validation/make_signals.py T1 T14     # selected cases
python validation/make_signals.py --out DIR  # another output directory
```

Each case is written as `<name>.npz` (signal, speed, shaft angle, and the truth: amplitude, phase and frequency of every order, the component stems, tachometer pulse times) and `<name>.json` (parameters, resolved random phases, seed, ROTA version). About 200 MB in total, mostly the two 48 kHz T5 files. `validation/signals/` is git-ignored. The tests do not need these files: they build the signals in memory from the same functions.

## Tests and tolerances

`tests/test_validation.py` runs the cases with pytest (a few seconds) and checks them against the truth stored in the signal. Expected amplitudes are always read from `order_amplitude` (or the stems), never from the parameter lists, because a resonance or a gain law changes the amplitude actually present in the signal (ROADMAP section 6, item 19). The expected reading of a map frame is the window-weighted mean of that truth over the frame.

The tolerances are the initial values of ROADMAP section 5.4, column "against analytic truth":

| Quantity | Tolerance | Where |
|---|---|---|
| Speed, median relative error | 0.1 % | T5, both pulse rates |
| Order amplitude, stationary segments (and, in these cases, away from the resonance) | 0.2 dB | T1 (order map and frequency map), T2, T14 |
| Order amplitude near a resonance | 1 dB | T2: frames in which the order is within 2 f0/Q of the resonance frequency f0 |

Further checks that belong to the cases but have no row in section 5.4: the number of missing pulses reported by `tacho_to_rpm` equals the number dropped (T5); the fixed tone is the largest line near its frequency in every frame of the frequency map (T2); the average order spectrum peaks at order 2 (T1); and with the same window duration as the angle-domain map has at top speed, a time-domain map cannot separate orders 2 and 2.5 at idle, reading them too high by more than 1 dB (T14, ROADMAP section 6, item 20).

## Not in v0.1

The black-box reference comparison (ROADMAP section 5.5, steps 2 and 3: running an external implementation on these signals, storing its output in `reference/`, and the `compare.py` report) is not part of v0.1. Only the comparison with analytic truth exists, and it is what continuous integration runs.
