"""Generate the v0.1 validation signals T1, T2, T5 and T14 (ROADMAP section 5.3).

Each case is one function that returns a :class:`rota.synth.SynthSignal`
built from explicit parameters and a fixed seed, so the signal and its
analytic truth are bit-identical on every run. The script can be run to write
the signals to disk, or imported (``tests/test_validation.py`` does this) to
build them in memory::

    python validation/make_signals.py            # writes validation/signals/
    python validation/make_signals.py T1 T14     # only these cases

Each signal is saved with :meth:`rota.synth.SynthSignal.save` as
``<name>.npz`` (arrays and truth) plus ``<name>.json`` (configuration, resolved
phases, seed, ROTA version). The output directory is git-ignored.

Units: time in s, frequency in Hz, rotational speed in rpm, amplitudes in Pa
(the signals are notionally sound pressure, but only the numbers matter).
Every amplitude given below is a *peak* amplitude, except ``noise_rms``. The
validation compares against ``SynthSignal.order_amplitude`` (the stored truth),
never against the parameter lists in this file.
"""

import argparse
from pathlib import Path

import numpy as np

from rota.synth import SynthSignal, rpm_profiles, synth_runup

__all__ = ["t1", "t2", "t5", "t14", "CASES", "main"]

# Default output directory: validation/signals next to this file.
SIGNALS_DIR = Path(__file__).resolve().parent / "signals"

# Case parameters shared with the tests (documented in each docstring).
T2_RESONANCE = {"freq": 200.0, "q": 8.0, "gain": 4.0}
T2_TONE = {"freq": 450.0, "amplitude": 0.30}
T5_SEEDS = {60: 5, 1: 4}  # chosen so that no end pulse is dropped (see t5)


def _time(duration, fs):
    return np.arange(int(round(duration * fs))) / fs


def t1():
    """T1: one order, constant amplitude, linear run-up.

    Examines the absolute accuracy of order amplitude against speed.

    * Sampling rate 16 kHz, duration 10 s.
    * Speed: linear, 1000 to 4000 rpm (300 rpm/s); the order is therefore
      at 33.3 to 133.3 Hz.
    * One order, number 2, peak amplitude 1 Pa (RMS 0.707 Pa), constant, random
      initial phase from the seed. No noise, no tones, no resonance.
    * Seed 1.

    Returns
    -------
    rota.synth.SynthSignal
        Mono signal with truth. Amplitudes are peak, in Pa.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis: Signal Analysis and
    Experimental Procedures, ch. 12 (run-up measurements). Wiley.
    """
    fs = 16000.0
    t = _time(10.0, fs)
    rpm = rpm_profiles(t, "linear", start=1000.0, stop=4000.0)
    return synth_runup(fs, rpm, [{"order": 2.0, "amplitude": 1.0}], seed=1)


def t2():
    """T2: several orders, a resonance, a fixed tone and broadband noise.

    Examines maps, slices and resonance handling.

    * Sampling rate 16 kHz, duration 12 s.
    * Speed: linear, 1000 to 5000 rpm (333.3 rpm/s, 5.56 rev/s^2).
    * Orders (peak amplitude before the resonance): 1 (0.50 Pa), 2 (1.00 Pa),
      3 (0.60 Pa, with a speed gain law: 0.3 at 1000 rpm, 1.0 at 2500 rpm,
      0.7 at 5000 rpm, linear in between) and 4.5 (0.40 Pa). Random initial
      phases from the seed.
    * One resonance at 200 Hz, Q = 8, peak gain 4 (linear). Order 4.5 sweeps
      through it near 2667 rpm (t = 5 s), order 3 near 4000 rpm (t = 9 s);
      order 2 ends 33 Hz below it (167 Hz) and order 1 stays far away.
      The truth is the quasi-static response (see
      :func:`rota.synth.synth_runup`), which is accurate when the sweep is slow
      compared with the ring-down time ``Q / (pi f0)`` = 12.7 ms. The fastest
      sweep is order 4.5 at 25 Hz/s: it needs 1 s to cross the 25 Hz
      half-power bandwidth, about 80 ring-down times.
    * One fixed tone at 450 Hz, 0.30 Pa peak before the resonance (the
      resonance changes it by +0.3 dB). Above order 4.5's top frequency
      (375 Hz), so no order crosses it.
    * White Gaussian noise, RMS 0.02 Pa before the resonance, shaped by it.
      The weakest order is about 30 dB above the noise inside its analysis
      band. The error that noise adds to a band-power order reading is the
      cross term between the line and the in-band noise; it is random. Over
      600 map frames and ten different seeds the worst error away from the
      resonance is 0.13 dB at these levels (0.21 dB with an order 4.5 of
      0.25 Pa, which is why that amplitude was not used).
    * Seed 2.

    Returns
    -------
    rota.synth.SynthSignal
        Mono signal with truth. Amplitudes are peak, in Pa (noise: RMS).

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis: Signal Analysis and
    Experimental Procedures, ch. 5-6 (second-order systems, resonances) and
    ch. 12 (orders). Wiley.
    """
    fs = 16000.0
    t = _time(12.0, fs)
    rpm = rpm_profiles(t, "linear", start=1000.0, stop=5000.0)
    orders = [
        {"order": 1.0, "amplitude": 0.50},
        {"order": 2.0, "amplitude": 1.00},
        {
            "order": 3.0,
            "amplitude": 0.60,
            "gain_rpm": [1000.0, 2500.0, 5000.0],
            "gain": [0.3, 1.0, 0.7],
        },
        {"order": 4.5, "amplitude": 0.40},
    ]
    return synth_runup(
        fs,
        rpm,
        orders,
        tones=[dict(T2_TONE)],
        resonances=[dict(T2_RESONANCE)],
        noise_rms=0.02,
        seed=2,
    )


def t5(ppr=60, seed=None):
    """T5: tachometer with several pulses per revolution, jitter and drop-outs.

    Examines the robustness of :func:`rota.speed.tacho_to_rpm`.

    * Tachometer sampled at 48 kHz, duration 15 s.
    * Speed: idle at 800 rpm for 2 s, raised-cosine ("s_curve", zero slope at
      both ends) run-up to 5000 rpm between 2 s and 10 s (peak slope
      825 rpm/s), then held at 5000 rpm for 5 s.
    * ``ppr`` pulses per revolution (default 60; ``t5(ppr=1)`` is the
      once-per-revolution variant), Gaussian timing jitter of standard
      deviation 5 us on each pulse, each pulse dropped with probability 0.5 %.
      At 60 ppr and 5000 rpm a pulse period is 200 us (9.6 samples).
    * The mix ``x`` holds a single order 2 of 1 Pa peak, so the signal is not
      degenerate; the case is about the tachometer channel.
    * Seed: ``T5_SEEDS[ppr]`` (60 -> 5, 1 -> 4) unless ``seed`` is given. The
      defaults are chosen so that neither the first nor the last pulse is
      dropped: a dropped end pulse leaves no gap in the pulse train and cannot
      be counted as missing by any detector.

    Parameters
    ----------
    ppr : int, default 60
        Tachometer pulses per revolution.
    seed : int, optional
        Overrides the default seed.

    Returns
    -------
    rota.synth.SynthSignal
        ``tacho`` holds the pulse signal, ``pulse_times``/``pulse_revs`` the
        truth of the pulses that were rendered, ``rpm`` the true speed in
        rpm, ``config["tacho_n_dropped"]`` the number of dropped pulses.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis: Signal Analysis and
    Experimental Procedures, sec. 12.2 (speed from tachometer signals). Wiley.
    """
    fs = 48000.0
    t = _time(15.0, fs)
    rpm = rpm_profiles(t, "s_curve", start=800.0, stop=5000.0, t_start=2.0, t_stop=10.0)
    if seed is None:
        seed = T5_SEEDS.get(int(ppr), 5)
    return synth_runup(
        fs,
        rpm,
        [{"order": 2.0, "amplitude": 1.0}],
        tacho_ppr=int(ppr),
        tacho_jitter=5e-6,
        tacho_drop=0.005,
        seed=seed,
    )


def t14(seed=14):
    """T14: start from idle, orders half an order apart.

    Examines order resolution at low speed: the angle-domain map keeps its
    resolution in orders at every speed, an equal-bandwidth time-domain map
    does not (ROADMAP section 6, item 20).

    * Sampling rate 16 kHz, duration 12 s.
    * Speed: idle at 800 rpm for 2 s, then a linear run-up to 3000 rpm between
      2 s and 11 s (244 rpm/s), held for the last second. At idle one
      revolution takes 75 ms and orders 2 and 2.5 are 6.7 Hz apart.
    * Orders: 1 (1.5 Pa peak), 2 (1.0 Pa) and 2.5 (1.0 Pa); constant
      amplitudes, random initial phases from the seed. Orders 2 and 2.5 have
      equal amplitude, so a map that cannot separate them reads about +3 dB
      on each. No noise.
    * Seed 14 (``seed`` can be changed to vary the relative phases).

    Parameters
    ----------
    seed : int, default 14
        Seed of the initial phases.

    Returns
    -------
    rota.synth.SynthSignal
        Mono signal with truth. Amplitudes are peak, in Pa.

    References
    ----------
    Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order
    tracking. Mechanical Systems and Signal Processing, 11(2), 187-205.
    Brandt, A. (2011). Noise and Vibration Analysis: Signal Analysis and
    Experimental Procedures, ch. 12. Wiley.
    """
    fs = 16000.0
    t = _time(12.0, fs)
    rpm = rpm_profiles(t, "linear", start=800.0, stop=3000.0, t_start=2.0, t_stop=11.0)
    orders = [
        {"order": 1.0, "amplitude": 1.5},
        {"order": 2.0, "amplitude": 1.0},
        {"order": 2.5, "amplitude": 1.0},
    ]
    return synth_runup(fs, rpm, orders, seed=seed)


# Case name -> builder. T5 is written twice (60 ppr and once per revolution).
CASES = {
    "T1": t1,
    "T2": t2,
    "T5": t5,
    "T5_ppr1": lambda: t5(ppr=1),
    "T14": t14,
}


def main(outdir=None, names=None):
    """Build the cases and save each one as ``<outdir>/<name>.npz`` and ``.json``.

    Parameters
    ----------
    outdir : path-like, optional
        Output directory (created if needed). Default: ``validation/signals``
        next to this file.
    names : sequence of str, optional
        Cases to write, from ``CASES``. Default: all.

    Returns
    -------
    dict
        ``{name: (npz_path, json_path)}`` of the files written.

    References
    ----------
    ROTA ROADMAP, section 5.5 (directory and workflow).
    """
    outdir = SIGNALS_DIR if outdir is None else Path(outdir)
    names = list(CASES) if not names else list(names)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        raise KeyError(f"unknown case(s) {unknown}; available: {list(CASES)}")
    outdir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in names:
        sig = CASES[name]()
        assert isinstance(sig, SynthSignal)
        written[name] = sig.save(outdir / name)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases", nargs="*", help=f"cases to write (default: all of {list(CASES)})")
    ap.add_argument("--out", default=None, help="output directory (default: validation/signals)")
    args = ap.parse_args()
    bad = [c for c in args.cases if c not in CASES]
    if bad:
        ap.error(f"unknown case(s) {bad}; available: {list(CASES)}")
    for case, (npz, js) in main(args.out, args.cases).items():
        print(f"{case}: {npz.name}, {js.name}")
