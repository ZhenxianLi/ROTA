"""Generate the validation signals T1-T14 (ROADMAP section 5.3) built so far.

v0.1: T1, T2, T5, T14. v0.2: T3, T4, T6, T8, T10, T12, T13.

Each case is one function that returns a :class:`rota_order.synth.SynthSignal`
built from explicit parameters and a fixed seed, so the signal and its
analytic truth are bit-identical on every run. The script can be run to write
the signals to disk, or imported (``tests/test_validation.py`` and
``tests/test_validation_v02.py`` do this) to build them in memory::

    python validation/make_signals.py            # writes validation/signals/
    python validation/make_signals.py T1 T14     # only these cases

Each signal is saved with :meth:`rota_order.synth.SynthSignal.save` as
``<name>.npz`` (arrays and truth) plus ``<name>.json`` (configuration, resolved
phases, seed, rota-order version). The output directory is git-ignored. T13 also
writes ``T13_references.npz`` with the perturbed speed references (see
:func:`t13_references`).

Units: time in s, frequency in Hz, rotational speed in rpm, amplitudes in Pa
(the signals are notionally sound pressure, but only the numbers matter).
Every amplitude given below is a *peak* amplitude, except ``noise_rms``. The
validation compares against ``SynthSignal.order_amplitude`` (the stored truth),
never against the parameter lists in this file.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

from rota_order.synth import SynthSignal, rpm_profiles, synth_runup

__all__ = [
    "CASES",
    "REFERENCES",
    "main",
    "t1",
    "t2",
    "t3",
    "t4",
    "t5",
    "t6",
    "t8",
    "t10",
    "t12",
    "t13",
    "t13_references",
    "t14",
]

# Default output directory: validation/signals next to this file.
SIGNALS_DIR = Path(__file__).resolve().parent / "signals"

# Case parameters shared with the tests (documented in each docstring).
T2_RESONANCE = {"freq": 200.0, "q": 8.0, "gain": 4.0}
T2_TONE = {"freq": 450.0, "amplitude": 0.30}
T5_SEEDS = {60: 5, 1: 4}  # chosen so that no end pulse is dropped (see t5)
T10_RANGE = (1200.0, 2800.0)  # search range of rpm_from_ridge in the T10 tests, rpm
# T13 speed-reference errors (see t13_references): relative to the true speed.
T13_BIAS = 0.005  # constant relative bias
T13_JITTER_RMS = 0.005  # rms of the band-limited random relative error
T13_JITTER_BAND = 20.0  # Hz, low-pass edge of the jitter
T13_JITTER_SEED = 13
T13_STEP = 0.01  # relative size of the step
T13_STEP_TIME = 6.0  # s, when the step happens


def _t2_orders():
    """The four orders of T2 (a fresh list on every call)."""
    return [
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


def _time(duration, fs):
    return np.arange(round(duration * fs)) / fs


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
    rota_order.synth.SynthSignal
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
      :func:`rota_order.synth.synth_runup`), which is accurate when the sweep is slow
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
    rota_order.synth.SynthSignal
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
    return synth_runup(
        fs,
        rpm,
        _t2_orders(),
        tones=[dict(T2_TONE)],
        resonances=[dict(T2_RESONANCE)],
        noise_rms=0.02,
        seed=2,
    )


def t3():
    """T3: exponential run-down (non-linear speed), four orders.

    Examines angle resampling and ridge tracking when the speed is not a
    linear function of time (ROADMAP T3): the sweep rate falls with the speed,
    so the order frequencies decelerate along the record and no fixed
    time-domain analysis follows them.

    * Sampling rate 4 kHz, duration 8 s (32 000 samples).
    * Speed: exponential run-down ``4000 * (1000 / 4000) ** (t / T)`` rpm, i.e.
      from 4000 to 1000 rpm with a constant relative rate of -17.3 % per
      second (-693 rpm/s at the start, -173 rpm/s at the end).
    * Orders (peak amplitude): 1 (0.80 Pa), 2 (1.00 Pa), 3 (0.60 Pa, with a
      speed gain law: 1.0 at 1000 rpm, 0.5 at 2500 rpm, 0.8 at 4000 rpm,
      linear in between, so the envelope has a kink and a dip) and 5
      (0.40 Pa). Frequencies up to 333 Hz (order 5 at 4000 rpm). The closest
      pair of orders is 1 and 2, 16.7 Hz apart at 1000 rpm. Random initial
      phases from the seed.
    * White Gaussian noise, 0.01 Pa RMS: the same power per Hz (5e-8 Pa^2/Hz)
      as the 0.02 Pa of T2 at 16 kHz. No tones, no resonance.
    * Seed 3.

    Returns
    -------
    rota_order.synth.SynthSignal
        Mono signal with truth. Amplitudes are peak, in Pa (noise: RMS).

    References
    ----------
    Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order
    tracking. Mechanical Systems and Signal Processing, 11(2), 187-205.
    Brandt, A. (2011). Noise and Vibration Analysis: Signal Analysis and
    Experimental Procedures, ch. 12. Wiley.
    """
    fs = 4000.0
    t = _time(8.0, fs)
    rpm = rpm_profiles(t, "exponential", start=4000.0, stop=1000.0)
    orders = [
        {"order": 1.0, "amplitude": 0.80},
        {"order": 2.0, "amplitude": 1.00},
        {
            "order": 3.0,
            "amplitude": 0.60,
            "gain_rpm": [1000.0, 2500.0, 4000.0],
            "gain": [1.0, 0.5, 0.8],
        },
        {"order": 5.0, "amplitude": 0.40},
    ]
    return synth_runup(fs, rpm, orders, noise_rms=0.01, seed=3)


def t4():
    """T4: two shafts whose orders cross.

    Examines the decoupling of crossing orders by the joint Vold-Kalman
    filter (ROADMAP T4): when two orders have the same frequency a single-order
    filter cannot tell them apart, a joint solution can because their
    frequency slopes differ.

    * Sampling rate 4 kHz, duration 8 s (32 000 samples).
    * Shaft 0 (column 0 of ``rpm``): linear run-up 1200 to 3600 rpm
      (+300 rpm/s). Shaft 1 (column 1): linear run-down 3600 to 1200 rpm
      (-300 rpm/s). The speeds are equal (2400 rpm) at t = 4 s.
    * Orders, in this order: shaft 0 order 2 (1.00 Pa peak), shaft 0 order 3
      (0.60 Pa), shaft 1 order 2 (0.80 Pa), shaft 1 order 3 (0.50 Pa).
      Constant amplitudes, random initial phases from the seed.
    * Crossings of the order frequencies (three events 1.6 s apart, relative
      sweep rates of 20 to 30 Hz/s): shaft 0 order 3 with shaft 1 order 2 at
      t = 2.4 s (96 Hz); the equal orders, 2 with 2 (80 Hz) and 3 with 3
      (120 Hz), at the same time t = 4.0 s; shaft 0 order 2 with shaft 1
      order 3 at t = 5.6 s (96 Hz). Highest frequency: 180 Hz.
    * White Gaussian noise, 0.01 Pa RMS. No tones, no resonance.
    * Seed 4.

    Returns
    -------
    rota_order.synth.SynthSignal
        Mono signal with truth; ``rpm`` has shape (n, 2). Amplitudes are peak,
        in Pa (noise: RMS).

    References
    ----------
    Vold, H., & Leuridan, J. (1993). High resolution order tracking at extreme
    slew rates, using Kalman tracking filters. SAE Technical Paper 931288.
    Vold, H., Mains, M., & Blough, J. (1997). Theoretical foundations for high
    performance order tracking with the Vold-Kalman tracking filter. SAE
    Technical Paper 972007.
    """
    fs = 4000.0
    t = _time(8.0, fs)
    rpm = np.column_stack([
        rpm_profiles(t, "linear", start=1200.0, stop=3600.0),
        rpm_profiles(t, "linear", start=3600.0, stop=1200.0),
    ])
    orders = [
        {"order": 2.0, "shaft": 0, "amplitude": 1.00},
        {"order": 3.0, "shaft": 0, "amplitude": 0.60},
        {"order": 2.0, "shaft": 1, "amplitude": 0.80},
        {"order": 3.0, "shaft": 1, "amplitude": 0.50},
    ]
    return synth_runup(fs, rpm, orders, noise_rms=0.01, seed=4)


def t5(ppr=60, seed=None):
    """T5: tachometer with several pulses per revolution, jitter and drop-outs.

    Examines the robustness of :func:`rota_order.speed.tacho_to_rpm`.

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
    rota_order.synth.SynthSignal
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


def t6(duration=1.0):
    """T6: fast run-up, high slew rate.

    Examines the limit of computed order tracking and of the Vold-Kalman
    filter at an extreme sweep rate (ROADMAP T6; Vold & Leuridan 1993): the
    speed changes by 500 % within the record, an analysis frame of the order
    map spans a large part of the speed range, and the amplitude of one order
    changes with the speed, so with the time.

    * Sampling rate 4 kHz, duration ``duration`` s (default 1.0: 4000
      samples).
    * Speed: linear 1000 to 6000 rpm. With the default duration the slew rate
      is 5000 rpm/s (83.3 rev/s^2); the order frequencies sweep at 167, 333
      and 500 Hz/s for orders 2, 4 and 6. At 1000 rpm one revolution takes
      60 ms, so the first 4-revolution frame (order resolution 0.25) lasts
      169 ms and spans 1000 to 1840 rpm. ``duration`` scales the slew rate
      (10 s gives 500 rpm/s, about the sweep rate of T1/T2).
    * Orders (peak amplitude): 2 (1.00 Pa), 4 (0.60 Pa, with a speed gain law:
      0.4 at 1000 rpm, 1.0 at 3000 rpm, 0.5 at 6000 rpm, linear in between,
      i.e. a hump that rises 2.5-fold and falls again within the run) and 6
      (0.30 Pa). Frequencies up to 600 Hz. The closest pair is orders 2 and 4,
      33.3 Hz apart at 1000 rpm. Random initial phases from the seed.
    * White Gaussian noise, 0.01 Pa RMS (same power per Hz as T2 at 16 kHz).
    * Seed 6.

    Parameters
    ----------
    duration : float, default 1.0
        Length of the run-up in s (changes the slew rate, not the speed range).

    Returns
    -------
    rota_order.synth.SynthSignal
        Mono signal with truth. Amplitudes are peak, in Pa (noise: RMS).

    References
    ----------
    Vold, H., & Leuridan, J. (1993). High resolution order tracking at extreme
    slew rates, using Kalman tracking filters. SAE Technical Paper 931288.
    Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order
    tracking. Mechanical Systems and Signal Processing, 11(2), 187-205.
    """
    fs = 4000.0
    t = _time(float(duration), fs)
    rpm = rpm_profiles(t, "linear", start=1000.0, stop=6000.0)
    orders = [
        {"order": 2.0, "amplitude": 1.00},
        {
            "order": 4.0,
            "amplitude": 0.60,
            "gain_rpm": [1000.0, 3000.0, 6000.0],
            "gain": [0.4, 1.0, 0.5],
        },
        {"order": 6.0, "amplitude": 0.30},
    ]
    return synth_runup(fs, rpm, orders, noise_rms=0.01, seed=6)


def t8(ratio=1.02, am_freq=(0.4, 0.25), am_depth=0.3, seed=8):
    """T8: twin shafts of nearly equal speed carrying the same order.

    Examines the resolvability limit (ROADMAP T8, section 6 item 7): the two
    shafts run at speeds that differ by ``ratio - 1`` (0.5 %, 2 % or 4 % for
    the three cases ``T8_r1005``, ``T8_r1020``, ``T8_r1040``), so the same
    order of the two is a pair of lines whose separation is the beat
    frequency. :func:`rota_order.extract.resolution_check` must warn when a setting
    cannot separate them.

    * Sampling rate 4 kHz, duration 10 s (40 000 samples).
    * Speed of shaft 0: linear 2400 to 3000 rpm (60 rpm/s). Shaft 1:
      ``ratio`` times shaft 0 at every sample (``rpm`` has shape (n, 2)).
    * One order on each shaft, order 4 (160 to 200 Hz): shaft 0 with 1.00 Pa
      peak, shaft 1 with 0.80 Pa. Each is amplitude-modulated:
      ``(1 + am_depth sin(2 pi f_am t))`` with ``am_depth = 0.3`` and
      ``am_freq = (0.40, 0.25)`` Hz for shaft 0 and 1, so the true envelope
      is not constant (the extraction has to follow it while separating the
      lines). Random initial phases from the seed.
    * Line separation ``4 (ratio - 1) rpm / 60``: with ratio 1.005 it is 0.80
      to 1.00 Hz, with 1.02 it is 3.2 to 4.0 Hz, with 1.04 it is 6.4 to 8.0 Hz
      (start to end of the record).
    * White Gaussian noise, 0.02 Pa RMS. No tones, no resonance.
    * Seed 8.

    Parameters
    ----------
    ratio : float, default 1.02
        Speed of shaft 1 over speed of shaft 0.
    am_freq : (float, float), default (0.4, 0.25)
        Modulation frequencies in Hz of the orders on shaft 0 and 1. The
        faster pair ``(0.8, 1.2)`` is used in the tests to put the envelope
        variation at the order of the beat frequency of the 0.5 % twins.
    am_depth : float, default 0.3
        Modulation depth of both orders.
    seed : int, default 8
        Seed of the phases and of the noise.

    Returns
    -------
    rota_order.synth.SynthSignal
        Mono signal with truth; ``rpm`` has shape (n, 2). Amplitudes are peak,
        in Pa (noise: RMS).

    References
    ----------
    Vold, H., Mains, M., & Blough, J. (1997). Theoretical foundations for high
    performance order tracking with the Vold-Kalman tracking filter. SAE
    Technical Paper 972007.
    Pan, M.-C., & Lin, Y.-F. (2006). Further exploration of Vold-Kalman
    filtering order tracking with shaft-speed information. I: Theoretical
    part, numerical implementation and parameter investigations. Mechanical
    Systems and Signal Processing, 20(5), 1134-1154.
    """
    fs = 4000.0
    t = _time(10.0, fs)
    base = rpm_profiles(t, "linear", start=2400.0, stop=3000.0)
    rpm = np.column_stack([base, float(ratio) * base])
    orders = [
        {
            "order": 4.0, "shaft": 0, "amplitude": 1.00,
            "am_depth": float(am_depth), "am_freq": float(am_freq[0]),
        },
        {
            "order": 4.0, "shaft": 1, "amplitude": 0.80,
            "am_depth": float(am_depth), "am_freq": float(am_freq[1]),
        },
    ]
    return synth_runup(fs, rpm, orders, noise_rms=0.02, seed=seed)


def t10():
    """T10: the true speed is partly outside the search range of the estimator.

    Examines :func:`rota_order.speed.rpm_from_ridge` (ROADMAP T10, section 6 item 1):
    when the true speed leaves the search range the estimator must say so
    (``at_bound``, a warning), not return the range boundary as if it were an
    estimate.

    * Sampling rate 8 kHz, duration 12 s (96 000 samples).
    * Speed: linear 600 to 3400 rpm (233 rpm/s, 3.9 rev/s^2). The tests search
      the range ``T10_RANGE`` = (1200, 2800) rpm (and, for a wider outside,
      (1500, 2500)), so the speed is below the range for the first 2.6 s, in
      it from 2.6 s to 9.4 s and above it for the last 2.6 s (for the first
      range).
    * Orders (constant peak amplitude): 1 (1.00 Pa), 2 (0.80 Pa), 3 (0.60 Pa)
      and 4 (0.50 Pa), the default comb of the estimator. Random initial
      phases from the seed.
    * White Gaussian noise, 0.05 Pa RMS. No tones, no resonance.
    * Seed 10.

    Returns
    -------
    rota_order.synth.SynthSignal
        Mono signal with truth. Amplitudes are peak, in Pa (noise: RMS).

    References
    ----------
    Peeters, C., Leclere, Q., Antoni, J., Lindahl, P., Donnal, J., Leeb, S.,
    & Helsen, J. (2019). Review and comparison of tacholess instantaneous
    speed estimation methods on experimental vibration data. Mechanical
    Systems and Signal Processing, 129, 407-436.
    Hermes, D. J. (1988). Measurement of pitch by subharmonic summation.
    Journal of the Acoustical Society of America, 83(1), 257-264.
    """
    fs = 8000.0
    t = _time(12.0, fs)
    rpm = rpm_profiles(t, "linear", start=600.0, stop=3400.0)
    orders = [
        {"order": 1.0, "amplitude": 1.00},
        {"order": 2.0, "amplitude": 0.80},
        {"order": 3.0, "amplitude": 0.60},
        {"order": 4.0, "amplitude": 0.50},
    ]
    return synth_runup(fs, rpm, orders, noise_rms=0.05, seed=10)


def t12():
    """T12: the T2 signal, used to test order removal.

    ROADMAP T12 is "T2 through ``remove_orders``": the signal is exactly T2
    (see :func:`t2`: orders 1, 2, 3, 4.5, a resonance, a fixed tone and
    noise, 16 kHz, 12 s, seed 2). The orders are removed with either extractor
    and the energy left at each removed order is measured against the stored
    truth (:func:`rota_order.metrics.order_residual_energy`).

    Returns
    -------
    rota_order.synth.SynthSignal
        The signal of :func:`t2`. Amplitudes are peak, in Pa (noise: RMS).

    References
    ----------
    Serra, X., & Smith, J. O. (1990). Spectral modeling synthesis: a sound
    analysis/synthesis system based on a deterministic plus stochastic
    decomposition. Computer Music Journal, 14(4), 12-24.
    """
    return t2()


def t13():
    """T13: a T2-like signal for tests with errors in the speed reference.

    The signal is clean; the errors are in the speed references returned by
    :func:`t13_references`. Examines how the order extractors degrade when the
    speed used for the carriers is wrong (ROADMAP T13), to give a figure for
    the speed accuracy that extraction needs.

    * Sampling rate 4 kHz, duration 12 s (48 000 samples).
    * Speed: linear 1000 to 5000 rpm, as T2.
    * Orders, tone and resonance exactly those of T2: orders 1 (0.50 Pa),
      2 (1.00 Pa), 3 (0.60 Pa with the same speed gain law) and 4.5 (0.40 Pa),
      a 450 Hz tone of 0.30 Pa, a 200 Hz resonance with Q = 8 and peak gain 4.
      Order 4.5 reaches 375 Hz at 5000 rpm.
    * White Gaussian noise, 0.01 Pa RMS: the power per Hz of T2 (0.02 Pa at
      16 kHz).
    * Seed 2 (the phases are those of T2; the noise differs because the
      sampling rate does).

    Returns
    -------
    rota_order.synth.SynthSignal
        Mono signal with truth. Amplitudes are peak, in Pa (noise: RMS).

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis: Signal Analysis and
    Experimental Procedures, ch. 12 (orders) and ch. 5-6 (resonances). Wiley.
    """
    fs = 4000.0
    t = _time(12.0, fs)
    rpm = rpm_profiles(t, "linear", start=1000.0, stop=5000.0)
    return synth_runup(
        fs,
        rpm,
        _t2_orders(),
        tones=[dict(T2_TONE)],
        resonances=[dict(T2_RESONANCE)],
        noise_rms=0.01,
        seed=2,
    )


def t13_references(
    sig,
    bias=T13_BIAS,
    jitter_rms=T13_JITTER_RMS,
    jitter_band=T13_JITTER_BAND,
    jitter_seed=T13_JITTER_SEED,
    step=T13_STEP,
    step_time=T13_STEP_TIME,
):
    """Speed references with errors for T13: truth, bias, jitter and step.

    All errors are relative to the true speed ``sig.rpm`` (rpm, shape (n,)):
    ``rpm_ref = rpm * (1 + e(t))``.

    * ``"truth"``: ``e = 0``.
    * ``"bias"``: ``e = bias`` (default 0.5 %), a constant scale error such as
      a wrong pulses-per-revolution constant or a calibration factor.
    * ``"jitter"``: ``e`` is Gaussian noise with RMS ``jitter_rms`` (default
      0.5 %), low-passed at ``jitter_band`` Hz (default 20) with a 4th-order
      Butterworth filter applied forward and backward to
      ``numpy.random.default_rng(jitter_seed)`` white noise (default seed 13;
      one second of extra noise at each end is generated and discarded, so
      the jitter is stationary from the first sample), then given zero mean
      and the exact RMS. The speed estimate of a tachometer or of a tacholess
      tracker fluctuates like this around the truth.
    * ``"step"``: ``e = step`` (default 1 %) from ``step_time`` s (default
      6 s, the middle of the record) on, zero before: a sudden offset such as
      a gear change that the reference does not follow.

    Parameters
    ----------
    sig : rota_order.synth.SynthSignal
        Signal whose true speed is perturbed (T13).
    bias, jitter_rms, step : float
        Relative sizes (0.005 is 0.5 %).
    jitter_band : float
        Low-pass edge of the jitter in Hz.
    jitter_seed : int
        Seed of the jitter.
    step_time : float
        Time of the step in s.

    Returns
    -------
    dict
        ``{"truth", "bias", "jitter", "step"}``, each an array of shape (n,)
        in rpm.

    References
    ----------
    Pan, M.-C., & Lin, Y.-F. (2006). Further exploration of Vold-Kalman
    filtering order tracking with shaft-speed information. I: Theoretical
    part, numerical implementation and parameter investigations. Mechanical
    Systems and Signal Processing, 20(5), 1134-1154.
    """
    rpm = np.asarray(sig.rpm, dtype=float)
    n = rpm.shape[0]
    # Filter a longer noise and keep the middle: the start-up transient of a zero-phase
    # filter applied to white noise is several times the RMS of the filtered noise.
    pad = round(sig.fs)
    white = np.random.default_rng(jitter_seed).standard_normal(n + 2 * pad)
    sos = sp_signal.butter(4, float(jitter_band), btype="lowpass", fs=sig.fs, output="sos")
    jit = sp_signal.sosfiltfilt(sos, white)[pad : pad + n]
    jit = (jit - jit.mean()) * (float(jitter_rms) / jit.std())
    return {
        "truth": rpm.copy(),
        "bias": rpm * (1.0 + float(bias)),
        "jitter": rpm * (1.0 + jit),
        "step": rpm * (1.0 + float(step) * (sig.t >= float(step_time))),
    }


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
    rota_order.synth.SynthSignal
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
    "T3": t3,
    "T4": t4,
    "T5": t5,
    "T5_ppr1": lambda: t5(ppr=1),
    "T6": t6,
    "T8_r1005": lambda: t8(1.005),
    "T8_r1020": lambda: t8(1.02),
    "T8_r1040": lambda: t8(1.04),
    "T10": t10,
    "T12": t12,
    "T13": t13,
    "T14": t14,
}

# Cases with extra arrays besides the signal: name -> function(sig) -> {name: array}.
# They are written to <outdir>/<name>_references.npz by main().
REFERENCES = {"T13": t13_references}


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
        ``{name: (npz_path, json_path)}`` of the files written. A case listed
        in ``REFERENCES`` (T13) also gets ``<name>_references.npz`` with its
        extra arrays, which is not part of the returned pair.

    References
    ----------
    rota-order ROADMAP, section 5.5 (directory and workflow).
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
        if name in REFERENCES:
            np.savez(outdir / f"{name}_references.npz", **REFERENCES[name](sig))
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
        extra = f", {case}_references.npz" if case in REFERENCES else ""
        print(f"{case}: {npz.name}, {js.name}{extra}")
