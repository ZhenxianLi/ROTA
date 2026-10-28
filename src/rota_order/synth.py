"""Synthetic test signals with analytic ground truth.

:func:`synth_runup` renders a run-up (or run-down) from data: a speed curve,
a list of orders with amplitude laws, optional fixed-frequency lines,
structural resonances, broadband noise and a tachometer. Every component is
rendered from stored truth: ``order_amplitude``, ``order_phase`` and
``order_frequency`` hold the exact peak envelope, total phase and
instantaneous frequency of each order, and ``stems`` holds each family of
components on its own. Analysis results can therefore be validated against
what was actually put into the signal, not against the speed curve and order
list alone, which would be circular (ROADMAP section 6, item 19). The shaft
angle is *defined* as :func:`rota_order.speed.rpm_to_phase` of the speed curve
(trapezoidal integral), so the truth phase and the analysis-side phase are
the same quantity by construction.

Units: time in s, frequency in Hz, rotational speed in rpm (or km/h for the
generic profiles), shaft angle in revolutions, phase in rad. All amplitudes
are peak amplitudes unless stated otherwise (noise is specified as RMS).
Output is mono (one channel) in v0.1.

:func:`synth_tone_noise` is separate: static listening-test stimuli (tone plus
noise at a stated broadband tone-to-noise power ratio and RMS level), with the
tone and noise stems returned on their own.

:func:`synth_scenario` chains the building blocks below into complete
recordings of a drive train with exact truth (``ice_runup``, ``ev_runup``,
``steady_cruise``): driveline speed with launch slip and shifts, an
angle-locked speed ripple, orders, combustion bursts, sidebands and inverter
carriers, broadband and transient noise, a modal transfer path and cabin
impulse response, a tachometer and an imperfect recorder. The building blocks
are re-exported here: :func:`drive_rpm`, :func:`angular_speed_ripple`,
:func:`angle_impulses`, :func:`sideband_orders`, :func:`carrier_components`,
:func:`modal_path`, :func:`modal_response`, :func:`room_impulse_response`,
:func:`apply_ir`, :func:`speed_noise`, :func:`transient_events`,
:func:`measurement_chain` and :func:`delay_signal`. All default scenario
parameters are illustrative values typical of passenger cars, rounded; they are
not measurements of any specific vehicle.
"""

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from ._broadband import synthesize_broadband
from ._channel import (
    _speed_rows,
    apply_ir,
    delay_signal,
    measurement_chain,
    modal_path,
    modal_response,
    room_impulse_response,
    speed_noise,
    transient_events,
)
from ._provenance import jsonable
from ._sources import (
    angle_impulses,
    angular_speed_ripple,
    carrier_components,
    drive_rpm,
    sideband_orders,
)
from .speed import rpm_to_phase, speed_to_rpm

__all__ = [
    "rpm_profiles",
    "synth_runup",
    "SynthSignal",
    "synth_tone_noise",
    "ToneNoiseSignal",
    "synth_scenario",
    "ScenarioSignal",
    "scenario_defaults",
    "SCENARIO_KINDS",
    "drive_rpm",
    "angular_speed_ripple",
    "angle_impulses",
    "sideband_orders",
    "carrier_components",
    "modal_path",
    "modal_response",
    "room_impulse_response",
    "apply_ir",
    "speed_noise",
    "transient_events",
    "measurement_chain",
    "delay_signal",
]

_KINDS = ("constant", "linear", "exponential", "power", "s_curve", "piecewise")
_TWO_PI = 2.0 * np.pi

_ORDER_KEYS = (
    "order", "amplitude", "shaft", "phase", "gain_rpm", "gain", "min_freq",
    "min_freq_taper", "am_depth", "am_freq", "fm_depth", "fm_freq", "fm_phase",
)
_TONE_KEYS = ("freq", "amplitude", "phase")
_RES_KEYS = ("freq", "q", "gain")


def rpm_profiles(
    t, kind="linear", *, start=None, stop=None, t_start=None, t_stop=None, exponent=2.0, knots=None
):
    """Generate a speed profile on a time axis.

    The profile is a function of the normalised time
    ``u = clip((t - t_start) / (t_stop - t_start), 0, 1)``, so it holds
    ``start`` before ``t_start`` (idle) and ``stop`` after ``t_stop``. The
    unit is whatever ``start`` and ``stop`` are given in (rpm or km/h).

    Kinds:

    * ``"constant"``: ``start`` everywhere.
    * ``"linear"``: ``start + (stop - start) * u``.
    * ``"exponential"``: ``start * (stop / start) ** u`` (constant relative
      rate; ``start`` and ``stop`` must be positive).
    * ``"power"``: ``start + (stop - start) * u ** exponent``.
    * ``"s_curve"``: raised cosine ``w = (1 - cos(pi u)) / 2``; zero slope at
      both ends.
    * ``"piecewise"``: linear interpolation through ``knots``, held at the
      end values outside the knot span (``start``, ``stop`` and the time
      limits are not used).

    Two-shaft profiles are ratios of one profile: build the primary shaft with
    this function and stack scaled copies, e.g.
    ``np.column_stack([r, 1.37 * r, 1.005 * r])`` (locked ratio, nearly
    equal speed); the result can be passed to :func:`synth_runup` as an
    ``(n, k)`` speed array. A piecewise profile gives a vehicle-speed schedule
    (such as 40, 100, 40 km/h), which :func:`rota_order.speed.speed_to_rpm` maps to
    engine speed.

    Parameters
    ----------
    t : array_like, shape (n,)
        Times in s.
    kind : str, default "linear"
        One of the kinds above.
    start, stop : float
        Values at ``t_start`` and ``t_stop``, in rpm or km/h.
    t_start, t_stop : float, optional
        Ramp limits in s. Default: ``t[0]`` and ``t[-1]``.
    exponent : float, default 2
        Exponent of the ``"power"`` kind (positive).
    knots : tuple of array_like, optional
        ``(times, values)`` of the ``"piecewise"`` kind; times in s must be
        strictly increasing.

    Returns
    -------
    ndarray, shape (n,)
        Profile values, in the unit of ``start`` and ``stop``. No amplitudes
        are involved.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis: Signal Analysis and
    Experimental Procedures, ch. 12 (run-up and run-down measurements). Wiley.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("t must be a non-empty, finite 1-D array of times in s")
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    if kind == "piecewise":
        if knots is None:
            raise ValueError("kind='piecewise' needs knots=(times, values)")
        try:
            kt, kv = (np.asarray(a, dtype=float) for a in knots)
        except (TypeError, ValueError) as err:
            raise ValueError("knots must be a pair (times, values) of 1-D arrays") from err
        if kt.ndim != 1 or kt.shape != kv.shape or kt.size < 2:
            raise ValueError("knots times and values must be 1-D, equal length, at least 2")
        if not (np.all(np.isfinite(kt)) and np.all(np.isfinite(kv))):
            raise ValueError("knots must be finite")
        if np.any(np.diff(kt) <= 0):
            raise ValueError("knot times must be strictly increasing")
        return np.interp(t, kt, kv)
    if knots is not None:
        raise ValueError("knots is only used with kind='piecewise'")
    if start is None:
        raise ValueError(f"kind={kind!r} needs start")
    start = float(start)
    if kind == "constant":
        return np.full(t.shape, start)
    if stop is None:
        raise ValueError(f"kind={kind!r} needs stop")
    stop = float(stop)
    if not (np.isfinite(start) and np.isfinite(stop)):
        raise ValueError("start and stop must be finite")
    t0 = float(t[0] if t_start is None else t_start)
    t1 = float(t[-1] if t_stop is None else t_stop)
    if not t1 > t0:
        raise ValueError("t_stop must be greater than t_start")
    u = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    if kind == "exponential":
        if start <= 0 or stop <= 0:
            raise ValueError("exponential profiles need positive start and stop")
        return np.where(u >= 1.0, stop, start * (stop / start) ** u)
    if kind == "linear":
        w = u
    elif kind == "power":
        if not exponent > 0:
            raise ValueError("exponent must be positive")
        w = u ** float(exponent)
    else:  # s_curve
        w = 0.5 * (1.0 - np.cos(np.pi * u))
    return start * (1.0 - w) + stop * w


@dataclass
class SynthSignal:
    """A synthetic signal together with its analytic truth.

    Units: time in s, frequency in Hz, speed in rpm, shaft angle in rev,
    phase in rad. Amplitudes are peak (noise: see ``stems``). ``n`` is the
    number of samples and ``K`` the number of orders.

    Attributes
    ----------
    fs : float
        Sampling rate in Hz.
    t : ndarray, shape (n,)
        Sample times in s (``arange(n) / fs``).
    x : ndarray, shape (n,)
        Mono mix, ``stems["orders"] + stems["tones"] + stems["noise"]``
        (evaluated in this order, so the identity holds bit for bit).
    rpm, rev : ndarray, shape (n,) or (n, k)
        Input speed in rpm and shaft angle in rev (one column per shaft),
        ``rev = rota_order.speed.rpm_to_phase(rpm, fs)``.
    orders : ndarray, shape (K,)
        Order numbers (cycles per revolution of the order's shaft).
    order_stems : ndarray, shape (n, K)
        Each order as rendered, ``order_amplitude * cos(order_phase)``.
    order_amplitude : ndarray, shape (n, K)
        Peak envelope of each order including amplitude, gain law, AM,
        frequency mask, resonance gain and Nyquist muting.
    order_phase : ndarray, shape (n, K)
        Total phase in rad: ``2*pi*order*rev + phi0 + FM term`` plus the
        resonance phase.
    order_frequency : ndarray, shape (n, K)
        Instantaneous frequency in Hz of the carrier,
        ``order * rpm/60 * (1 + fm_depth * sin(...))``, i.e. the derivative
        of ``order_phase`` over ``2*pi`` without the (quasi-static) resonance
        phase term.
    stems : dict
        ``"orders"`` (``order_stems.sum(axis=1)``), ``"tones"`` and
        ``"noise"``, each of shape (n,). The noise stem is after the
        envelope and resonances.
    tacho : ndarray, shape (n,) or None
        Tachometer signal (not part of ``x``).
    pulse_times, pulse_revs : ndarray
        Time in s of each rendered tachometer pulse (rising-edge 50 %
        crossing, jitter included) and its nominal shaft angle in rev
        (``k / ppr``); dropped pulses are absent. Empty without a tachometer.
    config : dict
        JSON-serialisable record of the inputs, with the resolved random
        phases, the seed (if an int) and the rota-order version.
    """

    fs: float
    t: np.ndarray
    x: np.ndarray
    rpm: np.ndarray
    rev: np.ndarray
    orders: np.ndarray
    order_stems: np.ndarray
    order_amplitude: np.ndarray
    order_phase: np.ndarray
    order_frequency: np.ndarray
    stems: dict
    tacho: np.ndarray | None
    pulse_times: np.ndarray
    pulse_revs: np.ndarray
    config: dict

    def save(self, path):
        """Write the arrays to ``<path>.npz`` and the config to ``<path>.json``.

        The JSON file is a provenance sidecar (parameters, resolved phases,
        seed, rota-order version); the NPZ holds ``fs``, ``t``, ``x``, ``rpm``,
        ``rev``, ``orders``, ``order_stems``, ``order_amplitude``,
        ``order_phase``, ``order_frequency``, ``stem_orders``, ``stem_tones``,
        ``stem_noise``, ``pulse_times``, ``pulse_revs``, ``tacho`` (only if
        present) and the config as a JSON string ``config_json``. A trailing
        ``.npz`` or ``.json`` on ``path`` is ignored.

        Returns
        -------
        (Path, Path)
            The NPZ and JSON paths.

        References
        ----------
        rota-order ROADMAP, design principle 7 (reproducibility sidecar).
        """
        p = Path(path)
        if p.suffix in (".npz", ".json"):
            p = p.with_suffix("")
        npz = p.with_name(p.name + ".npz")
        sidecar = p.with_name(p.name + ".json")
        text = json.dumps(self.config, indent=2)
        arrays = {
            "fs": np.asarray(self.fs),
            "t": self.t,
            "x": self.x,
            "rpm": self.rpm,
            "rev": self.rev,
            "orders": self.orders,
            "order_stems": self.order_stems,
            "order_amplitude": self.order_amplitude,
            "order_phase": self.order_phase,
            "order_frequency": self.order_frequency,
            "stem_orders": self.stems["orders"],
            "stem_tones": self.stems["tones"],
            "stem_noise": self.stems["noise"],
            "pulse_times": self.pulse_times,
            "pulse_revs": self.pulse_revs,
            "config_json": np.asarray(text),
        }
        if self.tacho is not None:
            arrays["tacho"] = self.tacho
        np.savez(npz, **arrays)
        sidecar.write_text(text + "\n", encoding="utf-8")
        return npz, sidecar


def _py(v):
    """Convert NumPy scalars and arrays to plain Python for JSON."""
    if isinstance(v, Mapping):
        return {str(k): _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, np.ndarray):
        return _py(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


def _specs(specs, name, allowed, required):
    """Validate a sequence of spec dicts and return shallow copies."""
    if specs is None:
        return []
    if isinstance(specs, Mapping):
        specs = [specs]
    out = []
    for i, s in enumerate(specs):
        if not isinstance(s, Mapping):
            raise TypeError(f"{name}[{i}] must be a dict, got {type(s).__name__}")
        extra = sorted(set(s) - set(allowed))
        if extra:
            raise ValueError(f"unknown key(s) {extra} in {name}[{i}]; allowed: {list(allowed)}")
        missing = [k for k in required if s.get(k) is None]
        if missing:
            raise ValueError(f"{name}[{i}] is missing required key(s) {missing}")
        out.append(dict(s))
    return out


def _num(spec, key, default, what, *, lo=None, hi=None, lo_open=False, hi_open=False):
    """Read a finite float from ``spec`` (None or absent -> ``default``)."""
    v = spec.get(key)
    if v is None:
        return default
    v = float(v)
    ok = np.isfinite(v)
    if lo is not None:
        ok &= v > lo if lo_open else v >= lo
    if hi is not None:
        ok &= v < hi if hi_open else v <= hi
    if not ok:
        raise ValueError(f"{what}: {key}={v!r} is out of range")
    return v


def _parse_orders(specs, n_shafts):
    out = []
    for i, s in enumerate(_specs(specs, "orders", _ORDER_KEYS, ("order",))):
        what = f"orders[{i}]"
        shaft = s.get("shaft", 0)
        if shaft is None:
            shaft = 0
        if int(shaft) != shaft or not 0 <= int(shaft) < n_shafts:
            raise ValueError(f"{what}: shaft={shaft!r} is not a column of rpm ({n_shafts})")
        gr, g = s.get("gain_rpm"), s.get("gain")
        if (gr is None) != (g is None):
            raise ValueError(f"{what}: gain_rpm and gain must be given together")
        if gr is not None:
            gr = np.atleast_1d(np.asarray(gr, dtype=float))
            g = np.atleast_1d(np.asarray(g, dtype=float))
            if gr.ndim != 1 or gr.shape != g.shape or gr.size < 1:
                raise ValueError(f"{what}: gain_rpm and gain must be 1-D of equal length")
            if not (np.all(np.isfinite(gr)) and np.all(np.isfinite(g)) and np.all(g >= 0)):
                raise ValueError(f"{what}: gain_rpm must be finite and gain finite and >= 0")
            if np.any(np.diff(gr) <= 0):
                raise ValueError(f"{what}: gain_rpm must be strictly increasing")
        spec = {
            "order": _num(s, "order", None, what, lo=0.0, lo_open=True),
            "amplitude": _num(s, "amplitude", 1.0, what, lo=0.0),
            "shaft": int(shaft),
            "phase": _num(s, "phase", None, what),
            "gain_rpm": gr,
            "gain": g,
            "min_freq": _num(s, "min_freq", None, what, lo=0.0),
            "min_freq_taper": _num(s, "min_freq_taper", 0.0, what, lo=0.0),
            "am_depth": _num(s, "am_depth", 0.0, what, lo=0.0, hi=1.0),
            "am_freq": _num(s, "am_freq", None, what, lo=0.0),
            "fm_depth": _num(s, "fm_depth", 0.0, what, lo=0.0, hi=1.0, hi_open=True),
            "fm_freq": _num(s, "fm_freq", None, what, lo=0.0),
            "fm_phase": _num(s, "fm_phase", 0.0, what),
        }
        if spec["min_freq"] is None and spec["min_freq_taper"] > 0:
            raise ValueError(f"{what}: min_freq_taper needs min_freq")
        if spec["am_depth"] > 0 and spec["am_freq"] is None:
            raise ValueError(f"{what}: am_depth needs am_freq")
        if spec["fm_depth"] > 0 and spec["fm_freq"] is None:
            raise ValueError(f"{what}: fm_depth needs fm_freq")
        out.append(spec)
    return out


def _parse_tones(specs):
    out = []
    for i, s in enumerate(_specs(specs, "tones", _TONE_KEYS, ("freq",))):
        what = f"tones[{i}]"
        out.append({
            "freq": _num(s, "freq", None, what, lo=0.0, lo_open=True),
            "amplitude": _num(s, "amplitude", 1.0, what, lo=0.0),
            "phase": _num(s, "phase", None, what),
        })
    return out


def _parse_resonances(specs, fs):
    out = []
    for i, s in enumerate(_specs(specs, "resonances", _RES_KEYS, _RES_KEYS)):
        what = f"resonances[{i}]"
        spec = {
            "freq": _num(s, "freq", None, what, lo=0.0, lo_open=True),
            "q": _num(s, "q", None, what, lo=0.0, lo_open=True),
            "gain": _num(s, "gain", None, what, lo=0.0, lo_open=True),
        }
        if spec["freq"] >= 0.5 * fs:
            raise ValueError(f"{what}: freq must be below fs/2")
        out.append(spec)
    return out


def _resonance_response(resonances, f):
    """Magnitude and unwrapped phase (rad) of the resonance product at ``f`` (Hz).

    Analog ``H(s) = (s^2 + g w0/Q s + w0^2) / (s^2 + w0/Q s + w0^2)`` at
    ``s = j 2 pi f`` for each resonance, multiplied. The phase is the sum of
    ``arg(num) - arg(den)`` with both arguments in [0, pi], so it is continuous
    in ``f``.
    """
    f = np.asarray(f, dtype=float)
    mag = np.ones(f.shape)
    phase = np.zeros(f.shape)
    w = _TWO_PI * f
    for r in resonances:
        w0 = _TWO_PI * r["freq"]
        d = w0 / r["q"]
        re = w0**2 - w**2
        num = re + 1j * (r["gain"] * d) * w
        den = re + 1j * d * w
        mag = mag * np.abs(num) / np.abs(den)
        phase = phase + np.angle(num) - np.angle(den)
    return mag, phase


def _resonance_sos(resonances, fs):
    """Digital resonances as second-order sections (bilinear, prewarped at f0).

    The analog prototype is designed at the prewarped frequency
    ``2 fs tan(pi f0 / fs)``, so the digital gain at ``f0`` equals ``gain``.
    """
    rows = []
    for r in resonances:
        w0 = 2.0 * fs * np.tan(np.pi * r["freq"] / fs)
        b, a = signal.bilinear(
            [1.0, r["gain"] * w0 / r["q"], w0**2], [1.0, w0 / r["q"], w0**2], fs
        )
        rows.append(np.concatenate((b, a)))
    return np.array(rows)


def _make_noise(rng, n, fs, rms, band):
    """White Gaussian noise, optionally band-limited, scaled to exact RMS."""
    if rms == 0.0:
        return np.zeros(n)
    w = rng.standard_normal(n)
    if band is not None:
        lo, hi = band
        if lo is None or lo <= 0:
            lo = None
        if lo is None and hi is None:
            raise ValueError("noise_band needs at least one edge")
        edges = [e for e in (lo, hi) if e is not None]
        if not all(0 < e < 0.5 * fs for e in edges) or (len(edges) == 2 and lo >= hi):
            raise ValueError("noise_band must satisfy 0 < lo < hi < fs/2")
        if lo is None:
            sos = signal.butter(4, hi, btype="lowpass", fs=fs, output="sos")
        elif hi is None:
            sos = signal.butter(4, lo, btype="highpass", fs=fs, output="sos")
        else:
            sos = signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        w = signal.sosfiltfilt(sos, w)
    return w * (rms / np.sqrt(np.mean(w**2)))


def _invert_rev(theta, rev, rpm, fs):
    """Times in s at which the shaft angle reaches ``theta`` (rev).

    ``rev`` is the trapezoidal integral of a piecewise-linear speed, i.e. a
    piecewise-quadratic function of time that is exact at the samples; each
    interval is inverted in closed form.
    """
    i = np.clip(np.searchsorted(rev, theta, side="right") - 1, 0, rev.size - 2)
    d = 60.0 * (theta - rev[i])  # rpm * s
    a = (rpm[i + 1] - rpm[i]) * fs  # rpm / s
    den = rpm[i] + np.sqrt(np.maximum(rpm[i] ** 2 + 2.0 * a * d, 0.0))
    tau = np.where(d > 0, 2.0 * d / np.where(den > 0, den, 1.0), 0.0)
    return i / fs + tau


def _pulse_train(t, fs, rise, fall):
    """Unit pulse train with two-sample linear edges centred on the edge times.

    Each edge is the ramp ``clip(0.5 + (t - t_e) * fs / 2, 0, 1)`` (full
    swing over two samples). Between the two samples that bracket ``t_e`` the
    sampled ramp is a straight line through ``(t_e, 0.5)``, so linear
    interpolation of the 50 % crossing returns ``t_e`` exactly. (A ramp of one
    sample would be clipped inside the bracket and bias the crossing by up to
    about 0.06 sample.) Vectorised: edges fully in the past are counted with
    one ``searchsorted``, the (at most a few) edges inside the ramp window of
    each sample are added explicitly.
    """
    edges = np.concatenate((rise, fall))
    sign = np.concatenate((np.ones(rise.size), -np.ones(fall.size)))
    o = np.argsort(edges, kind="stable")
    edges, sign = edges[o], sign[o]
    done = np.concatenate(([0.0], np.cumsum(sign)))
    half = 1.0 / fs
    j = np.searchsorted(edges, t - half, side="right")
    hi = np.searchsorted(edges, t + half, side="left")
    y = done[j]
    for m in range(int((hi - j).max(initial=0))):
        k = j + m
        sel = k < hi
        ks = k[sel]
        y[sel] += sign[ks] * np.clip(0.5 + (t[sel] - edges[ks]) * (0.5 * fs), 0.0, 1.0)
    return y


def synth_runup(
    fs,
    rpm,
    orders,
    *,
    tones=(),
    resonances=(),
    noise_rms=0.0,
    noise_band=None,
    noise_envelope=None,
    resonance_on_noise=True,
    tacho_ppr=None,
    tacho_shaft=0,
    tacho_jitter=0.0,
    tacho_drop=0.0,
    tacho_high=5.0,
    seed=None,
):
    """Synthesise a mono run-up signal with analytic truth.

    Model. With ``rev = rota_order.speed.rpm_to_phase(rpm, fs)`` (revolutions, 0 at
    the first sample; this integral is the definition of the truth phase), the
    k-th order of shaft ``s`` is ``x_k(t) = A_k(t) cos(Phi_k(t))`` with

    * ``Phi_k = 2 pi order rev_s + phi0 + FM + arg H(f_k)``,
    * ``FM`` the trapezoidal integral of ``2 pi order rpm_s/60 fm_depth
      sin(2 pi fm_freq t + fm_phase)``, so that the carrier frequency is
      ``f_k = order rpm_s/60 (1 + fm_depth sin(2 pi fm_freq t + fm_phase))``,
    * ``A_k = amplitude * gain(rpm_s) * (1 + am_depth sin(2 pi am_freq t))
      * mask(f_k) * |H(f_k)|``, and
    * ``A_k = 0`` wherever ``f_k >= fs/2``. Components above Nyquist are muted,
      never clipped to ``fs/2`` (which would create a false tone).

    ``gain`` is a piecewise-linear law of the shaft speed (``np.interp``, end
    values held). ``mask`` is 0 below ``min_freq``, 1 above
    ``min_freq + min_freq_taper`` and a raised-cosine ramp in between.

    Resonances are second-order sections with analog response
    ``H(s) = prod (s^2 + g w0/Q s + w0^2) / (s^2 + w0/Q s + w0^2)`` at
    ``s = j 2 pi f``, so ``|H(f0)| = g`` (linear peak gain, not dB). They act
    on orders and tones *quasi-statically*: each component is multiplied by
    ``|H|`` and phase-shifted by ``arg H`` at its instantaneous frequency.
    This is the definition of the truth; it is accurate when the component
    sweeps through the resonance slowly compared with the ring-down time
    ``Q / (pi f0)``. On noise (``resonance_on_noise``) the same sections are
    applied as filters: bilinear transform prewarped at ``f0``, so the digital
    gain at ``f0`` is also ``g``.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    rpm : array_like, shape (n,) or (n, k)
        Speed in rpm, one column per shaft; finite and non-negative. Its
        length sets the number of samples; ``t = arange(n) / fs``.
    orders : sequence of dict
        One dict per order. ``order`` (cycles per revolution, positive, may be
        fractional) is required. Optional: ``amplitude`` (peak, default 1),
        ``shaft`` (column of ``rpm``, default 0), ``phase`` (rad; None draws
        uniformly from the generator), ``gain_rpm`` and ``gain`` (together;
        speed in rpm of the order's shaft and linear gain), ``min_freq`` (Hz),
        ``min_freq_taper`` (Hz, default 0), ``am_depth`` (0 to 1) with
        ``am_freq`` (Hz), ``fm_depth`` (relative, 0 to <1) with ``fm_freq``
        (Hz) and ``fm_phase`` (rad, default 0). Unknown keys raise an error.
    tones : sequence of dict, optional
        Fixed-frequency lines: ``freq`` (Hz), ``amplitude`` (peak, default 1),
        ``phase`` (rad; None draws a random phase).
    resonances : sequence of dict, optional
        ``freq`` (Hz, below ``fs/2``), ``q`` and ``gain`` (linear peak gain).
    noise_rms : float, default 0
        RMS of the white Gaussian noise, before ``noise_envelope`` and
        resonances.
    noise_band : (float or None, float or None), optional
        ``(lo, hi)`` in Hz: band-limit the noise with a 4th-order Butterworth
        filter applied forward and backward (zero phase). A None or
        non-positive ``lo`` gives a low-pass, a None ``hi`` a high-pass. The
        noise is rescaled to ``noise_rms`` after filtering.
    noise_envelope : array_like, shape (n,), optional
        Multiplies the noise (e.g. a road or wind noise level law); the RMS of
        the stem is then ``noise_rms`` times the RMS of the envelope.
    resonance_on_noise : bool, default True
        Apply the resonances to the noise.
    tacho_ppr : int, optional
        Pulses per revolution; enables the tachometer. Pulse ``k`` (1, 2, ...)
        occurs at shaft angle ``k / ppr`` of shaft ``tacho_shaft``, located by
        inverting the shaft angle exactly (it is piecewise quadratic in time
        between samples). The tachometer is a separate channel and is not
        added to ``x``.
    tacho_shaft : int, default 0
        Column of ``rpm`` the tachometer is mounted on.
    tacho_jitter : float, default 0
        Standard deviation in s of Gaussian timing noise on each pulse.
    tacho_drop : float, default 0
        Probability of dropping each pulse.
    tacho_high : float, default 5
        Pulse height (low level 0). Pulses are high for 50 % of the local
        nominal interval; each edge is a two-sample linear ramp centred on the
        edge time, so a 50 % level crossing found by linear interpolation
        returns the pulse time exactly. At least 4 samples per pulse are
        required.
    seed : int, numpy.random.Generator or None
        Random source. Component phases, noise and tachometer errors use
        independent streams derived from it, so adding noise does not change
        the phases. The same seed gives bit-identical output.

    Returns
    -------
    SynthSignal
        See its attributes for shapes and conventions. ``x`` is mono; all
        amplitudes are peak (the noise stem has RMS ``noise_rms`` before the
        envelope and resonances).

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis: Signal Analysis and
    Experimental Procedures, ch. 12 (orders and run-ups) and ch. 5-6
    (second-order systems and resonances). Wiley.
    Boashash, B. (1992). Estimating and interpreting the instantaneous
    frequency of a signal, Part 1: Fundamentals. Proceedings of the IEEE,
    80(4), 520-538.
    Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal
    Processing, 3rd ed., sec. 7.1.2 (bilinear transformation) and ch. 7
    (Butterworth filters). Pearson.
    """
    from . import __version__

    fs = float(fs)
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a positive sampling rate in Hz")
    rpm_in = np.array(rpm, dtype=float)
    if rpm_in.ndim not in (1, 2):
        raise ValueError("rpm must have shape (n,) or (n, k)")
    n = rpm_in.shape[0]
    if n < 2:
        raise ValueError("rpm must have at least 2 samples")
    if not np.all(np.isfinite(rpm_in)) or np.any(rpm_in < 0):
        raise ValueError("rpm must be finite and non-negative")
    rev_in = rpm_to_phase(rpm_in, fs)
    R = rpm_in[:, None] if rpm_in.ndim == 1 else rpm_in
    REV = rev_in[:, None] if rpm_in.ndim == 1 else rev_in
    n_shafts = R.shape[1]
    t = np.arange(n) / fs
    nyq = 0.5 * fs

    ords = _parse_orders(orders, n_shafts)
    tns = _parse_tones(tones)
    res = _parse_resonances(resonances, fs)
    K = len(ords)
    noise_rms = float(noise_rms)
    if not (np.isfinite(noise_rms) and noise_rms >= 0):
        raise ValueError("noise_rms must be finite and non-negative")
    if noise_envelope is not None:
        noise_envelope = np.asarray(noise_envelope, dtype=float)
        if noise_envelope.shape != (n,) or not np.all(np.isfinite(noise_envelope)):
            raise ValueError(f"noise_envelope must be finite with shape ({n},)")

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**63 - 1, size=3, dtype=np.int64)
    rng_phase, rng_noise, rng_tacho = (np.random.default_rng(int(s)) for s in seeds)
    u_phase = rng_phase.uniform(0.0, _TWO_PI, size=K + len(tns))
    for i, sp in enumerate(ords + tns):
        if sp["phase"] is None:
            sp["phase"] = float(u_phase[i])

    # Orders.
    A = np.empty((n, K))
    PHI = np.empty((n, K))
    F = np.empty((n, K))
    S = np.empty((n, K))
    for k, sp in enumerate(ords):
        r = R[:, sp["shaft"]]
        o = sp["order"]
        f_car = o * r / 60.0
        phi = _TWO_PI * o * REV[:, sp["shaft"]] + sp["phase"]
        f = f_car
        if sp["fm_depth"] > 0:
            s = np.sin(_TWO_PI * sp["fm_freq"] * t + sp["fm_phase"])
            f = f_car * (1.0 + sp["fm_depth"] * s)
            integrand = _TWO_PI * f_car * sp["fm_depth"] * s
            fm = np.zeros(n)
            np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) / fs, out=fm[1:])
            phi = phi + fm
        a = np.full(n, sp["amplitude"])
        if sp["gain_rpm"] is not None:
            a *= np.interp(r, sp["gain_rpm"], sp["gain"])
        if sp["am_depth"] > 0:
            a *= 1.0 + sp["am_depth"] * np.sin(_TWO_PI * sp["am_freq"] * t)
        if sp["min_freq"] is not None:
            if sp["min_freq_taper"] > 0:
                x_ = np.clip((f - sp["min_freq"]) / sp["min_freq_taper"], 0.0, 1.0)
                a *= 0.5 * (1.0 - np.cos(np.pi * x_))
            else:
                a *= f >= sp["min_freq"]
        if res:
            mag, ph = _resonance_response(res, f)
            a *= mag
            phi = phi + ph
        a[f >= nyq] = 0.0
        A[:, k], PHI[:, k], F[:, k] = a, phi, f
        S[:, k] = a * np.cos(phi)
    stem_orders = S.sum(axis=1)

    # Fixed-frequency lines.
    stem_tones = np.zeros(n)
    for sp in tns:
        mag, ph = _resonance_response(res, sp["freq"])
        amp = float(sp["amplitude"] * mag) if sp["freq"] < nyq else 0.0
        stem_tones += amp * np.cos(_TWO_PI * sp["freq"] * t + sp["phase"] + float(ph))

    # Noise.
    noise = _make_noise(rng_noise, n, fs, noise_rms, noise_band)
    if noise_envelope is not None:
        noise = noise * noise_envelope
    if res and resonance_on_noise and noise_rms > 0:
        noise = signal.sosfilt(_resonance_sos(res, fs), noise)

    x = stem_orders + stem_tones + noise

    # Tachometer.
    tacho = None
    pulse_times = np.empty(0)
    pulse_revs = np.empty(0)
    n_dropped = 0
    if tacho_ppr is not None:
        if int(tacho_ppr) != tacho_ppr or tacho_ppr < 1:
            raise ValueError("tacho_ppr must be a positive integer")
        ppr = int(tacho_ppr)
        if int(tacho_shaft) != tacho_shaft or not 0 <= tacho_shaft < n_shafts:
            raise ValueError(f"tacho_shaft must be a column of rpm (0..{n_shafts - 1})")
        tacho_shaft = int(tacho_shaft)
        if not (np.isfinite(tacho_jitter) and tacho_jitter >= 0):
            raise ValueError("tacho_jitter must be non-negative (s)")
        if not 0 <= tacho_drop < 1:
            raise ValueError("tacho_drop must be in [0, 1)")
        if not tacho_high > 0:
            raise ValueError("tacho_high must be positive")
        rev_s, rpm_s = REV[:, tacho_shaft], R[:, tacho_shaft]
        if np.any(np.diff(rev_s) <= 0):
            raise ValueError("the tachometer shaft needs rpm > 0 at every sample")
        m = int(np.floor(rev_s[-1] * ppr + 1e-9))
        revs = np.arange(1, m + 1) / ppr
        nominal = _invert_rev(revs, rev_s, rpm_s, fs)
        width = 0.5 * 60.0 / (ppr * np.interp(nominal, t, rpm_s))
        if m and np.min(width) * fs < 2.0:
            raise ValueError("tacho pulses are shorter than 2 samples; lower ppr or raise fs")
        jitter = rng_tacho.standard_normal(m) * float(tacho_jitter)
        keep = rng_tacho.random(m) >= tacho_drop
        n_dropped = int(m - keep.sum())
        rise = (nominal + jitter)[keep]
        width = width[keep]
        pulse_revs = revs[keep]
        o = np.argsort(rise, kind="stable")
        rise, width, pulse_revs = rise[o], width[o], pulse_revs[o]
        pulse_times = rise
        tacho = float(tacho_high) * _pulse_train(t, fs, rise, rise + width)

    config = {
        "rota_order_version": __version__,
        "seed": int(seed) if isinstance(seed, (int, np.integer)) else None,
        "fs": fs,
        "n": int(n),
        "n_shafts": int(n_shafts),
        "orders": [dict(sp) for sp in ords],
        "tones": [dict(sp) for sp in tns],
        "resonances": [dict(sp) for sp in res],
        "noise_rms": noise_rms,
        "noise_band": None if noise_band is None else list(noise_band),
        "noise_envelope": noise_envelope is not None,
        "resonance_on_noise": bool(resonance_on_noise),
        "tacho_ppr": None if tacho_ppr is None else int(tacho_ppr),
        "tacho_shaft": int(tacho_shaft),
        "tacho_jitter": float(tacho_jitter),
        "tacho_drop": float(tacho_drop),
        "tacho_high": float(tacho_high),
        "tacho_n_dropped": n_dropped,
    }
    return SynthSignal(
        fs=fs,
        t=t,
        x=x,
        rpm=rpm_in,
        rev=rev_in,
        orders=np.array([sp["order"] for sp in ords], dtype=float),
        order_stems=S,
        order_amplitude=A,
        order_phase=PHI,
        order_frequency=F,
        stems={"orders": stem_orders, "tones": stem_tones, "noise": noise},
        tacho=tacho,
        pulse_times=pulse_times,
        pulse_revs=pulse_revs,
        config=_py(config),
    )


@dataclass
class ToneNoiseSignal:
    """A tone-plus-noise stimulus with its stems and configuration.

    Units: time in s, amplitudes in the unit of ``level_rms`` (e.g. Pa).
    ``n`` is the number of samples.

    Attributes
    ----------
    fs : float
        Sampling rate in Hz.
    t : ndarray, shape (n,)
        Sample times in s (``arange(n) / fs``).
    x : ndarray, shape (n,)
        The stimulus, ``tone + noise`` (bit for bit). Its RMS is ``level_rms``
        to rounding error.
    tone, noise : ndarray, shape (n,)
        The stems. Their mean-square values are exactly
        ``level_rms**2 * r / (1 + r)`` and ``level_rms**2 / (1 + r)`` with
        ``r = 10**(tnr_db / 10)``, to rounding error, and they are exactly
        orthogonal over the record (``sum(tone * noise) = 0`` to rounding
        error).
    config : dict
        JSON-serialisable record of the inputs, the resolved random phases,
        the seed (if an int), the stem powers and the rota-order version. Infinite
        ``tnr_db`` is stored as the string ``"inf"`` or ``"-inf"``.
    """

    fs: float
    t: np.ndarray
    x: np.ndarray
    tone: np.ndarray
    noise: np.ndarray
    config: dict


def _unit_rms(v):
    r = np.sqrt(np.mean(v**2))
    if not r > 0:
        raise ValueError("the requested component has no energy; check frequencies and bands")
    return v / r


def _json_float(v):
    return float(v) if np.isfinite(v) else ("inf" if v > 0 else "-inf")


def _random_phase_signal(rng, n, mag):
    """Real signal of length ``n`` with rfft magnitudes ``mag`` and random phases.

    The phases are uniform on [0, 2 pi) for every bin (drawn for all bins so
    that the stream does not depend on ``mag``); DC and, for even ``n``, the
    Nyquist bin are set to zero.
    """
    mag = np.array(mag, dtype=float)
    mag[0] = 0.0
    if n % 2 == 0:
        mag[-1] = 0.0
    phase = rng.uniform(0.0, _TWO_PI, size=mag.size)
    return np.fft.irfft(mag * np.exp(1j * phase), n)


def synth_tone_noise(
    fs,
    duration,
    level_rms,
    tnr_db,
    *,
    tone_freq=None,
    harmonics=(1.0,),
    tone_bandwidth_oct=0.0,
    noise="white",
    hump_fc=None,
    hump_sigma_oct=0.7,
    seed=None,
):
    """Synthesise a tone-plus-noise listening-test stimulus.

    The stimulus is ``x = tone + noise`` with the tone-to-noise ratio set on the
    *broadband* power. With total mean-square ``P = level_rms**2`` (RMS
    convention) and ``r = 10**(tnr_db / 10)``::

        tone power  = P * r / (1 + r)
        noise power = P / (1 + r)

    so ``tnr_db`` is ``10 log10`` of the ratio of the tone power to the power of
    the whole noise stem over 0 to ``fs/2``. This is **not** the ECMA-74
    Annex D tone-to-noise ratio, which relates the tone to the noise power in
    the critical band centred on the tone. For white noise the two differ by
    the constant ``10 log10((fs/2) / delta_f)`` with ``delta_f`` the critical
    bandwidth in Hz at the tone frequency: ``TNR_cb = tnr_db + 10 log10((fs/2)
    / delta_f)``. For the hump noise the in-band noise power must be computed
    from the spectrum, see the config.

    Exact level and ratio. Each stem is normalised to unit sample RMS and then
    scaled to the powers above; before scaling, the noise is made exactly
    orthogonal to the tone over the record (its projection onto the tone
    waveform is removed, a power change of about ``1/n``). The cross term of
    ``x`` therefore vanishes, so the RMS of ``x`` equals ``level_rms`` and
    the stem power ratio equals ``r`` to rounding error, rather than only on
    average. The random phases and the noise are drawn from independent
    streams of ``seed``, so for a given seed changing ``tnr_db`` or
    ``level_rms`` only rescales the stems (the same stimulus at another level
    or ratio).

    Tone. ``tone_freq`` is the fundamental in Hz. With
    ``tone_bandwidth_oct = 0`` the tone is a harmonic complex
    ``sum_k harmonics[k] cos(2 pi (k+1) tone_freq t + phi_k)`` with random
    phases ``phi_k`` (``harmonics=(1.0,)`` is a pure tone); ``harmonics`` are
    linear relative peak amplitudes of partial ``k+1``, so the partial peak
    amplitude is ``sqrt(2 * tone_power) * harmonics[k] / sqrt(sum(harmonics**2))``
    (exact for a record holding whole periods, otherwise within the
    normalisation to sample RMS). With ``tone_bandwidth_oct > 0`` each partial
    is replaced by narrow-band noise of that width in octaves, geometrically
    centred on the partial (edges ``f * 2**(+-width/2)``): a flat spectrum with
    random phases inside the band, zero outside, and band powers in proportion
    to ``harmonics**2``. The record needs at least 2 spectral bins in each band.

    Noise. ``"white"`` has a flat spectrum; ``"hump"`` has the power spectral
    density per Hz ``S(f) ~ exp(-0.5 * (log2(f / hump_fc) / hump_sigma_oct)**2)``,
    a Gaussian in log frequency whose PSD peak is at ``hump_fc`` and whose
    half-power (-3 dB) full width is ``2 sqrt(2 ln 2) * hump_sigma_oct``
    octaves (about 2.355 sigma). Both are made by an inverse FFT of the whole
    record with the stated magnitudes and random phases: the discrete spectrum
    follows the specification exactly (there is no chi-squared scatter in the
    periodogram), the amplitude distribution is close to Gaussian, DC and the
    Nyquist bin are empty, and the noise is periodic with the record length.

    Example anchor stimuli (60 dB SPL re 20 uPa is ``level_rms = 0.02`` Pa):
    ``synth_tone_noise(48000, 5, 0.02, np.inf, tone_freq=1000)`` is a pure tone;
    ``synth_tone_noise(48000, 5, 0.02, -np.inf)`` is white noise.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    duration : float
        Duration in s; the record has ``round(duration * fs)`` samples.
    level_rms : float
        RMS of the stimulus ``x``, in the unit of the result (e.g. Pa). No
        calibration constant is applied anywhere.
    tnr_db : float
        Broadband tone-to-noise power ratio in dB. ``inf`` gives the tone
        alone, ``-inf`` the noise alone (then ``tone_freq`` may be None).
    tone_freq : float, optional
        Fundamental in Hz, ``0 < tone_freq < fs/2``. Required unless
        ``tnr_db`` is ``-inf``. Every partial with non-zero weight (and its
        band edge, if ``tone_bandwidth_oct > 0``) must lie below ``fs/2``;
        none is muted or clipped silently.
    harmonics : sequence of float, default (1.0,)
        Linear relative amplitudes of partials 1, 2, ... (non-negative, not all
        zero).
    tone_bandwidth_oct : float, default 0
        Width in octaves of the narrow-band noise replacing each partial; 0
        gives sinusoids.
    noise : {"white", "hump"}
        Noise spectrum.
    hump_fc : float, optional
        Peak frequency in Hz of the hump noise, ``0 < hump_fc < fs/2``;
        required for ``noise="hump"`` and not allowed otherwise.
    hump_sigma_oct : float, default 0.7
        Standard deviation in octaves of the hump (positive).
    seed : int, numpy.random.Generator or None
        Random source for the phases and the noise. The same seed gives
        bit-identical output.

    Returns
    -------
    ToneNoiseSignal
        ``x``, the ``tone`` and ``noise`` stems, times and configuration. All
        amplitudes are in the unit of ``level_rms`` (RMS convention for the
        levels; partial amplitudes of a sinusoidal tone are peak values).

    References
    ----------
    ECMA-74 (Annex D: prominent discrete tones) and ISO 7779 (Annex D) for the
    critical-band tone-to-noise ratio that this broadband ratio is not.
    Zwicker, E., & Fastl, H. (2007). Psychoacoustics: Facts and Models, 3rd
    ed. Springer.
    Rice, S. O. (1944). Mathematical analysis of random noise. Bell System
    Technical Journal, 23(3), 282-332 (noise as a sum of random-phase
    components).
    """
    from . import __version__

    fs = float(fs)
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a positive sampling rate in Hz")
    duration = float(duration)
    if not (np.isfinite(duration) and duration > 0):
        raise ValueError("duration must be a positive time in s")
    n = round(duration * fs)
    if n < 16:
        raise ValueError("duration * fs must give at least 16 samples")
    level_rms = float(level_rms)
    if not (np.isfinite(level_rms) and level_rms >= 0):
        raise ValueError("level_rms must be finite and non-negative")
    tnr_db = float(tnr_db)
    if np.isnan(tnr_db):
        raise ValueError("tnr_db must not be NaN (use inf or -inf for a single component)")
    w = np.asarray(harmonics, dtype=float)
    if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("harmonics must be a non-empty 1-D sequence of non-negative weights, not all zero")
    bw = float(tone_bandwidth_oct)
    if not (np.isfinite(bw) and bw >= 0):
        raise ValueError("tone_bandwidth_oct must be finite and non-negative")
    if noise not in ("white", "hump"):
        raise ValueError(f"noise must be 'white' or 'hump', got {noise!r}")
    if noise == "hump":
        if hump_fc is None:
            raise ValueError("noise='hump' needs hump_fc")
        hump_fc = float(hump_fc)
        if not (np.isfinite(hump_fc) and 0 < hump_fc < 0.5 * fs):
            raise ValueError("hump_fc must satisfy 0 < hump_fc < fs/2")
        hump_sigma_oct = float(hump_sigma_oct)
        if not (np.isfinite(hump_sigma_oct) and hump_sigma_oct > 0):
            raise ValueError("hump_sigma_oct must be positive")
    elif hump_fc is not None:
        raise ValueError("hump_fc is only used with noise='hump'")
    if tone_freq is None:
        if tnr_db != -np.inf:
            raise ValueError("tone_freq is required unless tnr_db is -inf (noise only)")
    else:
        tone_freq = float(tone_freq)
        if not (np.isfinite(tone_freq) and 0 < tone_freq < 0.5 * fs):
            raise ValueError("tone_freq must satisfy 0 < tone_freq < fs/2")
        k_top = int(np.flatnonzero(w > 0).max()) + 1
        if not k_top * tone_freq * 2.0 ** (0.5 * bw) < 0.5 * fs:
            raise ValueError(
                "the highest partial (including its band edge) must lie below fs/2"
            )

    with np.errstate(over="ignore"):
        w_tone = 1.0 / (1.0 + 10.0 ** (-tnr_db / 10.0))
        w_noise = 1.0 / (1.0 + 10.0 ** (tnr_db / 10.0))
    power = level_rms**2

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**63 - 1, size=2, dtype=np.int64)
    rng_tone, rng_noise = (np.random.default_rng(int(s)) for s in seeds)
    t = np.arange(n) / fs
    freqs = np.arange(n // 2 + 1) * (fs / n)

    tone_u = noise_u = None
    phases = None
    if tone_freq is not None and w_tone > 0.0:
        if bw == 0.0:
            phases = rng_tone.uniform(0.0, _TWO_PI, size=w.size)
            v = np.zeros(n)
            for k in np.flatnonzero(w > 0):
                v += w[k] * np.cos(_TWO_PI * (k + 1) * tone_freq * t + phases[k])
        else:
            mag2 = np.zeros(freqs.size)
            for k in np.flatnonzero(w > 0):
                fc = (k + 1) * tone_freq
                band = (freqs >= fc * 2.0 ** (-0.5 * bw)) & (freqs <= fc * 2.0 ** (0.5 * bw))
                if band.sum() < 2:
                    raise ValueError(
                        f"the band of partial {k + 1} holds fewer than 2 spectral bins; "
                        "widen tone_bandwidth_oct or lengthen the record"
                    )
                mag2[band] += w[k] ** 2 / band.sum()
            v = _random_phase_signal(rng_tone, n, np.sqrt(mag2))
        tone_u = _unit_rms(v)
    if w_noise > 0.0:
        mag = np.ones(freqs.size)
        if noise == "hump":
            mag[1:] = np.exp(-((np.log2(freqs[1:] / hump_fc) / hump_sigma_oct) ** 2) / 4.0)
            if not np.sum(mag[1:-1] ** 2) > 0:
                raise ValueError("the hump is narrower than the frequency resolution")
        noise_u = _unit_rms(_random_phase_signal(rng_noise, n, mag))
    if tone_u is not None and noise_u is not None:
        noise_u = _unit_rms(noise_u - (noise_u @ tone_u) / (tone_u @ tone_u) * tone_u)

    tone = np.zeros(n) if tone_u is None else np.sqrt(power * w_tone) * tone_u
    noise_stem = np.zeros(n) if noise_u is None else np.sqrt(power * w_noise) * noise_u
    x = tone + noise_stem

    config = {
        "rota_order_version": __version__,
        "seed": int(seed) if isinstance(seed, (int, np.integer)) else None,
        "fs": fs,
        "duration": duration,
        "n": int(n),
        "level_rms": level_rms,
        "tnr_db": _json_float(tnr_db),
        "tnr_definition": "broadband power ratio of the tone and noise stems (not ECMA-74 critical-band TNR)",
        "tone_power": power * w_tone,
        "noise_power": power * w_noise,
        "tone_freq": tone_freq,
        "harmonics": w.tolist(),
        "tone_bandwidth_oct": bw,
        "tone_phases": None if phases is None else phases.tolist(),
        "noise": noise,
        "hump_fc": hump_fc,
        "hump_sigma_oct": hump_sigma_oct if noise == "hump" else None,
    }
    return ToneNoiseSignal(fs=fs, t=t, x=x, tone=tone, noise=noise_stem, config=_py(config))


# ==========================================================================
# Scenarios: complete synthetic recordings of a drive train
# ==========================================================================

SCENARIO_KINDS = ("ice_runup", "ev_runup", "steady_cruise")
_POWERTRAINS = ("ice", "ev")

_SPL_RE_1PA = 20.0 * np.log10(20e-6)  # 20 uPa in dB re 1 Pa (-93.98 dB)
_LAW_GRID = np.geomspace(50.0, 25000.0, 55)  # rpm grid on which speed laws are tabulated
_AMP_DB_PER_DOUBLING = 20.0 * np.log10(2.0)  # amplitude dB of a doubling
_PWR_DB_PER_DOUBLING = 10.0 * np.log10(2.0)  # power dB of a doubling (the exponent unit of a PSD)
# Names of the independent random streams: adding a component never changes the others.
_STREAMS = (
    "phases", "ripple", "orders", "combustion", "engine", "road", "rumble", "transients",
    "carrier", "tones", "cabin", "mix", "measurement",
)
_MEASUREMENT_KEYS = (
    "gain", "noise_floor_rms", "dc", "clip", "bits", "full_scale", "clock_drift_ppm", "seed",
)


def _speed_defaults(**kw):
    base = {
        "mode": "accel", "hold": 0.0, "start_kmh": 0.0, "accel_ms2": 3.0, "vmax_kmh": 200.0,
        "jerk_time": 0.6, "end_kmh": None, "ramp_time": None, "cruise_kmh": 90.0,
        "cruise_gear": 1,
    }
    base.update(kw)
    return base


def _cabin_defaults():
    return {
        "rt60": 0.1, "volume_m3": 3.5, "crossover_hz": None, "direct_delay": 0.005,
        "interchannel_delay": 0.0003, "ear_spacing_m": 0.17, "sound_speed": 343.0,
    }


def _transient_defaults():
    return [
        {"at": 0.42, "kind": "thump", "amplitude": 0.25, "decay": 0.06, "freq": 32.0},
        {"at": 0.78, "kind": "click", "amplitude": 0.04, "decay": 0.004},
    ]


def _tilt_block(level_db_spl, control, ref, slope, tilt, f_lo=200.0):
    """Receiver-level broadband noise: PSD ``f**b`` above ``f_lo``, level law in ``control``."""
    return {
        "level_db_spl": level_db_spl, "control": control, "ref": ref,
        "slope_db_per_doubling": slope, "tilt_db_per_octave": tilt, "f_lo": f_lo, "f_ref": 1000.0,
        "grain": 0.05,
    }


def _hump_block(level_db_spl, control, ref, slope, band_hz):
    """Receiver-level low-frequency hump: PSD Gaussian in log frequency, half power at ``band_hz``."""
    return {
        "level_db_spl": level_db_spl, "control": control, "ref": ref,
        "slope_db_per_doubling": slope, "band_hz": list(band_hz), "grain": 0.5,
    }


def _ice_defaults():
    def entry(name, order, level_db, slope=1.0):
        return {"name": name, "order": order, "level_db": level_db, "slope_db_per_krpm": slope}

    return {
        "duration": 18.0,
        "speed": _speed_defaults(
            mode="accel", start_kmh=15.49, accel_ms2=2.0, vmax_kmh=200.0, jerk_time=0.0,
            cruise_gear=5,
        ),
        "drive": {
            "gear_ratios": [3.5, 2.45, 1.71, 1.2, 0.84], "final_drive": 3.7,
            "wheel_circumference": 1.95, "start_gear": 2, "shift_rpm": 6000.0,
            "shift_speeds": None, "idle": 800.0, "redline": 6500.0, "shift_time": 0.4,
            "launch_slip": True, "launch_rpm": None,
        },
        "ripple": [{"order": 2.0, "amplitude": 0.003}, {"order": 4.0, "amplitude": 0.0008}],
        "load": 1.0,
        "orders": {
            "ref_rpm": 3600.0, "ref_amplitude": 0.37, "min_freq": 10.0, "min_freq_taper": 10.0,
            "list": [
                entry("firing", 2.0, 0.0, 3.0),
                entry("order_4", 4.0, -15.0),
                entry("order_6", 6.0, -24.0),
                entry("order_8", 8.0, -30.0),
                entry("order_3", 3.0, -25.0),
                entry("order_5", 5.0, -28.0),
                entry("order_7", 7.0, -32.0),
                entry("order_1", 1.0, -32.0),
                entry("order_0.5", 0.5, -38.0),
                entry("order_1.5", 1.5, -34.0),
                entry("order_2.5", 2.5, -36.0),
            ],
            "sidebands": None,
        },
        "combustion": {
            "per_rev": 2.0, "amplitude": 0.12, "ref_rpm": 3600.0, "slope_db_per_krpm": 1.0,
            "resonance_hz": 1900.0, "damping": 0.04, "jitter_rev": 0.004,
        },
        "engine_broadband": _tilt_block(75.0, "rpm", 3600.0, 3.0, -8.5),
        "road": _tilt_block(77.0, "speed", 46.5, 4.5, -8.5),
        "rumble": _hump_block(78.0, "speed", 46.5, 4.5, (20.0, 60.0)),
        "transients": _transient_defaults(),
        "path": {
            "gain": 1.0,
            "modes": [
                {"freq": 2200.0, "zeta": 0.4, "gain": 0.9744},
                {"freq": 45.0, "q": 3.0, "gain": 0.6746},
                {"freq": 175.0, "q": 4.0, "gain": 0.4256},
                {"freq": 80.0, "q": 6.0, "gain": -0.08},
            ],
        },
        "cabin": _cabin_defaults(),
        "tacho": {"ppr": 60, "jitter": 3e-6, "delay": 0.0015, "drop": 0.0, "high": 5.0},
    }


def _ev_defaults():
    def entry(order, level_db, slope):
        return {
            "name": f"motor_{order:g}", "order": order, "level_db": level_db,
            "slope_db_per_doubling": slope,
        }

    return {
        "duration": 10.0,
        "speed": _speed_defaults(
            mode="accel", start_kmh=2.0, accel_ms2=4.4, vmax_kmh=190.0, jerk_time=0.5
        ),
        "drive": {
            "gear_ratios": [12.0], "final_drive": 1.0, "wheel_circumference": 2.0,
            "start_gear": 1, "shift_rpm": None, "shift_speeds": [], "idle": 0.0,
            "redline": 15000.0, "shift_time": 0.0, "launch_slip": False, "launch_rpm": None,
        },
        "pole_pairs": 4,
        "ripple": [{"order": 12.0, "amplitude": 0.0004}],
        "load": 1.0,
        "motor": {
            "ref_rpm": 9000.0, "ref_amplitude": 0.0182,
            "list": [
                entry(2.0, 0.0, 5.0), entry(4.0, -6.0, 7.0), entry(8.0, -9.0, 6.0),
                entry(10.0, -12.0, 6.0), entry(12.0, -10.0, 2.0), entry(14.0, -15.0, 5.0),
            ],
        },
        "gear_mesh": {
            "teeth": 40, "level_db": -14.0, "slope_db_per_doubling": 3.0, "am_depth": 0.6,
            "am_freq": 0.5, "mesh2_order": 6.7, "mesh2_level_db": -8.0,
            "mesh2_slope_db_per_doubling": 5.0, "sidebands": None, "mod_order": 1.0,
            "modulation": "am",
        },
        "inverter": {
            "carrier_hz": 10000.0, "dither_hz": 800.0, "hop_s": [0.2, 0.5], "step_hz": 50.0,
            "tnr_db": 4.0, "bandwidth_hz": 20.0,
            "sidebands": [{"n": 2, "tnr_db": 3.0}, {"n": 4, "tnr_db": 2.0}],
        },
        "tones": [
            {"freq": 6300.0, "tnr_db": 10.0, "bandwidth_hz": 20.0},
            {"freq": 14200.0, "tnr_db": 6.0, "bandwidth_hz": 20.0},
        ],
        "road": _tilt_block(74.0, "speed", 90.0, 6.0, -9.0),
        "rumble": _hump_block(89.8, "speed", 90.0, 6.0, (25.0, 35.0)),
        "transients": _transient_defaults(),
        "path": {
            "gain": 1.0,
            "modes": [
                {"freq": 6000.0, "zeta": 0.5, "gain": 0.85},
                {"freq": 95.0, "q": 8.0, "gain": 0.10},
                {"freq": 1300.0, "q": 2.5, "gain": 0.25},
                {"freq": 2100.0, "q": 8.0, "gain": 0.12},
                {"freq": 3300.0, "q": 8.0, "gain": 0.10},
            ],
        },
        "cabin": _cabin_defaults(),
        "tacho": {"ppr": 48, "jitter": 1e-6, "delay": 0.0005, "drop": 0.0, "high": 5.0},
    }


def _cruise_defaults(powertrain="ice"):
    if powertrain not in _POWERTRAINS:
        raise ValueError(f"powertrain must be one of {_POWERTRAINS}, got {powertrain!r}")
    d = _ice_defaults() if powertrain == "ice" else _ev_defaults()
    d["duration"] = 10.0
    d["speed"].update(mode="constant", cruise_kmh=90.0, cruise_gear=5 if powertrain == "ice" else 1)
    d["load"] = 0.53 if powertrain == "ice" else 0.44
    d["transients"] = []
    d["powertrain"] = powertrain
    return d


def scenario_defaults(kind, powertrain="ice"):
    """Default parameters of a scenario kind, as a fresh nested ``dict``.

    The dictionary is what :func:`synth_scenario` starts from; its keyword
    overrides replace entries of it. Everything in it is a plain Python number,
    string, list, ``None`` or nested ``dict`` (JSON serialisable). The values are
    illustrative values typical of passenger cars, rounded; they are not
    measurements of any specific vehicle (see the Notes of
    :func:`synth_scenario` for what each block means).

    Parameters
    ----------
    kind : {"ice_runup", "ev_runup", "steady_cruise"}
        Scenario kind.
    powertrain : {"ice", "ev"}, default "ice"
        Only for ``steady_cruise``: the drive train whose settings the cruise
        uses (its dictionary also holds ``powertrain``).

    Returns
    -------
    dict
        Parameter blocks keyed by name. Amplitudes of tonal and impulsive
        components are peak values in Pa (peak convention) at the source,
        before the transfer path; levels of noise components are RMS sound
        pressure levels in dB re 20 uPa (``level_db_spl``) at the receiver.

    Raises
    ------
    ValueError
        For an unknown kind or powertrain.

    References
    ----------
    Heywood, J. B. (1988). Internal Combustion Engine Fundamentals. McGraw-Hill.
    Gieras, J. F., Wang, C., & Lai, J. C. (2006). Noise of Polyphase Electric
    Motors. CRC Press.
    """
    if kind not in SCENARIO_KINDS:
        raise ValueError(f"kind must be one of {SCENARIO_KINDS}, got {kind!r}")
    if kind == "steady_cruise":
        return _cruise_defaults(powertrain)
    return _ice_defaults() if kind == "ice_runup" else _ev_defaults()


def _merge_overrides(cfg, overrides, where="scenario"):
    """Replace entries of ``cfg`` in place; nested dicts merge key by key."""
    for key, val in overrides.items():
        if key not in cfg:
            raise ValueError(f"unknown parameter {key!r} in {where}; valid: {sorted(cfg)}")
        if isinstance(cfg[key], dict) and isinstance(val, Mapping):
            _merge_overrides(cfg[key], val, f"{where}.{key}")
        else:
            cfg[key] = copy.deepcopy(val)


def _scenario_streams(seed):
    """Integer seed to record (None for a generator) and the named random streams."""
    if isinstance(seed, np.random.Generator):
        base, used = int(seed.integers(0, 2**63 - 1)), None
    elif seed is None:
        base = int(np.random.default_rng().integers(0, 2**63 - 1))
        used = base
    elif isinstance(seed, (int, np.integer)) and not isinstance(seed, bool):
        base = used = int(seed)
    else:
        raise TypeError("seed must be an int, a numpy.random.Generator or None")
    children = np.random.SeedSequence(base).spawn(len(_STREAMS))
    return used, {name: np.random.default_rng(c) for name, c in zip(_STREAMS, children)}


def _block_seed(rng):
    return int(rng.integers(0, 2**63 - 1))


def _speed_profile(t, sp):
    """Vehicle speed in km/h: drag-limited acceleration, a linear ramp or a constant speed.

    ``accel``: ``dv/dt = a0 w(t) (1 - (v / vmax)**2)`` with a jerk-limiting ramp
    ``w`` (a raised cosine from 0 to 1 over ``jerk_time`` after ``hold``); with
    ``S(t)`` the integral of ``w`` the solution is closed form,
    ``v = vmax tanh(atanh(v0 / vmax) + a0 S / vmax)``. ``ramp``: linear from
    ``start_kmh`` to ``end_kmh`` after ``hold`` over ``ramp_time`` (the coast
    variant). ``constant``: ``cruise_kmh``.
    """
    mode = sp["mode"]
    if mode == "constant":
        v = float(sp["cruise_kmh"])
        if not (np.isfinite(v) and v >= 0):
            raise ValueError("speed.cruise_kmh must be a non-negative speed in km/h")
        return np.full(t.shape, v)
    hold = float(sp["hold"])
    if mode == "ramp":
        v0, v1 = float(sp["start_kmh"]), sp["end_kmh"]
        if v1 is None:
            raise ValueError("speed.mode='ramp' needs end_kmh")
        T = float(sp["ramp_time"]) if sp["ramp_time"] is not None else float(t[-1] - hold)
        if not (T > 0 and hold >= 0 and np.isfinite(v0) and np.isfinite(float(v1))):
            raise ValueError("speed ramp needs ramp_time > 0, hold >= 0 and finite speeds")
        u = np.clip((t - hold) / T, 0.0, 1.0)
        return v0 + (float(v1) - v0) * u
    if mode != "accel":
        raise ValueError(f"speed.mode must be 'accel', 'ramp' or 'constant', got {mode!r}")
    a0 = float(sp["accel_ms2"])
    vmax = float(sp["vmax_kmh"]) / 3.6
    v0 = float(sp["start_kmh"]) / 3.6
    tj = float(sp["jerk_time"])
    ok = all(np.isfinite(v) for v in (a0, vmax, v0, tj, hold))
    if not (ok and a0 > 0 and vmax > 0 and 0 <= v0 < vmax and tj >= 0 and hold >= 0):
        raise ValueError(
            "speed needs accel_ms2 > 0, vmax_kmh > start_kmh >= 0, jerk_time >= 0 and hold >= 0"
        )
    tau = np.maximum(t - hold, 0.0)
    if tj > 0:
        ramp = 0.5 * tau - tj * np.sin(np.pi * np.minimum(tau, tj) / tj) / (2.0 * np.pi)
        s = np.where(tau < tj, ramp, 0.5 * tj + (tau - tj))
    else:
        s = tau
    return 3.6 * vmax * np.tanh(np.arctanh(v0 / vmax) + a0 * s / vmax)


def _drive(t, speed_kmh, cfg):
    """Mean engine or motor speed from the vehicle speed; returns (drive dict, shift speeds)."""
    d = cfg["drive"]
    ratios = np.asarray(d["gear_ratios"], dtype=float)
    chain = dict(
        wheel_circumference=d["wheel_circumference"], gear_ratios=ratios, final_drive=d["final_drive"]
    )
    k = np.array([float(speed_to_rpm(1.0, gear=g, **chain)) for g in range(1, ratios.size + 1)])
    shift = d.get("shift_speeds")
    if shift is None:
        if ratios.size > 1:
            if d.get("shift_rpm") is None:
                raise ValueError("drive needs shift_rpm or shift_speeds for several gears")
            shift = np.asarray(float(d["shift_rpm"]) / k[:-1])
            g0 = int(d.get("start_gear", 1))
            if not 1 <= g0 <= ratios.size:
                raise ValueError(f"drive.start_gear must be in 1..{ratios.size}")
            if g0 > 1:  # already in gear g0 at the start: lower thresholds below the start speed
                shift[: g0 - 1] = 0.9 * float(speed_kmh[0]) * np.arange(1, g0) / g0
            shift = shift.tolist()
        else:
            shift = []
    shift = [float(s) for s in shift]
    sp = cfg["speed"]
    if sp["mode"] != "accel":
        if sp["mode"] == "constant":
            g = int(sp["cruise_gear"])
            if not 1 <= g <= ratios.size:
                raise ValueError(f"speed.cruise_gear must be in 1..{ratios.size}")
            gear = np.full(t.shape, g, dtype=int)
            rpm = speed_to_rpm(speed_kmh, gear=g, **chain)
        else:  # ramp: gears by the shift speeds, static relation
            gear = np.searchsorted(np.asarray(shift), speed_kmh, side="right") + 1
            rpm = speed_to_rpm(speed_kmh, gear=gear, **chain)
        return {
            "rpm": np.clip(rpm, d["idle"], d["redline"]), "gear": gear, "shift_times": np.empty(0),
            "shift_from": np.empty(0, dtype=int), "shift_to": np.empty(0, dtype=int),
            "in_shift": np.zeros(t.shape, dtype=bool), "slip": np.zeros(t.shape, dtype=bool),
            "launch_times": np.empty((0, 2)),
        }, shift
    out = drive_rpm(
        t, speed_kmh, gear_ratios=ratios, final_drive=d["final_drive"],
        wheel_circumference=d["wheel_circumference"], shift_speeds=shift, idle=d["idle"],
        redline=d["redline"], shift_time=d["shift_time"], launch_slip=d["launch_slip"],
        launch_rpm=d["launch_rpm"],
    )
    return out, shift


def _ripple_terms(entries, rng):
    terms = []
    for e in entries:
        phase = float(rng.uniform(0.0, _TWO_PI))
        terms.append((float(e["order"]), float(e["amplitude"]), phase))
    return terms


def _order_law(entry, ref_rpm):
    """Gain law of one order on the rpm grid: dB per 1000 rpm or dB per doubling of speed."""
    if entry.get("slope_db_per_doubling") is not None:
        g = (_LAW_GRID / ref_rpm) ** (float(entry["slope_db_per_doubling"]) / _AMP_DB_PER_DOUBLING)
    else:
        slope = float(entry.get("slope_db_per_krpm", 0.0))
        g = 10.0 ** (slope * (_LAW_GRID - ref_rpm) / 1000.0 / 20.0)
    return _LAW_GRID.copy(), g


def _transient_list(entries, duration):
    out = []
    for e in entries:
        e = dict(e)
        at = e.pop("at", None)
        if e.get("time") is None:
            if at is None:
                raise ValueError("a transient needs 'time' (s) or 'at' (fraction of the duration)")
            e["time"] = float(at) * duration
        out.append({k: v for k, v in e.items() if v is not None})
    return out


# --- receiver-level noise with a stated spectral shape -----------------------------------


def _noise_shape(block):
    """Shape function of a noise block: a tilted band above ``f_lo`` or a hump over ``band_hz``.

    The hump is Gaussian in log frequency (as the hump noise of :func:`synth_tone_noise`),
    ``exp(-0.5 (log2(f / f0) / sigma)**2)``, with the half-power points of the PSD at
    the two band edges, ``f0 = sqrt(lo hi)``; it falls off faster than any power law,
    so it does not add a skirt to the tilted broadband above it.
    """
    if block.get("band_hz") is not None:
        lo, hi = (float(v) for v in block["band_hz"])
        if not 0 < lo < hi:
            raise ValueError("band_hz must satisfy 0 < lo < hi")
        f0 = np.sqrt(lo * hi)
        sigma = 0.5 * np.log2(hi / lo) / np.sqrt(2.0 * np.log(2.0))

        def hump(f):
            f = np.asarray(f, dtype=float)
            out = np.zeros(f.shape)
            pos = f > 0
            out[pos] = np.exp(-0.5 * (np.log2(f[pos] / f0) / sigma) ** 2)
            return out

        return hump, 0.25 * lo
    b = float(block["tilt_db_per_octave"]) / _PWR_DB_PER_DOUBLING
    f_lo, f_ref = float(block["f_lo"]), float(block["f_ref"])
    if not 0 < f_lo:
        raise ValueError("f_lo must be positive")

    def tilt(f):
        f = np.asarray(f, dtype=float)
        out = np.zeros(f.shape)
        edge = np.clip((f / f_lo - 0.8) / 0.2, 0.0, 1.0)  # raised-cosine start from 0.8 f_lo
        pos = f > 0
        out[pos] = (f[pos] / f_ref) ** b * 0.5 * (1.0 - np.cos(np.pi * edge[pos]))
        return out

    return tilt, 0.5 * f_lo


def _noise_model(block, fs):
    """PSD model of a noise block: the shape scaled to the RMS level at the reference.

    The expected mean square of the synthesised noise is computed on the bins of
    one grain, as ``synthesize_broadband`` lays them out, so that the realised
    RMS equals the stated level in expectation.
    """
    grain = float(block["grain"])
    length = 2 * round(grain * fs / 2.0)
    if length < 8:
        raise ValueError("noise grain is shorter than 8 samples; increase grain or fs")
    hop = length // 2
    df = fs / length
    shape, hp = _noise_shape(block)
    f_bins = np.arange(hop + 1) * df
    w = (f_bins >= hp).astype(float)
    w[0] *= 0.5
    w[-1] *= 0.5
    p0 = float(np.sum(w * shape(f_bins)) * df)
    if not p0 > 0:
        raise ValueError("the noise band lies outside the frequency range of the grain")
    rms = 20e-6 * 10.0 ** (float(block["level_db_spl"]) / 20.0)
    return {
        "s0": rms**2 / p0, "shape": shape, "hp": hp, "grain": grain, "hop": hop, "df": df,
        "ref": float(block["ref"]), "slope": float(block["slope_db_per_doubling"]),
        "rms_ref": rms,
    }


def _noise_psd(model, f):
    """PSD in Pa**2/Hz of a noise model at its reference condition."""
    return model["s0"] * model["shape"](f)


def _render_noise(model, fs, n, control, seed):
    """One channel of noise whose level follows ``control`` by the stated power law."""
    f_grid = np.arange(4 * model["hop"] + 1) * (model["df"] / 4.0)
    base = _noise_psd(model, f_grid)
    slope = model["slope"]
    control = np.asarray(control, dtype=float)
    if slope == 0.0 or not float(control.max()) > 0.0:
        psd, ctrl, axis = base, None, None
        if slope != 0.0:
            return np.zeros(n)
    else:
        axis = _speed_rows(control, slope)
        kappa = slope / _PWR_DB_PER_DOUBLING
        with np.errstate(divide="ignore"):
            gain = np.where(axis > 0.0, (axis / model["ref"]) ** kappa, 0.0)
        psd, ctrl = np.outer(gain, base), control
    return synthesize_broadband(
        psd, f_grid, fs, n, control=ctrl, control_axis=axis, grain=model["grain"],
        highpass=model["hp"], seed=seed,
    )


def _ear_mix(w, fs, spacing, c, taps=513, nfft=4096):
    """Give the columns of independent noise ``w`` the diffuse-field coherence of receivers.

    The coherence between two receivers a distance ``d`` apart in a diffuse sound
    field is ``sinc(2 pi f d / c)`` (Cook et al. 1955; Jacobsen & Roisin 2000):
    close to 1 at low frequency and first zero at ``c / (2 d)``, 1 kHz for two ears
    17 cm apart. Each output channel is a frequency-dependent mixture of the
    independent inputs with the symmetric square root of the coherence matrix, so
    every channel keeps its own power spectrum.
    """
    n_ch = w.shape[1]
    if n_ch == 1 or spacing is None or float(spacing) <= 0.0:
        return w
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    idx = np.abs(np.subtract.outer(np.arange(n_ch), np.arange(n_ch)))
    gamma = np.sinc(2.0 * f[:, None, None] * float(spacing) * idx[None] / float(c))
    lam, vec = np.linalg.eigh(gamma)
    root = np.einsum("fik,fk,fjk->fij", vec, np.sqrt(np.maximum(lam, 0.0)), vec)
    win = np.hanning(taps + 2)[1:-1]
    mid = taps // 2
    out = np.zeros_like(w)
    for i in range(n_ch):
        for j in range(n_ch):
            h = np.fft.fftshift(np.fft.irfft(root[:, i, j], nfft))
            h = h[nfft // 2 - mid : nfft // 2 + mid + 1] * win
            out[:, i] += signal.oaconvolve(w[:, j], h, mode="same")
    return out


def _cabin_ir(fs, cabin, n_channels, seed):
    """Cabin impulse response: a smooth direct response below, a diffuse tail above the crossover.

    A small cabin is not a diffuse field at low frequency: below the Schroeder
    frequency ``2000 sqrt(T60 / V)`` (about 340 Hz for 0.1 s and 3.5 m**3) the
    response is smooth, and the resonances are those of the modal path. The
    impulse response is therefore the sum of a low-passed direct arrival at
    ``direct_delay + c * interchannel_delay`` (unit gain below the crossover, so the
    channels differ there only by the delay) and the high-passed part of a decaying-
    noise tail (``room_impulse_response``, mean-square response 1 above the
    crossover, independent realisations per channel given the diffuse-field
    coherence ``sinc(2 pi f d / c)`` between channels, ``_ear_mix``). The low-pass
    is a Hann-windowed sinc and the high-pass is the complement of the same
    low-pass, so the two parts add up without a gap or overlap.
    """
    rt60 = float(cabin["rt60"])
    f_x = cabin.get("crossover_hz")
    f_x = 2000.0 * np.sqrt(rt60 / float(cabin["volume_m3"])) if f_x is None else float(f_x)
    if not 0.0 < f_x < 0.25 * fs:
        raise ValueError("the cabin crossover (Schroeder) frequency must lie between 0 and fs/4")
    tail, tcfg = room_impulse_response(
        fs, rt60, n_channels=n_channels, direct_delay=cabin["direct_delay"],
        interchannel_delay=cabin["interchannel_delay"], seed=seed,
    )
    tail2 = tail[:, None] if tail.ndim == 1 else tail
    tail2 = _ear_mix(tail2, fs, cabin.get("ear_spacing_m"), cabin.get("sound_speed", 343.0))
    half = int(round(1.5 * fs / f_x))
    k = np.arange(-half, half + 1)
    lp = np.sinc(2.0 * f_x * k / fs) * np.hanning(2 * half + 3)[1:-1]
    lp /= lp.sum()  # unit gain at DC
    high = tail2 - signal.oaconvolve(tail2, lp[:, None], axes=0, mode="same")
    delays = np.asarray(tcfg["delays"], dtype=float)
    x = np.arange(tail2.shape[0])[:, None] / fs - delays[None, :]  # time from the arrival
    w = np.where(np.abs(x) < half / fs, 0.5 * (1.0 + np.cos(np.pi * x * fs / half)), 0.0)
    direct = np.sinc(2.0 * f_x * x) * w
    direct /= direct.sum(axis=0)
    h = direct + high
    return (h[:, 0] if n_channels == 1 else h), {**tcfg, "crossover_hz": f_x}


# --- dithered inverter carrier ---------------------------------------------------------


def _dither_track(n, fs, inv, rng):
    """Piecewise-constant carrier frequency track: random hops within ``fc +- dither``."""
    fc, dith = float(inv["carrier_hz"]), float(inv["dither_hz"])
    lo, hi = (float(v) for v in inv["hop_s"])
    step = float(inv["step_hz"])
    track = np.full(n, fc)
    hops = [0]
    freqs = [fc]
    i = 0
    while True:
        i += max(1, int(round(rng.uniform(lo, hi) * fs)))
        if i >= n:
            break
        f = fc + dith * rng.uniform(-1.0, 1.0)
        f = fc + step * np.round((f - fc) / step) if step > 0 else f
        track[i:] = f
        hops.append(i)
        freqs.append(f)
    return track, np.array(hops) / fs, np.array(freqs)


def _dithered_carrier(rev, fs, rpm, fc_track, terms, rng):
    """Components ``A_j cos(Phi_c + 2 pi q_j rev + phi_j)`` around a hopping carrier.

    ``Phi_c`` is the integral of the carrier frequency track ``fc(t)`` (phase
    continuous across the hops). The instantaneous frequency of component ``j`` is
    ``fc(t) + q_j rpm / 60`` (muted outside 0 and fs/2, never clipped), the same
    truth conventions as :func:`carrier_components`.
    """
    q = np.array([tq for tq, _ in terms], dtype=float)
    amp = np.array([a for _, a in terms], dtype=float)
    phase0 = rng.uniform(0.0, _TWO_PI, size=q.size)
    big_phi = _TWO_PI * (np.cumsum(fc_track) - fc_track) / fs
    phase = big_phi[:, None] + _TWO_PI * np.outer(rev, q) + phase0[None, :]
    freq = fc_track[:, None] + np.outer(rpm / 60.0, q)
    amplitude = np.where((freq > 0.0) & (freq < 0.5 * fs), amp[None, :], 0.0)
    stems = amplitude * np.cos(phase)
    return {
        "x": stems.sum(axis=1), "q": q, "frequency": freq, "amplitude": amplitude,
        "phase": phase, "phase0": phase0, "carrier_hz": fc_track,
    }


def _check_measurement(measurement):
    if measurement is None:
        return None
    if not isinstance(measurement, Mapping):
        raise TypeError("measurement must be a dict of measurement_chain arguments or None")
    extra = sorted(set(measurement) - set(_MEASUREMENT_KEYS))
    if extra:
        raise ValueError(f"unknown measurement key(s) {extra}; allowed: {list(_MEASUREMENT_KEYS)}")
    return dict(measurement)


@dataclass
class ScenarioSignal:
    """A complete synthetic recording of a drive train with its truth.

    Units: time in s, frequency in Hz, speed in rpm, shaft angle in rev, vehicle
    speed in km/h, sound pressure in Pa. Tonal and impulsive components carry
    peak amplitudes, noise components RMS levels. ``n`` is the number of
    samples of the truth arrays, ``ch`` the number of channels, ``K`` the number
    of source orders.

    Attributes
    ----------
    kind : str
        Scenario kind.
    fs : float
        Sampling rate in Hz.
    t : ndarray, shape (n,)
        Time axis of the truth arrays (``arange(n) / fs``) on the true clock.
    x : ndarray, shape (n_out,) or (n_out, ch)
        The recording: ``x_clean`` passed through the measurement chain. With an
        ideal recorder (``measurement=None``) ``x`` equals ``x_clean`` and
        ``n_out == n``; only a sampling-clock drift changes the length.
    x_clean : ndarray, shape (n,) or (n, ch)
        The sum of ``stems`` before the measurement chain, accumulated in the
        order of ``stems`` (``stem_sum()`` reproduces it bit for bit).
    t_true : ndarray, shape (n_out,)
        True time in s of every sample of ``x``; equals ``t`` without clock drift.
    clipped : ndarray of bool, same shape as ``x``
        Samples limited by the hard clip or the quantiser; all False for an
        ideal recorder.
    rpm : ndarray, shape (n,)
        True instantaneous engine or motor speed, including the angle-periodic
        speed ripple, ``60 d rev / dt`` at the samples.
    rpm_mean : ndarray, shape (n,)
        Speed without the ripple (from the vehicle speed through the driveline,
        with launch slip and shift transients).
    rev : ndarray, shape (n,)
        True shaft angle in revolutions (exact, from the ripple model). The
        source orders are rendered with the trapezoidal angle of ``rpm``
        (``rota_order.speed.rpm_to_phase``), which differs from ``rev`` by
        ``config["resolved"]["rev_deviation_max"]`` rev.
    speed_kmh : ndarray, shape (n,)
        Vehicle speed in km/h.
    gear : ndarray of int, shape (n,)
        Engaged gear (1-based; switches to the new gear at the shift time).
    tacho : ndarray, shape (n,) or None
        The recorded tachometer channel, delayed by ``tacho_delay`` (the tacho
        lags the audio). It is not part of ``x`` and not affected by the
        measurement chain (it has its own clock).
    pulse_times : ndarray
        Time in s of every pulse in ``tacho`` (rising-edge 50 % crossing; jitter
        and delay included). Empty without a tachometer.
    pulse_revs : ndarray
        Nominal shaft angle ``k / ppr`` of every pulse, in rev.
    stems : dict of ndarray
        Named components as they reach the receiver, each of the shape of
        ``x_clean``. ``x_clean`` is their sum (bit for bit, in dict order).
        Components that are off (parameter block ``None``) are absent. ICE
        (and an ICE cruise): ``orders``, ``combustion``, ``transients`` (through
        the path), then ``engine_broadband``, ``road``, ``rumble`` (generated at
        the receiver). EV (and an EV cruise): ``motor``, ``gear``, ``inverter``,
        ``tones``, ``transients`` (through the path), then ``road``, ``rumble``.
    source_stems : dict of ndarray
        The components that pass the transfer path, before it, each of shape
        (n,): the first group of ``stems`` (the broadband noises ``engine_broadband``,
        ``road`` and ``rumble`` are generated at the receiver and have no source
        stem).
    orders : ndarray, shape (K,)
        Source order numbers (cycles per revolution), from ``synth_runup``.
    order_names : list of str
        Name of each source order. ICE: ``firing`` (order 2), ``order_4``,
        ``order_6``, ``order_8``, ``order_3``, ``order_5``, ``order_7``,
        ``order_1``, ``order_0.5``, ``order_1.5``, ``order_2.5``. EV:
        ``motor_2``, ``motor_4``, ``motor_8``, ``motor_10``, ``motor_12``,
        ``motor_14`` (stem ``motor``), ``gear_mesh_40`` (the whine, ``gear_mesh.teeth``)
        and ``gear_mesh_6.7`` (stem ``gear``).
    order_stems, order_amplitude, order_phase, order_frequency : ndarray, shape (n, K)
        Truth of the source orders before the path (``rota_order.synth.SynthSignal``
        conventions): rendered order, peak envelope, total phase in rad and
        instantaneous frequency in Hz.
    ir : ndarray, shape (m,) or (m, ch)
        Cabin impulse response: a smooth low-passed direct arrival (unit gain below
        the Schroeder frequency) plus the high-passed diffuse tail (mean-square
        response 1 above it); see the Notes of :func:`synth_scenario`.
    events : dict
        ``shift_times``, ``shift_from``, ``shift_to``, ``in_shift``, ``slip``,
        ``launch_times`` (from ``drive_rpm``; empty when the record starts in
        gear), ``impulse_times``, ``impulse_rev`` (combustion events, ICE),
        ``carrier_hop_times``, ``carrier_hop_hz`` (EV) and ``transients`` (the
        resolved transient events).
    carrier : dict or None
        Truth of the inverter components (EV, source level): ``q`` (order
        offsets, 0 for the carrier), ``frequency`` (n, J) in Hz, ``amplitude``
        (n, J) peak, ``phase`` (n, J) in rad, ``phase0`` and ``carrier_hz`` (n,),
        the dithered carrier frequency track. The hop times and frequencies are
        in ``events["carrier_hop_times"]`` and ``events["carrier_hop_hz"]``.
    config : dict
        JSON-serialisable record: the resolved parameters, the seed, the rota-order
        version, the configuration of every building block and of the
        measurement chain.
    """

    kind: str
    fs: float
    t: np.ndarray
    x: np.ndarray
    x_clean: np.ndarray
    t_true: np.ndarray
    clipped: np.ndarray
    rpm: np.ndarray
    rpm_mean: np.ndarray
    rev: np.ndarray
    speed_kmh: np.ndarray
    gear: np.ndarray
    tacho: np.ndarray | None
    pulse_times: np.ndarray
    pulse_revs: np.ndarray
    stems: dict
    source_stems: dict
    orders: np.ndarray
    order_names: list
    order_stems: np.ndarray
    order_amplitude: np.ndarray
    order_phase: np.ndarray
    order_frequency: np.ndarray
    ir: np.ndarray
    events: dict
    carrier: dict | None
    config: dict

    @property
    def n(self):
        """Number of samples of the truth arrays."""
        return int(self.t.size)

    @property
    def n_channels(self):
        """Number of audio channels of ``x``."""
        return 1 if self.x.ndim == 1 else int(self.x.shape[1])

    def stem_sum(self, names=None):
        """Sum of stems, accumulated in the order of ``stems``.

        With ``names=None`` all stems are summed and the result equals
        ``x_clean`` bit for bit. A subset gives a partial mix (for example
        everything except the engine orders). Amplitude convention: the stems
        are summed as they are (peak for tonal, RMS level for noise).

        Parameters
        ----------
        names : sequence of str, optional
            Stems to include.

        Returns
        -------
        ndarray
            The mix, of the shape of the stems, in Pa.

        References
        ----------
        None; a plain sum of the stems.
        """
        chosen = [k for k in self.stems if names is None or k in names]
        if names is not None:
            missing = sorted(set(names) - set(self.stems))
            if missing:
                raise ValueError(f"unknown stem(s) {missing}; available: {list(self.stems)}")
        out = None
        for k in chosen:
            out = self.stems[k].copy() if out is None else out + self.stems[k]
        return np.zeros(self.x_clean.shape) if out is None else out

    def through_path(self, source):
        """Pass a source-level signal through the transfer path and the cabin.

        The modal path of the scenario (``modal_path``) followed by the cabin
        impulse response (``apply_ir``): exactly what each of the
        ``source_stems`` went through to become a stem. Use it for the stem of
        a single order, for example ``through_path(order_stems[:, k])``.

        Parameters
        ----------
        source : array_like, shape (n,)
            Signal at the source (Pa, any amplitude convention; the path is
            linear).

        Returns
        -------
        ndarray, shape (n,) or (n, ch)
            Signal at the receiver in the unit of ``source``.

        References
        ----------
        Ewins, D. J. (2000). Modal Testing: Theory, Practice and Application,
        2nd ed. Research Studies Press.
        Kuttruff, H. (2009). Room Acoustics, 5th ed. Spon Press.
        """
        p = self.config["parameters"]["path"]
        y, _ = modal_path(np.asarray(source, dtype=float), self.fs, p["modes"], gain=p["gain"])
        return apply_ir(y, self.ir)

    def path_response(self, f, channel=0):
        """Complex frequency response of modal path times cabin impulse response.

        The steady-state gain that a stationary sinusoid of frequency ``f``
        meets between the source and the receiver ``channel``: the digital
        modal response (``modal_response`` with ``fs``, times the path gain)
        times the discrete-time Fourier transform of the impulse response. The
        ratio is the same for peak and RMS amplitudes. The impulse response is
        evaluated directly (cost proportional to ``f.size * ir.shape[0]``).

        Parameters
        ----------
        f : array_like
            Frequencies in Hz.
        channel : int, default 0
            Receiver channel.

        Returns
        -------
        ndarray of complex, shape of ``f``
            Dimensionless transfer function.

        References
        ----------
        Ewins, D. J. (2000). Modal Testing: Theory, Practice and Application,
        2nd ed. Research Studies Press.
        Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal
        Processing, 3rd ed. Pearson.
        """
        f = np.asarray(f, dtype=float)
        p = self.config["parameters"]["path"]
        h = self.ir if self.ir.ndim == 1 else self.ir[:, int(channel)]
        flat = f.ravel()
        k = np.arange(h.size)
        resp = np.empty(flat.size, dtype=complex)
        step = max(1, int(2_000_000 // h.size))
        for s0 in range(0, flat.size, step):
            fr = flat[s0 : s0 + step]
            resp[s0 : s0 + step] = np.exp(-2j * np.pi * np.outer(fr / self.fs, k)) @ h
        return p["gain"] * modal_response(f, p["modes"], self.fs) * resp.reshape(f.shape)

    def quasi_static_amplitude(self, k, channel=0, step=0.005):
        """Peak amplitude of source order ``k`` at the receiver, quasi-statically.

        ``order_amplitude[:, k] * |H(f_k(t))|`` with ``H`` from
        :meth:`path_response` and ``f_k`` the instantaneous order frequency:
        what the order would read if the path responded instantly to the
        sweep. It neglects the memory of the resonances (ring-down), which
        matters where an order sweeps through a mode faster than the mode's
        bandwidth squared. ``|H|`` is evaluated every ``step`` seconds and
        interpolated linearly.

        Parameters
        ----------
        k : int
            Column of the source orders.
        channel : int, default 0
            Receiver channel.
        step : float, default 0.005
            Evaluation interval of ``|H|`` in s.

        Returns
        -------
        ndarray, shape (n,)
            Peak amplitude in Pa (peak convention).

        References
        ----------
        Brandt, A. (2011). Noise and Vibration Analysis, ch. 5-6, 12. Wiley.
        """
        tg = np.unique(np.concatenate((np.arange(0.0, self.t[-1], float(step)), self.t[-1:])))
        fg = np.interp(tg, self.t, self.order_frequency[:, int(k)])
        gain = np.abs(self.path_response(fg, channel))
        return self.order_amplitude[:, int(k)] * np.interp(self.t, tg, gain)

    def at_recording(self, a):
        """Resample a truth array of the true grid onto the recorded samples.

        With a clock drift the recording ``x`` has ``n_out`` samples at the true
        times ``t_true``; this interpolates a truth array of length ``n``
        (``rpm``, ``rev``, ``speed_kmh``, ...) linearly there. Without drift it
        returns the array unchanged.

        Parameters
        ----------
        a : array_like, shape (n,) or (n, k)
            Truth array on ``t``.

        Returns
        -------
        ndarray, shape (n_out,) or (n_out, k)
            Values at ``t_true`` (same unit as ``a``).

        References
        ----------
        None; linear interpolation.
        """
        a = np.asarray(a, dtype=float)
        if a.shape[0] != self.t.size:
            raise ValueError(f"a must have {self.t.size} samples along the first axis")
        if self.t_true.size == self.t.size and np.array_equal(self.t_true, self.t):
            return a.copy()
        if a.ndim == 1:
            return np.interp(self.t_true, self.t, a)
        return np.column_stack([np.interp(self.t_true, self.t, a[:, j]) for j in range(a.shape[1])])

    def save(self, path):
        """Write the arrays to ``<path>.npz`` and the configuration to ``<path>.json``.

        The NPZ holds ``fs``, ``t``, ``x``, ``x_clean``, ``t_true``,
        ``clipped``, ``rpm``, ``rpm_mean``, ``rev``, ``speed_kmh``, ``gear``,
        ``orders``, ``order_stems``, ``order_amplitude``, ``order_phase``,
        ``order_frequency``, ``ir``, ``pulse_times``, ``pulse_revs``, ``tacho``
        (if present), ``stem_<name>`` and ``source_<name>`` for every stem, the
        array-valued entries of ``events`` as ``event_<name>``, ``carrier_<name>``
        (EV) and the configuration as the JSON string ``config_json``. A
        trailing ``.npz`` or ``.json`` on ``path`` is ignored.

        Returns
        -------
        (Path, Path)
            The NPZ and JSON paths.

        References
        ----------
        rota-order ROADMAP, design principle 7 (reproducibility sidecar).
        """
        p = Path(path)
        if p.suffix in (".npz", ".json"):
            p = p.with_suffix("")
        npz = p.with_name(p.name + ".npz")
        sidecar = p.with_name(p.name + ".json")
        text = json.dumps(self.config, indent=2)
        arrays = {
            "fs": np.asarray(self.fs), "t": self.t, "x": self.x, "x_clean": self.x_clean,
            "t_true": self.t_true, "clipped": self.clipped, "rpm": self.rpm,
            "rpm_mean": self.rpm_mean, "rev": self.rev, "speed_kmh": self.speed_kmh,
            "gear": self.gear, "orders": self.orders, "order_stems": self.order_stems,
            "order_amplitude": self.order_amplitude, "order_phase": self.order_phase,
            "order_frequency": self.order_frequency, "ir": self.ir,
            "pulse_times": self.pulse_times, "pulse_revs": self.pulse_revs,
            "config_json": np.asarray(text),
        }
        if self.tacho is not None:
            arrays["tacho"] = self.tacho
        for name, s in self.stems.items():
            arrays[f"stem_{name}"] = s
        for name, s in self.source_stems.items():
            arrays[f"source_{name}"] = s
        for name, v in self.events.items():
            if isinstance(v, np.ndarray):
                arrays[f"event_{name}"] = v
        if self.carrier is not None:
            for name, v in self.carrier.items():
                arrays[f"carrier_{name}"] = np.asarray(v)
        np.savez(npz, **arrays)
        sidecar.write_text(text + "\n", encoding="utf-8")
        return npz, sidecar


def _sideband_specs(order, amplitude, mod_order, levels_db, modulation, rng):
    """Carrier phase and the sideband order specs of a mesh order (``sideband_orders``)."""
    phi_c, phi_m = (float(v) for v in rng.uniform(0.0, _TWO_PI, size=2))
    sb = sideband_orders(
        order, mod_order, levels_db, carrier_amplitude=amplitude, shaft=0, modulation=modulation,
        carrier_phase=phi_c, mod_phase=phi_m,
    )
    return phi_c, sb[1:]


def _floor_amplitude(models, f, tnr_db, bandwidth_hz, path_gain):
    """Source peak amplitude that stands ``tnr_db`` above the local broadband floor.

    The floor is the sum of the receiver-level noise PSDs at ``f`` at their reference
    condition, taken over ``bandwidth_hz``; the component is given the amplitude
    that reaches that level after the modal path (whose gain at ``f`` is divided out;
    the cabin impulse response then adds its random fading around this level).
    """
    floor = sum(float(_noise_psd(m, np.array([f]))[0]) for m in models.values())
    peak = np.sqrt(2.0 * floor * float(bandwidth_hz)) * 10.0 ** (float(tnr_db) / 20.0)
    return peak / path_gain(f)


def synth_scenario(
    kind, fs=48000.0, duration=None, *, seed=None, measurement=None, n_channels=1, **overrides
):
    """Synthesise a complete drive-train recording with exact truth.

    A scenario chains the building blocks of rota-order's test-signal toolbox into a
    recording that an analysis chain would meet in practice: a vehicle speed
    profile, the driveline that maps it to engine or motor speed (gear shifts,
    optional clutch-slip launch), a speed ripple locked to shaft angle, tonal
    orders, angle-locked impulses, discrete electrical tones, broadband and
    transient noise, a structural and cabin transfer path with memory, one or
    more receiver channels with diffuse-field coherence, a tachometer channel
    and an imperfect recorder. Every part is rendered from stored truth
    (:class:`ScenarioSignal`), so that analysis results can be checked against
    what was put into the signal and not against the speed curve and the order
    list alone.

    All default parameters are illustrative values typical of passenger cars,
    rounded; they are **not** measurements of any specific vehicle, machine or
    recording. Change them with keyword overrides (``scenario_defaults`` returns
    the parameter dictionary). Amplitudes of tonal and impulsive components are
    peak values in Pa (peak convention), noise levels are RMS sound pressure
    levels in dB re 20 uPa, as the individual parameters state.

    Signal flow (ICE; the EV replaces the engine sources, see Notes)::

        vehicle speed --drive_rpm--> rpm_mean --angular_speed_ripple--> rpm, rev
        rev --> orders (synth_runup)  -+
        rev --> combustion bursts      +--> modal path --> cabin IR --> stems
        transient events -------------+
        speed, rpm --> road, engine broadband, rumble (at the receiver) --> stems
        sum of stems --> measurement chain --> x
        rev --> tacho pulses --> delay --> tacho

    Parameters
    ----------
    kind : {"ice_runup", "ev_runup", "steady_cruise"}
        Scenario kind, see Notes. ``steady_cruise`` also accepts
        ``powertrain="ice"`` (default) or ``"ev"``, which selects the settings
        it starts from.
    fs : float, default 48000
        Sampling rate in Hz. It must allow at least 2 samples per tachometer
        pulse at the highest speed, path modes below ``fs/2`` and, for the EV,
        the inverter carrier (with its dither) and the tones below ``fs/2``;
        sidebands above ``fs/2`` are muted, never clipped.
    duration : float, optional
        Duration in s; default: the kind's ``duration`` parameter (18 s ICE,
        10 s EV, 10 s cruise).
    seed : int, numpy.random.Generator or None
        Random source. Every random draw (phases, ripple phases, noise, impulse
        jitter, impulse response, carrier hops, tachometer jitter, recorder
        noise) comes from its own stream derived from it, so the same integer
        seed gives a bit-identical result and adding a component does not
        change the others. An integer seed is recorded in ``config["seed"]``.
    measurement : dict, optional
        Arguments of :func:`rota_order.synth.measurement_chain` applied to the sum of
        the stems (``gain``, ``noise_floor_rms``, ``dc``, ``clip``, ``bits``,
        ``full_scale``, ``clock_drift_ppm``, ``seed``). ``None`` is an ideal
        recorder: ``x`` is then the exact sum of the stems. The tachometer
        channel does not pass through the chain (it has its own clock), so a
        clock drift is a drift between audio and tachometer.
    n_channels : int, default 1
        Number of receiver channels. With 2 or more, the cabin impulse
        response has the direct sound ``cabin.interchannel_delay`` later per
        channel, and both the reverberant tails and the receiver-level noises
        have the coherence of a diffuse field between receivers
        ``cabin.ear_spacing_m`` apart, ``sinc(2 pi f d / c)``: about 0.95 at
        200 Hz and a first zero at 1 kHz for ``d`` = 0.17 m. A simple stand-in
        for a binaural or two-microphone recording (no head-related transfer
        function).
    **overrides
        Replace entries of the parameter dictionary. A nested block given as a
        dict is merged key by key (``tacho={"ppr": 20}`` changes only ``ppr``);
        anything else replaces the entry (``orders={"list": [...]}`` replaces
        the order list, ``road=None`` switches a component off). Unknown names
        raise ``ValueError``.

    Returns
    -------
    ScenarioSignal
        Recording, true speed and angle, tachometer, stems before and after the
        path, source-order truth and configuration. ``x`` is (n,) for one channel
        and (n, ch) otherwise. The stems are accumulated in dict order into
        ``x_clean``, which equals ``x`` bit for bit for an ideal recorder; with a
        measurement chain ``x`` is the chain applied to ``x_clean`` (clock-drift
        resampling, gain, self-noise, DC, clipping, quantisation) and the sum
        identity holds only up to those impairments.

    Raises
    ------
    ValueError
        For an unknown kind, parameter or powertrain, an invalid value, or a
        configuration the building blocks reject (for example tachometer pulses
        shorter than 2 samples at the highest speed, a path mode above
        ``fs/2``).

    Notes
    -----
    *Model structure and literature.* The firing order of a four-stroke engine
    with ``Z`` cylinders is ``Z/2`` (the cycle spans two revolutions); the torque
    pulses of the cycle make the speed fluctuate at that order, and the
    harmonics, the half orders of the cycle and the crank orders follow
    (Heywood 1988). Gear-mesh orders carry sidebands spaced by the shaft order
    (``sideband_orders``; Randall 2011); a pulse-width-modulated inverter has a
    carrier with sidebands at multiples of the electrical frequency
    (Holmes & Lipo 2003; the dither of the carrier frequency is the usual
    spread-spectrum measure); the electromagnetic orders of a polyphase machine
    are multiples of the electrical frequency, which is ``p`` shaft orders for
    ``p`` pole pairs (Gieras et al. 2006). Vehicle speed follows the road-load
    equation with a quadratic drag term, ``dv/dt = a0 (1 - (v / vmax)**2)``, and
    the driveline maps it to engine speed (Gillespie 1992). Broadband levels
    follow straight lines in the logarithm of a control variable (the form used
    for tyre/road noise; Sandberg & Ejsmont 2002). The transfer path is a bank
    of second-order modes (Ewins 2000); the receivers are given the coherence of
    a diffuse field, ``sinc(2 pi f d / c)`` (Cook et al. 1955; Jacobsen & Roisin
    2000); the cabin reverberation follows a decaying-noise model with a stated
    reverberation time (Schroeder 1965; Kuttruff 2009).

    *ICE defaults* (``ice_runup``; a four-stroke four-cylinder engine). Speed:
    full-load run-up from 1200 rpm in second gear, about 500 rpm/s up to the
    6000 rpm shift point (acceleration ``a0`` = 2 m/s**2, ``vmax`` = 200 km/h),
    two upshifts in 18 s with gear ratios 3.5/2.45/1.71/1.2/0.84 (ratio step
    1.43, so the speed drops by about 30 % in a 0.4 s raised-cosine transient),
    idle 800 rpm, limiter 6500 rpm; ``speed.hold`` and ``speed.start_kmh`` = 0
    with ``drive.start_gear`` = 1 give the launch from idle with clutch slip
    instead. Speed ripple 0.30 % at order 2 and 0.08 % at order 4. Orders, level
    relative to order 2 at 3600 rpm: order 4 -15 dB, 6 -24 dB, 8 -30 dB; odd
    orders 3, 5, 7 at -25, -28, -32 dB; order 1 -32 dB and the half orders 0.5,
    1.5, 2.5 at -38, -34, -36 dB. Order 2 grows by 3 dB per 1000 rpm, all others
    by 1 dB per 1000 rpm. Combustion bursts (2 per revolution, jitter 0.004 rev)
    ring a structure resonance at 1.9 kHz (damping 0.04). The path has a hump
    at 45 Hz (Q = 3, +8 dB) and at 175 Hz (Q = 4, +6 dB) over the
    mid-frequency level, a dip near 80 Hz and a roll-off above 3 kHz. Broadband,
    at the receiver: a PSD tilt of -8.5 dB/octave from 200 Hz (engine
    broadband, +3 dB per doubling of rpm; road and wind, +4.5 dB per doubling
    of vehicle speed) and a low-frequency rumble hump over 20 to 60 Hz. The
    levels are calibrated to about 88 dB re 20 uPa unweighted at full load at
    mid speed (3600 rpm) with the orders 5 dB above the rest in energy
    (unweighted).

    *EV defaults* (``ev_runup``). Shaft speed 100 rpm per km/h (reduction 12,
    rolling circumference 2 m), full-load run-up from 2 km/h to about 130 km/h
    in 10 s (``a0`` = 4.4 m/s**2, ``vmax`` = 190 km/h); ``speed.mode="ramp"``
    with ``start_kmh`` = 120, ``end_kmh`` = 10 and a duration of 20 s is the
    coast-down variant. Motor orders (shaft orders, 4 pole pairs so the
    electrical frequency is order 4), level re the strongest and slope per
    doubling of speed: 2: 0 dB (+5), 4: -6 (+7), 8: -9 (+6), 10: -12 (+6), 12:
    -10 (+2, a broad resonance), 14: -15 (+5); a gear-mesh order 6.7 at -8 dB
    (+5); a gear whine at order 40 (``gear_mesh.teeth``) at -14 dB (+3) with a
    slow 0.5 Hz amplitude modulation of +-6 dB (depth 0.6); sidebands of the whine
    (``gear_mesh.sidebands``) are off. The inverter carrier is 10 kHz,
    dithered by up to +-0.8 kHz in hops every 0.2 to 0.5 s, with sidebands at
    carrier +- 2 and +- 4 times the electrical frequency (order 4), only
    slightly (2 to 4 dB) above the local floor in a 20 Hz band; two fixed tones
    at 6.3 and 14.2 kHz stand 10 and 6 dB above it. Broadband: a rumble hump
    over 25 to 35 Hz carrying at least 95 % of the energy below 200 Hz and a
    PSD tilt of -9 dB/octave above 200 Hz, both +6 dB per doubling of speed,
    about 90 dB re 20 uPa unweighted at 90 km/h. The tonal lines are weak:
    their A-weighted level is about 12 dB below the broadband at full load
    (much lower unweighted).

    *steady_cruise*: the ``powertrain`` settings at a constant speed
    (``speed.cruise_kmh`` = 90; 30 km/h is the other typical value; for the
    ICE choose ``cruise_gear``, 5 at 90 km/h and 3 at 30 km/h) for 10 s,
    without transients, with the tonal components lowered (``load``) so that
    they are about 20 dB below the broadband, A-weighted, at 90 km/h (at 30 km/h
    the ratio is about -21 dB for the ICE in third gear and -17 dB for the EV).
    The ripple is kept.

    *Common.* Path modes: a broad mode gives a flat gain from about 100 Hz to
    2 kHz. Cabin: reverberation time 0.1 s and volume 3.5 m**3 (an absorption
    area of 6 m**2 by Sabine's formula); below the Schroeder frequency
    ``2000 sqrt(T60 / V)`` = 338 Hz the response is a smooth direct arrival (unit
    gain; the resonances are those of the modal path), above it a diffuse tail
    whose channels have the coherence ``sinc(2 pi f d / c)``; direct sound at 5 ms
    and 0.3 ms per further channel, receivers 0.17 m apart (Schroeder 1962;
    Kuttruff 2009). Tachometer: ICE 60 pulses per revolution, 3 us RMS jitter, 1.5 ms
    delay; EV 48 pulses per revolution, 1 us, 0.5 ms; the pulses sit at the true
    shaft angles ``k / ppr`` (the delay moves every edge). Transients: a thump
    (32 Hz, 0.25 Pa peak, 0.06 s decay) at 42 % and a click (0.04 Pa, 4 ms) at
    78 % of the record, through the path.

    Amplitude and phase conventions of the truth are those of
    :class:`SynthSignal`: ``order_amplitude`` is the peak envelope and the
    complex envelope of an order at the receiver is, quasi-statically,
    ``order_amplitude * H(f) * exp(1j (order_phase - 2 pi order rev))`` with
    ``H`` from ``ScenarioSignal.path_response``. The first fraction of a second
    of every stem carries the start-up transient of the path (zero initial
    state).

    References
    ----------
    Heywood, J. B. (1988). Internal Combustion Engine Fundamentals. McGraw-Hill.
    Gillespie, T. D. (1992). Fundamentals of Vehicle Dynamics, ch. 2. SAE
    International.
    Randall, R. B. (2011). Vibration-based Condition Monitoring: Industrial,
    Automotive and Aerospace Applications. Wiley.
    Holmes, D. G., & Lipo, T. A. (2003). Pulse Width Modulation for Power
    Converters: Principles and Practice. IEEE Press / Wiley.
    Gieras, J. F., Wang, C., & Lai, J. C. (2006). Noise of Polyphase Electric
    Motors. CRC Press.
    Sandberg, U., & Ejsmont, J. A. (2002). Tyre/Road Noise Reference Book.
    Informex.
    Cook, R. K., Waterhouse, R. V., Berendt, R. D., Edelman, S., & Thompson, M. C.
    (1955). Measurement of correlation coefficients in reverberant sound fields.
    Journal of the Acoustical Society of America, 27(6), 1072-1077.
    Jacobsen, F., & Roisin, T. (2000). The coherence of reverberant sound fields.
    Journal of the Acoustical Society of America, 108(1), 204-210.
    Schroeder, M. R. (1962). Frequency-correlation functions of frequency
    responses in rooms. Journal of the Acoustical Society of America, 34(12),
    1819-1823.
    Schroeder, M. R. (1965). New method of measuring reverberation time. Journal
    of the Acoustical Society of America, 37(3), 409-412.
    Kuttruff, H. (2009). Room Acoustics, 5th ed. Spon Press.
    Ewins, D. J. (2000). Modal Testing: Theory, Practice and Application, 2nd
    ed. Research Studies Press.
    """
    from . import __version__

    if kind not in SCENARIO_KINDS:
        raise ValueError(f"kind must be one of {SCENARIO_KINDS}, got {kind!r}")
    fs = float(fs)
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a positive sampling rate in Hz")
    if isinstance(n_channels, bool) or int(n_channels) != n_channels or n_channels < 1:
        raise ValueError("n_channels must be a positive integer")
    n_ch = int(n_channels)
    overrides = dict(overrides)
    powertrain = overrides.pop("powertrain", "ice") if kind == "steady_cruise" else None
    cfg = scenario_defaults(kind, powertrain or "ice")
    _merge_overrides(cfg, overrides, kind)
    is_ev = kind == "ev_runup" or powertrain == "ev"
    duration = float(cfg["duration"] if duration is None else duration)
    if not (np.isfinite(duration) and duration > 0):
        raise ValueError("duration must be a positive time in s")
    cfg["duration"] = duration
    n = int(round(duration * fs))
    if n < 64:
        raise ValueError("duration * fs must give at least 64 samples")
    meas = _check_measurement(measurement)
    used_seed, rng = _scenario_streams(seed)
    t = np.arange(n) / fs

    # --- speed -------------------------------------------------------------------------
    speed_kmh = _speed_profile(t, cfg["speed"])
    drv, shift_speeds = _drive(t, speed_kmh, cfg)
    rpm_mean = np.asarray(drv["rpm"], dtype=float)
    ripple = _ripple_terms(cfg["ripple"], rng["ripple"])
    rpm_inst, rev = angular_speed_ripple(rpm_mean, fs, ripple=ripple)

    # --- tonal orders (synth_runup) ------------------------------------------------------
    load = float(cfg.get("load", 1.0))
    specs, names, group = [], [], []
    if not is_ev:
        ob = cfg["orders"]
        ref, a_ref = float(ob["ref_rpm"]), float(ob["ref_amplitude"]) * load
        mf = ob["min_freq"]
        taper = ob["min_freq_taper"] if mf is not None else 0.0
        for e in ob["list"]:
            gr, g = _order_law(e, ref)
            specs.append({
                "order": float(e["order"]), "amplitude": a_ref * 10.0 ** (float(e["level_db"]) / 20.0),
                "gain_rpm": gr, "gain": g, "min_freq": mf, "min_freq_taper": taper,
            })
            names.append(str(e.get("name", f"order_{e['order']:g}")))
            group.append("orders")
        sb = ob.get("sidebands")
        if sb is not None:
            k = names.index(sb["carrier"])
            phi_c, extra = _sideband_specs(
                specs[k]["order"], specs[k]["amplitude"], sb["mod_order"], sb["levels_db"],
                sb["modulation"], rng["phases"],
            )
            if sb["modulation"] is not None:
                specs[k]["phase"] = phi_c
            for s in extra:
                specs.append({**s, "gain_rpm": specs[k]["gain_rpm"], "gain": specs[k]["gain"],
                              "min_freq": mf, "min_freq_taper": taper})
                names.append(f"{names[k]}_sb_{s['order']:g}")
                group.append("orders")
    else:
        mb = cfg["motor"]
        ref, a_ref = float(mb["ref_rpm"]), float(mb["ref_amplitude"]) * load
        for e in mb["list"]:
            gr, g = _order_law(e, ref)
            specs.append({
                "order": float(e["order"]), "amplitude": a_ref * 10.0 ** (float(e["level_db"]) / 20.0),
                "gain_rpm": gr, "gain": g,
            })
            names.append(str(e.get("name", f"motor_{e['order']:g}")))
            group.append("motor")
        gm = cfg["gear_mesh"]
        if gm is not None:
            gr, g = _order_law({"slope_db_per_doubling": gm["slope_db_per_doubling"]}, ref)
            whine = {
                "order": float(gm["teeth"]), "amplitude": a_ref * 10.0 ** (float(gm["level_db"]) / 20.0),
                "gain_rpm": gr, "gain": g, "am_depth": float(gm["am_depth"]),
                "am_freq": float(gm["am_freq"]),
            }
            if gm.get("sidebands") is not None:
                phi_c, extra = _sideband_specs(
                    whine["order"], whine["amplitude"], gm["mod_order"], gm["sidebands"],
                    gm["modulation"], rng["phases"],
                )
                if gm["modulation"] is not None:
                    whine["phase"] = phi_c
            else:
                extra = []
            specs.append(whine)
            names.append(f"gear_mesh_{gm['teeth']:g}")
            group.append("gear")
            for s in extra:
                specs.append({**s, "gain_rpm": gr, "gain": g, "am_depth": whine["am_depth"],
                              "am_freq": whine["am_freq"]})
                names.append(f"gear_mesh_{gm['teeth']:g}_sb_{s['order']:g}")
                group.append("gear")
            if gm.get("mesh2_order") is not None:
                gr2, g2 = _order_law({"slope_db_per_doubling": gm["mesh2_slope_db_per_doubling"]}, ref)
                specs.append({
                    "order": float(gm["mesh2_order"]),
                    "amplitude": a_ref * 10.0 ** (float(gm["mesh2_level_db"]) / 20.0),
                    "gain_rpm": gr2, "gain": g2,
                })
                names.append(f"gear_mesh_{gm['mesh2_order']:g}")
                group.append("gear")
    tc = cfg["tacho"]
    tkw = {}
    if tc is not None:
        tkw = dict(
            tacho_ppr=tc["ppr"], tacho_jitter=tc["jitter"], tacho_drop=tc["drop"],
            tacho_high=tc["high"],
        )
    ss = synth_runup(fs, rpm_inst, specs, seed=_block_seed(rng["orders"]), **tkw)
    source = {}
    if not is_ev:
        source["orders"] = ss.stems["orders"]
    else:
        for name in ("motor", "gear"):
            idx = [i for i, g_ in enumerate(group) if g_ == name]
            if idx:
                source[name] = ss.order_stems[:, idx].sum(axis=1)

    # --- noise models (levels at the reference condition) ---------------------------------
    models = {}
    for name in ("engine_broadband", "road", "rumble"):
        if cfg.get(name) is not None:
            models[name] = _noise_model(cfg[name], fs)
    pth = cfg["path"]

    def path_gain(f):
        return float(np.abs(modal_response(f, pth["modes"], fs))) * float(pth["gain"])

    # --- other sources ----------------------------------------------------------------------
    events = {
        "shift_times": np.asarray(drv["shift_times"]), "shift_from": np.asarray(drv["shift_from"]),
        "shift_to": np.asarray(drv["shift_to"]), "in_shift": np.asarray(drv["in_shift"]),
        "slip": np.asarray(drv["slip"]), "launch_times": np.asarray(drv["launch_times"]),
    }
    blocks = {"synth_runup": ss.config}
    carrier = None
    if not is_ev:
        cb = cfg["combustion"]
        if cb is not None:
            gr, g = _order_law({"slope_db_per_krpm": cb["slope_db_per_krpm"]}, cb["ref_rpm"])
            imp = angle_impulses(
                rev, fs, per_rev=cb["per_rev"], amplitude=float(cb["amplitude"]) * load,
                resonance_hz=cb["resonance_hz"], damping=cb["damping"], amplitude_rpm=gr,
                amplitude_gain=g, rpm=rpm_inst, jitter_rev=cb["jitter_rev"],
                seed=_block_seed(rng["combustion"]),
            )
            source["combustion"] = imp["x"]
            events["impulse_times"] = imp["impulse_times"]
            events["impulse_rev"] = imp["impulse_rev"]
            blocks["combustion"] = {
                "n_events": int(imp["impulse_times"].size), "burst_samples": imp["burst_samples"],
            }
    else:
        inv = cfg["inverter"]
        if inv is not None:
            fc, dith = float(inv["carrier_hz"]), float(inv["dither_hz"])
            if not 0.0 < fc + dith < 0.5 * fs:
                raise ValueError("inverter carrier (plus dither) must lie below fs/2")
            if "road" not in models:
                raise ValueError("the inverter level is set against the broadband floor: keep road")
            a0 = _floor_amplitude(models, fc, inv["tnr_db"], inv["bandwidth_hz"], path_gain)
            terms = [(0.0, a0)]
            for sbd in inv["sidebands"]:
                q = float(sbd["n"]) * float(cfg["pole_pairs"])
                a = _floor_amplitude(models, fc, sbd["tnr_db"], inv["bandwidth_hz"], path_gain)
                terms += [(-q, a), (q, a)]
            track, hop_t, hop_f = _dither_track(n, fs, inv, rng["carrier"])
            cc = _dithered_carrier(rev, fs, rpm_inst, track, terms, rng["carrier"])
            source["inverter"] = cc["x"]
            carrier = {k_: cc[k_] for k_ in ("q", "frequency", "amplitude", "phase", "phase0", "carrier_hz")}
            events["carrier_hop_times"] = hop_t
            events["carrier_hop_hz"] = hop_f
        if cfg["tones"]:
            if "road" not in models:
                raise ValueError("the tone levels are set against the broadband floor: keep road")
            tone = np.zeros(n)
            phases = rng["tones"].uniform(0.0, _TWO_PI, size=len(cfg["tones"]))
            for tn, ph in zip(cfg["tones"], phases):
                f = float(tn["freq"])
                if not 0.0 < f < 0.5 * fs:
                    raise ValueError(f"tone at {f:g} Hz must lie below fs/2")
                a = _floor_amplitude(models, f, tn["tnr_db"], tn["bandwidth_hz"], path_gain)
                tone += a * np.cos(_TWO_PI * f * t + ph)
            source["tones"] = tone
    tr = cfg["transients"]
    if tr:
        evs = _transient_list(tr, duration)
        stem, tcfg = transient_events(fs, n, evs, seed=_block_seed(rng["transients"]))
        source["transients"] = stem
        events["transients"] = tcfg["events"]
        blocks["transients"] = tcfg

    # --- path, cabin and the receiver-level noises --------------------------------------------
    ir, ircfg = _cabin_ir(fs, cfg["cabin"], n_ch, _block_seed(rng["cabin"]))
    stems = {}
    pcfg = None
    for name, s in source.items():
        y, pcfg = modal_path(s, fs, pth["modes"], gain=pth["gain"])
        stems[name] = apply_ir(y, ir)
    blocks["modal_path"] = pcfg
    blocks["cabin_ir"] = ircfg
    cab = cfg["cabin"]
    controls = {"speed": speed_kmh, "rpm": rpm_mean}
    for name in ("engine_broadband", "road", "rumble"):
        if name not in models:
            continue
        cols = [
            _render_noise(models[name], fs, n, controls[cfg[name]["control"]], _block_seed(rng[{
                "engine_broadband": "engine", "road": "road", "rumble": "rumble"}[name]]))
            for _ in range(n_ch)
        ]
        w = _ear_mix(np.column_stack(cols), fs, cab.get("ear_spacing_m"), cab.get("sound_speed", 343.0))
        stems[name] = w[:, 0] if n_ch == 1 else w
        m = models[name]
        blocks[name] = {
            "rms_at_ref": m["rms_ref"], "ref": m["ref"], "slope_db_per_doubling": m["slope"],
            "grain": m["grain"], "psd_at_1khz": float(_noise_psd(m, np.array([1000.0]))[0]),
        }
    x_clean = None
    for s in stems.values():
        x_clean = s.copy() if x_clean is None else x_clean + s

    # --- recorder -------------------------------------------------------------------------------
    if meas is None:
        x, t_true, clipped, mcfg = x_clean.copy(), t.copy(), np.zeros(x_clean.shape, dtype=bool), None
    else:
        meas.setdefault("seed", _block_seed(rng["measurement"]))
        mc = measurement_chain(x_clean, fs, **meas)
        x, t_true, clipped, mcfg = mc["x"], mc["t_true"], mc["clipped"], mc["config"]

    # --- tachometer ------------------------------------------------------------------------------
    tacho, pulse_times, pulse_revs, delay = None, np.empty(0), np.empty(0), 0.0
    if tc is not None:
        delay = float(tc["delay"])
        if not np.isfinite(delay):
            raise ValueError("tacho.delay must be a finite time in s")
        if delay == 0.0:
            tacho = ss.tacho
        else:
            # Re-render the pulse train with every edge moved by the delay: exact edge times
            # (a band-limited fractional delay of a pulse train would move edges by up to 0.05
            # sample and ring). Widths: 50 % of the local pulse interval, as in synth_runup.
            rise = ss.pulse_times
            width = 0.5 * 60.0 / (float(tc["ppr"]) * np.interp(rise, t, rpm_inst))
            tacho = float(tc["high"]) * _pulse_train(t, fs, rise + delay, rise + delay + width)
        late = ss.pulse_times + delay
        inside = (late >= 0.0) & (late < t[-1])  # a moved edge may leave the record
        pulse_times = late[inside]
        pulse_revs = ss.pulse_revs[inside]

    gear = np.asarray(drv["gear"], dtype=int)
    config = {
        "rota_order_version": __version__,
        "kind": kind,
        "seed": used_seed,
        "fs": fs,
        "duration": duration,
        "n": n,
        "n_channels": n_ch,
        "parameters": cfg,
        "resolved": {
            "powertrain": "ev" if is_ev else "ice",
            "shift_speeds_kmh": shift_speeds,
            "ripple": [list(r) for r in ripple],
            "order_names": names,
            "orders": ss.orders,
            "tacho_delay": delay,
            "rev_deviation_max": float(np.max(np.abs(rev - ss.rev))),
            "stems": list(stems),
            "source_stems": list(source),
            "amplitude": "peak (tonal, impulsive) / rms (noise)",
        },
        "measurement": mcfg,
        "blocks": blocks,
    }
    return ScenarioSignal(
        kind=kind, fs=fs, t=t, x=x, x_clean=x_clean, t_true=t_true, clipped=clipped,
        rpm=rpm_inst, rpm_mean=rpm_mean, rev=rev, speed_kmh=speed_kmh, gear=gear, tacho=tacho,
        pulse_times=pulse_times, pulse_revs=pulse_revs, stems=stems, source_stems=source,
        orders=ss.orders, order_names=names, order_stems=ss.order_stems,
        order_amplitude=ss.order_amplitude, order_phase=ss.order_phase,
        order_frequency=ss.order_frequency, ir=ir, events=events, carrier=carrier,
        config=jsonable(config),
    )
