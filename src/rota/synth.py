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
angle is *defined* as :func:`rota.speed.rpm_to_phase` of the speed curve
(trapezoidal integral), so the truth phase and the analysis-side phase are
the same quantity by construction.

Units: time in s, frequency in Hz, rotational speed in rpm (or km/h for the
generic profiles), shaft angle in revolutions, phase in rad. All amplitudes
are peak amplitudes unless stated otherwise (noise is specified as RMS).
Output is mono (one channel) in v0.1.

:func:`synth_tone_noise` is separate: static listening-test stimuli (tone plus
noise at a stated broadband tone-to-noise power ratio and RMS level), with the
tone and noise stems returned on their own.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from .speed import rpm_to_phase

__all__ = ["rpm_profiles", "synth_runup", "SynthSignal", "synth_tone_noise", "ToneNoiseSignal"]

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
    (such as 40, 100, 40 km/h), which :func:`rota.speed.speed_to_rpm` maps to
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
        Profile values, in the unit of ``start`` and ``stop``.

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
        ``rev = rota.speed.rpm_to_phase(rpm, fs)``.
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
        phases, the seed (if an int) and the ROTA version.
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
        seed, ROTA version); the NPZ holds ``fs``, ``t``, ``x``, ``rpm``,
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
        ROTA ROADMAP, design principle 7 (reproducibility sidecar).
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

    Model. With ``rev = rota.speed.rpm_to_phase(rpm, fs)`` (revolutions, 0 at
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
        "rota_version": __version__,
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
        the seed (if an int), the stem powers and the ROTA version. Infinite
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
        "rota_version": __version__,
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
