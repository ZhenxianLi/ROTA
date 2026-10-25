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
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from .speed import rpm_to_phase

__all__ = ["rpm_profiles", "synth_runup", "SynthSignal"]

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
