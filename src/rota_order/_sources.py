"""Physically motivated source models for synthetic drive-train signals.

Building blocks that make the speed and the sources of :func:`rota_order.synth.synth_runup`
test signals more realistic, each with exact truth:

* :func:`drive_rpm`: engine speed from vehicle speed through a driveline, with
  clutch-slip launch and gear-shift transients.
* :func:`angular_speed_ripple`: angle-periodic speed fluctuation (cyclic
  irregularity), integrated exactly so that the shaft angle is the truth.
* :func:`angle_impulses`: impulsive events locked to shaft angle (combustion,
  gear-tooth impacts) ringing a fixed-frequency resonance.
* :func:`sideband_orders`: carrier order with modulation sidebands, as order
  specifications for :func:`rota_order.synth.synth_runup`.
* :func:`carrier_components`: components around a fixed carrier frequency whose
  sidebands move with shaft speed (inverter switching sidebands).

Units: time in s, frequency in Hz, rotational speed in rpm, shaft angle in
revolutions, vehicle speed in km/h, phase in rad. All amplitudes are peak
amplitudes. Every random draw goes through a seed that is an int or a
:class:`numpy.random.Generator`; the same seed gives bit-identical output.
"""

import numpy as np
from scipy import interpolate

from ._utils import runs
from .speed import rpm_to_phase, speed_to_rpm

__all__ = [
    "drive_rpm",
    "angular_speed_ripple",
    "angle_impulses",
    "sideband_orders",
    "carrier_components",
]

_TWO_PI = 2.0 * np.pi
_STANDSTILL_KMH = 0.5  # at or below this speed the vehicle counts as stationary
_TAIL_LEVEL = 1e-6  # burst support ends where the envelope is this fraction of its start
_CHUNK_ELEMENTS = 4_000_000  # events x burst samples rendered per vectorised block


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _vector(x, name, n=None):
    """Finite non-empty 1-D float array, optionally of length ``n``."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a non-empty, finite 1-D array")
    if n is not None and x.size != n:
        raise ValueError(f"{name} must have {n} samples, got {x.size}")
    return x


def _positive(value, name):
    v = float(value)
    if not (np.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be a positive, finite number")
    return v


def _raised_cosine(u):
    """Smooth 0-to-1 ramp ``(1 - cos(pi u)) / 2`` with zero slope at both ends."""
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(u, 0.0, 1.0)))


# --------------------------------------------------------------------------
# 1. engine speed from vehicle speed
# --------------------------------------------------------------------------


def drive_rpm(
    t,
    speed_kmh,
    *,
    gear_ratios,
    final_drive,
    wheel_circumference,
    shift_speeds,
    idle,
    redline,
    shift_time=0.3,
    launch_slip=True,
    launch_rpm=None,
):
    """Engine speed from vehicle speed with launch slip and gear-shift transients.

    The static relation is the driveline chain of :func:`rota_order.speed.speed_to_rpm`,
    ``rpm = v / 3.6 / C * 60 * G[g] * final_drive``, with the gear ``g``
    selected from the vehicle speed: gear ``g`` is engaged while
    ``shift_speeds[g-2] <= v < shift_speeds[g-1]`` (the same rule as
    ``speed_to_rpm``; the shift speed is the same up and down, so ``speed_kmh``
    should be smooth, otherwise a speed dithering around a shift speed shifts at
    every crossing). On top of it two transients are added.

    *Shifts.* A shift starts at the time ``ts`` at which the vehicle speed
    crosses the shift speed (linear interpolation between samples). From the
    engine speed ``r_start`` that the old gear gives at ``ts`` to the speed
    ``r_end`` that the new gear gives at ``ts + shift_time``, the engine speed
    follows a raised cosine,
    ``rpm = r_start + w(u) (r_end - r_start)``, ``u = (t - ts) / shift_time``,
    ``w(u) = (1 - cos(pi u)) / 2``. It is continuous at both ends, joins the
    static curve at ``ts + shift_time`` and is strictly monotone in between
    (falling for an upshift and rising for a downshift, as the new-gear speed
    is normally below and above the old-gear speed). If the next shift starts
    before the transient has finished, it starts from the value reached so far.
    The raised-cosine shape is a modelling choice (a clutch-open, free-running
    engine), not a measured law.

    *Launch.* When the vehicle sets off from standstill the clutch slips until
    the first gear gives ``launch_rpm``: the engine speed rises from ``idle`` at
    the onset of motion (the last sample at or below 0.5 km/h) along a raised
    cosine and meets the first-gear value ``launch_rpm`` exactly when the
    vehicle speed reaches it; it is never below the first-gear value. After
    that the clutch is locked and the static relation holds. Without
    ``launch_slip`` the static relation is clipped at ``idle`` (kink where it
    leaves ``idle``). No slip phase is modelled for a record that starts while
    the vehicle is already moving, nor for one that ends before lock-up.

    Finally the engine speed is clipped to ``[idle, redline]`` (idle at
    standstill, rev limiter).

    Parameters
    ----------
    t : array_like, shape (n,)
        Sample times in s, strictly increasing (need not be uniform).
    speed_kmh : array_like, shape (n,)
        Vehicle speed in km/h (not m/s); negative values map to ``idle``.
    gear_ratios : sequence of float
        Gear ratios, first gear first.
    final_drive : float
        Final-drive ratio.
    wheel_circumference : float
        Rolling circumference in m.
    shift_speeds : sequence of float
        Shift speeds in km/h, ``len(gear_ratios) - 1`` strictly increasing
        positive values (empty for a single gear).
    idle, redline : float
        Engine-speed limits in rpm, ``0 <= idle < redline``.
    shift_time : float, default 0.3
        Duration of a shift transient in s; 0 gives an instantaneous step.
    launch_slip : bool, default True
        Model the clutch-slip launch.
    launch_rpm : float, optional
        Engine speed in rpm at which the clutch is locked. Default:
        ``2 * idle``, limited to 80 % of the first-gear speed at the first
        shift speed. An explicit value must lie between ``idle`` and both
        ``redline`` and that first-gear speed.

    Returns
    -------
    dict
        ``rpm`` (n,): engine speed in rpm. ``gear`` (n,) int: selected gear
        (1-based), switching to the new gear at the shift time. ``shift_times``
        (m,): shift times in s, with ``shift_from`` and ``shift_to`` (m,) the
        gears. ``in_shift`` (n,) bool: samples inside a shift transient.
        ``slip`` (n,) bool: samples inside a launch slip phase.
        ``launch_times`` (j, 2): onset and lock-up time in s of each launch.
        No amplitudes are involved.

    References
    ----------
    Gillespie, T. D. (1992). Fundamentals of Vehicle Dynamics, ch. 2
    (acceleration performance: driveline ratios and rolling radius). SAE
    International.
    """
    t = _vector(t, "t")
    n = t.size
    if n < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing with at least 2 samples")
    v = _vector(speed_kmh, "speed_kmh", n)
    ratios = np.asarray(gear_ratios, dtype=float)
    if ratios.ndim != 1 or ratios.size < 1 or not (np.all(np.isfinite(ratios)) and np.all(ratios > 0)):
        raise ValueError("gear_ratios must be a non-empty 1-D sequence of positive ratios")
    ng = ratios.size
    final_drive = _positive(final_drive, "final_drive")
    circumference = _positive(wheel_circumference, "wheel_circumference")
    ss = np.asarray(shift_speeds, dtype=float).reshape(-1)
    if ss.size != ng - 1:
        raise ValueError("shift_speeds needs len(gear_ratios) - 1 values")
    if ss.size and not (np.all(np.isfinite(ss)) and ss[0] > 0 and np.all(np.diff(ss) > 0)):
        raise ValueError("shift_speeds must be positive and strictly increasing")
    idle, redline = float(idle), float(redline)
    if not (np.isfinite(idle) and np.isfinite(redline) and 0 <= idle < redline):
        raise ValueError("idle and redline must satisfy 0 <= idle < redline (rpm)")
    shift_time = float(shift_time)
    if not (np.isfinite(shift_time) and shift_time >= 0):
        raise ValueError("shift_time must be a non-negative time in s")

    chain = dict(wheel_circumference=circumference, gear_ratios=ratios, final_drive=final_drive)
    k = np.array([float(speed_to_rpm(1.0, gear=g, **chain)) for g in range(1, ng + 1)])  # rpm per km/h
    gear = np.searchsorted(ss, v, side="right") + 1
    r = np.array(speed_to_rpm(v, gear=gear, **chain))  # static, not yet limited

    # Shift transients.
    in_shift = np.zeros(n, dtype=bool)
    ev_t, ev_from, ev_to = [], [], []
    prev = None  # (start time, start value, end value) of the transient in progress
    for i0 in np.flatnonzero(np.diff(gear) != 0) + 1:
        a, b = int(gear[i0 - 1]), int(gear[i0])
        s_star = ss[a - 1] if b > a else ss[a - 2]  # first shift speed crossed
        tau = np.clip((s_star - v[i0 - 1]) / (v[i0] - v[i0 - 1]), 0.0, 1.0)
        ts = t[i0 - 1] + tau * (t[i0] - t[i0 - 1])
        ev_t.append(ts)
        ev_from.append(a)
        ev_to.append(b)
        if shift_time == 0.0:
            continue
        if prev is not None and ts < prev[0] + shift_time:
            start = prev[1] + _raised_cosine((ts - prev[0]) / shift_time) * (prev[2] - prev[1])
        else:
            start = k[a - 1] * s_star
        end = k[b - 1] * np.interp(ts + shift_time, t, v)
        j0 = np.searchsorted(t, ts, side="left")
        j1 = np.searchsorted(t, ts + shift_time, side="left")
        r[j0:j1] = start + _raised_cosine((t[j0:j1] - ts) / shift_time) * (end - start)
        in_shift[j0:j1] = True
        prev = (ts, start, end)

    # Launch slip.
    slip = np.zeros(n, dtype=bool)
    launches = []
    if launch_rpm is not None:
        launch_rpm = float(launch_rpm)
        first_gear_cap = k[0] * ss[0] if ss.size else redline
        if not (np.isfinite(launch_rpm) and idle < launch_rpm < min(redline, first_gear_cap)):
            raise ValueError(
                "launch_rpm must lie between idle and both redline and the first-gear "
                "speed at the first shift speed"
            )
        r_lock = launch_rpm
    else:
        r_lock = min(2.0 * idle, 0.8 * k[0] * ss[0]) if ss.size else min(2.0 * idle, 0.9 * redline)
    if launch_slip and r_lock > idle:
        below = (gear == 1) & (k[0] * v < r_lock)
        for a, b in zip(*runs(below)):
            if b >= n:  # the record ends before lock-up
                continue
            standing = np.flatnonzero(v[a:b] <= _STANDSTILL_KMH)
            if not standing.size:  # never stationary: a vehicle already moving does not launch
                continue
            j = a + int(standing[-1])
            dv = v[b] - v[b - 1]
            tau = np.clip((r_lock / k[0] - v[b - 1]) / dv, 0.0, 1.0) if dv > 0 else 1.0
            t_lock = t[b - 1] + tau * (t[b] - t[b - 1])
            t_on = t[j]
            if not t_lock > t_on:
                continue
            r_on = max(idle, k[0] * v[j])
            j1 = np.searchsorted(t, t_lock, side="left")
            rise = r_on + _raised_cosine((t[j:j1] - t_on) / (t_lock - t_on)) * (r_lock - r_on)
            r[j:j1] = np.maximum(r[j:j1], rise)
            slip[j:j1] = True
            launches.append((t_on, t_lock))

    return {
        "rpm": np.clip(r, idle, redline),
        "gear": gear,
        "shift_times": np.array(ev_t, dtype=float),
        "shift_from": np.array(ev_from, dtype=int),
        "shift_to": np.array(ev_to, dtype=int),
        "in_shift": in_shift,
        "slip": slip,
        "launch_times": np.array(launches, dtype=float).reshape(-1, 2),
    }


# --------------------------------------------------------------------------
# 2. angle-periodic speed ripple
# --------------------------------------------------------------------------


def _ripple_gain(theta, k, a, phi):
    """``g(theta) = 1 + sum a_k cos(2 pi k theta + phi_k)`` for angle ``theta`` in rev."""
    g = np.ones(np.shape(theta))
    for kk, aa, pp in zip(k, a, phi):
        g = g + aa * np.cos(_TWO_PI * kk * theta + pp)
    return g


def _common_period(orders, max_period=64):
    """Smallest integer ``P <= max_period`` (revolutions) with ``orders * P`` integers."""
    for period in range(1, max_period + 1):
        kp = orders * period
        if np.all(np.abs(kp - np.round(kp)) <= 1e-9 * np.maximum(1.0, kp)):
            return period
    raise ValueError(
        f"ripple orders must be integer multiples of 1/P with P <= {max_period} "
        "(the fluctuation must be periodic in shaft angle)"
    )


def angular_speed_ripple(rpm, fs, *, ripple, rev0=0.0):
    """Instantaneous speed with an angle-periodic fluctuation, and the exact shaft angle.

    The shaft angle ``theta`` (revolutions) obeys

    ``d theta / dt = Omega(t) * g(theta)``,
    ``g(theta) = 1 + sum_k a_k cos(2 pi k theta + phi_k)``,

    with ``Omega = rpm / 60`` in rev/s, piecewise linear between the samples, and
    ``theta(0) = rev0``. This is the cyclic irregularity of a reciprocating
    engine, whose torque pulses are locked to crank angle, seen as a speed
    fluctuation of a few tenths of a percent at the firing order and its
    harmonics. Because the fluctuation depends on angle and the nominal speed
    only on time, the equation separates: with the nominal angle
    ``psi = rota_order.speed.rpm_to_phase(rpm)`` (exact for the piecewise-linear
    speed) one has ``d theta / d psi = g(theta)``, i.e.
    ``psi + G(rev0) = G(theta)`` with ``G(theta) = integral of 1/g``. ``G`` is
    tabulated over one period in angle with Gauss-Legendre quadrature and
    inverted by cubic Hermite interpolation, whose slopes are known exactly
    (``g``), so ``rev`` solves the equation to about 1e-11 rev at every sample
    (checked against a closed-form solution and an adaptive ODE solver) and no
    step-by-step integration, with its accumulated error, is involved.

    Consequences that tests and users can rely on:

    * ``rpm_inst = 60 * d rev / dt`` at the samples, exactly: it is the speed a
      perfect tachometer on this shaft would report. The trapezoidal integral of
      ``rpm_inst / 60`` reproduces ``rev`` up to the sampling error of the
      ripple (second order in ripple frequency over sampling rate).
    * The nominal speed is the angle-average of the speed, not the time-average:
      the shaft dwells longer where it is slow, so over many cycles
      ``rev`` advances at ``Omega / <1/g>`` with ``<.>`` the average over one
      period in angle, i.e. slower than the nominal speed by a factor
      ``1 - sum(a_k**2) / 2`` to second order. For one term the exact factor is
      ``sqrt(1 - a**2)``.
    * With ``ripple=()`` the result is the nominal speed and
      ``rota_order.speed.rpm_to_phase(rpm, fs, rev0=rev0)``.

    Parameters
    ----------
    rpm : array_like, shape (n,)
        Nominal shaft speed in rpm at the samples, finite and non-negative.
    fs : float
        Sampling rate in Hz.
    ripple : sequence of (order, amplitude, phase)
        Fluctuation terms: ``order`` ``k > 0`` in cycles per revolution
        (e.g. the firing order, half orders allowed), ``amplitude`` the relative
        peak amplitude ``a_k`` of the speed fluctuation (dimensionless; 0.003 is
        0.3 %) and ``phase`` ``phi_k`` in rad, referenced to the absolute shaft
        angle. The orders must be integer multiples of ``1/P`` for some integer
        ``P <= 64`` (to 1e-9), so that the fluctuation is periodic in angle, and
        ``sum |a_k| < 1``.
    rev0 : float, default 0
        Shaft angle at the first sample, in revolutions.

    Returns
    -------
    rpm_inst : ndarray, shape (n,)
        Instantaneous speed in rpm, ``rpm * g(rev)``.
    rev : ndarray, shape (n,)
        Exact shaft angle in revolutions at the samples (non-decreasing). No
        amplitude convention applies; ``a_k`` are peak relative amplitudes.

    References
    ----------
    Heywood, J. B. (1988). Internal Combustion Engine Fundamentals. McGraw-Hill
    (cyclic torque pulsation of reciprocating engines).
    Rizzoni, G. (1989). Estimate of indicated torque from crankshaft speed
    fluctuations: a model for the dynamics of the IC engine. IEEE Transactions
    on Vehicular Technology, 38(3).
    Abramowitz, M., & Stegun, I. A. (1964). Handbook of Mathematical Functions,
    sec. 25.4 (Gauss-Legendre quadrature). National Bureau of Standards.
    de Boor, C. (1978). A Practical Guide to Splines (Hermite interpolation).
    Springer.
    """
    rpm = _vector(rpm, "rpm")
    n = rpm.size
    if n < 2:
        raise ValueError("rpm must have at least 2 samples")
    if np.any(rpm < 0):
        raise ValueError("rpm must be non-negative")
    fs = _positive(fs, "fs")
    rev0 = float(rev0)
    if not np.isfinite(rev0):
        raise ValueError("rev0 must be finite")
    terms = [tuple(float(c) for c in term) for term in ripple]
    if any(len(term) != 3 for term in terms):
        raise ValueError("ripple must be a sequence of (order, amplitude, phase)")
    psi = rpm_to_phase(rpm, fs)
    if not terms:
        return rpm.copy(), psi + rev0
    k, a, phi = (np.array(c) for c in zip(*terms))
    if not (np.all(np.isfinite(k)) and np.all(k > 0)):
        raise ValueError("ripple orders must be positive and finite")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(phi))):
        raise ValueError("ripple amplitudes and phases must be finite")
    if not np.sum(np.abs(a)) < 1.0:
        raise ValueError("sum of |amplitude| must be below 1 (the speed must stay positive)")
    period = _common_period(k)
    k = np.round(k * period) / period

    # One period of G(theta) = integral_0^theta dtheta' / g(theta'), cell by cell.
    cells_per_rev = max(512, int(np.ceil(128 * k.max())))
    m = cells_per_rev * period
    h = 1.0 / cells_per_rev
    nodes = np.arange(m + 1) * h
    xi, wq = np.polynomial.legendre.leggauss(6)
    pts = (nodes[:-1] + 0.5 * h)[:, None] + 0.5 * h * xi[None, :]
    cell = 0.5 * h * ((1.0 / _ripple_gain(pts, k, a, phi)) @ wq)
    big_g = np.concatenate(([0.0], np.cumsum(cell)))
    g_nodes = _ripple_gain(nodes, k, a, phi)
    g_period = big_g[-1]
    forward = interpolate.CubicHermiteSpline(nodes, big_g, 1.0 / g_nodes)  # G(theta)
    inverse = interpolate.CubicHermiteSpline(big_g, nodes, g_nodes)  # theta(G)

    n0 = np.floor(rev0 / period)
    gamma = n0 * g_period + float(forward(rev0 - n0 * period)) + psi
    cycles = np.floor(gamma / g_period)
    rev = period * cycles + inverse(np.clip(gamma - cycles * g_period, 0.0, g_period))
    return rpm * _ripple_gain(rev, k, a, phi), rev


# --------------------------------------------------------------------------
# 3. impulsive events locked to shaft angle
# --------------------------------------------------------------------------


def _times_at_angle(theta, rev, rpm, fs):
    """Times in s at which the shaft angle reaches ``theta`` (rev).

    On each sample interval the angle is the cubic Hermite interpolant of
    ``rev`` with slopes ``rpm / 60`` at the samples. That reproduces any
    quadratic exactly, so for a speed that is piecewise linear between samples
    (the trapezoidal angle of ``rota_order.speed.rpm_to_phase``) it is the angle itself.
    The cubic is inverted by Newton iteration from the linear guess.
    """
    n = rev.size
    i = np.clip(np.searchsorted(rev, theta, side="right") - 1, 0, n - 2)
    y0, d = rev[i], rev[i + 1] - rev[i]
    m0, m1 = rpm[i] / (60.0 * fs), rpm[i + 1] / (60.0 * fs)  # rev per sample interval
    c2 = 3.0 * d - 2.0 * m0 - m1
    c3 = m0 + m1 - 2.0 * d
    target = theta - y0
    u = np.clip(np.divide(target, d, out=np.zeros_like(target), where=d > 0), 0.0, 1.0)
    for _ in range(8):
        p = ((c3 * u + c2) * u + m0) * u
        dp = (3.0 * c3 * u + 2.0 * c2) * u + m0
        u = np.clip(u - np.divide(p - target, dp, out=np.zeros_like(p), where=dp > 0), 0.0, 1.0)
    return (i + u) / fs


def angle_impulses(
    rev,
    fs,
    *,
    per_rev,
    amplitude,
    resonance_hz,
    damping,
    amplitude_rpm=None,
    amplitude_gain=None,
    rpm=None,
    jitter_rev=0.0,
    seed=None,
):
    """Impulsive events locked to shaft angle, each ringing a fixed resonance.

    Combustion pulses and gear-tooth impacts occur at fixed shaft angles and
    excite a structural resonance whose frequency does not depend on speed. An
    event ``e`` at time ``te`` produces the burst

    ``x(t) = A_e * exp(-zeta w0 (t - te)) sin(wd (t - te)) / N`` for ``t >= te``,

    with ``w0 = 2 pi resonance_hz``, ``zeta = damping``,
    ``wd = w0 sqrt(1 - zeta**2)`` (the impulse response of a damped
    single-degree-of-freedom system) and the constant
    ``N = sqrt(1 - zeta**2) exp(-zeta acos(zeta) / sqrt(1 - zeta**2))``, the
    largest value of the burst envelope function, so that the continuous-time
    peak of the burst is exactly ``A_e`` (the first sample after ``te`` need not
    sample the maximum, so the sampled peak can be slightly lower).

    The events are located at the shaft angles ``k / per_rev`` (``k`` an
    integer), plus an optional Gaussian angle error. Their times are found by
    inverting the shaft angle: the angle is the cubic Hermite interpolant of
    ``rev`` with the slopes ``rpm / 60``, which equals the angle of
    :func:`rota_order.speed.rpm_to_phase` exactly for a piecewise-linear speed. The
    burst is evaluated analytically at the sample times for the sub-sample event
    time (no interpolated template), so the stem equals the sum of the analytic
    bursts up to rounding, apart from the burst tails, which are cut where the
    envelope is 1e-6 of its start (-120 dB). The events give families of orders
    at multiples of ``per_rev`` whose amplitudes follow the resonance, with the
    line at frequency ``f`` of strength ``2 A |H(f)| per_rev rpm / 60 / N`` for
    a slowly varying speed, ``H`` the transfer function
    ``wd / (w0**2 - w**2 + 2j zeta w0 w)`` of the burst.

    Events before the first sample are not rendered, so the first burst tails
    are missing for about one decay time.

    Parameters
    ----------
    rev : array_like, shape (n,)
        Shaft angle in revolutions at the samples (non-decreasing), e.g. from
        :func:`rota_order.speed.rpm_to_phase` or :func:`angular_speed_ripple`.
    fs : float
        Sampling rate in Hz.
    per_rev : float
        Events per revolution (positive; fractional values such as 1.5 allowed).
        The events are at the angles ``k / per_rev`` that lie in the record.
    amplitude : float
        Peak amplitude ``A`` of one burst, in the unit of the result (its sign
        sets the polarity).
    resonance_hz : float
        Undamped resonance frequency in Hz, ``0 < resonance_hz < fs/2``.
    damping : float
        Damping ratio ``zeta``, ``0 < zeta < 1``.
    amplitude_rpm, amplitude_gain : array_like, optional
        Amplitude law versus engine speed, given together: speeds in rpm
        (strictly increasing) and non-negative linear gains, interpolated
        linearly at the speed of each event (end values held). Without it the
        gain is 1.
    rpm : array_like, shape (n,), optional
        Speed in rpm at the samples, used for the slopes in the angle inversion
        and for the amplitude law; it should be the derivative of ``rev`` (pass
        ``rpm_inst`` of :func:`angular_speed_ripple` when ``rev`` comes from
        it). Default: ``60 * fs * gradient(rev)``.
    jitter_rev : float, default 0
        Standard deviation in revolutions of the Gaussian error added to each
        event angle (e.g. 0.002 rev is 0.7 degrees). Events pushed out of the
        record are dropped.
    seed : int, numpy.random.Generator or None
        Random source for the jitter.

    Returns
    -------
    dict
        ``x`` (n,): the stem, with peak burst amplitude ``A * gain``.
        ``impulse_times`` (m,): event times in s, jitter included (truth).
        ``impulse_rev`` (m,): the angle of each event in rev, jitter included;
        ``impulse_rev_nominal`` (m,): ``k / per_rev``. ``impulse_amplitude``
        (m,): peak amplitude of each burst. ``burst_samples``: length in
        samples of the rendered burst.

    References
    ----------
    Inman, D. J. (2014). Engineering Vibration, 4th ed. Pearson (impulse
    response of the damped single-degree-of-freedom system).
    Brandt, A. (2011). Noise and Vibration Analysis: Signal Analysis and
    Experimental Procedures, ch. 5-6 (second-order systems and resonances).
    Wiley.
    Randall, R. B. (2011). Vibration-based Condition Monitoring: Industrial,
    Automotive and Aerospace Applications. Wiley (impulsive excitation locked
    to shaft angle, in engines and gearboxes).
    """
    rev = _vector(rev, "rev")
    n = rev.size
    if n < 2:
        raise ValueError("rev must have at least 2 samples")
    if np.any(np.diff(rev) < 0):
        raise ValueError("rev must be non-decreasing")
    fs = _positive(fs, "fs")
    per_rev = _positive(per_rev, "per_rev")
    amplitude = float(amplitude)
    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    resonance_hz = _positive(resonance_hz, "resonance_hz")
    if not resonance_hz < 0.5 * fs:
        raise ValueError("resonance_hz must be below fs/2")
    damping = float(damping)
    if not (np.isfinite(damping) and 0 < damping < 1):
        raise ValueError("damping must satisfy 0 < damping < 1")
    jitter_rev = float(jitter_rev)
    if not (np.isfinite(jitter_rev) and jitter_rev >= 0):
        raise ValueError("jitter_rev must be non-negative (rev)")
    if (amplitude_rpm is None) != (amplitude_gain is None):
        raise ValueError("amplitude_rpm and amplitude_gain must be given together")
    if amplitude_rpm is not None:
        law_rpm = _vector(amplitude_rpm, "amplitude_rpm")
        law_gain = _vector(amplitude_gain, "amplitude_gain", law_rpm.size)
        if np.any(np.diff(law_rpm) <= 0) or np.any(law_gain < 0):
            raise ValueError("amplitude_rpm must increase strictly and amplitude_gain be >= 0")
    if rpm is None:
        rpm = 60.0 * fs * np.gradient(rev)
    else:
        rpm = _vector(rpm, "rpm", n)
    rpm = np.maximum(rpm, 0.0)

    # Event angles and times.
    k_first = int(np.ceil(rev[0] * per_rev - 1e-9))
    k_last = int(np.floor(rev[-1] * per_rev + 1e-9))
    nominal = np.arange(k_first, k_last + 1) / per_rev
    angle = nominal.copy()
    if jitter_rev > 0 and nominal.size:
        angle = angle + jitter_rev * np.random.default_rng(seed).standard_normal(nominal.size)
    keep = (angle >= rev[0]) & (angle <= rev[-1])
    nominal, angle = nominal[keep], angle[keep]
    te = _times_at_angle(angle, rev, rpm, fs)
    gain = np.ones(te.size)
    if amplitude_rpm is not None:
        gain = np.interp(np.interp(te, np.arange(n) / fs, rpm), law_rpm, law_gain)
    peak = amplitude * gain

    # Render the bursts: exp(s tau) with s = -zeta w0 + j wd, evaluated at the
    # sample times of a sub-sample event time, vectorised in chunks.
    w0 = _TWO_PI * resonance_hz
    s = complex(-damping * w0, w0 * np.sqrt(1.0 - damping**2))
    norm = np.sqrt(1.0 - damping**2) * np.exp(-damping * np.arccos(damping) / np.sqrt(1.0 - damping**2))
    length = min(n, int(np.ceil(-np.log(_TAIL_LEVEL) / (damping * w0) * fs)) + 1)
    template = np.exp(s * (np.arange(length) / fs))
    first = np.ceil(te * fs).astype(np.intp)  # first sample at or after the event
    lead = np.exp(s * (first / fs - te)) * (peak / norm)
    x = np.zeros(n)
    chunk = max(1, int(_CHUNK_ELEMENTS // length))
    for c0 in range(0, te.size, chunk):
        sl = slice(c0, c0 + chunk)
        vals = np.imag(lead[sl, None] * template[None, :])
        idx = first[sl, None] + np.arange(length)[None, :]
        ok = idx < n
        x += np.bincount(idx[ok], weights=vals[ok], minlength=n)

    return {
        "x": x,
        "impulse_times": te,
        "impulse_rev": angle,
        "impulse_rev_nominal": nominal,
        "impulse_amplitude": peak,
        "burst_samples": int(length),
    }


# --------------------------------------------------------------------------
# 4. carrier with modulation sidebands as orders
# --------------------------------------------------------------------------


def sideband_orders(
    carrier_order,
    mod_order,
    levels_db,
    *,
    carrier_amplitude=1.0,
    shaft=0,
    modulation=None,
    carrier_phase=0.0,
    mod_phase=0.0,
):
    """Order specifications of a carrier order with modulation sidebands.

    A tooth-meshing order modulated by a shaft rotation (eccentricity, a local
    fault, load variation) shows sidebands at ``carrier_order +- k * mod_order``
    spaced by the modulating order. The result lists the carrier and the
    sidebands as order dictionaries for :func:`rota_order.synth.synth_runup`
    (``orders=...``), in the order carrier, lower 1, upper 1, lower 2, upper 2,
    ... Because the sidebands are ordinary orders, their level, order and
    frequency are in the truth of the synthesised signal.

    The level of sideband ``k`` is ``levels_db`` dB relative to the carrier
    amplitude: a peak amplitude of ``carrier_amplitude * 10**(level / 20)``.
    For amplitude modulation ``(1 + m cos(2 pi q rev + phi_m)) cos(2 pi o rev
    + phi_c)`` the first sidebands have the relative level ``m / 2``, i.e.
    ``m = 2 * 10**(level / 20)``.

    Phases. With ``modulation=None`` no phase is set and ``synth_runup`` draws
    the phases of all components at random: the sideband levels are exact but
    the relation of the sidebands to the carrier is not that of a modulation.
    With ``modulation="am"`` the sidebands ``k`` have the phases
    ``phi_c +- k phi_m`` (in phase with the carrier envelope: for symmetric
    first sidebands pure amplitude modulation). With ``modulation="pm"``
    (phase modulation ``cos(Phi_c + beta sin(Phi_m))``, Bessel structure) the
    upper sideband has the phase ``phi_c + k phi_m`` and the lower sideband
    ``phi_c - k phi_m + k pi``, so that for a small index ``beta`` and levels
    ``beta / 2`` the first sidebands give ``cos(Phi_c) - beta sin(Phi_c)
    sin(Phi_m)``, where ``Phi_m = 2 pi q rev + phi_m`` (``q`` the modulating
    order). The sideband pair has the same level in both models; only the
    phases differ, and the level of one side can be set differently
    (``levels_db`` with two columns) for an asymmetric spectrum.

    A lower sideband at a negative order is folded to the positive order with
    the phase negated (a real signal); one at order zero (a DC term, not an
    order) is dropped.

    Parameters
    ----------
    carrier_order : float
        Carrier (e.g. gear-mesh) order, in cycles per revolution of ``shaft``
        (positive).
    mod_order : float
        Modulating order, positive.
    levels_db : array_like, shape (K,) or (K, 2)
        Sideband levels in dB relative to the carrier amplitude for
        ``k = 1 ... K``: one value per pair (both sidebands equal) or two
        columns (lower, upper). ``-inf`` gives an absent sideband.
    carrier_amplitude : float, default 1
        Peak amplitude of the carrier (non-negative).
    shaft : int, default 0
        Shaft index passed to the orders.
    modulation : {None, "am", "pm"}
        Phase relation of the sidebands to the carrier, see above.
    carrier_phase, mod_phase : float
        ``phi_c`` and ``phi_m`` in rad, used when ``modulation`` is not None.

    Returns
    -------
    list of dict
        ``1 + 2 K`` order dictionaries with the keys ``order``, ``amplitude``
        (peak, same unit as ``carrier_amplitude``), ``shaft`` and, with a
        modulation, ``phase`` (rad).

    References
    ----------
    Randall, R. B. (1982). A new method of modeling gear faults. Journal of
    Mechanical Design, 104(2).
    Randall, R. B. (2011). Vibration-based Condition Monitoring: Industrial,
    Automotive and Aerospace Applications. Wiley (amplitude and phase
    modulation sidebands of gear meshing).
    """
    carrier_order = _positive(carrier_order, "carrier_order")
    mod_order = _positive(mod_order, "mod_order")
    carrier_amplitude = float(carrier_amplitude)
    if not (np.isfinite(carrier_amplitude) and carrier_amplitude >= 0):
        raise ValueError("carrier_amplitude must be finite and non-negative")
    if int(shaft) != shaft or shaft < 0:
        raise ValueError("shaft must be a non-negative integer")
    if modulation not in (None, "am", "pm"):
        raise ValueError("modulation must be None, 'am' or 'pm'")
    levels = np.atleast_1d(np.asarray(levels_db, dtype=float))
    if levels.ndim == 1:
        levels = np.column_stack([levels, levels])
    if levels.ndim != 2 or levels.shape[1] != 2 or levels.shape[0] < 1:
        raise ValueError("levels_db must have shape (K,) or (K, 2) with K >= 1")
    if np.any(np.isnan(levels)) or np.any(levels == np.inf):
        raise ValueError("levels_db must not contain NaN or +inf")
    carrier_phase, mod_phase = float(carrier_phase), float(mod_phase)
    if not (np.isfinite(carrier_phase) and np.isfinite(mod_phase)):
        raise ValueError("carrier_phase and mod_phase must be finite")

    first = {"order": carrier_order, "amplitude": carrier_amplitude, "shaft": int(shaft)}
    if modulation is not None:
        first["phase"] = carrier_phase
    out = [first]
    amp = carrier_amplitude * 10.0 ** (levels / 20.0)
    for kk in range(1, levels.shape[0] + 1):
        for side, sign in ((0, -1), (1, 1)):
            order = carrier_order + sign * kk * mod_order
            phase = None
            if modulation is not None:
                phase = carrier_phase + sign * kk * mod_phase
                if modulation == "pm" and sign < 0:
                    phase += kk * np.pi
            if abs(order) <= 1e-12 * max(carrier_order, kk * mod_order):
                continue  # a DC term is not an order
            if order < 0:
                order = -order
                phase = None if phase is None else -phase
            spec = {"order": order, "amplitude": float(amp[kk - 1, side]), "shaft": int(shaft)}
            if phase is not None:
                spec["phase"] = float(phase)
            out.append(spec)
    return out


# --------------------------------------------------------------------------
# 5. components around a fixed carrier
# --------------------------------------------------------------------------


def carrier_components(rev, fs, rpm, *, carrier_hz, terms, seed=None):
    """Components around a fixed carrier frequency with speed-dependent sidebands.

    Pulse-width-modulated converters feeding a machine radiate noise at the
    switching (carrier) frequency and at sidebands displaced from it by
    multiples of the fundamental of the output voltage, which follows the
    rotor speed. Component ``j`` of ``terms = [(q_j, A_j), ...]`` is

    ``x_j(t) = A_j cos(Phi_j(t))``,
    ``Phi_j = 2 pi (carrier_hz t + q_j rev(t)) + phi_j``,

    with instantaneous frequency
    ``f_j = carrier_hz + q_j rpm / 60`` (``q_j = m * order``: a signed multiple
    of the rotation order, 0 for the carrier itself). The initial phases
    ``phi_j`` are drawn uniformly from the generator.

    Frequencies at or above ``fs/2`` or at or below 0 Hz are muted (amplitude
    set to 0), never clipped to the band edge, which would create a false tone
    (ROADMAP section 6, item 16). The truth ``frequency`` keeps the unmuted
    value. ``rpm`` must be the derivative of ``rev`` (``rpm_inst`` of
    :func:`angular_speed_ripple`, or the speed that ``rev`` was integrated from)
    so that the frequency truth is the derivative of the phase truth; it is
    used for nothing else.

    Parameters
    ----------
    rev : array_like, shape (n,)
        Shaft angle in revolutions at the samples (truth).
    fs : float
        Sampling rate in Hz.
    rpm : array_like, shape (n,)
        Shaft speed in rpm, consistent with ``rev``.
    carrier_hz : float
        Carrier (switching) frequency in Hz, non-negative.
    terms : sequence of (float, float)
        ``(q, amplitude)`` pairs: ``q`` the signed order offset in cycles per
        revolution (negative gives a lower sideband) and ``amplitude`` the
        peak amplitude (non-negative).
    seed : int, numpy.random.Generator or None
        Random source for the initial phases.

    Returns
    -------
    dict
        ``x`` (n,): the sum of the components (peak amplitudes).
        ``stems`` (n, J): each component after muting. ``amplitude`` (n, J):
        peak envelope (``A_j`` or 0 where muted). ``phase`` (n, J): total phase
        in rad. ``frequency`` (n, J): instantaneous frequency in Hz, the
        derivative of ``phase`` over ``2 pi``, not muted. ``phase0`` (J,):
        the drawn ``phi_j``. ``q`` (J,): the order offsets.

    References
    ----------
    Holmes, D. G., & Lipo, T. A. (2003). Pulse Width Modulation for Power
    Converters: Principles and Practice. IEEE Press / Wiley (carrier and
    sideband harmonics of PWM waveforms).
    Boashash, B. (1992). Estimating and interpreting the instantaneous
    frequency of a signal, Part 1: Fundamentals. Proceedings of the IEEE,
    80(4), 520-538.
    """
    rev = _vector(rev, "rev")
    n = rev.size
    rpm = _vector(rpm, "rpm", n)
    fs = _positive(fs, "fs")
    carrier_hz = float(carrier_hz)
    if not (np.isfinite(carrier_hz) and carrier_hz >= 0):
        raise ValueError("carrier_hz must be finite and non-negative (Hz)")
    tm = np.asarray(terms, dtype=float)
    if tm.ndim != 2 or tm.shape[1] != 2 or tm.shape[0] < 1:
        raise ValueError("terms must be a sequence of (q, amplitude) pairs")
    if not np.all(np.isfinite(tm)) or np.any(tm[:, 1] < 0):
        raise ValueError("terms must be finite with non-negative amplitudes")
    q, amp = tm[:, 0], tm[:, 1]
    phase0 = np.random.default_rng(seed).uniform(0.0, _TWO_PI, size=q.size)
    t = np.arange(n) / fs
    freq = carrier_hz + np.outer(rpm / 60.0, q)
    phase = _TWO_PI * (carrier_hz * t[:, None] + np.outer(rev, q)) + phase0[None, :]
    amplitude = np.where((freq > 0.0) & (freq < 0.5 * fs), amp[None, :], 0.0)
    stems = amplitude * np.cos(phase)
    return {
        "x": stems.sum(axis=1),
        "stems": stems,
        "amplitude": amplitude,
        "phase": phase,
        "frequency": freq,
        "phase0": phase0,
        "q": q,
    }
