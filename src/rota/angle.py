"""Angle-domain resampling (computed order tracking).

Units: time in s, speed in rpm, angle in revolutions, order in cycles per
revolution.
"""

from dataclasses import dataclass

import numpy as np
from scipy import interpolate, signal

from ._utils import as_2d, broadcast_rpm, restore
from .speed import rpm_to_phase

__all__ = ["angular_resample", "AngularSignal"]

# Cap on the intermediate (oversampled) angle-domain length, in samples.
_MAX_INTERMEDIATE = 200_000_000


@dataclass
class AngularSignal:
    """Signal sampled uniformly in shaft angle.

    Attributes
    ----------
    x : ndarray, shape (m,) or (m, ch)
        Signal values, same unit and amplitude as the input.
    rev : ndarray, shape (m,)
        Shaft angle of each sample in revolutions (``k / samples_per_rev``).
    t : ndarray, shape (m,)
        Time of each sample in s.
    rpm : ndarray, shape (m,)
        Speed at each sample in rpm.
    samples_per_rev : int
        Angular sampling rate in samples per revolution.
    max_order : float
        Highest order kept by the anti-aliasing filter.
    """

    x: np.ndarray
    rev: np.ndarray
    t: np.ndarray
    rpm: np.ndarray
    samples_per_rev: int
    max_order: float


def angular_resample(
    x,
    fs,
    rpm,
    *,
    samples_per_rev=None,
    max_order=None,
    min_rpm=None,
    delay=0.0,
    attenuation=80.0,
):
    """Resample a time signal to uniform shaft-angle increments.

    The shaft angle is obtained from ``rpm`` with :func:`rota.speed.rpm_to_phase`.
    To avoid aliasing whatever the speed, the signal is first interpolated
    (cubic spline) onto an intermediate angle grid dense enough that the
    time-domain Nyquist frequency at the *lowest* speed maps below the
    intermediate angular Nyquist order; there the interpolation is a local
    upsampling and creates no aliases. The intermediate signal is then
    low-pass filtered *in the angle domain* (linear-phase Kaiser FIR) and
    decimated to ``samples_per_rev``. Because the filter acts on order, not
    frequency, the anti-aliasing cut-off follows the speed: orders above
    ``max_order`` are removed at every speed, and nothing below it is lost
    at high speed.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal sampled at ``fs``. Channels are processed identically.
    fs : float
        Sampling rate in Hz.
    rpm : float or array_like, shape (n,)
        Shaft speed in rpm at the signal samples.
    samples_per_rev : int, optional
        Output angular sampling rate. Default: ``ceil(2.5 * max_order)``
        rounded up to an even number.
    max_order : float, optional
        Highest order to keep (pass band edge). Default:
        ``0.45 * fs / (max(rpm)/60)``, the highest order representable at the
        top speed; if ``samples_per_rev`` is given, ``0.4 * samples_per_rev``.
    min_rpm : float, optional
        Floor applied to ``rpm`` before integration. Default: 2 % of the
        maximum speed. The angle is wrong wherever the true speed is below the
        floor.
    delay : float, default 0
        Speed-reference delay in s, see :func:`rota.speed.rpm_to_phase`.
    attenuation : float, default 80
        Stop-band attenuation in dB of the angle-domain anti-aliasing filter.

    Returns
    -------
    AngularSignal
        Amplitudes are preserved sample by sample (peak and RMS conventions
        are unchanged); the first and last few revolutions carry filter edge
        effects of about ``numtaps / (2 * samples_per_rev * M)`` revolutions.

    References
    ----------
    Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order
    tracking. Mechanical Systems and Signal Processing, 11(2), 187-205.
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    Kaiser, J. F. (1974). Nonrecursive digital filter design using the
    I0-sinh window function. Proc. IEEE Int. Symp. Circuits and Systems,
    20-23.
    """
    X, was_1d = as_2d(x)
    n = X.shape[0]
    rpm = broadcast_rpm(rpm, n)
    rpm_max = float(rpm.max())
    if rpm_max <= 0:
        raise ValueError("rpm must be positive somewhere")
    if min_rpm is None:
        min_rpm = 0.02 * rpm_max
    if min_rpm <= 0:
        raise ValueError("min_rpm must be positive")

    if max_order is None:
        if samples_per_rev is not None:
            max_order = 0.4 * samples_per_rev
        else:
            max_order = 0.45 * fs / (rpm_max / 60.0)
    if samples_per_rev is None:
        samples_per_rev = int(np.ceil(2.5 * max_order))
        samples_per_rev += samples_per_rev % 2
    samples_per_rev = int(samples_per_rev)
    if max_order >= 0.5 * samples_per_rev:
        raise ValueError("max_order must be below samples_per_rev / 2")

    rpm_eff = np.maximum(rpm, min_rpm)
    rev = rpm_to_phase(rpm_eff, fs, delay=delay)
    rpm_eff = np.maximum(
        np.interp(np.arange(n) / fs + delay, np.arange(n) / fs, rpm), min_rpm
    )
    rpm_lo = float(rpm_eff.min())

    # Intermediate angular rate: an integer multiple M of the output rate with
    # at least fs * 60 / rpm_lo samples per revolution.
    M = max(1, int(np.ceil(fs * 60.0 / rpm_lo / samples_per_rev)))
    spr_hi = samples_per_rev * M
    m_hi = int(np.floor(rev[-1] * spr_hi)) + 1
    if m_hi * X.shape[1] > _MAX_INTERMEDIATE:
        raise MemoryError(
            "intermediate angle grid too large; raise min_rpm or lower fs"
        )
    rev_hi = np.arange(m_hi) / spr_hi
    t_hi = np.interp(rev_hi, rev, np.arange(n) / fs)
    spline = interpolate.make_interp_spline(np.arange(n), X, k=3, axis=0)
    x_hi = spline(t_hi * fs)

    if M > 1:
        nyq_hi = 0.5 * spr_hi
        edge = 0.5 * samples_per_rev
        width = (edge - max_order) / nyq_hi
        numtaps, beta = signal.kaiserord(attenuation, width)
        numtaps += 1 - numtaps % 2
        h = signal.firwin(numtaps, 0.5 * (max_order + edge) / nyq_hi, window=("kaiser", beta))
        h /= h.sum()
        y = signal.resample_poly(x_hi, 1, M, axis=0, window=h)
    else:
        y = x_hi
    m = y.shape[0]
    rev_out = np.arange(m) / samples_per_rev
    t_out = t_hi[::M][:m]
    rpm_out = np.interp(t_out, np.arange(n) / fs, rpm_eff)
    return AngularSignal(
        x=restore(y, was_1d),
        rev=rev_out,
        t=t_out,
        rpm=rpm_out,
        samples_per_rev=samples_per_rev,
        max_order=float(max_order),
    )
