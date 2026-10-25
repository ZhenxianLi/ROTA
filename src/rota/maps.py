"""RPM-frequency (Campbell) and RPM-order maps.

Both maps are short-time Fourier transforms: over time for the frequency map,
over shaft angle for the order map. Values are complex amplitude spectra
scaled so that a sinusoid of amplitude ``A`` (peak) that falls exactly on a
bin reads ``A`` (``amplitude="peak"``) or ``A/sqrt(2)``
(``amplitude="rms"``). Between bins the reading drops (scalloping); use
:func:`rota.slices.order_slice`, which sums power over a band, for accurate
order levels.

Phases are referenced to the start of the signal: for
``x = A cos(2 pi f t + phi)`` the frequency-map bin at ``f`` holds
``A exp(i phi)`` (peak scaling), and for ``x = A cos(2 pi o rev + phi)`` the
order-map bin at order ``o`` holds ``A exp(i phi)``, with ``rev`` the shaft
angle in revolutions from :func:`rota.speed.rpm_to_phase`.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal

from ._utils import as_2d, broadcast_rpm, check_amplitude, restore
from .angle import angular_resample

__all__ = ["rpm_frequency_map", "rpm_order_map", "Map"]


@dataclass
class Map:
    """Short-time spectrum tagged with speed.

    Attributes
    ----------
    values : ndarray, complex, shape (n_bins, n_frames) or (n_bins, n_frames, ch)
        Complex amplitude spectrum, in the input unit, with the amplitude
        convention given by ``amplitude``.
    axis : ndarray, shape (n_bins,)
        Bin centres: frequency in Hz (``domain="frequency"``) or order
        (``domain="order"``).
    domain : str
        ``"frequency"`` or ``"order"``.
    t : ndarray, shape (n_frames,)
        Frame centre times in s.
    rpm : ndarray, shape (n_frames,)
        Mean speed over each frame in rpm.
    rpm_min, rpm_max : ndarray, shape (n_frames,)
        Speed range within each frame.
    amplitude : str
        ``"peak"`` or ``"rms"``.
    enbw_bins : float
        Equivalent noise bandwidth of the window in bins. The amplitude of a
        component spread over bins ``k`` is ``sqrt(sum |values[k]|**2 / enbw_bins)``.
    window : str
        Window name.
    nperseg, hop : int
        Frame length and hop in samples (time or angle samples).
    """

    values: np.ndarray
    axis: np.ndarray
    domain: str
    t: np.ndarray
    rpm: np.ndarray
    rpm_min: np.ndarray
    rpm_max: np.ndarray
    amplitude: str
    enbw_bins: float
    window: str
    nperseg: int
    hop: int

    @property
    def magnitude(self):
        """Magnitude ``|values|`` in the map's amplitude convention."""
        return np.abs(self.values)

    @property
    def resolution(self):
        """Bin spacing in Hz or orders."""
        return float(self.axis[1] - self.axis[0])


def _stft(X, nperseg, hop, window, amplitude, sample_rate):
    """Frames, scaled complex spectra and frame start indices.

    ``sample_rate`` is samples per second (time) or per revolution (angle);
    the bin axis is ``k * sample_rate / nperseg``.
    """
    n = X.shape[0]
    if nperseg > n:
        raise ValueError("signal shorter than one frame")
    w = signal.get_window(window, nperseg)
    starts = np.arange(0, n - nperseg + 1, hop)
    frames = np.lib.stride_tricks.sliding_window_view(X, nperseg, axis=0)[starts]
    # frames: (n_frames, ch, nperseg)
    F = np.fft.rfft(frames * w, axis=-1)
    k = np.arange(F.shape[-1])
    F *= np.exp(-2j * np.pi * np.outer(starts, k) / nperseg)[:, None, :]
    scale = np.full(k.size, 2.0 / w.sum())
    scale[0] = 1.0 / w.sum()
    if nperseg % 2 == 0:
        scale[-1] = 1.0 / w.sum()
    if amplitude == "rms":
        scale /= np.sqrt(2.0)
        scale[0] = 1.0 / w.sum()
    F *= scale
    values = np.transpose(F, (2, 0, 1))  # (bins, frames, ch)
    axis = k * sample_rate / nperseg
    enbw = nperseg * np.sum(w**2) / np.sum(w) ** 2
    return values, axis, starts, float(enbw)


def _frame_stats(v, starts, nperseg):
    """Mean, min and max of ``v`` over each frame."""
    win = np.lib.stride_tricks.sliding_window_view(v, nperseg)[starts]
    return win.mean(axis=1), win.min(axis=1), win.max(axis=1)


def _hop(nperseg, overlap):
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    return max(1, int(round(nperseg * (1.0 - overlap))))


def rpm_frequency_map(
    x,
    fs,
    rpm,
    *,
    resolution=None,
    nperseg=None,
    overlap=0.75,
    window="hann",
    amplitude="rms",
    max_freq=None,
):
    """Frequency-RPM map (Campbell diagram) by short-time Fourier transform.

    Each time frame is tagged with the mean, minimum and maximum speed within
    it. The frequency resolution is the same at every speed, so at low speed
    a fixed-length window spans several orders; use :func:`rpm_order_map`
    when orders must be resolved.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal, any unit (e.g. Pa).
    fs : float
        Sampling rate in Hz.
    rpm : float or array_like, shape (n,)
        Speed in rpm at the signal samples.
    resolution : float, optional
        Bin spacing in Hz; sets ``nperseg = round(fs / resolution)``.
    nperseg : int, optional
        Frame length in samples (default: frames of 0.1 s).
    overlap : float, default 0.75
        Frame overlap fraction.
    window : str, default "hann"
        Window name accepted by :func:`scipy.signal.get_window` (periodic).
        Use ``"flattop"`` for scalloping-free peak readings.
    amplitude : {"rms", "peak"}, default "rms"
        Amplitude convention of the values.
    max_freq : float, optional
        Discard bins above this frequency in Hz.

    Returns
    -------
    Map
        With ``domain="frequency"`` and ``axis`` in Hz.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 9 and 12. Wiley.
    Allen, J. B., & Rabiner, L. R. (1977). A unified approach to short-time
    Fourier analysis and synthesis. Proceedings of the IEEE, 65(11),
    1558-1564.
    """
    check_amplitude(amplitude)
    X, was_1d = as_2d(x)
    rpm = broadcast_rpm(rpm, X.shape[0])
    if nperseg is None:
        nperseg = int(round(fs / resolution)) if resolution else int(round(0.1 * fs))
    hop = _hop(nperseg, overlap)
    values, axis, starts, enbw = _stft(X, nperseg, hop, window, amplitude, fs)
    if max_freq is not None:
        keep = axis <= max_freq
        values, axis = values[keep], axis[keep]
    r_mean, r_min, r_max = _frame_stats(rpm, starts, nperseg)
    return Map(
        values=restore(values, was_1d),
        axis=axis,
        domain="frequency",
        t=(starts + 0.5 * (nperseg - 1)) / fs,
        rpm=r_mean,
        rpm_min=r_min,
        rpm_max=r_max,
        amplitude=amplitude,
        enbw_bins=enbw,
        window=window,
        nperseg=nperseg,
        hop=hop,
    )


def rpm_order_map(
    x,
    fs,
    rpm,
    *,
    order_resolution=0.25,
    max_order=None,
    samples_per_rev=None,
    overlap=0.75,
    window="hann",
    amplitude="rms",
    min_rpm=None,
    delay=0.0,
):
    """Order-RPM map by short-time Fourier transform in the angle domain.

    The signal is resampled to uniform shaft angle with
    :func:`rota.angle.angular_resample` and transformed with frames of fixed
    length in revolutions, ``1 / order_resolution``. The order resolution is
    therefore the same at every speed: half orders stay separated at idle,
    which a fixed-length time window cannot achieve.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal, any unit (e.g. Pa).
    fs : float
        Sampling rate in Hz.
    rpm : float or array_like, shape (n,)
        Speed in rpm at the signal samples.
    order_resolution : float, default 0.25
        Bin spacing in orders; the frame spans ``1/order_resolution``
        revolutions.
    max_order, samples_per_rev, min_rpm, delay
        Passed to :func:`rota.angle.angular_resample`.
    overlap : float, default 0.75
        Frame overlap fraction (in angle).
    window : str, default "hann"
        Window name accepted by :func:`scipy.signal.get_window` (periodic).
    amplitude : {"rms", "peak"}, default "rms"
        Amplitude convention of the values.

    Returns
    -------
    Map
        With ``domain="order"`` and ``axis`` in orders, truncated at
        ``max_order``.

    References
    ----------
    Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order
    tracking. Mechanical Systems and Signal Processing, 11(2), 187-205.
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    """
    check_amplitude(amplitude)
    a = angular_resample(
        x, fs, rpm, samples_per_rev=samples_per_rev, max_order=max_order,
        min_rpm=min_rpm, delay=delay,
    )
    X, was_1d = as_2d(a.x)
    spr = a.samples_per_rev
    nperseg = int(round(spr / order_resolution))
    hop = _hop(nperseg, overlap)
    values, axis, starts, enbw = _stft(X, nperseg, hop, window, amplitude, spr)
    keep = axis <= a.max_order + 1e-9
    values, axis = values[keep], axis[keep]
    r_mean, r_min, r_max = _frame_stats(a.rpm, starts, nperseg)
    t_c = np.interp(starts + 0.5 * (nperseg - 1), np.arange(a.t.size), a.t)
    return Map(
        values=restore(values, was_1d),
        axis=axis,
        domain="order",
        t=t_c,
        rpm=r_mean,
        rpm_min=r_min,
        rpm_max=r_max,
        amplitude=amplitude,
        enbw_bins=enbw,
        window=window,
        nperseg=nperseg,
        hop=hop,
    )
