"""RPM-frequency (Campbell) and RPM-order maps.

Both maps are short-time Fourier transforms: over time for the frequency map,
over shaft angle for the order map. Values are complex amplitude spectra
scaled so that a sinusoid of amplitude ``A`` (peak) that falls exactly on a
bin reads ``A`` (``amplitude="peak"``) or ``A/sqrt(2)``
(``amplitude="rms"``). Between bins the reading drops (scalloping); use
:func:`rota_order.slices.order_slice`, which sums power over a band, for accurate
order levels.

Phases are referenced to the start of the signal: for
``x = A cos(2 pi f t + phi)`` the frequency-map bin at ``f`` holds
``A exp(i phi)`` (peak scaling), and for ``x = A cos(2 pi o rev + phi)`` the
order-map bin at order ``o`` holds ``A exp(i phi)``, with ``rev`` the shaft
angle in revolutions from :func:`rota_order.speed.rpm_to_phase`.

Both maps can optionally be aggregated per rpm bin (``rpm_bins``, ``segment``;
see :func:`bin_map`), which turns the per-frame map into one column per speed
bin with a frame count, for run-ups and run-downs where the same speed is
passed several times or where the speed is not monotonic (gear shifts).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from ._utils import as_2d, broadcast_rpm, check_amplitude, restore
from .angle import angular_resample

__all__ = ["rpm_frequency_map", "rpm_order_map", "Map", "bin_map", "BinnedMap"]


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
        if self.axis.size < 2:
            raise ValueError("map has fewer than two bins; resolution undefined")
        return float(self.axis[1] - self.axis[0])


@dataclass
class BinnedMap(Map):
    """A :class:`Map` aggregated per rpm bin by :func:`bin_map`.

    Behaves as a :class:`Map` with one column per populated rpm bin
    (``values`` has ``n_bins`` columns in place of ``n_frames``), so the
    functions of :mod:`rota_order.slices` accept it. In the inherited fields,
    ``t`` is the mean time of the frames in each bin, ``rpm`` their mean speed
    (the operating point of the bin; the bin centre is ``rpm_centre``) and
    ``rpm_min``, ``rpm_max`` the extreme speeds reached within them.

    Attributes
    ----------
    counts : ndarray of int, shape (n_bins,)
        Number of map frames aggregated in each bin.
    rpm_lo, rpm_hi : ndarray, shape (n_bins,)
        Lower (inclusive) and upper (exclusive, except for the last edge of
        the grid) edge of each bin in rpm.
    segment : str or None
        Segment that was selected: ``"runup"``, ``"rundown"`` or None (all
        frames); ``"auto"`` is resolved to one of the first two.
    phase : str
        ``"drop"``: values are real and non-negative (zero imaginary part);
        ``"circular"``: the amplitude carries the phase of the circular mean of
        the frame phases (see :func:`bin_map`).
    """

    counts: np.ndarray = field(default=None)
    rpm_lo: np.ndarray = field(default=None)
    rpm_hi: np.ndarray = field(default=None)
    segment: object = None
    phase: str = "drop"

    @property
    def rpm_centre(self):
        """Bin centres in rpm, ``(rpm_lo + rpm_hi) / 2``."""
        return 0.5 * (self.rpm_lo + self.rpm_hi)


_SEGMENTS = (None, "runup", "rundown", "auto")
_BIN_PHASES = ("drop", "circular")


def _segment_mask(rpm, segment):
    """Boolean mask of frames on a rising or falling stretch; resolved name.

    The direction of frame ``i`` is the sign of the central difference of the
    frame speeds (one-sided at the ends). Frames with zero slope belong to
    neither direction. There is no dead band, so the choice is memoryless
    (no hysteresis): it depends only on the neighbouring frames.
    """
    if segment == "auto":
        step = np.diff(rpm)
        segment = "runup" if np.sum(np.maximum(step, 0.0)) >= np.sum(np.maximum(-step, 0.0)) else "rundown"
    if rpm.size < 2:
        return np.zeros(rpm.size, bool), segment
    g = np.gradient(rpm)
    return (g > 0 if segment == "runup" else g < 0), segment


def _bin_edges(rpm_bins, r):
    """Strictly increasing bin edges from a width or an explicit edge array."""
    b = np.asarray(rpm_bins, dtype=float)
    if b.ndim == 0:
        w = float(b)
        if not (np.isfinite(w) and w > 0):
            raise ValueError("rpm_bins as a width must be a positive number of rpm")
        if r.size == 0:
            return np.array([0.0, w])
        k0 = np.floor(r.min() / w)
        k1 = np.floor(r.max() / w) + 1.0
        return w * np.arange(k0, k1 + 1.0)
    if b.ndim != 1 or b.size < 2 or not np.all(np.isfinite(b)) or np.any(np.diff(b) <= 0):
        raise ValueError("rpm_bins must be a positive width or a strictly increasing edge array (>= 2 edges)")
    return b


def bin_map(m, rpm_bins, *, segment=None, min_count=1, phase="drop"):
    """Aggregate the frames of a map per rpm bin (energy average).

    Frames are assigned to speed bins by their mean speed ``m.rpm`` and,
    optionally, restricted first to the rising (run-up) or falling
    (run-down) part of a non-monotonic speed record, so that a gear shift or
    a lift-off does not mix the opposite direction into the bins. Per bin and
    channel, the aggregate amplitude is the energy mean of the frames,
    ``sqrt(mean(|V|^2))``, which keeps the amplitude convention of the map
    (peak or RMS) and the meaning of ``enbw_bins``: a stationary component
    keeps its amplitude, and for stationary noise the result is the
    root-mean-square amplitude per bin, the quantity averaged over frames by
    :func:`rota_order.slices.order_spectrum`. Amplitudes of components that change
    with speed are averaged in energy over the speed range of the bin, so the
    bin should be narrow compared with the variation of the amplitude.

    Segment selection. The direction of a frame is the sign of the central
    difference of the frame speeds (one-sided at the first and last frame).
    ``"runup"`` keeps frames with positive slope, ``"rundown"`` those with
    negative slope; frames at a turning point or at constant speed belong to
    neither, and a segment keeps every stretch of its direction (both rises
    around a gear shift, for instance). ``"auto"`` picks the direction that
    covers the larger total speed change (sum of the positive against the
    sum of the negative frame-to-frame steps). The rule is memoryless
    (no hysteresis, no dead band): speed with jitter larger than the
    frame-to-frame change should be smoothed first.

    Parameters
    ----------
    m : Map
        Map from :func:`rpm_frequency_map` or :func:`rpm_order_map` (per
        frame; binning an already binned map is an error).
    rpm_bins : float or array_like
        A bin width in rpm (edges are multiples of the width, so that bins
        are comparable between runs), or strictly increasing bin edges in
        rpm. Bins are ``[lo, hi)``; the last edge of an edge array is
        inclusive. Frames outside the edges are discarded.
    segment : {None, "runup", "rundown", "auto"}, default None
        Frame selection described above; None keeps all frames.
    min_count : int, default 1
        Bins with fewer frames are dropped. Only populated bins are
        returned.
    phase : {"drop", "circular"}, default "drop"
        ``"drop"``: the phase is discarded and the returned values are real
        (stored as complex with zero imaginary part). ``"circular"``: the
        phase is that of the sum of the unit phasors ``V/|V|`` of the
        frames (circular mean, Mardia & Jupp 2000), zero where they cancel
        exactly. Phases of an order map are referenced to the signal start
        (see the module docstring), so a circular mean is meaningful for
        components locked to the shaft angle only; for frequency maps it is
        only meaningful for stationary tones.

    Returns
    -------
    BinnedMap
        One column per populated bin, in the amplitude convention of ``m``
        and the unit of the analysed signal, with ``counts`` frames per bin.
        ``values`` has the shape of ``m.values`` with ``n_bins`` in place of
        ``n_frames``.

    Raises
    ------
    ValueError
        For an invalid ``rpm_bins``, ``segment``, ``min_count`` or ``phase``,
        or if ``m`` is already binned.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    Mardia, K. V., & Jupp, P. E. (2000). Directional Statistics, ch. 2.
    Wiley.
    Welch, P. D. (1967). The use of fast Fourier transform for the estimation
    of power spectra. IEEE Transactions on Audio and Electroacoustics, 15(2),
    70-73.
    """
    if isinstance(m, BinnedMap):
        raise ValueError("map is already binned")
    if segment not in _SEGMENTS:
        raise ValueError(f"segment must be one of {_SEGMENTS}, got {segment!r}")
    if phase not in _BIN_PHASES:
        raise ValueError(f"phase must be one of {_BIN_PHASES}, got {phase!r}")
    if int(min_count) != min_count or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    rpm = np.asarray(m.rpm, dtype=float)
    keep = np.ones(rpm.size, bool)
    resolved = None
    if segment is not None:
        keep, resolved = _segment_mask(rpm, segment)
    edges = _bin_edges(rpm_bins, rpm[keep])
    nb = edges.size - 1
    idx = np.searchsorted(edges, rpm, side="right") - 1
    idx[rpm == edges[-1]] = nb - 1  # closed last edge
    keep &= (idx >= 0) & (idx < nb)

    v = np.asarray(m.values)
    out_bins, cols, counts = [], [], []
    t_o, r_o, rmin_o, rmax_o = [], [], [], []
    for b in range(nb):
        sel = np.flatnonzero(keep & (idx == b))
        if sel.size < min_count:
            continue
        vb = v[:, sel]
        amp = np.sqrt(np.mean(np.abs(vb) ** 2, axis=1))
        if phase == "circular":
            mag = np.abs(vb)
            unit = np.divide(vb, mag, out=np.zeros_like(vb, dtype=complex), where=mag > 0)
            z = unit.sum(axis=1)
            zm = np.abs(z)
            ph = np.divide(z, zm, out=np.ones_like(z, dtype=complex), where=zm > 0)
            cols.append(amp * ph)
        else:
            cols.append(amp.astype(complex))
        out_bins.append(b)
        counts.append(sel.size)
        t_o.append(np.mean(m.t[sel]))
        r_o.append(np.mean(rpm[sel]))
        rmin_o.append(np.min(m.rpm_min[sel]))
        rmax_o.append(np.max(m.rpm_max[sel]))
    ob = np.asarray(out_bins, dtype=int)
    if cols:
        values = np.stack(cols, axis=1)
    else:
        values = np.zeros(v.shape[:1] + (0,) + v.shape[2:], dtype=complex)
    return BinnedMap(
        values=values,
        axis=m.axis,
        domain=m.domain,
        t=np.asarray(t_o, dtype=float),
        rpm=np.asarray(r_o, dtype=float),
        rpm_min=np.asarray(rmin_o, dtype=float),
        rpm_max=np.asarray(rmax_o, dtype=float),
        amplitude=m.amplitude,
        enbw_bins=m.enbw_bins,
        window=m.window,
        nperseg=m.nperseg,
        hop=m.hop,
        counts=np.asarray(counts, dtype=int),
        rpm_lo=edges[ob],
        rpm_hi=edges[ob + 1],
        segment=resolved,
        phase=phase,
    )


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
    if amplitude == "rms":
        scale /= np.sqrt(2.0)
    # DC and Nyquist bins are one-sided: peak and RMS coincide there.
    scale[0] = 1.0 / w.sum()
    if nperseg % 2 == 0:
        scale[-1] = 1.0 / w.sum()
    F *= scale
    values = np.transpose(F, (2, 0, 1))  # (bins, frames, ch)
    axis = k * sample_rate / nperseg
    enbw = nperseg * np.sum(w**2) / np.sum(w) ** 2
    return values, axis, starts, float(enbw)


def _frame_stats(v, starts, nperseg):
    """Mean, min and max of ``v`` over each frame."""
    win = np.lib.stride_tricks.sliding_window_view(v, nperseg)[starts]
    return win.mean(axis=1), win.min(axis=1), win.max(axis=1)


def _check_binning(rpm_bins, segment):
    if rpm_bins is None and segment is not None:
        raise ValueError("segment selection needs rpm_bins (a bin width or bin edges)")


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
    rpm_bins=None,
    segment=None,
    min_count=1,
    bin_phase="drop",
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
    rpm_bins : float or array_like, optional
        Aggregate the frames per speed bin with :func:`bin_map`: a bin width
        in rpm or bin edges. The default None returns one column per time
        frame.
    segment : {None, "runup", "rundown", "auto"}, optional
        Monotonic segment selection before binning (needs ``rpm_bins``); see
        :func:`bin_map`.
    min_count : int, default 1
        Minimum number of frames per bin (``rpm_bins`` only).
    bin_phase : {"drop", "circular"}, default "drop"
        Phase of the binned values, see :func:`bin_map`.

    Returns
    -------
    Map
        With ``domain="frequency"`` and ``axis`` in Hz. A
        :class:`BinnedMap` (a :class:`Map` with per-bin ``counts``) when
        ``rpm_bins`` is given.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 9 and 12. Wiley.
    Allen, J. B., & Rabiner, L. R. (1977). A unified approach to short-time
    Fourier analysis and synthesis. Proceedings of the IEEE, 65(11),
    1558-1564.
    """
    check_amplitude(amplitude)
    _check_binning(rpm_bins, segment)
    X, was_1d = as_2d(x)
    rpm = broadcast_rpm(rpm, X.shape[0])
    if resolution is not None and not resolution > 0:
        raise ValueError("resolution must be positive (Hz)")
    if nperseg is None:
        nperseg = int(round(fs / resolution)) if resolution else int(round(0.1 * fs))
    hop = _hop(nperseg, overlap)
    values, axis, starts, enbw = _stft(X, nperseg, hop, window, amplitude, fs)
    if max_freq is not None:
        keep = axis <= max_freq
        values, axis = values[keep], axis[keep]
    r_mean, r_min, r_max = _frame_stats(rpm, starts, nperseg)
    m = Map(
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
    if rpm_bins is None:
        return m
    return bin_map(m, rpm_bins, segment=segment, min_count=min_count, phase=bin_phase)


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
    rpm_bins=None,
    segment=None,
    min_count=1,
    bin_phase="drop",
):
    """Order-RPM map by short-time Fourier transform in the angle domain.

    The signal is resampled to uniform shaft angle with
    :func:`rota_order.angle.angular_resample` and transformed with frames of fixed
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
    max_order, samples_per_rev, delay
        Passed to :func:`rota_order.angle.angular_resample`.
    min_rpm : float, optional
        Lowest speed of the map, in rpm. :func:`~rota_order.angle.angular_resample`
        floors the speed at ``min_rpm`` before integrating the angle, so a frame
        in which the true speed falls below it would be placed at the floor
        speed with a wrong shaft angle. Such frames are therefore left out of the
        map (they have no valid order content); the frames kept lie entirely at
        or above ``min_rpm``, so ``rpm_min >= min_rpm`` in the result.
        Default: 2 % of the maximum speed, which only matters near standstill.
    overlap : float, default 0.75
        Frame overlap fraction (in angle).
    window : str, default "hann"
        Window name accepted by :func:`scipy.signal.get_window` (periodic).
    amplitude : {"rms", "peak"}, default "rms"
        Amplitude convention of the values.
    rpm_bins : float or array_like, optional
        Aggregate the frames per speed bin with :func:`bin_map`: a bin width
        in rpm or bin edges. The default None returns one column per frame.
    segment : {None, "runup", "rundown", "auto"}, optional
        Monotonic segment selection before binning (needs ``rpm_bins``); see
        :func:`bin_map`.
    min_count : int, default 1
        Minimum number of frames per bin (``rpm_bins`` only).
    bin_phase : {"drop", "circular"}, default "drop"
        Phase of the binned values, see :func:`bin_map`.

    Returns
    -------
    Map
        With ``domain="order"`` and ``axis`` in orders, truncated at
        ``max_order``. A :class:`BinnedMap` (a :class:`Map` with per-bin
        ``counts``) when ``rpm_bins`` is given.

    References
    ----------
    Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order
    tracking. Mechanical Systems and Signal Processing, 11(2), 187-205.
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    """
    check_amplitude(amplitude)
    _check_binning(rpm_bins, segment)
    if not order_resolution > 0:
        raise ValueError("order_resolution must be positive")
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
    # Frames reaching below the speed floor of the angle (min_rpm) have a wrong angle and a
    # floored speed in a.rpm: leave them out.
    rpm_true = broadcast_rpm(rpm, np.asarray(x).shape[0])
    floor = 0.02 * float(rpm_true.max()) if min_rpm is None else float(min_rpm)
    k_t = np.arange(rpm_true.size) / fs
    rpm_a = np.interp(a.t, k_t, np.interp(k_t + delay, k_t, rpm_true))  # true speed at the angle samples
    _, true_min, _ = _frame_stats(rpm_a, starts, nperseg)
    ok = true_min >= floor * (1.0 - 1e-9)
    if not ok.any():
        raise ValueError(f"no frame lies at or above min_rpm = {floor:g} rpm")
    if not ok.all():
        starts, values = starts[ok], values[:, ok]
    r_mean, r_min, r_max = _frame_stats(a.rpm, starts, nperseg)
    t_c = np.interp(starts + 0.5 * (nperseg - 1), np.arange(a.t.size), a.t)
    m = Map(
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
    if rpm_bins is None:
        return m
    return bin_map(m, rpm_bins, segment=segment, min_count=min_count, phase=bin_phase)
