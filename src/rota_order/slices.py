"""Order slices and averaged spectra from RPM maps.

Both functions act on a :class:`rota_order.maps.Map` and keep its amplitude
convention (peak or RMS). Units: orders are cycles per revolution, frequency
in Hz, speed in rpm, time in s; amplitudes are in the unit of the analysed
signal.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["order_slice", "order_spectrum", "OrderSlice", "Spectrum"]

_METHODS = ("band", "peak")


@dataclass
class OrderSlice:
    """Level of one order versus speed, one value per map frame.

    Attributes
    ----------
    order : float
        Order of the slice.
    t : ndarray, shape (n_frames,)
        Frame centre times in s.
    rpm : ndarray, shape (n_frames,)
        Mean speed over each frame in rpm.
    amplitude : ndarray, shape (n_frames,) or (n_frames, ch)
        Order amplitude in the signal unit, in the convention given by
        ``amplitude_convention``. NaN where the band leaves the map's axis.
    complex : ndarray, complex, shape (n_frames,) or (n_frames, ch)
        Map value at the bin nearest the order (same convention; phase
        referenced to the start of the signal). NaN where ``amplitude`` is.
        The phase is that of the order only when the order falls on a bin
        (see :func:`order_slice`).
    amplitude_convention : str
        ``"peak"`` or ``"rms"``, copied from the map.
    bandwidth : float
        Total band width used, in the map's axis unit (orders or Hz),
        excluding the smear allowance of frequency maps.
    domain : str
        ``"order"`` or ``"frequency"``, domain of the map the slice came from.
    """

    order: float
    t: np.ndarray
    rpm: np.ndarray
    amplitude: np.ndarray
    complex: np.ndarray
    amplitude_convention: str
    bandwidth: float
    domain: str


@dataclass
class Spectrum:
    """Energy-averaged spectrum of a map.

    Attributes
    ----------
    axis : ndarray, shape (n_bins,)
        Order (``domain="order"``) or frequency in Hz (``domain="frequency"``).
    amplitude : ndarray, shape (n_bins,) or (n_bins, ch)
        ``sqrt(mean(|values|**2))`` over the selected frames, in the signal
        unit and the convention given by ``amplitude_convention``.
    domain : str
        ``"order"`` or ``"frequency"``.
    amplitude_convention : str
        ``"peak"`` or ``"rms"``, copied from the map.
    n_frames : int
        Number of frames averaged.
    enbw_bins : float
        Equivalent noise bandwidth in bins of the map's window (NaN if
        unknown). The amplitude of a line spread over bins ``k`` is
        ``sqrt(sum(amplitude[k]**2) / enbw_bins)``.
    """

    axis: np.ndarray
    amplitude: np.ndarray
    domain: str
    amplitude_convention: str
    n_frames: int
    enbw_bins: float = float("nan")


def _check_map_domain(m):
    if m.domain not in ("order", "frequency"):
        raise ValueError(f"unknown map domain {m.domain!r}")


def order_slice(m, order, *, bandwidth=None, method="band"):
    """Level of one order versus speed, read from an RPM map.

    For each map frame the order is located at ``order`` (order map) or at
    ``order * rpm / 60`` Hz (frequency map, ``rpm`` the frame's mean speed).
    The band around that centre has half-width ``bandwidth / 2``; in a
    frequency map it is widened by the distance the line moves during the
    frame, ``order * (rpm_max - rpm_min) / 60`` Hz in total, so that a line
    sweeping through the frame in a run-up stays inside.

    ``method="band"`` sums power over the bins of the band,
    ``amplitude = sqrt(sum(|values|**2) / enbw_bins)``. Because the window's
    power is spread over several bins, this reading does not depend on where
    the line falls between bins (no scalloping loss) and is also right for a
    line smeared by speed change. Broadband content inside the band adds to
    the result. ``method="peak"`` returns the largest ``|values|`` in the
    band, which shows the scalloping loss of the window (up to 1.42 dB for
    Hann) when the line is between bins.

    Frames whose band extends beyond the lowest or highest bin of the axis
    (above Nyquist or the map's ``max_freq`` / ``max_order``, or within the
    band half-width of DC, where the line is not resolved from its negative
    frequency image) give NaN.

    Parameters
    ----------
    m : rota_order.maps.Map
        Order or frequency map; ``values`` of shape ``(n_bins, n_frames)`` or
        ``(n_bins, n_frames, ch)``.
    order : float
        Order to extract, in cycles per revolution (> 0).
    bandwidth : float, optional
        Total band width in the map's axis unit (orders for an order map,
        Hz for a frequency map). Default: ``4 * m.resolution``, i.e. the bins
        within two bin spacings of the centre (five bins when the order falls
        on a bin, four when it falls midway). Neighbouring orders closer than
        about three bin spacings leak into the band; use a finer map.
    method : {"band", "peak"}, default "band"
        Power summation over the band, or maximum in the band.

    Returns
    -------
    OrderSlice
        ``amplitude`` is in the amplitude convention of the map (peak or
        RMS) and in the unit of the analysed signal, one value per frame (and
        channel). ``complex`` holds the map value at the bin nearest the
        order; its phase is referenced to the start of the signal and equals
        the phase of the order only if the order falls on a bin of an order
        map. For other cases (off-bin orders, frequency maps) it varies from
        frame to frame with the offset between line and bin; use it for
        magnitude-and-phase work only at bin centres.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 9 and 12. Wiley.
    Harris, F. J. (1978). On the use of windows for harmonic analysis with
    the discrete Fourier transform. Proceedings of the IEEE, 66(1), 51-83.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    _check_map_domain(m)
    order = float(order)
    if not order > 0:
        raise ValueError("order must be positive")
    res = m.resolution
    bandwidth = 4.0 * res if bandwidth is None else float(bandwidth)
    if not bandwidth > 0:
        raise ValueError("bandwidth must be positive")

    V = np.asarray(m.values)
    one_channel = V.ndim == 2
    if V.ndim not in (2, 3):
        raise ValueError("map values must have shape (n_bins, n_frames[, ch])")
    if one_channel:
        V = V[..., None]  # (bins, frames, ch)
    axis = np.asarray(m.axis, dtype=float)
    n_frames = V.shape[1]
    rpm = np.asarray(m.rpm, dtype=float)

    if m.domain == "order":
        centre = np.full(n_frames, order)
        half = np.full(n_frames, 0.5 * bandwidth)
    else:
        centre = order * rpm / 60.0
        smear = order * (np.asarray(m.rpm_max, float) - np.asarray(m.rpm_min, float)) / 60.0
        half = 0.5 * bandwidth + 0.5 * smear

    tol = 1e-6 * res
    dist = np.abs(axis[:, None] - centre[None, :])  # (bins, frames)
    nearest = np.argmin(dist, axis=0)
    frames = np.arange(n_frames)
    mask = dist <= half[None, :] + tol
    mask[nearest, frames] = True  # never empty, whatever the bandwidth
    outside = (centre + half > axis[-1] + tol) | (centre - half < axis[0] - tol)

    mag = np.abs(V)
    if method == "band":
        band = np.einsum("bf,bfc->fc", mask.astype(float), mag**2)
        amp = np.sqrt(band / m.enbw_bins)
    else:
        amp = np.max(mag, axis=0, where=mask[..., None], initial=0.0)
    cval = V[nearest, frames].astype(complex)  # (frames, ch)
    amp[outside] = np.nan
    cval[outside] = complex(np.nan, np.nan)
    if one_channel:
        amp, cval = amp[:, 0], cval[:, 0]
    return OrderSlice(
        order=order,
        t=np.array(m.t, dtype=float),
        rpm=np.array(m.rpm, dtype=float),
        amplitude=amp,
        complex=cval,
        amplitude_convention=m.amplitude,
        bandwidth=bandwidth,
        domain=m.domain,
    )


def order_spectrum(m, *, rpm_range=None, t_range=None):
    """Energy-averaged spectrum of a map over selected frames.

    ``amplitude = sqrt(mean over frames of |values|**2)`` per bin (and
    channel): the squared magnitudes are averaged, as in Welch's method, so
    the result is an RMS over frames of the map's amplitude and equals the
    constant level of a line whose amplitude does not change. For an order
    map this is the average order spectrum; lines that drift in order (or in
    frequency, for a frequency map) are smeared by the averaging.

    Parameters
    ----------
    m : rota_order.maps.Map
        Order or frequency map.
    rpm_range : (float, float), optional
        Keep frames with ``lo <= m.rpm <= hi`` (rpm); ``None`` for either
        limit leaves that side open.
    t_range : (float, float), optional
        Keep frames with ``lo <= m.t <= hi`` (s); ``None`` as above. If both
        ranges are given, frames must satisfy both.

    Returns
    -------
    Spectrum
        ``amplitude`` is in the amplitude convention of the map (peak or
        RMS) and the unit of the analysed signal, shape ``(n_bins,)`` or
        ``(n_bins, ch)``.

    Raises
    ------
    ValueError
        If no frame is selected.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 9 and 12. Wiley.
    Welch, P. D. (1967). The use of fast Fourier transform for the estimation
    of power spectra: a method based on time averaging over short, modified
    periodograms. IEEE Transactions on Audio and Electroacoustics, 15(2),
    70-73.
    Harris, F. J. (1978). On the use of windows for harmonic analysis with
    the discrete Fourier transform. Proceedings of the IEEE, 66(1), 51-83.
    """
    _check_map_domain(m)
    V = np.asarray(m.values)
    n_frames = V.shape[1]
    sel = np.ones(n_frames, dtype=bool)
    for name, rng, v in (("rpm_range", rpm_range, m.rpm), ("t_range", t_range, m.t)):
        if rng is None:
            continue
        lo, hi = rng
        lo = -np.inf if lo is None else float(lo)
        hi = np.inf if hi is None else float(hi)
        v = np.asarray(v, dtype=float)
        sel &= (v >= lo) & (v <= hi)  # NaN never selected
    n = int(sel.sum())
    if n == 0:
        raise ValueError("no frames selected; check rpm_range and t_range")
    Vs = V if n == n_frames else V[:, sel]
    power = np.mean(Vs.real**2 + Vs.imag**2, axis=1)
    return Spectrum(
        axis=np.array(m.axis, dtype=float),
        amplitude=np.sqrt(power),
        domain=m.domain,
        amplitude_convention=m.amplitude,
        n_frames=n,
        enbw_bins=float(m.enbw_bins),
    )


# Application layer: fixed-frequency slices, overall level, metrics along rpm,
# order attribution of frequency tracks.
from ._application import (  # noqa: E402
    FrequencyOrder,
    FrequencySlice,
    MetricVsRpm,
    OrderShare,
    OverallLevel,
    frequency_slice,
    frequency_to_order,
    metric_vs_rpm,
    order_share,
    overall_level,
)

__all__ += [
    "frequency_slice",
    "overall_level",
    "metric_vs_rpm",
    "frequency_to_order",
    "order_share",
    "FrequencySlice",
    "OverallLevel",
    "MetricVsRpm",
    "FrequencyOrder",
    "OrderShare",
]
