"""Order extraction, order removal and resolvability diagnostics.

Two extractors return the same :class:`OrderComponents` (peak envelopes,
``waveform = Re{envelope * exp(i 2 pi order rev)}``):

* :func:`order_waveform`: second-generation Vold-Kalman filter, joint over
  orders and shafts; best for crossing orders and close twin shafts.
* :func:`tracked_harmonics`: phase-locked short-time averaging over a window
  of fixed length in revolutions; much cheaper, for long multichannel records.

Check with :func:`resolution_check` whether the requested orders can be
separated with the chosen window or bandwidth before trusting the result.
Units: time in s, speed in rpm, frequency and bandwidth in Hz.
"""

import numpy as np

from ._components import OrderComponents
from ._resolution import ResolutionReport, ResolutionWarning, resolution_check
from ._tracked import tracked_harmonics
from ._vk import order_waveform

__all__ = [
    "order_waveform",
    "tracked_harmonics",
    "remove_orders",
    "residual",
    "resolution_check",
    "OrderComponents",
    "ResolutionReport",
    "ResolutionWarning",
]

_METHODS = {"vold-kalman": order_waveform, "tracked-harmonics": tracked_harmonics}


def _extract(x, fs, rpm, orders, method, shafts, kwargs):
    if method not in _METHODS:
        raise ValueError(f"method must be one of {tuple(_METHODS)}, got {method!r}")
    return _METHODS[method](x, fs, rpm, orders, shafts=shafts, **kwargs)


def remove_orders(x, fs, rpm, orders, *, method="vold-kalman", shafts=None, **kwargs):
    """Subtract selected orders from a signal.

    The orders are extracted with :func:`order_waveform` or
    :func:`tracked_harmonics` and their waveforms are subtracted sample by
    sample, so the result keeps everything else (other orders, broadband
    content) with its original amplitude and phase. Per channel and linear.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal, any unit (e.g. Pa).
    fs : float
        Sampling rate in Hz.
    rpm : array_like, shape (n,) or (n, n_shafts)
        Shaft speed(s) in rpm.
    orders : sequence of float
        Orders to remove.
    method : {"vold-kalman", "tracked-harmonics"}
        Extractor.
    shafts : sequence of int, optional
        Column of ``rpm`` for each order (default 0).
    **kwargs
        Passed to the extractor (e.g. ``bandwidth`` or ``window_revs``).

    Returns
    -------
    y : ndarray
        ``x`` minus the extracted orders, same shape and unit as ``x``.
    components : OrderComponents
        The removed components (peak envelopes).

    References
    ----------
    Vold, H., & Leuridan, J. (1993). High resolution order tracking at extreme
    slew rates, using Kalman tracking filters. SAE Technical Paper 931288.
    Antoni, J. (2009). Cyclostationarity by examples. Mechanical Systems and
    Signal Processing, 23(4), 987-1036.
    """
    x = np.asarray(x, dtype=float)
    comp = _extract(x, fs, rpm, orders, method, shafts, kwargs)
    return x - comp.total, comp


def residual(x, fs, rpm, orders, *, method="vold-kalman", shafts=None, **kwargs):
    """Broadband residual: the signal minus all orders of all shafts.

    Same computation as :func:`remove_orders`, intended for the complete set
    of orders (every shaft) so that the result is the order-free background,
    e.g. for order-plus-residual resynthesis. Returns ``(residual,
    components)``; ``residual`` is exactly ``x - components.total``, so the
    two add back to ``x`` up to floating-point rounding.
    Units and amplitude conventions as in :func:`remove_orders`.

    References
    ----------
    Serra, X., & Smith, J. O. (1990). Spectral modeling synthesis: a sound
    analysis/synthesis system based on a deterministic plus stochastic
    decomposition. Computer Music Journal, 14(4), 12-24.
    """
    return remove_orders(x, fs, rpm, orders, method=method, shafts=shafts, **kwargs)
