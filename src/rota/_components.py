"""Result container shared by the order extractors."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OrderComponents:
    """Order components extracted from a signal.

    For order ``k`` with carrier angle ``theta_k(t) = 2 pi orders[k] rev(t)``
    (``rev`` the shaft angle in revolutions of the order's shaft, from
    :func:`rota.speed.rpm_to_phase`)::

        waveforms[:, k] = Re{ envelopes[:, k] * exp(1j * theta_k) }

    so ``|envelopes|`` is the *peak* amplitude in the signal unit and
    ``angle(envelopes)`` the phase relative to the carrier. This matches the
    truth of :func:`rota.synth.synth_runup`:
    ``order_amplitude * exp(1j * (order_phase - 2 pi order rev))``.

    Attributes
    ----------
    orders : ndarray, shape (k,)
        Order numbers.
    shafts : ndarray of int, shape (k,)
        Column of ``rpm`` each order refers to.
    waveforms : ndarray, shape (n, k) or (n, k, ch)
        Real order waveforms, signal unit.
    envelopes : ndarray, complex, shape (n, k) or (n, k, ch)
        Complex peak envelopes, signal unit.
    rev : ndarray, shape (n,) or (n, n_shafts)
        Shaft angle(s) in revolutions used for the carriers.
    method : str
        Name of the extractor.
    params : dict
        Parameters used (JSON-serialisable).
    """

    orders: np.ndarray
    shafts: np.ndarray
    waveforms: np.ndarray
    envelopes: np.ndarray
    rev: np.ndarray
    method: str
    params: dict = field(default_factory=dict)

    @property
    def total(self):
        """Sum of all order waveforms, shape (n,) or (n, ch)."""
        return self.waveforms.sum(axis=1)
