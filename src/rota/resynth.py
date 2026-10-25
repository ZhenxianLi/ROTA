"""Order-based resynthesis: from analysis results back to sound.

Order tables (orders x rpm -> complex amplitude, explicit peak or RMS) are
built from order slices or extracted components and rendered by additive
synthesis along any speed curve, offline (:func:`synthesize_orders`) or in
blocks (:class:`OrderSynth`). Broadband content is rendered from a spectral
envelope (:func:`synthesize_broadband`). Every rendering can be documented
with a JSON sidecar (:func:`provenance`). ROTA does no audio I/O.

Units: time in s, speed in rpm, frequency in Hz; amplitudes in the unit of
the table (e.g. Pa). Rendered waveforms are real signals whose order
components have the peak amplitude given by the table (converted from RMS
where needed).
"""

from ._broadband import synthesize_broadband
from ._ordertable import OrderSynth, OrderTable, synthesize_orders
from ._provenance import provenance, stable_seed

__all__ = [
    "OrderTable",
    "synthesize_orders",
    "OrderSynth",
    "synthesize_broadband",
    "provenance",
    "stable_seed",
]
