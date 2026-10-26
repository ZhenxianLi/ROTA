"""Order-based resynthesis: from analysis results back to sound.

Order tables (orders x rpm -> complex amplitude, explicit peak or RMS) are
built from order slices or extracted components and rendered by additive
synthesis along any speed curve, offline (:func:`synthesize_orders`) or in
blocks (:class:`OrderSynth`). Broadband content is rendered from a spectral
envelope (:func:`synthesize_broadband`). Orders can be edited
(:func:`modify_orders`: gain, mute, solo, relabel, frequency offset), rendered
along a new speed curve (:func:`remap_rpm`) and recombined with the residual
under explicit gains (:func:`resynthesize`); :func:`stationary_excerpt` makes a
steady listening excerpt at a reference speed from a run-up. Every rendering
can be documented with a JSON sidecar (:func:`provenance`). ROTA does no audio
I/O.

Units: time in s, speed in rpm, frequency in Hz; amplitudes in the unit of
the table (e.g. Pa). Rendered waveforms are real signals whose order
components have the peak amplitude given by the table (converted from RMS
where needed).
"""

from ._broadband import synthesize_broadband
from ._excerpt import ExcerptResult, stationary_excerpt
from ._modify import modify_orders, remap_rpm, resynthesize
from ._ordertable import OrderSynth, OrderTable, synthesize_orders
from ._provenance import provenance, stable_seed

__all__ = [
    "OrderTable",
    "synthesize_orders",
    "OrderSynth",
    "synthesize_broadband",
    "modify_orders",
    "remap_rpm",
    "resynthesize",
    "stationary_excerpt",
    "ExcerptResult",
    "provenance",
    "stable_seed",
]
