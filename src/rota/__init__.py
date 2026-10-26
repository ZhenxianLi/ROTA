"""ROTA: rotational order tracking and analysis.

Submodules
----------
speed    Speed references: tachometer pulses, tacholess estimation, phase,
         control signals.
angle    Angle-domain (computed order tracking) resampling.
maps     RPM-frequency and RPM-order maps.
slices   Order spectra and order slices.
extract  Order extraction (Vold-Kalman, tracked harmonics), removal, residual.
resynth  Order tables, order and broadband resynthesis, provenance.
synth    Synthetic test signals with analytic ground truth.
metrics  Validation metrics.

Conventions
-----------
Time in seconds, sampling rates in Hz, rotational speed in rev/min (rpm),
shaft angle in revolutions, vehicle speed in km/h. Signals are arrays of
shape ``(n,)`` or ``(n, ch)``; every operator acts on each channel
independently and linearly. Amplitude conventions (peak or RMS) are stated
in each docstring and selected explicitly where a choice exists.
"""

__version__ = "0.3.0"

from . import angle, extract, maps, metrics, resynth, slices, speed, synth  # noqa: F401

__all__ = [
    "angle", "extract", "maps", "metrics", "resynth", "slices", "speed", "synth",
    "__version__",
]
