"""Optional thin bridges to open-source psychoacoustic toolboxes (since 0.6.0).

rota-order computes no psychoacoustic metric. These wrappers call the documented public
functions of the toolboxes on rota-order's conventions (input: instantaneous sound pressure in
Pa, mono; calibration factor; resampling to 48 kHz with :func:`rota_order.tonal.to_calibrated_48k`)
and return a :class:`TimeMetric`, whose method :meth:`TimeMetric.vs_rpm` bins it along rpm
with :func:`rota_order.slices.metric_vs_rpm`::

    from rota_order import bridges

    loud = bridges.mosqito.loudness(x, fs, calibration=cal)   # ISO 532-1:2017, sone
    n5 = loud.vs_rpm(rpm, t_rpm, bin_width=250.0)             # N5 per rpm bin

The quantity and the toolbox are named in the same line, and the binned result
(:class:`rota_order.slices.MetricVsRpm`) carries the ``name``, ``unit`` and ``source`` of the
metric. A bridge module is named exactly like the toolbox it wraps. A short alias is up to
you, for example ``from rota_order.bridges import sottek_hearing_model as shm``.

Submodules
----------
mosqito               MoSQITo (Green Forge Coop, Apache-2.0, ``pip install mosqito`` and
                      matplotlib): ISO 532-1:2017 time-varying loudness, DIN 45692:2009
                      sharpness, Daniel and Weber (1997) roughness.
sottek_hearing_model  sottek-hearing-model (Lotinga, AGPL-3.0-only, Python >= 3.11,
                      ``pip install sottek-hearing-model``): ECMA-418-2:2025 (4th edition)
                      tonality, loudness and roughness.
sqat                  SQAT (Felix Greco et al., GPL-3.0; some files BSD-3-Clause or MIT),
                      MATLAB, via the MATLAB Engine API for Python
                      (``pip install matlabengine==<matching MATLAB>``, licensed local
                      MATLAB + Signal Processing Toolbox, SQAT v2.0 checkout in
                      ROTA_ORDER_SQAT_PATH): ISO 532-1 loudness, DIN 45692 sharpness, Daniel and
                      Weber roughness, ECMA-418-2:2025 tonality.

The toolboxes are optional and are never vendored: ``pip install rota-order`` installs
only NumPy and SciPy, ``import rota_order.bridges`` works without them, and each toolbox is
imported inside the function that needs it. Without it the function raises ImportError
with the install command; rota-order itself does not depend on the toolbox. Mind the licence of
a toolbox before you install it (the Sottek hearing model is AGPL-3.0-only). rota-order does
not include, copy, modify or redistribute any toolbox code; each keeps its own licence,
and its authors ask to be cited as given in References and in each submodule.

Not wrapped: the ECMA-418-2 functions of MoSQITo (2nd edition only) and its
``tnr_ecma_st`` (rota-order has its own ECMA-418-1:2024 tone-to-noise ratio in
:mod:`rota_order.tonal`).

Metrics are perceptual quantities in the unit of the standard (sone, acum, asper, tu_HMS,
sone_HMS, asper_HMS), not amplitudes; no peak or RMS convention applies to them.

References
----------
ISO 532-1:2017; DIN 45692:2009; Daniel, P., & Weber, R. (1997). Acustica 83, 113-123;
ECMA-418-2:2025 (4th ed.).
Green Forge Coop (2024). MOSQITO (version 1.2.1). Zenodo.
https://doi.org/10.5281/zenodo.11026796 (Apache-2.0).
Lotinga, M. J. B., Torjussen, M., & Felix Greco, G. (2025). Verified implementations of the
Sottek psychoacoustic Hearing Model standardised sound quality metrics (ECMA-418-2
loudness, tonality and roughness). Forum Acusticum / Euronoise 2025, Malaga.
https://doi.org/10.61782/fa.2025.0951; sottek-hearing-model,
https://github.com/mlotinga/sottek-hearing-model (AGPL-3.0-only).
Felix Greco, G., Merino-Martínez, R., Osses, A., & Lotinga, M. J. B. (2026). SQAT: a sound
quality analysis toolbox for MATLAB (release v2.0). Zenodo.
https://doi.org/10.5281/zenodo.7934709 (GPL-3.0).
Felix Greco, G., Merino-Martínez, R., Osses, A., & Langer, S. C. (2023). SQAT: a
MATLAB-based toolbox for quantitative sound quality analysis. INTER-NOISE 2023, Chiba,
Japan. https://doi.org/10.3397/IN_2023_1075
Catalogue of psychoacoustic metrics and tools (LISQM): https://zhenxianli.github.io/LISQM/
"""

from . import mosqito, sottek_hearing_model, sqat
from ._common import TimeMetric

__all__ = ["TimeMetric", "mosqito", "sottek_hearing_model", "sqat"]
