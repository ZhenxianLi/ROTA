"""Validation metrics.

Speed-track error for validating speed estimators against a known reference
(:func:`track_error`), waveform agreement of an extracted order or source with
its truth (:func:`waveform_similarity`) and the energy that remains at an
order after it was removed (:func:`order_residual_energy`).
"""

import numpy as np

from ._utils import as_2d
from .maps import rpm_order_map
from .slices import order_slice

__all__ = ["track_error", "waveform_similarity", "order_residual_energy"]


def track_error(estimate, truth, *, thresholds=(0.03, 0.05), at_bound=None):
    """Relative error of an estimated speed track against a reference.

    The error per sample is ``|estimate - truth| / |truth|``. Samples where
    either value is NaN or infinite, or where ``truth`` is zero (relative
    error undefined), are excluded from every statistic and counted in
    ``n_invalid``.

    Parameters
    ----------
    estimate, truth : array_like, same shape
        Estimated and true speed in any one consistent unit (rpm, Hz, km/h,
        ...). ``truth`` may be a scalar for a constant-speed reference. No
        amplitude convention is involved.
    thresholds : sequence of float, default (0.03, 0.05)
        Relative-error limits (fractions, 0.03 is 3 %) for the ``within``
        fractions. A sample is within a limit if its error is at most the
        limit.
    at_bound : array_like of bool, optional
        Same shape as ``estimate``; True where the estimator reported that it
        touched a search bound. Sets ``bound_fraction``.

    Returns
    -------
    dict
        All errors are relative (dimensionless); NaN if no sample is valid.

        ``median_rel_error``, ``mean_rel_error``, ``max_rel_error``
            Statistics of the relative error over valid samples.
        ``within``
            ``{threshold: fraction of valid samples with error <= threshold}``.
        ``n``
            Number of valid samples used (``n + n_invalid`` is the total).
        ``n_invalid``
            Number of samples excluded.
        ``bound_fraction``
            Fraction of valid samples flagged by ``at_bound`` (only if given).

    References
    ----------
    Peeters, C., et al. (2019). Review and comparison of tacholess
    instantaneous speed estimation methods on experimental vibration data.
    Mechanical Systems and Signal Processing, 129, 407-436.
    """
    est = np.asarray(estimate, dtype=float)
    ref = np.asarray(truth, dtype=float)
    try:
        est, ref = np.broadcast_arrays(est, ref)
    except ValueError:
        raise ValueError("estimate and truth must have the same shape") from None
    shape = est.shape
    est, ref = est.ravel(), ref.ravel()
    thr = [float(t) for t in np.atleast_1d(np.asarray(thresholds, dtype=float))]
    if any(not t > 0 for t in thr):
        raise ValueError("thresholds must be positive")

    valid = np.isfinite(est) & np.isfinite(ref) & (ref != 0)
    n = int(valid.sum())
    n_invalid = int(valid.size - n)
    rel = np.abs(est[valid] - ref[valid]) / np.abs(ref[valid])

    if n:
        stats = (float(np.median(rel)), float(np.mean(rel)), float(np.max(rel)))
        # Tiny relative slack so that e.g. 1.03 * truth counts as within 3 %.
        within = {t: float(np.mean(rel <= t * (1.0 + 1e-9))) for t in thr}
    else:
        stats = (np.nan, np.nan, np.nan)
        within = {t: np.nan for t in thr}
    out = {
        "median_rel_error": stats[0],
        "mean_rel_error": stats[1],
        "max_rel_error": stats[2],
        "within": within,
        "n": n,
        "n_invalid": n_invalid,
    }
    if at_bound is not None:
        try:
            flag = np.broadcast_to(np.asarray(at_bound, dtype=bool), shape).ravel()
        except ValueError:
            raise ValueError("at_bound must have the same shape as estimate") from None
        out["bound_fraction"] = float(np.mean(flag[valid])) if n else np.nan
    return out


# --------------------------------------------------------------------------
# waveform_similarity
# --------------------------------------------------------------------------

_WEIGHTINGS = ("energy", "uniform")


def _pair_2d(a, b, name_a, name_b):
    """Two arrays of the same shape as float ``(n, ch)`` arrays."""
    A, one_a = as_2d(a)
    B, one_b = as_2d(b)
    if A.shape != B.shape or one_a != one_b:
        raise ValueError(f"{name_a} and {name_b} must have the same shape")
    return A, B


def _pearson(a, b):
    """Pearson correlation per channel of two ``(n, ch)`` arrays (NaN if degenerate)."""
    am = a - a.mean(axis=0)
    bm = b - b.mean(axis=0)
    saa = np.sum(am * am, axis=0)
    sbb = np.sum(bm * bm, axis=0)
    # A channel is degenerate (constant) if its spread is below 1e-12 of its level.
    ok = (saa > 1e-24 * np.sum(a * a, axis=0)) & (sbb > 1e-24 * np.sum(b * b, axis=0))
    ok &= (saa > 0) & (sbb > 0)
    r = np.full(a.shape[1], np.nan)
    r[ok] = np.sum(am * bm, axis=0)[ok] / np.sqrt(saa[ok] * sbb[ok])
    return np.clip(r, -1.0, 1.0)


def _nrmse(est, ref):
    """rms(est - ref) / rms(ref) per channel (NaN where ``ref`` is all zero)."""
    num = np.sqrt(np.mean((est - ref) ** 2, axis=0))
    den = np.sqrt(np.mean(ref**2, axis=0))
    out = np.full(est.shape[1], np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def _weights(ref, weighting):
    """Channel weights (sum 1) from the energy of ``ref`` or uniform."""
    ch = ref.shape[1]
    if weighting == "uniform":
        return np.full(ch, 1.0 / ch)
    e = np.sum(ref**2, axis=0)
    total = e.sum()
    return e / total if total > 0 else np.full(ch, np.nan)


def _aggregate(values, w):
    """Weighted mean over the channels with positive weight; NaN if any is NaN."""
    use = w > 0
    if not use.any() or np.any(np.isnan(values[use])):
        return float("nan")
    return float(np.sum(w[use] * values[use]) / np.sum(w[use]))


def waveform_similarity(estimate, truth, *, mixture=None, other=None, trim=0, weighting="energy"):
    """Agreement of an estimated waveform with its truth: correlation and NRMSE.

    For each channel, the Pearson correlation coefficient of the demeaned
    signals, ``corr = sum((e - mean e) (t - mean t)) / sqrt(sum((e - mean e)^2)
    sum((t - mean t)^2))``, and the normalised RMS error
    ``nrmse = rms(e - t) / rms(t)`` (RMS of the truth, not demeaned). The
    correlation is blind to gain and offset, the NRMSE is not: an estimate
    ``a * truth`` has ``corr = 1`` and ``nrmse = |a - 1|``. Channels in which
    the signal is constant (correlation) or the truth is zero (NRMSE) give NaN.

    The aggregate over channels is a weighted mean of the per-channel values.
    With ``weighting="energy"`` (default) the weight of a channel is its share
    of the energy of the truth, ``sum(truth**2)``; with ``"uniform"`` every
    channel counts the same. When the energy sits in a few channels (a source
    heard mainly by one microphone) a uniform mean is dominated by the
    near-silent channels, where the estimate is mostly noise, and badly
    underestimates the agreement where the source actually is (ROADMAP
    section 6, item 8). Channels with zero weight are left out of the
    aggregate; if a channel with positive weight has a NaN value the aggregate
    is NaN.

    Optional references answer two further questions. ``mixture`` is the
    unprocessed signal the estimate was extracted from: its agreement with the
    truth is the zero baseline (what the estimate has to beat). ``other`` is
    the truth of a different source: the correlation of the estimate with it
    measures how much of that source was captured by mistake (cross capture).
    The aggregate weights always come from the signal the estimate is compared
    with: the energy of ``truth`` for the main metrics and the baseline, the
    energy of ``other`` for the cross capture.

    Parameters
    ----------
    estimate, truth : array_like, shape (n,) or (n, ch)
        Estimated and true waveform in the same unit (e.g. Pa) and with the
        same calibration and time alignment. The correlation is dimensionless
        and the NRMSE is a ratio of RMS values, so neither depends on the
        amplitude convention (peak or RMS) as long as both signals share it.
    mixture : array_like, same shape, optional
        Unprocessed signal, compared with ``truth`` in the same way.
    other : array_like, same shape, optional
        Truth of another source, correlated with ``estimate``.
    trim : int, default 0
        Number of samples removed at both ends of every input before
        computing (filter edge effects of extractors).
    weighting : {"energy", "uniform"}, default "energy"
        Channel weights of the aggregates.

    Returns
    -------
    dict
        ``corr``, ``nrmse``
            Aggregates (float, dimensionless).
        ``corr_per_channel``, ``nrmse_per_channel``
            Arrays of shape ``(ch,)`` (``ch = 1`` for 1-D input).
        ``weights``
            Channel weights used for ``truth``, shape ``(ch,)``.
        ``n``
            Number of samples per channel after trimming.
        ``baseline_corr``, ``baseline_nrmse``, ``baseline_corr_per_channel``,
        ``baseline_nrmse_per_channel``
            Same metrics for ``mixture`` against ``truth`` (only if ``mixture``
            is given).
        ``cross_corr``, ``cross_corr_per_channel``
            Correlation of ``estimate`` with ``other`` (only if ``other`` is
            given).

    Raises
    ------
    ValueError
        For inputs of different shapes, non-finite values, invalid ``trim`` or
        ``weighting``.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 3 (correlation).
    Wiley.
    Pearson, K. (1895). Notes on regression and inheritance in the case of two
    parents. Proceedings of the Royal Society of London, 58, 240-242.
    """
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"weighting must be one of {_WEIGHTINGS}, got {weighting!r}")
    if int(trim) != trim or trim < 0:
        raise ValueError("trim must be a non-negative integer number of samples")
    trim = int(trim)
    E, T = _pair_2d(estimate, truth, "estimate", "truth")
    inputs = {"estimate": E, "truth": T}
    for name, arr in (("mixture", mixture), ("other", other)):
        if arr is not None:
            inputs[name] = _pair_2d(arr, truth, name, "truth")[0]
    n = E.shape[0]
    if n - 2 * trim < 2:
        raise ValueError("trim leaves fewer than 2 samples")
    for name, arr in inputs.items():
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"{name} contains non-finite values; remove them or use trim for edge samples"
            )
    sl = slice(trim, n - trim)
    E, T = E[sl], T[sl]

    w = _weights(T, weighting)
    corr_c = _pearson(E, T)
    nrmse_c = _nrmse(E, T)
    out = {
        "corr": _aggregate(corr_c, w),
        "nrmse": _aggregate(nrmse_c, w),
        "corr_per_channel": corr_c,
        "nrmse_per_channel": nrmse_c,
        "weights": w,
        "n": int(E.shape[0]),
    }
    if mixture is not None:
        M = inputs["mixture"][sl]
        bc, bn = _pearson(M, T), _nrmse(M, T)
        out.update(
            baseline_corr=_aggregate(bc, w),
            baseline_nrmse=_aggregate(bn, w),
            baseline_corr_per_channel=bc,
            baseline_nrmse_per_channel=bn,
        )
    if other is not None:
        O = inputs["other"][sl]
        xc = _pearson(E, O)
        out.update(cross_corr=_aggregate(xc, _weights(O, weighting)), cross_corr_per_channel=xc)
    return out


# --------------------------------------------------------------------------
# order_residual_energy
# --------------------------------------------------------------------------


def order_residual_energy(residual, reference, fs, rpm, order, *, order_resolution=0.125, rpm_range=None):
    """Energy left at an order after its removal, in dB relative to the reference.

    Both signals are analysed in the angle domain with
    :func:`rota.maps.rpm_order_map` (frames of ``1 / order_resolution``
    revolutions, Hann window, overlap 0.75) and the order is read with
    :func:`rota.slices.order_slice` in ``method="band"``, which sums the power
    over a band of four bins (``4 * order_resolution`` orders) around ``order``
    and is therefore independent of where the line falls between bins. With
    ``P_res`` and ``P_ref`` the band power summed over the selected frames (and
    over the channels), the result is ``10 log10(P_res / P_ref)``: the ratio
    of the energy of the order in the residual to its energy in the reference,
    with every frame weighted by its energy (a mean of energies, not of
    levels). 0 dB means nothing was removed, -20 dB (the tolerance of ROADMAP
    case T12) that 99 % of the energy of the order is gone. Other orders
    outside the band do not enter, except through the window's side lobes
    (below -31 dB for lines more than two bins away) and broadband noise inside
    the band, which sets a floor.

    The reference is typically the signal the order was removed from, or the
    true order stem; with the original signal as reference, other orders and
    noise in the band appear in the denominator too.

    Parameters
    ----------
    residual, reference : array_like, shape (n,) or (n, ch)
        Signal after removal of the order and the signal against which the
        remaining energy is measured, same shape, same unit (e.g. Pa), sampled
        at ``fs``. The result is a ratio of energies, independent of the unit
        and of the amplitude convention (peak or RMS).
    fs : float
        Sampling rate in Hz.
    rpm : float or array_like, shape (n,)
        Speed in rpm of the shaft the order refers to, at the signal samples.
    order : float
        Order to measure, in cycles per revolution (> 0).
    order_resolution : float, default 0.125
        Bin spacing of the order map in orders; the analysis band is four bins
        wide. Neighbouring orders closer than about ``3 * order_resolution``
        leak into the band.
    rpm_range : (float or None, float or None), optional
        Keep only frames whose mean speed lies in ``lo <= rpm <= hi`` (rpm);
        None leaves that side open.

    Returns
    -------
    float
        Residual energy of the order in dB relative to the reference
        (``-inf`` if the residual has no energy at the order).

    Raises
    ------
    ValueError
        If the inputs differ in shape, the order (with its band) lies above
        the highest order the sampling rate and speed allow, no frame is
        selected, or the reference has no energy at the order.

    References
    ----------
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 9 and 12. Wiley.
    Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order
    tracking. Mechanical Systems and Signal Processing, 11(2), 187-205.
    Harris, F. J. (1978). On the use of windows for harmonic analysis with the
    discrete Fourier transform. Proceedings of the IEEE, 66(1), 51-83.
    """
    R, F = _pair_2d(residual, reference, "residual", "reference")
    order = float(order)
    if not (np.isfinite(order) and order > 0):
        raise ValueError("order must be positive and finite")
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(F)):
        raise ValueError("residual and reference must be finite")
    if rpm_range is not None:
        try:
            lo, hi = rpm_range
        except (TypeError, ValueError):
            raise ValueError("rpm_range must be a pair (lo, hi)") from None
        lo = -np.inf if lo is None else float(lo)
        hi = np.inf if hi is None else float(hi)
        if lo > hi:
            raise ValueError("rpm_range must have lo <= hi")

    kw = {"order_resolution": order_resolution, "amplitude": "rms"}
    m_res = rpm_order_map(R, fs, rpm, **kw)
    m_ref = rpm_order_map(F, fs, rpm, **kw)
    s_res = order_slice(m_res, order, method="band")
    s_ref = order_slice(m_ref, order, method="band")
    p_res = np.sum(as_2d(s_res.amplitude)[0] ** 2, axis=1)  # band power per frame, all channels
    p_ref = np.sum(as_2d(s_ref.amplitude)[0] ** 2, axis=1)
    usable = np.isfinite(p_res) & np.isfinite(p_ref)
    if not usable.any():
        raise ValueError(
            "order (with its analysis band) lies outside the order axis: above the "
            "highest order that fs and the speed allow, or within two bins of zero"
        )
    sel = usable
    if rpm_range is not None:
        rpm_f = s_ref.rpm
        sel = usable & (rpm_f >= lo) & (rpm_f <= hi)
        if not sel.any():
            raise ValueError("no frame selected; check rpm_range")
    e_ref = float(np.sum(p_ref[sel]))
    e_res = float(np.sum(p_res[sel]))
    if not e_ref > 0:
        raise ValueError("the reference has no energy at this order")
    if e_res == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(e_res / e_ref))
