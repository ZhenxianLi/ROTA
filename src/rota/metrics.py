"""Validation metrics.

Speed-track error for validating speed estimators against a known reference.
"""

import numpy as np

__all__ = ["track_error"]


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
