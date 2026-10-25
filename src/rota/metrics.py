"""Validation metrics.

Speed-track error for validating speed estimators against a known reference
(:func:`track_error`), waveform agreement of an extracted order or source with
its truth (:func:`waveform_similarity`) and the energy that remains at an
order after it was removed (:func:`order_residual_energy`). For resynthesised
noise: the overlap-add conditions of a window (:func:`window_checks`), the
pumping detector (:func:`modulation_peak`) and envelope statistics
(:func:`envelope_stats`).
"""

from fractions import Fraction

import numpy as np
from scipy import signal

from ._utils import as_2d
from .maps import rpm_order_map
from .slices import order_slice

__all__ = [
    "track_error",
    "waveform_similarity",
    "order_residual_energy",
    "window_checks",
    "modulation_peak",
    "envelope_stats",
]


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


# --------------------------------------------------------------------------
# Checks for resynthesised noise: window_checks, modulation_peak, envelope_stats
# --------------------------------------------------------------------------

_RIPPLE_TOL_DB = 0.01  # ripple below which a window counts as COLA / power complementary


def _mono(x, name):
    """``x`` as a finite 1-D float array."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D (mono); analyse channels one at a time")
    if x.size < 2:
        raise ValueError(f"{name} must have at least 2 samples")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values")
    return x


def _positive(value, name):
    value = float(value)
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive and finite")
    return value


def _check_band(band, fs):
    """Validated ``(lo, hi)`` in Hz with ``0 < lo < hi < fs/2``."""
    try:
        lo, hi = (float(b) for b in band)
    except (TypeError, ValueError):
        raise ValueError("band must be a pair (lo, hi) in Hz") from None
    if not (0.0 < lo < hi < fs / 2.0):
        raise ValueError("band must satisfy 0 < lo < hi < fs/2")
    return lo, hi


def _bandpass(x, fs, band):
    """Zero-phase 4th-order Butterworth band-pass (applied forward and backward)."""
    sos = signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def window_checks(window, hop, *, n_frames=64):
    """Overlap-add conditions of a window: amplitude COLA and power complementarity.

    Frames of the window ``w`` (length ``L``) placed every ``hop = R`` samples
    add up to ``s1(n) = sum_k w(n - k R)``. A deterministic signal that is
    analysed and resynthesised with the window is reconstructed without
    ripple if ``s1`` is constant: the constant-overlap-add (COLA) condition
    (Allen & Rabiner 1977). Frames that carry *independent* random phases do
    not add in amplitude but in power, and the output power is constant if
    ``s2(n) = sum_k w(n - k R)**2`` is constant: the window is *power
    complementary* (the case of noise grains, ROADMAP section 6, item 11).
    The two conditions are different: the periodic Hann window is COLA at 50 %
    overlap but not power complementary, its square root is the reverse. See
    also Griffin & Lim (1984).

    Both sums are evaluated in the steady region, where the full set of
    overlapping frames is present (``L - R <= n < n_frames R``), where they
    are periodic in ``R``. The ripple is the ratio of the maximum to the
    minimum of the sum, expressed as the level change it would cause in the
    signal: ``20 log10`` for the amplitude sum ``s1``, ``10 log10`` for the
    power sum ``s2``. A window passes a condition if the ripple is below
    0.01 dB.

    Parameters
    ----------
    window : array_like, shape (L,)
        Window samples (for the periodic Hann window use
        ``scipy.signal.get_window("hann", L)``, not the symmetric
        ``numpy.hanning``). The window is dimensionless; the checks do not
        depend on its scale or on any amplitude convention.
    hop : int
        Hop between frames in samples (>= 1).
    n_frames : int, default 64
        Number of frames summed. Raised automatically to
        ``ceil(L / hop) + 2`` if the window needs more to reach a steady
        region.

    Returns
    -------
    dict
        ``cola_ripple_db``
            ``20 log10(max s1 / min s1)`` in the steady region (``inf`` if
            ``s1`` reaches 0 or below).
        ``power_ripple_db``
            ``10 log10(max s2 / min s2)`` in the steady region.
        ``cola``, ``power_complementary``
            bool, True if the respective ripple is below 0.01 dB.
        ``cola_constant``, ``power_constant``
            Mean of ``s1`` and ``s2`` in the steady region (``sum(w) / R``
            and ``sum(w**2) / R`` when the conditions hold; 1 for the
            normalised windows used for resynthesis).

    Raises
    ------
    ValueError
        For a window that is not 1-D with at least two finite samples, is all
        zero, a non-integer or non-positive ``hop``, or ``n_frames < 2``.

    References
    ----------
    Allen, J. B., & Rabiner, L. R. (1977). A unified approach to short-time
    Fourier analysis and synthesis. Proceedings of the IEEE, 65(11),
    1558-1564.
    Griffin, D. W., & Lim, J. S. (1984). Signal estimation from modified
    short-time Fourier transform. IEEE Transactions on Acoustics, Speech, and
    Signal Processing, 32(2), 236-243.
    """
    w = _mono(window, "window")
    if not np.any(w):
        raise ValueError("window is all zero")
    if int(hop) != hop or hop < 1:
        raise ValueError("hop must be a positive integer number of samples")
    hop = int(hop)
    if int(n_frames) != n_frames or n_frames < 2:
        raise ValueError("n_frames must be an integer >= 2")
    length = w.size
    n_frames = max(int(n_frames), -(-length // hop) + 2)

    total = (n_frames - 1) * hop + length
    s1 = np.zeros(total)
    s2 = np.zeros(total)
    w2 = w * w
    for k in range(n_frames):
        s1[k * hop : k * hop + length] += w
        s2[k * hop : k * hop + length] += w2
    # Steady region, cut to a whole number of hops so that the means are exact period averages.
    lo = max(length - hop, 0)
    hi = min(n_frames * hop, total)
    hi = lo + max((hi - lo) // hop, 1) * hop
    s1, s2 = s1[lo:hi], s2[lo:hi]

    def ripple(s, factor):
        top, bottom = float(s.max()), float(s.min())
        if bottom <= 0.0:
            return float("inf")
        return float(factor * np.log10(top / bottom))

    cola_db = ripple(s1, 20.0)
    power_db = ripple(s2, 10.0)
    return {
        "cola_ripple_db": cola_db,
        "power_ripple_db": power_db,
        "cola": bool(cola_db < _RIPPLE_TOL_DB),
        "power_complementary": bool(power_db < _RIPPLE_TOL_DB),
        "cola_constant": float(s1.mean()),
        "power_constant": float(s2.mean()),
    }


_MOD_TRIM_S = 0.25  # dropped at both ends of the envelope (filter and resampling transients)


def modulation_peak(x, fs, rate, *, band=(1000.0, 4000.0), env_fs=200.0, half_width=1.0):
    """Level of a periodic power modulation at ``rate`` above its neighbours (pumping detector).

    Overlap-add synthesis with a wrong window modulates the signal power at
    the frame rate (ROADMAP section 6, item 11). The detector band-passes
    ``x`` (4th-order Butterworth, zero phase), takes the power envelope
    ``|x_a|**2`` with ``x_a`` the analytic signal (Boashash 1992), decimates
    it to ``env_fs`` (polyphase, anti-aliased), and estimates its modulation
    spectrum with Welch's method (Hann window, 50 % overlap, mean removed per
    segment; Welch 1967). The bin spacing of that spectrum is ``half_width``,
    so a line at ``rate`` is confined to the excluded interval. The result is
    the level at ``rate`` in dB relative to the median level of the
    neighbouring modulation frequencies: those with
    ``half_width < |f - rate| <= 10 half_width``, at least ``3 half_width``
    above 0 Hz and below ``0.4 env_fs``.

    A signal without modulation at ``rate`` gives about 0 dB, a periodic
    modulation gives its excess over the envelope noise floor. The noise floor
    is the natural envelope fluctuation of band-limited noise, so the result is
    a *detection statistic*, not the modulation depth: for a stationary
    Gaussian signal of ``T`` seconds it scatters around 0 dB with a standard
    deviation of about ``3.3 dB / sqrt(T * half_width)`` (about 1 dB for
    ``T = 10`` s and ``half_width = 1`` Hz), and a single realisation
    exceeds 1 dB by chance. Raise ``half_width`` or lengthen the record to
    tighten it (at the price of a lower sensitivity for a given modulation
    depth).

    Parameters
    ----------
    x : array_like, shape (n,)
        Signal in any unit (the result is a ratio and does not depend on the
        scale or on the amplitude convention); mono, analyse channels one at
        a time.
    fs : float
        Sampling rate in Hz.
    rate : float
        Modulation frequency in Hz to test, e.g. the grain or frame rate
        ``fs / hop`` of an overlap-add synthesis.
    band : (float, float), default (1000, 4000)
        Pass band in Hz of the carrier whose power envelope is analysed, with
        ``0 < lo < hi < fs/2``. A wider band lowers the noise floor of the
        envelope and raises the sensitivity.
    env_fs : float, default 200.0
        Sampling rate in Hz of the decimated envelope; ``rate`` must lie
        below ``0.4 env_fs``. Must be below ``fs``.
    half_width : float, default 1.0
        Half width in Hz of the interval around ``rate`` that is excluded
        from the neighbours, and the bin spacing of the modulation spectrum.
        ``rate`` must be more than ``3 half_width`` above 0 Hz.

    Returns
    -------
    float
        Level of the modulation spectrum at ``rate`` minus the median level of
        its neighbours, in dB. NaN if the signal has no envelope fluctuation
        at all (e.g. exact zeros).

    Raises
    ------
    ValueError
        For invalid arguments, a record shorter than ``2 / half_width``
        seconds (after the 0.25 s trimmed at each end), or no neighbouring
        frequencies within range.

    References
    ----------
    Boashash, B. (1992). Estimating and interpreting the instantaneous
    frequency of a signal. I. Fundamentals. Proceedings of the IEEE, 80(4),
    520-538.
    Welch, P. D. (1967). The use of fast Fourier transform for the estimation
    of power spectra: A method based on time averaging over short, modified
    periodograms. IEEE Transactions on Audio and Electroacoustics, 15(2),
    70-73.
    Allen, J. B., & Rabiner, L. R. (1977). A unified approach to short-time
    Fourier analysis and synthesis. Proceedings of the IEEE, 65(11),
    1558-1564.
    """
    x = _mono(x, "x")
    fs = _positive(fs, "fs")
    rate = _positive(rate, "rate")
    env_fs = _positive(env_fs, "env_fs")
    half_width = _positive(half_width, "half_width")
    band = _check_band(band, fs)
    if env_fs >= fs:
        raise ValueError("env_fs must be below fs")
    if rate >= 0.4 * env_fs:
        raise ValueError("rate must be below 0.4 * env_fs; raise env_fs")
    if rate <= 3.0 * half_width:
        raise ValueError("rate must be more than 3 * half_width above 0 Hz")

    ratio = Fraction(env_fs / fs).limit_denominator(100000)
    up, down = ratio.numerator, ratio.denominator
    fs_env = fs * up / down

    carrier = _bandpass(x - x.mean(), fs, band)
    power = np.abs(signal.hilbert(carrier)) ** 2
    env = signal.resample_poly(power, up, down)
    trim = round(_MOD_TRIM_S * fs_env)
    env = env[trim : env.size - trim]

    nperseg = round(fs_env / half_width)
    if nperseg < 8:
        raise ValueError("half_width is too large for env_fs (fewer than 8 samples per segment)")
    if env.size < 2 * nperseg:
        raise ValueError(
            f"x is too short: need at least {2 * nperseg / fs_env + 2 * _MOD_TRIM_S:.2f} s "
            f"for half_width = {half_width} Hz"
        )
    # Zero padding (x8) puts a bin close to `rate` whatever the segment length.
    f, pxx = signal.welch(
        env, fs_env, window="hann", nperseg=nperseg, noverlap=nperseg // 2, nfft=8 * nperseg
    )
    offset = np.abs(f - rate)
    neighbours = (
        (offset > half_width)
        & (offset <= 10.0 * half_width)
        & (f >= 3.0 * half_width)
        & (f <= 0.4 * fs_env)
    )
    if np.count_nonzero(neighbours) < 8:
        raise ValueError("no neighbouring modulation frequencies in range; check rate and half_width")
    peak = float(pxx[np.argmin(offset)])
    floor = float(np.median(pxx[neighbours]))
    if not (peak > 0.0 and floor > 0.0):
        return float("nan")
    return float(10.0 * np.log10(peak / floor))


def envelope_stats(x, fs, *, band=None, percentiles=(5, 95), reference=None, nperseg=None):
    """Envelope swing, crest factor and spectral error of a (resynthesised) signal.

    The envelope is the magnitude of the analytic signal (Hilbert transform,
    Boashash 1992) of ``x`` with its mean removed, optionally after a
    zero-phase 4th-order Butterworth band-pass. Its percentiles are expressed
    in dB relative to the RMS of the analysed signal, so they do not depend
    on the scale of ``x``; for a Gaussian signal the envelope is
    Rayleigh-distributed and for a sine of peak amplitude ``A`` it is the
    constant ``A``, i.e. ``+3.01 dB`` re RMS. The swing is the difference
    between the highest and the lowest requested percentile: the span of
    level that the envelope covers, a measure of how strongly the signal
    fluctuates (a pure tone: 0 dB; Gaussian noise: 17.7 dB between the 5th and
    95th percentile; two equal tones: about 22 dB).

    The crest factor is the peak-to-RMS ratio of the analysed signal (the
    band-passed signal if ``band`` is given). If ``reference`` is given, the
    spectral error compares the Welch power spectral densities (Welch 1967,
    Hann window, 50 % overlap) of ``x`` and ``reference`` in dB, bin by bin,
    and returns the RMS of the difference over frequency within ``band``
    (above 0 Hz if ``band`` is None). It contains the scatter of the two
    estimates: for independent realisations of the same noise, about
    ``6 dB / sqrt(K)`` with ``K`` the number of averaged segments (the default
    ``nperseg`` gives 100 segments for 10 s of signal, 0.6 dB).

    Parameters
    ----------
    x : array_like, shape (n,)
        Signal in any unit, mono. The envelope statistics are ratios and do
        not depend on the unit or on the amplitude convention (peak or RMS).
    fs : float
        Sampling rate in Hz.
    band : (float, float), optional
        Pass band ``(lo, hi)`` in Hz, ``0 < lo < hi < fs/2``, applied before
        the envelope and crest factor, and restricting the spectral error.
    percentiles : sequence of float, default (5, 95)
        At least two percentiles in [0, 100] of the envelope to report.
    reference : array_like, shape (m,), optional
        Reference signal (same unit as ``x``, same ``fs``, any length) for the
        spectral error; e.g. the recording a resynthesis was made from.
    nperseg : int, optional
        Welch segment length in samples for the spectral error. Default:
        ``fs / 5`` (a 5 Hz bin spacing), limited to the shorter signal.

    Returns
    -------
    dict
        ``swing_db``
            ``20 log10`` of the ratio of the highest to the lowest percentile
            of the envelope (``inf`` if the lowest is zero).
        ``percentiles_db``
            ``{percentile: 20 log10(envelope percentile / RMS)}`` with float
            keys, in the order given.
        ``crest_factor_db``
            ``20 log10(max |x| / RMS)`` of the analysed signal.
        ``spectral_error_db``
            RMS over frequency of the dB difference of the PSDs of ``x`` and
            ``reference`` (only if ``reference`` is given; NaN if no bin has
            power in both).

    Raises
    ------
    ValueError
        For non-mono or non-finite input, ``x`` without energy, invalid
        ``band`` or ``percentiles``, or ``nperseg`` below 8.

    References
    ----------
    Boashash, B. (1992). Estimating and interpreting the instantaneous
    frequency of a signal. I. Fundamentals. Proceedings of the IEEE, 80(4),
    520-538.
    Welch, P. D. (1967). The use of fast Fourier transform for the estimation
    of power spectra: A method based on time averaging over short, modified
    periodograms. IEEE Transactions on Audio and Electroacoustics, 15(2),
    70-73.
    """
    x = _mono(x, "x")
    fs = _positive(fs, "fs")
    pct = np.atleast_1d(np.asarray(percentiles, dtype=float))
    if pct.ndim != 1 or pct.size < 2 or not np.all(np.isfinite(pct)) or np.any((pct < 0) | (pct > 100)):
        raise ValueError("percentiles must hold at least two values in [0, 100]")
    if np.unique(pct).size != pct.size:
        raise ValueError("percentiles must be distinct")
    if band is not None:
        band = _check_band(band, fs)
    if reference is not None:
        reference = _mono(reference, "reference")
    if nperseg is not None and (int(nperseg) != nperseg or nperseg < 8):
        raise ValueError("nperseg must be an integer >= 8")

    analysed = x - x.mean()
    if band is not None:
        analysed = _bandpass(analysed, fs, band)
    rms = float(np.sqrt(np.mean(analysed**2)))
    if not rms > 0.0:
        raise ValueError("x has no energy in the analysed band")
    env = np.abs(signal.hilbert(analysed))
    values = np.percentile(env, pct)
    with np.errstate(divide="ignore"):
        levels = 20.0 * np.log10(values / rms)
    out = {
        "swing_db": float(levels[np.argmax(pct)] - levels[np.argmin(pct)]),
        "percentiles_db": {float(p): float(v) for p, v in zip(pct, levels)},
        "crest_factor_db": float(20.0 * np.log10(np.max(np.abs(analysed)) / rms)),
    }
    if reference is not None:
        seg = int(nperseg) if nperseg is not None else max(8, round(fs / 5.0))
        seg = min(seg, x.size, reference.size)
        f, p_x = signal.welch(x, fs, window="hann", nperseg=seg)
        _, p_ref = signal.welch(reference, fs, window="hann", nperseg=seg)
        use = (p_x > 0) & (p_ref > 0) & (f > 0)
        if band is not None:
            use &= (f >= band[0]) & (f <= band[1])
        if use.any():
            diff = 10.0 * np.log10(p_x[use] / p_ref[use])
            out["spectral_error_db"] = float(np.sqrt(np.mean(diff**2)))
        else:
            out["spectral_error_db"] = float("nan")
    return out
