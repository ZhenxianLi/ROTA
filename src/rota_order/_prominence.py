"""Prominent discrete tones: tone-to-noise ratio and prominence ratio.

Implemented from the text of ECMA-418-1 (3rd edition, December 2024),
*Psychoacoustic metrics for ITT equipment - Part 1 (prominent discrete
tones)*, which holds the technical content of ECMA-74 Annex D (17th edition,
2019; from its 18th edition ECMA-74 refers to ECMA-418-1, and ISO 7779 Annex D
describes the same two methods). Other open-source implementations of these
metrics are listed in: Li, Z. LISQM - A List of Open-Source Implementations
of Psychoacoustic and Sound Quality Metrics,
https://zhenxianli.github.io/LISQM/.

Changes of the 3rd edition implemented here (clause 11 and Annex A; the
critical bands, the prominence ratio and both criteria are unchanged from the
2nd edition, 2022):

* Tone band (11.2, A.2). Its end points are the minimal points on either
  side of the tone peak within a window of 15 % of the critical bandwidth
  centred on the peak; an end point on the window edge is moved outwards to
  the nearest local minimum. The 2nd edition's fallback of a fixed band of
  15 % of the critical band was withdrawn.
* Tone power (Formula (A.1)): the power above the straight line joining the
  two end points, ``X_t = X_t' - (x_1 + x_2) N / 2`` over the ``N`` lines of
  the tone band.
* Masking noise (Formula (A.2)): ``X_n = (X_tot - X_t) delta_fc /
  delta_f_tot``. The 2nd edition divided by ``delta_f_tot - delta_f_t``; the
  noise under the tone now stays in ``X_n``.

Reading of the text of the 3rd edition where it is inconsistent:

* Formulae (A.7) and (A.8) still read ``X_t,p - X_t,s`` (as in the 2nd
  edition); the text of A.6 and Formulae (15) and (16) of clause 11 add the
  secondary tone: the sum is used.
* The tone spacing is still printed ``|delta_f_s - delta_f_p|``; it is read
  as ``|f_s - f_p|``. Formula (14) is used below 1 kHz and every tone in the
  band is proximate from 1 kHz up (the new NOTE of 11.6 accepts the small
  discontinuity at 1 kHz).
* Clause 10 still leaves ``f0 = 500 Hz`` between its two ranges; 12.2 (``f <=
  500 Hz``: arithmetic centring) is followed. The new NOTE of clause 10 calls
  the band-edge discontinuity near 500 Hz negligible.
* The discrete tone range is 89.1 Hz to 11 220 Hz in 4.1.1 NOTE 2 (the outer
  edges of the 100 Hz and 10 kHz one-third-octave bands) but 11 200 Hz in
  clause 10, Formulae (13) and (26) and Tables 2 and 3: the criteria are
  given for 89.1 Hz <= f < 11 200 Hz, as in the formulae.
* Formula (20) now reads ``f_2,M = f_1,M + delta_f_M`` (corrected from
  ``f_t + delta_f_M``, Formula (19) of the 2nd edition), as implemented.
* New misprints, resolved with the Annex B counterparts: Formula (21) reads
  ``C_L,0 f_t`` for ``C_L,1 f_t`` (B.5), and the condition of Formula (23)
  reads ``f_t <= 171.4 Hz`` for ``> 171.4 Hz`` (B.7 and the text).

Two layers:

* :func:`tone_prominence` evaluates one tone on a given one-sided power
  spectrum (frequency axis in Hz).
* :func:`order_prominence` evaluates orders of a rotating machine on the
  order-tracked (angle-domain) spectrum, so that an order is a stationary
  line inside every analysis frame even during a run-up, and maps each
  frame's order axis to Hz with the frame's mean speed for the critical-band
  and criterion calculations. :func:`components` condenses the result into
  one description per order.

Conventions. Powers are mean-square values (RMS convention) in the squared
signal unit (e.g. Pa^2); power spectral densities are one-sided, in signal
unit^2 / Hz. Levels are ``10 lg(power / ref**2)`` dB (``ref = 20e-6`` for a
signal in Pa gives dB SPL). Frequencies in Hz, speed in rpm, orders in cycles
per revolution, ratios and criteria in dB.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from ._utils import as_2d, broadcast_rpm
from .maps import rpm_order_map

__all__ = [
    "OrderComponent",
    "OrderProminence",
    "ToneProminence",
    "components",
    "critical_band_edges",
    "critical_bandwidth",
    "order_prominence",
    "prominence_criterion",
    "tone_prominence",
]

_METHODS = ("tnr", "pr")
_SCALINGS = ("psd", "power")

# Range in which the criteria of 11.5 / 12.6 are defined (ECMA-418-1:2024,
# Formulae (12), (13), (25), (26); 4.1.1 NOTE 2 gives 11 220 Hz, see the
# module notes).
_F_LO = 89.1
_F_HI = 11200.0
# Frequency at and below which the lower critical band is truncated at 20 Hz
# (ECMA-418-1, 12.3 and 12.5).
_F_TRUNC = 171.4
# Width of the window, as a fraction of the critical bandwidth centred on the
# tone peak, in which the end points of the tone band are sought; a wider
# tone band means the tone is not resolved (ECMA-418-1:2024, 11.2 and A.2).
_TONE_FRACTION = 0.15


# ---------------------------------------------------------------------------
# Critical bands and criteria
# ---------------------------------------------------------------------------


def critical_bandwidth(f):
    """Width of the critical band centred at ``f``.

    ``delta_fc = 25 + 75 (1 + 1.4 (f / 1000 Hz)^2)^0.69`` Hz (ECMA-418-1,
    Formula (2)); about 100 Hz below 500 Hz, 162.2 Hz at 1 kHz.

    Parameters
    ----------
    f : float or array_like
        Centre frequency in Hz.

    Returns
    -------
    float or ndarray
        Critical bandwidth in Hz (same shape as ``f``).

    No amplitudes are involved (frequencies in Hz).

    References
    ----------
    Ecma International (2024). ECMA-418-1, Psychoacoustic metrics for ITT
    equipment - Part 1 (prominent discrete tones), 3rd edition. Clauses 10
    (critical bands), 11 and Annex A (tone-to-noise ratio), 12 and Annex B
    (prominence ratio).
    Ecma International (2019). ECMA-74, Measurement of airborne noise emitted
    by information technology and telecommunications equipment, 17th
    edition, Annex D.
    ISO 7779:2018. Acoustics - Measurement of airborne noise emitted by
    information technology and telecommunications equipment, Annex D.
    """
    f = np.asarray(f, dtype=float)
    out = 25.0 + 75.0 * (1.0 + 1.4 * (f / 1000.0) ** 2) ** 0.69
    return float(out) if out.ndim == 0 else out


def critical_band_edges(f, *, adjacent=False):
    """Edges of the critical band centred on a tone at ``f``.

    The middle band (the critical band of the tone) has width
    ``delta_fc = critical_bandwidth(f)``. Up to 500 Hz it is arithmetically
    centred, ``f -+ delta_fc / 2``; above 500 Hz it is geometrically centred,
    ``f1 f2 = f**2`` with ``f2 - f1 = delta_fc``, i.e.
    ``f1 = -delta_fc / 2 + sqrt(delta_fc**2 + 4 f**2) / 2`` (ECMA-418-1,
    Formulae (3) to (8) and (17) to (20); at exactly 500 Hz the arithmetic
    form is used, as in 12.2).

    The adjacent bands of the prominence ratio method are contiguous with
    the middle band: the lower band spans ``[f1_L, f1]`` and the upper band
    ``[f2, f2_U]``, with ``f1_L`` and ``f2_U`` from the quadratic fits of
    ECMA-418-1, Formulae (21) and (22) (B.5 and B.6) with Tables 2 and 3 (for
    ``f <= 171.4`` Hz the lower band starts at 20 Hz). Outside 89.1 Hz to
    11 200 Hz the end rows of the tables are extrapolated.

    Parameters
    ----------
    f : float
        Tone frequency in Hz (positive).
    adjacent : bool, default False
        Also return the outer edges of the lower and upper bands.

    Returns
    -------
    tuple of float
        ``(f1, f2)`` in Hz, or ``(f1_L, f1, f2, f2_U)`` if ``adjacent``.

    No amplitudes are involved (frequencies in Hz).

    References
    ----------
    Ecma International (2024). ECMA-418-1, Psychoacoustic metrics for ITT
    equipment - Part 1 (prominent discrete tones), 3rd edition. Clauses 10
    (critical bands), 11 and Annex A (tone-to-noise ratio), 12 and Annex B
    (prominence ratio).
    Ecma International (2019). ECMA-74, Measurement of airborne noise emitted
    by information technology and telecommunications equipment, 17th
    edition, Annex D.
    ISO 7779:2018. Acoustics - Measurement of airborne noise emitted by
    information technology and telecommunications equipment, Annex D.
    """
    f = float(f)
    if not (np.isfinite(f) and f > 0):
        raise ValueError("tone frequency must be positive (Hz)")
    dfc = critical_bandwidth(f)
    if f <= 500.0:
        f1 = f - 0.5 * dfc
    else:
        f1 = -0.5 * dfc + 0.5 * np.sqrt(dfc**2 + 4.0 * f**2)
    f2 = f1 + dfc
    if not adjacent:
        return float(f1), float(f2)
    # Table 2 (lower band, lower edge) and Table 3 (upper band, upper edge).
    if f < _F_TRUNC:
        f1l = 20.0
    elif f <= 1600.0:
        f1l = -149.5 + 1.001 * f - 6.90e-5 * f**2
    else:
        f1l = 6.8 + 0.806 * f - 8.20e-6 * f**2
    if f <= 1600.0:
        f2u = 149.5 + 1.035 * f + 7.70e-5 * f**2
    else:
        f2u = 3.3 + 1.215 * f + 2.16e-5 * f**2
    return float(f1l), float(f1), float(f2), float(f2u)


def prominence_criterion(f, method="tnr"):
    """Frequency-dependent prominence criterion in dB.

    Tone-to-noise ratio (ECMA-418-1:2024, 11.5, Formulae (12) and (13)):
    ``8 + 8.33 lg(1000 / f)`` dB for 89.1 Hz <= f < 1000 Hz and 8 dB for
    1000 Hz <= f < 11 200 Hz. Prominence ratio (12.6, Formulae (25) and
    (26)): ``9 + 10 lg(1000 / f)`` dB below 1 kHz and 9 dB from 1 kHz to
    11 200 Hz. A tone is prominent when its ratio is equal to or greater than
    the criterion and it is audible (11.8, 12.8: an aural examination, not
    assessed here). Outside 89.1 Hz to 11 200 Hz the standard gives no
    criterion and NaN is returned.

    Parameters
    ----------
    f : float or array_like
        Tone frequency in Hz.
    method : {"tnr", "pr"}, default "tnr"
        Tone-to-noise ratio or prominence ratio criterion.

    Returns
    -------
    float or ndarray
        Criterion in dB (NaN outside the frequency range).

    No amplitudes are involved (criterion in dB, frequency in Hz).

    References
    ----------
    Ecma International (2024). ECMA-418-1, Psychoacoustic metrics for ITT
    equipment - Part 1 (prominent discrete tones), 3rd edition. Clauses 10
    (critical bands), 11 and Annex A (tone-to-noise ratio), 12 and Annex B
    (prominence ratio).
    Ecma International (2019). ECMA-74, Measurement of airborne noise emitted
    by information technology and telecommunications equipment, 17th
    edition, Annex D.
    ISO 7779:2018. Acoustics - Measurement of airborne noise emitted by
    information technology and telecommunications equipment, Annex D.
    """
    _check_method(method)
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(f < 1000.0, np.log10(1000.0 / f), 0.0)
    out = 8.0 + 8.33 * lg if method == "tnr" else 9.0 + 10.0 * lg
    out = np.where((f >= _F_LO) & (f < _F_HI), out, np.nan)
    return float(out) if out.ndim == 0 else out


def _check_method(method):
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")


def _proximity_spacing(fp):
    """Proximity spacing of two tones in Hz (ECMA-418-1:2024, Formula (14)).

    Infinite from 1 kHz up: there the two tones are always combined.
    """
    if fp >= 1000.0:
        return np.inf
    return 21.0 * 10.0 ** (1.2 * abs(np.log10(fp / 212.0)) ** 1.8)


def _main_lobe_half(window):
    """Half-width of the window's main lobe, rounded up to whole lines."""
    n = 64
    w = signal.get_window(window, n)
    W = np.abs(np.fft.rfft(w, 64 * n))
    k = np.arange(W.size) / 64.0
    # First local minimum of the magnitude response: the first null.
    d = np.diff(W)
    i = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))
    half = k[i[0] + 1] if i.size else 1.0
    return max(1, int(np.ceil(half - 1e-3)))


def _enbw(window):
    w = signal.get_window(window, 1024)
    return float(w.size * np.sum(w**2) / np.sum(w) ** 2)


# ---------------------------------------------------------------------------
# Core evaluation on one spectrum
# ---------------------------------------------------------------------------


_KEYS = (
    "peak_freq", "tone_lo", "tone_hi", "tone_bandwidth", "tone_power",
    "total_power", "noise_power", "tnr", "middle_power", "lower_power",
    "upper_power", "pr", "tone_band_exceeded",
)


def _peak_line(f, S, centre, band_idx, taken, search):
    """Highest free line within ``centre +- search`` inside ``band_idx``.

    Falls back to the free line nearest ``centre``; None if none is free.
    """
    cand = band_idx[(np.abs(f[band_idx] - centre) <= search) & ~taken[band_idx]]
    if cand.size == 0:
        near = band_idx[~taken[band_idx]]
        if near.size == 0:
            return None
        cand = near[[np.argmin(np.abs(f[near] - centre))]]
    return int(cand[np.argmax(S[cand])])


def _stands_out(S, p, band_idx, taken, guard, thr):
    """Whether line ``p`` exceeds the median of the other free band lines by ``thr``."""
    ref_idx = band_idx[(np.abs(band_idx - p) > guard) & ~taken[band_idx]]
    if ref_idx.size < 3:
        ref_idx = band_idx[~taken[band_idx]]
    return bool(S[p] > np.median(S[ref_idx]) * thr)


def _tone_band(S, p, m, b0, b1, taken):
    """End points of the tone band around the peak line ``p``.

    ECMA-418-1:2024, 11.2 and A.2: on each side of the peak, the lowest line
    within the window of ``m`` lines (15 % of the critical bandwidth centred
    on the peak; on ties the line nearest the peak, so the end points are the
    minimal points bracketing the peak). If it lies on the window edge, the
    nearest local minimum beyond the edge is used: the search moves outwards
    while the next line is lower. Lines outside ``[b0, b1]`` (the critical
    band) or already ``taken`` by another tone are never used.

    Returns ``(lo, hi)``, the inclusive end lines.
    """
    ends = []
    for step in (-1, 1):
        lines = []
        k = p + step
        while abs(k - p) <= m and b0 <= k <= b1 and not taken[k]:
            lines.append(k)
            k += step
        if lines:
            e = lines[int(np.argmin(S[lines]))]
            at_edge = abs(e - p) == m
        elif m == 0 and b0 <= p + step <= b1 and not taken[p + step]:
            # Window narrower than one line: its edge is the peak itself.
            e, at_edge = p + step, True
        else:
            e, at_edge = p, False
        if at_edge:
            while b0 <= e + step <= b1 and not taken[e + step] and S[e + step] < S[e]:
                e += step
        ends.append(e)
    return ends[0], ends[1]


def _tone(f, S, df, p, b0, b1, taken):
    """Tone band and tone power above the line joining its end points.

    ``X_t = X_t' - (x_1 + x_2) N / 2`` (ECMA-418-1:2024, Formula (A.1)),
    with ``X_t'`` the power of the ``N`` lines of the tone band and ``x_1``,
    ``x_2`` the powers of its end lines; negative values are set to zero.
    Also returns whether ``delta_f_t = N df`` exceeds 15 % of the critical
    bandwidth centred on the peak (11.2).
    """
    dfc = critical_bandwidth(f[p])
    m = int(np.floor(0.5 * _TONE_FRACTION * dfc / df + 1e-9))
    lo, hi = _tone_band(S, p, m, b0, b1, taken)
    n = hi - lo + 1
    xt = (S[lo:hi + 1].sum() - 0.5 * n * (S[lo] + S[hi])) * df
    exceeded = n * df > _TONE_FRACTION * dfc * (1 + 1e-9)
    return lo, hi, max(float(xt), 0.0), bool(exceeded)


def _analyse(f, S, df, ft, others, *, half_lobe, threshold_db, search_hz):
    """TNR and PR quantities of the tone at ``ft`` on the PSD ``S`` (1-D)."""
    out = dict.fromkeys(_KEYS, np.nan)
    out["tone_band_exceeded"] = False
    out["combined"] = ()
    dfc = critical_bandwidth(ft)
    f1l, f1, f2, f2u = critical_band_edges(ft, adjacent=True)
    lo_edge, hi_edge = f[0] - 0.5 * df, f[-1] + 0.5 * df
    thr = 10.0 ** (threshold_db / 10.0)
    search = half_lobe * df if search_hz is None else max(float(search_hz), 0.5 * df)

    # Tone-to-noise ratio (clause 11 / Annex A).
    band_idx = np.flatnonzero((f >= f1) & (f < f2))
    if f1 >= lo_edge and f2 <= hi_edge and band_idx.size >= 2 * half_lobe + 3:
        b0, b1 = int(band_idx[0]), int(band_idx[-1])
        taken = np.zeros(f.size, bool)
        guard = max(half_lobe + 1, int(np.ceil(0.5 * _TONE_FRACTION * dfc / df)))
        p = _peak_line(f, S, ft, band_idx, taken, search)
        lo, hi, xt, exceeded = _tone(f, S, df, p, b0, b1, taken)
        out["tone_band_exceeded"] = exceeded
        taken[lo:hi + 1] = True
        dft = (hi - lo + 1) * df
        xrem = 0.0
        combined = []
        fprox = _proximity_spacing(ft)
        for fo in others:
            if not (f1 <= fo < f2) or taken[np.argmin(np.abs(f - fo))]:
                continue
            q = _peak_line(f, S, fo, band_idx, taken, search)
            if q is None or not _stands_out(S, q, band_idx, taken, guard, thr):
                continue
            slo, shi, xs, _ = _tone(f, S, df, q, b0, b1, taken)
            taken[slo:shi + 1] = True
            if abs(fo - ft) < fprox:  # proximate: perceived as one tone (11.6)
                xt, dft = xt + xs, dft + (shi - slo + 1) * df
                combined.append(float(fo))
            else:  # separate tone: removed from the masking noise only
                xrem += xs
        xtot = S[band_idx].sum() * df
        dftot = band_idx.size * df
        # Formulae (A.2) and (16): the noise under the tone bands stays in X_n.
        xn = (xtot - xt - xrem) * dfc / dftot
        with np.errstate(divide="ignore", invalid="ignore"):
            tnr = 10.0 * np.log10(xt / xn)
        out.update(
            peak_freq=float(f[p]), tone_lo=float(f[lo] - 0.5 * df),
            tone_hi=float(f[hi] + 0.5 * df), tone_bandwidth=float(dft),
            tone_power=float(xt), total_power=float(xtot), noise_power=float(xn),
            tnr=float(tnr), combined=tuple(combined),
        )

    # Prominence ratio (clause 12 / Annex B).
    if f1l >= lo_edge and f2u <= hi_edge and f1l < f1:
        xm = S[(f >= f1) & (f < f2)].sum() * df
        xl = S[(f >= f1l) & (f < f1)].sum() * df
        xu = S[(f >= f2) & (f < f2u)].sum() * df
        if ft <= _F_TRUNC:
            # Truncated lower band: normalised to a 100 Hz wide band (12.5).
            xl = xl * 100.0 / (f1 - f1l)
        with np.errstate(divide="ignore", invalid="ignore"):
            pr = 10.0 * np.log10(xm / (0.5 * (xl + xu)))
        out.update(middle_power=float(xm), lower_power=float(xl),
                   upper_power=float(xu), pr=float(pr))
    return out


# ---------------------------------------------------------------------------
# Single spectrum
# ---------------------------------------------------------------------------


@dataclass
class ToneProminence:
    """Prominence of one discrete tone (ECMA-418-1).

    Scalars for a single-channel spectrum; arrays of shape ``(ch,)`` where
    marked *per channel* for a multichannel one. Powers are mean-square
    values (RMS convention) in signal unit^2; levels in dB re ``ref**2``.

    Attributes
    ----------
    tone_freq : float
        Frequency of the tone under investigation in Hz (the band centre).
    method : str
        ``"tnr"`` or ``"pr"``: the method behind ``ratio``, ``criterion`` and
        ``prominent``. Both ratios are always computed.
    critical_bandwidth : float
        ``delta_fc`` at ``tone_freq`` in Hz.
    band : tuple of float
        ``(f1, f2)``, edges of the critical band of the tone in Hz.
    lower_band, upper_band : tuple of float
        Edges in Hz of the adjacent bands of the prominence ratio.
    peak_freq : float or ndarray
        Frequency of the highest line of the tone (per channel).
    tone_band : tuple of (float or ndarray)
        ``(lo, hi)``: outer edges in Hz of the lines assigned to the tone
        (per channel).
    tone_bandwidth : float or ndarray
        ``delta_ft``: number of lines in the tone band times the line
        spacing, Hz (including combined proximate tones; per channel).
    tone_band_exceeded : bool or ndarray
        True where the tone band of the tone under investigation is wider
        than 15 % of the critical bandwidth centred on its peak: the tone is
        not resolved (ECMA-418-1:2024, 11.2 asks for a finer resolution, and
        its CAUTION advises using the prominence ratio as well).
    tone_power, total_power, noise_power : float or ndarray
        ``X_t`` (tone power above the line joining the end points of the
        tone band), ``X_tot`` (all lines of the critical band) and ``X_n``
        (masking noise ``(X_tot - X_t)`` scaled to the critical bandwidth),
        per channel.
    tone_level, noise_level : float or ndarray
        ``10 lg(X / ref**2)`` in dB for the tone and the masking noise.
    tnr : float or ndarray
        Tone-to-noise ratio ``Delta L_T = 10 lg(X_t / X_n)`` in dB.
    middle_power, lower_power, upper_power : float or ndarray
        ``X_M``, ``X_L`` (already normalised to 100 Hz for tones at or
        below 171.4 Hz) and ``X_U``.
    pr : float or ndarray
        Prominence ratio ``Delta L_P`` in dB.
    tnr_criterion, pr_criterion : float
        Criteria in dB at ``tone_freq`` (NaN outside 89.1 Hz to 11.2 kHz).
    ratio, criterion : float or ndarray
        The ratio and criterion of ``method``.
    prominent : bool or ndarray
        ``ratio >= criterion`` (False where either is NaN). Audibility
        (ECMA-418-1, 11.8 and 12.8) is not assessed.
    margin : float or ndarray
        ``ratio - criterion`` in dB.
    combined_freqs : tuple
        Other tones combined with this one as proximate (11.6), Hz.
    ref : float
        Reference value for the levels, signal unit.
    """

    tone_freq: float
    method: str
    critical_bandwidth: float
    band: tuple
    lower_band: tuple
    upper_band: tuple
    peak_freq: object
    tone_band: tuple
    tone_bandwidth: object
    tone_band_exceeded: object
    tone_power: object
    total_power: object
    noise_power: object
    tone_level: object
    noise_level: object
    tnr: object
    middle_power: object
    lower_power: object
    upper_power: object
    pr: object
    tnr_criterion: float
    pr_criterion: float
    ratio: object
    criterion: float
    prominent: object
    margin: object
    combined_freqs: tuple = ()
    ref: float = 1.0


def _db(x, ref):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.asarray(x, dtype=float) / ref**2)


def _uniform_axis(freqs):
    f = np.asarray(freqs, dtype=float)
    if f.ndim != 1 or f.size < 8:
        raise ValueError("freqs must be a 1-D axis with at least 8 lines")
    d = np.diff(f)
    df = float(np.mean(d))
    if not df > 0 or np.max(np.abs(d - df)) > 1e-6 * df:
        raise ValueError("freqs must be uniformly spaced and increasing (Hz)")
    return f, df


def tone_prominence(
    freqs,
    spectrum,
    tone_freq,
    *,
    method="tnr",
    scaling="psd",
    window="hann",
    enbw=None,
    other_tones=(),
    threshold_db=6.0,
    search_hz=None,
    ref=1.0,
):
    """Tone-to-noise ratio and prominence ratio of one tone (ECMA-418-1).

    Implements, on a one-sided spectrum with uniformly spaced lines, both
    methods of ECMA-418-1:2024 (3rd edition; formerly ECMA-74 Annex D), in
    the mean-square form of its Annexes A and B:

    * **Tone-to-noise ratio** (clause 11). The critical band of the tone
      ``[f1, f2)`` follows :func:`critical_band_edges`. ``X_tot`` is the sum
      of the power of the ``N_tot`` lines in the band (``delta_f_tot = N_tot
      df``). The peak line of the tone is the highest line near
      ``tone_freq``. The tone band runs between the minimal points on either
      side of the peak within a window of 15 % of the critical bandwidth
      centred on the peak; a minimal point on the window edge is moved
      outwards to the nearest local minimum (11.2, A.2). The tone power is
      the power above the straight line joining the end points,
      ``X_t = X_t' - (x_1 + x_2) N / 2`` with ``X_t'`` the power of the
      ``N`` lines of the tone band and ``x_1``, ``x_2`` that of its end lines
      (Formula (A.1)). The masking noise is the rest of the band power
      rescaled to the critical bandwidth, ``X_n = (X_tot - X_t) delta_fc /
      delta_f_tot`` (Formula (A.2)), and ``TNR = 10 lg(X_t / X_n)`` dB.
    * **Prominence ratio** (clause 12). ``X_M``, ``X_L`` and ``X_U`` are the
      powers of the lines in the middle band and in the contiguous lower and
      upper bands; for tones at or below 171.4 Hz the lower band starts at
      20 Hz and ``X_L`` is scaled by ``100 Hz / (f1 - 20 Hz)``.
      ``PR = 10 lg(X_M / ((X_L + X_U) / 2))`` dB.
    * **Criteria** from :func:`prominence_criterion`; ``prominent`` is
      ``ratio >= criterion``.

    Lines are assigned to bands by their centre frequency on half-open
    intervals ``[lo, hi)``, so adjacent bands never share a line. The window
    of the tone band is ``+- floor(0.075 delta_fc / df)`` lines around the
    peak, with ``delta_fc`` at the peak frequency. The tone power is not the
    power of the tone lines: the noise under the line joining the end points
    stays in the masking noise, so for a tone on a locally flat or linear
    noise floor ``X_t`` is the power of the sinusoid alone.

    Window and ENBW. The standard prescribes a Hanning window and a power
    spectral density. With ``scaling="psd"`` the spectrum is a density
    (unit^2/Hz, e.g. :func:`scipy.signal.welch` with ``scaling="density"``),
    and band powers are ``sum(S) df``: correct for noise *and* for a tone
    whatever the window. With ``scaling="power"`` it is a mean-square value
    per line (an autospectrum: a sinusoid of RMS ``a`` on a line reads
    ``a**2``; e.g. ``|values|**2`` of an RMS :class:`rota_order.maps.Map`), which
    is converted to a density by dividing by ``enbw * df``. ``window`` sets
    the default ENBW and the main-lobe half-width used to search the peak.
    The 15 % window holds the whole main lobe of a stationary tone when the
    critical band spans at least ``(half-width + 1) / 0.075`` lines (40 for
    Hann).

    Multiple tones (11.6, A.6). Other tones listed in ``other_tones`` that
    lie in the critical band and whose peak line exceeds the median of the
    other band lines by ``threshold_db`` get their own tone band and tone
    power by the same rule; that power is removed from the masking noise
    (Formula (16): ``X_n = [X_tot - (X_t,p + X_t,s)] delta_fc /
    delta_f_tot``). Those closer to ``tone_freq`` than the proximity spacing
    ``21 * 10**(1.2 |lg(f / 212)|**1.8)`` Hz (always, from 1 kHz up) are
    also added to the tone (Formula (15)). The band always stays centred on
    ``tone_freq``.

    Parameters
    ----------
    freqs : array_like, shape (n,)
        Line frequencies in Hz, uniformly spaced.
    spectrum : array_like, shape (n,) or (n, ch)
        One-sided PSD (unit^2/Hz) or mean-square per line (unit^2), see
        ``scaling``. Non-negative.
    tone_freq : float
        Frequency of the tone under investigation in Hz.
    method : {"tnr", "pr"}, default "tnr"
        Method behind ``ratio``, ``criterion`` and ``prominent``.
    scaling : {"psd", "power"}, default "psd"
        Spectral convention of ``spectrum`` (see above).
    window : str, default "hann"
        Window used to compute the spectrum (any name accepted by
        :func:`scipy.signal.get_window`).
    enbw : float, optional
        Equivalent noise bandwidth of the window in lines; default from
        ``window`` (1.5 for Hann). Only used with ``scaling="power"``.
    other_tones : sequence of float, default ()
        Frequencies in Hz of other discrete tones (see Multiple tones).
    threshold_db : float, default 6
        An other tone is treated only if its peak line exceeds the median of
        the other band lines by this much (dB).
    search_hz : float, optional
        The peak line of a tone is searched within ``frequency +- search_hz``
        (default: the main-lobe half-width).
    ref : float, default 1
        Reference for the levels in the signal unit (20e-6 Pa: dB SPL).

    Returns
    -------
    ToneProminence
        Ratios in dB, powers in unit^2 (RMS convention), levels in dB re
        ``ref**2``. Quantities that need lines outside ``freqs`` are NaN.

    References
    ----------
    Ecma International (2024). ECMA-418-1, Psychoacoustic metrics for ITT
    equipment - Part 1 (prominent discrete tones), 3rd edition. Clauses 10
    (critical bands), 11 and Annex A (tone-to-noise ratio), 12 and Annex B
    (prominence ratio).
    Ecma International (2019). ECMA-74, Measurement of airborne noise emitted
    by information technology and telecommunications equipment, 17th
    edition, Annex D.
    ISO 7779:2018. Acoustics - Measurement of airborne noise emitted by
    information technology and telecommunications equipment, Annex D.
    """
    _check_method(method)
    if scaling not in _SCALINGS:
        raise ValueError(f"scaling must be one of {_SCALINGS}, got {scaling!r}")
    f, df = _uniform_axis(freqs)
    S, was_1d = as_2d(spectrum)
    if S.shape[0] != f.size:
        raise ValueError("spectrum must have one row per frequency line")
    if not np.all(np.isfinite(S)) or np.any(S < 0):
        raise ValueError("spectrum must be finite and non-negative")
    ft = float(tone_freq)
    if not (np.isfinite(ft) and ft > 0):
        raise ValueError("tone_freq must be positive (Hz)")
    if not ref > 0:
        raise ValueError("ref must be positive")
    if scaling == "power":
        e = _enbw(window) if enbw is None else float(enbw)
        if not e > 0:
            raise ValueError("enbw must be positive (lines)")
        S = S / (e * df)
    half = _main_lobe_half(window)
    others = [float(o) for o in np.atleast_1d(np.asarray(other_tones, dtype=float))]
    res = [
        _analyse(f, S[:, c], df, ft, others, half_lobe=half,
                 threshold_db=threshold_db, search_hz=search_hz)
        for c in range(S.shape[1])
    ]

    def col(key, dtype=float):
        v = np.array([r[key] for r in res], dtype=dtype)
        return v[0].item() if was_1d else v

    f1l, f1, f2, f2u = critical_band_edges(ft, adjacent=True)
    tnr, pr = col("tnr"), col("pr")
    ct, cp = prominence_criterion(ft, "tnr"), prominence_criterion(ft, "pr")
    ratio, crit = (tnr, ct) if method == "tnr" else (pr, cp)
    margin = np.asarray(ratio) - crit
    with np.errstate(invalid="ignore"):
        prom = np.asarray(margin >= 0) & np.isfinite(margin)
    return ToneProminence(
        tone_freq=ft,
        method=method,
        critical_bandwidth=critical_bandwidth(ft),
        band=(f1, f2),
        lower_band=(f1l, f1),
        upper_band=(f2, f2u),
        peak_freq=col("peak_freq"),
        tone_band=(col("tone_lo"), col("tone_hi")),
        tone_bandwidth=col("tone_bandwidth"),
        tone_band_exceeded=col("tone_band_exceeded", bool),
        tone_power=col("tone_power"),
        total_power=col("total_power"),
        noise_power=col("noise_power"),
        tone_level=_db(col("tone_power"), ref)[()],
        noise_level=_db(col("noise_power"), ref)[()],
        tnr=tnr,
        middle_power=col("middle_power"),
        lower_power=col("lower_power"),
        upper_power=col("upper_power"),
        pr=pr,
        tnr_criterion=ct,
        pr_criterion=cp,
        ratio=ratio,
        criterion=crit,
        prominent=prom.item() if was_1d else prom,
        margin=margin.item() if was_1d else margin,
        combined_freqs=tuple(sorted({c for r in res for c in r["combined"]})),
        ref=float(ref),
    )


# ---------------------------------------------------------------------------
# Order-tracked prominence
# ---------------------------------------------------------------------------


@dataclass
class OrderProminence:
    """TNR and PR of orders versus speed, from the angle-domain spectrum.

    ``n`` is the number of analysis points (frames, or rpm bins when
    ``rpm_bins`` was given), ``k`` the number of orders, ``ch`` the number
    of channels; the channel axis is dropped for a 1-D signal. Powers are
    mean-square values (RMS convention), levels in dB re ``ref**2``,
    ratios and criteria in dB.

    Attributes
    ----------
    orders : ndarray, shape (k,)
        Orders evaluated.
    method : str
        ``"tnr"`` or ``"pr"``: the method behind ``ratio``, ``criterion``
        and ``prominent``.
    t : ndarray, shape (n,)
        Centre time of each point in s (mean of the frame centres in a bin).
    rpm : ndarray, shape (n,)
        Mean speed of each point in rpm; it maps the order axis to Hz.
    freq : ndarray, shape (n, k)
        Tone frequency ``order * rpm / 60`` in Hz.
    ratio, criterion : ndarray, shape (n, k, ch) and (n, k)
        Ratio and criterion of ``method``.
    prominent : ndarray of bool, shape (n, k, ch)
        ``ratio >= criterion``.
    tnr, pr : ndarray, shape (n, k, ch)
        Tone-to-noise and prominence ratios in dB.
    tnr_criterion, pr_criterion : ndarray, shape (n, k)
        Criteria in dB (NaN outside 89.1 Hz to 11.2 kHz).
    tone_level, noise_level : ndarray, shape (n, k, ch)
        Tone and masking-noise levels (TNR method) in dB re ``ref**2``.
    tone_bandwidth : ndarray, shape (n, k, ch)
        Width of the lines assigned to the order, Hz.
    tone_band_exceeded : ndarray of bool, shape (n, k, ch)
        Tone wider than 15 % of the critical band (not resolved).
    critical_bandwidth : ndarray, shape (n, k)
        ``delta_fc`` at the order frequency, Hz.
    line_spacing : ndarray, shape (n,)
        Line spacing in Hz, ``order_resolution * rpm / 60``.
    n_frames : ndarray of int, shape (n,)
        Frames averaged into each point (1 without binning).
    frame_revs : float
        Frame length in revolutions.
    order_resolution : float
        Line spacing in orders, ``1 / frame_revs``.
    window : str
        Window of the angle-domain transform.
    enbw_bins : float
        Its equivalent noise bandwidth in lines.
    rpm_bins : ndarray or None
        Bin edges in rpm, if the points are rpm bins.
    ref : float
        Reference value of the levels, signal unit.
    params : dict
        Other parameters used.
    """

    orders: np.ndarray
    method: str
    t: np.ndarray
    rpm: np.ndarray
    freq: np.ndarray
    ratio: np.ndarray
    criterion: np.ndarray
    prominent: np.ndarray
    tnr: np.ndarray
    pr: np.ndarray
    tnr_criterion: np.ndarray
    pr_criterion: np.ndarray
    tone_level: np.ndarray
    noise_level: np.ndarray
    tone_bandwidth: np.ndarray
    tone_band_exceeded: np.ndarray
    critical_bandwidth: np.ndarray
    line_spacing: np.ndarray
    n_frames: np.ndarray
    frame_revs: float
    order_resolution: float
    window: str
    enbw_bins: float
    rpm_bins: object = None
    ref: float = 1.0
    params: dict = field(default_factory=dict)


def _default_frame_revs(rpm, orders, resolution_fraction, min_lines):
    """Frame length in revolutions from the two resolution rules.

    1. Line spacing at most ``resolution_fraction`` of the tone frequency:
       in the angle domain this is ``order_resolution <= fraction * order``,
       i.e. ``frame_revs >= 1 / (fraction * order)``, at every speed.
    2. At least ``min_lines`` lines across the critical band at every speed:
       ``frame_revs >= min_lines * (rpm/60) / delta_fc(order * rpm/60)``.
    """
    r = np.unique(np.asarray(rpm, dtype=float)) / 60.0
    r = r[r > 0]
    need = 1.0
    for o in orders:
        need = max(need, 1.0 / (resolution_fraction * o))
        if r.size:
            need = max(need, float(np.max(min_lines * r / critical_bandwidth(o * r))))
    return float(np.ceil(need - 1e-9))


def order_prominence(
    x,
    fs,
    rpm,
    orders,
    *,
    method="tnr",
    order_resolution=None,
    frame_revs=None,
    resolution_fraction=0.01,
    min_lines=None,
    overlap=0.5,
    window="hann",
    rpm_bins=None,
    multiple_tones=True,
    threshold_db=6.0,
    ref=1.0,
    max_order=None,
    samples_per_rev=None,
    min_rpm=None,
    delay=0.0,
):
    """Tone-to-noise ratio and prominence ratio of orders versus speed.

    The signal is resampled to uniform shaft angle (:func:`rota_order.maps.
    rpm_order_map`) and transformed in frames of ``frame_revs`` revolutions,
    so an order is a stationary spectral line within every frame, even in a
    fast run-up. A fixed-length time window long enough to meet the
    resolution required by the standard at low frequency would instead see
    the order sweep across many lines; the tone then spreads into the band
    used for the masking noise and the TNR is biased low (it is the drift
    sensitivity of long-window tonal metrics). Each frame's order axis is
    mapped to Hz with the frame's mean speed ``r = rpm/60``: line ``o`` lies
    at ``o r`` Hz, and the PSD per Hz is ``|V|**2 / (enbw * order_resolution
    * r)`` with ``V`` the RMS-scaled order spectrum. :func:`tone_prominence`
    is then applied at ``order * r`` Hz (ECMA-418-1, clauses 10 to 12).

    Frame length rule (``frame_revs`` None). The longest of

    1. ``1 / (resolution_fraction * order)`` revolutions: ECMA-418-1 clause 7
       requires a line spacing below 1 % of the tone frequency
       (``resolution_fraction=0.01``; 0.0025 is recommended for the TNR). In
       the angle domain the spacing relative to the tone, ``order_resolution
       / order``, does not depend on speed.
    2. ``min_lines * r / delta_fc(order r)`` revolutions at the highest
       ``r / delta_fc`` of the run: at least ``min_lines`` lines across the
       critical band at every speed. The default, ``(main-lobe half-width +
       2) / 0.075`` lines (54 for Hann, 94 for a flat-top), makes the window
       of 15 % of the critical band in which ECMA-418-1:2024, 11.2 seeks the
       end points of the tone band hold an order's main lobe and two noise
       lines on each side. Example: a 140 Hz tone (critical band about
       101 Hz) as order 2 at 4200 rpm needs a line spacing of 1.9 Hz with
       Hann, i.e. 38 revolutions (0.54 s), and the 1 % rule asks for 50
       revolutions (0.71 s).

    Averaging. With ``rpm_bins`` the PSDs (per Hz, on the order axis) of all
    frames whose mean speed falls in a bin are averaged linearly, as the
    standard's RMS averaging, and mapped to Hz with the bin's mean speed;
    this lowers the scatter of the masking-noise estimate.

    Multiple orders. With ``multiple_tones`` the other requested orders are
    passed as ``other_tones``: those falling in an order's critical band are
    removed from its masking noise, or combined with it when proximate
    (ECMA-418-1, 11.6).

    One frame length for all orders. The frames, and so the points of the
    result (``t``, ``rpm``, ``n_frames``, ``line_spacing``), are shared by the
    orders, so ``frame_revs`` is one number and the default rule is evaluated
    for the order that needs the longest frame, usually the lowest. Higher
    orders are then analysed with a longer frame than their own minimum: their
    line spacing is finer than the standard requires, which keeps the
    requirements of clause 7 and 11.2, but the curve resolves speed changes
    coarsely (every frame spans the same number of revolutions, for example 53, whatever
    the order). A different frame length per order would
    give each order its own frames and speed points, which the result type does
    not hold. To get the speed resolution of a high order, call the function once per
    order (or group of orders of similar rule) and draw the results separately, e.g.
    ``[order_prominence(x, fs, rpm, [o], rpm_bins=...) for o in (6.7, 40)]``; the
    multiple-tone treatment then only sees the orders of the same call, so list
    together the orders that fall in one critical band.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, ch)
        Signal (e.g. sound pressure in Pa). Channels are independent.
    fs : float
        Sampling rate in Hz.
    rpm : float or array_like, shape (n,)
        Shaft speed in rpm at the samples.
    orders : float or sequence of float
        Orders to evaluate (positive).
    method : {"tnr", "pr"}, default "tnr"
        Method behind ``ratio``, ``criterion`` and ``prominent``.
    order_resolution : float, optional
        Line spacing in orders; same as ``frame_revs = 1 / order_resolution``.
    frame_revs : float, optional
        Frame length in revolutions (default: the rule above).
    resolution_fraction : float, default 0.01
        Line spacing as a fraction of the tone frequency, rule 1.
    min_lines : float, optional
        Minimum number of lines across the critical band, rule 2.
    overlap : float, default 0.5
        Frame overlap fraction (in angle).
    window : str, default "hann"
        Window of the angle-domain transform (the standard uses Hanning).
    rpm_bins : float or array_like, optional
        Bin width in rpm (float) or bin edges (array) for averaging.
    multiple_tones : bool, default True
        Treat the other orders as other tones (see Multiple orders).
    threshold_db, ref
        As in :func:`tone_prominence`.
    max_order, samples_per_rev, min_rpm, delay
        Passed to :func:`rota_order.angle.angular_resample`. The order axis must
        reach the upper band of the PR method (about ``1.2 order + 150 /
        r``); quantities needing lines beyond it are NaN.

    Returns
    -------
    OrderProminence
        Per point and order: ratios in dB, levels in dB re ``ref**2`` (RMS
        convention), prominence flags.

    References
    ----------
    Ecma International (2024). ECMA-418-1, Psychoacoustic metrics for ITT
    equipment - Part 1 (prominent discrete tones), 3rd edition. Clauses 10
    (critical bands), 11 and Annex A (tone-to-noise ratio), 12 and Annex B
    (prominence ratio).
    Ecma International (2019). ECMA-74, Measurement of airborne noise emitted
    by information technology and telecommunications equipment, 17th
    edition, Annex D.
    ISO 7779:2018. Acoustics - Measurement of airborne noise emitted by
    information technology and telecommunications equipment, Annex D.
    Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order
    tracking. Mechanical Systems and Signal Processing, 11(2), 187-205.
    """
    _check_method(method)
    orders = np.atleast_1d(np.asarray(orders, dtype=float))
    if orders.ndim != 1 or orders.size == 0 or np.any(~np.isfinite(orders)) or np.any(orders <= 0):
        raise ValueError("orders must be positive")
    X, was_1d = as_2d(x)
    rpm_arr = broadcast_rpm(rpm, X.shape[0])
    if not 0 < resolution_fraction < 1:
        raise ValueError("resolution_fraction must be in (0, 1)")
    half = _main_lobe_half(window)
    if min_lines is None:
        min_lines = float(np.ceil((half + 2) / (0.5 * _TONE_FRACTION) - 1e-9))
    if order_resolution is not None:
        if not order_resolution > 0:
            raise ValueError("order_resolution must be positive")
        if frame_revs is not None and not np.isclose(frame_revs * order_resolution, 1.0):
            raise ValueError("give order_resolution or frame_revs, not both inconsistent")
        frame_revs = 1.0 / order_resolution
    if frame_revs is None:
        frame_revs = _default_frame_revs(rpm_arr, orders, resolution_fraction, min_lines)
    if not frame_revs > 0:
        raise ValueError("frame_revs must be positive")
    m = rpm_order_map(
        X, fs, rpm_arr, order_resolution=1.0 / frame_revs, max_order=max_order,
        samples_per_rev=samples_per_rev, overlap=overlap, window=window,
        amplitude="rms", min_rpm=min_rpm, delay=delay,
    )
    dor = m.resolution
    V = np.abs(m.values) ** 2  # (bins, frames, ch): mean square per line
    r = m.rpm / 60.0
    with np.errstate(divide="ignore", invalid="ignore"):
        D = V / (m.enbw_bins * dor * r[None, :, None])  # PSD per Hz

    # Analysis points: frames or rpm bins.
    if rpm_bins is None:
        edges = None
        groups = [np.array([i]) for i in range(r.size)]
    else:
        rb = np.asarray(rpm_bins, dtype=float)
        if rb.ndim == 0:
            if not rb > 0:
                raise ValueError("rpm_bins width must be positive (rpm)")
            lo = np.floor(m.rpm.min() / rb) * rb
            edges = np.arange(lo, m.rpm.max() + rb, rb)
            if edges[-1] <= m.rpm.max():
                edges = np.append(edges, edges[-1] + rb)
        else:
            edges = rb
            if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
                raise ValueError("rpm_bins edges must be increasing")
        idx = np.digitize(m.rpm, edges) - 1
        groups = [np.flatnonzero(idx == b) for b in range(edges.size - 1)]
        groups = [g for g in groups if g.size]

    n, k, ch = len(groups), orders.size, X.shape[1]
    shape3 = (n, k, ch)
    out = {key: np.full(shape3, np.nan) for key in (
        "tnr", "pr", "tone_power", "noise_power", "tone_bandwidth")}
    exceeded = np.zeros(shape3, bool)
    t_pt = np.empty(n)
    rpm_pt = np.empty(n)
    nfr = np.empty(n, int)
    for j, g in enumerate(groups):
        rj = float(np.mean(r[g]))
        t_pt[j], rpm_pt[j], nfr[j] = float(np.mean(m.t[g])), 60.0 * rj, g.size
        if not rj > 0:
            continue
        S = D[:, g, :].mean(axis=1)  # (bins, ch)
        f = m.axis * rj
        df = dor * rj
        ok = np.isfinite(S).all(axis=0)
        for i, o in enumerate(orders):
            ft = o * rj
            others = [oo * rj for oo in orders if oo != o] if multiple_tones else []
            for c in range(ch):
                if not ok[c]:
                    continue
                res = _analyse(f, S[:, c], df, ft, others, half_lobe=half,
                               threshold_db=threshold_db, search_hz=None)
                for key, arr in out.items():
                    arr[j, i, c] = res[key]
                exceeded[j, i, c] = res["tone_band_exceeded"]

    freq = orders[None, :] * rpm_pt[:, None] / 60.0
    ct = prominence_criterion(freq, "tnr")
    cp = prominence_criterion(freq, "pr")
    ratio = out["tnr"] if method == "tnr" else out["pr"]
    crit = ct if method == "tnr" else cp
    with np.errstate(invalid="ignore"):
        prom = (ratio >= crit[:, :, None]) & np.isfinite(ratio) & np.isfinite(crit)[:, :, None]

    def sq(a):
        return a[..., 0] if was_1d else a

    return OrderProminence(
        orders=orders,
        method=method,
        t=t_pt,
        rpm=rpm_pt,
        freq=freq,
        ratio=sq(ratio),
        criterion=crit,
        prominent=sq(prom),
        tnr=sq(out["tnr"]),
        pr=sq(out["pr"]),
        tnr_criterion=ct,
        pr_criterion=cp,
        tone_level=sq(_db(out["tone_power"], ref)),
        noise_level=sq(_db(out["noise_power"], ref)),
        tone_bandwidth=sq(out["tone_bandwidth"]),
        tone_band_exceeded=sq(exceeded),
        critical_bandwidth=critical_bandwidth(freq),
        line_spacing=dor * rpm_pt / 60.0,
        n_frames=nfr,
        frame_revs=float(frame_revs),
        order_resolution=float(dor),
        window=window,
        enbw_bins=float(m.enbw_bins),
        rpm_bins=edges,
        ref=float(ref),
        params={
            "resolution_fraction": float(resolution_fraction),
            "min_lines": float(min_lines),
            "overlap": float(overlap),
            "multiple_tones": bool(multiple_tones),
            "threshold_db": float(threshold_db),
            "samples_per_rev": round(m.nperseg / frame_revs),
        },
    )


# ---------------------------------------------------------------------------
# Component description
# ---------------------------------------------------------------------------


@dataclass
class OrderComponent:
    """Summary of one order component over a run.

    Attributes
    ----------
    order : float
        Order (cycles per revolution).
    fundamental : float
        Order the harmonicity refers to.
    harmonic_number : int
        Nearest whole multiple ``max(1, round(order / fundamental))``.
    harmonicity : float
        ``1 - 2 |order / fundamental - harmonic_number|``, clipped to
        [0, 1]: 1 for an exact harmonic, 0 halfway between two harmonics.
    rpm_range : tuple of float
        Lowest and highest speed (rpm) at which the order is prominent
        (NaN, NaN if never).
    prominent_fraction : float
        Fraction of evaluated points at which the order is prominent.
    rpm_peak : float
        Speed (rpm) of the highest tone level.
    level : float
        Highest tone level in dB re ``ref**2`` (RMS convention).
    level_mean : float
        Energetic mean of the tone level over the evaluated points, dB.
    ratio_max : float
        Highest TNR or PR (``method``) in dB.
    margin_max : float
        Highest ``ratio - criterion`` in dB.
    bandwidth : float
        Median width of the lines assigned to the order, in orders (a pure
        order gives about the window's main-lobe width).
    bandwidth_hz : float
        The same in Hz (median over points).
    method : str
        ``"tnr"`` or ``"pr"``.
    """

    order: float
    fundamental: float
    harmonic_number: int
    harmonicity: float
    rpm_range: tuple
    prominent_fraction: float
    rpm_peak: float
    level: float
    level_mean: float
    ratio_max: float
    margin_max: float
    bandwidth: float
    bandwidth_hz: float
    method: str


def components(result, *, fundamental=None, channel=0):
    """Describe each order of an :func:`order_prominence` result.

    For every order: the speed range where it is prominent, its tone level
    (highest and energetic mean, dB re ``ref**2``, RMS convention), the
    highest ratio and margin over the criterion, a bandwidth estimate (the
    median width of the lines assigned to it, in orders and Hz) and a
    harmonicity index relative to a fundamental order. The harmonicity index
    is defined here (it is not part of ECMA-418-1, which evaluates each
    harmonic independently, 11.7): with ``h = order / fundamental`` and
    ``n = max(1, round(h))``, ``harmonicity = clip(1 - 2 |h - n|, 0, 1)``.

    Parameters
    ----------
    result : OrderProminence
        Output of :func:`order_prominence`.
    fundamental : float, optional
        Fundamental order (default: the lowest order of ``result``).
    channel : int, default 0
        Channel described, for a multichannel result.

    Returns
    -------
    list of OrderComponent
        One per order, in the order of ``result.orders``.

    References
    ----------
    Ecma International (2024). ECMA-418-1, Psychoacoustic metrics for ITT
    equipment - Part 1 (prominent discrete tones), 3rd edition. Clauses 10
    (critical bands), 11 and Annex A (tone-to-noise ratio), 12 and Annex B
    (prominence ratio).
    Ecma International (2019). ECMA-74, Measurement of airborne noise emitted
    by information technology and telecommunications equipment, 17th
    edition, Annex D.
    ISO 7779:2018. Acoustics - Measurement of airborne noise emitted by
    information technology and telecommunications equipment, Annex D.
    """
    if not isinstance(result, OrderProminence):
        raise TypeError("result must be an OrderProminence")
    f0 = float(np.min(result.orders)) if fundamental is None else float(fundamental)
    if not f0 > 0:
        raise ValueError("fundamental must be a positive order")

    def pick(a):
        a = np.asarray(a)
        return a[:, :, channel] if a.ndim == 3 else a

    ratio, prom = pick(result.ratio), pick(result.prominent)
    level, bw = pick(result.tone_level), pick(result.tone_bandwidth)
    margin = ratio - result.criterion
    out = []
    for i, o in enumerate(result.orders):
        h = o / f0
        nh = max(1, int(np.round(h)))
        harm = float(np.clip(1.0 - 2.0 * abs(h - nh), 0.0, 1.0))
        valid = np.isfinite(ratio[:, i])
        pr_i = prom[:, i] & valid
        rp = result.rpm[pr_i]
        rng_ = (float(rp.min()), float(rp.max())) if rp.size else (np.nan, np.nan)
        lv = level[:, i]
        lvf = np.isfinite(lv)
        if lvf.any():
            ipk = int(np.flatnonzero(lvf)[np.argmax(lv[lvf])])
            lmax, rpk = float(lv[ipk]), float(result.rpm[ipk])
            lmean = float(10.0 * np.log10(np.mean(10.0 ** (lv[lvf] / 10.0))))
        else:
            lmax = rpk = lmean = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            bw_o = bw[:, i] / (result.rpm / 60.0)
        bwf = np.isfinite(bw[:, i])
        out.append(OrderComponent(
            order=float(o),
            fundamental=f0,
            harmonic_number=nh,
            harmonicity=harm,
            rpm_range=rng_,
            prominent_fraction=float(pr_i.sum() / valid.sum()) if valid.any() else np.nan,
            rpm_peak=rpk,
            level=lmax,
            level_mean=lmean,
            ratio_max=float(np.nanmax(ratio[:, i])) if valid.any() else np.nan,
            margin_max=float(np.nanmax(margin[:, i])) if np.isfinite(margin[:, i]).any() else np.nan,
            bandwidth=float(np.median(bw_o[bwf])) if bwf.any() else np.nan,
            bandwidth_hz=float(np.median(bw[bwf, i])) if bwf.any() else np.nan,
            method=result.method,
        ))
    return out
