"""Order tables and additive order synthesis.

An :class:`OrderTable` holds the complex amplitude of a set of orders on a
speed grid. :func:`synthesize_orders` renders it for a given speed curve as a
sum of sinusoids whose phase is the continuous integral of the speed
(McAulay & Quatieri, 1986), and :class:`OrderSynth` does the same block by
block while keeping the phase accumulators between calls.

Units: time in s, frequency in Hz, rotational speed in rpm, shaft angle in
revolutions, phase in rad, order in cycles per revolution. Table values are
in the signal unit (for example Pa), in the amplitude convention stored in
``OrderTable.amplitude``. Synthesis output is always a sum of *peak*
amplitude sinusoids in the table's unit.

The magnitude and the phase of the table are interpolated separately, never
the real and imaginary parts: where the phase differs between neighbouring
grid points, interpolating real and imaginary parts averages vectors that
point in different directions and the magnitude collapses (ROADMAP section 6,
item 10).
"""

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._provenance import jsonable
from ._utils import check_amplitude
from .speed import rpm_to_phase

__all__ = ["OrderTable", "synthesize_orders", "OrderSynth"]

_TWO_PI = 2.0 * np.pi
_SQRT2 = float(np.sqrt(2.0))
_FORMAT_VERSION = 1
# Samples rendered at a time. Bounds the temporaries; the output does not depend on it.
_CHUNK = 1 << 16
_PHASE_MODES = ("random", "zero", "table")


# --------------------------------------------------------------------------
# Grid helpers
# --------------------------------------------------------------------------


def _check_grid(grid, name):
    g = np.array(grid, dtype=float)
    if g.ndim != 1 or g.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.all(np.isfinite(g)):
        raise ValueError(f"{name} must be finite")
    if np.any(np.diff(g) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return g


def _bin_edges(grid):
    """Edges of the bins around the grid points: midpoints, outer edges half a spacing out."""
    if grid.size < 2:
        raise ValueError("an rpm grid for binning needs at least two points")
    mid = 0.5 * (grid[1:] + grid[:-1])
    return np.concatenate(([2.0 * grid[0] - mid[0]], mid, [2.0 * grid[-1] - mid[-1]]))


def _bin_index(rpm, edges):
    """Bin number of each speed and a mask of the speeds that fall inside some bin."""
    m = edges.size - 1
    idx = np.searchsorted(edges, rpm, side="right") - 1
    idx = np.where(rpm == edges[-1], m - 1, idx)  # the outer edge belongs to the last bin
    valid = np.isfinite(rpm) & (idx >= 0) & (idx < m)
    return np.where(valid, idx, 0), valid


def _bin_complex(idx, valid, m, power, phasor):
    """Energy-averaged magnitude and mean-phasor phase per bin, as complex numbers.

    ``sqrt(mean(power))`` over the finite ``power`` samples of a bin is the
    magnitude. The phase is the angle of the mean of the unit phasors
    ``phasor / |phasor|`` (finite, non-zero ones only), so it is insensitive to
    wrapping; a bin without usable phasor gets phase 0. A bin without finite
    ``power`` is NaN.
    """
    ok = valid & np.isfinite(power)
    cnt = np.bincount(idx[ok], minlength=m)
    tot = np.bincount(idx[ok], weights=power[ok], minlength=m)
    has = cnt > 0
    mag = np.full(m, np.nan)
    mag[has] = np.sqrt(tot[has] / cnt[has])
    mod = np.abs(phasor)
    okp = valid & np.isfinite(phasor) & (mod > 0)
    unit = phasor[okp] / mod[okp]
    ur = np.bincount(idx[okp], weights=unit.real, minlength=m)
    ui = np.bincount(idx[okp], weights=unit.imag, minlength=m)
    out = np.full(m, complex(np.nan, np.nan))
    out[has] = mag[has] * np.exp(1j * np.arctan2(ui[has], ur[has]))
    return out


def _moving_average(grid, y, width):
    """NaN-aware boxcar average of ``y`` (k, m[, ch]) along the grid, ``width`` rpm wide.

    Grid point ``i`` is replaced by the mean of the finite values at points
    within ``width / 2`` of it. Points that are NaN stay NaN.
    """
    half = 0.5 * width
    eps = 1e-9 * width
    lo = np.searchsorted(grid, grid - half - eps, side="left")
    hi = np.searchsorted(grid, grid + half + eps, side="right")
    fin = np.isfinite(y)
    zero = np.zeros_like(y[:, :1])
    cs = np.concatenate([zero, np.cumsum(np.where(fin, y, 0.0), axis=1)], axis=1)
    cn = np.concatenate([zero, np.cumsum(fin, axis=1)], axis=1)
    tot = cs[:, hi] - cs[:, lo]
    cnt = cn[:, hi] - cn[:, lo]
    out = np.full(y.shape, np.nan)
    ok = fin & (cnt > 0)
    out[ok] = tot[ok] / cnt[ok]
    return out


class _GridInterp:
    """Piecewise-linear interpolation of ``y`` (k, m[, ch]) on an rpm grid.

    NaN grid points are skipped, values outside the grid are the end values,
    a row (and channel) without any finite point gives NaN.
    """

    def __init__(self, grid, y):
        self.mono = y.ndim == 2
        Y = y[..., None] if self.mono else y
        self.k, _, self.C = Y.shape
        self._pts = []
        for j in range(self.k):
            row = []
            for c in range(self.C):
                col = Y[j, :, c]
                ok = np.isfinite(col)
                row.append((grid[ok], col[ok]) if ok.any() else None)
            self._pts.append(row)

    def row(self, j, q):
        """Row ``j`` at the 1-D speeds ``q``: shape (len(q), ch), NaN where undefined."""
        out = np.full((q.size, self.C), np.nan)
        for c, pts in enumerate(self._pts[j]):
            if pts is not None:
                out[:, c] = np.interp(q, *pts)
        return out

    def all(self, q):
        """All rows at the speeds ``q``: shape (k, len(q)) or (k, len(q), ch)."""
        out = np.stack([self.row(j, q) for j in range(self.k)])
        return out[..., 0] if self.mono else out


# --------------------------------------------------------------------------
# OrderTable
# --------------------------------------------------------------------------


@dataclass(eq=False)
class OrderTable:
    """Complex order amplitudes on a speed grid.

    Row ``j`` describes a component whose instantaneous frequency is
    ``orders[j] * rpm / 60 + freq_offset[j]`` Hz. ``values[j, i]`` is its
    complex amplitude at ``rpm[i]``: the magnitude is the amplitude in the
    convention given by ``amplitude`` and the angle is the phase in rad
    relative to the carrier ``2 pi (orders[j] rev + freq_offset[j] t)``, with
    ``rev`` the shaft angle in revolutions from
    :func:`rota_order.speed.rpm_to_phase`. This is the phase that
    :func:`rota_order.slices.order_slice` reads on an order-map bin and that the
    envelopes of the order extractors (``OrderComponents.envelopes``) carry.

    Orders may be negative or fractional. A fixed offset ``freq_offset`` turns
    a row into a carrier sideband, for example the switching frequency of an
    inverter plus or minus a multiple of the shaft frequency.

    Parameters
    ----------
    orders : array_like, shape (k,)
        Order of each row, in cycles per revolution.
    rpm : array_like, shape (m,)
        Speed grid in rpm, strictly increasing.
    values : array_like, complex, shape (k, m) or (k, m, ch)
        Complex amplitudes in the signal unit. NaN marks grid points without
        data. Real input is accepted and treated as zero phase.
    amplitude : {"peak", "rms"}
        Amplitude convention of ``values``. Required: there is no default
        (ROADMAP section 6, item 14).
    freq_offset : float or array_like, shape (k,), optional
        Fixed frequency offset of each row in Hz (default 0).
    unit : str, default ""
        Unit of ``values``, for information (for example ``"Pa"``).
    meta : dict, optional
        Free-form, JSON-serialisable notes (non-finite floats are saved as
        strings).

    Notes
    -----
    All arrays are copied on construction. Missing data is NaN, never
    infinity.

    References
    ----------
    McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis based
    on a sinusoidal representation. IEEE Transactions on Acoustics, Speech,
    and Signal Processing, 34(4), 744-754.
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    """

    orders: np.ndarray
    rpm: np.ndarray
    values: np.ndarray
    amplitude: str
    freq_offset: np.ndarray | None = None
    unit: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        orders = np.array(self.orders, dtype=float)
        if orders.ndim != 1 or orders.size == 0:
            raise ValueError("orders must be a non-empty 1-D array")
        if not np.all(np.isfinite(orders)):
            raise ValueError("orders must be finite")
        rpm = _check_grid(self.rpm, "rpm")
        values = np.array(self.values, dtype=complex)
        k, m = orders.size, rpm.size
        if values.ndim not in (2, 3) or values.shape[:2] != (k, m) or values.size == 0:
            raise ValueError(
                f"values must have shape ({k}, {m}) or ({k}, {m}, ch), got {values.shape}"
            )
        if np.any(np.isinf(values)):
            raise ValueError("values must not contain infinity; use NaN for missing data")
        check_amplitude(self.amplitude)
        if self.freq_offset is None:
            offset = np.zeros(k)
        else:
            offset = np.array(self.freq_offset, dtype=float)
            if offset.ndim == 0:
                offset = np.full(k, float(offset))
            if offset.shape != (k,) or not np.all(np.isfinite(offset)):
                raise ValueError(f"freq_offset must be finite, a scalar or of shape ({k},)")
        if not isinstance(self.unit, str):
            raise TypeError("unit must be a string")
        if not isinstance(self.meta, Mapping):
            raise TypeError("meta must be a dict")
        self.orders, self.rpm, self.values, self.freq_offset = orders, rpm, values, offset
        self.amplitude = str(self.amplitude)
        self.meta = copy.deepcopy(dict(self.meta))

    def __repr__(self):
        ch = "" if self.values.ndim == 2 else f", ch={self.values.shape[2]}"
        return (
            f"OrderTable(k={self.orders.size}, m={self.rpm.size}{ch}, "
            f"rpm=[{self.rpm[0]:g}, {self.rpm[-1]:g}], amplitude={self.amplitude!r}, "
            f"unit={self.unit!r})"
        )

    @property
    def magnitude(self):
        """Magnitude ``|values|`` in the table's amplitude convention and unit."""
        return np.abs(self.values)

    @property
    def n_channels(self):
        """Number of channels, or ``None`` for a table of shape (k, m)."""
        return None if self.values.ndim == 2 else int(self.values.shape[2])

    # -- construction ------------------------------------------------------

    @classmethod
    def from_slices(cls, slices, rpm_grid, *, freq_offset=None, unit="", meta=None):
        """Build a table from order slices by binning their frames on a speed grid.

        Each :class:`rota_order.slices.OrderSlice` supplies one row. A frame goes
        to the grid point nearest to its speed, provided it lies within half
        a grid spacing of it (bins are bounded by the midpoints between grid
        points; the outermost bins reach half a spacing beyond the end
        points). In a bin the magnitude is the energy average
        ``sqrt(mean(amplitude**2))`` of the slice's ``amplitude``, and the
        phase is the angle of the mean of the unit phasors of the slice's
        ``complex`` (no phase wrapping problems; NaN frames are ignored).
        Bins without frames are NaN. The slices' amplitude convention (peak
        or RMS) and unit are kept.

        The phase is only meaningful for slices of an order map taken at an
        order that falls on a map bin (see :func:`rota_order.slices.order_slice`);
        use ``phase="random"`` when synthesising from other slices.

        Parameters
        ----------
        slices : sequence of rota_order.slices.OrderSlice
            One slice per row, all with the same amplitude convention and
            channel layout.
        rpm_grid : array_like, shape (m,)
            Target speed grid in rpm, strictly increasing, at least 2 points.
        freq_offset : float or array_like, shape (k,), optional
            Fixed frequency offset of each row in Hz (default 0).
        unit : str, default ""
            Unit of the analysed signal, for information.
        meta : dict, optional
            Notes to store in the table.

        Returns
        -------
        OrderTable
            ``orders`` taken from the slices, ``values`` of shape (k, m) or
            (k, m, ch).

        References
        ----------
        Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
        Welch, P. D. (1967). The use of fast Fourier transform for the
        estimation of power spectra: a method based on time averaging over
        short, modified periodograms. IEEE Transactions on Audio and
        Electroacoustics, 15(2), 70-73.
        """
        slices = list(slices)
        if not slices:
            raise ValueError("slices is empty")
        conventions = {s.amplitude_convention for s in slices}
        if len(conventions) != 1:
            raise ValueError(f"slices mix amplitude conventions {sorted(conventions)}")
        grid = _check_grid(rpm_grid, "rpm_grid")
        edges = _bin_edges(grid)
        m = grid.size
        rows, ndims, chans = [], set(), set()
        for s in slices:
            amp = np.asarray(s.amplitude, dtype=float)
            cpx = np.asarray(s.complex, dtype=complex)
            r = np.asarray(s.rpm, dtype=float)
            if amp.ndim not in (1, 2) or amp.shape != cpx.shape or amp.shape[0] != r.size:
                raise ValueError("slice amplitude, complex and rpm have inconsistent shapes")
            ndims.add(amp.ndim)
            chans.add(1 if amp.ndim == 1 else amp.shape[1])
            idx, valid = _bin_index(r, edges)
            A = amp[:, None] if amp.ndim == 1 else amp
            Cx = cpx[:, None] if cpx.ndim == 1 else cpx
            rows.append(
                np.stack(
                    [_bin_complex(idx, valid, m, A[:, c] ** 2, Cx[:, c]) for c in range(A.shape[1])],
                    axis=-1,
                )
            )
        if len(ndims) != 1 or len(chans) != 1:
            raise ValueError("slices differ in channel layout")
        values = np.stack(rows)  # (k, m, ch)
        if ndims == {1}:
            values = values[..., 0]
        notes = {"source": "from_slices"}
        notes.update(meta or {})
        return cls(
            orders=[s.order for s in slices],
            rpm=grid,
            values=values,
            amplitude=conventions.pop(),
            freq_offset=freq_offset,
            unit=unit,
            meta=notes,
        )

    @classmethod
    def from_components(cls, comp, rpm, rpm_grid, *, unit="", meta=None):
        """Build a table from per-sample order envelopes.

        ``comp.envelopes`` are complex *peak* envelopes (the conventions of
        ``OrderComponents``, the result type of the order extractors: waveform
        ``Re{envelope exp(i 2 pi order rev)}``). The samples are binned on the
        speed grid exactly as in :meth:`from_slices`: magnitude is
        ``sqrt(mean(|envelope|**2))``, phase the angle of the mean of the unit
        phasors, empty bins are NaN. The table's convention is ``"peak"``.

        Parameters
        ----------
        comp : OrderComponents
            Extractor result with ``orders``, ``shafts`` and ``envelopes`` of
            shape (n, k) or (n, k, ch).
        rpm : array_like, shape (n,) or (n, n_shafts)
            Speed in rpm at the samples. With one column per shaft, each row
            is binned against the speed of its own shaft (``comp.shafts``) and
            the shafts are recorded in ``meta["shafts"]``; a table is still
            evaluated against a single speed, so synthesise rows of different
            shafts with their own speed curves.
        rpm_grid : array_like, shape (m,)
            Target speed grid in rpm, strictly increasing, at least 2 points.
        unit : str, default ""
            Unit of the analysed signal, for information.
        meta : dict, optional
            Notes to store in the table.

        Returns
        -------
        OrderTable
            With ``amplitude="peak"``.

        References
        ----------
        Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
        McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis
        based on a sinusoidal representation. IEEE Transactions on Acoustics,
        Speech, and Signal Processing, 34(4), 744-754.
        """
        env = np.asarray(comp.envelopes, dtype=complex)
        if env.ndim not in (2, 3):
            raise ValueError("envelopes must have shape (n, k) or (n, k, ch)")
        n, k = env.shape[:2]
        orders = np.asarray(comp.orders, dtype=float)
        shafts = np.asarray(comp.shafts, dtype=int)
        if orders.shape != (k,) or shafts.shape != (k,):
            raise ValueError("comp.orders and comp.shafts must match the envelopes")
        r = np.asarray(rpm, dtype=float)
        if r.ndim == 1:
            if np.any(shafts != 0):
                raise ValueError(
                    "the components refer to several shafts; pass rpm with one column per shaft"
                )
            r = r[:, None]
        elif r.ndim != 2:
            raise ValueError("rpm must have shape (n,) or (n, n_shafts)")
        if r.shape[0] != n:
            raise ValueError("rpm and the envelopes differ in length")
        if np.any(shafts < 0) or np.any(shafts >= r.shape[1]):
            raise ValueError("comp.shafts refers to a column that rpm does not have")
        grid = _check_grid(rpm_grid, "rpm_grid")
        edges = _bin_edges(grid)
        E = env[..., None] if env.ndim == 2 else env  # (n, k, ch)
        rows = []
        for j in range(k):
            idx, valid = _bin_index(r[:, shafts[j]], edges)
            rows.append(
                np.stack(
                    [
                        _bin_complex(idx, valid, grid.size, np.abs(E[:, j, c]) ** 2, E[:, j, c])
                        for c in range(E.shape[2])
                    ],
                    axis=-1,
                )
            )
        values = np.stack(rows)
        if env.ndim == 2:
            values = values[..., 0]
        notes = {"source": "from_components", "method": str(comp.method)}
        if np.asarray(rpm).ndim == 2:
            notes["shafts"] = shafts.tolist()
        notes.update(meta or {})
        return cls(orders=orders, rpm=grid, values=values, amplitude="peak", unit=unit, meta=notes)

    # -- evaluation --------------------------------------------------------

    def _magnitude_grid(self, smooth_rpm=None):
        mag = np.abs(self.values)
        if smooth_rpm is None:
            return mag
        smooth_rpm = float(smooth_rpm)
        if not (np.isfinite(smooth_rpm) and smooth_rpm > 0):
            raise ValueError("smooth_rpm must be a positive width in rpm")
        return _moving_average(self.rpm, mag, smooth_rpm)

    def _phase_grid(self):
        """Unwrapped phase along the grid; NaN where the value is NaN or zero."""
        v = self.values
        V = v[..., None] if v.ndim == 2 else v
        ok = np.isfinite(V) & (np.abs(V) > 0)
        out = np.full(V.shape, np.nan)
        for j in range(V.shape[0]):
            for c in range(V.shape[2]):
                sel = ok[j, :, c]
                if sel.any():
                    out[j, sel, c] = np.unwrap(np.angle(V[j, sel, c]))
        return out[..., 0] if v.ndim == 2 else out

    @staticmethod
    def _query(rpm_query):
        q = np.asarray(rpm_query, dtype=float)
        if q.ndim > 1:
            raise ValueError("rpm_query must be a scalar or a 1-D array")
        return q.ndim == 0, np.atleast_1d(q)

    def magnitude_at(self, rpm_query, *, smooth_rpm=None):
        """Magnitude of every row at the given speeds.

        The magnitudes ``|values|`` on the grid are interpolated linearly;
        grid points with NaN are skipped, and outside the grid the end values
        are held. The real and imaginary parts are *not* interpolated: where
        the phase differs between grid points that would average vectors
        pointing in different directions and collapse the magnitude by tens
        of dB (ROADMAP section 6, item 10). A row (and channel) without a
        finite grid point gives NaN.

        Parameters
        ----------
        rpm_query : float or array_like, shape (n,)
            Speeds in rpm.
        smooth_rpm : float, optional
            Width in rpm of a NaN-aware moving average applied to the
            magnitudes along the grid first (grid points within
            ``smooth_rpm / 2`` of each other are averaged), to suppress
            scatter in a measured table.

        Returns
        -------
        ndarray
            Shape (k, n) or (k, n, ch) -- ``values`` with the speed axis
            evaluated at ``rpm_query`` (shape (k,) or (k, ch) for a scalar).
            In the table's amplitude convention (peak or RMS) and unit.

        References
        ----------
        McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis
        based on a sinusoidal representation. IEEE Transactions on Acoustics,
        Speech, and Signal Processing, 34(4), 744-754.
        """
        scalar, q = self._query(rpm_query)
        out = _GridInterp(self.rpm, self._magnitude_grid(smooth_rpm)).all(q)
        return out[:, 0] if scalar else out

    def phase_at(self, rpm_query):
        """Unwrapped phase of every row at the given speeds, in rad.

        On the grid the phase is ``angle(values)`` unwrapped along the speed
        axis (``numpy.unwrap``, over the grid points that have a finite,
        non-zero value, so a jump of more than pi between neighbouring usable
        points is read as a wrap), then interpolated linearly; outside the
        grid the end values are held. The result is continuous and may exceed
        +-pi. Rows or channels with no usable point give NaN.

        Parameters
        ----------
        rpm_query : float or array_like, shape (n,)
            Speeds in rpm.

        Returns
        -------
        ndarray
            Shape (k, n) or (k, n, ch) (shape (k,) or (k, ch) for a scalar),
            in rad, relative to the carrier ``2 pi (order rev + freq_offset
            t)``.

        References
        ----------
        McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis
        based on a sinusoidal representation. IEEE Transactions on Acoustics,
        Speech, and Signal Processing, 34(4), 744-754.
        """
        scalar, q = self._query(rpm_query)
        out = _GridInterp(self.rpm, self._phase_grid()).all(q)
        return out[:, 0] if scalar else out

    # -- conversion --------------------------------------------------------

    def _converted(self, target):
        factor = 1.0 if self.amplitude == target else (_SQRT2 if target == "peak" else 1 / _SQRT2)
        return OrderTable(
            self.orders, self.rpm, self.values * factor, target, self.freq_offset, self.unit, self.meta
        )

    def to_peak(self):
        """Table in the peak convention (RMS values multiplied by sqrt(2)).

        Returns
        -------
        OrderTable
            A copy with ``amplitude="peak"``; same unit, orders, grid and
            phases.

        References
        ----------
        Brandt, A. (2011). Noise and Vibration Analysis, ch. 2. Wiley.
        """
        return self._converted("peak")

    def to_rms(self):
        """Table in the RMS convention (peak values divided by sqrt(2)).

        Returns
        -------
        OrderTable
            A copy with ``amplitude="rms"``; same unit, orders, grid and
            phases.

        References
        ----------
        Brandt, A. (2011). Noise and Vibration Analysis, ch. 2. Wiley.
        """
        return self._converted("rms")

    # -- files -------------------------------------------------------------

    @staticmethod
    def _paths(path):
        p = Path(path)
        if p.suffix in (".npz", ".json"):
            p = p.with_suffix("")
        return p.with_name(p.name + ".npz"), p.with_name(p.name + ".json")

    def save(self, path):
        """Write the table to ``<path>.npz`` and a description to ``<path>.json``.

        The NPZ holds everything needed by :meth:`load` (``orders``, ``rpm``,
        ``values``, ``freq_offset``, ``amplitude``, ``unit`` and the notes as
        the JSON string ``meta_json``); complex values and NaN are stored
        exactly. The JSON file is a human-readable sidecar (format version,
        rota-order version, amplitude convention, unit, orders, offsets, speed grid,
        shape and notes) that :meth:`load` does not need. A trailing ``.npz``
        or ``.json`` on ``path`` is ignored.

        Returns
        -------
        (Path, Path)
            The NPZ and JSON paths.

        References
        ----------
        rota-order ROADMAP, design principle 7 (reproducibility sidecar).
        """
        from . import __version__

        npz, sidecar = self._paths(path)
        meta = jsonable(self.meta, "meta")
        np.savez(
            npz,
            format_version=np.asarray(_FORMAT_VERSION),
            orders=self.orders,
            rpm=self.rpm,
            values=self.values,
            freq_offset=self.freq_offset,
            amplitude=np.asarray(self.amplitude),
            unit=np.asarray(self.unit),
            meta_json=np.asarray(json.dumps(meta, allow_nan=False)),
        )
        description = {
            "format": "rota_order.OrderTable",
            "format_version": _FORMAT_VERSION,
            "rota_order_version": __version__,
            "amplitude": self.amplitude,
            "unit": self.unit,
            "shape": list(self.values.shape),
            "orders": self.orders,
            "freq_offset": self.freq_offset,
            "rpm": self.rpm,
            "meta": meta,
        }
        sidecar.write_text(
            json.dumps(jsonable(description, "description"), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return npz, sidecar

    @classmethod
    def load(cls, path):
        """Read a table written by :meth:`save`.

        The arrays come back bit for bit (complex values, NaN included);
        ``meta`` comes back as saved, that is as plain JSON types (tuples as
        lists, NumPy values as Python numbers, non-finite floats as strings).
        A trailing ``.npz`` or ``.json`` on ``path`` is ignored; only the NPZ
        is read.

        Returns
        -------
        OrderTable

        References
        ----------
        rota-order ROADMAP, design principle 7 (reproducibility sidecar).
        """
        npz, _ = cls._paths(path)
        with np.load(npz, allow_pickle=False) as z:
            version = int(z["format_version"])
            if version != _FORMAT_VERSION:
                raise ValueError(f"unsupported OrderTable file format version {version}")
            return cls(
                orders=z["orders"],
                rpm=z["rpm"],
                values=z["values"],
                amplitude=str(z["amplitude"]),
                freq_offset=z["freq_offset"],
                unit=str(z["unit"]),
                meta=json.loads(str(z["meta_json"])),
            )


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------


def _check_fs(fs):
    fs = float(fs)
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("fs must be a positive sampling rate in Hz")
    return fs


def _check_speed(rpm):
    r = np.array(rpm, dtype=float)
    if r.ndim != 1:
        raise ValueError("rpm must be a 1-D array of speeds in rpm")
    if not np.all(np.isfinite(r)):
        raise ValueError("rpm contains non-finite values; regularise it first")
    return r


def _resolve_phase(phase, k, seed):
    """Return (mode, start phases (k,), seed actually used)."""
    used = None
    if isinstance(phase, str):
        if phase not in _PHASE_MODES:
            raise ValueError(f"phase must be one of {_PHASE_MODES} or an array, got {phase!r}")
        mode = phase
    else:
        arr = np.array(phase, dtype=float)
        if arr.shape != (k,) or not np.all(np.isfinite(arr)):
            raise ValueError(f"an array phase must be finite with shape ({k},)")
        mode = "array"
    if isinstance(seed, np.random.Generator):
        rng = seed
    elif seed is None or (isinstance(seed, (int, np.integer)) and not isinstance(seed, bool)):
        used = None if seed is None else int(seed)
        if used is None and mode == "random":
            used = int(np.random.default_rng().integers(0, 2**63 - 1))
        rng = np.random.default_rng(used)
    else:
        raise TypeError("seed must be an int, a numpy.random.Generator or None")
    if mode == "random":
        start = rng.uniform(0.0, _TWO_PI, size=k)
    elif mode == "array":
        start = arr
    else:
        start = np.zeros(k)
    return mode, start, used


class _Plan:
    """Everything the renderer needs from a table, resolved once."""

    def __init__(self, table, phase, seed, smooth_rpm):
        if not isinstance(table, OrderTable):
            raise TypeError("table must be an OrderTable")
        self.k = table.orders.size
        self.C = 1 if table.values.ndim == 2 else table.values.shape[2]
        self.mono = table.values.ndim == 2
        self.orders = table.orders.copy()
        self.offset = table.freq_offset.copy()
        scale = _SQRT2 if table.amplitude == "rms" else 1.0  # synthesis is peak
        self.amp = _GridInterp(table.rpm, table._magnitude_grid(smooth_rpm) * scale)
        self.mode, self.start, self.seed = _resolve_phase(phase, self.k, seed)
        self.phase = _GridInterp(table.rpm, table._phase_grid()) if self.mode == "table" else None
        finite = np.isfinite(table.magnitude)
        self.empty = ~finite.reshape(self.k, -1).any(axis=1)
        self.unit = table.unit
        self.table_amplitude = table.amplitude
        self.smooth_rpm = None if smooth_rpm is None else float(smooth_rpm)

    def render(self, rpm, rev, idx, fs, want_stems):
        """Render samples ``idx`` (int array) with speed ``rpm`` and shaft angle ``rev``.

        Returns the sum (n, ch), the stems (n, k, ch) or None, and the number
        of muted samples of each row.
        """
        n = rpm.size
        t = idx / fs
        nyq = 0.5 * fs
        x = np.zeros((n, self.C))
        stems = np.zeros((n, self.k, self.C)) if want_stems else None
        muted = np.zeros(self.k, dtype=np.int64)
        for j in range(self.k):
            f = self.orders[j] * rpm / 60.0 + self.offset[j]
            live = (f > 0.0) & (f < nyq)  # muted, never clipped (ROADMAP section 6, item 16)
            muted[j] = n - int(live.sum())
            if not live.any():
                continue
            a = np.nan_to_num(self.amp.row(j, rpm), nan=0.0)
            theta = (_TWO_PI * (self.orders[j] * rev + self.offset[j] * t) + self.start[j])[:, None]
            if self.phase is not None:
                theta = theta + np.nan_to_num(self.phase.row(j, rpm), nan=0.0)
            y = (a * live[:, None]) * np.cos(theta)
            x += y
            if want_stems:
                stems[:, j, :] = y
        return x, stems, muted

    def info(self, fs, n, muted):
        return {
            "fs": fs,
            "n": int(n),
            "orders": self.orders.copy(),
            "freq_offset": self.offset.copy(),
            "phase_mode": self.mode,
            "phases": self.start.copy(),
            "seed": self.seed,
            "smooth_rpm": self.smooth_rpm,
            "table_amplitude": self.table_amplitude,
            "output_amplitude": "peak",
            "unit": self.unit,
            "muted_fraction": muted / max(int(n), 1),
            "empty_rows": self.empty.copy(),
        }


def synthesize_orders(
    table, rpm, fs, *, phase="random", seed=None, smooth_rpm=None, return_stems=False
):
    """Additive synthesis of an order table along a speed curve.

    Each row ``j`` of the table becomes the sinusoid ``A_j(t) cos(theta_j(t))``
    with the continuous phase

    ``theta_j(t) = 2 pi (order_j rev(t) + freq_offset_j t) + phi_j``,

    where ``rev`` is the shaft angle from
    :func:`rota_order.speed.rpm_to_phase` (the trapezoidal integral of ``rpm / 60``;
    ``t = arange(n) / fs``, ``rev = 0`` at the first sample) and
    ``A_j(t) = magnitude_at(rpm(t))`` is converted to a *peak* amplitude (RMS
    tables are multiplied by sqrt(2)). The output is the sum over the rows.
    Between speed grid points the table's magnitude and phase are interpolated
    separately (:meth:`OrderTable.magnitude_at`, :meth:`OrderTable.phase_at`),
    never the real and imaginary parts (ROADMAP section 6, item 10).

    A component whose instantaneous frequency
    ``order_j rpm(t) / 60 + freq_offset_j`` is not inside ``(0, fs / 2)`` is
    muted (amplitude 0) at that sample, never clipped to the band edge, which
    would create a false tone (ROADMAP section 6, item 16). Note that a row
    with a negative order and no offset therefore stays silent: use the
    positive order for it (a cosine at negative frequency is the same real
    signal with reversed phase) or add a carrier offset for sidebands.

    Parameters
    ----------
    table : OrderTable
        Orders, speed grid and complex amplitudes. A table with a channel axis
        gives a multichannel output.
    rpm : array_like, shape (n,)
        Speed in rpm at each output sample, finite.
    fs : float
        Sampling rate in Hz.
    phase : {"random", "zero", "table"} or array_like, shape (k,)
        Start phases ``phi_j`` of the rows: ``"random"`` draws them uniformly
        on ``[0, 2 pi)`` (seedable), ``"zero"`` uses 0, ``"table"`` uses 0
        and adds the table's phase ``phase_at(rpm(t))`` to ``theta_j`` (use it
        only for tables with meaningful phases, for example from on-bin order
        slices), an array gives the start phases in rad directly. With
        ``"random"``, ``"zero"`` or an array the same start phase is used in
        every channel; ``"table"`` keeps each channel's own phase.
    seed : int, numpy.random.Generator or None
        Source of the random start phases. ``None`` draws a fresh seed, which
        is returned in ``info["seed"]``; the same integer seed gives
        bit-identical output.
    smooth_rpm : float, optional
        Width in rpm of a NaN-aware moving average applied to the table
        magnitudes along the speed grid before interpolation (see
        :meth:`OrderTable.magnitude_at`).
    return_stems : bool, default False
        Also return each row on its own.

    Returns
    -------
    x : ndarray, shape (n,) or (n, ch)
        The synthesised signal: a sum of peak-amplitude sinusoids, in the
        table's unit.
    stems : ndarray, shape (n, k) or (n, k, ch)
        Only if ``return_stems``; ``stems.sum(axis=1)`` equals ``x``.
    info : dict
        ``phase_mode``, ``phases`` (start phases used, rad, shape (k,)),
        ``seed``, ``muted_fraction`` (fraction of samples muted per row),
        ``empty_rows`` (rows without any finite table value, which stay
        silent), ``smooth_rpm``, ``orders``, ``freq_offset``, ``fs``, ``n``,
        ``table_amplitude``, ``output_amplitude`` (always ``"peak"``) and
        ``unit``. All entries are JSON-convertible with
        :func:`rota_order.resynth.provenance`.

    References
    ----------
    McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis based
    on a sinusoidal representation. IEEE Transactions on Acoustics, Speech,
    and Signal Processing, 34(4), 744-754.
    Brandt, A. (2011). Noise and Vibration Analysis, ch. 12. Wiley.
    """
    fs = _check_fs(fs)
    r = _check_speed(rpm)
    n = r.size
    if n == 0:
        raise ValueError("rpm is empty")
    plan = _Plan(table, phase, seed, smooth_rpm)
    rev = rpm_to_phase(r, fs)
    x = np.empty((n, plan.C))
    stems = np.empty((n, plan.k, plan.C)) if return_stems else None
    muted = np.zeros(plan.k, dtype=np.int64)
    for s in range(0, n, _CHUNK):
        sl = slice(s, min(s + _CHUNK, n))
        xc, sc, mu = plan.render(r[sl], rev[sl], np.arange(sl.start, sl.stop), fs, return_stems)
        x[sl] = xc
        if return_stems:
            stems[sl] = sc
        muted += mu
    info = plan.info(fs, n, muted)
    if plan.mono:
        x = x[:, 0]
        stems = stems[:, :, 0] if return_stems else None
    return (x, stems, info) if return_stems else (x, info)


class OrderSynth:
    """Block-wise order synthesis with state kept between blocks.

    Feeds an :class:`OrderTable` with the speed of one block at a time and
    returns one block of audio per call, for real-time programs and
    simulators that own the audio loop (rota-order does no audio I/O and no wall
    clock: the caller controls the block size). The shaft angle is integrated
    with the same trapezoidal rule as :func:`rota_order.speed.rpm_to_phase`,
    continued across block boundaries (the last speed of the previous block
    is the left end of the first interval of the next one, and the running
    sum carries on from the previous block), and the start phases are fixed
    at construction. Any split of a speed curve into blocks therefore
    reproduces :func:`synthesize_orders` on the whole curve to rounding error
    of the cosine (about 1e-15 relative to the peak amplitude, whatever the
    duration, order or block sizes).

    The table is read once at construction (magnitudes, phases and, if
    requested, the smoothed magnitudes are precomputed); later changes to the
    table object have no effect. Output follows the same conventions as
    :func:`synthesize_orders`: peak-amplitude sinusoids in the table's unit,
    components outside ``(0, fs / 2)`` muted, shape (nb,) or (nb, ch).

    Changes of the table or of the parameters while running, and the
    cross-fading they would need to avoid clicks, are out of scope: construct
    a new synthesiser (or smooth the speed and gain you feed in).

    Parameters
    ----------
    table : OrderTable
        Orders, speed grid and complex amplitudes.
    fs : float
        Sampling rate in Hz, fixed for the life of the object.
    phase, seed
        As in :func:`synthesize_orders`. The start phases are drawn once at
        construction and kept by :meth:`reset`.
    smooth_rpm : float, optional
        As in :func:`synthesize_orders`.

    Attributes
    ----------
    fs : float
        Sampling rate in Hz.
    phases : ndarray, shape (k,)
        Start phases in rad.
    n_samples : int
        Samples rendered since construction or the last :meth:`reset`.

    References
    ----------
    McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis based
    on a sinusoidal representation. IEEE Transactions on Acoustics, Speech,
    and Signal Processing, 34(4), 744-754.
    """

    def __init__(self, table, fs, *, phase="random", seed=None, smooth_rpm=None):
        self.fs = _check_fs(fs)
        self._plan = _Plan(table, phase, seed, smooth_rpm)
        self.reset()

    @property
    def phases(self):
        """Start phases of the rows in rad, shape (k,)."""
        return self._plan.start.copy()

    def reset(self):
        """Return to the initial state: first sample, shaft angle 0.

        The start phases are kept, so a run after :meth:`reset` repeats the
        first one bit for bit.

        References
        ----------
        McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis
        based on a sinusoidal representation. IEEE Transactions on Acoustics,
        Speech, and Signal Processing, 34(4), 744-754.
        """
        self.n_samples = 0
        self._last_rpm = None
        self._rev = 0.0
        self._muted = np.zeros(self._plan.k, dtype=np.int64)

    def info(self):
        """Description of the run so far, as for :func:`synthesize_orders`.

        ``muted_fraction`` and ``n`` refer to the samples rendered since the
        last :meth:`reset`. The amplitude convention of the output is peak, in
        the table's unit.

        Returns
        -------
        dict
            As the ``info`` of :func:`synthesize_orders`.

        References
        ----------
        McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis
        based on a sinusoidal representation. IEEE Transactions on Acoustics,
        Speech, and Signal Processing, 34(4), 744-754.
        """
        return self._plan.info(self.fs, self.n_samples, self._muted)

    @staticmethod
    def _block_speed(rpm, n):
        if n is not None and (int(n) != n or n < 0):
            raise ValueError("n must be a non-negative integer")
        ramp = isinstance(rpm, tuple) or (
            np.ndim(rpm) == 1 and np.size(rpm) == 2 and n is not None and n != 2
        )
        if ramp:
            if np.size(rpm) != 2 or np.ndim(rpm) != 1:
                raise ValueError("a ramp is a pair (start, end) in rpm")
            if n is None:
                raise ValueError("n is required for a (start, end) ramp")
            start, end = (float(v) for v in rpm)
            r = np.linspace(start, end, int(n), endpoint=False)
        elif np.ndim(rpm) == 0:
            if n is None:
                raise ValueError("n is required for a constant speed")
            r = np.full(int(n), float(rpm))
        else:
            r = np.array(rpm, dtype=float)
            if r.ndim != 1:
                raise ValueError("rpm must be a scalar, a (start, end) pair or a 1-D array")
            if n is not None and r.size != n:
                raise ValueError(f"rpm has {r.size} samples but n={n}")
        if not np.all(np.isfinite(r)):
            raise ValueError("rpm contains non-finite values")
        return r

    def process(self, rpm, n=None):
        """Render the next block.

        Parameters
        ----------
        rpm : float, (float, float) or array_like, shape (nb,)
            Speed in rpm for the block: a scalar (constant over the block;
            needs ``n``), a tuple ``(start, end)`` (linear ramp over the
            block, ``numpy.linspace(start, end, n, endpoint=False)``, so
            ``end`` is the speed of the first sample of the next block and
            chained blocks give a continuous schedule; needs ``n``), or one
            speed per sample. A length-2 list or array counts as a ramp when
            ``n`` is given and differs from 2; use a tuple to force a ramp.
        n : int, optional
            Block length in samples; required for scalar and ramp input, and
            checked against the length of a per-sample array.

        Returns
        -------
        ndarray, shape (nb,) or (nb, ch)
            The block: peak-amplitude sinusoids in the table's unit.

        References
        ----------
        McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis
        based on a sinusoidal representation. IEEE Transactions on Acoustics,
        Speech, and Signal Processing, 34(4), 744-754.
        """
        plan = self._plan
        r = self._block_speed(rpm, n)
        nb = r.size
        if nb == 0:
            return np.zeros((0,) if plan.mono else (0, plan.C))
        idx = self.n_samples + np.arange(nb)
        # Trapezoidal rule of rota_order.speed.rpm_to_phase, with the same operations in the
        # same order, accumulated by a running sum that continues the previous block's.
        # A sequential cumulative sum is independent of where the blocks are cut, so the
        # shaft angle equals that of the whole speed curve to the last bit.
        if self._last_rpm is None:
            joined, first = r, np.arange(nb) / self.fs
            carry = 0.0
        else:
            joined = np.concatenate(([self._last_rpm], r))
            first = np.arange(self.n_samples - 1, self.n_samples + nb) / self.fs
            carry = self._rev
        inc = 0.5 * (joined[1:] + joined[:-1]) / 60.0 * np.diff(first)
        rev = np.cumsum(np.concatenate(([carry], inc)))
        if self._last_rpm is not None:
            rev = rev[1:]
        x, _, muted = plan.render(r, rev, idx, self.fs, False)
        self._muted += muted
        self.n_samples += nb
        self._last_rpm = float(r[-1])
        self._rev = float(rev[-1])
        return x[:, 0] if plan.mono else x
