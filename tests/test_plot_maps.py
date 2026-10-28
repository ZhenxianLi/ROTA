"""Tests of the map figures of rota.plot against analytic truth (rota.synth supplies the signals).

Needs Matplotlib (Agg backend, no pixel comparison); skipped without it. The
tests read the artists back from the Axes: the QuadMesh holds the drawn levels
and cell edges, the Line2D objects the order lines, the Annotations their labels.
"""

import dataclasses
import subprocess
import sys
import warnings

import numpy as np
import pytest

import rota.plot as rp
from rota.maps import Map, bin_map, rpm_frequency_map, rpm_order_map
from rota.plot._core import edges_from_centres, speed_envelope, to_db
from rota.slices import order_slice
from rota.speed import rpm_from_ridge, tacho_to_rpm
from rota.synth import rpm_profiles, synth_runup, synth_scenario

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

MICRO = "\u00b5"  # micro sign U+00B5
FS = 4000.0
DUR = 3.0
R0, R1 = 1200.0, 3000.0  # run-up, rpm
A_ORDER2 = 1.5  # peak amplitudes in Pa
A_ORDER4 = 0.4
A_TONE = 2.0  # fixed tone at 1000 Hz, on a bin of the 10 Hz frequency map
P_REF = 2e-5


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------
# Signals, maps and readers of the drawn artists
# --------------------------------------------------------------------------


def _t():
    return np.arange(round(DUR * FS)) / FS


def _runup_rpm():
    return rpm_profiles(_t(), "linear", start=R0, stop=R1)


def _signal(rpm, a4=A_ORDER4, tone=A_TONE):
    """Orders 2 and 4 (on bins of the order map) and a fixed 1000 Hz tone; no noise."""
    orders = [
        {"order": 2.0, "amplitude": A_ORDER2, "phase": 0.3},
        {"order": 4.0, "amplitude": a4, "phase": 1.1},
    ]
    tones = [{"freq": 1000.0, "amplitude": tone, "phase": 0.5}] if tone else ()
    return synth_runup(FS, rpm, orders, tones=tones, noise_rms=0.0, seed=1)


def _omap(x, rpm, amplitude="rms", order_resolution=0.25, **kw):
    return rpm_order_map(x, FS, rpm, order_resolution=order_resolution, max_order=8.0, amplitude=amplitude, **kw)


def _fmap(x, rpm, amplitude="rms", **kw):
    return rpm_frequency_map(x, FS, rpm, nperseg=400, amplitude=amplitude, max_freq=1500.0, **kw)


@pytest.fixture(scope="module")
def runup():
    return _signal(_runup_rpm())


@pytest.fixture(scope="module")
def omap(runup):
    return _omap(runup.x, runup.rpm)


@pytest.fixture(scope="module")
def fmap(runup):
    return _fmap(runup.x, runup.rpm)


@pytest.fixture(scope="module")
def updown():
    """1200 -> 3000 -> 1500 rpm: not monotonic."""
    rpm = rpm_profiles(_t(), "piecewise", knots=([0.0, 1.5, 3.0], [1200.0, 3000.0, 1500.0]))
    return _signal(rpm)


def _top(m, limit):
    """Top edge of the last bin kept by ``fmax``/``max_order`` = limit: the bin at the limit is whole."""
    n = int(np.count_nonzero(m.axis <= limit + 1e-9))
    return float(m.axis[n - 1] + 0.5 * m.resolution)


def _reversals(v):
    """Largest reversal of a run-up speed as a fraction of its range, and the number of falling steps."""
    v = np.asarray(v, dtype=float)
    return float(np.max(np.maximum.accumulate(v) - v) / (v.max() - v.min())), int(np.sum(np.diff(v) < 0))


@pytest.fixture(scope="module")
def tacho_speed(runup):
    """Speed of a linear run-up from a jittery tachometer, as a function of the frame times of a map.

    The jitter (60 us on every pulse) leaves reversals of about 1 % of the speed range.
    """
    sig = synth_runup(
        FS, _runup_rpm(), [{"order": 2.0, "amplitude": 1.0}], noise_rms=0.0, seed=1, tacho_ppr=4, tacho_jitter=6e-5
    )
    rpm = tacho_to_rpm(sig.tacho, FS, 4)
    return lambda frames: np.interp(frames, _t(), rpm)


@pytest.fixture(scope="module")
def ridge_speed(runup):
    """Speed track of rpm_from_ridge on a noisy run-up, as a function of the frame times of a map."""
    sig = synth_runup(
        FS,
        _runup_rpm(),
        [{"order": 2.0, "amplitude": 1.0}, {"order": 4.0, "amplitude": 0.5}],
        noise_rms=6.0,
        seed=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # quality warnings are not under test
        ridge = rpm_from_ridge(sig.x, FS, (600.0, 6000.0), orders=(2, 4))
    return lambda frames: np.interp(frames, ridge.t, ridge.rpm)


@pytest.fixture(scope="module")
def gear_shifts():
    """An ICE run-up with two upshifts (the speed drops by about 30 %), as engine speed per sample."""
    return synth_scenario("ice_runup", fs=8000.0, duration=18.0, seed=1, tacho={"ppr": 1})


@pytest.fixture(params=["order", "frequency"])
def case(request, omap, fmap):
    """(figure function, map) for both domains."""
    return (rp.rpm_order_map, omap) if request.param == "order" else (rp.rpm_frequency_map, fmap)


def _mesh(ax):
    meshes = [c for c in ax.collections if type(c).__name__ == "QuadMesh"]
    assert len(meshes) == 1
    return meshes[0]


def _levels(mesh, shape):
    """Drawn levels in dB as (n_rows, n_columns), NaN where the cell is masked."""
    return np.ma.asarray(mesh.get_array()).astype(float).filled(np.nan).reshape(shape)


def _gaps(mesh, shape):
    return np.ma.getmaskarray(mesh.get_array()).reshape(shape)


def _edges(mesh):
    xy = mesh.get_coordinates()  # (n_rows + 1, n_columns + 1, 2)
    return xy[0, :, 0], xy[:, 0, 1]


def _row(m, order):
    return int(np.argmin(np.abs(m.axis - order)))


def _colourbar_label(mesh):
    return mesh.colorbar.ax.get_ylabel()


def _same_levels(z, expected):
    """The drawn levels are the levels of the map values on the same cells, gaps included.

    Compared to rounding (rtol 1e-12), not bit for bit: NumPy 1.x computes the magnitude of
    complex data with SIMD loops that can differ by one ulp between calls on strided data.
    """
    np.testing.assert_allclose(z, expected, rtol=1e-12, atol=1e-12)  # NaN gaps must coincide


def _raw_rcparams():
    """The stored rcParams, read without the hooks of RcParams.__getitem__ (which resolve the backend)."""
    return dict(dict.items(matplotlib.rcParams))


# --------------------------------------------------------------------------
# What is drawn: the cells and levels of the map, unchanged
# --------------------------------------------------------------------------


def test_the_mesh_holds_the_levels_of_the_map_values_on_the_cells_of_the_map(case):
    draw, m = case
    ax = draw(m, **rp.SOUND_PRESSURE)
    mesh = _mesh(ax)
    z = _levels(mesh, m.values.shape)
    _same_levels(z, to_db(np.abs(m.values), P_REF))  # no resampling, no smoothing: the same cells and values
    _same_levels(z, 20.0 * np.log10(np.abs(m.values) / P_REF))
    assert z.shape == m.values.shape and not np.isnan(z).any()
    assert not _gaps(mesh, m.values.shape).any()
    x_edges, y_edges = _edges(mesh)
    assert x_edges.size == m.values.shape[1] + 1 and y_edges.size == m.values.shape[0] + 1
    # frames on the frame speed, bins around their centres (the DC cell starts at 0)
    np.testing.assert_allclose(x_edges, edges_from_centres(m.rpm))
    np.testing.assert_allclose(y_edges[1:-1], 0.5 * (m.axis[:-1] + m.axis[1:]))
    assert y_edges[0] == 0.0 and y_edges[-1] == pytest.approx(m.axis[-1] + 0.5 * m.resolution)
    assert np.all(x_edges[:-1] < m.rpm) and np.all(m.rpm < x_edges[1:])  # each frame inside its column
    assert mesh.get_cmap().name == "viridis"
    assert ax.get_xlim() == pytest.approx((x_edges[0], x_edges[-1]))
    assert ax.get_ylim() == pytest.approx((0.0, y_edges[-1]))


@pytest.mark.parametrize("amplitude, scale", [("rms", 1.0 / np.sqrt(2.0)), ("peak", 1.0)])
def test_order_map_shows_a_known_order_at_its_analytic_level(runup, amplitude, scale):
    m = _omap(runup.x, runup.rpm, amplitude)
    ax = rp.rpm_order_map(m, **rp.SOUND_PRESSURE)
    mesh = _mesh(ax)
    z = _levels(mesh, m.values.shape)
    k2, k4 = _row(m, 2.0), _row(m, 4.0)
    assert m.axis[k2] == pytest.approx(2.0) and m.axis[k4] == pytest.approx(4.0)  # on bins: no scalloping
    exp2 = 20.0 * np.log10(A_ORDER2 * scale / P_REF)
    exp4 = 20.0 * np.log10(A_ORDER4 * scale / P_REF)
    np.testing.assert_allclose(z[k2], exp2, atol=0.01)  # every frame, edge frames included
    np.testing.assert_allclose(z[k4], exp4, atol=0.01)
    # the strongest cell is order 2 and the colour scale ends there
    assert np.nanmax(z) == pytest.approx(exp2, abs=0.01)
    assert mesh.get_clim()[1] == np.nanmax(z)
    # the level by order_slice (band power, no scalloping) is the same
    s = order_slice(m, 2.0)
    np.testing.assert_allclose(to_db(s.amplitude[3:-3], P_REF), exp2, atol=0.01)
    assert f"{amplitude.upper() if amplitude == 'rms' else amplitude}" in _colourbar_label(mesh)


@pytest.mark.parametrize("amplitude, scale", [("rms", 1.0 / np.sqrt(2.0)), ("peak", 1.0)])
def test_campbell_diagram_shows_a_known_tone_at_its_analytic_level(runup, amplitude, scale):
    m = _fmap(runup.x, runup.rpm, amplitude)
    ax = rp.rpm_frequency_map(m, ref=1.0)  # dB re 1 (Pa)
    mesh = _mesh(ax)
    z = _levels(mesh, m.values.shape)
    kt = _row(m, 1000.0)
    assert m.axis[kt] == pytest.approx(1000.0)
    exp_tone = 20.0 * np.log10(A_TONE * scale)
    np.testing.assert_allclose(z[kt], exp_tone, atol=1e-3)  # on a bin
    assert np.nanmax(z) == pytest.approx(exp_tone, abs=1e-3) and mesh.get_clim()[1] == np.nanmax(z)
    # An order sweeping through the bins: the bin nearest to the line reads the truth minus the
    # scalloping of the Hann window (at most 1.42 dB), never more than the truth.
    exp2 = 20.0 * np.log10(A_ORDER2 * scale)
    nearest = [_row(m, 2.0 * r / 60.0) for r in m.rpm]
    d = z[nearest, np.arange(m.rpm.size)] - exp2
    assert d.max() < 0.05 and d.min() > -1.5
    s = order_slice(m, 2.0)  # the band level has no scalloping
    np.testing.assert_allclose(20.0 * np.log10(s.amplitude[3:-3]), exp2, atol=0.02)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


@pytest.mark.parametrize("amplitude, word", [("rms", "RMS"), ("peak", "peak")])
def test_colour_bar_label_names_reference_unit_and_convention(runup, amplitude, word):
    m = _omap(runup.x, runup.rpm, amplitude)
    mesh = _mesh(rp.rpm_order_map(m, **rp.SOUND_PRESSURE))
    assert _colourbar_label(mesh) == f"Level (dB re 20 {MICRO}Pa, {word})"
    mesh = _mesh(rp.rpm_order_map(m))  # the reference is explicit: the default is 1, without a unit
    assert _colourbar_label(mesh) == f"Level (dB re 1, {word})"
    mesh = _mesh(rp.rpm_order_map(m, ref=1e-6, unit="m/s\u00b2"))
    assert _colourbar_label(mesh) == f"Level (dB re 1 {MICRO}m/s\u00b2, {word})"
    mesh = _mesh(rp.rpm_frequency_map(_fmap(runup.x, runup.rpm, amplitude), **rp.SOUND_PRESSURE))
    assert _colourbar_label(mesh) == f"Level (dB re 20 {MICRO}Pa, {word})"


def test_axis_labels_of_the_maps(omap, fmap, updown):
    ax = rp.rpm_order_map(omap)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Speed (rpm)", "Order")
    ax = rp.rpm_frequency_map(fmap)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Speed (rpm)", "Frequency (Hz)")
    ax = rp.rpm_order_map(_omap(updown.x, updown.rpm))
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Time (s)", "Order")
    ax = rp.rpm_frequency_map(_fmap(updown.x, updown.rpm))
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Time (s)", "Frequency (Hz)")


def test_colour_bar_can_be_left_out(case):
    draw, m = case
    ax = draw(m, colorbar=False)
    assert len(ax.figure.axes) == 1 and _mesh(ax).colorbar is None
    ax = draw(m)
    assert len(ax.figure.axes) == 2


# --------------------------------------------------------------------------
# Colour range
# --------------------------------------------------------------------------


def test_colour_limits_span_the_dynamic_range_below_the_maximum_shown(case):
    draw, m = case
    ax = draw(m)
    mesh = _mesh(ax)
    top = float(np.nanmax(_levels(mesh, m.values.shape)))
    assert mesh.get_clim() == pytest.approx((top - 40.0, top))  # the default range is 40 dB
    mesh = _mesh(draw(m, dynamic_range=25.0))
    assert mesh.get_clim() == pytest.approx((top - 25.0, top))
    mesh = _mesh(draw(m, vmin=0.0))
    assert mesh.get_clim() == pytest.approx((0.0, top))
    mesh = _mesh(draw(m, vmax=top + 10.0))
    assert mesh.get_clim() == pytest.approx((top - 30.0, top + 10.0))
    mesh = _mesh(draw(m, vmin=-30.0, vmax=70.0, dynamic_range=5.0))  # explicit limits ignore the range
    assert mesh.get_clim() == (-30.0, 70.0)


def test_the_colour_scale_covers_only_what_is_shown(fmap):
    full = _mesh(rp.rpm_frequency_map(fmap))
    ax = rp.rpm_frequency_map(fmap, fmax=500.0)  # the 1000 Hz tone is cropped away
    mesh = _mesh(ax)
    n = int(np.count_nonzero(fmap.axis <= 500.0))
    z = _levels(mesh, (n, fmap.values.shape[1]))
    _same_levels(z, to_db(fmap.values[:n], 1.0))
    assert ax.get_ylim() == pytest.approx((0.0, _top(fmap, 500.0)))  # the last bin is whole
    assert mesh.get_clim()[1] == np.nanmax(z) and mesh.get_clim()[1] < full.get_clim()[1] - 1.0


def test_max_order_crops_the_order_map(omap):
    ax = rp.rpm_order_map(omap, max_order=6.0)
    n = int(np.count_nonzero(omap.axis <= 6.0 + 1e-9))
    z = _levels(_mesh(ax), (n, omap.values.shape[1]))
    _same_levels(z, to_db(omap.values[:n], 1.0))
    assert ax.get_ylim() == pytest.approx((0.0, _top(omap, 6.0)))  # the bin at 6 is whole
    ax = rp.rpm_order_map(omap, max_order=1e6)  # beyond the axis: everything, to the top edge of the last bin
    assert ax.get_ylim() == pytest.approx((0.0, omap.axis[-1] + 0.5 * omap.resolution))
    assert _levels(_mesh(ax), omap.values.shape).shape == omap.values.shape


def test_the_bin_at_the_limit_is_shown_whole_and_its_mark_label_is_drawn(omap, fmap):
    # max_order on a bin centre: the axis ends at the top edge of that bin, not in the middle of it
    ax = rp.rpm_order_map(omap, max_order=6.0, mark_orders=(2, 6))
    top = ax.get_ylim()[1]
    assert top == pytest.approx(6.0 + 0.5 * omap.resolution) and top > 6.0
    _x_edges, y_edges = _edges(_mesh(ax))
    assert y_edges[-1] == pytest.approx(top)  # the order-6 row is drawn up to its upper edge
    assert [t.get_text() for t in ax.texts] == ["2", "6"]  # the label of the order at the limit is there
    assert len(ax.lines) == 2
    label6 = next(t for t in ax.texts if t.get_text() == "6")
    assert label6.xy[1] == pytest.approx(6.0)  # anchored on its line; below it, as the top strip is too thin
    assert label6.get_va() == "top" and next(t for t in ax.texts if t.get_text() == "2").get_va() == "bottom"
    # the same for fmax on a bin centre of a frequency map
    k = int(np.argmin(np.abs(fmap.axis - 500.0)))
    assert fmap.axis[k] == pytest.approx(500.0)
    ax = rp.rpm_frequency_map(fmap, fmax=500.0)
    assert ax.get_ylim()[1] == pytest.approx(500.0 + 0.5 * fmap.resolution)
    # a limit between two centres keeps the bins whose centre is at or below it, whole
    ax = rp.rpm_order_map(omap, max_order=6.1)
    assert ax.get_ylim()[1] == pytest.approx(6.0 + 0.5 * omap.resolution)


def test_a_limit_below_the_lowest_bin_raises(fmap):
    cut = dataclasses.replace(fmap, values=fmap.values[10:], axis=fmap.axis[10:])
    with pytest.raises(ValueError, match="below the lowest bin"):
        rp.rpm_frequency_map(cut, fmax=5.0)
    assert plt.get_fignums() == []


def test_zero_amplitudes_are_gaps_and_raise_no_warning(omap):
    values = omap.values.copy()
    values[:, :5] = 0.0
    values[3, 10] = 0.0
    m = dataclasses.replace(omap, values=values)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ax = rp.rpm_order_map(m, **rp.SOUND_PRESSURE)
    mesh = _mesh(ax)
    z = _levels(mesh, m.values.shape)
    gaps = _gaps(mesh, m.values.shape)
    assert np.isnan(z[:, :5]).all() and np.isnan(z[3, 10])
    assert gaps[:, :5].all() and gaps[3, 10] and gaps.sum() == 5 * m.values.shape[0] + 1  # drawn as gaps
    _same_levels(z[:, 5:][~gaps[:, 5:]], to_db(np.abs(m.values[:, 5:]), P_REF)[~gaps[:, 5:]])
    assert mesh.get_clim()[1] == np.nanmax(z)  # the colour scale ignores the gaps


def test_a_map_without_a_level_raises_unless_vmax_is_given(omap):
    m = dataclasses.replace(omap, values=np.zeros_like(omap.values))
    with pytest.raises(ValueError, match="no finite level"):
        rp.rpm_order_map(m)
    assert plt.get_fignums() == []
    ax = rp.rpm_order_map(m, vmax=0.0)  # explicit limits draw the empty image
    mesh = _mesh(ax)
    assert mesh.get_clim() == (-40.0, 0.0) and _gaps(mesh, m.values.shape).all()


# --------------------------------------------------------------------------
# The x axis: speed or time
# --------------------------------------------------------------------------


def test_x_auto_draws_a_run_up_against_speed(case):
    draw, m = case
    assert np.all(np.diff(m.rpm) > 0)
    ax = draw(m)
    assert ax.get_xlabel() == "Speed (rpm)"
    x_edges, _ = _edges(_mesh(ax))
    np.testing.assert_allclose(x_edges, edges_from_centres(m.rpm))


def test_x_auto_draws_an_up_and_down_speed_against_time(updown):
    for m, draw in ((_omap(updown.x, updown.rpm), rp.rpm_order_map), (_fmap(updown.x, updown.rpm), rp.rpm_frequency_map)):
        assert not (np.all(np.diff(m.rpm) > 0) or np.all(np.diff(m.rpm) < 0))
        ax = draw(m)
        assert ax.get_xlabel() == "Time (s)"
        x_edges, _ = _edges(_mesh(ax))
        np.testing.assert_allclose(x_edges, edges_from_centres(m.t))
        assert np.all(np.diff(x_edges) > 0)
        assert ax.get_xlim() == pytest.approx((x_edges[0], x_edges[-1]))


def test_x_rpm_with_a_non_monotonic_speed_raises_and_suggests_binning(updown):
    for m, draw in ((_omap(updown.x, updown.rpm), rp.rpm_order_map), (_fmap(updown.x, updown.rpm), rp.rpm_frequency_map)):
        with pytest.raises(ValueError, match="bin_map") as err:
            draw(m, x="rpm")
        assert "x='time'" in str(err.value)
    assert plt.get_fignums() == []


def test_x_time_is_available_for_a_run_up(case):
    draw, m = case
    ax = draw(m, x="time")
    assert ax.get_xlabel() == "Time (s)"
    x_edges, _ = _edges(_mesh(ax))
    np.testing.assert_allclose(x_edges, edges_from_centres(m.t))


def test_a_coast_down_is_drawn_against_speed_with_the_axis_in_the_usual_direction():
    rpm = rpm_profiles(_t(), "linear", start=R1, stop=R0)
    sig = _signal(rpm)
    m = _omap(sig.x, sig.rpm)
    assert np.all(np.diff(m.rpm) < 0)
    ax = rp.rpm_order_map(m)
    assert ax.get_xlabel() == "Speed (rpm)"
    x_edges, _ = _edges(_mesh(ax))
    assert np.all(np.diff(x_edges) < 0)  # frames run from high to low speed ...
    xlim = ax.get_xlim()
    assert xlim[0] < xlim[1] and xlim == pytest.approx((x_edges.min(), x_edges.max()))  # ... the axis does not flip
    z = _levels(_mesh(ax), m.values.shape)
    _same_levels(z, to_db(m.values, 1.0))  # column j is still frame j


def test_a_speed_hold_before_the_run_up_is_a_run_up_with_cells_of_zero_width():
    rpm = rpm_profiles(_t(), "linear", start=R0, stop=R1, t_start=0.6, t_stop=DUR)
    sig = _signal(rpm)
    m = _omap(sig.x, sig.rpm)
    hold = np.diff(m.rpm) == 0.0
    assert hold.any()  # an idle hold: equal frame speeds, no reversal
    ax = rp.rpm_order_map(m)
    assert ax.get_xlabel() == "Speed (rpm)"
    x_edges, _ = _edges(_mesh(ax))
    assert np.all(np.diff(x_edges) >= 0)  # monotone; the frames of the hold share one speed and have no width
    assert np.sum(np.diff(x_edges) == 0.0) >= hold.sum() - 1
    assert _mesh(rp.rpm_order_map(m, x="rpm")).get_array().size == m.values.size
    assert rp.rpm_order_map(m, x="time").get_xlabel() == "Time (s)"  # the way to see the hold


def _assert_envelope_cells(mesh, rpm, increasing=True, tol=0.02):
    """The x edges of a mesh are monotone, tile the envelope of the speed and hold every frame's envelope value."""
    x_edges, _ = _edges(mesh)
    env = speed_envelope(rpm, tol)
    assert env is not None
    np.testing.assert_allclose(x_edges, edges_from_centres(env), rtol=0, atol=1e-9)
    d = np.diff(x_edges)
    assert np.all(d >= 0) if increasing else np.all(d <= 0)  # the cells never overlap
    lo, hi = np.minimum(x_edges[:-1], x_edges[1:]), np.maximum(x_edges[:-1], x_edges[1:])
    assert np.all((lo <= env) & (env <= hi))  # each frame stays in its own cell
    assert np.all(np.abs(env - rpm) <= tol * (rpm.max() - rpm.min()) * (1 + 1e-12))  # displaced by <= the reversal
    return x_edges


def _with_speed(m, speed):
    """The map with the measured speed at its frames."""
    return dataclasses.replace(m, rpm=speed(m.t))


@pytest.mark.parametrize("source", ["tacho_speed", "ridge_speed"])
def test_a_measured_speed_with_small_reversals_is_an_rpm_axis_for_maps(request, runup, source):
    speed = request.getfixturevalue(source)
    for draw, m in (
        (rp.rpm_order_map, _with_speed(_omap(runup.x, runup.rpm), speed)),
        (rp.rpm_frequency_map, _with_speed(_fmap(runup.x, runup.rpm), speed)),
    ):
        frac, n_falling = _reversals(m.rpm)
        assert n_falling > 0 and 0.0 < frac < 0.02  # real reversals, below the 2 % rule
        assert not np.all(np.diff(m.rpm) > 0)  # not strictly monotonic
        for x in ("auto", "rpm"):
            ax = draw(m, x=x)
            assert ax.get_xlabel() == "Speed (rpm)"
            mesh = _mesh(ax)
            x_edges = _assert_envelope_cells(mesh, m.rpm)
            assert ax.get_xlim() == pytest.approx((x_edges[0], x_edges[-1]))
            _same_levels(_levels(mesh, m.values.shape), to_db(m.values, 1.0))  # column j is still frame j


def test_a_measured_coast_down_with_small_reversals_is_an_rpm_axis(tacho_speed, runup):
    m = _omap(runup.x, runup.rpm)
    rpm = tacho_speed(m.t)[::-1].copy()  # frames in the order of a coast-down
    m = dataclasses.replace(m, rpm=rpm)
    assert not np.all(np.diff(rpm) < 0) and _reversals(rpm[::-1])[0] < 0.02
    ax = rp.rpm_order_map(m)
    assert ax.get_xlabel() == "Speed (rpm)"
    x_edges = _assert_envelope_cells(_mesh(ax), rpm, increasing=False)
    xlim = ax.get_xlim()
    assert xlim[0] < xlim[1] and xlim == pytest.approx((x_edges.min(), x_edges.max()))


def test_the_displacement_of_a_frame_is_bounded_by_the_reversal_and_monotonic_tol(tacho_speed, runup):
    m = _with_speed(_omap(runup.x, runup.rpm), tacho_speed)
    frac, _ = _reversals(m.rpm)
    env = speed_envelope(m.rpm)
    assert 0.0 < np.max(env - m.rpm) <= frac * (m.rpm.max() - m.rpm.min()) * (1 + 1e-12)
    # the bound is a tolerance: a tighter one refuses the speed, a looser one still accepts it
    assert rp.rpm_order_map(m, monotonic_tol=frac * 1.01).get_xlabel() == "Speed (rpm)"
    assert rp.rpm_order_map(m, monotonic_tol=frac * 0.99).get_xlabel() == "Time (s)"  # auto falls back
    with pytest.raises(ValueError, match="essentially monotonic.*bin_map"):
        rp.rpm_order_map(m, x="rpm", monotonic_tol=frac * 0.99)
    assert rp.rpm_frequency_map(_with_speed(_fmap(runup.x, runup.rpm), tacho_speed), monotonic_tol=0.5)
    assert rp.rpm_order_map(m, monotonic_tol=0.0).get_xlabel() == "Time (s)"  # tol 0: the strict rule
    sorted_m = dataclasses.replace(m, rpm=np.sort(m.rpm))
    assert rp.rpm_order_map(sorted_m, monotonic_tol=0.0).get_xlabel() == "Speed (rpm)"


@pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf, "x"])
def test_an_invalid_monotonic_tol_raises_before_a_figure_is_made(omap, fmap, bad):
    for func, m in ((rp.rpm_order_map, omap), (rp.rpm_frequency_map, fmap)):
        with pytest.raises(ValueError, match="monotonic_tol"):
            func(m, monotonic_tol=bad)
    with pytest.raises(ValueError, match="monotonic_tol"):
        rp.compare_maps({"a": omap}, monotonic_tol=bad)
    assert plt.get_fignums() == []


def test_a_gear_shift_run_up_and_an_up_and_down_speed_are_not_for_auto_or_maps(gear_shifts, updown):
    m = rpm_order_map(gear_shifts.x, 8000.0, gear_shifts.rpm, order_resolution=0.5, max_order=6.0, amplitude="rms")
    assert _reversals(m.rpm)[0] > 0.2  # the upshifts: drops of about 30 %
    assert speed_envelope(m.rpm) is None
    others = [(rp.rpm_order_map, m), (rp.rpm_order_map, _omap(updown.x, updown.rpm)),
              (rp.rpm_frequency_map, _fmap(updown.x, updown.rpm))]
    for draw, mm in others:
        ax = draw(mm)
        assert ax.get_xlabel() == "Time (s)"  # auto
        x_edges, _ = _edges(_mesh(ax))
        np.testing.assert_allclose(x_edges, edges_from_centres(mm.t))
        assert np.all(np.diff(x_edges) > 0)
        with pytest.raises(ValueError, match="essentially monotonic") as err:
            draw(mm, x="rpm")
        assert "bin_map" in str(err.value) and "x='time'" in str(err.value)
        with pytest.raises(ValueError, match="essentially monotonic"):
            rp.compare_maps({"a": mm, "b": mm}, x="rpm")
        assert rp.compare_maps({"a": mm}, x="auto")[1][0].get_xlabel() == "Time (s)"
    # even a generous tolerance does not make a 30 % drop a reversal
    assert rp.rpm_order_map(m, monotonic_tol=0.1).get_xlabel() == "Time (s)"
    assert len(plt.get_fignums()) == 3 + 3 + 1  # one figure per call that succeeded


def test_compare_maps_of_maps_on_a_measured_speed_share_the_envelope_cells(tacho_speed, runup):
    a = _with_speed(_omap(runup.x, runup.rpm), tacho_speed)
    b = dataclasses.replace(a, values=0.5 * a.values)
    _fig, axes = rp.compare_maps({"before": a, "after": b}, difference=True)
    assert [ax.get_xlabel() for ax in axes] == ["Speed (rpm)"] * 3
    for ax in axes:
        _assert_envelope_cells(_mesh(ax), a.rpm)
    z = _levels(_mesh(axes[2]), a.values.shape)
    np.testing.assert_allclose(z[np.isfinite(z)], 20.0 * np.log10(0.5), atol=1e-6)


def test_a_map_of_a_single_frame_cannot_be_an_image(omap):
    one = dataclasses.replace(
        omap, values=omap.values[:, :1], t=omap.t[:1], rpm=omap.rpm[:1], rpm_min=omap.rpm_min[:1], rpm_max=omap.rpm_max[:1]
    )
    with pytest.raises(ValueError, match="at least two frames"):
        rp.rpm_order_map(one)
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------
# Binned maps
# --------------------------------------------------------------------------


def test_binned_map_is_drawn_on_its_rpm_bin_edges(case):
    draw, m = case
    b = bin_map(m, 250.0)
    assert b.values.shape[1] < m.values.shape[1]
    ax = draw(b, **rp.SOUND_PRESSURE)
    mesh = _mesh(ax)
    x_edges, _ = _edges(mesh)
    np.testing.assert_array_equal(x_edges, np.append(b.rpm_lo, b.rpm_hi[-1]))  # the speed bins, not the frames
    np.testing.assert_array_equal(b.rpm_hi[:-1], b.rpm_lo[1:])  # adjacent bins here
    assert ax.get_xlabel() == "Speed (rpm)"
    assert ax.get_xlim() == (b.rpm_lo[0], b.rpm_hi[-1])
    z = _levels(mesh, b.values.shape)
    _same_levels(z, to_db(np.abs(b.values), P_REF))
    assert _colourbar_label(mesh) == f"Level (dB re 20 {MICRO}Pa, RMS)"
    assert _mesh(draw(b, x="rpm")).get_array().shape == mesh.get_array().shape  # x="rpm" is the same
    with pytest.raises(ValueError, match="speed bins"):
        draw(b, x="time")


def test_binned_map_is_drawn_against_speed_whatever_the_speed_profile(updown):
    m = _fmap(updown.x, updown.rpm, rpm_bins=250.0, segment="runup")
    assert rp.rpm_frequency_map(m).get_xlabel() == "Speed (rpm)"  # also where the frames are not monotonic


def test_binned_map_leaves_a_gap_where_no_frame_fell():
    # speed jumps from 1800 to 2700 rpm within 0.1 ms: the bins in between receive no frame
    rpm = rpm_profiles(_t(), "piecewise", knots=([0.0, 1.0, 1.0001, 3.0], [1200.0, 1800.0, 2700.0, 3000.0]))
    sig = _signal(rpm, tone=None)
    m = _fmap(sig.x, sig.rpm)
    b = bin_map(m, 100.0)
    edges = np.unique(np.concatenate([b.rpm_lo, b.rpm_hi]))
    assert edges.size - 1 > b.values.shape[1]  # bin_map dropped the empty bins
    ax = rp.rpm_frequency_map(b, **rp.SOUND_PRESSURE)
    mesh = _mesh(ax)
    x_edges, _ = _edges(mesh)
    np.testing.assert_array_equal(x_edges, edges)  # one contiguous set of cells
    shape = (b.values.shape[0], edges.size - 1)
    z, gaps = _levels(mesh, shape), _gaps(mesh, shape)
    populated = np.searchsorted(edges, b.rpm_lo)
    empty = np.setdiff1d(np.arange(edges.size - 1), populated)
    assert empty.size and gaps[:, empty].all()  # the stretch without a bin is a gap, not an interpolation
    assert not gaps[:, populated].any()
    _same_levels(z[:, populated], to_db(np.abs(b.values), P_REF))


def test_binned_map_without_a_populated_bin_cannot_be_drawn(fmap):
    empty = bin_map(fmap, [10000.0, 11000.0])  # no frame falls in the only bin
    assert empty.values.shape[1] == 0
    with pytest.raises(ValueError, match="no frames"):
        rp.rpm_frequency_map(empty)
    assert plt.get_fignums() == []


def test_a_map_whose_frames_do_not_match_its_values_is_rejected(omap):
    broken = dataclasses.replace(omap, rpm=omap.rpm[:-1])
    with pytest.raises(ValueError, match="do not match"):
        rp.rpm_order_map(broken)
    short_axis = dataclasses.replace(omap, axis=omap.axis[:-1])
    with pytest.raises(ValueError, match="axis of the map"):
        rp.rpm_order_map(short_axis)
    backwards = dataclasses.replace(omap, axis=omap.axis[::-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        rp.rpm_order_map(backwards)
    assert plt.get_fignums() == []


def test_binned_map_with_missing_bin_edges_is_rejected(fmap):
    b = bin_map(fmap, 250.0)
    with pytest.raises(ValueError, match="rpm_lo"):
        rp.rpm_frequency_map(dataclasses.replace(b, rpm_lo=None))
    overlapping = dataclasses.replace(b, rpm_hi=b.rpm_hi + 300.0)
    with pytest.raises(ValueError, match="overlap"):
        rp.rpm_frequency_map(overlapping)
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------
# Order lines
# --------------------------------------------------------------------------


def test_campbell_order_lines_follow_f_equals_order_times_rpm_over_60(fmap):
    ax = rp.rpm_frequency_map(fmap, mark_orders=(2, 4, 6))
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    assert len(ax.lines) == 3
    for line, order in zip(ax.lines, (2, 4, 6)):
        x, y = line.get_xdata(), line.get_ydata()
        np.testing.assert_allclose(y, order * x / 60.0, rtol=1e-12)
        assert x.min() == pytest.approx(x_lo) and x.max() == pytest.approx(x_hi)  # across the image
        assert y.min() >= y_lo and y.max() <= y_hi
        assert line.get_linewidth() <= 1.0 and line.get_color() == "white"  # thin, light
    assert [t.get_text() for t in ax.texts] == ["2", "4", "6"]
    for text, order in zip(ax.texts, (2, 4, 6)):
        px, py = text.xy  # at the right end of its line, inside the axes
        assert x_lo <= px <= x_hi and y_lo <= py <= y_hi
        assert px >= x_hi - 0.02 * (x_hi - x_lo)
        np.testing.assert_allclose(py, order * px / 60.0, rtol=1e-12)  # on the line


def test_campbell_order_lines_against_time_use_the_speed_of_each_frame(updown):
    m = _fmap(updown.x, updown.rpm)
    ax = rp.rpm_frequency_map(m, mark_orders=[2, 4])
    assert ax.get_xlabel() == "Time (s)" and len(ax.lines) == 2
    for line, order in zip(ax.lines, (2, 4)):
        np.testing.assert_array_equal(line.get_xdata(), m.t)  # the frame centres
        np.testing.assert_allclose(line.get_ydata(), order * m.rpm / 60.0, rtol=1e-12)
    assert [t.get_text() for t in ax.texts] == ["2", "4"]


def test_campbell_order_lines_on_a_run_up_drawn_against_time(fmap):
    ax = rp.rpm_frequency_map(fmap, x="time", mark_orders=(2, 4))
    assert ax.get_xlabel() == "Time (s)"
    for line, order in zip(ax.lines, (2, 4)):
        np.testing.assert_array_equal(line.get_xdata(), fmap.t)
        np.testing.assert_allclose(line.get_ydata(), order * fmap.rpm / 60.0, rtol=1e-12)


def test_campbell_order_lines_of_a_coast_down_run_with_the_axis():
    sig = _signal(rpm_profiles(_t(), "linear", start=R1, stop=R0))
    m = _fmap(sig.x, sig.rpm)
    ax = rp.rpm_frequency_map(m, mark_orders=(2, 6))
    x_lo, x_hi = ax.get_xlim()
    assert ax.get_xlabel() == "Speed (rpm)" and x_lo < x_hi
    for line, order in zip(ax.lines, (2, 6)):
        x, y = line.get_xdata(), line.get_ydata()
        np.testing.assert_allclose(y, order * x / 60.0, rtol=1e-12)
        assert x.min() == pytest.approx(x_lo) and x.max() == pytest.approx(x_hi)
    for text, order in zip(ax.texts, (2, 6)):
        assert text.xy[0] >= x_hi - 0.02 * (x_hi - x_lo)  # at the right end, where the speed is highest
        np.testing.assert_allclose(text.xy[1], order * text.xy[0] / 60.0, rtol=1e-12)


def test_campbell_order_lines_on_a_binned_map_are_against_speed(fmap):
    b = bin_map(fmap, 250.0)
    ax = rp.rpm_frequency_map(b, mark_orders=(2,))
    x, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    np.testing.assert_allclose(y, 2.0 * x / 60.0, rtol=1e-12)
    assert x.min() == pytest.approx(b.rpm_lo[0]) and x.max() == pytest.approx(b.rpm_hi[-1])


def test_campbell_order_lines_above_fmax_are_clipped(fmap):
    ax = rp.rpm_frequency_map(fmap, mark_orders=(2, 4, 6, 12), fmax=150.0)
    top = _top(fmap, 150.0)  # the bin at 150 Hz is whole: the axis ends at its upper edge
    assert ax.get_ylim() == (0.0, top) and top > 150.0
    # order 12 is above the top over the whole speed range: not drawn, and not labelled
    assert [t.get_text() for t in ax.texts] == ["2", "4", "6"] and len(ax.lines) == 3
    for line, order in zip(ax.lines, (2, 4, 6)):
        x, y = line.get_xdata(), line.get_ydata()
        np.testing.assert_allclose(y, order * x / 60.0, rtol=1e-12)
        assert y.max() <= top + 1e-9  # nothing above the top of the axes
    # order 4 reaches the top at 15 * top rpm and ends there, order 6 at 10 * top rpm; order 2 never does
    assert ax.lines[1].get_xdata().max() == pytest.approx(15.0 * top)
    assert ax.lines[1].get_ydata().max() == pytest.approx(top)
    assert ax.lines[2].get_xdata().max() == pytest.approx(10.0 * top)
    assert ax.lines[0].get_xdata().max() == pytest.approx(ax.get_xlim()[1])
    for text in ax.texts:  # labels stay inside the axes
        assert 0.0 <= text.xy[1] <= top - 0.06 * top + 1e-9


def test_campbell_order_lines_against_time_are_clipped_where_they_leave_the_axes(updown):
    m = _fmap(updown.x, updown.rpm)
    ax = rp.rpm_frequency_map(m, mark_orders=(4,), fmax=150.0)
    x, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    assert np.nanmax(y) == pytest.approx(_top(m, 150.0))  # 4 * 3000 / 60 = 200 Hz at the turn is cut at the top
    # pieces are separated by NaN where the line leaves and re-enters; inside, it is f = order * rpm / 60
    assert np.isnan(x).sum() == 1 and np.isnan(y).sum() == 1
    ok = np.isfinite(x)
    np.testing.assert_allclose(y[ok], 4.0 * np.interp(x[ok], m.t, m.rpm) / 60.0, rtol=1e-12)


def test_order_map_marks_orders_with_horizontal_lines(omap):
    ax = rp.rpm_order_map(omap, mark_orders=(2, 4.5, 12), max_order=8.0)
    assert len(ax.lines) == 2  # 12 is above max_order
    x_lo, x_hi = ax.get_xlim()
    for line, order in zip(ax.lines, (2.0, 4.5)):
        np.testing.assert_array_equal(line.get_ydata(), [order, order])
        np.testing.assert_allclose(line.get_xdata(), [x_lo, x_hi])
        assert line.get_linewidth() <= 1.0
    assert [t.get_text() for t in ax.texts] == ["2", "4.5"]
    for text, order in zip(ax.texts, (2.0, 4.5)):
        assert text.xy[1] == order and x_lo <= text.xy[0] <= x_hi  # at the right end, inside


def test_order_map_lines_against_time_are_horizontal_across_the_frames(updown):
    m = _omap(updown.x, updown.rpm)
    ax = rp.rpm_order_map(m, mark_orders=(2, 4))
    assert ax.get_xlabel() == "Time (s)"
    x_lo, x_hi = ax.get_xlim()
    for line, order in zip(ax.lines, (2.0, 4.0)):
        np.testing.assert_array_equal(line.get_ydata(), [order, order])
        np.testing.assert_allclose(line.get_xdata(), [x_lo, x_hi])


def test_orders_may_be_a_number_or_an_array_and_default_to_none(omap, fmap):
    assert len(rp.rpm_order_map(omap, mark_orders=2).lines) == 1
    assert len(rp.rpm_frequency_map(fmap, mark_orders=np.array([2, 4])).lines) == 2
    ax = rp.rpm_order_map(omap)
    assert len(ax.lines) == 0 and len(ax.texts) == 0


@pytest.mark.parametrize("orders", [0, -2, [2, 0], np.nan, "two", [[1, 2]], [2, np.inf]])
def test_invalid_orders_raise(omap, orders):
    with pytest.raises(ValueError, match="mark_orders"):
        rp.rpm_order_map(omap, mark_orders=orders)
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------
# Channels
# --------------------------------------------------------------------------


def test_channel_selects_the_channel_of_a_multichannel_map(runup, omap):
    x2 = np.column_stack([runup.x, 0.5 * runup.x])  # the second channel is 6.02 dB lower
    m2 = _omap(x2, runup.rpm)
    assert m2.values.shape == omap.values.shape + (2,)
    tops = []
    for channel in (0, 1):
        mesh = _mesh(rp.rpm_order_map(m2, channel=channel, **rp.SOUND_PRESSURE))
        z = _levels(mesh, m2.values.shape[:2])
        _same_levels(z, to_db(m2.values[:, :, channel], P_REF))
        tops.append(np.nanmax(z))
    assert tops[0] - tops[1] == pytest.approx(20.0 * np.log10(2.0), abs=1e-6)
    default = _levels(_mesh(rp.rpm_order_map(m2)), m2.values.shape[:2])
    _same_levels(default, to_db(m2.values[:, :, 0], 1.0))  # channel 0 is the default
    last = _levels(_mesh(rp.rpm_order_map(m2, channel=-1)), m2.values.shape[:2])
    _same_levels(last, to_db(m2.values[:, :, 1], 1.0))


def test_channel_out_of_range_names_the_number_of_channels(runup, omap):
    m2 = _omap(np.column_stack([runup.x, runup.x]), runup.rpm)
    with pytest.raises(IndexError, match="2 channels"):
        rp.rpm_order_map(m2, channel=2)
    with pytest.raises(IndexError, match="1 channel"):
        rp.rpm_order_map(omap, channel=1)  # a single-channel map has channel 0 only
    assert rp.rpm_order_map(omap, channel=0) is not None
    assert plt.get_fignums() == [plt.get_fignums()[0]]  # only the successful call left a figure


def test_channel_of_a_binned_multichannel_map(runup):
    m2 = _fmap(np.column_stack([runup.x, 0.5 * runup.x]), runup.rpm, rpm_bins=250.0)
    z0 = _levels(_mesh(rp.rpm_frequency_map(m2, channel=0)), m2.values.shape[:2])
    z1 = _levels(_mesh(rp.rpm_frequency_map(m2, channel=1)), m2.values.shape[:2])
    assert np.nanmax(z0) - np.nanmax(z1) == pytest.approx(20.0 * np.log10(2.0), abs=1e-6)


# --------------------------------------------------------------------------
# Arguments and errors
# --------------------------------------------------------------------------


def test_a_map_of_the_other_domain_names_the_right_function(omap, fmap):
    with pytest.raises(ValueError, match=r"rota\.plot\.rpm_order_map"):
        rp.rpm_frequency_map(omap)
    with pytest.raises(ValueError, match=r"rota\.plot\.rpm_frequency_map"):
        rp.rpm_order_map(fmap)
    bad = dataclasses.replace(omap, domain="angle")
    with pytest.raises(ValueError, match="unknown map domain"):
        rp.rpm_order_map(bad)
    assert plt.get_fignums() == []


def test_something_that_is_not_a_map_raises_type_error(omap):
    with pytest.raises(TypeError, match="compare_maps"):  # a list of maps: the way to draw several
        rp.rpm_order_map([omap, omap])
    with pytest.raises(TypeError, match="compare_maps"):
        rp.rpm_order_map({"a": omap})
    with pytest.raises(TypeError, match="Map"):
        rp.rpm_frequency_map(np.zeros((4, 4)))
    with pytest.raises(TypeError, match="Matplotlib Axes"):
        rp.rpm_order_map(omap, ax=np.array([1, 2]))
    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    "kwargs, exc, match",
    [
        ({"ref": 0.0}, ValueError, "ref"),
        ({"ref": -1.0}, ValueError, "ref"),
        ({"dynamic_range": -1.0}, ValueError, "dynamic_range"),
        ({"dynamic_range": np.nan}, ValueError, "dynamic_range"),
        ({"vmin": 100.0, "vmax": 50.0}, ValueError, "vmin"),
        ({"vmax": np.inf}, ValueError, "vmax"),
        ({"fmax": -5.0}, ValueError, "fmax"),
        ({"fmax": 0.0}, ValueError, "fmax"),
        ({"mark_orders": [2, -1]}, ValueError, "mark_orders"),
        ({"channel": 3}, IndexError, "channel"),
        ({"channel": 1.5}, TypeError, "integer"),
        ({"x": "speed"}, ValueError, "'auto', 'rpm' or 'time'"),
        ({"monotonic_tol": -0.01}, ValueError, "monotonic_tol"),
        ({"cmap": "no_such_colour_map"}, ValueError, "no_such_colour_map"),
    ],
)
def test_invalid_arguments_raise_and_leave_no_figure_behind(fmap, kwargs, exc, match):
    with pytest.raises(exc, match=match):
        rp.rpm_frequency_map(fmap, **kwargs)
    assert plt.get_fignums() == []


def test_options_are_keyword_only(omap):
    with pytest.raises(TypeError):
        rp.rpm_order_map(omap, None)
    with pytest.raises(TypeError):
        rp.compare_maps({"a": omap}, 1.0)


# --------------------------------------------------------------------------
# Composition: axes, figures, rcParams, style, show
# --------------------------------------------------------------------------


def test_a_given_axes_is_used_and_returned(case):
    draw, m = case
    fig, axs = plt.subplots(1, 2)
    out = draw(m, ax=axs[1])
    assert out is axs[1]
    assert plt.get_fignums() == [fig.number]  # no other figure was created
    assert len(axs[1].collections) == 1 and len(axs[0].collections) == 0
    assert len(fig.axes) == 3  # the two panels and the colour bar
    assert axs[1].get_xlabel() and axs[0].get_xlabel() == ""


def test_without_an_axes_one_new_figure_is_created(case):
    draw, m = case
    assert plt.get_fignums() == []
    ax = draw(m)
    assert plt.get_fignums() == [ax.figure.number]
    ax2 = draw(m)
    assert ax2 is not ax and len(plt.get_fignums()) == 2


def test_a_given_axes_needs_no_pyplot_for_object_oriented_use_in_subprocess():
    # A Figure that pyplot does not manage (a server, an embedded GUI) never imports pyplot.
    code = (
        "import io, sys\n"
        "import numpy as np\n"
        "import rota, rota.plot as rp\n"
        "from matplotlib.figure import Figure\n"
        "fs = 2000.0\n"
        "t = np.arange(round(1.5 * fs)) / fs\n"
        "rpm = rota.synth.rpm_profiles(t, 'linear', start=900.0, stop=2400.0)\n"
        "sig = rota.synth.synth_runup(fs, rpm, [{'order': 2.0, 'amplitude': 1.0}], seed=1)\n"
        "omap = rota.maps.rpm_order_map(sig.x, fs, sig.rpm, order_resolution=0.5, max_order=6.0)\n"
        "fmap = rota.maps.rpm_frequency_map(sig.x, fs, sig.rpm, resolution=10.0, max_freq=500.0)\n"
        "fig = Figure()\n"
        "ax = fig.subplots(1, 2)\n"
        "assert rp.rpm_order_map(omap, ax=ax[0], mark_orders=(2,), **rp.SOUND_PRESSURE) is ax[0]\n"
        "assert rp.rpm_frequency_map(fmap, ax=ax[1], mark_orders=(2,)) is ax[1]\n"
        "fig.savefig(io.BytesIO(), format='png')\n"
        "assert 'matplotlib.pyplot' not in sys.modules, 'pyplot was imported'\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


@pytest.mark.parametrize("which", ["order", "frequency", "compare", "compare_difference"])
def test_rcparams_backend_and_show_are_left_alone(omap, fmap, monkeypatch, which):
    def show(*args, **kwargs):
        raise AssertionError("plt.show() was called")

    monkeypatch.setattr(plt, "show", show)
    before = _raw_rcparams()
    backend = matplotlib.get_backend()
    if which == "order":
        rp.rpm_order_map(omap, **rp.SOUND_PRESSURE, mark_orders=(2, 4))
    elif which == "frequency":
        rp.rpm_frequency_map(fmap, **rp.SOUND_PRESSURE, mark_orders=(2, 4), fmax=500.0)
    else:
        rp.compare_maps({"a": omap, "b": omap}, **rp.SOUND_PRESSURE, mark_orders=(2,), difference=which.endswith("difference"))
    assert _raw_rcparams() == before
    assert matplotlib.get_backend() == backend


def test_the_active_style_is_used(omap):
    with plt.rc_context({"axes.labelsize": 17, "image.cmap": "magma"}):
        ax = rp.rpm_order_map(omap)
        assert ax.xaxis.label.get_fontsize() == 17  # not overridden by the figure function
        _fig, axes = rp.compare_maps({"a": omap, "b": omap})
        assert axes[0].xaxis.label.get_fontsize() == 17
    assert _mesh(ax).get_cmap().name == "viridis"  # the map colour map is the documented default


def test_no_warning_is_raised_by_any_of_the_figures(omap, fmap, updown):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rp.rpm_order_map(omap, **rp.SOUND_PRESSURE, mark_orders=(2, 4, 12))
        rp.rpm_frequency_map(fmap, **rp.SOUND_PRESSURE, mark_orders=(2, 4, 12), fmax=300.0)
        rp.rpm_frequency_map(_fmap(updown.x, updown.rpm), mark_orders=(2, 4), fmax=150.0)
        rp.rpm_frequency_map(bin_map(fmap, 250.0), mark_orders=(2,))
        rp.compare_maps([omap, omap], difference=True, mark_orders=(2, 4))


# --------------------------------------------------------------------------
# compare_maps
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def half(runup):
    """The whole signal at half the amplitude: every level 6.02 dB lower."""
    return _omap(0.5 * runup.x, runup.rpm)


@pytest.fixture(scope="module")
def quiet(runup):
    """Order 4 at a quarter of its amplitude (12.04 dB lower), everything else unchanged."""
    return _omap(_signal(runup.rpm, a4=A_ORDER4 / 4.0).x, runup.rpm)


def test_compare_maps_draws_one_panel_per_map_on_one_colour_scale(omap, half):
    fig, axes = rp.compare_maps({"quiet": half, "loud": omap}, **rp.SOUND_PRESSURE)
    assert type(fig).__name__ == "Figure" and axes.shape == (2,)
    assert [a.get_title() for a in axes] == ["quiet", "loud"]  # the dict keys, in order
    m0, m1 = _mesh(axes[0]), _mesh(axes[1])
    z0, z1 = _levels(m0, half.values.shape), _levels(m1, omap.values.shape)
    _same_levels(z0, to_db(half.values, P_REF))
    _same_levels(z1, to_db(omap.values, P_REF))
    top = max(np.nanmax(z0), np.nanmax(z1))
    assert top == np.nanmax(z1) and np.nanmax(z0) == pytest.approx(top - 20.0 * np.log10(2.0), abs=1e-6)
    assert m0.get_clim() == m1.get_clim() == pytest.approx((top - 40.0, top))  # vmax over all maps
    bars = [a for a in fig.axes if a not in axes]
    assert len(bars) == 1  # one colour bar
    assert bars[0].get_ylabel() == f"Level (dB re 20 {MICRO}Pa, RMS)"
    assert [a.get_xlabel() for a in axes] == ["Speed (rpm)", "Speed (rpm)"]
    assert axes[0].get_ylabel() == "Order" and axes[1].get_ylabel() == ""  # y shared: labelled once
    assert axes[0].get_shared_y_axes().joined(axes[0], axes[1])
    assert plt.get_fignums() == [fig.number]


def test_compare_maps_limits_and_dynamic_range(omap, half):
    _fig, axes = rp.compare_maps([omap, half], dynamic_range=20.0)
    top = np.nanmax(_levels(_mesh(axes[0]), omap.values.shape))
    assert _mesh(axes[0]).get_clim() == pytest.approx((top - 20.0, top))
    _fig, axes = rp.compare_maps([omap, half], vmin=10.0, vmax=90.0)
    assert _mesh(axes[0]).get_clim() == _mesh(axes[1]).get_clim() == (10.0, 90.0)


def test_compare_maps_list_gets_default_titles_and_options_are_passed_on(omap, half):
    fig, axes = rp.compare_maps([omap, half], mark_orders=(2, 4.5), max_order=6.0, colorbar=False, figsize=(7.0, 3.0))
    assert [a.get_title() for a in axes] == ["Map 1", "Map 2"]
    assert len(fig.axes) == 2  # colorbar=False
    assert tuple(fig.get_size_inches()) == (7.0, 3.0)
    for ax in axes:
        assert ax.get_ylim() == pytest.approx((0.0, _top(omap, 6.0)))
        assert [t.get_text() for t in ax.texts] == ["2", "4.5"] and len(ax.lines) == 2
    fig, axes = rp.compare_maps({"only": omap})  # a single map is a (degenerate) comparison
    assert axes.shape == (1,) and axes[0].get_title() == "only"


def test_compare_maps_difference_is_zero_for_identical_maps(omap):
    fig, axes = rp.compare_maps({"a": omap, "b": omap}, difference=True)
    assert axes.shape == (3,)
    assert axes[2].get_title() == "b \u2212 a (dB)"
    diff = _mesh(axes[2])
    z = _levels(diff, omap.values.shape)
    assert np.isfinite(z).all() and np.max(np.abs(z)) < 1e-9  # zero (to rounding, whatever the platform)
    assert diff.get_cmap().name == "RdBu_r" and diff.get_clim() == (-1.0, 1.0)  # centred on 0, 1 dB if all zero
    bars = sorted(a.get_ylabel() for a in fig.axes if a not in axes)
    assert bars == ["Level (dB re 1, RMS)", "Level difference (dB)"]  # levels and differences, one bar each
    # the map panels still share one scale, with the viridis map; the difference has its own
    assert _mesh(axes[0]).get_clim() == _mesh(axes[1]).get_clim() != diff.get_clim()


def test_compare_maps_difference_equals_the_known_level_change(omap, quiet):
    _fig, axes = rp.compare_maps({"before": omap, "after": quiet}, difference=True, mark_orders=(4,))
    assert axes[2].get_title() == "after \u2212 before (dB)"
    diff = _mesh(axes[2])
    z = _levels(diff, omap.values.shape)
    k2, k4 = _row(omap, 2.0), _row(omap, 4.0)
    np.testing.assert_allclose(z[k4, 3:-3], 20.0 * np.log10(0.25), atol=0.01)  # a quarter: -12.04 dB
    np.testing.assert_allclose(z[k2, 3:-3], 0.0, atol=0.01)  # order 2 is unchanged
    lo, hi = diff.get_clim()
    # symmetric about 0 dB, and wide enough for the 12 dB change (robust: the extreme 0.5 % of the shown
    # cells, here the edge cells of the weak order, saturate rather than set the scale)
    assert lo == -hi and hi == pytest.approx(np.percentile(_shown_abs(axes, omap.values.shape), 99.5)) and hi >= 12.0
    assert len(axes[2].lines) == 1 and axes[2].lines[0].get_color() == "black"  # visible on the pale centre
    np.testing.assert_array_equal(  # the difference is of the drawn levels
        z, _levels(_mesh(axes[1]), omap.values.shape) - _levels(_mesh(axes[0]), omap.values.shape)
    )


def _shown_abs(axes, shape, n_maps=2):
    """|difference| in dB of the cells in which at least one map is above the shared colour scale."""
    base = _levels(_mesh(axes[0]), shape)
    vmin = _mesh(axes[0]).get_clim()[0]
    out = []
    for k in range(1, n_maps):
        other = _levels(_mesh(axes[k]), shape)
        d = other - base
        out.append(np.abs(d[((base > vmin) | (other > vmin)) & np.isfinite(d)]))
    return np.concatenate(out)


def test_compare_maps_difference_scale_ignores_rounding_noise(omap):
    # a copy that differs by rounding only (1e-13 relative: about 1e-12 dB) is "no difference": the
    # scale stays at +-1 dB instead of painting the noise in saturated colours
    noisy = dataclasses.replace(omap, values=omap.values * (1.0 + 1e-13))
    _fig, axes = rp.compare_maps({"a": omap, "b": noisy}, difference=True)
    diff = _mesh(axes[2])
    z = _levels(diff, omap.values.shape)
    assert 0.0 < np.nanmax(np.abs(z)) < 1e-9
    assert diff.get_clim() == (-1.0, 1.0)
    # a real difference of 0.01 dB is still resolved on its own scale
    small = dataclasses.replace(omap, values=omap.values * 10 ** (0.01 / 20.0))
    _fig, axes = rp.compare_maps({"a": omap, "b": small}, difference=True)
    lo, hi = _mesh(axes[2]).get_clim()
    assert lo == -hi and hi == pytest.approx(0.01, rel=1e-6)


def test_compare_maps_difference_range_sets_the_symmetric_limits(omap, quiet):
    _fig, axes = rp.compare_maps([omap, quiet], difference=True, difference_range=5.0)
    assert _mesh(axes[2]).get_clim() == (-5.0, 5.0)
    with pytest.raises(ValueError, match="difference_range"):
        rp.compare_maps([omap, quiet], difference=True, difference_range=0.0)


def test_compare_maps_default_difference_range_is_robust_against_a_few_outlier_cells(omap, half):
    # every cell differs by -6.02 dB, except a handful of noise/edge cells that differ by +60 dB
    values = half.values.copy()
    values[4, 5] *= 1000.0
    values[6, 8] *= 1000.0
    values[10, 20] *= 1000.0
    noisy = dataclasses.replace(half, values=values)
    _fig, axes = rp.compare_maps({"a": omap, "b": noisy}, difference=True)
    z = _levels(_mesh(axes[2]), omap.values.shape)
    assert np.nanmax(np.abs(z)) > 50.0  # the outliers are there ...
    lo, hi = _mesh(axes[2]).get_clim()
    assert lo == -hi and hi == pytest.approx(20.0 * np.log10(2.0), abs=1e-6)  # ... but do not set the scale
    # difference_range= overrides the default
    _fig, axes = rp.compare_maps({"a": omap, "b": noisy}, difference=True, difference_range=60.0)
    assert _mesh(axes[2]).get_clim() == (-60.0, 60.0)


def test_compare_maps_default_difference_range_is_a_high_percentile_of_the_shown_cells_of_all_panels(omap, half, quiet):
    _fig, axes = rp.compare_maps({"a": omap, "b": half, "c": quiet}, difference=True)
    expected = np.percentile(_shown_abs(axes, omap.values.shape, n_maps=3), 99.5)
    assert _mesh(axes[3]).get_clim() == _mesh(axes[4]).get_clim() == pytest.approx((-expected, expected))
    # a few cells that differ in an otherwise identical pair: the percentile is zero, the largest one is used
    # (the cell must be shown: at least one level above the colour scale)
    values = omap.values.copy()
    strongest = np.unravel_index(np.argmax(np.abs(values)), values.shape)
    values[strongest] *= 10 ** (3.0 / 20.0)
    one = dataclasses.replace(omap, values=values)
    _fig, axes = rp.compare_maps({"a": omap, "b": one}, difference=True)
    assert _mesh(axes[2]).get_clim() == pytest.approx((-3.0, 3.0))


def test_compare_maps_three_maps_two_differences(omap, half, quiet):
    _fig, axes = rp.compare_maps({"a": omap, "b": half, "c": quiet}, difference=True)
    assert axes.shape == (5,)
    assert [a.get_title() for a in axes] == ["a", "b", "c", "b \u2212 a (dB)", "c \u2212 a (dB)"]
    z = _levels(_mesh(axes[3]), omap.values.shape)
    np.testing.assert_allclose(z[np.isfinite(z)], -20.0 * np.log10(2.0), atol=1e-6)  # half: -6.02 dB everywhere
    assert _mesh(axes[3]).get_clim() == _mesh(axes[4]).get_clim()  # one difference scale


def test_compare_maps_frequency_maps(fmap, runup):
    other = _fmap(0.5 * runup.x, runup.rpm)
    _fig, axes = rp.compare_maps({"a": fmap, "b": other}, difference=True, fmax=500.0, mark_orders=(2,))
    assert axes.shape == (3,)
    for ax in axes:
        assert ax.get_ylim() == pytest.approx((0.0, _top(fmap, 500.0))) and ax.get_ylabel() in ("Frequency (Hz)", "")
    z = _levels(_mesh(axes[2]), (int(np.count_nonzero(fmap.axis <= 500.0)), fmap.values.shape[1]))
    np.testing.assert_allclose(z[np.isfinite(z)], -20.0 * np.log10(2.0), atol=1e-6)
    assert axes[0].get_ylabel() == "Frequency (Hz)"


def test_compare_maps_channel_and_binned_maps(runup):
    m2 = _fmap(np.column_stack([runup.x, 0.5 * runup.x]), runup.rpm, rpm_bins=250.0)
    _fig, axes = rp.compare_maps({"a": m2, "b": m2}, channel=1, difference=True)
    z = _levels(_mesh(axes[0]), m2.values.shape[:2])
    _same_levels(z, to_db(m2.values[:, :, 1], 1.0))
    assert np.all(_levels(_mesh(axes[2]), m2.values.shape[:2])[np.isfinite(z)] == 0.0)
    with pytest.raises(IndexError, match="2 channels"):
        rp.compare_maps({"a": m2}, channel=2)


def test_compare_maps_side_by_side_does_not_need_equal_cells(omap, runup):
    coarse = _omap(runup.x, runup.rpm, order_resolution=0.5)
    _fig, axes = rp.compare_maps({"fine": omap, "coarse": coarse})  # no difference: any maps of one kind
    assert axes.shape == (2,)
    assert _mesh(axes[0]).get_array().size == omap.values.size
    assert _mesh(axes[1]).get_array().size == coarse.values.size
    assert not axes[0].get_shared_y_axes().joined(axes[0], axes[1])  # different axes: not shared
    assert axes[0].get_ylabel() == "Order" and axes[1].get_ylabel() == "Order"
    plt.close("all")
    with pytest.raises(ValueError, match=r"cannot take the difference 'coarse' - 'fine'"):
        rp.compare_maps({"fine": omap, "coarse": coarse}, difference=True)
    assert plt.get_fignums() == []


def test_compare_maps_difference_needs_the_same_frames_and_speeds(omap, fmap, runup):
    # same number of frames and bins, other speeds: the cells are not the same
    faster = _fmap(runup.x, 1.1 * runup.rpm)
    assert faster.values.shape == fmap.values.shape and not np.allclose(faster.rpm, fmap.rpm)
    with pytest.raises(ValueError, match="differ in axis, frames or speed"):
        rp.compare_maps({"a": fmap, "b": faster}, difference=True)
    _fig, axes = rp.compare_maps({"a": fmap, "b": faster})  # side by side is fine
    assert axes.shape == (2,)
    plt.close("all")
    # other speeds also change the number of frames of an order map (more revolutions)
    faster_orders = _omap(runup.x, 1.1 * runup.rpm)
    with pytest.raises(ValueError, match="differ in axis, frames or speed"):
        rp.compare_maps({"a": omap, "b": faster_orders}, difference=True)
    with pytest.raises(ValueError, match="at least two maps"):
        rp.compare_maps({"a": omap}, difference=True)
    binned = bin_map(omap, 250.0)
    with pytest.raises(ValueError, match="differ in axis, frames or speed"):
        rp.compare_maps({"frames": omap, "bins": binned}, difference=True)
    assert plt.get_fignums() == []


def test_compare_maps_draws_a_campbell_diagram_and_an_order_map_on_one_colour_scale(omap, fmap):
    fig, axes = rp.compare_maps(
        {"Campbell": fmap, "Order map": omap}, **rp.SOUND_PRESSURE, mark_orders=(2, 4), fmax=800.0, max_order=6.0
    )
    assert axes.shape == (2,) and [a.get_title() for a in axes] == ["Campbell", "Order map"]
    # each panel keeps its own domain: axis, label and limit (fmax for Hz, max_order for orders)
    assert axes[0].get_ylabel() == "Frequency (Hz)" and axes[1].get_ylabel() == "Order"
    assert axes[0].get_ylim() == pytest.approx((0.0, _top(fmap, 800.0)))
    assert axes[1].get_ylim() == pytest.approx((0.0, _top(omap, 6.0)))
    assert not axes[0].get_shared_y_axes().joined(axes[0], axes[1])
    nf = int(np.count_nonzero(fmap.axis <= 800.0))
    no = int(np.count_nonzero(omap.axis <= 6.0))
    zf = _levels(_mesh(axes[0]), (nf, fmap.values.shape[1]))
    zo = _levels(_mesh(axes[1]), (no, omap.values.shape[1]))
    _same_levels(zf, to_db(fmap.values[:nf], P_REF))
    _same_levels(zo, to_db(omap.values[:no], P_REF))
    # the colour scale is shared (vmax over both panels) and there is one colour bar
    top = max(np.nanmax(zf), np.nanmax(zo))
    assert _mesh(axes[0]).get_clim() == _mesh(axes[1]).get_clim() == pytest.approx((top - 40.0, top))
    bars = [a for a in fig.axes if a not in axes]
    assert len(bars) == 1 and bars[0].get_ylabel() == f"Level (dB re 20 {MICRO}Pa, RMS)"
    # mark_orders goes to both panels as appropriate: slanted rays in Hz, horizontal lines in orders
    assert [t.get_text() for t in axes[0].texts] == ["2", "4"] == [t.get_text() for t in axes[1].texts]
    ray = axes[0].lines[0]
    np.testing.assert_allclose(ray.get_ydata()[0] / ray.get_xdata()[0], 2.0 / 60.0)  # f = order * rpm / 60
    assert np.ptp(axes[1].lines[0].get_ydata()) == 0.0 and axes[1].lines[0].get_ydata()[0] == 2.0


def test_compare_maps_mixed_domains_take_each_limit_only_for_their_panels(omap, fmap):
    _fig, axes = rp.compare_maps([fmap, omap], fmax=500.0)  # no max_order: the order panel is whole
    assert axes[0].get_ylim() == pytest.approx((0.0, _top(fmap, 500.0)))
    assert axes[1].get_ylim()[1] == pytest.approx(omap.axis[-1] + 0.125)
    _fig, axes = rp.compare_maps([omap, fmap], max_order=6.0)  # the other way round
    assert axes[0].get_ylim() == pytest.approx((0.0, _top(omap, 6.0)))
    assert axes[1].get_ylim()[1] == pytest.approx(fmap.axis[-1] + 5.0)
    _fig, axes = rp.compare_maps([omap, fmap, omap])  # also with a repeated domain
    assert axes.shape == (3,)


def test_compare_maps_mixed_domains_do_not_share_the_y_axis_even_on_equal_numbers(fmap):
    as_orders = dataclasses.replace(fmap, domain="order")  # the same bin numbers, read as orders
    _fig, axes = rp.compare_maps([fmap, as_orders])
    assert axes[0].get_ylim() == axes[1].get_ylim()
    assert not axes[0].get_shared_y_axes().joined(axes[0], axes[1])
    assert axes[0].get_ylabel() == "Frequency (Hz)" and axes[1].get_ylabel() == "Order"


def test_compare_maps_difference_needs_one_domain(omap, fmap):
    with pytest.raises(ValueError, match="one domain.*frequency.*order"):
        rp.compare_maps({"order": omap, "frequency": fmap}, difference=True)
    assert plt.get_fignums() == []
    _fig, axes = rp.compare_maps({"order": omap, "frequency": fmap})  # side by side is fine
    assert axes.shape == (2,)


def test_compare_maps_refuses_to_mix_amplitude_conventions(omap, runup):
    peak = _omap(runup.x, runup.rpm, amplitude="peak")
    with pytest.raises(ValueError, match="amplitude conventions"):
        rp.compare_maps({"rms": omap, "peak": peak})
    assert plt.get_fignums() == []


def test_compare_maps_limit_option_must_match_a_domain_present(omap, fmap):
    with pytest.raises(ValueError, match="fmax does not apply to order maps"):
        rp.compare_maps([omap, omap], fmax=500.0)
    with pytest.raises(ValueError, match="max_order does not apply to frequency maps"):
        rp.compare_maps([fmap, fmap], max_order=6.0)
    assert plt.get_fignums() == []


def test_compare_maps_rejects_what_it_cannot_draw(omap):
    with pytest.raises(TypeError, match="Map"):
        rp.compare_maps({"a": omap, "b": np.zeros((3, 3))})
    with pytest.raises(ValueError, match="empty"):
        rp.compare_maps([])
    with pytest.raises(ValueError, match="ref"):
        rp.compare_maps([omap, omap], ref=0.0)
    with pytest.raises(ValueError, match="no_such_colour_map"):
        rp.compare_maps([omap, omap], cmap="no_such_colour_map")
    assert plt.get_fignums() == []  # a failed call leaves no figure, including one that failed while drawing


# --------------------------------------------------------------------------
# Documentation contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("func", [rp.rpm_frequency_map, rp.rpm_order_map, rp.compare_maps])
def test_docstrings_state_units_convention_and_references(func):
    doc = " ".join(func.__doc__.split())  # line wraps do not count
    for needle in ("Parameters", "Returns", "Raises", "References", "amplitude convention", "peak", "RMS", "dB"):
        assert needle in doc, f"{func.__name__}: {needle!r} missing"
    for author in ("Brandt", "Hunter", "Crameri"):
        assert author in doc
    assert "rpm" in doc and ("Hz" in doc or "order" in doc)


def test_the_package_exports_the_map_functions_and_the_sound_pressure_constant():
    assert set(rp.__all__) >= {"SOUND_PRESSURE", "rpm_frequency_map", "rpm_order_map", "compare_maps"}
    assert rp.SOUND_PRESSURE == {"ref": 2e-5, "unit": "Pa"}
    assert "rpm_order_map" in rp.__doc__ and "SOUND_PRESSURE" in rp.__doc__ and "Matplotlib" in rp.__doc__


def _large_map_with_a_narrow_change(change_db=-10.0, noise=0.03, n_rows=960, n_frames=40):
    """Two order maps (resolution 0.0625, orders up to 60): strong orders 2 to 5 over a weak noise floor.

    The second map has resynthesis-like noise on every cell and order 3 lowered by ``change_db``; the
    changed band is three rows of the 960, well below 1 % of the cells.
    """
    rng = np.random.default_rng(7)
    axis = 0.0625 * np.arange(n_rows)
    t = np.arange(n_frames) / 10.0
    rpm = np.linspace(1000.0, 3000.0, n_frames)
    floor = 1e-4 * (1.0 + rng.random((n_rows, n_frames)))  # about 80 dB below the orders: not shown
    for order in (2.0, 3.0, 4.0, 5.0):
        floor[np.abs(axis - order) < 0.07] = 1.0
    jitter = 1.0 + noise * rng.standard_normal(floor.shape)
    after = floor * jitter
    after[np.abs(axis - 3.0) < 0.07] *= 10.0 ** (change_db / 20.0)
    make = lambda v: Map(
        values=v.astype(complex), axis=axis, domain="order", t=t, rpm=rpm, rpm_min=rpm, rpm_max=rpm,
        amplitude="rms", enbw_bins=1.5, window="hann", nperseg=64, hop=16,
    )
    return make(floor), make(after), axis


def test_compare_maps_difference_scale_shows_a_narrow_change_in_a_large_map():
    before, after, axis = _large_map_with_a_narrow_change(-10.0)
    band = np.abs(axis - 3.0) < 0.07
    assert band.mean() < 0.01  # the real change covers less than 1 % of the cells
    _fig, axes = rp.compare_maps({"before": before, "after": after}, difference=True)
    diff = _mesh(axes[2])
    z = _levels(diff, before.values.shape)
    lo, hi = diff.get_clim()
    # the noise of the cells that no map panel shows (and of the shown cells) does not set the scale
    assert lo == -hi and hi >= 9.0
    assert np.nanmedian(z[band]) == pytest.approx(-10.0, abs=0.5)
    # the scale is the one of the shown cells: far above the noise of all cells
    assert np.percentile(np.abs(z[np.isfinite(z)]), 99) < 1.0


def test_compare_maps_difference_colour_bar_has_arrows_where_cells_saturate():
    before, after, _ = _large_map_with_a_narrow_change(-10.0)
    for rng_db, extend in ((4.0, "min"), (30.0, "neither")):  # the band is -10 dB: below -4 saturates
        _fig, axes = rp.compare_maps({"a": before, "b": after}, difference=True, difference_range=rng_db)
        assert _mesh(axes[2]).colorbar.extend == extend
    # the default scale: arrows exactly when a shown cell lies beyond the limits
    _fig, axes = rp.compare_maps({"a": before, "b": after}, difference=True)
    _lo, hi = _mesh(axes[2]).get_clim()
    shown = _shown_abs(axes, before.values.shape)
    assert (_mesh(axes[2]).colorbar.extend != "neither") == bool(shown.max() > hi)
    # identical maps: nothing saturates
    _fig, axes = rp.compare_maps({"a": before, "b": before}, difference=True)
    assert _mesh(axes[2]).colorbar.extend == "neither"


def test_compare_maps_difference_scale_follows_the_change_at_any_map_size():
    # the reviewer's case: the same 10 dB change must not read as a fraction of a dB on a larger map
    for n_rows in (130, 480, 960):
        before, after, _ = _large_map_with_a_narrow_change(-10.0, n_rows=n_rows)
        _fig, axes = rp.compare_maps({"a": before, "b": after}, difference=True)
        assert _mesh(axes[2]).get_clim()[1] >= 9.0, n_rows
