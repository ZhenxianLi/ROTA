"""Tests of the speed, extraction and tonal figures of rota.plot against analytic truth.

Needs Matplotlib (Agg backend, no pixel comparison); skipped without it. Signals come
from rota.synth; the tests read the artists back from the Axes (lines, patches, the
mesh) and compare the drawn data with the known truth: a tachometer speed with the
profile that was synthesised, a ridge on a known run-up, extracted orders of known
amplitude, the values the result holds.
"""

import subprocess
import sys
import warnings

import numpy as np
import pytest

import rota
import rota.plot._analysis as ra
from rota.extract import order_waveform
from rota.plot._core import SOUND_PRESSURE, to_db
from rota.resynth import OrderTable
from rota.slices import order_slice
from rota.speed import rpm_from_ridge, tacho_to_rpm
from rota.synth import rpm_profiles, synth_runup
from rota.tonal import band_levels_vs_rpm, order_prominence

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

MICRO = "µ"  # micro sign U+00B5
FS = 4000.0
DUR = 3.0
R0, R1 = 1200.0, 3000.0  # run-up, rpm
A2, A4 = 1.5, 0.4  # peak amplitudes of orders 2 and 4, Pa
P_REF = 2e-5


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------
# Signals and results (rota.synth supplies the truth)
# --------------------------------------------------------------------------


def _t():
    return np.arange(round(DUR * FS)) / FS


@pytest.fixture(scope="module")
def runup():
    rpm = rpm_profiles(_t(), "linear", start=R0, stop=R1)
    orders = [
        {"order": 2.0, "amplitude": A2, "phase": 0.3},
        {"order": 4.0, "amplitude": A4, "phase": 1.1},
    ]
    return synth_runup(FS, rpm, orders, tacho_ppr=1, noise_rms=0.0, seed=1)


@pytest.fixture(scope="module")
def comps(runup):
    return order_waveform(runup.x, FS, runup.rpm, [2.0, 4.0])


@pytest.fixture(scope="module")
def ridge_runup():
    rpm = rpm_profiles(_t(), "linear", start=R0, stop=R1)
    orders = [{"order": 2.0, "amplitude": A2, "phase": 0.3}, {"order": 4.0, "amplitude": A4, "phase": 1.1}]
    return synth_runup(FS, rpm, orders, noise_rms=0.05, seed=1)


def _ridge(sig, rpm_range):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # quality / search-range warnings are not under test
        return rpm_from_ridge(sig.x, FS, rpm_range, orders=(2, 4))


@pytest.fixture(scope="module")
def ridge_ok(ridge_runup):
    return _ridge(ridge_runup, (600.0, 6000.0))


@pytest.fixture(scope="module")
def ridge_bound(ridge_runup):
    """The search range ends below the true speed at the end of the run-up."""
    return _ridge(ridge_runup, (600.0, 2200.0))


@pytest.fixture(scope="module")
def prominence():
    """Order 4 becomes prominent at high speed (gain law), order 8 is prominent throughout."""
    fs = 8000.0
    t = np.arange(round(6.0 * fs)) / fs
    rpm = rpm_profiles(t, "linear", start=1200.0, stop=4800.0)
    orders = [
        {"order": 4.0, "amplitude": 0.5, "phase": 0.3, "gain_rpm": [1200.0, 4800.0], "gain": [0.05, 2.0]},
        {"order": 8.0, "amplitude": 0.5, "phase": 1.0},
    ]
    sig = synth_runup(fs, rpm, orders, noise_rms=0.2, seed=3)
    return order_prominence(sig.x, fs, rpm, [4.0, 8.0], rpm_bins=300.0, ref=P_REF)


@pytest.fixture(scope="module")
def bands(runup):
    return band_levels_vs_rpm(
        runup.x, FS, runup.rpm, rpm_bins=np.arange(R0, R1 + 1.0, 300.0), frame=0.1, fmax=1500.0
    )


def _raw_rcparams():
    """The stored rcParams, read without the hooks of RcParams.__getitem__ (which resolve the backend)."""
    return dict(dict.items(matplotlib.rcParams))


def _line(ax, label):
    lines = [ln for ln in ax.lines if ln.get_label() == label]
    assert len(lines) == 1, (label, [ln.get_label() for ln in ax.lines])
    return lines[0]


def _legend(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def _same(a, b, **kw):
    np.testing.assert_allclose(a, b, rtol=kw.pop("rtol", 1e-12), atol=kw.pop("atol", 1e-12), **kw)


# --------------------------------------------------------------------------
# rpm_vs_time
# --------------------------------------------------------------------------


def test_rpm_vs_time_draws_the_tacho_speed_of_a_known_profile(runup):
    t = _t()
    rpm = tacho_to_rpm(runup.tacho, FS, 1)
    ax = ra.rpm_vs_time(t, rpm)
    line = ax.lines[0]
    np.testing.assert_array_equal(line.get_xdata(), t)
    np.testing.assert_array_equal(line.get_ydata(), rpm)  # drawn as given
    mid = (t > 0.3) & (t < 2.7)
    np.testing.assert_allclose(line.get_ydata()[mid], runup.rpm[mid], rtol=5e-3)  # the speed that was put in
    assert ax.get_xlabel() == "Time (s)" and ax.get_ylabel() == "Speed (rpm)"
    assert ax.get_legend() is None  # nothing to name


def test_rpm_vs_time_label_and_dashed_reference(runup):
    t = _t()
    rpm = tacho_to_rpm(runup.tacho, FS, 1)
    ax = ra.rpm_vs_time(t, rpm, label="tacho", reference=(t[::100], runup.rpm[::100]))
    assert _legend(ax) == ["tacho", "reference"]
    ref = _line(ax, "reference")
    assert ref.get_linestyle() == "--"
    assert matplotlib.colors.to_hex(ref.get_color()) == "#404040"  # the neutral reference style ...
    assert ref.get_zorder() > _line(ax, "tacho").get_zorder()  # ... above the data
    assert ref.get_linewidth() < _line(ax, "tacho").get_linewidth()
    np.testing.assert_array_equal(ref.get_xdata(), t[::100])
    np.testing.assert_array_equal(ref.get_ydata(), runup.rpm[::100])
    assert _line(ax, "tacho").get_linestyle() == "-"
    ax2 = ra.rpm_vs_time(t, rpm, reference=(t, runup.rpm))  # a reference alone gets a legend
    assert _legend(ax2) == ["reference"]


def test_rpm_vs_time_nan_stays_a_gap():
    t = np.arange(6.0)
    rpm = np.array([1000.0, 1100.0, np.nan, 1300.0, 1400.0, 1500.0])
    ax = ra.rpm_vs_time(t, rpm)
    y = ax.lines[0].get_ydata()
    assert np.isnan(y[2]) and y.size == 6  # no interpolation, no dropped point


@pytest.mark.parametrize(
    "args, kwargs, match",
    [
        ((np.arange(3.0), np.arange(4.0)), {}, "same length"),
        ((np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(2, 3)), {}, "1-D"),
        ((np.array([]), np.array([])), {}, "non-empty"),
        ((np.array([0.0, np.nan]), np.array([1.0, 2.0])), {}, "finite"),
        ((np.arange(3.0), np.array([1.0, np.inf, 2.0])), {}, "finite"),
        ((np.arange(3.0), np.arange(3.0)), {"reference": np.arange(3.0)}, "pair"),
        ((np.arange(3.0), np.arange(3.0)), {"reference": (np.arange(3.0), np.arange(2.0))}, "same length"),
    ],
)
def test_rpm_vs_time_errors_leave_no_figure(args, kwargs, match):
    with pytest.raises(ValueError, match=match):
        ra.rpm_vs_time(*args, **kwargs)
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------
# rpm_from_ridge
# --------------------------------------------------------------------------


def test_ridge_track_of_a_known_run_up(ridge_ok, ridge_runup):
    assert not ridge_ok.at_bound.any() and (ridge_ok.confidence >= 0.1).all()
    ax = ra.rpm_from_ridge(ridge_ok)
    track = _line(ax, "ridge track")
    np.testing.assert_array_equal(track.get_xdata(), ridge_ok.t)
    np.testing.assert_array_equal(track.get_ydata(), ridge_ok.rpm)
    truth = np.interp(ridge_ok.t, _t(), ridge_runup.rpm)
    inner = (ridge_ok.t > 0.5) & (ridge_ok.t < 2.5)
    np.testing.assert_allclose(track.get_ydata()[inner], truth[inner], rtol=0.01)
    assert ax.get_xlabel() == "Time (s)" and ax.get_ylabel() == "Speed (rpm)"
    assert _legend(ax) == ["ridge track"]  # nothing doubtful: no markers, no search range
    assert len(ax.lines) == 1


def test_ridge_frames_at_a_bound_and_of_low_confidence_are_made_visible(ridge_bound, ridge_runup):
    r = ridge_bound
    bound = r.at_bound | r.beyond
    low = r.confidence < 0.1
    assert bound.sum() > 5 and low.any()  # the true speed leaves the range: the case under test
    ax = ra.rpm_from_ridge(r, reference=(_t(), ridge_runup.rpm))
    marks = _line(ax, "flagged at_bound")
    np.testing.assert_array_equal(marks.get_xdata(), r.t[bound])
    np.testing.assert_array_equal(marks.get_ydata(), r.rpm[bound])
    assert marks.get_linestyle() == "None" and marks.get_marker() == "o"
    weak = _line(ax, "confidence < 0.1")
    np.testing.assert_array_equal(weak.get_xdata(), r.t[low])
    np.testing.assert_array_equal(weak.get_ydata(), r.rpm[low])
    # the search range is drawn as two horizontal lines at its ends
    levels = sorted(float(ln.get_ydata()[0]) for ln in ax.lines if len(ln.get_ydata()) == 2 and ln.get_xdata()[0] == 0)
    assert levels == sorted(r.rpm_range)
    ref = _line(ax, "reference")
    assert ref.get_linestyle() == "--"
    np.testing.assert_array_equal(ref.get_ydata(), ridge_runup.rpm)
    assert _legend(ax) == ["ridge track", "flagged at_bound", "search range", "confidence < 0.1", "reference"]
    # the track itself is untouched by the markers
    np.testing.assert_array_equal(_line(ax, "ridge track").get_ydata(), r.rpm)


def test_ridge_track_can_be_drawn_with_markers_instead_of_a_line(ridge_ok, ridge_runup):
    ax = ra.rpm_from_ridge(ridge_ok, marker=".", reference=(_t(), ridge_runup.rpm))
    track = _line(ax, "ridge track")
    assert track.get_linestyle() == "None" and track.get_marker() == "."  # markers only, no line
    np.testing.assert_array_equal(track.get_xdata(), ridge_ok.t)
    np.testing.assert_array_equal(track.get_ydata(), ridge_ok.rpm)
    assert _legend(ax) == ["ridge track", "reference"]
    # the default is a line, and the markers of the flags are unchanged by the option
    plain = _line(ra.rpm_from_ridge(ridge_ok), "ridge track")
    assert plain.get_linestyle() == "-" and plain.get_marker() == "None"
    with pytest.raises(ValueError):
        ra.rpm_from_ridge(ridge_ok, marker="not-a-marker")


def _hex(colour):
    return matplotlib.colors.to_hex(colour)


def test_ridge_flag_markers_differ_in_shape_and_stay_visible_together(ridge_bound, ridge_runup):
    r = ridge_bound
    both = (r.at_bound | r.beyond) & ~(r.confidence >= 0.1)
    assert both.any()  # frames with both flags: the case under test
    ax = ra.rpm_from_ridge(r, reference=(_t(), ridge_runup.rpm))
    track, ring = _line(ax, "ridge track"), _line(ax, "flagged at_bound")
    cross, ref = _line(ax, "confidence < 0.1"), _line(ax, "reference")
    # a large open circle and, on top of it, a smaller cross: both show on a frame with both flags
    assert ring.get_marker() == "o" and ring.get_markerfacecolor() == "none"
    assert cross.get_marker() == "x"
    assert cross.get_markersize() < ring.get_markersize()
    assert ring.get_zorder() > track.get_zorder()
    assert ring.get_zorder() > ref.get_zorder()  # the flags stay above the dashed reference
    assert cross.get_zorder() > ring.get_zorder()
    assert np.isin(r.t[both], cross.get_xdata()).all() and np.isin(r.t[both], ring.get_xdata()).all()
    # explicit colours, none of them from the colour cycle, and distinct from the reference and from each other
    cycle = {_hex(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]}
    colours = {_hex(ring.get_markeredgecolor()), _hex(cross.get_color()), _hex(ref.get_color())}
    assert len(colours) == 3 and not colours & cycle
    assert _hex(track.get_color()) not in colours


def test_ridge_reference_is_neutral_and_above_the_track_whatever_the_colour_cycle(ridge_bound, ridge_runup):
    with plt.rc_context({"axes.prop_cycle": matplotlib.cycler(color=["#d62728", "#ff7f0e", "#2ca02c"])}):
        ax = ra.rpm_from_ridge(ridge_bound, reference=(_t(), ridge_runup.rpm))
    ref, track = _line(ax, "reference"), _line(ax, "ridge track")
    assert _hex(ref.get_color()) == "#404040" and ref.get_linestyle() == "--"
    assert ref.get_zorder() > track.get_zorder() and ref.get_linewidth() < track.get_linewidth()
    # the same style as the reference of rpm_vs_time
    ax2 = ra.rpm_vs_time(_t(), ridge_runup.rpm, reference=(_t(), ridge_runup.rpm))
    ref2 = _line(ax2, "reference")
    for prop in ("get_color", "get_linestyle", "get_linewidth", "get_zorder"):
        assert getattr(ref, prop)() == getattr(ref2, prop)(), prop


def test_ridge_label_names_the_track(ridge_ok, ridge_bound):
    assert _legend(ra.rpm_from_ridge(ridge_ok)) == ["ridge track"]  # the default
    ax = ra.rpm_from_ridge(ridge_ok, label="ridge, orders 2 and 4")
    assert _legend(ax) == ["ridge, orders 2 and 4"]
    assert _line(ax, "ridge, orders 2 and 4").get_xdata().size == ridge_ok.t.size
    ax = ra.rpm_from_ridge(ridge_bound, label="mine")
    assert _legend(ax)[0] == "mine" and "ridge track" not in _legend(ax)
    assert ra.rpm_from_ridge(ridge_ok, label=None).get_legend() is None  # nothing to name


def test_a_reference_label_drawn_twice_on_one_axes_is_listed_once(runup):
    _fig, ax = plt.subplots()
    t = _t()
    ra.rpm_vs_time(t, runup.rpm, ax=ax, label="a", reference=(t, runup.rpm))
    ra.rpm_vs_time(t, 1.1 * runup.rpm, ax=ax, label="b", reference=(t, 1.1 * runup.rpm))
    assert len(ax.lines) == 4
    assert _legend(ax) == ["a", "b", "reference"]


def test_ridge_show_unreliable_false_and_threshold(ridge_bound):
    r = ridge_bound
    ax = ra.rpm_from_ridge(r, show_unreliable=False)
    assert [ln.get_label() for ln in ax.lines] == ["ridge track"]
    ax = ra.rpm_from_ridge(r, min_confidence=0.5)
    weak = _line(ax, "confidence < 0.5")
    np.testing.assert_array_equal(weak.get_xdata(), r.t[~(r.confidence >= 0.5)])
    ax = ra.rpm_from_ridge(r, min_confidence=0.0)  # nothing is below zero
    assert not any(label.startswith("confidence") for label in _legend(ax))


def test_ridge_nan_confidence_counts_as_low(ridge_ok):
    conf = ridge_ok.confidence.copy()
    conf[3] = np.nan
    r = type(ridge_ok)(**{**ridge_ok.__dict__, "confidence": conf})
    weak = _line(ra.rpm_from_ridge(r), "confidence < 0.1")
    np.testing.assert_array_equal(weak.get_xdata(), [r.t[3]])


def test_ridge_errors(ridge_ok):
    with pytest.raises(TypeError, match="RidgeResult"):
        ra.rpm_from_ridge(ridge_ok.rpm)
    for bad in (-0.1, 1.5, np.nan, "x"):
        with pytest.raises(ValueError, match="min_confidence"):
            ra.rpm_from_ridge(ridge_ok, min_confidence=bad)
    with pytest.raises(ValueError, match="pair"):
        ra.rpm_from_ridge(ridge_ok, reference=ridge_ok.rpm)
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------
# order_components
# --------------------------------------------------------------------------


def test_extracted_orders_are_drawn_at_their_analytic_level_against_speed(comps, runup):
    ax = ra.order_components(comps, rpm=runup.rpm, **SOUND_PRESSURE)
    assert _legend(ax) == ["Order 2", "Order 4"]
    inner = slice(800, -800)  # away from the ends of the record
    for label, amp in (("Order 2", A2), ("Order 4", A4)):
        line = _line(ax, label)
        np.testing.assert_array_equal(line.get_xdata(), runup.rpm)
        np.testing.assert_allclose(line.get_ydata()[inner], 20.0 * np.log10(amp / P_REF), atol=0.05)
    # exactly the level of the envelopes
    _same(_line(ax, "Order 2").get_ydata(), to_db(np.abs(comps.envelopes[:, 0]), P_REF))
    assert ax.get_xlabel() == "Speed (rpm)"
    assert ax.get_ylabel() == f"Order level (dB re 20 {MICRO}Pa, peak)"


def test_order_components_linear_scale_and_time_axis(comps, runup):
    ax = ra.order_components(comps, x="time", scale="linear", unit="Pa", orders=4)
    assert len(ax.lines) == 1
    line = ax.lines[0]
    np.testing.assert_allclose(line.get_xdata(), _t())  # arange(n) / fs, fs from the components
    np.testing.assert_allclose(line.get_ydata()[800:-800], A4, rtol=2e-3)  # the peak amplitude
    assert ax.get_xlabel() == "Time (s)" and ax.get_ylabel() == "Order amplitude (Pa, peak)"
    assert ra.order_components(comps, x="time", scale="linear").get_ylabel() == "Order amplitude (peak)"


def test_order_components_subset_and_dB_reference_default(comps, runup):
    ax = ra.order_components(comps, rpm=runup.rpm, orders=[4.0, 2])  # in the order asked
    assert _legend(ax) == ["Order 4", "Order 2"]
    assert ax.get_ylabel() == "Order level (dB re 1, peak)"
    np.testing.assert_allclose(_line(ax, "Order 2").get_ydata()[800:-800], 20.0 * np.log10(A2), atol=0.05)


def test_order_components_rpm_at_other_times_is_interpolated_to_the_samples_inside_its_span(comps, ridge_ok, runup):
    t_frames = ridge_ok.t  # the times of a speed track, coarser than the samples
    rpm_frames = np.interp(t_frames, _t(), runup.rpm)  # a smooth speed at those times
    ax = ra.order_components(comps, rpm=rpm_frames, t_rpm=t_frames)
    x = _line(ax, "Order 2").get_xdata()
    inside = (_t() >= t_frames[0]) & (_t() <= t_frames[-1])
    assert 0 < inside.sum() < _t().size  # the frames start half a window after the first sample
    assert x.size == inside.sum()
    np.testing.assert_allclose(x, np.interp(_t()[inside], t_frames, rpm_frames))
    np.testing.assert_allclose(x, runup.rpm[inside], rtol=1e-3)  # linear run-up: the true speed
    y = _line(ax, "Order 2").get_ydata()
    _same(y, to_db(np.abs(comps.envelopes[inside, 0]), 1.0))  # the samples with a known speed, no others


def test_order_components_need_the_speed_to_use_rpm(comps):
    with pytest.raises(ValueError, match=r"x='rpm' needs the speed.*rpm=.*x='time'"):
        ra.order_components(comps)
    assert plt.get_fignums() == []


def test_order_components_rpm_must_fit_the_samples(comps, runup):
    with pytest.raises(ValueError, match="samples"):
        ra.order_components(comps, rpm=runup.rpm[:100])
    with pytest.raises(ValueError, match="t_rpm has"):
        ra.order_components(comps, rpm=runup.rpm[:10], t_rpm=np.arange(5.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        ra.order_components(comps, rpm=runup.rpm[:3], t_rpm=[0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="shape"):
        ra.order_components(comps, rpm=np.zeros((2, 2, 2)))
    assert plt.get_fignums() == []


def test_order_components_rpm_is_always_allowed_and_draws_the_true_speed(comps, runup):
    n = runup.rpm.size
    # a measured speed with small reversals (a tachometer, a ridge) ...
    tacho = rpm_profiles(_t(), "linear", start=R0, stop=R1) + 5.0 * np.sin(np.arange(n))
    assert np.any(np.diff(tacho) < 0)
    # ... and a speed that goes up and down: neither raises, the line is drawn at the true values
    up_down = rpm_profiles(_t(), "piecewise", knots=([0.0, 1.5, 3.0], [1200.0, 3000.0, 1500.0]))
    for speed in (tacho, up_down, np.full(n, 1800.0)):
        ax = ra.order_components(comps, rpm=speed)
        line = _line(ax, "Order 2")
        np.testing.assert_array_equal(line.get_xdata(), speed)  # drawn as given, in sample order
        _same(line.get_ydata(), to_db(np.abs(comps.envelopes[:, 0]), 1.0))
        assert ax.get_xlabel() == "Speed (rpm)"
    assert plt.get_fignums() != []


def test_order_components_errors(comps, runup):
    with pytest.raises(TypeError, match="OrderComponents"):
        ra.order_components(comps.envelopes)
    with pytest.raises(ValueError, match="x must be"):
        ra.order_components(comps, rpm=runup.rpm, x="angle")
    with pytest.raises(ValueError, match="scale must be"):
        ra.order_components(comps, rpm=runup.rpm, scale="log")
    with pytest.raises(ValueError, match="orders 6 are not in the result"):
        ra.order_components(comps, rpm=runup.rpm, orders=[2, 6])
    with pytest.raises(ValueError, match="orders must be"):
        ra.order_components(comps, rpm=runup.rpm, orders=[])
    with pytest.raises(ValueError, match="ref"):
        ra.order_components(comps, rpm=runup.rpm, ref=0.0)
    with pytest.raises(IndexError, match="1 channel"):
        ra.order_components(comps, rpm=runup.rpm, channel=1)
    no_fs = type(comps)(**{**comps.__dict__, "params": {}})
    with pytest.raises(ValueError, match="fs"):
        ra.order_components(no_fs, x="time")
    assert plt.get_fignums() == []


def test_order_components_channel(runup):
    x2 = np.stack([runup.x, 0.5 * runup.x], axis=1)
    c2 = order_waveform(x2, FS, runup.rpm, [2.0])
    assert c2.envelopes.shape[2] == 2
    for ch, amp in ((0, A2), (1, 0.5 * A2), (-1, 0.5 * A2)):
        ax = ra.order_components(c2, rpm=runup.rpm, channel=ch)
        np.testing.assert_allclose(ax.lines[0].get_ydata()[800:-800], 20.0 * np.log10(amp), atol=0.05)
        plt.close(ax.figure)


def test_order_components_two_shafts_are_drawn_against_their_own_speed():
    t = _t()
    rpm = np.stack(
        [rpm_profiles(t, "linear", start=1200.0, stop=3000.0), rpm_profiles(t, "linear", start=900.0, stop=1500.0)],
        axis=1,
    )
    orders = [{"order": 2.0, "amplitude": 1.0, "shaft": 0}, {"order": 3.0, "amplitude": 0.5, "shaft": 1}]
    sig = synth_runup(FS, rpm, orders, noise_rms=0.0, seed=2)
    c = order_waveform(sig.x, FS, rpm, [2.0, 3.0], shafts=[0, 1])
    ax = ra.order_components(c, rpm=rpm)
    assert _legend(ax) == ["Order 2 (shaft 0)", "Order 3 (shaft 1)"]
    np.testing.assert_array_equal(_line(ax, "Order 2 (shaft 0)").get_xdata(), rpm[:, 0])
    np.testing.assert_array_equal(_line(ax, "Order 3 (shaft 1)").get_xdata(), rpm[:, 1])
    np.testing.assert_allclose(_line(ax, "Order 3 (shaft 1)").get_ydata()[800:-800], 20.0 * np.log10(0.5), atol=0.1)
    with pytest.raises(ValueError, match="one column per shaft"):
        ra.order_components(c, rpm=rpm[:, 0])
    ax = ra.order_components(c, rpm=rpm[:, 0], orders=2)  # one shaft selected: a single speed is enough
    assert _legend(ax) == ["Order 2"]
    with pytest.raises(ValueError, match="shaft 1, but rpm has 1"):
        ra.order_components(c, rpm=rpm[:, :1])


# --------------------------------------------------------------------------
# order_table
# --------------------------------------------------------------------------


def _table(amplitude="rms", unit="Pa", offset=None):
    grid = np.arange(1200.0, 3001.0, 300.0)
    amps = np.array([[1.0], [0.25]]) * np.linspace(1.0, 2.0, grid.size)  # known magnitudes, orders 2 and 4
    phase = np.exp(1j * np.array([[0.3], [1.1]]))
    return OrderTable(
        orders=[2.0, 4.0], rpm=grid, values=amps * phase, amplitude=amplitude, unit=unit, freq_offset=offset
    ), amps


def test_order_table_rows_are_drawn_at_their_known_level():
    tbl, amps = _table()
    ax = ra.order_table(tbl, **SOUND_PRESSURE)
    assert _legend(ax) == ["Order 2", "Order 4"]
    for label, row in (("Order 2", 0), ("Order 4", 1)):
        line = _line(ax, label)
        np.testing.assert_array_equal(line.get_xdata(), tbl.rpm)
        _same(line.get_ydata(), 20.0 * np.log10(amps[row] / P_REF))
    assert ax.get_xlabel() == "Speed (rpm)"
    assert ax.get_ylabel() == f"Order level (dB re 20 {MICRO}Pa, RMS)"


@pytest.mark.parametrize("amplitude, text", [("rms", "RMS"), ("peak", "peak")])
def test_order_table_label_carries_convention_and_unit(amplitude, text):
    tbl, amps = _table(amplitude=amplitude, unit="m/s²")
    ax = ra.order_table(tbl, ref=1e-6)  # the unit of the table is used for the reference
    assert ax.get_ylabel() == f"Order level (dB re 1 {MICRO}m/s², {text})"
    ax = ra.order_table(tbl, ref=1.0, unit="Pa")  # an explicit unit wins
    assert ax.get_ylabel() == f"Order level (dB re 1 Pa, {text})"
    ax = ra.order_table(tbl, scale="linear")
    assert ax.get_ylabel() == f"Order amplitude (m/s², {text})"
    _same(ax.lines[1].get_ydata(), amps[1])  # the magnitude as it is
    assert ra.order_table(_table(unit="")[0], ref=2e-5).get_ylabel() == "Order level (dB re 2e-05, RMS)"


def test_order_table_from_slices_of_a_map_reads_the_analytic_level(runup):
    omap = rota.maps.rpm_order_map(runup.x, FS, runup.rpm, order_resolution=0.25, max_order=8.0, amplitude="rms")
    grid = np.arange(1500.0, 2701.0, 300.0)
    tbl = OrderTable.from_slices([order_slice(omap, 2.0), order_slice(omap, 4.0)], grid, unit="Pa")
    ax = ra.order_table(tbl, **SOUND_PRESSURE)
    for label, amp in (("Order 2", A2), ("Order 4", A4)):
        np.testing.assert_allclose(
            _line(ax, label).get_ydata(), 20.0 * np.log10(amp / np.sqrt(2.0) / P_REF), atol=0.05
        )


def test_order_table_subset_offsets_nan_gaps_and_channels():
    grid = np.array([1000.0, 2000.0, 3000.0])
    values = np.array(
        [
            [1.0, np.nan, 3.0],
            [0.5, 0.5, 0.0],  # zero: no level
            [2.0, 2.0, 2.0],
        ]
    )
    tbl = OrderTable(orders=[2.0, 2.0, 5.0], rpm=grid, values=values, amplitude="peak", freq_offset=[0.0, 300.0, 0.0])
    ax = ra.order_table(tbl, orders=2)  # both rows of order 2, whatever their offset
    assert _legend(ax) == ["Order 2", "Order 2 +300 Hz"]
    y0 = _line(ax, "Order 2").get_ydata()
    assert np.isnan(y0[1]) and y0[0] == 0.0 and y0[2] == pytest.approx(20.0 * np.log10(3.0))
    y1 = _line(ax, "Order 2 +300 Hz").get_ydata()
    assert np.isnan(y1[2]) and y1[0] == pytest.approx(20.0 * np.log10(0.5))  # a zero amplitude is a gap
    two = OrderTable(
        orders=[2.0], rpm=grid, values=np.stack([np.ones((1, 3)), 4.0 * np.ones((1, 3))], axis=2), amplitude="rms"
    )
    for ch, level in ((0, 0.0), (1, 20.0 * np.log10(4.0))):
        np.testing.assert_allclose(ra.order_table(two, channel=ch).lines[0].get_ydata(), level)


def test_order_table_single_speed_point_is_visible():
    tbl = OrderTable(orders=[2.0], rpm=[1500.0], values=[[1.0]], amplitude="rms")
    assert ra.order_table(tbl).lines[0].get_marker() == "o"


def test_order_table_errors():
    tbl, _ = _table()
    with pytest.raises(TypeError, match="OrderTable"):
        ra.order_table(tbl.values)
    with pytest.raises(ValueError, match="orders 3 are not in the result"):
        ra.order_table(tbl, orders=3)
    with pytest.raises(ValueError, match="scale must be"):
        ra.order_table(tbl, scale="x")
    with pytest.raises(ValueError, match="ref"):
        ra.order_table(tbl, ref=-1.0)
    with pytest.raises(IndexError, match="channel 2"):
        ra.order_table(tbl, channel=2)
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------
# order_prominence
# --------------------------------------------------------------------------


def _spans(ax):
    """(x_min, x_max, colour) of the shaded ranges."""
    out = []
    for patch in ax.patches:
        if hasattr(patch, "get_width"):  # a Rectangle (Matplotlib 3.9 and later)
            lo, hi = patch.get_x(), patch.get_x() + patch.get_width()
        else:  # a Polygon
            xy = patch.get_xy()
            lo, hi = xy[:, 0].min(), xy[:, 0].max()
        out.append((float(lo), float(hi), patch.get_facecolor()))
    return out


@pytest.mark.parametrize(
    "kind, ylabel", [("tnr", "Tone-to-noise ratio (dB)"), ("pr", "Prominence ratio (dB)")]
)
def test_prominence_ratio_and_criterion_are_those_of_the_result(prominence, kind, ylabel):
    p = prominence
    ax = ra.order_prominence(p, kind=kind)
    assert ax.get_ylabel() == ylabel and ax.get_xlabel() == "Speed (rpm)"
    ratio, crit = getattr(p, kind), getattr(p, kind + "_criterion")
    assert _legend(ax) == ["Order 4", "Order 8", "Order 4 prominent", "Order 8 prominent", "ECMA-418-1 criterion"]
    for label, j in (("Order 4", 0), ("Order 8", 1)):
        line = _line(ax, label)
        np.testing.assert_array_equal(line.get_xdata(), p.rpm)
        np.testing.assert_array_equal(line.get_ydata(), ratio[:, j])
        dashed = [
            ln for ln in ax.lines if ln.get_linestyle() == "--" and np.allclose(ln.get_ydata(), crit[:, j], equal_nan=True)
        ]
        assert len(dashed) == 1 and dashed[0].get_color() == line.get_color()
        np.testing.assert_array_equal(dashed[0].get_xdata(), p.rpm)


def test_prominence_criterion_is_the_ecma_curve(prominence):
    p = prominence
    ax = ra.order_prominence(p, kind="tnr", orders=4)
    dashed = next(ln for ln in ax.lines if ln.get_linestyle() == "--")
    f = 4.0 * p.rpm / 60.0  # tone frequency, Hz
    expected = np.where(f < 1000.0, 8.0 + 8.33 * np.log10(1000.0 / f), 8.0)  # ECMA-418-1, Formulae (12), (13)
    np.testing.assert_allclose(dashed.get_ydata(), expected, rtol=1e-12)
    ax = ra.order_prominence(p, kind="pr", orders=4)
    dashed = next(ln for ln in ax.lines if ln.get_linestyle() == "--")
    expected = np.where(f < 1000.0, 9.0 + 10.0 * np.log10(1000.0 / f), 9.0)  # Formulae (25), (26)
    np.testing.assert_allclose(dashed.get_ydata(), expected, rtol=1e-12)


def test_prominent_ranges_are_shaded_on_the_speed_bins(prominence):
    p = prominence
    assert p.method == "tnr" and 0 < p.prominent[:, 0].sum() < p.rpm.size  # order 4 becomes prominent
    ax = ra.order_prominence(p, kind="tnr")
    spans = _spans(ax)
    assert len(spans) == 3  # order 4: one run; order 8: two, the bin 2100 to 2400 rpm has no point (no frames)
    colours = {label: matplotlib.colors.to_rgb(_line(ax, label).get_color()) for label in ("Order 4", "Order 8")}
    first = int(np.argmax(p.prominent[:, 0]))
    assert (p.prominent[first:, 0]).all() and not p.prominent[:first, 0].any()
    bin_lo = lambda rpm: 1500.0 + 300.0 * np.floor((rpm - 1500.0) / 300.0)  # the 300 rpm bins from 1500
    c4 = tuple(np.round(colours["Order 4"], 6))
    (lo4, hi4), = [(c[0], c[1]) for c in spans if tuple(np.round(c[2][:3], 6)) == c4]
    assert (lo4, hi4) == (bin_lo(p.rpm[first]), bin_lo(p.rpm[-1]) + 300.0)  # from the edge of the first prominent bin
    c8 = tuple(np.round(colours["Order 8"], 6))
    runs8 = sorted((c[0], c[1]) for c in spans if tuple(np.round(c[2][:3], 6)) == c8)
    assert runs8 == [(1500.0, 2100.0), (2400.0, bin_lo(p.rpm[-1]) + 300.0)]  # adjacent bins merge, a missing bin splits
    assert all(c[2][3] == pytest.approx(0.12) for c in spans)  # light shading: overlapping orders stay distinct
    # the prominence ratio gives its own ranges (order 4 is not prominent in the lowest bins)
    ax = ra.order_prominence(p, kind="pr", orders=4)
    (lo, hi, _), = _spans(ax)
    ok = p.pr[:, 0] >= p.pr_criterion[:, 0]
    assert not ok.all() and ok[-1]
    assert (lo, hi) == (bin_lo(p.rpm[int(np.argmax(ok))]), bin_lo(p.rpm[-1]) + 300.0)


def test_prominent_shading_has_one_legend_entry_per_order_in_the_colour_of_the_order(prominence):
    ax = ra.order_prominence(prominence)
    handles, labels = ax.get_legend_handles_labels()
    shaded = {lab: h for h, lab in zip(handles, labels) if lab.endswith(" prominent")}
    assert list(shaded) == ["Order 4 prominent", "Order 8 prominent"]
    for name in ("Order 4", "Order 8"):
        patch = shaded[f"{name} prominent"]
        line = _line(ax, name)
        np.testing.assert_allclose(patch.get_facecolor()[:3], matplotlib.colors.to_rgb(line.get_color()))
        assert patch.get_facecolor()[3] == pytest.approx(0.12)
    # lines come first in the legend, then the shaded ranges, then the criterion
    assert _legend(ax)[-1] == "ECMA-418-1 criterion" and _legend(ax)[:2] == ["Order 4", "Order 8"]
    # only the selected order has a shaded range and an entry
    one = ra.order_prominence(prominence, kind="pr", orders=4)
    assert [lab for lab in _legend(one) if lab.endswith("prominent")] == ["Order 4 prominent"]
    assert len(_spans(one)) == 1


def test_prominence_without_criterion_draws_only_the_ratio(prominence):
    ax = ra.order_prominence(prominence, criterion=False)
    assert [ln.get_label() for ln in ax.lines] == ["Order 4", "Order 8"]
    assert len(ax.patches) == 0 and _legend(ax) == ["Order 4", "Order 8"]


def test_prominence_orders_and_channel():
    fs = 8000.0
    t = np.arange(round(4.0 * fs)) / fs
    rpm = rpm_profiles(t, "linear", start=1500.0, stop=4500.0)
    sig = synth_runup(fs, rpm, [{"order": 8.0, "amplitude": 0.5, "phase": 1.0}], noise_rms=0.2, seed=3)
    x2 = np.stack([sig.x, 0.1 * sig.x], axis=1)
    p = order_prominence(x2, fs, rpm, [8.0, 12.0], rpm_bins=500.0, ref=P_REF)
    assert p.tnr.ndim == 3
    ax = ra.order_prominence(p, orders=12.0, channel=1, criterion=False)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), p.tnr[:, 1, 1])
    assert _legend(ax) == ["Order 12"]
    ax = ra.order_prominence(p, orders=8, channel=-1, criterion=False)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), p.tnr[:, 0, 1])


def test_prominence_without_binning_uses_frame_edges():
    fs = 8000.0
    t = np.arange(round(4.0 * fs)) / fs
    rpm = rpm_profiles(t, "linear", start=1500.0, stop=4500.0)
    sig = synth_runup(fs, rpm, [{"order": 8.0, "amplitude": 0.5, "phase": 1.0}], noise_rms=0.2, seed=3)
    p = order_prominence(sig.x, fs, rpm, [8.0], ref=P_REF)
    assert p.rpm_bins is None and p.rpm.size >= 3
    ax = ra.order_prominence(p)
    ok = p.tnr[:, 0] >= p.tnr_criterion[:, 0]
    assert ok.any()
    lo, hi, _ = _spans(ax)[0]
    assert lo < p.rpm[int(np.argmax(ok))] < hi  # the shaded cell contains the frame's centre


def test_prominence_errors(prominence):
    with pytest.raises(TypeError, match="OrderProminence"):
        ra.order_prominence(prominence.tnr)
    with pytest.raises(ValueError, match="kind must be"):
        ra.order_prominence(prominence, kind="tonality")
    with pytest.raises(ValueError, match="orders 5 are not in the result"):
        ra.order_prominence(prominence, orders=5)
    with pytest.raises(IndexError, match="channel"):
        ra.order_prominence(prominence, channel=3)
    assert plt.get_fignums() == []


def test_prominence_points_with_a_speed_that_is_not_monotonic_are_drawn_as_they_are(prominence):
    """Lines are drawn at the true speeds; the shaded cells are built in sorted speed order and never overlap."""
    order = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]
    jumpy = type(prominence)(**{
        **prominence.__dict__,
        "rpm": prominence.rpm[order],
        "tnr": prominence.tnr[order],
        "tnr_criterion": prominence.tnr_criterion[order],
        "pr": prominence.pr[order],
        "pr_criterion": prominence.pr_criterion[order],
        "rpm_bins": None,
    })
    ax = ra.order_prominence(jumpy)
    line = _line(ax, "Order 4")
    np.testing.assert_array_equal(line.get_xdata(), jumpy.rpm)
    np.testing.assert_array_equal(line.get_ydata(), jumpy.tnr[:, 0])
    # the same points in sorted order give the same shaded ranges
    ref = ra.order_prominence(type(prominence)(**{**prominence.__dict__, "rpm_bins": None}))
    assert len(ax.patches) > 0
    assert _spans(ax) == _spans(ref)
    nan_speed = jumpy.rpm.copy()
    nan_speed[3] = np.nan
    ax = ra.order_prominence(type(prominence)(**{**jumpy.__dict__, "rpm": nan_speed}))  # no shading for that point
    assert ax.lines


# --------------------------------------------------------------------------
# band_levels_vs_rpm
# --------------------------------------------------------------------------


def _mesh(ax):
    meshes = [c for c in ax.collections if type(c).__name__ == "QuadMesh"]
    assert len(meshes) == 1
    return meshes[0]


def _image(mesh, shape):
    return np.ma.asarray(mesh.get_array()).astype(float).filled(np.nan).reshape(shape)


def test_band_image_holds_the_stored_levels_on_rpm_and_band_edges(bands):
    b = bands
    ax = ra.band_levels_vs_rpm(b)
    mesh = _mesh(ax)
    n_bins, n_bands = b.levels.shape
    z = _image(mesh, (n_bands, n_bins))
    np.testing.assert_array_equal(z, b.levels.T)  # levels as stored, bins along x, bands along y
    xy = mesh.get_coordinates()
    np.testing.assert_array_equal(xy[0, :, 0], b.rpm_edges)
    np.testing.assert_allclose(xy[:, 0, 1], np.append(b.frames.lower, b.frames.upper[-1]), rtol=1e-12)
    assert ax.get_xlabel() == "Speed (rpm)" and ax.get_ylabel() == "Frequency (Hz)"
    assert ax.get_yscale() == "log"
    assert mesh.get_clim() == pytest.approx((np.nanmax(b.levels) - 40.0, np.nanmax(b.levels)))
    # the order 2 line of the run-up (1200 to 3000 rpm: 40 to 100 Hz) sits in the 40 to 100 Hz bands
    in_line = (b.exact >= 40.0) & (b.exact <= 100.0)
    assert b.levels[:, in_line].max() > b.levels[:, b.exact > 400.0].max() + 20.0


def test_band_image_levels_follow_the_analytic_rms_level(runup):
    """A tone of known RMS inside one band gives that band the tone's level (dB re 20 uPa)."""
    rpm = np.full(round(2.0 * FS), 1800.0)
    sig = synth_runup(FS, rpm, [{"order": 10.0, "amplitude": 1.0, "phase": 0.0}], noise_rms=0.0, seed=1)  # 300 Hz
    b = band_levels_vs_rpm(sig.x, FS, rpm, rpm_bins=[1700.0, 1900.0], frame=0.25, ref=P_REF, fmax=1000.0)
    ax = ra.band_levels_vs_rpm(b, unit="Pa")
    z = _image(_mesh(ax), (b.exact.size, 1))[:, 0]
    k = int(np.argmax(z))
    assert b.nominal[k] == 315.0  # the band that holds 300 Hz
    assert z[k] == pytest.approx(20.0 * np.log10(1.0 / np.sqrt(2.0) / P_REF), abs=0.5)  # peak 1 -> RMS 0.707
    assert ax.figure.axes[-1].get_ylabel() == f"Band level (dB re 20 {MICRO}Pa, RMS)"


def test_band_image_ref_none_is_as_stored_and_a_given_ref_re_references_exactly(bands):
    assert bands.ref == 20e-6
    stored = _image(_mesh(ra.band_levels_vs_rpm(bands)), (bands.exact.size, bands.levels.shape[0]))
    np.testing.assert_array_equal(stored, bands.levels.T)
    ax = ra.band_levels_vs_rpm(bands, **SOUND_PRESSURE)  # the stored reference, given: the same numbers and a label
    np.testing.assert_array_equal(_image(_mesh(ax), stored.shape), stored)
    assert _mesh(ax).colorbar.ax.get_ylabel() == f"Band level (dB re 20 {MICRO}Pa, RMS)"
    # re-referenced to 1 uPa: L + 20 log10(20 uPa / 1 uPa) = L + 26.02 dB
    ax = ra.band_levels_vs_rpm(bands, ref=1e-6, unit="Pa")
    mesh = _mesh(ax)
    shift = 20.0 * np.log10(bands.ref / 1e-6)
    assert shift == pytest.approx(26.020599913279625)
    np.testing.assert_allclose(_image(mesh, stored.shape), stored + shift, rtol=0, atol=1e-12)
    assert mesh.colorbar.ax.get_ylabel() == f"Band level (dB re 1 {MICRO}Pa, RMS)"
    assert mesh.get_clim() == pytest.approx((np.nanmax(stored + shift) - 40.0, np.nanmax(stored + shift)))
    # unit labels only: no number changes
    z_unit = _image(_mesh(ra.band_levels_vs_rpm(bands, unit="m/s²")), stored.shape)
    np.testing.assert_array_equal(z_unit, stored)
    for bad in (0.0, -1.0, np.nan, np.inf, "x"):
        with pytest.raises(ValueError, match="ref"):
            ra.band_levels_vs_rpm(bands, ref=bad)


def test_band_image_given_ref_matches_the_analytic_level(runup):
    """A tone of RMS 1/sqrt(2) Pa reads -3.01 dB re 1 Pa and 90.97 dB re 20 uPa in its band."""
    rpm = np.full(round(2.0 * FS), 1800.0)
    sig = synth_runup(FS, rpm, [{"order": 10.0, "amplitude": 1.0, "phase": 0.0}], noise_rms=0.0, seed=1)
    b = band_levels_vs_rpm(sig.x, FS, rpm, rpm_bins=[1700.0, 1900.0], frame=0.25, ref=P_REF, fmax=1000.0)
    z_pa = _image(_mesh(ra.band_levels_vs_rpm(b, ref=1.0, unit="Pa")), (b.exact.size, 1))[:, 0]
    z_up = _image(_mesh(ra.band_levels_vs_rpm(b, **SOUND_PRESSURE)), (b.exact.size, 1))[:, 0]
    k = int(np.argmax(z_pa))
    assert z_pa[k] == pytest.approx(20.0 * np.log10(1.0 / np.sqrt(2.0)), abs=0.5)
    np.testing.assert_allclose(z_up - z_pa, 20.0 * np.log10(1.0 / P_REF), atol=1e-9)


def test_band_image_labels_and_ticks(bands):
    ax = ra.band_levels_vs_rpm(bands)
    mesh = _mesh(ax)
    assert mesh.colorbar.ax.get_ylabel() == "Band level (dB re 2e-05, RMS)"  # the unit is not stored in the result
    assert ra.band_levels_vs_rpm(bands, unit="Pa").figure.axes[-1].get_ylabel() == f"Band level (dB re 20 {MICRO}Pa, RMS)"
    anchor = int(np.argmin(np.abs(np.log(bands.exact / 1000.0))))  # the band nearest to 1 kHz
    stride, start = 3, anchor % 3  # a third-octave bank: every octave
    np.testing.assert_allclose(ax.get_yticks(), bands.exact[start::stride])
    assert bands.exact.size > 12  # a third-octave bank from 20 Hz to 1.6 kHz
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["31.5", "63", "125", "250", "500", "1k"]  # every octave, counted from 1 kHz
    sig = synth_runup(FS, np.full(4000, 1800.0), [{"order": 10.0}], seed=1)
    octaves = band_levels_vs_rpm(sig.x, FS, sig.rpm, rpm_bins=[1700.0, 1900.0], frame=0.25, fraction=1, fmax=1000.0)
    short_ax = ra.band_levels_vs_rpm(octaves)
    np.testing.assert_array_equal(octaves.nominal, [16.0, 31.5, 63.0, 125.0, 250.0, 500.0, 1000.0])
    assert [t.get_text() for t in short_ax.get_yticklabels()] == ["16", "31.5", "63", "125", "250", "500", "1k"]  # every band


def test_band_image_colour_options_and_colourbar(bands):
    top = float(np.nanmax(bands.levels))
    mesh = _mesh(ra.band_levels_vs_rpm(bands, dynamic_range=25.0))
    assert mesh.get_clim() == pytest.approx((top - 25.0, top))
    mesh = _mesh(ra.band_levels_vs_rpm(bands, vmin=0.0, vmax=100.0, cmap="magma"))
    assert mesh.get_clim() == (0.0, 100.0) and mesh.get_cmap().name == "magma"
    ax = ra.band_levels_vs_rpm(bands, colorbar=False)
    assert len(ax.figure.axes) == 1 and _mesh(ax).colorbar is None


def test_band_image_gaps_and_channel(runup):
    x2 = np.stack([runup.x, 0.1 * runup.x], axis=1)
    b = band_levels_vs_rpm(
        x2, FS, np.broadcast_to(runup.rpm[:, None], x2.shape)[:, 0], rpm_bins=np.arange(R0, R1 + 1.0, 600.0),
        frame=0.1, fmax=1500.0, min_count=100,
    )
    assert b.levels.ndim == 3 and np.isnan(b.levels).all()  # too few frames per bin: all gaps
    b = band_levels_vs_rpm(
        x2, FS, runup.rpm, rpm_bins=np.arange(R0, R1 + 1.0, 300.0), frame=0.1, fmax=1500.0
    )
    levels = b.levels.copy()
    levels[1, :, :] = np.nan  # one empty speed bin
    b = type(b)(**{**b.__dict__, "levels": levels})
    mesh = _mesh(ra.band_levels_vs_rpm(b, channel=1))
    z = _image(mesh, (levels.shape[1], levels.shape[0]))
    assert np.isnan(z[:, 1]).all()
    np.testing.assert_array_equal(z, levels[:, :, 1].T)
    z0 = _image(_mesh(ra.band_levels_vs_rpm(b)), z.shape)
    assert np.nanmax(z0) - np.nanmax(z) == pytest.approx(20.0, abs=0.5)  # channel 1 is 20 dB lower
    plt.close("all")
    all_gaps = type(b)(**{**b.__dict__, "levels": np.full_like(levels, np.nan)})
    with pytest.raises(ValueError, match="no finite level"):
        ra.band_levels_vs_rpm(all_gaps)
    assert plt.get_fignums() == []
    assert _mesh(ra.band_levels_vs_rpm(all_gaps, vmax=0.0)).get_clim() == (-40.0, 0.0)


def test_band_image_errors(bands):
    with pytest.raises(TypeError, match="BandRpmLevels"):
        ra.band_levels_vs_rpm(bands.levels)
    with pytest.raises(IndexError, match="channel"):
        ra.band_levels_vs_rpm(bands, channel=1)
    with pytest.raises(ValueError, match="dynamic_range"):
        ra.band_levels_vs_rpm(bands, dynamic_range=0.0)
    with pytest.raises(ValueError, match="rpm_edges"):
        ra.band_levels_vs_rpm(type(bands)(**{**bands.__dict__, "rpm_edges": bands.rpm_edges[::-1]}))
    with pytest.raises(ValueError, match="bands"):
        ra.band_levels_vs_rpm(type(bands)(**{**bands.__dict__, "nominal": bands.nominal[:-1]}))
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------
# Composition: axes, figures, rcParams, show, failures while drawing
# --------------------------------------------------------------------------


@pytest.fixture
def draws(runup, comps, ridge_ok, prominence, bands):
    """Every function with a call that draws, by name."""
    tbl, _ = _table()
    return {
        "rpm_vs_time": lambda **kw: ra.rpm_vs_time(_t(), runup.rpm, **kw),
        "rpm_from_ridge": lambda **kw: ra.rpm_from_ridge(ridge_ok, **kw),
        "order_components": lambda **kw: ra.order_components(comps, rpm=runup.rpm, **kw),
        "order_table": lambda **kw: ra.order_table(tbl, **kw),
        "order_prominence": lambda **kw: ra.order_prominence(prominence, **kw),
        "band_levels_vs_rpm": lambda **kw: ra.band_levels_vs_rpm(bands, **kw),
    }


NAMES = ["rpm_vs_time", "rpm_from_ridge", "order_components", "order_table", "order_prominence", "band_levels_vs_rpm"]


@pytest.mark.parametrize("name", NAMES)
def test_a_new_figure_is_created_only_without_ax(draws, name):
    assert plt.get_fignums() == []
    ax = draws[name]()
    assert len(plt.get_fignums()) == 1 and ax.figure.number in plt.get_fignums()
    fig, own = plt.subplots()
    assert draws[name](ax=own) is own  # drawn on the given Axes, which is returned
    assert len(plt.get_fignums()) == 2 and own.figure is fig
    assert own.has_data() or own.collections


@pytest.mark.parametrize("name", NAMES)
def test_two_results_can_share_one_axes(draws, name):
    _, ax = plt.subplots()
    draws[name](ax=ax)
    n1, c1 = len(ax.lines), len(ax.collections)
    draws[name](ax=ax)
    assert len(ax.lines) == 2 * n1 and len(ax.collections) == 2 * c1  # appended, not replaced
    assert len(plt.get_fignums()) == 1


@pytest.mark.parametrize("name", NAMES)
def test_ax_must_be_an_axes(draws, name):
    _, axes = plt.subplots(1, 2)
    with pytest.raises(TypeError, match="Axes"):
        draws[name](ax=axes)


@pytest.mark.parametrize("name", NAMES)
def test_rcparams_backend_and_show_are_left_alone(draws, name, monkeypatch):
    def show(*args, **kwargs):
        raise AssertionError("plt.show() was called")

    monkeypatch.setattr(plt, "show", show)
    before = _raw_rcparams()
    backend = matplotlib.get_backend()
    draws[name]()
    assert _raw_rcparams() == before
    assert matplotlib.get_backend() == backend


@pytest.mark.parametrize("name", NAMES[:-1])
def test_a_figure_created_here_is_closed_when_drawing_fails(draws, name, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("drawing failed")

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", broken)
    with pytest.raises(RuntimeError, match="drawing failed"):
        draws[name]()
    assert plt.get_fignums() == []
    _, own = plt.subplots()
    with pytest.raises(RuntimeError, match="drawing failed"):
        draws[name](ax=own)
    assert len(plt.get_fignums()) == 1  # the figure of the caller stays


def test_band_image_closes_its_figure_when_drawing_fails(bands):
    with pytest.raises(ValueError):
        ra.band_levels_vs_rpm(bands, cmap="no-such-colour-map")
    assert plt.get_fignums() == []


def test_the_active_style_and_colour_cycle_are_used(runup):
    with plt.rc_context({"axes.labelsize": 17, "axes.prop_cycle": matplotlib.cycler(color=["#123456", "#654321"])}):
        ax = ra.rpm_vs_time(_t(), runup.rpm, reference=(_t(), runup.rpm))
        assert ax.xaxis.label.get_fontsize() == 17
        assert matplotlib.colors.to_hex(ax.lines[0].get_color()) == "#123456"
        # the reference is neutral grey in every style and does not use up a colour of the cycle
        assert matplotlib.colors.to_hex(ax.lines[1].get_color()) == "#404040"
        assert matplotlib.colors.to_hex(ax.plot([0.0], [0.0])[0].get_color()) == "#654321"


def test_the_functions_run_on_a_figure_without_pyplot_in_a_subprocess():
    # A Figure that pyplot does not manage (a server, an embedded GUI) never imports pyplot.
    code = (
        "import io, sys\n"
        "import numpy as np\n"
        "import rota\n"
        "import rota.plot._analysis as ra\n"
        "from matplotlib.figure import Figure\n"
        "fs = 2000.0\n"
        "t = np.arange(round(1.5 * fs)) / fs\n"
        "rpm = rota.synth.rpm_profiles(t, 'linear', start=900.0, stop=2400.0)\n"
        "sig = rota.synth.synth_runup(fs, rpm, [{'order': 2.0, 'amplitude': 1.0}], tacho_ppr=1, seed=1)\n"
        "c = rota.extract.order_waveform(sig.x, fs, rpm, [2.0])\n"
        "b = rota.tonal.band_levels_vs_rpm(sig.x, fs, rpm, rpm_bins=[900.0, 1600.0, 2400.0], frame=0.1, fmax=500.0)\n"
        "fig = Figure()\n"
        "ax = fig.subplots(1, 3)\n"
        "assert ra.rpm_vs_time(t, rpm, ax=ax[0], label='x') is ax[0]\n"
        "assert ra.order_components(c, rpm=rpm, ax=ax[1]) is ax[1]\n"
        "assert ra.band_levels_vs_rpm(b, ax=ax[2]) is ax[2]\n"
        "fig.savefig(io.BytesIO(), format='png')\n"
        "assert 'matplotlib.pyplot' not in sys.modules, 'pyplot was imported'\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_docstrings_name_parameters_returns_and_references():
    for name in ra.__all__:
        doc = getattr(ra, name).__doc__
        for section in ("Parameters", "Returns", "References"):
            assert section in doc, (name, section)
        assert "peak" in doc.lower() or "rms" in doc.lower() or "no amplitude" in doc.lower(), name
