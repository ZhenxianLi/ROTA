"""Tests of the line figures of rota.plot against analytic truth (rota.synth supplies the signals).

Needs Matplotlib (Agg backend, no pixel comparison); skipped without it. The tests read the
drawn data back from the Axes (Line2D data, the polygons of the bands and shaded spans, the
labels) and compare it with the analytic level of the synthetic orders and tones.
"""

import dataclasses

import numpy as np
import pytest

import rota.plot._lines as lines
from rota.bridges import TimeMetric
from rota.maps import rpm_frequency_map, rpm_order_map
from rota.plot._core import SOUND_PRESSURE, to_db
from rota.slices import frequency_slice as compute_frequency_slice
from rota.slices import metric_vs_rpm, order_slice, order_spectrum, overall_level
from rota.synth import rpm_profiles, synth_runup

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

MICRO = "µ"  # micro sign U+00B5
PM_STD = "\u00b1 std"  # plus-minus sign U+00B1
FS = 4000.0
DUR = 3.0
R0, R1 = 1200.0, 3000.0  # run-up, rpm
A2 = 1.5  # peak amplitudes in Pa
A4 = 0.4
A_TONE = 2.0  # fixed tone at 1000 Hz, on a bin of the 10 Hz frequency map and 100 cycles per 0.1 s frame
P_REF = 2e-5
GAIN_DB = 6.0
MID = slice(3, -3)  # frames away from the edges of the run-up (the band is fully inside)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------
# Signals, results and readers of the drawn artists
# --------------------------------------------------------------------------


def _t():
    return np.arange(round(DUR * FS)) / FS


def _runup_rpm(stop=R1):
    return rpm_profiles(_t(), "linear", start=R0, stop=stop)


def _signal(rpm, a2=A2, a4=A4, tone=A_TONE):
    orders = [
        {"order": 2.0, "amplitude": a2, "phase": 0.3},
        {"order": 4.0, "amplitude": a4, "phase": 1.1},
    ]
    tones = [{"freq": 1000.0, "amplitude": tone, "phase": 0.5}] if tone else ()
    return synth_runup(FS, rpm, orders, tones=tones, noise_rms=0.0, seed=1)


def _omap(sig, amplitude="rms", x=None):
    x = sig.x if x is None else x
    return rpm_order_map(x, FS, sig.rpm, order_resolution=0.25, max_order=8.0, amplitude=amplitude)


def _fmap(sig, amplitude="rms"):
    return rpm_frequency_map(sig.x, FS, sig.rpm, nperseg=400, amplitude=amplitude, max_freq=1500.0)


def _gain():
    return 10.0 ** (GAIN_DB / 20.0)


@pytest.fixture(scope="module")
def runup():
    return _signal(_runup_rpm())


@pytest.fixture(scope="module")
def louder():
    """The same signal with order 2 louder by GAIN_DB."""
    return _signal(_runup_rpm(), a2=A2 * _gain())


@pytest.fixture(scope="module")
def omap(runup):
    return _omap(runup)


@pytest.fixture(scope="module")
def fmap(runup):
    return _fmap(runup)


@pytest.fixture(scope="module")
def updown():
    rpm = rpm_profiles(_t(), "piecewise", knots=([0.0, 1.5, 3.0], [1200.0, 3000.0, 1500.0]))
    return _signal(rpm)


@pytest.fixture(scope="module")
def slice2(omap):
    return order_slice(omap, 2.0)


@pytest.fixture(scope="module")
def slice2_louder(louder):
    return order_slice(_omap(louder), 2.0)


def _line(ax, i=0):
    return ax.lines[i]


def _ydata(ax, i=0):
    return np.asarray(_line(ax, i).get_ydata(), dtype=float)


def _xdata(ax, i=0):
    return np.asarray(_line(ax, i).get_xdata(), dtype=float)


def _legend(ax):
    return ax.get_legend_handles_labels()[1]


def _by_label(ax, label):
    return [ln for ln in ax.lines if ln.get_label() == label]


def _raw_rcparams():
    return dict(dict.items(matplotlib.rcParams))


def _exp_db(amplitude, convention="rms"):
    return 20.0 * np.log10(amplitude * (1.0 / np.sqrt(2.0) if convention == "rms" else 1.0) / P_REF)


# --------------------------------------------------------------------------
# order_slice and frequency_slice: what is drawn
# --------------------------------------------------------------------------


def test_order_slice_draws_the_analytic_level_of_a_known_order(slice2):
    ax = lines.order_slice(slice2, **SOUND_PRESSURE)
    assert len(ax.lines) == 1
    np.testing.assert_allclose(_xdata(ax), slice2.rpm)
    y = _ydata(ax)
    np.testing.assert_allclose(y, to_db(slice2.amplitude, P_REF), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y[MID], _exp_db(A2), atol=0.02)  # order 2: A2 peak = A2/sqrt(2) RMS
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Speed (rpm)", f"Order level (dB re 20 {MICRO}Pa, RMS)")
    assert ax.get_legend() is None  # one unlabelled result: no legend


@pytest.mark.parametrize("amplitude, word", [("rms", "RMS"), ("peak", "peak")])
def test_order_slice_labels_follow_the_amplitude_convention(runup, amplitude, word):
    s = order_slice(_omap(runup, amplitude), 4.0)
    ax = lines.order_slice(s, **SOUND_PRESSURE)
    assert ax.get_ylabel() == f"Order level (dB re 20 {MICRO}Pa, {word})"
    np.testing.assert_allclose(_ydata(ax)[MID], _exp_db(A4, amplitude), atol=0.02)
    ax = lines.order_slice(s, **SOUND_PRESSURE, scale="linear")
    assert ax.get_ylabel() == f"Order amplitude (Pa, {word})"
    assert lines.order_slice(s).get_ylabel() == f"Order level (dB re 1, {word})"
    assert lines.order_slice(s, scale="linear").get_ylabel() == f"Order amplitude ({word})"
    assert lines.order_slice(s, ref=1e-6, unit="m/s²").get_ylabel() == f"Order level (dB re 1 {MICRO}m/s², {word})"


def test_linear_scale_draws_the_amplitude_in_the_signal_unit(slice2):
    ax = lines.order_slice(slice2, scale="linear", unit="Pa")
    np.testing.assert_array_equal(_ydata(ax), slice2.amplitude)
    np.testing.assert_allclose(_ydata(ax)[MID], A2 / np.sqrt(2.0), rtol=2e-3)
    # the reference plays no role in the linear scale
    np.testing.assert_array_equal(_ydata(lines.order_slice(slice2, scale="linear", ref=2e-5)), slice2.amplitude)


def test_frequency_slice_draws_the_analytic_level_of_a_known_tone(fmap):
    s = compute_frequency_slice(fmap, 1000.0)
    ax = lines.frequency_slice(s, **SOUND_PRESSURE)
    np.testing.assert_allclose(_xdata(ax), s.rpm)
    np.testing.assert_allclose(_ydata(ax), to_db(s.amplitude, P_REF), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_ydata(ax)[MID], _exp_db(A_TONE), atol=0.02)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Speed (rpm)", f"Band level (dB re 20 {MICRO}Pa, RMS)")
    ax = lines.frequency_slice(s, scale="linear", unit="Pa")
    assert ax.get_ylabel() == "Band amplitude (Pa, RMS)"
    np.testing.assert_array_equal(_ydata(ax), s.amplitude)


def test_x_time_draws_against_the_frame_times(slice2):
    ax = lines.order_slice(slice2, x="time")
    np.testing.assert_array_equal(_xdata(ax), slice2.t)
    assert ax.get_xlabel() == "Time (s)"


def test_x_rpm_is_always_allowed_and_draws_the_true_speeds_even_where_the_line_doubles_back(updown):
    s = order_slice(_omap(updown), 2.0)
    assert not (np.all(np.diff(s.rpm) > 0) or np.all(np.diff(s.rpm) < 0))  # up and then down
    for call in (lambda: lines.order_slice(s), lambda: lines.order_slice(s, x="rpm"),
                 lambda: lines.frequency_slice(compute_frequency_slice(_fmap(updown), 1000.0))):
        ax = call()
        assert ax.get_xlabel() == "Speed (rpm)"
    ax = lines.order_slice(s)
    np.testing.assert_array_equal(_xdata(ax), s.rpm)  # the true frame speeds, in frame order
    np.testing.assert_array_equal(_ydata(ax), to_db(s.amplitude, 1.0))
    assert _xdata(ax).argmax() not in (0, s.rpm.size - 1)  # the line goes to the maximum and back
    ax = lines.order_slice(s, x="time", **SOUND_PRESSURE)
    np.testing.assert_array_equal(_xdata(ax), s.t)
    np.testing.assert_allclose(_ydata(ax)[MID], _exp_db(A2), atol=0.02)


def test_x_rpm_of_a_measured_speed_with_reversals_is_drawn_as_it_is():
    sig = _signal(_runup_rpm())
    rng = np.random.default_rng(3)
    s = order_slice(_omap(sig), 2.0)
    jittery = dataclasses.replace(s, rpm=s.rpm + rng.normal(0.0, 0.01 * (R1 - R0), s.rpm.size))
    assert np.any(np.diff(jittery.rpm) < 0)  # reversals
    ax = lines.order_slice(jittery)
    np.testing.assert_array_equal(_xdata(ax), jittery.rpm)


def test_nan_frames_and_zero_amplitudes_stay_gaps(fmap, slice2):
    # order 30 leaves the 1500 Hz map at the end of the run-up: those frames are NaN, the rest not
    s = order_slice(fmap, 30.0)
    assert np.isnan(s.amplitude).any() and np.isfinite(s.amplitude).any()
    for scale in ("db", "linear"):
        y = _ydata(lines.order_slice(s, scale=scale))
        assert np.array_equal(np.isnan(y), np.isnan(s.amplitude))
    # a zero amplitude has no level: a gap in dB, a zero in the linear scale
    amp = slice2.amplitude.copy()
    amp[10:13] = 0.0
    z = dataclasses.replace(slice2, amplitude=amp)
    y = _ydata(lines.order_slice(z))
    assert np.isnan(y[10:13]).all() and np.isfinite(np.delete(y, np.arange(10, 13))).all()
    np.testing.assert_array_equal(_ydata(lines.order_slice(z, scale="linear"))[10:13], 0.0)


def test_truth_is_drawn_dashed_in_the_same_scale_and_reference(runup, slice2):
    rpm = np.linspace(R0, R1, 7)
    truth = (rpm, np.full(7, A2 / np.sqrt(2.0)))
    ax = lines.order_slice(slice2, **SOUND_PRESSURE, truth=truth)
    (tl,) = _by_label(ax, "truth")
    assert tl.get_linestyle() == "--" and _legend(ax) == ["truth"]
    assert matplotlib.colors.to_hex(tl.get_color()) == "#404040"  # neutral dark grey
    np.testing.assert_allclose(tl.get_xdata(), rpm)
    np.testing.assert_allclose(tl.get_ydata(), to_db(truth[1], P_REF), rtol=1e-12)
    # the truth passes through the drawn level
    y = np.interp(rpm, _xdata(ax), _ydata(ax))
    np.testing.assert_allclose(y[1:-1], tl.get_ydata()[1:-1], atol=0.02)
    # linear scale, time axis, NaN stays a gap
    t = np.linspace(0.0, DUR, 5)
    ax = lines.order_slice(slice2, x="time", scale="linear", truth=(t, [1.0, np.nan, 2.0, 0.0, 3.0]))
    ty = np.asarray(_by_label(ax, "truth")[0].get_ydata(), dtype=float)
    np.testing.assert_array_equal(ty, [1.0, np.nan, 2.0, 0.0, 3.0])
    ax = lines.order_slice(slice2, truth=(t, [1.0, 0.0, 2.0, 4.0, 5.0]))
    assert np.isnan(np.asarray(_by_label(ax, "truth")[0].get_ydata(), dtype=float)[1])  # zero: no level
    ax = lines.order_slice({"a": slice2}, truth=(t, np.ones(5)))
    assert _legend(ax) == ["a", "truth"]


def _legend_texts(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def test_truth_is_drawn_above_the_data_and_thinner(slice2, fmap):
    rpm = np.linspace(R0, R1, 7)
    ax = lines.order_slice(slice2, truth=(rpm, np.full(7, A2 / np.sqrt(2.0))))
    data = ax.lines[0]  # the unlabelled data line, drawn first
    (tl,) = _by_label(ax, "truth")
    assert tl.get_zorder() > data.get_zorder()  # the truth never vanishes under an identical estimate
    assert tl.get_linewidth() < data.get_linewidth()
    # the draw order of the artists on the Axes follows zorder: the truth is painted last
    ordered = sorted(ax.get_children(), key=lambda a: a.get_zorder())
    assert ordered.index(tl) > ordered.index(data)
    # frequency slices draw the truth the same way
    fs = compute_frequency_slice(fmap, 1000.0)
    ax = lines.frequency_slice({"tone": fs}, truth=(rpm, np.full(7, A_TONE / np.sqrt(2.0))))
    (data,) = _by_label(ax, "tone")
    (tl,) = _by_label(ax, "truth")
    assert tl.get_zorder() > data.get_zorder() and tl.get_linewidth() < data.get_linewidth()
    assert tl.get_linestyle() == "--"


def test_truth_is_drawn_like_the_reference_of_the_speed_figures(slice2):
    from rota.plot import _analysis as analysis

    t = np.linspace(0.0, DUR, 5)
    ax = lines.order_slice(slice2, x="time", truth=(t, np.ones(5)))
    (tl,) = _by_label(ax, "truth")
    ax2 = analysis.rpm_vs_time(t, np.ones(5), reference=(t, np.ones(5)))
    (ref,) = _by_label(ax2, "reference")
    for prop in ("get_color", "get_linestyle", "get_linewidth", "get_zorder"):
        assert getattr(tl, prop)() == getattr(ref, prop)(), prop


def test_truth_dict_gives_one_curve_per_entry_in_a_darker_shade_of_its_colour(slice2, slice2_louder, fmap):
    rpm = np.linspace(R0, R1, 7)
    truth = {"base": (rpm, np.full(7, A2 / np.sqrt(2.0))), "louder": (rpm, np.full(7, _gain() * A2 / np.sqrt(2.0)))}
    ax = lines.order_slice({"base": slice2, "louder": slice2_louder}, **SOUND_PRESSURE, truth=truth)
    assert _legend(ax) == ["base", "louder", "base truth", "louder truth"]  # one clear entry per truth
    for key in ("base", "louder"):
        (data,) = _by_label(ax, key)
        (tl,) = _by_label(ax, f"{key} truth")
        base = np.array(matplotlib.colors.to_rgb(data.get_color()))
        np.testing.assert_allclose(tl.get_color(), 0.65 * base)  # 35 % towards black: darker than the line, same hue
        assert np.all(np.asarray(matplotlib.colors.to_rgb(tl.get_color())) <= base)
        assert tl.get_linestyle() == "--" and tl.get_zorder() > data.get_zorder()
        assert tl.get_linewidth() < data.get_linewidth()
        np.testing.assert_allclose(tl.get_ydata(), to_db(truth[key][1], P_REF), rtol=1e-12)
    assert _legend_texts(ax) == _legend(ax)
    # a subset of the entries is fine; a single pair is the neutral "truth" next to a dict
    ax = lines.order_slice({"base": slice2, "louder": slice2_louder}, truth={"louder": truth["louder"]})
    assert _legend(ax) == ["base", "louder", "louder truth"]
    ax = lines.order_slice({"base": slice2, "louder": slice2_louder}, truth=truth["base"])
    assert _legend(ax) == ["base", "louder", "truth"]
    # the same for frequency slices
    fsl = compute_frequency_slice(fmap, 1000.0)
    ax = lines.frequency_slice({"tone": fsl}, truth={"tone": (rpm, np.full(7, A_TONE / np.sqrt(2.0)))})
    assert _legend(ax) == ["tone", "tone truth"]


def test_truth_dict_errors(slice2, slice2_louder):
    pair = ([1.0, 2.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="not one of the labels"):
        lines.order_slice({"a": slice2}, truth={"b": pair})
    with pytest.raises(ValueError, match="needs labelled results"):
        lines.order_slice([slice2, slice2_louder], truth={"a": pair})
    with pytest.raises(ValueError, match="empty"):
        lines.order_slice({"a": slice2}, truth={})
    with pytest.raises(ValueError, match="truth"):
        lines.order_slice({"a": slice2}, truth={"a": ([1.0, 2.0], [1.0])})
    with pytest.raises(ValueError, match="truth cannot be combined"):
        lines.order_slice({"a": slice2, "b": slice2_louder}, relative_to="a", truth={"b": pair})
    assert plt.get_fignums() == []


def test_the_same_truth_label_drawn_by_several_calls_is_listed_once(slice2, slice2_louder):
    t = np.linspace(0.0, DUR, 5)
    _fig, ax = plt.subplots()
    lines.order_slice({"one": slice2}, ax=ax, x="time", truth=(t, np.ones(5)))
    lines.order_slice({"two": slice2_louder}, ax=ax, x="time", truth=(t, 2.0 * np.ones(5)))
    assert len(_by_label(ax, "truth")) == 2  # both curves are drawn ...
    assert _legend_texts(ax) == ["one", "two", "truth"]  # ... and listed once, after the data
    # data lines with equal labels are different lines: they are not merged
    _fig, ax = plt.subplots()
    lines.order_slice({"x": slice2}, ax=ax)
    lines.order_slice({"x": slice2_louder}, ax=ax)
    assert _legend_texts(ax) == ["x", "x"]


# --------------------------------------------------------------------------
# One result, lists, dicts, relative_to
# --------------------------------------------------------------------------


def test_list_and_dict_inputs(slice2, slice2_louder):
    ax = lines.order_slice([slice2, slice2_louder])
    assert len(ax.lines) == 2 and ax.get_legend() is None
    ax = lines.order_slice({"before": slice2, "after": slice2_louder}, **SOUND_PRESSURE)
    assert len(ax.lines) == 2 and _legend(ax) == ["before", "after"]
    d = _ydata(ax, 1) - _ydata(ax, 0)
    np.testing.assert_allclose(d[MID], GAIN_DB, atol=0.02)
    ax = lines.order_slice((slice2, slice2))
    assert len(ax.lines) == 2


@pytest.mark.parametrize("scale", ["db", "linear"])
def test_relative_to_gives_the_known_change_after_a_gain(slice2, slice2_louder, scale):
    entries = {"before": slice2, "after": slice2_louder}
    ax = lines.order_slice(entries, relative_to="before", scale=scale, **SOUND_PRESSURE)
    before, after = _ydata(ax, 0), _ydata(ax, 1)
    assert np.all(before[np.isfinite(before)] == 0.0)  # the reference itself is the zero line
    if scale == "db":
        np.testing.assert_allclose(after[MID], GAIN_DB, atol=0.02)  # +6 dB after the gain
        assert ax.get_ylabel() == "Order level relative to before (dB)"
    else:
        np.testing.assert_allclose(
            after, slice2_louder.amplitude - slice2.amplitude, rtol=1e-12, atol=1e-15
        )
        np.testing.assert_allclose(after[MID], A2 * (_gain() - 1.0) / np.sqrt(2.0), rtol=2e-3)
        assert ax.get_ylabel() == "Order amplitude minus before (Pa, RMS)"
    assert ax.get_xlabel() == "Speed (rpm)"


def test_relative_to_gives_an_exact_known_change_for_a_scaled_slice(slice2):
    entries = {"a": slice2, "b": dataclasses.replace(slice2, amplitude=slice2.amplitude * 10 ** (-3.0 / 20))}
    ax = lines.order_slice(entries, relative_to="a")
    np.testing.assert_allclose(_ydata(ax, 1)[MID], -3.0, atol=1e-9)


def test_relative_to_errors(runup, slice2, slice2_louder):
    entries = {"before": slice2, "after": slice2_louder}
    with pytest.raises(ValueError, match="not one of the labels"):
        lines.order_slice(entries, relative_to="missing")
    with pytest.raises(ValueError, match="needs labelled results"):
        lines.order_slice([slice2, slice2_louder], relative_to="before")
    with pytest.raises(ValueError, match="needs labelled results"):
        lines.order_slice(slice2, relative_to="before")
    other = order_slice(_omap(_signal(_runup_rpm(stop=2800.0))), 2.0)  # another run-up: other frame speeds
    with pytest.raises(ValueError, match="x grids differ"):
        lines.order_slice({"a": slice2, "b": other}, relative_to="a")
    short = dataclasses.replace(
        slice2, t=slice2.t[:-1], rpm=slice2.rpm[:-1], amplitude=slice2.amplitude[:-1]
    )
    with pytest.raises(ValueError, match="x grids differ"):
        lines.order_slice({"a": slice2, "b": short}, relative_to="a")
    with pytest.raises(ValueError, match="truth cannot be combined"):
        lines.order_slice(entries, relative_to="before", truth=([1.0, 2.0], [1.0, 1.0]))
    assert plt.get_fignums() == []


def test_mixed_amplitude_conventions_are_refused(runup):
    peak = order_slice(_omap(runup, "peak"), 2.0)
    rms = order_slice(_omap(runup, "rms"), 2.0)
    with pytest.raises(ValueError, match="amplitude conventions"):
        lines.order_slice({"p": peak, "r": rms})
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------
# Multichannel
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_channel(runup):
    x2 = np.column_stack([runup.x, 2.0 * runup.x])
    return _omap(runup, x=x2)


def test_channel_selects_the_channel(two_channel):
    s = order_slice(two_channel, 2.0)
    assert s.amplitude.shape[1] == 2
    y0 = _ydata(lines.order_slice(s, **SOUND_PRESSURE))
    y1 = _ydata(lines.order_slice(s, **SOUND_PRESSURE, channel=1))
    np.testing.assert_allclose(y0, to_db(s.amplitude[:, 0], P_REF), rtol=1e-12)
    np.testing.assert_allclose(y1, to_db(s.amplitude[:, 1], P_REF), rtol=1e-12)
    np.testing.assert_allclose((y1 - y0)[MID], 20.0 * np.log10(2.0), atol=1e-6)  # channel 1 is twice channel 0
    np.testing.assert_allclose(_ydata(lines.order_slice(s, **SOUND_PRESSURE, channel=-1)), y1)
    np.testing.assert_allclose(_ydata(lines.order_slice(s, scale="linear", channel=1)), s.amplitude[:, 1])
    f = compute_frequency_slice(two_channel, 3.0 * R0 / 60.0 * 2.0)  # frequency slices keep the channel axis
    assert f.amplitude.ndim == 2
    np.testing.assert_allclose(_ydata(lines.frequency_slice(f, channel=1)), to_db(f.amplitude[:, 1], 1.0), rtol=1e-12)
    plt.close("all")
    with pytest.raises(IndexError, match="2 channels"):
        lines.order_slice(s, channel=2)
    with pytest.raises(TypeError, match="integer"):
        lines.order_slice(s, channel="a")
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------
# order_spectrum
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spec(omap):
    return order_spectrum(omap)


@pytest.fixture(scope="module")
def spec_louder(louder):
    return order_spectrum(_omap(louder))


def _bin(axis, value):
    return int(np.argmin(np.abs(np.asarray(axis) - value)))


def test_order_spectrum_draws_the_analytic_levels_of_known_orders(spec):
    ax = lines.order_spectrum(spec, **SOUND_PRESSURE)
    np.testing.assert_array_equal(_xdata(ax), spec.axis)
    y = _ydata(ax)
    np.testing.assert_allclose(y, to_db(spec.amplitude, P_REF), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y[_bin(spec.axis, 2.0)], _exp_db(A2), atol=0.01)  # on bins: no scalloping
    np.testing.assert_allclose(y[_bin(spec.axis, 4.0)], _exp_db(A4), atol=0.01)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Order", f"Spectrum level (dB re 20 {MICRO}Pa, RMS)")
    ax = lines.order_spectrum(spec, scale="linear", unit="Pa")
    np.testing.assert_array_equal(_ydata(ax), spec.amplitude)
    assert ax.get_ylabel() == "Spectrum amplitude (Pa, RMS)"


def test_order_spectrum_of_peak_amplitudes_says_peak(runup):
    sp = order_spectrum(_omap(runup, "peak"))
    ax = lines.order_spectrum(sp, **SOUND_PRESSURE)
    assert ax.get_ylabel() == f"Spectrum level (dB re 20 {MICRO}Pa, peak)"
    np.testing.assert_allclose(_ydata(ax)[_bin(sp.axis, 2.0)], _exp_db(A2, "peak"), atol=0.01)


def test_frequency_spectrum_is_drawn_against_frequency(fmap):
    sp = order_spectrum(fmap)
    ax = lines.order_spectrum(sp)
    assert ax.get_xlabel() == "Frequency (Hz)"
    np.testing.assert_allclose(_ydata(ax)[_bin(sp.axis, 1000.0)], 20.0 * np.log10(A_TONE / np.sqrt(2.0)), atol=0.01)
    plt.close("all")
    with pytest.raises(ValueError, match="order map"):
        lines.order_spectrum(sp, mark_orders=(2, 4))
    assert plt.get_fignums() == []


def test_order_spectrum_before_after_and_relative_to(spec, spec_louder):
    ax = lines.order_spectrum({"before": spec, "after": spec_louder}, relative_to="before")
    after = _ydata(ax, 1)
    np.testing.assert_allclose(after[_bin(spec.axis, 2.0)], GAIN_DB, atol=0.01)  # order 2 gained 6 dB
    np.testing.assert_allclose(after[_bin(spec.axis, 4.0)], 0.0, atol=0.01)  # order 4 did not change
    assert ax.get_ylabel() == "Spectrum level relative to before (dB)"
    assert _legend(ax) == ["before", "after"]
    other = dataclasses.replace(spec, axis=spec.axis + 0.1)
    with pytest.raises(ValueError, match="x grids differ"):
        lines.order_spectrum({"a": spec, "b": other}, relative_to="a")
    with pytest.raises(ValueError, match="domains"):
        lines.order_spectrum([spec, dataclasses.replace(spec, domain="frequency")])


def test_orders_are_marked_inside_the_spectrum_only(spec):
    ax = lines.order_spectrum(spec, mark_orders=(2, 4, 20))  # 20 is above the highest order of the map
    vertical = [ln for ln in ax.lines if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == ln.get_xdata()[1]]
    assert sorted(float(ln.get_xdata()[0]) for ln in vertical) == [2.0, 4.0]
    assert sorted(a.get_text() for a in ax.texts) == ["2", "4"]
    assert ax.get_xlim()[1] < 20.0  # the axis was not stretched to the missing order
    ax = lines.order_spectrum(spec, mark_orders=3)
    assert [a.get_text() for a in ax.texts] == ["3"]
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="mark_orders"):
        lines.order_spectrum(spec, mark_orders=(0, 2))
    assert plt.get_fignums() == before


def test_spectrum_channel_and_gaps(two_channel):
    sp = order_spectrum(two_channel)
    y0 = _ydata(lines.order_spectrum(sp))
    y1 = _ydata(lines.order_spectrum(sp, channel=1))
    np.testing.assert_allclose(y1 - y0, 20.0 * np.log10(2.0), atol=1e-6)
    amp = sp.amplitude.copy()
    amp[5, 0] = 0.0
    y = _ydata(lines.order_spectrum(dataclasses.replace(sp, amplitude=amp)))
    assert np.isnan(y[5]) and np.isfinite(np.delete(y, 5)).all()


# --------------------------------------------------------------------------
# overall_level
# --------------------------------------------------------------------------

L_TONE = 20.0 * np.log10(A_TONE / np.sqrt(2.0) / 20e-6)  # RMS level of the 1000 Hz tone, dB re 20 uPa
EDGES = np.array([1000.0, 1400.0, 1800.0, 2200.0, 2600.0, 3001.0, 3500.0, 4000.0])  # the last two bins are empty


@pytest.fixture(scope="module")
def tone_only():
    rpm = _runup_rpm()
    return synth_runup(FS, rpm, [], tones=[{"freq": 1000.0, "amplitude": A_TONE}], noise_rms=0.0, seed=1)


@pytest.fixture(scope="module")
def olevel(tone_only):
    return overall_level(tone_only.x, FS, tone_only.rpm, rpm_bins=EDGES)


def test_overall_level_draws_the_stored_level_per_speed_bin(olevel):
    ax = lines.overall_level(olevel)
    np.testing.assert_array_equal(_xdata(ax), olevel.centres)
    y = _ydata(ax)
    filled = olevel.counts > 0
    np.testing.assert_array_equal(y[filled], olevel.binned_level[filled])  # drawn as stored
    np.testing.assert_allclose(y[filled], L_TONE, atol=0.01)  # a constant tone: its RMS level in every bin
    assert np.array_equal(np.isnan(y), ~filled) and (~filled).sum() == 2  # empty bins are gaps
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Speed (rpm)", "Level (dB re 2e-05, RMS)")  # the stored ref, no unit
    assert _line(ax).get_marker() == "o"
    assert lines.overall_level(olevel, **SOUND_PRESSURE).get_ylabel() == f"Level (dB re 20 {MICRO}Pa, RMS)"
    assert lines.overall_level(olevel, unit="Pa").get_ylabel() == "Level (dB re 20 \u00b5Pa, RMS)"  # unit only labels
    assert lines.overall_level(olevel, unit="m/s²").get_ylabel() == f"Level (dB re 20 {MICRO}m/s², RMS)"


def test_overall_level_names_the_weighting(tone_only):
    for w in ("A", "C"):
        ol = overall_level(tone_only.x, FS, tone_only.rpm, rpm_bins=EDGES, weighting=w)
        ax = lines.overall_level(ol, **SOUND_PRESSURE)
        assert ax.get_ylabel() == f"{w}-weighted level (dB re 20 {MICRO}Pa, RMS)"
        filled = ol.counts > 0
        np.testing.assert_array_equal(_ydata(ax)[filled], ol.binned_level[filled])  # identical numbers: stored ref 2e-5
    ol_a = overall_level(tone_only.x, FS, tone_only.rpm, rpm_bins=EDGES, weighting="A")
    ol_c = overall_level(tone_only.x, FS, tone_only.rpm, rpm_bins=EDGES, weighting="C")
    assert lines.overall_level({"A": ol_a, "C": ol_c}, **SOUND_PRESSURE).get_ylabel() == f"Level (dB re 20 {MICRO}Pa, RMS)"


def test_overall_level_frames(tone_only, olevel):
    ol = overall_level(tone_only.x, FS, tone_only.rpm)
    ax = lines.overall_level(ol, binned=False)
    np.testing.assert_array_equal(_xdata(ax), ol.t)
    np.testing.assert_allclose(_ydata(ax), L_TONE, atol=0.01)
    assert ax.get_xlabel() == "Time (s)" and _line(ax).get_marker() == "None"
    with pytest.raises(ValueError, match="no speed bins"):
        lines.overall_level(ol)  # binned=True needs rpm_bins
    with pytest.raises(ValueError, match="no speed bins"):
        lines.overall_level({"a": ol}, binned=True)
    # a silent frame (-inf) is a gap
    level = ol.level.copy()
    level[4] = -np.inf
    y = _ydata(lines.overall_level(dataclasses.replace(ol, level=level), binned=False))
    assert np.isnan(y[4]) and np.isfinite(np.delete(y, 4)).all()
    assert len(plt.get_fignums()) == 2  # the two calls that succeeded


def test_overall_level_relative_to_and_references(olevel):
    louder_by_3 = dataclasses.replace(olevel, binned_level=olevel.binned_level + 3.0)
    ax = lines.overall_level({"a": olevel, "b": louder_by_3}, relative_to="a")
    filled = olevel.counts > 0
    np.testing.assert_allclose(_ydata(ax, 1)[filled], 3.0, rtol=1e-12)
    assert np.isnan(_ydata(ax, 1)[~filled]).all()
    assert ax.get_ylabel() == "Level relative to a (dB)"
    with pytest.raises(ValueError, match="different references"):
        lines.overall_level([olevel, dataclasses.replace(olevel, ref=1e-6)])
    other_bins = dataclasses.replace(olevel, centres=olevel.centres + 5.0)
    with pytest.raises(ValueError, match="x grids differ"):
        lines.overall_level({"a": olevel, "b": other_bins}, relative_to="a")


def test_overall_level_ref_none_plots_as_stored_and_a_given_ref_re_references_exactly(olevel):
    assert olevel.ref == 2e-5
    filled = olevel.counts > 0
    stored = _ydata(lines.overall_level(olevel))
    np.testing.assert_array_equal(stored[filled], olevel.binned_level[filled])
    # the stored reference given explicitly: the same numbers, bit for bit
    same = _ydata(lines.overall_level(olevel, **SOUND_PRESSURE))
    np.testing.assert_array_equal(same, stored)
    # 1 Pa: L re 1 Pa = L re 20 uPa - 20 log10(1 / 2e-5) = L - 93.979...
    ax = lines.overall_level(olevel, ref=1.0, unit="Pa")
    shift = 20.0 * np.log10(olevel.ref / 1.0)
    np.testing.assert_allclose(_ydata(ax)[filled], olevel.binned_level[filled] + shift, rtol=0, atol=1e-12)
    assert shift == pytest.approx(-93.97940008672037)
    assert ax.get_ylabel() == "Level (dB re 1 Pa, RMS)"  # labelled with the given ref, not the stored one
    # the analytic truth: a tone of peak A_TONE has RMS A_TONE / sqrt(2) Pa
    np.testing.assert_allclose(_ydata(ax)[filled], 20.0 * np.log10(A_TONE / np.sqrt(2.0)), atol=0.01)
    # frames and a silent frame
    level = olevel.level.copy()
    level[2] = -np.inf
    ax = lines.overall_level(dataclasses.replace(olevel, level=level), ref=1e-6, binned=False)
    y = _ydata(ax)
    assert np.isnan(y[2])
    np.testing.assert_allclose(np.delete(y, 2), np.delete(level, 2) + 20.0 * np.log10(2e-5 / 1e-6), atol=1e-12)


def test_overall_level_given_ref_lets_results_with_different_stored_references_share_an_axis(olevel):
    in_pascal = dataclasses.replace(olevel, ref=1.0, binned_level=olevel.binned_level - 20.0 * np.log10(1.0 / 2e-5))
    with pytest.raises(ValueError, match="different references.*ref="):
        lines.overall_level([olevel, in_pascal])
    ax = lines.overall_level({"a": olevel, "b": in_pascal}, **SOUND_PRESSURE)
    filled = olevel.counts > 0
    np.testing.assert_allclose(_ydata(ax, 1)[filled], _ydata(ax, 0)[filled], atol=1e-9)  # the same sound
    ax = lines.overall_level({"a": olevel, "b": in_pascal}, ref=1.0, relative_to="a")  # differences: ref cancels
    np.testing.assert_allclose(_ydata(ax, 1)[filled], 0.0, atol=1e-9)
    for bad in (0.0, -1.0, np.nan, np.inf, "x"):
        with pytest.raises(ValueError, match="ref"):
            lines.overall_level(olevel, ref=bad)
    assert lines.overall_level(olevel, **SOUND_PRESSURE) is not None  # and a valid ref still works


def test_overall_level_channel(tone_only):
    x2 = np.column_stack([tone_only.x, 2.0 * tone_only.x])
    ol = overall_level(x2, FS, tone_only.rpm, rpm_bins=EDGES)
    y0 = _ydata(lines.overall_level(ol))
    y1 = _ydata(lines.overall_level(ol, channel=1))
    filled = ol.counts > 0
    np.testing.assert_allclose((y1 - y0)[filled], 20.0 * np.log10(2.0), atol=1e-9)
    with pytest.raises(IndexError):
        lines.overall_level(ol, channel=2)


# --------------------------------------------------------------------------
# metric_vs_rpm
# --------------------------------------------------------------------------

SLOPE = 0.01  # the synthetic metric is SLOPE * rpm, in "sone"


def _metric_inputs(gap=True, scale=1.0):
    t = _t()
    rpm = _runup_rpm()
    tm = np.arange(0.0, DUR, 0.005)
    values = scale * SLOPE * np.interp(tm, t, rpm)
    if gap:
        values[(tm > 1.0) & (tm < 1.4)] = np.nan  # no samples in the bin at 1900 rpm
    return values, tm, rpm, t


@pytest.fixture(scope="module")
def mvr():
    values, tm, rpm, t = _metric_inputs()
    r = metric_vs_rpm(values, tm, rpm, t, bin_width=200.0, min_count=3)
    return dataclasses.replace(r, name="loudness", unit="sone", source="Toolbox 1.0, test edition")


@pytest.fixture(scope="module")
def mvr_median():
    values, tm, rpm, t = _metric_inputs()
    r = metric_vs_rpm(values, tm, rpm, t, bin_width=200.0, min_count=3, aggregate="median")
    return dataclasses.replace(r, name="loudness", unit="sone")


def _band_points(ax, label_part):
    """(x, y) points, rounded, of the polygons of the band whose label contains ``label_part``."""
    out = set()
    for coll in ax.collections:
        if label_part in coll.get_label():
            for p in coll.get_paths():
                out.update((round(float(x), 6), round(float(y), 6)) for x, y in p.vertices)
    return out


def test_metric_vs_rpm_draws_value_per_bin_and_hides_invalid_bins(mvr):
    assert (~mvr.valid).sum() == 1 and mvr.valid.sum() >= 5
    ax = lines.metric_vs_rpm(mvr)
    np.testing.assert_array_equal(_xdata(ax), mvr.centres)
    y = _ydata(ax)
    assert np.array_equal(np.isnan(y), ~mvr.valid)  # invalid bin: a gap
    np.testing.assert_array_equal(y[mvr.valid], mvr.value[mvr.valid])
    # the metric is SLOPE * rpm, so the mean of a bin that is fully populated is SLOPE * its centre
    inner = mvr.valid.copy()
    inner[[0, -1]] = False
    inner[np.flatnonzero(~mvr.valid) + 1] = False  # the bin next to the gap is partly empty
    np.testing.assert_allclose(y[inner], SLOPE * mvr.centres[inner], atol=0.03)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Speed (rpm)", "loudness (sone)")
    assert _legend(ax) == ["Toolbox 1.0, test edition", "Toolbox 1.0, test edition (\u00b1 std)"]


def test_metric_vs_rpm_spread_band_is_the_std_or_half_the_iqr(mvr, mvr_median):
    assert mvr.spread_kind == "std" and mvr_median.spread_kind == "iqr"
    for r, text, half in ((mvr, "std", mvr.std), (mvr_median, "IQR", 0.5 * mvr_median.iqr)):
        ax = lines.metric_vs_rpm({"run": r})
        points = _band_points(ax, text)
        assert points, "no band drawn"
        band_text = PM_STD if text == "std" else "IQR"
        assert _legend(ax) == ["run", f"run ({band_text})"]
        for c, v, h in zip(r.centres[r.valid], r.value[r.valid], half[r.valid]):
            assert (round(float(c), 6), round(float(v - h), 6)) in points
            assert (round(float(c), 6), round(float(v + h), 6)) in points
        # the width of the IQR of SLOPE * rpm over a full bin of 200 rpm is half of the bin: 1 sone
        full = r.valid & (r.count == r.count.max())
        if r is mvr_median:
            np.testing.assert_allclose(r.iqr[full], SLOPE * 200.0 / 2.0, rtol=0.05)
        else:
            np.testing.assert_allclose(r.std[full], SLOPE * 200.0 / np.sqrt(12.0), rtol=0.03)
    ax = lines.metric_vs_rpm(mvr, spread=False)
    assert not ax.collections and _legend(ax) == ["Toolbox 1.0, test edition"]


def test_metric_vs_rpm_without_name_unit_or_source():
    values, tm, rpm, t = _metric_inputs(gap=False)
    r = metric_vs_rpm(values, tm, rpm, t, bin_width=200.0)  # name, unit and source are empty
    ax = lines.metric_vs_rpm(r, spread=False)
    assert ax.get_ylabel() == "Metric value" and ax.get_legend() is None
    assert lines.metric_vs_rpm(dataclasses.replace(r, unit="sone"), spread=False).get_ylabel() == "Metric value (sone)"
    assert lines.metric_vs_rpm(dataclasses.replace(r, name="N"), spread=False).get_ylabel() == "N"
    assert lines.metric_vs_rpm(r, spread=False, control_unit="km/h").get_xlabel() == "Speed (km/h)"


def test_a_metric_name_that_ends_with_its_unit_in_parentheses_is_not_given_it_twice(mvr, tmetric):
    named = dataclasses.replace(mvr, name="Loudness (sone)", unit="sone")
    assert lines.metric_vs_rpm(named, spread=False).get_ylabel() == "Loudness (sone)"
    ax = lines.metric_vs_rpm({"a": named, "b": named}, relative_to="a", spread=False)
    assert ax.get_ylabel() == "Loudness (sone) relative to a"
    assert lines.metric_vs_rpm(dataclasses.replace(mvr, name="Loudness", unit="sone"), spread=False).get_ylabel() == (
        "Loudness (sone)"
    )
    tm = dataclasses.replace(tmetric, name="Sharpness (acum)", unit="acum")
    assert lines.time_metric(tm).get_ylabel() == "Sharpness (acum)"
    assert lines.time_metric(dataclasses.replace(tm, name="Sharpness")).get_ylabel() == "Sharpness (acum)"


def test_metric_vs_rpm_dict_list_and_relative_to(mvr):
    plus = dataclasses.replace(mvr, value=mvr.value + 0.5, source="")
    ax = lines.metric_vs_rpm({"a": mvr, "b": plus}, spread=False)
    assert _legend(ax) == ["a", "b"]
    np.testing.assert_allclose((_ydata(ax, 1) - _ydata(ax, 0))[mvr.valid], 0.5, rtol=1e-9)
    ax = lines.metric_vs_rpm({"a": mvr, "b": plus}, relative_to="a")
    y = _ydata(ax, 1)
    np.testing.assert_allclose(y[mvr.valid], 0.5, rtol=1e-9)  # the difference, in the unit of the metric
    assert np.isnan(y[~mvr.valid]).all()
    assert ax.get_ylabel() == "loudness relative to a (sone)"
    assert not ax.collections  # the spread of a difference is not known: no band
    ax = lines.metric_vs_rpm([mvr, plus])
    assert len(ax.lines) == 2
    with pytest.raises(ValueError, match="x grids differ"):
        values, tm, rpm, t = _metric_inputs()
        other = metric_vs_rpm(values, tm, rpm, t, bin_width=250.0, min_count=3)
        lines.metric_vs_rpm({"a": mvr, "b": dataclasses.replace(other, name="loudness", unit="sone")}, relative_to="a")
    with pytest.raises(ValueError, match="different units"):
        lines.metric_vs_rpm([mvr, dataclasses.replace(mvr, unit="acum")])
    with pytest.raises(ValueError, match="spread_kind"):
        lines.metric_vs_rpm(dataclasses.replace(mvr, spread_kind="mad"))
    assert len(plt.get_fignums()) == 3  # only the three calls that succeeded left a figure


# --------------------------------------------------------------------------
# time_metric
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tmetric():
    values, tm, _rpm, _t_rpm = _metric_inputs()
    return TimeMetric(name="loudness", t=tm, values=values, unit="sone", skip_start=0.6, source="Toolbox 1.0")


def _span_x(ax):
    """x extent in data units of the shaded start-up span (a Rectangle or a Polygon, by version)."""
    (patch,) = [p for p in ax.patches if p.get_label() == "start-up excluded"]
    x = patch.get_patch_transform().transform(patch.get_path().vertices)[:, 0]
    return float(x.min()), float(x.max())


def test_time_metric_draws_values_against_time_and_shades_the_startup(tmetric):
    ax = lines.time_metric(tmetric)
    np.testing.assert_array_equal(_xdata(ax), tmetric.t)
    y = _ydata(ax)
    np.testing.assert_array_equal(y, tmetric.values)  # NaN stays NaN
    assert np.isnan(y).any() and np.isfinite(y).any()
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Time (s)", "loudness (sone)")
    assert _legend(ax) == ["Toolbox 1.0", "start-up excluded"]
    lo, hi = _span_x(ax)
    assert lo == pytest.approx(tmetric.t[0]) and hi == pytest.approx(tmetric.t[0] + 0.6)  # counted from t[0]
    ax.figure.canvas.draw()
    y0, y1 = ax.get_ylim()
    assert y0 > -1.0 and y1 < 40.0  # the shaded span does not stretch the y axis


def test_skip_start_is_counted_from_the_first_time():
    t = np.arange(0.0, 3.0, 0.01) + 0.25
    m = TimeMetric(name="n", t=t, values=np.ones_like(t), unit="sone", skip_start=0.5)
    lo, hi = _span_x(lines.time_metric(m))
    assert lo == pytest.approx(0.25) and hi == pytest.approx(0.75)
    # no start-up exclusion: no span and no legend; one beyond the data is clipped to the data
    ax = lines.time_metric(dataclasses.replace(m, skip_start=0.0))
    assert not ax.patches and ax.get_legend() is None
    assert _span_x(lines.time_metric(dataclasses.replace(m, skip_start=10.0)))[1] == pytest.approx(t[-1])


def test_time_metric_dict_relative_to_and_errors(tmetric):
    plus = dataclasses.replace(tmetric, values=tmetric.values + 0.25, source="")
    ax = lines.time_metric({"a": tmetric, "b": plus}, relative_to="a")
    y = _ydata(ax, 1)
    np.testing.assert_allclose(y[np.isfinite(tmetric.values)], 0.25, rtol=1e-9)
    assert np.isnan(y[np.isnan(tmetric.values)]).all()
    assert ax.get_ylabel() == "loudness relative to a (sone)"
    assert _legend(ax) == ["a", "b", "start-up excluded"] and len(ax.patches) == 1
    shifted = dataclasses.replace(tmetric, t=tmetric.t + 0.5)
    with pytest.raises(ValueError, match="x grids differ"):
        lines.time_metric({"a": tmetric, "b": shifted}, relative_to="a")
    with pytest.raises(ValueError, match="different units"):
        lines.time_metric([tmetric, dataclasses.replace(tmetric, unit="acum")])
    with pytest.raises(ValueError, match="not one of the labels"):
        lines.time_metric({"a": tmetric}, relative_to="b")
    assert len(plt.get_fignums()) == 1


# --------------------------------------------------------------------------
# Errors common to all functions
# --------------------------------------------------------------------------


def test_wrong_types_are_refused_with_the_function_to_use(omap, slice2, spec, olevel, mvr, tmetric):
    fslice = compute_frequency_slice(_fmap_for_types(), 1000.0)
    wrong = [
        (lines.order_slice, omap), (lines.order_slice, fslice), (lines.order_slice, [slice2, spec]),
        (lines.order_slice, {"a": slice2, "b": "x"}), (lines.order_slice, np.ones(5)),
        (lines.frequency_slice, slice2), (lines.order_spectrum, slice2), (lines.overall_level, mvr),
        (lines.metric_vs_rpm, tmetric), (lines.time_metric, mvr), (lines.time_metric, olevel),
        (lines.order_slice, None),
    ]
    for func, obj in wrong:
        with pytest.raises(TypeError, match=func.__name__):
            func(obj)
    with pytest.raises(TypeError, match=r"entry 'b'"):
        lines.order_slice({"a": slice2, "b": spec})
    assert plt.get_fignums() == []


def _fmap_for_types():
    return rpm_frequency_map(
        _signal(_runup_rpm()).x, FS, _runup_rpm(), nperseg=400, max_freq=1500.0
    )


def test_empty_containers_and_invalid_options(slice2, spec, olevel, mvr, tmetric):
    for func in (lines.order_slice, lines.frequency_slice, lines.order_spectrum, lines.overall_level,
                 lines.metric_vs_rpm, lines.time_metric):
        for empty in ([], {}, ()):
            with pytest.raises(ValueError, match="empty"):
                func(empty)
    for bad in (0.0, -1.0, np.nan, np.inf, "x", None):
        with pytest.raises(ValueError, match="ref"):
            lines.order_slice(slice2, ref=bad)
        with pytest.raises(ValueError, match="ref"):
            lines.order_spectrum(spec, ref=bad)
    for func, obj in ((lines.order_slice, slice2), (lines.order_spectrum, spec)):
        with pytest.raises(ValueError, match="scale"):
            func(obj, scale="log")
    for bad in ("auto", "freq", None):
        with pytest.raises(ValueError, match="x must be"):
            lines.order_slice(slice2, x=bad)
    for bad in ((1.0, 2.0, 3.0), ([1.0, 2.0], [1.0]), ([[1.0, 2.0]], [[1.0, 2.0]]), ([], []), ([1.0, 2.0], ["a", "b"]),
                5.0, ([1.0], [1j])):
        with pytest.raises(ValueError, match="truth"):
            lines.order_slice(slice2, truth=bad)
    assert plt.get_fignums() == []


def test_invalid_ax(slice2):
    _fig, axes = plt.subplots(1, 2)
    with pytest.raises(TypeError, match="Axes"):
        lines.order_slice(slice2, ax=axes)
    with pytest.raises(TypeError, match="Axes"):
        lines.time_metric(TimeMetric(name="n", t=np.arange(3.0), values=np.ones(3), unit="u"), ax="x")


# --------------------------------------------------------------------------
# Composition: axes, figures, rcParams, style, show
# --------------------------------------------------------------------------


@pytest.fixture
def calls(slice2, fmap, spec, olevel, mvr, tmetric):
    fslice = compute_frequency_slice(fmap, 1000.0)
    return {
        "order_slice": lambda **kw: lines.order_slice(slice2, **kw),
        "frequency_slice": lambda **kw: lines.frequency_slice(fslice, **kw),
        "order_spectrum": lambda **kw: lines.order_spectrum(spec, mark_orders=(2, 4), **kw),
        "overall_level": lambda **kw: lines.overall_level(olevel, **kw),
        "metric_vs_rpm": lambda **kw: lines.metric_vs_rpm(mvr, **kw),
        "time_metric": lambda **kw: lines.time_metric(tmetric, **kw),
    }


NAMES = ["order_slice", "frequency_slice", "order_spectrum", "overall_level", "metric_vs_rpm", "time_metric"]


@pytest.mark.parametrize("name", NAMES)
def test_a_given_axes_is_drawn_on_and_returned_without_a_new_figure(calls, name):
    fig, ax = plt.subplots()
    out = calls[name](ax=ax)
    assert out is ax and plt.get_fignums() == [fig.number]
    assert ax.lines and ax.get_xlabel() and ax.get_ylabel()
    ax2 = calls[name]()  # without ax: a new figure
    assert ax2 is not ax and len(plt.get_fignums()) == 2


@pytest.mark.parametrize("name", NAMES)
def test_rcparams_backend_and_show_are_left_alone(calls, name, monkeypatch):
    def show(*args, **kwargs):
        raise AssertionError("plt.show() was called")

    monkeypatch.setattr(plt, "show", show)
    before = _raw_rcparams()
    backend = matplotlib.get_backend()
    calls[name]()
    assert _raw_rcparams() == before
    assert matplotlib.get_backend() == backend


def test_two_slices_can_share_an_axes_with_the_colour_cycle(slice2, slice2_louder):
    fig, ax = plt.subplots()
    lines.order_slice(slice2, ax=ax)
    lines.order_slice(slice2_louder, ax=ax)
    assert len(ax.lines) == 2 and ax.lines[0].get_color() != ax.lines[1].get_color()  # the active cycle
    assert plt.get_fignums() == [fig.number]


def test_the_active_style_is_used(slice2):
    with plt.rc_context({"axes.labelsize": 17, "lines.linewidth": 3.5}):
        ax = lines.order_slice(slice2)
        assert ax.xaxis.label.get_fontsize() == 17 and _line(ax).get_linewidth() == 3.5


def test_a_failing_call_leaves_no_figure_and_does_not_touch_a_given_axes(slice2, calls):
    with pytest.raises(ValueError):
        lines.order_slice(slice2, x="time", relative_to="missing")
    with pytest.raises(IndexError):
        lines.order_slice(slice2, channel=3)
    assert plt.get_fignums() == []
    # a failure while drawing closes the figure that was created for the call
    with pytest.raises(Exception, match="boom"):
        lines._on_axes(None, lambda ax: (_ for _ in ()).throw(RuntimeError("boom")))
    assert plt.get_fignums() == []
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        lines.order_slice(slice2, ax=ax, scale="bad")
    assert plt.get_fignums() == [fig.number] and not ax.lines  # a given Axes is neither drawn on nor closed
    with pytest.raises(Exception, match="boom"):
        lines._on_axes(ax, lambda a: (_ for _ in ()).throw(RuntimeError("boom")))
    assert plt.get_fignums() == [fig.number]


def test_no_warning_is_raised(slice2, fmap, spec, olevel, mvr, tmetric, calls):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in NAMES:
            calls[name]()
        lines.order_slice({"a": slice2, "b": slice2}, relative_to="a", scale="linear")
        lines.order_spectrum({"a": spec, "b": spec}, relative_to="a", mark_orders=2)
        lines.metric_vs_rpm({"a": mvr, "b": mvr}, relative_to="a")
        lines.order_slice(order_slice(fmap, 30.0))  # NaN frames


def test_the_module_is_exported_for_the_package():
    assert sorted(lines.__all__) == sorted(NAMES)
    for name in lines.__all__:
        doc = getattr(lines, name).__doc__
        assert all(section in doc for section in ("Parameters", "Returns", "References")), name
