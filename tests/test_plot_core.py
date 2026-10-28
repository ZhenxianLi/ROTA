"""Tests of the shared helpers of rota_order.plot and of its optional Matplotlib dependency.

The helpers are pure NumPy and run without Matplotlib; the tests that need it skip,
and the two subprocess tests (import without Matplotlib, ImportError hint) run
everywhere.
"""

import subprocess
import sys
import warnings

import numpy as np
import pytest

import rota_order.plot as rp
from rota_order.plot import _core
from rota_order.plot._core import (
    SOUND_PRESSURE,
    as_items,
    check_finite,
    check_positive,
    check_ref,
    check_tol,
    choose_x,
    colour_limits,
    edges_from_centres,
    essentially_monotonic,
    label_index,
    level_label,
    level_reference,
    ref_text,
    relative,
    select_channel,
    speed_envelope,
    to_db,
)

MICRO = "\u00b5"  # micro sign, as written by ref_text


@pytest.fixture
def plt():
    """pyplot on the Agg backend; skips without Matplotlib; closes the figures."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    yield pyplot
    pyplot.close("all")


# --------------------------------------------------------------------------
# to_db and the reference checks
# --------------------------------------------------------------------------


def test_to_db_is_twenty_log10_of_the_ratio():
    assert to_db(2e-5, 2e-5) == pytest.approx(0.0, abs=1e-12)
    assert to_db(2e-4, 2e-5) == pytest.approx(20.0)
    assert to_db(1.0, 2e-5) == pytest.approx(93.97940008672037)  # 1 Pa is 94 dB re 20 uPa
    a = np.array([[1.0, 2.0, 0.5], [4.0, 0.25, 8.0]])
    out = to_db(a, 1.0)
    assert out.shape == a.shape
    np.testing.assert_allclose(out, 20.0 * np.log10(a), rtol=1e-14)
    np.testing.assert_allclose(to_db(a, 0.5), 20.0 * np.log10(a / 0.5), rtol=1e-14)
    assert isinstance(to_db(3.0, 1.0), float)  # a scalar stays a scalar


def test_to_db_takes_the_magnitude_of_complex_and_negative_values():
    assert to_db(3.0 + 4.0j, 1.0) == pytest.approx(20.0 * np.log10(5.0))
    assert to_db(-2.0, 1.0) == pytest.approx(to_db(2.0, 1.0))
    z = np.array([1.0 + 1.0j, -2.0j, 0.5])
    np.testing.assert_allclose(to_db(z, 2.0), 20.0 * np.log10(np.abs(z) / 2.0), rtol=1e-14)


def test_to_db_gives_nan_not_minus_inf_for_zero_and_raises_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = to_db(np.array([0.0, 1.0, np.nan, np.inf, 1e-320]), 1.0)
        assert np.isnan(to_db(0.0, 1.0))
        assert np.isnan(to_db(0j, 1.0))
        np.testing.assert_array_equal(np.isnan(to_db(np.array([0, 10]), 1.0)), [True, False])  # int input
        assert np.isinf(to_db(1e300, 1e-300)) and to_db(1e300, 1e-300) > 0  # overflow of the ratio
    assert np.isnan(out[0]) and np.isnan(out[2])
    assert out[1] == 0.0
    assert out[3] == np.inf
    assert np.isfinite(out[4])  # a denormal is still a level
    assert not np.any(out == -np.inf)


def test_to_db_makes_complex_input_contiguous_before_taking_the_magnitude(monkeypatch):
    # On NumPy 1.x, np.abs of a strided complex array falls back from its SIMD loop to the scalar
    # libm hypot (a result one ulp away in a third of the elements) when its output happens to be
    # placed right after the source range, which for a channel of a two-channel array overshoots the
    # buffer by one element. Which of the two results a call gets then depends on the heap, so the
    # levels of equal maps would differ from call to call, and the difference of identical maps would
    # not be zero. to_db hands np.abs a contiguous array, which cannot touch its output this way.
    z = (np.arange(24.0) + 1j * np.arange(24.0)[::-1]).reshape(2, 3, 4)
    view = z[:, :, 1]
    assert not view.flags.c_contiguous
    seen = []
    real_abs = np.abs

    def spy(a, *args, **kwargs):
        seen.append(bool(np.asarray(a).flags.c_contiguous))
        return real_abs(a, *args, **kwargs)

    monkeypatch.setattr(np, "abs", spy)
    out = to_db(view, 1.0)
    monkeypatch.undo()
    assert seen and all(seen)
    np.testing.assert_allclose(out, 20.0 * np.log10(np.abs(view)), rtol=1e-14)
    assert np.shares_memory(view, z) and view.flags.writeable  # the caller's array is not replaced


def test_np_abs_of_a_strided_complex_array_can_depend_on_where_its_output_is():
    # The NumPy behaviour behind the test above, made deterministic: the output array is placed
    # exactly at the end of the source range of a channel view. Where NumPy has the SIMD loop (1.x on
    # AVX-512 machines) this gives other values than a far-away output; elsewhere both agree. Either
    # way the values stay within rounding of the exact magnitude, so the figures are right to rounding.
    n = 3000
    rng = np.random.default_rng(7)
    z = np.zeros(2 * n + 1 + n // 2 + 8, dtype=complex)
    z[: 2 * n] = rng.standard_normal(2 * n) + 1j * rng.standard_normal(2 * n)
    view = z[1 : 2 * n : 2]  # a channel of a two-channel array: stride 32 bytes, range [16, 16 + 32 n)
    touching = z[2 * n + 1 :].view(float)[:n]  # starts exactly where the source range ends
    far = np.empty(n)
    out_touching = np.abs(view, out=touching)
    out_far = np.abs(view, out=far)
    exact = np.hypot(view.real, view.imag)
    for out in (out_touching, out_far):
        np.testing.assert_allclose(out, exact, rtol=1e-15, atol=0)  # both within 2 ulp of hypot
    # a contiguous copy gives the same values wherever the output is
    flat = np.ascontiguousarray(view)
    np.testing.assert_array_equal(np.abs(flat, out=np.empty(n)), np.abs(flat))


@pytest.mark.parametrize("ref", [0.0, -1.0, np.nan, np.inf, "x", None, [1.0, 2.0]])
def test_invalid_reference_raises_value_error(ref):
    for func in (lambda r: to_db(1.0, r), lambda r: ref_text(r, "Pa"), lambda r: level_label("L", r, "Pa"), check_ref):
        with pytest.raises(ValueError, match="ref"):
            func(ref)


def test_check_finite_and_check_positive_name_the_argument():
    assert check_finite(3, "n") == 3.0 and isinstance(check_finite(3, "n"), float)
    assert check_positive("2.5", "w") == 2.5
    for bad in (np.nan, np.inf, "a", None):
        with pytest.raises(ValueError, match="width"):
            check_finite(bad, "width")
    for bad in (0, -1.0, np.nan):
        with pytest.raises(ValueError, match="dynamic_range"):
            check_positive(bad, "dynamic_range")


# --------------------------------------------------------------------------
# ref_text and level_label
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref, unit, expected",
    [
        (2e-5, "Pa", f"20 {MICRO}Pa"),
        (1e-6, "m/s\u00b2", f"1 {MICRO}m/s\u00b2"),
        (1.0, "", "1"),
        (1, "Pa", "1 Pa"),
        (2e-3, "Pa", "2 mPa"),
        (1e-12, "W", "1 pW"),
        (1e-9, "m", "1 nm"),
        (1e3, "Pa", "1 kPa"),
        (2.5e6, "Pa", "2.5 MPa"),
        (1e9, "Hz", "1 GHz"),
        (100.0, "Pa", "100 Pa"),
        (1e-4, "Pa", f"100 {MICRO}Pa"),  # the number stays in [1, 1000)
        (0.5, "Pa", "500 mPa"),
        (1e-3, "Pa", "1 mPa"),  # a power of ten does not slip into the next prefix
        (1.25e-6, "Pa", f"1.25 {MICRO}Pa"),
        (2.50e-5 * 1.0, "Pa", f"25 {MICRO}Pa"),  # no trailing zeros
        (1e-13, "W", "1e-13 W"),  # below p: no prefix, exponent form
        (1e12, "Hz", "1e+12 Hz"),
        (1e-6, "", "1e-06"),  # no unit: the number as it is
        (0.001, None, "0.001"),
    ],
)
def test_ref_text(ref, unit, expected):
    assert ref_text(ref, unit) == expected


def test_ref_text_of_a_value_that_rounds_up_to_a_power_of_ten():
    # 0.99999999999e-3 is "1 mPa" to the ten digits printed, not "1000 uPa" (and 10 nines are kept as they are)
    assert ref_text(0.99999999999e-3, "Pa") == "1 mPa"
    assert ref_text(0.9999999999e-3, "Pa") == f"999.9999999 {MICRO}Pa"


def test_ref_text_uses_the_micro_sign_u00b5():
    assert "\u00b5" in ref_text(2e-5, "Pa") and "\u03bc" not in ref_text(2e-5, "Pa")


@pytest.mark.parametrize(
    "convention, expected",
    [
        ("rms", f"Order level (dB re 20 {MICRO}Pa, RMS)"),
        ("RMS", f"Order level (dB re 20 {MICRO}Pa, RMS)"),
        ("peak", f"Order level (dB re 20 {MICRO}Pa, peak)"),
        (None, f"Order level (dB re 20 {MICRO}Pa)"),
    ],
)
def test_level_label(convention, expected):
    assert level_label("Order level", 2e-5, "Pa", convention) == expected
    assert level_label("Order level", **SOUND_PRESSURE, convention=convention) == expected


def test_level_label_without_a_unit_and_with_a_bad_convention():
    assert level_label("Level", 1.0, "") == "Level (dB re 1)"
    assert level_label("Level", 1.0, "", "peak") == "Level (dB re 1, peak)"
    assert level_label("Acceleration level", 1e-6, "m/s\u00b2", "rms") == (
        f"Acceleration level (dB re 1 {MICRO}m/s\u00b2, RMS)"
    )
    with pytest.raises(ValueError, match="convention"):
        level_label("Level", 1.0, "", "average")


def test_sound_pressure_is_the_explicit_plain_dict():
    assert SOUND_PRESSURE == {"ref": 2e-5, "unit": "Pa"}
    assert type(SOUND_PRESSURE) is dict
    assert rp.SOUND_PRESSURE is SOUND_PRESSURE
    # unpacks into the ref and unit options
    assert ref_text(**SOUND_PRESSURE) == f"20 {MICRO}Pa"
    assert float(to_db(1.0, SOUND_PRESSURE["ref"])) == pytest.approx(93.9794, abs=1e-4)


# --------------------------------------------------------------------------
# colour_limits
# --------------------------------------------------------------------------


def test_colour_limits_default_to_a_span_below_the_maximum():
    a = np.array([[10.0, 55.0], [np.nan, -np.inf]])
    assert colour_limits(a) == (15.0, 55.0)  # 40 dB below the largest finite value
    assert colour_limits(a, dynamic_range=25.0) == (30.0, 55.0)
    assert colour_limits([a, a + 10.0]) == (25.0, 65.0)  # a shared scale over several images
    assert all(isinstance(v, float) for v in colour_limits(a))


def test_colour_limits_given_limits_win():
    a = np.array([10.0, 55.0])
    assert colour_limits(a, vmin=20.0) == (20.0, 55.0)
    assert colour_limits(a, vmax=70.0) == (30.0, 70.0)
    assert colour_limits(a, 10.0, vmin=0.0, vmax=100.0) == (0.0, 100.0)  # dynamic_range is not used


def test_colour_limits_errors():
    with pytest.raises(ValueError, match="dynamic_range"):
        colour_limits(np.array([1.0]), dynamic_range=0.0)
    with pytest.raises(ValueError, match="vmin"):
        colour_limits(np.array([1.0, 50.0]), vmin=60.0)
    with pytest.raises(ValueError, match="vmax"):
        colour_limits(np.array([1.0]), vmax=np.nan)
    with pytest.raises(ValueError, match="no finite level"):
        colour_limits(np.array([np.nan, np.inf]))
    assert colour_limits(np.array([np.nan]), vmax=0.0) == (-40.0, 0.0)  # nothing to scan when vmax is given


# --------------------------------------------------------------------------
# as_items, label_index, relative
# --------------------------------------------------------------------------


class _Result:
    """Stands for any result object (a dataclass of rota-order)."""


def test_as_items_one_result_a_list_a_tuple_and_a_dict():
    r1, r2 = _Result(), _Result()
    assert as_items(r1) == [(None, r1)]
    assert as_items([r1, r2]) == [(None, r1), (None, r2)]
    assert as_items((r1, r2)) == [(None, r1), (None, r2)]
    assert as_items({"before": r1, "after": r2}) == [("before", r1), ("after", r2)]  # insertion order
    assert as_items({1000: r1, 2.5: r2}) == [("1000", r1), ("2.5", r2)]  # keys through str()
    assert as_items(np.zeros(3))[0][0] is None  # an array is one result, not a container


def test_as_items_rejects_empty_containers():
    for empty in ([], (), {}):
        with pytest.raises(ValueError, match="empty"):
            as_items(empty)


def _pairs():
    x = np.array([1000.0, 1500.0, 2000.0, 2500.0, 3000.0])
    before = np.array([90.0, 91.0, 92.0, np.nan, 94.0])
    after = np.array([80.0, 81.0, 82.0, 83.0, np.nan])
    other = np.array([95.0, 96.0, 97.0, 98.0, 99.0])
    return [(x, before), (x.copy(), after), (x.copy(), other)], ["before", "after", "other"]


def test_relative_subtracts_the_reference_entry():
    pairs, labels = _pairs()
    out = relative(pairs, labels, "before")
    assert len(out) == 3
    for (x_in, _), (x_out, _) in zip(pairs, out):
        np.testing.assert_array_equal(x_out, x_in)
    np.testing.assert_array_equal(out[0][1], [0.0, 0.0, 0.0, np.nan, 0.0])  # the reference: zero, NaN stays
    np.testing.assert_array_equal(out[1][1], [-10.0, -10.0, -10.0, np.nan, np.nan])  # a gap in either is a gap
    np.testing.assert_array_equal(out[2][1], [5.0, 5.0, 5.0, np.nan, 5.0])
    np.testing.assert_array_equal(relative(pairs, labels, "other")[1][1], [-15.0, -15.0, -15.0, -15.0, np.nan])
    assert label_index(labels, "after") == 1


def test_relative_none_returns_the_entries_unchanged():
    pairs, labels = _pairs()
    out = relative(pairs, labels, None)
    assert out is not pairs and len(out) == 3
    for (x_a, y_a), (x_b, y_b) in zip(pairs, out):
        assert x_a is x_b and y_a is y_b
    assert relative(pairs, [None, None, None], None)  # unlabelled is fine without a reference


def test_relative_labels_are_compared_as_strings():
    x = np.arange(3.0)
    pairs = [(x, np.array([1.0, 2.0, 3.0])), (x, np.array([2.0, 4.0, 6.0]))]
    out = relative(pairs, ["1000", "2000"], 1000)  # a dict {1000: ..., 2000: ...} gives these labels
    np.testing.assert_array_equal(out[1][1], [1.0, 2.0, 3.0])


def test_relative_unknown_label_and_unlabelled_entries_raise():
    pairs, labels = _pairs()
    with pytest.raises(ValueError, match="not one of the labels"):
        relative(pairs, labels, "missing")
    with pytest.raises(ValueError, match="after") as err:  # the message lists what exists
        relative(pairs, labels, "missing")
    assert "before" in str(err.value)
    with pytest.raises(ValueError, match="dict"):
        relative(pairs, [None, None, None], "before")
    with pytest.raises(ValueError, match="same length"):
        relative(pairs, labels[:2], "before")


def test_relative_needs_the_same_x_grid_and_shape():
    pairs, _ = _pairs()
    x_shift = pairs[1][0] + 1.0
    with pytest.raises(ValueError, match="x grids differ"):
        relative([pairs[0], (x_shift, pairs[1][1])], ["before", "after"], "before")
    with pytest.raises(ValueError, match="x grids differ"):  # a different length is a different grid
        relative([pairs[0], (pairs[1][0][:4], pairs[1][1][:4])], ["before", "after"], "before")
    with pytest.raises(ValueError, match="differ in shape"):
        relative([pairs[0], (pairs[1][0], pairs[1][1][:, None])], ["before", "after"], "before")
    # grids that agree to rounding are the same grid
    x = pairs[0][0]
    ok = relative([pairs[0], (x * (1.0 + 1e-12), pairs[1][1])], ["before", "after"], "before")
    assert len(ok) == 2
    with pytest.raises(ValueError, match="x grids differ"):
        relative([pairs[0], (x * (1.0 + 1e-6), pairs[1][1])], ["before", "after"], "before")


def test_relative_infinite_levels_do_not_warn():
    x = np.arange(3.0)
    y = np.array([-np.inf, 1.0, np.inf])  # silent frames read -inf in OverallLevel
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = relative([(x, y), (x, y)], ["a", "b"], "a")
    assert np.isnan(out[1][1][0]) and out[1][1][1] == 0.0 and np.isnan(out[1][1][2])


# --------------------------------------------------------------------------
# select_channel
# --------------------------------------------------------------------------


def test_select_channel_on_the_channel_axis():
    v = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)  # (bins, frames, ch)
    np.testing.assert_array_equal(select_channel(v, 0, 2), v[:, :, 0])
    np.testing.assert_array_equal(select_channel(v, 3, 2), v[:, :, 3])
    np.testing.assert_array_equal(select_channel(v, -1, 2), v[:, :, 3])  # as in NumPy
    w = np.arange(10.0).reshape(5, 2)  # (frames, ch)
    np.testing.assert_array_equal(select_channel(w, 1, 1), w[:, 1])
    assert select_channel(v, 1, 2).base is not None  # a view, no copy of the map
    assert select_channel(np.int64(2), 0, 1) == 2  # 0-d


def test_select_channel_single_channel_arrays_pass_through():
    m = np.ones((4, 6))  # a map of one channel has no channel axis
    assert select_channel(m, 0, 2) is m or np.shares_memory(select_channel(m, 0, 2), m)
    np.testing.assert_array_equal(select_channel(m, -1, 2), m)
    s = np.arange(5.0)
    np.testing.assert_array_equal(select_channel(s, 0, 1), s)


def test_select_channel_out_of_range_names_the_number_of_channels():
    v = np.zeros((2, 3, 2))
    with pytest.raises(IndexError, match="2 channels"):
        select_channel(v, 2, 2)
    with pytest.raises(IndexError, match="2 channels"):
        select_channel(v, -3, 2)
    with pytest.raises(IndexError, match="1 channel") as err:  # a mono result
        select_channel(np.zeros((2, 3)), 1, 2)
    assert "channels" not in str(err.value)
    assert "channel 1" in str(err.value)


def test_select_channel_needs_an_integer():
    with pytest.raises(TypeError, match="integer"):
        select_channel(np.zeros((2, 3, 2)), 0.5, 2)
    with pytest.raises(TypeError, match="integer"):
        select_channel(np.zeros((2, 3, 2)), "0", 2)
    assert select_channel(np.zeros((2, 3, 2)), np.int64(1), 2).shape == (2, 3)  # NumPy integers are fine


# --------------------------------------------------------------------------
# choose_x
# --------------------------------------------------------------------------

T_FRAMES = np.linspace(0.1, 2.9, 15)


def _jittery_runup(n=15, amplitude=0.01, seed=0):
    """A run-up 1000 -> 3000 rpm with reversals of about ``amplitude`` of the range."""
    up = np.linspace(1000.0, 3000.0, n)
    return up + np.random.default_rng(seed).uniform(-1.0, 1.0, n) * amplitude * 2000.0 * 0.5


def test_choose_x_auto_gives_rpm_for_a_monotonic_speed():
    up = np.linspace(1000.0, 3000.0, 15)
    kind, values, label = choose_x(T_FRAMES, up, "auto")
    assert kind == "rpm" and label == "Speed (rpm)"
    np.testing.assert_array_equal(values, up)
    kind, values, label = choose_x(T_FRAMES, up[::-1], "auto")  # a coast-down
    assert kind == "rpm" and label == "Speed (rpm)"
    np.testing.assert_array_equal(values, up[::-1])
    assert choose_x(T_FRAMES, up)[0] == "rpm"  # "auto" is the default


def test_choose_x_auto_accepts_small_reversals_and_returns_the_true_values():
    up = np.linspace(1000.0, 3000.0, 15)
    up[7] = up[6] - 0.015 * 2000.0  # a 1.5 % reversal: below the 2 % default
    kind, values, _ = choose_x(T_FRAMES, up, "auto")
    assert kind == "rpm"
    np.testing.assert_array_equal(values, up)  # the measured speed, not the envelope
    assert choose_x(T_FRAMES, up, "rpm")[0] == "rpm"
    down = np.linspace(3000.0, 1000.0, 15)
    down[5] = down[4] + 0.015 * 2000.0  # a rise during a coast-down
    assert choose_x(T_FRAMES, down, "auto")[0] == "rpm"


def test_choose_x_tol_sets_the_largest_reversal():
    up = np.linspace(1000.0, 3000.0, 15)
    up[7] = up[6] - 0.03 * 2000.0  # a 3 % reversal (the frame falls 3 % of the range below its predecessor)
    assert choose_x(T_FRAMES, up)[0] == "time"  # default 2 %
    assert choose_x(T_FRAMES, up, tol=0.05)[0] == "rpm"
    assert choose_x(T_FRAMES, up, "rpm", tol=0.05)[0] == "rpm"
    with pytest.raises(ValueError, match="essentially monotonic"):
        choose_x(T_FRAMES, up, "rpm", tol=0.01)
    assert choose_x(T_FRAMES, np.linspace(1000.0, 3000.0, 15), tol=0.0)[0] == "rpm"
    for bad in (-0.1, np.nan, np.inf, "x", None):
        with pytest.raises(ValueError, match="monotonic_tol"):
            choose_x(T_FRAMES, up, tol=bad)


def test_choose_x_auto_gives_time_otherwise():
    up_down = np.concatenate([np.linspace(1000.0, 3000.0, 8), np.linspace(2800.0, 1500.0, 7)])
    kind, values, label = choose_x(T_FRAMES, up_down, "auto")
    assert kind == "time" and label == "Time (s)"
    np.testing.assert_array_equal(values, T_FRAMES)
    # a gear shift: a drop of about 30 % in the middle of a run-up
    shift = np.concatenate([np.linspace(2000.0, 6000.0, 8), np.linspace(4200.0, 6000.0, 7)])
    assert choose_x(T_FRAMES, shift)[0] == "time"
    # a speed that ends where it started has no direction, a constant speed no range
    assert choose_x(T_FRAMES, np.concatenate([np.linspace(1000, 2000, 8), np.linspace(2000, 1000, 7)]))[0] == "time"
    assert choose_x(T_FRAMES, np.full(15, 1500.0))[0] == "time"
    assert choose_x(T_FRAMES[:1], np.array([1500.0]))[0] == "time"  # a single frame has no direction
    nan_speed = np.linspace(1000.0, 3000.0, 15)
    nan_speed[4] = np.nan
    assert choose_x(T_FRAMES, nan_speed)[0] == "time"


def test_choose_x_a_plateau_is_essentially_monotonic():
    hold = np.concatenate([np.full(5, 1200.0), np.linspace(1300.0, 3000.0, 10)])  # idle, then a run-up
    assert choose_x(T_FRAMES, hold)[0] == "rpm"


def test_choose_x_rpm_with_a_non_monotonic_speed_raises_with_advice():
    up_down = np.concatenate([np.linspace(1000.0, 3000.0, 8), np.linspace(2800.0, 1500.0, 7)])
    with pytest.raises(ValueError, match="bin_map") as err:
        choose_x(T_FRAMES, up_down, "rpm")
    assert "x='time'" in str(err.value) and "essentially monotonic" in str(err.value)
    with pytest.raises(ValueError, match="bin_map"):
        choose_x(T_FRAMES, np.full(15, 1500.0), "rpm")
    nan_speed = np.linspace(1000.0, 3000.0, 15)
    nan_speed[4] = np.nan
    with pytest.raises(ValueError, match="bin_map"):
        choose_x(T_FRAMES, nan_speed, "rpm")


def test_choose_x_check_false_always_allows_rpm_for_lines():
    up_down = np.concatenate([np.linspace(1000.0, 3000.0, 8), np.linspace(2800.0, 1500.0, 7)])
    for speed in (up_down, np.full(15, 1500.0), np.where(np.arange(15) == 4, np.nan, up_down)):
        kind, values, label = choose_x(T_FRAMES, speed, "rpm", check=False)
        assert (kind, label) == ("rpm", "Speed (rpm)")
        np.testing.assert_array_equal(values, speed)  # the true values, however they run
    assert choose_x(T_FRAMES, up_down, "auto", check=False)[0] == "time"  # "auto" still applies the rule
    assert choose_x(T_FRAMES, up_down, "time", check=False)[0] == "time"


def test_choose_x_time_is_always_possible_and_x_is_checked():
    up = np.linspace(1000.0, 3000.0, 15)
    kind, values, label = choose_x(T_FRAMES, up, "time")
    assert (kind, label) == ("time", "Time (s)")
    np.testing.assert_array_equal(values, T_FRAMES)
    for bad in ("speed", "RPM", None, 1):
        with pytest.raises(ValueError, match="'auto', 'rpm' or 'time'"):
            choose_x(T_FRAMES, up, bad)


# --------------------------------------------------------------------------
# speed_envelope and essentially_monotonic
# --------------------------------------------------------------------------


def test_speed_envelope_is_the_running_maximum_of_a_run_up_and_the_running_minimum_of_a_coast_down():
    up = np.array([1000.0, 1100.0, 1090.0, 1200.0, 1195.0, 1190.0, 1300.0])
    env = speed_envelope(up, 0.1)  # reversals of 10 and 5 rpm in a range of 300 rpm
    np.testing.assert_array_equal(env, [1000.0, 1100.0, 1100.0, 1200.0, 1200.0, 1200.0, 1300.0])
    assert np.all(np.diff(env) >= 0)
    down = np.array([3000.0, 2900.0, 2910.0, 2800.0, 2805.0, 2700.0])
    env = speed_envelope(down, 0.1)
    np.testing.assert_array_equal(env, [3000.0, 2900.0, 2900.0, 2800.0, 2800.0, 2700.0])
    assert np.all(np.diff(env) <= 0)
    smooth = np.linspace(1000.0, 3000.0, 9)
    np.testing.assert_array_equal(speed_envelope(smooth), smooth)  # already monotone: unchanged


def test_speed_envelope_displaces_a_frame_by_at_most_its_reversal_and_the_tolerance():
    rng = np.random.default_rng(5)
    for tol in (0.005, 0.02, 0.1):
        v = np.linspace(1000.0, 3000.0, 400) + rng.uniform(-1.0, 1.0, 400) * tol * 2000.0 / 2.5
        env = speed_envelope(v, tol)
        assert env is not None
        span = v.max() - v.min()
        assert np.all(np.diff(env) >= 0) and np.all(env >= v)
        assert np.max(env - v) <= tol * span * (1 + 1e-12)  # the displacement bound
        np.testing.assert_array_equal(env[env == v], v[env == v])  # frames on the envelope do not move
    down = np.linspace(3000.0, 1000.0, 100) + np.sin(np.arange(100)) * 5.0  # a coast-down with ripple
    env = speed_envelope(down, 0.02)
    assert env is not None and np.all(np.diff(env) <= 0) and np.all(env <= down)
    assert np.max(down - env) <= 0.02 * (down.max() - down.min())


def test_speed_envelope_is_none_when_not_essentially_monotonic():
    t = np.linspace(0.0, 1.0, 50)
    for v in (
        np.concatenate([np.linspace(1000.0, 3000.0, 25), np.linspace(2900.0, 1500.0, 25)]),  # up and down
        np.concatenate([np.linspace(1000.0, 6000.0, 25), np.linspace(4000.0, 6500.0, 25)]),  # gear shift
        np.full(50, 1500.0),  # constant: no range
        np.concatenate([np.linspace(1000.0, 2000.0, 25), np.linspace(2000.0, 1000.0, 25)]),  # last == first
        np.where(t > 0.5, np.nan, 1000.0 + 2000.0 * t),  # not finite
        np.array([1500.0]),  # a single frame
        np.empty(0),
    ):
        assert speed_envelope(v) is None
        assert not essentially_monotonic(v)
    assert essentially_monotonic(1000.0 + 2000.0 * t)
    assert essentially_monotonic(3000.0 - 2000.0 * t)
    with pytest.raises(ValueError, match="monotonic_tol"):
        speed_envelope(1000.0 + 2000.0 * t, -1.0)


def test_check_tol():
    assert check_tol(0.02) == 0.02 and check_tol(0) == 0.0 and check_tol(np.float64(0.5)) == 0.5
    for bad in (-1e-9, np.nan, np.inf, "a", None, [1]):
        with pytest.raises(ValueError, match="monotonic_tol"):
            check_tol(bad)


def test_x_labels_are_the_documented_strings():
    assert _core.X_LABELS == {"rpm": "Speed (rpm)", "time": "Time (s)"}


# --------------------------------------------------------------------------
# edges_from_centres
# --------------------------------------------------------------------------


def test_edges_from_centres_even_spacing():
    np.testing.assert_allclose(edges_from_centres([0.0, 1.0, 2.0, 3.0]), [-0.5, 0.5, 1.5, 2.5, 3.5], rtol=0, atol=1e-15)
    np.testing.assert_allclose(edges_from_centres([0.0, 10.0]), [-5.0, 5.0, 15.0], rtol=0, atol=1e-15)


def test_edges_from_centres_uneven_and_decreasing_spacing():
    np.testing.assert_allclose(edges_from_centres([0.0, 1.0, 3.0]), [-0.5, 0.5, 2.0, 4.0])  # midpoints inside
    np.testing.assert_allclose(edges_from_centres([3.0, 2.0, 1.0]), [3.5, 2.5, 1.5, 0.5])  # decreasing
    c = np.array([900.0, 1000.0, 1300.0, 1900.0, 2100.0])
    for centres in (c, c[::-1]):
        e = edges_from_centres(centres)
        assert e.shape == (c.size + 1,)
        lo, hi = np.minimum(e[:-1], e[1:]), np.maximum(e[:-1], e[1:])
        assert np.all(lo < centres) and np.all(centres < hi)  # each centre inside its cell
        np.testing.assert_allclose(e[1:-1], 0.5 * (centres[:-1] + centres[1:]))  # the cells tile
        assert np.all(np.diff(e) > 0) if centres[0] < centres[-1] else np.all(np.diff(e) < 0)


def test_edges_from_centres_single_centre_and_errors():
    np.testing.assert_array_equal(edges_from_centres([5.0]), [4.5, 5.5])
    for bad in ([], [[1.0, 2.0]], [1.0, np.nan], [1.0, np.inf]):
        with pytest.raises(ValueError, match="cell centres"):
            edges_from_centres(bad)


# --------------------------------------------------------------------------
# Matplotlib access
# --------------------------------------------------------------------------


def test_pyplot_returns_matplotlib_pyplot(plt):
    assert _core.pyplot() is plt


def test_get_ax_creates_a_figure_only_without_an_axes(plt):
    assert plt.get_fignums() == []
    fig, ax = _core.get_ax(None)
    assert plt.get_fignums() == [fig.number] and ax.figure is fig
    assert _core.get_ax()[1].figure is not fig  # the default is None: another new figure
    plt.close("all")
    fig, ax = plt.subplots()
    n = len(plt.get_fignums())
    out_fig, out_ax = _core.get_ax(ax)
    assert out_ax is ax and out_fig is fig and len(plt.get_fignums()) == n


def test_get_ax_rejects_what_is_not_an_axes(plt):
    fig, axs = plt.subplots(1, 2)
    for bad in (axs, fig, "ax", 3):
        with pytest.raises(TypeError, match="Matplotlib Axes"):
            _core.get_ax(bad)


# --------------------------------------------------------------------------
# Optional Matplotlib: subprocess tests (run everywhere)
# --------------------------------------------------------------------------


def _run(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)


def test_import_rota_order_plot_does_not_import_matplotlib_in_subprocess():
    # Hidden: any import of Matplotlib while importing rota_order.plot would raise.
    code = (
        "import sys\n"
        "sys.modules['matplotlib'] = None\n"
        "import rota_order, rota_order.plot as rp\n"
        "assert {'SOUND_PRESSURE', 'rpm_order_map', 'rpm_frequency_map', 'compare_maps'} <= set(rp.__all__)\n"
        "assert sys.modules['matplotlib'] is None\n"
        "assert not [m for m in sys.modules if m.startswith('matplotlib.')]\n"
        "print('ok')\n"
    )
    r = _run(code)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
    # Not hidden: still not imported (importing rota_order and rota_order.plot loads no Matplotlib).
    code = "import sys, rota_order, rota_order.plot\nassert 'matplotlib' not in sys.modules\nprint('ok')\n"
    r = _run(code)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_figure_functions_raise_the_install_hint_without_matplotlib_in_subprocess():
    code = (
        "import sys\n"
        "sys.modules['matplotlib'] = None\n"
        "import numpy as np\n"
        "import rota_order, rota_order.plot as rp\n"
        "from rota_order.plot import _core\n"
        "fs = 2000.0\n"
        "t = np.arange(round(1.5 * fs)) / fs\n"
        "rpm = rota_order.synth.rpm_profiles(t, 'linear', start=900.0, stop=2400.0)\n"
        "sig = rota_order.synth.synth_runup(fs, rpm, [{'order': 2.0, 'amplitude': 1.0}], seed=1)\n"
        "omap = rota_order.maps.rpm_order_map(sig.x, fs, sig.rpm, order_resolution=0.5, max_order=6.0)\n"
        "fmap = rota_order.maps.rpm_frequency_map(sig.x, fs, sig.rpm, resolution=10.0, max_freq=500.0)\n"
        "calls = [_core.pyplot, lambda: _core.get_ax(None), lambda: rp.rpm_order_map(omap),\n"
        "         lambda: rp.rpm_frequency_map(fmap), lambda: rp.compare_maps({'a': omap, 'b': omap})]\n"
        "for call in calls:\n"
        "    try:\n"
        "        call()\n"
        "    except ImportError as e:\n"
        "        assert \"pip install 'rota-order[plot]'\" in str(e), str(e)\n"
        "        assert 'rota_order.plot needs Matplotlib' in str(e), str(e)\n"
        "        assert 'does not depend on it' in str(e), str(e)\n"
        "        assert isinstance(e.__cause__, ImportError), repr(e.__cause__)\n"
        "    else:\n"
        "        raise SystemExit('no ImportError')\n"
        "print('ok')\n"
    )
    r = _run(code)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


PLOT_FUNCTIONS = [
    "band_levels_vs_rpm", "compare_maps", "frequency_slice", "metric_vs_rpm", "order_components",
    "order_prominence", "order_slice", "order_spectrum", "order_table", "overall_level",
    "rpm_frequency_map", "rpm_from_ridge", "rpm_order_map", "rpm_vs_time", "time_metric",
]


def test_all_of_rota_order_plot_is_documented_and_present():
    assert PLOT_FUNCTIONS == sorted(PLOT_FUNCTIONS)
    assert rp.__all__ == ["SOUND_PRESSURE", *PLOT_FUNCTIONS]  # alphabetical after the constant
    for name in PLOT_FUNCTIONS:
        assert callable(getattr(rp, name))
        assert f"    {name} " in rp.__doc__, f"{name} is not in the Functions list of the module docstring"
    for name in rp.__all__:
        obj = getattr(rp, name)
        if not callable(obj):
            continue
        doc = obj.__doc__ or ""
        for section in ("Parameters", "Returns", "References"):
            assert section in doc, f"{name} has no {section} section"


def test_the_options_that_mean_the_same_have_the_same_names_and_defaults():
    import inspect

    expected = {
        "ax": None, "ref": 1.0, "unit": "", "channel": 0, "scale": "db", "relative_to": None,
        "dynamic_range": 40.0, "vmin": None, "vmax": None, "cmap": "viridis", "colorbar": True,
    }
    selects = {"order_components", "order_table", "order_prominence"}  # orders= selects rows
    marks = {"rpm_frequency_map", "rpm_order_map", "compare_maps", "order_spectrum"}  # mark_orders= draws lines
    for name in PLOT_FUNCTIONS:
        params = inspect.signature(getattr(rp, name)).parameters
        assert ("orders" in params) == (name in selects), name
        assert ("mark_orders" in params) == (name in marks), name
        if name in marks:
            assert params["mark_orders"].default is None and params["mark_orders"].kind is inspect.Parameter.KEYWORD_ONLY
        if name in selects:
            assert params["orders"].default is None
    stored_ref = {"overall_level", "band_levels_vs_rpm"}  # levels stored in dB: ref=None means "as stored"
    for name in PLOT_FUNCTIONS:
        params = inspect.signature(getattr(rp, name)).parameters
        first = next(iter(params.values()))
        assert first.kind in (first.POSITIONAL_ONLY, first.POSITIONAL_OR_KEYWORD)
        rest = list(params.values())[1:]
        if name == "rpm_vs_time":
            rest = rest[1:]  # (t, rpm)
        assert all(p.kind is p.KEYWORD_ONLY for p in rest), f"{name}: options must be keyword-only"
        for key, default in expected.items():
            if key in params:
                want = None if (key == "ref" and name in stored_ref) else default
                assert params[key].default == want, f"{name}: {key}"
        if name != "compare_maps":  # it builds its own figure and returns (fig, axes)
            assert list(params)[1 if name != "rpm_vs_time" else 2] == "ax", f"{name}: ax is the first option"
        else:
            assert "ax" not in params


def test_level_reference_re_references_stored_levels_exactly():
    assert level_reference(2e-5) == (2e-5, 0.0)
    assert level_reference(2e-5, 2e-5) == (2e-5, 0.0)  # the same reference: a shift of exactly zero
    ref, shift = level_reference(2e-5, 1.0)
    assert ref == 1.0 and shift == pytest.approx(20.0 * np.log10(2e-5))
    ref, shift = level_reference(1.0, 2e-5)
    assert ref == 2e-5 and shift == pytest.approx(93.97940008672037)
    a = 0.37  # an amplitude: re-referencing the level equals the level re the new reference
    level = to_db(a, 2e-5)
    assert level + level_reference(2e-5, 1e-6)[1] == pytest.approx(to_db(a, 1e-6), abs=1e-12)
    for bad in (0.0, -1.0, np.nan, np.inf, "x"):
        with pytest.raises(ValueError, match="ref"):
            level_reference(2e-5, bad)
        with pytest.raises(ValueError, match="ref"):
            level_reference(bad)
