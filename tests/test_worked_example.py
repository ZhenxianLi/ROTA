"""The analysis chain of examples/10_worked_analysis.py on a shortened version of its signal.

The example builds one problem into a synthetic electric-drive run-up (a structural resonance at
2900 Hz that the gear whine, order 40, passes at 4350 rpm) and finds it again from the signal alone.
This module runs the same functions on the first 3.6 s of the recording (200 to 5360 rpm, which holds
the whole passage of the resonance; the transients keep their times of the full record) and checks that the analysis recovers what was built in: the
order, the speed range, the excess over the ECMA-418-1 criterion and the reduction that removes it,
all against the truth computed from the stems of ``rota.synth``. Data come from ``rota.synth`` only.

The toolbox parts (MoSQITo loudness and sharpness, Sottek-model tonality, through ``rota.bridges``) run
on a 1.6 s excerpt around the passage and are skipped when the package is missing.
"""

import importlib.abc
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

import rota
from rota import bridges

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "10_worked_analysis.py"
DURATION = 3.6  # s of the scenario; the passage of the resonance is at 2.9 to 3.1 s
EXCERPT = (1.9, 3.5)  # s, the excerpt that goes to the toolboxes


def _load():
    spec = importlib.util.spec_from_file_location("worked_analysis_example", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def worked():
    return _load()


@pytest.fixture(scope="module")
def run(worked):
    return worked.analyse(duration=DURATION, log=lambda *_: None, reference=False)


# ---- helpers -----------------------------------------------------------------------------------


def test_prominent_range_edges(worked):
    rpm = np.array([1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0])
    lo, hi = worked.prominent_range(rpm, np.array([-4.0, -1.0, 3.0, 5.0, 1.0, -3.0]))
    assert lo == pytest.approx(1100.0 + 100.0 * 1.0 / 4.0)  # linear interpolation of the crossings
    assert hi == pytest.approx(1400.0 + 100.0 * 1.0 / 4.0)
    assert np.isnan(worked.prominent_range(rpm, np.full(6, -1.0))[0])
    # only the run around the maximum counts, not a second excursion above zero
    lo, hi = worked.prominent_range(rpm, np.array([2.0, -1.0, -1.0, 1.0, 4.0, -1.0]))
    assert lo > 1200.0


def test_write_wav_scaling_and_clip_guard(worked, tmp_path, monkeypatch):
    monkeypatch.setattr(worked, "AUDIO_DIR", tmp_path)
    fs = 48000.0
    x = 1.0 * np.sin(2 * np.pi * 1000.0 * np.arange(int(0.5 * fs)) / fs)  # 1 Pa peak
    worked.write_wav("tone.wav", x, fs)
    rate, data = wavfile.read(tmp_path / "tone.wav")
    assert rate == 48000 and data.dtype == np.int16 and data.ndim == 1
    assert np.max(data) == pytest.approx(32767.0 / worked.FULL_SCALE_PA, abs=2)  # 1 Pa -> 8192
    with pytest.raises(ValueError):
        worked.write_wav("loud.wav", 2.0 * worked.FULL_SCALE_PA * x, fs)


# ---- the analysis chain on the shortened signal ----------------------------------------------------


def test_speed_references_agree(run):
    st = run["speed_stats"]
    assert st["tacho_median"] < 5e-4  # tachometer against the true speed: median below 0.05 %
    assert st["ridge_median"] < 0.01  # tacholess against tachometer, above 2500 rpm: median below 1 %
    assert st["ridge_max"] < 0.05


def test_level_is_unchanged_by_the_48k_conversion(run):
    x, x48 = run["x"], run["x48"]
    d = 10 * np.log10(np.mean(x48**2) / np.mean(x**2))
    assert abs(d) < 0.01


def test_problem_order_is_found(run):
    tr = run["tr"]
    assert tr["order"] in run["cand"]
    assert run["problem"] == tr["order"]
    others = [s["tnr_excess"] for o, s in run["screen"].items() if o != run["problem"]]
    assert max(others) < 0.0 < run["screen"][run["problem"]]["tnr_excess"]  # only the whine is prominent


def test_prominence_agrees_with_the_truth(run):
    tr, fine = run["tr"], run["fine"]
    assert run["max_excess"] == pytest.approx(run["max_excess_truth"], abs=1.5)  # dB
    assert run["max_excess"] > 10.0  # the built-in problem is large
    assert abs(run["max_excess_rpm"] - tr["rpm_res"]) < 150.0  # the peak sits at the resonance
    near = (fine["rpm"] > run["window"][0] - 200.0) & (fine["rpm"] < run["window"][1] + 200.0)
    assert np.nanmedian(np.abs(fine["tnr"][near] - run["tnr_true"][near])) < 1.5
    lo, hi = run["window"]
    lo_t, hi_t = run["window_truth"]
    assert abs(hi - hi_t) < 150.0  # steep upper edge
    assert abs(lo - lo_t) < 500.0  # gentle lower slope: the edge moves by a few 100 rpm for 0.5 dB
    lo6, hi6 = run["window6"]
    lo6_t, hi6_t = run["window6_truth"]
    assert abs(lo6 - lo6_t) < 150.0 and abs(hi6 - hi6_t) < 150.0  # the range more than 6 dB above the criterion
    assert lo6 < tr["rpm_band"][0] and hi6 > tr["rpm_band"][1]  # it holds the half-power band of the resonance
    assert run["max_excess_pr"] > 0.0


def test_order_slice_level_against_the_truth(run):
    sc, tr = run["sc"], run["tr"]
    sl = run["screen"][run["problem"]]["slice"]
    truth = np.interp(sl.t, sc.t, sc.quasi_static_amplitude(tr["k"]) / np.sqrt(2.0))
    lo, hi = run["window"]
    sel = (sl.rpm >= lo) & (sl.rpm <= hi) & np.isfinite(sl.amplitude)
    err = 20 * np.log10(sl.amplitude[sel] / truth[sel])
    assert sel.sum() > 20
    assert np.median(np.abs(err)) < 0.5 and np.max(np.abs(err)) < 1.5  # dB, RMS convention on both sides


def test_strongest_line_is_on_the_problem_order(run):
    assert run["shares"][("range", run["problem"])] > 0.9
    assert run["shares"][("range", 34.0)] == 0.0  # control: an order without a source


def test_extraction_follows_the_true_stem(worked, run):
    sel = run["sc"].t > worked.EDGE_S
    sim = rota.metrics.waveform_similarity(run["comp"].waveforms[sel, 0], run["stem"][sel])
    assert sim["corr"] > 0.98
    assert np.allclose(run["resid"] + run["comp"].total, run["x"])  # extract.residual: x = residual + orders


def test_reduction_needed_and_what_if(run):
    p, red, checks = run["problem"], run["reductions"], run["checks"]
    need = np.ceil(run["max_excess_truth"])  # the truth says: reduce by this much to reach the criterion
    assert need <= red["required"] <= need + 3.0  # a little more, because the extracted order carries some noise
    assert checks["required"]["excess"] < 0.0  # below the criterion everywhere
    assert checks["required"]["range"][0] != checks["required"]["range"][0]  # NaN: no prominent range left
    # the round-valued reduction lowers the excess by about that much and leaves a narrower range
    assert checks["round"]["excess"] == pytest.approx(run["max_excess"] - red["round"], abs=1.5)
    lo, hi = run["window"]
    assert checks["round"]["range"][0] > lo and checks["round"]["range"][1] < hi
    for key in ("required", "round"):  # the order slice drops by the gain that was applied
        assert checks[key]["slice_change"] == pytest.approx(-red[key], abs=0.3)
        assert -3.0 < checks[key]["la_window"] - checks["original"]["la_window"] < 0.0
    assert p == run["tr"]["order"]


def test_modified_signal_differs_only_by_the_order(run):
    x, comp = run["x"], run["comp"]
    for key, gain_db in run["reductions"].items():
        gain = 10.0 ** (-gain_db / 20.0)
        assert np.allclose(run["variants"][key], x - (1.0 - gain) * comp.total, atol=1e-9)


# ---- toolboxes (optional) ---------------------------------------------------------------------------------


class _ImportFails(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Finds the given packages, as if they were installed, and raises ImportError when one is imported."""

    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, name, path=None, target=None):
        return importlib.util.spec_from_loader(name, self) if name in self.names else None

    def create_module(self, spec):
        return None  # the default module object

    def exec_module(self, module):
        raise ImportError(f"{module.__name__} is installed, but its import fails (simulated)")


@pytest.mark.parametrize("import_fails", [False, True], ids=["not-installed", "import-fails"])
def test_missing_toolboxes_are_skipped_with_a_message(worked, run, monkeypatch, import_fails):
    names = ("mosqito", "sottek_hearing_model")
    if import_fails:  # found by importlib.util.find_spec, as an installed package is, but its import raises
        for name in names:
            monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setattr(sys, "meta_path", [_ImportFails(names), *sys.meta_path])
        assert all(importlib.util.find_spec(name) is not None for name in names)
    else:  # not installed: the import raises ModuleNotFoundError
        for name in names:
            monkeypatch.setitem(sys.modules, name, None)
    lines = []
    out = worked.toolbox_metrics(run, log=lines.append, parallel=False)  # the bridges raise ImportError here
    assert out == {"mosqito": {}, "sottek": {}}
    text = "\n".join(lines)
    assert "MoSQITo is not installed or cannot be imported" in text
    assert "sottek-hearing-model is not installed or cannot be imported" in text
    assert "pip install mosqito" in text and "pip install sottek-hearing-model" in text  # the bridges' messages
    S = worked.summary(run, out)  # the summary and the conclusion still work, with ROTA's numbers only
    assert set(S["required"]) == {"L_A"}
    printed = []
    worked.print_conclusion(run, S, log=printed.append)
    assert any("problem order" in line for line in printed)
    assert any("cannot be imported" in line for line in printed)


@pytest.fixture(scope="module")
def excerpt(run):
    """The three signals of the run cut to EXCERPT (Pa at the rate of the scenario), with the speed and
    time axes of the cut; the bridges resample the signals to 48 kHz."""
    sc, rpm = run["sc"], run["rpm"]
    i0, i1 = int(EXCERPT[0] * sc.fs), int(EXCERPT[1] * sc.fs)
    x = {k: v[i0:i1] for k, v in run["variants"].items()}
    return x, sc.t[i0:i1] - sc.t[i0], rpm[i0:i1]


def _peak_bin(res, rpm_peak):
    return int(np.argmin(np.abs(res.centres - rpm_peak)))


def test_mosqito_loudness_and_sharpness_fall_at_the_passage(worked, run, excerpt):
    pytest.importorskip("matplotlib")  # mosqito imports it without declaring it
    pytest.importorskip("mosqito")
    x, t, rpm = excerpt
    b = {k: worked.bin_mosqito(worked.mosqito_metrics(v), t, rpm) for k, v in x.items() if k != "round"}
    j = _peak_bin(b["original"]["loudness"], run["tr"]["rpm_res"])
    n0, n1 = b["original"]["loudness"].value[j], b["required"]["loudness"].value[j]
    assert np.isfinite(n0) and np.isfinite(n1)
    assert n1 < 0.95 * n0  # loudness N5 at the peak bin: lower by more than 5 %
    s0, s1 = b["original"]["sharpness"].value[j], b["required"]["sharpness"].value[j]
    assert s1 < s0  # a tone at 2.9 kHz adds to the sharpness; removing it lowers it
    assert 0.0 < n1 and 0.0 < s1 < 3.0  # plausible units: sone and acum


def test_sottek_tonality_falls_at_the_passage(worked, run, excerpt):
    pytest.importorskip("matplotlib")
    pytest.importorskip("sottek_hearing_model")
    x, t, rpm = excerpt
    b = {}
    for k in ("original", "required"):
        ton = bridges.sottek_hearing_model.tonality(x[k], worked.FS)
        b[k] = worked.bin_sottek(ton, t, rpm, run["problem"])
    j = _peak_bin(b["original"]["tonality"], run["tr"]["rpm_res"])
    t0, t1 = b["original"]["tonality"].value[j], b["required"]["tonality"].value[j]
    assert np.isfinite(t0) and np.isfinite(t1)
    assert t0 > 0.5 and t1 < 0.7 * t0  # the whine sets the tonality at the passage; 15 dB less removes much of it
    o0, o1 = b["original"]["order_tonality"].value[j], b["required"]["order_tonality"].value[j]
    assert o1 < 0.7 * o0
