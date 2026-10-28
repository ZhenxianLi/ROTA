"""Tests for the reproducibility sidecar and the deterministic seeds."""

import datetime
import hashlib
import json
import pathlib
import platform

import numpy as np
import pytest
import scipy

import rota_order as ro
from rota_order._provenance import jsonable, provenance, stable_seed

# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_sidecar_is_written_with_versions_params_and_seed(tmp_path):
    rec = provenance(
        tmp_path / "stimulus_01",
        params={"fs": 48000.0, "phase": "random", "orders": [2, 4.5]},
        seed=1234,
        extra={"note": "unit test"},
    )
    path = tmp_path / "stimulus_01.json"
    assert path.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == rec  # the returned dict is what was written
    assert rec["rota_order_version"] == ro.__version__
    assert rec["python"] == platform.python_version()
    assert rec["numpy"] == np.__version__ and rec["scipy"] == scipy.__version__
    assert rec["platform"] == platform.platform()
    assert rec["params"] == {"fs": 48000.0, "phase": "random", "orders": [2, 4.5]}
    assert rec["seed"] == 1234 and rec["extra"] == {"note": "unit test"}
    assert rec["inputs"] == {}
    utc = datetime.timezone.utc
    stamp = datetime.datetime.strptime(rec["timestamp_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=utc)
    assert abs((datetime.datetime.now(utc) - stamp).total_seconds()) < 120


def test_sidecar_name_rules(tmp_path):
    provenance(tmp_path / "a", params={})
    provenance(tmp_path / "b.wav", params={})
    provenance(tmp_path / "c.json", params={})
    provenance(str(tmp_path / "d.JSON"), params={})
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["a.json", "b.wav.json", "c.json", "d.JSON"]
    with pytest.raises(FileNotFoundError):
        provenance(tmp_path / "missing_dir" / "x", params={})


def test_input_digests_match_hashlib(tmp_path):
    data = np.random.default_rng(0).bytes(3 * (1 << 20) + 17)  # more than one read chunk
    f = tmp_path / "recording.bin"
    f.write_bytes(data)
    a = np.random.default_rng(1).standard_normal((50, 3))
    z = a.astype(complex)[::2]  # non-contiguous view
    i = np.arange(7, dtype=np.int16)
    rec = provenance(
        tmp_path / "p",
        params={},
        inputs={"rec": f, "str_path": str(f), "a": a, "z": z, "i": i, "lst": [1.0, 2.0]},
    )
    inputs = rec["inputs"]
    assert inputs["rec"] == {
        "kind": "file",
        "name": "recording.bin",  # the directory is not recorded
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }
    assert inputs["str_path"]["sha256"] == inputs["rec"]["sha256"]
    assert str(tmp_path) not in json.dumps(rec)
    assert inputs["a"] == {
        "kind": "array",
        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
        "shape": [50, 3],
        "dtype": "<f8",
        "nbytes": a.nbytes,
    }
    assert inputs["z"]["sha256"] == hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
    assert inputs["z"]["shape"] == [25, 3] and inputs["z"]["dtype"] == "<c16"
    assert inputs["i"]["sha256"] == hashlib.sha256(i.tobytes()).hexdigest()
    assert inputs["i"]["dtype"] == "<i2" and inputs["i"]["shape"] == [7]
    assert inputs["lst"]["sha256"] == hashlib.sha256(np.array([1.0, 2.0]).tobytes()).hexdigest()
    # A different array or a different file content changes the digest.
    b = a.copy()
    b[0, 0] += 1e-12
    other = provenance(tmp_path / "q", params={}, inputs={"a": b})
    assert other["inputs"]["a"]["sha256"] != inputs["a"]["sha256"]


def test_inputs_given_as_sequence_or_single_item(tmp_path):
    f = tmp_path / "x.dat"
    f.write_bytes(b"abc")
    a = np.zeros(4)
    rec = provenance(tmp_path / "s", params={}, inputs=[f, a])
    assert list(rec["inputs"]) == ["input_0", "input_1"]
    assert rec["inputs"]["input_0"]["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert rec["inputs"]["input_1"]["kind"] == "array"
    assert list(provenance(tmp_path / "t", params={}, inputs=f)["inputs"]) == ["input_0"]
    assert list(provenance(tmp_path / "u", params={}, inputs=a)["inputs"]) == ["input_0"]
    empty = provenance(tmp_path / "v", params={}, inputs=np.zeros(0))
    assert empty["inputs"]["input_0"]["sha256"] == hashlib.sha256(b"").hexdigest()


def test_input_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        provenance(tmp_path / "p", params={}, inputs={"gone": tmp_path / "nope.wav"})
    with pytest.raises(TypeError, match="inputs|input"):
        provenance(tmp_path / "p", params={}, inputs={"x": 3.0})
    with pytest.raises(TypeError):
        provenance(tmp_path / "p", params={}, inputs={"x": np.array(["a", "b"])})
    with pytest.raises(TypeError):
        provenance(tmp_path / "p", params={}, inputs={"x": np.array([object()], dtype=object)})


def test_numpy_types_are_serialised(tmp_path):
    params = {
        "f32": np.float32(1.5),
        "i64": np.int64(7),
        "u8": np.uint8(3),
        "b": np.bool_(True),
        "arr": np.arange(3),
        "mat": np.eye(2),
        "tuple": (1, np.float64(2.5)),
        "nested": {1: np.array([[1, 2], [3, 4]]), "path": pathlib.PurePosixPath("a/b")},
        "c": 1 + 2j,
        "carr": np.array([1j]),
        "dtype": np.dtype("float32"),
        "none": None,
        "nan": float("nan"),
        "inf": np.inf,
        "ninf": np.float64("-inf"),
        "np_nan_in_array": np.array([1.0, np.nan]),
    }
    rec = provenance(tmp_path / "np", params=params, seed=np.int64(42))
    p = rec["params"]
    assert p["f32"] == 1.5 and type(p["f32"]) is float
    assert p["i64"] == 7 and type(p["i64"]) is int and p["u8"] == 3
    assert p["b"] is True
    assert p["arr"] == [0, 1, 2] and p["mat"] == [[1.0, 0.0], [0.0, 1.0]]
    assert p["tuple"] == [1, 2.5]
    assert p["nested"] == {"1": [[1, 2], [3, 4]], "path": "a/b"}
    assert p["c"] == {"real": 1.0, "imag": 2.0} and p["carr"] == [{"real": 0.0, "imag": 1.0}]
    assert p["dtype"] == "float32" and p["none"] is None
    assert p["nan"] == "nan" and p["inf"] == "inf" and p["ninf"] == "-inf"
    assert p["np_nan_in_array"] == [1.0, "nan"]
    assert rec["seed"] == 42 and type(rec["seed"]) is int
    # The file is strict JSON (no NaN/Infinity literals) and reads back identically.
    text = (tmp_path / "np.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == rec


def test_unserialisable_params_fail_loudly_with_their_location(tmp_path):
    with pytest.raises(TypeError, match=r"params\['rng'\]"):
        provenance(tmp_path / "bad", params={"rng": np.random.default_rng(0)})
    with pytest.raises(TypeError, match=r"params\['x'\]\[1\]"):
        provenance(tmp_path / "bad", params={"x": [1, object()]})
    with pytest.raises(TypeError, match="extra"):
        provenance(tmp_path / "bad", params={}, extra={"f": len})
    assert not (tmp_path / "bad.json").exists()  # nothing half-written


def test_jsonable_is_plain_python():
    out = jsonable({"a": np.arange(2), "b": (np.float64(1.0),)})
    assert json.loads(json.dumps(out)) == out
    assert isinstance(out["a"][0], int) and isinstance(out["b"][0], float)
    assert jsonable("text") == "text" and jsonable(True) is True and jsonable(None) is None


# --------------------------------------------------------------------------
# stable_seed
# --------------------------------------------------------------------------


def test_stable_seed_is_deterministic_63_bit_and_distinct():
    s = stable_seed("engine_run_03.wav", 1, 7)
    assert s == stable_seed("engine_run_03.wav", 1, 7)
    assert isinstance(s, int) and 0 <= s < 2**63
    # Fixed value: the definition (SHA-256, first 8 bytes big-endian, 63 bits) must not drift.
    digest = hashlib.sha256(b"engine_run_03.wav\x1f1\x1f7").digest()
    assert s == int.from_bytes(digest[:8], "big") & (2**63 - 1)
    keys = [
        ("a",),
        ("b",),
        ("a", "b"),
        ("ab",),
        ("a", "bc"),
        ("ab", "c"),
        ("a", 1),
        ("a", 2),
        ("a", 1.0),
        ("engine_run_03.wav", 0, 0),
        ("engine_run_03.wav", 0, 1),
        ("engine_run_03.wav", 1, 0),
        ("",),
        ("", ""),
    ]
    seeds = [stable_seed(*k) for k in keys]
    assert len(set(seeds)) == len(keys)
    assert stable_seed(1) == stable_seed("1")  # keys are converted with str()
    # 63 bits are really used: the high bit of the range occurs among a few hundred seeds.
    many = [stable_seed("k", i) for i in range(400)]
    assert max(many) >= 2**62 and min(many) < 2**62 and len(set(many)) == 400


def test_stable_seed_feeds_numpy_and_rejects_bad_keys():
    s = stable_seed("file", 3)
    a = np.random.default_rng(s).uniform(size=4)
    b = np.random.default_rng(stable_seed("file", 3)).uniform(size=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, np.random.default_rng(stable_seed("file", 4)).uniform(size=4))
    with pytest.raises(ValueError):
        stable_seed()
    with pytest.raises(ValueError):
        stable_seed("a\x1fb")


def test_seed_from_name_reproduces_a_synthesised_stimulus(tmp_path):
    """The workflow of ROADMAP section 6, item 15: name -> seed -> stimulus -> sidecar."""
    from rota_order._ordertable import OrderTable, synthesize_orders

    table = OrderTable([2.0, 4.0], [500.0, 3000.0], np.ones((2, 2)), "peak", unit="Pa")
    rpm = np.linspace(800.0, 2500.0, 4000)
    seed = stable_seed("stimulus_07", "ch0")
    x1, info = synthesize_orders(table, rpm, 8000.0, seed=seed)
    rec = provenance(tmp_path / "stimulus_07", params=info, seed=seed, inputs={"rpm": rpm})
    assert rec["seed"] == seed and rec["params"]["seed"] == seed
    x2, _ = synthesize_orders(table, rpm, 8000.0, seed=stable_seed("stimulus_07", "ch0"))
    assert np.array_equal(x1, x2)
    assert rec["inputs"]["rpm"]["sha256"] == hashlib.sha256(rpm.tobytes()).hexdigest()
