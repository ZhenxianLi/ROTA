"""Reproducibility sidecars and deterministic seeds.

A synthesised stimulus is only as useful as the record of how it was made.
:func:`provenance` writes that record next to the audio file as JSON (ROTA
version, library versions, parameters, seed and a digest of every input), and
:func:`stable_seed` derives a seed from names (file, channel, index) so that a
stimulus can be rebuilt from its name alone.
"""

import datetime
import hashlib
import json
import os
import platform
from collections.abc import Mapping
from pathlib import Path

import numpy as np

__all__ = ["provenance", "stable_seed", "jsonable"]

_CHUNK = 1 << 20
_SEED_BITS = 63
_KEY_SEPARATOR = "\x1f"  # ASCII unit separator


def jsonable(obj, _where="value"):
    """Convert ``obj`` to plain Python objects that ``json`` can write.

    NumPy scalars and arrays become Python scalars and nested lists, tuples
    become lists, mapping keys become strings, path-like objects become
    strings, complex numbers become ``{"real": ..., "imag": ...}`` and
    non-finite floats become the strings ``"nan"``, ``"inf"`` and ``"-inf"``
    (strict JSON has no NaN). Anything else raises ``TypeError`` rather than
    being silently turned into text, because a sidecar that cannot be read
    back is worse than none.

    Parameters
    ----------
    obj : object
        Value to convert.

    Returns
    -------
    object
        Built from ``dict``, ``list``, ``str``, ``int``, ``float``, ``bool``
        and ``None`` only.

    Raises
    ------
    TypeError
        For an object that has no JSON form; the message names its location.

    References
    ----------
    Bray, T. (2017). The JavaScript Object Notation (JSON) data interchange
    format. RFC 8259, IETF.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.generic):
        return jsonable(obj.item(), _where)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        v = float(obj)
        if np.isfinite(v):
            return v
        return "nan" if np.isnan(v) else ("inf" if v > 0 else "-inf")
    if isinstance(obj, complex):
        return {"real": jsonable(obj.real, _where), "imag": jsonable(obj.imag, _where)}
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist(), _where)
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v, f"{_where}[{k!r}]") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v, f"{_where}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, np.dtype):
        return str(obj)
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    raise TypeError(f"{type(obj).__name__} at {_where} is not JSON-serialisable")


def _sha256_file(path):
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _sha256_array(a):
    if a.dtype.hasobject:
        raise TypeError("object arrays cannot be hashed reproducibly")
    flat = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
    return hashlib.sha256(flat).hexdigest()


def _digest(item, name):
    """Digest record of one input: a file path or an array."""
    if isinstance(item, (str, os.PathLike)):
        path = Path(item)
        sha, size = _sha256_file(path)
        return {"kind": "file", "name": path.name, "sha256": sha, "size": size}
    if isinstance(item, (np.ndarray, list, tuple)):
        a = np.asarray(item)
        if a.dtype.kind not in "biufcmM":
            raise TypeError(f"input {name!r}: arrays must be numeric, got dtype {a.dtype}")
        return {
            "kind": "array",
            "sha256": _sha256_array(a),
            "shape": list(a.shape),
            "dtype": a.dtype.str,
            "nbytes": int(a.nbytes),
        }
    raise TypeError(f"input {name!r}: expected a file path or an array, got {type(item).__name__}")


def provenance(path, *, params, seed=None, inputs=None, extra=None):
    """Write a JSON sidecar that records how a stimulus was made.

    The file ``path`` (``.json`` is appended unless ``path`` already ends in
    it, so ``stimulus.wav`` gets ``stimulus.wav.json``) holds one object with
    the keys

    ``rota_version``, ``timestamp_utc`` (``YYYY-MM-DDTHH:MM:SSZ``),
    ``platform``, ``python``, ``numpy``, ``scipy``
        Where and when it was made.
    ``params``
        The caller's parameters, converted with :func:`jsonable`.
    ``seed``
        The caller's seed (or ``None``).
    ``inputs``
        One record per input, keyed by name. A file path gives
        ``{"kind": "file", "name", "sha256", "size"}`` with the SHA-256 of the
        file contents and its size in bytes; only the file name is kept, not
        the directory, so that sidecars can be shared. An array gives
        ``{"kind": "array", "sha256", "shape", "dtype", "nbytes"}`` with the
        SHA-256 of its bytes in C order (the digest of
        ``numpy.ascontiguousarray(a).tobytes()``).
    ``extra``
        Anything else the caller wants to keep, converted like ``params``.

    Parameters
    ----------
    path : str or path-like
        Sidecar location. The directory must exist.
    params : object
        JSON-serialisable parameters (NumPy scalars and arrays are converted;
        non-finite floats are written as strings). The values are the
        caller's: use the same units and amplitude convention (peak or RMS)
        as the call being recorded.
    seed : int, optional
        Seed used for the random parts, recorded as given.
    inputs : mapping, sequence, path-like or ndarray, optional
        Inputs to fingerprint: a mapping ``name -> path or array``, a
        sequence of paths and arrays (named ``input_0``, ``input_1``, ...), or
        a single path or array (named ``input_0``).
    extra : object, optional
        Further JSON-serialisable notes.

    Returns
    -------
    dict
        The record exactly as written (plain Python types only).

    Raises
    ------
    TypeError
        If ``params`` or ``extra`` contain an object without a JSON form, or
        an input is neither a path nor a numeric array.
    FileNotFoundError
        If an input path does not exist.

    References
    ----------
    ROTA ROADMAP, design principle 7 and section 6, item 15 (reproducibility
    sidecar).
    National Institute of Standards and Technology (2015). Secure Hash
    Standard (SHS). FIPS PUB 180-4.
    """
    import scipy

    from . import __version__

    if inputs is None:
        named = {}
    elif isinstance(inputs, Mapping):
        named = {str(k): v for k, v in inputs.items()}
    elif isinstance(inputs, (list, tuple)):
        named = {f"input_{i}": v for i, v in enumerate(inputs)}
    else:
        named = {"input_0": inputs}
    record = {
        "rota_version": __version__,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "params": jsonable(params, "params"),
        "seed": jsonable(seed, "seed"),
        "inputs": {name: _digest(item, name) for name, item in named.items()},
        "extra": jsonable(extra, "extra"),
    }
    p = Path(path)
    if p.suffix.lower() != ".json":
        p = p.with_name(p.name + ".json")
    p.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def stable_seed(*keys):
    """Deterministic 63-bit seed derived from names.

    The seed is the first eight bytes of the SHA-256 digest of
    ``"\\x1f".join(str(k) for k in keys)`` (ASCII unit separator, so that
    ``("ab", "c")`` and ``("a", "bc")`` differ), read big-endian and masked to
    63 bits. It is a pure function of the keys: the same file name, channel
    and index give the same random phases on every machine and run, and
    different keys give unrelated seeds. The value fits a signed 64-bit
    integer and is accepted by :func:`numpy.random.default_rng`.

    Parameters
    ----------
    *keys : object
        One or more keys, typically a file name, a channel and an index.
        Each is converted with ``str``, so ``1`` and ``"1"`` are the same key
        while ``1`` and ``1.0`` are not.

    Returns
    -------
    int
        Seed in ``[0, 2**63)``. No amplitudes are involved.

    Raises
    ------
    ValueError
        If no key is given, or a key contains the separator character.

    References
    ----------
    National Institute of Standards and Technology (2015). Secure Hash
    Standard (SHS). FIPS PUB 180-4.
    ROTA ROADMAP, section 6, item 15 (seeds derived from file names).
    """
    if not keys:
        raise ValueError("stable_seed needs at least one key")
    parts = [str(k) for k in keys]
    if any(_KEY_SEPARATOR in s for s in parts):
        raise ValueError("keys must not contain the separator character '\\x1f'")
    digest = hashlib.sha256(_KEY_SEPARATOR.join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << _SEED_BITS) - 1)
