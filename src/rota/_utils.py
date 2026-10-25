"""Internal helpers shared across modules."""

import numpy as np

_AMPLITUDES = ("peak", "rms")


def as_2d(x):
    """Return ``x`` as a float array of shape ``(n, ch)`` and whether it was 1-D."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[:, None], True
    if x.ndim == 2:
        return x, False
    raise ValueError("signal must have shape (n,) or (n, ch)")


def restore(y, was_1d):
    """Undo :func:`as_2d` on the last axis."""
    return y[..., 0] if was_1d else y


def check_amplitude(amplitude):
    if amplitude not in _AMPLITUDES:
        raise ValueError(f"amplitude must be one of {_AMPLITUDES}, got {amplitude!r}")
    return amplitude


def time_axis(n, fs=None, t=None):
    """Return a time vector of length ``n`` from ``fs`` or an explicit ``t``."""
    if t is not None:
        t = np.asarray(t, dtype=float)
        if t.shape != (n,):
            raise ValueError("t must have the same length as the signal")
        if np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing")
        return t
    if fs is None or fs <= 0:
        raise ValueError("give a positive sampling rate fs or a time vector t")
    return np.arange(n) / float(fs)


def broadcast_rpm(rpm, n):
    """Return rpm as a float array of length ``n`` (scalars are broadcast)."""
    rpm = np.asarray(rpm, dtype=float)
    if rpm.ndim == 0:
        rpm = np.full(n, float(rpm))
    if rpm.shape != (n,):
        raise ValueError(f"rpm must be a scalar or have shape ({n},)")
    if not np.all(np.isfinite(rpm)):
        raise ValueError("rpm contains non-finite values; regularise it first")
    return rpm


def runs(mask):
    """Start and stop (exclusive) indices of contiguous True runs in ``mask``."""
    m = np.concatenate(([False], np.asarray(mask, bool), [False]))
    d = np.diff(m.astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)
