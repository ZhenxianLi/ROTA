"""Export two synthetic run-ups for SQAT (MATLAB) as calibrated 48 kHz signals in Pa.

Step 1 of the SQAT bridge (see README.md in this folder):

1. ``export_signals.py`` (this script) writes the signals, their speed and metadata,
2. ``run_sqat.m`` runs SQAT on them in MATLAB,
3. ``import_results.py`` bins the SQAT time series along rpm with ROTA.

Scenarios, both from ``rota.synth.synth_scenario`` with a fixed seed, one receiver channel and
an ideal recorder (``measurement=None``, so the signal is the exact sum of its stems):

* ``ice_runup``: four-cylinder engine at full load in second gear, 1200 to about 5800 rpm.
  The duration is shortened from the default 18 s to 9 s so that the record ends before the
  first upshift (at 9.19 s): a metric binned along rpm then belongs to one gear only.
* ``ev_runup``: electric drive, single-speed reduction, 200 to about 12000 rpm in 10 s
  (all defaults).

Files written to ``build/sqat_bridge/signals/`` (``build/`` is git-ignored), for each scenario:

* ``<name>.mat`` (MATLAB v5, readable by ``load`` in every release): ``x`` (column, sound
  pressure in Pa, double; no scaling, no normalisation), ``fs`` (48000 Hz), ``t_rpm`` and
  ``rpm`` (speed in rpm without the angle-locked ripple, ``rpm_mean``, at 1 kHz), ``speed_kmh``
  and ``gear`` (same 1 kHz grid), ``name``, ``seed``, ``duration``.
* ``<name>.json``: the same metadata plus the RMS and peak sound pressure, the overall level
  in dB re 20 uPa and the ROTA version.

Amplitude convention: ``x`` is instantaneous sound pressure in Pa (RMS and peak values in the
JSON are stated as such). SQAT expects exactly this: a calibrated signal in Pa.

Usage, from anywhere::

    python examples/09_sqat_bridge/export_signals.py
    python examples/09_sqat_bridge/export_signals.py --out DIR

References
----------
ISO 532-1:2017, section 5 (input: calibrated sound pressure signal in Pa).
ECMA-418-2, 4th edition (2025), section 5 (input: sound pressure in Pa, 48 kHz).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

import rota
from rota.synth import synth_scenario

FS = 48000.0           # Hz, the rate ECMA-418-2 and SQAT work at (no resampling needed)
SPEED_FS = 1000.0      # Hz, rate of the exported speed track
P_REF = 20e-6          # Pa, reference sound pressure

# name -> (seed, duration in s or None for the default, overrides of synth_scenario)
SCENARIOS = {
    "ice_runup": (11, 9.0, {}),
    "ev_runup": (12, None, {}),
}

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "build" / "sqat_bridge" / "signals"


def export(name: str, out: Path) -> dict:
    seed, duration, overrides = SCENARIOS[name]
    sig = synth_scenario(name, fs=FS, duration=duration, seed=seed, measurement=None,
                         n_channels=1, **overrides)
    x = np.asarray(sig.x, dtype=float).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise RuntimeError(f"{name}: signal contains non-finite samples")

    step = int(round(sig.fs / SPEED_FS))
    idx = np.arange(0, sig.t.size, step)
    rms = float(np.sqrt(np.mean(x ** 2)))
    meta = {
        "name": name,
        "kind": sig.kind,
        "seed": seed,
        "fs": float(sig.fs),
        "duration": float(x.size / sig.fs),
        "n_samples": int(x.size),
        "unit": "Pa",
        "rms_pa": rms,
        "peak_pa": float(np.max(np.abs(x))),
        "level_db_spl": float(20 * np.log10(rms / P_REF)),
        "rpm_start": float(sig.rpm_mean[0]),
        "rpm_end": float(sig.rpm_mean[-1]),
        "gears": [int(g) for g in np.unique(sig.gear)],
        "speed_fs": SPEED_FS,
        "speed_source": "rpm_mean (driveline speed without the angle-locked ripple)",
        "overrides": {"duration": duration, **overrides},
        "rota_version": rota.__version__,
        "generator": "rota.synth.synth_scenario",
    }
    out.mkdir(parents=True, exist_ok=True)
    savemat(
        out / f"{name}.mat",
        {
            "x": x[:, None],
            "fs": float(sig.fs),
            "t_rpm": sig.t[idx][:, None],
            "rpm": np.asarray(sig.rpm_mean, dtype=float)[idx][:, None],
            "speed_kmh": np.asarray(sig.speed_kmh, dtype=float)[idx][:, None],
            "gear": np.asarray(sig.gear, dtype=float)[idx][:, None],
            "name": name,
            "seed": float(seed),
            "duration": meta["duration"],
        },
        format="5",
        do_compression=True,
    )
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    ap.add_argument("names", nargs="*", default=list(SCENARIOS), help="scenarios to export")
    args = ap.parse_args()
    for name in args.names:
        m = export(name, args.out)
        print(f"{name}: {m['duration']:.2f} s at {m['fs']:.0f} Hz, {m['level_db_spl']:.1f} dB SPL "
              f"(RMS {m['rms_pa']:.3f} Pa, peak {m['peak_pa']:.3f} Pa), "
              f"{m['rpm_start']:.0f} to {m['rpm_end']:.0f} rpm, gear(s) {m['gears']}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
