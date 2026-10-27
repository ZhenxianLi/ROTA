"""Bin the SQAT time series of run_sqat.m along rpm with ROTA and draw them.

Step 3 of the SQAT bridge (see README.md in this folder). For each scenario exported by
``export_signals.py`` it reads

* ``build/sqat_bridge/signals/<name>.mat``: the speed track (``t_rpm``, ``rpm``),
* ``build/sqat_bridge/results/<name>_sqat.mat``: the SQAT time series written by ``run_sqat.m``
  (ISO 532-1 loudness in sone, DIN 45692 sharpness in acum, Daniel & Weber roughness in asper,
  ECMA-418-2 tonality in tu_HMS),

and bins every metric along rpm with ``rota.slices.metric_vs_rpm`` (arithmetic mean per bin,
run-up segment, start-up transient of each model skipped, see ``METRICS``). The metrics are
binned as SQAT returns them; no amplitude convention applies (they are not levels).

Written:

* ``build/sqat_bridge/results/binned.csv`` and ``binned.md``: bin centre, value, standard
  deviation and sample count for every scenario and metric,
* ``docs/images/09_sqat_bridge.png`` (when at least one result exists): each metric against
  rpm, the samples SQAT returned as dots and the bin means with their standard deviation.

Without results the script says so and exits; with ``--strict`` it then returns exit status 1
(as it does whenever a metric of a scenario is missing), which is what CI uses.

Usage, from anywhere::

    python examples/09_sqat_bridge/import_results.py [--strict]

References
----------
ISO 532-1:2017. Acoustics - Methods for calculating loudness - Part 1: Zwicker method.
DIN 45692:2009. Measurement technique for the simulation of the auditory sensation of sharpness.
Daniel, P., & Weber, R. (1997). Psychoacoustical roughness: implementation of an optimized
model. Acustica, 83(1), 113-123.
ECMA-418-2, 4th edition (2025). Psychoacoustic metrics for ITT equipment - Part 2 (methods for
describing human perception based on the Sottek Hearing Model).
Felix Greco, G., Merino-Martinez, R., Osses, A., & Lotinga, M. J. B. (2026). SQAT: a sound
quality analysis toolbox for MATLAB. Zenodo. doi:10.5281/zenodo.7934709.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from rota.slices import metric_vs_rpm

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "build" / "sqat_bridge"
SIG_DIR = BASE / "signals"
RES_DIR = BASE / "results"
FIG_PATH = ROOT / "docs" / "images" / "09_sqat_bridge.png"

SCENARIOS = ("ice_runup", "ev_runup")
# Bin width in rpm per scenario (ICE 1200 to 5900 rpm, EV 200 to 12900 rpm).
BIN_WIDTH = {"ice_runup": 250.0, "ev_runup": 500.0}
# key -> (time field, value field, label, unit, skip_start in s). skip_start follows the
# start-up transients stated in the docstring of metric_vs_rpm and in ECMA-418-2 6.2.9 (0.304 s).
METRICS = {
    "loudness": ("loudness_t", "loudness_sone", "Loudness ISO 532-1", "sone", 0.6),
    "sharpness": ("sharpness_t", "sharpness_acum", "Sharpness DIN 45692", "acum", 0.6),
    "roughness": ("roughness_t", "roughness_asper", "Roughness Daniel & Weber", "asper", 1.5),
    "tonality": ("tonality_t", "tonality_tu", "Tonality ECMA-418-2", "tu_HMS", 0.6),
}


def _vec(d, key):
    if key not in d:
        return None
    v = np.asarray(d[key], dtype=float).reshape(-1)
    return v if v.size else None


def bin_scenario(name):
    """Return {metric: (MetricVsRpm, t, values)} for one scenario and a list of problems."""
    problems = []
    sig_file, res_file = SIG_DIR / f"{name}.mat", RES_DIR / f"{name}_sqat.mat"
    if not sig_file.exists():
        return {}, [f"{name}: signal file {sig_file.name} missing (run export_signals.py)"]
    if not res_file.exists():
        return {}, [f"{name}: SQAT result {res_file.name} missing (run run_sqat.m)"]
    sig = loadmat(sig_file)
    res = loadmat(res_file)
    t_rpm, rpm = _vec(sig, "t_rpm"), _vec(sig, "rpm")
    out = {}
    for key, (tk, vk, _label, _unit, skip) in METRICS.items():
        t, v = _vec(res, tk), _vec(res, vk)
        if t is None or v is None:
            msg = res.get(f"{key}_msg")
            msg = str(np.squeeze(msg)) if msg is not None else "no output"
            problems.append(f"{name}: {key} missing ({msg})")
            continue
        n = min(t.size, v.size)
        t, v = t[:n], v[:n]
        keep = np.r_[True, np.diff(t) > 0]          # metric_vs_rpm needs a strictly increasing time
        t, v = t[keep], v[keep]
        try:
            out[key] = (metric_vs_rpm(v, t, rpm, t_rpm, segment="runup", bin_width=BIN_WIDTH[name],
                                      min_count=3, skip_start=skip, aggregate="mean"), t, v)
        except ValueError as exc:
            problems.append(f"{name}: {key} could not be binned ({exc})")
    return out, problems


def write_tables(results):
    RES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, metrics in results.items():
        for key, (b, _t, _v) in metrics.items():
            for c, val, sd, cnt in zip(b.centres, b.value, b.std, b.count):
                if cnt > 0:
                    rows.append((name, key, METRICS[key][3], c, val, sd, int(cnt)))
    with open(RES_DIR / "binned.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "metric", "unit", "rpm_centre", "mean", "std", "count"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.0f}", f"{r[4]:.6g}", f"{r[5]:.4g}", r[6]])

    lines = ["# SQAT metrics binned along rpm (rota.slices.metric_vs_rpm, mean per bin)", ""]
    for name, metrics in results.items():
        if not metrics:
            continue
        keys = list(metrics)
        lines.append(f"## {name} (bin width {BIN_WIDTH[name]:.0f} rpm)")
        lines.append("")
        lines.append("| rpm | " + " | ".join(f"{METRICS[k][2]} ({METRICS[k][3]})" for k in keys) + " |")
        lines.append("|---" * (len(keys) + 1) + "|")
        centres = metrics[keys[0]][0].centres
        for i, c in enumerate(centres):
            cells = []
            for k in keys:
                b = metrics[k][0]
                j = np.flatnonzero(np.isclose(b.centres, c))
                cells.append("" if j.size == 0 or not np.isfinite(b.value[j[0]]) else f"{b.value[j[0]]:.3g}")
            if any(cells):
                lines.append(f"| {c:.0f} | " + " | ".join(cells) + " |")
        lines.append("")
    (RES_DIR / "binned.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def make_figure(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in SCENARIOS if results.get(n)]
    fig, axes = plt.subplots(len(METRICS), len(names), figsize=(8.0, 9.0), dpi=150,
                             squeeze=False, constrained_layout=True)
    for j, name in enumerate(names):
        for i, (key, (_tk, _vk, label, unit, _skip)) in enumerate(METRICS.items()):
            ax = axes[i, j]
            if key not in results[name]:
                ax.text(0.5, 0.5, "no SQAT output", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            b, _t, v = results[name][key]
            ax.plot(b.control[b.used], v[b.used], ".", ms=1.5, color="0.7", label="SQAT samples")
            ok = b.valid
            ax.fill_between(b.centres[ok], (b.value - b.std)[ok], (b.value + b.std)[ok],
                            color="C0", alpha=0.25, lw=0)
            ax.plot(b.centres[ok], b.value[ok], "o-", ms=3, color="C0", label="bin mean (± std)")
            ax.set_ylabel(f"{label}\n({unit})", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(name, fontsize=9)
            if i == len(METRICS) - 1:
                ax.set_xlabel("Speed (rpm)", fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("SQAT metrics along rpm (rota.synth scenarios)", fontsize=10)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strict", action="store_true",
                    help="exit with status 1 unless every metric of every scenario was binned")
    args = ap.parse_args()
    results, problems = {}, []
    for name in SCENARIOS:
        res, prob = bin_scenario(name)
        results[name] = res
        problems += prob
    for p in problems:
        print("missing:", p)
    if not any(results.values()):
        print("No SQAT results found in", RES_DIR, "- run run_sqat.m first (see README.md).")
        return 1 if args.strict else 0
    print(write_tables(results))
    make_figure(results)
    print("figure:", FIG_PATH)
    print("tables:", RES_DIR / "binned.md", RES_DIR / "binned.csv")
    return 1 if (args.strict and problems) else 0


if __name__ == "__main__":
    sys.exit(main())
