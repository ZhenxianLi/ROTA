# 09: SQAT bridge (MATLAB)

This example links ROTA to [SQAT](https://github.com/ggrecow/SQAT), the open-source Sound Quality Analysis Toolbox for MATLAB, and shows how psychoacoustic metrics vary with rpm in two synthetic run-ups. ROTA writes the signals, SQAT computes the metrics over time, and ROTA bins them along the speed with `rota.slices.metric_vs_rpm`. SQAT is an optional, external dependency: it is never vendored in ROTA, and only its documented public functions are called (the SQAT README and the help header of each function), with every parameter given explicitly. In continuous integration it runs the same way as the black-box MATLAB reference (`validation/README.md`): on a GitHub-hosted runner, where MathWorks licenses MATLAB automatically for a public repository.

## SQAT

* Version: release `v2.0`, pinned in the workflow to commit `94f489c053d1366da05d5e62838c3f54ac7715c3`.
* Licence: GPL-3.0 for the toolbox as a whole (`LICENSE`); individual files carry their own licence, stated in their headers and in the table of the SQAT README (for instance `Loudness_ISO532_1` BSD-3-Clause, `Roughness_Daniel1997` MIT, `Sharpness_DIN45692` and `Tonality_ECMA418_2` GPL-3.0). ROTA does not copy or redistribute any of it; the workflow clones it at run time.
* MATLAB products: MATLAB and the Signal Processing Toolbox (required by the ECMA-418-2 functions).
* Cite: Felix Greco, G., Merino-Martínez, R., Osses, A., & Lotinga, M. J. B. (2026). SQAT: a sound quality analysis toolbox for MATLAB. Zenodo. doi:10.5281/zenodo.7934709 (cite the release used).

## Signals

Two scenarios of `rota.synth.synth_scenario`, one receiver channel, an ideal recorder (the signal is the exact sum of its stems) and a fixed seed, so every run analyses bit-identical signals:

| Name | Seed | Duration | Speed | Overrides |
|---|---|---|---|---|
| `ice_runup` | 11 | 9 s | 1200 to about 5900 rpm, second gear | duration shortened from 18 s so that the record ends before the first upshift (9.19 s); a metric binned along rpm then belongs to one gear |
| `ev_runup` | 12 | 10 s | 200 to about 12900 rpm, single-speed reduction | none |

`export_signals.py` writes `build/sqat_bridge/signals/<name>.mat` (MATLAB v5): `x`, the sound pressure in Pa at 48 kHz as a double column (no scaling or normalisation; SQAT expects a calibrated signal in Pa), the speed `rpm` (the driveline speed `rpm_mean`, without the angle-locked ripple) with its time axis `t_rpm` at 1 kHz, `speed_kmh` and `gear` on the same grid, and `<name>.json` with the metadata (seed, overrides, RMS and peak pressure, level in dB re 20 uPa, ROTA version). `build/` is git-ignored.

## SQAT calls

`run_sqat.m` adds SQAT to the path with its documented initialiser `startup_SQAT`, then for each signal:

| Metric | Function | Arguments | Output used |
|---|---|---|---|
| Loudness, ISO 532-1:2017 (Zwicker), time-varying | `Loudness_ISO532_1` | `(x, 48000, 0, 2, 0.5, false)`: free field, time-varying method, 0.5 s skipped in SQAT's own statistics, no figure | `time`, `InstantaneousLoudness` (sone), `InstantaneousLoudnessLevel` (phon) |
| Sharpness, DIN 45692:2009 | `Sharpness_DIN45692_from_loudness` | `(InstantaneousSpecificLoudness, 'DIN45692', time, 0.5, false)`, from the loudness call above; if that call fails, `Sharpness_DIN45692(x, 48000, 'DIN45692', 0, 2, 0.5, false, false)` | `time`, `InstantaneousSharpness` (acum) |
| Roughness, Daniel & Weber (1997) | `Roughness_Daniel1997` | `(x, 48000, 1.5, false)` | `time`, `InstantaneousRoughness` (asper) |
| Tonality, ECMA-418-2 4th edition (2025), Sottek Hearing Model | `Tonality_ECMA418_2` | `(x, 48000, 'free-frontal', 0.304, false)`: free-field frontal outer and middle ear filter, 0.304 s skipped in the statistics (the minimum of ECMA-418-2 section 6.2.9) | `timeOut`, `tonalityTDep` (tu_HMS), `tonalityTDepFreqs` (Hz) |

The `time_skip` arguments only affect SQAT's statistics (N5, mean and so on); the time series are saved complete, and the start-up transients are skipped when binning. Every call is wrapped in `try`/`catch`: a failing call is logged and the others continue. `run_sqat.m` saves `build/sqat_bridge/results/<name>_sqat.mat` (MAT v7, readable by `scipy.io.loadmat`) with the time series, the call made, its duration and any warning or error, and writes `versions.txt` (MATLAB release, toolboxes, SQAT tag and commit) and `log.txt`.

Not run here: Aures tonality, the ECMA-418-2 loudness and roughness, fluctuation strength and psychoacoustic annoyance (SQAT has them; ECMA-418-2 fluctuation strength is not part of the pinned release).

## Binning along rpm

`import_results.py` reads the speed of each signal and the SQAT time series and calls `rota.slices.metric_vs_rpm(metric, t_metric, rpm, t_rpm, segment="runup", bin_width=..., min_count=3, skip_start=..., aggregate="mean")`: bins of 250 rpm (ICE) and 500 rpm (EV), the arithmetic mean per bin (the metrics are not levels, so no energy averaging), and the start of each series skipped for the transient of its model: 0.6 s for loudness, sharpness and tonality, 1.5 s for roughness. It writes `build/sqat_bridge/results/binned.csv` and `binned.md` (bin centre, mean, standard deviation, number of samples) and the figure `docs/images/09_sqat_bridge.png`: each metric against rpm, the samples SQAT returned as grey dots and the bin means with their standard deviation. Without results it says so and exits; `--strict` makes it return exit status 1 when any metric of a scenario is missing.

## Running it in CI

Actions, then `sqat-bridge`, then Run workflow; it also runs on pushes that change `examples/09_sqat_bridge/**` or `.github/workflows/sqat-bridge.yml`. The job installs ROTA (`pip install -e ".[plot]"`), runs `export_signals.py`, fetches SQAT at the pinned commit into `build/sqat_bridge/SQAT`, sets up MATLAB with the Signal Processing Toolbox (`matlab-actions/setup-matlab`), runs `run('examples/09_sqat_bridge/run_sqat.m')`, then `import_results.py --strict`, and uploads `build/sqat_bridge/results/` (the `<name>_sqat.mat` files, `binned.csv`, `binned.md`, the figure, `log.txt`, `versions.txt` and the signal metadata) as the artifact `sqat-bridge`. The binned values and the log are added to the job summary.

**The SQAT output is not committed.** As for the MATLAB reference, it stays in the CI artifact (and `build/` is git-ignored).

## Running it locally

```
python examples/09_sqat_bridge/export_signals.py
git clone --branch v2.0 --depth 1 https://github.com/ggrecow/SQAT build/sqat_bridge/SQAT
matlab -batch "run('examples/09_sqat_bridge/run_sqat.m')"
python examples/09_sqat_bridge/import_results.py
```

`run_sqat.m` finds the repository from its own location; set the environment variable `SQAT_DIR` to use a SQAT copy elsewhere. Matplotlib is needed for the figure only.

## References

* ISO 532-1:2017. Acoustics - Methods for calculating loudness - Part 1: Zwicker method.
* DIN 45692:2009. Measurement technique for the simulation of the auditory sensation of sharpness.
* Daniel, P., & Weber, R. (1997). Psychoacoustical roughness: implementation of an optimized model. Acustica, 83(1), 113-123.
* ECMA-418-2, 4th edition (2025). Psychoacoustic metrics for ITT equipment - Part 2 (methods for describing human perception based on the Sottek Hearing Model).
* Felix Greco, G., Merino-Martínez, R., Osses, A., & Langer, S. C. (2023). SQAT: a MATLAB-based toolbox for quantitative sound quality analysis. INTER-NOISE 2023. doi:10.3397/IN_2023_1075.
* Lotinga, M. J. B., Torjussen, M., & Felix Greco, G. (2025). Verified implementations of the Sottek psychoacoustic Hearing Model standardised sound quality metrics (ECMA-418-2 loudness, tonality and roughness). Forum Acusticum 2025. doi:10.61782/fa.2025.0951.
