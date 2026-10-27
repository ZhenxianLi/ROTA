"""Bridge to MoSQITo (optional external dependency): tone-to-noise ratio and metrics along rpm.

The whole module is skipped when MoSQITo is not installed, so the suite passes without it. Data come
from rota.synth only. Two things are checked:

* On a steady tone in noise, ROTA's ``tone_prominence`` and MoSQITo's ``tnr_ecma_st`` give the same
  tone-to-noise ratio within 1.5 dB, and ROTA's agrees with the analytic truth of the stimulus
  (tone power over the noise power in the critical band, from the stems of ``synth_tone_noise``).
  Convention differences: ROTA takes the tone frequency and a Hann periodogram as PSD, MoSQITo
  detects the tones itself from the signal (its reported frequency lies up to one line above the
  true one); ROTA follows ECMA-418-1 (3rd edition, 2024), MoSQITo is catalogued against ECMA-74:2019.
  The stimulus has an exact flat noise spectrum. With Gaussian noise (periodogram scatter) MoSQITo
  1.2.1 was found to return 2 to 3 dB for some tones between about 130 and 380 Hz, which is not
  tested here (see examples/07_mosqito_bridge.py).
* ``rota.slices.metric_vs_rpm`` runs on MoSQITo's time-varying loudness and sharpness (the time
  axis and the units are the ones MoSQITo documents) and puts them on the right speed bins.
"""

import subprocess
import sys

import numpy as np
import pytest
from scipy import signal

pytest.importorskip("matplotlib")  # imported by mosqito, not declared by it
pytest.importorskip("mosqito")

import rota  # noqa: E402
from mosqito.sq_metrics import loudness_zwtv, sharpness_din_from_loudness, tnr_ecma_st  # noqa: E402

FS = 48000.0  # the rate MoSQITo's time-varying loudness needs


def _db(x):
    return 10.0 * np.log10(x)


@pytest.mark.parametrize("tnr_cb", [8.0, 15.0, 22.0])
@pytest.mark.parametrize("f0", [130.0, 500.0, 1000.0, 4000.0])
def test_tone_prominence_agrees_with_mosqito_tnr(f0, tnr_cb):
    # Critical-band TNR set through the broadband ratio (synth_tone_noise docstring):
    # TNR_cb = tnr_db + 10 log10((fs / 2) / critical_bandwidth(f0)), white noise.
    dfc = rota.tonal.critical_bandwidth(f0)
    s = rota.synth.synth_tone_noise(FS, 2.0, 0.05, tnr_cb - _db(0.5 * FS / dfc), tone_freq=f0, seed=2)
    truth = _db(np.mean(s.tone ** 2) / (np.mean(s.noise ** 2) / (0.5 * FS) * dfc))
    assert truth == pytest.approx(tnr_cb, abs=0.05)  # the stimulus is what it says

    f, psd = signal.welch(s.x, FS, window="hann", nperseg=int(FS), scaling="density")  # 1 Hz lines, Pa^2/Hz
    ours = rota.tonal.tone_prominence(f, psd, f0).tnr
    assert ours == pytest.approx(truth, abs=1.0)

    _t_tnr, tnr, _prominent, freqs = tnr_ecma_st(s.x, FS, prominence=False)  # Pa in, tones detected
    tnr, freqs = np.asarray(tnr, dtype=float).ravel(), np.asarray(freqs, dtype=float).ravel()  # object arrays
    assert tnr.size == 1, "MoSQITo should find the one tone"
    assert abs(freqs[0] - f0) <= 1.0
    assert abs(tnr[0] - ours) <= 1.5


@pytest.fixture(scope="module")
def loudness_runup():
    """A 2.5 s run-up (1500 to 4500 rpm, 48 kHz) whose level rises by 10.5 dB, and MoSQITo's loudness."""
    n = int(2.5 * FS)
    t = np.arange(n) / FS
    rpm = rota.synth.rpm_profiles(t, "linear", start=1500.0, stop=4500.0)
    sig = rota.synth.synth_runup(
        FS, rpm, orders=[{"order": 2, "amplitude": 0.5}, {"order": 4, "amplitude": 0.2}], noise_rms=0.02, seed=5
    )
    x = sig.x * np.linspace(0.3, 1.0, n)
    n_sone, n_spec, _bark, t_n = loudness_zwtv(x, FS)
    return {"t": t, "rpm": sig.rpm, "N": np.asarray(n_sone), "N_spec": n_spec, "t_n": np.asarray(t_n)}


def test_metric_vs_rpm_on_mosqito_loudness(loudness_runup):
    d = loudness_runup
    assert d["N"].shape == d["t_n"].shape  # one value per MoSQITo time step (2 ms)
    m = rota.slices.metric_vs_rpm(
        d["N"], d["t_n"], d["rpm"], d["t"], bin_width=250.0, skip_start=0.6, aggregate=("percentile", 95), min_count=3
    )
    assert m.segment == "runup" and m.aggregate == "percentile" and m.q == 95.0
    assert m.valid.sum() >= 8
    assert int(m.count.sum()) == int(m.used.sum())
    assert d["t_n"][m.used].min() >= d["t_n"][0] + 0.6 - 1e-9  # the start-up transient is skipped
    assert m.centres[m.valid].min() > 1500.0 and m.centres[m.valid].max() < 4500.0

    # The binning is what it says: recompute the percentile of each bin from the samples that were used.
    ctrl, used_n = m.control[m.used], d["N"][m.used]
    for j in np.flatnonzero(m.valid):
        in_bin = (ctrl >= m.edges[j]) & (ctrl < m.edges[j + 1])
        assert m.value[j] == pytest.approx(np.percentile(used_n[in_bin], 95.0), rel=1e-12)

    # The level rises by 10.5 dB over the run-up (and the order frequency with it): the loudness must follow.
    v = m.value[m.valid]
    assert v[-1] > 1.5 * v[0]
    assert np.all(np.diff(v) > -0.05 * v[:-1])


def test_metric_vs_rpm_on_mosqito_sharpness(loudness_runup):
    d = loudness_runup
    s = np.asarray(sharpness_din_from_loudness(d["N"], d["N_spec"], weighting="din"))  # DIN 45692, acum
    assert s.shape == d["N"].shape
    m = rota.slices.metric_vs_rpm(
        s, d["t_n"], d["rpm"], d["t"], bin_width=250.0, skip_start=0.6, aggregate="median", min_count=3
    )
    assert m.valid.sum() >= 8
    v = m.value[m.valid]
    assert np.all(np.isfinite(v)) and np.all((v > 0.1) & (v < 10.0))  # acum, plausible for a 100 to 600 Hz tone


def test_rota_does_not_import_mosqito():
    code = "import sys, rota; assert 'mosqito' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
