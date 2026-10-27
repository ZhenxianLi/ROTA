"""Speed from a tachometer and from the signal alone, against the true speed.

The signal of ``01_order_map.py`` comes with a tachometer of 12 pulses per
revolution (30 us timing jitter, about 0.3 % of the pulses lost). Two speed
estimates are compared with the speed that went into the signal:
``rota.speed.tacho_to_rpm`` (pulse edges to rpm) and ``rota.speed.rpm_from_ridge``
(tacholess, from the signal alone, with a per-frame ``confidence`` and an
``at_bound`` flag).

Figure: (a) the speed curves, (b) their relative errors, (c, d) the tacholess
estimator run again with a search range that is deliberately too narrow (600 to
3500 rpm): above the limit the estimate does not sit on the bound but jumps to a
wrong value, and only ``at_bound`` tells it reliably (``confidence`` falls in some
of those frames, not in all).

Settings that the data call for (see the docstring of ``rpm_from_ridge``):
``orders=(1.5, 3, 4.5, 6)``, because with the default integer comb the half-order
signal locks at exactly 1.5 times the speed; ``whiten=None``, because the fixed
250 Hz tone sits on a comb tooth at a speed above the search range and, with
whitening, briefly outscores the orders there (``moving=True`` would also remove
it). The default ``whiten="auto"`` widens the whitening with the sweep of the
lines and tracks this run-up as well.

No amplitude convention is involved (the tacholess estimator uses ratios of
spectral levels only). Speeds are in rpm, times in s.

Run:  python examples/02_tacho_and_tacholess_speed.py   (needs matplotlib, about 2 s)
Writes docs/images/02_tacho_and_tacholess_speed.png.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import rota

FS = 8000.0  # sampling rate, Hz
DURATION = 6.0  # s
IDLE_RPM, TOP_RPM = 900.0, 4500.0
T_RUNUP = 1.2  # end of the idle hold, s
PPR = 12  # tachometer pulses per revolution
COMB = (1.5, 3.0, 4.5, 6.0)  # comb orders of the tacholess estimator
FIG_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "images" / "02_tacho_and_tacholess_speed.png"
)

ORDERS = [
    {"order": 0.5, "amplitude": 0.15},
    {"order": 1.0, "amplitude": 0.35},
    {"order": 1.5, "amplitude": 1.0, "gain_rpm": [IDLE_RPM, TOP_RPM], "gain": [0.7, 1.3]},
    {"order": 3.0, "amplitude": 0.5},
    {"order": 4.5, "amplitude": 0.3},
    {"order": 6.0, "amplitude": 0.2},
]


def build_signal():
    """Idle hold, then linear run-up; orders, resonance, tone, noise and a tachometer."""
    t = np.arange(round(DURATION * FS)) / FS
    rpm = rota.synth.rpm_profiles(
        t, "linear", start=IDLE_RPM, stop=TOP_RPM, t_start=T_RUNUP, t_stop=DURATION
    )
    return rota.synth.synth_runup(
        FS,
        rpm,
        ORDERS,
        tones=[{"freq": 250.0, "amplitude": 0.12}],
        resonances=[{"freq": 180.0, "q": 6.0, "gain": 4.0}],
        noise_rms=0.05,
        tacho_ppr=PPR,
        tacho_jitter=30e-6,  # s
        tacho_drop=0.003,  # probability of losing a pulse
        seed=7,
    )


def ridge(sig, rpm_range):
    """Tacholess estimate; returns the result, the truth at the frame times and the warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = rota.speed.rpm_from_ridge(sig.x, FS, rpm_range, orders=COMB, whiten=None)
    truth = np.interp(res.t, sig.t, sig.rpm)
    return res, truth, [str(w.message) for w in caught]


def shade(ax, t, flag, label=None):
    """Grey bands over the frames where ``flag`` is True."""
    ax.fill_between(t, 0, 1, where=flag, step="mid", transform=ax.get_xaxis_transform(),
                    color="0.6", alpha=0.3, lw=0, label=label)


def main():
    plt.switch_backend("Agg")  # file output only; no display needed
    sig = build_signal()

    # Tachometer to rpm, at every sample.
    rpm_tacho, info = rota.speed.tacho_to_rpm(sig.tacho, FS, ppr=PPR, return_info=True)
    e_tacho = rota.metrics.track_error(rpm_tacho, sig.rpm)

    # Tacholess, search range wide enough, and one that is too narrow.
    res, truth, warn_ok = ridge(sig, (600.0, 6000.0))
    e_ridge = rota.metrics.track_error(res.rpm, truth, at_bound=res.at_bound)
    narrow, truth_n, warn_narrow = ridge(sig, (600.0, 3500.0))
    outside = truth_n > narrow.rpm_range[1]  # frames where the true speed left the range

    print(f"Signal: {DURATION:g} s at {FS:g} Hz, {IDLE_RPM:g} -> {TOP_RPM:g} rpm; "
          f"tacho {PPR} pulses/rev, {sig.pulse_times.size} pulses kept, "
          f"{sig.config['tacho_n_dropped']} dropped")
    print(f"tacho_to_rpm      : {info['n_missing']} missing pulses filled, {info['n_spurious']} "
          f"spurious removed; relative error over {e_tacho['n']} samples: median "
          f"{100 * e_tacho['median_rel_error']:.3f} %, max {100 * e_tacho['max_rel_error']:.2f} %")
    print(f"rpm_from_ridge    : range {res.rpm_range[0]:g}-{res.rpm_range[1]:g} rpm, {res.t.size} frames; "
          f"relative error median {100 * e_ridge['median_rel_error']:.3f} %, "
          f"max {100 * e_ridge['max_rel_error']:.2f} %, within 3 %: "
          f"{100 * e_ridge['within'][0.03]:.1f} %, at_bound {100 * res.bound_fraction:.1f} %, "
          f"median confidence {np.median(res.confidence):.2f}, warnings: {len(warn_ok)}")
    ok = ~outside & ~narrow.at_bound
    rel = np.abs(narrow.rpm / truth_n - 1.0)
    print(f"too-narrow range  : {narrow.rpm_range[0]:g}-{narrow.rpm_range[1]:g} rpm, true speed above the "
          f"range in {outside.sum()} of {outside.size} frames; flagged at_bound there: "
          f"{100 * narrow.at_bound[outside].mean():.0f} % ({narrow.beyond[outside].sum()} of them "
          f"by the out-of-range test 'beyond'), wrongly flagged inside: "
          f"{100 * narrow.at_bound[~outside].mean():.0f} %")
    print(f"                    unflagged in-range frames: max error {100 * rel[ok].max():.2f} %; "
          f"confidence median {np.median(narrow.confidence[ok]):.2f} (trusted) versus "
          f"{np.median(narrow.confidence[outside]):.2f} (true speed above the range); "
          f"RuntimeWarning raised: {len(warn_narrow) > 0}")

    # ---- figure ----------------------------------------------------------------------------
    blue, orange, grey = "#0072B2", "#D55E00", "0.45"
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig, (ax_a, ax_b, ax_c, ax_d) = plt.subplots(
        4, 1, figsize=(8.0, 9.0), dpi=150, sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2.3, 1.3, 2.0, 1.2]},
    )

    ax_a.plot(sig.t, sig.rpm, color="k", lw=2.6, alpha=0.35, label="true speed (synth)")
    ax_a.plot(sig.t, rpm_tacho, color=blue, lw=1.0, label="tacho_to_rpm")
    ax_a.plot(res.t, res.rpm, ls="none", marker="o", ms=3.0, color=orange, mfc="none",
              label="rpm_from_ridge, 600 to 6000 rpm")
    ax_a.set_title("(a) Speed", loc="left")
    ax_a.set_ylabel("Speed (rpm)")
    ax_a.legend(loc="upper left")

    ax_b.axhline(0.0, color="k", lw=0.6)
    ax_b.plot(sig.t, 100.0 * (rpm_tacho / sig.rpm - 1.0), color=blue, lw=0.8, label="tacho_to_rpm")
    ax_b.plot(res.t, 100.0 * (res.rpm / truth - 1.0), ls="none", marker="o", ms=3.0,
              color=orange, mfc="none", label="rpm_from_ridge")
    ax_b.set_title("(b) Relative error of the speed", loc="left")
    ax_b.set_ylabel("Error (%)")
    ax_b.set_ylim(-2.5, 2.5)
    ax_b.legend(loc="upper right", ncol=2)

    ax_c.plot(sig.t, sig.rpm, color="k", lw=2.6, alpha=0.35, label="true speed (synth)")
    ax_c.plot(narrow.t, narrow.rpm, ls="none", marker="o", ms=3.0, color=orange, mfc="none",
              label="rpm_from_ridge, 600 to 3500 rpm")
    ax_c.axhline(narrow.rpm_range[1], color=grey, lw=0.8, ls="--", label="upper limit of rpm_range")
    shade(ax_c, narrow.t, narrow.at_bound, "at_bound")
    ax_c.annotate("wrong value, not on the bound:\nflagged by at_bound (beyond)", (5.2, 2700.0),
                  xytext=(3.3, 1500.0), fontsize=7, ha="center", va="center",
                  arrowprops={"arrowstyle": "->", "lw": 0.6})
    ax_c.set_title("(c) Search range too narrow: the estimate leaves the truth above the limit", loc="left")
    ax_c.set_ylabel("Speed (rpm)")
    ax_c.legend(loc="upper left")

    ax_d.plot(res.t, res.confidence, color=grey, lw=1.0, label="confidence, range (a)")
    ax_d.plot(narrow.t, narrow.confidence, color=orange, lw=1.2, label="confidence, range (c)")
    shade(ax_d, narrow.t, narrow.at_bound)
    ax_d.set_title("(d) Confidence of the tacholess estimate (0 to 1)", loc="left")
    ax_d.set_ylabel("Confidence")
    ax_d.set_ylim(-0.04, 1.08)  # a clean track sits at 1: keep it off the frame
    ax_d.set_xlabel("Time (s)")
    ax_d.legend(loc="lower left", ncol=2)
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.grid(alpha=0.3)
        ax.set_xlim(0.0, DURATION)

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Figure written to {FIG_PATH.relative_to(FIG_PATH.parent.parent.parent)}")


if __name__ == "__main__":
    main()
