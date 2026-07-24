"""Does the identity transition line up with the model's learning?

Stacked, shared x = pretraining step:
  top    - cross-seed identifiability (activations, unembedding)
  bottom - held-out cross-entropy loss (mean over seed models, +/- range)

The shaded band marks where token identity emerges (chance -> ceiling). It sits
on the steep part of the loss curve: identifiable cross-seed geometry appears in
the rapid-loss phase and saturates as loss enters its slow tail. (Note most of
the raw CE drop, 11 -> ~7, happens *before* identity emerges — the transition
coincides with the later CE 7 -> 6 portion, not the very first drop.)

    python -m emergence.plot_loss
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
INK = "#1a1a2e"
MUTED = "#8a8a99"
ACT = "#2f6fed"
UNEMBED = "#e8710a"
LOSS = "#6a4c93"


def main():
    ic = json.load(open(os.path.join(OUT_DIR, "curve.json")))
    lc = json.load(open(os.path.join(OUT_DIR, "loss_curve.json")))
    chance = 1 / ic["meta"]["subset_size"]

    irows = [r for r in ic["curve"] if "act" in r and "unembed" in r]
    isteps = [r["step"] for r in irows]
    xs = np.array([s if s > 0 else 0.5 for s in isteps])
    act = np.array([np.mean(r["act"]["subset_acc"]) for r in irows])
    une = np.array([np.mean(r["unembed"]["subset_acc"]) for r in irows])

    lsteps = [r["step"] for r in lc["curve"]]
    lx = np.array([s if s > 0 else 0.5 for s in lsteps])
    ce = np.array([np.mean(r["ce"]) for r in lc["curve"]])
    ce_lo = np.array([np.min(r["ce"]) for r in lc["curve"]])
    ce_hi = np.array([np.max(r["ce"]) for r in lc["curve"]])

    # emergence window: identity leaves chance (>0.3) -> reaches ceiling (>0.85)
    k0 = int(np.argmax(act > 0.3))
    k1 = int(np.argmax(act > 0.85))
    w0 = float(np.sqrt(xs[k0 - 1] * xs[k0]))
    w1 = float(np.sqrt(xs[k1] * xs[min(k1 + 1, len(xs) - 1)]))

    fig, (ax_i, ax_l) = plt.subplots(2, 1, figsize=(7.6, 5.8), dpi=200, sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
    fig.patch.set_facecolor("white")

    for ax in (ax_i, ax_l):
        ax.set_facecolor("white")
        ax.axvspan(w0, w1, color=MUTED, alpha=0.12, lw=0, zorder=1)

    # top: identifiability
    ax_i.axhline(chance, ls=(0, (4, 4)), lw=1.2, color=MUTED, zorder=2)
    ax_i.plot(xs, une, "-s", color=UNEMBED, lw=2.0, ms=4.5, markeredgecolor="white",
              markeredgewidth=0.8, zorder=3, label="unembedding")
    ax_i.plot(xs, act, "-o", color=ACT, lw=2.2, ms=5, markeredgecolor="white",
              markeredgewidth=0.9, zorder=4, label="activations")
    ax_i.set_ylim(0, 1.02)
    ax_i.set_ylabel("identifiability", fontsize=10, color=INK)
    ax_i.text(xs[0], chance + 0.02, "chance", ha="left", va="bottom", fontsize=8, color=MUTED)
    ax_i.legend(loc="center right", frameon=True, fontsize=9,
                labelcolor=INK).get_frame().set_edgecolor(MUTED)

    # bottom: held-out loss
    ax_l.fill_between(lx, ce_lo, ce_hi, color=LOSS, alpha=0.18, lw=0, zorder=2)
    ax_l.plot(lx, ce, "-o", color=LOSS, lw=2.2, ms=5, markeredgecolor="white",
              markeredgewidth=0.9, zorder=3)
    ax_l.set_ylabel("held-out CE loss\n(nats)", fontsize=10, color=INK)
    ax_l.set_xlabel("pretraining step", fontsize=10.5, color=INK)
    ax_l.set_xscale("log")

    for ax in (ax_i, ax_l):
        ax.grid(True, axis="y", color=MUTED, alpha=0.16, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_color(INK)

    fig.text(0.125, 0.975, "Identifiable geometry emerges in the rapid-loss phase",
             fontsize=12.5, color=INK, ha="left", va="bottom")
    fig.text(0.125, 0.925, f"{ic['meta']['model']} · cross-seed · held-out loss over "
             f"{lc['meta']['n_seeds']} seed models", fontsize=9.5, color=MUTED, ha="left", va="bottom")

    out = os.path.join(OUT_DIR, "loss_vs_identity.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
