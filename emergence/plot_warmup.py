"""Confound check: is the emergence window just the LR warmup?

Pythia-160m warms the learning rate linearly over the first 1430 steps (1% of
143000), then cosine-decays 6e-4 -> 6e-5. Our identity-emergence window (~step
256 -> 1000, saturating ~1000-2000) sits *inside* warmup and saturates right
around warmup end -- so absolute step numbers are schedule-dependent and should
not be reified. (The internal structure -- depth ordering, family ordering -- is
a single-scalar-per-step schedule cannot explain, so that part is unaffected.)

    python -m emergence.plot_warmup
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
LR_C = "#2e7d4f"

MAX_LR, MIN_LR = 6e-4, 6e-5
WARMUP, TOTAL = 1430, 143000


def lr_at(step):
    step = np.asarray(step, dtype=float)
    warm = MAX_LR * step / WARMUP
    dr = np.clip((step - WARMUP) / (TOTAL - WARMUP), 0, 1)
    cos = MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + np.cos(np.pi * dr))
    return np.where(step < WARMUP, warm, cos)


def main():
    d = json.load(open(os.path.join(OUT_DIR, "curve.json")))
    meta = d["meta"]
    chance = 1 / meta["subset_size"]
    rows = [r for r in d["curve"] if "act" in r and "unembed" in r]
    steps = [r["step"] for r in rows]
    xs = np.array([s if s > 0 else 0.5 for s in steps])
    act = np.array([np.mean(r["act"]["subset_acc"]) for r in rows])
    une = np.array([np.mean(r["unembed"]["subset_acc"]) for r in rows])

    fig, (ax_i, ax_l) = plt.subplots(2, 1, figsize=(7.6, 5.8), dpi=200, sharex=True,
                                     gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.12})
    fig.patch.set_facecolor("white")
    for ax in (ax_i, ax_l):
        ax.set_facecolor("white")
        ax.axvspan(0.32, WARMUP, color=MUTED, alpha=0.12, lw=0, zorder=1)
        ax.axvline(WARMUP, ls=(0, (3, 3)), lw=1.2, color=MUTED, zorder=2)

    # top: identifiability
    ax_i.axhline(chance, ls=(0, (4, 4)), lw=1.2, color=MUTED, zorder=2)
    ax_i.plot(xs, une, "-s", color=UNEMBED, lw=2.0, ms=4.5, markeredgecolor="white",
              markeredgewidth=0.8, zorder=3, label="unembedding")
    ax_i.plot(xs, act, "-o", color=ACT, lw=2.2, ms=5, markeredgecolor="white",
              markeredgewidth=0.9, zorder=4, label="activations")
    ax_i.set_ylim(0, 1.02)
    ax_i.set_ylabel("identifiability", fontsize=10, color=INK)
    ax_i.text(WARMUP, 1.05, "warmup ends (1430)", ha="center", va="bottom",
              fontsize=8.5, color=MUTED, style="italic",
              transform=ax_i.get_xaxis_transform())
    ax_i.text(xs[0], chance + 0.02, "chance", ha="left", va="bottom", fontsize=8, color=MUTED)
    ax_i.legend(loc="center right", frameon=True, fontsize=9,
                labelcolor=INK).get_frame().set_edgecolor(MUTED)

    # bottom: LR schedule
    grid = np.logspace(0, np.log10(TOTAL), 500)
    ax_l.plot(grid, lr_at(grid) * 1e4, "-", color=LR_C, lw=2.2, zorder=3)
    ax_l.plot(xs[xs >= 1], lr_at(xs[xs >= 1]) * 1e4, "o", color=LR_C, ms=4.5,
              markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    ax_l.set_ylabel("learning rate\n(×1e-4)", fontsize=10, color=INK)
    ax_l.set_xlabel("pretraining step", fontsize=10.5, color=INK)
    ax_l.set_xscale("log")
    ax_l.set_ylim(0, 6.6)

    for ax in (ax_i, ax_l):
        ax.grid(True, axis="y", color=MUTED, alpha=0.16, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_color(INK)

    fig.text(0.125, 0.975, "The emergence window is the LR warmup — step numbers are schedule-dependent",
             fontsize=11, color=INK, ha="left", va="bottom")
    fig.text(0.125, 0.93, f"{meta['model']} · cross-seed · linear warmup to step 1430, then cosine decay",
             fontsize=9, color=MUTED, ha="left", va="bottom")

    out = os.path.join(OUT_DIR, "warmup.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
