"""A high CKA does not imply recoverable identity.

Two panels (activations, unembedding). Each overlays, over training step and on
one shared 0-1 axis:
  - identifiability: label-free exact token recovery (subset-Gram matching)
  - CKA: the standard aggregate representational-similarity scalar

Both are cross-seed, mean over seed pairs. The honest point is NOT that CKA is
blind to the timing (normalized to its own range, CKA rises in the same window).
It is that CKA's absolute scale is uninterpretable: it is already ~0.87
(unembedding) or ~0.65 (activations) at random init (dotted reference line) and
its whole span up to convergence is small/non-monotonic — so the same CKA value
covers both unrecoverable and perfectly recoverable identity. Exact recovery,
anchored at a known chance floor, separates what CKA compresses.

    python -m emergence.plot_cka
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
IDENT = "#2f6fed"   # exact identifiability (hero)
CKA = "#e8710a"     # aggregate similarity

PANELS = [("act", "activations"), ("unembed", "unembedding")]


def main():
    data = json.load(open(os.path.join(OUT_DIR, "curve.json")))
    meta, curve = data["meta"], data["curve"]
    chance = 1 / meta["subset_size"]
    rows = [r for r in curve if all(f in r for f, _ in PANELS)]
    steps = [r["step"] for r in rows]
    xs = np.array([s if s > 0 else 0.5 for s in steps])
    n_pairs = len(rows[0]["act"]["subset_acc"])

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), dpi=200, sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (fam, label) in zip(axes, PANELS):
        ax.set_facecolor("white")
        ident = np.array([np.mean(r[fam]["subset_acc"]) for r in rows])
        cka = np.array([np.mean(r[fam]["cka"]) for r in rows])
        cka_init = float(cka[0])  # CKA of two independently random-init nets

        # reference lines: identity's chance floor, and CKA's random-init floor
        ax.axhline(chance, ls=(0, (4, 4)), lw=1.2, color=MUTED, zorder=2)
        ax.axhline(cka_init, ls=(0, (1, 2)), lw=1.4, color=CKA, alpha=0.8, zorder=2)
        ax.text(xs[0], cka_init + 0.02, f"init {cka_init:.2f}",
                ha="left", va="bottom", fontsize=8, color=CKA)

        ax.plot(xs, cka, "-s", color=CKA, lw=2.0, ms=4.5, markeredgecolor="white",
                markeredgewidth=0.8, zorder=3, label="CKA (aggregate similarity)")
        ax.plot(xs, ident, "-o", color=IDENT, lw=2.2, ms=5, markeredgecolor="white",
                markeredgewidth=0.9, zorder=4, label="identifiability (exact recovery)")

        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("pretraining step", fontsize=10, color=INK)
        ax.set_title(label, fontsize=11.5, color=INK, pad=8)
        ax.grid(True, axis="y", color=MUTED, alpha=0.16, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_color(INK)

    axes[0].set_ylabel("cross-seed score (0–1)", fontsize=10, color=INK)
    axes[0].text(0.34, chance + 0.02, "chance", ha="left", va="bottom", fontsize=8, color=MUTED)
    axes[1].legend(loc="lower right", frameon=True, fontsize=9, borderpad=0.7,
                   labelcolor=INK).get_frame().set_edgecolor(MUTED)

    fig.suptitle("A high CKA does not imply recoverable identity",
                 x=0.02, y=1.0, ha="left", fontsize=12.5, color=INK)
    pair_note = f"mean of {n_pairs} seed pairs" if n_pairs > 1 else "cross-seed"
    fig.text(0.02, 0.925, f"{meta['model']} · {meta['n_tokens']} concept tokens · {pair_note}"
             " · dotted = CKA at random init", fontsize=9.5, color=MUTED, ha="left")

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = os.path.join(OUT_DIR, "cka_vs_identifiability.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
