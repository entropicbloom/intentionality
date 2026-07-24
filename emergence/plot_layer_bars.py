"""Two-panel bar chart: how identifiability is organized across layers at two
training snapshots — the emergence onset vs the end of training.

Left (step 256): identifiability increases monotonically with depth — recoverable
geometry appears deep-first. Right (final step): an inverted-U — mid-depth layers
are the most seed-stable while the deepest and shallowest sit lower.

    python -m emergence.plot_layer_bars
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

# the two snapshots to contrast
SNAPSHOTS = [(256, "step 256", "emergence onset"),
             (143000, "step 143,000", "end of training")]


def main():
    data = json.load(open(os.path.join(OUT_DIR, "layer_curve.json")))
    meta, curve = data["meta"], data["curve"]
    layers = meta["layers"]
    chance = 1 / meta["subset_size"]
    by_step = {r["step"]: r for r in curve}

    cmap = plt.cm.Blues
    colors = [cmap(p) for p in np.linspace(0.45, 1.0, len(layers))]
    x = np.arange(len(layers))
    n_pairs = len(by_step[SNAPSHOTS[0][0]][f"L{layers[0]}"])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), dpi=200, sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (step, label, tag) in zip(axes, SNAPSHOTS):
        ax.set_facecolor("white")
        per_layer = [by_step[step][f"L{l}"] for l in layers]          # list of per-pair lists
        means = np.array([np.mean(v) for v in per_layer])
        los = np.array([np.min(v) for v in per_layer])
        his = np.array([np.max(v) for v in per_layer])
        ax.bar(x, means, width=0.72, color=colors, zorder=3)
        if n_pairs > 1:
            ax.errorbar(x, means, yerr=[means - los, his - means], fmt="none",
                        ecolor=INK, elinewidth=1.2, capsize=3, zorder=5)
            for xi, v in zip(x, per_layer):  # individual pair estimates
                ax.scatter([xi] * len(v), v, s=11, color=INK, alpha=0.5,
                           linewidths=0, zorder=6)
        for xi, m, top in zip(x, means, his):
            ax.text(xi, top + 0.025, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK)

        ax.axhline(chance, ls=(0, (4, 4)), lw=1.3, color=MUTED, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.set_xlabel("layer (depth)", fontsize=10, color=INK)
        ax.set_title(f"{label}", fontsize=11.5, color=INK, pad=16, loc="center")
        ax.text(0.5, 1.005, tag, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color=MUTED)

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_color(INK)
        ax.grid(True, axis="y", color=MUTED, alpha=0.16, lw=0.8, zorder=0)

    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("cross-seed token identifiability", fontsize=10, color=INK)
    axes[1].text(len(layers) - 1, chance + 0.015, "chance", ha="right", va="bottom",
                 fontsize=8.5, color=MUTED)

    fig.suptitle("How layers organize: deep-first at onset, mid-peaked at convergence",
                 fontsize=12, color=INK, x=0.02, ha="left", y=1.0)
    pair_note = (f"mean of {n_pairs} seed pairs (whiskers = range, dots = pairs)"
                 if n_pairs > 1 else "cross-seed")
    subtitle = f"{meta['model']} · activation identifiability · subset metric · {pair_note}"
    fig.text(0.02, 0.925, subtitle, fontsize=9.5, color=MUTED, ha="left")

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out = os.path.join(OUT_DIR, "layer_bars.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
