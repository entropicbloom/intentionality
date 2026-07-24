"""Layer-resolved emergence figure.

One cross-seed identifiability curve per probed layer, colored by a sequential
single-hue ramp (light = shallow, dark = deep). Shows how the emergence timing
and plateau height of recoverable geometry vary with depth.

    python -m emergence.plot_layers
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

INK = "#1a1a2e"
MUTED = "#8a8a99"


def main():
    data = json.load(open(os.path.join(OUT_DIR, "layer_curve.json")))
    meta, curve = data["meta"], data["curve"]
    layers = meta["layers"]
    chance = 1 / meta["subset_size"]

    steps = [r["step"] for r in curve]
    xs = [s if s > 0 else 0.5 for s in steps]

    # sequential single-hue ramp: light (shallow) -> dark (deep)
    cmap = plt.cm.Blues
    positions = np.linspace(0.45, 1.0, len(layers))  # avoid pale-on-white for L1
    colors = [cmap(p) for p in positions]

    fig, ax = plt.subplots(figsize=(7.6, 4.5), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axhline(chance, ls=(0, (4, 4)), lw=1.4, color=MUTED, zorder=1)
    ax.text(xs[0], chance - 0.035, "chance", ha="left", va="top", fontsize=9, color=MUTED)

    n_pairs = len(curve[0][f"L{layers[0]}"]) if curve else 1
    for li, layer in enumerate(layers):
        ys = [np.mean(r[f"L{layer}"]) for r in curve]
        ax.plot(xs, ys, "-", lw=2.0, color=colors[li], zorder=3)
        ax.plot(xs, ys, "o", ms=4.5, color=colors[li], markeredgecolor="white",
                markeredgewidth=0.8, zorder=4)

    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("pretraining step", fontsize=10.5, color=INK)
    ax.set_ylabel("cross-seed token identifiability", fontsize=10.5, color=INK)
    ax.set_title("Recoverable geometry emerges deep layers first",
                 fontsize=11.5, color=INK, pad=24, loc="left")
    subtitle = (f"{meta['model']} · activations by layer (of {meta['n_layers']}) · "
                f"cross-seed · subset metric")
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=MUTED, ha="left", va="bottom")

    # colorbar as the depth legend (sequential encoding)
    norm = Normalize(vmin=min(layers), vmax=max(layers))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=layers, pad=0.015, aspect=22)
    cbar.set_label("layer (depth)", fontsize=9.5, color=INK)
    cbar.ax.tick_params(labelsize=8.5, colors=INK)
    cbar.outline.set_edgecolor(MUTED)

    ax.grid(True, which="major", axis="y", color=MUTED, alpha=0.18, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(INK)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "emergence_layers.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
