"""Layer x step heatmap + layer-averaged marginal line.

Top: rows = layer (depth, shallow at bottom), columns = pretraining step
(log-spaced), color = cross-seed token identifiability (mean over seed pairs).
Bottom (shared x): mean identifiability across layers, with a shaded +/-1 SD
band showing how much layers disagree at each step — wide at the emergence onset
(deep-first) and narrow once every layer has organized.

    python -m emergence.plot_heatmap
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
LINE = "#2f6fed"


def fmt_step(s):
    return f"{s // 1000}k" if s >= 1000 else str(s)


def main():
    data = json.load(open(os.path.join(OUT_DIR, "layer_curve.json")))
    meta, curve = data["meta"], data["curve"]
    layers = meta["layers"]
    steps = [r["step"] for r in curve]
    chance = 1 / meta["subset_size"]
    n_pairs = len(curve[0][f"L{layers[0]}"])
    n = len(steps)

    grid = np.array([[np.mean(curve[j][f"L{l}"]) for j in range(n)]
                     for l in layers])          # [layer, step], layer ascending
    layer_mean = grid.mean(axis=0)              # avg across layers, per step
    layer_sd = grid.std(axis=0)

    vmin, vmax = chance, float(grid.max())
    fig = plt.figure(figsize=(8.6, 6.4), dpi=200)
    gs = fig.add_gridspec(2, 2, width_ratios=[32, 1], height_ratios=[3.0, 1.5],
                          hspace=0.10, wspace=0.03)
    ax_h = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax_l = fig.add_subplot(gs[1, 0], sharex=ax_h)
    fig.patch.set_facecolor("white")

    # --- heatmap ---
    im = ax_h.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                     vmin=vmin, vmax=vmax)
    for i in range(len(layers)):
        for j in range(n):
            v = grid[i, j]
            norm = (v - vmin) / (vmax - vmin + 1e-9)
            ax_h.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                      fontsize=7, color="black" if norm > 0.55 else "white")
    ax_h.set_yticks(range(len(layers)))
    ax_h.set_yticklabels([f"L{l}" for l in layers], fontsize=9, color=INK)
    ax_h.set_ylabel("layer (depth)", fontsize=10.5, color=INK)
    ax_h.tick_params(axis="x", length=0, labelbottom=False)
    ax_h.tick_params(axis="y", length=0)
    for s in ax_h.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(f"identifiability (floor = chance {chance:.2f})", fontsize=9, color=INK)
    cbar.ax.tick_params(labelsize=8, colors=INK)
    cbar.outline.set_edgecolor(MUTED)

    # --- marginal line: mean across layers +/- 1 SD ---
    x = np.arange(n)
    ax_l.axhline(chance, ls=(0, (4, 4)), lw=1.2, color=MUTED, zorder=1)
    ax_l.fill_between(x, layer_mean - layer_sd, layer_mean + layer_sd,
                      color=LINE, alpha=0.18, lw=0, zorder=2)
    ax_l.plot(x, layer_mean, "-o", color=LINE, lw=2.0, ms=4.5,
              markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax_l.set_ylim(0, 1.02)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels([fmt_step(s) for s in steps], fontsize=8.5, color=INK)
    ax_l.set_xlabel("pretraining step", fontsize=10.5, color=INK)
    ax_l.set_ylabel("mean over\nlayers", fontsize=9.5, color=INK)
    ax_l.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for t in ax_l.get_xticklabels() + ax_l.get_yticklabels():
        t.set_color(INK)
    ax_l.grid(True, axis="y", color=MUTED, alpha=0.16, lw=0.8)
    for side in ("top", "right"):
        ax_l.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_l.spines[side].set_color(MUTED)
    ax_l.text(0, chance + 0.03, "chance", ha="left", va="bottom", fontsize=8, color=MUTED)
    ax_l.text(0.985, 0.30, "shaded = ±1 SD across layers", transform=ax_l.transAxes,
              ha="right", va="bottom", fontsize=8, color=MUTED)

    ax_h.set_title("The emergence wave: recoverable geometry across layer x step",
                   fontsize=11.5, color=INK, pad=22, loc="left")
    pair_note = f"mean of {n_pairs} seed pairs" if n_pairs > 1 else "cross-seed"
    ax_h.text(0.0, 1.02, f"{meta['model']} · activation identifiability · subset metric · {pair_note}",
              transform=ax_h.transAxes, fontsize=9.5, color=MUTED, ha="left", va="bottom")

    out = os.path.join(OUT_DIR, "emergence_heatmap.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
