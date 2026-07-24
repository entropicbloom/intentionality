"""3D surface: recoverability as a landscape over layer x step.

Base plane = (pretraining step, layer); height & color = cross-seed token
identifiability. A flat low plain on the left (chance), a cliff rising around
step 256-1000 (steeper/earlier at the back = deep layers), and a high plateau
with a mid-depth ridge toward the front-right. Exploratory companion to the
heatmap; the grid is cubic-interpolated for a smooth surface.

    python -m emergence.plot_surface3d
"""

import json
import os

import numpy as np
from scipy.ndimage import zoom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3d projection)

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
INK = "#1a1a2e"
MUTED = "#8a8a99"


def fmt_step(s):
    return f"{s // 1000}k" if s >= 1000 else str(s)


def main():
    data = json.load(open(os.path.join(OUT_DIR, "layer_curve.json")))
    meta, curve = data["meta"], data["curve"]
    layers = meta["layers"]
    steps = [r["step"] for r in curve]
    chance = 1 / meta["subset_size"]
    n_pairs = len(curve[0][f"L{layers[0]}"])

    grid = np.array([[np.mean(curve[j][f"L{l}"]) for j in range(len(steps))]
                     for l in layers])  # [layer, step], layer ascending

    # cubic upsample for a smooth landscape (clip overshoot back into [0,1])
    zf = np.clip(zoom(grid, (6, 4), order=3), 0.0, 1.0)
    ny, nx = zf.shape
    xf = np.linspace(0, len(steps) - 1, nx)
    yf = np.linspace(0, len(layers) - 1, ny)
    Xf, Yf = np.meshgrid(xf, yf)

    fig = plt.figure(figsize=(8.6, 5.6), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    surf = ax.plot_surface(Xf, Yf, zf, cmap="viridis", vmin=chance, vmax=float(grid.max()),
                           linewidth=0, antialiased=True, rcount=ny, ccount=nx)

    xticks = list(range(0, len(steps), 2))
    ax.set_xticks(xticks)
    ax.set_xticklabels([fmt_step(steps[i]) for i in xticks], fontsize=8, color=INK)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8.5, color=INK)
    ax.set_zlim(0, 1.0)
    ax.set_zticks([0, chance, 0.5, 1.0])
    ax.set_zticklabels(["0", "chance", "0.5", "1.0"], fontsize=8, color=INK)

    ax.set_xlabel("pretraining step", fontsize=10, color=INK, labelpad=8)
    ax.set_ylabel("layer (depth)", fontsize=10, color=INK, labelpad=6)
    ax.set_zlabel("identifiability", fontsize=10, color=INK, labelpad=4)

    ax.view_init(elev=30, azim=-58)
    ax.set_box_aspect((1.5, 1.0, 0.7))
    try:
        ax.set_proj_type("persp")
    except Exception:
        pass
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("white")
        pane.pane.set_edgecolor(MUTED)
        pane.pane.set_alpha(0.15)

    pair_note = f"mean of {n_pairs} seed pairs" if n_pairs > 1 else "cross-seed"
    fig.suptitle("Recoverability landscape over layer x step",
                 x=0.05, y=0.96, ha="left", fontsize=12.5, color=INK)
    fig.text(0.05, 0.905, f"{meta['model']} · activation identifiability · {pair_note} · "
             "height = color = identifiability", fontsize=9.5, color=MUTED, ha="left")

    out = os.path.join(OUT_DIR, "emergence_surface3d.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
