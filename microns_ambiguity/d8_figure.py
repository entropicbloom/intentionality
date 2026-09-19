"""Diagram of the dihedral symmetries of the orientation ring (D8) and what the anisotropy removes.

Orientation is periodic in 180 degrees, so the eight 22.5-degree classes sit on a ring at angle 2*theta.
A circulant class-Gram is invariant under the 8 rotations and 8 reflections of that ring (D8, 16 elements).
The 90-degree anisotropy marks one point, which removes the rotations but not the reflection through
that point, so 45 and 135 degrees stay interchangeable.
Run: .venv/bin/python -m microns_ambiguity.d8_figure  -> outputs/paper/fig_d8_symmetries.{png,pdf}
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "paper")
K = 8
ORI = np.arange(K) * 180 / K                    # 0, 22.5, ..., 157.5
ANG = np.deg2rad(2 * ORI)                       # position on the ring: doubled angle, 0 deg at the top
POS = np.stack([np.sin(ANG), np.cos(ANG)], 1)   # (x, y), clockwise from the top
CARD = "#1f4e79"; OBL = "#c9a227"; GREY = "#8a8a8a"; MARK = "#c0392b"


def ring(ax, title, highlight=None, node_colors=None):
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.75, 1.55)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=GREY, lw=1.2, zorder=1))
    for k in range(K):
        x, y = POS[k]
        c = node_colors[k] if node_colors else (CARD if k % 4 == 0 else (OBL if k % 4 == 2 else GREY))
        r = 0.16 if highlight == k else 0.11
        ax.add_patch(plt.Circle((x, y), r, color=c, zorder=3, ec="white", lw=1.5))
        lx, ly = 1.32 * x, 1.32 * y
        lab = f"{ORI[k]:g}°"
        ax.text(lx, ly, lab, ha="center", va="center", fontsize=9, zorder=4)
    ax.set_title(title, fontsize=10.5, pad=8)


def arc_arrow(ax, k_from, k_to, color, style="-", rad=0.25, lw=1.6):
    a = FancyArrowPatch(POS[k_from] * 0.82, POS[k_to] * 0.82, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                        mutation_scale=12, color=color, lw=lw, linestyle=style, zorder=2)
    ax.add_patch(a)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 4.2), gridspec_kw=dict(wspace=0.35))

    # (a) the ring
    ax = axes[0]; ring(ax, "(a) eight orientation classes on a ring")
    ax.text(0, -1.62, "orientation is periodic in 180°, so the ring\nposition is the doubled angle; 0° and 90° are opposite",
            ha="center", va="top", fontsize=8, color="0.3")

    # (b) rotations
    ax = axes[1]; ring(ax, "(b) 8 rotations: shift every class")
    for k in range(K):
        arc_arrow(ax, k, (k + 1) % K, CARD, rad=-0.35)
    ax.text(0, -1.62, "a circulant class-Gram looks the same after any\nshift of the labels around the ring (8, with identity)",
            ha="center", va="top", fontsize=8, color="0.3")

    # (c) reflections
    ax = axes[2]; ring(ax, "(c) 8 reflections: mirror across an axis")
    ax.plot([0, 0], [-1.25, 1.25], color=MARK, lw=1.2, ls="--", zorder=1)
    for a, b in [(1, 7), (2, 6), (3, 5)]:
        arc_arrow(ax, a, b, OBL, rad=0.0); arc_arrow(ax, b, a, OBL, rad=0.0)
    ax.text(0, -1.62, "the mirror through 0° and 90° swaps 45° with 135°;\nseven other axes give the other reflections",
            ha="center", va="top", fontsize=8, color="0.3")

    # (d) what the anisotropy fixes
    ax = axes[3]; ring(ax, "(d) one marked class: no rotations,\none mirror left", highlight=4)
    ax.add_patch(plt.Circle(POS[4], 0.22, fill=False, color=MARK, lw=1.8, zorder=2))
    ax.annotate("most self-similar\nclass", xy=(POS[4][0] + 0.2, POS[4][1] + 0.08), xytext=(0.95, -0.35), fontsize=8, color=MARK, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=MARK, lw=0.8), zorder=5)
    ax.plot([0, 0], [-1.25, 1.25], color=MARK, lw=1.2, ls="--", zorder=1)
    for a_, b_ in [(1, 7), (2, 6), (3, 5)]:
        arc_arrow(ax, a_, b_, OBL, style="--", rad=0.0); arc_arrow(ax, b_, a_, OBL, style="--", rad=0.0)
    ax.text(0, -1.62, "the frame's rotation is fixed; the reflection through\n90° survives and still swaps each class with its mirror",
            ha="center", va="top", fontsize=8, color="0.3")

    fig.suptitle("The symmetries of the orientation ring (D8: 8 rotations × 2 reflections = 16) and what the cortical anisotropy removes",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_d8_symmetries.{ext}"), dpi=200, bbox_inches="tight")
    print("wrote fig_d8_symmetries")


if __name__ == "__main__":
    main()
