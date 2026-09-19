"""What an unresolved reflection does to the decoder's output, model against measurement.

The decoder outputs a unit vector at the doubled angle, (cos 2θ, sin 2θ). If it cannot tell θ from its
mirror image 180° − θ, the best output under squared error is the average of the two candidate vectors,
which lies on the horizontal axis (the 0°/90° axis of the doubled-angle plane) and shrinks toward zero as
θ approaches 45°. The decoded angle then falls on the nearer cardinal and the error equals the distance
to that cardinal: zero at 0° and 90°, 45° at the obliques.
Left: the doubled-angle plane, the two candidates per class and their average (model), with the measured
mean output vector per class from the saved twin predictions. Right: error against true orientation, the
model's triangle and the measured per-bin errors on both substrates.
Run: .venv/bin/python -m microns_ambiguity.reflection_figure -> outputs/paper/fig_reflection_readout.{png,pdf}
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs", "paper"); PREDS = os.path.join(OUT, "..", "preds")
K = 8; CENT = np.arange(K) * 180 / K
TWIN = "#e0a030"; IV = "#3a6ea5"; MODEL = "#c0392b"; GREY = "#8a8a8a"


def per_bin(tag):
    z = np.load(os.path.join(PREDS, tag + ".npz")); P, y = z["P"], z["y"]
    ang = (0.5 * np.degrees(np.arctan2(P[:, 1], P[:, 0]))) % 180
    err = np.abs(((ang - y + 90) % 180) - 90)
    b = np.round(y / (180 / K)).astype(int) % K
    return np.array([err[b == k].mean() for k in range(K)]), np.array([P[b == k].mean(0) for k in range(K)])


def main():
    e_tw, v_tw = per_bin("c_is_17M_sp2"); e_iv, v_iv = per_bin("c_iv_17M_sp2")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw=dict(width_ratios=[1, 1.25]))

    # (a) doubled-angle plane
    ax.set_aspect("equal"); ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=GREY, lw=1)); ax.axhline(0, color=GREY, lw=0.8, ls=":"); ax.axvline(0, color=GREY, lw=0.8, ls=":")
    for k in range(K):
        th = np.deg2rad(2 * CENT[k]); u = np.array([np.cos(th), np.sin(th)]); m = np.array([np.cos(th), -np.sin(th)])   # candidate and its mirror
        avg = (u + m) / 2
        ax.plot(*u, "o", color=GREY, ms=5, zorder=3); ax.text(1.16 * u[0], 1.16 * u[1], f"{CENT[k]:g}°", ha="center", va="center", fontsize=8.5)
        if k not in (0, 4):
            ax.plot([u[0], m[0]], [u[1], m[1]], color=MODEL, lw=0.8, ls="--", zorder=1)
        ax.annotate("", xy=avg, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=MODEL, lw=1.4, mutation_scale=10), zorder=4)
        ax.annotate("", xy=v_tw[k], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=TWIN, lw=1.6, mutation_scale=10), zorder=5)
    ax.plot([], [], color=MODEL, lw=1.4, label="model: average of θ and its mirror 180° − θ")
    ax.plot([], [], color=TWIN, lw=1.6, label="measured: mean decoder output per class (twin)")
    ax.plot([], [], color=MODEL, lw=0.8, ls="--", label="the two candidates")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.set_xlabel("cos 2θ   (0° ↔ 90° axis)", fontsize=9, labelpad=2); ax.set_ylabel("sin 2θ   (45° ↔ 135° axis)", fontsize=9)
    ax.set_title("(a) the output plane: mirror pairs average onto the cardinal axis", fontsize=10)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])

    # (b) error vs orientation
    th = np.linspace(0, 180, 721); model = np.minimum(th % 90, 90 - th % 90)
    bx.plot(th, model, color=MODEL, lw=1.6, label="model: distance to the nearer cardinal")
    bx.plot(CENT, e_tw, "o-", color=TWIN, lw=1.2, ms=6, label="measured, twin (per 22.5° bin)")
    bx.plot(CENT, e_iv, "s-", color=IV, lw=1.2, ms=5, label="measured, in vivo")
    bx.axhline(45, color=GREY, lw=1, ls="--"); bx.text(178, 46, "chance 45°", ha="right", va="bottom", fontsize=8, color="0.3")
    bx.set_xticks(CENT); bx.set_xticklabels([f"{c:g}" for c in CENT]); bx.set_xlim(-8, 168); bx.set_ylim(0, 50)
    bx.set_xlabel("true preferred orientation (°)"); bx.set_ylabel("mean angular error (°)")
    bx.set_title("(b) error grows with the distance to the surviving mirror axis", fontsize=10)
    bx.legend(fontsize=8, loc="upper left", frameon=False)
    for s in ("top", "right"): bx.spines[s].set_visible(False)

    fig.suptitle("One unresolved reflection explains the cardinal readout", fontsize=11.5, y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_reflection_readout.{ext}"), dpi=200, bbox_inches="tight")
    print("wrote fig_reflection_readout; twin per-bin", np.round(e_tw, 1), "in vivo", np.round(e_iv, 1))


if __name__ == "__main__":
    main()
