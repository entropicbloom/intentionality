"""Model-stitching across checkpoints — an INCONCLUSIVE functional probe.

Top: held-out CE of the stitched seed1->seed2 network at layer L, for three maps
(identity, orthogonal Procrustes, learned linear) plus seed2's solo floor. The
one clean, competence-free fact: an orthogonal rotation suffices to graft one
seed into another (procrustes works; raw identity fails and worsens with
training) — a rotation-equivalence result, but static, with no bearing on the
emergence threshold.

Bottom: two derived curves that we initially read as "interchangeable content
emerges" — but every loss-based readout here is confounded by model competence,
which changes fastest exactly in the emergence window:
  - procrustes reaches the solo floor *before* identity emerges (penalty ~0 at
    step 64-256) -> vacuous, the floor is garbage there (ppl ~5000).
  - penalty (procrustes - solo) grows later -> confounded: the floor becomes
    demanding as the model improves, not necessarily "seeds diverge".
  - content (shuffle - procrustes) grows -> confounded by content-*amount*, not
    graft fidelity.
So this does NOT establish that the threshold marks a functional change. Kept as
an honest negative; the clean results are the geometric ones.

    python -m emergence.plot_stitch
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
SOLO = "#555555"
IDENT = "#c44e52"
SHUF = "#6a4c93"
PROC = "#2f6fed"
CONTENT = "#2e7d4f"
PENALTY = "#e8710a"


def emergence_window():
    d = json.load(open(os.path.join(OUT_DIR, "curve.json")))
    rows = [r for r in d["curve"] if "act" in r]
    steps = [r["step"] for r in rows]
    xs = np.array([s if s > 0 else 0.5 for s in steps])
    act = np.array([np.mean(r["act"]["subset_acc"]) for r in rows])
    k0, k1 = int(np.argmax(act > 0.3)), int(np.argmax(act > 0.85))
    return float(np.sqrt(xs[k0 - 1] * xs[k0])), float(np.sqrt(xs[k1] * xs[min(k1 + 1, len(xs) - 1)]))


def main():
    d = json.load(open(os.path.join(OUT_DIR, "stitch_curve.json")))
    meta, curve = d["meta"], d["curve"]
    steps = [r["step"] for r in curve]
    xs = np.array([s if s > 0 else 0.5 for s in steps])
    solo = np.array([r["solo_B"] for r in curve])
    ident = np.array([r["identity"] for r in curve])
    proc = np.array([r["procrustes"] for r in curve])
    shuf = np.array([r["shuffle"] for r in curve])
    w0, w1 = emergence_window()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 6.4), dpi=200, sharex=True,
                                   gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.13})
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.axvspan(w0, w1, color=MUTED, alpha=0.12, lw=0, zorder=1)

    # top: CE curves
    for y, c, lab, mk in [(ident, IDENT, "identity (raw plug)", "^"),
                          (shuf, SHUF, "shuffle (competence-matched null)", "s"),
                          (proc, PROC, "procrustes (orthogonal align)", "o"),
                          (solo, SOLO, "seed2 solo (floor)", None)]:
        style = "--" if lab.endswith("(floor)") else "-"
        ax1.plot(xs, y, style, color=c, lw=2.0, zorder=3,
                 marker=mk, ms=4.5, markeredgecolor="white", markeredgewidth=0.7, label=lab)
    ax1.set_ylabel("stitched held-out\nCE loss (nats)", fontsize=10, color=INK)
    ax1.legend(loc="center left", frameon=True, fontsize=8.5,
               labelcolor=INK).get_frame().set_edgecolor(MUTED)

    # bottom: confound-free quantities
    ax2.axhline(0, lw=1, color=MUTED, zorder=1)
    ax2.plot(xs, shuf - proc, "-o", color=CONTENT, lw=2.2, ms=4.5, markeredgecolor="white",
             markeredgewidth=0.8, zorder=3, label="content transmitted (shuffle − proc)")
    ax2.plot(xs, proc - solo, "-o", color=PENALTY, lw=2.2, ms=4.5, markeredgecolor="white",
             markeredgewidth=0.8, zorder=3, label="residual penalty (proc − solo)")
    ax2.set_ylabel("nats", fontsize=10, color=INK)
    ax2.set_xlabel("pretraining step", fontsize=10.5, color=INK)
    ax2.set_xscale("log")
    ax2.legend(loc="center left", frameon=True, fontsize=8.5,
               labelcolor=INK).get_frame().set_edgecolor(MUTED)

    for ax in (ax1, ax2):
        ax.grid(True, axis="y", color=MUTED, alpha=0.16, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_color(INK)
    ax1.text(np.sqrt(w0 * w1), ax1.get_ylim()[1], "identity emerges", ha="center",
             va="top", fontsize=8.5, color=MUTED, style="italic")

    fig.text(0.125, 0.975, "Loss-based stitching is competence-confounded — no clean functional threshold",
             fontsize=11.5, color=INK, ha="left", va="bottom")
    fig.text(0.125, 0.93, f"{meta['model']} · seed1→seed2 at layer {meta['layer']} · "
             "held-out CE · clean fact: rotation grafts (identity fails); curves confounded by competence",
             fontsize=8.5, color=MUTED, ha="left", va="bottom")

    out = os.path.join(OUT_DIR, "stitch.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
