"""Report figure: when does meaning become recoverable from LLM representations?

Cross-seed label-free identifiability (robust subset-Gram metric) of contextual
activations and the unembedding interface, as a function of pretraining step.
Reads left-to-right as "at random init nothing is recoverable; identity
crystallizes early in training, activations before the output interface".

    python -m emergence.plot
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

INK = "#1a1a2e"       # primary text
MUTED = "#8a8a99"     # axes / grid / reference
ACT = "#2f6fed"       # activations (categorical slot 1)
UNEMBED = "#e8710a"   # unembedding (categorical slot 2) — CVD-safe validated pair


def main():
    data = json.load(open(os.path.join(OUT_DIR, "curve.json")))
    meta, curve = data["meta"], data["curve"]
    chance = 1 / meta["subset_size"]

    rows = [r for r in curve if "act" in r and "unembed" in r]
    steps = [r["step"] for r in rows]
    acc = [r["act"]["subset_acc"] for r in rows]
    une = [r["unembed"]["subset_acc"] for r in rows]
    xs = [s if s > 0 else 0.5 for s in steps]  # step 0 -> sit on the log axis

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axhline(chance, ls=(0, (4, 4)), lw=1.4, color=MUTED, zorder=1)
    ax.text(xs[0], chance - 0.035, "chance", ha="left", va="top", fontsize=9, color=MUTED)

    for ys, color in [(une, UNEMBED), (acc, ACT)]:
        ax.plot(xs, ys, "-", lw=2.2, color=color, zorder=3)
        ax.plot(xs, ys, "o", ms=6, color=color, markeredgecolor="white",
                markeredgewidth=1.2, zorder=4)

    handles = [plt.Line2D([], [], color=ACT, lw=2.2, marker="o", ms=6,
                          markeredgecolor="white", markeredgewidth=1.2, label="activations"),
               plt.Line2D([], [], color=UNEMBED, lw=2.2, marker="o", ms=6,
                          markeredgecolor="white", markeredgewidth=1.2, label="unembedding")]
    leg = ax.legend(handles=handles, loc="center right", frameon=True, fontsize=10.5,
                    borderpad=0.8, handlelength=1.6, labelcolor=INK)
    leg.get_frame().set_edgecolor(MUTED)
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.95)

    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(right=xs[-1] * 1.6)
    ax.set_xlabel("pretraining step", fontsize=10.5, color=INK)
    ax.set_ylabel("cross-seed token identifiability", fontsize=10.5, color=INK)

    ax.set_title("Identifiable geometry emerges early — activations before the output interface",
                 fontsize=11.5, color=INK, pad=24, loc="left")
    subtitle = (f"{meta['model']} · activations at layer {meta['layer_idx']}/{meta['n_layers']} "
                f"(≈⅔ depth) · {meta['n_tokens']} concept tokens · cross-seed")
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=MUTED, ha="left", va="bottom")

    ax.grid(True, which="major", axis="y", color=MUTED, alpha=0.18, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(INK)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "emergence.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
