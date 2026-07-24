"""Identifiability vs weight-distance travelled — the schedule-agnostic view.

Raw step is confounded by the LR warmup. Re-plotting against ||theta - theta(0)||
(the actual distance the weights have moved) removes that: identity still
crystallizes early -- within the first ~12% of the total weight-space journey --
and then persists (slightly decaying) while the model travels ~8x further. The
ordering (activations before unembedding) is unchanged, since any monotone
reparametrization of x preserves which curve is higher.

    python -m emergence.plot_distance
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
WARMUP_STEP = 1430


def main():
    ic = json.load(open(os.path.join(OUT_DIR, "curve.json")))
    dc = json.load(open(os.path.join(OUT_DIR, "distance_curve.json")))
    chance = 1 / ic["meta"]["subset_size"]
    dist_by_step = {r["step"]: r["dist"] for r in dc["curve"]}
    total = max(dist_by_step.values())

    rows = [r for r in ic["curve"] if "act" in r and "unembed" in r and r["step"] in dist_by_step]
    steps = [r["step"] for r in rows]
    x = np.array([dist_by_step[s] for s in steps])
    act = np.array([np.mean(r["act"]["subset_acc"]) for r in rows])
    une = np.array([np.mean(r["unembed"]["subset_acc"]) for r in rows])

    # warmup end in distance units (interpolate between bracketing checkpoints)
    ds = sorted(dist_by_step.items())
    xs_s = [s for s, _ in ds]
    xs_d = [d for _, d in ds]
    warm_dist = float(np.interp(WARMUP_STEP, xs_s, xs_d))

    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axvline(warm_dist, ls=(0, (3, 3)), lw=1.2, color=MUTED, zorder=1)
    ax.text(warm_dist, 0.02, " warmup ends", ha="left", va="bottom", fontsize=8,
            color=MUTED, style="italic")
    ax.axhline(chance, ls=(0, (4, 4)), lw=1.2, color=MUTED, zorder=1)
    ax.text(x[0], chance + 0.02, "chance", ha="left", va="bottom", fontsize=8, color=MUTED)

    ax.plot(x, une, "-s", color=UNEMBED, lw=2.0, ms=5, markeredgecolor="white",
            markeredgewidth=0.8, zorder=3, label="unembedding")
    ax.plot(x, act, "-o", color=ACT, lw=2.2, ms=5.5, markeredgecolor="white",
            markeredgewidth=0.9, zorder=4, label="activations")

    # annotate a few checkpoints with their step number
    for s in (256, 1000, 8000, 143000):
        if s in dist_by_step:
            xi = dist_by_step[s]
            yi = float(np.mean([r for r in rows if r["step"] == s][0]["act"]["subset_acc"]))
            ax.annotate(f"step {s:,}", (xi, yi), textcoords="offset points",
                        xytext=(4, -12), fontsize=7.5, color=INK)
            ax.plot([xi], [yi], "o", ms=3, color=INK, zorder=5)

    ax.set_ylim(0, 1.02)
    ax.set_xlabel("weight-space distance travelled  ‖θ − θ₀‖", fontsize=10.5, color=INK)
    ax.set_ylabel("cross-seed token identifiability", fontsize=10.5, color=INK)
    ax.set_title("Schedule-agnostic axis: identity crystallizes in the first ~12% of the journey",
                 fontsize=11, color=INK, pad=22, loc="left")
    ax.text(0.0, 1.02, f"{ic['meta']['model']} · cross-seed · x = distance moved from init "
            f"(total {total:.0f}); ordering is unchanged from the step axis",
            transform=ax.transAxes, fontsize=9, color=MUTED, ha="left", va="bottom")
    ax.legend(loc="center right", frameon=True, fontsize=9,
              labelcolor=INK).get_frame().set_edgecolor(MUTED)

    ax.grid(True, axis="y", color=MUTED, alpha=0.16, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(INK)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "distance.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
