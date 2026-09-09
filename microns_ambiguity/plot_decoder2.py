"""Scaling figures for the per-neuron decoder (outputs/decoder2.json)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .config import OUT

CEIL = {("func_iv", "ori"): 0.41, ("func_is", "ori"): 0.49, ("func_iv", "rf"): 0.32, ("func_is", "rf"): 0.67}
BASE = {"ori": 0.255, "rf": 0.0, "rf_dist": 0.0}
COL = {"func_iv": "#1f618d", "func_is": "#5dade2"}


def main():
    d = json.load(open(OUT / "decoder2.json"))
    runs = [v | {"tag": k} for k, v in d.items() if v["epochs"] == 10]     # matched budget only
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, con, metric in zip(axes, ["ori", "rf", "rf_dist"], ["acc", "r2", "r2"]):
        for sub in ["func_iv", "func_is"]:
            for params, mk, lab in [(0.4e6, "o", "0.3M params"), (2.3e6, "s", "2.2M"), (7.3e6, "^", "7.2M")]:
                pts = sorted([(r["n"], r[metric]) for r in runs if r["con"] == con and r["sub"] == sub and abs(r["params"] - params) < 0.15e6])
                if pts:
                    ax.plot(*zip(*pts), marker=mk, color=COL[sub], label=f"{'in vivo' if sub == 'func_iv' else 'twin'}, {lab}")
            if (sub, con) in CEIL:
                ax.axhline(CEIL[(sub, con)], color=COL[sub], ls=":", lw=1)
        ax.axhline(BASE[con], color="grey", lw=0.8)
        ax.set_xscale("log", base=2); ax.set_xlabel("neurons per population")
        ax.set_ylabel("accuracy" if metric == "acc" else "R²"); ax.set_title({"ori": "orientation (8 classes)", "rf": "RF position (x, y)", "rf_dist": "RF distance from centre"}[con])
        ax.legend(fontsize=7)
    fig.suptitle("Label-free per-neuron decoding from the population correlation matrix (dotted: labelled-reference ceiling; grey: majority class / mean)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "decoder2_scaling.png", dpi=150); plt.close(fig)
    # overfitting example
    if "big_twin_n512_long" in d:
        h = d["big_twin_n512_long"]["history"]
        fig, ax = plt.subplots(figsize=(5, 3.2)); ax2 = ax.twinx()
        ax.plot([e["acc"] for e in h], color="#1f618d", marker="o", ms=3, label="validation accuracy")
        ax2.plot([e["loss"] for e in h], color="#c0392b", label="training loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy", color="#1f618d"); ax2.set_ylabel("training loss", color="#c0392b")
        ax.set_title("2.2M model, 512 neurons, 20 epochs x 4000 populations: memorises training neurons", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "decoder2_overfit.png", dpi=150); plt.close(fig)
    print("written")


if __name__ == "__main__":
    main()
