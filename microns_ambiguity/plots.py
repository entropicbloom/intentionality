"""Figures from outputs/*.json -> outputs/*.png"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import OUT

SUB_LABEL = {"struct_out": "synaptic", "struct_out_adp": "proximity",
             "struct_out_rewired": "rewired", "struct_in": "synaptic",
             "struct_in_adp": "proximity", "struct_in_rewired": "rewired",
             "func_iv": "func. in vivo", "func_is": "func. twin", "soma": "soma dist."}
CON_LABEL = {"ori": "orientation (K=8)", "rf": "RF position (3x3)", "rf_resid": "RF minus retinotopic trend (3x3)", "soma_xz": "cortical position (3x3)",
             "depth": "depth (K=4)", "layer": "layer (K=3)", "area": "area (K=3)"}
COL = {"struct_in": "#c0392b", "struct_in_adp": "#e59866", "struct_in_rewired": "#f5cba7",
       "func_iv": "#1f618d", "func_is": "#5dade2", "soma": "#7f8c8d",
       "struct_out": "#c0392b", "struct_out_adp": "#e59866", "struct_out_rewired": "#f5cba7"}


def load(name):
    p = OUT / f"{name}.json"
    return json.load(open(p)) if p.exists() else {}


def fig_geometric(geo, subs, contents, fname, title):
    fig, axes = plt.subplots(1, len(contents), figsize=(3.1 * len(contents), 3.8), sharey=True)
    for ax, con in zip(np.atleast_1d(axes), contents):
        xs, hs, es, cs, nulls = [], [], [], [], []
        for i, s in enumerate(subs):
            r = geo.get(f"{s}|{con}")
            if r is None: continue
            xs.append(i); hs.append(r["acc"]); es.append(r["acc_sd"] / np.sqrt(r["n_splits"])); cs.append(COL[s])
            nulls.append((i, r["null_indep"]["acc"], r["null_fixed"]["acc"]))
        ax.bar(xs, hs, yerr=es, color=cs, width=0.7)
        for i, ni, nf in nulls:
            ax.plot([i - 0.35, i + 0.35], [nf, nf], color="k", lw=1.2)
        K = geo[f"{subs[0]}|{con}"]["K"] if f"{subs[0]}|{con}" in geo else None
        if K: ax.axhline(1 / K, color="grey", ls=":", lw=1)
        ax.set_xticks(range(len(subs))); ax.set_xticklabels([SUB_LABEL[s] for s in subs], fontsize=7, rotation=35, ha="right")
        ax.set_title(CON_LABEL[con], fontsize=10); ax.set_ylim(0, 1.05)
    np.atleast_1d(axes)[0].set_ylabel("class-identity accuracy\n(geometric matching)")
    fig.suptitle(title + "  (black tick: fixed-group null; dotted: 1/K)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150); plt.close(fig)


def fig_perm_hist(geo, keys, fname):
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3))
    for ax, k in zip(np.atleast_1d(axes), keys):
        r = geo.get(k)
        if r is None: continue
        e = r["example"]; edges = np.array(e["edges"]); h = np.array(e["hist"])
        ax.bar(edges[:-1], h, width=np.diff(edges), color="#aab7b8", align="edge")
        ax.axvline(e["d_id"], color="red", lw=2, label=f"true labels (rank {e['rank_id']})")
        ax.set_title(k.replace("|", " / "), fontsize=9); ax.set_xlabel("Frobenius distance to reference"); ax.legend(fontsize=7)
        ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150); plt.close(fig)


def fig_class_grams(geo, keys, fname):
    fig, axes = plt.subplots(2, len(keys), figsize=(3.2 * len(keys), 6))
    for j, k in enumerate(keys):
        r = geo.get(k)
        if r is None: continue
        M = np.array(r["class_gram"]); np.fill_diagonal(M, np.nan)
        ax = axes[0, j]; im = ax.imshow(M, cmap="viridis"); ax.set_title(k.replace("|", " / "), fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046); ax.set_xlabel("class"); ax.set_ylabel("class")
        ax = axes[1, j]; im = ax.imshow(np.array(r["marginal"]), vmin=0, vmax=1, cmap="magma")
        ax.set_title(f"posterior p(test class -> ref class)\nH={r['H_class_bits']:.2f}/{r['H_class_max']:.2f} bits", fontsize=8)
        ax.set_xlabel("reference class"); ax.set_ylabel("test class"); plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150); plt.close(fig)


def fig_decoder(dec, fname):
    subs = ["struct_in", "struct_in_adp", "struct_in_rewired", "func_iv", "func_is", "soma"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, con, metric, chance in zip(axes, ["ori", "rf", "layer", "area"], ["acc", "r2", "acc", "acc"], [1 / 8, 0, 1 / 3, 1 / 3]):
        for i, s in enumerate(subs):
            vals = [v[metric] for k, v in dec.items() if k.startswith(f"{s}|{con}|full|")]
            if not vals: continue
            ax.bar(i, np.mean(vals), yerr=np.std(vals) if len(vals) > 1 else 0, color=COL[s], width=0.7)
            to = [v[metric] for k, v in dec.items() if k.startswith(f"{s}|{con}|target_only|")]
            if to: ax.plot([i - 0.3, i + 0.3], [np.mean(to)] * 2, color="k", lw=1.5)
            sh = [v[metric] for k, v in dec.items() if k.startswith(f"{s}|{con}|shuffled|")]
            if sh: ax.plot([i - 0.3, i + 0.3], [np.mean(sh)] * 2, color="k", lw=1.5, ls="--")
        ax.axhline(chance, color="grey", ls=":", lw=1)
        ax.set_xticks(range(len(subs))); ax.set_xticklabels([SUB_LABEL[s] for s in subs], fontsize=7, rotation=35, ha="right")
        ax.set_title(f"{CON_LABEL[con]}: {'accuracy' if metric == 'acc' else 'R²'}", fontsize=10)
        ax.set_ylim(min(0, ax.get_ylim()[0]), 1.05)
    fig.suptitle("Learned relational decoder, hidden population labels (solid tick: target-only ablation; dashed: shuffled labels)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150); plt.close(fig)


def fig_spectral(spec, fname):
    subs = ["struct_in", "struct_in_adp", "struct_in_rewired", "func_iv", "func_is", "soma"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, con in zip(axes, ["rf", "soma_xz", "ori"]):
        for i, s in enumerate(subs):
            r = spec.get(f"{s}|{con}")
            if r is None: continue
            ax.bar(i - 0.2, r["procrustes_top2"], width=0.38, color=COL[s], alpha=0.55)
            ax.bar(i + 0.2, r["linear_m50"], width=0.38, color=COL[s])
            ax.plot([i - 0.4, i + 0.4], [r["null_linear_m50"]] * 2, color="k", lw=1.2)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(range(len(subs))); ax.set_xticklabels([SUB_LABEL[s] for s in subs], fontsize=7, rotation=35, ha="right")
        ax.set_title(CON_LABEL.get(con, con).split(" (")[0], fontsize=10); ax.set_ylim(-0.1, 1)
    axes[0].set_ylabel("held-out R²")
    fig.suptitle("Reference-free recovery: kernel-PCA of the population Gram; light = top-2 axes up to rotation/scale, dark = linear readout of 50 axes, tick = shuffled null", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150); plt.close(fig)


def fig_transfer(tr, fname):
    subs = ["func_iv", "func_is", "struct_in", "struct_in_adp", "soma"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, con in zip(axes, ["rf", "ori"]):
        M = np.full((len(subs), len(subs)), np.nan)
        for i, X in enumerate(subs):
            for j, Y in enumerate(subs):
                r = tr.get(f"{con}|{X}->{Y}")
                if r: M[i, j] = r["acc"]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        for i in range(len(subs)):
            for j in range(len(subs)):
                if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="w" if M[i, j] < 0.6 else "k", fontsize=8)
        ax.set_xticks(range(len(subs))); ax.set_xticklabels([SUB_LABEL[s] for s in subs], rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(subs))); ax.set_yticklabels([SUB_LABEL[s] for s in subs], fontsize=7)
        ax.set_xlabel("test substrate"); ax.set_ylabel("reference substrate"); ax.set_title(CON_LABEL[con], fontsize=10)
    plt.colorbar(im, ax=axes, fraction=0.03, label="class accuracy")
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_symmetry(sym, fname):
    subs = [s for s in ["func_iv", "func_is", "struct_in", "struct_in_adp", "soma"] if s in sym]
    fig, axes = plt.subplots(1, len(subs) + 1, figsize=(3.2 * (len(subs) + 1), 3.4))
    for ax, s in zip(axes, subs):
        M = np.array(sym[s]["class_gram"]); np.fill_diagonal(M, np.nan)
        im = ax.imshow(M, cmap="viridis"); ax.set_title(f"{SUB_LABEL[s]}\ncirculant var. {sym[s]['frac_var_circulant']:.2f}", fontsize=8)
        ax.set_xticks(range(8)); ax.set_xticklabels([f"{22.5*i:g}" for i in range(8)], fontsize=6)
        ax.set_yticks(range(8)); ax.set_yticklabels([f"{22.5*i:g}" for i in range(8)], fontsize=6)
    ax = axes[-1]
    w = 0.25
    for k, (name, c) in enumerate([("raw", "#1f618d"), ("circulant", "#e59866")]):
        ax.bar(np.arange(len(subs)) + (k - 0.5) * w, [sym[s][f"acc_{name}"] for s in subs], width=w, color=c, label=f"{name} (abs.)")
        ax.scatter(np.arange(len(subs)) + (k - 0.5) * w, [sym[s][f"acc_mod_{name}"] for s in subs], color="k", s=10, zorder=3)
    ax.axhline(1 / 8, color="grey", ls=":"); ax.set_xticks(range(len(subs))); ax.set_xticklabels([SUB_LABEL[s] for s in subs], fontsize=6, rotation=35, ha="right")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=6); ax.set_title("matching accuracy after projecting\nclass-Gram (dots: modulo D8)", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150); plt.close(fig)


def main():
    geo, dec, spec, tr, sym = load("geometric"), load("decoder"), load("spectral"), load("transfer"), load("symmetry")
    if geo:
        fig_geometric(geo, ["struct_in", "struct_in_adp", "struct_in_rewired", "func_iv", "func_is", "soma"],
                      ["ori", "rf", "rf_resid", "soma_xz", "layer", "area"], "geometric_main.png",
                      "Class identity from relational structure alone (12,894 coregistered neurons; postsynaptic view)")
        fig_geometric(geo, ["struct_out", "struct_out_adp", "struct_out_rewired"], ["ori", "rf", "soma_xz", "layer", "area"],
                      "geometric_axons.png", "148 proofread axons: relations = shared synaptic targets (K reduced to 4)")
        fig_perm_hist(geo, ["func_iv|ori", "struct_in|rf", "struct_in_adp|rf", "struct_in|ori"], "perm_distances.png")
        fig_class_grams(geo, ["func_iv|ori", "struct_in|rf", "struct_in_adp|rf"], "class_grams.png")
    if dec: fig_decoder(dec, "decoder.png")
    if spec: fig_spectral(spec, "spectral.png")
    if tr: fig_transfer(tr, "transfer.png")
    if sym: fig_symmetry(sym, "symmetry.png")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
