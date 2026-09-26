"""Do the 25 LLMs still agree once sentence meaning is regressed out?

For each variant, each LLM's 45 off-diagonal cosines are regressed on those of a
"meaning" model (MiniLM, GPT-2 medium, or both) and the residuals kept. We then
correlate each LLM's residuals with the mean residuals of the other 24 LLMs
(leave one model out) and of the LLMs from other families (leave one family out).

GPT-2 medium uses the final token at layer 20; its matrix is computed here with
the recipe in baseline.py.
"""

import os

import numpy as np

from baseline import OUT, SETS, load_sentences, rep_causal_lm, similarity
from identify import IU, VARIANTS, family, load_variant

GPT2M_LAYER = 20


def meaning_matrices():
    mats = {"minilm": {v: np.loadtxt(os.path.join(OUT, f"baseline_minilm_{v}.csv"), delimiter=",")
                       for v in VARIANTS}}
    cache = os.path.join(OUT, f"baseline_acts_gpt2medium_L{GPT2M_LAYER}.npy")
    sents = load_sentences()
    if os.path.exists(cache):
        X = np.load(cache).astype(np.float64)
    else:
        X = rep_causal_lm("gpt2-medium", GPT2M_LAYER)(sum((sents[s][0] for s in SETS), []))
        np.save(cache, X.astype(np.float32))
    bounds = np.cumsum([0] + [len(sents[s][0]) for s in SETS])
    acts = {s: X[bounds[i]:bounds[i + 1]] for i, s in enumerate(SETS)}
    cats = {s: sents[s][1] for s in SETS}
    mats["gpt2medium"] = {v: similarity(acts, cats, v) for v in VARIANTS}
    return mats


def residuals(Y, Z):
    """Y: (models, 45); Z: (45, k) predictors. Least-squares residuals per model."""
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(Z1, Y.T, rcond=None)
    return Y - (Z1 @ beta).T, 1 - ((Y - (Z1 @ beta).T).var(1) / Y.var(1))


def agreement(E, names):
    loo, lofo = [], []
    for m in range(len(E)):
        loo.append(np.corrcoef(E[m], np.delete(E, m, 0).mean(0))[0, 1])
        other = [j for j in range(len(E)) if family(names[j]) != family(names[m])]
        lofo.append(np.corrcoef(E[m], E[other].mean(0))[0, 1])
    return np.array(loo), np.array(lofo)


def main():
    meaning = meaning_matrices()
    lines = ["Per variant: R^2 of the meaning model(s) for each LLM (median), and agreement",
             "(Pearson r) of each LLM with the mean of the others, median [min, max],",
             "leaving out one model (LOO) or its whole family (LOFO).", ""]
    for v in VARIANTS:
        names, mats = load_variant(VARIANTS[v])
        Y = mats[:, IU[0], IU[1]]
        loo, lofo = agreement(Y, names)
        lines.append(f"=== {v} ===")
        lines.append(f"{'no removal':24s} R2   -    LOO {np.median(loo):.2f} [{loo.min():.2f}, {loo.max():.2f}]"
                     f"   LOFO {np.median(lofo):.2f} [{lofo.min():.2f}, {lofo.max():.2f}]")
        preds = {"minilm": ["minilm"], "gpt2medium": ["gpt2medium"],
                 "minilm+gpt2medium": ["minilm", "gpt2medium"]}
        for label, keys in preds.items():
            Z = np.column_stack([meaning[k][v][IU] for k in keys])
            E, r2 = residuals(Y, Z)
            loo, lofo = agreement(E, names)
            lines.append(f"{'minus ' + label:24s} R2 {np.median(r2):.2f}  LOO {np.median(loo):.2f} "
                         f"[{loo.min():.2f}, {loo.max():.2f}]   LOFO {np.median(lofo):.2f} "
                         f"[{lofo.min():.2f}, {lofo.max():.2f}]")
        # which entries carry the shared residual (both meaning models removed)
        Z = np.column_stack([meaning[k][v][IU] for k in ["minilm", "gpt2medium"]])
        E, _ = residuals(Y, Z)
        mean_e, se = E.mean(0), E.std(0, ddof=1) / np.sqrt(len(E))
        from identify import CONCEPTS
        pairs = [f"{CONCEPTS[i]}-{CONCEPTS[j]}" for i, j in zip(*IU)]
        top = np.argsort(-np.abs(mean_e / se))[:6]
        lines.append("  largest shared residuals (LLMs vs meaning; mean, t): " +
                     ", ".join(f"{pairs[k]} {mean_e[k]:+.2f} ({mean_e[k] / se[k]:+.1f})" for k in top))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "residual.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
