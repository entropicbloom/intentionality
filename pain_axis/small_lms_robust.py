"""Robustness of small_lms.py to the meaning predictors: embedders only (no GPT-2
medium) with squared terms, and all four linear only. Prints r_rem per model at the
layer chosen by r_gram, raw and whitened."""
import numpy as np
import torch

from identify import family
from small_lms import (IU, LMS, MEANING_LAYER, MEANING_LM, OUT, SETS, VARIANTS, hidden_states,
                       load_sentences, load_variant, meaning_embeddings, residuals, similarity, tag)

sents = load_sentences()
texts = sum((sents[s][0] for s in SETS), [])
bounds = np.cumsum([0] + [len(sents[s][0]) for s in SETS])
cats = {s: sents[s][1] for s in SETS}


def gram(X, v):
    return similarity({s: np.asarray(X[bounds[i]:bounds[i + 1]], dtype=np.float64)
                       for i, s in enumerate(SETS)}, cats, v)[IU]


embs = meaning_embeddings(texts)
gm = hidden_states(MEANING_LM, texts, torch.float64)[:, MEANING_LAYER]
hid = {n: hidden_states(n, texts) for n in LMS}
lines = []
for label, use_gm, sq in [("embedders only, +squares", False, True), ("all 4, linear only", True, False)]:
    lines.append(f"--- meaning predictors: {label}")
    for v in ["raw", "whitened"]:
        names, mats = load_variant(VARIANTS[v])
        Y = mats[:, IU[0], IU[1]]
        base = np.column_stack([gram(X, v) for X in embs.values()] + ([gram(gm, v)] if use_gm else []))
        if sq:
            base = np.column_stack([base, base ** 2])
        E, r2 = residuals(Y, base)
        fams = np.array([family(n) for n in names])
        lofo = np.median([np.corrcoef(E[m], E[fams != fams[m]].mean(0))[0, 1] for m in range(len(E))])
        lines.append(f"  {v}: meaning R2 {np.median(r2):.2f}, LLM reference (LOFO) {lofo:.2f}")
        for n, (size, fam) in sorted(LMS.items(), key=lambda kv: kv[1][0]):
            ref = E[fams != fam].mean(0)
            rows = [(np.corrcoef(g, Y.mean(0))[0, 1], np.corrcoef(residuals(g[None], base)[0][0], ref)[0, 1])
                    for g in (gram(hid[n][:, L], v) for L in range(1, hid[n].shape[1]))]
            lines.append(f"    {tag(n):26s} {size:5.2f}  r_rem {max(rows)[1]:+.2f}")
text = "\n".join(lines)
print(text)
with open(f"{OUT}/small_lms_robust.txt", "w") as fh:
    fh.write(text + "\n")
