"""Robustness of residual.py: add all-mpnet-base-v2 and bge-large-en-v1.5 as meaning
models, squared terms, and a null with the same number of random predictors."""
import sys, os, numpy as np
sys.path.insert(0, '.')
from baseline import *
from residual import meaning_matrices, residuals, agreement
from identify import IU, VARIANTS, load_variant
from sentence_transformers import SentenceTransformer
sents = load_sentences(); texts = sum((sents[s][0] for s in SETS), [])
bounds = np.cumsum([0] + [len(sents[s][0]) for s in SETS]); cats = {s: sents[s][1] for s in SETS}
M = meaning_matrices()
for name in ["sentence-transformers/all-mpnet-base-v2", "BAAI/bge-large-en-v1.5"]:
    X = SentenceTransformer(name).encode(texts, batch_size=64).astype(np.float64)
    acts = {s: X[bounds[i]:bounds[i+1]] for i, s in enumerate(SETS)}
    M[name.split('/')[-1]] = {v: similarity(acts, cats, v) for v in VARIANTS}
for v in VARIANTS:
    names, mats = load_variant(VARIANTS[v]); Y = mats[:, IU[0], IU[1]]
    llm_mean = Y.mean(0)
    print(f"=== {v} ===")
    for k in M:
        print(f"  {k:24s} r with LLM mean {np.corrcoef(M[k][v][IU], llm_mean)[0,1]:.2f}")
    keys = list(M)
    Zlin = np.column_stack([M[k][v][IU] for k in keys])
    Zq = np.column_stack([Zlin, Zlin**2])
    for label, Z in [("all 4 linear", Zlin), ("all 4 + squares", Zq)]:
        E, r2 = residuals(Y, Z); loo, lofo = agreement(E, names)
        print(f"  minus {label:18s} ({Z.shape[1]} predictors) R2 {np.median(r2):.2f}  LOO {np.median(loo):.2f}  LOFO {np.median(lofo):.2f}")
    # null: same number of random predictors
    rng = np.random.default_rng(0); nl = []
    for _ in range(200):
        E, _ = residuals(Y, rng.standard_normal((45, Zq.shape[1]))); nl.append(np.median(agreement(E, names)[0]))
    print(f"  minus {Zq.shape[1]} random predictors: LOO median {np.mean(nl):.2f}")
