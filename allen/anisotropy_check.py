"""Per-class anisotropy factor on Allen grating relations (2026-09-19, not in the arXiv v1).

Same definition as the MICrONS Table 2: mean correlation between two cells of the same
orientation class, divided by the mean of that quantity over the other classes. Relations are
the 40 drifting-grating condition means (session DG), labels the static-grating preferred
orientation (6 classes, 30 degrees). Pooled over the 33 mice and restricted to within-mouse pairs,
with a cell bootstrap for the peak class.
Result (2026-09-19): peak 0 deg, factor 1.29 pooled / 1.21 within mouse, 0 deg the peak in 500/500
bootstraps; label mode ties 0 and 90 deg at 19 %. MICrONS twin for comparison: 1.28 at 90 deg.
Run: .venv/bin/python -m allen.anisotropy_check
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from allen.data import Allen
from microns_ambiguity.data import zscore_rows
from microns_ambiguity.relational import cosine_gram


def factors(G, lab, idx, K):
    C = np.full(K, np.nan); cnt = np.zeros(K)
    for c in range(K):
        m = idx[lab[idx] == c]
        if len(m) < 2:
            continue
        sub = G[np.ix_(m, m)]; C[c] = sub[~np.eye(len(m), dtype=bool)].mean(); cnt[c] = len(m)
    fac = np.array([C[c] / np.nanmean(np.delete(C, c)) for c in range(K)])
    return C, fac, cnt


def main(n_boot=500):
    ds = Allen(movie="nm1", ori_source="sg", session="DG")
    K, step = ds.K, ds.step
    G = cosine_gram(zscore_rows(ds.R)); lab = ds.ori_class; ok = ds.ori_ok & (lab >= 0); idx = np.flatnonzero(ok)
    C, fac, cnt = factors(G, lab, idx, K)
    print(f"pooled: label fraction {np.round(cnt / cnt.sum(), 2)}, same-class corr {np.round(C, 3)}, factor {np.round(fac, 2)}, "
          f"peak {int(np.nanargmax(fac)) * step:.0f} deg, label mode {int(np.argmax(cnt)) * step:.0f} deg")
    Cm = np.zeros(K); nm = np.zeros(K)
    for mouse in ds.mice:
        for c in range(K):
            m = np.flatnonzero(ok & (ds.mouse == mouse) & (lab == c))
            if len(m) < 2:
                continue
            sub = G[np.ix_(m, m)]; Cm[c] += sub[~np.eye(len(m), dtype=bool)].sum(); nm[c] += len(m) * (len(m) - 1)
    Cm /= nm; facm = np.array([Cm[c] / np.mean(np.delete(Cm, c)) for c in range(K)])
    print(f"within-mouse pairs: same-class corr {np.round(Cm, 3)}, factor {np.round(facm, 2)}, peak {int(np.argmax(facm)) * step:.0f} deg")
    rng = np.random.default_rng(0); peaks = []
    for _ in range(n_boot):
        b = np.unique(rng.choice(idx, len(idx), replace=True)); peaks.append(int(np.nanargmax(factors(G, lab, b, K)[1])))
    peaks = np.array(peaks)
    print("bootstrap P(peak class):", {f"{c * step:.0f}": round(float((peaks == c).mean()), 2) for c in range(K)})


if __name__ == "__main__":
    main()
