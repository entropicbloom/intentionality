"""Grating-substrate diagnostics: orientation signal in relations built from
drifting-grating responses (session A), labels from static gratings (session B).
Writes outputs/dg_diagnostics.json."""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import RidgeClassifierCV

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from microns_ambiguity.relational import cosine_gram
from microns_ambiguity.data import zscore_rows
from allen.data import Allen, ROOT

OUT = os.path.join(ROOT, "allen", "outputs")


def main():
    res = {}
    for movie in ["nm1", "nm3", "both"]:
        ds = Allen(session="DG", movie=movie); ok = ds.ori_ok & (ds.ori_class >= 0); idx = np.flatnonzero(ok); y = ds.ori_class
        G = cosine_gram(zscore_rows(ds.R)); Gi = G[np.ix_(idx, idx)]
        dori = np.abs(((ds.ori[idx][:, None] - ds.ori[idx][None, :]) + 90) % 180 - 90); same = ds.mouse[idx][:, None] == ds.mouse[idx][None, :]; off = ~np.eye(len(idx), dtype=bool)
        sig = dict(same_d0=float(Gi[same & off & (dori == 0)].mean()), same_d90=float(Gi[same & off & (dori >= 60)].mean()),
                   cross_d0=float(Gi[~same & (dori == 0)].mean()), cross_d90=float(Gi[~same & (dori >= 60)].mean()))
        rng = np.random.default_rng(0); perm = rng.permutation(idx); tr, va = perm[: len(idx) // 2], perm[len(idx) // 2:]
        clf = RidgeClassifierCV(alphas=np.logspace(-1, 4, 11)).fit(G[np.ix_(tr, tr)], y[tr]); within = float((clf.predict(G[np.ix_(va, tr)]) == y[va]).mean())
        lomo = []
        for m in ds.mice:
            te = idx[ds.mouse[idx] == m]; trm = idx[ds.mouse[idx] != m]
            if len(te) < 20: continue
            c = RidgeClassifierCV(alphas=np.logspace(-1, 4, 11)).fit(G[np.ix_(trm, trm)], y[trm]); lomo.append(float((c.predict(G[np.ix_(te, trm)]) == y[te]).mean()))
        res[movie] = dict(n=int(len(idx)), mice=len(ds.mice), signal=sig, ridge_within=within, ridge_lomo=float(np.mean(lomo)), ridge_lomo_sd=float(np.std(lomo)),
                          majority=float(np.bincount(y[va]).max() / len(va)), K=ds.K)
        print(f"[{movie:4s}] n={len(idx)} | corr same-mouse dOri=0 {sig['same_d0']:+.3f} dOri>=60 {sig['same_d90']:+.3f} | cross-mouse {sig['cross_d0']:+.3f} / {sig['cross_d90']:+.3f} | ridge within {within:.3f} LOMO {np.mean(lomo):.3f}+-{np.std(lomo):.3f} (majority {res[movie]['majority']:.3f})", flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "dg_diagnostics.json"), "w"))


if __name__ == "__main__":
    main()
