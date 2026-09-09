"""Allen session C: receptive-field content from natural-movie correlations.
(a) RF-distance dependence of correlation, within and across mice
(b) labelled ridge ceilings for relative RF (within random halves; leave-one-mouse-out)
(c) class-level cross-animal matching on a per-mouse 2x2 quantile grid of relative RF
    (pooled reference = all other mice; test = one mouse), accuracy and accuracy
    modulo grid flips/transposes; shuffled-label null."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from microns_ambiguity.geometric import acc, acc_modulo, all_perms, argmin_tiebreak, perm_distances
from microns_ambiguity.relational import bin_grid, class_gram, cosine_gram, grid_group
from microns_ambiguity.data import zscore_rows
from allen.data import Allen, ROOT

OUT = os.path.join(ROOT, "allen", "outputs")


def z(M):
    return (M - M.mean()) / (M.std() + 1e-9)


def main(movie="both", grid=(2, 2), min_cells=20, n_null=50):
    ds = Allen(session="C", movie=movie); s = ds.summary(); print({k: v for k, v in s.items() if "per_mouse" not in k}, flush=True)
    G = cosine_gram(zscore_rows(ds.R)); ok = ds.rf_rel_ok; idx = np.flatnonzero(ok)
    res = {}
    # (a)
    same = ds.mouse[idx][:, None] == ds.mouse[idx][None, :]; off = ~np.eye(len(idx), dtype=bool)
    rfd = np.linalg.norm(ds.rf[idx][:, None] - ds.rf[idx][None, :], axis=-1); Gi = G[np.ix_(idx, idx)]
    res["corr_by_rf_distance"] = {f"{lo}-{hi}": dict(same=float(Gi[same & off & (rfd >= lo) & (rfd < hi)].mean()), cross=float(Gi[~same & (rfd >= lo) & (rfd < hi)].mean()))
                                  for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]}
    print("corr by RF distance (deg):", {k: (round(v["same"], 4), round(v["cross"], 4)) for k, v in res["corr_by_rf_distance"].items()}, flush=True)
    # (b)
    y = ds.rf_rel; rng = np.random.default_rng(0); perm = rng.permutation(idx); tr, va = perm[: len(idx) // 2], perm[len(idx) // 2:]
    reg = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(G[np.ix_(tr, tr)], y[tr]); P = reg.predict(G[np.ix_(va, tr)])
    r2w = (1 - ((P - y[va]) ** 2).sum(0) / ((y[va] - y[va].mean(0)) ** 2).sum(0)).mean()
    lomo = []
    for m in ds.mice:
        te = idx[ds.mouse[idx] == m]; trm = idx[ds.mouse[idx] != m]
        if len(te) < min_cells: continue
        reg = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(G[np.ix_(trm, trm)], y[trm]); P = reg.predict(G[np.ix_(te, trm)])
        lomo.append((1 - ((P - y[te]) ** 2).sum(0) / ((y[te] - y[te].mean(0)) ** 2).sum(0)).mean())
    res["ceiling"] = dict(within_r2=float(r2w), lomo_r2_mean=float(np.mean(lomo)), lomo_r2_sd=float(np.std(lomo)), n_mice=len(lomo))
    print("labelled ridge ceiling (relative RF): within", round(r2w, 3), " leave-one-mouse-out", round(np.mean(lomo), 3), "+-", round(np.std(lomo), 3), flush=True)
    # (c) per-mouse grid classes
    K = grid[0] * grid[1]; group = grid_group(grid); P_ = all_perms(K)
    lab = np.full(ds.n, -1)
    for m in ds.mice:
        sel = np.flatnonzero((ds.mouse == m) & ok)
        if len(sel) >= min_cells: lab[sel] = bin_grid(ds.rf_rel[sel], grid)
    mice = [m for m in ds.mice if ((ds.mouse == m) & (lab >= 0)).sum() >= min_cells]
    Ms = {m: class_gram(G[np.ix_(np.flatnonzero((ds.mouse == m) & (lab >= 0)), np.flatnonzero((ds.mouse == m) & (lab >= 0)))], lab[(ds.mouse == m) & (lab >= 0)], K) for m in mice}
    pooled = {}
    for m in mice:
        others = np.flatnonzero(np.isin(ds.mouse, [o for o in mice if o != m]) & (lab >= 0))
        MA = class_gram(G[np.ix_(others, others)], lab[others], K)
        D = perm_distances(z(MA), z(Ms[m])); best = P_[argmin_tiebreak(D, rng)].astype(int)
        sel = np.flatnonzero((ds.mouse == m) & (lab >= 0)); nulls = []
        for _ in range(n_null):
            sl = rng.permutation(lab[sel]); Mn = class_gram(G[np.ix_(sel, sel)], sl, K); Dn = perm_distances(z(MA), z(Mn))
            bn = P_[argmin_tiebreak(Dn, rng)].astype(int); nulls.append((acc(bn, K), acc_modulo(bn, group)))
        pooled[m] = dict(acc=acc(best, K), acc_mod=acc_modulo(best, group), null_acc=float(np.mean([x[0] for x in nulls])), null_mod=float(np.mean([x[1] for x in nulls])), n=int(len(sel)))
    res["pooled"] = pooled
    res["summary"] = dict(K=K, grid=grid, n_mice=len(mice), acc=float(np.mean([v["acc"] for v in pooled.values()])), acc_mod=float(np.mean([v["acc_mod"] for v in pooled.values()])),
                          null_acc=float(np.mean([v["null_acc"] for v in pooled.values()])), null_mod=float(np.mean([v["null_mod"] for v in pooled.values()])), chance=1 / K)
    print("cross-animal RF grid matching (pooled reference -> test mouse):", res["summary"], flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, f"rf_{movie}.json"), "w"))


if __name__ == "__main__":
    main(movie=sys.argv[1] if len(sys.argv) > 1 else "both")
