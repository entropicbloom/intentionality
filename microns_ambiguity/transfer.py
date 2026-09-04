"""Cross-substrate transfer (the paper's cross-architecture test): reference
class-Gram from substrate X on neuron half A, test class-Gram from substrate Y
on half B.  Class-Grams are z-scored so substrates with different similarity
scales are comparable.  Writes outputs/transfer.json."""
from __future__ import annotations

import json
import sys

import numpy as np

from .config import K_ORI, N_SPLITS, OUT, RF_GRID, SEED
from .data import Dataset
from .geometric import acc, acc_modulo, all_perms, perm_distances, stratified_half_split
from .relational import bin_grid, bin_orientation, class_gram, dihedral_group, grid_group
from .run_geometric import gram_for


def z(M):
    return (M - M.mean()) / (M.std() + 1e-9)


def main(subs=None, n_splits=N_SPLITS // 2):
    ds = Dataset(); rng = np.random.default_rng(SEED)
    subs = subs or ["func_iv", "func_is", "struct_in", "struct_in_adp", "soma"]
    grams = {}
    rows = ds.post_idx                          # common neuron set: recipients of proofread axons
    for s in subs:
        r, G = gram_for(ds, s, rng)
        pos = {v: i for i, v in enumerate(r)}
        sel = np.array([pos[v] for v in rows])
        grams[s] = G[np.ix_(sel, sel)]
    contents = {}
    lab = bin_orientation(ds.ori[rows], K_ORI); contents["ori"] = (lab, ds.ori_ok[rows], K_ORI, dihedral_group(K_ORI))
    ok = ds.rf_ok[rows]; lab = np.full(len(rows), -1); lab[ok] = bin_grid(ds.rf[rows][ok], RF_GRID)
    contents["rf"] = (lab, ok, RF_GRID[0] * RF_GRID[1], grid_group(RF_GRID))
    results = {}
    for con, (lab, ok, K, group) in contents.items():
        keep = np.flatnonzero(ok); lk = lab[keep]
        for X in subs:
            for Y in subs:
                accs, mods, hits = [], [], []
                r2 = np.random.default_rng(SEED + 1)
                for _ in range(n_splits):
                    A, B = stratified_half_split(lk, r2)
                    MA = z(class_gram(grams[X][np.ix_(keep[A], keep[A])], lk[A], K))
                    MB = z(class_gram(grams[Y][np.ix_(keep[B], keep[B])], lk[B], K))
                    D = perm_distances(MA, MB); best = all_perms(K)[int(np.argmin(D))].astype(int)
                    accs.append(acc(best, K)); mods.append(acc_modulo(best, group))
                    hits.append(float(best.tolist() == list(range(K))))
                results[f"{con}|{X}->{Y}"] = dict(acc=float(np.mean(accs)), acc_mod=float(np.mean(mods)), hit=float(np.mean(hits)), K=K, n=int(len(keep)))
                print(f"{con:4s} {X:14s}->{Y:14s} acc={np.mean(accs):.3f} mod={np.mean(mods):.3f} hit={np.mean(hits):.2f}", flush=True)
    OUT.mkdir(exist_ok=True); json.dump(results, open(OUT / "transfer.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
