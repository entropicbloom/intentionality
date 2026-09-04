"""Protocol 1: class-level geometric matching across substrates x contents.
Writes outputs/geometric.json."""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from .ars import ars_fano
from .config import K_ORI, N_SPLITS, OUT, RF_GRID, SEED
from .data import Dataset, rewire_within_adp
from .geometric import run_matching
from .relational import (bin_grid, bin_orientation, class_gram, cosine_gram,
                         dihedral_group, grid_group, label_classes, soma_kernel)

SMALL = {"struct_out", "struct_out_adp", "struct_out_rewired"}


def gram_for(ds, name, rng):
    if name == "soma":
        rows, S = ds.substrate("soma")
        return rows, soma_kernel(S)
    if name.endswith("_rewired"):
        base = name.replace("_rewired", "")
        rows, F = ds.substrate(base, W_syn=rewire_within_adp(ds, rng))
        return rows, cosine_gram(F)
    rows, F = ds.substrate(name)
    return rows, cosine_gram(F)


def labels_for(ds, rows, content, small):
    vals, ok = ds.content(content)
    ok = ok[rows]
    v = vals[rows]
    if content == "ori":
        K = 4 if small else K_ORI
        lab = bin_orientation(v, K)
        return lab, ok, K, dihedral_group(K), "circular"
    if content in ("rf", "rf_resid", "soma_xz"):
        grid = (2, 2) if small else RF_GRID
        lab = np.full(len(rows), -1)
        lab[ok] = bin_grid(v[ok], grid)
        return lab, ok, grid[0] * grid[1], grid_group(grid), "grid"
    if content == "depth":
        K = 4
        q = np.quantile(v[ok], np.linspace(0, 1, K + 1)[1:-1])
        return np.searchsorted(q, v), ok, K, None, "ordinal"
    if content == "layer":
        names = ["L2/3", "L4", "L5"]
        lab = np.full(len(rows), -1); lab[ok] = label_classes(v[ok], names)
        return lab, ok, 3, None, "nominal"
    if content == "area":
        names = ["V1", "RL", "AL"]
        lab = np.full(len(rows), -1); lab[ok] = label_classes(v[ok], names)
        return lab, ok, 3, None, "nominal"
    raise KeyError(content)


def main(substrates=None, contents=None, n_splits=N_SPLITS, tag="geometric"):
    ds = Dataset()
    rng = np.random.default_rng(SEED)
    substrates = substrates or ["struct_out", "struct_out_adp", "struct_out_rewired",
                                "struct_in", "struct_in_adp", "struct_in_rewired",
                                "func_iv", "func_is", "soma"]
    contents = contents or ["ori", "rf", "soma_xz", "depth", "layer", "area"]
    results = {}
    for sub in substrates:
        t0 = time.time()
        rows, G = gram_for(ds, sub, rng)
        print(f"[{sub}] gram {G.shape} in {time.time()-t0:.1f}s", flush=True)
        for con in contents:
            lab, ok, K, group, kind = labels_for(ds, rows, con, sub in SMALL)
            keep = ok & (lab >= 0)
            Gk, lk = G[np.ix_(keep, keep)], lab[keep]
            counts = np.bincount(lk, minlength=K)
            if counts.min() < 4:
                print(f"  {con}: skipped (min class count {counts.min()})"); continue
            t0 = time.time()
            res = run_matching(Gk, lk, K, n_splits, SEED, group=group)
            # two nulls: (fixed) labels shuffled once, i.e. arbitrary but fixed neuron
            # groups - conservative, keeps group-level idiosyncrasy; (indep) labels
            # shuffled independently within each half - pure chance.
            nf = run_matching(Gk, rng.permutation(lk), K, max(50, n_splits // 4), SEED + 1, group=group)
            ni = run_matching(Gk, lk, K, max(50, n_splits // 4), SEED + 2, group=group, shuffle_within_halves=True)
            res.update(n=int(keep.sum()), class_counts=counts.tolist(), kind=kind,
                       ars_fano=ars_fano(res["acc"], K), chance=1.0 / K,
                       null_fixed={k: nf[k] for k in ("acc", "acc_sd", "acc_mod", "hit", "ars_posterior")},
                       null_indep={k: ni[k] for k in ("acc", "acc_sd", "acc_mod", "hit", "ars_posterior")},
                       class_gram=class_gram(Gk, lk, K).tolist())
            null = nf
            results[f"{sub}|{con}"] = res
            print(f"  {con:8s} K={K} n={keep.sum():5d} acc={res['acc']:.3f}±{res['acc_sd']:.3f} "
                  f"mod={res['acc_mod']:.3f} hit={res['hit']:.2f} ARSpost={res['ars_posterior']:.3f} "
                  f"| nullfix acc={nf['acc']:.3f} mod={nf['acc_mod']:.3f} nullind acc={ni['acc']:.3f} mod={ni['acc_mod']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{tag}.json", "w") as f:
        json.dump(results, f)
    return results


if __name__ == "__main__":
    subs = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    cons = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    tag = sys.argv[3] if len(sys.argv) > 3 else "geometric"
    main(subs, cons, tag=tag)
