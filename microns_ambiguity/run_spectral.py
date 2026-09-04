"""Protocol 3 runner. Writes outputs/spectral.json."""
from __future__ import annotations

import json
import sys

import numpy as np

from .ars import ars_gaussian
from .config import OUT, SEED
from .data import Dataset
from .run_geometric import gram_for
from .spectral import evaluate


def targets(ds, rows):
    out = {}
    v, ok = ds.content("rf"); out["rf"] = (v[rows], ok[rows])
    v, ok = ds.content("soma_xz"); out["soma_xz"] = (v[rows], ok[rows])
    th = np.deg2rad(ds.ori[rows]) * 2
    out["ori"] = (np.stack([np.cos(th), np.sin(th)], 1), ds.ori_ok[rows])
    return out


def main(substrates=None, max_n=6000):
    ds = Dataset(); rng = np.random.default_rng(SEED)
    substrates = substrates or ["func_iv", "func_is", "struct_in", "struct_in_adp", "struct_in_rewired", "soma"]
    results = {}
    for sub in substrates:
        rows, G = gram_for(ds, sub, rng)
        if len(rows) > max_n:                      # eigendecomposition cost
            sel = np.sort(rng.choice(len(rows), max_n, replace=False)); rows, G = rows[sel], G[np.ix_(sel, sel)]
        for con, (y, ok) in targets(ds, rows).items():
            keep = np.flatnonzero(ok)
            res = evaluate(G[np.ix_(keep, keep)], y[keep], seed=SEED)
            res["n"] = int(len(keep)); res["ars_gaussian_linear_m50"] = ars_gaussian(res["linear_m50"])
            results[f"{sub}|{con}"] = res
            print(f"[{sub}|{con}] n={len(keep)} procrustes2={res['procrustes_top2']:.3f} "
                  f"lin10={res['linear_m10']:.3f} lin50={res['linear_m50']:.3f} | null lin50={res['null_linear_m50']:.3f}", flush=True)
    OUT.mkdir(exist_ok=True)
    json.dump(results, open(OUT / "spectral.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
