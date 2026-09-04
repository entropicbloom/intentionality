"""Protocol 2 runner. Writes outputs/decoder.json (merging into existing)."""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from .ars import ars_fano, ars_gaussian
from .config import DEC_POP, DEC_SEEDS, K_ORI, OUT, SEED
from .data import Dataset, rewire_within_adp
from .decoder import train_decoder
from .geometric import stratified_half_split
from .relational import bin_orientation, cosine_gram, label_classes, soma_kernel


def build(ds, sub, rng):
    if sub == "soma":
        rows, S = ds.substrate("soma"); return rows, soma_kernel(S)
    if sub.endswith("_rewired"):
        rows, F = ds.substrate(sub.replace("_rewired", ""), W_syn=rewire_within_adp(ds, rng))
        return rows, cosine_gram(F)
    rows, F = ds.substrate(sub); return rows, cosine_gram(F)


def target(ds, rows, con):
    vals, ok = ds.content(con); v, ok = vals[rows], ok[rows]
    if con == "ori":
        return bin_orientation(v, K_ORI), ok, "class", K_ORI
    if con in ("rf", "rf_resid", "soma_xz"):
        return v, ok, "reg", None
    if con == "layer":
        lab = np.full(len(rows), -1); lab[ok] = label_classes(v[ok], ["L2/3", "L4", "L5"]); return lab, ok, "class", 3
    if con == "area":
        lab = np.full(len(rows), -1); lab[ok] = label_classes(v[ok], ["V1", "RL", "AL"]); return lab, ok, "class", 3
    raise KeyError(con)


def main(substrates, contents, variants=("full",), seeds=DEC_SEEDS, pop=DEC_POP, **kw):
    ds = Dataset(); rng = np.random.default_rng(SEED)
    OUT.mkdir(exist_ok=True); path = OUT / "decoder.json"
    results = json.load(open(path)) if path.exists() else {}
    for sub in substrates:
        rows, G = build(ds, sub, rng)
        for con in contents:
            y, ok, task, K = target(ds, rows, con)
            keep = np.flatnonzero(ok & (y >= 0)) if task == "class" else np.flatnonzero(ok)
            strat = y[keep] if task == "class" else np.zeros(len(keep), int)
            for variant in variants:
                for seed in range(seeds):
                    key = f"{sub}|{con}|{variant}|s{seed}" + (f"|n{pop}" if pop != DEC_POP else "")
                    if key in results: print("skip", key); continue
                    tr, va = stratified_half_split(strat, np.random.default_rng(seed))
                    tr, va = keep[tr], keep[va]
                    yy = y.copy()
                    if variant == "shuffled":
                        yy[keep] = np.random.default_rng(seed + 7).permutation(y[keep])
                    t0 = time.time(); print(f"[{key}] n={len(keep)}", flush=True)
                    m = train_decoder(G, yy, task, tr, va, seed=seed, n=pop, target_only=(variant == "target_only"), **kw)
                    m.update(n=int(len(keep)), K=K, task=task, pop=pop, seconds=time.time() - t0)
                    m["ars"] = ars_fano(m["acc"], K) if task == "class" else ars_gaussian(m["r2"])
                    results[key] = m
                    json.dump(results, open(path, "w"))
                    print(f"  -> {key}: " + " ".join(f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, float)), flush=True)
    return results


if __name__ == "__main__":
    subs = sys.argv[1].split(","); cons = sys.argv[2].split(",")
    variants = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("full",)
    seeds = int(sys.argv[4]) if len(sys.argv) > 4 else DEC_SEEDS
    pop = int(sys.argv[5]) if len(sys.argv) > 5 else DEC_POP
    main(subs, cons, variants, seeds, pop)
