"""Sweep runner for decoder2. Usage:
python -m microns_ambiguity.run_decoder2 <tag> <substrate> <content> [n=128] [dim=128] [layers=2] [relbias=0] [epochs=10] [pops=2000] [batch=32] [rowproj=1] [device=cpu]
Appends to outputs/decoder2.json under key <tag>."""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from .config import K_ORI, OUT, SEED
from .data import Dataset
from .decoder2 import train
from .geometric import stratified_half_split
from .relational import bin_orientation, label_classes


def main(tag, sub, con, **kw):
    ds = Dataset()
    F = ds.func_iv if sub == "func_iv" else ds.func_is
    if sub == "func_is" and kw.pop("pca", 0):
        from .config import DATA
        F = np.load(DATA / "func_is_pca512.npy")   # 512 PCs of the twin responses; Gram preserved to r > 0.99
    if con == "ori":
        y, ok, task = bin_orientation(ds.ori, K_ORI), ds.ori_ok, "class"
    elif con == "rf":
        y, ok, task = ds.rf, ds.rf_ok, "reg"
    elif con == "rf_dist":
        y, ok, task = np.linalg.norm(ds.rf - np.median(ds.rf[ds.rf_ok], 0), axis=1), ds.rf_ok, "reg"
    elif con == "area":
        lab = np.full(ds.n, -1); lab[ds.area_ok] = label_classes(ds.area[ds.area_ok], ["V1", "RL", "AL"]); y, ok, task = lab, ds.area_ok, "class"
    keep = np.flatnonzero(ok)
    strat = y[keep] if task == "class" else np.zeros(len(keep), int)
    tr, va = stratified_half_split(strat, np.random.default_rng(SEED)); tr, va = keep[tr], keep[va]
    t0 = time.time(); print(f"[{tag}] {sub} {con} n_neurons={len(keep)} {kw}", flush=True)
    m = train(F, y, task, tr, va, **kw)
    m.update(sub=sub, con=con, seconds=time.time() - t0, n_neurons=int(len(keep)))
    OUT.mkdir(exist_ok=True); p = OUT / "decoder2.json"
    d = json.load(open(p)) if p.exists() else {}
    d[tag] = m; json.dump(d, open(p, "w"))
    print(f"  -> {tag}: " + " ".join(f"{k}={v:.3f}" for k, v in m.items() if k in ("acc", "r2", "loss", "acc_modD", "sel_metric", "val_at_last_epoch") or "avg" in k) + (f" best_epoch={m['best_epoch']}" if "best_epoch" in m else "") + f" params={m['params']} {m['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    tag, sub, con = sys.argv[1:4]
    kw = {}
    for a in sys.argv[4:]:
        k, v = a.split("="); kw[k] = v if k == "device" else (bool(int(v)) if k in ("rel_bias", "row_proj") else (float(v) if k in ("lr", "early_stop", "dropout", "cond_frac", "gram_drop") else int(v)))
    pca = kw.pop("pca", 0)
    if pca: kw["pca"] = 1
    main(tag, sub, con, **kw)
