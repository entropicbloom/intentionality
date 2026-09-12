"""Sweep runner for decoder2. Usage:
python -m microns_ambiguity.run_decoder2 <tag> <substrate> <content> [n=128] [dim=128] [layers=2] [relbias=0] [epochs=10] [pops=2000] [batch=32] [rowproj=1] [device=cpu]
    [split_seed=<int>]  seed of the stratified neuron half split (default: config SEED)
    [bins=1|2]          cross-stimulus test (2: control with the same half on both sides): training/selection Grams from a random half of the stimulus bins,
                        test Grams from the other half (disjoint); twin uses the raw 4999-bin responses, not the PCs
    [input_mode=act]    raw-activity reference: tokens from response vectors instead of Gram rows (rel_bias=0: no Gram at all; layers=0: linear per-neuron readout)
    [label_rot=1]       labels of each training population rotated by a random angle (no frame in the labels; report err_modD)
    [ori_weight=1]      loss weighted by inverse label density (orientation-balanced training)
    [area_train=V1 area_test=RL]  cross-area transfer: training half restricted to one area, test half to another
    [save_preds=1]      save averaged per-neuron predictions to outputs/preds/<tag>.npz (idx, P, y, scan, area)
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
    split_seed = kw.pop("split_seed", SEED); bins = kw.pop("bins", 0); save_preds = kw.pop("save_preds", 0)
    if bins:                                     # bins=1: disjoint halves of the stimulus bins for training and test Grams
        kw.pop("pca", None); perm = np.random.default_rng(SEED + 11).permutation(F.shape[1]); h = len(perm) // 2
        if bins == 2: F = F[:, perm[:h]]         # bins=2: control, the same half for both (isolates the cost of fewer bins)
        else: F, kw["F_eval"] = F[:, perm[:h]], F[:, perm[h:2 * h]]   # equal widths (the activity input layer needs them)
    if sub == "func_is" and kw.pop("pca", 0):
        from .config import DATA
        F = np.load(DATA / "func_is_pca512.npy")   # 512 PCs of the twin responses; Gram preserved to r > 0.99
    if con == "ori":
        y, ok, task = bin_orientation(ds.ori, K_ORI), ds.ori_ok, "class"
    elif con == "oricirc":                       # continuous preferred orientation, circular regression
        y = ds.ori.astype(float).copy(); y[~ds.ori_ok] = np.nan; ok, task = ds.ori_ok, "circ"
    elif con == "rf":
        y, ok, task = ds.rf, ds.rf_ok, "reg"
    elif con == "rf_dist":
        y, ok, task = np.linalg.norm(ds.rf - np.median(ds.rf[ds.rf_ok], 0), axis=1), ds.rf_ok, "reg"
    elif con == "area":
        lab = np.full(ds.n, -1); lab[ds.area_ok] = label_classes(ds.area[ds.area_ok], ["V1", "RL", "AL"]); y, ok, task = lab, ds.area_ok, "class"
    keep = np.flatnonzero(ok)
    strat = y[keep] if task == "class" else np.zeros(len(keep), int)
    tr, va = stratified_half_split(strat, np.random.default_rng(split_seed)); tr, va = keep[tr], keep[va]
    area_train, area_test = kw.pop("area_train", ""), kw.pop("area_test", "")     # cross-area transfer: restrict the halves by cortical area
    if area_train: tr = tr[ds.area[tr] == area_train]
    if area_test: va = va[ds.area[va] == area_test]
    t0 = time.time(); print(f"[{tag}] {sub} {con} n_neurons={len(keep)} split_seed={split_seed} bins={bins} {({k: v for k, v in kw.items() if k != 'F_eval'})}", flush=True)
    m = train(F, y, task, tr, va, return_preds=bool(save_preds), **{k: v for k, v in kw.items()})
    if save_preds:
        pr = m.pop("preds"); (OUT / "preds").mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "preds" / f"{tag}.npz", idx=pr["idx"], P=pr["P"], y=np.asarray(y, float)[pr["idx"]], scan=ds.scan[pr["idx"]].astype(str), area=ds.area[pr["idx"]].astype(str))
    m.pop("preds", None); m.pop("F_eval", None)
    m.update(sub=sub, con=con, seconds=time.time() - t0, n_neurons=int(len(keep)), split_seed=int(split_seed), bins=int(bins), area_train=area_train, area_test=area_test, n_train=int(len(tr)), n_test=int(len(va)))
    OUT.mkdir(exist_ok=True); p = OUT / "decoder2.json"
    d = json.load(open(p)) if p.exists() else {}
    d[tag] = m; json.dump(d, open(p, "w"))
    print(f"  -> {tag}: " + " ".join(f"{k}={v:.3f}" for k, v in m.items() if k in ("acc", "r2", "err", "within15", "err_modD", "loss", "acc_modD", "sel_metric", "val_at_last_epoch") or "avg" in k) + (f" best_epoch={m['best_epoch']}" if "best_epoch" in m else "") + f" params={m['params']} {m['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    tag, sub, con = sys.argv[1:4]
    kw = {}
    for a in sys.argv[4:]:
        k, v = a.split("="); kw[k] = v if k in ("device", "input_mode", "area_train", "area_test") else (bool(int(v)) if k in ("rel_bias", "row_proj", "label_rot", "ori_weight") else (float(v) if k in ("lr", "early_stop", "dropout", "cond_frac", "gram_drop", "aug_prob") else int(v)))
    pca = kw.pop("pca", 0)
    if pca: kw["pca"] = 1
    main(tag, sub, con, **kw)
