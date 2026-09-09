"""Per-neuron decoder on Allen data, three regimes:
  within : train/val = disjoint halves of neurons, populations sampled within a mouse
  cross  : train mice / test mice disjoint (leave-k-mice-out); populations within a mouse
  pooled : populations sampled across all mice (correlations across mice are
           meaningful because the movies are shared)
Reuses microns_ambiguity.decoder2 (label-free, dense supervision)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from microns_ambiguity import decoder2 as d2
from microns_ambiguity.relational import bin_orientation
from allen.data import Allen, ROOT

OUT = os.path.join(ROOT, "allen", "outputs")


class MouseSampler(d2.Sampler):
    """Populations drawn within one mouse (mouse chosen at random, weighted by size)."""
    def __init__(self, Fn, pools, n, rng, mean, std):
        self.pools = [p for p in pools if len(p) >= n]; w = np.array([len(p) for p in self.pools], float)
        self.Fn, self.n, self.rng, self.mean, self.std, self.w = Fn, n, rng, mean, std, w / w.sum()
        self.pool = np.concatenate(self.pools)

    def batch(self, B):
        import torch
        idx = np.stack([self.rng.choice(self.pools[self.rng.choice(len(self.pools), p=self.w)], self.n, replace=False) for _ in range(B)])
        it = torch.as_tensor(idx, device=self.Fn.device); X = self.Fn[it]
        G = torch.bmm(X, X.transpose(1, 2)); G = (G - self.mean) / self.std; G.diagonal(dim1=1, dim2=2).zero_()
        return idx, G


def main(tag, regime="cross", n=128, test_frac=0.3, movie="both", ori_source="sg", content="ori", session="A", **kw):
    ds = Allen(movie=movie, ori_source=ori_source, session=session)
    if content == "ori":
        y = ds.ori_class.copy(); y[~ds.ori_ok] = -1; task = "class"
    elif content == "rf":
        y = ds.rf.copy(); y[~ds.rf_ok] = np.nan; task = "reg"
    elif content == "rf_rel":
        y = ds.rf_rel.copy(); y[~ds.rf_rel_ok] = np.nan; task = "reg"
    elif content == "rf_dist":
        y = ds.rf_dist.copy(); task = "reg"
    # every neuron is a token; only labelled neurons are supervised / scored
    keep = np.ones(ds.n, bool)
    rng = np.random.default_rng(kw.get("seed", 0))
    mice = np.array(ds.mice)
    if regime == "cross":
        test_mice = rng.choice(mice, max(1, int(len(mice) * test_frac)), replace=False)
        tr = np.flatnonzero(keep & ~np.isin(ds.mouse, test_mice)); va = np.flatnonzero(keep & np.isin(ds.mouse, test_mice))
    else:
        idx = np.flatnonzero(keep); perm = rng.permutation(idx); tr, va = perm[: len(idx) // 2], perm[len(idx) // 2:]
    pools_tr = [np.intersect1d(tr, np.flatnonzero(ds.mouse == m)) for m in mice]
    pools_va = [np.intersect1d(va, np.flatnonzero(ds.mouse == m)) for m in mice]
    print(f"[{tag}] regime={regime} train neurons={len(tr)} val neurons={len(va)} mice={len(mice)}", flush=True)
    if regime in ("cross", "within"):
        # monkey-patch samplers so populations stay within a mouse
        orig = d2.Sampler
        class S(orig):
            def __init__(self, Fn, pool, n_, rng_, mean, std):
                # restrict whatever pool train() hands us (train / validation / selection) to within-mouse populations
                pools = [np.intersect1d(np.asarray(pool), np.flatnonzero(ds.mouse == m)) for m in mice]
                self._ms = MouseSampler(Fn, pools, n_, rng_, mean, std); self.Fn, self.n, self.rng, self.mean, self.std = Fn, n_, rng_, mean, std
                self.pool = self._ms.pool
            def batch(self, B):
                return self._ms.batch(B)
        d2.Sampler = S
    try:
        m = d2.train(ds.R, y, task, tr, va, n=n, **kw)
    finally:
        if regime in ("cross", "within"): d2.Sampler = orig
    m.update(regime=regime, movie=movie, content=content, session=session, n_train=int(len(tr)), n_val=int(len(va)), test_mice=[str(x) for x in (test_mice if regime == "cross" else [])])
    os.makedirs(OUT, exist_ok=True); p = os.path.join(OUT, "decoder.json"); d = json.load(open(p)) if os.path.exists(p) else {}
    d[tag] = m; json.dump(d, open(p, "w"))
    key = "acc" if task == "class" else "r2"
    print(f"  -> {tag}: {key}={m[key]:.3f} avg32={m.get(key + '_avg32', 0):.3f}" + (f" best_epoch={m['best_epoch']}" if "best_epoch" in m else ""), flush=True)


if __name__ == "__main__":
    tag, regime = sys.argv[1], sys.argv[2]; kw = {}
    for a in sys.argv[3:]:
        k, v = a.split("="); kw[k] = v if k in ("device", "movie", "ori_source", "content", "session") else (float(v) if k in ("lr", "early_stop", "dropout", "test_frac") else (bool(int(v)) if k == "rel_bias" else int(v)))
    main(tag, regime, **kw)
