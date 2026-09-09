"""Per-neuron relational decoder, second iteration.

Differences from decoder.py:
 * populations' Grams are computed on the fly from the (N x d) response matrix,
   so population size is limited by compute, not by holding an N x N matrix;
 * every token is supervised (dense); inputs standardized;
 * optional 'relbias' architecture: the Gram enters attention as a learned
   per-head additive bias on the logits (a graph transformer over the
   relational matrix), in addition to the row-projection tokens;
 * model width / depth / batch are parameters.
No labels ever enter the input.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn


class RelAttention(nn.Module):
    def __init__(self, dim, heads, rel_bias):
        super().__init__()
        self.h, self.dk = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim); self.out = nn.Linear(dim, dim)
        self.rel_bias = rel_bias
        if rel_bias:
            self.bias_scale = nn.Parameter(torch.ones(heads) * 0.5)

    def forward(self, x, G):
        B, n, d = x.shape
        q, k, v = self.qkv(x).view(B, n, 3, self.h, self.dk).unbind(2)
        att = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.dk)
        if self.rel_bias:
            att = att + self.bias_scale.view(1, -1, 1, 1) * G.unsqueeze(1)
        att = att.softmax(-1)
        y = torch.einsum("bhij,bjhd->bihd", att, v).reshape(B, n, d)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, dim, heads, rel_bias, dropout=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.att = RelAttention(dim, heads, rel_bias)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, G):
        x = x + self.drop(self.att(self.n1(x), G))
        return x + self.drop(self.mlp(self.n2(x)))


class RelDecoder(nn.Module):
    def __init__(self, n_tokens, out_dim, dim=128, heads=4, layers=2, rel_bias=False, row_proj=True, dropout=0.1):
        super().__init__()
        self.row_proj = row_proj
        # token init: projected Gram row (+ 3 permutation-invariant row statistics)
        self.inp = nn.Linear((n_tokens if row_proj else 0) + 3, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, rel_bias, dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))

    def forward(self, G):                                  # G: (B, n, n) standardized
        stats = torch.stack([G.mean(-1), G.std(-1), (G ** 3).mean(-1)], -1)
        x = torch.cat([G, stats], -1) if self.row_proj else stats
        x = self.inp(x)
        for b in self.blocks:
            x = b(x, G)
        return self.head(x)                                 # (B, n, out)


class Sampler:
    """Populations of n neurons from a pool; Gram computed from normalized features."""
    def __init__(self, Fn: torch.Tensor, pool, n, rng, mean, std):
        self.Fn, self.pool, self.n, self.rng, self.mean, self.std = Fn, np.asarray(pool), n, rng, mean, std

    def batch(self, B):
        idx = np.stack([self.rng.choice(self.pool, self.n, replace=False) for _ in range(B)])
        it = torch.as_tensor(idx, device=self.Fn.device)
        X = self.Fn[it]                                     # (B, n, d)
        G = torch.bmm(X, X.transpose(1, 2))
        G = (G - self.mean) / self.std
        G.diagonal(dim1=1, dim2=2).zero_()
        return idx, G


def normalize_features(F):
    F = np.asarray(F, np.float32); F = F - F.mean(1, keepdims=True)
    nrm = np.linalg.norm(F, axis=1, keepdims=True); nrm[nrm == 0] = 1
    return F / nrm


def train(F, y, task, train_idx, val_idx, n=128, dim=128, heads=4, layers=2, rel_bias=False, row_proj=True,
          epochs=10, pops_per_epoch=2000, batch=32, lr=1e-3, seed=0, device="cpu", verbose=True, val_pops=200,
          heads_=None, early_stop=0.0, dropout=0.1):
    """F: (N, d) responses; y: labels (N,) int or (N, k) float. Dense supervision.
    early_stop: fraction of the TRAINING neurons held out as a selection set; the
    reported validation metric is taken at the epoch that is best on that set, so
    the validation neurons never influence model selection."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    sel_idx = None
    if early_stop > 0:
        train_idx = rng.permutation(np.asarray(train_idx)); k = int(len(train_idx) * early_stop)
        sel_idx, train_idx = train_idx[:k], train_idx[k:]
    Fn = torch.as_tensor(normalize_features(F), device=device)
    # global off-diagonal statistics of the correlation, from a random sample
    s = rng.choice(len(F), min(2000, len(F)), replace=False); Gs = Fn[s] @ Fn[s].T
    off = Gs[~torch.eye(len(s), dtype=torch.bool, device=device)]
    mean, std = float(off.mean()), float(off.std())
    if task == "class":
        K = int(y.max()) + 1; yt = torch.as_tensor(y, device=device, dtype=torch.long); out_dim = K
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1); labelled = y >= 0
    else:
        y = np.atleast_2d(y.T).T.astype(np.float32); labelled = np.isfinite(y).all(1)
        mu, sd = np.nanmean(y[train_idx], 0), np.nanstd(y[train_idx], 0)
        yt = torch.as_tensor(np.nan_to_num((y - mu) / sd), device=device); out_dim = y.shape[1]
        lab_t = torch.as_tensor(labelled, device=device)
        def loss_fn(out, tgt, _idx=None):
            m = lab_t[_idx]
            return ((out[m] - tgt[m]) ** 2).mean() if m.any() else (out * 0).sum()
    model = RelDecoder(n, out_dim, dim, heads, layers, rel_bias, row_proj, dropout=dropout).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (pops_per_epoch // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    tr = Sampler(Fn, train_idx, n, rng, mean, std); va = Sampler(Fn, val_idx, n, np.random.default_rng(seed + 100), mean, std)
    # selection populations are drawn from selection + training neurons (the
    # selection slice alone can be smaller than n); only selection neurons are scored
    se = None
    if sel_idx is not None:
        pool = sel_idx if len(sel_idx) >= n else np.concatenate([sel_idx, train_idx])
        se = Sampler(Fn, pool, n, np.random.default_rng(seed + 200), mean, std)

    def evaluate(reps=1, sampler=None, pool_idx=None, score_only=None):
        """reps>1: cover the pool `reps` times with disjoint populations and
        average each neuron's prediction (logits / standardized values)."""
        sampler = sampler or va; pool_idx = val_idx if pool_idx is None else pool_idx
        model.eval(); N = len(y); S = np.zeros((N, out_dim), np.float32); C = np.zeros(N)
        with torch.no_grad():
            if reps == 1:
                for _ in range(max(1, val_pops // batch)):
                    idx, G = sampler.batch(batch); out = model(G).cpu().numpy()
                    np.add.at(S, idx.reshape(-1), out.reshape(-1, out_dim)); np.add.at(C, idx.reshape(-1), 1)
            else:
                pool = np.asarray(pool_idx); r = np.random.default_rng(seed + 7)
                for _ in range(reps):
                    perm = r.permutation(pool); perm = perm[: (len(perm) // n) * n].reshape(-1, n)
                    for b in range(0, len(perm), batch):
                        idx = perm[b:b + batch]; it = torch.as_tensor(idx, device=device); X = Fn[it]
                        G = torch.bmm(X, X.transpose(1, 2)); G = (G - mean) / std; G.diagonal(dim1=1, dim2=2).zero_()
                        out = model(G).cpu().numpy()
                        np.add.at(S, idx.reshape(-1), out.reshape(-1, out_dim)); np.add.at(C, idx.reshape(-1), 1)
        model.train()
        if score_only is not None:
            keep = np.zeros(N, bool); keep[np.asarray(score_only)] = True; C = C * keep
        C = C * labelled                           # score labelled neurons only
        T = np.flatnonzero(C); P = S[T] / C[T, None]
        if task == "class":
            pred, true = P.argmax(1), y[T]
            # accuracy modulo the dihedral relabelings of a circular K-class content:
            # does the decoder know the structure but not the frame?
            K = out_dim; best = 0.0
            for k in range(K):
                for sgn in (1, -1):
                    best = max(best, float(((sgn * pred + k) % K == true).mean()))
            return dict(acc=float((pred == true).mean()), acc_modD=best)
        Y = (y[T] - mu) / sd; r2 = 1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
        return dict(r2=float(r2.mean()), r2_dims=r2.tolist())

    hist, t0 = [], time.time()
    for ep in range(epochs):
        tot = 0.0
        for _ in range(pops_per_epoch // batch):
            idx, G = tr.batch(batch)
            out = model(G).reshape(-1, out_dim); it = torch.as_tensor(idx.reshape(-1), device=device)
            loss = loss_fn(out, yt[it]) if task == "class" else loss_fn(out, yt[it], it)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item()
        if str(device) == "mps":
            torch.mps.empty_cache()          # the MPS caching allocator otherwise grows across epochs
        m = evaluate(); m["loss"] = tot / (pops_per_epoch // batch); m["t"] = time.time() - t0
        if se is not None:
            ms = evaluate(sampler=se, pool_idx=sel_idx, score_only=sel_idx); m["sel"] = ms.get("acc", ms.get("r2"))
        hist.append(m)
        if verbose:
            print(f"    ep{ep} loss={m['loss']:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in m.items() if k in ("acc", "r2", "sel")) + f" ({m['t']:.0f}s)", flush=True)
    final = dict(hist[-1])
    if se is not None:                       # early stopping: report the epoch best on the selection set
        key = "acc" if task == "class" else "r2"
        b = int(np.argmax([h["sel"] for h in hist])); final.update({key: hist[b][key], "best_epoch": b, "sel_metric": hist[b]["sel"]})
        final["val_at_last_epoch"] = hist[-1][key]
    for reps in (8, 32):
        m = evaluate(reps); final[f"acc_avg{reps}" if task == "class" else f"r2_avg{reps}"] = m.get("acc", m.get("r2"))
    if verbose:
        print("    test-time averaging: " + " ".join(f"{k}={v:.3f}" for k, v in final.items() if "avg" in k), flush=True)
    final.update(history=hist, n=n, dim=dim, layers=layers, rel_bias=rel_bias, row_proj=row_proj,
                                         params=nparam, pops_per_epoch=pops_per_epoch, epochs=epochs, batch=batch)
    return final
