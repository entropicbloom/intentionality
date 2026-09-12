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
    def __init__(self, n_tokens, out_dim, dim=128, heads=4, layers=2, rel_bias=False, row_proj=True, dropout=0.1, input_mode="gram", act_dim=0):
        super().__init__()
        self.row_proj, self.input_mode = row_proj, input_mode
        # token init: "gram": projected Gram row (+ 3 permutation-invariant row statistics);
        # "act": the neuron's own (row-normalised) response vector, the raw-activity reference (no Gram in the input)
        self.inp = nn.Linear(act_dim if input_mode == "act" else (n_tokens if row_proj else 0) + 3, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, rel_bias, dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))

    def forward(self, G, X=None):                          # G: (B, n, n) standardized; X: (B, n, d) row-normalised responses
        if self.input_mode == "act":
            x = self.inp(X * math.sqrt(X.shape[-1]))       # unit-norm rows scaled to unit variance per feature
        else:
            stats = torch.stack([G.mean(-1), G.std(-1), (G ** 3).mean(-1)], -1)
            x = torch.cat([G, stats], -1) if self.row_proj else stats
            x = self.inp(x)
        for b in self.blocks:
            x = b(x, G)
        return self.head(x)                                 # (B, n, out)


# label-free augmentation of the relations, applied to TRAINING populations only
# (train() switches it on around its gradient steps and off for every evaluation):
#   cond_frac < 1: each population's Gram is recomputed from a random subset of the
#                  feature dimensions (stimulus conditions / time bins), rows re-standardised;
#   gram_drop > 0: random Gram entries are zeroed (zero = the mean correlation).
AUG = dict(cond_frac=1.0, gram_drop=0.0, aug_prob=1.0, active=False)   # aug_prob: fraction of training populations augmented
BINPERM = dict(on=False)   # bin_perm: permute the stimulus bins of every population (same permutation for all its neurons),
                           # at training and test; keeps every bin-permutation-invariant statistic, destroys stimulus alignment


def permute_bins(X):
    if not BINPERM["on"]: return X
    B, n, d = X.shape; order = torch.rand(B, d, device=X.device).argsort(1)
    return torch.gather(X, 2, order.unsqueeze(1).expand(B, n, d))


def gram_from_features(X, mean, std, rng=None):
    """X: (B, n, d) row-normalised features -> standardised Gram (B, n, n), diagonal zeroed."""
    if AUG["active"] and AUG["cond_frac"] < 1:
        B, n, d = X.shape; k = max(2, int(round(d * AUG["cond_frac"])))
        keep = torch.zeros(B, 1, d, device=X.device)
        cols = torch.rand(B, d, device=X.device).argsort(1)[:, :k]
        keep.scatter_(2, cols.unsqueeze(1), 1.0)
        if AUG["aug_prob"] < 1:                                   # un-augmented populations keep every condition
            full = (torch.rand(B, 1, 1, device=X.device) >= AUG["aug_prob"]).float()
            keep = torch.maximum(keep, full); k = keep.sum(-1, keepdim=True)
        X = X * keep; X = X - X.sum(-1, keepdim=True) / k * keep
        X = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    G = torch.bmm(X, X.transpose(1, 2))
    G = (G - mean) / std
    if AUG["active"] and AUG["gram_drop"] > 0:
        G = G * (torch.rand_like(G) >= AUG["gram_drop"])
    G.diagonal(dim1=1, dim2=2).zero_()
    return G


class Sampler:
    """Populations of n neurons from a pool; Gram computed from normalized features."""
    def __init__(self, Fn: torch.Tensor, pool, n, rng, mean, std):
        self.Fn, self.pool, self.n, self.rng, self.mean, self.std = Fn, np.asarray(pool), n, rng, mean, std

    def batch(self, B):
        idx = np.stack([self.rng.choice(self.pool, self.n, replace=False) for _ in range(B)])
        it = torch.as_tensor(idx, device=self.Fn.device)
        X = self.Fn[it]                                     # (B, n, d)
        return idx, gram_from_features(X, self.mean, self.std), X


def normalize_features(F):
    F = np.asarray(F, np.float32); F = F - F.mean(1, keepdims=True)
    nrm = np.linalg.norm(F, axis=1, keepdims=True); nrm[nrm == 0] = 1
    return F / nrm


def train(F, y, task, train_idx, val_idx, n=128, dim=128, heads=4, layers=2, rel_bias=False, row_proj=True,
          epochs=10, pops_per_epoch=2000, batch=32, lr=1e-3, seed=0, device="cpu", verbose=True, val_pops=200,
          heads_=None, early_stop=0.0, dropout=0.1, sel_idx=None, sel_reps=1, avg_reps=(8, 32), return_preds=False, cover_groups=None, cond_frac=1.0, gram_drop=0.0, aug_prob=1.0, F_eval=None, input_mode="gram", label_rot=False, ori_weight=False, bin_perm=False, sel_modD=False):
    """F: (N, d) responses; y: labels (N,) int or (N, k) float. Dense supervision.
    early_stop: fraction of the TRAINING neurons held out as a selection set; the
    reported validation metric is taken at the epoch that is best on that set, so
    the validation neurons never influence model selection.
    sel_reps>1: selection metric averaged over sel_reps population covers (less noisy).
    The model state at the best selection epoch is restored before test-time averaging.
    return_preds: also return per-neuron averaged logits (max avg_reps) for the validation set.
    cover_groups: (N,) group id per neuron; if given, the averaged evaluation forms its
    populations within a group (e.g. within a mouse), matching a group-restricted Sampler.
    F_eval: optional (N, d') features used ONLY for the validation-set Grams (cross-stimulus
    test: training and selection Grams from F, test Grams from disjoint stimulus bins).
    input_mode: "gram" (default) or "act": tokens from each neuron's response vector instead of its
    Gram row, the raw-activity reference decoder; with rel_bias the Gram still enters attention,
    without it the Gram is not used at all; layers=0 makes it a linear per-neuron readout.
    label_rot (circ): each training population's labels are rotated by a random angle, so the
    labels carry no frame; selection and the reported error use the frame-corrected err_modD.
    ori_weight (circ): loss weight per labelled neuron = inverse density of its 15° label bin."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    AUG.update(cond_frac=cond_frac, gram_drop=gram_drop, aug_prob=aug_prob, active=False); BINPERM["on"] = bool(bin_perm)
    if sel_idx is not None:                     # caller-provided selection set (e.g. held-out mice)
        sel_idx = np.asarray(sel_idx); train_idx = np.setdiff1d(np.asarray(train_idx), sel_idx)
    elif early_stop > 0:
        train_idx = rng.permutation(np.asarray(train_idx)); k = int(len(train_idx) * early_stop)
        sel_idx, train_idx = train_idx[:k], train_idx[k:]
    Fn = torch.as_tensor(normalize_features(F), device=device)
    # global off-diagonal statistics of the correlation, from a random sample
    s = rng.choice(len(F), min(2000, len(F)), replace=False); Gs = Fn[s] @ Fn[s].T
    off = Gs[~torch.eye(len(s), dtype=torch.bool, device=device)]
    mean, std = float(off.mean()), float(off.std())
    Fn_ev, mean_ev, std_ev = Fn, mean, std
    if F_eval is not None:                      # standardisation statistics of the eval Grams from the eval features
        Fn_ev = torch.as_tensor(normalize_features(F_eval), device=device); Ge = Fn_ev[s] @ Fn_ev[s].T
        offe = Ge[~torch.eye(len(s), dtype=torch.bool, device=device)]; mean_ev, std_ev = float(offe.mean()), float(offe.std())
    ang = None
    if task == "circ":                          # orientation as a circular regression: target (cos 2θ, sin 2θ), θ in degrees mod 180
        ang = np.asarray(y, np.float64); th = np.deg2rad(2 * ang)
        y = np.c_[np.cos(th), np.sin(th)].astype(np.float32); y[~np.isfinite(ang)] = np.nan
    if task == "class":
        K = int(y.max()) + 1; yt = torch.as_tensor(y, device=device, dtype=torch.long); out_dim = K
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1); labelled = y >= 0
    elif task == "circ":
        labelled = np.isfinite(y).all(1); mu, sd = np.zeros(2, np.float32), np.ones(2, np.float32)
        yt = torch.as_tensor(np.nan_to_num(y), device=device); out_dim = 2
        lab_t = torch.as_tensor(labelled, device=device)
        w_np = np.ones(len(y), np.float32)
        if ori_weight:                          # inverse density of the label's 15° bin, over the labelled training neurons
            b = ((np.nan_to_num(ang) % 180) // 15).astype(int); cnt = np.bincount(b[np.asarray(train_idx)][labelled[np.asarray(train_idx)]], minlength=12).astype(float)
            w_np = (cnt.mean() / np.maximum(cnt, 1))[b].astype(np.float32)
        w_t = torch.as_tensor(w_np, device=device)
        def loss_fn(out, tgt, _idx=None, _B=None):
            m = lab_t[_idx]
            if not m.any(): return (out * 0).sum()
            if label_rot:                       # frame-free loss: align each population's predictions to its targets by the best
                B = _B; n_ = out.shape[0] // B  # rotation (and reflection) before the error, so only relative structure is learned
                o = out.view(B, n_, 2); t = tgt.view(B, n_, 2); mm = m.view(B, n_).float()
                best = None
                for sgn in (1.0, -1.0):
                    oc = torch.stack([o[..., 0], sgn * o[..., 1]], -1)                      # optional reflection
                    z_re = (mm * (t[..., 0] * oc[..., 0] + t[..., 1] * oc[..., 1])).sum(1); z_im = (mm * (t[..., 1] * oc[..., 0] - t[..., 0] * oc[..., 1])).sum(1)
                    a = torch.atan2(z_im, z_re).detach()[:, None]; c, s_ = torch.cos(a), torch.sin(a)
                    ro = torch.stack([c * oc[..., 0] - s_ * oc[..., 1], s_ * oc[..., 0] + c * oc[..., 1]], -1)
                    e = (((ro - t) ** 2).sum(-1) * mm).sum(1) / mm.sum(1).clamp_min(1)     # per-population error after alignment
                    best = e if best is None else torch.minimum(best, e)
                return best.mean()
            w = w_t[_idx][m]; return (((out[m] - tgt[m]) ** 2).mean(1) * w).sum() / w.sum()
    else:
        y = np.atleast_2d(y.T).T.astype(np.float32); labelled = np.isfinite(y).all(1)
        mu, sd = np.nanmean(y[train_idx], 0), np.nanstd(y[train_idx], 0)
        yt = torch.as_tensor(np.nan_to_num((y - mu) / sd), device=device); out_dim = y.shape[1]
        lab_t = torch.as_tensor(labelled, device=device)
        def loss_fn(out, tgt, _idx=None):
            m = lab_t[_idx]
            return ((out[m] - tgt[m]) ** 2).mean() if m.any() else (out * 0).sum()
    model = RelDecoder(n, out_dim, dim, heads, layers, rel_bias, row_proj, dropout=dropout, input_mode=input_mode, act_dim=Fn.shape[1]).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (pops_per_epoch // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    tr = Sampler(Fn, train_idx, n, rng, mean, std); va = Sampler(Fn_ev, val_idx, n, np.random.default_rng(seed + 100), mean_ev, std_ev)
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
        AUG["active"] = False; model.eval(); N = len(y); S = np.zeros((N, out_dim), np.float32); C = np.zeros(N)
        with torch.no_grad():
            if reps == 1:
                for _ in range(max(1, val_pops // batch)):
                    idx, G, X = sampler.batch(batch); out = model(G, permute_bins(X)).cpu().numpy()
                    np.add.at(S, idx.reshape(-1), out.reshape(-1, out_dim)); np.add.at(C, idx.reshape(-1), 1)
            else:
                pool = np.asarray(pool_idx); r = np.random.default_rng(seed + 7)
                subpools = [pool] if cover_groups is None else [pool[cover_groups[pool] == g] for g in np.unique(cover_groups[pool])]
                subpools = [p for p in subpools if len(p) >= n]
                for _ in range(reps):
                    perm = np.concatenate([r.permutation(p)[: (len(p) // n) * n].reshape(-1, n) for p in subpools])
                    perm = perm[r.permutation(len(perm))]
                    for b in range(0, len(perm), batch):
                        idx = perm[b:b + batch]; it = torch.as_tensor(idx, device=device); X = sampler.Fn[it]
                        out = model(gram_from_features(X, sampler.mean, sampler.std), permute_bins(X)).cpu().numpy()
                        np.add.at(S, idx.reshape(-1), out.reshape(-1, out_dim)); np.add.at(C, idx.reshape(-1), 1)
        model.train()
        if score_only is not None:
            keep = np.zeros(N, bool); keep[np.asarray(score_only)] = True; C = C * keep
        C = C * labelled                           # score labelled neurons only
        T = np.flatnonzero(C); P = S[T] / C[T, None]
        preds = dict(idx=T, P=P)
        if task == "class":
            pred, true = P.argmax(1), y[T]
            # accuracy modulo the dihedral relabelings of a circular K-class content:
            # does the decoder know the structure but not the frame?
            K = out_dim; best = 0.0
            for k in range(K):
                for sgn in (1, -1):
                    best = max(best, float(((sgn * pred + k) % K == true).mean()))
            return dict(acc=float((pred == true).mean()), acc_modD=best, preds=preds)
        if task == "circ":
            th_hat = np.rad2deg(np.arctan2(P[:, 1], P[:, 0])) / 2; th = ang[T]
            d = np.abs((th_hat - th + 90) % 180 - 90)                  # angular error in [0, 90]
            # frame check: error after the best global rotation (and reflection) of the predictions
            best = d.mean()
            for sgn in (1, -1):
                z = np.exp(1j * np.deg2rad(2 * (sgn * th_hat - th))); off = np.rad2deg(np.angle(z.mean())) / 2
                best = min(best, np.abs((sgn * th_hat - off - th + 90) % 180 - 90).mean())
            return dict(err=float(d.mean()), within15=float((d <= 15).mean()), err_modD=float(best), preds=preds)
        Y = (y[T] - mu) / sd; r2 = 1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
        return dict(r2=float(r2.mean()), r2_dims=r2.tolist(), preds=preds)

    hist, t0, best_state = [], time.time(), None
    for ep in range(epochs):
        tot = 0.0
        AUG["active"] = True
        for _ in range(pops_per_epoch // batch):
            idx, G, X = tr.batch(batch)
            out = model(G, permute_bins(X)).reshape(-1, out_dim); it = torch.as_tensor(idx.reshape(-1), device=device)
            tgt = yt[it]
            if label_rot and task == "circ":    # rotate each population's targets by a random angle (2φ on the (cos 2θ, sin 2θ) circle)
                phi = torch.rand(idx.shape[0], 1, device=device) * 2 * math.pi; c, s_ = torch.cos(phi), torch.sin(phi)
                t = tgt.view(idx.shape[0], -1, 2); tgt = torch.stack([c * t[..., 0] - s_ * t[..., 1], s_ * t[..., 0] + c * t[..., 1]], -1).reshape(-1, 2)
            loss = loss_fn(out, tgt) if task == "class" else (loss_fn(out, tgt, it, idx.shape[0]) if (task == "circ" and label_rot) else loss_fn(out, tgt, it))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item()
        AUG["active"] = False
        if str(device) == "mps":
            torch.mps.empty_cache()          # the MPS caching allocator otherwise grows across epochs
        m = evaluate(); m.pop("preds", None); m["loss"] = tot / (pops_per_epoch // batch); m["t"] = time.time() - t0
        if se is not None:
            ms = evaluate(reps=sel_reps, sampler=se, pool_idx=sel_idx, score_only=sel_idx); m["sel"] = -(ms["err_modD"] if (label_rot or sel_modD) else ms["err"]) if task == "circ" else ms.get("acc", ms.get("r2"))
            if best_state is None or m["sel"] > max(h["sel"] for h in hist):
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        hist.append(m)
        if verbose:
            print(f"    ep{ep} loss={m['loss']:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in m.items() if k in ("acc", "r2", "err", "within15", "sel")) + f" ({m['t']:.0f}s)", flush=True)
    final = dict(hist[-1]); final.pop("preds", None)
    if se is not None:                       # early stopping: report the epoch best on the selection set
        key = {"class": "acc", "circ": "err"}.get(task, "r2")
        b = int(np.argmax([h["sel"] for h in hist])); final.update({key: hist[b][key], "best_epoch": b, "sel_metric": hist[b]["sel"]})
        if task == "circ": final.update(within15=hist[b]["within15"], err_modD=hist[b]["err_modD"])
        final["val_at_last_epoch"] = hist[-1][key]
        model.load_state_dict(best_state)      # test-time averaging uses the selected model
    for reps in avg_reps:
        m = evaluate(reps); k_ = {"class": "acc", "circ": "err"}.get(task, "r2"); final[f"{k_}_avg{reps}"] = m[k_]
        if return_preds and reps == max(avg_reps):
            final["preds"] = m["preds"]
    if verbose:
        print("    test-time averaging: " + " ".join(f"{k}={v:.3f}" for k, v in final.items() if "avg" in k), flush=True)
    final.update(history=hist, n=n, dim=dim, layers=layers, rel_bias=rel_bias, row_proj=row_proj, input_mode=input_mode, label_rot=label_rot, ori_weight=ori_weight, bin_perm=bin_perm, sel_modD=sel_modD, cond_frac=cond_frac, gram_drop=gram_drop, aug_prob=aug_prob,
                 heads=heads, dropout=dropout, lr=lr, seed=seed, early_stop=early_stop, sel_reps=sel_reps, n_sel=(int(len(sel_idx)) if sel_idx is not None else 0),
                                         params=nparam, pops_per_epoch=pops_per_epoch, epochs=epochs, batch=batch)
    return final
