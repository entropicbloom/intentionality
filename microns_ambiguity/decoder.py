"""Protocol 2: learned relational decoder with hidden population labels.

Mirrors the paper's transformer decoder: a sampled population of n neurons is
presented as its n x n relational (Gram) matrix, rows as tokens, no positional
encoding; the target is token 0; the decoder predicts the target's content.
Training populations are drawn from one half of the neurons, validation
populations from the other half, so the decoder must extract population-
geometry regularities that transfer to unseen neurons.

'target_only' ablation: all pairwise relations not involving the target are
removed (only row/column 0 and the diagonal survive), the paper's local-vs-
global control.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import (DEC_BATCH, DEC_DIM, DEC_EPOCHS, DEC_HEADS, DEC_LAYERS, DEC_LR,
                     DEC_POP, DEC_TRAIN_SAMPLES, DEC_VAL_SAMPLES)


def get_device():
    # MPS was ~2.5x slower than CPU for this small model on an M3; stay on CPU.
    return torch.device("cpu")


class GramDecoder(nn.Module):
    def __init__(self, n_tokens, out_dim, dim=DEC_DIM, heads=DEC_HEADS, layers=DEC_LAYERS):
        super().__init__()
        self.inp = nn.Linear(n_tokens, dim)
        self.is_target = nn.Parameter(torch.zeros(1, 1, dim))
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))

    def forward(self, G):                     # G: (B, n, n)
        h = self.inp(G)
        h = torch.cat([h[:, :1] + self.is_target, h[:, 1:]], 1)
        h = self.enc(h)
        return self.head(h[:, 0])


class PopulationSampler:
    def __init__(self, G: torch.Tensor, pool: np.ndarray, n: int, rng: np.random.Generator,
                 target_only=False):
        self.G, self.pool, self.n, self.rng, self.target_only = G, pool, n, rng, target_only

    def batch(self, B: int, targets: np.ndarray | None = None):
        if targets is None:
            targets = self.rng.choice(self.pool, B)
        others = np.stack([self.rng.choice(self.pool[self.pool != t], self.n - 1, replace=False)
                           for t in targets])
        idx = np.concatenate([targets[:, None], others], 1)
        idx_t = torch.as_tensor(idx, device=self.G.device)
        sub = self.G[idx_t[:, :, None], idx_t[:, None, :]]
        if self.target_only:
            mask = torch.zeros(self.n, self.n, device=sub.device, dtype=torch.bool)
            mask[0, :] = True; mask[:, 0] = True
            mask |= torch.eye(self.n, device=sub.device, dtype=torch.bool)
            sub = sub * mask
        return targets, sub


def train_decoder(G: np.ndarray, y: np.ndarray, task: str, train_idx, val_idx, seed=0,
                  n=DEC_POP, epochs=DEC_EPOCHS, train_samples=DEC_TRAIN_SAMPLES,
                  val_samples=DEC_VAL_SAMPLES, target_only=False, device=None, verbose=True):
    """task: 'class' (y int labels) or 'reg' (y float, (N, d)). Returns metrics."""
    device = device or get_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Gt = torch.as_tensor(np.ascontiguousarray(G, dtype=np.float32), device=device)
    if task == "class":
        K = int(y.max()) + 1
        yt = torch.as_tensor(y, device=device, dtype=torch.long)
        model = GramDecoder(n, K).to(device)
        loss_fn = nn.CrossEntropyLoss()
    else:
        y = np.atleast_2d(y.T).T.astype(np.float32)
        mu, sd = y[train_idx].mean(0), y[train_idx].std(0)
        yt = torch.as_tensor((y - mu) / sd, device=device)
        model = GramDecoder(n, y.shape[1]).to(device)
        loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=DEC_LR, weight_decay=1e-4)
    steps = epochs * (train_samples // DEC_BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, DEC_LR, total_steps=steps, pct_start=0.1)
    tr = PopulationSampler(Gt, np.asarray(train_idx), n, rng, target_only)
    va = PopulationSampler(Gt, np.asarray(val_idx), n, np.random.default_rng(seed + 100), target_only)

    def evaluate():
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for _ in range(val_samples // DEC_BATCH):
                t, sub = va.batch(DEC_BATCH)
                preds.append(model(sub).cpu().numpy()); trues.append(t)
        model.train()
        P = np.concatenate(preds); T = np.concatenate(trues)
        if task == "class":
            return dict(acc=float((P.argmax(1) == y[T]).mean()))
        Y = ((y[T] - mu) / sd)
        r2 = 1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
        return dict(r2=float(r2.mean()), r2_dims=r2.tolist())

    hist = []
    for ep in range(epochs):
        tot = 0.0
        for _ in range(train_samples // DEC_BATCH):
            t, sub = tr.batch(DEC_BATCH)
            out = model(sub)
            loss = loss_fn(out, yt[torch.as_tensor(t, device=device)])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        m = evaluate(); m["loss"] = tot / (train_samples // DEC_BATCH); hist.append(m)
        if verbose:
            print(f"    ep{ep} loss={m['loss']:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in m.items() if k not in ('loss', 'r2_dims')), flush=True)
    final = dict(hist[-1])
    final["history"] = [dict(h) for h in hist]
    return final
