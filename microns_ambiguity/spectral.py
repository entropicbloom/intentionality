"""Protocol 3: reference-free recovery.  No other population, no labelled
reference: the Gram of the test population alone is embedded (kernel PCA) and
we ask how much of the content is present up to a similarity transform
(rotation/reflection/scale of the top-2 axes: the automorphisms a relational
structure cannot fix) or up to a linear readout of the top-m axes."""
from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import RidgeCV


def kernel_pca(G: np.ndarray, m: int):
    n = G.shape[0]
    H = np.eye(n, dtype=np.float32) - 1.0 / n
    Gc = H @ G @ H
    w, V = np.linalg.eigh(Gc.astype(np.float64))
    order = np.argsort(w)[::-1][:m]
    return V[:, order] * np.sqrt(np.clip(w[order], 0, None)), w[order]


def r2(pred, y):
    ss = ((y - y.mean(0)) ** 2).sum(0)
    return float((1 - ((pred - y) ** 2).sum(0) / ss).mean())


def procrustes_r2(E2, y, train, test):
    """Fit rotation/reflection + scale + shift on train neurons, score on test."""
    mu_e, mu_y = E2[train].mean(0), y[train].mean(0)
    A, B = E2[train] - mu_e, y[train] - mu_y
    R, s = orthogonal_procrustes(A, B)
    scale = s / (A ** 2).sum()
    pred = (E2[test] - mu_e) @ R * scale + mu_y
    return r2(pred, y[test])


def linear_r2(E, y, train, test):
    m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(E[train], y[train])
    return r2(m.predict(E[test]), y[test])


def evaluate(G, y, ms=(2, 10, 50), n_rep=10, seed=0):
    """y: (n, d) continuous content. Returns dict of R2 for procrustes(top-2),
    linear(top-m), and their shuffled-label nulls."""
    rng = np.random.default_rng(seed)
    E, w = kernel_pca(G, max(ms))
    out = {f"linear_m{m}": [] for m in ms}
    out.update(procrustes_top2=[], null_procrustes_top2=[], null_linear_m50=[])
    n = len(y)
    for _ in range(n_rep):
        perm = rng.permutation(n); train, test = perm[: n // 2], perm[n // 2:]
        out["procrustes_top2"].append(procrustes_r2(E[:, :2], y, train, test))
        for m in ms:
            out[f"linear_m{m}"].append(linear_r2(E[:, :m], y, train, test))
        ys = y[rng.permutation(n)]
        out["null_procrustes_top2"].append(procrustes_r2(E[:, :2], ys, train, test))
        out["null_linear_m50"].append(linear_r2(E[:, :max(ms)], ys, train, test))
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res.update({k + "_sd": float(np.std(v)) for k, v in out.items()})
    res["eigen_top"] = (w[:10] / w.sum()).tolist()
    return res
