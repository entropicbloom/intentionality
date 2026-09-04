"""Class-level geometric matching (the paper's Gram-matching protocol).

Reference and test are disjoint stratified halves of the neuron population.
The K x K class-Gram of the reference carries the class identities; the test
class-Gram is presented with class identities hidden and we search all K!
relabelings for the one closest in Frobenius distance.

Beyond the paper's exact-match accuracy we also compute
 * a posterior over relabelings, p(g) ∝ exp(-d_g^2 / 2 tau^2), with tau set to
   the typical distance of the *correct* relabeling across splits (split-half
   noise).  Its entropy is an estimate of H(I | R, C) over relabelings, and its
   per-class marginals give the ambiguity of each class' identity.
 * accuracy modulo a symmetry group (rotations/reflections of the orientation
   circle; flips of the RF grid) - the relabelings a difference-only wiring
   rule can never resolve.
"""
from __future__ import annotations

from itertools import permutations
from math import lgamma

import numpy as np

from .relational import class_gram

_PERM_CACHE: dict[int, np.ndarray] = {}


def all_perms(K: int) -> np.ndarray:
    if K not in _PERM_CACHE:
        _PERM_CACHE[K] = np.array(list(permutations(range(K))), dtype=np.int16)
    return _PERM_CACHE[K]


def argmin_tiebreak(D: np.ndarray, rng: np.random.Generator) -> int:
    """argmin with random choice among exact ties (a circulant class-Gram is
    invariant under every rotation; np.argmin would silently pick the identity,
    which is listed first)."""
    cands = np.flatnonzero(D <= D.min() + 1e-6 * max(1.0, float(D.min())))
    return int(rng.choice(cands))


def perm_distances(M_ref: np.ndarray, M_test: np.ndarray) -> np.ndarray:
    """Frobenius distance of M_ref to every relabeling of M_test.
    perm[c] = reference class assigned to test class c."""
    K = M_ref.shape[0]
    P = all_perms(K).astype(np.int64)
    # relabeled test matrix R with R[perm[c], perm[d]] = M_test[c, d]
    inv = np.argsort(P, axis=1)                      # inv[perm[c]] = c
    T = M_test[inv[:, :, None], inv[:, None, :]]      # (K!, K, K)
    D = np.sqrt(((T - M_ref[None]) ** 2).sum((1, 2)))
    return D


def stratified_half_split(labels: np.ndarray, rng: np.random.Generator):
    A, B = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        h = len(idx) // 2
        A.append(idx[:h]); B.append(idx[h:])
    return np.concatenate(A), np.concatenate(B)


def match_once(G: np.ndarray, labels: np.ndarray, K: int, rng: np.random.Generator,
               shuffle_within_halves=False):
    A, B = stratified_half_split(labels, rng)
    lA, lB = labels[A], labels[B]
    if shuffle_within_halves:      # pure-chance null: class = unrelated random groups in A and B
        lA, lB = rng.permutation(lA), rng.permutation(lB)
    M_A = class_gram(G[np.ix_(A, A)], lA, K)
    M_B = class_gram(G[np.ix_(B, B)], lB, K)
    D = perm_distances(M_A, M_B)
    P = all_perms(K)
    ident = np.arange(K)
    i_id = int(np.flatnonzero((P == ident).all(1))[0])
    best = argmin_tiebreak(D, rng)
    return dict(perm=P[best].astype(int), D=D, i_id=i_id, d_id=float(D[i_id]),
                d_best=float(D[best]), rank_id=int((D < D[i_id]).sum()))


def acc(perm, K):
    return float((perm == np.arange(K)).mean())


def acc_modulo(perm, group: np.ndarray | None):
    K = len(perm)
    if group is None:
        return acc(perm, K)
    return max(float((g[perm] == np.arange(K)).mean()) for g in group)


def posterior_entropy(D: np.ndarray, tau: float, K: int):
    """Entropy (bits) of p(g) ∝ exp(-D^2/(2 tau^2)) over all K! relabelings, and
    the mean per-class marginal entropy (bits) of 'which reference class does
    test class c map to'."""
    logp = -(D.astype(np.float64) ** 2) / (2 * tau ** 2)
    logp -= logp.max()
    p = np.exp(logp); p /= p.sum()
    H = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    P = all_perms(K)
    marg = np.zeros((K, K))
    for c in range(K):
        np.add.at(marg[c], P[:, c].astype(int), p)
    Hc = -(np.where(marg > 0, marg * np.log2(np.where(marg > 0, marg, 1)), 0)).sum(1)
    return H, float(Hc.mean()), marg


def log2_factorial(K):
    return lgamma(K + 1) / np.log(2)


def run_matching(G, labels, K, n_splits, seed, group=None, tau=None, shuffle_within_halves=False):
    """Full protocol over random splits. Returns summary dict."""
    rng = np.random.default_rng(seed)
    runs = [match_once(G, labels, K, rng, shuffle_within_halves) for _ in range(n_splits)]
    accs = np.array([acc(r["perm"], K) for r in runs])
    accs_mod = np.array([acc_modulo(r["perm"], group) for r in runs])
    hits = np.array([r["rank_id"] == 0 for r in runs], float)
    d_id = np.array([r["d_id"] for r in runs])
    if tau is None:
        # per-entry noise of the difference of two independent class-Gram
        # estimates: E[d_id^2] = K^2 * tau^2 under iid Gaussian entry noise
        tau = float(np.sqrt((d_id ** 2).mean() / K ** 2))
    Hs, Hcs = [], []
    marg = np.zeros((K, K))
    for r in runs:
        H, Hc, m = posterior_entropy(r["D"], tau, K)
        Hs.append(H); Hcs.append(Hc); marg += m / n_splits
    Hs, Hcs = np.array(Hs), np.array(Hcs)
    return dict(
        K=K, n_splits=n_splits, tau=tau,
        acc=float(accs.mean()), acc_sd=float(accs.std()),
        acc_mod=float(accs_mod.mean()), acc_mod_sd=float(accs_mod.std()),
        hit=float(hits.mean()),
        rank_id_median=float(np.median([r["rank_id"] for r in runs])),
        H_perm_bits=float(Hs.mean()), H_perm_max=log2_factorial(K),
        H_class_bits=float(Hcs.mean()), H_class_max=float(np.log2(K)),
        ars_posterior=float(1 - Hcs.mean() / np.log2(K)),
        marginal=marg.tolist(),
        confusion=np.mean([np.eye(K)[r["perm"]] for r in runs], 0).tolist(),
        example=_example(runs[0]),
    )


def _example(r):
    D = r["D"]
    hist, edges = np.histogram(D, bins=80)
    return dict(hist=hist.tolist(), edges=edges.tolist(), d_id=r["d_id"], d_best=r["d_best"],
                rank_id=r["rank_id"], n_perms=int(len(D)))
