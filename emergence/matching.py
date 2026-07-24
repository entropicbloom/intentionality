"""Label-free identifiability from relational (Gram) structure.

Given two models' concept vectors for the same word set, ask whether token
identity can be recovered from the cosine-Gram matrices alone — no labels,
only relational structure. Two readouts:

  - subset_match: exhaustive permutation matching on random size-k sub-Grams
    (the robust headline metric; chance = 1/k).
  - profile_match: a single global permutation over all K tokens via
    sorted-similarity-profile Hungarian init + iterative refinement
    (harder, noisier; chance = 1/K).
  - cka: centered-kernel-alignment scalar between the two Grams (the standard
    aggregate-similarity metric, for contrast).
"""

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment


def gram(j):
    """K x K cosine-similarity (Gram) matrix of the row vectors in j."""
    n = j / (np.linalg.norm(j, axis=1, keepdims=True) + 1e-12)
    return n @ n.T


def cka(g_a, g_b):
    """Centered kernel alignment between two Gram matrices."""
    k = g_a.shape[0]
    h = np.eye(k) - np.ones((k, k)) / k
    ga, gb = h @ g_a @ h, h @ g_b @ h
    return float((ga * gb).sum() / (np.linalg.norm(ga) * np.linalg.norm(gb) + 1e-12))


def _refine(g_a, g_b, perm, n_refine=100):
    """Iterate Hungarian on the linearized quadratic objective until fixpoint."""
    k = g_a.shape[0]
    for _ in range(n_refine):
        rows, cols = linear_sum_assignment(-(g_a @ g_b[:, perm].T))
        new_perm = np.empty(k, dtype=int)
        new_perm[rows] = cols
        if (new_perm == perm).all():
            break
        perm = new_perm
    return perm


def profile_match(g_a, g_b, n_refine=100):
    """Full-K label-free matching. Returns (accuracy, perm)."""
    sig_a = np.sort(g_a, axis=1)
    sig_b = np.sort(g_b, axis=1)
    cost = ((sig_a[:, None, :] - sig_b[None, :, :]) ** 2).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(g_a.shape[0], dtype=int)
    perm[rows] = cols
    perm = _refine(g_a, g_b, perm, n_refine)
    acc = float((perm == np.arange(len(perm))).mean())
    return acc, perm


def subset_match(g_a, g_b, rng, subset_size=8, n_subsets=200):
    """Exhaustive permutation matching on random subset_size-token sub-Grams.
    Returns (mean per-token accuracy, exact-hit rate). Chance = 1/subset_size."""
    k = g_a.shape[0]
    perms = np.array(list(itertools.permutations(range(subset_size))))
    accs, hits = [], 0
    for _ in range(n_subsets):
        idx = rng.choice(k, size=subset_size, replace=False)
        a = g_a[np.ix_(idx, idx)]
        b = g_b[np.ix_(idx, idx)]
        bp = b[perms[:, :, None], perms[:, None, :]]
        dists = ((a[None] - bp) ** 2).sum(axis=(1, 2))
        best = perms[dists.argmin()]
        accs.append((best == np.arange(subset_size)).mean())
        hits += int((best == np.arange(subset_size)).all())
    return float(np.mean(accs)), hits / n_subsets
