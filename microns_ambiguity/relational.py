"""Relational structure: Gram matrices and class-level Grams."""
from __future__ import annotations

import numpy as np


def cosine_gram(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, np.float32)
    nrm = np.linalg.norm(F, axis=1, keepdims=True)
    nrm[nrm == 0] = 1
    Fn = F / nrm
    return Fn @ Fn.T


def soma_kernel(S: np.ndarray, sigma_um: float = 100.0) -> np.ndarray:
    d2 = ((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * sigma_um ** 2)).astype(np.float32)


def class_gram(G: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    """K x K matrix of mean pairwise relation between classes (diagonal:
    within-class mean over distinct pairs)."""
    C = np.zeros((len(labels), K), np.float32)
    C[np.arange(len(labels)), labels] = 1
    S = C.T @ G @ C
    cnt = C.sum(0)
    N = np.outer(cnt, cnt)
    np.fill_diagonal(N, cnt * (cnt - 1))
    np.fill_diagonal(S, np.diag(S) - (np.diag(G) @ C))
    N[N == 0] = 1
    return S / N


def bin_orientation(theta_deg: np.ndarray, K: int) -> np.ndarray:
    """Bins centred on multiples of 180/K so that the reflection theta -> -theta
    maps bin c to (-c) mod K."""
    w = 180.0 / K
    return (np.floor((theta_deg + w / 2) / w).astype(int)) % K


def bin_grid(xy: np.ndarray, grid=(3, 3)) -> np.ndarray:
    """Quantile grid bins over 2-D content. bin = ix * ny + iy."""
    nx, ny = grid
    qx = np.quantile(xy[:, 0], np.linspace(0, 1, nx + 1)[1:-1])
    qy = np.quantile(xy[:, 1], np.linspace(0, 1, ny + 1)[1:-1])
    ix = np.searchsorted(qx, xy[:, 0])
    iy = np.searchsorted(qy, xy[:, 1])
    return ix * ny + iy


def dihedral_group(K: int):
    """All relabelings of K circular bins under rotation and reflection."""
    base = np.arange(K)
    out = []
    for k in range(K):
        out.append((base + k) % K)
        out.append((-base + k) % K)
    return np.unique(np.array(out), axis=0)


def grid_group(grid=(3, 3)):
    """Relabelings of an nx x ny grid under x-flip, y-flip and (if square)
    transpose: the symmetries a translation-invariant, isotropic wiring rule
    would leave unresolved."""
    nx, ny = grid
    idx = np.arange(nx * ny).reshape(nx, ny)
    mats = [idx, idx[::-1], idx[:, ::-1], idx[::-1, ::-1]]
    if nx == ny:
        mats += [m.T for m in mats]
    return np.unique(np.array([m.ravel() for m in mats]), axis=0)


def label_classes(values: np.ndarray, names: list[str]) -> np.ndarray:
    lut = {n: i for i, n in enumerate(names)}
    return np.array([lut[v] for v in values])
