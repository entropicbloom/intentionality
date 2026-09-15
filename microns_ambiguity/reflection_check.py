"""Reflection check (2026-09-15): does the residual anisotropy fix the rotation only, or the reflection too?

Three measurements, none of which train anything:
  1. decoder outputs by true class on the saved 17M predictions: under an unresolved reflection about the
     cardinal axes the two candidate outputs for an oblique neuron are opposite unit vectors, so the MSE-optimal
     output is the zero vector and the angle falls on a cardinal;
  2. the reflection-antisymmetric share of the class-Gram residual on cortex (only that share could fix the reflection);
  3. the source of the residual in the synthetic local-field model: systematic (the weight peaked at 90 degrees) or
     the shared random draw of the field, measured by comparing residuals across neuron halves with the same or
     disjoint frames, with the peaked or a uniform weight.
Run: .venv/bin/python -m microns_ambiguity.reflection_check
"""
from __future__ import annotations

import numpy as np

K = 8


def near(a, c, w=180 / (2 * K)):
    return np.abs(((a - c + 90) % 180) - 90) < w


def classgram(R, pref, idx):
    Rz = R[idx] - R[idx].mean(1, keepdims=True)
    Rz /= np.linalg.norm(Rz, axis=1, keepdims=True)
    G = Rz @ Rz.T
    cls = ((pref[idx] + 90 / K) // (180 / K)).astype(int) % K
    C = np.zeros((K, K))
    for a in range(K):
        for b in range(K):
            m = np.outer(cls == a, cls == b)
            if a == b:
                m &= ~np.eye(len(idx), dtype=bool)
            C[a, b] = G[m].mean()
    return C


def resid(C):
    circ = np.zeros_like(C)
    for d in range(K):
        circ += np.mean([C[a, (a + d) % K] for a in range(K)]) * np.roll(np.eye(K), d, axis=1)
    return C - circ


def reflect(M):                       # theta -> 180 - theta, i.e. class k -> -k mod K (fixes 0 and 90 degrees)
    j = [(-a) % K for a in range(K)]
    return M[np.ix_(j, j)]


def outputs_by_class(tag):
    z = np.load(f"microns_ambiguity/outputs/preds/{tag}.npz")
    P, y = z["P"], z["y"]
    ang = (0.5 * np.degrees(np.arctan2(P[:, 1], P[:, 0]))) % 180
    print(tag)
    for c in (0, 45, 90, 135):
        m = near(y, c)
        frac = " ".join(f"{q}:{near(ang[m], q).mean():.2f}" for q in (0, 45, 90, 135))
        print(f"  true {c:3d} n={m.sum():4d} | predicted near {frac} | mean output ({P[m, 0].mean():+.2f}, {P[m, 1].mean():+.2f})")


def cortex_residual():
    from .data import Dataset
    ds = Dataset()
    for name, R in (("twin", ds.func_is), ("in vivo", ds.func_iv)):
        for area in ("V1", "all"):
            idx = np.where(ds.ori_ok & ((ds.area == area) if area != "all" else ds.area_ok))[0]
            C = classgram(R.astype(np.float64), ds.ori, idx)
            r = resid(C)
            anti = (r - reflect(r)) / 2
            print(f"{name:8s} {area:4s} n={len(idx):5d} residual fraction {(r ** 2).sum() / ((C - C.mean()) ** 2).sum():.2f}"
                  f"  reflection-antisymmetric share {(anti ** 2).sum() / (r ** 2).sum():.2f}")


def synthetic_residual(k=1.0, seed=0):
    from . import synthetic as S

    def halves(R, pref, split_frames):
        perm = np.random.default_rng(seed).permutation(len(pref)); h = len(pref) // 2
        Ra, Rb = (R[:, : R.shape[1] // 2], R[:, R.shape[1] // 2:]) if split_frames else (R, R)
        Ca, Cb = classgram(Ra, pref, perm[:h]), classgram(Rb, pref, perm[h:])
        ra, rb = resid(Ca), resid(Cb)
        return (ra ** 2).sum() / ((Ca - Ca.mean()) ** 2).sum(), np.corrcoef(ra.ravel(), rb.ravel())[0, 1]

    for flat in (0, 1):
        R, pref = S.make(n=6000, T=120, cocorr=2, cocorr_k=k, seed=seed, cocorr_flat=flat)
        for split in (False, True):
            f, c = halves(R, pref, split)
            print(f"local field k={k} weight={'uniform' if flat else 'peaked at 90'} frames={'disjoint' if split else 'shared'}:"
                  f" residual fraction {f:.3f}, corr(residual train half, residual test half) {c:+.2f}")


if __name__ == "__main__":
    for t in ("c_is_17M_sp1", "c_iv_17M_sp1"):
        outputs_by_class(t)
    cortex_residual()
    synthetic_residual()
