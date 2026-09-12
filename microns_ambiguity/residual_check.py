"""Is the non-circulant residual of the orientation class-Gram signal?
(a) reproducibility across random neuron halves (same stimulus: cannot rule out a stimulus-sample origin),
(b) reproducibility across disjoint halves of the stimulus bins (rules it out),
(c) size of the residual on class-balanced subsamples (rules out neuron-count bias as its origin).
Usage: python -m microns_ambiguity.residual_check"""
import numpy as np

from .config import K_ORI
from .data import Dataset


def class_gram(Fn, cls, K):
    G = Fn @ Fn.T
    return np.array([[G[np.ix_(cls == i, cls == j)][~np.eye((cls == i).sum(), (cls == j).sum(), dtype=bool)].mean() if i == j else G[np.ix_(cls == i, cls == j)].mean() for j in range(K)] for i in range(K)])


def circulant_part(C):
    K = len(C); out = np.zeros_like(C)
    for d in range(K):
        vals = [C[i, (i + d) % K] for i in range(K)]; m = np.mean(vals)
        for i in range(K): out[i, (i + d) % K] = m
    return (out + out.T) / 2


def normalise(F):
    F = F - F.mean(1, keepdims=True); return F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)


def resid(Fn, cls, K):
    C = class_gram(Fn, cls, K); R = C - circulant_part(C); return C, R


def corr(a, b):
    m = ~np.eye(len(a), dtype=bool); return np.corrcoef(a[m], b[m])[0, 1]


def main():
    ds = Dataset(); K = K_ORI; ok = ds.ori_ok; ang = ds.ori[ok] % 180
    cls = ((ang + 90 / K) // (180 / K) % K).astype(int)
    for name, F in [("in vivo", ds.func_iv[ok]), ("twin", ds.func_is[ok])]:
        Fn = normalise(F.astype(np.float64)); C, R = resid(Fn, cls, K)
        frac = (R ** 2).sum() / ((C - C.mean()) ** 2).sum()
        # (a) neuron halves
        ra = []
        for seed in range(10):
            rng = np.random.default_rng(seed); p = rng.permutation(len(cls)); h = len(p) // 2
            _, R1 = resid(Fn[p[:h]], cls[p[:h]], K); _, R2 = resid(Fn[p[h:]], cls[p[h:]], K); ra.append(corr(R1, R2))
        # (b) disjoint stimulus halves (all neurons)
        rb = []
        for seed in range(10):
            rng = np.random.default_rng(100 + seed); q = rng.permutation(F.shape[1]); h = F.shape[1] // 2
            _, R1 = resid(normalise(F[:, q[:h]].astype(np.float64)), cls, K); _, R2 = resid(normalise(F[:, q[h:2 * h]].astype(np.float64)), cls, K); rb.append(corr(R1, R2))
        # (c) class-balanced subsamples
        rc, fc = [], []
        m = np.bincount(cls).min()
        for seed in range(10):
            rng = np.random.default_rng(200 + seed); idx = np.concatenate([rng.choice(np.flatnonzero(cls == k), m, replace=False) for k in range(K)])
            Cb, Rb = resid(Fn[idx], cls[idx], K); fc.append((Rb ** 2).sum() / ((Cb - Cb.mean()) ** 2).sum()); rc.append(corr(Rb, R))
        nb = np.array([C[i, (i + 1) % K] for i in range(K)])
        print(f"{name}: residual = {frac:.3f} of class-Gram variance; neighbour corr by class {np.round(nb, 3)}")
        print(f"  (a) residual corr between neuron halves      : {np.mean(ra):.2f} ± {np.std(ra):.2f}")
        print(f"  (b) residual corr between stimulus-bin halves: {np.mean(rb):.2f} ± {np.std(rb):.2f}  ({F.shape[1] // 2} bins per half)")
        print(f"  (c) class-balanced ({m}/class): residual fraction {np.mean(fc):.3f}, corr with full residual {np.mean(rc):.2f}")


if __name__ == "__main__":
    main()
