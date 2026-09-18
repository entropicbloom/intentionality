"""Same-scan versus cross-scan correlation at matched receptive-field distance (Methods, MICrONS paragraph).

6,000 random receptive-field-labelled neurons; all pairs binned into deciles of receptive-field-centre
distance; per decile, the mean correlation of same-scan pairs divided by that of cross-scan pairs.
The ratio is reported over the six nearest deciles, where both means are clearly positive; at larger
distances the correlations cross zero and the ratio is undefined.
Run: .venv/bin/python -m microns_ambiguity.scan_effect
"""
from __future__ import annotations

import numpy as np


def main(n=6000, seed=0):
    from .data import Dataset
    ds = Dataset()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.where(ds.rf_ok)[0], n, replace=False))
    rf, scan = ds.rf[idx], ds.scan[idx]
    D = np.linalg.norm(rf[:, None, :] - rf[None, :, :], axis=-1)
    iu = np.triu_indices(n, 1)
    d, same = D[iu], (scan[:, None] == scan[None, :])[iu]
    edges = np.quantile(d, np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, 9)
    for name, R in (("in vivo", ds.func_iv), ("twin", ds.func_is)):
        Rz = R[idx].astype(np.float64)
        Rz -= Rz.mean(1, keepdims=True)
        Rz /= np.linalg.norm(Rz, axis=1, keepdims=True)
        c = (Rz @ Rz.T)[iu]
        ratio = [c[(b == k) & same].mean() / c[(b == k) & ~same].mean() for k in range(10)]
        print(f"{name:8s} same/cross ratio by distance decile: {np.round(ratio, 2)}  (six nearest: {min(ratio[:6]):.2f} to {max(ratio[:6]):.2f})")


if __name__ == "__main__":
    main()
