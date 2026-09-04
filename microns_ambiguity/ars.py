"""Ambiguity Reduction Scores, as defined in the paper (Fano / Gaussian bounds)."""
import numpy as np


def _hb(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def ars_fano(acc: float, K: int) -> float:
    e = 1 - acc
    return float(1 - (_hb(e) + e * np.log2(max(K - 1, 1))) / np.log2(K))


def ars_gaussian(r2: float) -> float:
    r2 = min(max(r2, 0.0), 1 - 1e-6)
    return float(np.log2(1 / (1 - r2)) / np.log2(2 * np.pi * np.e))
