"""Class-balance check on saved orientation predictions (MICrONS outputs/preds or allen/outputs/preds).
Reports the raw error, the error per true-orientation bin, the balanced error (mean over bins), the best
constant predictor (what the uneven label distribution alone buys), and where the predictions pile up.
Usage: python -m microns_ambiguity.balance_check <path.npz> [<path.npz> ...] [bins=8]"""
import sys

import numpy as np


def ang_err(a, b):
    return np.abs((a - b + 90) % 180 - 90)


def check(path, K=8):
    z = np.load(path); P, y = z["P"], z["y"]
    if y.ndim == 2: y = y[:, 0]
    ok = np.isfinite(y); P, y = P[ok], y[ok] % 180
    th = (np.rad2deg(np.arctan2(P[:, 1], P[:, 0])) / 2) % 180
    d = ang_err(th, y); w = 180 / K
    yb = ((y + w / 2) // w % K).astype(int); tb = ((th + w / 2) // w % K).astype(int)   # bins centred on multiples of w
    per = np.array([d[yb == k].mean() for k in range(K)]); frac = np.bincount(yb, minlength=K) / len(y)
    grid = np.arange(0, 180, 1.0); const = min(ang_err(g, y).mean() for g in grid)
    pred_frac = np.bincount(tb, minlength=K) / len(y)
    print(f"{path.split('/')[-1]}: n={len(y)} err={d.mean():.1f} balanced_err={per.mean():.1f} best_constant={const:.1f} chance=45")
    print("  true-bin centre : " + " ".join(f"{k * w:5.0f}" for k in range(K)))
    print("  label fraction  : " + " ".join(f"{f:5.2f}" for f in frac))
    print("  error in bin    : " + " ".join(f"{e:5.1f}" for e in per))
    print("  pred fraction   : " + " ".join(f"{f:5.2f}" for f in pred_frac))


if __name__ == "__main__":
    K = 8; paths = []
    for a in sys.argv[1:]:
        if a.startswith("bins="): K = int(a.split("=")[1])
        else: paths.append(a)
    for p in paths: check(p, K)
