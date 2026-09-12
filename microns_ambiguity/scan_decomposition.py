"""Scan / area decomposition of saved MICrONS predictions (outputs/preds/<tag>.npz).
For RF (reg): R² on absolute coordinates vs R² after centring predictions and truth per scan
(and per area); a scan-identity readout scores 0 after centring.
For orientation (circ): mean angular error vs the error of an area-prior predictor (each neuron
gets its area's circular mean) and the error after removing a per-area rotation of the predictions.
Usage: python -m microns_ambiguity.scan_decomposition <tag> [<tag> ...]"""
import sys

import numpy as np

from .config import OUT


def r2(P, Y):
    return float(np.mean(1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)))


def centred(P, Y, g):
    P, Y = P.copy(), Y.copy()
    for k in np.unique(g):
        m = g == k; P[m] -= P[m].mean(0); Y[m] -= Y[m].mean(0)
    return P, Y


def ang_err(a, b):
    return np.abs((a - b + 90) % 180 - 90)


def circ_mean(th):
    return np.rad2deg(np.angle(np.exp(1j * np.deg2rad(2 * th)).mean())) / 2


def decompose(tag):
    z = np.load(OUT / "preds" / f"{tag}.npz", allow_pickle=True); P, y, scan, area = z["P"], z["y"], z["scan"].astype(str), z["area"].astype(str)
    out = dict(tag=tag, n=int(len(y)), scans=int(len(np.unique(scan))))
    if y.ndim == 2:                                   # RF: predictions are standardised; put them back on the label scale
        ok = np.isfinite(y).all(1); P, y, scan, area = P[ok], y[ok], scan[ok], area[ok]
        Pd = P * y.std(0) + y.mean(0)                   # test-set scale (the training mean/std are not stored; R² is scale-invariant up to this)
        out["r2_abs"] = r2(Pd, y)
        for name, g in [("scan", scan), ("area", area)]:
            Pc, Yc = centred(Pd, y, g); out[f"r2_within_{name}"] = r2(Pc, Yc)
            out[f"r2_between_{name}"] = r2(np.stack([Pd[g == k].mean(0) for k in np.unique(g)]), np.stack([y[g == k].mean(0) for k in np.unique(g)])) if len(np.unique(g)) > 2 else float("nan")
            # scan-identity-only predictor: each neuron gets its group's true mean
            out[f"r2_groupmean_{name}"] = r2(np.stack([y[g == k].mean(0) for k in g]), y)
            out[f"var_between_{name}"] = float(1 - ((Yc ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)).mean())
    else:                                            # orientation
        th_hat = np.rad2deg(np.arctan2(P[:, 1], P[:, 0])) / 2; ok = np.isfinite(y); th_hat, y, scan, area = th_hat[ok], y[ok], scan[ok], area[ok]
        out["err"] = float(ang_err(th_hat, y).mean())
        for name, g in [("scan", scan), ("area", area)]:
            prior = np.array([circ_mean(y[g == k]) for k in g]); out[f"err_groupprior_{name}"] = float(ang_err(prior, y).mean())
            adj = th_hat.copy()
            for k in np.unique(g):
                m = g == k; adj[m] = (th_hat[m] - circ_mean(th_hat[m] - y[m])) % 180
            out[f"err_within_{name}"] = float(ang_err(adj, y).mean())
        out["err_per_area"] = {str(k): float(ang_err(th_hat[area == k], y[area == k]).mean()) for k in np.unique(area)}
        out["n_per_area"] = {str(k): int((area == k).sum()) for k in np.unique(area)}
    return out


if __name__ == "__main__":
    for t in sys.argv[1:]:
        d = decompose(t); print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()), flush=True)
