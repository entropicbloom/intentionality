"""Synthetic orientation populations for the mechanism test.
Model neurons with von Mises tuning respond to a movie-like sequence of T oriented frames; the
decoder then sees only the Gram of sampled populations, exactly as for MICrONS.  Three switches,
each on/off, give eight cells:
  count   cardinal excess in the preferred orientations (half the neurons drawn near 0° / 90°)
  sharp   sharper tuning at the cardinal orientations (kappa 2 -> 4 at 0°/90°, 2 at obliques)
  stim    cardinal excess in the stimulus (half the frames near 0° / 90°)
With every switch off the class-Gram is circulant in expectation, so the labels are fixed only up
to rotation and reflection; the frame check (err vs err_modD) says whether a switch fixes the frame.
Usage: python -m microns_ambiguity.synthetic <tag> count=0|1 sharp=0|1 stim=0|1 [decoder kwargs]"""
import json
import sys
import time

import numpy as np

from .config import OUT
from .decoder2 import train


def vonmises_pref(n, rng, cardinal):
    th = rng.uniform(0, 180, n)
    if cardinal:                                  # half the neurons near a cardinal axis (sd ~ 12°)
        k = n // 2; th[:k] = (rng.choice([0.0, 90.0], k) + rng.normal(0, 12, k)) % 180
    return th


def make(n=6000, T=120, count=0, sharp=0, stim=0, noise=0.5, seed=0, amp_var=0, sharp_k=4.0, gain=0):
    rng = np.random.default_rng(seed)
    pref = vonmises_pref(n, rng, count)
    kappa = np.full(n, 2.0)
    if sharp: kappa = 2.0 + (sharp_k - 2.0) * np.cos(np.deg2rad(2 * pref)) ** 2   # sharp_k at 0°/90°, 2 at 45°/135°
    g = 1.0 + 0.5 * np.cos(np.deg2rad(2 * pref)) ** 2 if gain else np.ones(n)          # gain: cardinal neurons respond 1.5x more strongly
    phi = rng.permutation((np.arange(T) + rng.uniform(0, 1, T)) * 180 / T)  # frame orientations: even coverage of the circle
    if stim:                                                                # cardinal excess in the stimulus: half the frames near 0° / 90°
        k = T // 2; phi[:k] = (rng.choice([0.0, 90.0], k) + rng.normal(0, 12, k)) % 180
    amp = rng.gamma(2.0, 1.0, T) if amp_var else np.ones(T)                 # frame-to-frame contrast (off by default: it breaks the symmetry by finite sampling)
    d = np.deg2rad(2 * (pref[:, None] - phi[None, :]))
    R = g[:, None] * amp[None, :] * np.exp(kappa[:, None] * (np.cos(d) - 1)) + noise * rng.normal(size=(n, T))
    return R.astype(np.float32), pref.astype(np.float32)


def main(tag, count=0, sharp=0, stim=0, n_neurons=6000, T=120, noise=0.5, data_seed=0, amp_var=0, sharp_k=4.0, gain=0, **kw):
    F, pref = make(n_neurons, T, count, sharp, stim, noise, data_seed, amp_var, sharp_k, gain)
    rng = np.random.default_rng(data_seed); perm = rng.permutation(n_neurons); h = n_neurons // 2
    tr, va = perm[:h], perm[h:]
    t0 = time.time(); print(f"[{tag}] synthetic count={count} sharp={sharp} stim={stim} n={n_neurons} T={T} {kw}", flush=True)
    m = train(F, pref, "circ", tr, va, **kw)
    m.update(synthetic=dict(count=count, sharp=sharp, stim=stim, n_neurons=n_neurons, T=T, noise=noise, data_seed=data_seed, amp_var=amp_var, sharp_k=sharp_k, gain=gain), seconds=time.time() - t0)
    m.pop("preds", None); OUT.mkdir(exist_ok=True); p = OUT / "synthetic.json"
    d = json.load(open(p)) if p.exists() else {}; d[tag] = m; json.dump(d, open(p, "w"))
    print(f"  -> {tag}: err={m['err']:.2f} err_modD={m['err_modD']:.2f} within15={m['within15']:.3f} best_epoch={m.get('best_epoch')} {m['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    tag = sys.argv[1]; kw = {}
    for a in sys.argv[2:]:
        k, v = a.split("="); kw[k] = v if k in ("device", "input_mode") else (bool(int(v)) if k in ("rel_bias", "row_proj", "label_rot", "ori_weight") else (float(v) if k in ("lr", "early_stop", "dropout", "noise", "sharp_k") else int(v)))
    main(tag, **kw)
