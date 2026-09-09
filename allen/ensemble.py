"""Ensemble saved per-neuron predictions of several decoder runs that share a mouse split.
Usage: python -m allen.ensemble tag1 tag2 ...  (classification: mean softmax; regression: mean)"""
import os, sys
import numpy as np
from allen.data import ROOT
D = os.path.join(ROOT, "allen", "outputs", "preds")

def main(tags):
    runs = [np.load(os.path.join(D, t + ".npz")) for t in tags]
    idx = runs[0]["idx"]; assert all((r["idx"] == idx).all() for r in runs), "runs must share the test set"
    y = runs[0]["y"]
    if y.ndim == 1 and np.issubdtype(y.dtype, np.integer):
        Ps = [np.exp(r["P"] - r["P"].max(1, keepdims=True)) for r in runs]; Ps = [p / p.sum(1, keepdims=True) for p in Ps]
        for t, p in zip(tags, Ps): print(f"  {t}: {(p.argmax(1) == y).mean():.3f}")
        print(f"ensemble of {len(tags)}: acc={(np.mean(Ps, 0).argmax(1) == y).mean():.3f}")
    else:
        P = np.mean([r["P"] for r in runs], 0)
        print("regression ensemble: mean prediction saved; R2 needs the standardisation used in training (see decoder2)")

if __name__ == "__main__":
    main(sys.argv[1:])
