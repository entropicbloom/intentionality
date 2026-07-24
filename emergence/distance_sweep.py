"""Weight-space distance travelled per checkpoint — a schedule-agnostic x-axis.

Raw step is confounded by the LR warmup (early steps move the weights barely at
all). A more honest progress coordinate is how far the weights have actually
moved from initialization: ||theta(step) - theta(0)||_2 over all parameters.
This absorbs the step-size confound directly (tiny-LR warmup steps contribute
little distance). We also record cumulative LR (integral of the schedule) as a
cheaper proxy for the same correction.

    python -m emergence.distance_sweep
"""

import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from emergence.sweep import SEEDS, STEPS, OUT_DIR

REPO = SEEDS["seedA"]
MAX_LR, MIN_LR, WARMUP, TOTAL = 6e-4, 6e-5, 1430, 143000


def cumulative_lr(step):
    """Integral of the LR schedule from 0 to `step` (in lr*step units)."""
    if step <= 0:
        return 0.0
    fine = np.arange(0, step + 1)
    warm = MAX_LR * fine / WARMUP
    dr = np.clip((fine - WARMUP) / (TOTAL - WARMUP), 0, 1)
    cos = MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + np.cos(np.pi * dr))
    lr = np.where(fine < WARMUP, warm, cos)
    return float(lr.sum())


def state_dict_at(step):
    m = AutoModelForCausalLM.from_pretrained(REPO, revision=f"step{step}")
    sd = {k: v.detach().float().clone() for k, v in m.state_dict().items()}
    del m
    return sd


def main():
    print(f"loading reference theta(0) for {REPO} ...", flush=True)
    ref = state_dict_at(0)
    norm0 = float(torch.sqrt(sum((v * v).sum() for v in ref.values())))

    curve = []
    for step in STEPS:
        sd = state_dict_at(step)
        sq = 0.0
        for k, v in sd.items():
            sq += float(((v - ref[k]) ** 2).sum())
        dist = float(np.sqrt(sq))
        curve.append({"step": step, "dist": round(dist, 4),
                      "rel_dist": round(dist / norm0, 5),
                      "cum_lr": round(cumulative_lr(step), 4)})
        print(f"step {step:>6}  ||dTheta||={dist:8.2f}  rel={dist/norm0:.4f}  "
              f"cumLR={cumulative_lr(step):.3f}", flush=True)
        del sd

    meta = {"model": "pythia-160m", "seed": "seed1", "norm_theta0": round(norm0, 2)}
    out = os.path.join(OUT_DIR, "distance_curve.json")
    with open(out, "w") as f:
        json.dump({"meta": meta, "curve": curve}, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
