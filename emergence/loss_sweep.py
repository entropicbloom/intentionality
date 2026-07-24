"""Held-out cross-entropy per checkpoint — a training-dynamics reference axis.

Does the identifiability transition line up with the model's actual learning?
For each cached checkpoint we compute mean next-token cross-entropy on a fixed
held-out English sample (forward pass only), averaged over the seed models.
Pythia's loss falls steeply over the first ~1-2k steps then plateaus; overlaying
this against the identifiability curve tests whether recoverable geometry emerges
during the rapid-loss phase.

    python -m emergence.loss_sweep --mode extract   # -> per (seed,step) cache
    python -m emergence.loss_sweep --mode curve      # -> outputs/loss_curve.json
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emergence.sweep import SEEDS, STEPS, SWEEP_DIR, OUT_DIR, get_device

# Fixed held-out prose (never in a template; just a neutral eval sample).
HELDOUT = [
    "The harbor filled slowly with morning light as the fishing boats returned.",
    "Economists disagree about whether the new policy will reduce inflation.",
    "She opened the old notebook and found a pressed flower between the pages.",
    "The telescope revealed faint spiral arms in the distant galaxy.",
    "After the election, the council met to discuss the coming budget.",
    "A gentle rain fell on the quiet streets long after midnight.",
    "The recipe called for fresh basil, ripe tomatoes, and a little salt.",
    "Engineers tested the bridge under load before opening it to traffic.",
    "He explained the theorem carefully, drawing diagrams on the board.",
    "The children built a sandcastle and watched the tide wash it away.",
    "Researchers published their findings after years of careful experiments.",
    "The train slowed as it approached the station on the far side of the valley.",
]


def cache_path(seed, step):
    return os.path.join(SWEEP_DIR, f"{seed}-step{step}-loss.json")


def heldout_ce(model, tokenizer, device):
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for text in HELDOUT:
            enc = tokenizer(text, return_tensors="pt").to(device)
            ids = enc["input_ids"]
            if ids.shape[1] < 2:
                continue
            out = model(input_ids=ids, labels=ids)
            n = ids.shape[1] - 1  # predicted positions
            total_loss += float(out.loss) * n
            total_tok += n
    return total_loss / total_tok  # mean CE in nats


def run_extract():
    os.makedirs(SWEEP_DIR, exist_ok=True)
    device = get_device()
    for step in STEPS:
        for seed_tag, repo in SEEDS.items():
            if os.path.exists(cache_path(seed_tag, step)):
                print(f"  [skip] {seed_tag} step{step}")
                continue
            print(f"  eval {seed_tag} step{step} ...", flush=True)
            rev = f"step{step}"
            tk = AutoTokenizer.from_pretrained(repo, revision=rev)
            model = AutoModelForCausalLM.from_pretrained(repo, revision=rev).eval().to(device)
            ce = heldout_ce(model, tk, device)
            with open(cache_path(seed_tag, step), "w") as f:
                json.dump({"seed": seed_tag, "step": step, "ce": ce}, f)
            del model
            if device.type == "mps":
                torch.mps.empty_cache()


def run_curve():
    curve = []
    for step in STEPS:
        ces = []
        for seed_tag in SEEDS:
            p = cache_path(seed_tag, step)
            if os.path.exists(p):
                ces.append(json.load(open(p))["ce"])
        if not ces:
            continue
        curve.append({"step": step, "ce": [round(c, 4) for c in ces]})
        print(f"step {step:>6}  CE={np.mean(ces):.3f}  ppl={np.exp(np.mean(ces)):.1f}")
    meta = {"model": "pythia-160m", "n_seeds": len(SEEDS), "n_heldout": len(HELDOUT)}
    out = os.path.join(OUT_DIR, "loss_curve.json")
    with open(out, "w") as f:
        json.dump({"meta": meta, "curve": curve}, f, indent=2)
    print(f"\nsaved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["extract", "curve"], required=True)
    args = ap.parse_args()
    run_extract() if args.mode == "extract" else run_curve()


if __name__ == "__main__":
    main()
