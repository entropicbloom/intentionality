"""Layer-resolved emergence sweep.

Same cross-seed identifiability curve as sweep.py, but for activations at every
other layer (1, 3, ..., 11) instead of a single ⅔-depth layer — one curve per
depth. All probed layers come from the same forward pass, so this costs no more
model passes than the single-layer sweep.

    python -m emergence.layer_sweep --mode extract
    python -m emergence.layer_sweep --mode curve      # -> outputs/layer_curve.json

Then: python -m emergence.plot_layers
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emergence.tokens import select_single_token_words
from emergence.matching import gram, subset_match
from emergence.sweep import (SEEDS, PAIRS, STEPS, SWEEP_DIR, OUT_DIR, TEMPLATES,
                             ACT_BATCH, SUBSET_SIZE, N_SUBSETS, get_device)

LAYERS = [1, 3, 5, 7, 9, 11]


def npz_path(seed, step):
    return os.path.join(SWEEP_DIR, f"{seed}-step{step}-actlayers.npz")


def extract_layers(model, tokenizer, words, layers, device):
    """Return [len(layers), K, d] mean activations at each probed layer."""
    acc = torch.zeros(len(layers), len(words), model.config.hidden_size, dtype=torch.float64)
    with torch.no_grad():
        for template in TEMPLATES:
            texts = [template.replace("{w}", " " + w) for w in words]
            for start in range(0, len(texts), ACT_BATCH):
                batch = texts[start:start + ACT_BATCH]
                enc = tokenizer(batch, return_tensors="pt", padding=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc, output_hidden_states=True)
                lengths = enc["attention_mask"].sum(dim=1) - 1  # word is last token
                for li, layer in enumerate(layers):
                    h = out.hidden_states[layer]
                    for i, pos in enumerate(lengths):
                        acc[li, start + i] += h[i, pos].cpu().double()
    return (acc / len(TEMPLATES)).float().numpy()


def run_extract():
    os.makedirs(SWEEP_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(SEEDS["seedA"])
    words, _ = select_single_token_words(tok)
    print(f"{len(words)} tokens; layers {LAYERS}; {len(STEPS)} steps x {len(SEEDS)} seeds")
    device = get_device()
    for step in STEPS:
        for seed_tag, repo in SEEDS.items():
            if os.path.exists(npz_path(seed_tag, step)):
                print(f"  [skip] {seed_tag} step{step}")
                continue
            print(f"  extracting {seed_tag} step{step} ...", flush=True)
            rev = f"step{step}"
            tk = AutoTokenizer.from_pretrained(repo, revision=rev)
            if tk.pad_token is None:
                tk.pad_token = tk.eos_token
            model = AutoModelForCausalLM.from_pretrained(repo, revision=rev).eval().to(device)
            n_layers = model.config.num_hidden_layers
            j = extract_layers(model, tk, words, LAYERS, device)
            np.savez(npz_path(seed_tag, step), j=j, layers=np.array(LAYERS),
                     words=np.array(words), step=step, n_layers=n_layers)
            del model
            if device.type == "mps":
                torch.mps.empty_cache()


def run_curve():
    """Per-layer identifiability aggregated across all cross-seed PAIRS; each
    layer's value is a list (one per pair) for mean + spread in the plots."""
    curve = []
    n_layers = None
    for step in STEPS:
        if not all(os.path.exists(npz_path(s, step)) for pair in PAIRS for s in pair):
            continue
        loaded = {s: np.load(npz_path(s, step), allow_pickle=True)
                  for pair in PAIRS for s in pair}
        n_layers = int(next(iter(loaded.values()))["n_layers"])
        row = {"step": step}
        for li, layer in enumerate(LAYERS):
            per_pair = []
            for a, b in PAIRS:
                g_a, g_b = gram(loaded[a]["j"][li]), gram(loaded[b]["j"][li])
                rng = np.random.default_rng(0)
                acc, _ = subset_match(g_a, g_b, rng, SUBSET_SIZE, N_SUBSETS)
                per_pair.append(round(acc, 4))
            row[f"L{layer}"] = per_pair
        curve.append(row)
        cells = "  ".join(f"L{l}={np.mean(row[f'L{l}']):.2f}" for l in LAYERS)
        print(f"step {step:>6}  {cells}")

    meta = {"model": "pythia-160m", "n_layers": n_layers, "layers": LAYERS,
            "subset_size": SUBSET_SIZE, "pairs": [f"{a}|{b}" for a, b in PAIRS]}
    out = os.path.join(OUT_DIR, "layer_curve.json")
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
