"""Training-dynamics emergence sweep.

When during pretraining does convergent (cross-seed) representational geometry
crystallize, and does the timing differ between contextual activations and the
static unembedding interface?

Pythia publishes ~154 checkpoints per model (revision="step{N}"). We sweep two
independently-seeded pythia-160m runs over a step grid, extract concept vectors
at each checkpoint, and run label-free Gram matching at each step. Output: a
per-step identifiability curve, one line per direction family.

    python -m emergence.sweep --mode extract    # cache vectors per checkpoint
    python -m emergence.sweep --mode curve       # build outputs/curve.json

Extraction caches one npz per (seed, step, family); the curve phase and re-runs
need no recompute. Each checkpoint is downloaded into the HF cache once.
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emergence.tokens import select_single_token_words
from emergence.matching import gram, cka, subset_match, profile_match

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
SWEEP_DIR = os.path.join(OUT_DIR, "sweep")

# Two independently pretrained 160m runs (shared NeoX tokenizer + config).
SEEDS = {
    "seedA": "EleutherAI/pythia-160m-seed1",
    "seedB": "EleutherAI/pythia-160m-seed2",
}

# Available revisions: step0,1,2,4,...,512 (log2), then every 1000 to 143000.
STEPS = [0, 8, 64, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000,
         128000, 143000]

LAYER_FRAC = 2.0 / 3.0
FAMILIES = ["act", "unembed"]

# Neutral carrier templates; the concept word is the final token.
TEMPLATES = [
    "My favorite word is{w}",
    "She wrote a single word on the board:{w}",
    "The story kept returning to the idea of{w}",
]
ACT_BATCH = 16
SUBSET_SIZE = 8
N_SUBSETS = 200


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def npz_path(seed, step, family):
    return os.path.join(SWEEP_DIR, f"{seed}-step{step}-{family}.npz")


def extract_activations(model, tokenizer, words, layer_idx, device):
    acc = torch.zeros(len(words), model.config.hidden_size, dtype=torch.float64)
    with torch.no_grad():
        for template in TEMPLATES:
            texts = [template.replace("{w}", " " + w) for w in words]
            for start in range(0, len(texts), ACT_BATCH):
                batch = texts[start:start + ACT_BATCH]
                enc = tokenizer(batch, return_tensors="pt", padding=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc, output_hidden_states=True)
                h = out.hidden_states[layer_idx]
                lengths = enc["attention_mask"].sum(dim=1) - 1  # word is last token
                for i, pos in enumerate(lengths):
                    acc[start + i] += h[i, pos].cpu().double()
    return (acc / len(TEMPLATES)).float().numpy()


def extract_for_checkpoint(seed_tag, repo, step, token_ids, words):
    device = get_device()
    revision = f"step{step}"
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, revision=revision).eval().to(device)
    n_layers = model.config.num_hidden_layers
    layer_idx = max(1, round(LAYER_FRAC * n_layers))

    families = {
        "act": extract_activations(model, tokenizer, words, layer_idx, device),
        "unembed": model.get_output_embeddings().weight.detach().cpu().numpy()[token_ids],
    }
    for family, mat in families.items():
        np.savez(npz_path(seed_tag, step, family), j=mat, words=np.array(words),
                 token_ids=np.array(token_ids), model=repo, step=step,
                 layer_idx=layer_idx, n_layers=n_layers)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()


def run_extract():
    os.makedirs(SWEEP_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(SEEDS["seedA"])
    words, token_ids = select_single_token_words(tok)
    print(f"{len(words)} concept tokens; {len(STEPS)} steps x {len(SEEDS)} seeds")
    for step in STEPS:
        for seed_tag, repo in SEEDS.items():
            if all(os.path.exists(npz_path(seed_tag, step, f)) for f in FAMILIES):
                print(f"  [skip] {seed_tag} step{step} (cached)")
                continue
            print(f"  extracting {seed_tag} step{step} ...", flush=True)
            extract_for_checkpoint(seed_tag, repo, step, token_ids, words)


def _load(seed, step, family):
    return np.load(npz_path(seed, step, family), allow_pickle=True)


def run_curve():
    curve, meta = [], {}
    for step in STEPS:
        row = {"step": step}
        for family in FAMILIES:
            pa, pb = npz_path("seedA", step, family), npz_path("seedB", step, family)
            if not (os.path.exists(pa) and os.path.exists(pb)):
                continue
            da, db = _load("seedA", step, family), _load("seedB", step, family)
            g_a, g_b = gram(da["j"]), gram(db["j"])
            rng = np.random.default_rng(0)
            sub_acc, sub_hit = subset_match(g_a, g_b, rng, SUBSET_SIZE, N_SUBSETS)
            full_acc, _ = profile_match(g_a, g_b)
            row[family] = {"subset_acc": round(sub_acc, 4), "subset_hit": round(sub_hit, 4),
                           "full_acc": round(full_acc, 4), "cka": round(cka(g_a, g_b), 4)}
            if family == "act" and not meta:
                meta = {"model": "pythia-160m", "layer_idx": int(da["layer_idx"]),
                        "n_layers": int(da["n_layers"]), "n_tokens": int(len(da["words"])),
                        "subset_size": SUBSET_SIZE}
        curve.append(row)
        cells = "  ".join(f"{f}: sub={row[f]['subset_acc']:.2f}" for f in FAMILIES if f in row)
        print(f"step {step:>6}  {cells}")

    out = os.path.join(OUT_DIR, "curve.json")
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
