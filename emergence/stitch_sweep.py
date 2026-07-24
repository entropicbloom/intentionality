"""Model stitching across checkpoints — does matched geometry mean the
representations are functionally interchangeable?

At each checkpoint we splice seed1 and seed2: run seed1 up to layer L, map its
residual stream into seed2's space, and let seed2's remaining layers + head
finish. We measure held-out cross-entropy of the stitched network. Three maps,
increasingly permissive:
  - identity   : plug raw activations in (null; different bases -> should fail)
  - procrustes : orthogonal rotation aligning the two spaces (the map our
                 rotation-invariant identifiability metric implies)
  - linear     : unconstrained affine map (upper bound)

Reference floor = seed2's own held-out CE (a "perfect" self-stitch). The gap
stitched - solo is the functional cost of swapping seed1 in.

Confound control: around emergence both models also get individually better, so
a raw stitched-loss drop could just be "both got good". To isolate genuine
interchangeability we add a competence-matched null, `shuffle`: the SAME seed1
activations and Procrustes map, but with token positions shuffled — identical
model competence, correspondence destroyed. The confound-free signal is
shuffle_ce - procrustes_ce (functional content the alignment transmits): if
`procrustes` falls to the solo floor while `shuffle` stays high around the
emergence window, that is genuine interchangeability, not general improvement.

Maps are fit on a separate corpus and evaluated on the held-out set. Caveat:
d_model=768 vs a modest fit corpus, so the orthogonal fit is somewhat
underdetermined — read the trend, not absolute values.

    python -m emergence.stitch_sweep --mode extract
    python -m emergence.stitch_sweep --mode curve
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emergence.sweep import SEEDS, STEPS, SWEEP_DIR, OUT_DIR, get_device
from emergence.loss_sweep import HELDOUT, heldout_ce

LAYER = 8               # stitch point (hidden_states[LAYER]); matches the act metric
PAIR = ("seedA", "seedB")

# Fitting corpus for the stitch map — disjoint from HELDOUT, longer for more tokens.
FIT_TEXT = [
    "The committee reviewed the proposal carefully before reaching a decision about the funding.",
    "In the early hours the city was quiet, and only a few cars moved along the wide avenues.",
    "Scientists have long debated the origins of the phenomenon, gathering evidence from many fields.",
    "The garden in summer was full of bees, and the roses climbed along the weathered stone wall.",
    "He read the letter twice, folded it slowly, and placed it back inside the wooden drawer.",
    "Modern processors contain billions of transistors arranged in intricate and repeating patterns.",
    "The orchestra tuned their instruments as the audience settled into the darkened concert hall.",
    "Farmers watched the sky anxiously, hoping the rain would arrive before the crops were lost.",
    "The novel follows three generations of a family through war, migration, and slow reconciliation.",
    "A small boat crossed the bay at dawn, its sail catching the first pale light of the morning.",
    "The lecture covered the history of the language, from its earliest roots to its modern forms.",
    "Engineers spent months testing the prototype under conditions far harsher than it would ever face.",
    "The market was crowded with vendors selling fruit, cloth, spices, and freshly baked bread.",
    "Over the winter the lake froze solid, and children skated across it every afternoon after school.",
    "The report concluded that further study was needed before any firm recommendation could be made.",
    "She sketched the mountains from memory, the jagged peaks softened by a haze of distant cloud.",
]


def cache_path(step):
    return os.path.join(SWEEP_DIR, f"stitch-L{LAYER}-step{step}.json")


def collect_acts(model, tokenizer, texts, layer, device):
    rows = []
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors="pt").to(device)
            h = model(**enc, output_hidden_states=True).hidden_states[layer][0]
            rows.append(h.cpu().double().numpy())
    return np.concatenate(rows, 0)  # [N_tok, d]


def fit_maps(hA, hB):
    mA, mB = hA.mean(0), hB.mean(0)
    Ac, Bc = hA - mA, hB - mB
    U, _, Vt = np.linalg.svd(Ac.T @ Bc)
    R = U @ Vt  # orthogonal, Ac @ R ~= Bc
    HA = np.concatenate([hA, np.ones((len(hA), 1))], 1)
    W, *_ = np.linalg.lstsq(HA, hB, rcond=None)  # affine (d+1, d)
    return {"identity": ("identity",),
            "procrustes": ("procrustes", mA, R, mB),
            "linear": ("linear", W)}


def apply_map(h, m):
    kind = m[0]
    if kind == "identity":
        return h
    if kind == "procrustes":
        _, mA, R, mB = m
        return (h - mA) @ R + mB
    _, W = m
    return np.concatenate([h, np.ones((len(h), 1))], 1) @ W


def stitched_ce(modelA, modelB, tokenizer, texts, layer, m, device, shuffle=False):
    block = modelB.gpt_neox.layers[layer - 1]
    rng = np.random.default_rng(0)
    total_loss, total_tok = 0.0, 0
    for t in texts:
        enc = tokenizer(t, return_tensors="pt").to(device)
        ids = enc["input_ids"]
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            hA = modelA(**enc, output_hidden_states=True).hidden_states[layer][0]
        hA = hA.cpu().double().numpy()
        if shuffle:  # competence-matched null: destroy the token correspondence
            hA = hA[rng.permutation(hA.shape[0])]
        stitched = apply_map(hA, m)
        mdtype = next(modelB.parameters()).dtype  # checkpoints load as fp16
        st = torch.tensor(stitched, device=device).to(mdtype).unsqueeze(0)

        def hook(_mod, _inp, out):
            return (st,) + tuple(out[1:]) if isinstance(out, tuple) else st

        handle = block.register_forward_hook(hook)
        try:
            with torch.no_grad():
                loss = float(modelB(**enc, labels=ids).loss)
            n = ids.shape[1] - 1
            total_loss += loss * n
            total_tok += n
        finally:
            handle.remove()
    return total_loss / total_tok


def run_extract():
    os.makedirs(SWEEP_DIR, exist_ok=True)
    device = torch.device("cpu")  # fp16 stitch is unstable on MPS; CPU is fast enough here
    repoA, repoB = SEEDS[PAIR[0]], SEEDS[PAIR[1]]
    for step in STEPS:
        if os.path.exists(cache_path(step)):
            print(f"  [skip] step{step}")
            continue
        print(f"  stitch step{step} ...", flush=True)
        rev = f"step{step}"
        tok = AutoTokenizer.from_pretrained(repoA, revision=rev)
        modelA = AutoModelForCausalLM.from_pretrained(repoA, revision=rev).eval().to(device)
        modelB = AutoModelForCausalLM.from_pretrained(repoB, revision=rev).eval().to(device)

        hA = collect_acts(modelA, tok, FIT_TEXT, LAYER, device)
        hB = collect_acts(modelB, tok, FIT_TEXT, LAYER, device)
        maps = fit_maps(hA, hB)
        rec = {"step": step, "solo_B": heldout_ce(modelB, tok, device)}
        for name, m in maps.items():
            rec[name] = stitched_ce(modelA, modelB, tok, HELDOUT, LAYER, m, device)
        # competence-matched null: procrustes map on position-shuffled activations
        rec["shuffle"] = stitched_ce(modelA, modelB, tok, HELDOUT, LAYER,
                                     maps["procrustes"], device, shuffle=True)
        with open(cache_path(step), "w") as f:
            json.dump(rec, f)
        print(f"    solo={rec['solo_B']:.2f}  id={rec['identity']:.2f}  "
              f"proc={rec['procrustes']:.2f}  lin={rec['linear']:.2f}  "
              f"shuf={rec['shuffle']:.2f}", flush=True)
        del modelA, modelB
        if device.type == "mps":
            torch.mps.empty_cache()


def run_curve():
    curve = []
    for step in STEPS:
        p = cache_path(step)
        if os.path.exists(p):
            curve.append(json.load(open(p)))
    meta = {"model": "pythia-160m", "layer": LAYER, "pair": "seed1|seed2",
            "n_heldout": len(HELDOUT), "n_fit_tokens": None}
    out = os.path.join(OUT_DIR, "stitch_curve.json")
    with open(out, "w") as f:
        json.dump({"meta": meta, "curve": curve}, f, indent=2)
    for r in curve:
        print(f"step {r['step']:>6}  solo={r['solo_B']:.2f}  id={r['identity']:.2f}  "
              f"proc={r['procrustes']:.2f}  lin={r['linear']:.2f}")
    print(f"\nsaved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["extract", "curve"], required=True)
    args = ap.parse_args()
    run_extract() if args.mode == "extract" else run_curve()


if __name__ == "__main__":
    main()
