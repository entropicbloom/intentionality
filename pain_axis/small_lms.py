"""Do small causal LMs, read like the 25 LLMs, show the LLM-specific remainder?

residual.py found that after regressing each LLM's 45 Gram entries on sentence-meaning
models, the residuals ("remainder") still agree across the 25 LLMs, also across
families. Here we read small open causal LMs (0.1B-1.7B) the way the LLMs are read
(hidden state at the final ":" token, every layer), build their Grams with the authors'
recipe, and ask whether their own remainder after the same meaning predictors
correlates with the mean remainder of the LLMs outside their family.

Layer: chosen per model and variant by the r of its Gram with the 25-LLM mean Gram,
not by the remainder. We also give the median over the upper half of layers.
Chance: r between the residual of a random 45-vector and the same reference remainder.
Reference: each LLM's remainder vs the mean remainder of the other families (LOFO).

Meaning predictors: MiniLM, mpnet, bge-large (mean pooled) and GPT-2 medium layer 20
(final token), each linear + squared. gpt2-medium runs in float64 (in float32 its
output layer returned NaN on this CPU; its hidden states agree with float64 to 1e-6).
The others run in float32; for Qwen2.5-0.5B the float32 and float64 Grams agree
to < 1e-4.
"""

import gc
import os

import numpy as np
import torch

from baseline import OUT, SETS, load_sentences, similarity
from identify import IU, VARIANTS, family, load_variant
from residual import residuals

# name: (parameters in B, family among the 25 LLMs to exclude from the reference, or None)
LMS = {
    "gpt2": (0.12, None),
    "EleutherAI/pythia-160m": (0.16, None),
    "HuggingFaceTB/SmolLM2-360M": (0.36, None),
    "EleutherAI/pythia-410m": (0.41, None),
    "Qwen/Qwen2.5-0.5B": (0.49, "Qwen"),
    "Qwen/Qwen3-0.6B-Base": (0.6, "Qwen"),
    "gpt2-large": (0.77, None),
    "allenai/OLMo-2-0425-1B": (1.5, None),
    "TinyLlama/TinyLlama_v1.1": (1.1, "Llama"),
    "microsoft/phi-1_5": (1.4, "Phi"),
    "EleutherAI/pythia-1.4b": (1.4, None),
    "gpt2-xl": (1.6, None),
    "Qwen/Qwen2.5-1.5B": (1.5, "Qwen"),
    "stabilityai/stablelm-2-1_6b": (1.6, None),
    "HuggingFaceTB/SmolLM2-1.7B": (1.7, None),
    "Qwen/Qwen3-1.7B-Base": (1.7, "Qwen"),
}
MEANING_LM, MEANING_LAYER = "gpt2-medium", 20


def tag(name):
    return name.split("/")[-1].replace(".", "")


def hidden_states(name, texts, dtype=torch.float32):
    """Final-token hidden state of every layer, (sentences, layers + 1, d)."""
    suffix = "_f32" if dtype == torch.float32 else ""
    cache = os.path.join(OUT, f"hidden_{tag(name)}{suffix}.npy")
    if os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, dtype=dtype).eval()
    out = []
    with torch.no_grad():
        for t in texts:
            hs = model(**tok(t, return_tensors="pt"), output_hidden_states=True).hidden_states
            out.append(torch.stack([h[0, -1] for h in hs]).float().numpy())
    out = np.array(out)
    np.save(cache, out)
    del model
    gc.collect()
    print(f"  ran {name}: {out.shape}", flush=True)
    return np.load(cache, mmap_mode="r")


def meaning_embeddings(texts):
    from sentence_transformers import SentenceTransformer
    embs = {}
    for name in ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2",
                 "BAAI/bge-large-en-v1.5"]:
        path = os.path.join(OUT, f"meaning_{tag(name)}.npy")
        if not os.path.exists(path):
            np.save(path, SentenceTransformer(name).encode(texts, batch_size=64).astype(np.float32))
        embs[tag(name)] = np.load(path)
    return embs


def main():
    sents = load_sentences()
    texts = sum((sents[s][0] for s in SETS), [])
    bounds = np.cumsum([0] + [len(sents[s][0]) for s in SETS])
    cats = {s: sents[s][1] for s in SETS}

    def gram(X, v):
        acts = {s: np.asarray(X[bounds[i]:bounds[i + 1]], dtype=np.float64) for i, s in enumerate(SETS)}
        return similarity(acts, cats, v)[IU]

    embs = meaning_embeddings(texts)
    meaning_lm = hidden_states(MEANING_LM, texts, torch.float64)[:, MEANING_LAYER]
    hid = {n: hidden_states(n, texts) for n in LMS}
    rng = np.random.default_rng(0)

    lines = ["Small causal LMs read at the final ':' token. Per model and variant:",
             "  layer   chosen by r of the model's Gram with the 25-LLM mean Gram",
             "  r_gram  that r",
             "  r_rem   r of the model's remainder with the mean remainder of the LLMs outside its family",
             "  r_rem_upper  median r_rem over the upper half of layers",
             "Chance: 95% of |r| for the residual of a random 45-vector. Reference: each LLM vs the",
             "mean remainder of the other families (LOFO), median [min, max].", ""]
    for v in VARIANTS:
        names, mats = load_variant(VARIANTS[v])
        Y = mats[:, IU[0], IU[1]]
        base = np.column_stack([gram(X, v) for X in embs.values()] + [gram(meaning_lm, v)])
        base = np.column_stack([base, base ** 2])
        E, r2 = residuals(Y, base)
        fams = np.array([family(n) for n in names])
        lofo = [np.corrcoef(E[m], E[fams != fams[m]].mean(0))[0, 1] for m in range(len(E))]
        null = [np.corrcoef(residuals(rng.standard_normal((1, 45)), base)[0][0], E.mean(0))[0, 1]
                for _ in range(5000)]
        lines.append(f"=== {v} ===  meaning R2 {np.median(r2):.2f};  LLM reference (LOFO) "
                     f"{np.median(lofo):.2f} [{min(lofo):.2f}, {max(lofo):.2f}];  "
                     f"chance ±{np.percentile(np.abs(null), 95):.2f}")
        lines.append(f"  {'model':26s} {'size':>5s} {'layer':>6s} {'r_gram':>7s} {'r_rem':>6s} {'r_rem_upper':>11s}")
        for n, (size, fam) in sorted(LMS.items(), key=lambda kv: kv[1][0]):
            ref = E[fams != fam].mean(0)
            rows = []
            for L in range(1, hid[n].shape[1]):
                g = gram(hid[n][:, L], v)
                rows.append((np.corrcoef(g, Y.mean(0))[0, 1],
                             np.corrcoef(residuals(g[None], base)[0][0], ref)[0, 1], L))
            best = max(rows)
            upper = np.median([r[1] for r in rows[len(rows) // 2:]])
            lines.append(f"  {tag(n):26s} {size:5.2f} {'L' + str(best[2]):>6s} {best[0]:7.2f} "
                         f"{best[1]:+6.2f} {upper:+11.2f}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "small_lms.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
