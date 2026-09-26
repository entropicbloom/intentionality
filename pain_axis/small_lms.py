"""Do small causal LMs, read like the 25 LLMs, show the LLM-specific remainder?

residual.py found that after regressing each LLM's 45 Gram entries on sentence-meaning
models, the residuals ("remainder") still agree across the 25 LLMs. Here we read
Qwen2.5-0.5B and Qwen2.5-1.5B the way the LLMs are read (hidden state at the final
":" token, every layer), build their Grams with the authors' recipe, and ask whether
their own remainder after the same meaning predictors correlates with the mean LLM
remainder.

Meaning predictors: MiniLM, mpnet, bge-large (mean pooled) and GPT-2 medium layer 20
(final token), each linear + squared. Forward passes run in float64: in float32 on
this CPU, gpt2-medium's output layer returned NaN and Qwen's hidden states differed
from float64 by up to 7%.
"""

import os

import numpy as np
import torch

from baseline import OUT, SETS, load_sentences, similarity
from identify import IU, VARIANTS, load_variant
from residual import agreement, residuals

LMS = ["gpt2-medium", "Qwen/Qwen2.5-0.5B"]


def tag(name):
    return name.split("/")[-1].replace(".", "")


def hidden_states(name, texts):
    """Final-token hidden state of every layer, (sentences, layers + 1, d)."""
    cache = os.path.join(OUT, f"hidden_{tag(name)}.npy")
    old = os.path.join(OUT, f"nexttoken_{tag(name)}.npz")  # earlier run stored them there
    if not os.path.exists(cache) and os.path.exists(old):
        np.save(cache, np.load(old)["hidden"])
    if os.path.exists(cache):
        return np.load(cache)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, dtype=torch.float64).eval()
    out = []
    with torch.no_grad():
        for t in texts:
            hs = model(**tok(t, return_tensors="pt"), output_hidden_states=True).hidden_states
            out.append(torch.stack([h[0, -1] for h in hs]).float().numpy())
    out = np.array(out)
    np.save(cache, out)
    return out


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
    hid = {n: hidden_states(n, texts) for n in LMS}

    lines = ["Per layer of each small LM (final-token hidden state): r of its Gram with the",
             "25-LLM mean Gram, and r of its remainder (after the meaning predictors) with the",
             "mean LLM remainder. For reference, LLM remainders agree with each other at the",
             "LOO value given per variant.", ""]
    for v in VARIANTS:
        names, mats = load_variant(VARIANTS[v])
        Y = mats[:, IU[0], IU[1]]
        base = np.column_stack([gram(X, v) for X in embs.values()] + [gram(hid["gpt2-medium"][:, 20], v)])
        base = np.column_stack([base, base ** 2])
        E, r2 = residuals(Y, base)
        loo, _ = agreement(E, names)
        shared = E.mean(0)
        lines.append(f"=== {v} === meaning R2 {np.median(r2):.2f}; LLM remainder agreement LOO {np.median(loo):.2f}")
        for n in LMS[1:]:
            row = []
            for L in range(1, hid[n].shape[1]):
                g = gram(hid[n][:, L], v)
                gres = residuals(g[None], base)[0][0]
                row.append((L, np.corrcoef(g, Y.mean(0))[0, 1], np.corrcoef(gres, shared)[0, 1]))
            lines.append(f"  {tag(n)}  layer: r with LLM mean / remainder r")
            lines.append("    " + "  ".join(f"L{L} {a:.2f}/{b:+.2f}" for L, a, b in row))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "small_lms.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
