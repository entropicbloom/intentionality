"""Does the Pain-axis direction-similarity matrix come from the sentences alone?

We rebuild the 10 directions of Tagliabue, Dung & Berg (arXiv:2609.16247) with their
recipe (scripts/3.3_validation/03_similarity_one_model.py in their repo), but from
representations that have no reason to contain a pain axis:

  tfidf          TF-IDF vector of the sentence
  minilm         all-MiniLM-L6-v2 sentence embedding (mean pooled)
  gpt2_L6        GPT-2 small, residual stream after block 6, final token
  pythia160m_L6  Pythia-160m, residual stream after block 6, final token
  gpt2rand_L6    GPT-2 small architecture with random weights, final token
  gpt2rand_mean  same, mean over tokens

and compare each resulting 10x10 matrix with the mean over their 25 LLMs:
Pearson r of the off-diagonal entries, and the rank of the true labelling when
the baseline matrix is matched to the LLM mean over all 10! relabellings.

Their layer is chosen per model by cross-validation; we use a fixed middle layer.
"""

import json
import os

import numpy as np
import torch

from identify import CONCEPTS, IU, K, VARIANTS, all_perms, load_variant, scores

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
DS = os.environ.get("PAIN_AXIS_DATASETS", os.path.join(HERE, "datasets"))

DENOISE_VARIANCE = 0.5
POOL_SETS = ["S1_1P", "S2_1P", "ControlSupplement_1P"]
PAIN_CATS = ["A1", "A2", "A3", "A4", "A5"]
CONTROL_CATS = ["B", "C1", "C2", "D", "E"]
SETS = POOL_SETS + ["Arousal_1P", "Random_1P", "Numb_1P", "SD_sadness_1P"]


def load_sentences():
    d = json.load(open(os.path.join(DS, "3.1_pain_and_control_datasets.json")))["datasets"]
    d.update(json.load(open(os.path.join(DS, "3.1_sadness_dataset.json")))["datasets"])
    return {s: ([x["prompt"] for x in d[s]["sentences"]],
                np.array([x["category"] for x in d[s]["sentences"]])) for s in SETS}


# ---------------------------------------------------------------- representations

def rep_tfidf(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer().fit_transform(texts).toarray()


def rep_minilm(texts):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(texts, batch_size=64)


def rep_causal_lm(name, layer, pool="final", random_init=False):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    def f(texts):
        tok = AutoTokenizer.from_pretrained(name)
        if random_init:
            torch.manual_seed(0)
            model = AutoModel.from_config(AutoConfig.from_pretrained(name))
        else:
            model = AutoModel.from_pretrained(name)
        model.eval()
        out = []
        with torch.no_grad():
            for t in texts:
                ids = tok(t, return_tensors="pt")
                h = model(**ids, output_hidden_states=True).hidden_states[layer][0]
                out.append((h[-1] if pool == "final" else h.mean(0)).numpy())
        return np.array(out)
    return f


REPS = {
    "tfidf": rep_tfidf,
    "minilm": rep_minilm,
    "gpt2_L6": rep_causal_lm("gpt2", 6),
    "pythia160m_L6": rep_causal_lm("EleutherAI/pythia-160m", 6),
    "gpt2rand_L6": rep_causal_lm("gpt2", 6, random_init=True),
    "gpt2rand_mean": rep_causal_lm("gpt2", 6, pool="mean", random_init=True),
}


# ---------------------------------------------------------------- their recipe

def denoise_basis(acts, mean):
    X = acts - mean
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    cumvar = np.cumsum(S ** 2) / (S ** 2).sum()
    return Vt[:min(int(np.searchsorted(cumvar, DENOISE_VARIANCE)) + 1, len(Vt))]


def project_out(v, basis):
    return v - basis.T @ (basis @ v)


def similarity(acts, cats, variant):
    """acts[set] = (n, d) array; cats[set] = category per row. Returns the 10x10 matrix."""
    def rows(s, c=None):
        return acts[s] if c is None else acts[s][np.isin(cats[s], c)]

    if variant == "whitened":
        std = np.concatenate([rows(s, ["D"]) for s in POOL_SETS]).std(0)
        floor = np.median(std[std > 0]) * 1e-3
        std = np.where(std > floor, std, floor)
        acts = {s: a / std for s, a in acts.items()}

    neutral = np.concatenate([rows(s, ["D"]) for s in POOL_SETS])
    nmean = neutral.mean(0)
    if variant == "alldenoise":
        allc = np.concatenate([rows(s, CONTROL_CATS) for s in POOL_SETS])
        basis = denoise_basis(allc, allc.mean(0))
    else:
        basis = denoise_basis(neutral, nmean)

    def control_vec(a):
        return project_out(a.mean(0) - nmean, basis)

    def pain_vec(s):
        c = rows(s, CONTROL_CATS)
        return project_out(rows(s, PAIN_CATS).mean(0) - c.mean(0), denoise_basis(c, c.mean(0)))

    def pooled(c):
        return np.concatenate([rows(s, [c]) for s in POOL_SETS])

    v = {"S1_pain": pain_vec("S1_1P"), "S2_pain": pain_vec("S2_1P"),
         "Fear": control_vec(pooled("B")), "NegEmotion": control_vec(pooled("C1")),
         "NegWorld": control_vec(pooled("C2")), "BodySens": control_vec(pooled("E")),
         "Arousal": control_vec(rows("Arousal_1P")), "Random": control_vec(rows("Random_1P")),
         "Numb": control_vec(rows("Numb_1P")), "Sadness": control_vec(rows("SD_sadness_1P"))}
    V = np.array([v[c] / np.linalg.norm(v[c]) for c in CONCEPTS])
    return V @ V.T


# ---------------------------------------------------------------- comparison

def main():
    os.makedirs(OUT, exist_ok=True)
    sents = load_sentences()
    texts = sum((sents[s][0] for s in SETS), [])
    bounds = np.cumsum([0] + [len(sents[s][0]) for s in SETS])
    cats = {s: sents[s][1] for s in SETS}
    perms = all_perms()

    llm = {}
    for variant in VARIANTS:
        names, mats = load_variant(VARIANTS[variant])
        mean = mats.mean(0)
        loo = [np.corrcoef(mats[m][IU], np.delete(mats, m, 0).mean(0)[IU])[0, 1]
               for m in range(len(mats))]
        llm[variant] = (mean, np.min(loo), np.median(loo))

    lines = ["r = Pearson r of the 45 off-diagonal entries with the mean over the 25 LLMs;",
             "rank = rank of the true labelling among 10! when matching to that mean (1 = identified).",
             "LLM reference: each LLM vs the mean of the other 24, min / median r:"]
    lines += [f"  {v:10s} {llm[v][1]:.3f} / {llm[v][2]:.3f}" for v in VARIANTS]
    lines.append(f"\n{'representation':15s} " + " ".join(f"{v + ' r':>14s} {'rank':>7s}" for v in VARIANTS))
    for rep, fn in REPS.items():
        cache = os.path.join(OUT, f"baseline_acts_{rep}.npy")
        if os.path.exists(cache):
            X = np.load(cache)
        else:
            X = np.asarray(fn(texts), dtype=np.float64)
            np.save(cache, X.astype(np.float32))
        acts = {s: X[bounds[i]:bounds[i + 1]].astype(np.float64) for i, s in enumerate(SETS)}
        cells = []
        for variant in VARIANTS:
            M = similarity(acts, cats, variant)
            np.savetxt(os.path.join(OUT, f"baseline_{rep}_{variant}.csv"), M, fmt="%.4f",
                       delimiter=",", header=",".join(CONCEPTS))
            r = np.corrcoef(M[IU], llm[variant][0][IU])[0, 1]
            sc = scores(M, llm[variant][0], perms)
            rank = int((sc > sc[0]).sum()) + 1
            cells.append(f"{r:14.3f} {rank:7d}")
        lines.append(f"{rep:15s} " + " ".join(cells))
        print(lines[-1], flush=True)
    with open(os.path.join(OUT, "baseline.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
