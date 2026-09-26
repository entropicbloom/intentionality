"""Label-free identification of concept directions across LLMs.

Data: per-model 10x10 cosine-similarity matrices between the concept
directions of Tagliabue, Dung & Berg, "The Pain Axis" (arXiv:2609.16247),
copied from github.com/valen-research/Pain-axis
(results/3.3_validation/cosine_similarity, MIT licence) into data/.

Question: given only one model's similarity matrix with its row/column
labels removed, can the labels be recovered by matching the matrix to the
average matrix of the other models? We search all 10! relabellings.

Because the multiset of off-diagonal entries does not change under
relabelling, maximising the inner product with the reference is the same
as maximising the Pearson correlation of the off-diagonal entries, so we
report that correlation.

Two held-out schemes:
  model   reference = mean of the other 24 models
  family  reference = mean of the models from other families
"""

import glob
import itertools
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")

CONCEPTS = ["S1_pain", "S2_pain", "Fear", "NegEmotion", "NegWorld",
            "BodySens", "Arousal", "Random", "Numb", "Sadness"]
K = len(CONCEPTS)
VARIANTS = {"raw": "similarity_", "alldenoise": "similarity_alldenoise_",
            "whitened": "similarity_whitened_"}
IU = np.triu_indices(K, 1)


def family(name):
    return name.split("_")[0]


def load_variant(prefix):
    files = []
    for f in sorted(glob.glob(os.path.join(DATA, prefix + "*.csv"))):
        base = os.path.basename(f)
        if "MEAN" in base:
            continue
        if prefix == "similarity_" and re.match(r"similarity_(alldenoise|whitened)_", base):
            continue
        files.append(f)
    names, mats = [], []
    for f in files:
        with open(f) as fh:
            header = fh.readline().strip().split(",")[1:]
        assert header == CONCEPTS, (f, header)
        names.append(re.sub(r"_L\d+$", "", os.path.basename(f)[len(prefix):-4]))
        mats.append(np.loadtxt(f, delimiter=",", skiprows=1, usecols=range(1, K + 1)))
    return names, np.array(mats)


def all_perms():
    return np.array(list(itertools.permutations(range(K))), dtype=np.int8)


def offdiag_centered(M):
    """Symmetric matrix with zero diagonal and off-diagonal entries centred and scaled
    so that the inner product of two such matrices (upper triangle) is a Pearson r."""
    v = M[IU] - M[IU].mean()
    v = v / np.linalg.norm(v)
    Z = np.zeros((K, K))
    Z[IU] = v
    return Z + Z.T


def scores(A, R, perms, chunk=400_000):
    """Pearson r between off-diagonals of A relabelled by each permutation and R.
    perms[n, i] = which row of A is assigned label i."""
    A, R = offdiag_centered(A), offdiag_centered(R)
    out = np.empty(len(perms))
    for s in range(0, len(perms), chunk):
        P = perms[s:s + chunk].astype(np.intp)
        Ap = A[P[:, :, None], P[:, None, :]]
        out[s:s + chunk] = np.einsum("nij,ij->n", Ap, R) / 2
    return out


def run(variant, scheme, perms):
    names, mats = load_variant(VARIANTS[variant])
    ident = 0  # itertools.permutations yields the identity first
    assert (perms[ident] == np.arange(K)).all()
    rows, confusion = [], np.zeros((K, K), int)
    for m, name in enumerate(names):
        if scheme == "model":
            ref_idx = [j for j in range(len(names)) if j != m]
        else:
            ref_idx = [j for j in range(len(names)) if family(names[j]) != family(name)]
        R = mats[ref_idx].mean(0)
        sc = scores(mats[m], R, perms)
        order = np.argsort(-sc)
        best = perms[order[0]]
        r_true = sc[ident]
        rank_true = int((sc > r_true).sum()) + 1
        r_other = sc[order[1]] if order[0] == ident else sc[order[0]]
        # cheapest single swap of the true labelling
        swaps = []
        for i, j in itertools.combinations(range(K), 2):
            p = np.arange(K)
            p[[i, j]] = p[[j, i]]
            idx = perm_index(p)
            swaps.append((r_true - sc[idx], CONCEPTS[i], CONCEPTS[j]))
        swaps.sort()
        for i in range(K):
            confusion[i, best[i]] += 1  # label i was given to the row of concept best[i]
        rows.append(dict(model=name, n_ref=len(ref_idx), r_true=r_true, rank_true=rank_true,
                         r_best_other=r_other, margin=r_true - r_other,
                         n_correct=int((best == np.arange(K)).sum()),
                         wrong=[f"{CONCEPTS[i]}<-{CONCEPTS[best[i]]}" for i in range(K) if best[i] != i],
                         cheapest_swap=swaps[0]))
    return names, rows, confusion


_FACT = [1] * (K + 1)
for _i in range(1, K + 1):
    _FACT[_i] = _FACT[_i - 1] * _i


def perm_index(p):
    """Lexicographic index of permutation p (matches itertools.permutations order)."""
    p = list(p)
    idx, avail = 0, list(range(K))
    for pos, v in enumerate(p):
        r = avail.index(v)
        idx += r * _FACT[K - 1 - pos]
        avail.pop(r)
    return idx


def main():
    os.makedirs(OUT, exist_ok=True)
    perms = all_perms()
    assert perm_index(perms[123456]) == 123456
    lines = []
    for scheme in ["model", "family"]:
        for variant in VARIANTS:
            names, rows, conf = run(variant, scheme, perms)
            n_id = sum(r["rank_true"] == 1 for r in rows)
            per_concept = np.diag(conf)
            head = (f"\n=== held out: {scheme}, variant: {variant} ===\n"
                    f"true labelling ranked first for {n_id}/{len(rows)} models "
                    f"(out of {len(perms):,} labellings); "
                    f"concepts correct {per_concept.sum()}/{len(rows) * K}\n"
                    "per concept: " + ", ".join(f"{c} {n}" for c, n in zip(CONCEPTS, per_concept)))
            lines.append(head)
            lines.append(f"{'model':32s} {'r_true':>6s} {'rank':>5s} {'margin':>7s} {'ok':>3s}  "
                         f"cheapest swap (drop in r)   errors")
            for r in rows:
                d, a, b = r["cheapest_swap"]
                lines.append(f"{r['model']:32s} {r['r_true']:6.3f} {r['rank_true']:5d} "
                             f"{r['margin']:+7.3f} {r['n_correct']:3d}  "
                             f"{a}/{b} ({d:.3f}){'':4s} {' '.join(r['wrong'])}")
            np.savetxt(os.path.join(OUT, f"confusion_{scheme}_{variant}.csv"), conf, fmt="%d",
                       delimiter=",", header=",".join(CONCEPTS))
            print("\n".join(lines[-len(rows) - 2:]), flush=True)
    with open(os.path.join(OUT, "identify.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
