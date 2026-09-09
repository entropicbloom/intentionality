"""Cross-animal class-level matching on Allen data.
Reference class-Gram from mouse A (labels known), test class-Gram from mouse B
(class identities hidden): does the relational signature of '45 deg' transfer
between animals?  Also within-mouse split-half for comparison, and a pooled
reference (all other mice) -> single test mouse."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from microns_ambiguity.geometric import acc, acc_modulo, all_perms, argmin_tiebreak, perm_distances, stratified_half_split
from microns_ambiguity.relational import bin_orientation, class_gram, cosine_gram, dihedral_group
from microns_ambiguity.data import zscore_rows
from allen.data import Allen, ROOT

OUT = os.path.join(ROOT, "allen", "outputs")


def z(M):
    return (M - M.mean()) / (M.std() + 1e-9)


def main(min_per_class=8, n_rep=20, movie="both", ori_source="sg", session="A"):
    ds = Allen(movie=movie, ori_source=ori_source, session=session); print({k: v for k, v in ds.summary().items() if "per_mouse" not in k}, flush=True)
    K = ds.K; G = cosine_gram(zscore_rows(ds.R)); lab = ds.ori_class; group = dihedral_group(K); P = all_perms(K)
    rng = np.random.default_rng(0)
    ok_m = {}
    for m in ds.mice:
        idx = np.flatnonzero((ds.mouse == m) & ds.ori_ok)
        if len(idx) and np.bincount(lab[idx], minlength=K).min() >= min_per_class:
            ok_m[m] = idx
    mice = list(ok_m); print("mice usable:", len(mice), flush=True)
    res = {"within": {}, "cross": {}, "pooled": {}}
    # within-mouse split-half
    for m in mice:
        idx = ok_m[m]; a = []
        for _ in range(n_rep):
            A, B = stratified_half_split(lab[idx], rng)
            MA = class_gram(G[np.ix_(idx[A], idx[A])], lab[idx[A]], K); MB = class_gram(G[np.ix_(idx[B], idx[B])], lab[idx[B]], K)
            D = perm_distances(MA, MB); best = P[argmin_tiebreak(D, rng)].astype(int); a.append((acc(best, K), acc_modulo(best, group)))
        res["within"][m] = dict(acc=float(np.mean([x[0] for x in a])), acc_mod=float(np.mean([x[1] for x in a])), n=int(len(idx)))
    # cross-mouse: full mouse A -> full mouse B
    Ms = {m: class_gram(G[np.ix_(ok_m[m], ok_m[m])], lab[ok_m[m]], K) for m in mice}
    cross = np.full((len(mice), len(mice)), np.nan); cross_mod = cross.copy()
    for i, a_ in enumerate(mice):
        for j, b_ in enumerate(mice):
            if i == j: continue
            D = perm_distances(z(Ms[a_]), z(Ms[b_])); best = P[argmin_tiebreak(D, rng)].astype(int)
            cross[i, j] = acc(best, K); cross_mod[i, j] = acc_modulo(best, group)
    res["cross"] = dict(mice=mice, acc=cross.tolist(), acc_mod=cross_mod.tolist(),
                        mean_acc=float(np.nanmean(cross)), mean_acc_mod=float(np.nanmean(cross_mod)))
    # pooled reference: all other mice -> test mouse (leave-one-mouse-out)
    for i, m in enumerate(mice):
        others = np.concatenate([ok_m[o] for o in mice if o != m])
        MA = class_gram(G[np.ix_(others, others)], lab[others], K)
        D = perm_distances(z(MA), z(Ms[m])); best = P[argmin_tiebreak(D, rng)].astype(int)
        # null: shuffle labels within the test mouse
        nl = []
        for _ in range(n_rep):
            sl = rng.permutation(lab[ok_m[m]]); Mn = class_gram(G[np.ix_(ok_m[m], ok_m[m])], sl, K)
            Dn = perm_distances(z(MA), z(Mn)); bn = P[argmin_tiebreak(Dn, rng)].astype(int); nl.append(acc(bn, K))
        res["pooled"][m] = dict(acc=acc(best, K), acc_mod=acc_modulo(best, group), null_acc=float(np.mean(nl)), n=int(len(ok_m[m])))
    res["summary"] = dict(within_mean=float(np.mean([v["acc"] for v in res["within"].values()])),
                          cross_mean=res["cross"]["mean_acc"], cross_mod_mean=res["cross"]["mean_acc_mod"],
                          pooled_mean=float(np.mean([v["acc"] for v in res["pooled"].values()])),
                          pooled_mod_mean=float(np.mean([v["acc_mod"] for v in res["pooled"].values()])),
                          pooled_null_mean=float(np.mean([v["null_acc"] for v in res["pooled"].values()])), chance=1 / K, K=K, movie=movie)
    print(json.dumps(res["summary"], indent=1))
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, f"geometric_{session}_{movie}_{ori_source}.json"), "w"))


if __name__ == "__main__":
    main(movie=sys.argv[1] if len(sys.argv) > 1 else "both", ori_source=sys.argv[2] if len(sys.argv) > 2 else "sg", session=sys.argv[3] if len(sys.argv) > 3 else "A")
