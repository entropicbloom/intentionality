"""Automorphism analysis for orientation content.

If relations between neurons depended only on the orientation *difference*
(a circulant class-Gram), no relational decoder could tell 0 deg from 45 deg:
the dihedral group D_K would be an automorphism of the relational structure and
H(I|R) would be at least log2(2K) bits.  We measure how far the empirical
class-Gram is from circulant, how much posterior mass the dihedral relabelings
receive, and what happens to matching when the anisotropy is projected out.
Writes outputs/symmetry.json."""
from __future__ import annotations

import json
import sys

import numpy as np

from .config import K_ORI, N_SPLITS, OUT, SEED
from .data import Dataset
from .geometric import acc, acc_modulo, all_perms, argmin_tiebreak, perm_distances, stratified_half_split
from .relational import bin_orientation, class_gram, dihedral_group
from .run_geometric import gram_for


def circulant_projection(M):
    K = M.shape[0]
    out = np.zeros_like(M)
    for d in range(K):
        idx = [(c, (c + d) % K) for c in range(K)]
        v = np.mean([M[i, j] for i, j in idx])
        for i, j in idx:
            out[i, j] = v
    return out


def main(subs=None, n_splits=N_SPLITS // 2):
    ds = Dataset(); rng = np.random.default_rng(SEED)
    subs = subs or ["func_iv", "func_is", "struct_in", "struct_in_adp", "soma"]
    K = K_ORI; group = dihedral_group(K); P = all_perms(K)
    gidx = [int(np.flatnonzero((P == g).all(1))[0]) for g in group]
    results = {}
    for sub in subs:
        rows, G = gram_for(ds, sub, rng)
        ok = ds.ori_ok[rows]; keep = np.flatnonzero(ok)
        lab = bin_orientation(ds.ori[rows][keep], K); Gk = G[np.ix_(keep, keep)]
        M = class_gram(Gk, lab, K); C = circulant_projection(M)
        # a symmetric circulant matrix is invariant under the whole dihedral group
        # (rotations and the reflection c -> -c), so one projection covers D_K.
        var = ((M - M.mean()) ** 2).sum()
        res = dict(K=K, n=int(len(keep)),
                   frac_var_circulant=float(1 - ((M - C) ** 2).sum() / var),
                   class_gram=M.tolist(), circulant=C.tolist(), anisotropy=(M - C).tolist())
        # matching on raw vs. projected class-Grams
        r2 = np.random.default_rng(SEED + 3)
        stats = {"raw": [], "circulant": [], "null_raw": [], "null_circulant": []}; mass = np.zeros(len(group)); d_ids = []
        i_id = int(np.flatnonzero((P == np.arange(K)).all(1))[0])
        for _ in range(n_splits):
            A, B = stratified_half_split(lab, r2)
            for null in (False, True):
                lA, lB = (r2.permutation(lab[A]), r2.permutation(lab[B])) if null else (lab[A], lab[B])
                MA = class_gram(Gk[np.ix_(A, A)], lA, K); MB = class_gram(Gk[np.ix_(B, B)], lB, K)
                if not null:
                    d_ids.append(perm_distances(MA, MB)[i_id])
                for name, f in (("raw", lambda x: x), ("circulant", circulant_projection)):
                    Dp = perm_distances(f(MA), f(MB)); best = P[argmin_tiebreak(Dp, r2)].astype(int)
                    stats[("null_" if null else "") + name].append((acc(best, K), acc_modulo(best, group)))
        tau = float(np.sqrt(np.mean(np.square(d_ids)) / K ** 2))
        r2 = np.random.default_rng(SEED + 3)
        for _ in range(n_splits):
            A, B = stratified_half_split(lab, r2)
            MA = class_gram(Gk[np.ix_(A, A)], lab[A], K); MB = class_gram(Gk[np.ix_(B, B)], lab[B], K)
            D = perm_distances(MA, MB)
            logp = -(D.astype(float) ** 2) / (2 * tau ** 2); logp -= logp.max(); p = np.exp(logp); p /= p.sum()
            mass += p[gidx] / n_splits
        for name, v in stats.items():
            v = np.array(v); res[f"acc_{name}"] = float(v[:, 0].mean()); res[f"acc_mod_{name}"] = float(v[:, 1].mean())
        res["dihedral_posterior_mass"] = mass.tolist(); res["dihedral_elements"] = group.tolist()
        res["posterior_mass_identity"] = float(mass[[i for i, g in enumerate(group) if (g == np.arange(K)).all()][0]])
        res["posterior_mass_group_total"] = float(mass.sum())
        results[sub] = res
        print(f"[{sub}] n={len(keep)} circ.var={res['frac_var_circulant']:.3f} "
              f"acc raw={res['acc_raw']:.3f}(mod {res['acc_mod_raw']:.3f}) circ={res['acc_circulant']:.3f}(mod {res['acc_mod_circulant']:.3f}) "
              f"| null raw={res['acc_null_raw']:.3f}(mod {res['acc_mod_null_raw']:.3f}) circ={res['acc_null_circulant']:.3f}(mod {res['acc_mod_null_circulant']:.3f}) "
              f"| post mass id={res['posterior_mass_identity']:.3f} D_K={res['posterior_mass_group_total']:.3f}", flush=True)
    OUT.mkdir(exist_ok=True); json.dump(results, open(OUT / "symmetry.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
