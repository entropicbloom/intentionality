"""Print markdown tables from outputs/*.json for the README."""
import json
import numpy as np
from .config import OUT


def load(n):
    p = OUT / f"{n}.json"; return json.load(open(p)) if p.exists() else {}


def main():
    geo, dec, spec, tr, sym = (load(n) for n in ["geometric", "decoder", "spectral", "transfer", "symmetry"])
    if geo:
        print("\n### Geometric matching (class identity from relational structure)\n")
        print("| substrate | content | K | n | acc | acc mod sym | exact hit | H(class) bits / max | ARS post | ARS Fano | null fixed | null indep |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k, r in geo.items():
            s, c = k.split("|")
            print(f"| {s} | {c} | {r['K']} | {r['n']} | {r['acc']:.2f} ± {r['acc_sd']:.2f} | {r['acc_mod']:.2f} | {r['hit']:.2f} | "
                  f"{r['H_class_bits']:.2f} / {r['H_class_max']:.2f} | {r['ars_posterior']:.2f} | {r['ars_fano']:.2f} | "
                  f"{r['null_fixed']['acc']:.2f} | {r['null_indep']['acc']:.2f} |")
    if dec:
        print("\n### Learned decoder (hidden population labels)\n")
        print("| substrate | content | variant | n | metric | value (mean over seeds) | ARS |")
        print("|---|---|---|---|---|---|---|")
        groups = {}
        for k, v in dec.items():
            s, c, var, seed = k.split("|"); groups.setdefault((s, c, var), []).append(v)
        for (s, c, var), vs in groups.items():
            m = "acc" if vs[0]["task"] == "class" else "r2"
            vals = [v[m] for v in vs]
            print(f"| {s} | {c} | {var} | {vs[0]['n']} | {m} | {np.mean(vals):.3f} ± {np.std(vals):.3f} (n={len(vals)}) | {np.mean([v['ars'] for v in vs]):.2f} |")
    if spec:
        print("\n### Reference-free (kernel PCA of the test population's Gram)\n")
        print("| substrate | content | n | top-2 up to rotation/scale R² | linear 10 axes R² | linear 50 axes R² | null (50) |")
        print("|---|---|---|---|---|---|---|")
        for k, r in spec.items():
            s, c = k.split("|")
            print(f"| {s} | {c} | {r['n']} | {r['procrustes_top2']:.3f} | {r['linear_m10']:.3f} | {r['linear_m50']:.3f} | {r['null_linear_m50']:.3f} |")
    if tr:
        print("\n### Cross-substrate transfer (reference -> test), class accuracy\n")
        for con in ["rf", "ori"]:
            subs = sorted({k.split("|")[1].split("->")[0] for k in tr if k.startswith(con)})
            print(f"\n{con}:\n\n| ref \\ test | " + " | ".join(subs) + " |"); print("|---" * (len(subs) + 1) + "|")
            for X in subs:
                print(f"| {X} | " + " | ".join(f"{tr[f'{con}|{X}->{Y}']['acc']:.2f}" for Y in subs) + " |")
    if sym:
        print("\n### Orientation symmetry analysis\n")
        print("| substrate | n | circulant var. frac | acc raw (mod D8) | acc after circulant projection (mod D8) | null raw (mod) | null circulant (mod) | posterior mass identity | mass on D8 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for s, r in sym.items():
            print(f"| {s} | {r['n']} | {r['frac_var_circulant']:.2f} | {r['acc_raw']:.2f} ({r['acc_mod_raw']:.2f}) | {r['acc_circulant']:.2f} ({r['acc_mod_circulant']:.2f}) | "
                  f"{r['acc_null_raw']:.2f} ({r['acc_mod_null_raw']:.2f}) | {r['acc_null_circulant']:.2f} ({r['acc_mod_null_circulant']:.2f}) | "
                  f"{r['posterior_mass_identity']:.2f} | {r['posterior_mass_group_total']:.2f} |")


if __name__ == "__main__":
    main()
