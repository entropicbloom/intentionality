"""Markdown tables for the per-neuron decoder runs (outputs/decoder2.json)."""
import json
import numpy as np
from .config import OUT


def main():
    d = json.load(open(OUT / "decoder2.json"))
    print("\n### Early-stopped headline runs (512 neurons, 2.2M params, 12 ep x 2500 pops, 3 seeds)\n")
    print("| substrate | content | metric | mean ± sd (n) | best epoch |")
    print("|---|---|---|---|---|")
    groups = {}
    for k, v in d.items():
        if k.startswith("es_") and "s" in k.split("_")[-1]:
            groups.setdefault(k.rsplit("_s", 1)[0], []).append(v)
    for g, vs in sorted(groups.items()):
        m = "acc" if "acc" in vs[0] else "r2"; vals = [v[m] for v in vs]
        print(f"| {vs[0]['sub']} | {vs[0]['con']} | {m} | {np.mean(vals):.3f} ± {np.std(vals):.3f} ({len(vals)}) | {[v.get('best_epoch') for v in vs]} |")
    print("\n### All runs\n")
    print("| tag | substrate | content | n | params | epochs x pops | metric | value | avg32 | early stop |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, v in d.items():
        m = "acc" if "acc" in v else "r2"
        print(f"| {k} | {v['sub']} | {v['con']} | {v['n']} | {v['params']/1e6:.1f}M | {v['epochs']}x{v['pops_per_epoch']} | {m} | {v[m]:.3f} | {v.get(m+'_avg32', float('nan')):.3f} | {'ep'+str(v['best_epoch']) if 'best_epoch' in v else '-'} |")


if __name__ == "__main__":
    main()
