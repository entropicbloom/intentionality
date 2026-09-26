"""Layer sweep: r between the 10x10 matrix of small real LMs (final token, each layer)
and the 25-LLM mean, for the three variants. Picking the best layer by this r is optimistic."""
import sys, os, numpy as np, torch
sys.path.insert(0, '.')
from baseline import *
from transformers import AutoModel, AutoTokenizer
sents = load_sentences(); texts = sum((sents[s][0] for s in SETS), [])
bounds = np.cumsum([0] + [len(sents[s][0]) for s in SETS]); cats = {s: sents[s][1] for s in SETS}
llm = {v: load_variant(VARIANTS[v])[1].mean(0) for v in VARIANTS}
for name in ["gpt2", "EleutherAI/pythia-160m", "gpt2-medium"]:
    tok = AutoTokenizer.from_pretrained(name); model = AutoModel.from_pretrained(name).eval()
    H = []
    with torch.no_grad():
        for t in texts:
            hs = model(**tok(t, return_tensors="pt"), output_hidden_states=True).hidden_states
            H.append(torch.stack([h[0, -1] for h in hs]).numpy())
    H = np.array(H, dtype=np.float64)
    for L in range(1, H.shape[1]):
        acts = {s: H[bounds[i]:bounds[i+1], L] for i, s in enumerate(SETS)}
        print(name, L, " ".join(f"{v} {np.corrcoef(similarity(acts, cats, v)[IU], llm[v][IU])[0,1]:.3f}" for v in VARIANTS), flush=True)
