# Representational-identity emergence

When during pretraining does convergent representational geometry crystallize —
and does standard similarity (CKA) see it?

We take two **independently seeded** pythia-160m runs and, at each of ~14
published checkpoints (`step0` → `step143000`), ask a label-free question: can
the identity of concept tokens be recovered from **relational structure alone**?
For each model we build the cosine-Gram matrix of a fixed 120-word concept set
and match the two Grams across seeds — no labels, only the shape of the
similarity structure (the method of `gram_matrix_decoder`).

Two direction families are compared:

- **activations** — the contextual residual-stream vector for each word at layer
  ⅔ depth (mean over 3 neutral templates); a stimulus representation.
- **unembedding** — the token's row in the output-embedding matrix; the static
  output interface.

## Findings

![emergence curve](outputs/emergence.png)

- **Identifiable geometry emerges as an early transition.** Both families sit at
  chance through ~step 64, then rise to near-ceiling by ~step 1000 — under 1% of
  the training run.
- **Staggered by family.** Activation identity crystallizes first (recoverable at
  step 256 while the unembedding is still at chance); the output interface catches
  up by step ~1000.
- **CKA is largely blind to it.** The unembedding's CKA is ~0.87 at random init
  and stays high across the entire window where identifiability goes from chance
  to ceiling — aggregate similarity does not track when identity becomes
  recoverable. (See `cka` in `outputs/curve.json`.)

Metric note: the plotted line is the robust **subset** metric — 200 random
8-token sub-Grams matched exhaustively over all 8! permutations, chance = 1/8.
The harder full-120 metric and CKA are also stored in `curve.json`.

Caveats: one architecture (160m), one seed pair. The transition *sharpness* and
the family ordering rest on a coarse step grid — densify steps 64–512 and add
seed pairs before making a phase-transition or ordering claim.

## Run

    pip install -r requirements.txt
    python -m emergence.sweep --mode extract    # downloads/caches checkpoints, ~minutes
    python -m emergence.sweep --mode curve       # -> outputs/curve.json
    python -m emergence.plot                     # -> outputs/emergence.png

Extraction caches one npz per (seed, step, family) under `outputs/sweep/`
(gitignored); re-runs skip cached checkpoints.
