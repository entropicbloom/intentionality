# Representational ambiguity in mouse visual cortex (MICrONS)

Does the relational structure of a real cortical population fix what its
neurons represent, and up to which symmetries?  This applies the
relational-decoding framework of arXiv:2512.11000 (§4.4 there names the
biological test) to cortex: take neurons whose content is known
(receptive-field position, preferred orientation), hide the labels, and ask
whether the content can be recovered from the *relations* among neurons alone —
from synaptic connectivity and from functional covariation — first at the class
level with a labelled reference, then per neuron with no reference at all.

## Data

Ding, Fahey, Papadopoulos et al. 2025 functional-connectomics release of the
MICrONS mm³ volume (`functional_connectomics/node_and_edge_properties/v1` on
the public `bossdb-open-data` bucket; one mouse, 13 two-photon scans):

* **12,894 coregistered neurons** with in-vivo preferred orientation and gOSI
  (Monet stimuli), digital-twin receptive-field centre (STA fit, stimulus
  coordinates in [-1, 1]), soma position, layer, area (V1 / RL / AL / LM),
  trial-averaged **in-vivo responses to the shared oracle natural-movie clips**
  (120 samples) and **digital-twin responses to a shared movie** (4,999 samples).
* **148 proofread presynaptic axons** with 8,128 synapses onto 4,811 of the
  coregistered neurons, plus 287,243 **axon-dendrite-proximity (ADP)** pairs:
  postsynaptic dendrites the axon passed within reach of but did *not*
  contact.  Connected ∪ ADP is the axon's *potential* connectivity.

Sanity check that in-vivo oracle responses are comparable across scans:
signal correlation of pairs with overlapping RFs (< 0.1 apart) is 0.062 within
a scan and 0.041 across scans, versus ≈ 0 for far pairs in both cases; same
pattern for orientation (0.078 / 0.065 for Δori < 15°, ≈ -0.01 for Δori > 75°).

## Relational substrates

Each substrate is a neuron × feature matrix; the relational structure is its
cosine Gram (for responses: z-scored rows, so cosine = signal correlation).

| name | rows | relation between two neurons |
|---|---|---|
| `struct_in` | 4,811 postsynaptic neurons | overlap of the sets of proofread axons that synapse onto them (148-dim, synapse counts) |
| `struct_in_adp` | same neurons | overlap of the sets of proofread axons that *could* have contacted them (potential connectivity) |
| `struct_in_rewired` | same neurons | synapses redrawn uniformly within each axon's potential targets, out-degree and synapse-count multiset preserved (proximity-constrained null connectome) |
| `struct_out` / `_adp` / `_rewired` | the 148 axons | overlap of their synaptic (resp. potential, rewired) target sets over all 12,894 neurons |
| `func_iv` | all 12,894 | in-vivo signal correlation (oracle clips) |
| `func_is` | all 12,894 | digital-twin signal correlation (shared movie) |
| `soma` | all 12,894 | Gaussian kernel of soma distance (σ = 100 µm): the purely spatial null |

## Contents (what a neuron represents)

* `ori` — in-vivo preferred orientation, gOSI ≥ 0.25 (n = 5,287), binned into
  K = 8 classes of 22.5° centred on multiples of 22.5°.
* `rf` — receptive-field centre, digital-twin test correlation ≥ 0.2
  (n = 11,326), binned into a 3 × 3 quantile grid (K = 9) or used as 2-D
  regression target.
* `rf_resid` — RF centre minus the local retinotopic map (mean RF of the 50
  nearest same-area somata; the map explains 51–54 % of RF variance).  What
  remains is the local scatter of retinotopy, which cortical location cannot
  predict by construction.
* Anatomical positive controls: `soma_xz` (tangential cortical position,
  3 × 3), `depth` (K = 4), `layer` (K = 3), `area` (V1 / RL / AL, K = 3).

## Protocols

**P1 — geometric matching (paper §3.1.2).**  Neurons are split into two
stratified halves.  The reference half's K × K *class-Gram* (mean relation
between neurons of class a and class b) carries the labels; the test half's
class-Gram is presented with class identities hidden and all K! relabelings
are searched for the smallest Frobenius distance.  Accuracy = fraction of
classes assigned correctly; "hit" = whole permutation correct.  200 random
splits.  Additions to the paper's protocol:

* *Posterior over relabelings*, p(g) ∝ exp(−d_g² / 2τ²), with τ the per-entry
  split-half noise of the correct relabeling.  Its per-class marginal entropy
  is a direct estimate of H(I | R, C) in bits; `ARS post` = 1 − H / log₂K.
  The paper's Fano bound (`ARS Fano`) is reported alongside.
* *Accuracy modulo a symmetry group* — rotations and reflections of the
  orientation circle (D₈), flips/transposes of the RF grid — the relabelings
  a relation that depends only on content *difference* can never resolve.
* Two nulls.  `null indep`: labels shuffled independently inside each half
  (pure chance, 1/K).  `null fixed`: labels shuffled once, i.e. *arbitrary but
  fixed* neuron groups.  The second is conservative: fixed groups can stay
  identifiable across halves through degree heterogeneity alone, so it
  measures content-specific signal beyond "some particular set of neurons".

**P2 — learned decoder with hidden population labels (paper §2.1.4).**  A
population of *n* neurons is sampled; its *n × n* Gram is fed row-wise, as
tokens without positional encoding, to a transformer that predicts each
token's content.  Training populations come from one half of the neurons,
validation populations from the other, so the decoder must learn
population-geometry regularities that transfer to unseen neurons.  The first
version (`decoder.py`: 48 neurons, token-0 supervision) and its iteration
(`decoder2.py`: on-the-fly Grams, dense supervision, optional relational
attention bias, early stopping on a held-out slice of the training neurons)
are both reported in §4.  Ablation `target_only` removes every relation not
involving the target (the paper's local-vs-global control); `shuffled` trains
on permuted labels; `anchored` (diagnostic only) gives the other tokens their
true labels.

**P3 — reference-free recovery.**  Kernel PCA of the test population's own
Gram; content is read out from the top-2 axes up to rotation/reflection/scale
(orthogonal Procrustes fitted on half the neurons, scored on the other half),
or linearly from the top-10 / top-50 axes.  No other population, no reference
class-Gram, no labels in the embedding.

**Transfer.**  Reference class-Gram from substrate X on one half, test
class-Gram from substrate Y on the other half (z-scored), the paper's
cross-architecture test recast as cross-substrate.

**Symmetry.**  For orientation, the class-Gram is projected onto its
circulant part (relation depends only on Δori; a symmetric circulant matrix is
invariant under the whole dihedral group D₈).  The variance it explains, the
matching accuracy that survives the projection (with random tie-breaking among
the rotations it cannot distinguish), and the posterior mass on the 16
dihedral relabelings quantify how much of orientation identity is fixed by
anisotropy rather than by the difference structure.

## Run

    uv venv --python 3.13 .venv && uv pip install numpy pandas scipy scikit-learn matplotlib torch pyarrow
    # data: see data/ (downloaded from s3://bossdb-open-data/iarpa_microns/minnie/functional_data/...)
    python -m microns_ambiguity.run_geometric
    python -m microns_ambiguity.run_decoder func_iv,func_is,struct_in,struct_in_adp,struct_in_rewired,soma ori,rf full 2
    python -m microns_ambiguity.run_spectral
    python -m microns_ambiguity.transfer
    python -m microns_ambiguity.symmetry
    python -m microns_ambiguity.plots && python -m microns_ambiguity.summarize
    # per-neuron decoder iteration (see data/sweep*.sh for the exact runs)
    python -m microns_ambiguity.run_decoder2 <tag> func_is rf n=512 dim=256 layers=4 rel_bias=1 device=mps batch=8 epochs=20 pops_per_epoch=4000 early_stop=0.15 pca=1
    python -m microns_ambiguity.plot_decoder2 && python -m microns_ambiguity.summarize_decoder2


## Results

All numbers: 200 random half-splits unless noted; "null" is the pure-chance
null (labels shuffled independently in the two halves) and "fixed" the
arbitrary-fixed-group null.  Full tables: `python -m microns_ambiguity.summarize`.

### 1. Class identity from relational structure (geometric matching)

![geometric matching](outputs/geometric_main.png)

| substrate | orientation (K=8) | RF position (K=9) | RF minus local map (K=9) | cortical pos. (K=9) | layer (K=3) | area (K=3) |
|---|---|---|---|---|---|---|
| synaptic (shared inputs, n=4,811) | 0.14 (chance) | **0.83** | 0.16 (chance) | 1.00 | 0.97 | 1.00 |
| proximity (potential inputs) | 0.31 | **0.99** | 0.16 (chance)_ADP | 1.00 | 1.00 | 1.00 |
| rewired within reach (null connectome) | 0.20 | **0.32** | 0.16 (chance)_REWIRED | 0.92 | 0.70 | 0.88 |
| functional, in vivo (n=12,894) | **1.00** | 1.00 | 1.00 | 1.00 | 0.94 | 1.00 |
| functional, digital twin | 0.93 (1.00 mod D₈) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| soma distance (spatial null) | 0.21 (0.91 mod D₈) | 1.00 | 0.64 | 1.00 | 1.00 | 1.00 |
| chance / fixed-group null | 0.13 / 0.13–0.19 | 0.11 / 0.08–0.18 | 0.11 / 0.06–0.12 | 0.11 / 0.06–0.10 | 0.33 / 0.15–0.45 | 0.33 / 0.19–0.67 |

* **Functional relational structure fixes every content we tested.**  From
  signal correlations alone, with class identities hidden, the 8 orientation
  classes and the 9 RF-position classes are recovered exactly in 100 % of
  splits (in vivo), with posterior entropy ≈ 0 bits: H(I | R, C) = 0 in the
  paper's terms.  The permutation-distance picture is the one the paper shows
  for dropout networks — the true labelling is separated from all 40,319
  alternatives (`perm_distances.png`, left).

* **Synaptic relational structure fixes *where* a neuron looks, not *which
  orientation* it prefers.**  Shared-input overlap identifies RF-position
  classes at 83 % (posterior entropy 0.24 of 3.17 bits) but orientation at
  chance.  The failure is not lack of like-to-like wiring: after projecting the
  orientation class-Gram onto its Δori-only (circulant) part, synaptic
  relations identify orientation *modulo rotations/reflections* at 87 %
  (null 45 %; §3).  There is a Δori-dependent structure in who shares inputs
  with whom — but it is symmetric, so it cannot say which class is 0°.

* **Most, but not all, of the RF content in synapses is cortical location.**
  Potential connectivity (which axons *could* have contacted a dendrite) does
  better than actual synapses (99 % vs 83 %), and so does raw soma distance
  (100 %).  Cortical position is trivially recoverable from any of these
  (100 %).  The proximity-constrained rewired connectome — same axons, same
  out-degrees, partners redrawn among each axon's reachable dendrites — is
  the fair comparison at matched sparsity: 32 % vs 83 %.  Synaptic
  *specificity* therefore adds RF information beyond proximity, consistent
  with Ding et al.'s like-to-like result, here stated as identifiability of
  content rather than as a connection-probability ratio.  The residual-RF
  column (RF minus the mean RF of the 50 nearest same-area somata; the local
  map explains 51–54 % of RF variance) is the direct test of whether any of
  this survives once cortical location is removed: it does not.  Synaptic relations
  identify the local RF scatter at chance (0.16; proximity 0.27, mostly
  residual spatial structure, since soma distance alone still gives 0.64),
  while functional relations identify it perfectly (1.00 in vivo and in the
  twin).  At this sparsity, synaptic relational structure carries RF content
  at the level of the retinotopic map, not of the individual neuron; functional
  relational structure carries both.

* **The 148 proofread axons (target-set overlap) are too few for K > 4**
  (`geometric_axons.png`); only area (84 %) and cortical position (52 %) rise
  clearly above the nulls.  Their fixed-group null for area is 79 % under
  potential connectivity — an arbitrary fixed set of 50 axons is almost as
  identifiable as a real area from where their axons go.  This is why the
  fixed-group null is reported everywhere: a fixed set of specific neurons is
  consistently "itself" across halves through degree heterogeneity alone, and
  it stays above chance even at n = 12,894 for unbalanced classes (area,
  in vivo: 0.67).  The content-specific claim is always relative to it.

### 2. Cross-substrate transfer (the paper's cross-architecture test)

![transfer](outputs/transfer.png)

Reference class-Gram from one substrate, test class-Gram from another (disjoint
neurons, z-scored).  RF-position geometry transfers across every substrate:
functional → synaptic 0.93, synaptic → functional 0.95, proximity → functional
0.95, soma → digital twin 0.95.  Retinotopy imposes one relational geometry on
who covaries with whom, who shares inputs with whom, and who sits next to
whom.  Orientation transfers between in-vivo and digital-twin covariance only
*modulo* D₈ (accuracy 0.24, 1.00 modulo rotation/reflection): both carry the
same Δori structure but different anisotropies, so the twin's absolute
orientation frame is not the animal's.  Orientation does not transfer to or
from any structural substrate.

### 3. What fixes absolute orientation: anisotropy, not the difference rule

![symmetry](outputs/symmetry.png)

| substrate | circulant share of class-Gram variance | accuracy raw (mod D₈) | after circulant projection (mod D₈) | null (mod D₈) |
|---|---|---|---|---|
| functional, in vivo | 0.71 | 1.00 (1.00) | 0.11 (1.00) | 0.13 (0.41) |
| functional, digital twin | 0.81 | 0.94 (1.00) | 0.10 (1.00) | 0.13 (0.39) |
| synaptic | 0.21 | 0.16 (0.42) | 0.13 (0.87) | 0.12 (0.40) |
| proximity | 0.20 | 0.24 (0.45) | 0.14 (1.00) | 0.14 (0.39) |
| soma distance | 0.55 | 0.20 (0.91) | 0.11 (1.00) | 0.12 (0.39) |

A relation that depends only on Δori is invariant under the 16 rotations and
reflections of the orientation circle; such a structure can fix orientation
only up to D₈, leaving log₂16 = 4 bits of ambiguity.  Projecting the
functional class-Gram onto its circulant part does exactly that: absolute
accuracy collapses to chance (0.11) while accuracy modulo D₈ stays at 1.00.
The 29 % of variance that is *not* circulant — a cardinal bias whose strength
differs by area (V1: 25 % of neurons within ±11° of horizontal; AL: 29 % near
vertical) — is what makes absolute orientation identifiable, and the posterior
puts mass 1.00 on the identity and 0 on the other 15 dihedral relabelings.
Soma distance identifies orientation modulo D₈ at 0.91 and puts 0.61 of its
posterior mass on the dihedral group: the area-wise cardinal bias, laid out
along the cortical axis, gives the *difference* structure of orientation a
spatial signature but not its absolute frame.  This is the empirical version
of the automorphism argument: residual ambiguity = the symmetry group of the
relational structure, and it is broken by anisotropy, not by more relations.

### 4. Per-neuron decoding with no labels and no reference

The paper's transformer decoder, applied to sampled populations: a population
of *n* neurons is presented as its *n × n* correlation matrix, rows as tokens,
and the decoder predicts each token's content.  Training populations come from
one half of the neurons, validation populations from the other; nothing but
the correlation matrix enters the input.  The first version (48 neurons,
0.3M parameters, one supervised token per population) was at the class-prior
baseline for everything except a small orientation gain.  Iterating on it:

![decoder scaling](outputs/decoder2_scaling.png)

| change | orientation (twin) | RF (x, y) (twin) | note |
|---|---|---|---|
| original: 48 neurons, 0.3M, token-0 supervision | 0.305 (in vivo) | 0.007 | class prior 0.255 / R² 0 |
| standardized inputs | 0.303 | 0.012 | no effect |
| all tokens supervised (dense) | 0.325 | 0.015 | small gain, no extra compute |
| on-the-fly Grams, 128–1024 neurons, 0.3M | 0.34–0.36 | 0.02–0.07 | size helps RF slowly, orientation saturates |
| relational attention bias | = | = | no effect |
| test-time averaging over 32 populations | +0.00–0.01 | +0.00–0.01 | predictions already consistent |
| **2.2M parameters** (width 256, 4 layers), 512 neurons | **0.416** | **0.279** | the largest single gain; RF from 0.07 |
| 7.2M parameters | 0.420 | 0.449 (20 epochs, early-stopped) | RF still improving with capacity |
| 20 epochs × 4000 populations | 0.426 peak → 0.367 end | 0.437 | orientation memorises the ~2,600 training neurons; RF does not |
| frame-free target (distance from RF centre), 0.3M | – | 0.288 vs 0.046 for (x, y) | the small model recovers relative position, not the frame |

**Headline numbers** (512 neurons, 2.2M parameters, early-stopped on a
held-out 20 % of the *training* neurons, 3 seeds; the validation neurons never
influence model selection):

| content | in vivo | twin | labelled-reference ceiling (in vivo / twin) | baseline |
|---|---|---|---|---|
| orientation, 8 classes | 0.368 ± 0.007 | 0.406 ± 0.010 | 0.41 / 0.49 | 0.255 (majority class) |
| RF (x, y), R² | 0.200 ± 0.003 | 0.341 ± 0.005 | 0.32 / 0.67 | 0 |
| RF distance from centre, R² | 0.241 (1 seed) | 0.351 ± 0.001 | – | 0 |
| best single run (7.2M) | – | 0.420 / 0.449 | | |

**GPU runs (one L40S, 8 hours).**  Larger models, 512-neuron populations,
batch 64, 5,000 populations per epoch, best epoch restored from a checkpoint,
selection metric averaged over 4 population covers of the 20 % selection
slice.  The label-free relation augmentation that helps on Allen (below) was
also tried: recomputing each training population's Gram from a random half
of the feature dimensions, and zeroing 20 % of Gram entries.

| content, substrate | 2.2M (laptop, 3 seeds) | 17M | 57M | augmented 17M | labelled reference |
|---|---|---|---|---|---|
| orientation, twin | 0.406 ± 0.010 | 0.421 | (queued, not run) | 0.399 (half the PCs), 0.409 (Gram dropout) | 0.49 |
| orientation, in vivo | 0.368 ± 0.007 | **0.388** | – | 0.374 (half the stimulus bins) | 0.41 |
| RF (x, y), twin | 0.341 ± 0.005 | 0.466 / 0.487 (2 seeds; 0.476 / 0.495 averaged) | **0.507** (0.512 averaged) | 0.061 (half the PCs) | 0.67 |
| RF (x, y), in vivo | 0.200 ± 0.003 | (plain run cut by budget) | – | **0.247** (half the stimulus bins) | 0.32 |

Capacity is still the lever: twin RF goes 0.34 → 0.47 → 0.51 from 2.2M to
57M parameters, and the 17M model lifts in-vivo orientation to within two
points of the labelled linear reference.  Every large run memorises the
training neurons (final training loss 0.001–0.06) and is selected at epoch
4–17, so all of the gain comes from early stopping on a better model, not
from longer training.  Augmentation does not help MICrONS: subsampling the
512 twin principal components destroys the Gram (PCs are not exchangeable
conditions), and halving the 120 in-vivo stimulus bins costs 1–2 points on
orientation.  The exception is in-vivo RF, where the augmented 17M model
reaches 0.247 against 0.200; the un-augmented 17M comparison was cut when the
budget ran out, so how much of that is capacity and how much augmentation is
open.  Seed spread at 17M is about 2 points (twin RF 0.466 vs 0.487).

The *labelled-reference ceiling* is a ridge readout from each validation
neuron's correlations to all ~6,000 labelled training neurons — the same
correlations, plus a fully labelled anchor set.  In vivo, the label-free
decoder reaches 90 % of that ceiling for orientation and 63 % for RF; on
twin correlations 83 % and 51 % (67 % with the 7.2M model).  Orientation's
ceiling is itself capped by label noise: in-vivo and digital-twin preferred
orientations agree on only 70 % of neurons at 8 classes.

**What limited the first version, in order.**  (i) Capacity: the 0.3M model
could not find the anisotropy that fixes the absolute RF frame — it recovered
distance from centre at R² 0.29 while (x, y) stayed at 0.05; the 2.2M model
recovers (x, y) at 0.28–0.34 and its predictions are in the true frame
(orientation accuracy modulo D₈ equals plain accuracy for every run).
(ii) Data quality: twin correlations have ~6× less per-pair noise than 120-bin
in-vivo correlations, and the labelled ceilings show the same gap
independently of any decoder (RF 0.32 vs 0.67).  (iii) Population size: real
but saturating by 512 neurons for both contents once capacity is adequate.
(iv) Regularisation: orientation (5,287 labelled neurons) memorises within a
few epochs and needs early stopping; RF (11,326) does not; dropout 0.25 hurt.

![overfitting](outputs/decoder2_overfit.png)

**Reading.**  A decoder that sees only the correlation matrix of ~500 cortical
neurons, with no labels and no reference population, assigns orientation
preference at most of what a fully labelled readout achieves and places
receptive fields on the screen at R² 0.34–0.45.  Recovery follows the symmetry
structure of the relations: frame-free content first, absolute content once
the model is large enough to use the weak anisotropies.  This is the per-neuron
form of the class-level result in §1–3, and it matches the MNIST
input-neuron subset curve (R² rising from 0.23 at 4 neurons to 0.84 at 784).

### 5. Reference-free recovery

![spectral](outputs/spectral.png)

| substrate | RF: top-2 up to rotation/scale | RF: linear, 50 axes | orientation: linear, 50 axes | cortical position: linear, 50 axes |
|---|---|---|---|---|
| synaptic | 0.00 | 0.06 | 0.01 | 0.26 |
| proximity | 0.07 | 0.37 | 0.05 | 0.87 |
| rewired | 0.00 | 0.01 | 0.00 | 0.13 |
| functional, in vivo | 0.01 | 0.28 | 0.38 | 0.32 |
| functional, digital twin | 0.00 | 0.52 | 0.55 | 0.39 |
| soma distance | 0.11 | 0.50 | 0.06 | 0.97 |

Held-out R² of content read out from the kernel-PCA embedding of the test
population's own Gram (no reference population, no labels in the embedding;
shuffled-label nulls are ≤ 0.01).  The leading two axes of no substrate are a
retinotopic map up to a similarity transform (R² ≤ 0.11): unlike the pixel
covariance of an image, the dominant relational axes in cortex are not visual
space (for soma distance they are cortical space, R² 0.97 for position, and
retinotopy is only their correlate).  Content is present *linearly* in the top
50 axes — RF at 0.52, orientation at 0.55 from digital-twin covariance —
i.e. it is in the relational structure but not as its principal geometry.
Reference-free recovery is therefore weaker than reference-based recovery by a
wide margin here, the opposite of the MNIST input layer, and the "up to
automorphism" reading needs a labelled reference or a learned decoder to pick
out the content-bearing subspace.

## What this says

1. **Relational structure fixes content in a real cortical population.**
   At the class level, orientation and RF classes are recovered exactly from
   signal correlations with identities hidden (H(I|R,C) ≈ 0 bits); per neuron,
   with no labels and no reference, a decoder recovers orientation at 83–90 %
   of the labelled ceiling and RF position at R² 0.34–0.45.

2. **Content is fixed up to the automorphism group of the relational
   structure, and anisotropy is what breaks it.**  A Δori-only class-Gram
   identifies orientation only modulo rotation/reflection; the cardinal bias
   fixes the frame.  The per-neuron decoder shows the same thing dynamically:
   small models recover frame-free RF distance long before absolute position.

3. **Synaptic relations, at connectome scale, carry RF at the map level and
   orientation only as a difference structure.**  Not a proxy for functional
   relations in this volume.

4. **What limits per-neuron recovery is capacity and per-pair noise, not the
   idea.**  The gap to the labelled ceiling closed by a factor of ~10 across
   the iteration; the remaining gap is largest where per-pair noise is largest.

Open, and needed before this is a paper: a second animal for the class-level
symmetry result (within one animal, two halves share every statistic; the
claim that "45°" has a relational signature needs a cross-animal reference —
the Allen data below supply a partial one: orientation crosses animals
there through shared-stimulus grating correlations, not through natural-movie
or within-circuit ones); in-vivo per-pair noise, which caps
everything in vivo; and seeds on the larger-model runs.

## Allen Brain Observatory: a second dataset, 33 mice

Visual Coding 2P (allensdk), VISp excitatory containers with ≥ 150
orientation-labelled cells: 36 containers, 33 mice, 9,281 cells (session A)
and 8,367 cells (session C).  Relations: signal correlations of trial-averaged
dF/F to natural movies that are identical across all mice (session A: movies
one + three, 4,500 bins; session C: one + two, 1,800 bins).  Contents:
preferred orientation from static gratings (6 classes, cell table), receptive
field centre from locally sparse noise (session C; 1,504 cells with a
significant RF, 23–112 per mouse).  Code in `allen/`.

**Regimes.**  Two independent choices define an Allen decoder run.  The
*split* decides where the test neurons come from: the training animals
(each animal's cells are halved into training and test neurons; this tests
generalisation to new neurons only) or held-out animals (whole animals are
kept out of training and model selection; this tests generalisation to new
brains).  The *population* decides what a single training or test sample
is: 128–1024 neurons drawn from one animal, so the Gram is a within-circuit
correlation matrix, or drawn from several animals, so most Gram entries are
between-animal correlations that exist only because all mice saw the same
stimuli.  The code names:

| | single-animal populations | mixed populations |
|---|---|---|
| test neurons from the training animals | `within` | `pooledwithin` |
| test neurons from held-out animals | `cross` | `pooledcross` |

MICrONS is one animal, so every MICrONS result is in the `within` cell.  The
cross-animal claims below are `cross` (strictest) or `pooledcross`, always
named.

**Orientation is nearly absent from natural-movie correlations in this
dataset.**  Same-orientation pairs correlate 0.005 more than orthogonal pairs
(MICrONS in vivo: 0.09); a fully labelled ridge readout reaches 0.236 against
a 0.198 majority rate; the class-level test is at chance within and across
mice (33 mice, K = 6); the label-free decoder is at the majority rate in every
regime (within-mouse 0.171, cross-mouse 0.168, pooled cross-mouse 0.205).
Removing shared population fluctuations (mean or top principal components) and
coarser temporal binning do not change this.  Static- and drifting-grating
orientations of the same cells agree within 15° for only 45–67 %, so labels
are part of it, but the ceiling says the correlations themselves carry little.
The relational signature of orientation that MICrONS shows is therefore not a
generic property of natural-movie covariance; it may depend on the stimulus
set, the deconvolved responses, or the twin-cleaned labels there.

**With grating relations, orientation does cross animals, but only through
between-animal correlations.**  Session A also contains drifting gratings (8
directions × 5 temporal frequencies, identical across mice).  Building the
relations from the 40 blank-subtracted condition means instead of the movies
(labels unchanged: static gratings from session B, so labels and relations use
different stimuli) puts orientation back into the substrate: same-orientation
pairs correlate 0.05 more than orthogonal pairs within a mouse and 0.044 more
across mice; a labelled ridge readout reaches 0.35 within and 0.35 ± 0.06
leave-one-mouse-out (majority 0.20).  Using the 40 × 60 condition time courses
instead gives the same ridge ceiling but four times weaker pair correlations,
so the condition means are the substrate below.

| orientation, K = 6, 33 mice | movie relations | grating relations | null |
|---|---|---|---|
| class-level, within-mouse split-half | 0.167 | 0.206 | 0.167 |
| class-level, one mouse → another | 0.164 | 0.181 | 0.167 |
| class-level, pooled reference → held-out mouse | 0.192 | 0.258 | 0.173 (shuffle) |
| same, modulo D6 | 0.515 | 0.631 | 0.5 |
| labelled ridge, leave-one-mouse-out | 0.236 | 0.346 | 0.198 (majority) |
| label-free decoder, populations within one held-out mouse (`cross`) | 0.168 | 0.180 (0.244 averaged) | 0.19 |
| label-free decoder, `pooledcross`, 128 neurons, 12 epochs, first protocol, 3 mouse splits | 0.205 | 0.229 ± 0.020 (0.245 ± 0.005 averaged) | 0.19 |
| same, fixed selection (best-epoch checkpoint, metric averaged over 4 covers), 30 epochs, 3 seeds on one split | | 0.269–0.281 | |
| same, 256 neurons, 3 seeds | | 0.286–0.294 | |
| GPU: 256 neurons, 17M params, 60 epochs | | 0.288 (0.303 averaged) | |
| GPU: 256 neurons, 85M params | | 0.300 (0.296) | |
| GPU: 512 neurons, 85M params, 100 epochs | | 0.310 (0.305) | |
| **GPU: 256 neurons, 17M params, Gram from a random half of the 40 conditions** | | **0.310 (0.306)**, selected at epoch 35 of 60 | |
| label-free decoder, `pooledwithin` (test neurons from the training animals, mixed populations), 128 / 256 neurons | | 0.277 (0.307 averaged) / 0.281 (0.288) | |

The null for the decoder rows is the training-set majority class applied to the
test mice (0.187–0.188 over the three splits); chance is 0.167.  Every grating
number is above its movie counterpart, and the pooled-cross decoder is a
label-free orientation readout across animals: trained on 24 mice, selected on
6, it assigns orientation to neurons of 9 unseen mice at 0.31 against 0.19,
three quarters of the labelled linear reference's excess over baseline.

What moved the number, in order.  (i) *Model selection*: the first protocol
selected on single-population accuracy of 5 mice and stopped runs at epoch
0–5; restoring the best-epoch weights and averaging the selection metric over
4 population covers moved 0.25 → 0.28 at no other change.  (ii) *Population
256 instead of 128*: +1–2 points; 512 and 1024 add nothing (0.283, 0.291 with
the 17M model).  (iii) *Capacity*: 2M → 17M → 85M gives 0.27 → 0.29 → 0.30
without augmentation, selected at epoch 13–16 of 60 with training loss near
0.25, i.e. the model memorises the ~6,700 training neurons.  (iv) *Label-free
augmentation*: recomputing each training population's Gram from a random
subset of the 40 conditions (test Grams use all 40).  With the 2M model 75 %
of conditions helps (0.272 → 0.292), 50 % hurts (0.247) and 30 % breaks it
(0.214); with the 17M model 50 % gives the best run, 0.310, selected at epoch
35 instead of 13.  The augmentation works only with capacity: the small
model cannot fit the noisier training Grams, the large one can and stops
memorising.  Gram-entry dropout (20 %) gives +1 point alone and nothing on
top of subsampling.  Dropout 0.2 instead of 0.1 does nothing.  (v) Averaging
each neuron's prediction over 32 sampled populations: +0–1.5 points, and it
stabilises the score across seeds (spread ±0.005 instead of ±0.02).

The `pooledwithin` row is the control for the population axis: with the same
mixed populations but test neurons from the *training* animals, the score is
0.277 (0.307 averaged) at 128 neurons and 0.281 (0.288) at 256, the same as
with held-out animals.  Orientation
transfer to new brains costs nothing; the whole difference between the
`cross` and `pooledcross` rows is single-animal versus mixed populations,
not generalisation.

What the two regimes measure differs, and the difference is the finding.  In
`cross` all 128 neurons come from one animal, so the Gram is a within-circuit
correlation matrix and the decoder must transfer across animals' internal
matrices; that stays at the baseline, for movies and for gratings.  In
`pooledcross` most Gram entries relate a neuron in one mouse to a neuron in
another, which exist only because all mice saw the same conditions, and they
measure how similar two tuning profiles are.  The decoder still never sees the
condition indices, only correlations, so the absolute frame (which class is
0°) must come from the population's relational structure, as in MICrONS.  So
in the Allen data: a neuron's orientation is readable, without labels, from
where it sits in a correlation geometry that spans animals, and that geometry
is shared enough across mice to transfer; it is *not* readable from its
relations inside its own network alone.  For RF position (next paragraph) both
readings transfer.

**Receptive-field position is present and crosses animals.**  Correlation
falls with RF distance within a mouse (0.127 at < 5° to 0.068 beyond 40°) and
across mice (0.049 to 0.011): two neurons in different animals covary when
they look at the same part of the screen.

| test (session C, absolute screen coordinates) | R² |
|---|---|
| labelled ridge, random halves | 0.27 |
| labelled ridge, leave-one-mouse-out (place a held-out mouse's neurons) | 0.20 |
| label-free decoder, `pooledcross` (held-out mice, mixed populations), 128 neurons | 0.11 (0.20 averaged over 32 populations) |
| same, 256 neurons; GPU 256 neurons 2M / 17M / augmented / 512 / 1024 neurons | 0.18 (**0.23** averaged); 0.22 / 0.21 / 0.22 / 0.21 / 0.09 |
| **label-free decoder, `cross` (held-out mice, single-animal populations)**, 128 neurons, 2M | **0.273** (0.267 averaged); 0.281 with 85 % condition subsampling; **0.286** with 800-population epochs and lr 5e-4 (peak at epoch 1) |
| label-free decoder, `pooledwithin` (training mice, mixed populations), 128 neurons | 0.155 (0.165 averaged); trains on half as many RF labels |
| same, 17M | 0.176 |
| relative RF (within the mouse's own patch), any method | ≈ 0 |

The cross-animal decoder recovers *where on the screen* a held-out mouse's
neurons look, above the labelled cross-animal reference (0.20), using only
the correlations among neurons of held-out animals.  For RF the two
population regimes rank the other way round from orientation: single-animal
populations (`cross`, 0.27–0.28) beat mixed ones (`pooledcross`, 0.22), so
the within-circuit relation is the better carrier of screen position while
the between-animal relation is the only carrier of orientation.  The `cross`
RF runs peak at epoch 0 (about 80 gradient steps) and decay to 0.10 by epoch
40 as the training loss falls from 0.9 to 0.15: the content is learned
immediately and then the ~800 RF-labelled training cells are memorised.
Capacity hurts here (17M: 0.18) and neither augmentation nor larger mixed
populations help (1024 neurons: 0.09).  With 800-population epochs and a
lower learning rate the peak is resolved at epoch 1 (about 50 steps): 0.286
plain, 0.278 with mild augmentation, 0.260 with strong augmentation.  The
same fine schedule does not help mixed populations (0.15 vs 0.22).  What no method
recovers is the layout *within* a mouse: with ~45 RF-labelled cells per mouse
the internal retinotopic map is not estimable here, so the per-mouse grid
class test is at its null and relative-RF readouts are at zero.

All held-out-animal regimes select the model on held-out *mice* (25 % of the
training mice), never on test mice; `pooledwithin` selects on a 25 % slice of
the training neurons.  The `within` cell (training animals, single-animal
populations) failed under the new selection code (the per-mouse selection
slices are smaller than a population) and is not reported.

## Caveats

* One animal, one volume, one proofreading set (148 axons; postsynaptic
  neurons see ≈ 1.7 proofread inputs on average).  The structural substrates
  are extremely sparse; the rewired null is the only density-matched control.
* RF centres and digital-twin responses are model-derived (Wang et al. 2025);
  orientation and in-vivo covariance are not.  Orientation labels are gated
  at gOSI ≥ 0.25 (5,287 of 12,894 neurons).
* In-vivo oracle responses are compared across 13 scans; same-scan pairs are
  ≈ 50 % more correlated than cross-scan pairs at matched RF distance, a
  session effect that the stratified splits do not remove but that cannot
  create orientation- or RF-specific class structure.
* The class-level protocol is the paper's; it hides class *identities* but
  keeps class *membership* (which neurons belong together).  The per-neuron
  decoder (§4) removes that too.
* Twin correlations come from a model fitted to the same recordings; the
  in-vivo results are the claim about cortex and the twin results show what
  cleaner data would give.  Single seeds for the 7.2M-parameter runs; early
  stopping for the 1024-neuron and 7.2M orientation runs used a selection slice
  smaller than one population and is therefore contaminated by training
  neurons (last-epoch values are reported alongside).
* ARS from the permutation posterior assumes iid Gaussian entry noise on the
  class-Gram; τ is calibrated from the split-half distance of the correct
  labelling.  It agrees with the Fano bound where both are informative.
