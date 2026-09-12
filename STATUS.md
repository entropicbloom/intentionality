# Study log: relational decoding in mouse visual cortex (MICrONS + Allen)

Results tables: `microns_ambiguity/README.md`. Paper: `microns_ambiguity/paper/main.tex`. Numbers: `microns_ambiguity/outputs/decoder2.json`, `allen/outputs/decoder.json`.

## In flight

In flight (2026-09-12 08:00 UTC): raw-activity reference decoders on pod 5 `vqnpxaxi140zwu` (A40 48 GB, $0.49/h, ssh `-p 22065 root@69.30.85.40`, scratchpad `pod5.env`), queue `microns_ambiguity/sweeps/queue_pod5.sh`, log `logs/queue_pod5.log`, marker `QUEUE_POD5_DONE`: tags `a_{iv,is}_17M_{ori,rf}` (tokens from response vectors, plain attention, no Gram; `input_mode=act rel_bias=0`), `ab_{iv,is}_17M_ori` (activity tokens + Gram attention bias), `lin_{iv,is}_{ori,rf}` (layers=0: linear per-neuron readout of the response vector). All save predictions. When done: pull `decoder2.json` + `preds`, terminate the pod, add the numbers to the controls appendix (reference decoder paragraph + table rows), README §4b and the discussion (how much relations alone lose vs the activity). Previously (03:00 UTC): Both RunPod pods are terminated (balance $5.30). Everything below under "Done 2026-09-11/12" is merged into `allen/outputs/decoder.json`, `microns_ambiguity/outputs/decoder2.json`, the `preds/` directories (git-ignored), the figures in `microns_ambiguity/outputs/paper/`, the paper (`microns_ambiguity/paper/main.tex`, 16 pages, copy on the Desktop), `paper_draft.md` and the README. Open items that need new code: Allen single-animal diversity control (mixed populations from a pool as small as one mouse), synthetic von Mises experiment (cardinal count bias vs correlation anisotropy), raw-activity reference decoder (how much relations alone lose), appendix B/C tables, affiliation and repo URL.

## Log of decisions and results (2026-09-11/12, newest last)

Reporting decision for the regression results (2026-09-11): main text uses mean absolute angular error in degrees only, with a 45° chance line (uniform errors on 0–90° average 45°) and one predicted-vs-true scatter per dataset. Allen panels also mark the ~7.5° label-quantisation floor (30° label grid). Within-15° fraction goes to the appendix table only; no normalised score.

Reviewer follow-ups queued on the same pod (2026-09-11, `microns_ambiguity/sweeps/sweep_reviewer.sh`, log `logs/reviewer.log`, marker `SWEEP_REVIEWER_DONE`; waits for `SWEEP_CIRC_MICRONS_DONE`): extra MICrONS neuron splits with saved predictions (`c_is_17M_sp1/sp2`, `c_iv_17M_sp1/sp2`), RF with saved predictions (`r_iv_17M_s0`, `r_is_17M_s0`), cross-stimulus disjoint-bin test (`c_iv_17M_bins`, `c_is_17M_bins`; new `bins=1` flag -> `F_eval` in `train()`), architecture ablations (`c_is_17M_statsbias` row_proj=0 rel_bias=1, `c_is_17M_statsonly` row_proj=0 rel_bias=0). Predictions land in `microns_ambiguity/outputs/preds/<tag>.npz` (idx, P, y, scan, area); analyse with `python -m microns_ambiguity.scan_decomposition <tags>` (within-scan / within-area R², area-prior error). Also queued: `allen/extra_circ_plain.sh` -> `c_pc_2M_plain_sp1/sp2` (marker `EXTRA_CIRC_PLAIN_DONE`, log `logs/extra_circ_plain.log`) so the Allen 2x2 can use plain 2M runs in every cell and augmentation can leave the paper (it gives no gain in angular error: 17M cf50 35.6 vs plain 36.5, 2M plain 35.2).

Pod 3 (`bofozxleuvk0d5`, 4090) stopped when the balance hit zero on 2026-09-11 ~21:00 UTC with the follow-up sweep partly done; its host had no free GPU to resume on. Replacement pod 4: `sofz6n1s1x8yq8`, A100 80GB PCIe, $1.59/h, ssh `-p 18803 root@216.81.151.3` (scratchpad `pod4.env`), created by `pod4_bootstrap.sh` from `create_pod.json` + `data_bundle.tgz` + `code_bundle.tgz`. Queue `queue_pod4.sh` (log `logs/queue_pod4.log`, marker `QUEUE_POD4_DONE`): population-size runs, then the reviewer follow-ups in reverse order; the four plain-2M Allen runs failed on macOS `._*` files in the tar and are re-queued in `queue_pod4_allen.sh` (log `logs/queue_pod4_allen.log`, marker `QUEUE_POD4_ALLEN_DONE`). Pod 3 can be terminated once pod 4's results are pulled (its volume only holds duplicates). The three Allen tags `c_pc_17M_n512`, `c_pc_57M_cf50`, `c_pc_17M_n1024` in the local decoder.json are reconstructed from log lines (flag `reconstructed_from_log`), no history.

Follow-up results (pod 4, 2026-09-12, circular regression, 17M twin unless noted; chance 45°): stats-only (row_proj=0, rel_bias=0) 37.7°; equivariant (row_proj=0, rel_bias=1) 19.6° vs standard 20.2° -> the order-dependent row projection is unnecessary. Cross-stimulus (disjoint halves of the stimulus bins for training vs test Grams): twin 21.2° (raw 4999 bins, 2500 per half) vs 20.2°; in vivo 34.2° vs 25.4°, and the same-half 60-bin control gives 25.3°, so the in vivo cost is real and comes from the small stimulus sample (60 bins), not from fewer bins per se. Neuron splits: in vivo sp2 24.6° (sp0 seeds 25.4±0.5). Population size in error: in vivo 0.3M 27.4/26.9/27.8/29.5° at 128/256/512/1024 (n1024 with sel_reps=1); twin 2.2M 20.9/20.9/21.6° at 256/512/1024.

Balance / prior checks on saved predictions (`microns_ambiguity/balance_check.py`, `scan_decomposition.py`): MICrONS labels are uneven (0° 0.19, 90° 0.25, obliques 0.06–0.10 of neurons); best constant predictor 41.3°, scan-prior predictor 39.0°, area-prior 38.7°, so priors buy ≤ 6° of the 25° gain. Balanced error (mean over 8 true-orientation bins): twin 24.5° (raw 20.3°), in vivo 27.9° (raw 24.6°). Per-scan / per-area rotation removal changes the error by < 0.2°; per-area errors 19–22° (twin). Pattern on both datasets: cardinal bins decoded well (twin 0°: 11°, 90°: 12°; Allen 0°: 25°), oblique bins near chance (twin 45°/135°: 37–39°; Allen 120°/150°: 45°), and 63–72 % of predictions land on 0° or 90°. Allen labels are flat (0.14–0.20), best constant 43.9°, balanced 36.4° vs raw 35.7°. Cross-stimulus in vivo per area: V1 33.4°, RL 37.0°, AL 42.7°.

RF scan decomposition (`scan_decomposition.py r_is_17M_s0`, twin, R² 0.499 absolute): within-scan 0.498, within-area 0.468; a scan-mean predictor scores 0.022, an area-mean predictor 0.087. MICrONS RF is a neuron-level within-scan readout, unlike Allen. In vivo (r_iv_17M_s0, R² 0.249 absolute): within-scan 0.253, within-area 0.221. Written into the paper's controls appendix.

Horizontal-vs-vertical readout (post hoc, from saved preds; keep next to the angular error, never instead): nearest cardinal axis correct for 87 % (twin) / 79 % (in vivo) of all neurons, chance 56 %; for neurons within ±15° of an axis (56 % of them) 94 % / 86 %; Allen pooledcross (3 splits) 62 % of all cells, 68 % of the cardinal-labelled ones (chance 51–52 %). Four-way 0/45/90/135: twin 0.66, in vivo 0.59 (majority 0.40); Allen 0.32 (majority 0.33). Figs 2 and 5 now carry these coarse readouts as their right panel instead of the scatters (user request 2026-09-12): obliques unresolved on Allen. Label distributions: `microns_ambiguity/outputs/label_distribution.png` (MICrONS bimodal at 0°/90°, 44 % of neurons within ±11°; V1 peaks at 90°, AL at 0°; Allen flat 14–19 %).

Smoke-test fact: only 2 % of the MICrONS RF label variance is between scans, 9 % between areas, so a scan-identity readout bounds at R² ~0.02.## In-flight

Class-Gram residual checks (`microns_ambiguity/residual_check.py`, 2026-09-12): the non-circulant residual holds 0.29 (in vivo) / 0.19 (twin) of the class-Gram variance; it correlates 0.94 between random neuron halves, 0.84 ± 0.11 (in vivo, 60 bins per half) / 0.99 (twin, 2,499 per half) between disjoint stimulus-bin halves, and 0.99 with the residual on class-balanced subsamples (330 neurons per class, same fraction). So the anisotropy is signal, not a stimulus-sample artefact, and not neuron-count bias. In vivo the neighbour correlation peaks at 90° (0.073) with 0° only average (0.033); twin peaks at 0°/90°/157° (~0.05). Synthetic generator note: a random 120-frame stimulus sample or random frame amplitudes alone produce a residual of the same size (0.2–0.3 range in neighbour correlation) in an otherwise isotropic model; the isotropic baseline needs even orientation coverage and constant amplitude (range 0.01).

Per-area class-Gram residuals (twin / in vivo, `residual_check` helpers): residual fraction V1 0.16 / 0.30, RL 0.32 / 0.31, AL 0.72 / 0.68; residual correlation between areas V1–RL 0.77 / 0.78, V1–AL 0.31 / 0.22, RL–AL 0.63 / 0.53. Twin neighbour-correlation profiles: V1 highest at 0° and 90° (0.054, 0.057), AL highest at 158°/0° (0.195, 0.166) and lowest at 45° (0.059). The cross-area transfer gap follows the residual similarity: V1->RL 1.6°, V1->AL 27°.

## Run ledger (circular-regression era, 2026-09-11 onward)

Every run writes config + per-epoch history + final metrics to `microns_ambiguity/outputs/decoder2.json` (MICrONS) or `allen/outputs/decoder.json` (Allen) under its tag; per-neuron predictions (idx, P, y[, scan, area]) go to `<...>/outputs/preds/<tag>.npz`; sweep scripts in `microns_ambiguity/sweeps/` and `allen/`; pod logs in `microns_ambiguity/outputs/logs_pod4/` (pod 5 logs to be added). Chance for orientation error is 45°.

| experiment | tags | purpose | status |
|---|---|---|---|
| MICrONS standard (17M, 512 neurons, 3 seeds) | `c_{is,iv}_17M_s{0,1,2}` | headline orientation error | done: twin 20.2±0.4, in vivo 25.4±0.5 |
| MICrONS capacity | `c_{is,iv}_2M_s*`, `c_is_57M_s0`, `c_{is,iv}_03M_s0` | Fig A1 | done |
| MICrONS population size | `c_iv_03M_n{128,256,1024}`, `c_is_2M_n{256,1024}` | Fig A2 | done |
| MICrONS neuron splits (preds saved) | `c_{is,iv}_17M_sp{1,2}` | split uncertainty; scan/balance/coarse readouts | done |
| MICrONS RF (preds saved) | `r_{is,iv}_17M_s0` | scan decomposition | done: within-scan R² = absolute |
| cross-stimulus (Gram) | `c_{is,iv}_17M_bins`, `c_iv_17M_bins60` | disjoint stimulus bins; 60-bin control | done: twin 21.2, in vivo 34.2 / 25.3 |
| architecture ablations | `c_is_17M_statsbias`, `c_is_17M_statsonly` | equivariant decoder; stats only | done: 19.6 / 37.7 |
| raw-activity references | `a_{is,iv}_17M_{ori,rf}`, `ab_{is,iv}_17M_ori`, `lin_{is,iv}_{ori,rf}` | ceiling: activity tokens; + Gram bias; linear per-neuron | done: see README §4b |
| cross-stimulus (activity) | `a_{is,iv}_17M_bins` | does the activity decoder learn relations implicitly | done: in vivo 49.5° (39.7° frame-corrected; Gram 34.2°), twin 41.0° (Gram 21.2°) -> no, it reads stimulus-aligned features |
| Allen 2x2 (2.2M plain) | `c_{wi,pw,cr}_2M_sp{0,1,2}`, `c_pc_2M_plain{,_sp1,_sp2,_s1,_s2}` | Fig 5, Table 2, Fig A4 | done |
| Allen sweeps | `c_pc_17M_cf50_sp0_s*`, `c_pc_17M_cf50_sp{1,2}`, `c_pc_{2M,17M,57M}_plain`, `c_pc_2M_cf{75,50,30}`, `c_pc_17M_cf85`, `c_pc_57M_cf50`, `c_pc_17M_n{512,1024}` | capacity / augmentation / population size | done (3 tags reconstructed from logs) |
| Allen activity decoder | `act_pc_2M_sp{0,1,2}`, `act_cr_2M_sp{0,1,2}` | fair cross-animal comparison; single-animal cell with activity tokens | done: pooledcross 32.4 / 33.7 / 34.3° (Gram 35.2 / 37.0 / 38.4°); cross (single-animal) 32.8 / 33.6 / 34.9° (Gram 43.1 / 43.1 / 42.7°) -> the single-animal failure is about within-animal relations, not the neurons |
| rotated labels | `m_rot2_{is,iv}` (alignment-invariant loss: each population's predictions aligned to its targets by the best rotation/reflection before the error) | what the symmetric part alone gives per neuron (report err_modD) | done: twin 25.4° raw / 19.9° after alignment (standard 20.2°), in vivo 30.9° / 25.5° (standard 25.4°): the circulant part carries the full per-neuron structure, the residual adds only the frame; raw well below 45° = the decoder anchors a frame on the data unasked. `m_rot_is` (43.3 / 40.6°) used a pointwise loss on rotated targets, whose optimum is degenerate: invalid, ignore; `m_rot_iv` killed |
| orientation-balanced loss | `m_w_{is,iv}` | are obliques recoverable when weighted | twin done: raw 21.4°, balanced 25.3° (standard 20.3 / 24.5); oblique bins 34–41° unchanged, 71 % of predictions still cardinal -> obliques are not recoverable, the readout limit is in the data; in vivo: raw 25.8°, balanced 29.1° (standard 24.6 / 27.9), oblique bins 36–38° unchanged, predictions less piled on cardinals (0.20 / 0.29 vs 0.31 / 0.39) but no better |
| synthetic von Mises populations | `syn_c{0,1}s{0,1}t{0,1}` in `microns_ambiguity/outputs/synthetic.json` (`microns_ambiguity/synthetic.py`) | which switch fixes the frame: count bias / cardinal sharpening / stimulus bias; baseline circulant to 0.01 | seed 0 (raw / frame-corrected): none 56.1 / 23.8 (structure without frame, as predicted); stim 15.1 / 15.1; sharp 50.3 / 26.0; sharp+stim 4.3 / 4.2; count 38.4 / 32.9; count+stim 46.8 / 43.2 (odd: structure lost too, needs a seed); count+sharp 29.4 / 26.7; all three 45.9 / 43.6 (degenerate like count+stim: every count+stim cell collapses at an early epoch, systematic or not to be seen with seed 1). Second seed `syn_*_d1` queued at the end of the chain |
| stimulus-agnostic activity decoder | `bp_{iv,is}_17M_{ori,rf}` (`input_mode=act bin_perm=1`) | upper bound on all bin-permutation-invariant statistics (pairwise and higher); gap to the Gram decoder = content beyond pairwise relations (plus per-neuron marginals) | done: ori 27.1° / 24.5° (Gram 25.4 / 20.2), RF 0.04 / 0.08 (Gram 0.23 / 0.48) -> no content beyond pairwise; explicit Gram beats implicit relations, strongly for RF |
| bin-permuted per-neuron control | `bp0_{iv,is}_ori` (`input_mode=act bin_perm=1 layers=0`) | shuffled vector, no population context: if it matches `bp_*`, the gain over the Gram is intrinsic per-neuron statistics, not relations | done: 40.3° / 40.2° (near chance) -> the bp_* readout comes from relations computed across the population |
| within-scan populations (MICrONS) | `ws_{is,iv}_17M` (`within_scan=1`, n=128, sel_reps=1) | the MICrONS analogue of Allen's single-animal cell: is orientation readable from relations inside one scan | queued (pod 5, end of chain) |
| Allen training-pool size | `tp_pc_2M_n128_p{140,280,560,all}` (`train_pool=N`, pooledcross split 0, n=128) | reviewer's diversity confound: mixed populations from a pool as small as one mouse | queued (pod 5, end of chain) |
| synthetic variants | `syn_sharp8`, `syn_gain`, `syn_gain_stim` (`sharp_k=8`, `gain=1`) | which cortical anisotropies can fix the frame at all | queued (pod 5, end of chain) |
| cross-area transfer (twin) | `m_{V1toRL,V1toV1}_is` (n=512), `m_{V1toAL,ALtoV1,V1toV1}_is_n128`, `m_{RLtoV1,RLtoRL,V1toRL,V1toV1}_is_n256` | is the frame source area-specific (raw vs err_modD gap) | **V1->AL (n=128): 58.2° raw / 31.6° frame-corrected (gap 27°, raw worse than chance = wrong frame); V1->V1 n=128: 22.2 / 22.2; V1->V1 n=512: 21.0 / 21.0; V1->RL n=512: 26.2 / 24.6.** The frame anchor learned on V1 misplaces AL, whose anisotropy peaks at 0° instead of 90°: the frame source is the area-specific anisotropy. `m_ALtoV1` did not run (AL training pool < 128 after the selection slice); `m_RLtoV1_is`, `m_RLtoRL_is` at n=512 crashed (RL pool ~450) and rerun at n=256 at the end of the chain |

## Related work notes (for the paper; all references verified against publisher records on 2026-09-11)

ML neighbours (verified 2026-09-11, now cited in the intro): NeuPRINT (Mi, Le, He, Shlizerman, Sümbül, NeurIPS 2023): time-invariant per-neuron embedding from population dynamics, decodes cell type. NuCLR (Arora, Lachi, Knight, Azabou, Richards, Hurwitz, Siegle, Dyer, arXiv 2512.01199, Dec 2025): contrastive self-supervised neuron identity from population context, permutation-equivariant spatiotemporal transformer on binned spike trains (20 ms bins); tasks are cell type (Allen Neuropixels optotagging, Bugeon) and brain region (IBL, Steinmetz); zero-shot to unseen animals; no tuning targets, no MICrONS, no relations-only input. POYO (Azabou et al., NeurIPS 2023): one decoder across sessions/animals via spike tokens with unit embeddings. Positioning: same genre (per-neuron property from population context, transfer to unseen animals), different input (relations only, stimulus discarded), different target (stimulus content), plus the symmetry explanation. Natural reviewer request: a raw-activity decoder as reference for how much relations alone lose.
The intro lineage (paragraph 2 of `paper/main.tex`), one line each on what the source claims and what we take from it:
- Shepard & Chipman 1970 (Cogn. Psychol. 1:1-17): second-order isomorphism; a representation need not resemble its
  object, the relations among representations should mirror the relations among objects.
- Edelman 1998 (BBS 21(4):449-467): representation is representation of similarities; content is a position in a
  similarity structure ("chorus of prototypes"). The cleanest statement of the idea we build on.
- Kriegeskorte, Mur & Bandettini 2008 (Front. Syst. Neurosci. 2:4); Kriegeskorte & Kievit 2013 (TICS 17(8):401-412):
  RSA; similarity structure is the level at which brains, models and species are compared, independent of coordinates.
- Haxby et al. 2011 (Neuron 72(2):404-416): hyperalignment; a shared representational space across subjects built from
  response structure under a shared movie. Closest precedent for our across-brain regime, at subject level.
- Sucholutsky et al. 2023 (arXiv:2310.13018); Huh et al. 2024 (ICML, PMLR 235:20617-20642, "Position: the platonic
  representation hypothesis"): representational alignment across systems via relations among representations.
- Tsodyks et al. 1999 (Science 286:1943-1946); Kenet et al. 2003 (Nature 425:954-956): spontaneous activity in the dark
  reproduces evoked orientation maps; a single neuron's spike-triggered population pattern matches the map of its
  preferred orientation. THE key precedent. What it lacks: reading the tuning out required the evoked map (a labelled
  reference); correlations sort neurons by orientation and order the groups on the circle but do not say which group is
  vertical. Our additions: (1) no reference (the anisotropy pins the frame; symmetry section), (2) per-neuron, learned,
  thousands of identified neurons, (3) across animals. Our setting is weaker than theirs in one respect: relations from
  responses to a shared stimulus with its identity discarded, not spontaneous activity (open question noted below).
- Berkes et al. 2011 (Science 331:83-87): spontaneous activity statistics converge to evoked statistics over
  development. NOT an orientation-map result; cite only for "matches the statistics of evoked activity" (fixed).
- Ko et al. 2011 (Nature 473:87-91); Ding et al. 2025 (Nature 640(8058):459-469): like-to-like connectivity.
- Safaie et al. 2023 (Nature 623:765-771): latent dynamics preserved across animals, aligned without matched neurons.
- Lyre 2022 (Neurosci. Conscious. 2022(1):niac012): neurophenomenal structuralism, qualities individuated by position in
  quality spaces; carries the structuralist claim. Kleiner & Ludwig 2024 (Synthese 203(3):89): formal definition of a
  mathematical structure of experience; cite as definition, not as the claim (fixed).
- Oizumi, Lim & Kanai 2026 (PNAS Nexus 5(9):pgag261): equivariant encoders -> rigid group-inherited orbits ("attributes",
  universal) + plastic quotient ("signatures", individual). Our circulant part = rigid, anisotropy = what pins the frame.
- Kawakita et al. 2024 (Sci. Rep. 14:15917): Gromov-Wasserstein unsupervised alignment of similarity structures; the
  continuous relaxation of our exhaustive permutation matching.
- Lässig 2025 (arXiv:2512.11000, q-bio.NC): our framework paper; cited after the lineage, never first.
Decisions: no consciousness framing; the framework paper and Oizumi appear in one discussion paragraph; the
spontaneous-activity papers get a sentence in the discussion as the precedent.

## Possible next steps
- Spontaneous-activity relations (Allen sessions have a grey-screen block): Kenet 2003 / Tsodyks 1999 show spontaneous
  correlations group neurons by orientation (the circulant part). The open question for a label-free readout is whether
  the anisotropy (cardinal bias, area differences) that pins the absolute frame is also present in spontaneous
  correlations. Within-animal cells only (no shared time axis across mice). Needs a re-download to extract the traces.
- Ensembles over seeds on a shared split (predictions saved in allen/outputs/preds; `python -m allen.ensemble tag1 tag2`).
- Labelled reference ("ceiling") is not settled: ridge is one linear labelled model and the decoder
  beat it on RF.  Open checks (deferred, user wants label-free decoding first): ridge accuracy vs
  OSI gate and vs same-session (drifting-grating) labels to separate label noise from substrate
  noise; nonlinear labelled readouts (kNN, MLP, labelled-token transformer); zero the Gram
  diagonal in the ridge fit.  Call it "labelled reference" in the paper, not ceiling.
- Decoder: average populations across scans in MICrONS (targets the session effect).
- Allen: seeds for the 256-neuron / 30-epoch / 7.2M runs; larger selection set (sel_frac) so
  512-neuron runs do not early-stop at epoch 2.
- Paper: write up.  Structure agreed: lead with cortex, method re-introduced, no consciousness framing.

## Regimes (keep the two axes apart)
Split: test neurons from training animals (`within`, `pooledwithin`) vs held-out animals
(`cross`, `pooledcross`).  Population: single-animal (`within`, `cross`) vs mixed
(`pooledwithin`, `pooledcross`).  See README "Regimes" and the docstring of `allen/run_decoder.py`.

## Caveats to carry
- Machine has ~2.5 GB free RAM (a Virtualization process holds 2.3 GB); keep MPS batch <= 16, n <= 512.
- Allen orientation labels: SG vs DG agree within 15 deg for 45-67 % of cells.
- Model selection in Allen runs uses held-out mice (`sel_frac`), never test mice.

## Regression re-run plan (2026-09-11, done; kept for the record)

Decision: orientation is decoded as a circular regression (target (cos 2θ, sin 2θ), metric = mean absolute angular
error in degrees, `within15` = fraction within 15°, `err_modD` = error after the best global rotation/reflection, i.e.
the frame check). Classification stays only for the class-level symmetry analysis and one appendix table. RF unchanged.
Allen labels are quantised at 30° (static gratings): quantisation floor ~7.5°; say so in the paper. Baseline: 45° (uniform).
Code: `task == "circ"` in `microns_ambiguity/decoder2.py`; content `oricirc` in `microns_ambiguity/run_decoder2.py` and
`allen/run_decoder.py`. Smoke-tested locally on CPU.
Pod: RunPod RTX 4090, id bofozxleuvk0d5, $0.74/h, host/port in scratchpad `pod3.env` (else query the GraphQL API with
the key in `.env`). Two detached streams on the pod, logs in `/workspace/intentionality/logs/circ_microns.log` and
`circ_allen.log`; sweeps `microns_ambiguity/sweeps/sweep_circ_microns.sh` (15 runs, tags `c_is_*`, `c_iv_*`) and
`allen/sweep_circ_allen.sh` (24 runs, tags `c_pc_*`, `c_pw_*`, `c_wi_*`, `c_cr_*`); end markers SWEEP_CIRC_MICRONS_DONE /
SWEEP_CIRC_ALLEN_DONE. Expected ~3.5 h wall, ~$3.
When done: pull `allen/outputs/decoder.json`, `microns_ambiguity/outputs/decoder2.json`, `allen/outputs/preds`, `logs`
(tar over ssh; do NOT print anything else into the tar stream), merge into the local JSONs (dict update), terminate the
pod (GraphQL podTerminate), then: (1) update `microns_ambiguity/paper_figures.py` so Figs 2, 5, A1, A2, A3, A4, A5
plot angular error for orientation (lower is better; baseline line at 45°), keep RF panels; (2) update tables and text in
`microns_ambiguity/paper/main.tex` and `paper_draft.md` (Table 1, Table 2, Sections 3.1, 3.3, 3.5, abstract numbers);
(3) recompile, copy PDF to ~/Desktop/relational_decoding_draft.pdf, commit, push.

## Done
- Session A (natural movies one+three) cached for 36 containers / 33 mice: `allen_data/cache/`
- Session C (locally sparse noise RF + movies one+two): `allen_data/cache_C/`
- Orientation from natural-movie correlations: negative (README "Allen" section)
- Cross-animal RF placement: positive, R² 0.23 pooled-cross 256 neurons (`allen/outputs/decoder.json`)

## Done: grating-substrate test (2026-09-09)
Relations from drifting-grating condition means (session A, 40 conditions), labels from
static gratings (session B).  Results in README "Allen" section, `allen/outputs/dg_diagnostics.json`,
`allen/outputs/geometric_DG_nm1_sg.json`, `allen/outputs/decoder.json` (tags `dg_*`).
Headline: pooled-cross label-free decoder 0.229 +- 0.020 (0.245 +- 0.005 averaged over 32
populations) on 9 held-out mice vs 0.19 baseline; labelled LOMO ceiling 0.35; within-mouse
(`cross`) regime stays at baseline.  Sweeps: `allen/sweep_dg.sh`, `allen/sweep_dg2.sh`.

## GPU sessions (2026-09-09/10, RunPod L40S, ~14 h, ~$13; pod terminated, balance $6.76 left)
Code and data were copied to the pod; results merged back into `allen/outputs/decoder.json`,
`microns_ambiguity/outputs/decoder2.json`, `allen/outputs/preds/`, logs in `allen/outputs/gpu_logs/`.
Sweep scripts: `allen/sweep_gpu*.sh`, `microns_ambiguity/sweeps/sweep_gpu*.sh` (paths are pod paths).
Headline: Allen pooled-cross orientation 0.310 (17M + 50 % condition subsampling; 57M plain also 0.310);
Allen cross RF 0.27-0.28 (single-animal populations beat mixed 0.22); MICrONS twin RF 0.507 (57M),
in-vivo ori 0.388 (17M), in-vivo RF 0.247 (17M augmented).  Details in README §4 and Allen section.
Lessons: (1) launch pod jobs with `nohup setsid ... &` inside a subshell; queue loops launched from
an ssh session die with it.  (2) never `pkill -f <pattern>` from an ssh one-liner whose own command
line contains the pattern (it kills the session).  (3) two big runs share 44 GB: batch 32 for 57M+
models, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.  (4) budget: $1.09/h burned faster than
planned because both streams ran; check balance every hour.

## Consolidated conclusions (2026-09-10, after the second GPU session)
- Orientation: mixed-population cells work (pooledwithin 0.28, pooledcross 0.29/0.24/0.26 over 3 splits,
  baseline 0.19); single-animal cells at baseline on every split, seen or unseen animal.  Headline
  config (17M + 50 % condition subsampling) = 2M standard config within noise; ensemble 0.30 on split 0.
- RF: absolute R2 is split-dependent (cross 0.27 / -0.10 / -0.46; pooledcross 0.21 / 0.04 / -0.21).
  Decomposition of saved predictions: per-mouse mean screen position is recovered (r 0.36-0.71 over 27
  mouse-level predictions, permutation p <= 0.03), within-mouse layout is not (r ~ 0.1).  Report the
  mouse-level correlation, not R2.  The earlier "RF crosses animals at R2 0.23" was split 0 only.
- MICrONS: capacity pays on RF (twin 0.48 +- 0.01 at 17M, 0.51 at 57M; in vivo 0.225 plain, 0.262 with
  75 % bin subsampling) and not on orientation (0.42 at 7M/17M/57M twin; in vivo 0.38 at 17M).
  Every large run memorises; gains come from early stopping on a bigger model.  Seeds: `g12_*`.
- Levers that do nothing: population size past 256 (Allen) / 512 (MICrONS), dropout, Gram dropout,
  partial augmentation, augmentation on PCA features (destructive), stacking augmentation on 57M.

## Third GPU session (2026-09-10, RTX 4090, 15 min, $0.15)
Splits 1 and 2 for the training-animal cells (`g13_*`): within ori 0.216 / 0.217, pooledwithin ori 0.283 / 0.283,
within RF 0.359 / 0.364, pooledwithin RF 0.123 / 0.157.  Every 2x2 cell now has three splits.  Balance $6.61.
