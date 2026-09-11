# Allen work: status and next steps (written 2026-09-09, before context compaction)

## IN FLIGHT (2026-09-11): orientation as circular regression, re-run of every plotted configuration
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

## Related work notes (for the paper)
- Oizumi, Lim & Kanai, "Principal Bundle Geometry of Qualia: Understanding the Quality of Consciousness from Symmetry",
  PNAS Nexus 5(9) pgag261 (2026), https://academic.oup.com/pnasnexus/article/5/9/pgag261/8788187 .  Theory only:
  equivariant encoders split the state space into a group-inherited, universal part ("qualia attributes") and a
  learnable, individual part ("signatures"); proposes Gromov-Wasserstein alignment / TDA for cross-subject tests.
  Use: theoretical framing for the symmetry result (circulant part = group-fixed structure identical up to
  relabelling; the anisotropy = what pins the frame).  Their objects are similarity structures over stimuli across
  subjects; ours are correlation structures over neurons, decoded per neuron and tested across animals.  One paragraph.
- Oizumi group's unsupervised Gromov-Wasserstein alignment of similarity structures across individuals (e.g. Kawakita
  et al.): the continuous relaxation of our exhaustive permutation matching in the class-level test; cite in methods.

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
