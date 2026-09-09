# Allen work: status and next steps (written 2026-09-09, before context compaction)

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

## GPU session (2026-09-09, RunPod L40S, 8 h, $9.5)
Code and data were copied to the pod; results merged back into `allen/outputs/decoder.json`,
`microns_ambiguity/outputs/decoder2.json`, `allen/outputs/preds/`, logs in `allen/outputs/gpu_logs/`.
Sweep scripts: `allen/sweep_gpu*.sh`, `microns_ambiguity/sweeps/sweep_gpu*.sh` (paths are pod paths).
Headline: Allen pooled-cross orientation 0.310 (17M + 50 % condition subsampling; 85M plain also 0.310);
Allen cross RF 0.27-0.28 (single-animal populations beat mixed 0.22); MICrONS twin RF 0.507 (57M),
in-vivo ori 0.388 (17M), in-vivo RF 0.247 (17M augmented).  Details in README §4 and Allen section.
Lessons: (1) launch pod jobs with `nohup setsid ... &` inside a subshell; queue loops launched from
an ssh session die with it.  (2) never `pkill -f <pattern>` from an ssh one-liner whose own command
line contains the pattern (it kills the session).  (3) two big runs share 44 GB: batch 32 for 57M+
models, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.  (4) budget: $1.09/h burned faster than
planned because both streams ran; check balance every hour.

## Runs cut by the budget (queue them first on the next GPU session)
- Allen: `allen/sweep_gpu8.sh` (fine-grained-epoch RF, `cross` and `pooledcross`, with augmentation);
  seeds 1,2 and splits 1,2 of the best config (`g10_d512L8_cf50_*` in `allen/sweep_gpu5.sh`); 85M + cf50.
- MICrONS: plain 17M in-vivo RF (comparison for the augmented 0.247), 57M twin orientation, 17M twin RF seed 2.
- Fix the `within` regime: per-mouse selection slices < n; select on a slice of whole mice or on
  pooled populations of the selection slice.

## Possible next steps
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
