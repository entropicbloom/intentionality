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

## Possible next steps
- Labelled reference ("ceiling") is not settled: ridge is one linear labelled model and the decoder
  beat it on RF.  Open checks (deferred, user wants label-free decoding first): ridge accuracy vs
  OSI gate and vs same-session (drifting-grating) labels to separate label noise from substrate
  noise; nonlinear labelled readouts (kNN, MLP, labelled-token transformer); zero the Gram
  diagonal in the ridge fit.  Call it "labelled reference" in the paper, not ceiling.
- Decoder: average populations across scans in MICrONS (targets the session effect).
- Allen: seeds for the 256-neuron / 30-epoch / 7.2M runs; larger selection set (sel_frac) so
  512-neuron runs do not early-stop at epoch 2.
- Paper: write up.  Structure agreed: lead with cortex, method re-introduced, no consciousness framing.

## Caveats to carry
- Machine has ~2.5 GB free RAM (a Virtualization process holds 2.3 GB); keep MPS batch <= 16, n <= 512.
- Allen orientation labels: SG vs DG agree within 15 deg for 45-67 % of cells.
- Model selection in Allen runs uses held-out mice (`sel_frac`), never test mice.
