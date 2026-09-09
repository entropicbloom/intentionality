# Allen work: status and next steps (written 2026-09-09, before context compaction)

## Done
- Session A (natural movies one+three) cached for 36 containers / 33 mice: `allen_data/cache/`
- Session C (locally sparse noise RF + movies one+two): `allen_data/cache_C/`
- Orientation from natural-movie correlations: negative (README "Allen" section)
- Cross-animal RF placement: positive, R² 0.23 pooled-cross 256 neurons (`allen/outputs/decoder.json`)

## In flight: grating-substrate test
Question: is orientation recoverable across animals when relations are built from
drifting-grating responses (session A, 8 dir x 5 TF, shared across mice) instead of movies?
Labels stay from static gratings (session B, cell table) so labels and relations use
different stimuli.

1. `allen/fetch.py - DG` re-downloads session A and caches `allen_data/cache_DG/<exp>.npz`
   with r_nm1 = condition means (cells x 40) and r_nm3 = condition time courses (cells x 2400).
   Log: `allen_data/fetch_DG.log` (ends with `doneDG`).
2. When done, run in `.venv`:
   - `python -m allen.run_dg`                      -> outputs/dg_diagnostics.json (signal + ridge ceilings)
   - `python -m allen.run_geometric nm1 sg DG`      -> class-level within / cross / pooled matching
   - `python -m allen.run_decoder dg_ori_cross cross content=ori session=DG movie=nm1 n=128 dim=256 layers=4 rel_bias=1 epochs=12 pops_per_epoch=2500 batch=16 device=mps`
   - same with `regime=pooledcross`, and with `movie=both`
   Use `export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45` for MPS runs.
3. Interpret: if class-level cross-mouse accuracy > chance (1/6) and the decoder beats the
   majority rate (~0.20) on held-out mice, the symmetry / per-neuron orientation result
   replicates across animals and the paper is journal-ready; else the paper is
   "RF crosses animals, orientation is dataset-dependent". Add to README, commit, push.

## Caveats to carry
- Machine has ~2.5 GB free RAM (a Virtualization process holds 2.3 GB); keep MPS batch <= 16, n <= 512.
- Allen orientation labels: SG vs DG agree within 15 deg for 45-67 % of cells.
- Model selection in Allen runs uses held-out mice (`sel_frac`), never test mice.
