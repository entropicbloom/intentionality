#!/bin/bash
# frame-source follow-ups: finer-resolution residual ablation and class-balanced populations (twin first, then in vivo)
cd /workspace/intentionality
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="python -m microns_ambiguity.run_decoder2"
S17="dim=512 layers=8 epochs=30 batch=64 rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000 n=512 save_preds=1 seed=0"
$R abl_resid24_is func_is oricirc pca=1 ablate=resid ablate_k=24 $S17
$R bal_is         func_is oricirc pca=1 balance_pop=1 $S17
$R bal_abl_is     func_is oricirc pca=1 balance_pop=1 ablate=resid $S17
$R bal_abl24_is   func_is oricirc pca=1 balance_pop=1 ablate=resid ablate_k=24 $S17
$R abl_resid24_iv func_iv oricirc ablate=resid ablate_k=24 $S17
$R bal_abl24_iv   func_iv oricirc balance_pop=1 ablate=resid ablate_k=24 $S17
echo QUEUE_POD6C_DONE
