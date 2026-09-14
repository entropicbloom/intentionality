#!/bin/bash
# rerun of the seed-2 runs lost when queue_pod6a.sh died at 00:08 UTC 2026-09-14
cd /workspace/intentionality; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="python -m microns_ambiguity.run_decoder2"
s=2
S17="dim=512 layers=8 epochs=30 batch=64 rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000 n=512 save_preds=1"
$R bp_is_17M_ori_s$s func_is oricirc pca=1 input_mode=act rel_bias=0 bin_perm=1 seed=$s $S17
$R c_iv_17M_bins_s$s func_iv oricirc bins=1 seed=$s $S17
$R c_is_17M_bins_s$s func_is oricirc bins=1 seed=$s $S17
$R c_is_17M_statsbias_s$s func_is oricirc pca=1 row_proj=0 seed=$s $S17
echo QUEUE_POD6G_DONE
