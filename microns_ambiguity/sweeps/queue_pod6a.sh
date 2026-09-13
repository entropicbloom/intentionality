#!/bin/bash
# v2 real-data runs: residual ablation, rotation-angle preds, matched single-scan training, extra seeds for the reference/controls tables
cd /workspace/intentionality; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="python -m microns_ambiguity.run_decoder2"
S17="dim=512 layers=8 epochs=30 batch=64 rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000 n=512 save_preds=1"
# A. residual ablation (and the circulant-removal control)
$R abl_resid_is func_is oricirc pca=1 ablate=resid seed=0 $S17
$R abl_resid_iv func_iv oricirc       ablate=resid seed=0 $S17
$R abl_circ_is  func_is oricirc pca=1 ablate=circ  seed=0 $S17
$R abl_circ_iv  func_iv oricirc       ablate=circ  seed=0 $S17
# B. cross-area with saved predictions (rotation angle)
$R m_V1toAL_is_n128 func_is oricirc pca=1 area_train=V1 area_test=AL n=128 sel_reps=1 seed=0 dim=512 layers=8 epochs=30 batch=64 rel_bias=1 device=cuda early_stop=0.2 pops_per_epoch=5000 save_preds=1
$R m_V1toV1_is_n128 func_is oricirc pca=1 area_train=V1 area_test=V1 n=128 sel_reps=1 seed=0 dim=512 layers=8 epochs=30 batch=64 rel_bias=1 device=cuda early_stop=0.2 pops_per_epoch=5000 save_preds=1
$R m_V1toRL_is func_is oricirc pca=1 area_train=V1 area_test=RL n=512 sel_reps=1 seed=0 dim=512 layers=8 epochs=30 batch=64 rel_bias=1 device=cuda early_stop=0.2 pops_per_epoch=5000 save_preds=1
# C. matched single-circuit training: one scan, 140 training neurons, within-scan populations, test on the other scans
W="dim=512 layers=8 epochs=30 batch=64 seed=0 rel_bias=1 device=cuda pops_per_epoch=5000 within_scan=1 n=128 sel_reps=1 train_pool=140"
for sc in 6-4 5-6 4-7; do $R ts_is_$sc func_is oricirc pca=1 train_scan=$sc $W; $R ts_iv_$sc func_iv oricirc train_scan=$sc $W; done
# D. extra seeds for the reference and controls tables
for s in 1 2; do
  $R a_iv_17M_ori_s$s func_iv oricirc       input_mode=act rel_bias=0 seed=$s $S17
  $R a_is_17M_ori_s$s func_is oricirc pca=1 input_mode=act rel_bias=0 seed=$s $S17
  L="dim=512 layers=0 epochs=30 batch=64 seed=$s n=512 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000 input_mode=act rel_bias=0"
  $R lin_iv_ori_s$s func_iv oricirc       $L; $R lin_is_ori_s$s func_is oricirc pca=1 $L; $R lin_iv_rf_s$s func_iv rf $L; $R lin_is_rf_s$s func_is rf pca=1 $L
  $R bp_iv_17M_ori_s$s func_iv oricirc       input_mode=act rel_bias=0 bin_perm=1 seed=$s $S17
  $R bp_is_17M_ori_s$s func_is oricirc pca=1 input_mode=act rel_bias=0 bin_perm=1 seed=$s $S17
  $R c_iv_17M_bins_s$s func_iv oricirc bins=1 seed=$s $S17
  $R c_is_17M_bins_s$s func_is oricirc bins=1 seed=$s $S17
  $R c_is_17M_statsbias_s$s func_is oricirc pca=1 row_proj=0 seed=$s $S17
done
echo QUEUE_POD6A_DONE
