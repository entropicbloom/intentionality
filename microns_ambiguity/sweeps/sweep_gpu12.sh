#!/bin/bash
# Consolidation (MICrONS): seeds for the 17M headline configuration, in vivo first.
cd /workspace/intentionality
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000 batch=64 n=512 dim=512 layers=8"
R="python -m microns_ambiguity.run_decoder2"
$R g12_iv_ori_s1  func_iv ori epochs=30 seed=1 $C
$R g12_iv_rf_s1   func_iv rf  epochs=40 seed=1 $C
$R g12_iv_ori_s2  func_iv ori epochs=30 seed=2 $C
$R g12_iv_rf_s2   func_iv rf  epochs=40 seed=2 $C
$R g12_is_ori_s1  func_is ori pca=1 epochs=30 seed=1 $C
$R g12_is_ori_s2  func_is ori pca=1 epochs=30 seed=2 $C
echo SWEEP_GPU12_DONE
