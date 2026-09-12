#!/bin/bash
# Pod 4 queue: everything the stopped pod 3 may not have finished, least-likely-finished first.
cd /workspace/intentionality; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="python -m microns_ambiguity.run_decoder2"
A="python -m allen.run_decoder"
C="rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000"
B="content=oricirc session=DG movie=nm1 rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 pops_per_epoch=5000 batch=64"
S17="dim=512 layers=8 epochs=30 batch=64 seed=0 n=512 $C"
# (a) plain 2M Allen splits and seeds (2x2 table cell, split figure)
for sp in 1 2; do $A c_pc_2M_plain_sp$sp pooledcross n=256 dim=256 layers=4 epochs=60 split_seed=$sp seed=0 $B; done
for s in 1 2; do $A c_pc_2M_plain_s$s pooledcross n=256 dim=256 layers=4 epochs=60 split_seed=0 seed=$s $B; done
# (b) population-size curves in angular error
for n in 128 256 1024; do $R c_iv_03M_n$n func_iv oricirc dim=128 layers=2 epochs=20 batch=64 seed=0 n=$n $C; done
for n in 256 1024; do $R c_is_2M_n$n func_is oricirc pca=1 dim=256 layers=4 epochs=30 batch=64 seed=0 n=$n $C; done
# (c) reviewer follow-ups, reverse order of the pod-3 sweep
$R c_is_17M_statsonly func_is oricirc pca=1 row_proj=0 rel_bias=0 dim=512 layers=8 epochs=30 batch=64 seed=0 n=512 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000
$R c_is_17M_statsbias func_is oricirc pca=1 row_proj=0 rel_bias=1 dim=512 layers=8 epochs=30 batch=64 seed=0 n=512 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000
$R c_is_17M_bins  func_is oricirc bins=1 save_preds=1 $S17
$R c_iv_17M_bins  func_iv oricirc bins=1 save_preds=1 $S17
$R c_iv_17M_sp2   func_iv oricirc       split_seed=2 save_preds=1 $S17
$R c_is_17M_sp2   func_is oricirc pca=1 split_seed=2 save_preds=1 $S17
$R r_is_17M_s0    func_is rf      pca=1              save_preds=1 $S17
$R r_iv_17M_s0    func_iv rf                         save_preds=1 $S17
$R c_iv_17M_sp1   func_iv oricirc       split_seed=1 save_preds=1 $S17
$R c_is_17M_sp1   func_is oricirc pca=1 split_seed=1 save_preds=1 $S17
echo QUEUE_POD4_DONE
