#!/bin/bash
# GPU sweep 9: MICrONS twin, plain (no PC subsampling): capacity, seeds, 1024-neuron populations.
cd /workspace/intentionality
C="rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000"
R="python -m microns_ambiguity.run_decoder2"
$R g9_is_rf_n512_d768L12     func_is rf  pca=1 n=512  dim=768 layers=12 epochs=40 batch=64 $C
$R g9_is_rf_n512_d512L8_s1   func_is rf  pca=1 n=512  dim=512 layers=8  epochs=40 batch=64 seed=1 $C
$R g9_is_rf_n512_d512L8_s2   func_is rf  pca=1 n=512  dim=512 layers=8  epochs=40 batch=64 seed=2 $C
$R g9_is_rf_n1024_d512L8     func_is rf  pca=1 n=1024 dim=512 layers=8  epochs=40 batch=32 $C
$R g9_is_ori_n512_d768L12    func_is ori pca=1 n=512  dim=768 layers=12 epochs=30 batch=64 $C
$R g9_is_ori_n512_d512L8_s1  func_is ori pca=1 n=512  dim=512 layers=8  epochs=30 batch=64 seed=1 $C
$R g9_is_ori_n512_d512L8_s2  func_is ori pca=1 n=512  dim=512 layers=8  epochs=30 batch=64 seed=2 $C
echo SWEEP_GPU9_DONE
