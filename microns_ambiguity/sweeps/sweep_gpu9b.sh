#!/bin/bash
cd /workspace/intentionality
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "run_decoder2 g9_is_rf_n512_d512L8_s[2]" > /dev/null; do sleep 20; done
C="rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000"
R="python -m microns_ambiguity.run_decoder2"
$R g9_iv_rf_n512_d512L8       func_iv rf n=512 dim=512 layers=8 epochs=40 batch=64 $C
$R g9_iv_rf_n512_d512L8_cf75  func_iv rf n=512 dim=512 layers=8 epochs=40 batch=64 cond_frac=0.75 $C
$R g9_is_ori_n512_d768L12     func_is ori pca=1 n=512 dim=768 layers=12 epochs=30 batch=32 $C
echo SWEEP_GPU9B_DONE
