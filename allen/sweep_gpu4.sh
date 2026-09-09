#!/bin/bash
# GPU sweep 4: Allen receptive-field position, pooled-cross across mice (session C, movies one+two).
cd /workspace/intentionality
C="content=rf session=C movie=both rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 split_seed=0 seed=0 pops_per_epoch=5000 batch=64 epochs=60"
R="python -m allen.run_decoder"
$R g4_rf_n256_d256L4       pooledcross n=256 dim=256 layers=4 $C
$R g4_rf_n256_d256L4_cf50  pooledcross n=256 dim=256 layers=4 cond_frac=0.5 $C
$R g4_rf_n512_d512L8_cf50  pooledcross n=512 dim=512 layers=8 cond_frac=0.5 $C
$R g4_rf_n256_d512L8_cf50_gd20 pooledcross n=256 dim=512 layers=8 cond_frac=0.5 gram_drop=0.2 $C
$R g4_rf_n1024_d512L8_cf50 pooledcross n=1024 dim=512 layers=8 cond_frac=0.5 batch=32 $C
echo SWEEP_GPU4_DONE
