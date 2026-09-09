#!/bin/bash
# GPU sweep 10: Allen orientation, best configuration (25M, cond_frac 0.5): seeds for an ensemble, other splits, 85M + augmentation.
cd /workspace/intentionality
C="content=ori session=DG movie=nm1 rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 pops_per_epoch=5000 batch=64 epochs=60 n=256 cond_frac=0.5"
R="python -m allen.run_decoder"
$R g10_d512L8_cf50_s1   pooledcross dim=512 layers=8  split_seed=0 seed=1 $C
$R g10_d512L8_cf50_s2   pooledcross dim=512 layers=8  split_seed=0 seed=2 $C
$R g10_d512L8_cf50_sp1  pooledcross dim=512 layers=8  split_seed=1 seed=1 $C
$R g10_d512L8_cf50_sp2  pooledcross dim=512 layers=8  split_seed=2 seed=2 $C
$R g10_d768L12_cf50     pooledcross dim=768 layers=12 split_seed=0 seed=0 $C
echo SWEEP_GPU10_DONE
