#!/bin/bash
# priority Allen runs: seeds + splits of the best configuration (25M, cond_frac 0.5), then mild augmentation
cd /workspace/intentionality
C="content=ori session=DG movie=nm1 rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 pops_per_epoch=5000 batch=64 epochs=60 n=256 dim=512 layers=8"
R="python -m allen.run_decoder"
$R g10_d512L8_cf50_s1   pooledcross cond_frac=0.5 split_seed=0 seed=1 $C
$R g10_d512L8_cf50_s2   pooledcross cond_frac=0.5 split_seed=0 seed=2 $C
$R g10_d512L8_cf50_sp1  pooledcross cond_frac=0.5 split_seed=1 seed=1 $C
$R g10_d512L8_cf50_sp2  pooledcross cond_frac=0.5 split_seed=2 seed=2 $C
$R g5_d512L8_cf85       pooledcross cond_frac=0.85 split_seed=0 seed=0 $C
$R g5_d512L8_cf75_ap50  pooledcross cond_frac=0.75 aug_prob=0.5 split_seed=0 seed=0 $C
echo SWEEP_GPU5_DONE
