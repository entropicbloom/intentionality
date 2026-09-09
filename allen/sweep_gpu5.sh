#!/bin/bash
# GPU sweep 5: augmentation refinement at the 25M capacity (mild subsampling, partial augmentation).
cd /workspace/intentionality
C="content=ori session=DG movie=nm1 rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 split_seed=0 seed=0 pops_per_epoch=5000 batch=64 epochs=60 n=256 dim=512 layers=8"
R="python -m allen.run_decoder"
$R g5_d512L8_cf85          pooledcross cond_frac=0.85 $C
$R g5_d512L8_cf75          pooledcross cond_frac=0.75 $C
$R g5_d512L8_cf75_ap50     pooledcross cond_frac=0.75 aug_prob=0.5 $C
$R g5_d512L8_cf50_ap50     pooledcross cond_frac=0.5 aug_prob=0.5 $C
$R g5_d512L8_cf75_gd10     pooledcross cond_frac=0.75 gram_drop=0.1 $C
$R g5_d512L8_cf85_s1       pooledcross cond_frac=0.85 seed=1 $C
$R g5_d512L8_cf85_s2       pooledcross cond_frac=0.85 seed=2 $C
echo SWEEP_GPU5_DONE
