#!/bin/bash
# GPU sweep 2: label-free regularisation (condition subsampling, Gram dropout) at moderate capacity.
cd /workspace/intentionality
C="content=ori session=DG movie=nm1 rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 split_seed=0 seed=0 pops_per_epoch=5000 batch=64 epochs=60"
R="python -m allen.run_decoder"
$R g2_n256_d256L4_cf75      pooledcross n=256 dim=256 layers=4 cond_frac=0.75 $C
$R g2_n256_d256L4_cf50      pooledcross n=256 dim=256 layers=4 cond_frac=0.5 $C
$R g2_n256_d256L4_gd20      pooledcross n=256 dim=256 layers=4 gram_drop=0.2 $C
$R g2_n256_d256L4_cf50_gd20 pooledcross n=256 dim=256 layers=4 cond_frac=0.5 gram_drop=0.2 $C
$R g2_n256_d512L8_cf50      pooledcross n=256 dim=512 layers=8 cond_frac=0.5 $C
$R g2_n256_d512L8_cf50_gd20 pooledcross n=256 dim=512 layers=8 cond_frac=0.5 gram_drop=0.2 $C
$R g2_n256_d256L4_cf30      pooledcross n=256 dim=256 layers=4 cond_frac=0.3 $C
echo SWEEP_GPU2_DONE
