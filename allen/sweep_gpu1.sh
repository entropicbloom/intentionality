#!/bin/bash
# GPU sweep 1: Allen orientation, pooled-cross, grating relations. Fixed protocol (split 0, sel_reps=4, sel_frac=0.25).
cd /workspace/intentionality
C="content=ori session=DG movie=nm1 rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 split_seed=0 seed=0"
R="python -m allen.run_decoder"
$R g1_n256_d256L4_e60   pooledcross n=256  dim=256 layers=4  epochs=60  pops_per_epoch=5000 batch=64 $C
$R g1_n256_d512L8_e60   pooledcross n=256  dim=512 layers=8  epochs=60  pops_per_epoch=5000 batch=64 $C
$R g1_n512_d512L8_e60   pooledcross n=512  dim=512 layers=8  epochs=60  pops_per_epoch=5000 batch=64 $C
$R g1_n256_d768L12_e60  pooledcross n=256  dim=768 layers=12 epochs=60  pops_per_epoch=5000 batch=64 $C
$R g1_n1024_d512L8_e60  pooledcross n=1024 dim=512 layers=8  epochs=60  pops_per_epoch=5000 batch=32 $C
$R g1_n512_d768L12_e100 pooledcross n=512  dim=768 layers=12 epochs=100 pops_per_epoch=5000 batch=64 $C
$R g1_n256_d512L8_e60_do2 pooledcross n=256 dim=512 layers=8 epochs=60 pops_per_epoch=5000 batch=64 dropout=0.2 $C
echo SWEEP_GPU1_DONE
