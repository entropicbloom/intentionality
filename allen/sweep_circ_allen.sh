#!/bin/bash
# Circular-regression orientation on Allen: the 2x2 with three splits, the headline seeds, and the scaling / augmentation points.
cd /workspace/intentionality
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B="content=oricirc session=DG movie=nm1 rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 pops_per_epoch=5000 batch=64"
R="python -m allen.run_decoder"
# headline: pooledcross, 17M + 50 % conditions, 256 neurons, 60 epochs
$R c_pc_17M_cf50_sp0_s0 pooledcross n=256 dim=512 layers=8 epochs=60 cond_frac=0.5 split_seed=0 seed=0 $B
$R c_pc_17M_cf50_sp0_s1 pooledcross n=256 dim=512 layers=8 epochs=60 cond_frac=0.5 split_seed=0 seed=1 $B
$R c_pc_17M_cf50_sp0_s2 pooledcross n=256 dim=512 layers=8 epochs=60 cond_frac=0.5 split_seed=0 seed=2 $B
$R c_pc_17M_cf50_sp1    pooledcross n=256 dim=512 layers=8 epochs=60 cond_frac=0.5 split_seed=1 seed=1 $B
$R c_pc_17M_cf50_sp2    pooledcross n=256 dim=512 layers=8 epochs=60 cond_frac=0.5 split_seed=2 seed=2 $B
# the other three cells, 2M standard, 30 epochs, three splits
for sp in 0 1 2; do
  $R c_pw_2M_sp$sp pooledwithin n=256 dim=256 layers=4 epochs=30 split_seed=$sp seed=$sp $B
  $R c_wi_2M_sp$sp within       n=128 dim=256 layers=4 epochs=30 split_seed=$sp seed=$sp $B
  $R c_cr_2M_sp$sp cross        n=128 dim=256 layers=4 epochs=30 split_seed=$sp seed=$sp $B
done
# scaling and augmentation (split 0, seed 0, pooledcross, 256 neurons, 60 epochs)
$R c_pc_2M_plain   pooledcross n=256  dim=256 layers=4  epochs=60 split_seed=0 seed=0 $B
$R c_pc_17M_plain  pooledcross n=256  dim=512 layers=8  epochs=60 split_seed=0 seed=0 $B
$R c_pc_57M_plain  pooledcross n=256  dim=768 layers=12 epochs=60 split_seed=0 seed=0 content=oricirc session=DG movie=nm1 rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 pops_per_epoch=5000 batch=32
$R c_pc_2M_cf75    pooledcross n=256  dim=256 layers=4  epochs=60 cond_frac=0.75 split_seed=0 seed=0 $B
$R c_pc_2M_cf50    pooledcross n=256  dim=256 layers=4  epochs=60 cond_frac=0.5  split_seed=0 seed=0 $B
$R c_pc_2M_cf30    pooledcross n=256  dim=256 layers=4  epochs=60 cond_frac=0.3  split_seed=0 seed=0 $B
$R c_pc_17M_cf85   pooledcross n=256  dim=512 layers=8  epochs=60 cond_frac=0.85 split_seed=0 seed=0 $B
$R c_pc_57M_cf50   pooledcross n=256  dim=768 layers=12 epochs=60 cond_frac=0.5  split_seed=0 seed=0 content=oricirc session=DG movie=nm1 rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 pops_per_epoch=5000 batch=32
$R c_pc_17M_n512   pooledcross n=512  dim=512 layers=8  epochs=60 split_seed=0 seed=0 $B
$R c_pc_17M_n1024  pooledcross n=1024 dim=512 layers=8  epochs=60 split_seed=0 seed=0 content=oricirc session=DG movie=nm1 rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 pops_per_epoch=5000 batch=32
echo SWEEP_CIRC_ALLEN_DONE
