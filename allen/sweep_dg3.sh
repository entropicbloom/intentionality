#!/bin/zsh
# Fixed protocol: best-epoch checkpoint, selection averaged over 4 population covers of 6 held-out mice.
cd /Users/flaessig/Documents/repos/intentionality
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45
C="content=ori session=DG movie=nm1 rel_bias=1 batch=16 device=mps sel_reps=4 sel_frac=0.25 pops_per_epoch=2500"
R=".venv/bin/python -m allen.run_decoder"
# A. base 128, 30 epochs: 3 seeds on split 0 (ensemble), splits 1 and 2
$=R dg3_128_sp0_s0 pooledcross n=128 dim=256 layers=4 epochs=30 split_seed=0 seed=0 $=C
$=R dg3_128_sp0_s1 pooledcross n=128 dim=256 layers=4 epochs=30 split_seed=0 seed=1 $=C
$=R dg3_128_sp0_s2 pooledcross n=128 dim=256 layers=4 epochs=30 split_seed=0 seed=2 $=C
$=R dg3_128_sp1    pooledcross n=128 dim=256 layers=4 epochs=30 split_seed=1 seed=1 $=C
$=R dg3_128_sp2    pooledcross n=128 dim=256 layers=4 epochs=30 split_seed=2 seed=2 $=C
# E. cross regime, corrected within-mouse averaging
$=R dg3_cross_128  cross       n=128 dim=256 layers=4 epochs=30 split_seed=0 seed=0 $=C
# B. 256 neurons, 30 epochs, 3 seeds on split 0
$=R dg3_256_sp0_s0 pooledcross n=256 dim=256 layers=4 epochs=30 split_seed=0 seed=0 $=C
$=R dg3_256_sp0_s1 pooledcross n=256 dim=256 layers=4 epochs=30 split_seed=0 seed=1 $=C
$=R dg3_256_sp0_s2 pooledcross n=256 dim=256 layers=4 epochs=30 split_seed=0 seed=2 $=C
# C. big model + 256 neurons
$=R dg3_big256_sp0 pooledcross n=256 dim=384 layers=6 epochs=30 split_seed=0 seed=0 $=C
# D. 512 neurons with the fixed selection
$=R dg3_512_sp0    pooledcross n=512 dim=256 layers=4 epochs=20 split_seed=0 seed=0 content=ori session=DG movie=nm1 rel_bias=1 batch=8 device=mps sel_reps=4 sel_frac=0.25 pops_per_epoch=1500
echo SWEEP_DG3_DONE
