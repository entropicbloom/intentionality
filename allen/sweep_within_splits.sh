#!/bin/zsh
# splits 1 and 2 for the two training-animal cells (MPS), standard 2M configuration, 30 epochs
cd /Users/flaessig/Documents/repos/intentionality
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45
C="content=ori session=DG movie=nm1 rel_bias=1 device=mps sel_reps=4 sel_frac=0.25 pops_per_epoch=2500 batch=16 epochs=30 dim=256 layers=4"
R=".venv/bin/python -m allen.run_decoder"
$=R g13_ori_within_n128_sp1        within       n=128 split_seed=1 seed=1 $=C
$=R g13_ori_within_n128_sp2        within       n=128 split_seed=2 seed=2 $=C
$=R g13_ori_pooledwithin_n256_sp1  pooledwithin n=256 split_seed=1 seed=1 $=C
$=R g13_ori_pooledwithin_n256_sp2  pooledwithin n=256 split_seed=2 seed=2 $=C
echo SWEEP13_DONE
