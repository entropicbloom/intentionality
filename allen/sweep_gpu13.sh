#!/bin/bash
# splits 1 and 2 for the training-animal cells (orientation and RF), matching the split-0 configurations
cd /workspace/intentionality
O="content=ori session=DG movie=nm1 rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 pops_per_epoch=5000 batch=64 epochs=30 dim=256 layers=4"
F="content=rf session=C movie=both rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 pops_per_epoch=5000 batch=64 epochs=30 dim=256 layers=4"
R="python -m allen.run_decoder"
$R g13_ori_within_n128_sp1       within       n=128 split_seed=1 seed=1 $O
$R g13_ori_within_n128_sp2       within       n=128 split_seed=2 seed=2 $O
$R g13_ori_pooledwithin_n256_sp1 pooledwithin n=256 split_seed=1 seed=1 $O
$R g13_ori_pooledwithin_n256_sp2 pooledwithin n=256 split_seed=2 seed=2 $O
$R g13_rf_within_n128_sp1        within       n=128 split_seed=1 seed=1 $F
$R g13_rf_within_n128_sp2        within       n=128 split_seed=2 seed=2 $F
$R g13_rf_pooledwithin_n128_sp1  pooledwithin n=128 split_seed=1 seed=1 $F
$R g13_rf_pooledwithin_n128_sp2  pooledwithin n=128 split_seed=2 seed=2 $F
echo SWEEP_GPU13_DONE
