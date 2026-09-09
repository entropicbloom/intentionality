#!/bin/bash
# Consolidation (Allen): the fixed `within` cell, and seeds / splits for every reported held-out-animal number.
cd /workspace/intentionality
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B="rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 pops_per_epoch=5000 batch=64 dim=256 layers=4"
R="python -m allen.run_decoder"
# within cell (training animals, single-animal populations), orientation and RF
$R g11_ori_within_n128      within content=ori session=DG movie=nm1 n=128 epochs=30 split_seed=0 seed=0 $B
$R g11_rf_within_n128       within content=rf  session=C  movie=both n=128 epochs=30 split_seed=0 seed=0 $B
# RF cross (single-animal, held-out mice), fine schedule: seeds and splits
F="content=rf session=C movie=both rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 pops_per_epoch=800 batch=32 epochs=40 lr=0.0005 dim=256 layers=4 n=128"
$R g11_rf_cross_fine_s1     cross split_seed=0 seed=1 $F
$R g11_rf_cross_fine_s2     cross split_seed=0 seed=2 $F
$R g11_rf_cross_fine_sp1    cross split_seed=1 seed=1 $F
$R g11_rf_cross_fine_sp2    cross split_seed=2 seed=2 $F
# RF pooledcross (mixed, held-out mice), 256 neurons, 2M: splits
P="content=rf session=C movie=both rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 pops_per_epoch=5000 batch=64 epochs=60 dim=256 layers=4 n=256"
$R g11_rf_pooledcross_sp1   pooledcross split_seed=1 seed=1 $P
$R g11_rf_pooledcross_sp2   pooledcross split_seed=2 seed=2 $P
# orientation cross (single-animal, held-out mice): splits
O="content=ori session=DG movie=nm1 rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 pops_per_epoch=5000 batch=64 epochs=30 dim=256 layers=4 n=128"
$R g11_ori_cross_sp1        cross split_seed=1 seed=1 $O
$R g11_ori_cross_sp2        cross split_seed=2 seed=2 $O
echo SWEEP_GPU11_DONE
