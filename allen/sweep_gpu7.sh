#!/bin/bash
# GPU sweep 7: the 2x2 of split (training vs held-out animals) x population (single-animal vs mixed).
cd /workspace/intentionality
C="rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 split_seed=0 seed=0 pops_per_epoch=5000 batch=64 epochs=30 dim=256 layers=4"
R="python -m allen.run_decoder"
$R g7_ori_within_n128       within       content=ori session=DG movie=nm1 n=128 $C
$R g7_ori_pooledwithin_n128 pooledwithin content=ori session=DG movie=nm1 n=128 $C
$R g7_ori_pooledwithin_n256 pooledwithin content=ori session=DG movie=nm1 n=256 $C
$R g7_rf_within_n128        within       content=rf session=C movie=both n=128 $C
$R g7_rf_pooledwithin_n128  pooledwithin content=rf session=C movie=both n=128 $C
echo SWEEP_GPU7_DONE
