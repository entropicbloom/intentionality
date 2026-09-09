#!/bin/bash
# GPU sweep 8: RF with fine epoch granularity (peak is reached within ~100 steps) and augmentation.
cd /workspace/intentionality
C="content=rf session=C movie=both rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 split_seed=0 seed=0 pops_per_epoch=800 batch=32 epochs=40 lr=0.0005 dim=256 layers=4"
R="python -m allen.run_decoder"
$R g8_rf_cross_fine            cross       n=128 $C
$R g8_rf_cross_fine_cf75ap50   cross       n=128 cond_frac=0.75 aug_prob=0.5 $C
$R g8_rf_cross_fine_cf50gd20   cross       n=128 cond_frac=0.5 gram_drop=0.2 $C
$R g8_rf_pooledcross_fine      pooledcross n=256 $C
$R g8_rf_pooledcross_fine_cf75ap50 pooledcross n=256 cond_frac=0.75 aug_prob=0.5 $C
echo SWEEP_GPU8_DONE
