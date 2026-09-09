#!/bin/bash
cd /workspace/intentionality
C="content=rf session=C movie=both rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 split_seed=0 seed=0 pops_per_epoch=5000 batch=64 epochs=40"
python -m allen.run_decoder g6_rf_cross_n128 cross n=128 dim=256 layers=4 $C
python -m allen.run_decoder g6_rf_cross_n128_cf85 cross n=128 dim=256 layers=4 cond_frac=0.85 $C
python -m allen.run_decoder g6_rf_cross_n128_d512L8 cross n=128 dim=512 layers=8 $C
echo SWEEP_GPU6_DONE
