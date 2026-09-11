#!/bin/bash
# Circular-regression orientation on MICrONS: 17M x 3 seeds (both substrates), 2.2M x 3 seeds, 57M twin, 0.3M points.
cd /workspace/intentionality
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000 n=512"
R="python -m microns_ambiguity.run_decoder2"
for s in 0 1 2; do
  $R c_is_17M_s$s func_is oricirc pca=1 dim=512 layers=8 epochs=30 batch=64 seed=$s $C
  $R c_iv_17M_s$s func_iv oricirc       dim=512 layers=8 epochs=30 batch=64 seed=$s $C
done
for s in 0 1 2; do
  $R c_is_2M_s$s func_is oricirc pca=1 dim=256 layers=4 epochs=30 batch=64 seed=$s $C
  $R c_iv_2M_s$s func_iv oricirc       dim=256 layers=4 epochs=30 batch=64 seed=$s $C
done
$R c_is_57M_s0 func_is oricirc pca=1 dim=768 layers=12 epochs=30 batch=32 seed=0 $C
$R c_is_03M_s0 func_is oricirc pca=1 dim=128 layers=2 epochs=20 batch=64 seed=0 $C
$R c_iv_03M_s0 func_iv oricirc       dim=128 layers=2 epochs=20 batch=64 seed=0 $C
echo SWEEP_CIRC_MICRONS_DONE
