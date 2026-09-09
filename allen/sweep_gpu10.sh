#!/bin/bash
# lower priority: 85M + augmentation
cd /workspace/intentionality
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="content=ori session=DG movie=nm1 rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 pops_per_epoch=5000 batch=32 epochs=60 n=256 cond_frac=0.5 split_seed=0 seed=0"
python -m allen.run_decoder g10_d768L12_cf50 pooledcross dim=768 layers=12 $C
echo SWEEP_GPU10_DONE
