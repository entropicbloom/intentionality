#!/bin/bash
# leak control for the residual ablation: random label-keyed offsets of the residual's size (K=8 and K=24), twin
cd /workspace/intentionality; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="python -m microns_ambiguity.run_decoder2"
S17="dim=512 layers=8 epochs=30 batch=64 rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000 n=512 save_preds=1"
$R abl_null_is   func_is oricirc pca=1 ablate=null ablate_k=8  seed=0 $S17
$R abl_null24_is func_is oricirc pca=1 ablate=null ablate_k=24 seed=0 $S17
echo QUEUE_POD6D_DONE
