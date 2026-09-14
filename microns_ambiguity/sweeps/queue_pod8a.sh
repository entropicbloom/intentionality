#!/bin/bash
# v4: Gram row without the three row statistics (complement of the row-statistics-only control), twin and in vivo
cd /workspace/intentionality; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="python -m microns_ambiguity.run_decoder2"
S17="dim=512 layers=8 epochs=30 batch=64 rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000 n=512 save_preds=1"
$R c_is_17M_nostats func_is oricirc pca=1 use_stats=0 seed=0 $S17
$R c_iv_17M_nostats func_iv oricirc       use_stats=0 seed=0 $S17
echo QUEUE_POD8A_DONE
