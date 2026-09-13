#!/bin/bash
# class-balanced populations (clean frame-source test; replaces the leaking offset ablations of streams c/d)
cd /workspace/intentionality; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="python -m microns_ambiguity.run_decoder2"
S17="dim=512 layers=8 epochs=30 batch=64 rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000 n=512 save_preds=1"
$R bal_iv    func_iv oricirc       balance_pop=1 seed=0 $S17
$R bal_is_s1 func_is oricirc pca=1 balance_pop=1 seed=1 $S17
$R bal_iv_s1 func_iv oricirc       balance_pop=1 seed=1 $S17
echo QUEUE_POD6E_DONE
