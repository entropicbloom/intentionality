#!/bin/bash
# v3 real-data runs, stream c: seeds for the within-scan rows and a matched mixed-scan reference at 128 neurons (all areas, both substrates)
cd /workspace/intentionality; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="python -m microns_ambiguity.run_decoder2"
W="dim=512 layers=8 epochs=30 batch=64 rel_bias=1 device=cuda pops_per_epoch=5000 within_scan=1 n=128 sel_reps=1 early_stop=0 save_preds=1"
M="dim=512 layers=8 epochs=30 batch=64 rel_bias=1 device=cuda early_stop=0.2 pops_per_epoch=5000 n=128 sel_reps=1 save_preds=1"
for s in 0 1 2; do
  $R c_is_17M_n128_s$s func_is oricirc pca=1 seed=$s $M
  $R c_iv_17M_n128_s$s func_iv oricirc       seed=$s $M
done
for s in 1 2; do
  $R ws_is_17M_s$s func_is oricirc pca=1 seed=$s $W
  $R ws_iv_17M_s$s func_iv oricirc       seed=$s $W
done
echo QUEUE_POD7C_DONE
