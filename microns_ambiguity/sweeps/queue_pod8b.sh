#!/bin/bash
# v4 synthetic: local-field like-tuned co-fluctuation (cocorr=2; residual shape matches cortex), three strengths x five seeds
cd /workspace/intentionality; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Y="python -m microns_ambiguity.synthetic"
for k in 0.5 1 2; do
  for ds in 0 1 2 3 4; do
    D="n=512 dim=256 layers=4 epochs=20 batch=64 seed=$ds rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000"
    $Y syn5_local${k}_d$ds cocorr=2 cocorr_k=$k n_neurons=6000 T=120 data_seed=$ds $D
  done
done
echo QUEUE_POD8B_DONE
