#!/bin/bash
# synthetic: shared variability among like-tuned neurons peaked at 90° (the dominant-class co-correlation seen in V1/AL), isotropic stimulus, uniform counts
cd /workspace/intentionality; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Y="python -m microns_ambiguity.synthetic"
for ds in 0 1 2 3 4; do
  D="n=512 dim=256 layers=4 epochs=20 batch=64 seed=$ds rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000"
  $Y syn5_cocorr_d$ds cocorr=1 cocorr_k=0.5 n_neurons=6000 T=120 data_seed=$ds $D
done
for ds in 0 1; do
  D="n=512 dim=256 layers=4 epochs=20 batch=64 seed=$ds rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000"
  for k in 0.25 1; do $Y syn5_cocorr${k}_d$ds cocorr=1 cocorr_k=$k n_neurons=6000 T=120 data_seed=$ds $D; done
  $Y syn5_cocorr_stim_d$ds cocorr=1 cocorr_k=0.5 stim=1 n_neurons=6000 T=120 data_seed=$ds $D
done
echo QUEUE_POD6F_DONE
