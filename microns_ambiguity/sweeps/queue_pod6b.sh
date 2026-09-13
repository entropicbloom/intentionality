#!/bin/bash
# v2 synthetic block: 8 cells x 5 seeds selected on the raw error, plus strength sweeps of neuron-side anisotropy with an isotropic stimulus
cd /workspace/intentionality; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Y="python -m microns_ambiguity.synthetic"
for ds in 0 1 2 3 4; do
  D="n=512 dim=256 layers=4 epochs=20 batch=64 seed=$ds rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000"
  for c in 0 1; do for s in 0 1; do for t in 0 1; do
    $Y syn5_c${c}s${s}t${t}_d$ds count=$c sharp=$s stim=$t n_neurons=6000 T=120 data_seed=$ds $D
  done; done; done
done
for ds in 0 1; do
  D="n=512 dim=256 layers=4 epochs=20 batch=64 seed=$ds rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000"
  for k in 4 8 16 32; do $Y syn5_sharp${k}_d$ds sharp=1 sharp_k=$k n_neurons=6000 T=120 data_seed=$ds $D; done
  for g in 0.5 1 2 4; do $Y syn5_gain${g}_d$ds gain_k=$g n_neurons=6000 T=120 data_seed=$ds $D; done
done
echo QUEUE_POD6B_DONE
