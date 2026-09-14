#!/bin/bash
# v3 synthetic: like-tuned co-fluctuation at the in-vivo-matched strength (k=1.2, factor 1.75 vs in vivo V1 1.77), plus seeds to complete k=1.0 (factor 1.68) and k=0.25 (factor 1.09) to five
cd /workspace/intentionality; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Y="python -m microns_ambiguity.synthetic"
for ds in 0 1 2 3 4; do
  D="n=512 dim=256 layers=4 epochs=20 batch=64 seed=$ds rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000"
  $Y syn5_cocorr1.2_d$ds cocorr=1 cocorr_k=1.2 n_neurons=6000 T=120 data_seed=$ds $D
done
for ds in 2 3 4; do
  D="n=512 dim=256 layers=4 epochs=20 batch=64 seed=$ds rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000"
  $Y syn5_cocorr1_d$ds cocorr=1 cocorr_k=1 n_neurons=6000 T=120 data_seed=$ds $D
  $Y syn5_cocorr0.25_d$ds cocorr=1 cocorr_k=0.25 n_neurons=6000 T=120 data_seed=$ds $D
done
echo QUEUE_POD7B_DONE
