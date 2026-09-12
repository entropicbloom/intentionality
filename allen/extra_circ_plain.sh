#!/bin/bash
# (ran on pod 3 after sweep_circ_allen.sh; re-run on pod 4 via microns_ambiguity/sweeps/queue_pod4_allen.sh)
cd /workspace/intentionality
while ! grep -q SWEEP_CIRC_ALLEN_DONE logs/circ_allen.log; do sleep 60; done
for sp in 1 2; do
  python -m allen.run_decoder c_pc_2M_plain_sp$sp pooledcross n=256 dim=256 layers=4 epochs=60 split_seed=$sp seed=0 content=oricirc session=DG movie=nm1 rel_bias=1 device=cuda sel_reps=4 sel_frac=0.25 pops_per_epoch=5000 batch=64
done
echo EXTRA_CIRC_PLAIN_DONE
