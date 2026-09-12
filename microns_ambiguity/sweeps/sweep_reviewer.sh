#!/bin/bash
# Follow-ups to the reviewer's concerns (all MICrONS, 17M standard config, circular regression unless rf).
# Starts once the MICrONS regression sweep is done; runs alongside the Allen remainder.
cd /workspace/intentionality
while ! grep -q SWEEP_CIRC_MICRONS_DONE logs/circ_microns.log; do sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="python -m microns_ambiguity.run_decoder2"
B="dim=512 layers=8 epochs=30 batch=64 seed=0 rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000 n=512"
# 1+3. extra neuron splits with saved predictions (scan/area decomposition, per-orientation error, class-balanced score)
$R c_is_17M_sp1   func_is oricirc pca=1 split_seed=1 save_preds=1 $B
$R c_iv_17M_sp1   func_iv oricirc       split_seed=1 save_preds=1 $B
$R r_iv_17M_s0    func_iv rf                         save_preds=1 $B
$R r_is_17M_s0    func_is rf      pca=1              save_preds=1 $B
$R c_is_17M_sp2   func_is oricirc pca=1 split_seed=2 save_preds=1 $B
$R c_iv_17M_sp2   func_iv oricirc       split_seed=2 save_preds=1 $B
# 4. cross-stimulus test: training Grams from half the stimulus bins, test Grams from the other half
$R c_iv_17M_bins  func_iv oricirc bins=1 save_preds=1 $B
$R c_is_17M_bins  func_is oricirc bins=1 save_preds=1 $B
# 2. architecture ablations on the twin: equivariant (row statistics + attention bias) and statistics only
$R c_is_17M_statsbias func_is oricirc pca=1 row_proj=0 rel_bias=1 dim=512 layers=8 epochs=30 batch=64 seed=0 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000 n=512
$R c_is_17M_statsonly func_is oricirc pca=1 row_proj=0 rel_bias=0 dim=512 layers=8 epochs=30 batch=64 seed=0 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000 n=512
echo SWEEP_REVIEWER_DONE
