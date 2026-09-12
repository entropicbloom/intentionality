#!/bin/bash
# Raw-activity reference decoders: same transformer and protocol, tokens from response vectors instead of Gram rows.
cd /workspace/intentionality; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="python -m microns_ambiguity.run_decoder2"
S17="dim=512 layers=8 epochs=30 batch=64 seed=0 n=512 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000 save_preds=1"
# (a) activity tokens, plain attention, no Gram anywhere
$R a_iv_17M_ori func_iv oricirc       input_mode=act rel_bias=0 $S17
$R a_is_17M_ori func_is oricirc pca=1 input_mode=act rel_bias=0 $S17
$R a_iv_17M_rf  func_iv rf            input_mode=act rel_bias=0 $S17
$R a_is_17M_rf  func_is rf      pca=1 input_mode=act rel_bias=0 $S17
# (b) activity tokens + Gram as attention bias
$R ab_iv_17M_ori func_iv oricirc       input_mode=act rel_bias=1 $S17
$R ab_is_17M_ori func_is oricirc pca=1 input_mode=act rel_bias=1 $S17
# (c) linear per-neuron readout of the response vector (no population context)
L="dim=512 layers=0 epochs=30 batch=64 seed=0 n=512 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000"
$R lin_iv_ori func_iv oricirc       input_mode=act rel_bias=0 $L
$R lin_is_ori func_is oricirc pca=1 input_mode=act rel_bias=0 $L
$R lin_iv_rf  func_iv rf            input_mode=act rel_bias=0 $L
$R lin_is_rf  func_is rf      pca=1 input_mode=act rel_bias=0 $L
echo QUEUE_POD5_DONE
