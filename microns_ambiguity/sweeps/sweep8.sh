#!/bin/zsh
cd /Users/flaessig/Documents/repos/intentionality
until grep -q SWEEP7_DONE microns_ambiguity/data/decoder2_s7.log; do sleep 20; done
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45
P=.venv/bin/python
$P -m microns_ambiguity.run_decoder2 big_iv_n512_rf func_iv rf n=512 rel_bias=1 dim=256 layers=4 device=mps batch=8
$P -m microns_ambiguity.run_decoder2 big_iv_n512_rfdist func_iv rf_dist n=512 rel_bias=1 dim=256 layers=4 device=mps batch=8
echo SWEEP8_DONE
