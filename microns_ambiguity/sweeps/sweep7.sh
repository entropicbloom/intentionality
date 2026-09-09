#!/bin/zsh
cd /Users/flaessig/Documents/repos/intentionality
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45
P=.venv/bin/python
$P -m microns_ambiguity.run_decoder2 big_twin_n512_long func_is ori n=512 rel_bias=1 dim=256 layers=4 device=mps batch=8 epochs=20 pops_per_epoch=4000 pca=1
$P -m microns_ambiguity.run_decoder2 big_twin_n512_rf_long func_is rf n=512 rel_bias=1 dim=256 layers=4 device=mps batch=8 epochs=20 pops_per_epoch=4000 pca=1
$P -m microns_ambiguity.run_decoder2 big_iv_n512 func_iv ori n=512 rel_bias=1 dim=256 layers=4 device=mps batch=8
$P -m microns_ambiguity.run_decoder2 big_twin_n1024 func_is ori n=1024 rel_bias=1 dim=256 layers=4 device=mps batch=4 pops_per_epoch=1000 pca=1
$P -m microns_ambiguity.run_decoder2 big_twin_n1024_rf func_is rf n=1024 rel_bias=1 dim=256 layers=4 device=mps batch=4 pops_per_epoch=1000 pca=1
$P -m microns_ambiguity.run_decoder2 huge_twin_n512 func_is ori n=512 rel_bias=1 dim=384 layers=6 device=mps batch=4 pca=1
$P -m microns_ambiguity.run_decoder2 huge_twin_n512_rfdist func_is rf_dist n=512 rel_bias=1 dim=384 layers=6 device=mps batch=4 pca=1
echo SWEEP7_DONE
