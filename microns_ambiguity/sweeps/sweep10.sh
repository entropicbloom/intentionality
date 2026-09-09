#!/bin/zsh
# RF: capacity, population, regularization at the 20-epoch budget
cd /Users/flaessig/Documents/repos/intentionality
until grep -q SWEEP9_DONE microns_ambiguity/data/decoder2_s9.log; do sleep 20; done
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45
P=.venv/bin/python
$P -m microns_ambiguity.run_decoder2 huge_twin_n512_rf_long func_is rf n=512 rel_bias=1 dim=384 layers=6 device=mps batch=4 epochs=20 pops_per_epoch=4000 early_stop=0.15 pca=1
$P -m microns_ambiguity.run_decoder2 big_twin_n1024_rf_long func_is rf n=1024 rel_bias=1 dim=256 layers=4 device=mps batch=4 epochs=20 pops_per_epoch=2000 early_stop=0.15 pca=1
$P -m microns_ambiguity.run_decoder2 big_twin_n512_rf_drop func_is rf n=512 rel_bias=1 dim=256 layers=4 device=mps batch=8 epochs=20 pops_per_epoch=4000 early_stop=0.15 dropout=0.25 pca=1
$P -m microns_ambiguity.run_decoder2 huge_twin_n512_ori_es func_is ori n=512 rel_bias=1 dim=384 layers=6 device=mps batch=4 epochs=16 pops_per_epoch=3000 early_stop=0.15 pca=1
echo SWEEP10_DONE
