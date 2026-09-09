#!/bin/zsh
cd /Users/flaessig/Documents/repos/intentionality
until grep -q SWEEP5_DONE microns_ambiguity/data/decoder2_s5.log; do sleep 20; done
P=.venv/bin/python
$P -m microns_ambiguity.run_decoder2 huge_twin_n512 func_is ori n=512 rel_bias=1 dim=384 layers=6 device=mps batch=8
$P -m microns_ambiguity.run_decoder2 big_twin_n512_long func_is ori n=512 rel_bias=1 dim=256 layers=4 device=mps batch=16 epochs=20 pops_per_epoch=4000
$P -m microns_ambiguity.run_decoder2 huge_twin_n512_rfdist func_is rf_dist n=512 rel_bias=1 dim=384 layers=6 device=mps batch=8
$P -m microns_ambiguity.run_decoder2 big_iv_n512 func_iv ori n=512 rel_bias=1 dim=256 layers=4 device=mps batch=16
echo SWEEP6_DONE
