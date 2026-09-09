#!/bin/zsh
cd /Users/flaessig/Documents/repos/intentionality
until grep -q SWEEP3_DONE microns_ambiguity/data/decoder2_s3.log; do sleep 20; done
P=.venv/bin/python
$P -m microns_ambiguity.run_decoder2 twin_n1024_rf func_is rf n=1024 rel_bias=1 device=mps
$P -m microns_ambiguity.run_decoder2 twin_n1024_rfdist func_is rf_dist n=1024 rel_bias=1 device=mps
$P -m microns_ambiguity.run_decoder2 twin_n512_long func_is ori n=512 rel_bias=1 device=mps epochs=20 pops_per_epoch=4000
$P -m microns_ambiguity.run_decoder2 twin_n512_rfdist_long func_is rf_dist n=512 rel_bias=1 device=mps epochs=20 pops_per_epoch=4000
echo SWEEP4_DONE
