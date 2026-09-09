#!/bin/zsh
cd /Users/flaessig/Documents/repos/intentionality
until grep -q SWEEP2_DONE microns_ambiguity/data/decoder2_s2.log; do sleep 20; done
P=.venv/bin/python
$P -m microns_ambiguity.run_decoder2 twin_n256_rfdist func_is rf_dist n=256 rel_bias=1
$P -m microns_ambiguity.run_decoder2 n512_rfdist func_iv rf_dist n=512 rel_bias=1 device=mps
$P -m microns_ambiguity.run_decoder2 n256_modD func_iv ori n=256 rel_bias=1
$P -m microns_ambiguity.run_decoder2 twin_n512_rfdist func_is rf_dist n=512 rel_bias=1 device=mps
echo SWEEP3_DONE
