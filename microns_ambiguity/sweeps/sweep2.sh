#!/bin/zsh
cd /Users/flaessig/Documents/repos/intentionality
until grep -q SWEEP1_DONE microns_ambiguity/data/decoder2.log; do sleep 20; done
P=.venv/bin/python
$P -m microns_ambiguity.run_decoder2 twin_n256 func_is ori n=256 rel_bias=1
$P -m microns_ambiguity.run_decoder2 twin_n256_rf func_is rf n=256 rel_bias=1
$P -m microns_ambiguity.run_decoder2 n256_avg func_iv ori n=256 rel_bias=1
$P -m microns_ambiguity.run_decoder2 n256_avg_rf func_iv rf n=256 rel_bias=1
$P -m microns_ambiguity.run_decoder2 twin_n512 func_is ori n=512 rel_bias=1 device=mps
$P -m microns_ambiguity.run_decoder2 twin_n512_rf func_is rf n=512 rel_bias=1 device=mps
$P -m microns_ambiguity.run_decoder2 big_n256 func_iv ori n=256 rel_bias=1 dim=256 layers=4 device=mps
$P -m microns_ambiguity.run_decoder2 big_twin_n256 func_is ori n=256 rel_bias=1 dim=256 layers=4 device=mps
echo SWEEP2_DONE
