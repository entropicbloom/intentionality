#!/bin/zsh
cd /Users/flaessig/Documents/repos/intentionality
P=.venv/bin/python
$P -m microns_ambiguity.run_decoder2 n128_row func_iv ori n=128 rel_bias=0
$P -m microns_ambiguity.run_decoder2 n128_row_rf func_iv rf n=128 rel_bias=0
$P -m microns_ambiguity.run_decoder2 n128_rel func_iv ori n=128 rel_bias=1
$P -m microns_ambiguity.run_decoder2 n128_rel_rf func_iv rf n=128 rel_bias=1
$P -m microns_ambiguity.run_decoder2 n256_rel func_iv ori n=256 rel_bias=1
$P -m microns_ambiguity.run_decoder2 n256_rel_rf func_iv rf n=256 rel_bias=1
$P -m microns_ambiguity.run_decoder2 n512_rel func_iv ori n=512 rel_bias=1 device=mps
$P -m microns_ambiguity.run_decoder2 n512_rel_rf func_iv rf n=512 rel_bias=1 device=mps
$P -m microns_ambiguity.run_decoder2 n1024_rel func_iv ori n=1024 rel_bias=1 device=mps
$P -m microns_ambiguity.run_decoder2 n1024_rel_rf func_iv rf n=1024 rel_bias=1 device=mps
echo SWEEP1_DONE
