#!/bin/zsh
cd /Users/flaessig/Documents/repos/intentionality
P=.venv/bin/python
$P -m microns_ambiguity.run_decoder2 big_twin_n512 func_is ori n=512 rel_bias=1 dim=256 layers=4 device=mps batch=16 pops_per_epoch=2000
$P -m microns_ambiguity.run_decoder2 big_twin_n512_rfdist func_is rf_dist n=512 rel_bias=1 dim=256 layers=4 device=mps batch=16
$P -m microns_ambiguity.run_decoder2 huge_twin_n256 func_is ori n=256 rel_bias=1 dim=384 layers=6 device=mps batch=16
$P -m microns_ambiguity.run_decoder2 big_twin_n512_rf func_is rf n=512 rel_bias=1 dim=256 layers=4 device=mps batch=16
$P -m microns_ambiguity.run_decoder2 big_twin_n256_long func_is ori n=256 rel_bias=1 dim=256 layers=4 device=mps batch=16 epochs=20 pops_per_epoch=4000
$P -m microns_ambiguity.run_decoder2 huge_twin_n256_rfdist func_is rf_dist n=256 rel_bias=1 dim=384 layers=6 device=mps batch=16
$P -m microns_ambiguity.run_decoder2 twin_n1024_rfdist func_is rf_dist n=1024 rel_bias=1 device=mps batch=8 pops_per_epoch=1000
echo SWEEP5_DONE
