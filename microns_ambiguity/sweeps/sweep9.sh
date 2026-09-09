#!/bin/zsh
# headline configurations, early-stopped on a held-out training slice, 3 seeds
cd /Users/flaessig/Documents/repos/intentionality
until grep -q SWEEP8_DONE microns_ambiguity/data/decoder2_s8.log; do sleep 20; done
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45
P=.venv/bin/python
for s in 0 1 2; do
  $P -m microns_ambiguity.run_decoder2 es_twin_ori_s$s func_is ori n=512 rel_bias=1 dim=256 layers=4 device=mps batch=8 epochs=12 pops_per_epoch=2500 early_stop=0.2 seed=$s pca=1
  $P -m microns_ambiguity.run_decoder2 es_twin_rf_s$s func_is rf n=512 rel_bias=1 dim=256 layers=4 device=mps batch=8 epochs=12 pops_per_epoch=2500 early_stop=0.2 seed=$s pca=1
  $P -m microns_ambiguity.run_decoder2 es_iv_ori_s$s func_iv ori n=512 rel_bias=1 dim=256 layers=4 device=mps batch=8 epochs=12 pops_per_epoch=2500 early_stop=0.2 seed=$s
  $P -m microns_ambiguity.run_decoder2 es_iv_rf_s$s func_iv rf n=512 rel_bias=1 dim=256 layers=4 device=mps batch=8 epochs=12 pops_per_epoch=2500 early_stop=0.2 seed=$s
  $P -m microns_ambiguity.run_decoder2 es_twin_rfdist_s$s func_is rf_dist n=512 rel_bias=1 dim=256 layers=4 device=mps batch=8 epochs=12 pops_per_epoch=2500 early_stop=0.2 seed=$s pca=1
done
echo SWEEP9_DONE
