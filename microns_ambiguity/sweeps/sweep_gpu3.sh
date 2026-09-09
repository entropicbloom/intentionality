#!/bin/bash
# GPU sweep 3: MICrONS per-neuron decoder, capacity + label-free relation augmentation.
# Selection: 20 % slice of the training neurons, metric averaged over 4 covers; best epoch restored.
cd /workspace/intentionality
C="rel_bias=1 device=cuda early_stop=0.2 sel_reps=4 pops_per_epoch=5000 batch=64"
R="python -m microns_ambiguity.run_decoder2"
$R g3_is_ori_n512_d512L8      func_is ori pca=1 n=512 dim=512 layers=8  epochs=40 $C
$R g3_is_ori_n512_d512L8_cf50 func_is ori pca=1 n=512 dim=512 layers=8  epochs=40 cond_frac=0.5 $C
$R g3_is_ori_n512_d512L8_gd20 func_is ori pca=1 n=512 dim=512 layers=8  epochs=40 gram_drop=0.2 $C
$R g3_is_rf_n512_d512L8       func_is rf  pca=1 n=512 dim=512 layers=8  epochs=40 $C
$R g3_is_rf_n512_d512L8_cf50  func_is rf  pca=1 n=512 dim=512 layers=8  epochs=40 cond_frac=0.5 $C
$R g3_iv_ori_n512_d512L8      func_iv ori       n=512 dim=512 layers=8  epochs=40 $C
$R g3_iv_ori_n512_d512L8_cf50 func_iv ori       n=512 dim=512 layers=8  epochs=40 cond_frac=0.5 $C
$R g3_iv_rf_n512_d512L8_cf50  func_iv rf        n=512 dim=512 layers=8  epochs=40 cond_frac=0.5 $C
$R g3_is_rf_n512_d768L12_cf50 func_is rf  pca=1 n=512 dim=768 layers=12 epochs=60 cond_frac=0.5 $C
$R g3_is_ori_n512_d768L12_cf50 func_is ori pca=1 n=512 dim=768 layers=12 epochs=60 cond_frac=0.5 $C
echo SWEEP_GPU3_DONE
