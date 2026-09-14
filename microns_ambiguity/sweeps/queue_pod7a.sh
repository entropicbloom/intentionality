#!/bin/bash
# v3 real-data runs, stream a: balanced-training cross-area transfer (prior-free rotation test) and seeds for the cross-area rows
cd /workspace/intentionality; mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="python -m microns_ambiguity.run_decoder2"
C="dim=512 layers=8 epochs=30 batch=64 rel_bias=1 device=cuda early_stop=0.2 pops_per_epoch=5000 save_preds=1 sel_reps=1"
# A. training populations class-balanced (balance_train=1), test populations unrestricted: does the V1 decoder still rotate AL by ~90 degrees without a training prior?
for s in 0 1; do
  $R bt_V1toAL_is_n128_s$s func_is oricirc pca=1 area_train=V1 area_test=AL n=128 balance_train=1 seed=$s $C
  $R bt_V1toV1_is_n128_s$s func_is oricirc pca=1 area_train=V1 area_test=V1 n=128 balance_train=1 seed=$s $C
  $R bt_V1toRL_is_s$s       func_is oricirc pca=1 area_train=V1 area_test=RL n=512 balance_train=1 seed=$s $C
done
# B. seeds 1, 2 for the standard cross-area rows
for s in 1 2; do
  $R m_V1toAL_is_n128_s$s func_is oricirc pca=1 area_train=V1 area_test=AL n=128 seed=$s $C
  $R m_V1toV1_is_n128_s$s func_is oricirc pca=1 area_train=V1 area_test=V1 n=128 seed=$s $C
  $R m_V1toRL_is_s$s       func_is oricirc pca=1 area_train=V1 area_test=RL n=512 seed=$s $C
done
echo QUEUE_POD7A_DONE
