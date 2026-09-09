#!/bin/zsh
cd /Users/flaessig/Documents/repos/intentionality
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45
COMMON="content=ori session=DG n=128 dim=256 layers=4 rel_bias=1 epochs=12 pops_per_epoch=2500 batch=16 device=mps"
.venv/bin/python -m allen.run_decoder dg_ori_cross_nm1 cross movie=nm1 $=COMMON
.venv/bin/python -m allen.run_decoder dg_ori_pooledcross_nm1 pooledcross movie=nm1 $=COMMON
.venv/bin/python -m allen.run_decoder dg_ori_cross_both cross movie=both $=COMMON
.venv/bin/python -m allen.run_decoder dg_ori_pooledcross_both pooledcross movie=both $=COMMON
echo SWEEP_DG_DONE
