#!/bin/zsh
cd /Users/flaessig/Documents/repos/intentionality
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45
C="content=ori session=DG movie=nm1 rel_bias=1 batch=16 device=mps"
.venv/bin/python -m allen.run_decoder dg_pc_s1 pooledcross n=128 dim=256 layers=4 epochs=12 pops_per_epoch=2500 seed=1 $=C
.venv/bin/python -m allen.run_decoder dg_pc_s2 pooledcross n=128 dim=256 layers=4 epochs=12 pops_per_epoch=2500 seed=2 $=C
.venv/bin/python -m allen.run_decoder dg_pc_n256 pooledcross n=256 dim=256 layers=4 epochs=12 pops_per_epoch=2500 $=C
.venv/bin/python -m allen.run_decoder dg_pc_big pooledcross n=128 dim=384 layers=6 epochs=20 pops_per_epoch=2500 $=C
.venv/bin/python -m allen.run_decoder dg_pc_long pooledcross n=128 dim=256 layers=4 epochs=30 pops_per_epoch=2500 $=C
.venv/bin/python -m allen.run_decoder dg_pc_n512 pooledcross n=512 dim=256 layers=4 epochs=12 pops_per_epoch=1500 batch=8 $=C
echo SWEEP_DG2_DONE
