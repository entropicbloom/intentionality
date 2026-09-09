#!/bin/bash
cd /workspace/intentionality
while ! grep -q SWEEP_GPU9B_DONE logs/gpu9.log; do sleep 60; done
microns_ambiguity/sweeps/sweep_gpu12.sh > logs/gpu12.log 2>&1
