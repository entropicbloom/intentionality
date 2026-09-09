#!/bin/bash
cd /workspace/intentionality
( while pgrep -f sweep_gpu1.sh > /dev/null; do sleep 30; done; allen/../microns_ambiguity/sweeps/sweep_gpu3.sh > logs/gpu3.log 2>&1 ) &
( while pgrep -f sweep_gpu2.sh > /dev/null; do sleep 30; done; allen/sweep_gpu4.sh > logs/gpu4.log 2>&1 ) &
wait
