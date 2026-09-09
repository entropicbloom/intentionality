#!/bin/bash
cd /workspace/intentionality
allen/sweep_gpu4.sh > logs/gpu4.log 2>&1
allen/sweep_gpu7.sh > logs/gpu7.log 2>&1
allen/sweep_gpu8.sh > logs/gpu8.log 2>&1
allen/sweep_gpu5.sh > logs/gpu5.log 2>&1
allen/sweep_gpu10.sh > logs/gpu10.log 2>&1
echo ALLEN_STREAM_DONE >> logs/gpu10.log
