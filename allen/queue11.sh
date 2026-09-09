#!/bin/bash
cd /workspace/intentionality
while ! grep -q ALLEN_STREAM_DONE logs/gpu10.log; do sleep 60; done
allen/sweep_gpu11.sh > logs/gpu11.log 2>&1
