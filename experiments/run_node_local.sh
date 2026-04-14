#!/bin/bash
# Per-node script: starts MPS locally, runs AI background, runs L1.
# Called via srun on each node.
# Args: <label> <cells> <ai_script> [ai_args...]

LABEL=$1
CELLS=$2
AI_SCRIPT=$3
shift 3
AI_ARGS="$@"

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH
IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_graph.py

# Start MPS locally on this node
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${LABEL}_$$
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${LABEL}_$$
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d 2>/dev/null
sleep 1

# Start AI background on this node
AI_PID=""
if [ -n "$AI_SCRIPT" ]; then
    PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $AI_SCRIPT $AI_ARGS &
    AI_PID=$!
    sleep 8
    echo "[$(hostname)] AI PID=$AI_PID started"
fi

# Run L1 on this node
shifter --image=$IMAGE python3 $L1 ${LABEL}_$(hostname) $CELLS 1

# Cleanup
if [ -n "$AI_PID" ]; then
    kill $AI_PID 2>/dev/null
    wait $AI_PID 2>/dev/null
fi
echo quit | nvidia-cuda-mps-control 2>/dev/null
