#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J scale1n_v2
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scale1n_v2_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scale1n_v2_%j.err

# Shell background & + MPS — same method as Exp-17

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_graph.py
AI=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_realistic_ai_stress.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py

CELLS=8

echo "============================================================"
echo "Scale 1N: CUDA Graph ${CELLS}cell + AI (40GB, MPS)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

start_mps() {
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${1}_$$
    export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${1}_$$
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
    nvidia-cuda-mps-control -d 2>/dev/null
    sleep 1
}
stop_mps() {
    echo quit | nvidia-cuda-mps-control 2>/dev/null
    sleep 1
}

run_mode() {
    local label=$1
    local ai_cmd=$2
    shift 2

    echo "=========================================="
    echo "MODE: $label"
    echo "=========================================="
    start_mps $label

    local AI_PID=""
    if [ -n "$ai_cmd" ]; then
        PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $ai_cmd "$@" &
        AI_PID=$!
        sleep 8
        echo "AI PID=$AI_PID"
    fi

    shifter --image=$IMAGE python3 $L1 $label $CELLS 1

    if [ -n "$AI_PID" ]; then
        kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
    fi
    stop_mps
    echo ""
}

run_mode "s1n_baseline" ""
run_mode "s1n_neuralrx" "$AI" 0 120 neural_rx 1.0
run_mode "s1n_gpt2" "$GPT2" 0 120
run_mode "s1n_resnet" "$RESNET" 0 120 128

echo "ALL COMPLETE"
