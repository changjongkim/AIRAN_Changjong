#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J bora_eval
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/bora_eval_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/bora_eval_%j.err

# BORA Eval: Shell background + MPS (proven method)
# AI throughput saved to file by AI process

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_graph.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py
CELLS=8
RESULT_DIR=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results

echo "============================================================"
echo "BORA Eval: 40GB, Shell background + MPS"
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

run_eval() {
    local config=$1
    local ai_script=$2
    local ai_name=$3
    shift 3
    local ai_args="$@"

    local L1_SM=100; local AI_SM=100
    case $config in
        A) ;;
        B) L1_SM=40; AI_SM=60 ;;
        C) L1_SM=40; AI_SM=60 ;;
        D) L1_SM=40; AI_SM=60 ;;
    esac

    local label="eval_${config}_${ai_name}"
    echo "=========================================="
    echo "Config $config + $ai_name (SM L1=${L1_SM}% AI=${AI_SM}%)"
    echo "=========================================="

    start_mps $label

    # AI: shell background, output to file for throughput capture
    local AI_PID=""
    local AI_LOG="${RESULT_DIR}/ai_log_${label}.txt"
    if [ -n "$ai_script" ]; then
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$AI_SM PYTHONPATH=$PYTHONPATH \
            shifter --image=$IMAGE python3 $ai_script $ai_args > $AI_LOG 2>&1 &
        AI_PID=$!
        sleep 8
        echo "AI ($ai_name) PID=$AI_PID SM=${AI_SM}%"
    fi

    # L1: foreground with SM limit
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$L1_SM \
        shifter --image=$IMAGE python3 $L1 $label $CELLS 1

    # Kill AI and capture throughput
    if [ -n "$AI_PID" ]; then
        kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
        echo "--- AI throughput ($ai_name) ---"
        grep -E "done|iters|it/s|inf/s|running" $AI_LOG 2>/dev/null || echo "  (no output)"
    fi

    stop_mps
    echo ""
}

# Baseline
run_eval A "" "baseline"

# Config A: no protection (worst case)
run_eval A "$GPT2" "gpt2" 0 60
run_eval A "$RESNET" "resnet" 0 60 128

# Config B: MIG-like 40:60
run_eval B "$GPT2" "gpt2" 0 60
run_eval B "$RESNET" "resnet" 0 60 128

# Config C: MIG + Priority
run_eval C "$GPT2" "gpt2" 0 60
run_eval C "$RESNET" "resnet" 0 60 128

echo "============================================================"
echo "ALL COMPLETE"
echo "============================================================"
