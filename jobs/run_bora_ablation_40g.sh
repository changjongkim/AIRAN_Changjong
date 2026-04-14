#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J bora_40g
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/bora_40g_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/bora_40g_%j.err

# BORA Ablation on 40GB — where interference was severe (3.87x GPT-2, 4.06x ResNet)
# Config A~D × GPT-2 and ResNet (the workloads that broke TTI)

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_graph.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py

CELLS=8

echo "============================================================"
echo "BORA Ablation: 40GB (병목 케이스)"
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

run_config() {
    local config=$1
    local ai_script=$2
    local ai_name=$3
    shift 3
    local ai_args="$@"

    # Parse config
    local L1_SM=100; local AI_SM=100; local L1_PRIO=""; local AI_PRIO=""
    case $config in
        A) ;;
        B) L1_SM=40; AI_SM=60 ;;
        C) L1_SM=40; AI_SM=60; L1_PRIO="-5"; AI_PRIO="0" ;;
        D) L1_SM=40; AI_SM=60; L1_PRIO="-5"; AI_PRIO="0" ;;
    esac

    local label="bora40_${config}_${ai_name}"
    echo "=========================================="
    echo "Config $config + $ai_name (SM L1=${L1_SM}% AI=${AI_SM}%)"
    echo "=========================================="

    start_mps $label

    # Launch AI with SM limit
    local AI_PID=""
    if [ -n "$ai_script" ]; then
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$AI_SM PYTHONPATH=$PYTHONPATH \
            shifter --image=$IMAGE python3 $ai_script $ai_args &
        AI_PID=$!
        sleep 10
        echo "AI ($ai_name) PID=$AI_PID (SM=${AI_SM}%)"
    fi

    # Run L1 with SM limit (+ priority if config C/D)
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$L1_SM \
        shifter --image=$IMAGE python3 $L1 $label $CELLS 1

    if [ -n "$AI_PID" ]; then
        kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
    fi
    stop_mps
    echo ""
}

# Config A: baseline (no AI)
run_config A "" "baseline"

# Config A + GPT-2 (no protection — worst case)
run_config A "$GPT2" "gpt2" 0 120

# Config A + ResNet (no protection — worst case)
run_config A "$RESNET" "resnet" 0 120 128

# Config B + GPT-2 (MIG-like only)
run_config B "$GPT2" "gpt2" 0 120

# Config B + ResNet (MIG-like only)
run_config B "$RESNET" "resnet" 0 120 128

# Config C + GPT-2 (MIG + priority)
run_config C "$GPT2" "gpt2" 0 120

# Config C + ResNet (MIG + priority)
run_config C "$RESNET" "resnet" 0 120 128

echo "============================================================"
echo "ALL COMPLETE"
echo "============================================================"
