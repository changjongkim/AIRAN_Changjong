#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J thresh_v2
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/thresh_v2_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/thresh_v2_%j.err

# ===================================================================
# Threshold v2: Realistic LLM workloads
# Compare GPT-2 (124M) vs Qwen-7B vs HBM stress
# to find where real AI workloads fall on the interference curve
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_only.py
HBM=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
QWEN=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_qwen7b_stress.py

echo "============================================================"
echo "Threshold v2: GPT-2 vs Qwen-7B vs HBM stress"
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
    local ai_script=$2
    shift 2

    echo "=========================================="
    echo "MODE: $label"
    echo "=========================================="

    start_mps $label

    local AI_PID=""
    if [ -n "$ai_script" ]; then
        shifter --image=$IMAGE python3 $ai_script "$@" &
        AI_PID=$!
        sleep 15  # extra time for large model loading
        echo "AI worker PID=$AI_PID"
    fi

    shifter --image=$IMAGE python3 $L1 $label

    if [ -n "$AI_PID" ]; then
        kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
    fi

    stop_mps
    echo ""
}

# Baseline
run_mode "v2_baseline" ""

# GPT-2 (124M) — lightweight LLM
run_mode "v2_gpt2_b4" "$GPT2" 0 180

# Qwen-7B — realistic large LLM
run_mode "v2_qwen7b" "$QWEN" 0 180

# HBM stress reference points
run_mode "v2_hbm_0.5GB" "$HBM" 0 120 0.5
run_mode "v2_hbm_2GB"   "$HBM" 0 120 2.0

# Final baseline
run_mode "v2_baseline_final" ""

echo "============================================================"
echo "ALL MODES COMPLETE"
echo "============================================================"
