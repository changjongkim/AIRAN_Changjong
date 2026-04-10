#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J poc_compare
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/poc_compare_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/poc_compare_%j.err

# ===================================================================
# NVIDIA AI-RAN PoC Paper Comparison Experiment
# ===================================================================
# Paper setup:  GH200, MIG 40:60, 4T4R, 100MHz/30kHz, LLM inference
# Our setup:    A100, MPS emulation, 1T2R×multi-cell, 273PRB/30kHz,
#               GPT-2/ResNet-50/HBM stress
#
# Modes:
#   1. baseline         — L1 solo (RAN only)
#   2. same_gpu_hbm     — L1 + HBM stress on GPU0 (bandwidth interference)
#   3. same_gpu_gpt2    — L1 + GPT-2 on GPU0 (LLM, matching paper)
#   4. same_gpu_resnet  — L1 + ResNet-50 on GPU0 (compute-heavy AI)
#   5. diff_gpu_hbm     — L1:GPU0 + HBM:GPU1 (isolation baseline)
#   6. diff_gpu_gpt2    — L1:GPU0 + GPT-2:GPU1 (isolation baseline)
#   7. diff_gpu_resnet  — L1:GPU0 + ResNet-50:GPU1 (isolation baseline)
#   8. baseline_final   — consistency check
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_only.py
HBM=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py

echo "============================================================"
echo "NVIDIA AI-RAN PoC Comparison Experiment"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
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

# run_mode <label> <ai_script> <ai_gpu> [ai_extra_args...]
run_mode() {
    local label=$1
    local ai_script=$2
    local ai_gpu=$3
    shift 3
    local ai_args="$@"

    echo "=========================================="
    echo "MODE: $label"
    echo "=========================================="

    start_mps $label

    local AI_PID=""
    if [ -n "$ai_script" ]; then
        shifter --image=$IMAGE python3 $ai_script $ai_gpu 120 $ai_args &
        AI_PID=$!
        sleep 8  # warmup time for AI
        echo "AI worker PID=$AI_PID on GPU$ai_gpu"
    fi

    shifter --image=$IMAGE python3 $L1 $label

    if [ -n "$AI_PID" ]; then
        kill $AI_PID 2>/dev/null
        wait $AI_PID 2>/dev/null
    fi

    stop_mps
    echo ""
}

# ====== Experiments ======

# Baseline
run_mode "baseline" "" ""

# Same GPU (GPU0) — interference
run_mode "same_gpu_hbm"    "$HBM"    0 8
run_mode "same_gpu_gpt2"   "$GPT2"   0
run_mode "same_gpu_resnet" "$RESNET" 0

# Different GPU (GPU1) — isolation control
run_mode "diff_gpu_hbm"    "$HBM"    1 8
run_mode "diff_gpu_gpt2"   "$GPT2"   1
run_mode "diff_gpu_resnet" "$RESNET" 1

# Final baseline
run_mode "baseline_final" "" ""

echo ""
echo "============================================================"
echo "ALL MODES COMPLETE"
echo "============================================================"
echo ""

# Summary: extract all results
echo "SUMMARY:"
echo "--------------------------------------------------------------"
for f in /pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results/exp_baseline_*.json \
         /pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results/exp_same_gpu_*.json \
         /pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results/exp_diff_gpu_*.json \
         /pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results/exp_baseline_final_*.json; do
    [ -f "$f" ] && python3 -c "
import json, sys
d=json.load(open('$f'))
s=d['stats']
print(f\"{d['label']:<24} RX={s['mean_ms']:>7.3f}ms  P99={s['p99_ms']:>7.3f}ms  miss={d['miss_1ms']*100:>5.1f}%\")
" 2>/dev/null
done
