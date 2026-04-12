#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J thresh_72b
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/thresh_72b_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/thresh_72b_%j.err

# ===================================================================
# Threshold with Qwen-72B (4GPU tensor parallel)
#
# Qwen-72B uses ALL 4 GPUs for tensor parallelism (~34GB/GPU)
# L1 runs on GPU0 → shares HBM with Qwen-72B's shard on GPU0
#
# This is the realistic AI-RAN scenario:
#   - Large LLM uses entire GPU cluster
#   - L1 must coexist on the same GPUs
#   - Long context (seq=2048) + batch > 1 = heavy bandwidth
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_only.py
QWEN72B=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_qwen72b_stress.py

echo "============================================================"
echo "Threshold: Qwen-72B (4GPU TP) + L1 on GPU0"
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

# 1. Baseline — L1 alone on GPU0
echo "=========================================="
echo "MODE: baseline"
echo "=========================================="
start_mps baseline
shifter --image=$IMAGE python3 $L1 72b_baseline
stop_mps
echo ""

# 2. Qwen-72B (batch=1, seq=2048) + L1
echo "=========================================="
echo "MODE: qwen72b_b1_s2048"
echo "=========================================="
start_mps qwen72b_b1
PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH shifter --image=$IMAGE python3 $QWEN72B 300 1 2048 &
AI_PID=$!
echo "Waiting for Qwen-72B to load (may take 60-90s)..."
sleep 90
echo "Qwen-72B PID=$AI_PID, measuring L1..."
shifter --image=$IMAGE python3 $L1 72b_qwen72b_b1_s2048
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# 3. Qwen-72B (batch=4, seq=2048) — heavier
echo "=========================================="
echo "MODE: qwen72b_b4_s2048"
echo "=========================================="
start_mps qwen72b_b4
PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH shifter --image=$IMAGE python3 $QWEN72B 300 4 2048 &
AI_PID=$!
sleep 90
echo "Qwen-72B PID=$AI_PID, measuring L1..."
shifter --image=$IMAGE python3 $L1 72b_qwen72b_b4_s2048
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# 4. Baseline final
echo "=========================================="
echo "MODE: baseline_final"
echo "=========================================="
start_mps final
shifter --image=$IMAGE python3 $L1 72b_baseline_final
stop_mps

echo ""
echo "============================================================"
echo "ALL MODES COMPLETE"
echo "============================================================"
