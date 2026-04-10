#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J threshold
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/threshold_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/threshold_%j.err

# ===================================================================
# Threshold Experiment: Find the bandwidth level where L1 breaks
#
# Sweeps HBM copy size from 0.1GB to 8GB in fine steps
# to find exact threshold where L1 deadline miss starts
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_only.py
HBM=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py

echo "============================================================"
echo "Threshold Experiment: HBM Bandwidth Interference Curve"
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

# Baseline first
echo "=========================================="
echo "MODE: baseline"
echo "=========================================="
start_mps baseline
shifter --image=$IMAGE python3 $L1 thresh_baseline
stop_mps
echo ""

# Sweep HBM copy sizes: fine-grained from 0.1GB to 8GB
for size in 0.1 0.2 0.5 1.0 2.0 4.0 8.0; do
    label="thresh_hbm_${size}GB"
    echo "=========================================="
    echo "MODE: $label"
    echo "=========================================="
    start_mps $label
    shifter --image=$IMAGE python3 $HBM 0 120 $size &
    AI_PID=$!
    sleep 5
    echo "HBM stress ${size}GB PID=$AI_PID"
    shifter --image=$IMAGE python3 $L1 $label
    kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
    stop_mps
    echo ""
done

# Final baseline
echo "=========================================="
echo "MODE: baseline_final"
echo "=========================================="
start_mps final
shifter --image=$IMAGE python3 $L1 thresh_baseline_final
stop_mps
echo ""

echo "============================================================"
echo "ALL MODES COMPLETE"
echo "============================================================"
