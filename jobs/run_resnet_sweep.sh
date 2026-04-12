#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J resnet_sweep
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/resnet_sweep_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/resnet_sweep_%j.err

# ===================================================================
# ResNet-50 Batch Size Sweep
# ResNet uses more continuous bandwidth than LLMs due to
# frequent activation read/write per conv layer.
# Larger batch → larger activations → more bandwidth.
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_only.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py

echo "============================================================"
echo "ResNet-50 Batch Size Sweep — Interference Threshold"
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

# Baseline
echo "=========================================="
echo "MODE: baseline"
echo "=========================================="
start_mps baseline
shifter --image=$IMAGE python3 $L1 resnet_baseline
stop_mps
echo ""

# Sweep batch sizes: 1, 8, 16, 32, 64, 128, 256
for bs in 1 8 16 32 64 128 256; do
    label="resnet_bs${bs}"
    echo "=========================================="
    echo "MODE: $label"
    echo "=========================================="
    start_mps $label
    PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $RESNET 0 120 $bs &
    AI_PID=$!
    sleep 8
    echo "ResNet bs=$bs PID=$AI_PID"
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
shifter --image=$IMAGE python3 $L1 resnet_baseline_final
stop_mps

echo ""
echo "============================================================"
echo "ALL MODES COMPLETE"
echo "============================================================"
