#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J neuralrx
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/neuralrx_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/neuralrx_%j.err

# ===================================================================
# REAL Neural Receiver (TensorRT) + L1 Interference
# This is the actual AI-RAN in-line AI workload
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_heavy.py
NRX=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_neural_rx_stress.py

TX=4; RX=4; CELLS=8; MCS=2

echo "============================================================"
echo "Neural Receiver (TensorRT) + L1 Interference (40GB)"
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

# 1. Baseline
echo "=========================================="
echo "MODE: baseline (L1 only)"
echo "=========================================="
start_mps baseline
shifter --image=$IMAGE python3 $L1 nrx_baseline $TX $RX $CELLS $MCS
stop_mps
echo ""

# 2. L1 + Neural Receiver (same GPU)
echo "=========================================="
echo "MODE: L1 + Neural Receiver (real AI-RAN)"
echo "=========================================="
start_mps nrx
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $NRX 0 180 &
AI_PID=$!
sleep 30  # TRT build takes time
echo "Neural Rx PID=$AI_PID"
shifter --image=$IMAGE python3 $L1 nrx_with_neuralrx $TX $RX $CELLS $MCS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# 3. Baseline final
echo "=========================================="
echo "MODE: baseline_final"
echo "=========================================="
start_mps final
shifter --image=$IMAGE python3 $L1 nrx_baseline_final $TX $RX $CELLS $MCS
stop_mps

echo ""
echo "ALL COMPLETE"
