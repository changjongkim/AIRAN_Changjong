#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J scaled_nrx
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scaled_nrx_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scaled_nrx_%j.err

# ===================================================================
# Scaled Neural Receiver: 4ant → 16ant → 64ant + L1 Interference
# Simulates real Massive MIMO Neural Receiver at different scales
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_heavy.py
NRX=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_scaled_neural_rx.py

TX=4; RX=4; CELLS=8; MCS=2

echo "============================================================"
echo "Scaled Neural Receiver + L1 (${CELLS}cells, 40GB)"
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
shifter --image=$IMAGE python3 $L1 snrx_baseline $TX $RX $CELLS $MCS
stop_mps
echo ""

# Each Neural Rx scale
for scale in small medium large xlarge; do
    label="snrx_${scale}"
    echo "=========================================="
    echo "MODE: Neural Rx ${scale}"
    echo "=========================================="
    start_mps $label
    PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $NRX 0 120 $scale &
    AI_PID=$!
    sleep 8
    echo "Neural Rx ($scale) PID=$AI_PID"
    shifter --image=$IMAGE python3 $L1 $label $TX $RX $CELLS $MCS
    kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
    stop_mps
    echo ""
done

# Baseline final
echo "=========================================="
echo "MODE: baseline_final"
echo "=========================================="
start_mps final
shifter --image=$IMAGE python3 $L1 snrx_baseline_final $TX $RX $CELLS $MCS
stop_mps

echo ""
echo "ALL COMPLETE"
