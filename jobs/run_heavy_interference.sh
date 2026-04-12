#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J heavy_intf
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/heavy_intf_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/heavy_intf_%j.err

# ===================================================================
# Heavy L1 (4T4R × 20cell) + AI Interference
# Realistic gNB-level L1 load + various AI workloads
#
# Modes:
#   1. 4T4R×20cell baseline
#   2. 4T4R×20cell + HBM stress 4GB (same GPU)
#   3. 4T4R×20cell + Qwen-72B (same node, 4GPU TP)
#   4. 4T4R×20cell + ResNet-50 bs=128 (same GPU)
#   5. 4T4R×20cell + GPT-2 (same GPU)
#   6. 4T4R×20cell baseline_final
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_heavy.py
HBM=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
QWEN=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_qwen72b_stress.py

# L1 config: 4T4R × 20 cells
TX=4; RX=4; CELLS=20; MCS=2

echo "============================================================"
echo "Heavy L1 Interference: ${TX}T${RX}R × ${CELLS}cells + AI"
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

# 1. Baseline
echo "=========================================="
echo "MODE: baseline (${TX}T${RX}R × ${CELLS}cells)"
echo "=========================================="
start_mps baseline
shifter --image=$IMAGE python3 $L1 heavy_baseline $TX $RX $CELLS $MCS
stop_mps
echo ""

# 2. + HBM stress 4GB (same GPU0)
echo "=========================================="
echo "MODE: + HBM stress 4GB"
echo "=========================================="
start_mps hbm
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $HBM 0 180 4 &
AI_PID=$!
sleep 8
shifter --image=$IMAGE python3 $L1 heavy_hbm4g $TX $RX $CELLS $MCS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# 3. + Qwen-72B (4GPU TP, same node)
echo "=========================================="
echo "MODE: + Qwen-72B (4GPU TP)"
echo "=========================================="
start_mps qwen
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $QWEN 300 1 2048 &
AI_PID=$!
echo "Waiting 120s for Qwen-72B to load..."
sleep 120
shifter --image=$IMAGE python3 $L1 heavy_qwen72b $TX $RX $CELLS $MCS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# 4. + ResNet-50 bs=128 (same GPU0)
echo "=========================================="
echo "MODE: + ResNet-50 bs=128"
echo "=========================================="
start_mps resnet
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $RESNET 0 180 128 &
AI_PID=$!
sleep 10
shifter --image=$IMAGE python3 $L1 heavy_resnet128 $TX $RX $CELLS $MCS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# 5. + GPT-2 (same GPU0)
echo "=========================================="
echo "MODE: + GPT-2"
echo "=========================================="
start_mps gpt2
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $GPT2 0 180 &
AI_PID=$!
sleep 10
shifter --image=$IMAGE python3 $L1 heavy_gpt2 $TX $RX $CELLS $MCS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# 6. Baseline final
echo "=========================================="
echo "MODE: baseline_final"
echo "=========================================="
start_mps final
shifter --image=$IMAGE python3 $L1 heavy_baseline_final $TX $RX $CELLS $MCS
stop_mps

echo ""
echo "============================================================"
echo "ALL MODES COMPLETE"
echo "============================================================"
