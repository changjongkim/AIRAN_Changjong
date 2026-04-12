#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J heavy_40g
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/heavy_40g_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/heavy_40g_%j.err

# ===================================================================
# Heavy L1 on 40GB GPU — matches NVIDIA PoC paper's GPU class
# 4T4R × 20cell on 40GB = HBM ~44% (matches PoC's 30-40%)
#
# 3 experiment sets:
#   1. Interference: baseline / HBM / GPT-2 / ResNet / Qwen-7B
#   2. SM% sweep: 100% → 10%
#   3. Threshold: HBM 0.1GB → 4GB
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_heavy.py
HBM=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
QWEN7B=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_qwen7b_stress.py

TX=4; RX=4; CELLS=20; MCS=2

echo "============================================================"
echo "Heavy L1 on 40GB GPU (realistic PoC comparison)"
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

# ====== Part 1: Interference ======
echo ""
echo "====== Part 1: Interference ======"

# Baseline
echo "=========================================="
echo "MODE: baseline"
echo "=========================================="
start_mps baseline
shifter --image=$IMAGE python3 $L1 g40_baseline $TX $RX $CELLS $MCS
stop_mps
echo ""

# HBM stress 2GB (40GB에서 2GB는 상당한 비율)
echo "=========================================="
echo "MODE: + HBM 2GB"
echo "=========================================="
start_mps hbm
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $HBM 0 180 2 &
AI_PID=$!; sleep 8
shifter --image=$IMAGE python3 $L1 g40_hbm2g $TX $RX $CELLS $MCS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# GPT-2 (same GPU)
echo "=========================================="
echo "MODE: + GPT-2"
echo "=========================================="
start_mps gpt2
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $GPT2 0 180 &
AI_PID=$!; sleep 10
shifter --image=$IMAGE python3 $L1 g40_gpt2 $TX $RX $CELLS $MCS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# ResNet-50 bs=128
echo "=========================================="
echo "MODE: + ResNet bs128"
echo "=========================================="
start_mps resnet
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $RESNET 0 180 128 &
AI_PID=$!; sleep 10
shifter --image=$IMAGE python3 $L1 g40_resnet128 $TX $RX $CELLS $MCS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# Qwen-7B (fits in single 40GB GPU)
echo "=========================================="
echo "MODE: + Qwen-7B"
echo "=========================================="
start_mps qwen7b
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $QWEN7B 0 180 &
AI_PID=$!; sleep 15
shifter --image=$IMAGE python3 $L1 g40_qwen7b $TX $RX $CELLS $MCS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# ====== Part 2: SM% Sweep ======
echo ""
echo "====== Part 2: SM% Sweep ======"

for sm in 100 50 30 20 10; do
    echo "=========================================="
    echo "MODE: SM=${sm}%"
    echo "=========================================="
    start_mps sm$sm
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$sm shifter --image=$IMAGE \
        python3 $L1 g40_sm${sm} $TX $RX $CELLS $MCS
    stop_mps
    echo ""
done

# ====== Part 3: Threshold ======
echo ""
echo "====== Part 3: Threshold ======"

for size in 0.1 0.5 1.0 2.0; do
    label="g40_thresh_${size}GB"
    echo "=========================================="
    echo "MODE: HBM ${size}GB"
    echo "=========================================="
    start_mps $label
    PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $HBM 0 120 $size &
    AI_PID=$!; sleep 5
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
shifter --image=$IMAGE python3 $L1 g40_baseline_final $TX $RX $CELLS $MCS
stop_mps

echo ""
echo "============================================================"
echo "ALL COMPLETE"
echo "============================================================"
