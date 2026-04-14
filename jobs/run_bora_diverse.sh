#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J bora_diverse
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/bora_diverse_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/bora_diverse_%j.err

# BORA vs NVIDIA baseline with diverse AI workloads
# Workloads from Exp-15/17: Neural Rx, GPT-2, ResNet, HBM stress, Matmul

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_graph.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py
AI_REAL=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_realistic_ai_stress.py
HBM=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py
CELLS=8
RESULT_DIR=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results

echo "============================================================"
echo "BORA vs Baseline: Diverse AI Workloads (40GB)"
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

run_compare() {
    local label=$1
    local config=$2  # B=nvidia, C=bora
    local ai_cmd=$3
    shift 3
    local ai_args="$@"

    local L1_SM=40; local AI_SM=60
    local AI_LOG="${RESULT_DIR}/ai_tp_${label}.txt"
    local config_name=$([ "$config" = "B" ] && echo "NVIDIA" || echo "BORA")

    echo "=========================================="
    echo "$config_name: $label"
    echo "=========================================="

    start_mps $label

    # AI background with throughput logging
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$AI_SM PYTHONPATH=$PYTHONPATH \
        shifter --image=$IMAGE bash -c "
cd /tmp
python3 $ai_cmd $ai_args &
AI_INNER=\$!
sleep 5
# Monitor AI iterations by checking process
START=\$(date +%s)
while kill -0 \$AI_INNER 2>/dev/null; do sleep 1; done
END=\$(date +%s)
echo \"duration \$((END-START))\" > $AI_LOG
" > ${RESULT_DIR}/ai_out_${label}.txt 2>&1 &
    AI_PID=$!
    sleep 8
    echo "AI PID=$AI_PID"

    # L1 foreground
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$L1_SM \
        shifter --image=$IMAGE python3 $L1 $label $CELLS 1

    kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null

    # Parse AI output
    echo "--- AI output ---"
    grep -E "done|iters|it/s|inf/s|TTI/s" ${RESULT_DIR}/ai_out_${label}.txt 2>/dev/null | tail -2

    stop_mps
    echo ""
}

# ====== Neural Rx (high) — in-line AI ======
run_compare "nv_neuralrx"   B "$AI_REAL" 0 75 neural_rx 1.0
run_compare "bora_neuralrx" C "$AI_REAL" 0 75 neural_rx 1.0

# ====== GPT-2 — LLM serving ======
run_compare "nv_gpt2"   B "$GPT2" 0 75
run_compare "bora_gpt2" C "$GPT2" 0 75

# ====== ResNet-50 bs128 — video analytics ======
run_compare "nv_resnet"   B "$RESNET" 0 75 128
run_compare "bora_resnet" C "$RESNET" 0 75 128

# ====== HBM stress 2GB — bandwidth extreme ======
run_compare "nv_hbm"   B "$HBM" 0 75 2
run_compare "bora_hbm" C "$HBM" 0 75 2

# ====== Matmul high — compute+bandwidth mixed ======
run_compare "nv_matmul"   B "$AI_REAL" 0 75 matmul 1.0
run_compare "bora_matmul" C "$AI_REAL" 0 75 matmul 1.0

echo "============================================================"
echo "ALL COMPLETE"
echo "============================================================"
