#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J prio_realist
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/prio_realist_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/prio_realist_%j.err

# ===================================================================
# Stream Priority with Realistic AI-RAN Workloads
# 4T4R × 8cell L1 + Neural Receiver / Video Analytics / Continuous Matmul
# Compare default vs HIGH priority for each
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_heavy.py
AI=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_realistic_ai_stress.py

TX=4; RX=4; CELLS=8; MCS=2

echo "============================================================"
echo "Realistic AI Workloads + Stream Priority (40GB, ${CELLS}cells)"
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
    local ai_type=$2
    local ai_intensity=$3

    echo "=========================================="
    echo "MODE: $label"
    echo "=========================================="

    start_mps $label

    if [ -n "$ai_type" ]; then
        PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE \
            python3 $AI 0 180 $ai_type $ai_intensity &
        AI_PID=$!
        sleep 8
        echo "AI PID=$AI_PID ($ai_type, intensity=$ai_intensity)"
    fi

    shifter --image=$IMAGE python3 $L1 $label $TX $RX $CELLS $MCS

    if [ -n "$ai_type" ]; then
        kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
    fi

    stop_mps
    echo ""
}

# Baseline
run_mode "real_baseline" "" ""

# Neural Receiver — low/medium/high intensity
run_mode "real_neuralrx_low"  "neural_rx" "0.2"
run_mode "real_neuralrx_mid"  "neural_rx" "0.5"
run_mode "real_neuralrx_high" "neural_rx" "1.0"

# Video Analytics — medium
run_mode "real_video_mid" "video" "0.5"

# Continuous Matmul — medium/high (bandwidth-heavy)
run_mode "real_matmul_mid"  "matmul" "0.5"
run_mode "real_matmul_high" "matmul" "1.0"

# Baseline final
run_mode "real_baseline_final" "" ""

echo "============================================================"
echo "ALL COMPLETE"
echo "============================================================"
