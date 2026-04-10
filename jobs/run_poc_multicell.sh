#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J poc_multicell
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/poc_multicell_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/poc_multicell_%j.err

# ===================================================================
# Multi-cell PoC Comparison
# Increases L1 GPU usage by running multiple cell pipelines
# to match NVIDIA PoC paper's 30-40% GPU utilization
#
# Each cuPHY cell uses ~770MB HBM → 8 cells ≈ 6.2GB (15-20% of 40GB)
# With channel processing overhead → ~30-40% GPU activity
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_multicell.py
HBM=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py

echo "============================================================"
echo "Multi-cell PoC Comparison (matching NVIDIA AI-RAN paper)"
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

NCELLS=8  # 8 cells to approach 30-40% GPU usage

# run_mode <label> <ai_script> <ai_gpu> [ai_args...]
run_mode() {
    local label=$1
    local ai_script=$2
    local ai_gpu=$3
    shift 3

    echo "=========================================="
    echo "MODE: $label (${NCELLS} cells)"
    echo "=========================================="

    start_mps $label

    local AI_PID=""
    if [ -n "$ai_script" ]; then
        shifter --image=$IMAGE python3 $ai_script $ai_gpu 180 "$@" &
        AI_PID=$!
        sleep 8
        echo "AI worker PID=$AI_PID on GPU$ai_gpu"
    fi

    shifter --image=$IMAGE python3 $L1 $label $NCELLS

    if [ -n "$AI_PID" ]; then
        kill $AI_PID 2>/dev/null
        wait $AI_PID 2>/dev/null
    fi

    stop_mps
    echo ""
}

# ====== 8-cell experiments ======

# Baselines
run_mode "8cell_baseline" "" ""

# Same GPU interference
run_mode "8cell_same_hbm"    "$HBM"    0 4
run_mode "8cell_same_gpt2"   "$GPT2"   0
run_mode "8cell_same_resnet" "$RESNET" 0

# Different GPU (isolation control)
run_mode "8cell_diff_hbm"    "$HBM"    1 4
run_mode "8cell_diff_gpt2"   "$GPT2"   1
run_mode "8cell_diff_resnet" "$RESNET" 1

# Final baseline
run_mode "8cell_baseline_final" "" ""

echo ""
echo "============================================================"
echo "ALL MODES COMPLETE"
echo "============================================================"
