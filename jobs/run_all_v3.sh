#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J phase0_v3
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/phase0_v3_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/phase0_v3_%j.err

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1_SCRIPT=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_only.py
HBM_SCRIPT=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py

echo "GPU info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Helper: start MPS fresh
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

# Helper: run L1 measurement with optional background stress
# Usage: run_mode <label> [stress_gpu stress_gb]
run_mode() {
    local label=$1
    local stress_gpu=$2
    local stress_gb=$3

    echo "=========================================="
    echo "MODE: $label"
    echo "=========================================="

    start_mps $label

    # Launch HBM stress in background if requested
    local STRESS_PID=""
    if [ -n "$stress_gpu" ]; then
        shifter --image=$IMAGE python3 $HBM_SCRIPT $stress_gpu 120 $stress_gb &
        STRESS_PID=$!
        sleep 5  # let stress warm up
        echo "HBM stress PID=$STRESS_PID on GPU$stress_gpu (${stress_gb}GB)"
    fi

    # Run L1 measurement
    shifter --image=$IMAGE python3 $L1_SCRIPT $label

    # Kill stress
    if [ -n "$STRESS_PID" ]; then
        kill $STRESS_PID 2>/dev/null
        wait $STRESS_PID 2>/dev/null
    fi

    stop_mps
    echo ""
}

# ====== Single GPU experiments ======

# 1. Baseline — L1 alone
run_mode "baseline"

# 2. No MIG — L1 + HBM stress both on GPU0 (worst case)
run_mode "noMIG_hbm_gpu0" 0 8

# 3. Different GPU — L1 on GPU0 + HBM stress on GPU1
run_mode "diffGPU_hbm_gpu1" 1 8

# 4. MIG-like: L1 on GPU0, stress on GPU0 with smaller allocation
run_mode "sameGPU_hbm_4GB" 0 4
run_mode "sameGPU_hbm_2GB" 0 2
run_mode "sameGPU_hbm_1GB" 0 1

# 5. Final baseline (consistency check)
run_mode "baseline_final"

echo ""
echo "=========================================="
echo "ALL MODES COMPLETE"
echo "=========================================="
