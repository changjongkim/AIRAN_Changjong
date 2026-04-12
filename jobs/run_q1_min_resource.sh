#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J q1_minres
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/q1_minres_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/q1_minres_%j.err

# ===================================================================
# Q1: What is the MINIMUM GPU resource L1 needs to meet SLA?
#
# Sweep SM% from 100% down to 5%:
#   - At each level, measure L1 latency
#   - Find the "knee point" where SLA breaks
#   - Then: remaining SM% can be given to AI
#
# Also test: L1 with reduced SM% + AI workload simultaneously
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_sm_limited.py
L1_MULTI=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_multicell.py
QWEN=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_qwen72b_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py

echo "============================================================"
echo "Q1: Minimum L1 Resource (SM% Sweep)"
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

# ====== Part 1: L1-only SM% sweep (1 cell) ======
echo ""
echo "====== Part 1: L1 SM% Sweep (1 cell) ======"
echo ""

for sm in 100 80 60 50 40 30 20 14 10 5; do
    label="sm${sm}_1cell"
    echo "=========================================="
    echo "MODE: $label (SM=${sm}%)"
    echo "=========================================="
    start_mps $label
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$sm shifter --image=$IMAGE python3 $L1 $label
    stop_mps
    echo ""
done

# ====== Part 2: L1 SM% sweep with AI co-running (Qwen-72B) ======
echo ""
echo "====== Part 2: L1 SM% + Qwen-72B (same GPU) ======"
echo ""

# Pre-launch Qwen-72B (takes ~4min to load)
start_mps corun
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $QWEN 600 1 2048 &
QWEN_PID=$!
echo "Waiting 150s for Qwen-72B to load..."
sleep 150

for sm in 100 60 40 20 10; do
    label="sm${sm}_1cell_with72b"
    echo "=========================================="
    echo "MODE: $label (L1 SM=${sm}% + Qwen-72B)"
    echo "=========================================="
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$sm shifter --image=$IMAGE python3 $L1 $label
    echo ""
done

kill $QWEN_PID 2>/dev/null; wait $QWEN_PID 2>/dev/null
stop_mps

echo ""
echo "============================================================"
echo "ALL MODES COMPLETE"
echo "============================================================"
