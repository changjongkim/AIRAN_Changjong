#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J mn_72b
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/mn_72b_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/mn_72b_%j.err

# ===================================================================
# Multi-Node Qwen-72B Experiment (2 Nodes × 4 GPU-80GB)
#
# Node 0: L1 (cuPHY) on GPU0
# Node 1: Qwen-72B inference (4GPU tensor parallel)
#
# Modes:
#   1. baseline         — L1 solo on Node0
#   2. same_node_72b    — L1 on Node0:GPU0 + Qwen-72B on Node0 (4GPU)
#   3. diff_node_72b    — L1 on Node0:GPU0 + Qwen-72B on Node1 (4GPU)
#   4. baseline_final
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_only.py
QWEN=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_qwen72b_stress.py

NODE0=$(scontrol show hostname $SLURM_JOB_NODELIST | head -1)
NODE1=$(scontrol show hostname $SLURM_JOB_NODELIST | tail -1)

echo "============================================================"
echo "Multi-Node Qwen-72B Experiment"
echo "Node0 (L1): $NODE0"
echo "Node1 (AI): $NODE1"
echo "============================================================"
srun -N1 -n1 --nodelist=$NODE0 nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null
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
echo "MODE: baseline (L1 solo on Node0)"
echo "=========================================="
start_mps baseline
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE python3 $L1 mn72b_baseline
stop_mps
echo ""

# 2. Same Node — L1 on Node0:GPU0 + Qwen-72B on Node0 (all 4 GPUs)
echo "=========================================="
echo "MODE: same_node_72b (L1+Qwen-72B both on Node0)"
echo "=========================================="
start_mps same
srun -N1 -n1 --nodelist=$NODE0 env PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $QWEN 300 1 2048 &
AI_PID=$!
echo "Waiting 120s for Qwen-72B to load on Node0..."
sleep 120
echo "Qwen-72B PID=$AI_PID, measuring L1 on Node0:GPU0..."
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE python3 $L1 mn72b_same_node
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# 3. Diff Node — L1 on Node0:GPU0 + Qwen-72B on Node1
echo "=========================================="
echo "MODE: diff_node_72b (L1:Node0, Qwen-72B:Node1)"
echo "=========================================="
start_mps diff
srun -N1 -n1 --nodelist=$NODE1 env PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $QWEN 300 1 2048 &
AI_PID=$!
echo "Waiting 120s for Qwen-72B to load on Node1..."
sleep 120
echo "Qwen-72B PID=$AI_PID on Node1, measuring L1 on Node0:GPU0..."
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE python3 $L1 mn72b_diff_node
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# 4. Baseline final
echo "=========================================="
echo "MODE: baseline_final"
echo "=========================================="
start_mps final
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE python3 $L1 mn72b_baseline_final
stop_mps

echo ""
echo "============================================================"
echo "ALL MODES COMPLETE"
echo "============================================================"
