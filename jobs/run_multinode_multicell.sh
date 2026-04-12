#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J mn_multicell
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/mn_multicell_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/mn_multicell_%j.err

# ===================================================================
# Multi-Node Multi-Cell Experiment (2 Nodes × 4 GPU)
# Node 0: L1 (8 cells) on GPU0
# Node 1: AI workload on GPU0
#
# Modes:
#   1. baseline         — L1 solo
#   2. same_gpu_gpt2    — L1 + GPT-2 on Node0:GPU0
#   3. same_gpu_hbm     — L1 + HBM on Node0:GPU0
#   4. diff_gpu_gpt2    — L1 on Node0:GPU0, GPT-2 on Node0:GPU1
#   5. diff_node_gpt2   — L1 on Node0:GPU0, GPT-2 on Node1:GPU0
#   6. diff_node_hbm    — L1 on Node0:GPU0, HBM on Node1:GPU0
#   7. baseline_final   — consistency check
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_multicell.py
HBM=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py

NCELLS=8
NODE0=$(scontrol show hostname $SLURM_JOB_NODELIST | head -1)
NODE1=$(scontrol show hostname $SLURM_JOB_NODELIST | tail -1)

echo "============================================================"
echo "Multi-Node Multi-Cell Experiment"
echo "Node0 (L1): $NODE0"
echo "Node1 (AI): $NODE1"
echo "L1 config: ${NCELLS} cells"
echo "============================================================"
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

# 1. Baseline — L1 solo on Node0
echo "=========================================="
echo "MODE: baseline (L1 ${NCELLS}cells solo)"
echo "=========================================="
start_mps baseline
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE python3 $L1 mn_baseline $NCELLS
stop_mps
echo ""

# 2. Same GPU — L1 + GPT-2 both on Node0:GPU0
echo "=========================================="
echo "MODE: same_gpu_gpt2"
echo "=========================================="
start_mps same_gpt2
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE python3 $GPT2 0 180 &
AI_PID=$!
sleep 10
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE python3 $L1 mn_same_gpu_gpt2 $NCELLS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# 3. Same GPU — L1 + HBM both on Node0:GPU0
echo "=========================================="
echo "MODE: same_gpu_hbm"
echo "=========================================="
start_mps same_hbm
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE python3 $HBM 0 180 4 &
AI_PID=$!
sleep 8
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE python3 $L1 mn_same_gpu_hbm $NCELLS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# 4. Diff GPU — L1 on Node0:GPU0, GPT-2 on Node0:GPU1
echo "=========================================="
echo "MODE: diff_gpu_gpt2"
echo "=========================================="
start_mps diff_gpu
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE python3 $GPT2 1 180 &
AI_PID=$!
sleep 10
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE python3 $L1 mn_diff_gpu_gpt2 $NCELLS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# 5. Diff Node — L1 on Node0:GPU0, GPT-2 on Node1:GPU0
echo "=========================================="
echo "MODE: diff_node_gpt2"
echo "=========================================="
start_mps diff_node_gpt2
srun -N1 -n1 --nodelist=$NODE1 shifter --image=$IMAGE python3 $GPT2 0 180 &
AI_PID=$!
sleep 10
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE python3 $L1 mn_diff_node_gpt2 $NCELLS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# 6. Diff Node — L1 on Node0:GPU0, HBM on Node1:GPU0
echo "=========================================="
echo "MODE: diff_node_hbm"
echo "=========================================="
start_mps diff_node_hbm
srun -N1 -n1 --nodelist=$NODE1 shifter --image=$IMAGE python3 $HBM 0 180 4 &
AI_PID=$!
sleep 8
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE python3 $L1 mn_diff_node_hbm $NCELLS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# 7. Final baseline
echo "=========================================="
echo "MODE: baseline_final"
echo "=========================================="
start_mps baseline_final
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE python3 $L1 mn_baseline_final $NCELLS
stop_mps
echo ""

echo "============================================================"
echo "ALL MODES COMPLETE"
echo "============================================================"
