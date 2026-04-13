#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J feasibility
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/feasibility_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/feasibility_%j.err

# ===================================================================
# Feasibility tests for TTI-aware AI scheduling
#   Test 1: L1 TTI timing (how much idle time exists?)
#   Test 2: CUDA stream priority (can we prioritize L1 over AI?)
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb

echo "============================================================"
echo "Feasibility: TTI-aware AI Scheduling"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== Test 1: L1 TTI Timing ==="
shifter --image=$IMAGE \
    python3 /pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/test_tti_timing.py

echo ""
echo "=== Test 2: CUDA Stream Priority ==="
shifter --image=$IMAGE \
    python3 /pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/test_stream_priority.py

echo ""
echo "ALL TESTS COMPLETE"
