#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J prio_real
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/prio_real_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/prio_real_%j.err

# ===================================================================
# CUDA Stream Priority with REAL cuPHY + REAL AI
# 4T4R × 4cell and 4T4R × 8cell
# Compare default vs high priority for each AI workload
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH
export HF_HOME=/pscratch/sd/s/sgkim/kcj/AI-RAN/datasets/models
export TRANSFORMERS_OFFLINE=1

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
SCRIPT=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/test_priority_real.py

echo "============================================================"
echo "CUDA Stream Priority: Real cuPHY + Real AI (40GB)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Test with 4 cells (fits in TTI better)
for cells in 4 8; do
    echo "====== ${cells} cells ======"

    # Baseline (no AI)
    echo "--- baseline ---"
    shifter --image=$IMAGE python3 $SCRIPT prio_${cells}c_baseline $cells none default
    echo ""

    # Each AI type: default vs high priority
    for ai in hbm gpt2 resnet; do
        echo "--- ${ai} default priority ---"
        shifter --image=$IMAGE python3 $SCRIPT prio_${cells}c_${ai}_default $cells $ai default
        echo ""

        echo "--- ${ai} HIGH priority ---"
        shifter --image=$IMAGE python3 $SCRIPT prio_${cells}c_${ai}_high $cells $ai high
        echo ""
    done
done

echo "ALL COMPLETE"
