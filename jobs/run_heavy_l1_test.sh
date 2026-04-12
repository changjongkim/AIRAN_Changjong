#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J heavy_l1
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/heavy_l1_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/heavy_l1_%j.err

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

echo "Testing heavy L1 configs..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

shifter --image=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb \
    python3 /pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/test_heavy_l1.py
