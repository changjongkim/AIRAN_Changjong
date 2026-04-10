#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -J phase0_unified
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/unified_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/unified_%j.err

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

shifter --image=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb \
    python3 /pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/exp_unified.py
