#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -J exp2_ai_intf
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/exp2_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/exp2_%j.err

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

# Start MPS daemon for GPU sharing between L1 and AI workloads
nvidia-cuda-mps-control -d
echo "MPS daemon started"

shifter --image=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb \
    python3 /pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/exp2_ai_interference.py

# Stop MPS daemon
echo quit | nvidia-cuda-mps-control
echo "MPS daemon stopped"
