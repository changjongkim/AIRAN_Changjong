#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J phase0_4gpu
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/phase0_4gpu_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/phase0_4gpu_%j.err

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

# Start MPS for GPU sharing
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$$
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$$
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
sleep 2
echo "MPS started"

# Show GPU topology
nvidia-smi topo -m

shifter --image=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb \
    python3 /pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/exp_phase0_4gpu.py

echo quit | nvidia-cuda-mps-control
echo "MPS stopped"
