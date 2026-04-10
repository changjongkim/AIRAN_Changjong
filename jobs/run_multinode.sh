#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J phase0_2node
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/multinode_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/multinode_%j.err

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
SCRIPT=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/exp_multinode.py

echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"

# Run: Rank 0 (Node 0) = L1, Rank 1 (Node 1) = AI workload
# Test with ResNet-50 on remote node
srun --mpi=pmi2 shifter --image=$IMAGE \
    python3 $SCRIPT --ai-workload resnet --cells 1 --tx 1 --rx 2 --mcs 2
