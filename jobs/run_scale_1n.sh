#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J scale_1n
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scale_1n_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scale_1n_%j.err

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
SCRIPT=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_node_l1_ai.py
DUR=30

echo "============================================================"
echo "Scale test: 1 Node (CUDA Graph baseline)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# L1 only
shifter --image=$IMAGE python3 $SCRIPT scale1n_baseline 8 none 0 $DUR
# L1 + Neural Rx high
shifter --image=$IMAGE python3 $SCRIPT scale1n_neuralrx 8 neural_rx 1.0 $DUR
# L1 + GPT-2
shifter --image=$IMAGE python3 $SCRIPT scale1n_gpt2 8 gpt2 1.0 $DUR
# L1 + ResNet
shifter --image=$IMAGE python3 $SCRIPT scale1n_resnet 8 resnet 1.0 $DUR

echo "ALL COMPLETE"
