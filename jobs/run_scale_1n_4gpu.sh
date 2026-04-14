#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J s1n4g
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/s1n4g_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/s1n4g_%j.err

# 1 Node × 4 GPU: each GPU runs L1 + AI
LOCAL=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_node_4gpu.sh
CELLS=8

echo "============================================================"
echo "Scale: 1 Node × 4 GPU (40GB)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "=== baseline ==="
bash $LOCAL s1n4g_baseline $CELLS none
echo ""

echo "=== neuralrx ==="
bash $LOCAL s1n4g_neuralrx $CELLS neuralrx
echo ""

echo "=== gpt2 ==="
bash $LOCAL s1n4g_gpt2 $CELLS gpt2
echo ""

echo "=== resnet ==="
bash $LOCAL s1n4g_resnet $CELLS resnet
echo ""

echo "ALL COMPLETE"
