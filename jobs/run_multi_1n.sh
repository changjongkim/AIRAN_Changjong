#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J multi_1n
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/multi_1n_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/multi_1n_%j.err

LOCAL=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_node_multi_ai.sh
CELLS=8

echo "============================================================"
echo "1N × 4GPU: L1 + Multi-AI services (40GB)"
echo "  GPU0: L1 + Neural Rx | GPU1: L1 + GPT-2"
echo "  GPU2: L1 + ResNet    | GPU3: L1 + Neural Rx"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "=== baseline ==="
bash $LOCAL m1n_baseline $CELLS none
echo ""
echo "=== full AI ==="
bash $LOCAL m1n_full $CELLS full
echo ""
echo "ALL COMPLETE"
