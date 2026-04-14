#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J q32b_1n
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/q32b_1n_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/q32b_1n_%j.err

# 1N × 4GPU: L1 + Qwen-32B tensor parallel (NVLink)
LOCAL=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_node_qwen32b.sh
CELLS=8

echo "============================================================"
echo "1N × 4GPU: L1 + Qwen-32B TP (NVLink, 40GB)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "=== baseline ==="
bash $LOCAL q32b1n_baseline $CELLS none
echo ""
echo "=== Qwen-32B 4GPU TP ==="
bash $LOCAL q32b1n_qwen32b $CELLS qwen32b
echo ""
echo "ALL COMPLETE"
