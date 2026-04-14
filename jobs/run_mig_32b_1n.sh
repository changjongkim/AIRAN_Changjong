#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J mig32_1n
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/mig32_1n_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/mig32_1n_%j.err

LOCAL=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_node_qwen_mig.sh
echo "============================================================"
echo "1N: MIG-like 40:60 + Qwen-32B TP (40GB)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "=== baseline (L1 40% SM) ==="
bash $LOCAL mig32_1n_base 8 32b none
echo ""
echo "=== L1(40%) + Qwen-32B(60%) ==="
bash $LOCAL mig32_1n_qwen 8 32b qwen
echo ""
echo "ALL COMPLETE"
