#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J bora_1n
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/bora_1n_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/bora_1n_%j.err

# BORA Ablation: Config A → B → C → D on 1 Node with Qwen-72B
LOCAL=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_bora_node.sh
CELLS=8

echo "============================================================"
echo "BORA Ablation: 1N × 4GPU (80GB, Qwen-72B)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

for config in A B C D; do
    echo ""
    echo "=== Config $config ==="
    if [ "$config" = "A" ]; then
        bash $LOCAL bora_${config} $CELLS $config none
    else
        bash $LOCAL bora_${config} $CELLS $config 72b
    fi
done

echo ""
echo "ALL CONFIGS COMPLETE"
