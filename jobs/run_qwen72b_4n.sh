#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J q72b_4n
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/q72b_4n_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/q72b_4n_%j.err

# 4N × 4GPU: each node runs L1 + Qwen-72B TP independently
LOCAL=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_node_qwen72b.sh
CELLS=8

echo "============================================================"
echo "4N × 4GPU: L1 + Qwen-72B TP (NVLink, 40GB)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "============================================================"

echo "=== baseline ==="
srun --mpi=pmi2 bash $LOCAL q72b4n_baseline $CELLS none
echo ""
echo "=== Qwen-72B 4GPU TP ==="
srun --mpi=pmi2 bash $LOCAL q72b4n_qwen72b $CELLS qwen72b
echo ""
echo "ALL COMPLETE"
