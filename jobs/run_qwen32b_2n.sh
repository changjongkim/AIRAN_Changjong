#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J q32b_2n
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/q32b_2n_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/q32b_2n_%j.err

# 2N × 4GPU: each node runs L1 + Qwen-32B TP independently
LOCAL=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_node_qwen32b.sh
CELLS=8

echo "============================================================"
echo "2N × 4GPU: L1 + Qwen-32B TP (NVLink, 40GB)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "============================================================"

echo "=== baseline ==="
srun --mpi=pmi2 bash $LOCAL q32b2n_baseline $CELLS none
echo ""
echo "=== Qwen-32B 4GPU TP ==="
srun --mpi=pmi2 bash $LOCAL q32b2n_qwen32b $CELLS qwen32b
echo ""
echo "ALL COMPLETE"
