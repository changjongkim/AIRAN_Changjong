#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J multi_2n
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/multi_2n_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/multi_2n_%j.err

LOCAL=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_node_multi_ai.sh
CELLS=8

echo "============================================================"
echo "2N × 4GPU: L1 + Multi-AI services (40GB)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "============================================================"

echo "=== baseline ==="
srun --mpi=pmi2 bash $LOCAL m2n_baseline $CELLS none
echo ""
echo "=== full AI ==="
srun --mpi=pmi2 bash $LOCAL m2n_full $CELLS full
echo ""
echo "ALL COMPLETE"
