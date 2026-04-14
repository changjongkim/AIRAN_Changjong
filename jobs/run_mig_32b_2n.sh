#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J mig32_2n
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/mig32_2n_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/mig32_2n_%j.err

LOCAL=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_node_qwen_mig.sh
echo "============================================================"
echo "2N: MIG-like 40:60 + Qwen-32B TP (40GB)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "============================================================"
echo "=== baseline ==="
srun --mpi=pmi2 bash $LOCAL mig32_2n_base 8 32b none
echo ""
echo "=== L1(40%) + Qwen-32B(60%) ==="
srun --mpi=pmi2 bash $LOCAL mig32_2n_qwen 8 32b qwen
echo ""
echo "ALL COMPLETE"
