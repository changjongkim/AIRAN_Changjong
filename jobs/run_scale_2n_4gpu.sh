#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J s2n4g
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/s2n4g_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/s2n4g_%j.err

# 2 Nodes × 4 GPU: each GPU runs L1 + AI
LOCAL=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_node_4gpu.sh
CELLS=8

echo "============================================================"
echo "Scale: 2 Nodes × 4 GPU (40GB)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "============================================================"

for mode in "baseline none" "neuralrx neuralrx" "gpt2 gpt2" "resnet resnet"; do
    read name ai_type <<< "$mode"
    echo "=== $name ==="
    srun --mpi=pmi2 bash $LOCAL s2n4g_${name} $CELLS $ai_type
    echo ""
done

echo "ALL COMPLETE"
