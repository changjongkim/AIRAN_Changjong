#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J scale2n_v4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scale2n_v4_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scale2n_v4_%j.err

# srun sends run_node_local.sh to each node
# Each node locally: starts MPS → AI background → L1 → cleanup

LOCAL=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_node_local.sh
AI=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_realistic_ai_stress.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py
CELLS=8

echo "============================================================"
echo "Scale 2N v4: per-node MPS (40GB)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "============================================================"

# Baseline
echo "=== baseline ==="
srun --mpi=pmi2 bash $LOCAL s2v4_baseline $CELLS ""
echo ""

# Neural Rx high
echo "=== neuralrx ==="
srun --mpi=pmi2 bash $LOCAL s2v4_neuralrx $CELLS "$AI" 0 120 neural_rx 1.0
echo ""

# GPT-2
echo "=== gpt2 ==="
srun --mpi=pmi2 bash $LOCAL s2v4_gpt2 $CELLS "$GPT2" 0 120
echo ""

# ResNet
echo "=== resnet ==="
srun --mpi=pmi2 bash $LOCAL s2v4_resnet $CELLS "$RESNET" 0 120 128
echo ""

echo "ALL COMPLETE"
