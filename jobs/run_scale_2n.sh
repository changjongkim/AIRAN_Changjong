#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J scale_2n
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scale_2n_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scale_2n_%j.err

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
SCRIPT=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_node_l1_ai.py
DUR=30

echo "============================================================"
echo "Scale test: 2 Nodes (each: L1 CUDA Graph + AI)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "============================================================"

# Each mode: both nodes run simultaneously
for mode in "baseline none 0" "neuralrx neural_rx 1.0" "gpt2 gpt2 1.0" "resnet resnet 1.0"; do
    read name ai_type intensity <<< "$mode"
    echo "=== $name ==="
    srun --mpi=pmi2 shifter --image=$IMAGE \
        python3 $SCRIPT scale2n_${name} 8 $ai_type $intensity $DUR
    echo ""
done

echo "ALL COMPLETE"
