#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J multinode_hbm
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/multinode_hbm_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/multinode_hbm_%j.err

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
SCRIPT=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/exp_multinode_hbm.py

echo "Nodes: $SLURM_JOB_NODELIST"

# Run 4 modes sequentially on the SAME 2-node allocation
for mode in baseline same_gpu diff_gpu diff_node; do
    echo ""
    echo "========== MODE: $mode =========="
    srun --mpi=pmi2 shifter --image=$IMAGE python3 $SCRIPT --mode $mode
    sleep 3
done

echo ""
echo "All modes complete."
