#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J multinode_v2
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/multinode_v2_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/multinode_v2_%j.err

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
SCRIPT=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/exp_multinode_hbm.py

echo "Nodes: $SLURM_JOB_NODELIST"
echo ""

# Each mode gets a fresh MPS + srun
run_mode() {
    local mode=$1
    echo "========== $mode =========="
    # Start MPS on node 0 for same_gpu mode
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${mode}_$$
    export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${mode}_$$
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
    nvidia-cuda-mps-control -d 2>/dev/null
    sleep 1
    srun --mpi=pmi2 shifter --image=$IMAGE python3 $SCRIPT --mode $mode
    echo quit | nvidia-cuda-mps-control 2>/dev/null
    sleep 3
    echo ""
}

run_mode baseline
run_mode same_gpu
run_mode diff_gpu
run_mode diff_node

echo "All modes complete."
