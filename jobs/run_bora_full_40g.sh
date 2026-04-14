#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J bora_full
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/bora_full_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/bora_full_%j.err

# BORA Full Eval: L1 latency + AI throughput on 40GB
# Config A~D × GPT-2/ResNet

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:/pscratch/sd/s/sgkim/kcj/AI-RAN:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
SCRIPT=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_bora_full.py
DUR=20

echo "============================================================"
echo "BORA Full Eval: 40GB (L1 latency + AI throughput)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

start_mps() {
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${1}_$$
    export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${1}_$$
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
    nvidia-cuda-mps-control -d 2>/dev/null
    sleep 1
}
stop_mps() {
    echo quit | nvidia-cuda-mps-control 2>/dev/null
    sleep 1
}

for config in A B C D; do
    for ai in gpt2 resnet; do
        label="full_${config}_${ai}"
        echo "=== Config $config + $ai ==="
        start_mps $label
        shifter --image=$IMAGE python3 $SCRIPT $label 8 $config $ai $DUR
        stop_mps
        echo ""
    done
done

# Baseline (no AI) for reference
echo "=== Config A baseline ==="
start_mps baseline
shifter --image=$IMAGE python3 $SCRIPT full_A_none 8 A none $DUR
stop_mps

echo ""
echo "ALL COMPLETE"
