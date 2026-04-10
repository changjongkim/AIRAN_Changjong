#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J real_4gpu
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/real_4gpu_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/real_4gpu_%j.err

# ===================================================================
# Realistic 4-GPU Experiment
# ALL 4 GPUs run L1 simultaneously (like a real multi-cell gNB)
# Then AI is added to ONE GPU → measure interference on that GPU
# while other GPUs serve as comparison
#
# Modes:
#   1. all_L1_only      — 4 GPUs all running L1, no AI (baseline)
#   2. gpu0_gets_resnet  — GPU0: L1+ResNet, GPU1-3: L1 only
#   3. gpu0_gets_gpt2    — GPU0: L1+GPT2,   GPU1-3: L1 only
#   4. gpu0_gets_hbm     — GPU0: L1+HBM,    GPU1-3: L1 only
#   5. all_L1_only_final — consistency check
#
# We measure GPU0's L1 latency in each mode.
# GPU1-3 run L1 as background load (simulating real multi-GPU gNB).
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_multicell.py
HBM=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py

NCELLS=4  # cells per GPU (4 GPUs × 4 cells = 16 cells total)

echo "============================================================"
echo "Realistic 4-GPU: All GPUs busy with L1"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
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

# L1 background on GPU1,2,3 (separate processes, run for duration)
# These simulate "other cells on other GPUs" in a real gNB
start_background_l1() {
    local duration=$1
    echo "Starting background L1 on GPU1,2,3 (${NCELLS} cells each, ${duration}s)..."
    for gpu in 1 2 3; do
        CUDA_VISIBLE_DEVICES=$gpu shifter --image=$IMAGE \
            python3 -c "
import os, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES']='0'  # remapped inside container
sys.path.insert(0, '/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages')
import torch
torch.cuda.set_device(0)
# Simple GPU busy loop (simulates L1 load without cuPHY conflict)
n = 50000000
a = torch.randn(n, device='cuda')
b = torch.randn(n, device='cuda')
start = time.time()
while time.time() - start < $duration:
    c = a * b + a  # element-wise ops (SM + bandwidth usage)
print(f'[BG L1 GPU$gpu] done', flush=True)
" &
    done
    sleep 3
}

kill_background() {
    jobs -p | xargs -r kill 2>/dev/null
    wait 2>/dev/null
    sleep 2
}

# ====== Mode 1: All GPUs L1 only (baseline) ======
echo "=========================================="
echo "MODE: all_L1_only (baseline)"
echo "=========================================="
start_mps baseline
start_background_l1 180
CUDA_VISIBLE_DEVICES=0 shifter --image=$IMAGE python3 $L1 real4gpu_baseline $NCELLS
kill_background
stop_mps
echo ""

# ====== Mode 2: GPU0 gets ResNet-50 ======
echo "=========================================="
echo "MODE: gpu0_gets_resnet (all GPUs busy + ResNet on GPU0)"
echo "=========================================="
start_mps resnet
start_background_l1 180
# AI on GPU0
CUDA_VISIBLE_DEVICES=0 shifter --image=$IMAGE python3 $RESNET 0 180 &
AI_PID=$!
sleep 10
echo "ResNet PID=$AI_PID on GPU0"
CUDA_VISIBLE_DEVICES=0 shifter --image=$IMAGE python3 $L1 real4gpu_resnet_gpu0 $NCELLS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
kill_background
stop_mps
echo ""

# ====== Mode 3: GPU0 gets GPT-2 ======
echo "=========================================="
echo "MODE: gpu0_gets_gpt2 (all GPUs busy + GPT-2 on GPU0)"
echo "=========================================="
start_mps gpt2
start_background_l1 180
CUDA_VISIBLE_DEVICES=0 shifter --image=$IMAGE python3 $GPT2 0 180 &
AI_PID=$!
sleep 10
echo "GPT-2 PID=$AI_PID on GPU0"
CUDA_VISIBLE_DEVICES=0 shifter --image=$IMAGE python3 $L1 real4gpu_gpt2_gpu0 $NCELLS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
kill_background
stop_mps
echo ""

# ====== Mode 4: GPU0 gets HBM stress ======
echo "=========================================="
echo "MODE: gpu0_gets_hbm (all GPUs busy + HBM on GPU0)"
echo "=========================================="
start_mps hbm
start_background_l1 180
CUDA_VISIBLE_DEVICES=0 shifter --image=$IMAGE python3 $HBM 0 180 2 &
AI_PID=$!
sleep 8
echo "HBM PID=$AI_PID on GPU0"
CUDA_VISIBLE_DEVICES=0 shifter --image=$IMAGE python3 $L1 real4gpu_hbm_gpu0 $NCELLS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
kill_background
stop_mps
echo ""

# ====== Mode 5: Baseline final ======
echo "=========================================="
echo "MODE: all_L1_only_final (consistency check)"
echo "=========================================="
start_mps final
start_background_l1 180
CUDA_VISIBLE_DEVICES=0 shifter --image=$IMAGE python3 $L1 real4gpu_baseline_final $NCELLS
kill_background
stop_mps
echo ""

echo "============================================================"
echo "ALL MODES COMPLETE"
echo "============================================================"
