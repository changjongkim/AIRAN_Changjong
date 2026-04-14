#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J bora_div2
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/bora_div2_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/bora_div2_%j.err

# Exact same method as bora_vs (proven: GPT-2 2.6x, ResNet 14% improvement)
# Extended to more workloads

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_graph.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py
AI_REAL=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_realistic_ai_stress.py
HBM=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py
CELLS=8
RESULT_DIR=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results

echo "============================================================"
echo "BORA vs Baseline: Diverse (40GB) — same method as bora_vs"
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

# Exact same pattern as run_bora_vs_baseline.sh
run_compare() {
    local label=$1
    local config=$2  # B=nvidia, C=bora
    local ai_script=$3
    shift 3
    local ai_args="$@"

    local L1_SM=40; local AI_SM=60
    local AI_LOG="${RESULT_DIR}/ai_tp_${label}.txt"

    echo "=========================================="
    echo "$label (Config $config)"
    echo "=========================================="

    start_mps $label

    # AI: exactly like bora_vs — shifter python3 directly, write throughput to file
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$AI_SM PYTHONPATH=$PYTHONPATH \
        shifter --image=$IMAGE python3 -c "
import sys, time, os
sys.path.insert(0, '/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages')
os.environ['HF_HOME'] = '/pscratch/sd/s/sgkim/kcj/AI-RAN/datasets/models'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

ai_script = '$ai_script'
ai_args = '$ai_args'.split()
log_file = '$AI_LOG'

# Import and run the actual AI workload inline
import torch
c = 0; start = time.time()

if 'gpt2' in ai_script:
    from transformers import GPT2LMHeadModel, GPT2Config
    cfg = GPT2Config()
    model = GPT2LMHeadModel(cfg).cuda().eval()
    dummy = torch.randint(0, cfg.vocab_size, (4, 512), device='cuda')
    with torch.no_grad():
        for _ in range(3): model(dummy)
    torch.cuda.synchronize()
    with torch.no_grad():
        while time.time() - start < 75:
            model(dummy); c += 1
            if c % 20 == 0:
                with open(log_file, 'w') as f: f.write(f'{c} {time.time()-start:.1f}\n')

elif 'resnet' in ai_script:
    import torchvision.models as models
    model = models.resnet50(weights=None).cuda().eval()
    dummy = torch.randn(64, 3, 224, 224, device='cuda')
    with torch.no_grad():
        for _ in range(5): model(dummy)
    torch.cuda.synchronize()
    with torch.no_grad():
        while time.time() - start < 75:
            model(dummy); c += 1
            if c % 20 == 0:
                with open(log_file, 'w') as f: f.write(f'{c} {time.time()-start:.1f}\n')

elif 'realistic' in ai_script and 'neural_rx' in '$ai_args':
    import torch.nn as nn
    model = nn.Sequential(
        nn.Conv1d(128, 256, 7, padding=3), nn.ReLU(),
        nn.Conv1d(256, 256, 7, padding=3), nn.ReLU(),
        nn.Conv1d(256, 128, 3, padding=1),
    ).cuda().eval()
    inp = torch.randn(16, 128, 4096, device='cuda')
    with torch.no_grad():
        for _ in range(5): model(inp)
    torch.cuda.synchronize()
    with torch.no_grad():
        while time.time() - start < 75:
            out = model(inp)
            inp[:,:out.shape[1],:out.shape[2]].copy_(out)
            c += 1
            if c % 100 == 0:
                with open(log_file, 'w') as f: f.write(f'{c} {time.time()-start:.1f}\n')

elif 'realistic' in ai_script and 'matmul' in '$ai_args':
    size = 8192
    A = torch.randn(size, size, device='cuda', dtype=torch.float16)
    B = torch.randn(size, size, device='cuda', dtype=torch.float16)
    C = torch.empty(size, size, device='cuda', dtype=torch.float16)
    torch.cuda.synchronize()
    while time.time() - start < 75:
        torch.mm(A, B, out=C); c += 1
        if c % 20 == 0:
            with open(log_file, 'w') as f: f.write(f'{c} {time.time()-start:.1f}\n')

elif 'hbm' in ai_script:
    n = int(2e9 / 4)
    src = torch.randn(n, dtype=torch.float32, device='cuda')
    dst = torch.empty_like(src)
    torch.cuda.synchronize()
    while time.time() - start < 75:
        dst.copy_(src); src.copy_(dst); c += 1
        if c % 100 == 0:
            with open(log_file, 'w') as f: f.write(f'{c} {time.time()-start:.1f}\n')

elapsed = time.time() - start
with open(log_file, 'w') as f: f.write(f'{c} {elapsed:.1f}\n')
print(f'[AI] done: {c} iters in {elapsed:.1f}s ({c/elapsed:.1f} it/s)', flush=True)
" > /dev/null 2>&1 &
    AI_PID=$!
    sleep 10
    echo "AI PID=$AI_PID (SM=${AI_SM}%)"

    # L1: foreground
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$L1_SM \
        shifter --image=$IMAGE python3 $L1 $label $CELLS 1

    sleep 3
    kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null

    # AI throughput
    if [ -f "$AI_LOG" ]; then
        read iters elapsed < $AI_LOG
        tp=$(echo "scale=1; $iters / $elapsed" | bc 2>/dev/null || echo "?")
        echo "AI throughput: $iters iters / ${elapsed}s = ${tp} it/s"
    else
        echo "AI throughput: (no data)"
    fi

    stop_mps
    echo ""
}

# NVIDIA baseline (Config B) vs BORA (Config C) for each workload

run_compare "nv_neuralrx"   B "$AI_REAL"  0 75 neural_rx 1.0
run_compare "bora_neuralrx" C "$AI_REAL"  0 75 neural_rx 1.0

run_compare "nv_gpt2"   B "$GPT2"   0 75
run_compare "bora_gpt2" C "$GPT2"   0 75

run_compare "nv_resnet"   B "$RESNET" 0 75 128
run_compare "bora_resnet" C "$RESNET" 0 75 128

run_compare "nv_hbm"   B "$HBM"     0 75 2
run_compare "bora_hbm" C "$HBM"     0 75 2

run_compare "nv_matmul"   B "$AI_REAL" 0 75 matmul 1.0
run_compare "bora_matmul" C "$AI_REAL" 0 75 matmul 1.0

echo "============================================================"
echo "ALL COMPLETE"
echo "============================================================"
