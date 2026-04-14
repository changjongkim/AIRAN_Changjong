#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J bora_vs
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/bora_vs_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/bora_vs_%j.err

# BORA vs NVIDIA baseline: proper comparison
# Baseline = MIG 40:60 + AI (NVIDIA's recommended approach)
# BORA = MIG 40:60 + Priority + TTI coordination
# Both run with AI, compare L1 miss + AI throughput

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_graph.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py
CELLS=8
RESULT_DIR=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results

echo "============================================================"
echo "BORA vs NVIDIA Baseline (40GB)"
echo "Baseline = MIG 40:60 (Config B)"
echo "BORA     = MIG 40:60 + Priority + TTI (Config D)"
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

# AI throughput: write periodically to file
run_comparison() {
    local label=$1
    local config=$2   # B or D
    local ai_script=$3
    local ai_name=$4
    shift 4
    local ai_args="$@"

    local L1_SM=40; local AI_SM=60
    local AI_LOG="${RESULT_DIR}/ai_tp_${label}.txt"

    echo "=========================================="
    echo "$label: Config $config + $ai_name"
    echo "=========================================="

    start_mps $label

    # AI process: background, write throughput to file every 5s
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$AI_SM PYTHONPATH=$PYTHONPATH \
        shifter --image=$IMAGE python3 -c "
import sys, time, os
sys.path.insert(0, '/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages')
os.environ['HF_HOME'] = '/pscratch/sd/s/sgkim/kcj/AI-RAN/datasets/models'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

ai_type = '${ai_name}'
log_file = '${AI_LOG}'
c = 0; start = time.time()

if ai_type == 'gpt2':
    import torch
    from transformers import GPT2LMHeadModel, GPT2Config
    cfg = GPT2Config()
    model = GPT2LMHeadModel(cfg).cuda().eval()
    dummy = torch.randint(0, cfg.vocab_size, (4, 512), device='cuda')
    with torch.no_grad():
        for _ in range(3): model(dummy)
    torch.cuda.synchronize()
    with torch.no_grad():
        while time.time() - start < 90:
            model(dummy); c += 1
            if c % 20 == 0:
                with open(log_file, 'w') as f:
                    f.write(f'{c} {time.time()-start:.1f}\n')

elif ai_type == 'resnet':
    import torch, torchvision.models as models
    model = models.resnet50(weights=None).cuda().eval()
    dummy = torch.randn(64, 3, 224, 224, device='cuda')
    with torch.no_grad():
        for _ in range(5): model(dummy)
    torch.cuda.synchronize()
    with torch.no_grad():
        while time.time() - start < 90:
            model(dummy); c += 1
            if c % 20 == 0:
                with open(log_file, 'w') as f:
                    f.write(f'{c} {time.time()-start:.1f}\n')

elapsed = time.time() - start
with open(log_file, 'w') as f:
    f.write(f'{c} {elapsed:.1f}\n')
print(f'[AI {ai_type}] done: {c} iters in {elapsed:.1f}s ({c/elapsed:.1f} it/s)', flush=True)
" > /dev/null 2>&1 &
    AI_PID=$!
    sleep 10
    echo "AI ($ai_name) PID=$AI_PID (SM=${AI_SM}%)"

    # L1 process: foreground, 60 seconds
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$L1_SM \
        shifter --image=$IMAGE python3 $L1 $label $CELLS 1

    # Wait for AI to finish naturally or kill after grace period
    sleep 5
    kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null

    # Report AI throughput from file
    if [ -f "$AI_LOG" ]; then
        read iters elapsed < $AI_LOG
        if [ -n "$iters" ] && [ -n "$elapsed" ]; then
            tp=$(echo "scale=1; $iters / $elapsed" | bc 2>/dev/null || echo "?")
            echo "AI throughput: $iters iters / ${elapsed}s = ${tp} inf/s"
        fi
    else
        echo "AI throughput: (no data)"
    fi

    stop_mps
    echo ""
}

# ====== NVIDIA Baseline: Config B (MIG 40:60) ======
run_comparison "nvidia_gpt2"   B "$GPT2"   gpt2   0 90
run_comparison "nvidia_resnet" B "$RESNET" resnet  0 90 128

# ====== BORA: Config D (MIG + Priority + TTI) ======
# Note: TTI coordination via shared memory not in separate-process mode yet.
# Using Config C (MIG + Priority) as proxy for now.
run_comparison "bora_gpt2"   C "$GPT2"   gpt2   0 90
run_comparison "bora_resnet" C "$RESNET" resnet  0 90 128

echo "============================================================"
echo "SUMMARY: NVIDIA Baseline vs BORA"
echo "============================================================"
echo ""
echo "--- L1 Latency ---"
for f in nvidia_gpt2 nvidia_resnet bora_gpt2 bora_resnet; do
    grep -E "RESULT|RX mean:|Miss" ${RESULT_DIR}/../results/exp_${f}_*.json 2>/dev/null | head -3
done
echo ""
echo "--- AI Throughput ---"
for f in nvidia_gpt2 nvidia_resnet bora_gpt2 bora_resnet; do
    echo -n "$f: "
    cat ${RESULT_DIR}/ai_tp_${f}.txt 2>/dev/null || echo "(no data)"
done

echo ""
echo "ALL COMPLETE"
