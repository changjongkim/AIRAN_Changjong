#!/bin/bash
# Per-node: 4 GPU, each GPU runs L1 (40% SM) + Qwen TP (60% SM)
# MIG-like isolation via CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
# Args: <label> <cells> <model> <ai_mode>
# model: 32b / 72b
# ai_mode: none / qwen

LABEL=$1
CELLS=$2
MODEL=$3
AI_MODE=${4:-"none"}

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH
export HF_HOME=/pscratch/sd/s/sgkim/kcj/hf_cache
export TRANSFORMERS_OFFLINE=1
IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_graph.py
HOSTNAME=$(hostname)

if [ "$MODEL" = "72b" ]; then
    MODEL_NAME="Qwen/Qwen2.5-72B"
else
    MODEL_NAME="Qwen/Qwen2.5-32B"
fi

echo "[$HOSTNAME] $LABEL: ${CELLS}cells/GPU, model=$MODEL_NAME, AI=$AI_MODE"

# Start MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${LABEL}_$$
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${LABEL}_$$
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d 2>/dev/null
sleep 1

# Start Qwen with 60% SM (AI partition)
AI_PID=""
if [ "$AI_MODE" = "qwen" ]; then
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=60 PYTHONPATH=$PYTHONPATH \
    shifter --image=$IMAGE python3 -c "
import os, sys, time
os.environ['HF_HOME'] = '/pscratch/sd/s/sgkim/kcj/hf_cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages')
import torch
from transformers import AutoModelForCausalLM
print('[${MODEL}B] Loading model (4GPU TP, SM=60%)...', flush=True)
model = AutoModelForCausalLM.from_pretrained(
    '${MODEL_NAME}', torch_dtype=torch.float16, device_map='auto')
model.eval()
dummy = torch.randint(0, 32000, (1, 1024), device='cuda:0')
with torch.no_grad():
    for _ in range(2): model(dummy)
torch.cuda.synchronize()
for i in range(4):
    free, total = torch.cuda.mem_get_info(i)
    print(f'  GPU{i}: {(total-free)/1e9:.1f}/{total/1e9:.1f}GB', flush=True)
print('[${MODEL}B] Running inference (SM=60%)...', flush=True)
c = 0; start = time.time()
with torch.no_grad():
    while time.time() - start < 180:
        model(dummy); c += 1
        if c % 10 == 0:
            print(f'[${MODEL}B] {c} iters, {c/(time.time()-start):.1f} it/s', flush=True)
print(f'[${MODEL}B] done: {c} iters', flush=True)
" &
    AI_PID=$!
    echo "[$HOSTNAME] Waiting 120s for ${MODEL}B to load..."
    sleep 120
    echo "[$HOSTNAME] ${MODEL}B PID=$AI_PID (SM=60%)"
fi

# Run L1 on each GPU with 40% SM (L1 partition)
L1_PIDS=()
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40 \
        shifter --image=$IMAGE python3 $L1 ${LABEL}_${HOSTNAME}_g${gpu} $CELLS 1 &
    L1_PIDS+=($!)
done
for pid in "${L1_PIDS[@]}"; do wait $pid; done

# Cleanup
if [ -n "$AI_PID" ]; then
    kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
fi
echo quit | nvidia-cuda-mps-control 2>/dev/null
echo "[$HOSTNAME] Done"
