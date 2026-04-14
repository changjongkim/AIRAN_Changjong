#!/bin/bash
# Per-node: 4 GPU, each runs L1, Qwen-32B runs across all 4 GPUs (tensor parallel)
# Args: <label> <cells> <ai_mode>
# ai_mode: none / qwen32b

LABEL=$1
CELLS=$2
AI_MODE=${3:-"none"}

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH
export HF_HOME=/pscratch/sd/s/sgkim/kcj/hf_cache
export TRANSFORMERS_OFFLINE=1
IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_graph.py
HOSTNAME=$(hostname)

echo "[$HOSTNAME] $LABEL: ${CELLS}cells/GPU, AI=$AI_MODE"

# Start MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${LABEL}_$$
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${LABEL}_$$
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d 2>/dev/null
sleep 1

# Start Qwen-32B 4GPU tensor parallel (background)
AI_PID=""
if [ "$AI_MODE" = "qwen32b" ]; then
    PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 -c "
import os, sys, time
os.environ['HF_HOME'] = '/pscratch/sd/s/sgkim/kcj/hf_cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages')
import torch
from transformers import AutoModelForCausalLM
print('[Qwen-32B] Loading model (4GPU TP)...', flush=True)
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-32B', torch_dtype=torch.float16, device_map='auto')
model.eval()
dummy = torch.randint(0, 32000, (1, 1024), device='cuda:0')
with torch.no_grad():
    for _ in range(2): model(dummy)
torch.cuda.synchronize()
for i in range(4):
    free, total = torch.cuda.mem_get_info(i)
    print(f'  GPU{i}: {(total-free)/1e9:.1f}/{total/1e9:.1f}GB', flush=True)
print('[Qwen-32B] Running inference loop...', flush=True)
c = 0; start = time.time()
with torch.no_grad():
    while time.time() - start < 180:
        model(dummy); c += 1
        if c % 10 == 0:
            print(f'[Qwen-32B] {c} iters, {c/(time.time()-start):.1f} it/s', flush=True)
print(f'[Qwen-32B] done: {c} iters', flush=True)
" &
    AI_PID=$!
    echo "[$HOSTNAME] Waiting 120s for Qwen-32B to load..."
    sleep 120
    echo "[$HOSTNAME] Qwen-32B PID=$AI_PID"
fi

# Run L1 on each GPU simultaneously
L1_PIDS=()
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu \
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
