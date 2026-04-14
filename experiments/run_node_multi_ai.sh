#!/bin/bash
# Per-node: 4 GPU, each GPU runs L1 + different AI service
# GPU0: L1 + Neural Rx (high)
# GPU1: L1 + GPT-2 (LLM serving)
# GPU2: L1 + ResNet (video analytics)
# GPU3: L1 + Neural Rx (high)
# Args: <label> <cells>

LABEL=$1
CELLS=$2
AI_MODE=${3:-"full"}  # full = all AI, none = baseline

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH
IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_graph.py
NRX=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_realistic_ai_stress.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py
HOSTNAME=$(hostname)

echo "[$HOSTNAME] $LABEL: ${CELLS}cells/GPU, AI=$AI_MODE"

# Start MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${LABEL}_$$
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${LABEL}_$$
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d 2>/dev/null
sleep 1

# Start AI on each GPU — different service per GPU
AI_PIDS=()
if [ "$AI_MODE" = "full" ]; then
    # GPU0: Neural Rx high
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH \
        shifter --image=$IMAGE python3 $NRX 0 180 neural_rx 1.0 &
    AI_PIDS+=($!)

    # GPU1: GPT-2
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PYTHONPATH \
        shifter --image=$IMAGE python3 $GPT2 0 180 &
    AI_PIDS+=($!)

    # GPU2: ResNet-50
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PYTHONPATH \
        shifter --image=$IMAGE python3 $RESNET 0 180 128 &
    AI_PIDS+=($!)

    # GPU3: Neural Rx high
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PYTHONPATH \
        shifter --image=$IMAGE python3 $NRX 0 180 neural_rx 1.0 &
    AI_PIDS+=($!)

    sleep 10
    echo "[$HOSTNAME] AI started: GPU0=NeuralRx, GPU1=GPT2, GPU2=ResNet, GPU3=NeuralRx"
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
for pid in "${AI_PIDS[@]}"; do
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
done
echo quit | nvidia-cuda-mps-control 2>/dev/null
echo "[$HOSTNAME] Done"
