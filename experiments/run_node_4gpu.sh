#!/bin/bash
# Per-node script: 4 GPU, each GPU runs L1, AI is distributed across all 4 GPUs.
# Called via srun on each node.
# Args: <label> <cells_per_gpu> <ai_type> [ai_args...]
# ai_type: none / neuralrx / gpt2 / resnet

LABEL=$1
CELLS=$2
AI_TYPE=$3
shift 3
AI_EXTRA="$@"

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH
IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_graph.py
AI=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_realistic_ai_stress.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py
HOSTNAME=$(hostname)

echo "[$HOSTNAME] Starting: $LABEL, ${CELLS}cells/GPU, AI=$AI_TYPE"

# Start MPS locally
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${LABEL}_$$
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${LABEL}_$$
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d 2>/dev/null
sleep 1

# Start AI on each of 4 GPUs (background)
AI_PIDS=()
if [ "$AI_TYPE" != "none" ]; then
    for gpu in 0 1 2 3; do
        if [ "$AI_TYPE" = "neuralrx" ]; then
            CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PYTHONPATH \
                shifter --image=$IMAGE python3 $AI $gpu 120 neural_rx 1.0 &
            AI_PIDS+=($!)
        elif [ "$AI_TYPE" = "gpt2" ]; then
            CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PYTHONPATH \
                shifter --image=$IMAGE python3 $GPT2 $gpu 120 &
            AI_PIDS+=($!)
        elif [ "$AI_TYPE" = "resnet" ]; then
            CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PYTHONPATH \
                shifter --image=$IMAGE python3 $RESNET $gpu 120 128 &
            AI_PIDS+=($!)
        fi
    done
    sleep 10
    echo "[$HOSTNAME] AI ($AI_TYPE) started on GPU0~3, PIDs: ${AI_PIDS[@]}"
fi

# Run L1 on each of 4 GPUs (parallel, wait for all)
L1_PIDS=()
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu \
        shifter --image=$IMAGE python3 $L1 ${LABEL}_${HOSTNAME}_g${gpu} $CELLS 1 &
    L1_PIDS+=($!)
done

for pid in "${L1_PIDS[@]}"; do
    wait $pid
done

# Cleanup
for pid in "${AI_PIDS[@]}"; do
    kill $pid 2>/dev/null
    wait $pid 2>/dev/null
done
echo quit | nvidia-cuda-mps-control 2>/dev/null
echo "[$HOSTNAME] Done"
