#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J scale4n_v2
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scale4n_v2_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scale4n_v2_%j.err

# Each node: Shell background & + MPS — identical to 1N setup

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_graph.py
AI=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_realistic_ai_stress.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py

CELLS=8
NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))

echo "============================================================"
echo "Scale 4N: CUDA Graph ${CELLS}cell + AI (40GB, MPS)"
echo "Nodes: ${NODES[@]}"
echo "============================================================"

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

run_mode() {
    local label=$1
    local ai_cmd=$2
    shift 2

    echo "=========================================="
    echo "MODE: $label (4 nodes)"
    echo "=========================================="
    start_mps $label

    # Start AI on all 4 nodes
    local AI_PIDS=()
    if [ -n "$ai_cmd" ]; then
        for i in 0 1 2 3; do
            srun -N1 -n1 --nodelist=${NODES[$i]} env PYTHONPATH=$PYTHONPATH \
                shifter --image=$IMAGE python3 $ai_cmd "$@" &
            AI_PIDS+=($!)
        done
        sleep 10
        echo "AI on all 4 nodes"
    fi

    # L1 on all 4 nodes simultaneously
    local L1_PIDS=()
    for i in 0 1 2 3; do
        srun -N1 -n1 --nodelist=${NODES[$i]} \
            shifter --image=$IMAGE python3 $L1 ${label}_n${i} $CELLS 1 &
        L1_PIDS+=($!)
    done
    for pid in "${L1_PIDS[@]}"; do wait $pid; done

    for pid in "${AI_PIDS[@]}"; do
        kill $pid 2>/dev/null; wait $pid 2>/dev/null
    done
    stop_mps
    echo ""
}

run_mode "s4n_baseline" ""
run_mode "s4n_neuralrx" "$AI" 0 120 neural_rx 1.0
run_mode "s4n_gpt2" "$GPT2" 0 120
run_mode "s4n_resnet" "$RESNET" 0 120 128

echo "ALL COMPLETE"
