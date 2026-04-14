#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J scale2n_v2
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scale2n_v2_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scale2n_v2_%j.err

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
NODE0=$(scontrol show hostname $SLURM_JOB_NODELIST | head -1)
NODE1=$(scontrol show hostname $SLURM_JOB_NODELIST | tail -1)

echo "============================================================"
echo "Scale 2N: CUDA Graph ${CELLS}cell + AI (40GB, MPS)"
echo "Node0: $NODE0, Node1: $NODE1"
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
    echo "MODE: $label (both nodes)"
    echo "=========================================="
    start_mps $label

    # Node 0: AI background + L1
    local AI0=""
    local AI1=""
    if [ -n "$ai_cmd" ]; then
        srun -N1 -n1 --nodelist=$NODE0 env PYTHONPATH=$PYTHONPATH \
            shifter --image=$IMAGE python3 $ai_cmd "$@" &
        AI0=$!
        srun -N1 -n1 --nodelist=$NODE1 env PYTHONPATH=$PYTHONPATH \
            shifter --image=$IMAGE python3 $ai_cmd "$@" &
        AI1=$!
        sleep 10
        echo "AI on Node0 PID=$AI0, Node1 PID=$AI1"
    fi

    # L1 on both nodes simultaneously
    srun -N1 -n1 --nodelist=$NODE0 \
        shifter --image=$IMAGE python3 $L1 ${label}_n0 $CELLS 1 &
    L1_0=$!
    srun -N1 -n1 --nodelist=$NODE1 \
        shifter --image=$IMAGE python3 $L1 ${label}_n1 $CELLS 1 &
    L1_1=$!
    wait $L1_0 $L1_1

    if [ -n "$AI0" ]; then
        kill $AI0 $AI1 2>/dev/null
        wait $AI0 $AI1 2>/dev/null
    fi
    stop_mps
    echo ""
}

run_mode "s2n_baseline" ""
run_mode "s2n_neuralrx" "$AI" 0 120 neural_rx 1.0
run_mode "s2n_gpt2" "$GPT2" 0 120
run_mode "s2n_resnet" "$RESNET" 0 120 128

echo "ALL COMPLETE"
