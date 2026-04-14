#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J graph_intf
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/graph_intf_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/graph_intf_%j.err

# ===================================================================
# CUDA Graph multi-cell L1 + AI interference
# 8cell GRAPH (0.475ms) vs 8cell STREAM (1.305ms)
# + realistic AI workloads
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_graph.py
AI=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_realistic_ai_stress.py
HBM=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py

CELLS=8

echo "============================================================"
echo "CUDA Graph L1 (${CELLS}cell) + AI Interference (40GB)"
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

run_mode() {
    local label=$1
    local graph=$2  # 0 or 1
    local ai_cmd=$3
    shift 3

    local mode_str=$([ "$graph" = "1" ] && echo "GRAPH" || echo "STREAM")
    echo "=========================================="
    echo "MODE: $label ($mode_str)"
    echo "=========================================="

    start_mps $label

    local AI_PID=""
    if [ -n "$ai_cmd" ]; then
        PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $ai_cmd "$@" &
        AI_PID=$!
        sleep 8
        echo "AI PID=$AI_PID"
    fi

    shifter --image=$IMAGE python3 $L1 $label $CELLS $graph

    if [ -n "$AI_PID" ]; then
        kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
    fi

    stop_mps
    echo ""
}

# ====== STREAM mode baselines ======
run_mode "gintf_stream_baseline" 0 ""
run_mode "gintf_stream_neuralrx_high" 0 "$AI" 0 120 neural_rx 1.0
run_mode "gintf_stream_gpt2" 0 "$GPT2" 0 120
run_mode "gintf_stream_resnet" 0 "$RESNET" 0 120 128
run_mode "gintf_stream_hbm" 0 "$HBM" 0 120 2

# ====== GRAPH mode — same workloads ======
run_mode "gintf_graph_baseline" 1 ""
run_mode "gintf_graph_neuralrx_high" 1 "$AI" 0 120 neural_rx 1.0
run_mode "gintf_graph_gpt2" 1 "$GPT2" 0 120
run_mode "gintf_graph_resnet" 1 "$RESNET" 0 120 128
run_mode "gintf_graph_hbm" 1 "$HBM" 0 120 2

echo "============================================================"
echo "ALL COMPLETE"
echo "============================================================"
