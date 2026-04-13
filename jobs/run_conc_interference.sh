#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J conc_intf
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/conc_intf_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/conc_intf_%j.err

# ===================================================================
# CONCURRENT L1 (4T4R × 20cell) + AI Interference on 40GB
# All cells launch without per-cell sync → real SM contention
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_heavy_concurrent.py
HBM=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py
QWEN7B=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_qwen7b_stress.py

TX=4; RX=4; CELLS=20; MCS=2

echo "============================================================"
echo "CONCURRENT L1 Interference: ${TX}T${RX}R × ${CELLS}cells (40GB)"
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

echo "=========================================="; echo "MODE: baseline"
start_mps baseline
shifter --image=$IMAGE python3 $L1 conc_baseline $TX $RX $CELLS $MCS
stop_mps; echo ""

echo "=========================================="; echo "MODE: + HBM 2GB"
start_mps hbm
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $HBM 0 180 2 &
AI_PID=$!; sleep 8
shifter --image=$IMAGE python3 $L1 conc_hbm2g $TX $RX $CELLS $MCS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps; echo ""

echo "=========================================="; echo "MODE: + GPT-2"
start_mps gpt2
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $GPT2 0 180 &
AI_PID=$!; sleep 10
shifter --image=$IMAGE python3 $L1 conc_gpt2 $TX $RX $CELLS $MCS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps; echo ""

echo "=========================================="; echo "MODE: + ResNet bs128"
start_mps resnet
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $RESNET 0 180 128 &
AI_PID=$!; sleep 10
shifter --image=$IMAGE python3 $L1 conc_resnet128 $TX $RX $CELLS $MCS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps; echo ""

echo "=========================================="; echo "MODE: + Qwen-7B"
start_mps qwen7b
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $QWEN7B 0 180 &
AI_PID=$!; sleep 15
shifter --image=$IMAGE python3 $L1 conc_qwen7b $TX $RX $CELLS $MCS
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps; echo ""

echo "=========================================="; echo "MODE: baseline_final"
start_mps final
shifter --image=$IMAGE python3 $L1 conc_baseline_final $TX $RX $CELLS $MCS
stop_mps

echo ""; echo "ALL COMPLETE"
