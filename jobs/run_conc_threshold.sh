#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J conc_thresh
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/conc_thresh_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/conc_thresh_%j.err

# CONCURRENT L1 Threshold — bandwidth interference with concurrent cells
export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_heavy_concurrent.py
HBM=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py
TX=4; RX=4; CELLS=20; MCS=2

echo "============================================================"
echo "CONCURRENT Threshold: ${TX}T${RX}R × ${CELLS}cells (40GB)"
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

echo "=========================================="; echo "MODE: baseline"
start_mps baseline
shifter --image=$IMAGE python3 $L1 conc_thresh_baseline $TX $RX $CELLS $MCS
stop_mps; echo ""

for size in 0.1 0.5 1.0 2.0; do
    label="conc_thresh_${size}GB"
    echo "=========================================="; echo "MODE: HBM ${size}GB"
    start_mps $label
    PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $HBM 0 120 $size &
    AI_PID=$!; sleep 5
    shifter --image=$IMAGE python3 $L1 $label $TX $RX $CELLS $MCS
    kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
    stop_mps; echo ""
done

echo "=========================================="; echo "MODE: baseline_final"
start_mps final
shifter --image=$IMAGE python3 $L1 conc_thresh_baseline_final $TX $RX $CELLS $MCS
stop_mps

echo ""; echo "ALL COMPLETE"
