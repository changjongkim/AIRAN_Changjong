#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J g40_sm
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/g40_sm_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/g40_sm_%j.err

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_heavy.py
TX=4; RX=4; CELLS=20; MCS=2

echo "============================================================"
echo "40GB SM% Sweep: ${TX}T${RX}R × ${CELLS}cells"
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

for sm in 100 50 30 20 10; do
    echo "=========================================="; echo "MODE: SM=${sm}%"
    start_mps sm$sm
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$sm shifter --image=$IMAGE \
        python3 $L1 g40_sm${sm} $TX $RX $CELLS $MCS
    stop_mps; echo ""
done

echo "ALL COMPLETE"
