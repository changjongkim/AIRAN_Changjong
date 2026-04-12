#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J heavy_sm
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/heavy_sm_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/heavy_sm_%j.err

# ===================================================================
# Heavy L1 (4T4R × 20cell) SM% Sweep
# Now L1 uses real GPU resources — SM% reduction should matter
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_heavy.py

TX=4; RX=4; CELLS=20; MCS=2

echo "============================================================"
echo "Heavy L1 SM% Sweep: ${TX}T${RX}R × ${CELLS}cells"
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

for sm in 100 80 60 50 40 30 20 14 10; do
    label="heavy_sm${sm}"
    echo "=========================================="
    echo "MODE: SM=${sm}% (${TX}T${RX}R × ${CELLS}cells)"
    echo "=========================================="
    start_mps $label
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$sm shifter --image=$IMAGE \
        python3 $L1 $label $TX $RX $CELLS $MCS
    stop_mps
    echo ""
done

echo "============================================================"
echo "ALL MODES COMPLETE"
echo "============================================================"
