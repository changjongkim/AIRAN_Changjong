#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J mig_sweep_v2
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/mig_sweep_v2_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/mig_sweep_v2_%j.err

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
SCRIPT=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/mig_emu_single.py

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

# Each mode gets a fresh MPS server → no crash propagation
run_mode() {
    local name=$1
    shift
    echo "========== $name =========="
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${name}_$$
    export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${name}_$$
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
    nvidia-cuda-mps-control -d 2>/dev/null
    sleep 1
    shifter --image=$IMAGE python3 $SCRIPT "$@"
    echo quit | nvidia-cuda-mps-control 2>/dev/null
    sleep 2
    echo ""
}

run_mode "baseline"       --l1-profile 7g --ai-workload none
run_mode "noMIG_hbm"      --l1-profile 7g --ai-profile 7g --ai-workload hbm_stress
run_mode "MIG_4g3g_hbm"   --l1-profile 4g --ai-profile 3g --ai-workload hbm_stress
run_mode "MIG_3g4g_hbm"   --l1-profile 3g --ai-profile 4g --ai-workload hbm_stress
run_mode "MIG_2g4g_hbm"   --l1-profile 2g --ai-profile 4g --ai-workload hbm_stress
run_mode "MIG_1g4g_hbm"   --l1-profile 1g --ai-profile 4g --ai-workload hbm_stress
run_mode "baseline_final"  --l1-profile 7g --ai-workload none

echo "All modes complete."
