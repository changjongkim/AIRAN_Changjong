#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J scale2n_v3
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scale2n_v3_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/scale2n_v3_%j.err

# Use ssh to run on each node directly (not srun)
# This ensures AI and L1 share the same GPU via MPS on each node

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

# Common env for ssh commands
ENVSETUP="export cuBB_SDK=$cuBB_SDK; export SITE=$SITE; export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:\$PYTHONPATH; export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_\$\$; export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_\$\$; mkdir -p \$CUDA_MPS_PIPE_DIRECTORY \$CUDA_MPS_LOG_DIRECTORY; nvidia-cuda-mps-control -d; sleep 1"

echo "============================================================"
echo "Scale 2N v3: ssh-based (40GB, MPS per node)"
echo "Node0: $NODE0, Node1: $NODE1"
echo "============================================================"

run_mode() {
    local label=$1
    local ai_script=$2
    shift 2
    local ai_args="$@"

    echo "=========================================="
    echo "MODE: $label"
    echo "=========================================="

    for node in $NODE0 $NODE1; do
        # Start MPS on each node
        ssh $node "export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${label}_$$; export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${label}_$$; mkdir -p \$CUDA_MPS_PIPE_DIRECTORY \$CUDA_MPS_LOG_DIRECTORY; nvidia-cuda-mps-control -d" 2>/dev/null &
    done
    sleep 2

    # Start AI on each node (background)
    if [ -n "$ai_script" ]; then
        for node in $NODE0 $NODE1; do
            ssh $node "export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:\$PYTHONPATH; export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${label}_$$; shifter --image=$IMAGE python3 $ai_script $ai_args" &
        done
        sleep 10
        echo "AI started on both nodes"
    fi

    # Run L1 on each node (foreground, wait for both)
    ssh $NODE0 "export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:\$PYTHONPATH; export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${label}_$$; shifter --image=$IMAGE python3 $L1 ${label}_n0 $CELLS 1" &
    PID0=$!
    ssh $NODE1 "export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:\$PYTHONPATH; export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${label}_$$; shifter --image=$IMAGE python3 $L1 ${label}_n1 $CELLS 1" &
    PID1=$!
    wait $PID0 $PID1

    # Kill AI and MPS
    for node in $NODE0 $NODE1; do
        ssh $node "pkill -f 'run_realistic_ai\|run_gpt2\|run_resnet'; echo quit | nvidia-cuda-mps-control" 2>/dev/null &
    done
    wait
    sleep 2
    echo ""
}

run_mode "s2nv3_baseline" ""
run_mode "s2nv3_neuralrx" "$AI" 0 120 neural_rx 1.0
run_mode "s2nv3_gpt2" "$GPT2" 0 120
run_mode "s2nv3_resnet" "$RESNET" 0 120 128

echo "ALL COMPLETE"
