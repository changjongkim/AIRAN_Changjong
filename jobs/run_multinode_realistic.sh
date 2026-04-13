#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J mn_real
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/mn_real_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/mn_real_%j.err

# ===================================================================
# Realistic Multi-Node AI-RAN Orchestration Experiment
#
# Scenario: 2-node AI-RAN cluster, 40GB GPUs
#   Node 0: Urban gNB (20 cells, heavy L1)
#   Node 1: Suburban gNB (5 cells, light L1)
#
# Modes:
#   1. Both L1 only (baseline)
#   2. AI on Node 0 (busy node) → interference on heavy L1
#   3. AI on Node 1 (idle node) → no interference expected
#   4. AI on both → worst case
#   5. Flash crowd: AI on Node 1, then migrate to Node 0 mid-run
#   6. Smart placement: AI on Node 1 only (optimal)
#   7. Both L1 only (final baseline)
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_heavy_timed.py
HBM=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py
GPT2=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_gpt2_stress.py

NODE0=$(scontrol show hostname $SLURM_JOB_NODELIST | head -1)
NODE1=$(scontrol show hostname $SLURM_JOB_NODELIST | tail -1)

# Node 0: heavy (20 cells), Node 1: light (5 cells)
N0_CELLS=20; N1_CELLS=5
TX=4; RX=4; DUR=45

echo "============================================================"
echo "Realistic Multi-Node AI-RAN Orchestration"
echo "  Node 0 (urban):    $NODE0 — ${N0_CELLS} cells (heavy)"
echo "  Node 1 (suburban): $NODE1 — ${N1_CELLS} cells (light)"
echo "============================================================"
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

# ====== Mode 1: Both L1 only (baseline) ======
echo "=========================================="
echo "MODE 1: Both L1 only (baseline)"
echo "=========================================="
start_mps m1
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE \
    python3 $L1 mn_m1_node0_heavy $TX $RX $N0_CELLS $DUR &
PID0=$!
srun -N1 -n1 --nodelist=$NODE1 shifter --image=$IMAGE \
    python3 $L1 mn_m1_node1_light $TX $RX $N1_CELLS $DUR &
PID1=$!
wait $PID0 $PID1
stop_mps
echo ""

# ====== Mode 2: AI (GPT-2) on Node 0 (busy node) ======
echo "=========================================="
echo "MODE 2: AI on Node 0 (busy node → interference)"
echo "=========================================="
start_mps m2
# AI on Node 0
srun -N1 -n1 --nodelist=$NODE0 env PYTHONPATH=$PYTHONPATH \
    shifter --image=$IMAGE python3 $GPT2 0 120 &
AI_PID=$!
sleep 10
# L1 on both nodes
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE \
    python3 $L1 mn_m2_node0_heavy_withAI $TX $RX $N0_CELLS $DUR &
PID0=$!
srun -N1 -n1 --nodelist=$NODE1 shifter --image=$IMAGE \
    python3 $L1 mn_m2_node1_light_noAI $TX $RX $N1_CELLS $DUR &
PID1=$!
wait $PID0 $PID1
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# ====== Mode 3: AI (GPT-2) on Node 1 (idle node) ======
echo "=========================================="
echo "MODE 3: AI on Node 1 (idle node → less interference)"
echo "=========================================="
start_mps m3
srun -N1 -n1 --nodelist=$NODE1 env PYTHONPATH=$PYTHONPATH \
    shifter --image=$IMAGE python3 $GPT2 0 120 &
AI_PID=$!
sleep 10
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE \
    python3 $L1 mn_m3_node0_heavy_noAI $TX $RX $N0_CELLS $DUR &
PID0=$!
srun -N1 -n1 --nodelist=$NODE1 shifter --image=$IMAGE \
    python3 $L1 mn_m3_node1_light_withAI $TX $RX $N1_CELLS $DUR &
PID1=$!
wait $PID0 $PID1
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# ====== Mode 4: AI on both nodes ======
echo "=========================================="
echo "MODE 4: AI on both nodes (worst case)"
echo "=========================================="
start_mps m4
srun -N1 -n1 --nodelist=$NODE0 env PYTHONPATH=$PYTHONPATH \
    shifter --image=$IMAGE python3 $GPT2 0 120 &
AI0=$!
srun -N1 -n1 --nodelist=$NODE1 env PYTHONPATH=$PYTHONPATH \
    shifter --image=$IMAGE python3 $GPT2 0 120 &
AI1=$!
sleep 10
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE \
    python3 $L1 mn_m4_node0_heavy_withAI $TX $RX $N0_CELLS $DUR &
PID0=$!
srun -N1 -n1 --nodelist=$NODE1 shifter --image=$IMAGE \
    python3 $L1 mn_m4_node1_light_withAI $TX $RX $N1_CELLS $DUR &
PID1=$!
wait $PID0 $PID1
kill $AI0 $AI1 2>/dev/null; wait $AI0 $AI1 2>/dev/null
stop_mps
echo ""

# ====== Mode 5: HBM stress on busy node vs idle node ======
echo "=========================================="
echo "MODE 5: HBM stress on Node 0 (busy) vs Node 1 (idle)"
echo "=========================================="
start_mps m5
# HBM on Node 0
srun -N1 -n1 --nodelist=$NODE0 env PYTHONPATH=$PYTHONPATH \
    shifter --image=$IMAGE python3 $HBM 0 120 2 &
HBM_PID=$!
sleep 5
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE \
    python3 $L1 mn_m5_node0_heavy_withHBM $TX $RX $N0_CELLS $DUR &
PID0=$!
srun -N1 -n1 --nodelist=$NODE1 shifter --image=$IMAGE \
    python3 $L1 mn_m5_node1_light_noHBM $TX $RX $N1_CELLS $DUR &
PID1=$!
wait $PID0 $PID1
kill $HBM_PID 2>/dev/null; wait $HBM_PID 2>/dev/null
stop_mps
echo ""

# ====== Mode 6: Baseline final ======
echo "=========================================="
echo "MODE 6: Both L1 only (final baseline)"
echo "=========================================="
start_mps m6
srun -N1 -n1 --nodelist=$NODE0 shifter --image=$IMAGE \
    python3 $L1 mn_m6_node0_final $TX $RX $N0_CELLS $DUR &
PID0=$!
srun -N1 -n1 --nodelist=$NODE1 shifter --image=$IMAGE \
    python3 $L1 mn_m6_node1_final $TX $RX $N1_CELLS $DUR &
PID1=$!
wait $PID0 $PID1
stop_mps

echo ""
echo "============================================================"
echo "ALL MODES COMPLETE"
echo "============================================================"
