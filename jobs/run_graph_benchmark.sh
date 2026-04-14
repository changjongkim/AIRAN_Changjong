#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J graph_bench
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/graph_bench_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/graph_bench_%j.err

# ===================================================================
# CUDA Graph vs Stream benchmark — multi-cell cuPHY
# Compare actual L1 latency between stream mode and graph mode
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
SCRIPT=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_graph.py

echo "============================================================"
echo "CUDA Graph vs Stream: Multi-cell cuPHY Benchmark"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

for cells in 1 2 4 8; do
    for mode in 0 1; do
        mode_str=$([ "$mode" = "1" ] && echo "GRAPH" || echo "STREAM")
        label="bench_${cells}c_${mode_str}"
        echo "=== ${cells} cells, ${mode_str} ==="
        shifter --image=$IMAGE python3 $SCRIPT $label $cells $mode
        echo ""
    done
done

echo "ALL COMPLETE"
