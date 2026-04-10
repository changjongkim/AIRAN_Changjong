#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J test_mig
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/test_mig_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/test_mig_%j.err

echo "=== Before MIG ==="
nvidia-smi -q | grep -A3 "MIG Mode"
nvidia-smi

echo ""
echo "=== Trying to enable MIG ==="
nvidia-smi -i 0 -mig 1 2>&1
echo "Exit code: $?"

echo ""
echo "=== After MIG enable attempt ==="
nvidia-smi -q | grep -A3 "MIG Mode"

# If MIG enabled, try creating partitions
# 3g.20gb (L1) + 4g.20gb (AI) = full GPU
echo ""
echo "=== Trying MIG partitions ==="
nvidia-smi mig -i 0 -cgi 9,5 -C 2>&1
echo "Exit code: $?"

echo ""
echo "=== MIG status ==="
nvidia-smi mig -i 0 -lgi 2>&1
nvidia-smi mig -i 0 -lci 2>&1
nvidia-smi 2>&1

# Cleanup
echo ""
echo "=== Cleanup ==="
nvidia-smi mig -i 0 -dci 2>&1
nvidia-smi mig -i 0 -dgi 2>&1
nvidia-smi -i 0 -mig 0 2>&1
