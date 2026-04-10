#!/bin/bash
# Submit MIG emulator experiments as separate jobs.
# Each partition+workload combo is independent.

SCRIPT=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/mig_emu_single.py
OUTDIR=/pscratch/sd/s/sgkim/kcj/AI-RAN/jobs
IMAGE="docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb"

submit_job() {
    local name=$1
    local args=$2

    script=$(mktemp $OUTDIR/job_mig_${name}_XXXX.sh)
    cat > $script << JOBEOF
#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J mig_${name}
#SBATCH -o ${OUTDIR}/mig_${name}_%j.out
#SBATCH -e ${OUTDIR}/mig_${name}_%j.err

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=\$cuBB_SDK/pyaerial/src:\$SITE:\$PYTHONPATH

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_\$\$
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_\$\$
mkdir -p \$CUDA_MPS_PIPE_DIRECTORY \$CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
sleep 2

shifter --image=${IMAGE} python3 ${SCRIPT} ${args}

echo quit | nvidia-cuda-mps-control
JOBEOF

    jid=$(sbatch $script 2>&1 | awk '{print $4}')
    echo "  $name → $jid"
}

echo "Submitting MIG emulator jobs..."

# Baseline — no AI
submit_job "baseline" "--l1-profile 7g --ai-workload none"

# No MIG (full GPU sharing) — worst case
submit_job "noMIG_hbm"    "--l1-profile 7g --ai-profile 7g --ai-workload hbm_stress"
submit_job "noMIG_resnet"  "--l1-profile 7g --ai-profile 7g --ai-workload resnet"
submit_job "noMIG_gpt2"    "--l1-profile 7g --ai-profile 7g --ai-workload gpt2"

# MIG 4g+3g (balanced)
submit_job "4g3g_hbm"    "--l1-profile 4g --ai-profile 3g --ai-workload hbm_stress"
submit_job "4g3g_resnet"  "--l1-profile 4g --ai-profile 3g --ai-workload resnet"
submit_job "4g3g_gpt2"    "--l1-profile 4g --ai-profile 3g --ai-workload gpt2"

# MIG 3g+4g (more AI)
submit_job "3g4g_hbm"    "--l1-profile 3g --ai-profile 4g --ai-workload hbm_stress"
submit_job "3g4g_resnet"  "--l1-profile 3g --ai-profile 4g --ai-workload resnet"
submit_job "3g4g_gpt2"    "--l1-profile 3g --ai-profile 4g --ai-workload gpt2"

# MIG 2g+4g (L1 constrained)
submit_job "2g4g_hbm"    "--l1-profile 2g --ai-profile 4g --ai-workload hbm_stress"
submit_job "2g4g_resnet"  "--l1-profile 2g --ai-profile 4g --ai-workload resnet"

# MIG 1g+4g (L1 severely constrained)
submit_job "1g4g_hbm"    "--l1-profile 1g --ai-profile 4g --ai-workload hbm_stress"
submit_job "1g4g_resnet"  "--l1-profile 1g --ai-profile 4g --ai-workload resnet"

echo ""
echo "Total: 14 jobs submitted"
