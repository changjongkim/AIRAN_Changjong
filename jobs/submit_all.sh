#!/bin/bash
# Submit all Phase 0 experiment modes as separate jobs.
# Each job is independent — no MPS crash propagation.

SCRIPT=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/exp_single_mode.py
OUTDIR=/pscratch/sd/s/sgkim/kcj/AI-RAN/jobs

COMMON="
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
"

ENV="
export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=\$cuBB_SDK/pyaerial/src:\$SITE:\$PYTHONPATH
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_\$\$
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_\$\$
mkdir -p \$CUDA_MPS_PIPE_DIRECTORY \$CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
sleep 2
"

CLEANUP='echo quit | nvidia-cuda-mps-control'
IMAGE="docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb"

# L1 configs: "cells tx rx mcs label"
CONFIGS=(
    "1 1 2 2"
    "4 2 4 10"
    "8 4 8 15"
)

# Modes
MODES=(baseline hbm_same hbm_diff resnet_same resnet_diff gpt2_same gpt2_diff)

for cfg in "${CONFIGS[@]}"; do
    read -r cells tx rx mcs <<< "$cfg"
    tag="${cells}c_${tx}T${rx}R_mcs${mcs}"

    for mode in "${MODES[@]}"; do
        jobname="p0_${tag}_${mode}"
        script=$(mktemp $OUTDIR/job_${jobname}_XXXX.sh)

        cat > $script << JOBEOF
#!/bin/bash
${COMMON}
#SBATCH -J ${jobname}
#SBATCH -o ${OUTDIR}/${jobname}_%j.out
#SBATCH -e ${OUTDIR}/${jobname}_%j.err

${ENV}

shifter --image=${IMAGE} \\
    python3 ${SCRIPT} \\
    --mode ${mode} --cells ${cells} --tx ${tx} --rx ${rx} --mcs ${mcs}

${CLEANUP}
JOBEOF

        jid=$(sbatch $script 2>&1 | awk '{print $4}')
        echo "Submitted $jobname → $jid"
    done
done

echo ""
echo "Total: $((${#CONFIGS[@]} * ${#MODES[@]})) jobs submitted"
