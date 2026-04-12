#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J q2_dynamic
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/q2_dynamic_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/q2_dynamic_%j.err

# ===================================================================
# Q2: Dynamic Workload — Flash Crowd Simulation
#
# Simulates traffic spike: L1 is running calmly, then suddenly
# AI workload starts (simulating "AI was migrated here" or
# "new AI tenant added"). Measures how quickly L1 is affected.
#
# Modes:
#   1. baseline (L1 solo, steady state)
#   2. L1 running → start ResNet mid-measurement → observe transition
#   3. L1 running → start HBM stress mid-measurement → observe transition
#   4. L1 running → start Qwen-72B mid-measurement → observe loading impact
#   5. L1 + AI running → kill AI mid-measurement → observe recovery
# ===================================================================

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
HBM=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_hbm_stress.py
RESNET=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_resnet_stress.py
QWEN=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_qwen72b_stress.py

echo "============================================================"
echo "Q2: Dynamic Workload — Flash Crowd Simulation"
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

# Dynamic L1 measurement script — measures per-iteration latency
# and records timestamps so we can see the transition point
DYN_SCRIPT=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_dynamic.py
# (dynamic script is now a proper file, not inline heredoc)
cat > /dev/null << 'PYEOF'
import os, sys, json, time, datetime
sys.stdout.reconfigure(line_buffering=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

NUM_WARMUP = 5
NUM_ITERATIONS = 300  # More iterations to capture transition
RESULTS_DIR = "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results"
ESNO_DB = 10.0
NUM_OFDM_SYMBOLS = 14; FFT_SIZE = 4096; CYCLIC_PREFIX_LENGTH = 288
SUBCARRIER_SPACING = 30e3; NUM_GUARD_SUBCARRIERS = (410, 410)
NUM_SLOTS_PER_FRAME = 20; NUM_PRBS = 273; START_PRB = 0; START_SYM = 2
NUM_SYMBOLS = 12; DMRS_SCRM_ID = 41
DMRS_SYMS = [0,0,1,0,0,0,0,0,0,0,0,0,0,0]
DMRS_MAX_LEN = 1; DMRS_ADD_LN_POS = 0; NUM_DMRS_CDM_GRPS_NO_DATA = 2
RNTI = 1234; SCID = 0; DATA_SCID = 0

def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "dynamic"
    import numpy as np, cupy as cp, tensorflow as tf, sionna
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    if len(gpus) > 1: tf.config.set_visible_devices(gpus[0], "GPU")
    cp.cuda.Device(0).use()
    from aerial.phy5g.pdsch import PdschTxPipelineFactory
    from aerial.phy5g.pusch import PuschRxPipelineFactory
    from aerial.phy5g.config import *
    from aerial.phy5g.ldpc import get_mcs, get_tb_size, random_tb
    from aerial.util.cuda import get_cuda_stream
    from aerial.pycuphy.types import PuschLdpcKernelLaunch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("/tmp", "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments"))
    from utils.timing import CudaTimer

    num_tx, num_rx, mcs_index = 1, 2, 2
    rg = sionna.phy.ofdm.ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS, fft_size=FFT_SIZE,
        subcarrier_spacing=SUBCARRIER_SPACING, num_tx=num_tx, num_streams_per_tx=1,
        cyclic_prefix_length=CYCLIC_PREFIX_LENGTH, num_guard_carriers=NUM_GUARD_SUBCARRIERS, dc_null=False)
    rg_map = sionna.phy.ofdm.ResourceGridMapper(rg)
    rm_g = sionna.phy.ofdm.RemoveNulledSubcarriers(rg)
    ch = sionna.phy.channel.OFDMChannel(
        sionna.phy.channel.RayleighBlockFading(num_rx=1, num_rx_ant=num_rx, num_tx=1, num_tx_ant=num_tx),
        rg, add_awgn=True, normalize_channel=True, return_channel=False)
    def apply_ch(tx_t, No):
        t = tf.experimental.dlpack.from_dlpack(tx_t.toDlpack())
        t = tf.transpose(t, (2, 1, 0)); t = tf.reshape(t, (num_tx, -1))[None, :, None, :]
        t = rg_map(t); r = ch(t, No); r = rm_g(r)[0, 0]; r = tf.transpose(r, (2, 1, 0))
        return cp.from_dlpack(tf.experimental.dlpack.to_dlpack(r))

    stream = get_cuda_stream()
    mo, cr = get_mcs(mcs_index, 1)
    tb_sz = get_tb_size(mod_order=mo, code_rate=cr, dmrs_syms=DMRS_SYMS, num_prbs=NUM_PRBS,
                        start_sym=START_SYM, num_symbols=NUM_SYMBOLS, num_layers=1)
    No = pow(10.0, -ESNO_DB / 10.0)
    cid = 41
    pcw = PdschCwConfig(mcs_table=0, mcs_index=mcs_index, code_rate=int(cr*10), mod_order=mo)
    pue = PdschUeConfig(cw_configs=[pcw], scid=SCID, dmrs_scrm_id=DMRS_SCRM_ID, layers=1, dmrs_ports=1, rnti=RNTI, data_scid=DATA_SCID)
    pcfg = PdschConfig(ue_configs=[pue], num_dmrs_cdm_grps_no_data=NUM_DMRS_CDM_GRPS_NO_DATA,
                       start_prb=START_PRB, num_prbs=NUM_PRBS, dmrs_syms=DMRS_SYMS, start_sym=START_SYM, num_symbols=NUM_SYMBOLS)
    txp = PdschTxPipelineFactory().create(AerialPdschTxConfig(cell_id=cid, num_tx_ant=num_tx), stream)
    uue = PuschUeConfig(scid=SCID, layers=1, dmrs_ports=1, rnti=RNTI, data_scid=DATA_SCID, mcs_table=0,
                        mcs_index=mcs_index, code_rate=int(cr*10), mod_order=mo, tb_size=tb_sz//8)
    ucfg = [PuschConfig(ue_configs=[uue], num_dmrs_cdm_grps_no_data=NUM_DMRS_CDM_GRPS_NO_DATA,
                        dmrs_scrm_id=DMRS_SCRM_ID, start_prb=START_PRB, num_prbs=NUM_PRBS,
                        dmrs_syms=DMRS_SYMS, dmrs_max_len=DMRS_MAX_LEN, dmrs_add_ln_pos=DMRS_ADD_LN_POS,
                        start_sym=START_SYM, num_symbols=NUM_SYMBOLS)]
    rxp = PuschRxPipelineFactory().create(
        AerialPuschRxConfig(cell_id=cid, num_rx_ant=num_rx, enable_pusch_tdi=0, eq_coeff_algo=1,
                            ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL), stream)

    timer = CudaTimer()
    latencies = []
    timestamps = []
    t0 = time.time()

    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        slot = i % NUM_SLOTS_PER_FRAME
        tb = cp.array(random_tb(mod_order=mo, code_rate=cr, dmrs_syms=DMRS_SYMS, num_prbs=NUM_PRBS,
                                start_sym=START_SYM, num_symbols=NUM_SYMBOLS, num_layers=1), dtype=cp.uint8, order="F")
        tx_t = txp(slot=slot, tb_inputs=[tb], config=[pcfg])
        rx_t = apply_ch(tx_t, No)
        timer.start(); rxp(slot=slot, rx_slot=rx_t, config=ucfg); timer.stop()
        if i >= NUM_WARMUP:
            lat = timer.elapsed_ms()
            latencies.append(lat)
            timestamps.append(time.time() - t0)
            if (i - NUM_WARMUP) % 50 == 0:
                print(f"  [{i-NUM_WARMUP}/{NUM_ITERATIONS}] t={timestamps[-1]:.1f}s RX={lat:.3f}ms", flush=True)

    arr = np.array(latencies)
    print(f"\nRESULT: {label}", flush=True)
    print(f"  RX mean:  {np.mean(arr):.3f} ms", flush=True)
    print(f"  RX P99:   {np.percentile(arr, 99):.3f} ms", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"exp_{label}_{ts}.json")
    with open(out, "w") as f:
        json.dump({"label": label, "latencies": latencies, "timestamps": timestamps,
                    "mean_ms": float(np.mean(arr)), "p99_ms": float(np.percentile(arr, 99))}, f, indent=2)
    print(f"Saved: {out}", flush=True)

if __name__ == "__main__":
    main()
PYEOF

# ====== Mode 1: Baseline (steady state) ======
echo "=========================================="
echo "MODE: baseline_steady"
echo "=========================================="
start_mps baseline
shifter --image=$IMAGE python3 $DYN_SCRIPT dyn_baseline
stop_mps
echo ""

# ====== Mode 2: L1 running, then HBM stress starts at t=10s ======
echo "=========================================="
echo "MODE: flash_hbm (HBM starts mid-run)"
echo "=========================================="
start_mps flash_hbm
# Start L1 in background
shifter --image=$IMAGE python3 $DYN_SCRIPT dyn_flash_hbm &
L1_PID=$!
# Wait 10 seconds, then start HBM stress
sleep 10
echo ">>> INJECTING HBM STRESS at t=10s <<<"
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $HBM 0 60 4 &
HBM_PID=$!
# Wait for L1 to finish
wait $L1_PID
kill $HBM_PID 2>/dev/null; wait $HBM_PID 2>/dev/null
stop_mps
echo ""

# ====== Mode 3: L1 running, then ResNet starts at t=10s ======
echo "=========================================="
echo "MODE: flash_resnet (ResNet starts mid-run)"
echo "=========================================="
start_mps flash_resnet
shifter --image=$IMAGE python3 $DYN_SCRIPT dyn_flash_resnet &
L1_PID=$!
sleep 10
echo ">>> INJECTING ResNet-50 bs=128 at t=10s <<<"
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $RESNET 0 60 128 &
AI_PID=$!
wait $L1_PID
kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
stop_mps
echo ""

# ====== Mode 4: HBM stress running, then KILL at t=10s (recovery test) ======
echo "=========================================="
echo "MODE: recovery_hbm (HBM killed mid-run)"
echo "=========================================="
start_mps recovery
PYTHONPATH=$PYTHONPATH shifter --image=$IMAGE python3 $HBM 0 120 4 &
HBM_PID=$!
sleep 8
shifter --image=$IMAGE python3 $DYN_SCRIPT dyn_recovery_hbm &
L1_PID=$!
sleep 10
echo ">>> KILLING HBM STRESS at t=10s <<<"
kill $HBM_PID 2>/dev/null; wait $HBM_PID 2>/dev/null
wait $L1_PID
stop_mps
echo ""

echo "============================================================"
echo "ALL MODES COMPLETE"
echo "============================================================"
