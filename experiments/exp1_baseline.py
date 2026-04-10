"""
Experiment 1: L1 Baseline Latency Measurement
==============================================
Measures cuPHY L1 pipeline latency without any AI workload interference.

Measures:
- PUSCH Rx (full fused pipeline) per-slot latency
- PUSCH Rx (separable pipeline) per-component latency
- PDSCH Tx per-slot latency
- Overall TTI processing time

Usage:
  shifter --image=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb bash -c '
  export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
  export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
  export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH
  python3 /pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/exp1_baseline.py
  '
"""
import os
import sys
import json
import time
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cupy as cp
import tensorflow as tf

# Configure TF to not grab all GPU memory.
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

from aerial.phy5g.pdsch import PdschTx, PdschTxPipelineFactory
from aerial.phy5g.pusch import PuschRx, PuschRxPipelineFactory
from aerial.phy5g.pusch import SeparablePuschRx, SeparablePuschRxPipelineFactory
from aerial.phy5g.config import (
    AerialPuschRxConfig, AerialPdschTxConfig,
    PdschConfig, PdschUeConfig, PdschCwConfig,
    PuschConfig, PuschUeConfig,
)
from aerial.phy5g.ldpc import get_mcs, get_tb_size, random_tb
from aerial.util.cuda import get_cuda_stream
from aerial.pycuphy.types import PuschLdpcKernelLaunch

sys.path.insert(0, os.path.dirname(__file__))
from utils.timing import CudaTimer, LatencyTracker

# ============================================================
# Configuration
# ============================================================
NUM_WARMUP = 50        # Warmup iterations (not measured)
NUM_ITERATIONS = 1000  # Measured iterations
RESULTS_DIR = "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results"

# 5G NR parameters (same as pyAerial example)
NUM_OFDM_SYMBOLS = 14
FFT_SIZE = 4096
CYCLIC_PREFIX_LENGTH = 288
SUBCARRIER_SPACING = 30e3
NUM_GUARD_SUBCARRIERS = (410, 410)
NUM_SLOTS_PER_FRAME = 20

NUM_TX_ANT = 1
NUM_RX_ANT = 2
CELL_ID = 41
NUM_PRBS = 273
START_PRB = 0
START_SYM = 2
NUM_SYMBOLS = 12
LAYERS = 1
MCS_INDEX = 2
MCS_TABLE = 0
DMRS_PORTS = 1
DMRS_SCRM_ID = 41
DMRS_SYMS = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
DMRS_MAX_LEN = 1
DMRS_ADD_LN_POS = 0
NUM_DMRS_CDM_GRPS_NO_DATA = 2
RNTI = 1234
SCID = 0
DATA_SCID = 0

# Channel
CARRIER_FREQUENCY = 3.5e9
ESNO_DB = 10.0  # Fixed Es/No for baseline (high SNR, no errors)


def setup_channel():
    """Set up Sionna Rayleigh channel."""
    import sionna

    resource_grid = sionna.phy.ofdm.ResourceGrid(
        num_ofdm_symbols=NUM_OFDM_SYMBOLS,
        fft_size=FFT_SIZE,
        subcarrier_spacing=SUBCARRIER_SPACING,
        num_tx=NUM_TX_ANT,
        num_streams_per_tx=1,
        cyclic_prefix_length=CYCLIC_PREFIX_LENGTH,
        num_guard_carriers=NUM_GUARD_SUBCARRIERS,
        dc_null=False,
        pilot_pattern=None,
        pilot_ofdm_symbol_indices=None,
    )
    resource_grid_mapper = sionna.phy.ofdm.ResourceGridMapper(resource_grid)
    remove_guard = sionna.phy.ofdm.RemoveNulledSubcarriers(resource_grid)

    ch_model = sionna.phy.channel.RayleighBlockFading(
        num_rx=1, num_rx_ant=NUM_RX_ANT, num_tx=1, num_tx_ant=NUM_TX_ANT
    )
    channel = sionna.phy.channel.OFDMChannel(
        ch_model, resource_grid, add_awgn=True, normalize_channel=True, return_channel=False
    )

    def apply_channel(tx_tensor, No):
        tx_tensor = tf.experimental.dlpack.from_dlpack(tx_tensor.toDlpack())
        tx_tensor = tf.transpose(tx_tensor, (2, 1, 0))
        tx_tensor = tf.reshape(tx_tensor, (NUM_TX_ANT, -1))[None, :, None, :]
        tx_tensor = resource_grid_mapper(tx_tensor)
        rx_tensor = channel(tx_tensor, No)
        rx_tensor = remove_guard(rx_tensor)
        rx_tensor = rx_tensor[0, 0]
        rx_tensor = tf.transpose(rx_tensor, (2, 1, 0))
        rx_tensor = tf.experimental.dlpack.to_dlpack(rx_tensor)
        return cp.from_dlpack(rx_tensor)

    return apply_channel


def setup_pipelines():
    """Create TX and RX pipelines."""
    cuda_stream = get_cuda_stream()
    mod_order, code_rate = get_mcs(MCS_INDEX, MCS_TABLE + 1)
    tb_size = get_tb_size(
        mod_order=mod_order, code_rate=code_rate, dmrs_syms=DMRS_SYMS,
        num_prbs=NUM_PRBS, start_sym=START_SYM, num_symbols=NUM_SYMBOLS, num_layers=LAYERS
    )

    # TX config
    pdsch_cw_config = PdschCwConfig(mcs_table=MCS_TABLE, mcs_index=MCS_INDEX,
                                     code_rate=int(code_rate * 10), mod_order=mod_order)
    pdsch_ue_config = PdschUeConfig(cw_configs=[pdsch_cw_config], scid=SCID,
                                     dmrs_scrm_id=DMRS_SCRM_ID, layers=LAYERS,
                                     dmrs_ports=DMRS_PORTS, rnti=RNTI, data_scid=DATA_SCID)
    pdsch_config = PdschConfig(ue_configs=[pdsch_ue_config],
                                num_dmrs_cdm_grps_no_data=NUM_DMRS_CDM_GRPS_NO_DATA,
                                start_prb=START_PRB, num_prbs=NUM_PRBS,
                                dmrs_syms=DMRS_SYMS, start_sym=START_SYM, num_symbols=NUM_SYMBOLS)
    pdsch_tx_config = AerialPdschTxConfig(cell_id=CELL_ID, num_tx_ant=NUM_TX_ANT)
    tx_pipeline = PdschTxPipelineFactory().create(pdsch_tx_config, cuda_stream)

    # RX config
    pusch_ue_config = PuschUeConfig(scid=SCID, layers=LAYERS, dmrs_ports=DMRS_PORTS,
                                     rnti=RNTI, data_scid=DATA_SCID, mcs_table=MCS_TABLE,
                                     mcs_index=MCS_INDEX, code_rate=int(code_rate * 10),
                                     mod_order=mod_order, tb_size=tb_size // 8)
    pusch_configs = [PuschConfig(ue_configs=[pusch_ue_config],
                                  num_dmrs_cdm_grps_no_data=NUM_DMRS_CDM_GRPS_NO_DATA,
                                  dmrs_scrm_id=DMRS_SCRM_ID, start_prb=START_PRB,
                                  num_prbs=NUM_PRBS, dmrs_syms=DMRS_SYMS,
                                  dmrs_max_len=DMRS_MAX_LEN, dmrs_add_ln_pos=DMRS_ADD_LN_POS,
                                  start_sym=START_SYM, num_symbols=NUM_SYMBOLS)]

    pusch_rx_config = AerialPuschRxConfig(
        cell_id=CELL_ID, num_rx_ant=NUM_RX_ANT, enable_pusch_tdi=0, eq_coeff_algo=1,
        ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL
    )
    fused_pipeline = PuschRxPipelineFactory().create(pusch_rx_config, cuda_stream)
    separable_pipeline = SeparablePuschRxPipelineFactory().create(pusch_rx_config, cuda_stream)

    return {
        "tx_pipeline": tx_pipeline,
        "fused_rx": fused_pipeline,
        "separable_rx": separable_pipeline,
        "pdsch_config": pdsch_config,
        "pusch_configs": pusch_configs,
        "mod_order": mod_order,
        "code_rate": code_rate,
        "cuda_stream": cuda_stream,
    }


def run_baseline():
    """Run baseline L1 latency measurement."""
    print("=" * 60)
    print("Experiment 1: L1 Baseline Latency Measurement")
    print("=" * 60)
    print(f"Warmup: {NUM_WARMUP}, Iterations: {NUM_ITERATIONS}")
    print()

    # Setup
    print("[1/4] Setting up channel model...")
    apply_channel = setup_channel()

    print("[2/4] Creating L1 pipelines...")
    ctx = setup_pipelines()
    No = pow(10.0, -ESNO_DB / 10.0)

    # Trackers
    tx_tracker = LatencyTracker("PDSCH_Tx")
    fused_rx_tracker = LatencyTracker("PUSCH_Rx_Fused")
    separable_rx_tracker = LatencyTracker("PUSCH_Rx_Separable")
    tti_tracker = LatencyTracker("TTI_Total")
    channel_tracker = LatencyTracker("Channel")

    timer = CudaTimer()
    total_iters = NUM_WARMUP + NUM_ITERATIONS

    print(f"[3/4] Running {total_iters} iterations (first {NUM_WARMUP} warmup)...")
    for i in range(total_iters):
        is_measuring = i >= NUM_WARMUP
        slot_number = i % NUM_SLOTS_PER_FRAME

        # Generate random transport block
        tb_input_np = random_tb(
            mod_order=ctx["mod_order"], code_rate=ctx["code_rate"],
            dmrs_syms=DMRS_SYMS, num_prbs=NUM_PRBS,
            start_sym=START_SYM, num_symbols=NUM_SYMBOLS, num_layers=LAYERS
        )
        tb_input = cp.array(tb_input_np, dtype=cp.uint8, order="F")

        tti_start = time.perf_counter()

        # --- TX ---
        timer.start()
        tx_tensor = ctx["tx_pipeline"](
            slot=slot_number, tb_inputs=[tb_input], config=[ctx["pdsch_config"]]
        )
        timer.stop()
        if is_measuring:
            tx_tracker.record(timer.elapsed_ms())

        # --- Channel ---
        ch_start = time.perf_counter()
        rx_tensor = apply_channel(tx_tensor, No)
        ch_end = time.perf_counter()
        if is_measuring:
            channel_tracker.record((ch_end - ch_start) * 1000)

        # --- Fused RX ---
        timer.start()
        tb_crcs, tbs = ctx["fused_rx"](
            slot=slot_number, rx_slot=rx_tensor, config=ctx["pusch_configs"]
        )
        timer.stop()
        if is_measuring:
            fused_rx_tracker.record(timer.elapsed_ms())

        # --- Separable RX ---
        timer.start()
        tb_crcs2, tbs2 = ctx["separable_rx"](
            slot=slot_number, rx_slot=rx_tensor, config=ctx["pusch_configs"]
        )
        timer.stop()
        if is_measuring:
            separable_rx_tracker.record(timer.elapsed_ms())

        tti_end = time.perf_counter()
        if is_measuring:
            tti_tracker.record((tti_end - tti_start) * 1000)

        if is_measuring and (i - NUM_WARMUP) % 200 == 0:
            print(f"  [{i - NUM_WARMUP}/{NUM_ITERATIONS}] "
                  f"TTI={tti_tracker.latencies[-1]:.2f}ms "
                  f"FusedRx={fused_rx_tracker.latencies[-1]:.2f}ms")

    # --- Results ---
    print()
    print("[4/4] Results")
    print("=" * 60)

    # GPU info
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    gpu_mem = cp.cuda.Device(0).mem_info
    print(f"GPU: {gpu_name}")
    print(f"HBM: {gpu_mem[1] / 1e9:.1f} GB total, {gpu_mem[0] / 1e9:.1f} GB free")
    print()

    trackers = [tx_tracker, channel_tracker, fused_rx_tracker, separable_rx_tracker, tti_tracker]
    results = {}
    for t in trackers:
        s = t.stats()
        results[t.name] = s
        print(f"--- {t.name} ---")
        print(f"  Mean:  {s['mean_ms']:.3f} ms")
        print(f"  Std:   {s['std_ms']:.3f} ms")
        print(f"  P50:   {s['p50_ms']:.3f} ms")
        print(f"  P95:   {s['p95_ms']:.3f} ms")
        print(f"  P99:   {s['p99_ms']:.3f} ms")
        print(f"  Jitter: {s['jitter']:.4f}")
        print(f"  Deadline miss (>1ms): {t.deadline_miss_rate(1.0)*100:.1f}%")
        print()

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"exp1_baseline_{timestamp}.json")

    output = {
        "experiment": "exp1_baseline",
        "timestamp": timestamp,
        "config": {
            "num_warmup": NUM_WARMUP,
            "num_iterations": NUM_ITERATIONS,
            "num_prbs": NUM_PRBS,
            "mcs_index": MCS_INDEX,
            "num_rx_ant": NUM_RX_ANT,
            "num_tx_ant": NUM_TX_ANT,
            "esno_db": ESNO_DB,
        },
        "results": results,
        "raw_latencies": {
            t.name: t.latencies for t in trackers
        },
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    run_baseline()
