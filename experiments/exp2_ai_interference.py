"""
Experiment 2: AI Interference Mode
====================================
Measures L1 pipeline latency while AI workloads run concurrently via MPS.

Launches AI workload (ResNet-50 or GPT-2) as a background process sharing
the same GPU via CUDA MPS, then measures L1 latency degradation.

Usage:
  shifter --image=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb bash -c '
  export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
  export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
  export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH
  python3 /pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/exp2_ai_interference.py
  '
"""
import os
import sys
import json
import time
import signal
import datetime
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cupy as cp
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

from aerial.phy5g.pdsch import PdschTxPipelineFactory
from aerial.phy5g.pusch import PuschRxPipelineFactory
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
NUM_WARMUP = 20
NUM_ITERATIONS = 500
RESULTS_DIR = "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results"
ESNO_DB = 10.0

# AI workloads to test
WORKLOADS = {
    "none": None,
    "resnet50": "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/workloads/resnet50_loop.py",
    "gpt2": "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/workloads/gpt2_loop.py",
}

# Wait time (seconds) after launching AI workload before measuring
AI_WARMUP_TIME = 30

# 5G NR parameters (same as exp1)
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


def setup_channel():
    """Set up Sionna Rayleigh channel."""
    import sionna
    resource_grid = sionna.phy.ofdm.ResourceGrid(
        num_ofdm_symbols=NUM_OFDM_SYMBOLS, fft_size=FFT_SIZE,
        subcarrier_spacing=SUBCARRIER_SPACING, num_tx=NUM_TX_ANT,
        num_streams_per_tx=1, cyclic_prefix_length=CYCLIC_PREFIX_LENGTH,
        num_guard_carriers=NUM_GUARD_SUBCARRIERS, dc_null=False,
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
    rx_pipeline = PuschRxPipelineFactory().create(pusch_rx_config, cuda_stream)

    return {
        "tx_pipeline": tx_pipeline,
        "rx_pipeline": rx_pipeline,
        "pdsch_config": pdsch_config,
        "pusch_configs": pusch_configs,
        "mod_order": mod_order,
        "code_rate": code_rate,
        "cuda_stream": cuda_stream,
    }


def launch_ai_workload(workload_script, duration_sec):
    """Launch AI workload as a subprocess. Returns Popen object."""
    env = os.environ.copy()
    proc = subprocess.Popen(
        ["python3", workload_script, str(duration_sec)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def measure_with_workload(ctx, apply_channel, No, workload_name):
    """Run L1 pipeline with an optional background AI workload."""
    workload_script = WORKLOADS[workload_name]
    ai_proc = None

    if workload_script:
        # Estimate total time needed
        estimated_duration = AI_WARMUP_TIME + (NUM_WARMUP + NUM_ITERATIONS) * 0.01 + 30
        print(f"  Launching {workload_name} background workload...")
        ai_proc = launch_ai_workload(workload_script, int(estimated_duration))
        print(f"  Waiting {AI_WARMUP_TIME}s for AI workload to warm up...")
        time.sleep(AI_WARMUP_TIME)

    tti_tracker = LatencyTracker(f"TTI_{workload_name}")
    rx_tracker = LatencyTracker(f"RX_{workload_name}")
    tx_tracker = LatencyTracker(f"TX_{workload_name}")
    timer = CudaTimer()

    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        is_measuring = i >= NUM_WARMUP
        slot_number = i % NUM_SLOTS_PER_FRAME

        tb_input_np = random_tb(
            mod_order=ctx["mod_order"], code_rate=ctx["code_rate"],
            dmrs_syms=DMRS_SYMS, num_prbs=NUM_PRBS,
            start_sym=START_SYM, num_symbols=NUM_SYMBOLS, num_layers=LAYERS
        )
        tb_input = cp.array(tb_input_np, dtype=cp.uint8, order="F")

        tti_start = time.perf_counter()

        # TX
        timer.start()
        tx_tensor = ctx["tx_pipeline"](
            slot=slot_number, tb_inputs=[tb_input], config=[ctx["pdsch_config"]]
        )
        timer.stop()
        if is_measuring:
            tx_tracker.record(timer.elapsed_ms())

        # Channel
        rx_tensor = apply_channel(tx_tensor, No)

        # RX
        timer.start()
        tb_crcs, tbs = ctx["rx_pipeline"](
            slot=slot_number, rx_slot=rx_tensor, config=ctx["pusch_configs"]
        )
        timer.stop()
        if is_measuring:
            rx_tracker.record(timer.elapsed_ms())

        tti_end = time.perf_counter()
        if is_measuring:
            tti_tracker.record((tti_end - tti_start) * 1000)

    # Stop AI workload
    if ai_proc:
        ai_proc.terminate()
        try:
            ai_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            ai_proc.kill()
        stdout = ai_proc.stdout.read().decode()
        if stdout.strip():
            print(f"  [{workload_name} output]: {stdout.strip().split(chr(10))[-1]}")

    return {
        "workload": workload_name,
        "tti": tti_tracker.stats(),
        "tx": tx_tracker.stats(),
        "rx": rx_tracker.stats(),
        "tti_deadline_miss_1ms": tti_tracker.deadline_miss_rate(1.0),
        "raw_tti": tti_tracker.latencies,
        "raw_rx": rx_tracker.latencies,
    }


def run_ai_interference():
    """Run AI interference experiment."""
    print("=" * 60)
    print("Experiment 2: AI Interference Mode")
    print("=" * 60)
    print(f"Workloads: {list(WORKLOADS.keys())}")
    print(f"Iterations per workload: {NUM_ITERATIONS} (warmup: {NUM_WARMUP})")
    print()

    print("[1/3] Setting up channel model...")
    apply_channel = setup_channel()

    print("[2/3] Creating L1 pipelines...")
    ctx = setup_pipelines()
    No = pow(10.0, -ESNO_DB / 10.0)

    print("[3/3] Running measurements...")
    all_results = []

    for workload_name in WORKLOADS:
        print(f"\n--- Workload: {workload_name} ---")
        result = measure_with_workload(ctx, apply_channel, No, workload_name)
        all_results.append(result)

        tti = result["tti"]
        rx = result["rx"]
        print(f"  TTI:  mean={tti['mean_ms']:.3f}ms  p99={tti['p99_ms']:.3f}ms  "
              f"jitter={tti['jitter']:.4f}")
        print(f"  RX:   mean={rx['mean_ms']:.3f}ms  p99={rx['p99_ms']:.3f}ms")
        print(f"  Deadline miss (>1ms): {result['tti_deadline_miss_1ms']*100:.1f}%")

    # Summary table
    baseline = all_results[0]["tti"]["mean_ms"]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Workload':>12} | {'TTI mean':>10} | {'TTI P99':>10} | {'RX mean':>10} | "
          f"{'RX P99':>10} | {'Jitter':>8} | {'Miss%':>6} | {'Slowdown':>9}")
    print("-" * 95)
    for r in all_results:
        tti, rx = r["tti"], r["rx"]
        slowdown = tti["mean_ms"] / baseline if baseline > 0 else 0
        print(f"{r['workload']:>12} | {tti['mean_ms']:>8.3f}ms | "
              f"{tti['p99_ms']:>8.3f}ms | {rx['mean_ms']:>8.3f}ms | "
              f"{rx['p99_ms']:>8.3f}ms | {tti['jitter']:>8.4f} | "
              f"{r['tti_deadline_miss_1ms']*100:>5.1f}% | {slowdown:>8.2f}x")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"exp2_ai_interference_{timestamp}.json")

    output = {
        "experiment": "exp2_ai_interference",
        "timestamp": timestamp,
        "config": {
            "workloads": list(WORKLOADS.keys()),
            "num_warmup": NUM_WARMUP,
            "num_iterations": NUM_ITERATIONS,
            "ai_warmup_time": AI_WARMUP_TIME,
            "num_prbs": NUM_PRBS,
            "mcs_index": MCS_INDEX,
            "esno_db": ESNO_DB,
        },
        "results": [{k: v for k, v in r.items() if not k.startswith("raw_")} for r in all_results],
        "raw_latencies": {r["workload"]: r["raw_tti"] for r in all_results},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    run_ai_interference()
