"""
Phase 0: GPU Interference Experiment (MPS + Multiprocessing)
=============================================================
Correctly measures L1 vs AI interference using separate OS processes
sharing the same GPU via NVIDIA MPS — matching real AI-RAN architecture.

Process A (main): cuPHY L1 pipeline + latency measurement
Process B (child): AI workload (ResNet-50 / GPT-2 / HBM stress)

Usage:
  # MPS must be started BEFORE this script (see job script)
  python3 exp_phase0.py
"""
import os
import sys
import gc
import json
import time
import signal
import datetime
import multiprocessing as mp

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HOME"] = "/pscratch/sd/s/sgkim/kcj/AI-RAN/datasets/models"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ============================================================
# Configuration
# ============================================================
NUM_WARMUP = 10
NUM_ITERATIONS = 100
RESULTS_DIR = "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results"
ESNO_DB = 10.0

# 5G NR parameters
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


# ============================================================
# AI Workload functions (run in separate PROCESS, not thread)
# ============================================================
def ai_worker_resnet50(ready_event, stop_event, batch_size=32):
    """ResNet-50 inference loop in a separate process."""
    import torch
    import torchvision.models as models

    torch.cuda.set_device(0)
    model = models.resnet50(weights=None).cuda().eval()
    dummy = torch.randn(batch_size, 3, 224, 224, device="cuda")

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    torch.cuda.synchronize()

    ready_event.set()  # Signal main process we're ready
    count = 0
    with torch.no_grad():
        while not stop_event.is_set():
            model(dummy)
            count += 1
    print(f"    [ResNet-50 bs={batch_size}] {count} iters completed", flush=True)


def ai_worker_gpt2(ready_event, stop_event, batch_size=4, seq_len=512):
    """GPT-2 inference loop in a separate process."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2Config

    torch.cuda.set_device(0)
    config = GPT2Config()
    model = GPT2LMHeadModel(config).cuda().eval()
    dummy = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")

    with torch.no_grad():
        for _ in range(5):
            model(dummy)
    torch.cuda.synchronize()

    ready_event.set()
    count = 0
    with torch.no_grad():
        while not stop_event.is_set():
            model(dummy)
            count += 1
    print(f"    [GPT-2 bs={batch_size}] {count} iters completed", flush=True)


def ai_worker_hbm_stress(ready_event, stop_event, size_gb=8.0):
    """HBM bandwidth stress in a separate process."""
    import torch

    torch.cuda.set_device(0)
    num_elements = int(size_gb * 1e9 / 4)
    src = torch.randn(num_elements, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    torch.cuda.synchronize()

    ready_event.set()
    count = 0
    while not stop_event.is_set():
        dst.copy_(src)
        src.copy_(dst)
        count += 1
    print(f"    [HBM_Stress {size_gb}GB] {count} copy iters completed", flush=True)


# ============================================================
# L1 Pipeline (runs in main process)
# ============================================================
def setup_l1():
    """Set up channel + cuPHY pipelines. Returns (apply_channel, ctx)."""
    import numpy as np
    import cupy as cp
    import tensorflow as tf
    import sionna

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
    from aerial.phy5g.ldpc import get_mcs, get_tb_size
    from aerial.util.cuda import get_cuda_stream
    from aerial.pycuphy.types import PuschLdpcKernelLaunch

    # Channel
    resource_grid = sionna.phy.ofdm.ResourceGrid(
        num_ofdm_symbols=NUM_OFDM_SYMBOLS, fft_size=FFT_SIZE,
        subcarrier_spacing=SUBCARRIER_SPACING, num_tx=NUM_TX_ANT,
        num_streams_per_tx=1, cyclic_prefix_length=CYCLIC_PREFIX_LENGTH,
        num_guard_carriers=NUM_GUARD_SUBCARRIERS, dc_null=False,
    )
    rg_mapper = sionna.phy.ofdm.ResourceGridMapper(resource_grid)
    rm_guard = sionna.phy.ofdm.RemoveNulledSubcarriers(resource_grid)
    ch_model = sionna.phy.channel.RayleighBlockFading(
        num_rx=1, num_rx_ant=NUM_RX_ANT, num_tx=1, num_tx_ant=NUM_TX_ANT
    )
    channel = sionna.phy.channel.OFDMChannel(
        ch_model, resource_grid, add_awgn=True, normalize_channel=True, return_channel=False
    )

    def apply_channel(tx_tensor, No):
        tx_t = tf.experimental.dlpack.from_dlpack(tx_tensor.toDlpack())
        tx_t = tf.transpose(tx_t, (2, 1, 0))
        tx_t = tf.reshape(tx_t, (NUM_TX_ANT, -1))[None, :, None, :]
        tx_t = rg_mapper(tx_t)
        rx_t = channel(tx_t, No)
        rx_t = rm_guard(rx_t)
        rx_t = rx_t[0, 0]
        rx_t = tf.transpose(rx_t, (2, 1, 0))
        rx_t = tf.experimental.dlpack.to_dlpack(rx_t)
        return cp.from_dlpack(rx_t)

    # Pipelines
    cuda_stream = get_cuda_stream()
    mod_order, code_rate = get_mcs(MCS_INDEX, MCS_TABLE + 1)
    tb_size = get_tb_size(
        mod_order=mod_order, code_rate=code_rate, dmrs_syms=DMRS_SYMS,
        num_prbs=NUM_PRBS, start_sym=START_SYM, num_symbols=NUM_SYMBOLS, num_layers=LAYERS
    )

    pdsch_cw = PdschCwConfig(mcs_table=MCS_TABLE, mcs_index=MCS_INDEX,
                              code_rate=int(code_rate * 10), mod_order=mod_order)
    pdsch_ue = PdschUeConfig(cw_configs=[pdsch_cw], scid=SCID, dmrs_scrm_id=DMRS_SCRM_ID,
                              layers=LAYERS, dmrs_ports=DMRS_PORTS, rnti=RNTI, data_scid=DATA_SCID)
    pdsch_cfg = PdschConfig(ue_configs=[pdsch_ue],
                             num_dmrs_cdm_grps_no_data=NUM_DMRS_CDM_GRPS_NO_DATA,
                             start_prb=START_PRB, num_prbs=NUM_PRBS,
                             dmrs_syms=DMRS_SYMS, start_sym=START_SYM, num_symbols=NUM_SYMBOLS)
    tx_cfg = AerialPdschTxConfig(cell_id=CELL_ID, num_tx_ant=NUM_TX_ANT)
    tx_pipe = PdschTxPipelineFactory().create(tx_cfg, cuda_stream)

    pusch_ue = PuschUeConfig(scid=SCID, layers=LAYERS, dmrs_ports=DMRS_PORTS, rnti=RNTI,
                              data_scid=DATA_SCID, mcs_table=MCS_TABLE, mcs_index=MCS_INDEX,
                              code_rate=int(code_rate * 10), mod_order=mod_order,
                              tb_size=tb_size // 8)
    pusch_cfgs = [PuschConfig(ue_configs=[pusch_ue],
                               num_dmrs_cdm_grps_no_data=NUM_DMRS_CDM_GRPS_NO_DATA,
                               dmrs_scrm_id=DMRS_SCRM_ID, start_prb=START_PRB,
                               num_prbs=NUM_PRBS, dmrs_syms=DMRS_SYMS,
                               dmrs_max_len=DMRS_MAX_LEN, dmrs_add_ln_pos=DMRS_ADD_LN_POS,
                               start_sym=START_SYM, num_symbols=NUM_SYMBOLS)]
    rx_cfg = AerialPuschRxConfig(cell_id=CELL_ID, num_rx_ant=NUM_RX_ANT,
                                  enable_pusch_tdi=0, eq_coeff_algo=1,
                                  ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL)
    rx_pipe = PuschRxPipelineFactory().create(rx_cfg, cuda_stream)

    return apply_channel, {
        "tx_pipe": tx_pipe, "rx_pipe": rx_pipe,
        "pdsch_cfg": pdsch_cfg, "pusch_cfgs": pusch_cfgs,
        "mod_order": mod_order, "code_rate": code_rate,
    }


def measure_l1(apply_channel, ctx, No, label):
    """Run L1 pipeline and return latency stats."""
    import cupy as cp
    from aerial.phy5g.ldpc import random_tb

    sys.path.insert(0, os.path.dirname(__file__))
    from utils.timing import CudaTimer, LatencyTracker

    tx_tracker = LatencyTracker(f"TX_{label}")
    rx_tracker = LatencyTracker(f"RX_{label}")
    timer = CudaTimer()

    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        measuring = i >= NUM_WARMUP
        slot = i % NUM_SLOTS_PER_FRAME

        tb_np = random_tb(mod_order=ctx["mod_order"], code_rate=ctx["code_rate"],
                          dmrs_syms=DMRS_SYMS, num_prbs=NUM_PRBS,
                          start_sym=START_SYM, num_symbols=NUM_SYMBOLS, num_layers=LAYERS)
        tb = cp.array(tb_np, dtype=cp.uint8, order="F")

        # TX
        timer.start()
        tx = ctx["tx_pipe"](slot=slot, tb_inputs=[tb], config=[ctx["pdsch_cfg"]])
        timer.stop()
        if measuring:
            tx_tracker.record(timer.elapsed_ms())

        # Channel
        rx = apply_channel(tx, No)

        # RX — this is the critical L1 measurement
        timer.start()
        ctx["rx_pipe"](slot=slot, rx_slot=rx, config=ctx["pusch_cfgs"])
        timer.stop()
        if measuring:
            rx_tracker.record(timer.elapsed_ms())

    return {
        "label": label,
        "tx": tx_tracker.stats(),
        "rx": rx_tracker.stats(),
        "rx_miss_1ms": rx_tracker.deadline_miss_rate(1.0),
        "raw_rx": rx_tracker.latencies,
    }


# ============================================================
# Main
# ============================================================
def run_mode(apply_channel, ctx, No, label, worker_fn=None, worker_kwargs=None):
    """Run one experiment mode with optional background AI process."""
    print(f"\n=== {label} ===", flush=True)

    proc = None
    ready_event = None
    stop_event = None

    if worker_fn:
        ready_event = mp.Event()
        stop_event = mp.Event()
        kwargs = worker_kwargs or {}
        kwargs["ready_event"] = ready_event
        kwargs["stop_event"] = stop_event
        proc = mp.Process(target=worker_fn, kwargs=kwargs)
        proc.start()
        print(f"  Waiting for AI worker to warm up...", flush=True)
        ready_event.wait(timeout=120)
        print(f"  AI worker ready. Measuring L1...", flush=True)

    result = measure_l1(apply_channel, ctx, No, label)

    if proc:
        stop_event.set()
        proc.join(timeout=15)
        if proc.is_alive():
            proc.kill()

    rx = result["rx"]
    print(f"  RX: mean={rx['mean_ms']:.3f}ms  p95={rx['p95_ms']:.3f}ms  "
          f"p99={rx['p99_ms']:.3f}ms  jitter={rx['jitter']:.4f}  "
          f"miss(>1ms)={result['rx_miss_1ms']*100:.1f}%", flush=True)
    return result


def main():
    import torch

    print("=" * 70)
    print("Phase 0: GPU Interference Experiment (MPS + Multiprocessing)")
    print("=" * 70)

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {props.name}")
    print(f"HBM: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
    print(f"Config: warmup={NUM_WARMUP}, iterations={NUM_ITERATIONS}")

    print("\n[1/2] Setting up L1 pipelines + channel...")
    apply_channel, ctx = setup_l1()
    No = pow(10.0, -ESNO_DB / 10.0)

    # Experiment modes
    modes = [
        ("A1_baseline", None, None),
        ("B1_hbm_stress_8GB", ai_worker_hbm_stress, {"size_gb": 8.0}),
        ("B2_hbm_stress_16GB", ai_worker_hbm_stress, {"size_gb": 16.0}),
        ("C1_resnet50_bs16", ai_worker_resnet50, {"batch_size": 16}),
        ("C2_resnet50_bs32", ai_worker_resnet50, {"batch_size": 32}),
        ("C3_gpt2_bs4", ai_worker_gpt2, {"batch_size": 4}),
        ("A2_baseline_final", None, None),
    ]

    print(f"[2/2] Running {len(modes)} modes...", flush=True)
    all_results = []

    for label, worker_fn, worker_kwargs in modes:
        result = run_mode(apply_channel, ctx, No, label, worker_fn, worker_kwargs)
        all_results.append(result)
        gc.collect()
        time.sleep(2)

    # Summary
    baseline_rx = all_results[0]["rx"]["mean_ms"]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<24} | {'RX mean':>8} | {'RX P95':>8} | {'RX P99':>8} | "
          f"{'Jitter':>7} | {'Miss%':>6} | {'vs base':>8}")
    print("-" * 85)
    for r in all_results:
        rx = r["rx"]
        ratio = rx["mean_ms"] / baseline_rx if baseline_rx > 0 else 0
        print(f"{r['label']:<24} | {rx['mean_ms']:>6.3f}ms | {rx['p95_ms']:>6.3f}ms | "
              f"{rx['p99_ms']:>6.3f}ms | {rx['jitter']:>7.4f} | "
              f"{r['rx_miss_1ms']*100:>5.1f}% | {ratio:>7.2f}x")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"exp_phase0_{ts}.json")
    output = {
        "experiment": "phase0_mps",
        "timestamp": ts,
        "gpu": props.name,
        "hbm_total_gb": round(total / 1e9, 1),
        "config": {"num_warmup": NUM_WARMUP, "num_iterations": NUM_ITERATIONS,
                    "num_prbs": NUM_PRBS, "mcs_index": MCS_INDEX, "esno_db": ESNO_DB},
        "results": [{k: v for k, v in r.items() if k != "raw_rx"} for r in all_results],
        "raw_latencies": {r["label"]: r["raw_rx"] for r in all_results},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
