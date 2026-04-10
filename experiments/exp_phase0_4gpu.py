"""
Phase 0: 4-GPU Interference Experiment
========================================
1 Node, 4x A100 GPU. Compares L1 performance under different GPU sharing configs.

Modes:
  A1: L1 on GPU0 alone (baseline)
  B1: L1 on GPU0 + HBM stress on GPU0 (same GPU — MPS sharing)
  B2: L1 on GPU0 + HBM stress on GPU1 (separate GPU — isolation control)
  B3: L1 on GPU0 + HBM stress on GPU0,1,2,3 (all GPUs busy)
  C1: L1 on GPU0 + ResNet-50 on GPU0 (same GPU — compute interference)
  C2: L1 on GPU0 + ResNet-50 on GPU1 (separate GPU — isolation control)
  C3: L1 on GPU0 + ResNet-50 on GPU0,1,2,3 (all GPUs busy)
  A2: L1 on GPU0 alone (final baseline — consistency check)
"""
import os
import sys
import gc
import json
import time
import datetime
import multiprocessing as mp

sys.stdout.reconfigure(line_buffering=True)

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
L1_GPU = 0  # L1 always runs on GPU 0

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
# AI Workers (separate processes, explicit GPU assignment)
# ============================================================
def worker_hbm_stress(ready_event, stop_event, gpu_id, size_gb=8.0):
    """HBM bandwidth stress on a specific GPU."""
    import torch
    torch.cuda.set_device(gpu_id)
    n = int(size_gb * 1e9 / 4)
    src = torch.randn(n, dtype=torch.float32, device=f"cuda:{gpu_id}")
    dst = torch.empty_like(src)
    torch.cuda.synchronize(gpu_id)
    ready_event.set()
    count = 0
    while not stop_event.is_set():
        dst.copy_(src)
        src.copy_(dst)
        count += 1
    print(f"    [HBM_Stress GPU{gpu_id} {size_gb}GB] {count} iters", flush=True)


def worker_resnet50(ready_event, stop_event, gpu_id, batch_size=32):
    """ResNet-50 inference on a specific GPU."""
    import torch
    import torchvision.models as models
    torch.cuda.set_device(gpu_id)
    model = models.resnet50(weights=None).to(f"cuda:{gpu_id}").eval()
    dummy = torch.randn(batch_size, 3, 224, 224, device=f"cuda:{gpu_id}")
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    torch.cuda.synchronize(gpu_id)
    ready_event.set()
    count = 0
    with torch.no_grad():
        while not stop_event.is_set():
            model(dummy)
            count += 1
    print(f"    [ResNet50 GPU{gpu_id} bs={batch_size}] {count} iters", flush=True)


# ============================================================
# L1 Pipeline (always on GPU 0)
# ============================================================
def setup_l1():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(L1_GPU)

    import numpy as np
    import cupy as cp
    import tensorflow as tf
    import sionna

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

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

        timer.start()
        tx = ctx["tx_pipe"](slot=slot, tb_inputs=[tb], config=[ctx["pdsch_cfg"]])
        timer.stop()
        if measuring:
            tx_tracker.record(timer.elapsed_ms())

        rx = apply_channel(tx, No)

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
# Mode runner
# ============================================================
def run_mode(apply_channel, ctx, No, label, workers_spec=None):
    """
    Run one mode. workers_spec is a list of (worker_fn, kwargs_dict) for each background process.
    Each kwargs_dict must include 'gpu_id'.
    """
    print(f"\n=== {label} ===", flush=True)

    procs = []
    stop_events = []

    if workers_spec:
        for worker_fn, kwargs in workers_spec:
            ready = mp.Event()
            stop = mp.Event()
            kwargs["ready_event"] = ready
            kwargs["stop_event"] = stop
            p = mp.Process(target=worker_fn, kwargs=kwargs)
            p.start()
            procs.append(p)
            stop_events.append(stop)
            gpu_id = kwargs.get("gpu_id", "?")
            print(f"  Started worker on GPU{gpu_id}, waiting for warmup...", flush=True)
            ready.wait(timeout=120)

        print(f"  All {len(procs)} workers ready. Measuring L1 on GPU{L1_GPU}...", flush=True)

    result = measure_l1(apply_channel, ctx, No, label)

    # Stop all workers
    for stop in stop_events:
        stop.set()
    for p in procs:
        p.join(timeout=15)
        if p.is_alive():
            p.kill()

    rx = result["rx"]
    print(f"  RX: mean={rx['mean_ms']:.3f}ms  p95={rx['p95_ms']:.3f}ms  "
          f"p99={rx['p99_ms']:.3f}ms  jitter={rx['jitter']:.4f}  "
          f"miss(>1ms)={result['rx_miss_1ms']*100:.1f}%", flush=True)
    return result


# ============================================================
# Main
# ============================================================
def main():
    import torch

    num_gpus = torch.cuda.device_count()
    print("=" * 70)
    print("Phase 0: 4-GPU Interference Experiment")
    print("=" * 70)
    print(f"GPUs available: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU{i}: {props.name}, {total/1e9:.1f}GB total, {free/1e9:.1f}GB free")
    print(f"L1 pipeline runs on GPU{L1_GPU}")
    print(f"Config: warmup={NUM_WARMUP}, iterations={NUM_ITERATIONS}")

    print("\n[1/2] Setting up L1 on GPU0...")
    apply_channel, ctx = setup_l1()
    No = pow(10.0, -ESNO_DB / 10.0)

    modes = [
        # (label, workers_spec)
        # workers_spec = [(worker_fn, {gpu_id, ...}), ...]

        ("A1_baseline", None),

        # HBM stress: same GPU vs different GPU
        ("B1_hbm8G_sameGPU0",
         [(worker_hbm_stress, {"gpu_id": 0, "size_gb": 8.0})]),

        ("B2_hbm8G_diffGPU1",
         [(worker_hbm_stress, {"gpu_id": 1, "size_gb": 8.0})]),

        ("B3_hbm8G_allGPUs",
         [(worker_hbm_stress, {"gpu_id": i, "size_gb": 8.0}) for i in range(num_gpus)]),

        # ResNet-50: same GPU vs different GPU
        ("C1_resnet_sameGPU0",
         [(worker_resnet50, {"gpu_id": 0, "batch_size": 32})]),

        ("C2_resnet_diffGPU1",
         [(worker_resnet50, {"gpu_id": 1, "batch_size": 32})]),

        ("C3_resnet_allGPUs",
         [(worker_resnet50, {"gpu_id": i, "batch_size": 32}) for i in range(num_gpus)]),

        ("A2_baseline_final", None),
    ]

    print(f"[2/2] Running {len(modes)} modes...", flush=True)
    all_results = []

    for label, workers_spec in modes:
        try:
            result = run_mode(apply_channel, ctx, No, label, workers_spec)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            all_results.append({"label": label, "rx": {"mean_ms": -1, "p95_ms": -1,
                                "p99_ms": -1, "jitter": -1}, "rx_miss_1ms": -1, "raw_rx": []})
        gc.collect()
        time.sleep(3)

    # Summary
    baseline_rx = all_results[0]["rx"]["mean_ms"]
    print("\n" + "=" * 70)
    print("SUMMARY (1 Node, 4 GPU)")
    print("=" * 70)
    print(f"{'Mode':<24} | {'RX mean':>8} | {'RX P95':>8} | {'RX P99':>8} | "
          f"{'Jitter':>7} | {'Miss%':>6} | {'vs base':>8}")
    print("-" * 85)
    for r in all_results:
        rx = r["rx"]
        if rx["mean_ms"] < 0:
            print(f"{r['label']:<24} | {'CRASH':>8} |")
            continue
        ratio = rx["mean_ms"] / baseline_rx if baseline_rx > 0 else 0
        print(f"{r['label']:<24} | {rx['mean_ms']:>6.3f}ms | {rx['p95_ms']:>6.3f}ms | "
              f"{rx['p99_ms']:>6.3f}ms | {rx['jitter']:>7.4f} | "
              f"{r['rx_miss_1ms']*100:>5.1f}% | {ratio:>7.2f}x")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"exp_phase0_4gpu_{ts}.json")

    import torch as _torch
    output = {
        "experiment": "phase0_4gpu",
        "timestamp": ts,
        "num_gpus": num_gpus,
        "gpu": _torch.cuda.get_device_properties(0).name,
        "config": {"num_warmup": NUM_WARMUP, "num_iterations": NUM_ITERATIONS,
                    "num_prbs": NUM_PRBS, "mcs_index": MCS_INDEX, "esno_db": ESNO_DB},
        "results": [{k: v for k, v in r.items() if k != "raw_rx"} for r in all_results],
        "raw_latencies": {r["label"]: r.get("raw_rx", []) for r in all_results},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
