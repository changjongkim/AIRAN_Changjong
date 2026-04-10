"""
Phase 0: Realistic GPU Interference Experiment
================================================
Multi-cell L1 (multiple cuPHY pipeline instances) to simulate realistic
GPU utilization, then measure interference from actual AI workloads.

Scales L1 load by increasing:
  - Number of cells (each cell = separate cuPHY pipeline instance)
  - MIMO antenna count (4x4, 8x4)
  - Higher MCS (more compute per PRB)

Then measures interference from realistic AI workloads (ResNet-50, GPT-2)
on the same GPU vs different GPU.

1 Node, 4x A100-SXM4-40GB, MPS enabled.
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
L1_GPU = 0

# NR parameters
NUM_OFDM_SYMBOLS = 14
FFT_SIZE = 4096
CYCLIC_PREFIX_LENGTH = 288
SUBCARRIER_SPACING = 30e3
NUM_GUARD_SUBCARRIERS = (410, 410)
NUM_SLOTS_PER_FRAME = 20
NUM_PRBS = 273
START_PRB = 0
START_SYM = 2
NUM_SYMBOLS = 12
DMRS_SCRM_ID = 41
DMRS_SYMS = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
DMRS_MAX_LEN = 1
DMRS_ADD_LN_POS = 0
NUM_DMRS_CDM_GRPS_NO_DATA = 2
RNTI = 1234
SCID = 0
DATA_SCID = 0


# ============================================================
# AI Workers (separate processes)
# ============================================================
def _setup_child_env():
    """Ensure child process has correct PYTHONPATH."""
    import sys
    site = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
    if site not in sys.path:
        sys.path.insert(0, site)


def worker_resnet50(ready_event, stop_event, gpu_id, batch_size=32):
    _setup_child_env()
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


def worker_gpt2(ready_event, stop_event, gpu_id, batch_size=4, seq_len=512):
    _setup_child_env()
    import torch
    from transformers import GPT2LMHeadModel, GPT2Config
    torch.cuda.set_device(gpu_id)
    config = GPT2Config()
    model = GPT2LMHeadModel(config).to(f"cuda:{gpu_id}").eval()
    dummy = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=f"cuda:{gpu_id}")
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
    torch.cuda.synchronize(gpu_id)
    ready_event.set()
    count = 0
    with torch.no_grad():
        while not stop_event.is_set():
            model(dummy)
            count += 1
    print(f"    [GPT2 GPU{gpu_id} bs={batch_size}] {count} iters", flush=True)


def worker_hbm_stress(ready_event, stop_event, gpu_id, size_gb=8.0):
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


# ============================================================
# L1 Pipeline setup — supports multi-cell, configurable MIMO
# ============================================================
def create_l1_config(num_tx_ant, num_rx_ant, num_cells, mcs_index, mcs_table):
    """Create L1 pipeline configuration for given antenna/cell count."""
    # Don't set CUDA_VISIBLE_DEVICES — child processes need access to all GPUs.
    # Instead, pin cuPHY/TF to GPU0 via TF config and CuPy device setting.
    import numpy as np
    import cupy as cp
    import tensorflow as tf
    import sionna

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Pin TF to GPU0 only
    tf.config.set_visible_devices(gpus[L1_GPU], "GPU")
    cp.cuda.Device(L1_GPU).use()

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

    # Channel (one per config, reused across cells)
    resource_grid = sionna.phy.ofdm.ResourceGrid(
        num_ofdm_symbols=NUM_OFDM_SYMBOLS, fft_size=FFT_SIZE,
        subcarrier_spacing=SUBCARRIER_SPACING, num_tx=num_tx_ant,
        num_streams_per_tx=1, cyclic_prefix_length=CYCLIC_PREFIX_LENGTH,
        num_guard_carriers=NUM_GUARD_SUBCARRIERS, dc_null=False,
    )
    rg_mapper = sionna.phy.ofdm.ResourceGridMapper(resource_grid)
    rm_guard = sionna.phy.ofdm.RemoveNulledSubcarriers(resource_grid)
    ch_model = sionna.phy.channel.RayleighBlockFading(
        num_rx=1, num_rx_ant=num_rx_ant, num_tx=1, num_tx_ant=num_tx_ant
    )
    channel = sionna.phy.channel.OFDMChannel(
        ch_model, resource_grid, add_awgn=True, normalize_channel=True, return_channel=False
    )

    def apply_channel(tx_tensor, No):
        tx_t = tf.experimental.dlpack.from_dlpack(tx_tensor.toDlpack())
        tx_t = tf.transpose(tx_t, (2, 1, 0))
        tx_t = tf.reshape(tx_t, (num_tx_ant, -1))[None, :, None, :]
        tx_t = rg_mapper(tx_t)
        rx_t = channel(tx_t, No)
        rx_t = rm_guard(rx_t)
        rx_t = rx_t[0, 0]
        rx_t = tf.transpose(rx_t, (2, 1, 0))
        rx_t = tf.experimental.dlpack.to_dlpack(rx_t)
        return cp.from_dlpack(rx_t)

    cuda_stream = get_cuda_stream()
    mod_order, code_rate = get_mcs(mcs_index, mcs_table + 1)
    layers = min(num_tx_ant, num_rx_ant)  # layers = min(tx, rx)
    # For simplicity use 1 layer to avoid precoding complexity
    layers = 1
    dmrs_ports = 1
    tb_size = get_tb_size(
        mod_order=mod_order, code_rate=code_rate, dmrs_syms=DMRS_SYMS,
        num_prbs=NUM_PRBS, start_sym=START_SYM, num_symbols=NUM_SYMBOLS, num_layers=layers
    )

    # Create N cell pipelines
    cells = []
    for cell_idx in range(num_cells):
        cell_id = 41 + cell_idx

        pdsch_cw = PdschCwConfig(mcs_table=mcs_table, mcs_index=mcs_index,
                                  code_rate=int(code_rate * 10), mod_order=mod_order)
        pdsch_ue = PdschUeConfig(cw_configs=[pdsch_cw], scid=SCID, dmrs_scrm_id=DMRS_SCRM_ID,
                                  layers=layers, dmrs_ports=dmrs_ports, rnti=RNTI, data_scid=DATA_SCID)
        pdsch_cfg = PdschConfig(ue_configs=[pdsch_ue],
                                 num_dmrs_cdm_grps_no_data=NUM_DMRS_CDM_GRPS_NO_DATA,
                                 start_prb=START_PRB, num_prbs=NUM_PRBS,
                                 dmrs_syms=DMRS_SYMS, start_sym=START_SYM, num_symbols=NUM_SYMBOLS)
        tx_cfg = AerialPdschTxConfig(cell_id=cell_id, num_tx_ant=num_tx_ant)
        tx_pipe = PdschTxPipelineFactory().create(tx_cfg, cuda_stream)

        pusch_ue = PuschUeConfig(scid=SCID, layers=layers, dmrs_ports=dmrs_ports, rnti=RNTI,
                                  data_scid=DATA_SCID, mcs_table=mcs_table, mcs_index=mcs_index,
                                  code_rate=int(code_rate * 10), mod_order=mod_order,
                                  tb_size=tb_size // 8)
        pusch_cfgs = [PuschConfig(ue_configs=[pusch_ue],
                                   num_dmrs_cdm_grps_no_data=NUM_DMRS_CDM_GRPS_NO_DATA,
                                   dmrs_scrm_id=DMRS_SCRM_ID, start_prb=START_PRB,
                                   num_prbs=NUM_PRBS, dmrs_syms=DMRS_SYMS,
                                   dmrs_max_len=DMRS_MAX_LEN, dmrs_add_ln_pos=DMRS_ADD_LN_POS,
                                   start_sym=START_SYM, num_symbols=NUM_SYMBOLS)]
        rx_cfg = AerialPuschRxConfig(cell_id=cell_id, num_rx_ant=num_rx_ant,
                                      enable_pusch_tdi=0, eq_coeff_algo=1,
                                      ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL)
        rx_pipe = PuschRxPipelineFactory().create(rx_cfg, cuda_stream)

        cells.append({
            "tx_pipe": tx_pipe, "rx_pipe": rx_pipe,
            "pdsch_cfg": pdsch_cfg, "pusch_cfgs": pusch_cfgs,
        })

    import torch
    free, total = torch.cuda.mem_get_info(0)
    print(f"  L1 config: {num_cells} cells, {num_tx_ant}T{num_rx_ant}R MIMO, MCS {mcs_index}")
    print(f"  GPU0 HBM after L1 setup: {(total-free)/1e9:.1f}/{total/1e9:.1f} GB "
          f"({(total-free)/total*100:.0f}%)", flush=True)

    return apply_channel, cells, {
        "mod_order": mod_order, "code_rate": code_rate, "layers": layers,
    }


def measure_multicell_l1(apply_channel, cells, params, No, label):
    """Run all cells per iteration, measure total processing time."""
    import cupy as cp
    from aerial.phy5g.ldpc import random_tb
    sys.path.insert(0, os.path.dirname(__file__))
    from utils.timing import CudaTimer, LatencyTracker

    rx_tracker = LatencyTracker(f"RX_{label}")
    total_tracker = LatencyTracker(f"Total_{label}")
    timer = CudaTimer()

    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        measuring = i >= NUM_WARMUP
        slot = i % NUM_SLOTS_PER_FRAME

        tb_np = random_tb(mod_order=params["mod_order"], code_rate=params["code_rate"],
                          dmrs_syms=DMRS_SYMS, num_prbs=NUM_PRBS,
                          start_sym=START_SYM, num_symbols=NUM_SYMBOLS,
                          num_layers=params["layers"])
        tb = cp.array(tb_np, dtype=cp.uint8, order="F")

        total_start = time.perf_counter()
        rx_total_ms = 0

        # Process all cells sequentially (like real multi-cell gNB)
        for cell in cells:
            tx = cell["tx_pipe"](slot=slot, tb_inputs=[tb], config=[cell["pdsch_cfg"]])
            rx_tensor = apply_channel(tx, No)

            timer.start()
            cell["rx_pipe"](slot=slot, rx_slot=rx_tensor, config=cell["pusch_cfgs"])
            timer.stop()
            rx_total_ms += timer.elapsed_ms()

        total_end = time.perf_counter()

        if measuring:
            rx_tracker.record(rx_total_ms)
            total_tracker.record((total_end - total_start) * 1000)

    return {
        "label": label,
        "rx": rx_tracker.stats(),
        "total": total_tracker.stats(),
        "rx_miss_1ms": rx_tracker.deadline_miss_rate(1.0),
        "raw_rx": rx_tracker.latencies,
    }


# ============================================================
# Mode runner
# ============================================================
def run_mode(apply_channel, cells, params, No, label, workers_spec=None):
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
            print(f"  Started worker on GPU{kwargs.get('gpu_id','?')}...", flush=True)
            ready.wait(timeout=120)
        print(f"  All {len(procs)} workers ready.", flush=True)

    try:
        result = measure_multicell_l1(apply_channel, cells, params, No, label)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        result = {"label": label, "rx": {"mean_ms": -1, "p95_ms": -1, "p99_ms": -1,
                  "jitter": -1}, "rx_miss_1ms": -1, "raw_rx": []}

    for stop in stop_events:
        stop.set()
    for p in procs:
        p.join(timeout=15)
        if p.is_alive():
            p.kill()

    rx = result["rx"]
    if rx["mean_ms"] > 0:
        print(f"  RX({len(cells)}cells): mean={rx['mean_ms']:.3f}ms  "
              f"p95={rx['p95_ms']:.3f}ms  p99={rx['p99_ms']:.3f}ms  "
              f"jitter={rx['jitter']:.4f}  miss(>1ms)={result['rx_miss_1ms']*100:.1f}%",
              flush=True)
    return result


# ============================================================
# Main
# ============================================================
def main():
    import torch

    num_gpus = torch.cuda.device_count()
    print("=" * 70)
    print("Phase 0: Realistic Multi-Cell GPU Interference Experiment")
    print("=" * 70)
    print(f"GPUs: {num_gpus}x {torch.cuda.get_device_properties(0).name}")
    for i in range(num_gpus):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU{i}: {free/1e9:.1f}/{total/1e9:.1f} GB free")
    print(f"Config: warmup={NUM_WARMUP}, iterations={NUM_ITERATIONS}")

    # L1 configurations to test: (num_tx, num_rx, num_cells, mcs_index, mcs_table)
    l1_configs = [
        (1, 2, 1, 2, 0, "1cell_2x1_mcs2"),      # Tiny (current)
        (2, 4, 4, 10, 0, "4cell_4x2_mcs10"),     # Medium
        (4, 8, 8, 15, 0, "8cell_8x4_mcs15"),     # Realistic
    ]

    No = pow(10.0, -ESNO_DB / 10.0)
    all_results = []

    for num_tx, num_rx, num_cells, mcs_idx, mcs_tbl, config_name in l1_configs:
        print(f"\n{'='*70}")
        print(f"L1 CONFIG: {config_name}")
        print(f"{'='*70}", flush=True)

        try:
            apply_channel, cells, params = create_l1_config(
                num_tx, num_rx, num_cells, mcs_idx, mcs_tbl)
        except Exception as e:
            print(f"  SETUP FAILED: {e}", flush=True)
            continue

        # Modes for this config
        modes = [
            (f"{config_name}_baseline", None),
            (f"{config_name}_hbm8G_same", [(worker_hbm_stress, {"gpu_id": 0, "size_gb": 8.0})]),
            (f"{config_name}_hbm8G_diff", [(worker_hbm_stress, {"gpu_id": 1, "size_gb": 8.0})]),
            (f"{config_name}_resnet_same", [(worker_resnet50, {"gpu_id": 0, "batch_size": 32})]),
            (f"{config_name}_resnet_diff", [(worker_resnet50, {"gpu_id": 1, "batch_size": 32})]),
            (f"{config_name}_gpt2_same", [(worker_gpt2, {"gpu_id": 0, "batch_size": 4})]),
            (f"{config_name}_gpt2_diff", [(worker_gpt2, {"gpu_id": 1, "batch_size": 4})]),
        ]

        for label, workers_spec in modes:
            result = run_mode(apply_channel, cells, params, No, label, workers_spec)
            result["config_name"] = config_name
            result["num_cells"] = num_cells
            result["mimo"] = f"{num_tx}T{num_rx}R"
            result["mcs"] = mcs_idx
            all_results.append(result)
            gc.collect()
            time.sleep(2)

        # Cleanup cell pipelines
        del cells, apply_channel
        gc.collect()
        import torch as _t
        _t.cuda.empty_cache()
        time.sleep(3)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'Mode':<14} | {'RX mean':>8} | {'RX P99':>8} | "
          f"{'Miss%':>6} | {'vs base':>8}")
    print("-" * 80)

    # Group by config
    baselines = {}
    for r in all_results:
        rx = r["rx"]
        if rx["mean_ms"] < 0:
            print(f"{r.get('config_name','?'):<20} {r['label'].split('_')[-1] if '_' in r['label'] else r['label']:<14} | {'CRASH':>8} |")
            continue
        cn = r.get("config_name", "")
        short = r["label"].replace(cn + "_", "")
        if "baseline" in short:
            baselines[cn] = rx["mean_ms"]
        base = baselines.get(cn, rx["mean_ms"])
        ratio = rx["mean_ms"] / base if base > 0 else 0
        print(f"{cn:<20} {short:<14} | {rx['mean_ms']:>6.3f}ms | "
              f"{rx['p99_ms']:>6.3f}ms | {r['rx_miss_1ms']*100:>5.1f}% | {ratio:>7.2f}x")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"exp_phase0_realistic_{ts}.json")
    output = {
        "experiment": "phase0_realistic",
        "timestamp": ts,
        "num_gpus": num_gpus,
        "config": {"num_warmup": NUM_WARMUP, "num_iterations": NUM_ITERATIONS, "esno_db": ESNO_DB},
        "results": [{k: v for k, v in r.items() if k != "raw_rx"} for r in all_results],
        "raw_latencies": {r["label"]: r.get("raw_rx", []) for r in all_results},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
