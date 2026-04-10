"""
Unified Phase 0 Experiment
===========================
Single script that runs all 3 modes on the same node, same process:
  Mode A: L1 Baseline (no interference)
  Mode B: HBM Bandwidth Stress (active memcpy, not idle zeros)
  Mode C: AI Compute Interference (ResNet-50 / GPT-2 in background threads)

All modes share the same pipeline instance, same GPU, same measurement code.

Usage:
  shifter --image=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb bash -c '
  export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
  export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
  export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH
  python3 /pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/exp_unified.py
  '
"""
import os
import sys
import gc
import json
import time
import threading
import datetime

# Force unbuffered stdout so we can see progress in SLURM output
sys.stdout.reconfigure(line_buffering=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HOME"] = "/pscratch/sd/s/sgkim/kcj/AI-RAN/datasets/models"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import cupy as cp
import torch
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
# Background interference workers (run in threads)
# ============================================================
class InterferenceWorker:
    """Base class for background GPU interference."""

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self._iter_count = 0

    def start(self):
        self._stop.clear()
        self._iter_count = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Wait for worker to warm up
        time.sleep(3)
        print(f"    [{self.name}] running (warmup done)")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        print(f"    [{self.name}] stopped after {self._iter_count} iters")

    def _run(self):
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__


class HBMBandwidthStress(InterferenceWorker):
    """Continuously copies data in HBM to create bandwidth contention."""

    def __init__(self, size_gb=8.0):
        super().__init__()
        self.size_gb = size_gb
        self.num_elements = int(size_gb * 1e9 / 4)  # float32

    def _run(self):
        # Allocate two large tensors, continuously copy between them
        src = torch.randn(self.num_elements, dtype=torch.float32, device="cuda")
        dst = torch.empty_like(src)
        torch.cuda.synchronize()

        while not self._stop.is_set():
            dst.copy_(src)
            src.copy_(dst)
            self._iter_count += 1
            # Don't synchronize every iter — let it pipeline and saturate bandwidth

    @property
    def name(self):
        return f"HBM_BW_Stress_{self.size_gb}GB"


class HBMFillPassive(InterferenceWorker):
    """Just fills HBM without active bandwidth usage (control experiment)."""

    def __init__(self, target_percent=85):
        super().__init__()
        self.target_percent = target_percent

    def _run(self):
        total = torch.cuda.get_device_properties(0).total_memory
        current_free = torch.cuda.mem_get_info()[0]
        target_used = int(total * self.target_percent / 100)
        current_used = total - current_free
        to_alloc = max(0, target_used - current_used)

        tensors = []
        chunk = 256 * 1024 * 1024
        while to_alloc > chunk:
            tensors.append(torch.zeros(chunk // 4, dtype=torch.float32, device="cuda"))
            to_alloc -= chunk
        if to_alloc > 0:
            tensors.append(torch.zeros(to_alloc // 4, dtype=torch.float32, device="cuda"))

        free, total = torch.cuda.mem_get_info()
        print(f"    [HBM_Fill] {(total-free)/1e9:.1f}/{total/1e9:.1f} GB "
              f"({(total-free)/total*100:.0f}%)")

        # Just hold memory, do nothing
        while not self._stop.is_set():
            time.sleep(0.1)
            self._iter_count += 1

        del tensors

    @property
    def name(self):
        return f"HBM_Fill_{self.target_percent}pct"


class ResNet50Worker(InterferenceWorker):
    """ResNet-50 inference in a background thread."""

    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def _run(self):
        import torchvision.models as models
        model = models.resnet50(weights=None).cuda().eval()
        dummy = torch.randn(self.batch_size, 3, 224, 224, device="cuda")

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                model(dummy)
        torch.cuda.synchronize()

        with torch.no_grad():
            while not self._stop.is_set():
                model(dummy)
                self._iter_count += 1

    @property
    def name(self):
        return f"ResNet50_bs{self.batch_size}"


class GPT2Worker(InterferenceWorker):
    """GPT-2 inference in a background thread."""

    def __init__(self, batch_size=4, seq_len=512):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len

    def _run(self):
        from transformers import GPT2LMHeadModel, GPT2Config
        config = GPT2Config()
        model = GPT2LMHeadModel(config).cuda().eval()
        dummy = torch.randint(0, config.vocab_size,
                              (self.batch_size, self.seq_len), device="cuda")

        with torch.no_grad():
            for _ in range(5):
                model(dummy)
        torch.cuda.synchronize()

        with torch.no_grad():
            while not self._stop.is_set():
                model(dummy)
                self._iter_count += 1

    @property
    def name(self):
        return f"GPT2_bs{self.batch_size}"


# ============================================================
# L1 Pipeline setup
# ============================================================
def setup_channel():
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


# ============================================================
# Measurement loop
# ============================================================
def measure_l1(ctx, apply_channel, No, label="baseline"):
    """Run L1 pipeline and collect latency measurements."""
    tx_tracker = LatencyTracker(f"TX_{label}")
    rx_tracker = LatencyTracker(f"RX_{label}")
    tti_tracker = LatencyTracker(f"TTI_{label}")
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

        # RX (this is the critical L1 path)
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

    # HBM snapshot
    free, total = torch.cuda.mem_get_info()
    hbm_used_pct = (total - free) / total * 100

    return {
        "label": label,
        "tx": tx_tracker.stats(),
        "rx": rx_tracker.stats(),
        "tti": tti_tracker.stats(),
        "rx_deadline_miss_1ms": rx_tracker.deadline_miss_rate(1.0),
        "hbm_used_pct": round(hbm_used_pct, 1),
        "raw_rx": rx_tracker.latencies,
        "raw_tx": tx_tracker.latencies,
    }


# ============================================================
# Main experiment
# ============================================================
def run_unified():
    print("=" * 70)
    print("Unified Phase 0 Experiment")
    print("=" * 70)

    # GPU info
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {props.name}")
    print(f"HBM: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
    print(f"Warmup: {NUM_WARMUP}, Iterations: {NUM_ITERATIONS}")
    print()

    print("[1/2] Setting up L1 pipelines + channel...")
    apply_channel = setup_channel()
    ctx = setup_pipelines()
    No = pow(10.0, -ESNO_DB / 10.0)

    # Define experiment modes
    modes = [
        ("A1_baseline", None),
        ("B1_hbm_fill_85pct", HBMFillPassive(target_percent=85)),
        ("B2_hbm_bw_stress_4GB", HBMBandwidthStress(size_gb=4.0)),
        ("B3_hbm_bw_stress_16GB", HBMBandwidthStress(size_gb=16.0)),
        ("C1_resnet50_bs16", ResNet50Worker(batch_size=16)),
        ("C1_resnet50_bs32", ResNet50Worker(batch_size=32)),
        ("C2_gpt2_bs4", GPT2Worker(batch_size=4)),
        ("A2_baseline_final", None),  # Re-measure baseline at the end
    ]

    print(f"[2/2] Running {len(modes)} modes...\n")
    all_results = []

    for mode_name, worker in modes:
        print(f"=== {mode_name} ===")

        # Start interference if any
        if worker:
            worker.start()

        # Measure L1
        result = measure_l1(ctx, apply_channel, No, label=mode_name)
        all_results.append(result)

        # Stop interference
        if worker:
            worker.stop()
            # Cleanup GPU memory
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

        rx = result["rx"]
        print(f"  RX:  mean={rx['mean_ms']:.3f}ms  p95={rx['p95_ms']:.3f}ms  "
              f"p99={rx['p99_ms']:.3f}ms  jitter={rx['jitter']:.4f}")
        print(f"  HBM used: {result['hbm_used_pct']}%  "
              f"RX deadline miss (>1ms): {result['rx_deadline_miss_1ms']*100:.1f}%")
        print()

    # ============================================================
    # Summary
    # ============================================================
    baseline_rx = all_results[0]["rx"]["mean_ms"]

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<28} | {'RX mean':>8} | {'RX P95':>8} | {'RX P99':>8} | "
          f"{'Jitter':>7} | {'Miss%':>6} | {'HBM%':>5} | {'vs base':>8}")
    print("-" * 95)
    for r in all_results:
        rx = r["rx"]
        ratio = rx["mean_ms"] / baseline_rx if baseline_rx > 0 else 0
        print(f"{r['label']:<28} | {rx['mean_ms']:>6.3f}ms | {rx['p95_ms']:>6.3f}ms | "
              f"{rx['p99_ms']:>6.3f}ms | {rx['jitter']:>7.4f} | "
              f"{r['rx_deadline_miss_1ms']*100:>5.1f}% | {r['hbm_used_pct']:>4.0f}% | "
              f"{ratio:>7.2f}x")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"exp_unified_{timestamp}.json")

    output = {
        "experiment": "unified_phase0",
        "timestamp": timestamp,
        "gpu": props.name,
        "hbm_total_gb": round(total / 1e9, 1),
        "config": {
            "num_warmup": NUM_WARMUP,
            "num_iterations": NUM_ITERATIONS,
            "num_prbs": NUM_PRBS,
            "mcs_index": MCS_INDEX,
            "esno_db": ESNO_DB,
        },
        "results": [{k: v for k, v in r.items() if not k.startswith("raw_")}
                     for r in all_results],
        "raw_latencies": {r["label"]: r["raw_rx"] for r in all_results},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    run_unified()
