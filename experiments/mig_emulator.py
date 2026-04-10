"""
MIG Emulator using MPS
=======================
Emulates NVIDIA MIG partitioning via MPS controls:
  - CUDA_MPS_ACTIVE_THREAD_PERCENTAGE → SM partitioning
  - torch.cuda.set_per_process_memory_fraction() → Memory partitioning
  - Bandwidth sharing → naturally occurs (same as real MIG)

A100-80GB MIG profiles (reference):
  1g.10gb:  14 SM (14%), 10GB
  2g.20gb:  28 SM (29%), 20GB
  3g.40gb:  42 SM (43%), 40GB
  4g.40gb:  56 SM (57%), 40GB
  7g.80gb:  98 SM (100%), 80GB

Valid 2-way splits: 3g+4g (43%+57%), 1g+2g+4g, etc.
"""
import os
import sys
import gc
import json
import time
import datetime
import subprocess
import multiprocessing as mp

sys.stdout.reconfigure(line_buffering=True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HOME"] = "/pscratch/sd/s/sgkim/kcj/AI-RAN/datasets/models"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"

RESULTS_DIR = "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results"
NUM_WARMUP = 10
NUM_ITERATIONS = 100
ESNO_DB = 10.0

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
# MIG-like partition definitions
# Emulates real A100 MIG profiles
# ============================================================
MIG_PROFILES = {
    # name: (sm_percent, mem_fraction, description)
    "7g.80gb": (100, 1.0,  "Full GPU — no partitioning"),
    "4g.40gb": (57,  0.50, "4/7 SM, 40/80 GB"),
    "3g.40gb": (43,  0.50, "3/7 SM, 40/80 GB"),
    "2g.20gb": (29,  0.25, "2/7 SM, 20/80 GB"),
    "1g.10gb": (14,  0.12, "1/7 SM, 10/80 GB"),
}

# Valid 2-way partition combos (L1_profile, AI_profile)
PARTITION_CONFIGS = [
    # (l1_profile, ai_profile, description)
    ("7g.80gb", None,       "No AI — L1 uses full GPU"),
    ("4g.40gb", "3g.40gb",  "MIG 4g+3g split (57%+43%)"),
    ("3g.40gb", "4g.40gb",  "MIG 3g+4g split (43%+57%)"),
    ("2g.20gb", "4g.40gb",  "MIG 2g+4g split (29%+57%) — L1 constrained"),
    ("1g.10gb", "4g.40gb",  "MIG 1g+4g split (14%+57%) — L1 severely constrained"),
]


# ============================================================
# AI Worker — runs in separate process with SM/memory limits
# ============================================================
def ai_worker(ready_event, stop_event, workload_type, sm_percent, mem_fraction):
    """AI workload with MIG-emulated resource limits."""
    if SITE not in sys.path:
        sys.path.insert(0, SITE)

    import torch
    torch.cuda.set_device(0)

    # Apply memory limit (MIG emulation)
    if mem_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(mem_fraction, 0)

    # Note: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is set via environment
    # before this process starts (in the launcher)

    if workload_type == "resnet":
        import torchvision.models as models
        model = models.resnet50(weights=None).cuda().eval()
        dummy = torch.randn(32, 3, 224, 224, device="cuda")
        with torch.no_grad():
            for _ in range(10):
                model(dummy)
        torch.cuda.synchronize()
        ready_event.set()
        c = 0
        with torch.no_grad():
            while not stop_event.is_set():
                model(dummy)
                c += 1
        print(f"  [AI ResNet SM={sm_percent}% Mem={mem_fraction:.0%}] {c} iters", flush=True)

    elif workload_type == "gpt2":
        from transformers import GPT2LMHeadModel, GPT2Config
        cfg = GPT2Config()
        model = GPT2LMHeadModel(cfg).cuda().eval()
        dummy = torch.randint(0, cfg.vocab_size, (4, 512), device="cuda")
        with torch.no_grad():
            for _ in range(5):
                model(dummy)
        torch.cuda.synchronize()
        ready_event.set()
        c = 0
        with torch.no_grad():
            while not stop_event.is_set():
                model(dummy)
                c += 1
        print(f"  [AI GPT2 SM={sm_percent}% Mem={mem_fraction:.0%}] {c} iters", flush=True)

    elif workload_type == "hbm_stress":
        # Use only allowed memory fraction
        total_mem = torch.cuda.get_device_properties(0).total_memory
        alloc_bytes = int(total_mem * mem_fraction * 0.7)  # 70% of allowed
        n = alloc_bytes // 4
        src = torch.randn(n, dtype=torch.float32, device="cuda")
        dst = torch.empty_like(src)
        torch.cuda.synchronize()
        ready_event.set()
        c = 0
        while not stop_event.is_set():
            dst.copy_(src)
            src.copy_(dst)
            c += 1
        print(f"  [AI HBM SM={sm_percent}% Mem={mem_fraction:.0%} "
              f"size={alloc_bytes/1e9:.1f}GB] {c} iters", flush=True)

    elif workload_type == "none":
        ready_event.set()
        while not stop_event.is_set():
            time.sleep(0.1)


# ============================================================
# L1 Pipeline
# ============================================================
def setup_and_measure_l1(label):
    """Set up L1 pipeline and measure latency. Returns stats dict."""
    import numpy as np
    import cupy as cp
    import tensorflow as tf
    import sionna

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    if len(gpus) > 1:
        tf.config.set_visible_devices(gpus[0], "GPU")
    cp.cuda.Device(0).use()

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

    num_tx, num_rx, mcs_index = 1, 2, 2

    rg = sionna.phy.ofdm.ResourceGrid(
        num_ofdm_symbols=NUM_OFDM_SYMBOLS, fft_size=FFT_SIZE,
        subcarrier_spacing=SUBCARRIER_SPACING, num_tx=num_tx,
        num_streams_per_tx=1, cyclic_prefix_length=CYCLIC_PREFIX_LENGTH,
        num_guard_carriers=NUM_GUARD_SUBCARRIERS, dc_null=False)
    rg_map = sionna.phy.ofdm.ResourceGridMapper(rg)
    rm_g = sionna.phy.ofdm.RemoveNulledSubcarriers(rg)
    ch = sionna.phy.channel.OFDMChannel(
        sionna.phy.channel.RayleighBlockFading(
            num_rx=1, num_rx_ant=num_rx, num_tx=1, num_tx_ant=num_tx),
        rg, add_awgn=True, normalize_channel=True, return_channel=False)

    def apply_channel(tx_t, No):
        t = tf.experimental.dlpack.from_dlpack(tx_t.toDlpack())
        t = tf.transpose(t, (2, 1, 0))
        t = tf.reshape(t, (num_tx, -1))[None, :, None, :]
        t = rg_map(t)
        r = ch(t, No)
        r = rm_g(r)[0, 0]
        r = tf.transpose(r, (2, 1, 0))
        return cp.from_dlpack(tf.experimental.dlpack.to_dlpack(r))

    stream = get_cuda_stream()
    mo, cr = get_mcs(mcs_index, 1)
    tb_size = get_tb_size(mod_order=mo, code_rate=cr, dmrs_syms=DMRS_SYMS,
                          num_prbs=NUM_PRBS, start_sym=START_SYM,
                          num_symbols=NUM_SYMBOLS, num_layers=1)
    No = pow(10.0, -ESNO_DB / 10.0)

    cid = 41
    pcw = PdschCwConfig(mcs_table=0, mcs_index=mcs_index, code_rate=int(cr*10), mod_order=mo)
    pue = PdschUeConfig(cw_configs=[pcw], scid=SCID, dmrs_scrm_id=DMRS_SCRM_ID,
                        layers=1, dmrs_ports=1, rnti=RNTI, data_scid=DATA_SCID)
    pcfg = PdschConfig(ue_configs=[pue], num_dmrs_cdm_grps_no_data=NUM_DMRS_CDM_GRPS_NO_DATA,
                       start_prb=START_PRB, num_prbs=NUM_PRBS, dmrs_syms=DMRS_SYMS,
                       start_sym=START_SYM, num_symbols=NUM_SYMBOLS)
    tx = PdschTxPipelineFactory().create(
        AerialPdschTxConfig(cell_id=cid, num_tx_ant=num_tx), stream)
    uue = PuschUeConfig(scid=SCID, layers=1, dmrs_ports=1, rnti=RNTI,
                        data_scid=DATA_SCID, mcs_table=0, mcs_index=mcs_index,
                        code_rate=int(cr*10), mod_order=mo, tb_size=tb_size//8)
    ucfg = [PuschConfig(ue_configs=[uue], num_dmrs_cdm_grps_no_data=NUM_DMRS_CDM_GRPS_NO_DATA,
                        dmrs_scrm_id=DMRS_SCRM_ID, start_prb=START_PRB, num_prbs=NUM_PRBS,
                        dmrs_syms=DMRS_SYMS, dmrs_max_len=DMRS_MAX_LEN,
                        dmrs_add_ln_pos=DMRS_ADD_LN_POS, start_sym=START_SYM,
                        num_symbols=NUM_SYMBOLS)]
    rx = PuschRxPipelineFactory().create(
        AerialPuschRxConfig(cell_id=cid, num_rx_ant=num_rx, enable_pusch_tdi=0,
                            eq_coeff_algo=1,
                            ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL),
        stream)

    import torch
    free, total = torch.cuda.mem_get_info(0)
    print(f"  L1 HBM: {(total-free)/1e9:.1f}/{total/1e9:.1f}GB", flush=True)

    # Measure
    rx_trk = LatencyTracker(f"RX_{label}")
    tx_trk = LatencyTracker(f"TX_{label}")
    timer = CudaTimer()

    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        m = i >= NUM_WARMUP
        slot = i % NUM_SLOTS_PER_FRAME
        tb_np = random_tb(mod_order=mo, code_rate=cr, dmrs_syms=DMRS_SYMS,
                          num_prbs=NUM_PRBS, start_sym=START_SYM,
                          num_symbols=NUM_SYMBOLS, num_layers=1)
        tb = cp.array(tb_np, dtype=cp.uint8, order="F")

        timer.start()
        tx_t = tx(slot=slot, tb_inputs=[tb], config=[pcfg])
        timer.stop()
        if m:
            tx_trk.record(timer.elapsed_ms())

        rx_t = apply_channel(tx_t, No)

        timer.start()
        rx(slot=slot, rx_slot=rx_t, config=ucfg)
        timer.stop()
        if m:
            rx_trk.record(timer.elapsed_ms())

    return {
        "rx": rx_trk.stats(),
        "tx": tx_trk.stats(),
        "rx_miss_1ms": rx_trk.deadline_miss_rate(1.0),
        "raw_rx": rx_trk.latencies,
    }


# ============================================================
# Run one partition config
# ============================================================
def run_partition(l1_profile, ai_profile, ai_workload, description):
    """Run one MIG-emulated partition configuration."""
    l1_sm, l1_mem, l1_desc = MIG_PROFILES[l1_profile]

    label = f"{l1_profile}"
    if ai_profile:
        ai_sm, ai_mem, ai_desc = MIG_PROFILES[ai_profile]
        label += f"+{ai_profile}_{ai_workload}"
    else:
        ai_sm, ai_mem = 0, 0
        label += "_solo"

    print(f"\n{'='*60}", flush=True)
    print(f"MIG Partition: {label}", flush=True)
    print(f"  L1:  {l1_profile} ({l1_sm}% SM, {l1_mem:.0%} mem)", flush=True)
    if ai_profile:
        print(f"  AI:  {ai_profile} ({ai_sm}% SM, {ai_mem:.0%} mem) — {ai_workload}", flush=True)
    print(f"  {description}", flush=True)
    print(f"{'='*60}", flush=True)

    # Launch AI worker with SM/memory limits
    proc = None
    stop_event = None

    if ai_profile and ai_workload != "none":
        ready = mp.Event()
        stop_event = mp.Event()

        # Set SM limit for AI worker via environment
        env = os.environ.copy()
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(ai_sm)

        proc = mp.Process(
            target=ai_worker,
            kwargs={
                "ready_event": ready,
                "stop_event": stop_event,
                "workload_type": ai_workload,
                "sm_percent": ai_sm,
                "mem_fraction": ai_mem,
            }
        )
        # Set SM limit before starting
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(ai_sm)
        proc.start()
        # Reset for L1 process
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(l1_sm)

        print(f"  Waiting for AI worker...", flush=True)
        if not ready.wait(timeout=120):
            print(f"  AI worker timeout!", flush=True)
            proc.kill()
            return None
        print(f"  AI worker ready. Measuring L1...", flush=True)

    # Measure L1
    result = setup_and_measure_l1(label)
    result["label"] = label
    result["l1_profile"] = l1_profile
    result["ai_profile"] = ai_profile
    result["ai_workload"] = ai_workload
    result["l1_sm_pct"] = l1_sm
    result["ai_sm_pct"] = ai_sm

    # Stop AI worker
    if proc:
        stop_event.set()
        proc.join(timeout=15)
        if proc.is_alive():
            proc.kill()

    rx = result["rx"]
    print(f"\n  RESULT: RX mean={rx['mean_ms']:.3f}ms  "
          f"p95={rx['p95_ms']:.3f}ms  p99={rx['p99_ms']:.3f}ms  "
          f"jitter={rx['jitter']:.4f}  miss(>1ms)={result['rx_miss_1ms']*100:.1f}%",
          flush=True)

    return result


# ============================================================
# Main
# ============================================================
def main():
    import torch

    print("=" * 60, flush=True)
    print("MIG Emulator — GPU Partitioning Experiment", flush=True)
    print("=" * 60, flush=True)

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
    print(f"HBM: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
    print(f"Config: warmup={NUM_WARMUP}, iterations={NUM_ITERATIONS}")

    # AI workloads to test with each partition
    ai_workloads = ["resnet", "hbm_stress"]

    all_results = []

    # 1. Baseline — full GPU, no AI
    result = run_partition("7g.80gb", None, "none", "Full GPU baseline")
    if result:
        all_results.append(result)

    # 2. No partitioning — L1 + AI on full GPU (MPS, no SM limit)
    for wl in ai_workloads:
        result = run_partition("7g.80gb", "7g.80gb", wl,
                               f"No MIG — both use full GPU (worst case)")
        if result:
            all_results.append(result)
        gc.collect()
        time.sleep(3)

    # 3. MIG-emulated partitions
    for l1_prof, ai_prof, desc in PARTITION_CONFIGS:
        if ai_prof is None:
            continue  # skip full GPU solo (already done)
        for wl in ai_workloads:
            result = run_partition(l1_prof, ai_prof, wl, desc)
            if result:
                all_results.append(result)
            gc.collect()
            time.sleep(3)

    # Summary
    baseline_rx = all_results[0]["rx"]["mean_ms"] if all_results else 1

    print(f"\n{'='*70}", flush=True)
    print("SUMMARY — MIG Emulator Results", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Config':<30} {'AI':<12} | {'RX mean':>8} | {'RX P99':>8} | "
          f"{'Jitter':>7} | {'Miss%':>6} | {'vs base':>8}", flush=True)
    print("-" * 90, flush=True)

    for r in all_results:
        rx = r["rx"]
        ratio = rx["mean_ms"] / baseline_rx if baseline_rx > 0 else 0
        ai = r.get("ai_workload", "none")
        cfg = f"L1:{r['l1_profile']}"
        if r.get("ai_profile"):
            cfg += f" AI:{r['ai_profile']}"
        print(f"{cfg:<30} {ai:<12} | {rx['mean_ms']:>6.3f}ms | "
              f"{rx['p99_ms']:>6.3f}ms | {rx['jitter']:>7.4f} | "
              f"{r['rx_miss_1ms']*100:>5.1f}% | {ratio:>7.2f}x", flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"exp_mig_emulator_{ts}.json")
    output = {
        "experiment": "mig_emulator",
        "timestamp": ts,
        "gpu": props.name,
        "gpu_sms": props.multi_processor_count,
        "hbm_gb": round(total / 1e9, 1),
        "mig_profiles": {k: {"sm_pct": v[0], "mem_frac": v[1], "desc": v[2]}
                         for k, v in MIG_PROFILES.items()},
        "config": {"num_warmup": NUM_WARMUP, "num_iterations": NUM_ITERATIONS},
        "results": [{k: v for k, v in r.items() if k != "raw_rx"} for r in all_results],
        "raw": {r["label"]: r.get("raw_rx", []) for r in all_results},
    }
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
