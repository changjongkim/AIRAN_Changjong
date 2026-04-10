"""
MIG Emulator — HBM Stress Sweep (Single Job)
==============================================
Runs all MIG partition configs with HBM stress sequentially on the SAME node.
Guarantees identical hardware for all comparisons.

Modes (sequential):
  1. baseline      — L1 solo, full GPU
  2. noMIG_hbm     — L1 + HBM stress, no SM limit (worst case)
  3. 4g+3g_hbm     — L1(57% SM) + HBM stress(43% SM)
  4. 3g+4g_hbm     — L1(43% SM) + HBM stress(57% SM)
  5. 2g+4g_hbm     — L1(29% SM) + HBM stress(57% SM)
  6. 1g+4g_hbm     — L1(14% SM) + HBM stress(57% SM)
  7. baseline_final — L1 solo again (consistency check)
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
SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"

NUM_WARMUP = 10
NUM_ITERATIONS = 100
RESULTS_DIR = "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results"
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

# MIG profiles: (sm_percent, mem_fraction)
PROFILES = {
    "7g": (100, 0.90),
    "4g": (57,  0.45),
    "3g": (43,  0.45),
    "2g": (29,  0.22),
    "1g": (14,  0.10),
}


# ============================================================
# HBM Stress Worker
# ============================================================
def hbm_worker(ready_event, stop_event, sm_pct, mem_frac):
    import torch
    torch.cuda.set_device(0)
    if mem_frac < 0.9:
        torch.cuda.set_per_process_memory_fraction(mem_frac, 0)
    total = torch.cuda.get_device_properties(0).total_memory
    alloc = int(total * mem_frac * 0.5)  # Use 50% of allowed fraction
    n = alloc // 4
    src = torch.randn(n, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    torch.cuda.synchronize()
    ready_event.set()
    c = 0
    while not stop_event.is_set():
        dst.copy_(src)
        src.copy_(dst)
        c += 1
    print(f"    [HBM stress SM={sm_pct}% alloc={alloc/1e9:.1f}GB] {c} iters", flush=True)


# ============================================================
# L1 Setup (done once, reused across modes)
# ============================================================
def setup_l1():
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
    from aerial.phy5g.ldpc import get_mcs, get_tb_size
    from aerial.util.cuda import get_cuda_stream
    from aerial.pycuphy.types import PuschLdpcKernelLaunch

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

    def apply_ch(tx_t, No):
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
    tb_sz = get_tb_size(mod_order=mo, code_rate=cr, dmrs_syms=DMRS_SYMS,
                        num_prbs=NUM_PRBS, start_sym=START_SYM,
                        num_symbols=NUM_SYMBOLS, num_layers=1)
    cid = 41
    pcw = PdschCwConfig(mcs_table=0, mcs_index=mcs_index, code_rate=int(cr*10), mod_order=mo)
    pue = PdschUeConfig(cw_configs=[pcw], scid=SCID, dmrs_scrm_id=DMRS_SCRM_ID,
                        layers=1, dmrs_ports=1, rnti=RNTI, data_scid=DATA_SCID)
    pcfg = PdschConfig(ue_configs=[pue], num_dmrs_cdm_grps_no_data=NUM_DMRS_CDM_GRPS_NO_DATA,
                       start_prb=START_PRB, num_prbs=NUM_PRBS, dmrs_syms=DMRS_SYMS,
                       start_sym=START_SYM, num_symbols=NUM_SYMBOLS)
    txp = PdschTxPipelineFactory().create(
        AerialPdschTxConfig(cell_id=cid, num_tx_ant=num_tx), stream)
    uue = PuschUeConfig(scid=SCID, layers=1, dmrs_ports=1, rnti=RNTI,
                        data_scid=DATA_SCID, mcs_table=0, mcs_index=mcs_index,
                        code_rate=int(cr*10), mod_order=mo, tb_size=tb_sz//8)
    ucfg = [PuschConfig(ue_configs=[uue], num_dmrs_cdm_grps_no_data=NUM_DMRS_CDM_GRPS_NO_DATA,
                        dmrs_scrm_id=DMRS_SCRM_ID, start_prb=START_PRB, num_prbs=NUM_PRBS,
                        dmrs_syms=DMRS_SYMS, dmrs_max_len=DMRS_MAX_LEN,
                        dmrs_add_ln_pos=DMRS_ADD_LN_POS, start_sym=START_SYM,
                        num_symbols=NUM_SYMBOLS)]
    rxp = PuschRxPipelineFactory().create(
        AerialPuschRxConfig(cell_id=cid, num_rx_ant=num_rx, enable_pusch_tdi=0,
                            eq_coeff_algo=1,
                            ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL),
        stream)

    return apply_ch, txp, rxp, pcfg, ucfg, mo, cr


# ============================================================
# Measure L1
# ============================================================
def measure(apply_ch, txp, rxp, pcfg, ucfg, mo, cr, No, label):
    import cupy as cp
    from aerial.phy5g.ldpc import random_tb
    sys.path.insert(0, os.path.dirname(__file__))
    from utils.timing import CudaTimer, LatencyTracker

    trk = LatencyTracker(label)
    timer = CudaTimer()
    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        m = i >= NUM_WARMUP
        slot = i % NUM_SLOTS_PER_FRAME
        tb = cp.array(
            random_tb(mod_order=mo, code_rate=cr, dmrs_syms=DMRS_SYMS,
                      num_prbs=NUM_PRBS, start_sym=START_SYM,
                      num_symbols=NUM_SYMBOLS, num_layers=1),
            dtype=cp.uint8, order="F")
        tx_t = txp(slot=slot, tb_inputs=[tb], config=[pcfg])
        rx_t = apply_ch(tx_t, No)
        timer.start()
        rxp(slot=slot, rx_slot=rx_t, config=ucfg)
        timer.stop()
        if m:
            trk.record(timer.elapsed_ms())
    return trk


# ============================================================
# Run one mode
# ============================================================
def run_mode(apply_ch, txp, rxp, pcfg, ucfg, mo, cr, No,
             label, l1_prof, ai_prof=None):
    print(f"\n--- {label} ---", flush=True)

    l1_sm, _ = PROFILES[l1_prof]
    proc = None
    stop_event = None

    if ai_prof:
        ai_sm, ai_mem = PROFILES[ai_prof]
        ready = mp.Event()
        stop_event = mp.Event()

        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(ai_sm)
        proc = mp.Process(target=hbm_worker, kwargs={
            "ready_event": ready, "stop_event": stop_event,
            "sm_pct": ai_sm, "mem_frac": ai_mem,
        })
        proc.start()
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(l1_sm)

        if not ready.wait(timeout=60):
            print(f"  AI worker timeout!", flush=True)
            proc.kill()
            return None
        print(f"  L1={l1_prof}({l1_sm}%SM) + AI={ai_prof}({ai_sm}%SM) HBM stress", flush=True)
    else:
        print(f"  L1={l1_prof}({l1_sm}%SM) solo", flush=True)

    trk = measure(apply_ch, txp, rxp, pcfg, ucfg, mo, cr, No, label)

    if proc:
        stop_event.set()
        proc.join(timeout=15)
        if proc.is_alive():
            proc.kill()

    s = trk.stats()
    miss = trk.deadline_miss_rate(1.0)
    print(f"  RX: mean={s['mean_ms']:.3f}ms  p95={s['p95_ms']:.3f}ms  "
          f"p99={s['p99_ms']:.3f}ms  jitter={s['jitter']:.4f}  "
          f"miss>1ms={miss*100:.1f}%", flush=True)

    return {"label": label, "l1": l1_prof, "ai": ai_prof,
            "l1_sm": l1_sm, "ai_sm": PROFILES[ai_prof][0] if ai_prof else 0,
            "stats": s, "miss_1ms": miss, "raw": trk.latencies}


# ============================================================
# Main
# ============================================================
def main():
    import torch

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)

    print("=" * 60, flush=True)
    print("MIG Emulator — HBM Stress Sweep (Same Node)", flush=True)
    print("=" * 60, flush=True)
    print(f"GPU: {props.name} ({props.multi_processor_count} SMs)", flush=True)
    print(f"HBM: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free", flush=True)
    print(f"Iterations: {NUM_ITERATIONS} (warmup: {NUM_WARMUP})", flush=True)

    print("\nSetting up L1 pipeline...", flush=True)
    apply_ch, txp, rxp, pcfg, ucfg, mo, cr = setup_l1()
    No = pow(10.0, -ESNO_DB / 10.0)

    modes = [
        ("baseline",      "7g", None),
        ("noMIG_hbm",     "7g", "7g"),
        ("MIG_4g+3g_hbm", "4g", "3g"),
        ("MIG_3g+4g_hbm", "3g", "4g"),
        ("MIG_2g+4g_hbm", "2g", "4g"),
        ("MIG_1g+4g_hbm", "1g", "4g"),
        ("baseline_final","7g", None),
    ]

    results = []
    for label, l1, ai in modes:
        r = run_mode(apply_ch, txp, rxp, pcfg, ucfg, mo, cr, No, label, l1, ai)
        if r:
            results.append(r)
        gc.collect()
        time.sleep(2)

    # Summary
    base = results[0]["stats"]["mean_ms"] if results else 1
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Mode':<20} {'L1 SM%':>7} {'AI SM%':>7} | {'RX mean':>8} | "
          f"{'RX P99':>8} | {'Jitter':>7} | {'Miss%':>6} | {'vs base':>8}", flush=True)
    print("-" * 85, flush=True)
    for r in results:
        s = r["stats"]
        ratio = s["mean_ms"] / base if base > 0 else 0
        print(f"{r['label']:<20} {r['l1_sm']:>6}% {r['ai_sm']:>6}% | "
              f"{s['mean_ms']:>6.3f}ms | {s['p99_ms']:>6.3f}ms | "
              f"{s['jitter']:>7.4f} | {r['miss_1ms']*100:>5.1f}% | "
              f"{ratio:>7.2f}x", flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"exp_mig_hbm_sweep_{ts}.json")
    with open(out, "w") as f:
        json.dump({
            "experiment": "mig_hbm_sweep",
            "timestamp": ts,
            "gpu": props.name,
            "gpu_sms": props.multi_processor_count,
            "hbm_gb": round(total / 1e9, 1),
            "config": {"num_warmup": NUM_WARMUP, "num_iterations": NUM_ITERATIONS},
            "results": [{k: v for k, v in r.items() if k != "raw"} for r in results],
            "raw": {r["label"]: r["raw"] for r in results},
        }, f, indent=2)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
