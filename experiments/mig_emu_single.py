"""
MIG Emulator — Single Mode
============================
Runs ONE MIG partition config per job. No crash propagation.

Usage:
  python3 mig_emu_single.py --l1-profile 4g --ai-profile 3g --ai-workload resnet
  python3 mig_emu_single.py --l1-profile 7g --ai-workload none  # baseline
"""
import os
import sys
import gc
import json
import time
import argparse
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
    "7g": (100, 0.95),
    "4g": (57,  0.50),
    "3g": (43,  0.50),
    "2g": (29,  0.25),
    "1g": (14,  0.12),
}


# ============================================================
# AI Worker
# ============================================================
def ai_worker(ready_event, stop_event, workload_type, sm_pct, mem_frac):
    if SITE not in sys.path:
        sys.path.insert(0, SITE)

    import torch
    torch.cuda.set_device(0)
    if mem_frac < 0.95:
        torch.cuda.set_per_process_memory_fraction(mem_frac, 0)

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
        print(f"  [AI ResNet SM={sm_pct}%] {c} iters", flush=True)

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
        print(f"  [AI GPT2 SM={sm_pct}%] {c} iters", flush=True)

    elif workload_type == "hbm_stress":
        total = torch.cuda.get_device_properties(0).total_memory
        alloc = int(total * mem_frac * 0.6)
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
        print(f"  [AI HBM SM={sm_pct}% {alloc/1e9:.1f}GB] {c} iters", flush=True)


# ============================================================
# L1 Pipeline
# ============================================================
def run_l1(label):
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
    No = pow(10.0, -ESNO_DB / 10.0)

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

    import torch
    free, total = torch.cuda.mem_get_info(0)
    print(f"  L1 HBM: {(total-free)/1e9:.1f}/{total/1e9:.1f}GB", flush=True)

    rx_trk = LatencyTracker(f"RX_{label}")
    timer = CudaTimer()

    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        m = i >= NUM_WARMUP
        slot = i % NUM_SLOTS_PER_FRAME
        tb_np = random_tb(mod_order=mo, code_rate=cr, dmrs_syms=DMRS_SYMS,
                          num_prbs=NUM_PRBS, start_sym=START_SYM,
                          num_symbols=NUM_SYMBOLS, num_layers=1)
        tb = cp.array(tb_np, dtype=cp.uint8, order="F")
        tx_t = txp(slot=slot, tb_inputs=[tb], config=[pcfg])
        rx_t = apply_ch(tx_t, No)
        timer.start()
        rxp(slot=slot, rx_slot=rx_t, config=ucfg)
        timer.stop()
        if m:
            rx_trk.record(timer.elapsed_ms())

    return rx_trk


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l1-profile", required=True, choices=PROFILES.keys())
    parser.add_argument("--ai-profile", default=None, choices=list(PROFILES.keys()))
    parser.add_argument("--ai-workload", default="none",
                        choices=["none", "resnet", "gpt2", "hbm_stress"])
    args = parser.parse_args()

    import torch
    props = torch.cuda.get_device_properties(0)
    l1_sm, l1_mem = PROFILES[args.l1_profile]

    label = f"MIG_L1-{args.l1_profile}"
    if args.ai_profile and args.ai_workload != "none":
        ai_sm, ai_mem = PROFILES[args.ai_profile]
        label += f"_AI-{args.ai_profile}_{args.ai_workload}"
    else:
        ai_sm, ai_mem = 0, 0
        label += "_solo"

    print("=" * 60, flush=True)
    print(f"MIG Emulator: {label}", flush=True)
    print(f"GPU: {props.name} ({props.multi_processor_count} SMs)", flush=True)
    print(f"L1:  {args.l1_profile} ({l1_sm}% SM, {l1_mem:.0%} mem)", flush=True)
    if args.ai_profile and args.ai_workload != "none":
        print(f"AI:  {args.ai_profile} ({ai_sm}% SM, {ai_mem:.0%} mem) — {args.ai_workload}", flush=True)
    print("=" * 60, flush=True)

    # Launch AI worker
    proc = None
    stop_event = None

    if args.ai_profile and args.ai_workload != "none":
        ready = mp.Event()
        stop_event = mp.Event()

        # Set AI SM limit via MPS env
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(ai_sm)

        proc = mp.Process(target=ai_worker, kwargs={
            "ready_event": ready, "stop_event": stop_event,
            "workload_type": args.ai_workload,
            "sm_pct": ai_sm, "mem_frac": ai_mem,
        })
        proc.start()

        # Set L1 SM limit
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(l1_sm)

        print("Waiting for AI worker...", flush=True)
        if not ready.wait(timeout=120):
            print("AI worker timeout!", flush=True)
            proc.kill()
            return
        print("AI worker ready. Measuring L1...", flush=True)

    # Measure L1
    rx_trk = run_l1(label)

    # Stop AI
    if proc:
        stop_event.set()
        proc.join(timeout=15)
        if proc.is_alive():
            proc.kill()

    # Results
    s = rx_trk.stats()
    miss = rx_trk.deadline_miss_rate(1.0)

    print(f"\n{'='*60}", flush=True)
    print(f"RESULT: {label}")
    print(f"  RX mean:  {s['mean_ms']:.3f} ms")
    print(f"  RX P50:   {s['p50_ms']:.3f} ms")
    print(f"  RX P95:   {s['p95_ms']:.3f} ms")
    print(f"  RX P99:   {s['p99_ms']:.3f} ms")
    print(f"  Jitter:   {s['jitter']:.4f}")
    print(f"  Miss>1ms: {miss*100:.1f}%")
    print(f"{'='*60}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"exp_{label}_{ts}.json")
    with open(out, "w") as f:
        json.dump({
            "label": label, "l1_profile": args.l1_profile,
            "ai_profile": args.ai_profile, "ai_workload": args.ai_workload,
            "l1_sm_pct": l1_sm, "ai_sm_pct": ai_sm,
            "stats": s, "miss_1ms": miss, "raw": rx_trk.latencies,
        }, f, indent=2)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
