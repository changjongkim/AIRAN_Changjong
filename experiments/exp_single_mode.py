"""
Phase 0: Single Mode Experiment
=================================
Runs ONE interference mode per job. Avoids MPS crash propagation.

Usage:
  python3 exp_single_mode.py --mode baseline
  python3 exp_single_mode.py --mode hbm_same --hbm-gb 8
  python3 exp_single_mode.py --mode hbm_diff --hbm-gb 8
  python3 exp_single_mode.py --mode resnet_same
  python3 exp_single_mode.py --mode resnet_diff
  python3 exp_single_mode.py --mode gpt2_same
  python3 exp_single_mode.py --mode gpt2_diff
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
# Workers
# ============================================================
def _child_env():
    if SITE not in sys.path:
        sys.path.insert(0, SITE)


def worker_hbm(ready_event, stop_event, gpu_id, size_gb):
    import torch
    torch.cuda.set_device(gpu_id)
    n = int(size_gb * 1e9 / 4)
    src = torch.randn(n, dtype=torch.float32, device=f"cuda:{gpu_id}")
    dst = torch.empty_like(src)
    torch.cuda.synchronize(gpu_id)
    ready_event.set()
    c = 0
    while not stop_event.is_set():
        dst.copy_(src); src.copy_(dst); c += 1
    print(f"  [HBM GPU{gpu_id} {size_gb}GB] {c} iters", flush=True)


def worker_resnet(ready_event, stop_event, gpu_id, batch_size):
    _child_env()
    import torch
    import torchvision.models as models
    torch.cuda.set_device(gpu_id)
    model = models.resnet50(weights=None).to(f"cuda:{gpu_id}").eval()
    dummy = torch.randn(batch_size, 3, 224, 224, device=f"cuda:{gpu_id}")
    with torch.no_grad():
        for _ in range(10): model(dummy)
    torch.cuda.synchronize(gpu_id)
    ready_event.set()
    c = 0
    with torch.no_grad():
        while not stop_event.is_set(): model(dummy); c += 1
    print(f"  [ResNet GPU{gpu_id} bs{batch_size}] {c} iters", flush=True)


def worker_gpt2(ready_event, stop_event, gpu_id, batch_size):
    _child_env()
    import torch
    from transformers import GPT2LMHeadModel, GPT2Config
    torch.cuda.set_device(gpu_id)
    cfg = GPT2Config()
    model = GPT2LMHeadModel(cfg).to(f"cuda:{gpu_id}").eval()
    dummy = torch.randint(0, cfg.vocab_size, (batch_size, 512), device=f"cuda:{gpu_id}")
    with torch.no_grad():
        for _ in range(5): model(dummy)
    torch.cuda.synchronize(gpu_id)
    ready_event.set()
    c = 0
    with torch.no_grad():
        while not stop_event.is_set(): model(dummy); c += 1
    print(f"  [GPT2 GPU{gpu_id} bs{batch_size}] {c} iters", flush=True)


# ============================================================
# L1
# ============================================================
def setup_l1(num_tx, num_rx, num_cells, mcs_index, mcs_table=0):
    import numpy as np
    import cupy as cp
    import tensorflow as tf
    import sionna

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
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

    rg = sionna.phy.ofdm.ResourceGrid(
        num_ofdm_symbols=NUM_OFDM_SYMBOLS, fft_size=FFT_SIZE,
        subcarrier_spacing=SUBCARRIER_SPACING, num_tx=num_tx,
        num_streams_per_tx=1, cyclic_prefix_length=CYCLIC_PREFIX_LENGTH,
        num_guard_carriers=NUM_GUARD_SUBCARRIERS, dc_null=False)
    rg_map = sionna.phy.ofdm.ResourceGridMapper(rg)
    rm_g = sionna.phy.ofdm.RemoveNulledSubcarriers(rg)
    ch = sionna.phy.channel.OFDMChannel(
        sionna.phy.channel.RayleighBlockFading(num_rx=1, num_rx_ant=num_rx, num_tx=1, num_tx_ant=num_tx),
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
    mo, cr = get_mcs(mcs_index, mcs_table + 1)
    layers = 1
    tb_size = get_tb_size(mod_order=mo, code_rate=cr, dmrs_syms=DMRS_SYMS,
                          num_prbs=NUM_PRBS, start_sym=START_SYM,
                          num_symbols=NUM_SYMBOLS, num_layers=layers)

    cells = []
    for ci in range(num_cells):
        cid = 41 + ci
        pcw = PdschCwConfig(mcs_table=mcs_table, mcs_index=mcs_index,
                            code_rate=int(cr*10), mod_order=mo)
        pue = PdschUeConfig(cw_configs=[pcw], scid=SCID, dmrs_scrm_id=DMRS_SCRM_ID,
                            layers=layers, dmrs_ports=1, rnti=RNTI, data_scid=DATA_SCID)
        pcfg = PdschConfig(ue_configs=[pue], num_dmrs_cdm_grps_no_data=NUM_DMRS_CDM_GRPS_NO_DATA,
                           start_prb=START_PRB, num_prbs=NUM_PRBS, dmrs_syms=DMRS_SYMS,
                           start_sym=START_SYM, num_symbols=NUM_SYMBOLS)
        tx = PdschTxPipelineFactory().create(
            AerialPdschTxConfig(cell_id=cid, num_tx_ant=num_tx), stream)
        uue = PuschUeConfig(scid=SCID, layers=layers, dmrs_ports=1, rnti=RNTI,
                            data_scid=DATA_SCID, mcs_table=mcs_table, mcs_index=mcs_index,
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
        cells.append({"tx": tx, "rx": rx, "pcfg": pcfg, "ucfg": ucfg})

    import torch
    free, total = torch.cuda.mem_get_info(0)
    print(f"L1: {num_cells} cells, {num_tx}T{num_rx}R, MCS {mcs_index}, "
          f"HBM {(total-free)/1e9:.1f}/{total/1e9:.1f}GB ({(total-free)/total*100:.0f}%)", flush=True)
    return apply_channel, cells, {"mo": mo, "cr": cr, "layers": layers}


def measure(apply_channel, cells, params, No, label):
    import cupy as cp
    from aerial.phy5g.ldpc import random_tb
    sys.path.insert(0, os.path.dirname(__file__))
    from utils.timing import CudaTimer, LatencyTracker

    rx_trk = LatencyTracker(f"RX_{label}")
    timer = CudaTimer()

    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        m = i >= NUM_WARMUP
        slot = i % NUM_SLOTS_PER_FRAME
        tb_np = random_tb(mod_order=params["mo"], code_rate=params["cr"],
                          dmrs_syms=DMRS_SYMS, num_prbs=NUM_PRBS, start_sym=START_SYM,
                          num_symbols=NUM_SYMBOLS, num_layers=params["layers"])
        tb = cp.array(tb_np, dtype=cp.uint8, order="F")

        rx_ms = 0
        for cell in cells:
            tx_t = cell["tx"](slot=slot, tb_inputs=[tb], config=[cell["pcfg"]])
            rx_t = apply_channel(tx_t, No)
            timer.start()
            cell["rx"](slot=slot, rx_slot=rx_t, config=cell["ucfg"])
            timer.stop()
            rx_ms += timer.elapsed_ms()
        if m:
            rx_trk.record(rx_ms)

    return rx_trk


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["baseline", "hbm_same", "hbm_diff",
                                 "resnet_same", "resnet_diff", "gpt2_same", "gpt2_diff"])
    parser.add_argument("--cells", type=int, default=1)
    parser.add_argument("--tx", type=int, default=1)
    parser.add_argument("--rx", type=int, default=2)
    parser.add_argument("--mcs", type=int, default=2)
    parser.add_argument("--hbm-gb", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import torch
    num_gpus = torch.cuda.device_count()
    print("=" * 60, flush=True)
    print(f"Phase 0 | mode={args.mode} | {args.cells}cell {args.tx}T{args.rx}R MCS{args.mcs}")
    print(f"GPUs: {num_gpus}x {torch.cuda.get_device_properties(0).name}")
    print("=" * 60, flush=True)

    apply_channel, cells, params = setup_l1(args.tx, args.rx, args.cells, args.mcs)
    No = pow(10.0, -ESNO_DB / 10.0)

    # Launch worker if needed
    proc = None
    stop_event = None
    worker_gpu = 0 if "same" in args.mode else 1

    worker_map = {
        "hbm_same": (worker_hbm, {"gpu_id": 0, "size_gb": args.hbm_gb}),
        "hbm_diff": (worker_hbm, {"gpu_id": 1, "size_gb": args.hbm_gb}),
        "resnet_same": (worker_resnet, {"gpu_id": 0, "batch_size": args.batch_size}),
        "resnet_diff": (worker_resnet, {"gpu_id": 1, "batch_size": args.batch_size}),
        "gpt2_same": (worker_gpt2, {"gpu_id": 0, "batch_size": 4}),
        "gpt2_diff": (worker_gpt2, {"gpu_id": 1, "batch_size": 4}),
    }

    if args.mode in worker_map:
        fn, kwargs = worker_map[args.mode]
        ready = mp.Event()
        stop_event = mp.Event()
        kwargs["ready_event"] = ready
        kwargs["stop_event"] = stop_event
        proc = mp.Process(target=fn, kwargs=kwargs)
        proc.start()
        print(f"Worker started on GPU{kwargs['gpu_id']}, waiting...", flush=True)
        ready.wait(timeout=120)
        print("Worker ready.", flush=True)

    # Measure
    label = f"{args.cells}c_{args.tx}T{args.rx}R_mcs{args.mcs}_{args.mode}"
    rx_trk = measure(apply_channel, cells, params, No, label)

    # Stop worker
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
        json.dump({"label": label, "mode": args.mode, "config": vars(args),
                    "stats": s, "miss_1ms": miss, "raw": rx_trk.latencies}, f, indent=2)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
