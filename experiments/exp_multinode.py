"""
Phase 0: Multi-Node Experiment (2 Nodes)
==========================================
Compares L1 interference across GPU isolation levels:
  - Same GPU (MPS sharing)
  - Same node, different GPU
  - Different node entirely

Node 0: Always runs L1 (cuPHY)
Node 1: Runs AI workload (controlled via MPI)

Usage:
  srun -N 2 --ntasks-per-node=1 shifter ... python3 exp_multinode.py --ai-workload resnet
"""
import os
import sys
import json
import time
import argparse
import datetime
import socket

sys.stdout.reconfigure(line_buffering=True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HOME"] = "/pscratch/sd/s/sgkim/kcj/AI-RAN/datasets/models"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

NUM_WARMUP = 10
NUM_ITERATIONS = 100
RESULTS_DIR = "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results"
ESNO_DB = 10.0

# NR parameters (same as single mode)
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
# AI workload (runs on designated node/GPU)
# ============================================================
def run_ai_workload(workload_type, gpu_id, duration_sec):
    """Run AI workload for a fixed duration. Called on the AI node."""
    import torch
    torch.cuda.set_device(gpu_id)
    hostname = socket.gethostname()
    print(f"[AI Worker] {workload_type} on {hostname}:GPU{gpu_id}", flush=True)

    if workload_type == "resnet":
        import torchvision.models as models
        model = models.resnet50(weights=None).to(f"cuda:{gpu_id}").eval()
        dummy = torch.randn(32, 3, 224, 224, device=f"cuda:{gpu_id}")
        with torch.no_grad():
            for _ in range(10):
                model(dummy)
        torch.cuda.synchronize(gpu_id)
        print(f"[AI Worker] ResNet-50 warmup done, running for {duration_sec}s...", flush=True)
        count = 0
        start = time.time()
        with torch.no_grad():
            while time.time() - start < duration_sec:
                model(dummy)
                count += 1
        print(f"[AI Worker] ResNet-50 done: {count} iters, "
              f"{count/(time.time()-start):.1f} it/s", flush=True)

    elif workload_type == "gpt2":
        from transformers import GPT2LMHeadModel, GPT2Config
        cfg = GPT2Config()
        model = GPT2LMHeadModel(cfg).to(f"cuda:{gpu_id}").eval()
        dummy = torch.randint(0, cfg.vocab_size, (4, 512), device=f"cuda:{gpu_id}")
        with torch.no_grad():
            for _ in range(5):
                model(dummy)
        torch.cuda.synchronize(gpu_id)
        print(f"[AI Worker] GPT-2 warmup done, running for {duration_sec}s...", flush=True)
        count = 0
        start = time.time()
        with torch.no_grad():
            while time.time() - start < duration_sec:
                model(dummy)
                count += 1
        print(f"[AI Worker] GPT-2 done: {count} iters, "
              f"{count/(time.time()-start):.1f} it/s", flush=True)

    elif workload_type == "hbm":
        n = int(8e9 / 4)
        src = torch.randn(n, dtype=torch.float32, device=f"cuda:{gpu_id}")
        dst = torch.empty_like(src)
        torch.cuda.synchronize(gpu_id)
        print(f"[AI Worker] HBM stress warmup done, running for {duration_sec}s...", flush=True)
        count = 0
        start = time.time()
        while time.time() - start < duration_sec:
            dst.copy_(src)
            src.copy_(dst)
            count += 1
        print(f"[AI Worker] HBM stress done: {count} copy iters", flush=True)

    elif workload_type == "none":
        print(f"[AI Worker] Idle for {duration_sec}s...", flush=True)
        time.sleep(duration_sec)

    else:
        raise ValueError(f"Unknown workload: {workload_type}")


# ============================================================
# L1 measurement (runs on Node 0)
# ============================================================
def run_l1_measurement(num_cells, num_tx, num_rx, mcs_index):
    """Set up and run L1 pipeline, return latency tracker."""
    import numpy as np
    import cupy as cp
    import tensorflow as tf
    import sionna

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
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
    mo, cr = get_mcs(mcs_index, 1)
    layers = 1
    tb_size = get_tb_size(mod_order=mo, code_rate=cr, dmrs_syms=DMRS_SYMS,
                          num_prbs=NUM_PRBS, start_sym=START_SYM,
                          num_symbols=NUM_SYMBOLS, num_layers=layers)
    No = pow(10.0, -ESNO_DB / 10.0)

    cells = []
    for ci in range(num_cells):
        cid = 41 + ci
        pcw = PdschCwConfig(mcs_table=0, mcs_index=mcs_index, code_rate=int(cr*10), mod_order=mo)
        pue = PdschUeConfig(cw_configs=[pcw], scid=SCID, dmrs_scrm_id=DMRS_SCRM_ID,
                            layers=layers, dmrs_ports=1, rnti=RNTI, data_scid=DATA_SCID)
        pcfg = PdschConfig(ue_configs=[pue], num_dmrs_cdm_grps_no_data=NUM_DMRS_CDM_GRPS_NO_DATA,
                           start_prb=START_PRB, num_prbs=NUM_PRBS, dmrs_syms=DMRS_SYMS,
                           start_sym=START_SYM, num_symbols=NUM_SYMBOLS)
        tx = PdschTxPipelineFactory().create(
            AerialPdschTxConfig(cell_id=cid, num_tx_ant=num_tx), stream)
        uue = PuschUeConfig(scid=SCID, layers=layers, dmrs_ports=1, rnti=RNTI,
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
        cells.append({"tx": tx, "rx": rx, "pcfg": pcfg, "ucfg": ucfg})

    import torch
    free, total = torch.cuda.mem_get_info(0)
    print(f"[L1] {num_cells} cells, {num_tx}T{num_rx}R, MCS {mcs_index}, "
          f"HBM {(total-free)/1e9:.1f}/{total/1e9:.1f}GB", flush=True)

    # Measure
    rx_trk = LatencyTracker("RX")
    timer = CudaTimer()

    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        m = i >= NUM_WARMUP
        slot = i % NUM_SLOTS_PER_FRAME
        tb_np = random_tb(mod_order=mo, code_rate=cr, dmrs_syms=DMRS_SYMS,
                          num_prbs=NUM_PRBS, start_sym=START_SYM,
                          num_symbols=NUM_SYMBOLS, num_layers=layers)
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
# Main — MPI-based coordination
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai-workload", default="resnet",
                        choices=["none", "resnet", "gpt2", "hbm"])
    parser.add_argument("--ai-gpu", type=int, default=0, help="GPU id for AI on its node")
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--tx", type=int, default=2)
    parser.add_argument("--rx", type=int, default=4)
    parser.add_argument("--mcs", type=int, default=10)
    args = parser.parse_args()

    # Determine rank from SLURM
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    hostname = socket.gethostname()

    print(f"[Rank {rank}] {hostname} | world_size={world_size} | "
          f"ai_workload={args.ai_workload}", flush=True)

    if world_size < 2:
        print("ERROR: Need at least 2 ranks (2 nodes). Use srun -N 2 --ntasks-per-node=1",
              flush=True)
        sys.exit(1)

    # Estimate L1 measurement duration
    est_duration = (NUM_WARMUP + NUM_ITERATIONS) * 0.5 + 60  # generous estimate

    if rank == 0:
        # Node 0: Run L1 measurement
        print(f"\n[Rank 0] Running L1 on {hostname}:GPU0", flush=True)
        # Wait for AI worker to warm up
        time.sleep(15)
        print(f"[Rank 0] Starting L1 measurement...", flush=True)

        rx_trk = run_l1_measurement(args.cells, args.tx, args.rx, args.mcs)

        s = rx_trk.stats()
        miss = rx_trk.deadline_miss_rate(1.0)
        label = (f"{args.cells}c_{args.tx}T{args.rx}R_mcs{args.mcs}_"
                 f"multinode_{args.ai_workload}_gpu{args.ai_gpu}")

        print(f"\n{'='*60}", flush=True)
        print(f"RESULT: {label}")
        print(f"  L1 node: {hostname}")
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
            json.dump({"label": label, "l1_node": hostname,
                        "config": vars(args), "stats": s,
                        "miss_1ms": miss, "raw": rx_trk.latencies}, f, indent=2)
        print(f"Saved: {out}", flush=True)

    else:
        # Node 1+: Run AI workload
        print(f"\n[Rank {rank}] Running AI={args.ai_workload} on "
              f"{hostname}:GPU{args.ai_gpu}", flush=True)
        run_ai_workload(args.ai_workload, args.ai_gpu, int(est_duration))

    print(f"[Rank {rank}] Done.", flush=True)


if __name__ == "__main__":
    main()
