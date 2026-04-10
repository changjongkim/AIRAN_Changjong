"""
Multi-Node HBM Stress Experiment (2 Nodes)
============================================
Compares L1 latency with HBM stress on:
  1. Same GPU (Node0:GPU0)
  2. Same node different GPU (Node0:GPU1)
  3. Different node (Node1:GPU0)

All measured from the SAME L1 pipeline on Node0:GPU0.

Usage: srun -N 2 --ntasks-per-node=1 shifter ... python3 exp_multinode_hbm.py --mode <mode>
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


def run_hbm_stress(gpu_id, duration_sec):
    """Run HBM bandwidth stress on given GPU for fixed duration."""
    import torch
    torch.cuda.set_device(gpu_id)
    hostname = socket.gethostname()
    total = torch.cuda.get_device_properties(gpu_id).total_memory
    alloc = int(total * 0.4)
    n = alloc // 4
    src = torch.randn(n, dtype=torch.float32, device=f"cuda:{gpu_id}")
    dst = torch.empty_like(src)
    torch.cuda.synchronize(gpu_id)
    print(f"[Rank {os.environ.get('SLURM_PROCID','?')}] {hostname}:GPU{gpu_id} "
          f"HBM stress {alloc/1e9:.1f}GB running for {duration_sec}s", flush=True)
    c = 0
    start = time.time()
    while time.time() - start < duration_sec:
        dst.copy_(src)
        src.copy_(dst)
        c += 1
    print(f"[Rank {os.environ.get('SLURM_PROCID','?')}] HBM stress done: {c} iters", flush=True)


def run_l1():
    """Run L1 pipeline and return latency stats."""
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
    hostname = socket.gethostname()
    print(f"[L1] {hostname}:GPU0 HBM {(total-free)/1e9:.1f}/{total/1e9:.1f}GB", flush=True)

    trk = LatencyTracker("RX")
    timer = CudaTimer()
    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        m = i >= NUM_WARMUP
        slot = i % NUM_SLOTS_PER_FRAME
        tb = cp.array(random_tb(mod_order=mo, code_rate=cr, dmrs_syms=DMRS_SYMS,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["baseline", "same_gpu", "diff_gpu", "diff_node"])
    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1"))
    hostname = socket.gethostname()

    print(f"[Rank {rank}] {hostname} | mode={args.mode} | world={world}", flush=True)

    est_duration = (NUM_WARMUP + NUM_ITERATIONS) * 0.5 + 60

    if args.mode == "baseline":
        # Single node, no stress
        if rank == 0:
            print(f"\n=== baseline (L1 solo) ===", flush=True)
            trk = run_l1()
        else:
            time.sleep(est_duration)

    elif args.mode == "same_gpu":
        # Rank 0: L1 on GPU0 + HBM stress on GPU0 (via MPS subprocess)
        if rank == 0:
            import multiprocessing as mp
            mp.set_start_method("spawn", force=True)
            ready = mp.Event()
            stop = mp.Event()

            def _stress(ready_event, stop_event):
                import torch
                torch.cuda.set_device(0)
                n = int(torch.cuda.get_device_properties(0).total_memory * 0.3 / 4)
                s = torch.randn(n, dtype=torch.float32, device="cuda")
                d = torch.empty_like(s)
                torch.cuda.synchronize()
                ready_event.set()
                c = 0
                while not stop_event.is_set():
                    d.copy_(s); s.copy_(d); c += 1

            proc = mp.Process(target=_stress, args=(ready, stop))
            proc.start()
            ready.wait(timeout=60)

            print(f"\n=== same_gpu (L1+HBM on GPU0) ===", flush=True)
            trk = run_l1()

            stop.set()
            proc.join(timeout=10)
            if proc.is_alive(): proc.kill()
        else:
            time.sleep(est_duration)

    elif args.mode == "diff_gpu":
        # Rank 0: L1 on GPU0, stress on GPU1
        if rank == 0:
            import multiprocessing as mp
            mp.set_start_method("spawn", force=True)
            ready = mp.Event()
            stop = mp.Event()

            def _stress(ready_event, stop_event):
                import torch
                torch.cuda.set_device(1)
                n = int(torch.cuda.get_device_properties(1).total_memory * 0.3 / 4)
                s = torch.randn(n, dtype=torch.float32, device="cuda:1")
                d = torch.empty_like(s)
                torch.cuda.synchronize(1)
                ready_event.set()
                c = 0
                while not stop_event.is_set():
                    d.copy_(s); s.copy_(d); c += 1

            proc = mp.Process(target=_stress, args=(ready, stop))
            proc.start()
            ready.wait(timeout=60)

            print(f"\n=== diff_gpu (L1:GPU0, HBM:GPU1) ===", flush=True)
            trk = run_l1()

            stop.set()
            proc.join(timeout=10)
            if proc.is_alive(): proc.kill()
        else:
            time.sleep(est_duration)

    elif args.mode == "diff_node":
        # Rank 0: L1, Rank 1: HBM stress
        if rank == 0:
            time.sleep(10)  # let rank1 start stress
            print(f"\n=== diff_node (L1:Node0, HBM:Node1) ===", flush=True)
            trk = run_l1()
        else:
            run_hbm_stress(gpu_id=0, duration_sec=int(est_duration))

    # Save results (rank 0 only)
    if rank == 0:
        s = trk.stats()
        miss = trk.deadline_miss_rate(1.0)
        label = f"multinode_{args.mode}"

        print(f"\n{'='*60}", flush=True)
        print(f"RESULT: {label} ({hostname})", flush=True)
        print(f"  RX mean:  {s['mean_ms']:.3f} ms", flush=True)
        print(f"  RX P95:   {s['p95_ms']:.3f} ms", flush=True)
        print(f"  RX P99:   {s['p99_ms']:.3f} ms", flush=True)
        print(f"  Jitter:   {s['jitter']:.4f}", flush=True)
        print(f"  Miss>1ms: {miss*100:.1f}%", flush=True)
        print(f"{'='*60}", flush=True)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(RESULTS_DIR, f"exp_{label}_{ts}.json")
        with open(out, "w") as f:
            json.dump({"label": label, "mode": args.mode, "hostname": hostname,
                        "stats": s, "miss_1ms": miss, "raw": trk.latencies}, f, indent=2)
        print(f"Saved: {out}", flush=True)

    print(f"[Rank {rank}] Done.", flush=True)


if __name__ == "__main__":
    main()
