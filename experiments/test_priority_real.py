"""
Stream Priority with REAL cuPHY + REAL AI workloads.
Tests if CUDA stream priority reduces bandwidth interference
when actual cuPHY L1 (4T4R multi-cell) runs with GPT-2/ResNet/HBM stress.

Usage: python3 test_priority_real.py <label> <num_cells> <ai_type> <priority_mode>
  ai_type: none / hbm / gpt2 / resnet
  priority_mode: default / high
"""
import os, sys, json, time, datetime, threading
sys.stdout.reconfigure(line_buffering=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

import numpy as np
import cupy as cp
import torch
from cuda import cudart

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

RESULTS_DIR = "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results"
NUM_WARMUP = 10
NUM_ITERATIONS = 50


def run_ai_background(ai_type, stop_event, gpu_id=0, priority_mode="default"):
    """Run AI workload in background thread with specified stream priority."""
    torch.cuda.set_device(gpu_id)

    # Create stream with priority
    if priority_mode == "high":
        # AI gets LOW priority (0 = lowest)
        err, ai_stream = cudart.cudaStreamCreateWithPriority(
            cudart.cudaStreamNonBlocking, 0)
    else:
        err, ai_stream = cudart.cudaStreamCreate()

    if ai_type == "hbm":
        n = int(2e9 / 4)
        src = torch.randn(n, dtype=torch.float32, device="cuda")
        dst = torch.empty_like(src)
        torch.cuda.synchronize()
        c = 0
        while not stop_event.is_set():
            with torch.cuda.stream(torch.cuda.ExternalStream(int(ai_stream))):
                dst.copy_(src)
                src.copy_(dst)
            c += 1

    elif ai_type == "gpt2":
        from transformers import GPT2LMHeadModel, GPT2Config
        cfg = GPT2Config()
        model = GPT2LMHeadModel(cfg).cuda().eval()
        dummy = torch.randint(0, cfg.vocab_size, (4, 512), device="cuda")
        with torch.no_grad():
            for _ in range(3):
                model(dummy)
        torch.cuda.synchronize()
        c = 0
        with torch.no_grad():
            while not stop_event.is_set():
                with torch.cuda.stream(torch.cuda.ExternalStream(int(ai_stream))):
                    model(dummy)
                c += 1

    elif ai_type == "resnet":
        import torchvision.models as models
        model = models.resnet50(weights=None).cuda().eval()
        dummy = torch.randn(64, 3, 224, 224, device="cuda")
        with torch.no_grad():
            for _ in range(5):
                model(dummy)
        torch.cuda.synchronize()
        c = 0
        with torch.no_grad():
            while not stop_event.is_set():
                with torch.cuda.stream(torch.cuda.ExternalStream(int(ai_stream))):
                    model(dummy)
                c += 1


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "priority_test"
    num_cells = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ai_type = sys.argv[3] if len(sys.argv) > 3 else "none"
    priority_mode = sys.argv[4] if len(sys.argv) > 4 else "default"

    num_tx, num_rx = 4, 4
    mcs_index = 2

    # Create L1 stream with priority
    if priority_mode == "high":
        # L1 gets HIGH priority (-5 = highest on A100)
        err, l1_stream_raw = cudart.cudaStreamCreateWithPriority(
            cudart.cudaStreamNonBlocking, -5)
        l1_stream = l1_stream_raw
    else:
        l1_stream = get_cuda_stream()

    mo, cr = get_mcs(mcs_index, 1)
    tb_sz = get_tb_size(mod_order=mo, code_rate=cr,
                        dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                        num_prbs=273, start_sym=2, num_symbols=12, num_layers=1)

    cells = []
    for ci in range(num_cells):
        cid = 41 + ci
        pcw = PdschCwConfig(mcs_table=0, mcs_index=mcs_index,
                            code_rate=int(cr*10), mod_order=mo)
        pue = PdschUeConfig(cw_configs=[pcw], scid=0, dmrs_scrm_id=41,
                            layers=1, dmrs_ports=1, rnti=1234, data_scid=0)
        pcfg = PdschConfig(ue_configs=[pue], num_dmrs_cdm_grps_no_data=2,
                           start_prb=0, num_prbs=273,
                           dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                           start_sym=2, num_symbols=12)
        txp = PdschTxPipelineFactory().create(
            AerialPdschTxConfig(cell_id=cid, num_tx_ant=num_tx), l1_stream)
        uue = PuschUeConfig(scid=0, layers=1, dmrs_ports=1, rnti=1234,
                            data_scid=0, mcs_table=0, mcs_index=mcs_index,
                            code_rate=int(cr*10), mod_order=mo, tb_size=tb_sz//8)
        ucfg = [PuschConfig(ue_configs=[uue], num_dmrs_cdm_grps_no_data=2,
                            dmrs_scrm_id=41, start_prb=0, num_prbs=273,
                            dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                            dmrs_max_len=1, dmrs_add_ln_pos=0,
                            start_sym=2, num_symbols=12)]
        rxp = PuschRxPipelineFactory().create(
            AerialPuschRxConfig(cell_id=cid, num_rx_ant=num_rx, enable_pusch_tdi=0,
                                eq_coeff_algo=1,
                                ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL),
            l1_stream)
        cells.append({"tx": txp, "rx": rxp, "pcfg": pcfg, "ucfg": ucfg})

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU: {props.name}", flush=True)
    print(f"L1: {num_cells}cells {num_tx}T{num_rx}R, AI: {ai_type}, priority: {priority_mode}",
          flush=True)
    print(f"HBM: {(total-free)/1e9:.1f}/{total/1e9:.1f}GB", flush=True)

    # Pre-gen inputs
    tb = cp.array(random_tb(mod_order=mo, code_rate=cr,
                            dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                            num_prbs=273, start_sym=2, num_symbols=12, num_layers=1),
                  dtype=cp.uint8, order="F")
    rx_inputs = []
    for cell in cells:
        tx_out = cell["tx"](slot=0, tb_inputs=[tb], config=[cell["pcfg"]])
        if num_rx > tx_out.shape[2]:
            repeats = (num_rx // tx_out.shape[2]) + 1
            rx_in = cp.concatenate([tx_out] * repeats, axis=2)[:, :, :num_rx]
        else:
            rx_in = tx_out[:, :, :num_rx]
        noise = 0.01 * (cp.random.randn(*rx_in.shape, dtype=cp.float32) +
                1j * cp.random.randn(*rx_in.shape, dtype=cp.float32)).astype(cp.complex64)
        rx_inputs.append(rx_in + noise)

    # Start AI background
    stop_event = threading.Event()
    ai_thread = None
    if ai_type != "none":
        ai_thread = threading.Thread(target=run_ai_background,
                                     args=(ai_type, stop_event, 0, priority_mode),
                                     daemon=True)
        ai_thread.start()
        time.sleep(5)
        print(f"AI background ({ai_type}) started", flush=True)

    # Measure L1
    start_ev = cp.cuda.Event()
    end_ev = cp.cuda.Event()
    latencies = []

    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        start_ev.record()
        for ci, cell in enumerate(cells):
            cell["rx"](slot=0, rx_slot=rx_inputs[ci], config=cell["ucfg"])
        end_ev.record()
        end_ev.synchronize()
        if i >= NUM_WARMUP:
            latencies.append(cp.cuda.get_elapsed_time(start_ev, end_ev))

    # Stop AI
    if ai_thread:
        stop_event.set()
        ai_thread.join(timeout=10)

    arr = np.array(latencies)
    print(f"\nRESULT: {label}", flush=True)
    print(f"  RX mean:  {np.mean(arr):.3f} ms", flush=True)
    print(f"  RX P95:   {np.percentile(arr, 95):.3f} ms", flush=True)
    print(f"  RX P99:   {np.percentile(arr, 99):.3f} ms", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"exp_{label}_{ts}.json")
    with open(out, "w") as f:
        json.dump({"label": label, "num_cells": num_cells, "ai_type": ai_type,
                    "priority_mode": priority_mode,
                    "mean_ms": float(np.mean(arr)),
                    "p95_ms": float(np.percentile(arr, 95)),
                    "p99_ms": float(np.percentile(arr, 99)),
                    "raw": [float(x) for x in latencies]}, f, indent=2)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
