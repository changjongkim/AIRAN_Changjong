"""
Heavy L1 with timed per-iteration output — for multi-node coordination.
Runs for a fixed duration (not fixed iterations) so nodes can sync.
Usage: python3 run_l1_heavy_timed.py <label> <num_tx> <num_rx> <num_cells> <duration_sec>
"""
import os, sys, json, time, datetime
sys.stdout.reconfigure(line_buffering=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

import numpy as np
import cupy as cp
import torch
import socket

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


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "timed"
    num_tx = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    num_rx = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    num_cells = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    duration_sec = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    mcs_index = 2

    hostname = socket.gethostname()
    mo, cr = get_mcs(mcs_index, 1)
    tb_sz = get_tb_size(mod_order=mo, code_rate=cr,
                        dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                        num_prbs=273, start_sym=2, num_symbols=12, num_layers=1)

    cells = []
    for ci in range(num_cells):
        cid = 41 + ci
        stream = get_cuda_stream()
        pcw = PdschCwConfig(mcs_table=0, mcs_index=mcs_index, code_rate=int(cr*10), mod_order=mo)
        pue = PdschUeConfig(cw_configs=[pcw], scid=0, dmrs_scrm_id=41,
                            layers=1, dmrs_ports=1, rnti=1234, data_scid=0)
        pcfg = PdschConfig(ue_configs=[pue], num_dmrs_cdm_grps_no_data=2,
                           start_prb=0, num_prbs=273,
                           dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                           start_sym=2, num_symbols=12)
        txp = PdschTxPipelineFactory().create(
            AerialPdschTxConfig(cell_id=cid, num_tx_ant=num_tx), stream)
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
            stream)
        cells.append({"tx": txp, "rx": rxp, "pcfg": pcfg, "ucfg": ucfg})

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    print(f"[{hostname}] GPU: {props.name}, L1: {num_cells}cells {num_tx}T{num_rx}R, "
          f"HBM: {(total-free)/1e9:.1f}/{total/1e9:.1f}GB ({(total-free)/total*100:.0f}%)",
          flush=True)

    # Pre-generate RX inputs
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

    # Measure for fixed duration
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    latencies = []
    timestamps = []
    t0 = time.time()

    # Warmup
    for _ in range(5):
        for ci, cell in enumerate(cells):
            cell["rx"](slot=0, rx_slot=rx_inputs[ci], config=cell["ucfg"])

    print(f"[{hostname}] Measuring for {duration_sec}s...", flush=True)
    t0 = time.time()
    while time.time() - t0 < duration_sec:
        start_event.record()
        for ci, cell in enumerate(cells):
            cell["rx"](slot=0, rx_slot=rx_inputs[ci], config=cell["ucfg"])
        end_event.record()
        end_event.synchronize()
        lat = cp.cuda.get_elapsed_time(start_event, end_event)
        latencies.append(lat)
        timestamps.append(time.time() - t0)

    arr = np.array(latencies)
    print(f"\n[{hostname}] RESULT: {label}", flush=True)
    print(f"  Iterations: {len(arr)}", flush=True)
    print(f"  RX mean:  {np.mean(arr):.3f} ms", flush=True)
    print(f"  RX P95:   {np.percentile(arr, 95):.3f} ms", flush=True)
    print(f"  RX P99:   {np.percentile(arr, 99):.3f} ms", flush=True)
    print(f"  Miss>1ms: {np.sum(arr > 1.0) / len(arr) * 100:.1f}%", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"exp_{label}_{ts}.json")
    with open(out, "w") as f:
        json.dump({"label": label, "hostname": hostname, "num_cells": num_cells,
                    "num_tx": num_tx, "num_rx": num_rx,
                    "mean_ms": float(np.mean(arr)), "p95_ms": float(np.percentile(arr, 95)),
                    "p99_ms": float(np.percentile(arr, 99)),
                    "latencies": [float(x) for x in latencies],
                    "timestamps": [float(x) for x in timestamps]}, f, indent=2)
    print(f"[{hostname}] Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
