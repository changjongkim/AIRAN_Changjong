"""
Heavy L1 measurement — 4T4R/4T8R multi-cell, no Sionna.
Uses cuPHY TX → expand antennas + noise → cuPHY RX.
Usage: python3 run_l1_heavy.py <label> <num_tx> <num_rx> <num_cells> [mcs_index]
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

NUM_WARMUP = 10
NUM_ITERATIONS = 100
RESULTS_DIR = "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results"


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "heavy"
    num_tx = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    num_rx = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    num_cells = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    mcs_index = int(sys.argv[5]) if len(sys.argv) > 5 else 2

    stream = get_cuda_stream()
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
    hbm_pct = (total - free) / total * 100
    print(f"GPU: {props.name} ({props.multi_processor_count} SMs)", flush=True)
    print(f"L1: {num_cells} cells, {num_tx}T{num_rx}R, MCS {mcs_index}", flush=True)
    print(f"HBM: {(total-free)/1e9:.1f}/{total/1e9:.1f}GB ({hbm_pct:.0f}%)", flush=True)

    # Pre-generate TX signals + RX inputs
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

    # Measure
    trk = LatencyTracker(label)
    timer = CudaTimer()
    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        m = i >= NUM_WARMUP
        rx_ms = 0
        for ci, cell in enumerate(cells):
            timer.start()
            cell["rx"](slot=0, rx_slot=rx_inputs[ci], config=cell["ucfg"])
            timer.stop()
            rx_ms += timer.elapsed_ms()
        if m:
            trk.record(rx_ms)

    s = trk.stats()
    miss = trk.deadline_miss_rate(1.0)
    print(f"\nRESULT: {label}", flush=True)
    print(f"  RX mean:  {s['mean_ms']:.3f} ms ({s['mean_ms']/num_cells:.3f} ms/cell)", flush=True)
    print(f"  RX P95:   {s['p95_ms']:.3f} ms", flush=True)
    print(f"  RX P99:   {s['p99_ms']:.3f} ms", flush=True)
    print(f"  Jitter:   {s['jitter']:.4f}", flush=True)
    print(f"  Miss>1ms: {miss*100:.1f}%", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"exp_{label}_{ts}.json")
    with open(out, "w") as f:
        json.dump({"label": label, "num_cells": num_cells, "num_tx": num_tx,
                    "num_rx": num_rx, "mcs": mcs_index, "hbm_pct": round(hbm_pct, 1),
                    "stats": s, "miss_1ms": miss, "raw": trk.latencies}, f, indent=2)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
