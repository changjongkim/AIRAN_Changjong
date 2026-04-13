"""
Step 1: L1 TTI Timing — How long does L1 take per TTI?
Measures exact L1 burst duration and idle time available for AI.
"""
import os, sys, time
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


def measure_l1_timing(num_tx, num_rx, num_cells, label):
    """Measure L1 burst duration for given config."""
    stream = get_cuda_stream()
    mo, cr = get_mcs(2, 1)
    tb_sz = get_tb_size(mod_order=mo, code_rate=cr,
                        dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                        num_prbs=273, start_sym=2, num_symbols=12, num_layers=1)

    cells = []
    for ci in range(num_cells):
        cid = 41 + ci
        pcw = PdschCwConfig(mcs_table=0, mcs_index=2, code_rate=int(cr*10), mod_order=mo)
        pue = PdschUeConfig(cw_configs=[pcw], scid=0, dmrs_scrm_id=41,
                            layers=1, dmrs_ports=1, rnti=1234, data_scid=0)
        pcfg = PdschConfig(ue_configs=[pue], num_dmrs_cdm_grps_no_data=2,
                           start_prb=0, num_prbs=273,
                           dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                           start_sym=2, num_symbols=12)
        txp = PdschTxPipelineFactory().create(
            AerialPdschTxConfig(cell_id=cid, num_tx_ant=num_tx), stream)
        uue = PuschUeConfig(scid=0, layers=1, dmrs_ports=1, rnti=1234,
                            data_scid=0, mcs_table=0, mcs_index=2,
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

    # Pre-generate inputs
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

    # Measure per-cell and total L1 burst time
    start_ev = cp.cuda.Event()
    end_ev = cp.cuda.Event()

    per_cell_times = []
    total_times = []

    for iteration in range(50):
        # Total L1 burst
        start_ev.record()
        for ci, cell in enumerate(cells):
            cell["rx"](slot=0, rx_slot=rx_inputs[ci], config=cell["ucfg"])
        end_ev.record()
        end_ev.synchronize()
        total_ms = cp.cuda.get_elapsed_time(start_ev, end_ev)

        if iteration >= 10:  # skip warmup
            total_times.append(total_ms)

    total_arr = np.array(total_times)

    TTI_MS = 1.0  # 30kHz SCS = 1ms TTI (0.5ms slot)
    idle_ms = max(0, TTI_MS - np.mean(total_arr))
    idle_pct = idle_ms / TTI_MS * 100

    print(f"  {label}: L1 burst={np.mean(total_arr):.3f}ms "
          f"(P99={np.percentile(total_arr, 99):.3f}ms)  "
          f"TTI idle={idle_ms:.3f}ms ({idle_pct:.0f}%)", flush=True)

    # Cleanup
    del cells, rx_inputs
    import gc; gc.collect()
    torch.cuda.empty_cache()

    return {
        "label": label, "num_cells": num_cells, "num_tx": num_tx, "num_rx": num_rx,
        "burst_mean_ms": float(np.mean(total_arr)),
        "burst_p99_ms": float(np.percentile(total_arr, 99)),
        "tti_ms": TTI_MS,
        "idle_ms": float(idle_ms),
        "idle_pct": float(idle_pct),
    }


def main():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}", flush=True)
    print(f"TTI = 1ms (30kHz SCS)", flush=True)
    print("", flush=True)

    configs = [
        (1, 2, 1, "1T2R_1cell"),
        (1, 2, 4, "1T2R_4cell"),
        (1, 2, 8, "1T2R_8cell"),
        (4, 4, 1, "4T4R_1cell"),
        (4, 4, 4, "4T4R_4cell"),
        (4, 4, 8, "4T4R_8cell"),
        (4, 4, 20, "4T4R_20cell"),
    ]

    print("=== L1 Burst Duration vs TTI Budget ===", flush=True)
    results = []
    for num_tx, num_rx, num_cells, label in configs:
        try:
            r = measure_l1_timing(num_tx, num_rx, num_cells, label)
            results.append(r)
        except Exception as e:
            print(f"  {label}: FAIL — {e}", flush=True)

    print("", flush=True)
    print("=== SUMMARY ===", flush=True)
    print(f"{'Config':<18} {'L1 burst':>9} {'TTI idle':>9} {'Idle %':>7} {'AI feasible?':>14}",
          flush=True)
    print("-" * 60, flush=True)
    for r in results:
        feasible = "YES" if r["idle_ms"] > 0.1 else "TIGHT" if r["idle_ms"] > 0 else "NO"
        print(f"{r['label']:<18} {r['burst_mean_ms']:>7.3f}ms {r['idle_ms']:>7.3f}ms "
              f"{r['idle_pct']:>5.0f}%   {feasible:>12}", flush=True)

    import json, datetime, os
    os.makedirs("/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results/exp_tti_timing_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
