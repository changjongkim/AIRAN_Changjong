"""
Heavy L1 scale test v2 — multi-cell with proper signal generation.
Uses cuPHY TX pipeline per cell, then adds noise for RX input.
No Sionna dependency.
"""
import os, sys, time, json, datetime
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
from utils.timing import CudaTimer

RESULTS_DIR = "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results"
stream = get_cuda_stream()
timer = CudaTimer()

NUM_WARMUP = 5
NUM_MEASURE = 30

# Test configs: (num_tx, num_rx, num_cells, mcs_index)
configs = [
    (1, 2, 1, 2, "1T2R_1cell_mcs2"),
    (1, 2, 4, 2, "1T2R_4cell_mcs2"),
    (1, 2, 8, 2, "1T2R_8cell_mcs2"),
    (1, 2, 16, 2, "1T2R_16cell_mcs2"),
    (1, 2, 20, 2, "1T2R_20cell_mcs2"),
    (1, 4, 1, 2, "1T4R_1cell_mcs2"),
    (1, 4, 4, 2, "1T4R_4cell_mcs2"),
    (1, 4, 8, 2, "1T4R_8cell_mcs2"),
    (1, 4, 16, 2, "1T4R_16cell_mcs2"),
    (1, 8, 1, 2, "1T8R_1cell_mcs2"),
    (1, 8, 4, 2, "1T8R_4cell_mcs2"),
    (1, 8, 8, 2, "1T8R_8cell_mcs2"),
    (1, 16, 1, 2, "1T16R_1cell_mcs2"),
    (1, 16, 4, 2, "1T16R_4cell_mcs2"),
    (1, 2, 8, 15, "1T2R_8cell_mcs15"),
    (1, 2, 16, 15, "1T2R_16cell_mcs15"),
]

print("=" * 60, flush=True)
print("Heavy L1 Scale Test v2", flush=True)
print("=" * 60, flush=True)

props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name} ({props.multi_processor_count} SMs)", flush=True)
free, total = torch.cuda.mem_get_info(0)
print(f"HBM: {total/1e9:.1f}GB", flush=True)
print(f"Warmup: {NUM_WARMUP}, Measure: {NUM_MEASURE}", flush=True)
print("", flush=True)

results = []

for num_tx, num_rx, num_cells, mcs_index, label in configs:
    try:
        mo, cr = get_mcs(mcs_index, 1)
        tb_sz = get_tb_size(mod_order=mo, code_rate=cr,
                            dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                            num_prbs=273, start_sym=2, num_symbols=12, num_layers=1)

        # Create cell pipelines
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

        free_after, _ = torch.cuda.mem_get_info(0)
        hbm_pct = (total - free_after) / total * 100

        tb = cp.array(random_tb(mod_order=mo, code_rate=cr,
                                dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                num_prbs=273, start_sym=2, num_symbols=12, num_layers=1),
                      dtype=cp.uint8, order="F")

        # Pre-generate TX signals for all cells
        tx_signals = []
        for cell in cells:
            tx_t = cell["tx"](slot=0, tb_inputs=[tb], config=[cell["pcfg"]])
            # Expand to match RX antenna count by repeating + adding noise
            # tx_t shape: (subcarriers, symbols, num_tx)
            # Need: (subcarriers, symbols, num_rx)
            if num_rx > num_tx:
                rx_input = cp.concatenate([tx_t] * num_rx, axis=2)[:, :, :num_rx]
            else:
                rx_input = tx_t[:, :, :num_rx]
            # Add noise
            noise = 0.1 * (cp.random.randn(*rx_input.shape, dtype=cp.float32) +
                           1j * cp.random.randn(*rx_input.shape, dtype=cp.float32)).astype(cp.complex64)
            rx_input = rx_input + noise
            tx_signals.append(rx_input)

        # Warmup + Measure
        latencies = []
        for i in range(NUM_WARMUP + NUM_MEASURE):
            rx_ms = 0
            for ci, cell in enumerate(cells):
                timer.start()
                cell["rx"](slot=0, rx_slot=tx_signals[ci], config=cell["ucfg"])
                timer.stop()
                rx_ms += timer.elapsed_ms()
            if i >= NUM_WARMUP:
                latencies.append(rx_ms)

        arr = np.array(latencies)
        print(f"[OK] {label}: {num_cells}cells × {num_tx}T{num_rx}R MCS{mcs_index}  "
              f"HBM={hbm_pct:.0f}%  RX={np.mean(arr):.3f}ms  "
              f"({np.mean(arr)/num_cells:.3f}ms/cell)  "
              f"P99={np.percentile(arr,99):.3f}ms", flush=True)

        results.append({
            "label": label, "num_cells": num_cells,
            "num_tx": num_tx, "num_rx": num_rx, "mcs": mcs_index,
            "hbm_pct": round(hbm_pct, 1),
            "mean_ms": float(np.mean(arr)),
            "per_cell_ms": float(np.mean(arr) / num_cells),
            "p99_ms": float(np.percentile(arr, 99)),
        })

        # Cleanup
        del cells, tx_signals
        import gc; gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[FAIL] {label}: {e}", flush=True)
        results.append({"label": label, "error": str(e)})

# Summary
print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"{'Config':<24} {'HBM%':>5} | {'RX mean':>9} | {'per cell':>9} | {'P99':>9}", flush=True)
print("-" * 65, flush=True)
for r in results:
    if "error" in r:
        print(f"{r['label']:<24} FAIL: {r['error'][:40]}", flush=True)
    else:
        print(f"{r['label']:<24} {r['hbm_pct']:>4.0f}% | {r['mean_ms']:>7.3f}ms | "
              f"{r['per_cell_ms']:>7.3f}ms | {r['p99_ms']:>7.3f}ms", flush=True)

# Save
os.makedirs(RESULTS_DIR, exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out = os.path.join(RESULTS_DIR, f"exp_heavy_l1_v2_{ts}.json")
with open(out, "w") as f:
    json.dump({"results": results}, f, indent=2)
print(f"\nSaved: {out}", flush=True)
