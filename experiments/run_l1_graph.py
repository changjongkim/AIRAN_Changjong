"""
L1 with Multi-cell CUDA Graph — proper parallel execution.
Uses pyAerial's _pusch_config_to_cuphy for correct dynamic params,
then patches procModeBmsk for Graph mode.

Usage: python3 run_l1_graph.py <label> <num_cells> <graph_mode>
  graph_mode: 0=stream, 1=graph
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

from aerial.pycuphy import _pycuphy as pycuphy
from aerial.pycuphy.util import get_pusch_stat_prms, get_pusch_dyn_prms_phase_2
from aerial.pycuphy.types import (
    CellStatPrm, PuschLdpcKernelLaunch,
)
from aerial.phy5g.config import (
    _pusch_config_to_cuphy, PuschConfig, PuschUeConfig,
    AerialPdschTxConfig, PdschConfig, PdschUeConfig, PdschCwConfig,
)
from aerial.phy5g.pdsch import PdschTxPipelineFactory
from aerial.phy5g.ldpc import get_mcs, get_tb_size, random_tb
from aerial.util.cuda import get_cuda_stream, check_cuda_errors
from cuda import cudart

RESULTS_DIR = "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results"
NUM_WARMUP = 10
NUM_ITERATIONS = 50

DMRS_SYMS = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "graph_test"
    num_cells = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    graph_mode = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    num_tx = 4
    num_rx = 4
    mcs_index = 2
    mo, cr = get_mcs(mcs_index, 1)
    tb_sz = get_tb_size(mod_order=mo, code_rate=cr, dmrs_syms=DMRS_SYMS,
                        num_prbs=273, start_sym=2, num_symbols=12, num_layers=1)
    mode_str = "GRAPH" if graph_mode else "STREAM"

    stream = get_cuda_stream()

    # Create multi-cell pipeline
    cell_stat_prms = [CellStatPrm(
        phyCellId=np.uint16(41 + ci), nRxAnt=np.uint16(num_rx),
        nTxAnt=np.uint16(num_tx), nRxAntSrs=np.uint16(num_rx),
        nPrbUlBwp=np.uint16(273), nPrbDlBwp=np.uint16(273), mu=np.uint8(1),
    ) for ci in range(num_cells)]

    base_stat = get_pusch_stat_prms(
        cell_id=41, num_rx_ant=num_rx, num_tx_ant=num_tx,
        ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL)
    multicell_stat = base_stat._replace(
        nMaxCells=np.uint16(num_cells),
        nMaxCellsPerSlot=np.uint16(num_cells),
        cellStatPrms=cell_stat_prms,
        enableDeviceGraphLaunch=np.uint8(1 if graph_mode else 0))
    pipeline = pycuphy.PuschPipeline(multicell_stat, stream)

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU: {props.name}", flush=True)
    print(f"L1: {num_cells}cells 4T4R {mode_str}, HBM={(total-free)/1e9:.1f}GB", flush=True)

    # Generate RX data per cell using TX pipeline
    pusch_configs_per_cell = []
    rx_slots = []
    for ci in range(num_cells):
        cid = 41 + ci
        pcw = PdschCwConfig(mcs_table=0, mcs_index=mcs_index,
                            code_rate=int(cr*10), mod_order=mo)
        pue = PdschUeConfig(cw_configs=[pcw], scid=0, dmrs_scrm_id=41,
                            layers=1, dmrs_ports=1, rnti=1234, data_scid=0)
        pcfg = PdschConfig(ue_configs=[pue], num_dmrs_cdm_grps_no_data=2,
                           start_prb=0, num_prbs=273, dmrs_syms=DMRS_SYMS,
                           start_sym=2, num_symbols=12)
        txp = PdschTxPipelineFactory().create(
            AerialPdschTxConfig(cell_id=cid, num_tx_ant=num_tx), stream)
        tb = cp.array(random_tb(mod_order=mo, code_rate=cr, dmrs_syms=DMRS_SYMS,
                                num_prbs=273, start_sym=2, num_symbols=12, num_layers=1),
                      dtype=cp.uint8, order="F")
        tx_out = txp(slot=0, tb_inputs=[tb], config=[pcfg])
        if num_rx > tx_out.shape[2]:
            reps = (num_rx // tx_out.shape[2]) + 1
            rx_in = cp.concatenate([tx_out] * reps, axis=2)[:, :, :num_rx]
        else:
            rx_in = tx_out[:, :, :num_rx]
        noise = 0.01 * (cp.random.randn(*rx_in.shape, dtype=cp.float32) +
                1j * cp.random.randn(*rx_in.shape, dtype=cp.float32)).astype(cp.complex64)
        rx_slots.append(cp.asfortranarray(rx_in + noise))
        del txp

        uue = PuschUeConfig(scid=0, layers=1, dmrs_ports=1, rnti=1234,
                            data_scid=0, mcs_table=0, mcs_index=mcs_index,
                            code_rate=int(cr*10), mod_order=mo, tb_size=tb_sz//8)
        pusch_configs_per_cell.append(PuschConfig(
            ue_configs=[uue], num_dmrs_cdm_grps_no_data=2,
            dmrs_scrm_id=41, start_prb=0, num_prbs=273, dmrs_syms=DMRS_SYMS,
            dmrs_max_len=1, dmrs_add_ln_pos=0, start_sym=2, num_symbols=12))

    # Build dynamic params using pyAerial's function
    dyn_prms = _pusch_config_to_cuphy(
        cuda_stream=stream, rx_data=rx_slots, slot=0,
        pusch_configs=pusch_configs_per_cell)

    # Patch procModeBmsk for graph mode
    if graph_mode:
        dyn_prms = dyn_prms._replace(procModeBmsk=np.uint64(1))

    # Setup phase 1
    pipeline.setup_pusch_rx(dyn_prms)

    # Setup phase 2 — HARQ buffers
    num_ues = num_cells
    harq_buffers = []
    for ue_idx in range(num_ues):
        hsz = dyn_prms.dataOut.harqBufferSizeInBytes[ue_idx]
        hbuf = check_cuda_errors(cudart.cudaMalloc(hsz))
        check_cuda_errors(cudart.cudaMemsetAsync(hbuf, 0, hsz, stream))
        check_cuda_errors(cudart.cudaStreamSynchronize(stream))
        harq_buffers.append(hbuf)

    dyn_prms2 = get_pusch_dyn_prms_phase_2(dyn_prms, harq_buffers)
    pipeline.setup_pusch_rx(dyn_prms2)

    # Measure
    start_ev = cp.cuda.Event()
    end_ev = cp.cuda.Event()
    latencies = []

    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        pipeline.setup_pusch_rx(dyn_prms)
        pipeline.setup_pusch_rx(dyn_prms2)

        start_ev.record()
        pipeline.run_pusch_rx()
        end_ev.record()
        end_ev.synchronize()

        if i >= NUM_WARMUP:
            latencies.append(cp.cuda.get_elapsed_time(start_ev, end_ev))

    for hbuf in harq_buffers:
        check_cuda_errors(cudart.cudaFree(hbuf))

    arr = np.array(latencies)
    print(f"\nRESULT: {label}", flush=True)
    print(f"  Mode:     {mode_str}", flush=True)
    print(f"  Cells:    {num_cells}", flush=True)
    print(f"  RX mean:  {np.mean(arr):.3f} ms ({np.mean(arr)/num_cells:.3f} ms/cell)", flush=True)
    print(f"  RX P95:   {np.percentile(arr, 95):.3f} ms", flush=True)
    print(f"  RX P99:   {np.percentile(arr, 99):.3f} ms", flush=True)
    miss = float(np.sum(arr > 1.0) / len(arr))
    print(f"  Miss>1ms: {miss*100:.1f}%", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"exp_{label}_{ts}.json")
    with open(out, "w") as f:
        json.dump({"label": label, "num_cells": num_cells, "graph_mode": graph_mode,
                    "mode": mode_str, "mean_ms": float(np.mean(arr)),
                    "per_cell_ms": float(np.mean(arr) / num_cells),
                    "p95_ms": float(np.percentile(arr, 95)),
                    "p99_ms": float(np.percentile(arr, 99)),
                    "miss_1ms": miss, "raw": [float(x) for x in latencies]}, f, indent=2)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
