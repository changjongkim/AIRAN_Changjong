"""
Per-node L1 (CUDA Graph) + AI workload measurement.
Each node runs this independently. Results collected per-node.

Usage: python3 run_node_l1_ai.py <label> <num_cells> <ai_type> <ai_intensity> <duration_sec>
  ai_type: none / neural_rx / gpt2 / resnet
"""
import os, sys, json, time, datetime, threading, socket
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
from aerial.pycuphy.types import CellStatPrm, PuschLdpcKernelLaunch
from aerial.phy5g.config import (
    _pusch_config_to_cuphy, PuschConfig, PuschUeConfig,
    AerialPdschTxConfig, PdschConfig, PdschUeConfig, PdschCwConfig,
)
from aerial.phy5g.pdsch import PdschTxPipelineFactory
from aerial.phy5g.ldpc import get_mcs, get_tb_size, random_tb
from aerial.util.cuda import get_cuda_stream, check_cuda_errors
from cuda import cudart

RESULTS_DIR = "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results"
DMRS_SYMS = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def run_ai_background(ai_type, intensity, stop_event):
    """Background AI on same GPU."""
    if ai_type == "neural_rx":
        import torch.nn as nn
        channels = int(64 * intensity)
        input_size = int(4096 * intensity)
        batch = max(1, int(16 * intensity))
        model = nn.Sequential(
            nn.Conv1d(channels, channels*2, 7, padding=3), nn.ReLU(),
            nn.Conv1d(channels*2, channels*2, 7, padding=3), nn.ReLU(),
            nn.Conv1d(channels*2, channels, 3, padding=1),
        ).cuda().eval()
        inp = torch.randn(batch, channels, input_size, device="cuda")
        with torch.no_grad():
            for _ in range(5): model(inp)
        torch.cuda.synchronize()
        c = 0
        with torch.no_grad():
            while not stop_event.is_set():
                out = model(inp)
                inp[:, :out.shape[1], :out.shape[2]].copy_(out)
                c += 1
    elif ai_type == "gpt2":
        from transformers import GPT2LMHeadModel, GPT2Config
        cfg = GPT2Config()
        model = GPT2LMHeadModel(cfg).cuda().eval()
        dummy = torch.randint(0, cfg.vocab_size, (4, 512), device="cuda")
        with torch.no_grad():
            for _ in range(3): model(dummy)
        torch.cuda.synchronize()
        c = 0
        with torch.no_grad():
            while not stop_event.is_set(): model(dummy); c += 1
    elif ai_type == "resnet":
        import torchvision.models as models
        model = models.resnet50(weights=None).cuda().eval()
        dummy = torch.randn(64, 3, 224, 224, device="cuda")
        with torch.no_grad():
            for _ in range(5): model(dummy)
        torch.cuda.synchronize()
        c = 0
        with torch.no_grad():
            while not stop_event.is_set(): model(dummy); c += 1
    elif ai_type == "none":
        while not stop_event.is_set(): time.sleep(0.1)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "node_test"
    num_cells = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    ai_type = sys.argv[3] if len(sys.argv) > 3 else "none"
    ai_intensity = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    duration_sec = int(sys.argv[5]) if len(sys.argv) > 5 else 45

    hostname = socket.gethostname()
    num_tx, num_rx, mcs_index = 4, 4, 2
    mo, cr = get_mcs(mcs_index, 1)
    tb_sz = get_tb_size(mod_order=mo, code_rate=cr, dmrs_syms=DMRS_SYMS,
                        num_prbs=273, start_sym=2, num_symbols=12, num_layers=1)
    stream = get_cuda_stream()

    # Multi-cell CUDA Graph pipeline
    cell_stat_prms = [CellStatPrm(
        phyCellId=np.uint16(41+ci), nRxAnt=np.uint16(num_rx),
        nTxAnt=np.uint16(num_tx), nRxAntSrs=np.uint16(num_rx),
        nPrbUlBwp=np.uint16(273), nPrbDlBwp=np.uint16(273), mu=np.uint8(1),
    ) for ci in range(num_cells)]

    base_stat = get_pusch_stat_prms(cell_id=41, num_rx_ant=num_rx, num_tx_ant=num_tx,
        ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL)
    multicell_stat = base_stat._replace(
        nMaxCells=np.uint16(num_cells), nMaxCellsPerSlot=np.uint16(num_cells),
        cellStatPrms=cell_stat_prms, enableDeviceGraphLaunch=np.uint8(1))
    pipeline = pycuphy.PuschPipeline(multicell_stat, stream)

    # Generate RX data
    pusch_configs = []
    rx_slots = []
    for ci in range(num_cells):
        txp = PdschTxPipelineFactory().create(
            AerialPdschTxConfig(cell_id=41+ci, num_tx_ant=num_tx), stream)
        pcw = PdschCwConfig(mcs_table=0, mcs_index=mcs_index, code_rate=int(cr*10), mod_order=mo)
        pue = PdschUeConfig(cw_configs=[pcw], scid=0, dmrs_scrm_id=41,
                            layers=1, dmrs_ports=1, rnti=1234, data_scid=0)
        pcfg = PdschConfig(ue_configs=[pue], num_dmrs_cdm_grps_no_data=2,
                           start_prb=0, num_prbs=273, dmrs_syms=DMRS_SYMS,
                           start_sym=2, num_symbols=12)
        tb = cp.array(random_tb(mod_order=mo, code_rate=cr, dmrs_syms=DMRS_SYMS,
                                num_prbs=273, start_sym=2, num_symbols=12, num_layers=1),
                      dtype=cp.uint8, order="F")
        tx_out = txp(slot=0, tb_inputs=[tb], config=[pcfg])
        reps = (num_rx // tx_out.shape[2]) + 1
        rx_in = cp.concatenate([tx_out]*reps, axis=2)[:,:,:num_rx]
        noise = 0.01*(cp.random.randn(*rx_in.shape, dtype=cp.float32) +
                1j*cp.random.randn(*rx_in.shape, dtype=cp.float32)).astype(cp.complex64)
        rx_slots.append(cp.asfortranarray(rx_in + noise))
        del txp
        uue = PuschUeConfig(scid=0, layers=1, dmrs_ports=1, rnti=1234, data_scid=0,
                            mcs_table=0, mcs_index=mcs_index, code_rate=int(cr*10),
                            mod_order=mo, tb_size=tb_sz//8)
        pusch_configs.append(PuschConfig(ue_configs=[uue], num_dmrs_cdm_grps_no_data=2,
            dmrs_scrm_id=41, start_prb=0, num_prbs=273, dmrs_syms=DMRS_SYMS,
            dmrs_max_len=1, dmrs_add_ln_pos=0, start_sym=2, num_symbols=12))

    dyn_prms = _pusch_config_to_cuphy(cuda_stream=stream, rx_data=rx_slots,
                                       slot=0, pusch_configs=pusch_configs)
    dyn_prms = dyn_prms._replace(procModeBmsk=np.uint64(1))
    pipeline.setup_pusch_rx(dyn_prms)

    harq_buffers = []
    for ue_idx in range(num_cells):
        hsz = dyn_prms.dataOut.harqBufferSizeInBytes[ue_idx]
        hbuf = check_cuda_errors(cudart.cudaMalloc(hsz))
        check_cuda_errors(cudart.cudaMemsetAsync(hbuf, 0, hsz, stream))
        check_cuda_errors(cudart.cudaStreamSynchronize(stream))
        harq_buffers.append(hbuf)
    dyn_prms2 = get_pusch_dyn_prms_phase_2(dyn_prms, harq_buffers)
    pipeline.setup_pusch_rx(dyn_prms2)

    free, total = torch.cuda.mem_get_info(0)
    print(f"[{hostname}] L1: {num_cells}cells GRAPH, AI: {ai_type}({ai_intensity}), "
          f"HBM={(total-free)/1e9:.1f}/{total/1e9:.1f}GB", flush=True)

    # Start AI
    stop_event = threading.Event()
    if ai_type != "none":
        t = threading.Thread(target=run_ai_background,
                             args=(ai_type, ai_intensity, stop_event), daemon=True)
        t.start()
        time.sleep(5)

    # Measure for duration
    start_ev = cp.cuda.Event()
    end_ev = cp.cuda.Event()
    latencies = []
    t0 = time.time()

    # Warmup
    for _ in range(10):
        pipeline.setup_pusch_rx(dyn_prms)
        pipeline.setup_pusch_rx(dyn_prms2)
        pipeline.run_pusch_rx()

    t0 = time.time()
    while time.time() - t0 < duration_sec:
        pipeline.setup_pusch_rx(dyn_prms)
        pipeline.setup_pusch_rx(dyn_prms2)
        start_ev.record()
        pipeline.run_pusch_rx()
        end_ev.record()
        end_ev.synchronize()
        latencies.append(cp.cuda.get_elapsed_time(start_ev, end_ev))

    stop_event.set()
    for hbuf in harq_buffers:
        check_cuda_errors(cudart.cudaFree(hbuf))

    arr = np.array(latencies)
    miss = float(np.sum(arr > 1.0) / len(arr))
    print(f"[{hostname}] RESULT: {label}", flush=True)
    print(f"  Iters:    {len(arr)}", flush=True)
    print(f"  RX mean:  {np.mean(arr):.3f} ms", flush=True)
    print(f"  RX P95:   {np.percentile(arr, 95):.3f} ms", flush=True)
    print(f"  RX P99:   {np.percentile(arr, 99):.3f} ms", flush=True)
    print(f"  Miss>1ms: {miss*100:.1f}%", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"exp_{label}_{hostname}_{ts}.json")
    with open(out, "w") as f:
        json.dump({"label": label, "hostname": hostname, "num_cells": num_cells,
                    "ai_type": ai_type, "ai_intensity": ai_intensity,
                    "mean_ms": float(np.mean(arr)), "p95_ms": float(np.percentile(arr, 95)),
                    "p99_ms": float(np.percentile(arr, 99)), "miss_1ms": miss,
                    "num_iters": len(arr), "raw": [float(x) for x in latencies]}, f, indent=2)
    print(f"[{hostname}] Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
