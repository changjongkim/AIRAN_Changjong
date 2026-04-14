"""
BORA Full Evaluation — L1 latency + AI throughput simultaneously.
Measures both sides of the trade-off:
  - L1: per-iteration latency, TTI miss rate
  - AI: total inference count during measurement period (throughput)

Usage: python3 run_bora_full.py <label> <cells> <config> <ai_type> <duration_sec>
  config: A/B/C/D
  ai_type: none/gpt2/resnet/neuralrx
"""
import os, sys, json, time, datetime, socket, threading
sys.stdout.reconfigure(line_buffering=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

import numpy as np
import cupy as cp
import torch
from cuda import cudart
from multiprocessing import shared_memory

RESULTS_DIR = "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results"


def parse_config(config_str):
    """Parse BORA config string into parameters."""
    configs = {
        "A": {"sm_l1": 100, "sm_ai": 100, "prio_l1": 0, "prio_ai": 0, "tti": False},
        "B": {"sm_l1": 40, "sm_ai": 60, "prio_l1": 0, "prio_ai": 0, "tti": False},
        "C": {"sm_l1": 40, "sm_ai": 60, "prio_l1": -5, "prio_ai": 0, "tti": False},
        "D": {"sm_l1": 40, "sm_ai": 60, "prio_l1": -5, "prio_ai": 0, "tti": True},
    }
    return configs[config_str]


def run_ai_worker(ai_type, stop_event, result_dict, cfg):
    """AI worker: runs inference and counts throughput."""
    # Create stream with priority
    err, ai_stream = cudart.cudaStreamCreateWithPriority(1, cfg["prio_ai"])

    # TTI coordination
    shm = None
    if cfg["tti"]:
        try:
            shm = shared_memory.SharedMemory(name="bora_tti", create=False, size=8)
        except:
            pass

    if ai_type == "gpt2":
        os.environ["HF_HOME"] = "/pscratch/sd/s/sgkim/kcj/AI-RAN/datasets/models"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import GPT2LMHeadModel, GPT2Config
        model_cfg = GPT2Config()
        model = GPT2LMHeadModel(model_cfg).cuda().eval()
        dummy = torch.randint(0, model_cfg.vocab_size, (4, 512), device="cuda")
        with torch.no_grad():
            for _ in range(3):
                model(dummy)
        torch.cuda.synchronize()
        c = 0
        with torch.no_grad():
            while not stop_event.is_set():
                if shm and shm.buf[0] == 1:
                    time.sleep(0.0001)
                    continue
                with torch.cuda.stream(torch.cuda.ExternalStream(int(ai_stream))):
                    model(dummy)
                c += 1
        result_dict["ai_iters"] = c

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
                if shm and shm.buf[0] == 1:
                    time.sleep(0.0001)
                    continue
                with torch.cuda.stream(torch.cuda.ExternalStream(int(ai_stream))):
                    model(dummy)
                c += 1
        result_dict["ai_iters"] = c

    elif ai_type == "neuralrx":
        import torch.nn as nn
        model = nn.Sequential(
            nn.Conv1d(128, 256, 7, padding=3), nn.ReLU(),
            nn.Conv1d(256, 256, 7, padding=3), nn.ReLU(),
            nn.Conv1d(256, 128, 3, padding=1),
        ).cuda().eval()
        inp = torch.randn(16, 128, 4096, device="cuda")
        with torch.no_grad():
            for _ in range(5):
                model(inp)
        torch.cuda.synchronize()
        c = 0
        with torch.no_grad():
            while not stop_event.is_set():
                if shm and shm.buf[0] == 1:
                    time.sleep(0.0001)
                    continue
                with torch.cuda.stream(torch.cuda.ExternalStream(int(ai_stream))):
                    out = model(inp)
                    inp[:, :out.shape[1], :out.shape[2]].copy_(out)
                c += 1
        result_dict["ai_iters"] = c

    else:
        result_dict["ai_iters"] = 0


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "bora_test"
    cells = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    config_str = sys.argv[3] if len(sys.argv) > 3 else "A"
    ai_type = sys.argv[4] if len(sys.argv) > 4 else "none"
    duration = int(sys.argv[5]) if len(sys.argv) > 5 else 30

    cfg = parse_config(config_str)
    hostname = socket.gethostname()

    # L1 stream with priority
    err, l1_stream = cudart.cudaStreamCreateWithPriority(1, cfg["prio_l1"])

    # TTI shared memory
    shm = None
    if cfg["tti"]:
        try:
            shm = shared_memory.SharedMemory(name="bora_tti", create=True, size=8)
            shm.buf[0] = 0
        except FileExistsError:
            shm = shared_memory.SharedMemory(name="bora_tti", create=False, size=8)

    # Setup L1 pipeline (CUDA Graph, multi-cell)
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

    num_tx, num_rx, mcs = 4, 4, 2
    mo, cr = get_mcs(mcs, 1)
    tb_sz = get_tb_size(mod_order=mo, code_rate=cr,
                        dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                        num_prbs=273, start_sym=2, num_symbols=12, num_layers=1)

    cell_stat_prms = [CellStatPrm(
        phyCellId=np.uint16(41+ci), nRxAnt=np.uint16(num_rx),
        nTxAnt=np.uint16(num_tx), nRxAntSrs=np.uint16(num_rx),
        nPrbUlBwp=np.uint16(273), nPrbDlBwp=np.uint16(273), mu=np.uint8(1),
    ) for ci in range(cells)]

    base_stat = get_pusch_stat_prms(cell_id=41, num_rx_ant=num_rx, num_tx_ant=num_tx,
        ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL)
    multicell_stat = base_stat._replace(
        nMaxCells=np.uint16(cells), nMaxCellsPerSlot=np.uint16(cells),
        cellStatPrms=cell_stat_prms, enableDeviceGraphLaunch=np.uint8(1))
    pipeline = pycuphy.PuschPipeline(multicell_stat, l1_stream)

    pusch_configs = []
    rx_slots = []
    for ci in range(cells):
        txp = PdschTxPipelineFactory().create(
            AerialPdschTxConfig(cell_id=41+ci, num_tx_ant=num_tx), l1_stream)
        pcw = PdschCwConfig(mcs_table=0, mcs_index=mcs, code_rate=int(cr*10), mod_order=mo)
        pue = PdschUeConfig(cw_configs=[pcw], scid=0, dmrs_scrm_id=41,
                            layers=1, dmrs_ports=1, rnti=1234, data_scid=0)
        pcfg = PdschConfig(ue_configs=[pue], num_dmrs_cdm_grps_no_data=2,
                           start_prb=0, num_prbs=273,
                           dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                           start_sym=2, num_symbols=12)
        tb = cp.array(random_tb(mod_order=mo, code_rate=cr,
                                dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
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
                            mcs_table=0, mcs_index=mcs, code_rate=int(cr*10),
                            mod_order=mo, tb_size=tb_sz//8)
        pusch_configs.append(PuschConfig(ue_configs=[uue], num_dmrs_cdm_grps_no_data=2,
            dmrs_scrm_id=41, start_prb=0, num_prbs=273,
            dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
            dmrs_max_len=1, dmrs_add_ln_pos=0, start_sym=2, num_symbols=12))

    dyn_prms = _pusch_config_to_cuphy(cuda_stream=l1_stream, rx_data=rx_slots,
                                       slot=0, pusch_configs=pusch_configs)
    dyn_prms = dyn_prms._replace(procModeBmsk=np.uint64(1))
    pipeline.setup_pusch_rx(dyn_prms)

    harq_buffers = []
    for ue_idx in range(cells):
        hsz = dyn_prms.dataOut.harqBufferSizeInBytes[ue_idx]
        hbuf = check_cuda_errors(cudart.cudaMalloc(hsz))
        check_cuda_errors(cudart.cudaMemsetAsync(hbuf, 0, hsz, l1_stream))
        check_cuda_errors(cudart.cudaStreamSynchronize(l1_stream))
        harq_buffers.append(hbuf)
    dyn_prms2 = get_pusch_dyn_prms_phase_2(dyn_prms, harq_buffers)
    pipeline.setup_pusch_rx(dyn_prms2)

    free, total = torch.cuda.mem_get_info(0)
    print(f"[{hostname}] Config {config_str}: AI={ai_type}, "
          f"SM(L1={cfg['sm_l1']}%,AI={cfg['sm_ai']}%), "
          f"Prio(L1={cfg['prio_l1']},AI={cfg['prio_ai']}), TTI={'ON' if cfg['tti'] else 'OFF'}, "
          f"HBM={(total-free)/1e9:.1f}/{total/1e9:.1f}GB", flush=True)

    # Start AI in background thread
    stop_event = threading.Event()
    ai_result = {"ai_iters": 0}
    ai_thread = None
    if ai_type != "none":
        ai_thread = threading.Thread(target=run_ai_worker,
                                     args=(ai_type, stop_event, ai_result, cfg),
                                     daemon=True)
        ai_thread.start()
        time.sleep(5)

    # Measure L1 for fixed duration
    start_ev = cp.cuda.Event()
    end_ev = cp.cuda.Event()
    latencies = []

    # Warmup
    for _ in range(10):
        if shm: shm.buf[0] = 1
        pipeline.setup_pusch_rx(dyn_prms)
        pipeline.setup_pusch_rx(dyn_prms2)
        pipeline.run_pusch_rx()
        if shm: shm.buf[0] = 0

    t0 = time.time()
    while time.time() - t0 < duration:
        if shm: shm.buf[0] = 1  # L1 ACTIVE
        pipeline.setup_pusch_rx(dyn_prms)
        pipeline.setup_pusch_rx(dyn_prms2)
        start_ev.record()
        pipeline.run_pusch_rx()
        end_ev.record()
        end_ev.synchronize()
        if shm: shm.buf[0] = 0  # L1 IDLE
        latencies.append(cp.cuda.get_elapsed_time(start_ev, end_ev))

    measure_time = time.time() - t0

    # Stop AI
    stop_event.set()
    if ai_thread:
        ai_thread.join(timeout=10)

    # Cleanup
    for hbuf in harq_buffers:
        check_cuda_errors(cudart.cudaFree(hbuf))
    if shm:
        shm.close()
        try:
            shm.unlink()
        except:
            pass

    # Results
    arr = np.array(latencies)
    miss = float(np.sum(arr > 1.0) / len(arr))
    ai_throughput = ai_result["ai_iters"] / measure_time if measure_time > 0 else 0

    print(f"\nRESULT: {label}", flush=True)
    print(f"  Config:       {config_str}", flush=True)
    print(f"  L1 RX mean:   {np.mean(arr):.3f} ms", flush=True)
    print(f"  L1 RX P99:    {np.percentile(arr, 99):.3f} ms", flush=True)
    print(f"  TTI miss:     {miss*100:.1f}%", flush=True)
    print(f"  AI type:      {ai_type}", flush=True)
    print(f"  AI throughput: {ai_throughput:.1f} inf/s", flush=True)
    print(f"  AI total:     {ai_result['ai_iters']} iters in {measure_time:.1f}s", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"exp_{label}_{ts}.json")
    with open(out, "w") as f:
        json.dump({
            "label": label, "config": config_str, "ai_type": ai_type,
            "hostname": hostname,
            "l1_mean_ms": float(np.mean(arr)),
            "l1_p99_ms": float(np.percentile(arr, 99)),
            "tti_miss": miss,
            "ai_throughput_ips": ai_throughput,
            "ai_total_iters": ai_result["ai_iters"],
            "measure_time_s": measure_time,
            "raw_latencies": [float(x) for x in latencies],
        }, f, indent=2)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
