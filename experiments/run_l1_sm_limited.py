"""
L1 with SM% limit — measures L1 latency when GPU compute is restricted.
Simulates MIG partition where L1 gets only X% of SMs.

Usage: python3 run_l1_sm_limited.py <label> <sm_percent>
  sm_percent: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE (1-100)
"""
import os
import sys
import json
import time
import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

NUM_WARMUP = 10
NUM_ITERATIONS = 100
RESULTS_DIR = "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results"
ESNO_DB = 10.0

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


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unnamed"
    # SM% is set externally via CUDA_MPS_ACTIVE_THREAD_PERCENTAGE

    import numpy as np
    import cupy as cp
    import tensorflow as tf
    import sionna

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    if len(gpus) > 1:
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
    props = torch.cuda.get_device_properties(0)
    sm_pct = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100")
    print(f"GPU: {props.name} ({props.multi_processor_count} SMs)", flush=True)
    print(f"SM limit: {sm_pct}%", flush=True)

    trk = LatencyTracker(label)
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

    s = trk.stats()
    miss = trk.deadline_miss_rate(1.0)
    print(f"\nRESULT: {label}", flush=True)
    print(f"  SM%:      {sm_pct}%", flush=True)
    print(f"  RX mean:  {s['mean_ms']:.3f} ms", flush=True)
    print(f"  RX P50:   {s['p50_ms']:.3f} ms", flush=True)
    print(f"  RX P95:   {s['p95_ms']:.3f} ms", flush=True)
    print(f"  RX P99:   {s['p99_ms']:.3f} ms", flush=True)
    print(f"  Jitter:   {s['jitter']:.4f}", flush=True)
    print(f"  Miss>1ms: {miss*100:.1f}%", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"exp_{label}_{ts}.json")
    with open(out, "w") as f:
        json.dump({"label": label, "sm_pct": int(sm_pct),
                    "stats": s, "miss_1ms": miss, "raw": trk.latencies}, f, indent=2)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
