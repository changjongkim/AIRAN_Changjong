"""
L1 Dynamic measurement — records per-iteration latency with timestamps.
Used to observe transition when AI workload starts/stops mid-run.
Usage: python3 run_l1_dynamic.py <label>
"""
import os, sys, json, time, datetime
sys.stdout.reconfigure(line_buffering=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

import numpy as np
import cupy as cp

NUM_WARMUP = 5
NUM_ITERATIONS = 300
RESULTS_DIR = "/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results"
ESNO_DB = 10.0


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "dynamic"

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
    from utils.timing import CudaTimer

    num_tx, num_rx, mcs_index = 1, 2, 2
    rg = sionna.phy.ofdm.ResourceGrid(
        num_ofdm_symbols=14, fft_size=4096, subcarrier_spacing=30e3,
        num_tx=num_tx, num_streams_per_tx=1, cyclic_prefix_length=288,
        num_guard_carriers=(410, 410), dc_null=False)
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
    tb_sz = get_tb_size(mod_order=mo, code_rate=cr,
                        dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                        num_prbs=273, start_sym=2, num_symbols=12, num_layers=1)
    No = pow(10.0, -ESNO_DB / 10.0)

    cid = 41
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

    timer = CudaTimer()
    latencies = []
    timestamps = []
    t0 = time.time()

    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        slot = i % 20
        tb = cp.array(random_tb(mod_order=mo, code_rate=cr,
                                dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                num_prbs=273, start_sym=2, num_symbols=12, num_layers=1),
                      dtype=cp.uint8, order="F")
        tx_t = txp(slot=slot, tb_inputs=[tb], config=[pcfg])
        rx_t = apply_ch(tx_t, No)
        timer.start()
        rxp(slot=slot, rx_slot=rx_t, config=ucfg)
        timer.stop()
        if i >= NUM_WARMUP:
            lat = timer.elapsed_ms()
            latencies.append(lat)
            timestamps.append(time.time() - t0)
            if (i - NUM_WARMUP) % 50 == 0:
                print(f"  [{i-NUM_WARMUP}/{NUM_ITERATIONS}] t={timestamps[-1]:.1f}s RX={lat:.3f}ms",
                      flush=True)

    arr = np.array(latencies)
    print(f"\nRESULT: {label}", flush=True)
    print(f"  RX mean:  {np.mean(arr):.3f} ms", flush=True)
    print(f"  RX P99:   {np.percentile(arr, 99):.3f} ms", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"exp_{label}_{ts}.json")
    with open(out, "w") as f:
        json.dump({"label": label, "latencies": latencies, "timestamps": timestamps,
                    "mean_ms": float(np.mean(arr)),
                    "p99_ms": float(np.percentile(arr, 99))}, f, indent=2)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
