"""Test heavy L1 config without Sionna — random input directly to cuPHY."""
import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

import cupy as cp
import torch

from aerial.phy5g.pusch import PuschRxPipelineFactory
from aerial.phy5g.pdsch import PdschTxPipelineFactory
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

stream = get_cuda_stream()
timer = CudaTimer()

# Test configs: (num_tx, num_rx, num_cells, label)
configs = [
    (1, 2, 1, "1T2R_1cell"),
    (1, 4, 1, "1T4R_1cell"),
    (1, 4, 4, "1T4R_4cell"),
    (1, 4, 8, "1T4R_8cell"),
    (1, 8, 8, "1T8R_8cell"),
    (1, 16, 8, "1T16R_8cell"),
    (1, 16, 16, "1T16R_16cell"),
    (1, 16, 20, "1T16R_20cell"),
]

for num_tx, num_rx, num_cells, label in configs:
    try:
        mo, cr = get_mcs(2, 1)
        tb_sz = get_tb_size(mod_order=mo, code_rate=cr,
                            dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                            num_prbs=273, start_sym=2, num_symbols=12, num_layers=1)

        cells = []
        for ci in range(num_cells):
            cid = 41 + ci
            # TX pipeline
            pcw = PdschCwConfig(mcs_table=0, mcs_index=2, code_rate=int(cr*10), mod_order=mo)
            pue = PdschUeConfig(cw_configs=[pcw], scid=0, dmrs_scrm_id=41,
                                layers=1, dmrs_ports=1, rnti=1234, data_scid=0)
            pcfg = PdschConfig(ue_configs=[pue], num_dmrs_cdm_grps_no_data=2,
                               start_prb=0, num_prbs=273,
                               dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                               start_sym=2, num_symbols=12)
            txp = PdschTxPipelineFactory().create(
                AerialPdschTxConfig(cell_id=cid, num_tx_ant=num_tx), stream)

            # RX pipeline
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

        free, total = torch.cuda.mem_get_info(0)
        hbm_pct = (total - free) / total * 100

        # Warmup + measure
        tb = cp.array(random_tb(mod_order=mo, code_rate=cr,
                                dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                num_prbs=273, start_sym=2, num_symbols=12, num_layers=1),
                      dtype=cp.uint8, order="F")

        latencies = []
        for i in range(15):
            rx_ms = 0
            for cell in cells:
                tx_t = cell["tx"](slot=0, tb_inputs=[tb], config=[cell["pcfg"]])
                # Use TX output directly as RX input (skip channel — measures pure GPU load)
                timer.start()
                cell["rx"](slot=0, rx_slot=tx_t, config=cell["ucfg"])
                timer.stop()
                rx_ms += timer.elapsed_ms()
            if i >= 5:  # skip warmup
                latencies.append(rx_ms)

        import numpy as np
        arr = np.array(latencies)
        print(f"[OK] {label}: {num_cells}cells × {num_tx}T{num_rx}R  "
              f"HBM={hbm_pct:.0f}%  RX mean={np.mean(arr):.3f}ms  "
              f"({np.mean(arr)/num_cells:.3f}ms/cell)", flush=True)

        # Cleanup
        del cells
        import gc; gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[FAIL] {label}: {e}", flush=True)
        break
