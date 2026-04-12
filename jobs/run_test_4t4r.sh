#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH -J test_4t4r
#SBATCH -o /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/test_4t4r_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/test_4t4r_%j.err

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

shifter --image=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb \
python3 << 'EOF'
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages")
sys.stdout.reconfigure(line_buffering=True)

import cupy as cp
import torch
from aerial.phy5g.pdsch import PdschTxPipelineFactory
from aerial.phy5g.pusch import PuschRxPipelineFactory
from aerial.phy5g.config import *
from aerial.phy5g.ldpc import get_mcs, get_tb_size, random_tb
from aerial.util.cuda import get_cuda_stream
from aerial.pycuphy.types import PuschLdpcKernelLaunch

stream = get_cuda_stream()
mo, cr = get_mcs(2, 1)

configs = [
    (1, 2), (1, 4), (1, 8), (1, 16),
    (2, 2), (2, 4), (2, 8),
    (4, 4), (4, 8), (4, 16),
]

for num_tx, num_rx in configs:
    try:
        tx_cfg = AerialPdschTxConfig(cell_id=41, num_tx_ant=num_tx)
        tx_pipe = PdschTxPipelineFactory().create(tx_cfg, stream)

        rx_cfg = AerialPuschRxConfig(cell_id=41, num_rx_ant=num_rx,
                    enable_pusch_tdi=0, eq_coeff_algo=1,
                    ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL)
        rx_pipe = PuschRxPipelineFactory().create(rx_cfg, stream)

        tb_sz = get_tb_size(mod_order=mo, code_rate=cr,
                    dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                    num_prbs=273, start_sym=2, num_symbols=12, num_layers=1)
        pcw = PdschCwConfig(mcs_table=0, mcs_index=2, code_rate=int(cr*10), mod_order=mo)
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
        tx_out = tx_pipe(slot=0, tb_inputs=[tb], config=[pcfg])

        # Expand TX to match RX antennas
        if num_rx > tx_out.shape[2]:
            repeats = (num_rx // tx_out.shape[2]) + 1
            rx_in = cp.concatenate([tx_out] * repeats, axis=2)[:, :, :num_rx]
        else:
            rx_in = tx_out[:, :, :num_rx]
        noise = 0.01 * (cp.random.randn(*rx_in.shape, dtype=cp.float32) +
                1j * cp.random.randn(*rx_in.shape, dtype=cp.float32)).astype(cp.complex64)
        rx_in = rx_in + noise

        uue = PuschUeConfig(scid=0, layers=1, dmrs_ports=1, rnti=1234,
                    data_scid=0, mcs_table=0, mcs_index=2,
                    code_rate=int(cr*10), mod_order=mo, tb_size=tb_sz//8)
        ucfg = [PuschConfig(ue_configs=[uue], num_dmrs_cdm_grps_no_data=2,
                    dmrs_scrm_id=41, start_prb=0, num_prbs=273,
                    dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                    dmrs_max_len=1, dmrs_add_ln_pos=0,
                    start_sym=2, num_symbols=12)]
        rx_pipe(slot=0, rx_slot=rx_in, config=ucfg)

        free, total = torch.cuda.mem_get_info(0)
        print(f"[OK] {num_tx}T{num_rx}R: TX={tx_out.shape} RX_in={rx_in.shape} HBM={(total-free)/1e9:.1f}GB",
              flush=True)
        del tx_pipe, rx_pipe
        import gc; gc.collect(); torch.cuda.empty_cache()

    except Exception as e:
        err = str(e)[:80]
        print(f"[FAIL] {num_tx}T{num_rx}R: {err}", flush=True)
EOF
