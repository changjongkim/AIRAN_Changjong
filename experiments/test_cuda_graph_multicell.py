"""
Test: Multi-cell cuPHY with CUDA Graph mode.
Uses get_pusch_stat_prms() then patches nMaxCells and enableDeviceGraphLaunch.
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
from aerial.pycuphy.util import get_pusch_stat_prms
from aerial.pycuphy.types import (
    CellStatPrm, PuschLdpcKernelLaunch,
)
from aerial.util.cuda import get_cuda_stream


def test_multicell_graph():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.multi_processor_count} SMs)", flush=True)

    num_tx = 4
    num_rx = 4

    for num_cells in [1, 2, 4, 8]:
        for graph_mode in [False, True]:
            mode_str = "GRAPH" if graph_mode else "STREAM"
            label = f"{num_cells}cell_{mode_str}"

            try:
                stream = get_cuda_stream()

                # Get base stat prms (1 cell) using the existing helper
                base_stat = get_pusch_stat_prms(
                    cell_id=41,
                    num_rx_ant=num_rx,
                    num_tx_ant=num_tx,
                    ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL,
                )

                # Build multi-cell stat prms by patching fields
                cell_stat_prms = []
                for ci in range(num_cells):
                    cell_stat_prm = CellStatPrm(
                        phyCellId=np.uint16(41 + ci),
                        nRxAnt=np.uint16(num_rx),
                        nTxAnt=np.uint16(num_tx),
                        nRxAntSrs=np.uint16(num_rx),
                        nPrbUlBwp=np.uint16(273),
                        nPrbDlBwp=np.uint16(273),
                        mu=np.uint8(1),
                    )
                    cell_stat_prms.append(cell_stat_prm)

                multicell_stat = base_stat._replace(
                    nMaxCells=np.uint16(num_cells),
                    nMaxCellsPerSlot=np.uint16(num_cells),
                    cellStatPrms=cell_stat_prms,
                    enableDeviceGraphLaunch=np.uint8(1 if graph_mode else 0),
                )

                pipeline = pycuphy.PuschPipeline(multicell_stat, stream)

                free, total = torch.cuda.mem_get_info(0)
                print(f"[OK] {label}: pipeline created, "
                      f"HBM={(total-free)/1e9:.1f}GB", flush=True)

                del pipeline
                import gc; gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"[FAIL] {label}: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    test_multicell_graph()
