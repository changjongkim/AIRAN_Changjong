#!/bin/bash
# Per-node BORA experiment runner.
# Runs L1 (cuPHY CUDA Graph) + AI (Qwen-72B TP) with specified BORA config.
# Args: <label> <cells> <config> <model>
# config: A/B/C/D/E
# model: 72b/32b/none

LABEL=$1
CELLS=$2
CONFIG=$3
MODEL=${4:-"none"}

export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:/pscratch/sd/s/sgkim/kcj/AI-RAN:$PYTHONPATH
export HF_HOME=/pscratch/sd/s/sgkim/kcj/hf_cache
export TRANSFORMERS_OFFLINE=1
IMAGE=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
L1=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/run_l1_graph.py
HOSTNAME=$(hostname)

# Parse BORA config into flags
L1_SM=100; AI_SM=100; L1_PRIO=0; AI_PRIO=0; TTI_FLAG="off"
case $CONFIG in
    A) ;; # all off
    B) L1_SM=40; AI_SM=60 ;;
    C) L1_SM=40; AI_SM=60; L1_PRIO=-5; AI_PRIO=0 ;;
    D) L1_SM=40; AI_SM=60; L1_PRIO=-5; AI_PRIO=0; TTI_FLAG="on" ;;
    E) L1_SM=40; AI_SM=60; L1_PRIO=-5; AI_PRIO=0; TTI_FLAG="on" ;; # +cluster handled at job level
esac

if [ "$MODEL" = "72b" ]; then
    MODEL_NAME="Qwen/Qwen2.5-72B"
elif [ "$MODEL" = "32b" ]; then
    MODEL_NAME="Qwen/Qwen2.5-32B"
fi

echo "[$HOSTNAME] BORA Config $CONFIG: SM(L1=${L1_SM}%,AI=${AI_SM}%) Priority(L1=${L1_PRIO},AI=${AI_PRIO}) TTI=$TTI_FLAG Model=$MODEL"

# Start MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${LABEL}_$$
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${LABEL}_$$
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d 2>/dev/null
sleep 1

# Start AI with SM limit and priority
AI_PID=""
if [ "$MODEL" != "none" ]; then
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$AI_SM PYTHONPATH=$PYTHONPATH \
    shifter --image=$IMAGE python3 -c "
import os, sys, time
os.environ['HF_HOME'] = '/pscratch/sd/s/sgkim/kcj/hf_cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages')
sys.path.insert(0, '/pscratch/sd/s/sgkim/kcj/AI-RAN')
import torch
from cuda import cudart

# Create AI stream with priority
ai_prio = ${AI_PRIO}
err, ai_stream = cudart.cudaStreamCreateWithPriority(1, ai_prio)  # non-blocking
print(f'[AI] Stream priority={ai_prio}, SM=${AI_SM}%', flush=True)

from transformers import AutoModelForCausalLM
print('[AI] Loading ${MODEL_NAME}...', flush=True)
model = AutoModelForCausalLM.from_pretrained(
    '${MODEL_NAME}', torch_dtype=torch.float16, device_map='auto')
model.eval()
dummy = torch.randint(0, 32000, (1, 1024), device='cuda:0')
with torch.no_grad():
    for _ in range(2): model(dummy)
torch.cuda.synchronize()
for i in range(4):
    free, total = torch.cuda.mem_get_info(i)
    print(f'  GPU{i}: {(total-free)/1e9:.1f}/{total/1e9:.1f}GB', flush=True)

# TTI coordination (Level 3: shared memory polling)
tti_enabled = '${TTI_FLAG}' == 'on'
shm = None
if tti_enabled:
    from multiprocessing import shared_memory
    try:
        shm = shared_memory.SharedMemory(name='bora_tti_flag', create=False, size=8)
        print('[AI] TTI coordination: ON (polling shared flag)', flush=True)
    except:
        print('[AI] TTI coordination: flag not found, running free', flush=True)
        tti_enabled = False

print('[AI] Running inference...', flush=True)
c = 0; start = time.time()
with torch.no_grad():
    while time.time() - start < 180:
        # TTI check: if L1 active, use smaller batch
        if tti_enabled and shm is not None and shm.buf[0] == 1:
            # L1 active → skip this iteration (throttle)
            time.sleep(0.0001)
            continue
        model(dummy); c += 1
        if c % 10 == 0:
            print(f'[AI] {c} iters, {c/(time.time()-start):.1f} it/s', flush=True)
elapsed = time.time() - start
print(f'[AI] done: {c} iters in {elapsed:.1f}s ({c/elapsed:.2f} it/s)', flush=True)
" &
    AI_PID=$!
    echo "[$HOSTNAME] Waiting 120s for AI to load..."
    sleep 120
    echo "[$HOSTNAME] AI PID=$AI_PID"
fi

# Run L1 on each GPU with SM limit and priority
# TTI coordination: L1 sets shared flag before/after kernel
L1_PIDS=()
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$L1_SM \
        shifter --image=$IMAGE python3 -c "
import os, sys, json, time, datetime, socket
sys.stdout.reconfigure(line_buffering=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages')
sys.path.insert(0, '/pscratch/sd/s/sgkim/kcj/AI-RAN')
os.environ['cuBB_SDK'] = '/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran'
os.environ['PYTHONPATH'] = os.environ.get('cuBB_SDK','') + '/pyaerial/src:' + os.environ.get('PYTHONPATH','')
sys.path.insert(0, os.environ['cuBB_SDK'] + '/pyaerial/src')

import numpy as np, cupy as cp, torch
from cuda import cudart

# Create L1 stream with priority
l1_prio = ${L1_PRIO}
err, l1_stream_raw = cudart.cudaStreamCreateWithPriority(1, l1_prio)

from aerial.pycuphy import _pycuphy as pycuphy
from aerial.pycuphy.util import get_pusch_stat_prms, get_pusch_dyn_prms_phase_2
from aerial.pycuphy.types import CellStatPrm, PuschLdpcKernelLaunch
from aerial.phy5g.config import _pusch_config_to_cuphy, PuschConfig, PuschUeConfig, AerialPdschTxConfig, PdschConfig, PdschUeConfig, PdschCwConfig
from aerial.phy5g.pdsch import PdschTxPipelineFactory
from aerial.phy5g.ldpc import get_mcs, get_tb_size, random_tb
from aerial.util.cuda import get_cuda_stream, check_cuda_errors

hn = socket.gethostname()
cells = ${CELLS}; num_tx = 4; num_rx = 4; mcs = 2
mo, cr = get_mcs(mcs, 1)
tb_sz = get_tb_size(mod_order=mo, code_rate=cr, dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0], num_prbs=273, start_sym=2, num_symbols=12, num_layers=1)

stream = l1_stream_raw
cell_stat_prms = [CellStatPrm(phyCellId=np.uint16(41+ci), nRxAnt=np.uint16(num_rx), nTxAnt=np.uint16(num_tx), nRxAntSrs=np.uint16(num_rx), nPrbUlBwp=np.uint16(273), nPrbDlBwp=np.uint16(273), mu=np.uint8(1)) for ci in range(cells)]
base_stat = get_pusch_stat_prms(cell_id=41, num_rx_ant=num_rx, num_tx_ant=num_tx, ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL)
multicell_stat = base_stat._replace(nMaxCells=np.uint16(cells), nMaxCellsPerSlot=np.uint16(cells), cellStatPrms=cell_stat_prms, enableDeviceGraphLaunch=np.uint8(1))
pipeline = pycuphy.PuschPipeline(multicell_stat, stream)

pusch_configs = []; rx_slots = []
for ci in range(cells):
    txp = PdschTxPipelineFactory().create(AerialPdschTxConfig(cell_id=41+ci, num_tx_ant=num_tx), stream)
    pcw = PdschCwConfig(mcs_table=0, mcs_index=mcs, code_rate=int(cr*10), mod_order=mo)
    pue = PdschUeConfig(cw_configs=[pcw], scid=0, dmrs_scrm_id=41, layers=1, dmrs_ports=1, rnti=1234, data_scid=0)
    pcfg = PdschConfig(ue_configs=[pue], num_dmrs_cdm_grps_no_data=2, start_prb=0, num_prbs=273, dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0], start_sym=2, num_symbols=12)
    tb = cp.array(random_tb(mod_order=mo, code_rate=cr, dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0], num_prbs=273, start_sym=2, num_symbols=12, num_layers=1), dtype=cp.uint8, order='F')
    tx_out = txp(slot=0, tb_inputs=[tb], config=[pcfg])
    reps = (num_rx // tx_out.shape[2]) + 1
    rx_in = cp.concatenate([tx_out]*reps, axis=2)[:,:,:num_rx]
    noise = 0.01*(cp.random.randn(*rx_in.shape, dtype=cp.float32)+1j*cp.random.randn(*rx_in.shape, dtype=cp.float32)).astype(cp.complex64)
    rx_slots.append(cp.asfortranarray(rx_in+noise))
    del txp
    uue = PuschUeConfig(scid=0, layers=1, dmrs_ports=1, rnti=1234, data_scid=0, mcs_table=0, mcs_index=mcs, code_rate=int(cr*10), mod_order=mo, tb_size=tb_sz//8)
    pusch_configs.append(PuschConfig(ue_configs=[uue], num_dmrs_cdm_grps_no_data=2, dmrs_scrm_id=41, start_prb=0, num_prbs=273, dmrs_syms=[0,0,1,0,0,0,0,0,0,0,0,0,0,0], dmrs_max_len=1, dmrs_add_ln_pos=0, start_sym=2, num_symbols=12))

dyn_prms = _pusch_config_to_cuphy(cuda_stream=stream, rx_data=rx_slots, slot=0, pusch_configs=pusch_configs)
dyn_prms = dyn_prms._replace(procModeBmsk=np.uint64(1))
pipeline.setup_pusch_rx(dyn_prms)
harq_buffers = []
for ue_idx in range(cells):
    hsz = dyn_prms.dataOut.harqBufferSizeInBytes[ue_idx]
    hbuf = check_cuda_errors(cudart.cudaMalloc(hsz))
    check_cuda_errors(cudart.cudaMemsetAsync(hbuf, 0, hsz, stream))
    check_cuda_errors(cudart.cudaStreamSynchronize(stream))
    harq_buffers.append(hbuf)
dyn_prms2 = get_pusch_dyn_prms_phase_2(dyn_prms, harq_buffers)
pipeline.setup_pusch_rx(dyn_prms2)

# TTI coordination: shared memory flag
tti_enabled = '${TTI_FLAG}' == 'on'
shm = None
if tti_enabled:
    from multiprocessing import shared_memory
    try:
        shm = shared_memory.SharedMemory(name='bora_tti_flag', create=True, size=8)
        shm.buf[0] = 0
    except FileExistsError:
        shm = shared_memory.SharedMemory(name='bora_tti_flag', create=False, size=8)

start_ev = cp.cuda.Event(); end_ev = cp.cuda.Event()
latencies = []
for i in range(60):
    if tti_enabled and shm: shm.buf[0] = 1  # L1 ACTIVE
    pipeline.setup_pusch_rx(dyn_prms)
    pipeline.setup_pusch_rx(dyn_prms2)
    start_ev.record()
    pipeline.run_pusch_rx()
    end_ev.record(); end_ev.synchronize()
    if tti_enabled and shm: shm.buf[0] = 0  # L1 IDLE
    if i >= 10: latencies.append(cp.cuda.get_elapsed_time(start_ev, end_ev))

for hbuf in harq_buffers: check_cuda_errors(cudart.cudaFree(hbuf))
arr = np.array(latencies); miss = float(np.sum(arr>1.0)/len(arr))
label = '${LABEL}_' + hn + '_g${gpu}'
print(f'RESULT: {label}', flush=True)
print(f'  Config:   ${CONFIG} (SM L1=${L1_SM}% AI=${AI_SM}%, Prio L1=${L1_PRIO} AI=${AI_PRIO}, TTI=${TTI_FLAG})', flush=True)
print(f'  RX mean:  {np.mean(arr):.3f} ms ({np.mean(arr)/cells:.3f} ms/cell)', flush=True)
print(f'  RX P99:   {np.percentile(arr,99):.3f} ms', flush=True)
print(f'  Miss>1ms: {miss*100:.1f}%', flush=True)
os.makedirs('/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results', exist_ok=True)
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
with open(f'/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results/exp_{label}_{ts}.json', 'w') as f:
    json.dump({'label':label,'config':'${CONFIG}','mean_ms':float(np.mean(arr)),'p99_ms':float(np.percentile(arr,99)),'miss_1ms':miss,'raw':[float(x) for x in latencies]}, f, indent=2)
if shm:
    shm.close()
    try: shm.unlink()
    except: pass
" &
    L1_PIDS+=($!)
done
for pid in "${L1_PIDS[@]}"; do wait $pid; done

if [ -n "$AI_PID" ]; then
    kill $AI_PID 2>/dev/null; wait $AI_PID 2>/dev/null
fi
echo quit | nvidia-cuda-mps-control 2>/dev/null
echo "[$HOSTNAME] Done"
