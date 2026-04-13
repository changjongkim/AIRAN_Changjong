"""
Actual Neural Receiver stress — runs the real pyAerial Neural Receiver model
(ONNX → TensorRT) in a loop as background AI workload.

This is the REAL AI-RAN in-line AI workload: Neural Network replaces
channel estimation + equalization in the L1 pipeline.

Usage: python3 run_neural_rx_stress.py <gpu_id> <duration_sec>
"""
import os, sys, time
sys.stdout.reconfigure(line_buffering=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

import numpy as np
import cupy as cp

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120

# Build TRT engine from ONNX — try multiple paths (Shifter remaps)
_candidates = [
    os.path.join(os.environ.get("cuBB_SDK", ""), "pyaerial/models"),
    "/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran/pyaerial/models",
]
MODEL_DIR = None
for _c in _candidates:
    _f = os.path.join(_c, "neural_rx.onnx")
    if os.path.exists(_f):
        MODEL_DIR = _c
        break
if MODEL_DIR is None:
    print(f"[NeuralRx] FAIL: cannot find neural_rx.onnx", flush=True)
    sys.exit(1)
onnx_file = os.path.join(MODEL_DIR, "neural_rx.onnx")
trt_file = "/tmp/neural_rx.trt"

print(f"[NeuralRx] Building TRT engine from {onnx_file}...", flush=True)
cmd = (f"trtexec --onnx={onnx_file} --saveEngine={trt_file} --skipInference "
       f"--inputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,int32:chw,int32:chw "
       f"--outputIOFormats=fp32:chw,fp32:chw "
       f"--shapes=rx_slot_real:1x3276x12x4,rx_slot_imag:1x3276x12x4,"
       f"h_hat_real:1x4914x1x4,h_hat_imag:1x4914x1x4 "
       f"> /dev/null 2>&1")
ret = os.system(cmd)
if ret != 0:
    print(f"[NeuralRx] FAIL: trtexec returned {ret}", flush=True)
    sys.exit(1)
print(f"[NeuralRx] TRT engine built.", flush=True)

from aerial.phy5g.algorithms import TrtEngine, TrtTensorPrms

trt_engine = TrtEngine(
    trt_model_file=trt_file,
    max_batch_size=1,
    input_tensors=[
        TrtTensorPrms('rx_slot_real', (3276, 12, 4), np.float32),
        TrtTensorPrms('rx_slot_imag', (3276, 12, 4), np.float32),
        TrtTensorPrms('h_hat_real', (4914, 1, 4), np.float32),
        TrtTensorPrms('h_hat_imag', (4914, 1, 4), np.float32),
        TrtTensorPrms('active_dmrs_ports', (1,), np.float32),
        TrtTensorPrms('dmrs_ofdm_pos', (3,), np.int32),
        TrtTensorPrms('dmrs_subcarrier_pos', (6,), np.int32),
    ],
    output_tensors=[
        TrtTensorPrms('output_1', (8, 1, 3276, 12), np.float32),
        TrtTensorPrms('output_2', (1, 3276, 12, 8), np.float32),
    ]
)

# Create dummy input tensors (random IQ samples)
input_tensors = {
    "rx_slot_real": np.random.randn(1, 3276, 12, 4).astype(np.float32),
    "rx_slot_imag": np.random.randn(1, 3276, 12, 4).astype(np.float32),
    "h_hat_real": np.random.randn(1, 4914, 1, 4).astype(np.float32),
    "h_hat_imag": np.random.randn(1, 4914, 1, 4).astype(np.float32),
    "active_dmrs_ports": np.ones((1, 1), dtype=np.float32),
    "dmrs_ofdm_pos": np.array([[0, 5, 10]], dtype=np.int32),
    "dmrs_subcarrier_pos": np.array([[0, 2, 4, 6, 8, 10]], dtype=np.int32),
}

# Warmup
for _ in range(5):
    trt_engine.run(input_tensors)
print(f"[NeuralRx] Warmup done. Running for {duration}s...", flush=True)

c = 0
start = time.time()
while time.time() - start < duration:
    trt_engine.run(input_tensors)
    c += 1

elapsed = time.time() - start
print(f"[NeuralRx] done: {c} inferences in {elapsed:.1f}s "
      f"({c/elapsed:.0f} inf/s, {elapsed/c*1000:.2f}ms/inf)", flush=True)
