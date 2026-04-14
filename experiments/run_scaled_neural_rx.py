"""
Scaled Neural Receiver stress — creates PyTorch CNN models at various
Massive MIMO scales and runs them as background AI workload.

Scales:
  small:  4 ant, 0.5MB  — current pyAerial demo level
  medium: 16 ant, 2MB   — 16T16R gNB
  large:  64 ant, 9MB   — 64T64R Massive MIMO
  xlarge: 64 ant, 33MB  — large Massive MIMO Neural Rx

Usage: python3 run_scaled_neural_rx.py <gpu_id> <duration_sec> <scale>
  scale: small / medium / large / xlarge
"""
import sys
import os
import time

SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

import torch
import torch.nn as nn

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120
scale = sys.argv[3] if len(sys.argv) > 3 else "large"

torch.cuda.set_device(gpu_id)

CONFIGS = {
    "small":  {"num_ant": 4,  "hidden": 64,  "layers": 5, "batch": 1},
    "medium": {"num_ant": 16, "hidden": 128, "layers": 5, "batch": 1},
    "large":  {"num_ant": 64, "hidden": 256, "layers": 5, "batch": 1},
    "xlarge": {"num_ant": 64, "hidden": 512, "layers": 5, "batch": 1},
}

cfg = CONFIGS[scale]
num_ant = cfg["num_ant"]
hidden = cfg["hidden"]
batch = cfg["batch"]

# Build Neural Receiver model
# Input: IQ samples (batch × num_ant*2 × subcarriers*symbols)
# mimics: read IQ → Conv layers → output LLRs
layers = []
in_ch = num_ant * 2  # real + imag
for i in range(cfg["layers"]):
    out_ch = hidden if i < cfg["layers"] - 1 else num_ant
    k = 7 if i < 2 else 5 if i < 4 else 3
    layers.append(nn.Conv1d(in_ch, out_ch, k, padding=k // 2))
    if i < cfg["layers"] - 1:
        layers.append(nn.ReLU())
    in_ch = out_ch

model = nn.Sequential(*layers).to(f"cuda:{gpu_id}").eval()

# Input: subcarriers × OFDM symbols flattened
seq_len = 3276 * 14  # 273 PRBs × 12 subcarriers × 14 symbols
# For memory, cap seq_len for large models
if scale in ["large", "xlarge"]:
    seq_len = 3276 * 4  # 4 symbols per inference call (more realistic)

iq_input = torch.randn(batch, num_ant * 2, seq_len, device=f"cuda:{gpu_id}")

params = sum(p.numel() for p in model.parameters())
input_bytes = iq_input.nelement() * 4

# Warmup
with torch.no_grad():
    for _ in range(10):
        output = model(iq_input)
torch.cuda.synchronize(gpu_id)

free, total = torch.cuda.mem_get_info(gpu_id)
print(f"[ScaledNRx] scale={scale} ant={num_ant} hidden={hidden} "
      f"params={params:,} ({params*4/1e6:.1f}MB)", flush=True)
print(f"[ScaledNRx] input={list(iq_input.shape)} ({input_bytes/1e6:.1f}MB) "
      f"seq_len={seq_len}", flush=True)
print(f"[ScaledNRx] HBM={(total-free)/1e9:.1f}/{total/1e9:.1f}GB, "
      f"running {duration}s", flush=True)

c = 0
start = time.time()
with torch.no_grad():
    while time.time() - start < duration:
        output = model(iq_input)
        # Force write back (simulates writing decoded output to HBM)
        iq_input[:, :num_ant, :output.shape[2]].copy_(output)
        c += 1

elapsed = time.time() - start
bw_per_iter = (input_bytes + output.nelement() * 4 + params * 4) / 1e9
print(f"[ScaledNRx] done: {c} iters, {c/elapsed:.1f} inf/s, "
      f"{elapsed/c*1000:.2f}ms/inf, ~{bw_per_iter*c/elapsed:.1f}GB/s", flush=True)
