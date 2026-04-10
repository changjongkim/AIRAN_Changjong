"""
HBM Stress — standalone. Run as background process.
Usage: python3 run_hbm_stress.py <gpu_id> <duration_sec> [alloc_gb]
"""
import sys
import time
import torch

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120
alloc_gb = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0

torch.cuda.set_device(gpu_id)
n = int(alloc_gb * 1e9 / 4)
src = torch.randn(n, dtype=torch.float32, device=f"cuda:{gpu_id}")
dst = torch.empty_like(src)
torch.cuda.synchronize(gpu_id)
print(f"[HBM stress] GPU{gpu_id} alloc={alloc_gb}GB running for {duration}s", flush=True)

c = 0
start = time.time()
while time.time() - start < duration:
    dst.copy_(src)
    src.copy_(dst)
    c += 1
print(f"[HBM stress] done: {c} iters", flush=True)
