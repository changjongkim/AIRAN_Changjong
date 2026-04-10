"""
ResNet-50 inference stress — standalone background process.
Usage: python3 run_resnet_stress.py <gpu_id> <duration_sec> [batch_size]
"""
import sys
import os
import time

SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

import torch
import torchvision.models as models

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32

torch.cuda.set_device(gpu_id)
model = models.resnet50(weights=None).to(f"cuda:{gpu_id}").eval()
dummy = torch.randn(batch_size, 3, 224, 224, device=f"cuda:{gpu_id}")

with torch.no_grad():
    for _ in range(10):
        model(dummy)
torch.cuda.synchronize(gpu_id)

free, total = torch.cuda.mem_get_info(gpu_id)
print(f"[ResNet-50] GPU{gpu_id} bs={batch_size} HBM={(total-free)/1e9:.1f}/{total/1e9:.1f}GB, "
      f"running for {duration}s", flush=True)

c = 0
start = time.time()
with torch.no_grad():
    while time.time() - start < duration:
        model(dummy)
        c += 1
print(f"[ResNet-50] done: {c} iters, {c/(time.time()-start):.1f} it/s", flush=True)
