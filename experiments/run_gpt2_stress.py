"""
GPT-2 LLM inference stress — standalone background process.
Matches NVIDIA AI-RAN PoC paper's AI workload (LLM inference).
Usage: python3 run_gpt2_stress.py <gpu_id> <duration_sec>
"""
import sys
import os
import time

os.environ["HF_HOME"] = "/pscratch/sd/s/sgkim/kcj/AI-RAN/datasets/models"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

import torch
from transformers import GPT2LMHeadModel, GPT2Config

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120

torch.cuda.set_device(gpu_id)
cfg = GPT2Config()
model = GPT2LMHeadModel(cfg).to(f"cuda:{gpu_id}").eval()
dummy = torch.randint(0, cfg.vocab_size, (4, 512), device=f"cuda:{gpu_id}")

with torch.no_grad():
    for _ in range(5):
        model(dummy)
torch.cuda.synchronize(gpu_id)

free, total = torch.cuda.mem_get_info(gpu_id)
print(f"[GPT-2] GPU{gpu_id} HBM={(total-free)/1e9:.1f}/{total/1e9:.1f}GB, "
      f"running for {duration}s", flush=True)

c = 0
start = time.time()
with torch.no_grad():
    while time.time() - start < duration:
        model(dummy)
        c += 1
print(f"[GPT-2] done: {c} iters, {c/(time.time()-start):.1f} it/s", flush=True)
