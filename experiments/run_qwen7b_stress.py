"""
Qwen2.5-7B LLM inference stress — standalone background process.
Realistic large LLM workload for AI-RAN interference testing.
Usage: python3 run_qwen7b_stress.py <gpu_id> <duration_sec>
"""
import sys
import os
import time

os.environ["HF_HOME"] = "/pscratch/sd/s/sgkim/kcj/hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120

torch.cuda.set_device(gpu_id)

model_name = "Qwen/Qwen2.5-7B"
print(f"[Qwen-7B] Loading model on GPU{gpu_id}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=f"cuda:{gpu_id}",
)
model.eval()

# Generate input
dummy = torch.randint(0, 32000, (1, 512), device=f"cuda:{gpu_id}")

# Warmup
with torch.no_grad():
    for _ in range(3):
        model(dummy)
torch.cuda.synchronize(gpu_id)

free, total = torch.cuda.mem_get_info(gpu_id)
print(f"[Qwen-7B] GPU{gpu_id} HBM={(total-free)/1e9:.1f}/{total/1e9:.1f}GB, "
      f"running for {duration}s", flush=True)

c = 0
start = time.time()
with torch.no_grad():
    while time.time() - start < duration:
        model(dummy)
        c += 1
elapsed = time.time() - start
print(f"[Qwen-7B] done: {c} iters, {c/elapsed:.1f} it/s", flush=True)
