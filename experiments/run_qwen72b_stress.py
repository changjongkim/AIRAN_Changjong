"""
Qwen2.5-72B LLM inference stress — 4GPU tensor parallel.
Realistic large LLM serving workload for AI-RAN interference testing.

This model uses ~136GB in FP16 → requires 4x A100-40GB or 2x A100-80GB.
Uses HuggingFace device_map="auto" for tensor parallelism.

Usage: python3 run_qwen72b_stress.py <duration_sec> [batch_size] [seq_len]
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

duration = int(sys.argv[1]) if len(sys.argv) > 1 else 120
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4
seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 2048

print(f"[Qwen-72B] Loading model (4GPU TP, batch={batch_size}, seq={seq_len})...", flush=True)
load_start = time.time()

model_name = "Qwen/Qwen2.5-72B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",  # auto distribute across all GPUs
)
model.eval()

load_time = time.time() - load_start
print(f"[Qwen-72B] Model loaded in {load_time:.1f}s", flush=True)

# Print per-GPU memory usage
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  GPU{i}: {(total-free)/1e9:.1f}/{total/1e9:.1f}GB used", flush=True)

# Generate input on first device
first_device = next(model.parameters()).device
dummy = torch.randint(0, 32000, (batch_size, seq_len), device=first_device)

# Warmup
print(f"[Qwen-72B] Warmup...", flush=True)
with torch.no_grad():
    for _ in range(2):
        model(dummy)
torch.cuda.synchronize()
print(f"[Qwen-72B] Warmup done. Running for {duration}s", flush=True)

c = 0
start = time.time()
with torch.no_grad():
    while time.time() - start < duration:
        model(dummy)
        c += 1
        if c % 5 == 0:
            elapsed = time.time() - start
            print(f"[Qwen-72B] {c} iters, {c/elapsed:.2f} it/s", flush=True)

elapsed = time.time() - start
print(f"[Qwen-72B] done: {c} iters in {elapsed:.1f}s ({c/elapsed:.2f} it/s)", flush=True)

# Final memory
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  GPU{i}: {(total-free)/1e9:.1f}/{total/1e9:.1f}GB used", flush=True)
