"""GPT-2 infinite inference loop. Run as a separate MPS process."""
import os
import sys
import time
import torch
from transformers import GPT2LMHeadModel, GPT2Config

BATCH_SIZE = 4
SEQ_LEN = 512
DEVICE = "cuda"


def run_loop(duration_sec=None):
    """Run GPT-2 inference in a loop.

    Args:
        duration_sec: Run for this many seconds, or forever if None.
    """
    # Use default GPT-2 config (124M params) — no need to download weights
    config = GPT2Config()
    model = GPT2LMHeadModel(config).to(DEVICE).eval()
    dummy_input = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(dummy_input)
    torch.cuda.synchronize()

    count = 0
    start = time.time()
    print(f"[GPT-2] Starting inference loop (batch={BATCH_SIZE}, seq={SEQ_LEN})", flush=True)

    with torch.no_grad():
        while True:
            model(dummy_input)
            count += 1
            elapsed = time.time() - start
            if count % 50 == 0:
                print(f"[GPT-2] {count} iters, {count/elapsed:.1f} it/s", flush=True)
            if duration_sec and elapsed >= duration_sec:
                break

    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"[GPT-2] Done: {count} iters in {elapsed:.1f}s ({count/elapsed:.1f} it/s)")


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_loop(duration)
