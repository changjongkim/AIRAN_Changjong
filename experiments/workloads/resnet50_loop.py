"""ResNet-50 infinite inference loop. Run as a separate MPS process."""
import os
import sys
import time
import signal
import torch
import torchvision.models as models

BATCH_SIZE = 32
DEVICE = "cuda"


def run_loop(duration_sec=None):
    """Run ResNet-50 inference in a loop.

    Args:
        duration_sec: Run for this many seconds, or forever if None.
    """
    model = models.resnet50(weights=None).to(DEVICE).eval()
    dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224, device=DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)
    torch.cuda.synchronize()

    count = 0
    start = time.time()
    print(f"[ResNet-50] Starting inference loop (batch={BATCH_SIZE})", flush=True)

    with torch.no_grad():
        while True:
            model(dummy_input)
            count += 1
            elapsed = time.time() - start
            if count % 100 == 0:
                print(f"[ResNet-50] {count} iters, {count/elapsed:.1f} it/s", flush=True)
            if duration_sec and elapsed >= duration_sec:
                break

    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"[ResNet-50] Done: {count} iters in {elapsed:.1f}s ({count/elapsed:.1f} it/s)")


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_loop(duration)
