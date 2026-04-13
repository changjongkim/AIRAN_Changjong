"""
Realistic AI-RAN workload simulators.
Mimics actual AI workloads that would run alongside L1 in a real AI-RAN deployment.

Types:
  neural_rx   — Neural Receiver: read IQ samples → CNN inference → write output (every TTI)
  video       — Video Analytics: read frame → object detection → repeat
  continuous  — Continuous matmul: simulates any bandwidth+compute mixed workload

Usage: python3 run_realistic_ai_stress.py <gpu_id> <duration_sec> <type> [intensity]
  intensity: 0.0~1.0 controls bandwidth/compute ratio
"""
import sys
import os
import time

SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

import torch

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120
workload_type = sys.argv[3] if len(sys.argv) > 3 else "neural_rx"
intensity = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

torch.cuda.set_device(gpu_id)


def neural_receiver(duration_sec, intensity):
    """
    Simulates Neural Receiver: every 1ms (TTI), reads IQ samples from HBM,
    runs a small CNN, writes output. This creates a periodic bandwidth pattern.

    intensity controls model size:
      0.2 = small model (low bandwidth)
      0.5 = medium model
      1.0 = large model (high bandwidth, ~Neural Receiver for Massive MIMO)
    """
    # Model size scales with intensity
    channels = int(64 * intensity)
    input_size = int(4096 * intensity)  # IQ samples per antenna
    batch = int(16 * intensity)

    # Simple CNN that mimics Neural Receiver pattern
    model = torch.nn.Sequential(
        torch.nn.Conv1d(channels, channels * 2, 7, padding=3),
        torch.nn.ReLU(),
        torch.nn.Conv1d(channels * 2, channels * 2, 7, padding=3),
        torch.nn.ReLU(),
        torch.nn.Conv1d(channels * 2, channels, 3, padding=1),
    ).cuda().eval()

    # Input: IQ samples (batch × antennas × samples)
    iq_input = torch.randn(batch, channels, input_size, device="cuda")

    with torch.no_grad():
        for _ in range(5):
            model(iq_input)
    torch.cuda.synchronize()

    free, total = torch.cuda.mem_get_info(gpu_id)
    print(f"[NeuralRx] GPU{gpu_id} intensity={intensity} "
          f"batch={batch} ch={channels} size={input_size} "
          f"HBM={(total-free)/1e9:.1f}GB, running {duration_sec}s", flush=True)

    c = 0
    start = time.time()
    with torch.no_grad():
        while time.time() - start < duration_sec:
            # Read IQ → Inference → Write output (every iteration = 1 TTI)
            output = model(iq_input)
            # Force memory write (simulates writing decoded bits back)
            iq_input[:, :, :output.shape[2]].copy_(output)
            c += 1

    elapsed = time.time() - start
    print(f"[NeuralRx] done: {c} TTIs in {elapsed:.1f}s "
          f"({c/elapsed:.0f} TTI/s, {elapsed/c*1000:.2f}ms/TTI)", flush=True)


def video_analytics(duration_sec, intensity):
    """
    Simulates real-time video analytics: continuous frame reading + object detection.
    Each iteration: read large image tensor → run detection model → repeat.
    """
    import torchvision.models as models

    batch = max(1, int(4 * intensity))
    res = int(224 + 576 * intensity)  # 224 (low) to 800 (high)

    model = models.resnet50(weights=None).cuda().eval()
    frame = torch.randn(batch, 3, res, res, device="cuda")

    with torch.no_grad():
        for _ in range(5):
            model(frame)
    torch.cuda.synchronize()

    free, total = torch.cuda.mem_get_info(gpu_id)
    print(f"[VideoAI] GPU{gpu_id} intensity={intensity} "
          f"batch={batch} res={res} "
          f"HBM={(total-free)/1e9:.1f}GB, running {duration_sec}s", flush=True)

    c = 0
    start = time.time()
    with torch.no_grad():
        while time.time() - start < duration_sec:
            model(frame)
            c += 1

    elapsed = time.time() - start
    print(f"[VideoAI] done: {c} frames in {elapsed:.1f}s "
          f"({c/elapsed:.1f} fps, {elapsed/c*1000:.2f}ms/frame)", flush=True)


def continuous_matmul(duration_sec, intensity):
    """
    Continuous matrix multiplication — tunable bandwidth/compute mix.
    Controls how much HBM bandwidth vs SM compute is used.

    intensity controls matrix size:
      0.2 = small matrices (compute-bound)
      1.0 = large matrices (bandwidth-bound)
    """
    size = int(1024 + 7168 * intensity)  # 1K to 8K

    A = torch.randn(size, size, device="cuda", dtype=torch.float16)
    B = torch.randn(size, size, device="cuda", dtype=torch.float16)
    C = torch.empty(size, size, device="cuda", dtype=torch.float16)

    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info(gpu_id)
    bw_per_iter = 3 * size * size * 2 / 1e9  # read A, B, write C (fp16)
    print(f"[Matmul] GPU{gpu_id} intensity={intensity} "
          f"size={size} bw/iter={bw_per_iter:.1f}GB "
          f"HBM={(total-free)/1e9:.1f}GB, running {duration_sec}s", flush=True)

    c = 0
    start = time.time()
    while time.time() - start < duration_sec:
        torch.mm(A, B, out=C)
        c += 1

    elapsed = time.time() - start
    actual_bw = c * bw_per_iter / elapsed
    print(f"[Matmul] done: {c} iters, {actual_bw:.1f} GB/s effective bandwidth", flush=True)


if __name__ == "__main__":
    if workload_type == "neural_rx":
        neural_receiver(duration, intensity)
    elif workload_type == "video":
        video_analytics(duration, intensity)
    elif workload_type == "matmul":
        continuous_matmul(duration, intensity)
    else:
        print(f"Unknown workload type: {workload_type}", flush=True)
        sys.exit(1)
