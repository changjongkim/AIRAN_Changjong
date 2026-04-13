"""
Step 2: CUDA Stream Priority — Can we prioritize L1 over AI?
Tests if high-priority stream (L1) preempts low-priority stream (AI).

If this works: L1 runs immediately even when AI is using GPU.
If this doesn't work: need different approach (time-slicing, kernel splitting).
"""
import os, sys, time, json, datetime
sys.stdout.reconfigure(line_buffering=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
SITE = "/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

import numpy as np
import cupy as cp
import torch


def test_stream_priority():
    """Test if CUDA stream priority actually affects execution order."""

    # Check priority range
    least_priority = torch.cuda.get_device_properties(0).major  # placeholder
    import ctypes
    libcudart = ctypes.CDLL("libcudart.so")
    least = ctypes.c_int()
    greatest = ctypes.c_int()
    libcudart.cudaDeviceGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest))
    print(f"Stream priority range: {greatest.value} (highest) to {least.value} (lowest)", flush=True)

    # Create high and low priority streams
    high_stream = cp.cuda.Stream(non_blocking=True)
    low_stream = cp.cuda.Stream(non_blocking=True)

    # Test 1: AI running alone (baseline)
    print("\n=== Test 1: AI alone (baseline) ===", flush=True)
    n = 50_000_000
    a = cp.random.randn(n, dtype=cp.float32)
    b = cp.random.randn(n, dtype=cp.float32)
    c = cp.empty_like(a)

    start_ev = cp.cuda.Event()
    end_ev = cp.cuda.Event()

    ai_times = []
    for _ in range(20):
        start_ev.record(low_stream)
        with low_stream:
            for _ in range(10):
                c = a * b + a  # simulated AI computation
        end_ev.record(low_stream)
        end_ev.synchronize()
        ai_times.append(cp.cuda.get_elapsed_time(start_ev, end_ev))

    ai_alone = np.mean(ai_times[5:])
    print(f"  AI alone: {ai_alone:.3f}ms", flush=True)

    # Test 2: L1 (small kernel) alone
    print("\n=== Test 2: L1-like kernel alone ===", flush=True)
    l1_data = cp.random.randn(1_000_000, dtype=cp.float32)
    l1_out = cp.empty_like(l1_data)

    l1_times = []
    for _ in range(20):
        start_ev.record(high_stream)
        with high_stream:
            l1_out = cp.fft.fft(l1_data)  # FFT = L1-like operation
        end_ev.record(high_stream)
        end_ev.synchronize()
        l1_times.append(cp.cuda.get_elapsed_time(start_ev, end_ev))

    l1_alone = np.mean(l1_times[5:])
    print(f"  L1 alone: {l1_alone:.3f}ms", flush=True)

    # Test 3: L1 (high priority) while AI (low priority) is running
    print("\n=== Test 3: L1 with AI background (same default priority) ===", flush=True)
    l1_with_ai_default = []
    for _ in range(20):
        # Launch AI on default stream
        with low_stream:
            for _ in range(100):
                c = a * b + a

        # Immediately launch L1 on another stream
        start_ev.record(high_stream)
        with high_stream:
            l1_out = cp.fft.fft(l1_data)
        end_ev.record(high_stream)
        end_ev.synchronize()
        l1_with_ai_default.append(cp.cuda.get_elapsed_time(start_ev, end_ev))

    l1_default = np.mean(l1_with_ai_default[5:])
    print(f"  L1 with AI (default priority): {l1_default:.3f}ms "
          f"(vs alone: {l1_alone:.3f}ms, ratio: {l1_default/l1_alone:.2f}x)", flush=True)

    # Test 4: Use actual CUDA stream priority
    print("\n=== Test 4: L1 (HIGH priority) with AI (LOW priority) ===", flush=True)

    # Create prioritized streams via CUDA runtime
    from cuda import cudart
    err, hi_stream = cudart.cudaStreamCreateWithPriority(
        cudart.cudaStreamNonBlocking, greatest.value)
    err, lo_stream = cudart.cudaStreamCreateWithPriority(
        cudart.cudaStreamNonBlocking, least.value)

    hi_cp = cp.cuda.ExternalStream(int(hi_stream))
    lo_cp = cp.cuda.ExternalStream(int(lo_stream))

    l1_with_ai_priority = []
    for _ in range(20):
        # AI on low priority
        with lo_cp:
            for _ in range(100):
                c = a * b + a

        # L1 on high priority
        start_ev.record(stream=hi_cp)
        with hi_cp:
            l1_out = cp.fft.fft(l1_data)
        end_ev.record(stream=hi_cp)
        end_ev.synchronize()
        l1_with_ai_priority.append(cp.cuda.get_elapsed_time(start_ev, end_ev))

    l1_priority = np.mean(l1_with_ai_priority[5:])
    print(f"  L1 with AI (HIGH priority): {l1_priority:.3f}ms "
          f"(vs alone: {l1_alone:.3f}ms, ratio: {l1_priority/l1_alone:.2f}x)", flush=True)

    # Test 5: Bandwidth contention with priority
    print("\n=== Test 5: L1 with HBM stress (priority vs no priority) ===", flush=True)

    stress_src = cp.random.randn(int(2e9 / 4), dtype=cp.float32)
    stress_dst = cp.empty_like(stress_src)

    # HBM stress + L1 on default streams
    l1_with_hbm_default = []
    for _ in range(20):
        with low_stream:
            stress_dst.data.copy_from_device(stress_src.data, stress_src.nbytes)

        start_ev.record(high_stream)
        with high_stream:
            l1_out = cp.fft.fft(l1_data)
        end_ev.record(high_stream)
        end_ev.synchronize()
        l1_with_hbm_default.append(cp.cuda.get_elapsed_time(start_ev, end_ev))

    l1_hbm_default = np.mean(l1_with_hbm_default[5:])

    # HBM stress + L1 with priority
    l1_with_hbm_priority = []
    for _ in range(20):
        with lo_cp:
            stress_dst.data.copy_from_device(stress_src.data, stress_src.nbytes)

        start_ev.record(stream=hi_cp)
        with hi_cp:
            l1_out = cp.fft.fft(l1_data)
        end_ev.record(stream=hi_cp)
        end_ev.synchronize()
        l1_with_hbm_priority.append(cp.cuda.get_elapsed_time(start_ev, end_ev))

    l1_hbm_priority = np.mean(l1_with_hbm_priority[5:])

    print(f"  L1 + HBM (default): {l1_hbm_default:.3f}ms ({l1_hbm_default/l1_alone:.2f}x)",
          flush=True)
    print(f"  L1 + HBM (priority): {l1_hbm_priority:.3f}ms ({l1_hbm_priority/l1_alone:.2f}x)",
          flush=True)
    if l1_hbm_default > 0:
        improvement = (l1_hbm_default - l1_hbm_priority) / l1_hbm_default * 100
        print(f"  Priority improvement: {improvement:.1f}%", flush=True)

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    print(f"{'Scenario':<35} {'L1 latency':>12} {'vs alone':>10} {'Priority helps?':>16}",
          flush=True)
    print("-" * 75, flush=True)
    print(f"{'L1 alone':<35} {l1_alone:>10.3f}ms {'1.00x':>10}", flush=True)
    print(f"{'L1 + AI (default priority)':<35} {l1_default:>10.3f}ms "
          f"{l1_default/l1_alone:>9.2f}x", flush=True)
    print(f"{'L1 + AI (HIGH priority)':<35} {l1_priority:>10.3f}ms "
          f"{l1_priority/l1_alone:>9.2f}x "
          f"{'YES' if l1_priority < l1_default * 0.9 else 'NO':>14}", flush=True)
    print(f"{'L1 + HBM (default priority)':<35} {l1_hbm_default:>10.3f}ms "
          f"{l1_hbm_default/l1_alone:>9.2f}x", flush=True)
    print(f"{'L1 + HBM (HIGH priority)':<35} {l1_hbm_priority:>10.3f}ms "
          f"{l1_hbm_priority/l1_alone:>9.2f}x "
          f"{'YES' if l1_hbm_priority < l1_hbm_default * 0.9 else 'NO':>14}", flush=True)

    # Save
    results = {
        "l1_alone_ms": float(l1_alone),
        "ai_alone_ms": float(ai_alone),
        "l1_with_ai_default_ms": float(l1_default),
        "l1_with_ai_priority_ms": float(l1_priority),
        "l1_with_hbm_default_ms": float(l1_hbm_default),
        "l1_with_hbm_priority_ms": float(l1_hbm_priority),
        "priority_range": [greatest.value, least.value],
    }
    os.makedirs("/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results/exp_stream_priority_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results", flush=True)


if __name__ == "__main__":
    test_stream_priority()
