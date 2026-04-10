"""CUDA event-based timing utilities."""
import time
import numpy as np
import cupy as cp


class CudaTimer:
    """Measure GPU kernel execution time using CUDA events."""

    def __init__(self):
        self.start_event = cp.cuda.Event()
        self.end_event = cp.cuda.Event()

    def start(self, stream=None):
        self.start_event.record(stream)

    def stop(self, stream=None):
        self.end_event.record(stream)
        self.end_event.synchronize()

    def elapsed_ms(self):
        return cp.cuda.get_elapsed_time(self.start_event, self.end_event)


class LatencyTracker:
    """Collect and analyze latency measurements."""

    def __init__(self, name=""):
        self.name = name
        self.latencies = []

    def record(self, latency_ms):
        self.latencies.append(latency_ms)

    def stats(self):
        if not self.latencies:
            return {}
        arr = np.array(self.latencies)
        return {
            "name": self.name,
            "count": len(arr),
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "jitter": float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0,
        }

    def deadline_miss_rate(self, deadline_ms=1.0):
        if not self.latencies:
            return 0.0
        arr = np.array(self.latencies)
        return float(np.sum(arr > deadline_ms) / len(arr))

    def reset(self):
        self.latencies = []
