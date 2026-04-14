"""
[T] TTI-aware Coordination Controller — BORA's core mechanism

Coordinates HBM access between L1 and AI processes using signals.
During L1 active window: AI throttles HBM access.
During L1 idle window: AI runs at full bandwidth.

Three coordination levels:
  Level 1: Stream Priority (implicit, via PriorityController)
  Level 2: CUDA Event (GPU-level, explicit sync)
  Level 3: Shared Memory Flag (CPU-level, flexible throttle)
"""
import os
import time
from multiprocessing import shared_memory
from cuda import cudart


class TTICoordinator:
    """ON/OFF capable TTI-aware inter-process coordination.

    Core idea: instead of static HBM bank partitioning (impossible in SW),
    BORA coordinates the *timing* of HBM access between L1 and AI processes,
    indirectly minimizing bank conflict frequency.
    """

    def __init__(self, enabled=True, tti_ms=1.0, l1_budget_ms=0.5,
                 shm_name="bora_tti_flag"):
        self.enabled = enabled
        self.tti_ms = tti_ms
        self.l1_budget_ms = l1_budget_ms
        self.ai_budget_ms = tti_ms - l1_budget_ms
        self.shm_name = shm_name
        self._owns_shm = False

        if not enabled:
            self.l1_active_event = None
            self.l1_idle_event = None
            self.shm = None
            return

        # Level 2: CUDA events for GPU-level coordination
        err, self.l1_active_event = cudart.cudaEventCreate()
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to create l1_active_event: {err}")

        err, self.l1_idle_event = cudart.cudaEventCreate()
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to create l1_idle_event: {err}")

        # Level 3: Shared memory flag for CPU-level coordination
        # Flag: 0 = L1 IDLE, 1 = L1 ACTIVE
        try:
            self.shm = shared_memory.SharedMemory(name=shm_name, create=True, size=8)
            self._owns_shm = True
            self.shm.buf[0] = 0  # initially idle
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=shm_name, create=False, size=8)
            self._owns_shm = False

    # ---- L1 process calls these ----

    def l1_start(self, l1_stream=None):
        """Signal: L1 kernel is starting (HBM access begins)."""
        if not self.enabled:
            return
        # Level 2: record event on L1 stream
        if l1_stream is not None and self.l1_active_event is not None:
            cudart.cudaEventRecord(self.l1_active_event, l1_stream)
        # Level 3: set shared flag
        if self.shm is not None:
            self.shm.buf[0] = 1

    def l1_end(self, l1_stream=None):
        """Signal: L1 kernel is done (HBM access ends)."""
        if not self.enabled:
            return
        # Level 2: record idle event
        if l1_stream is not None and self.l1_idle_event is not None:
            cudart.cudaEventRecord(self.l1_idle_event, l1_stream)
        # Level 3: clear shared flag
        if self.shm is not None:
            self.shm.buf[0] = 0

    # ---- AI process calls these ----

    def ai_wait_for_idle(self, ai_stream=None):
        """AI waits until L1 is idle before heavy HBM access.
        Level 2: GPU-level synchronization via CUDA event."""
        if not self.enabled:
            return
        if ai_stream is not None and self.l1_idle_event is not None:
            cudart.cudaStreamWaitEvent(ai_stream, self.l1_idle_event, 0)

    def ai_should_throttle(self):
        """Check if AI should throttle (L1 is active).
        Level 3: CPU-level polling via shared memory.
        Returns True if L1 is currently active → AI should reduce bandwidth."""
        if not self.enabled:
            return False
        if self.shm is not None:
            return self.shm.buf[0] == 1
        return False

    def ai_adaptive_batch(self, default_batch, throttled_batch):
        """Return appropriate batch size based on L1 state.
        When L1 active: use smaller batch → less HBM bandwidth.
        When L1 idle: use full batch → max throughput."""
        if not self.enabled:
            return default_batch
        if self.ai_should_throttle():
            return throttled_batch
        return default_batch

    # ---- Cleanup ----

    def close(self):
        """Cleanup resources."""
        try:
            if self.l1_active_event is not None:
                cudart.cudaEventDestroy(self.l1_active_event)
            if self.l1_idle_event is not None:
                cudart.cudaEventDestroy(self.l1_idle_event)
            if self.shm is not None:
                self.shm.close()
                if self._owns_shm:
                    self.shm.unlink()
        except Exception:
            pass

    def __del__(self):
        self.close()
