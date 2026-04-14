"""
[P] Stream Priority Controller

Creates CUDA streams with different priorities for L1 and AI.
L1 gets highest priority → GPU scheduler executes L1 kernels first.
"""
from cuda import cudart


class PriorityController:
    """ON/OFF capable CUDA stream priority controller."""

    def __init__(self, enabled=True, l1_priority=-5, ai_priority=0):
        self.enabled = enabled

        if enabled:
            err, self.l1_stream = cudart.cudaStreamCreateWithPriority(
                cudart.cudaStreamNonBlocking, l1_priority)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"Failed to create L1 priority stream: {err}")

            err, self.ai_stream = cudart.cudaStreamCreateWithPriority(
                cudart.cudaStreamNonBlocking, ai_priority)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"Failed to create AI priority stream: {err}")
        else:
            err, self.l1_stream = cudart.cudaStreamCreate()
            err, self.ai_stream = cudart.cudaStreamCreate()

    def get_l1_stream(self):
        return self.l1_stream

    def get_ai_stream(self):
        return self.ai_stream

    def __del__(self):
        try:
            cudart.cudaStreamDestroy(self.l1_stream)
            cudart.cudaStreamDestroy(self.ai_stream)
        except Exception:
            pass
