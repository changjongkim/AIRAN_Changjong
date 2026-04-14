"""
[S] SM Limiter — MIG-like SM partitioning via MPS

Sets CUDA_MPS_ACTIVE_THREAD_PERCENTAGE to limit SM usage per process.
Emulates MIG 40:60 (or any ratio) without requiring MIG hardware support.
"""
import os


class SMLimiter:
    """ON/OFF capable MIG-like SM partitioning."""

    def __init__(self, enabled=True, l1_pct=40, ai_pct=60):
        self.enabled = enabled
        self.l1_pct = l1_pct if enabled else 100
        self.ai_pct = ai_pct if enabled else 100

    def apply_l1(self):
        """Set SM limit for L1 process. Call before launching L1."""
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.l1_pct)

    def apply_ai(self):
        """Set SM limit for AI process. Call before launching AI."""
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.ai_pct)

    def get_l1_env(self):
        """Get environment dict for L1 subprocess."""
        env = os.environ.copy()
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.l1_pct)
        return env

    def get_ai_env(self):
        """Get environment dict for AI subprocess."""
        env = os.environ.copy()
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.ai_pct)
        return env
