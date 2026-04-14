"""
BORA Configuration — ON/OFF switches for each mechanism.
Maps to experiment configs A-E.
"""
from dataclasses import dataclass


@dataclass
class BORAConfig:
    """Configuration for BORA orchestrator.

    Each mechanism can be independently enabled/disabled.
    This enables ablation study (Config A-E).
    """
    # [P] Stream Priority: L1=high(-5), AI=low(0)
    priority_enabled: bool = False
    l1_priority: int = -5   # highest on A100
    ai_priority: int = 0    # lowest

    # [S] SM Limit: MIG-like partitioning via MPS
    sm_limit_enabled: bool = False
    l1_sm_pct: int = 40     # L1 gets 40% of SMs
    ai_sm_pct: int = 60     # AI gets 60% of SMs

    # [T] TTI-aware Coordination: signal-based L1/AI scheduling
    tti_coord_enabled: bool = False
    tti_ms: float = 1.0     # TTI period (30kHz SCS)
    l1_budget_ms: float = 0.5  # L1 active window budget

    # [C] Cluster Placement: HBM-pressure-aware distribution
    cluster_enabled: bool = False

    @staticmethod
    def config_a():
        """Config A: all OFF (baseline)."""
        return BORAConfig()

    @staticmethod
    def config_b():
        """Config B: SM limit only (MIG-like 40:60)."""
        return BORAConfig(sm_limit_enabled=True)

    @staticmethod
    def config_c():
        """Config C: SM limit + Priority."""
        return BORAConfig(sm_limit_enabled=True, priority_enabled=True)

    @staticmethod
    def config_d():
        """Config D: SM limit + Priority + TTI coordination."""
        return BORAConfig(sm_limit_enabled=True, priority_enabled=True,
                          tti_coord_enabled=True)

    @staticmethod
    def config_e():
        """Config E: BORA full (all ON)."""
        return BORAConfig(sm_limit_enabled=True, priority_enabled=True,
                          tti_coord_enabled=True, cluster_enabled=True)

    def name(self):
        parts = []
        if self.sm_limit_enabled: parts.append("S")
        if self.priority_enabled: parts.append("P")
        if self.tti_coord_enabled: parts.append("T")
        if self.cluster_enabled: parts.append("C")
        return "+".join(parts) if parts else "baseline"
