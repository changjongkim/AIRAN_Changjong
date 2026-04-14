"""
BORA: Bandwidth-Oriented RAN-AI Orchestrator

Manages HBM bandwidth contention between hard real-time L1 (cuPHY)
and best-effort AI workloads on shared GPU clusters.

Mechanisms (all independently ON/OFF):
  P: Stream Priority — L1=high, AI=low CUDA stream priority
  S: SM Limit — MIG-like SM% partitioning via MPS
  T: TTI-aware Coordination — signal-based L1/AI HBM access scheduling
  C: Cluster Placement — HBM-pressure-aware workload distribution
"""
from .priority import PriorityController
from .sm_limit import SMLimiter
from .tti_coordinator import TTICoordinator
from .config import BORAConfig
