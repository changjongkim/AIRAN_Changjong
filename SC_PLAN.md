# SC Paper: TTI-Aware HBM-Bandwidth-Oriented GPU Orchestration

## 논문 제목

**"TTI-Aware HBM-Bandwidth-Oriented GPU Orchestration for Real-Time and AI Workloads on Shared GPU Clusters"**

---

## 1. 시스템 이름: **BORA** (Bandwidth-Oriented RAN-AI orchestrator)

---

## 2. 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                    BORA Orchestrator                      │
│                                                          │
│  ┌─────────────────────┐  ┌───────────────────────────┐ │
│  │  Time Axis           │  │  Space Axis               │ │
│  │  TTI-aware           │  │  HBM-pressure-aware       │ │
│  │  Bandwidth Budget    │  │  Cluster Scheduler        │ │
│  │  Controller          │  │                           │ │
│  │                      │  │                           │ │
│  │  Per-GPU, per-TTI:   │  │  Per-cluster:             │ │
│  │  - L1 active → AI   │  │  - Profile AI bandwidth   │ │
│  │    throttle          │  │  - Classify: light/heavy  │ │
│  │  - L1 idle → AI     │  │  - Place heavy AI on      │ │
│  │    full speed        │  │    L1-light nodes         │ │
│  │                      │  │  - Rebalance on demand    │ │
│  └─────────────────────┘  └───────────────────────────┘ │
│                                                          │
│  ON/OFF switches:                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────────┐ │
│  │ Priority │ │ SM Limit │ │ Cluster Placement        │ │
│  │ [ON/OFF] │ │ [ON/OFF] │ │ [STATIC/DYNAMIC]         │ │
│  └──────────┘ └──────────┘ └──────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## 3. ON/OFF 설계 — 각 메커니즘 독립 평가

모든 최적화 메커니즘을 독립적으로 ON/OFF 가능하게 설계하여,
각각의 contribution을 개별적으로 측정 가능.

### 3.1 메커니즘 목록

| ID | 메커니즘 | ON일 때 | OFF일 때 (baseline) |
|----|---------|---------|-------------------|
| **P** | Stream Priority | L1=high, AI=low priority | 둘 다 default priority |
| **S** | SM Limit (MIG-like) | L1=40% SM, AI=60% SM | 둘 다 100% SM |
| **T** | TTI-aware Throttle | L1 active 중 AI launch 지연 | AI 항상 실행 |
| **C** | Cluster Placement | bandwidth-aware 배치 | 균등 배치 (round-robin) |

### 3.2 실험 조합 (A/B testing)

```
Config A: 모두 OFF          = baseline (현재 MPS, 격리 없음)
Config B: S만 ON             = MIG-like 40:60
Config C: S + P ON           = MIG-like + priority
Config D: S + P + T ON       = MIG-like + priority + TTI throttle
Config E: S + P + T + C ON   = BORA full (우리 시스템)
Config F: C만 ON             = cluster placement만 (시간축 없이)

→ 각 단계별 개선을 독립적으로 보여줌
→ "어떤 메커니즘이 얼마나 기여하는가" ablation study
```

## 4. 구현 계획

### 4.1 Phase 1-A: Stream Priority Controller [P]

```python
class PriorityController:
    """ON/OFF 가능한 CUDA stream priority 제어."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        if enabled:
            # L1: highest priority (-5 on A100)
            self.l1_stream = cudart.cudaStreamCreateWithPriority(
                cudart.cudaStreamNonBlocking, -5)
            # AI: lowest priority (0 on A100)
            self.ai_stream = cudart.cudaStreamCreateWithPriority(
                cudart.cudaStreamNonBlocking, 0)
        else:
            # 둘 다 default
            self.l1_stream = cudart.cudaStreamCreate()
            self.ai_stream = cudart.cudaStreamCreate()
    
    def get_l1_stream(self):
        return self.l1_stream
    
    def get_ai_stream(self):
        return self.ai_stream
```

이미 검증된 데이터:
- 합성 실험: priority로 93% 간섭 감소
- 실제 cuPHY: 15-27% 감소

### 4.2 Phase 1-B: SM Limiter [S]

```python
class SMLimiter:
    """ON/OFF 가능한 MIG-like SM 비율 제한."""
    
    def __init__(self, enabled=True, l1_pct=40, ai_pct=60):
        self.enabled = enabled
        self.l1_pct = l1_pct if enabled else 100
        self.ai_pct = ai_pct if enabled else 100
    
    def get_l1_env(self):
        return {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(self.l1_pct)}
    
    def get_ai_env(self):
        return {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(self.ai_pct)}
```

이미 검증된 데이터:
- MIG-like 40:60으로 간섭 3.70x → 1.99x (46% 감소)

### 4.3 Phase 1-C: TTI-aware Throttle Controller [T]

```python
class TTIThrottleController:
    """ON/OFF 가능한 TTI-aware AI throttling.
    L1 active window 동안 AI kernel launch를 지연."""
    
    def __init__(self, enabled=True, tti_ms=1.0, l1_budget_ms=0.5):
        self.enabled = enabled
        self.tti_ms = tti_ms
        self.l1_budget_ms = l1_budget_ms
        self.ai_budget_ms = tti_ms - l1_budget_ms
    
    def should_launch_ai(self, time_in_tti_ms):
        """L1 active window인지 확인."""
        if not self.enabled:
            return True  # always launch
        # L1 budget 이후에만 AI launch 허용
        return time_in_tti_ms > self.l1_budget_ms
    
    def throttle_ai(self):
        """L1 active 중 AI를 잠시 멈춤."""
        if not self.enabled:
            return
        # AI stream에 wait event 삽입
        # 또는 AI batch size를 줄여서 bandwidth 사용 감소
```

구현 방법 후보:
1. **CUDA Event 기반**: L1 완료 event를 AI stream에서 wait
2. **Kernel launch rate 제한**: L1 active 중 AI kernel launch를 Python에서 지연
3. **AI batch size 동적 조절**: L1 active 중 batch 줄여서 bandwidth 감소

### 4.4 Phase 1-D: Cluster Placement Scheduler [C]

```python
class ClusterScheduler:
    """ON/OFF 가능한 HBM-pressure-aware placement."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.node_profiles = {}  # node → L1 load + AI bandwidth
    
    def profile_node(self, node_id, l1_cells, ai_bandwidth_gbps):
        """각 노드의 L1 부하 + AI bandwidth 프로파일."""
        self.node_profiles[node_id] = {
            "l1_cells": l1_cells,
            "ai_bw": ai_bandwidth_gbps,
            "pressure": l1_cells * ai_bandwidth_gbps  # 간단한 pressure metric
        }
    
    def place_ai(self, ai_workload):
        """AI workload를 pressure가 낮은 노드에 배치."""
        if not self.enabled:
            return "round_robin"  # 균등 배치
        
        # pressure가 낮은 노드 선택
        sorted_nodes = sorted(self.node_profiles.items(), 
                             key=lambda x: x[1]["pressure"])
        return sorted_nodes[0][0]  # 가장 여유 있는 노드
```

## 5. 실험 계획

### 5.1 실험 매트릭스

```
워크로드:
  L1: cuPHY CUDA Graph 8cell, 4T4R
  AI: Qwen-72B 4GPU tensor parallel (NVLink)

스케일: 1N, 2N, 4N (각 4GPU)

Configs:
  A: 모두 OFF (baseline)
  B: S만 ON (MIG-like 40:60)
  C: S+P ON (MIG + priority)
  D: S+P+T ON (MIG + priority + TTI throttle)
  E: S+P+T+C ON (BORA full)

측정:
  - L1 latency: per-GPU mean, P95, P99
  - TTI miss rate: % of iterations > 1ms
  - AI throughput: Qwen-72B inference/sec (per-node, total cluster)
  - GPU utilization: %
  - HBM bandwidth utilization: GB/s

총 실험 수: 5 configs × 3 scales = 15 실험
```

### 5.2 이미 있는 데이터 매핑

```
Config A (모두 OFF):
  → Exp-17 graph_intf: 72B 없지만 GPT-2/ResNet 데이터 있음
  → Exp-18 multi-node: 독립 AI, 격리 없음 데이터 있음

Config B (S만 ON):
  → Exp-19 mig72: 32B/72B MIG-like 40:60, 1/2/4N 데이터 있음 ✅

Config C (S+P):
  → test_priority_real.py: cuPHY + priority 데이터 있음 (하지만 MIG-like와 결합 안 함)
  → 추가 실험 필요

Config D (S+P+T):
  → 아직 없음, TTI throttle 구현 필요

Config E (BORA full):
  → 아직 없음, cluster scheduler 구현 필요
```

### 5.3 실험 순서

```
Step 1: Config C 실험 (S+P)
  → 기존 MIG-like 40:60 + stream priority 결합
  → 72B 1/2/4N에서 miss rate 변화 측정
  → 예상: Config B의 miss 10% → Config C에서 5%?

Step 2: TTI throttle 구현 + Config D 실험 (S+P+T)
  → L1 active window 동안 AI throttle
  → 예상: miss 5% → 2%?

Step 3: Cluster scheduler 구현 + Config E 실험 (BORA full)
  → bandwidth-aware 배치
  → 4N에서 worst-case GPU miss 균등화
  → 예상: miss 2% → <1%?

Step 4: 전체 비교 (A vs B vs C vs D vs E)
  → ablation study figure
  → AI throughput vs TTI miss rate trade-off curve
```

## 6. 예상 결과 Figure

### Figure 1: TTI Miss Rate vs Config (Ablation)

```
TTI miss rate (%)
  12 |  ██
  10 |  ██
   8 |  ██  ██
   6 |  ██  ██
   4 |  ██  ██  ██
   2 |  ██  ██  ██  ██
   0 |  ██  ██  ██  ██  ██
     |  A   B   C   D   E
     
  A=baseline, B=MIG, C=+Priority, D=+TTI, E=+Cluster (BORA)
```

### Figure 2: AI Throughput vs TTI Miss Rate (Trade-off)

```
AI throughput (inf/s)
     |           E●
     |        D●
     |     C●
     |  B●
     | A●
     |__________________
     0  2  4  6  8  10  TTI miss (%)
     
  → BORA(E)가 Pareto optimal: 높은 AI throughput + 낮은 miss
```

### Figure 3: Multi-node Scalability

```
Total AI throughput
     |              E-4N ●
     |         E-2N ●
     |    E-1N ●
     |   B-4N ●
     | B-1N ●
     |__________________
     1N  2N  3N  4N  nodes
     
  → BORA(E)가 더 steep하게 스케일 (per-node miss가 낮으니까)
```

## 7. 타임라인

```
Week 1: Config C 실험 (MIG + Priority)
  - priority controller를 기존 MIG-like 스크립트에 통합
  - 72B 1/2/4N 실험

Week 2: TTI throttle 구현 + Config D
  - CUDA event 기반 L1 window 감지
  - AI launch rate 제한 구현
  - 72B 실험

Week 3: Cluster scheduler 구현 + Config E
  - bandwidth 프로파일링
  - placement policy
  - 4N 실험

Week 4: 전체 비교 + AI throughput 측정
  - Config A~E ablation
  - trade-off curve

Week 5-6: 성능 모델 + 추가 실험
  - bandwidth → miss 예측 수식
  - 스케일 확장 (가능하면 8N)

Week 7-8: 논문 작성
  - SC 포맷 (12 pages)
```

## 8. 일반성 포장

```
본문 구조:
  "We present BORA, a GPU orchestration framework for environments where
   periodic hard-real-time tasks share GPU resources with best-effort 
   AI workloads."

  Section 2: Problem — GPU HBM bandwidth contention
  Section 3: Case Study — AI-RAN (5G L1 + LLM inference)
  Section 4: BORA Design — TTI-aware + cluster-wide
  Section 5: Evaluation — NERSC Perlmutter, 16 A100 GPUs
  Section 6: Generalization — other real-time + AI scenarios
  
일반화 가능한 시나리오:
  - 자율주행: sensor processing (real-time) + planning AI (best-effort)
  - 금융: market data processing (real-time) + risk model (best-effort)
  - 로봇: control loop (real-time) + perception AI (best-effort)
  
→ AI-RAN을 대표 케이스로 깊게 평가하되,
   "periodic real-time + AI on shared GPU" 일반 프레임워크로 포지셔닝
```
