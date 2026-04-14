# SC Paper: TTI-Aware HBM-Bandwidth-Oriented GPU Orchestration

## 논문 제목

**"TTI-Aware HBM-Bandwidth-Oriented GPU Orchestration for Real-Time and AI Workloads on Shared GPU Clusters"**

---

## 1. 시스템 이름: **BORA** (Bandwidth-Oriented RAN-AI orchestrator)

---

## 2. 핵심 아이디어: 프로세스 간 HBM 접근 Coordination

### 왜 MIG로는 부족한가

```
MIG (정적 분할):
  GPU를 물리적으로 나눔 → SM, Memory 격리
  하지만 HBM bandwidth는 여전히 공유
  → L1과 AI가 동시에 HBM을 때리면 bank conflict 발생
  → MIG는 이걸 막을 수 없음
  
  그리고 정적 → L1이 idle일 때 파티션 낭비
```

### BORA의 접근: 동적 coordination

```
직접적인 HBM bank partitioning은 현재 GPU 아키텍처에서 불가능하다.
대신 BORA는 프로세스 간 HBM 접근 시점을 조율(coordinate)하여
bandwidth contention을 최소화한다.
이는 bank conflict 빈도를 간접적으로 줄이는 효과를 가진다.

핵심 차이:
  MIG:  "너는 이 파티션, 나는 저 파티션" (정적, 공간 분할, 낭비)
  BORA: "지금 내가 쓰니까 너 잠깐 기다려" (동적, 시간 조율, 효율적)
```

### 프로세스 간 Signal 기반 Coordination

```
┌─ L1 Process ──────────────────────────────────────┐
│                                                    │
│  L1 kernel start → SIGNAL: "L1 ACTIVE"            │
│  ▓▓▓▓▓▓▓▓▓▓ (GPU compute, HBM read/write)        │
│  L1 kernel end   → SIGNAL: "L1 IDLE"              │
│                                                    │
└────────────────────────────────────────────────────┘
         ↕ IPC signal (CUDA event / shared memory)
┌─ AI Process ──────────────────────────────────────┐
│                                                    │
│  WAIT for "L1 IDLE" signal                         │
│  ████████████████ (AI inference, HBM full speed)   │
│  CHECK for "L1 ACTIVE" signal → throttle/pause     │
│                                                    │
└────────────────────────────────────────────────────┘

시간축:
├── TTI 0 (1ms) ────────────────┤── TTI 1 (1ms) ──...
│▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░│▓▓▓▓░░░░░░░░░░░░░...
│ L1  │ AI (full bandwidth)    │ L1  │ AI            
│     │                        │     │
│ → AI가 HBM을 안 씀           │     │
│   (bank conflict 최소)       │     │
```

### 구현 메커니즘

```
Signal 전달 방법 (후보):
  1. CUDA Event: L1 stream에 event record → AI stream에서 wait
     → GPU 레벨에서 직접 동기화, 가장 낮은 latency
     
  2. Shared Memory Flag: L1이 flag 세팅 → AI가 polling
     → CPU 레벨, 유연하지만 latency 높음
     
  3. MPS + Stream Priority: L1=high priority stream
     → GPU scheduler가 자동으로 L1 우선 → 암묵적 coordination
     → 가장 간단하지만 제어 정밀도 낮음

BORA는 세 가지를 조합:
  - Stream Priority: 항상 ON (기본 보호막)
  - CUDA Event: TTI-aware fine-grained coordination
  - Shared Flag: cluster-wide 상태 공유
```

## 3. 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                      BORA Orchestrator                        │
│                                                              │
│  ┌── Per-GPU: Bandwidth Coordinator ──────────────────────┐ │
│  │                                                        │ │
│  │  ┌─────────────┐  ┌────────────┐  ┌────────────────┐ │ │
│  │  │ Priority    │  │ SM Limit   │  │ TTI-aware      │ │ │
│  │  │ Controller  │  │ (MIG-like) │  │ Throttle       │ │ │
│  │  │ [ON/OFF]    │  │ [ON/OFF]   │  │ [ON/OFF]       │ │ │
│  │  │             │  │            │  │                │ │ │
│  │  │ L1=high     │  │ L1=40%SM   │  │ L1 active →   │ │ │
│  │  │ AI=low      │  │ AI=60%SM   │  │ AI pause      │ │ │
│  │  └─────────────┘  └────────────┘  │ L1 idle →     │ │ │
│  │                                    │ AI resume     │ │ │
│  │  Signal: CUDA Event / Shared Mem   └────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌── Per-Cluster: Placement Scheduler ────────────────────┐ │
│  │  [STATIC / DYNAMIC]                                    │ │
│  │                                                        │ │
│  │  Profile: per-node L1 load + AI bandwidth pressure     │ │
│  │  Policy:  heavy AI → L1-light nodes                    │ │
│  │           light AI → L1-heavy nodes OK                 │ │
│  │  Rebalance: worst-case GPU miss → migrate AI shard     │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## 3. ON/OFF 설계 — 각 메커니즘 독립 평가

모든 최적화 메커니즘을 독립적으로 ON/OFF 가능하게 설계하여,
각각의 contribution을 개별적으로 측정 가능.

### 4.1 메커니즘 목록

| ID | 메커니즘 | 수준 | ON일 때 | OFF일 때 (baseline) |
|----|---------|------|---------|-------------------|
| **P** | Stream Priority | GPU | L1=high(-5), AI=low(0) priority | 둘 다 default(0) |
| **S** | SM Limit (MIG-like) | GPU | L1=40% SM, AI=60% SM | 둘 다 100% SM |
| **T** | TTI-aware Coordination | GPU | L1 active → signal → AI pause/throttle; L1 idle → AI resume | AI 항상 실행 (signal 무시) |
| **C** | Cluster Placement | Cluster | HBM-pressure-aware 배치 | 균등 배치 (round-robin) |

**T가 BORA의 핵심 contribution** — 프로세스 간 signal 기반 동적 coordination.
P, S는 기존 메커니즘(MIG/MPS), C는 클러스터 레벨 확장.
T가 없으면 "정적 분할"에 불과하고, T가 있어야 "동적 조율"이 됨.

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

### 5.3 Phase 1-C: TTI-aware Coordination Controller [T] — 핵심 메커니즘

**BORA의 핵심: L1과 AI 프로세스 간 HBM 접근 시점 조율.**

```python
class TTICoordinator:
    """ON/OFF 가능한 TTI-aware 프로세스 간 coordination.
    L1 active window 동안 AI의 HBM 접근을 억제하여
    bank conflict 빈도를 간접적으로 최소화."""
    
    def __init__(self, enabled=True, tti_ms=1.0, l1_budget_ms=0.5):
        self.enabled = enabled
        self.tti_ms = tti_ms
        self.l1_budget_ms = l1_budget_ms
        
        if enabled:
            # IPC signal: CUDA event for GPU-level coordination
            self.l1_active_event = cudart.cudaEventCreate()
            self.l1_idle_event = cudart.cudaEventCreate()
            # Shared memory flag for CPU-level coordination
            self.shm_flag = shared_memory.SharedMemory(create=True, size=1)
    
    def l1_start(self, l1_stream):
        """L1 process calls this when L1 kernel begins."""
        if not self.enabled:
            return
        cudart.cudaEventRecord(self.l1_active_event, l1_stream)
        self.shm_flag.buf[0] = 1  # L1 ACTIVE
    
    def l1_end(self, l1_stream):
        """L1 process calls this when L1 kernel completes."""
        if not self.enabled:
            return
        cudart.cudaEventRecord(self.l1_idle_event, l1_stream)
        self.shm_flag.buf[0] = 0  # L1 IDLE
    
    def ai_wait_for_idle(self, ai_stream):
        """AI process waits until L1 is idle before heavy HBM access."""
        if not self.enabled:
            return  # no coordination, AI runs freely
        # GPU-level: AI stream waits for L1 idle event
        cudart.cudaStreamWaitEvent(ai_stream, self.l1_idle_event)
    
    def ai_should_throttle(self):
        """AI process checks if L1 is currently active."""
        if not self.enabled:
            return False
        return self.shm_flag.buf[0] == 1  # L1 is active → throttle
```

**세 가지 coordination 수준:**

```
Level 1: Stream Priority (항상 ON, 암묵적)
  → GPU scheduler가 L1 커널을 AI보다 먼저 실행
  → 별도 signal 없이 자동
  → 효과: 간섭 15-93% 감소 (이미 실측)

Level 2: CUDA Event (GPU 레벨, 명시적)
  → L1 idle event를 AI stream에서 wait
  → AI는 L1이 끝난 후에만 heavy kernel launch
  → GPU-to-GPU latency ~수 us
  
Level 3: Shared Memory Flag (CPU 레벨, 유연)
  → L1이 flag 세팅 → AI가 polling하여 batch size 조절
  → CPU-to-CPU, latency ~수십 us
  → AI의 bandwidth 사용량을 연속적으로 조절 가능

BORA는 세 수준을 조합:
  Priority: 기본 보호막 (worst case에도 L1 우선)
  CUDA Event: TTI 단위 fine-grained coordination
  Shared Flag: AI throughput 동적 조절 (batch size 등)
```

**기존 MIG와의 차이:**

```
MIG (NVIDIA 권장):
  정적 공간 분할 → SM/Memory 격리 → bandwidth는 공유
  → "항상 40:60으로 나눠놓고, 나머지는 운에 맡김"
  → L1 idle 시간에 40% 파티션 낭비

BORA:
  동적 시간 조율 → bandwidth 접근 시점 조율
  → "L1이 쓸 때 AI는 기다리고, L1이 안 쓸 때 AI가 전체를 씀"
  → L1 idle 시간에 AI가 100% GPU 활용
  → bank conflict을 간접적으로 최소화
```

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
