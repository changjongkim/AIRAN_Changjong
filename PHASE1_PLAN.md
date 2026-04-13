# Phase 1: AI-RAN Bandwidth-aware Resource Management System

## 논문 한 줄 요약

> AI-RAN에서 L1과 AI의 GPU 공존 시, 병목은 SM이 아닌 HBM Bandwidth이며,
> 이를 인지하는 3계층 최적화(코드/모델/오케스트레이터)로
> L1 SLA를 보장하면서 AI 처리량을 극대화한다.

---

## 1. Baseline 결과 — "왜 이 연구가 필요한가"의 정량적 근거

### 1.1 문제 1: Bandwidth 경합이 L1을 파괴한다

**실험: 4T4R × 20cell, A100-40GB (HBM 44%, NVIDIA PoC 수준)**

| AI Workload | L1 RX mean | vs baseline | 의미 |
|---|---|---|---|
| baseline (L1 solo) | 15.068ms | 1.00x | |
| + GPT-2 (124M, same GPU) | 58.278ms | **3.87x** | LLM inference로 L1 4배 느려짐 |
| + ResNet-50 bs128 | 61.202ms | **4.06x** | CNN inference도 4배 |
| + Qwen-7B (7B, same GPU) | 22.324ms | **1.48x** | 작은 LLM도 영향 |
| + HBM stress 2GB | 611.403ms | **40.57x** | bandwidth 포화 시 치명적 |

**→ 현실적 AI workload(GPT-2, ResNet)가 L1을 3.87~4.06x 느리게 만듦.**
NVIDIA PoC 논문은 "간섭 없다"고 했지만, **정량적으로 측정하면 간섭이 존재.**

### 1.2 문제 2: MIG의 SM 격리는 효과 없음 — Bandwidth가 진짜 병목

**실험: SM% sweep (4T4R × 20cell, 40GB)**

| SM% | RX mean | vs 100% |
|-----|---------|---------|
| 100% | 16.474ms | 1.00x |
| 50% | 16.421ms | 1.00x |
| 20% | 16.570ms | 1.01x |
| 10% | 17.010ms | 1.03x |

**→ SM을 10%로 줄여도 L1 변화 없음.** MIG가 제공하는 SM 격리는 무의미.

**원인 (cuPHY 소스코드 분석으로 규명):**
```cpp
// pusch_rx.cpp:5555 — cuPHY 내부에서 매 cell마다 강제 동기화
cudaStreamSynchronize(phase2Stream);
```
→ cell 간 직렬 실행 → SM 경합 자체가 발생하지 않음
→ **SM이 아닌 HBM Bandwidth가 L1-AI 간 유일한 경합 자원**

### 1.3 문제 3: Bandwidth 간섭은 L1 부하에 비례하여 악화

**실험: HBM threshold (1cell vs 20cell, 동일 stress)**

| HBM Copy | 1cell 간섭 | 20cell 간섭 | 악화 비율 |
|----------|-----------|------------|----------|
| 0.1GB | 1.62x | **4.42x** | 2.7배 |
| 0.5GB | 4.01x | **11.66x** | 2.9배 |
| 1.0GB | 6.73x | **21.12x** | 3.1배 |
| 2.0GB | 12.07x | **38.32x** | 3.2배 |

**→ L1 부하(cell 수)가 높을수록 같은 bandwidth stress에 3배 더 민감.**
→ 실제 기지국(20+ cell)에서는 가벼운 AI도 문제가 될 수 있음.

### 1.4 문제 4: cuPHY/pyAerial 코드 비효율이 L1 시간을 낭비

**cuPHY 소스코드에서 발견한 NVIDIA 자체 TODO:**

```python
# pyAerial pusch_rx.py:345 — NVIDIA 주석:
# TODO: Move this out of the real-time pipeline.
harq_buffer = cudart.cudaMalloc(harq_buffer_size)    # 매 TTI마다 GPU malloc
cudart.cudaStreamSynchronize(self.cuda_stream)        # 매 TTI마다 sync
```

```cpp
// pusch_rx.cpp:4190 — NVIDIA 주석:
// @todo: Potential optimization opportunity: carry out all the output
// transfers by using a single cudaMemcpyAsync.
```

**→ L1이 불필요한 오버헤드로 느림 → AI에 줄 수 있는 시간이 줄어듦**
→ 이걸 고치면 L1이 빨라지고 → AI 시간 증가 → GPU 활용률 향상

### 1.5 문제 5: 간섭 시작/해소는 즉각적 — reactive 정책이 유효

**실험: Dynamic flash crowd (1cell)**

| Mode | RX mean | vs baseline |
|------|---------|-------------|
| baseline | 1.932ms | 1.00x |
| HBM 투입 (t=10s) | 40.354ms | **20.9x** (즉시 간섭) |
| HBM 제거 (t=10s) | 1.949ms | **1.01x** (즉시 복구) |

**→ AI를 중단하면 L1이 즉시 baseline으로 복구. 전환 지연 ~0.**
→ 복잡한 예측 모델 없이도 "SLA 위반 감지 → AI 중단" 정책이 유효.

---

## 2. 3계층 최적화 아키텍처

```
Layer 3: Bandwidth-aware Orchestrator (클러스터 레벨)
  ┌──────────────────────────────────────────────────┐
  │  "AI workload X를 어느 노드/GPU에 배치할까?"      │
  │  간섭 모델로 예측 → SLA 내 최적 배치 결정         │
  │  SLA 위반 감지 → AI 즉시 중단/migration           │
  │                                                    │
  │  입력: 각 노드의 L1 부하 + AI bandwidth 프로파일   │
  │  출력: AI 배치 결정 + migration 트리거             │
  └──────────────────────────────────────────────────┘
                        ↑ uses
Layer 2: Bandwidth Interference Model (예측 엔진)
  ┌──────────────────────────────────────────────────┐
  │  L1_latency = baseline(cells) × (1 + α × BW)    │
  │                                                    │
  │  α = f(num_cells, hbm_total_gb, antenna_config)  │
  │    α(1cell, 80GB)  ≈ 5.7 / GB                    │
  │    α(20cell, 80GB) ≈ 15.0 / GB                   │
  │    α(20cell, 40GB) ≈ 19.2 / GB                   │
  │                                                    │
  │  입력: AI workload의 실측 bandwidth (GB/s)         │
  │  출력: 예상 L1 latency 증가율                      │
  └──────────────────────────────────────────────────┘
                        ↑ enables
Layer 1: cuPHY System Optimization (코드 레벨)
  ┌──────────────────────────────────────────────────┐
  │  a. HARQ buffer pre-allocation → malloc 오버헤드 제거│
  │  b. D2H copy 통합 → DMA 효율 증가                 │
  │  c. (Optional) Multi-cell CUDA Graph → 병렬화     │
  │                                                    │
  │  효과: L1이 빨라짐 → idle 시간 증가 → AI 시간 증가 │
  │  + 간섭 감소 (L1-AI 겹치는 시간 줄어듦)            │
  └──────────────────────────────────────────────────┘
```

---

## 3. 구현 및 실험 계획

### Phase 1-A: 코드 레벨 최적화 (Layer 1)

#### 실험 A1: HARQ Buffer Pre-allocation

**현재 코드 (pusch_rx.py:345-380):**
```python
# 매 호출마다:
harq_buffer = cudart.cudaMalloc(harq_buffer_size)
cudart.cudaStreamSynchronize(self.cuda_stream)
# ... 사용 ...
cudart.cudaFree(harq_buffer)
```

**최적화:**
```python
# 초기화 시 한번만:
self.harq_pool = cudart.cudaMalloc(max_harq_buffer_size * max_ues)

# 매 호출마다:
harq_buffer = self.harq_pool + offset  # 포인터 산술, malloc 없음
# cudaStreamSynchronize 제거 가능 (이전 데이터 없으므로 memset 불필요)
```

**실험 설계:**
```
Baseline: 현재 pyAerial, 4T4R × 20cell, 40GB
  → per-cell latency, 총 latency, 100 iterations

최적화 후: HARQ pool allocator 적용
  → 동일 조건에서 per-cell latency 비교

측정:
  - per-cell latency 감소량 (예상: 50~100us/cell)
  - 20cell 총 latency 감소량 (예상: 1~2ms)
  - AI 간섭 변화 (L1이 빨라지면 간섭도 줄어드는가?)
```

#### 실험 A2: D2H Copy 통합

**현재 (pusch_rx.cpp:4190+):**
- TB payload, CB CRC, TB CRC를 각각 개별 cudaMemcpyAsync
- NVIDIA TODO: "single cudaMemcpyAsync로 통합 가능"

**최적화:**
- 출력 버퍼를 연속 메모리로 할당
- 1회 cudaMemcpyAsync로 전체 D2H copy

**실험 설계:**
```
Baseline: 현재 코드, Nsight Systems로 D2H 시간 측정
최적화 후: 통합 D2H, 동일 측정
측정: D2H 오버헤드 감소율
```

#### 실험 A3: 코드 최적화 전후 AI 간섭 비교

```
Baseline:    현재 코드 + GPT-2 = 3.87x 간섭 (40GB, 20cell)
최적화 후:   HARQ + D2H 최적화 + GPT-2 = ?x 간섭

가설: L1이 빨라지면 (예: 15ms → 12ms) AI와 겹치는 시간이 줄어
     간섭도 감소할 것 (예: 3.87x → 3.0x?)

이게 확인되면:
→ "코드 최적화가 AI-RAN 간섭을 완화한다"는 새로운 contribution
→ "L1을 빠르게 만드는 것이 GPU 공유의 가장 직접적 해결책"
```

### Phase 1-B: Bandwidth 간섭 모델 (Layer 2)

#### 실험 B1: AI Workload별 실제 Bandwidth 프로파일

```
nvidia-smi dmon -s m (memory bandwidth utilization) 으로:

  GPT-2 inference (batch=4, seq=512):
    → HBM read bandwidth: ??? GB/s
    → HBM write bandwidth: ??? GB/s

  ResNet-50 bs=128:
    → HBM read bandwidth: ??? GB/s
    → HBM write bandwidth: ??? GB/s

  Qwen-7B inference:
    → HBM read bandwidth: ??? GB/s
    → HBM write bandwidth: ??? GB/s

  HBM stress 1GB:
    → HBM read bandwidth: ??? GB/s (≈ max)
    → HBM write bandwidth: ??? GB/s (≈ max)
```

이 데이터가 모델의 BW 변수에 들어감.

#### 실험 B2: 간섭 모델 피팅

```
이미 있는 데이터로 α 계산:

  L1_latency = baseline × (1 + α × BW_effective)

데이터 포인트:
  - HBM stress 0.1GB ~ 8GB → BW 실측 (B1에서)
  - 1cell, 20cell, 40GB, 80GB → 4개 환경

각 환경에서 α 피팅:
  α(1cell, 80GB)  = (latency/baseline - 1) / BW
  α(20cell, 80GB) = ...
  α(20cell, 40GB) = ...

α의 일반화:
  α = β × num_cells / hbm_total_gb
  → β를 피팅하면 임의 환경에서 예측 가능
```

#### 실험 B3: 모델 검증

```
예측:
  GPT-2의 BW = X GB/s (B1에서 측정)
  α(20cell, 40GB) = 19.2/GB
  예상 간섭 = 1 + 19.2 × X = ???

실측: 3.87x

오차 = |예상 - 실측| / 실측
→ 오차 <15%면 모델 유효
```

### Phase 1-C: Bandwidth-aware 오케스트레이터 (Layer 3)

#### 실험 C1: 오케스트레이터 시뮬레이터

```
시뮬레이션 환경:
  4 노드 × 4 GPU, 각 노드 L1 부하가 시간에 따라 변화
  트래픽 패턴: 24시간 주기 (새벽 low, 낮 high, 저녁 peak)
  AI workload: GPT-2, ResNet-50, Qwen-7B 혼합

비교:
  A) 정적 MIG 40:60 (NVIDIA 방식)
     - 각 GPU를 40% L1, 60% AI로 고정
     - 트래픽 낮을 때: L1 파티션 40% 놀고 있음
     - 트래픽 peak: L1에 40%로 부족할 수 있음

  B) 동적 오케스트레이터 (우리 제안)
     - 간섭 모델로 각 GPU의 AI 수용 가능량 실시간 계산
     - 트래픽 낮을 때: L1에 20%, AI에 80%
     - 트래픽 peak: L1에 100%, AI 중단 또는 다른 노드로 migration
     - migration 비용 모델 포함

측정:
  - 전체 AI 처리량 (tokens/s, images/s)
  - L1 SLA 위반율 (latency > deadline)
  - GPU 활용률 (%)
```

#### 실험 C2: Migration 비용 실측

```
Perlmutter에서 실측:
  GPU0 → GPU1 (같은 노드, NVLink):
    Qwen-7B (14GB) 전송 시간: ??? ms
    ResNet-50 (~100MB) 전송 시간: ??? ms

  Node0 → Node1 (Slingshot interconnect):
    Qwen-7B (14GB) 전송 시간: ??? ms

이 데이터가 오케스트레이터의 migration 결정에 사용됨:
  migration_benefit = interference_reduction × remaining_time
  migration_cost = transfer_time + warmup_time
  → benefit > cost 일 때만 migration
```

---

## 4. 예상 논문 결과 (Expected Outcomes)

```
1. 코드 최적화:
   "HARQ pre-alloc + D2H 통합으로 L1 latency 15~20% 감소
    → AI 간섭 3.87x → 3.0x로 완화"

2. 간섭 모델:
   "L1_latency = baseline × (1 + α × BW)
    α = β × cells / hbm_gb
    → AI workload의 bandwidth만 알면 간섭을 15% 오차 내로 예측"

3. 오케스트레이터:
   "동적 오케스트레이터가 정적 MIG 대비
    AI 처리량 40~60% 향상, L1 SLA 위반율 0%"
```

---

## 5. 타임라인

```
Week 1-2: Phase 1-A (코드 최적화)
  - HARQ pre-alloc 구현 + 실험 A1
  - D2H copy 통합 + 실험 A2
  - 간섭 변화 측정 + 실험 A3

Week 3: Phase 1-B (간섭 모델)
  - bandwidth 프로파일 실측 + 실험 B1
  - α 피팅 + 실험 B2
  - 모델 검증 + 실험 B3

Week 4-5: Phase 1-C (오케스트레이터)
  - 시뮬레이터 구현 + 실험 C1
  - migration 비용 실측 + 실험 C2
  - 정적 MIG vs 동적 오케스트레이터 비교

Week 6: 논문 작성
```
