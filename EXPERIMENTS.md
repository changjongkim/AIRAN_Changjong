# Phase 0 실험 계획

## 목표
"GPU에서 5G L1 처리와 AI 워크로드가 공존할 때, 실제로 간섭이 발생하는가?" 검증

## 실험 1: L1 Baseline (간섭 없음)

**목적**: 순수 L1 커널 성능 기준점 확보

```
cuPHY L1 파이프라인만 단독 실행
├── PUSCH Rx (Uplink): FFT → 채널추정 → MIMO detection → LDPC decode
├── PDSCH Tx (Downlink): LDPC encode → Modulation → Precoding → IFFT
└── 반복 실행 (1000+ iterations)
```

**측정 항목**:
- 커널별 실행 시간 (FFT, ChEst, LDPC encode/decode, MIMO 등)
- 전체 TTI 처리 시간 (P50, P95, P99)
- GPU utilization, HBM bandwidth, SM occupancy
- HBM 사용량 baseline

**방법**: pyAerial `example_pusch_simulation.ipynb` 기반으로 커널 타이밍 코드 추가

## 실험 2: AI 간섭 모드

**목적**: AI 워크로드가 L1 latency에 미치는 영향 측정

```
L1 파이프라인 (MPS Process 1)  +  AI inference (MPS Process 2)
```

| AI 워크로드 | 특성 | 왜 이걸 고르나 |
|-------------|------|----------------|
| ResNet-50 inference | SM-heavy, HBM 적음 | compute 간섭 측정 |
| GPT-2 inference | HBM-heavy, attention | 메모리 대역폭 간섭 측정 |
| LLaMA-7B inference | HBM 대량 점유 | 대형 모델 실제 시나리오 |

**측정 항목**:
- L1 커널별 latency 변화 (실험 1 baseline 대비)
- Jitter = stdev(TTI latency) / mean(TTI latency)
- TTI deadline miss rate = count(latency > 1ms) / total
- AI 워크로드 throughput 영향

**방법**: MPS로 두 프로세스 동시 실행, CUDA event로 각각 타이밍

## 실험 3: HBM 포화 모드

**목적**: HBM 사용량이 L1 성능에 미치는 영향 정량화

```
HBM 점유율을 단계적으로 올리며 L1 성능 측정
├── 0%  (baseline = 실험 1)
├── 30% (작은 AI 모델 수준)
├── 50% (중간 모델)
├── 70% (큰 모델)
├── 85% (LLM 수준)
└── 95% (극한 상황)
```

**방법**: `torch.zeros(SIZE).cuda()`로 HBM 점유 후 L1 실행

**측정 항목**:
- HBM 점유율별 L1 커널 latency 변화 곡선
- HBM bandwidth utilization 변화
- 성능이 급격히 떨어지는 임계점 (knee point) 식별

## 실험 순서와 산출물

```
Week 1: 실험 1 (Baseline)
  └── L1 커널별 latency 분포표 + CDF 그래프

Week 2: 실험 3 (HBM 포화) — 실험 2보다 단순해서 먼저
  └── HBM 점유율 vs L1 latency 곡선

Week 3: 실험 2 (AI 간섭)
  └── 워크로드 조합별 jitter + deadline miss rate 표

최종 산출물: "간섭이 실재한다"는 데이터 → Go/No-Go 판단
```

## 실험 코드 구조

```
/pscratch/sd/s/sgkim/kcj/AI-RAN/
├── experiments/
│   ├── exp1_baseline.py        # L1 단독 latency 측정
│   ├── exp2_ai_interference.py # L1 + AI 동시 실행
│   ├── exp3_hbm_saturation.py  # HBM 점유율별 L1 성능
│   ├── workloads/
│   │   ├── resnet50_loop.py    # ResNet-50 무한 inference
│   │   ├── gpt2_loop.py       # GPT-2 무한 inference
│   │   └── hbm_filler.py      # HBM 점유 유틸리티
│   └── utils/
│       ├── timing.py           # CUDA event 타이밍 래퍼
│       └── plotting.py         # CDF, latency 시각화
├── results/                    # 실험 결과 저장
└── jobs/
    ├── run_exp1.sh             # SLURM job script
    ├── run_exp2.sh
    └── run_exp3.sh
```
