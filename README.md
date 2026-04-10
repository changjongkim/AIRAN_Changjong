# AI-RAN GPU Resource Contention Research

## 1. Introduction

### 문제: GPU 위의 두 세계 — Hard Real-time RAN vs Best-effort AI

5G/6G 기지국의 물리계층(L1) 처리가 CPU에서 GPU로 이동하고 있다.
NVIDIA Aerial(cuPHY)은 FFT, 채널추정, MIMO detection, LDPC 디코딩을
GPU CUDA 커널로 가속하여, 기존 CPU 기반 대비 수십 배의 처리량을 달성한다.

그러나 L1 전용으로 GPU를 할당하면 **GPU 활용률이 낮다**.
L1은 TTI(0.5~1ms) 주기로 burst 처리 후 idle 상태가 되므로,
GPU의 상당 부분이 놀고 있는 셈이다.

**AI-RAN**은 이 idle 시간에 AI 워크로드(추론, 채널 예측, 스케줄링 최적화 등)를
같은 GPU에서 함께 돌려 활용률을 극대화하자는 아이디어다.

문제는 **L1과 AI의 성격이 완전히 다르다는 것**:

| | L1 (RAN) | AI (Inference) |
|---|---|---|
| Deadline | **0.5ms** (hard real-time) | 수십~수백 ms (best-effort) |
| 1μs라도 늦으면 | 패킷 에러, 기지국 장애 | 처리량 감소 (허용 가능) |
| GPU 사용 패턴 | 짧은 burst, 주기적 | 연속적, bandwidth-heavy |

기존 GPU 스케줄링 연구는 10~100ms 단위의 응답 시간을 다루지만,
L1의 **0.5ms deadline**은 차원이 다른 문제다.

### NVIDIA의 현재 해법: MIG (Multi-Instance GPU)

NVIDIA는 MIG를 통해 GPU를 하드웨어 수준으로 분할하여 L1과 AI를 격리한다.
SM(compute)과 Memory(capacity)는 MIG로 완전히 나눌 수 있다.

**그러나 MIG에는 세 가지 한계가 있다**:

1. **HBM Bandwidth 공유**: MIG가 SM과 메모리는 나누지만,
   HBM bandwidth는 파티션 간에 여전히 공유된다.
   AI가 bandwidth를 많이 쓰면 L1 커널의 메모리 접근이 느려질 수 있다.

2. **정적 파티셔닝**: MIG 프로파일은 한번 설정하면 고정.
   트래픽이 적을 때도 L1 파티션이 자원을 점유하고,
   트래픽이 몰릴 때 AI 파티션을 줄일 수 없다.
   프로파일 변경에는 GPU reset이 필요하여 실시간 대응 불가.

3. **제한된 파티션 크기**: A100은 7등분(1g, 2g, 3g, 4g, 7g)만 가능.
   워크로드 특성에 맞는 세밀한 리소스 조절이 불가능하다.

### 우리의 연구 질문

> MIG가 SM 격리는 해주지만, (1) bandwidth는 여전히 공유되고,
> (2) 파티션이 정적이라 트래픽 변화에 대응 못하고,
> (3) L1 idle 시간에 자원이 낭비된다.
>
> **이 세 가지 한계를 극복하는 동적 리소스 관리 시스템을 어떻게 설계할 것인가?**

### 연구의 차별점 (Novelty)

1. **워크로드의 특수성**: 일반 GPU 공유가 아닌, 0.5ms hard real-time L1과
   best-effort AI의 공존이라는 극단적 시나리오를 다룸

2. **MIG 한계의 실측**: "MIG면 충분하다"는 통념을 깨고, bandwidth 공유로 인한
   간섭을 AI-RAN 워크로드에서 정량적으로 증명 (53~104x latency 증가 실측)

3. **동적 오케스트레이션**: 정적 MIG를 넘어, L1 트래픽 변화에 따라
   무중단으로 AI 워크로드를 투입/회수하는 멀티 노드 전략 제안

## 2. Background

### GPU 공유 메커니즘: MIG vs MPS

GPU에서 여러 워크로드를 동시에 돌리는 방법은 크게 세 가지다:

```
1. Time-slicing (기본)
   Process A → GPU 독점 → 완료 → Process B → GPU 독점
   한 번에 하나만 실행. 동시성 없음.

2. MPS (Multi-Process Service)
   Process A ─┐
              ├→ 같은 GPU에서 동시 실행 (SM을 소프트웨어로 분할)
   Process B ─┘
   CUDA context를 공유하여 여러 프로세스가 GPU를 진짜 동시에 사용.

3. MIG (Multi-Instance GPU)
   ┌─ MIG Instance 1 ─┐ ┌─ MIG Instance 2 ─┐
   │ 42 SM, 20GB       │ │ 56 SM, 20GB       │
   │ (하드웨어 격리)    │ │ (하드웨어 격리)    │
   └──────────────────┘ └──────────────────┘
   GPU를 물리적으로 분할. SM과 메모리 완전 격리. Fault isolation 제공.
```

| | Time-slicing | MPS | MIG |
|---|---|---|---|
| 동시 실행 | ❌ | ✅ | ✅ |
| SM 격리 | N/A | 소프트웨어 (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`) | 하드웨어 (고정 파티션) |
| Memory 격리 | N/A | 소프트웨어 (`set_per_process_memory_fraction`) | 하드웨어 |
| HBM Bandwidth | 독점 (교대) | **공유** | **공유** (핵심 한계) |
| Fault isolation | ✅ (프로세스 독립) | ❌ (한 프로세스 crash → 전체 영향) | ✅ |
| 동적 변경 | 즉시 | 즉시 | GPU reset 필요 |

**핵심**: MIG든 MPS든 **HBM bandwidth는 항상 공유**된다. 이것이 AI-RAN에서
L1과 AI가 같은 GPU에 있을 때 간섭이 발생하는 근본 원인이다.

### 본 연구에서의 MPS 활용 (MIG 에뮬레이션)

Perlmutter에서 MIG 활성화가 불가(admin 권한 필요)하므로, MPS를 이용하여
MIG의 동작을 에뮬레이션한다:

```
MIG 4g.40gb 파티션 에뮬레이션:
  → CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=57  (SM 57% = 4/7)
  → torch.cuda.set_per_process_memory_fraction(0.5)  (Memory 50%)
  → HBM bandwidth: 자연스럽게 공유 (MIG에서도 동일)
```

SM과 Memory는 MPS로 제한하고, bandwidth 공유는 MIG와 동일하게 발생하므로,
**bandwidth 간섭 측정**에 대해서는 MIG와 동등한 실험 조건을 구성할 수 있다.

### NVIDIA AI-RAN PoC (참조 논문)

NVIDIA는 GH200 서버에서 MIG 40:60 분할로 5G RAN + LLM 동시 실행을 시연하였다.

| 항목 | NVIDIA PoC 설정 |
|------|----------------|
| Server | NVIDIA MGX GH200 (Grace-Hopper), 2 GPU |
| GPU 분할 | MIG 40:60 (GPU 1을 2개 인스턴스로 분할) |
| RAN | NVIDIA Aerial SDK, 4T4R, 100MHz, 30kHz SCS |
| AI Workload | LLM inference (NeMo/NIM), 챗봇, 디지털 휴먼 |
| GPU 활용률 | RAN only: 30-40% → RAN+AI: ~100% |
| 주장 | MIG 격리로 RAN 성능 저하 없이 AI 병행 가능 |

**이 논문이 보여주지 않은 것**:
- HBM bandwidth 공유에 의한 정량적 간섭 수치
- MIG 파티션 크기별 L1 latency 변화
- Stress 강도에 따른 간섭 곡선 (threshold)
- 멀티 노드 환경에서의 간섭 전파 여부

### 본 연구와의 비교

| | NVIDIA PoC | 본 연구 |
|---|---|---|
| **Hardware** | GH200 (Grace-Hopper) | A100-SXM4 (Perlmutter) |
| **GPU Sharing** | MIG (하드웨어 격리) | MPS (소프트웨어 에뮬레이션) |
| **RAN Stack** | Aerial SDK (cuPHY) | Aerial SDK (cuPHY) — 동일 |
| **RAN Config** | 4T4R, 100MHz, 30kHz | 1T2R, 273PRB, 30kHz (동일 bandwidth) |
| **AI Workload** | LLM (NeMo/NIM) | GPT-2, ResNet-50, HBM stress |
| **측정 방식** | GPU utilization % | **L1 kernel latency (CUDA events)** |
| **핵심 질문** | "동시 실행 가능한가?" | **"간섭이 얼마나 발생하는가?"** |
| **결과** | "간섭 없다" (정성적) | **정량적 간섭 수치 제시** |
| **Contribution** | PoC 시연 | **bandwidth 공유 한계 실측 + 모델링** |

### Cell의 개념과 GPU 부하

5G 기지국(gNB)은 여러 개의 **Cell**을 동시에 서비스한다.
Cell은 하나의 독립적인 무선 커버리지 영역으로, 각 Cell이
자기만의 안테나 세트와 주파수 자원을 가진다.

**핵심: 각 Cell은 독립적인 L1 파이프라인을 GPU에서 돌려야 한다.**

```
1 Cell이 매 TTI (1ms)마다 하는 일:

  수신 (PUSCH Rx): 안테나 → FFT → 채널추정 → MIMO detection → LDPC decode
  송신 (PDSCH Tx): LDPC encode → Modulation → Precoding → IFFT → 안테나

  → 이 전체 파이프라인이 Cell 하나당 하나씩 필요
  → Cell이 N개면 GPU가 N배의 연산을 해야 함
```

즉 Cell 수 = GPU 부하의 배수. 안테나 수(MIMO)도 마찬가지:
- **1T2R** (안테나 1송신 2수신): 행렬 연산 작음
- **4T4R**: 행렬이 4배 커짐 → FFT, 채널추정, MIMO detection 모두 연산량 증가
- **64T64R** (Massive MIMO): 행렬이 64배 → GPU를 거의 다 씀

**Cell 수가 늘어나면 GPU 부하가 증가한다**:

| 구성 | Cell 수 | 안테나 | GPU HBM 사용 | GPU 활용률 |
|------|---------|--------|-------------|-----------|
| 우리 실험 (lightweight) | 1 | 1T2R | ~770MB (~2%) | ~6% |
| 우리 실험 (realistic) | 8 | 1T2R×8 | ~6.2GB (~15%) | ~30% |
| NVIDIA PoC 논문 | - | 4T4R | - | 30-40% |
| 실제 상용 기지국 | 20 | 4T4R×20 | 대부분 | 70-90% |

각 Cell은 독립적인 cuPHY 파이프라인 인스턴스를 사용하며,
Cell당 ~770MB의 GPU HBM을 점유한다. Cell이 많을수록 GPU 리소스 경쟁이 심해지고,
**AI 워크로드와의 간섭도 커진다.**

### AI-RAN Architecture

```
┌─────────────── AI-RAN Node (GPU Server) ───────────────┐
│                                                         │
│  ┌─────────────────── GPU (A100) ──────────────────┐   │
│  │                                                  │   │
│  │   ┌─── MIG Partition 1 ───┐ ┌─ MIG Partition 2 ─┐  │
│  │   │   L1 (cuPHY)         │ │  AI Workload      │   │
│  │   │   - FFT/IFFT         │ │  - Inference      │   │
│  │   │   - Channel Est.     │ │  - Training       │   │
│  │   │   - MIMO Detection   │ │  - Optimization   │   │
│  │   │   - LDPC Dec/Enc     │ │                   │   │
│  │   └──────────────────────┘ └───────────────────┘   │
│  │              ↕ HBM Bandwidth (공유!) ↕              │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  CPU: L2/L3 (OAI), Scheduling, Control Plane           │
└─────────────────────────────────────────────────────────┘
```

### L1 처리의 시간적 특성

```
TTI (1ms) 내 L1 burst 패턴:

시간 →
├──── TTI 0 ────┤──── TTI 1 ────┤──── TTI 2 ────┤
│▓▓▓░░░░░░░░░░░│▓▓▓░░░░░░░░░░░│▓▓▓░░░░░░░░░░░│
│ L1  │  idle   │ L1  │  idle   │ L1  │  idle   │
│burst│ (낭비)  │burst│ (낭비)  │burst│ (낭비)  │

▓ = L1 처리 (GPU active, ~0.3-1ms)
░ = GPU idle (AI를 넣을 수 있는 시간)

→ AI-RAN의 핵심 아이디어: idle 시간에 AI를 실행
→ 문제: L1 burst가 올 때 AI가 GPU를 놓아주지 않으면? → 간섭 발생
```

### Phase 0 실험으로 확인한 사실

| 실험 | 결과 | 의미 |
|------|------|------|
| 같은 GPU, MPS 공유 | **54.6x 느려짐** | 격리 없는 공유는 치명적 |
| 다른 GPU, 완전 분리 | **1.06x (변화 없음)** | GPU 분리하면 간섭 제거 (하지만 자원 낭비) |
| HBM idle 점유 | **변화 없음** | 메모리 점유만으로는 간섭 안 생김 |
| HBM bandwidth 경쟁 | **53~104x 느려짐** | bandwidth가 진짜 간섭 원인 |
| cuPHY + PyTorch MPS | **CUDA crash** | 멀티테넌시 격리 자체가 불완전 |

---

## 3. Environment Setup

### Perlmutter 환경 정보

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA A100-PCIE-40GB |
| Driver | 580.105.08 |
| CUDA | 12.9 (module: cudatoolkit/12.9) |
| Container Runtime | **Shifter** (NERSC 공식, 호스트 uid로 실행) |
| MPS | 사용 가능 (`nvidia-cuda-mps-control`) |
| MIG | Disabled (admin 권한 필요) |

### 디렉토리 구조

```
/pscratch/sd/s/sgkim/kcj/AI-RAN/
├── aerial-cuda-accelerated-ran/   # NVIDIA Aerial SDK 소스 (GitHub clone)
│   ├── cuPHY/                     # GPU-accelerated 5G PHY (L1)
│   ├── cuMAC/                     # GPU-accelerated L2 scheduler
│   ├── pyaerial/                  # Python API + ML/AI integration
│   ├── testBenches/               # GPU testbench (cubb_gpu_test_bench)
│   ├── testVectors/               # Test vector 저장소
│   ├── 5GModel/                   # MATLAB 기반 5G 파형 생성
│   └── build/                     # 빌드 결과물 (scratch에 영구 보존)
├── pyaerial_install/              # pyAerial pip install prefix
└── README.md                      # 이 파일
```

### 설치 과정

### Step 1: 소스 클론

```bash
mkdir -p /pscratch/sd/s/sgkim/kcj/AI-RAN
cd /pscratch/sd/s/sgkim/kcj/AI-RAN
git clone https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git
```

### Step 2: Shifter 이미지 Pull

```bash
shifterimg pull docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
```

Pull 후 READY 확인:
```bash
shifterimg images | grep aerial
# perlmutter docker  READY  ...  nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
```

> Shifter는 Docker 이미지를 Squashfs로 변환하므로 pull에 10~30분 소요.
> podman-hpc와 달리 **호스트 uid로 실행**되어 scratch 파일시스템에 직접 접근 가능.

컨테이너에 포함된 것:
- CUDA 12.9.1, Python 3.10.12
- LDPC decoder cubins (sm_80, sm_86, sm_90)
- Nsight Systems CLI 2025.3.1
- 빌드 의존성 (CLI11, HDF5 1.10.7, fmt, ZMQ, yaml, pybind11, CuPy 등)

### Step 3: CMake Configure + 빌드

Shifter 안에서 scratch 소스를 직접 빌드. `/tmp` 복사 불필요.

```bash
shifter --image=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb bash -c '
export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
cd $cuBB_SDK

# Configure — Python 3.10을 명시적으로 지정해야 함 (호스트 3.11과 충돌 방지)
cmake -Bbuild -GNinja \
  -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native \
  -DNVIPC_FMTLOG_ENABLE=OFF \
  -DASIM_CUPHY_SRS_OUTPUT_FP32=ON \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DPython3_INCLUDE_DIR=/usr/include/python3.10 \
  -DPython3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so \
  -DPYTHON_EXECUTABLE=/usr/bin/python3

# Build cuPHY + Python bindings (185 targets, ~5-10분)
cmake --build build -t _pycuphy pycuphycpp
'
```

> **중요**: `-DPython3_EXECUTABLE=/usr/bin/python3` 등 Python 3.10 경로를 반드시 지정.
> Shifter는 호스트 PATH를 상속하므로, Perlmutter의 conda Python 3.11 헤더를 잡으면
> `PyFrameObject incomplete type` 에러로 pybind11 빌드가 실패함.

빌드 결과물:
- `build/pyaerial/_pycuphy.cpython-310-x86_64-linux-gnu.so` (Python C extension)
- `build/pyaerial/libpycuphycpp.so` (C++ shared library)
- `build/cuPHY/src/cuphy/libcuphy.so` (cuPHY core)

### Step 4: pyAerial 패키지 설치

```bash
shifter --image=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb bash -c '
export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export BUILD_ID=25.3.0
cd $cuBB_SDK
pip install -e pyaerial/ --prefix /pscratch/sd/s/sgkim/kcj/AI-RAN/pyaerial_install
'
```

> `BUILD_ID` 필수 (없으면 RuntimeError).
> Shifter는 read-only 파일시스템이므로 시스템 site-packages에 쓸 수 없음 → `--prefix` 사용.

### Step 5: ML 의존성 설치 (Sionna, PyTorch, TensorFlow)

pyAerial 노트북 실행 및 AI 간섭 실험을 위해 ML 패키지를 scratch에 설치:

```bash
shifter --image=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb bash -c '
export PIP_TARGET=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
mkdir -p $PIP_TARGET
pip install torch==2.7.1 --target $PIP_TARGET
pip install "tensorflow[and-cuda]==2.19.0" sionna==1.0.2 --target $PIP_TARGET
'
```

### 실행 방법

빌드 완료 후, 아래 명령어로 전체 환경 진입:

```bash
shifter --image=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb bash -c '
export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH
cd $cuBB_SDK

# 여기서 python3 실행
python3 your_script.py
'
```

환경 검증 스크립트:
```bash
shifter --image=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb bash -c '
export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export SITE=/pscratch/sd/s/sgkim/kcj/AI-RAN/site-packages
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$SITE:$PYTHONPATH
python3 /pscratch/sd/s/sgkim/kcj/AI-RAN/test_env.py
'
```

### 테스트 결과 (2026-04-07 확인)

### 테스트 명령어

```bash
shifter --image=docker:nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb bash -c '
export cuBB_SDK=/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran
export PYTHONPATH=$cuBB_SDK/pyaerial/src:$PYTHONPATH
python3 << "PYEOF"
import aerial
print("[PASS] 1. aerial import OK")

import aerial.pycuphy
attrs = [x for x in dir(aerial.pycuphy) if not x.startswith("_")]
print(f"[PASS] 2. pycuphy loaded — {len(attrs)} symbols")

from aerial.pycuphy import _pycuphy
all_attrs = [x for x in dir(_pycuphy) if not x.startswith("_")]
print(f"[PASS] 3. _pycuphy C extension — {len(all_attrs)} symbols")

key = ["LdpcEncoder","LdpcDecoder","PuschRxConfig","PdschTxConfig","SrsRxConfig"]
found = [k for k in key if k in all_attrs]
missing = [k for k in key if k not in all_attrs]
print(f"[INFO] 4. Key C classes found: {found}")
if missing: print(f"[WARN]    Not in C bindings: {missing}")

import importlib, pkgutil
subs = [name for _, name, _ in pkgutil.iter_modules(aerial.pycuphy.__path__)]
print(f"[INFO] 5. pycuphy submodules: {subs}")
for s in subs:
    try:
        importlib.import_module(f"aerial.pycuphy.{s}")
        print(f"[PASS]    aerial.pycuphy.{s}")
    except Exception as e:
        print(f"[FAIL]    aerial.pycuphy.{s}: {type(e).__name__}: {e}")

try:
    import cupy as cp
    print("[PASS] 6. CuPy GPU access OK")
except Exception as e:
    print(f"[SKIP] 6. CuPy: {e}")
PYEOF
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
'
```

### 실제 출력

```
[PASS] 1. aerial import OK
[PASS] 2. _pycuphy C extension — 131 symbols
[PASS] 3. PyTorch 2.7.1+cu126, CUDA=True
       GPU: NVIDIA A100-PCIE-40GB, 42GB
[PASS] 4. TensorFlow 2.19.0, GPUs=1
[PASS] 5. Sionna 1.0.2
[PASS] 6. CuPy 13.4.1

=== ALL DEPENDENCIES READY ===
```

### 결과 요약

| 항목 | 결과 |
|------|------|
| CMake configure | PASS |
| cuPHY + Python bindings 빌드 (185/185) | PASS |
| pyAerial install | PASS |
| `import aerial` + `_pycuphy` C extension | PASS (131 symbols, LdpcEncoder/Decoder) |
| PyTorch 2.7.1 + CUDA | PASS |
| TensorFlow 2.19.0 + GPU | PASS |
| Sionna 1.0.2 | PASS |
| CuPy 13.4.1 + GPU | PASS |

### 알려진 이슈

### 1. Python 버전 충돌 (Shifter)
- Shifter는 호스트 PATH를 상속하므로 Perlmutter의 conda Python 3.11이 cmake에 잡힘
- **해결**: cmake에 `-DPython3_EXECUTABLE=/usr/bin/python3` 등 컨테이너 Python 경로 명시

### 2. Test Vector 부재
- `cubb_gpu_test_bench`에 필요한 `TV_cuphy_*.h5` 파일은 레포에 미포함
- MATLAB `5GModel`로 생성해야 하지만 Perlmutter에 MATLAB 없음
- **해결**: pyAerial 노트북 사용 (Sionna 기반 자체 신호 생성, MATLAB 불필요)

### 3. Shifter read-only 파일시스템
- 컨테이너 내부에 패키지 설치 불가
- **해결**: `pip install --target` 으로 scratch에 설치 + `PYTHONPATH`로 참조

### 4. TensorFlow 초기화 warning
- cuFFT/cuDNN/cuBLAS factory 중복 등록 warning 출력 (무시 가능, 동작에 영향 없음)

## 4. Phase 0 실험 결과

### 실험 환경

| 항목 | 값 |
|------|-----|
| Node | 1 Node (4x A100-SXM4-40GB/80GB) |
| GPU Sharing | NVIDIA MPS (별도 OS 프로세스, 셸 레벨 제어) |
| L1 Pipeline | cuPHY Fused PUSCH Rx (pyAerial) |
| Channel Model | Sionna Rayleigh Block Fading |
| NR Config | 273 PRBs, MCS 2, 1T2R MIMO, 30kHz SCS |
| AI Workloads | HBM stress, GPT-2 (LLM), ResNet-50 (CNN) |
| Measurement | CUDA Events, 100 iterations (10 warmup) |

### 실험 설계

```
┌─ Shell Process A ────────────────┐
│  shifter → cuPHY L1 → 측정       │──┐
└──────────────────────────────────┘  │  MPS로 같은 GPU 공유
┌─ Shell Process B (background &) ─┐  │
│  shifter → AI workload 무한 루프  │──┘
└──────────────────────────────────┘
```

> Python multiprocessing은 MPS Error 807, pickle error 등 불안정.
> **셸 레벨에서 `&`로 background process 관리**가 가장 안정적.
> 모드마다 MPS를 fresh restart하여 crash 전파 방지.

### Exp-1: HBM Bandwidth Stress 크기별 간섭 곡선 (2026-04-10)

같은 노드, 같은 GPU에서 stress 크기를 1~8GB로 변화시키며 간섭 정량화.

| Mode | RX mean | vs baseline | 비고 |
|------|---------|-------------|------|
| **baseline** | **2.208ms** | 1.00x | L1 단독 |
| **8GB stress same GPU** | **122.770ms** | **55.6x** | 극심한 간섭 |
| **4GB stress same GPU** | **61.535ms** | **27.9x** | |
| **2GB stress same GPU** | **31.333ms** | **14.2x** | |
| **1GB stress same GPU** | **16.326ms** | **7.4x** | |
| 8GB stress diff GPU | 2.101ms | 0.95x | 간섭 없음 |
| baseline_final | 2.238ms | 1.01x | 일관성 확인 |

**핵심**: stress 크기와 간섭은 **선형 관계** (1GB=7.4x, 2GB=14.2x, 4GB=27.9x, 8GB=55.6x).
다른 GPU로 분리하면 간섭 완전 제거.

### Exp-2: PoC 비교 — GPT-2, ResNet-50, HBM (2026-04-10)

NVIDIA AI-RAN PoC 논문과 동일한 워크로드 유형으로 비교.
같은 노드에서 same GPU vs diff GPU를 직접 비교.

| Mode | RX mean | vs baseline | 비고 |
|------|---------|-------------|------|
| **baseline** | **2.052ms** | 1.00x | L1 단독 |
| **same GPU + HBM stress** | **95.700ms** | **46.6x** | bandwidth 포화 (극단) |
| **same GPU + GPT-2** | **2.121ms** | **1.03x** | LLM inference — 간섭 미미 |
| **same GPU + ResNet-50** | **2.683ms** | **1.31x** | CNN inference — 약간의 간섭 |
| diff GPU + HBM stress | 2.216ms | 1.08x | 분리 → 간섭 없음 |
| diff GPU + GPT-2 | 2.123ms | 1.03x | 분리 → 변화 없음 |

### Exp-3: Multi-Node (2 Nodes, 2026-04-10)

| Mode | RX mean | vs baseline |
|------|---------|-------------|
| baseline (L1 solo, Node0) | 2.060ms | 1.00x |
| diff_node (L1:Node0, HBM:Node1) | 1.951ms | 0.95x (변화 없음) |

### 핵심 발견

1. **HBM bandwidth stress는 L1을 7~56배 느리게 만듦 (크기에 비례)**
   - stress와 간섭의 선형 관계 확인
   - 이는 MIG에서도 bandwidth가 공유되므로 동일하게 발생할 수 있는 문제

2. **현실적 AI workload (GPT-2, ResNet-50)는 간섭이 미미 (1.03~1.31x)**
   - NVIDIA PoC 논문의 "간섭 없다" 주장과 부합
   - **단, 이는 현재 L1 설정이 가벼운 1T2R/1cell (GPU ~6%)이기 때문**
   - 실제 기지국 (4T4R, multi-cell, GPU 30-40%) 에서는 간섭이 커질 수 있음

3. **GPU 분리 (다른 GPU / 다른 노드) 하면 모든 경우 간섭 완전 제거**
   - 같은 GPU 내부 리소스 경쟁만이 간섭 원인

4. **Baseline 일관성 확인** — 실험 시작과 끝의 baseline 차이 <2%

### 한계 및 다음 실험

> **현재 L1 (1T2R, 1cell)은 GPU를 ~6%만 사용하여 비현실적으로 가벼움.**
> NVIDIA PoC 논문에서 RAN only가 30-40% GPU를 사용하는 것과 비교하면,
> 현재 설정에서 GPT-2/ResNet-50의 간섭이 미미한 것은 당연한 결과.
>
> **4T4R multi-cell로 L1 부하를 높여야 현실적 간섭 수준을 측정할 수 있음.**

### Exp-4: MIG Emulator (MPS 기반)

**목적**: NVIDIA MIG 파티셔닝의 효과와 한계를 실측. 실제 AI-RAN의 핵심 메커니즘.

**배경**: NVIDIA AI-RAN은 MIG를 사용하여 GPU를 L1 파티션과 AI 파티션으로 하드웨어 분리:
- SM (compute) → MIG가 완전 격리
- Memory (capacity) → MIG가 완전 격리
- **HBM bandwidth → MIG에서도 공유** (이게 잠재적 한계)

**MIG 에뮬레이션 방법**: Perlmutter에서 MIG 활성화 불가 (root 권한 필요)하여 MPS로 에뮬레이션:

| MIG 기능 | 에뮬레이션 | 정확도 |
|----------|-----------|--------|
| SM 파티셔닝 | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` | 높음 |
| Memory 파티셔닝 | `torch.cuda.set_per_process_memory_fraction()` | 높음 |
| Bandwidth 공유 | 자연스럽게 발생 (MIG에서도 동일) | 동일 |

**A100 MIG 프로파일 (에뮬레이션 대상)**:

| Profile | SM | Memory | SM% |
|---------|-----|--------|-----|
| 1g.10gb | 14 SM | 10 GB | 14% |
| 2g.20gb | 28 SM | 20 GB | 29% |
| 3g.40gb | 42 SM | 40 GB | 43% |
| 4g.40gb | 56 SM | 40 GB | 57% |
| 7g.80gb | 98 SM | 80 GB | 100% |

**실험 조합**:

```
1.  L1(7g) solo                    — Full GPU baseline
2.  L1(7g) + AI(7g) ResNet         — No MIG (최악)
3.  L1(7g) + AI(7g) HBM stress     — No MIG (최악)
4.  L1(4g) + AI(3g) ResNet         — MIG 균형 분할
5.  L1(4g) + AI(3g) HBM stress
6.  L1(3g) + AI(4g) ResNet         — AI에 더 할당
7.  L1(3g) + AI(4g) HBM stress
8.  L1(2g) + AI(4g) ResNet         — L1 제한
9.  L1(2g) + AI(4g) HBM stress
10. L1(1g) + AI(4g) ResNet         — L1 극도 제한
11. L1(1g) + AI(4g) HBM stress
```

**검증 포인트**:
- MIG SM 파티셔닝으로 compute 간섭이 줄어드는가?
- HBM bandwidth는 여전히 공유 → MIG에서도 bandwidth 간섭이 남는가?
- L1 파티션 크기를 줄이면 간섭은 없지만 L1 자체가 느려지는 트레이드오프
- 최적 파티션 크기는 workload에 따라 동적으로 변해야 → 정적 MIG의 한계

**실험 코드**:

```bash
# MIG 에뮬레이터 실행
sbatch /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/run_mig_emu.sh

# 개별 모드 실행 (job간 MPS crash 격리)
sbatch /pscratch/sd/s/sgkim/kcj/AI-RAN/jobs/submit_all.sh
```

## 6. 연구 방향 정리

### Phase 0 → Phase 1 전환 근거

```
Phase 0에서 확인한 것:
  ✅ GPU 내부 간섭은 실재한다 (53~104x latency 증가)
  ✅ GPU 분리하면 간섭 제거 (하지만 자원 낭비)
  ✅ MPS 멀티테넌시에서 cuPHY+PyTorch 메모리 충돌 발생
  🔄 MIG 파티셔닝 효과 실측 중 (에뮬레이터)

Phase 1에서 할 것:
  - MIG 파티셔닝의 정량적 효과 및 한계 분석
  - 수학적 모델: MIG 파티션 조합별 L1 SLA 보장 확률
  - 동적 파티셔닝 알고리즘 제안 (정적 MIG 대비 개선)
```

### 핵심 연구 질문

> "MIG가 SM 격리는 해주지만, (1) bandwidth는 여전히 공유되고,
> (2) 파티션이 정적이라 트래픽 변화에 대응 못하고,
> (3) L1 idle 시간에 자원이 낭비된다.
> 이 세 가지 한계를 극복하는 동적 리소스 관리 시스템을 어떻게 설계할 것인가?"
