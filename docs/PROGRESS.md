# newch 작업 요약

> 기준일: 2026-11-01 · 버전 0.2.0 ([PyPI](https://pypi.org/project/newch/) 공개, `pip install newch`) · C 엔진 약 4,400줄, Python 패키지 약 2,100줄, 테스트 약 1,300줄

## 1. 목표

**저사양 기기에서도 학습까지 할 수 있는, 가벼운 범용 머신러닝 라이브러리**입니다.
싱글보드 컴퓨터(라즈베리파이, 오렌지파이 등)는 "그 위에서도 돌아가야 하는" 호환 대상이고, 이 프로젝트의 중심은 아닙니다.

여러 프레임워크의 장점을 층별로 하나씩 가져옵니다.

| 출처 | 가져온 것 | 버린 것 |
|---|---|---|
| PyTorch | 즉시 실행, 테이프 기반 autograd, `Module`, `state_dict`, `DataLoader` | 수천 개의 연산자, CUDA |
| Keras | `compile` → `fit` 고수준 API, 콜백, `save_weights` | 멀티 백엔드 |
| scikit-learn | `fit`/`predict` 인터페이스, 고전 ML *(예정)* | 수백 개의 알고리즘 전부 |
| TensorFlow | 학습한 모델을 어디서나 실행 (가중치 파일, 정적 바이너리, int8) | 분산 학습, 서빙 |
| XLA | 연산 융합, 정적 메모리 계획 *(예정, 훅만 준비)* | LLVM 코드 생성 |

## 2. 구조

```
┌──────────────────────────────────────────────────────────────┐
│ L5 고수준     Keras식 Model.compile/fit/evaluate/predict, 콜백 │  ✅ Python
│              DataLoader 스트리밍, save/load_weights, quantize_ │  ✅ Python
│              sklearn식 Estimator                              │  ⏳
├──────────────────────────────────────────────────────────────┤
│ L4 라이브러리  net(Linear, Conv2d, Pool, BatchNorm, Dropout,    │  ✅ C + Python
│              Embedding, 활성화), optim(SGD, Adam)             │
├──────────────────────────────────────────────────────────────┤
│ L3 즉시 실행   텐서, 연산 레지스트리(34개), autograd, gradcheck  │  ✅ C
├──────────────────────────────────────────────────────────────┤
│ L2 컴파일러    추적 → IR → 융합, 메모리 계획                    │  ⏳ (ctx->trace 훅)
├──────────────────────────────────────────────────────────────┤
│ L1 커널       ref + NEON + AVX2/FMA, 런타임 디스패치, 스레드 풀  │  ✅ C
└──────────────────────────────────────────────────────────────┘
```

```
newch/
├── engine/                   C11 엔진
│   ├── include/nc.h          공개 C API (NC_API 함수만 공유 라이브러리에서 노출)
│   ├── src/core/             컨텍스트, 메모리, 텐서, 연산 레지스트리, 파일 입출력(io.c)
│   ├── src/autograd/         nc__apply, 테이프, backward, gradcheck
│   ├── src/ops/              elementwise, linalg, reduce, shape, nn, conv, quant
│   ├── src/kernels/          디스패치, ref/, neon/, x86/(AVX2)
│   ├── src/runtime/          CPU 감지, 스레드 풀 + 병렬 GEMM
│   ├── src/nn/, src/optim/   C 모듈, 옵티마이저
│   └── tests/test_main.c     C 테스트
├── newch/                    Python 패키지 (ctypes, 의존성 0)
│   ├── tensor.py             Tensor, Context, 생성·함수형 연산, 텐서 파일
│   ├── net.py                Module, 레이어, Model, Sequential, 양자화
│   ├── data.py               Dataset, TensorDataset, FileArray, DataLoader
│   └── optim.py, training.py, callbacks.py
├── tests/                    Python 테스트 (test_python.py, test_features.py)
├── examples/                 train_spiral.c, mnist_cnn.py
├── docs/                     PROGRESS.md, FORMAT.md
├── tools/build.py            zig cc 기반 멀티 타깃 빌드
├── pyproject.toml, setup.py  pip 설치 (빌드 시 zig로 엔진 컴파일)
├── CMakeLists.txt            기기에서 직접 빌드할 때
└── .github/workflows/ci.yml  CI
```

## 3. 핵심 설계 결정과 이유

| 결정 | 이유 | 대가 |
|---|---|---|
| **엔진은 C11** | 어떤 CPU·OS에도 컴파일러가 있고, 런타임 의존성이 없습니다 | 메모리 관리를 직접 설계해야 합니다 |
| **메모리 = 영구 풀 + 스텝 아레나** | 활성값은 아레나에 두고 `step_end()`로 O(1) 일괄 해제합니다. 학습 중 메모리가 스텝마다 일정합니다 | 스텝 안에서는 중간값을 해제하지 않으므로, 최대 메모리가 배치 크기에 비례합니다 |
| **영구 텐서는 항상 데이터를 소유** (영구 모드에서는 뷰를 만들지 않음) | 영구 텐서 하나를 해제해도 다른 텐서가 망가지지 않으므로, Python이 GC 시점에 안전하게 해제할 수 있습니다 | 영구 모드의 reshape 등은 복사합니다 |
| **에러 = 첫 에러 유지 + NULL 전파** | C에서도 순전파 전체를 마지막에 한 번만 확인하면 됩니다. 연산은 절대 abort하지 않습니다 | 첫 에러 이후의 연산은 모두 NULL이 됩니다 (`nc_clear_error`로 복구) |
| **연산을 데이터로 기술** (`nc_op_def`) | 모든 연산이 `nc__apply()` 하나를 거치므로, 나중에 컴파일러가 같은 정보로 그래프를 분석할 수 있습니다 | 연산마다 정형화된 코드를 써야 합니다 |
| **계층형 커널 디스패치** | 레퍼런스에서 시작해 CPU가 지원하는 계층(NEON, AVX2)이 항목별로 덮어씁니다. AVX2는 함수 단위 target 속성이라 기본 빌드는 모든 x86-64에서 돕니다 | 커널 테이블을 관리해야 합니다 |
| **병렬화는 GEMM 단위** | 출력 원소 하나를 스레드 하나가 계산하므로 결과가 스레드 수와 무관하게 비트 단위로 같습니다. 작은 GEMM(약 200만 곱셈 미만)은 직렬로 실행합니다 | 원소별 연산은 아직 단일 스레드입니다 |
| **합성곱 = im2col + GEMM, 역전파 때 im2col 재계산** | GEMM 최적화를 그대로 재사용합니다. 샘플별 열 버퍼를 저장하지 않아 메모리가 작습니다 | 역전파에서 im2col을 한 번 더 계산합니다 |
| **Python은 ctypes만** | 저사양 기기에 빌드 도구나 무거운 의존성을 요구하지 않습니다 | 호출당 약 1µs 오버헤드 |
| **스트리밍 데이터 = mmap 파일** (`FileArray`) | RAM보다 큰 데이터셋도 배치 하나만큼의 메모리로 학습합니다 | 데이터를 원시 이진 파일로 한 번 변환해야 합니다 |
| **zig cc로 빌드** | pip 패키지 하나로 Windows에서 ARM·RISC-V 리눅스까지 크로스 컴파일되고, `pip install`에도 시스템 컴파일러가 필요 없습니다 | 기기에서는 CMake + gcc 경로도 함께 유지합니다 |

## 4. 구현된 기능

### C 엔진
- **텐서:** f32 / i32 / i8, 최대 6차원, stride 뷰, numpy식 브로드캐스팅
- **연산 34개** (미분 가능한 연산은 전부 gradcheck 통과)
  - 원소별: `add sub mul div neg exp log sqrt relu sigmoid tanh scale add_scalar pow_scalar dropout`
  - 선형대수: `matmul`(전치 뷰는 복사 없이 처리), `linear`(fused `x @ Wᵀ + b`)
  - 합성곱·풀링: `conv2d`(stride, padding), `max_pool2d`, `avg_pool2d`
  - 정규화: `batch_norm` (학습: 배치 통계 + 이동 평균 갱신, 평가: 이동 평균 사용)
  - 축소: `sum mean max argmax`
  - shape: `reshape transpose contiguous detach index_select`
  - NN: `log_softmax softmax cross_entropy`, 합성 `mse_loss`
  - 양자화(추론): `nc_quantize_rows`(행별 대칭 int8), `linear_int8`(동적 양자화, int32 누적)
- **autograd:** 테이프 기반 역전파, 손실 여러 개, 그래디언트 누적, no-grad, detach
- **커널:** 레퍼런스(K/N 캐시 블로킹), ARM NEON(4×8 GEMM, 전치 GEMM), **x86 AVX2+FMA**(4×16 GEMM, 전치 GEMM, int8 GEMM, 원소별)
- **스레드 풀:** Windows/POSIX, 원자적 청크 분배(코어 속도가 달라도 균형), 중첩 호출은 자동 직렬화
- **메모리:** `nc_release`(개별 해제), `nc_arena_trim`(남는 아레나를 OS에 반환), 기본 청크 1MB, 통계
- **저장/불러오기:** `nc_save` / `nc_load` (NCW1 포맷, UTF-8 경로, 이름 순서 무관, shape/dtype 검사)
- **공개 심볼만 노출:** `-fvisibility=hidden` + `NC_API`, 릴리스 공유 라이브러리 strip

### Python 패키지
- **PyTorch식:** `Tensor`(연산자, `@`, 인덱싱, `.backward()`, `.copy_()`), `net.Module`(파라미터·버퍼 자동 탐색, `state_dict`, `load_state_dict`), `Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `BatchNorm1d/2d`, `Dropout`, `Embedding`, 활성화, `Flatten`, 함수형 API(`conv2d`, `batch_norm`, `dropout`, …), `optim.SGD/Adam`, `no_grad()`
- **Keras식:** `Sequential`, `compile`, `fit`(배열 / DataLoader 스트리밍 / FileArray 자동 스트리밍), `evaluate`, `predict`, `summary()`, `save_weights`/`load_weights`, `History`, `EarlyStopping`, `LambdaCallback`
- **데이터:** `Dataset`, `TensorDataset`, `DataLoader`(shuffle, drop_last, generator), `FileArray`(mmap, `create()`로 한 행씩 기록)
- **양자화:** `nc.quantize_(model)` → 모든 `Linear`를 `QuantizedLinear`로 교체
- **메모리 안전:** 해제된 임시 텐서 접근 시 `NcError`. 영구 텐서는 마지막 참조가 사라지면 자동 해제되고, 뷰와 grad는 원본을 살려 둡니다
- **설치:** `pip install .` → `py3-none-<플랫폼>` 휠 (Python 버전 무관)

## 5. 이름 규칙

**원칙: 이름이 같으면 동작도 같아야 합니다.** 고치는 비용이 작으면 동작을 원본과 똑같이 맞추고, 의미가 원래 다르면 비슷하지만 다른 이름을 씁니다.
지원하지 않는 경우(`padding="same"`, `dim≠0`인 `index_select` 등)는 **조용히 다르게 동작하지 않고 에러를 냅니다.**

### 동작을 원본과 맞춘 것 (이름 유지)

| API | 원본 | 비고 |
|---|---|---|
| `net.Linear` | torch | weight `[out, in]`, `y = x @ Wᵀ + b`, 같은 초기화 |
| `net.Conv2d`, `MaxPool2d`, `AvgPool2d` | torch | NCHW, 같은 초기화. dilation/groups 없음 (인자로 받지 않음) |
| `net.BatchNorm1d/2d` | torch | 버퍼 이름 `running_mean`, `running_var`, `num_batches_tracked`, 이동 분산은 불편 추정 |
| `net.Dropout`, `nc.dropout` | torch | 역드롭아웃, eval 모드에서는 항등 |
| `state_dict`, `load_state_dict(strict)` | torch | 같은 키 규칙. torch `state_dict`를 numpy로 바꾸면 그대로 불러옵니다 |
| `save_weights`, `load_weights` | Keras | 가중치만 저장 |
| `Dataset`, `TensorDataset`, `DataLoader` | torch | 같은 기본값(`batch_size=1`, `shuffle=False`), 튜플 샘플은 리스트로 묶음 |
| `Tensor.argmax()`, `size()`, `flatten()`, `.T`, `index_select(dim, index)` | torch | |
| `softmax(dim)`, `sum/mean/amax(dim)` | torch | numpy식 `axis=`도 허용 |
| `manual_seed` | torch | RNG만 재시드 |
| `optim.SGD`, `optim.Adam` | torch | 같은 기본값과 갱신식 |
| `Model.evaluate(return_dict)`, `EarlyStopping`, `LambdaCallback` | Keras | |

**라이브러리 전체 규칙:** 정수 데이터는 int32가 됩니다 (torch는 int64). 엔진에 int64가 없기 때문입니다.

### 의미가 달라서 다른 이름을 쓴 것

| 원본 이름 | newch 이름 | 이유 |
|---|---|---|
| `torch.nn` | **`nc.net`** | PyTorch에 없는 Keras식 `Model`까지 들어 있습니다 |
| `torch.nn.Parameter(tensor)` | **`nc.parameter(init)`** | 클래스가 아니라 함수이고, shape 튜플도 받습니다 |
| `Tensor.max(dim)` (값+인덱스) | **`Tensor.amax(dim)`** | 값만 반환 = torch `amax` |
| `Tensor.view()` (절대 복사 안 함) | **`reshape()`만 제공** | 필요하면 복사합니다 |
| `Tensor.T` (3차원 이상) | **`Tensor.mT`** | 마지막 두 축 교환 = torch `mT` |
| Keras `"sparse_categorical_crossentropy"` | **`"cross_entropy"`만** | Keras 기본은 확률 입력, 우리는 로짓 입력 |
| `torch.save` / `torch.load` (pickle, 임의 객체) | **`nc.save_tensors` / `nc.load_tensors`** | 텐서 딕셔너리만 다룹니다 |
| `torch.ao.quantization.quantize_dynamic` (사본 반환) | **`nc.quantize_(model)`** | 제자리 변경입니다. 끝의 `_`는 torch의 in-place 표기 관례를 따랐습니다 |
| `numpy.memmap` | **`newch.data.FileArray`** | 읽기 전용, 행 단위 배치 읽기에 맞춘 다른 API입니다 |

### newch에만 있는 개념

`nc.step_end()`, `nc.persistent()`, `nc.release()`, `nc.Context(threads=…)`, `Context.trim()`, `nc.mem_stats()`

## 6. 평가 후 개선 (v0.1 → v0.2)

| 지적된 약점 | 조치 | 확인 |
|---|---|---|
| Conv/Pool 없음 | `conv2d`, `max_pool2d`, `avg_pool2d` + 레이어 | gradcheck, MNIST 98.1% |
| 저장/불러오기 없음 | NCW1 포맷, `state_dict`, `save/load_weights`, torch 가중치 가져오기 | 저장 후 예측 완전 일치 |
| Dropout, BatchNorm 없음 | 연산 + 레이어 + 버퍼 | gradcheck(학습/평가), 이동 통계 값 검사 |
| 단일 스레드, 캐시 블로킹 없음, x86 SIMD 없음 | 스레드 풀 + 병렬 GEMM, K/N 블로킹, AVX2/FMA 커널 | 아래 성능 표, 스레드 1개/4개 결과 비트 일치 |
| 데이터셋 전체를 메모리에 올림 | `FileArray` + `DataLoader` 스트리밍, `fit(loader)` | MNIST 60,000장 스트리밍 학습 |
| int8 양자화 없음 | `quantize_rows`, `linear_int8`, `quantize_()`, AVX2 int8 GEMM | 상대 오차 0.6%, MNIST 정확도 유지, 가중치 81→36KB |
| 영구 텐서가 계속 쌓임 | Python GC 시 자동 해제, 영구 뷰 제거 | 반복 생성 후 메모리가 원래 값으로 복귀 |
| 아레나 최소 4MB | 기본 1MB + `trim()` | trim 후 예약 0 |
| `pip install` 불가 | pyproject + setup.py (zig로 빌드) | 깨끗한 venv(numpy 없음)에 휠 설치 후 학습 |
| 내부 심볼 노출 | `NC_API` + `-fvisibility=hidden` + strip | DLL에서 `nc__*` 접근 불가, `.so`에 `nc__` 0개 |
| CI 없음 | GitHub Actions (Windows/Linux/macOS, QEMU ARM·RISC-V, CMake, 휠) | 작업 12개 모두 통과 |
| ARM 실기기 검증 없음 | Apple Silicon(macOS CI)과 QEMU(aarch64, armv7)에서 NEON 커널 실행 | 레퍼런스와 결과 일치. 오렌지파이 실측은 보류 |

### 성능 (x86 PC, 12스레드, 정방 행렬곱)

| 크기 | v0.1 | v0.2 | numpy(BLAS) | v0.2 / numpy 시간 |
|---|---|---|---|---|
| 64 | 32 GFLOPS | 72 GFLOPS | 99 GFLOPS | 1.4배 |
| 128 | — | 79 GFLOPS | 67 GFLOPS | 0.9배 (더 빠름) |
| 256 | 53 GFLOPS | 202 GFLOPS | 279 GFLOPS | 1.4배 |
| 512 | 52 GFLOPS | 325 GFLOPS | 360 GFLOPS | 1.1배 |
| 1024 | — | 307 GFLOPS | 409 GFLOPS | 1.3배 |

## 7. 검증 상태

| 항목 | 결과 |
|---|---|
| C 테스트 | **20개 테스트 통과** (연산별 gradcheck 40여 건, 합성곱·풀링·배치 정규화, 드롭아웃 통계, int8 정확도와 커널 일치, 저장/불러오기와 에러, 스레드 결정성, 아레나 축소, 학습 2종 등) |
| Python 테스트 | **28개 통과** (기존 17개 + CNN 학습, 레이어 shape와 모드, BN 버퍼, 저장 후 예측 일치, strict/불일치 에러, torch 형식 state_dict, 양자화, FileArray 스트리밍 fit, 일반 Dataset 묶음, GC 자동 해제, 스레드/trim) |
| 디버그 빌드 | `-O0` + UBSan 통과 |
| 실제 데이터 | MNIST CNN 1 에포크 18초, 테스트 98.13%, int8 후 98.13% |
| 패키징 | 휠 빌드 → 별도 venv 설치(numpy 없음) → 학습 성공 |
| 크로스 컴파일 | x86_64 Windows/Linux, aarch64(gnu/musl/v8.2), armv7(gnu/musl), riscv64(gnu/musl). 경고 0 |
| CI (GitHub Actions) | Windows, Ubuntu, macOS(Apple Silicon)에서 C·Python 테스트, QEMU aarch64·armv7·riscv64, CMake+gcc, 휠 빌드. 모든 테스트는 해제 메모리 덮어쓰기 모드(`NC_DEBUG_POISON=1`)로 실행 |

**CI에서 찾아 고친 버그:** Python에서 `lib.f(임시텐서().ptr)` 형태로 쓰면 C 호출 전에 영구 임시 텐서가 해제되는
use-after-free가 있었습니다 (7곳). Windows 할당기에서는 우연히 동작했고 Linux/macOS에서 드러났습니다.
수정 후 `NC_DEBUG_POISON`으로 모든 플랫폼에서 재현되게 만들고 회귀 테스트를 추가했습니다.

**아직 확인하지 못한 것**
- 저사양 ARM 보드(오렌지파이 등) 실측: 보류. NEON 커널의 정확성은 Apple Silicon과 QEMU에서 확인했지만, Cortex-A 코어에서의 **속도**는 측정하지 않았습니다
- 리눅스·macOS에서의 `pip install`(sdist) 설치 경로

## 8. 빌드와 사용

```bash
pip install .                           # 설치 (시스템 C 컴파일러 불필요)
pip install ziglang                     # 개발 시
python tools/build.py                   # 빌드 + C/Python 테스트
python tools/build.py --target aarch64-static   # 정적 ARM 바이너리
python tools/build.py --target all      # 모든 타깃
python tools/build.py --debug           # -O0 + UBSan
python examples/mnist_cnn.py            # MNIST 다운로드 → 스트리밍 학습 → 저장 → 양자화
```

## 9. 알려진 제약

- **스텝 안에서는 메모리를 해제하지 않습니다.** 최대 메모리가 배치 크기에 비례합니다 (MNIST CNN, 평가 배치 256에서 아레나 최대 65MB). 저메모리 기기에서는 배치를 줄이세요. 근본 해결은 그래프 컴파일러의 메모리 계획입니다
- dtype: f32(연산), i32(라벨·인덱스), i8(양자화 가중치). f16, int64 없음
- `matmul`의 두 번째 피연산자는 2차원만, `softmax` 계열은 마지막 축만, `index_select`는 축 0만
- Conv2d: dilation, groups, `padding="same"` 없음. 양자화는 `Linear`만
- 원소별 연산과 im2col은 단일 스레드입니다
- `predict()`는 numpy 입력이면 ndarray, 아니면 리스트를 반환합니다
- 옵티마이저 상태는 저장하지 않습니다 (Keras `save_weights`와 같음)

## 10. 다음 단계 후보

1. **scikit-learn식 고전 ML:** 선형/로지스틱 회귀, 결정 트리, k-means, 전처리, `Pipeline`
2. **그래프 컴파일러:** 추적 → IR → 원소별 연산 융합, 정적 메모리 계획 (스텝 내 메모리 재사용)
3. **ARM 실측:** 오렌지파이에서 `aarch64-static/nc_tests` 실행, NEON int8(dotprod) 커널
4. **원소별 연산·im2col 병렬화, RISC-V RVV 커널**
5. **옵티마이저 상태 저장**(학습 재개), 학습률 스케줄러
