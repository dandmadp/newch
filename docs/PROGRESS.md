# newch 작업 요약

> 기준일: 2026-11-02 · 버전 0.3.1 (PyPI에는 0.3.0까지 공개) · C 엔진 약 4,900줄, Python 패키지 약 3,200줄, 테스트 약 1,800줄

## 0.3.1 정리: 한 가지 일에 한 가지 방법

0.3.0까지 torch·Keras·scikit-learn·numpy의 API를 함께 따라 하면서 같은 일을 하는 길이 여러 개 생겼습니다.
0.3.1은 **torch식을 기본**으로 두고, Keras는 `fit()` 학습 루프 하나로 줄였습니다. 공개 API를 지우는 변경이라
0.3.0 코드는 아래처럼 바꿔야 합니다.

| 0.3.0 | 0.3.1 |
|---|---|
| `nc.relu`, `nc.conv2d`, `nc.cross_entropy` 등 신경망 함수 18개 (`F`와 중복) | `import newch.net.functional as F` 후 `F.relu` 등. `nc.*`에는 텐서·수학(`exp`, `sigmoid`, `tanh` 등)·형태 함수만 |
| `compile("adam")` = Keras 기본값(eps 1e-7), `optim.Adam()` = torch 기본값 | 이름은 `nc.optim`의 같은 클래스를 그 기본값으로 만듦. compile 기본 optimizer는 `"adam"` |
| `nc.optim._KerasRMSprop`, `Optimizer.learning_rate` | `nc.optim.RMSprop`, `Optimizer.lr` |
| 콜백 `LearningRateScheduler`, `ReduceLROnPlateau` | `nc.optim.lr_scheduler.*` (새로 `ReduceLROnPlateau` 추가), fit에서는 `LambdaCallback`으로 step |
| `model.save_weights(p)` / `model.load_weights(p)` | `nc.save_tensors(model.state_dict(), p)` / `model.load_state_dict(nc.load_tensors(p))` |
| `compile(loss="mean_squared_error")` | `loss="mse"` |
| `compile(loss_weights=...)` 인수 | 삭제 (원래 지원하지 않아 에러만 냈음) |
| `newch.ml` (scikit-learn식 7종 + 지표) | 삭제. scikit-learn을 쓰고 `np.asarray(tensor)`로 주고받기 |

결과: Python 패키지 약 3,740줄 → 약 3,000줄, 최상위 공개 이름 60개 → 41개, 콜백 7개 → 5개.

## 1. 목표

**저사양 기기에서도 학습까지 할 수 있는, 가벼운 범용 머신러닝 라이브러리**입니다.
싱글보드 컴퓨터(라즈베리파이, 오렌지파이 등)는 "그 위에서도 돌아가야 하는" 호환 대상이고, 이 프로젝트의 중심은 아닙니다.

여러 프레임워크의 장점을 층별로 하나씩 가져옵니다.

| 출처 | 가져온 것 | 버린 것 |
|---|---|---|
| PyTorch | 즉시 실행, 테이프 기반 autograd, `Module`, `state_dict`, `DataLoader` | 수천 개의 연산자, CUDA |
| Keras | `fit` 한 가지 학습 루프와 콜백 몇 개 (0.3.1부터 얇게 축소) | 멀티 백엔드, 콜백 체계 전체, 문자열 기본값 표 |
| scikit-learn | (0.3.0의 `newch.ml`은 0.3.1에서 제거: 핵심 밖이고 순수 Python 구현이라 느림) | 전부 |
| TensorFlow | 학습한 모델을 어디서나 실행 (가중치 파일, 정적 바이너리, int8) | 분산 학습, 서빙 |
| XLA | 연산 융합, 정적 메모리 계획 *(예정, 훅만 준비)* | LLVM 코드 생성 |
| numpy | `axis=` 별칭, `np.asarray(tensor)` 연동 | 전체 ndarray API |

**가져오지 않은 것과 이유** (저사양 목표 기준)
- MLX의 지연 실행·통합 메모리: 단일 CPU 엔진에서는 이득이 없고, 즉시 실행 디버깅이 더 중요합니다
- PyTorch 활성값 체크포인팅: 스텝 아레나는 스텝 안에서 메모리를 해제하지 않으므로 최대 메모리가 줄지 않습니다. 그래프 컴파일러의 메모리 계획 이후에 의미가 생깁니다
- TensorFlow 분산 학습·서빙, Keras 멀티 백엔드, 전체 모델 직렬화: 크기 대비 이득이 작습니다

## 2. 구조

```
┌──────────────────────────────────────────────────────────────┐
│ L5 고수준     Model.compile/fit/evaluate/predict, 콜백 4개      │  ✅ Python
│              DataLoader 스트리밍, state_dict 저장, quantize_   │  ✅ Python
├──────────────────────────────────────────────────────────────┤
│ L4 라이브러리  net(Linear, Conv2d, Pool, BatchNorm, Dropout,    │  ✅ C + Python
│              Embedding, 활성화 10종), optim(SGD, Adam, AdamW, │
│              그래디언트 누적, 클리핑), 학습률 스케줄러          │
├──────────────────────────────────────────────────────────────┤
│ L3 즉시 실행   텐서, 연산 레지스트리(45개), autograd, gradcheck  │  ✅ C
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
├── examples/                 train_spiral.c, mnist_cnn.py (Keras식), mnist_torch_style.py (PyTorch식)
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
| **메모리 = 영구 풀 + 스텝 아레나** | 활성값은 아레나에 두고 `step_end()`로 O(1) 일괄 해제합니다. 학습 중 메모리가 스텝마다 일정합니다 | 최대 메모리가 배치 크기에 비례합니다 |
| **역전파 중 해제** | 노드의 역전파가 끝나면 그 노드의 임시값, 출력 그래디언트, (아무도 쥐고 있지 않으면) 활성값을 아레나에 돌려주고 같은 스텝 안에서 재사용합니다. Python은 지금 참조 중인 텐서 목록을 넘겨서 그 텐서는 지키고, 해제된 그래프로 두 번째 역전파를 하면 에러가 납니다 (PyTorch의 `retain_graph`와 같음) | 중간 텐서의 `.grad`는 역전파 뒤 None (PyTorch와 같음). 블록 반환·재할당 때문에 스텝이 2~7% 느려집니다 |
| **큰 블록은 따로, 캐시에 예산** | 256KB 이상 할당은 청크 밖의 개별 블록으로 두고, 해제·스텝 종료 시 크기별 캐시에 넣어 다음 스텝이 재사용합니다. 캐시는 "직전 스텝의 최대 사용량"을 넘지 않게 오래된 것부터 OS에 돌려줍니다 | 캐시가 없으면 매 스텝 OS 할당과 페이지 폴트가 생기고, 예산이 없으면 한 스텝에 쓰인 모든 크기의 합이 남습니다 |
| **추론 중 해제** | 기록되지 않는 연산(no_grad 등)은 끝나자마자 자기 임시 버퍼와 저장값을 돌려주고, Python은 no_grad에서 만든 임시 텐서를 아무도 참조하지 않게 되면 `nc_release_temp`로 돌려줍니다. 테이프가 쓰는 데이터(뷰면 데이터 주인)는 `ON_TAPE` 표시로 지킵니다 | 학습 모드(그래디언트 켜짐)의 순전파는 그대로 역전파 끝까지 남습니다 |
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
- **연산 45개** (미분 가능한 연산은 전부 gradcheck 통과)
  - 원소별: `add sub mul div neg exp log sqrt relu sigmoid tanh scale add_scalar pow_scalar dropout`
  - 활성화: `leaky_relu relu6 hardsigmoid hardswish silu gelu gelu_tanh` (PyTorch 정의, 꺾이는 점의 그래디언트도 PyTorch 규칙)
  - 선형대수: `matmul`(전치 뷰는 복사 없이 처리), `linear`(fused `x @ Wᵀ + b`)
  - 비교: `compare`(eq/ne/lt/le/gt/ge, 브로드캐스팅, f32·i32 혼합, 결과 i32 0/1), i32 `sum`(개수 세기)
  - 결합·변환: `cat`(입력 6개까지, Python `cat`은 더 많으면 나눠 호출), `to_dtype`(f32↔i32↔i8, 정수로 바꾸면 그래디언트 끊김)
  - 합성곱·풀링: `conv2d`(stride, padding), `max_pool2d`, `avg_pool2d`
  - 정규화: `batch_norm` (학습: 배치 통계 + 이동 평균 갱신, 평가: 이동 평균 사용)
  - 축소: `sum mean max argmax`
  - shape: `reshape transpose contiguous detach index_select`
  - NN: `log_softmax softmax cross_entropy nll_loss`, 합성 `mse_loss`
  - 양자화(추론): `nc_quantize_rows`(행별 대칭 int8), `linear_int8`(동적 양자화, int32 누적)
- **autograd:** 테이프 기반 역전파, 손실 여러 개, 그래디언트 누적, no-grad, detach
- **커널:** 레퍼런스(K/N 캐시 블로킹), ARM NEON(4×8 GEMM, 전치 GEMM), **x86 AVX2+FMA**(4×16 GEMM, 전치 GEMM, int8 GEMM, 원소별). AVX2·NEON 모두 **벡터 exp/sigmoid/tanh/erf**(다항식 근사, libm 대비 상대 오차 1.4e-7 이하, erf는 절대 오차 5e-7 이하)를 가지며 sigmoid, tanh, exp, SiLU, GELU, softmax, cross_entropy가 이를 씁니다
- **스레드 풀:** Windows/POSIX, 원자적 청크 분배(코어 속도가 달라도 균형), 중첩 호출은 자동 직렬화
- **메모리:** `nc_release`(개별 해제), `nc_arena_trim`(남는 아레나를 OS에 반환), 기본 청크 1MB, 통계. `nc_backward_ex`(역전파 중 해제, `NC_BACKWARD_FREE_GRAPH` / `NC_BACKWARD_RETAIN`). MNIST CNN 배치 64 한 스텝: 최대 사용 22.7 → 14.5MiB, **실제 예약 29.4 → 15.2MiB (−48%)**. `nc_release_temp`와 기록 안 되는 연산의 임시 버퍼 즉시 해제로 **평가(no_grad, 배치 256)가 스텝 안에서 쌓이지 않음**: 5번 순전파 최대 323 → 25MiB, `examples/mnist_cnn.py` 전체 최대 65 → 25MiB, 속도 차이 없음
- **저장/불러오기:** `nc_save` / `nc_load` (NCW1 포맷, UTF-8 경로, 이름 순서 무관, shape/dtype 검사)
- **공개 심볼만 노출:** `-fvisibility=hidden` + `NC_API`, 릴리스 공유 라이브러리 strip

### Python 패키지
- **PyTorch식:** `Tensor`(연산자, `@`, 인덱싱, `.backward()`, `.copy_()`), `net.Module`(파라미터·버퍼 자동 탐색, `state_dict`, `load_state_dict`), `Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `BatchNorm1d/2d`, `Dropout`, `Embedding`, 활성화, `Flatten`, 함수형 API(`conv2d`, `batch_norm`, `dropout`, …), `optim.SGD/Adam`, `no_grad()`
- **PyTorch 모델 코드 그대로:** `import newch as torch`, `import newch.net as nn`, `import newch.net.functional as F`로 바꾸면 `class Net(nn.Module)` + `super().__init__()`, `F.relu`/`F.log_softmax`/`F.nll_loss`, `torch.flatten`/`cat`/`stack`, `view`/`view_as`/`unsqueeze`/`squeeze`/`permute`, `eq`와 비교 연산자, `.float()`/`.to(device)`, `nn.ModuleList`/`ModuleDict`/`Identity`, `nn.CrossEntropyLoss`/`NLLLoss`/`MSELoss`, `model.apply()`, `register_buffer`, `nn.init.*`를 쓰는 일반적인 학습 스크립트가 돕니다 (`examples/mnist_torch_style.py`: 1 에포크 14초, 테스트 97.8%)
- **`fit` 학습 루프:** `Sequential`, `compile`, `fit`(배열 / DataLoader 스트리밍 / FileArray 자동 스트리밍), `evaluate`, `predict`, `summary()`, `History`, `EarlyStopping`, `LambdaCallback`
- **데이터:** `Dataset`, `TensorDataset`, `DataLoader`(shuffle, drop_last, generator), `FileArray`(mmap, `create()`로 한 행씩 기록)
- **양자화:** `nc.quantize_(model)` → 모든 `Linear`를 `QuantizedLinear`로 교체
- **저메모리 학습:** `gradient_accumulation_steps=n`(모든 옵티마이저, Keras 3과 같은 의미)으로 배치 B의 활성값 메모리로 배치 n·B처럼 학습합니다. `AdamW`, `net.utils.clip_grad_norm_`
- **학습률·체크포인트:** `optim.lr_scheduler`(StepLR, ExponentialLR, CosineAnnealingLR, LambdaLR), `ReduceLROnPlateau`(스케줄러), 콜백 `ModelCheckpoint`(가중치만). fit에서는 `LambdaCallback`으로 스케줄러를 step
- **numpy 연동:** `np.asarray(tensor)` (`__array__`, 항상 복사)
- **메모리 안전:** 해제된 임시 텐서 접근 시 `NcError`. 영구 텐서는 마지막 참조가 사라지면 자동 해제되고, 뷰와 grad는 원본을 살려 둡니다
- **설치:** `pip install .` → `py3-none-<플랫폼>` 휠 (Python 버전 무관)

## 5. 이름 규칙

**원칙: 이름이 같으면 동작도 같아야 합니다.** 고치는 비용이 작으면 동작을 원본과 똑같이 맞추고, 의미가 원래 다르면 비슷하지만 다른 이름을 씁니다.
지원하지 않는 경우(`padding="same"`, `dim≠0`인 `index_select` 등)는 **조용히 다르게 동작하지 않고 에러를 냅니다.**

**인수 규칙:**
- 위치 인수의 순서와 기본값은 원본과 같습니다. 그래서 `fit(x, y, 32, 10)`이나 `SGD(p, 0.1, 0.9, 0, 1e-4)`처럼 위치로 넘겨도 원본과 같은 뜻이 됩니다.
- 원본에는 있지만 구현하지 않은 인수(`dilation`, `groups`, `dampening`, `amsgrad`, `sampler`, `sample_weight` 등)도 같은 자리에 두고, 기본값이 아니면 `NotImplementedError`를 냅니다.
- newch에만 있는 인수(`gradient_accumulation_steps`, `ctx`, `seed`, `generator`, `compile(lr=)`)는 키워드로만 받습니다.

### 동작을 원본과 맞춘 것 (이름 유지)

| API | 원본 | 비고 |
|---|---|---|
| `net.Linear` | torch | weight `[out, in]`, `y = x @ Wᵀ + b`, 같은 초기화. `ctx`는 키워드 전용 |
| `net.Conv2d`, `MaxPool2d`, `AvgPool2d` (+ 함수형) | torch | NCHW, 같은 초기화, torch와 같은 인수 순서. `dilation≠1`, `groups≠1`, `padding_mode≠"zeros"`, `ceil_mode`, `return_indices`, `count_include_pad=False`, `divisor_override`는 에러 |
| `net.BatchNorm1d/2d` | torch | 버퍼 이름 `running_mean`, `running_var`, `num_batches_tracked`, 이동 분산은 불편 추정. `momentum=None`은 누적 평균. `track_running_stats=False`는 에러 |
| `net.Dropout`, `nc.dropout` | torch | 역드롭아웃, eval 모드에서는 항등. `inplace=True`는 에러 |
| `net.LeakyReLU`, `ReLU6`, `Hardsigmoid`, `Hardswish`, `SiLU`, `GELU(approximate)` + 함수형 `nc.leaky_relu` 등 | torch | 같은 정의와 인수. `inplace=True`는 에러 |
| `net.Embedding` | torch | weight ~ N(0, 1). `padding_idx`, `max_norm`은 에러 |
| `Module.zero_grad(set_to_none=True)`, `Optimizer.zero_grad(set_to_none=True)` | torch | 기본값은 grad를 None으로 (버퍼는 재사용해서 매 스텝 할당 없음). 옵티마이저는 grad가 None인 파라미터를 건너뜀 |
| `cross_entropy(input, target, ..., reduction, label_smoothing)` | torch | `reduction`은 "mean"/"sum", `weight`, `ignore_index≠-100`, `reduction="none"`은 에러 |
| `mse_loss(input, target, reduction)` | torch | "mean"/"sum"/"none" |
| `full(size, fill_value, *, dtype)`, `ones(*size, dtype)`, `zeros` | torch | dtype을 생략하면 값에서 정함 (정수 → int32) |
| `state_dict`, `load_state_dict(strict)` | torch | 같은 키 규칙. torch `state_dict`를 numpy로 바꾸면 그대로 불러옵니다 |
| `state_dict` + `nc.save_tensors` / `nc.load_tensors` | torch | 저장 경로는 이 한 가지 (torch.save의 pickle 대신 텐서 파일) |
| `Dataset`, `TensorDataset`, `DataLoader` | torch | 같은 인수 순서와 기본값, `collate_fn` 지원. `sampler`, `batch_sampler`, `num_workers≠0`, `pin_memory`는 에러. `generator`는 `random.Random` (키워드 전용) |
| `Tensor.argmax(dim, keepdim)`, `size()`, `flatten()`, `.T`, `index_select(dim, index)` | torch | `dim=None`에 `keepdim=True`는 에러 |
| `softmax(dim)`, `sum/mean/amax(dim)` | torch | numpy식 `axis=`도 허용. 함수형 `softmax`/`log_softmax`와 `net.Softmax`는 `dim=None`이면 torch의 암묵 규칙 + 경고, `Tensor.softmax(dim)`은 dim 필수 |
| `manual_seed` | torch | RNG만 재시드 |
| `optim.SGD`, `optim.Adam`, `optim.AdamW`, `optim.RMSprop` | torch | 같은 인수 순서, 기본값, 갱신식 (AdamW `weight_decay=0.01`, 분리된 감쇠). `dampening`, `nesterov`, `amsgrad`, `centered`, `maximize`는 에러 |
| `gradient_accumulation_steps` (옵티마이저 인자) | Keras 3 | n번째 `step()`마다 평균 그래디언트로 갱신. 그 사이 `zero_grad()`는 합을 유지 |
| `net.utils.clip_grad_norm_` (`net.clip_grad_norm_`) | torch | 2-노름만 지원 (`norm_type≠2`는 에러), 클리핑 전 노름을 반환 |
| `optim.lr_scheduler.StepLR`, `ExponentialLR`, `CosineAnnealingLR`, `LambdaLR` | torch | 파라미터 그룹 1개. Cosine은 닫힌 식(값은 torch와 같음). `last_epoch≠-1`은 에러 |
| `Model.optimizer` | Keras | compile()이 만든 옵티마이저 |
| `Model.compile(optimizer="adam", loss=None, metrics=None, *, lr=None)` | newch | 0.3.1부터 newch의 편의 함수 (Keras compile과 기본값이 다르므로 Keras 이름 규칙 대상에서 뺌). 이름("adam", "adamw", "sgd", "rmsprop")은 `nc.optim`의 같은 클래스를 그 기본값으로 만듦. `loss`는 "cross_entropy", "mse" 또는 함수·손실 모듈. loss 없이 fit/evaluate하면 에러 |
| `Model.fit`, `evaluate`, `predict` | Keras | Keras 인수 순서 (`x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, ...`), `batch_size=None`은 32, `initial_epoch` 지원. `class_weight`, `sample_weight`, `steps_per_epoch`, `steps`, evaluate/predict의 `callbacks`는 에러 |
| `optim.lr_scheduler.ReduceLROnPlateau` | torch | `step(metric)`, threshold rel/abs, cooldown, min_lr, eps (torch와 같은 판정) |
| `ModelCheckpoint` | Keras | `ModelCheckpoint`는 가중치만 저장하므로 `save_weights_only=True`를 명시해야 함 (Keras 기본값 False는 에러), NCW1 파일, `save_freq="epoch"`만 |
| `EarlyStopping(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights, start_from_epoch)` | Keras | Keras와 같은 순서와 판정 규칙. 모니터 값이 없으면 경고 |
| `LambdaCallback(on_epoch_begin, on_epoch_end, on_train_begin, on_train_end)` | Keras | 배치 단위 훅은 없음 |
| `Tensor.__array__` | numpy | 항상 복사 (`copy=False`는 에러) |
| `nc.net.functional` (`F.relu`, `F.nll_loss`, …) | torch.nn.functional | 같은 이름과 인수. `inplace=True`, `weight`, `ignore_index≠-100`은 에러 |
| `nc.net.init` (`uniform_`, `normal_`, `constant_`, `zeros_`, `ones_`, `xavier_*`, `kaiming_*`, `calculate_gain`) | torch.nn.init | 같은 분포와 공식. 난수는 newch RNG라 같은 시드라도 값은 다름 |
| `nn.ModuleList`, `ModuleDict`, `Identity`, `CrossEntropyLoss`, `NLLLoss`, `MSELoss` | torch | 손실 모듈은 함수형과 같은 인수 규칙 |
| `Module.apply`, `named_modules`, `named_children`, `register_buffer`, `register_parameter`, `to`, `cpu` | torch | `to`는 CPU 장치만 (그 밖은 에러), dtype 변경은 에러. `register_buffer(persistent=False)`는 에러 |
| `Tensor.view`, `view_as`, `unsqueeze`, `squeeze`, `permute`, `dim`, `to`, `float`, `int`, `cpu` | torch | `view`는 절대 복사하지 않음: 연속이 아니면 에러 (torch와 같음), `persistent()` 안에서도 에러. `permute`는 전치 뷰를 이어 붙인 뷰 |
| `eq`/`ne`/`lt`/`le`/`gt`/`ge`와 비교 연산자 | torch | 결과 dtype만 int32 (위의 라이브러리 규칙). 텐서는 여전히 해시 가능 (dict 키로 사용 가능) |
| `nc.cat`, `nc.stack`, `nc.flatten`, `nc.argmax`, `nc.nll_loss` | torch | |

**라이브러리 전체 규칙:** 정수 데이터는 int32가 됩니다 (torch는 int64). 엔진에 int64가 없기 때문입니다. 같은 이유로 `.long()`도 int32이고, bool dtype이 없어서 비교 결과(`eq`, `==` 등)는 int32 0/1입니다 (`.sum().item()`은 torch처럼 정수).

### 의미가 달라서 다른 이름을 쓴 것

| 원본 이름 | newch 이름 | 이유 |
|---|---|---|
| `torch.nn` | **`nc.net`** | PyTorch에 없는 Keras식 `Model`까지 들어 있습니다 (`import newch.net as nn`으로 쓰면 됩니다) |
| `torch.nn.Parameter(tensor)` | **`nc.parameter(init)`** | 클래스가 아니라 함수이고, shape 튜플도 받습니다 |
| `Tensor.max(dim)` (값+인덱스) | **`Tensor.amax(dim)`** | 값만 반환 = torch `amax` |
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
| 저장/불러오기 없음 | NCW1 포맷, `state_dict`, `save_tensors`/`load_tensors`, torch 가중치 가져오기 | 저장 후 예측 완전 일치 |
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

## 7. 활성화 함수 비교 (저사양 기준)

`python tools/bench_act.py`로 다시 잴 수 있습니다. x86 PC(AVX2) 한 코어 기준이며, ARM 보드 실측은 보류 중입니다.

#### Throughput (1048576 elements, median of 9, M elements/s)

| activation | forward | fwd+bwd | fwd+bwd vs relu |
|---|---:|---:|---:|
| relu | 4760 | 302 | 1.00x |
| leaky_relu | 4713 | 293 | 0.97x |
| relu6 | 5068 | 330 | 1.09x |
| hardswish | 5331 | 339 | 1.12x |
| hardsigmoid | 4974 | 308 | 1.02x |
| silu | 2208 | 250 | 0.83x |
| gelu | 1286 | 192 | 0.64x |
| gelu_tanh | 1241 | 211 | 0.70x |
| tanh | 2290 | 321 | 1.06x |
| sigmoid | 3248 | 303 | 1.00x |

#### MNIST MLP 784-256-128-10, Adam 1e-3, batch 64, 2 epochs, 3 seeds

| activation | test acc (mean ± sd) | int8 acc | int8 change (pt) | s/epoch |
|---|---:|---:|---:|---:|
| relu | 96.82 ± 0.50 | 96.82 | +0.00 | 1.6 |
| leaky_relu | 97.00 ± 0.28 | 96.99 | -0.01 | 1.7 |
| relu6 | 96.88 ± 0.46 | 96.89 | +0.00 | 1.7 |
| hardswish | 96.88 ± 0.19 | 96.88 | +0.00 | 1.6 |
| hardsigmoid | 94.96 ± 0.10 | 94.95 | -0.00 | 1.8 |
| silu | 96.96 ± 0.16 | 96.96 | -0.00 | 1.6 |
| gelu | 96.98 ± 0.26 | 96.98 | -0.01 | 1.6 |
| gelu_tanh | 96.99 ± 0.26 | 96.98 | -0.00 | 1.8 |
| tanh | 96.41 ± 0.28 | 96.41 | -0.00 | 1.6 |
| sigmoid | 95.26 ± 0.13 | 95.26 | +0.00 | 1.6 |

**결론**
- 정확도: ReLU, LeakyReLU, ReLU6, Hardswish, SiLU, GELU는 이 규모에서 표준편차 안의 차이입니다. Hardsigmoid와 Sigmoid를 은닉층에 쓰면 눈에 띄게 낮습니다
- int8: 활성값을 행마다 동적으로 양자화하므로 어떤 함수든 정확도 변화가 ±0.02%p 안입니다. 범위가 고정된 함수(ReLU6 등)의 이점은 정적 양자화(예정)에서 나타납니다
- 속도: exp가 필요한 함수(SiLU, sigmoid, tanh, GELU)는 libm 호출만 쓰면 ReLU보다 순전파가 수십 배 느렸습니다. 벡터 exp/erf 커널로 순전파가 13~32배 빨라졌고 (SiLU 105 → 2,200M 원소/초), MLP 에포크 시간도 SiLU 2.0초·GELU-tanh 2.2초에서 ReLU와 같은 1.6초 안팎이 되었습니다. exp가 없는 소형 코어용 대안이 필요하면 Hardswish(SiLU와 같은 모양, 곱셈과 비교만)를 권합니다
- **권장:** 기본은 ReLU, 정확도를 조금 더 원하면 SiLU 또는 GELU, NEON이 약하거나 없는 아주 작은 코어에서는 Hardswish

## 8. 검증 상태

| 항목 | 결과 |
|---|---|
| C 테스트 | **23개 테스트 통과** (추론 해제: 같은 결과·최대 메모리 1/3·테이프와 뷰는 보호, 역전파 해제 3모드의 그래디언트 비트 일치·최대 메모리 감소·재역전파 에러, nll_loss·cat·to_dtype gradcheck, 비교 연산과 i32 sum, 활성화 7종 gradcheck와 알려진 값, 벡터 exp/sigmoid/tanh/erf와 libm 비교, AdamW, RMSprop, 그래디언트 누적, 클리핑, grad None 처리 추가. 스레드 결정성 테스트가 열 분할 GEMM 경로를 실제로 거치도록 수정) — 기존: 연산별 gradcheck 40여 건, 합성곱·풀링·배치 정규화, 드롭아웃 통계, int8 정확도와 커널 일치, 저장/불러오기와 에러, 스레드 결정성, 아레나 축소, 학습 2종 등) |
| Python 테스트 | **65개 통과** (추론 메모리 3개: 반복 순전파가 쌓이지 않음·참조 중인 텐서와 뷰 유지·no_grad 텐서를 나중에 학습에 써도 안전 / 0.3.1: `newch.ml` 테스트 7개 제거, `ReduceLROnPlateau` 스케줄러 1개 추가 / 역전파 메모리 3개: 해제해도 그래디언트 동일·참조 중인 출력 유지·`retain_graph` / `test_torch_style.py` 10개: PyTorch 예제 구조 그대로 CNN 학습 90% 이상, 텐서 메서드, 컨테이너, 손실 모듈, nn.init, 장치 이동 / `test_extras.py` 26개: 활성화 레이어 값과 그래디언트, 인수 순서·기본값 호환 10개 포함, AdamW, 누적=큰 배치 동일성, 클리핑, 스케줄러, Keras 콜백, numpy 연동, `newch.ml` 전체 / 기존 17개 + CNN 학습, 레이어 shape와 모드, BN 버퍼, 저장 후 예측 일치, strict/불일치 에러, torch 형식 state_dict, 양자화, FileArray 스트리밍 fit, 일반 Dataset 묶음, GC 자동 해제, 스레드/trim) |
| 디버그 빌드 | `-O0` + UBSan 통과 |
| 실제 데이터 | MNIST CNN 1 에포크 18초, 테스트 98.13%, int8 후 98.13% |
| 패키징 | 휠 빌드 → 별도 venv 설치(numpy 없음) → 학습 성공 |
| 크로스 컴파일 | x86_64 Windows/Linux, aarch64(gnu/musl/v8.2), armv7(gnu/musl), riscv64(gnu/musl). 경고 0 |
| CI (GitHub Actions) | Windows, Ubuntu, macOS(Apple Silicon)에서 C·Python 테스트, QEMU aarch64·armv7·riscv64, CMake+gcc, 휠 빌드. 모든 테스트는 해제 메모리 덮어쓰기 모드(`NC_DEBUG_POISON=1`)로 실행 |

**CI에서 찾아 고친 버그:** Python에서 `lib.f(임시텐서().ptr)` 형태로 쓰면 C 호출 전에 영구 임시 텐서가 해제되는
use-after-free가 있었습니다 (7곳). Windows 할당기에서는 우연히 동작했고 Linux/macOS에서 드러났습니다.
수정 후 `NC_DEBUG_POISON`으로 모든 플랫폼에서 재현되게 만들고 회귀 테스트를 추가했습니다.

0.3.0 CI에서는 **스레드 수에 따라 행렬곱 결과가 1비트 달라지는** 문제가 CMake+gcc 빌드에서만 드러났습니다. 열 분할 시 청크 경계가 16열 타일 중간에 놓이면 경계 옆 열이 스칼라 경로로 계산되는데, gcc는 이 경로를 벡터 경로와 다르게 반올림합니다(clang은 우연히 같음). 이제 행은 4행, 열은 16열 타일 단위로만 나눕니다. `-ffp-contract=off` 빌드로 로컬에서 재현하고 수정을 확인했습니다.

**아직 확인하지 못한 것**
- 저사양 ARM 보드(오렌지파이 등) 실측: 보류. NEON 커널의 정확성은 Apple Silicon과 QEMU에서 확인했지만, Cortex-A 코어에서의 **속도**는 측정하지 않았습니다
- 리눅스·macOS에서의 `pip install`(sdist) 설치 경로

## 9. 빌드와 사용

```bash
pip install .                           # 설치 (시스템 C 컴파일러 불필요)
pip install ziglang                     # 개발 시
python tools/build.py                   # 빌드 + C/Python 테스트
python tools/build.py --target aarch64-static   # 정적 ARM 바이너리
python tools/build.py --target all      # 모든 타깃
python tools/build.py --debug           # -O0 + UBSan
python examples/mnist_cnn.py            # MNIST 다운로드 → 스트리밍 학습 → 저장 → 양자화
```

## 10. 알려진 제약

- **학습 순전파의 메모리는 배치 크기에 비례합니다.** 역전파 중 해제로 학습 스텝의 실제 메모리는 절반 가까이 줄었고 no_grad 순전파는 다 쓴 활성값을 바로 돌려주지만, 그래디언트를 켠 순전파는 역전파까지 모든 활성값이 남습니다. 저메모리 기기에서는 배치를 줄이거나 `gradient_accumulation_steps`를 쓰세요. 순전파 중 재사용은 그래프 컴파일러의 메모리 계획 단계에서 다룹니다
- dtype: f32(연산), i32(라벨·인덱스), i8(양자화 가중치). f16, int64 없음
- `matmul`의 두 번째 피연산자는 2차원만, `softmax` 계열은 마지막 축만, `index_select`는 축 0만
- Conv2d: dilation, groups, `padding="same"` 없음. 양자화는 `Linear`만
- 원소별 연산과 im2col은 단일 스레드입니다
- 벡터 exp/erf는 근사이므로 결과가 레퍼런스(libm) 빌드와 마지막 비트에서 다를 수 있습니다 (`force_reference=True`면 libm)
- `predict()`는 numpy 입력이면 ndarray, 아니면 리스트를 반환합니다
- 옵티마이저 상태는 저장하지 않습니다 (`state_dict()`는 모델 것만)

## 11. 다음 단계 후보

1. **방향성 토론 후 결정** (메모리 예산 학습, C 파일 하나로 내보내기, 기기 위 지속 학습 등)
2. **그래프 컴파일러:** 추적 → IR → 원소별 연산 융합, 정적 메모리 계획 (스텝 내 메모리 재사용)
3. **ARM 실측:** 오렌지파이에서 `aarch64-static/nc_tests` 실행, NEON int8(dotprod) 커널
4. **원소별 연산·im2col 병렬화, RISC-V RVV 커널**
5. **옵티마이저 상태 저장**(학습 재개, 전원이 끊기기 쉬운 보드에 유용)
