# newch

저사양 기기에서도 **학습**까지 할 수 있는 작은 머신러닝 라이브러리입니다. 엔진은 C11이고, Python에서는
PyTorch 스타일의 텐서·autograd·모듈을 쓰며, 짧은 학습 루프 `fit()`을 함께 제공합니다. Python 의존성은 없습니다
(numpy가 있으면 사용합니다). 싱글보드 컴퓨터는 호환 대상 중 하나입니다.

- 네이티브 라이브러리 약 220KB (Windows), aarch64 리눅스용 약 110KB
- MNIST CNN: 1 에포크 14초, 테스트 정확도 97~98% (x86 PC, 디스크에서 스트리밍)
- 학습 스텝의 메모리: 역전파가 다 쓴 값을 바로 돌려줘서 MNIST CNN 배치 64 기준 29MiB → 15MiB
- 행렬곱: 멀티스레드 + AVX2로 numpy(BLAS)의 0.9~1.5배 시간

## 설치

```
pip install newch            # Windows x86-64는 바로 설치, 그 밖의 플랫폼은 설치 중 엔진을 컴파일합니다
pip install "newch[numpy]"   # numpy 연동까지
```

소스에서 설치할 때도 시스템 C 컴파일러는 필요 없습니다 (pip 패키지 `ziglang`을 씁니다).
저장소에서는 `pip install .`, 개발할 때는 `pip install ziglang` 후 `python tools/build.py`를 실행합니다.
라이선스는 Apache-2.0입니다 ([LICENSE](https://github.com/dandmadp/newch/blob/main/LICENSE), [NOTICE](https://github.com/dandmadp/newch/blob/main/NOTICE)).

## Python 사용 예

PyTorch 모델 코드가 import만 바꾸면 그대로 돕니다. 모든 연산은 C 엔진에서 실행됩니다.

```python
import newch as torch                 # 텐서, 수학, 메모리
import newch.net as nn                # 모듈 (torch.nn)
import newch.net.functional as F      # 활성화, 합성곱, 손실 등 신경망 함수 (torch.nn.functional)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        return F.log_softmax(self.fc(torch.flatten(x, 1)), dim=1)

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for data, target in loader:
    optimizer.zero_grad()
    loss = F.nll_loss(model(data), target)
    loss.backward()
    optimizer.step()
    torch.step_end()                      # newch만의 한 줄: 이 스텝의 임시 텐서를 한 번에 해제

torch.save_tensors(model.state_dict(), "model.ncw")       # 다른 기기에서
model.load_state_dict(torch.load_tensors("model.ncw"))    # 이렇게 불러옵니다
```

전체 학습·평가 루프는 [examples/mnist_torch_style.py](https://github.com/dandmadp/newch/blob/main/examples/mnist_torch_style.py)에 있습니다.

**짧게 쓰고 싶을 때 (`fit`):** 같은 모델에 학습 루프를 맡깁니다.

```python
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Dropout(0.25), nn.Linear(16 * 14 * 14, 10),
)
model.compile(optimizer="adam", loss="cross_entropy", metrics=["accuracy"])   # "adam" = nc.optim.Adam 그대로
history = model.fit(x, y, batch_size=64, epochs=5, validation_split=0.1,
                    callbacks=[torch.callbacks.EarlyStopping(patience=2)])
loss, acc = model.evaluate(x_test, y_test)
```

**RAM보다 큰 데이터셋:** 샘플을 디스크 파일에 저장하고 `FileArray`로 열면 배치 단위로만 읽습니다.
픽셀처럼 0~255인 데이터는 `uint8`로 두면 float32의 1/4 크기이고, 배치마다 `.float()`로 바꿉니다.

```python
from newch.data import DataLoader, FileArray, TensorDataset

x = FileArray.create("x.u8", rows, row_shape=(1, 28, 28), dtype=torch.uint8)   # 한 행씩 기록
y = FileArray.create("y.u8", labels, row_shape=(), dtype=torch.uint8)
for xb, yb in DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True):
    loss = F.cross_entropy(model(xb.float() / 255), yb)
    ...
```

**학습 재개:** 옵티마이저와 스케줄러도 PyTorch와 같은 `state_dict()`를 가지며, 체크포인트 전체를 파일 하나에 저장합니다.

```python
torch.save_tensors({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch}, "ckpt.ncw")
ckpt = torch.load_tensors("ckpt.ncw")
model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["optimizer"])
```

**PyTorch 가중치 가져오기:** 레이어 이름과 레이아웃이 같으므로 numpy로 바꿔서 넣으면 됩니다.

```python
model.load_state_dict({k: v.numpy() for k, v in torch_model.state_dict().items()})
```

**메모리가 작을 때:** `gradient_accumulation_steps`로 작은 배치를 여러 번 모아 한 번에 갱신합니다.

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, gradient_accumulation_steps=4)   # 배치 16 x 4 = 64처럼 학습
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2)
model.compile(optimizer=opt, loss="cross_entropy", metrics=["accuracy"])
model.fit(x, y, batch_size=16, validation_split=0.1, callbacks=[
    torch.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: sched.step(logs["val_loss"])),
    torch.callbacks.ModelCheckpoint("best.ncw", save_best_only=True, save_weights_only=True)])
```

**활성화 함수:** `ReLU`, `LeakyReLU`, `ReLU6`, `Hardsigmoid`, `Hardswish`, `SiLU`, `GELU`, `Sigmoid`, `Tanh` (PyTorch와 같은 정의).
exp가 필요한 함수도 AVX2·NEON 벡터 커널로 ReLU와 비슷한 속도로 학습합니다. 비교 결과는
[docs/PROGRESS.md](https://github.com/dandmadp/newch/blob/main/docs/PROGRESS.md)의 "활성화 함수 비교"에 있습니다.

**int8 양자화 (추론):** `newch.quantize_(model)`은 모든 `Linear`를 int8 가중치로 바꿉니다 (가중치 메모리 1/4).

**스레드:** 기본값은 모든 코어입니다. `newch.set_default_context(newch.Context(threads=2))`로 바꿀 수 있습니다.

전체 예제는 [examples/mnist_cnn.py](https://github.com/dandmadp/newch/blob/main/examples/mnist_cnn.py)에 있습니다 (다운로드 → 디스크 변환 → 스트리밍 학습 → 저장 → 양자화).

PyTorch나 Keras와 이름이 같은 API는 동작도 같고, 의미가 다른 경우에는 다른 이름을 씁니다
(목록은 [docs/PROGRESS.md](https://github.com/dandmadp/newch/blob/main/docs/PROGRESS.md)의 "이름 규칙").

## C 사용 예

```c
nc_ctx *ctx = nc_init(&(nc_config){.threads = 0});            /* 0 = 모든 코어 */
nc_module *layers[] = {nc_linear(ctx, 2, 64, 1), nc_act(ctx, NC_ACT_RELU), nc_linear(ctx, 64, 3, 1)};
nc_module *model = nc_sequential(ctx, 3, layers);
nc_tensor *params[16];
int np = nc_parameters(model, params, 16);
nc_optim *opt = nc_adam(ctx, params, np, (nc_optim_opts){.lr = 1e-3f});

for (int step = 0; step < steps; step++) {
    nc_tensor *x = nc_from_data(ctx, NC_F32, batch_x, 2, (int64_t[]){B, 2});
    nc_tensor *y = nc_from_data(ctx, NC_I32, batch_y, 1, (int64_t[]){B});
    nc_tensor *loss = nc_cross_entropy(nc_forward(model, x), y);
    nc_optim_zero_grad(opt);
    nc_backward(loss);
    nc_optim_step(opt);
    if (!nc_ok(ctx)) { puts(nc_error(ctx)); break; }   /* 에러는 한 번에 확인 */
    nc_step_end(ctx);                                   /* 임시 텐서 전부 해제, O(1) */
}
const char *names[] = {"w", "b"};
nc_save(ctx, "model.ncw", 2, names, params);
nc_free_ctx(ctx);
```

## 지원 플랫폼

| 타깃 (`tools/build.py --target`) | 대표 기기 | 커널 |
|---|---|---|
| `host` | 개발 PC (Windows / Linux / macOS) | x86: AVX2+FMA (런타임 감지), ARM: NEON |
| `aarch64`, `aarch64-static` | Pi 3/4/5, Orange Pi 5, RK35xx, Apple Silicon | NEON |
| `armv7`, `armv7-static` | Pi 2/3 (32비트 OS), 구형 Allwinner | NEON (런타임 확인) |
| `riscv64`, `riscv64-static` | VisionFive 2, Milk-V, BPI-F3 | 레퍼런스 (RVV 예정) |

`*-static`은 musl 정적 링크 바이너리라서 어떤 리눅스 기기에든 복사해서 바로 실행할 수 있습니다.
기기에서 직접 빌드할 때는 `cmake -B build && cmake --build build -j && ctest --test-dir build`.

## 핵심 설계

- **메모리:** 파라미터 등 영구 텐서와 스텝 아레나(활성값, 중간 그래디언트)를 나눕니다. `step_end()`가 아레나를 O(1)로
  되돌리고, `trim()`이 남는 메모리를 OS에 반환합니다. Python에서는 영구 텐서를 마지막 참조가 사라질 때 자동으로 해제합니다.
  역전파는 다 쓴 활성값과 중간 그래디언트를 바로 돌려줘서, MNIST CNN 학습 스텝의 실제 메모리가 29MiB에서 15MiB로 줄었습니다
- **에러:** 연산은 abort하지 않습니다. 첫 에러를 기록하고 NULL을 전파하며, Python에서는 `NcError`가 됩니다
- **연산은 데이터로 기술:** 모든 연산이 `nc_op_def`(infer/view/forward/backward)로 등록되고 `nc__apply()`를 거칩니다
- **커널 디스패치:** 레퍼런스 테이블에서 시작해 CPU가 지원하는 계층(NEON, AVX2)이 항목별로 덮어씁니다.
  `test_kernel_parity`가 모든 계층을 레퍼런스와 비교합니다
- **스레드:** 큰 GEMM만 행 또는 열 단위로 나눠 병렬 실행합니다. 출력 원소 하나는 스레드 하나가 계산하므로 결과가 스레드 수와
  무관하게 비트 단위로 같습니다

## 디렉터리

```
engine/include/nc.h        공개 C API (NC_API로 표시된 함수만 공유 라이브러리에서 노출)
engine/src/core/           컨텍스트, 메모리, 텐서, 레지스트리, 파일 입출력
engine/src/autograd/       nc__apply, 테이프, backward, gradcheck
engine/src/ops/            elementwise, linalg, reduce, shape, nn, conv, quant
engine/src/kernels/        디스패치, ref/, neon/, x86/(AVX2)
engine/src/runtime/        CPU 감지, 스레드 풀
engine/tests/              C 테스트
newch/                     Python 패키지 (tensor, net/{functional,init}, optim, data, training, callbacks)
tests/                     Python 테스트
examples/                  train_spiral.c, mnist_cnn.py (Keras식), mnist_torch_style.py (PyTorch식)
docs/                      PROGRESS.md (작업 요약), FORMAT.md (가중치 파일 포맷)
tools/build.py             zig cc 기반 멀티 타깃 빌드
.github/workflows/ci.yml   CI (Windows/Linux/macOS, QEMU ARM·RISC-V, 휠)
```
