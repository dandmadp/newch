# 배포와 추론

학습한 모델을 작은 기기에서 돌리기 위한 내용입니다. 저장 파일, int8 양자화, 지원 플랫폼, C API를 다룹니다.

```python
import newch as torch
import newch.net as nn
```

## 저장 파일 (NCW1)

`torch.save_tensors(dict, path)`는 텐서를 단순한 바이너리 형식(NCW1)으로 씁니다. 텐서마다
이름, dtype, 모양, 원시 데이터가 순서대로 들어가므로 C API(`nc_load`)나 다른 언어에서도 쉽게 읽을 수 있습니다.
형식은 [docs/FORMAT.md](https://github.com/dandmadp/newch/blob/main/docs/FORMAT.md)에 있습니다.

```python
model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
torch.save_tensors(model.state_dict(), "model.ncw")
print(list(torch.load_tensors("model.ncw")))
```

불러오기는 파일의 크기·dtype·모양을 모두 검사하므로, 손상되었거나 조작된 파일은 `NcError`를 냅니다.

## int8 양자화

`quantize_(model)`은 모든 `Linear`를 int8 가중치(출력 행마다 scale) 버전으로 바꿉니다. 가중치 메모리가 1/4이 되고,
행렬곱은 int8 커널로 돕니다. 추론 전용입니다.

```python
x = torch.randn(8, 16)
before = model(x).tolist()
torch.quantize_(model)
after = model(x).tolist()
print(model[0])
print(max(abs(a - b) for ra, rb in zip(before, after) for a, b in zip(ra, rb)))
```

양자화한 모델도 `state_dict()`로 저장하고, 같은 구조를 만들어 `quantize_()`한 뒤 불러오면 됩니다.

## 스레드와 성능

- 기본값은 모든 코어입니다. 작은 기기에서 다른 작업과 나눠 쓰려면 `torch.set_default_context(torch.Context(threads=2))`.
- 결과는 스레드 수와 무관하게 비트 단위로 같으므로, PC에서 검증한 결과를 기기에서 그대로 재현할 수 있습니다.
- x86은 AVX2+FMA 커널(런타임 감지), ARM은 NEON 커널을 씁니다. `torch.default_context().backend`로 확인합니다.

## 지원 플랫폼

| 타깃 (`tools/build.py --target`) | 대표 기기 | 커널 |
|---|---|---|
| `host` | 개발 PC (Windows / Linux / macOS) | x86: AVX2+FMA, ARM: NEON |
| `aarch64`, `aarch64-static` | Raspberry Pi 3/4/5, Orange Pi 5, RK35xx | NEON |
| `armv7`, `armv7-static` | Raspberry Pi 2/3 (32비트 OS) | NEON (런타임 확인) |
| `riscv64`, `riscv64-static` | VisionFive 2, Milk-V | 레퍼런스 |

`*-static`은 musl 정적 링크 바이너리라 어떤 리눅스 기기에든 복사해서 실행할 수 있습니다. 크로스 컴파일에도 시스템
컴파일러는 필요 없습니다(`pip install ziglang`).

## C API

Python 없이 C에서 바로 쓸 수 있습니다. 공개 API는
[engine/include/nc.h](https://github.com/dandmadp/newch/blob/main/engine/include/nc.h) 하나입니다.

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
nc_free_ctx(ctx);
```

- 연산은 abort하지 않습니다. 첫 에러를 컨텍스트에 기록하고 이후 연산은 NULL을 돌려주므로, 스텝 끝에 한 번 확인하면 됩니다.
- C 옵티마이저 생성 함수는 `nc_optim_opts`의 0을 "기본값"으로 읽습니다(beta1 0.9, eps 1e-8 등).
  정확히 0을 쓰려면 생성한 뒤 `nc_optim_set_opts`로 넣습니다.
- 전체 예제는 [examples/train_spiral.c](https://github.com/dandmadp/newch/blob/main/examples/train_spiral.c)입니다.

```python
torch.step_end()
```
