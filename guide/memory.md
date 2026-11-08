# 메모리 모델

newch가 PyTorch와 가장 다른 부분입니다. 작은 기기에서 메모리를 예측할 수 있게 하려고, 텐서를 **임시 텐서**와
**영구 텐서** 두 종류로 나눕니다.

| | 임시 텐서 | 영구 텐서 |
|---|---|---|
| 무엇 | 연산 결과 대부분 (활성값, 중간값, 배치 데이터) | 파라미터, 버퍼, `persistent()` 안에서 만든 텐서 |
| 메모리 | 스텝 아레나 (큰 블록에서 잘라 씀) | 텐서마다 따로 할당 |
| 해제 | `step_end()`가 한 번에 (O(1)) | Python에서 마지막 참조가 사라질 때 |

## step_end()

```python
import newch as torch
import newch.net as nn
import newch.net.functional as F

model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for step in range(3):
    x = torch.randn(8, 4)                    # 임시 텐서: 이번 스텝에서만 씀
    target = torch.randn(8, 2)
    opt.zero_grad()
    loss = F.mse_loss(model(x), target)
    loss.backward()
    opt.step()
    print(step, loss.item())                 # 값은 step_end() 전에 꺼냅니다
    torch.step_end()                         # 이 스텝의 임시 텐서를 모두 해제
```

`step_end()` 뒤에 그 스텝의 임시 텐서를 쓰면 해제된 메모리를 읽는 대신 분명한 에러가 납니다.

```python
x = torch.randn(3)
torch.step_end()
try:
    x.sum()
except torch.NcError as e:
    print("에러:", e)
```

`fit()`, `predict()`, `evaluate()`는 배치마다 알아서 `step_end()`를 부릅니다.

## persistent()

여러 스텝에 걸쳐 쓸 텐서는 `persistent()` 안에서 만듭니다.

```python
with torch.persistent():
    dataset_x = torch.randn(100, 4)
    mean = dataset_x.mean(0)
torch.step_end()
print(mean.shape)            # 아직 살아 있음
torch.release(dataset_x)     # 다 썼으면 바로 해제할 수도 있습니다 (안 하면 참조가 사라질 때 해제)
```

파라미터(`nn.Linear`의 `weight` 등)와 옵티마이저 상태는 원래 영구입니다.

## 역전파와 추론 중의 해제

- 역전파는 노드를 지날 때마다 다 쓴 활성값과 중간 그래디언트를 돌려줍니다. Python에서 아직 참조하는 텐서는
  지우지 않습니다. MNIST CNN(배치 64)의 학습 스텝 메모리가 이 덕분에 29MiB에서 15MiB로 줄었습니다.
- `no_grad()` 안에서는 Python이 더 이상 참조하지 않는 중간값을 즉시 해제하므로, 긴 추론도 한 스텝 안에서 메모리가 쌓이지
  않습니다.

## 메모리 측정

```python
torch.default_context().reset_peak()
with torch.no_grad():
    y = model(torch.randn(256, 4))
stats = torch.mem_stats()
print({k: v for k, v in stats.items()})
torch.step_end()
```

| 항목 | 뜻 |
|---|---|
| `arena_bytes_in_use` | 지금 쓰는 임시 메모리 |
| `arena_peak_bytes` | `reset_peak()` 이후 최대 임시 메모리 |
| `arena_bytes_reserved` | 아레나가 OS에서 받아 둔 메모리 (다음 스텝에 재사용) |
| `persistent_bytes` | 영구 텐서 전체 |

유난히 큰 스텝이 지나간 뒤 받아 둔 메모리를 OS에 돌려주려면 `torch.default_context().trim()`을 부릅니다.

## Context와 스레드

모든 메모리는 `Context`가 가집니다. 보통은 기본 컨텍스트 하나로 충분합니다.

```python
ctx = torch.Context(seed=1, threads=2)       # threads=0(기본)은 모든 코어
torch.set_default_context(ctx)
print(ctx.threads, ctx.backend)              # 쓰는 스레드 수와 커널 (avx2, neon, ref)
```

- 큰 행렬곱, 원소별 연산, 합성곱, 풀링을 여러 스레드로 나눕니다. 출력 원소 하나는 늘 한 스레드가 계산하므로
  **결과가 스레드 수와 무관하게 비트 단위로 같습니다.**
- 컨텍스트를 버리면(참조가 모두 사라지면) 엔진 메모리와 워커 스레드도 정리됩니다.
- `torch.manual_seed(n)`은 기본 컨텍스트의 난수 생성기를 다시 시드합니다.
