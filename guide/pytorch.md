# PyTorch와 다른 점

newch의 원칙은 하나입니다. **PyTorch나 Keras와 이름이 같은 API는 동작도 같습니다.** 동작이 다르면 다른 이름을 쓰고
(`amax`, `quantize_`, `newch.net`), 지원하지 않는 인수나 경우는 조용히 다르게 동작하지 않고 에러를 냅니다.
PyTorch 코드를 옮길 때 알아 둘 차이는 아래와 같습니다.

## 꼭 바꿔야 하는 것

| PyTorch | newch | 이유 |
|---|---|---|
| `import torch.nn as nn` | `import newch.net as nn` | 모듈 패키지 이름이 다름 |
| `import torch.nn.functional as F` | `import newch.net.functional as F` | |
| (없음) | 스텝 끝에 `torch.step_end()` | 임시 텐서를 한 번에 해제 ([메모리 모델](memory.md)) |
| 여러 스텝에 쓰는 텐서 | `with torch.persistent():` 안에서 만들기 | 같은 이유 |
| `torch.save(obj, path)` / `torch.load` | `torch.save_tensors(obj, path)` / `torch.load_tensors` | pickle 대신 단순한 텐서 파일 |
| `x.max()`, `x.min()` | `x.amax()`, `x.argmax()` | torch의 `max(dim)`은 값과 인덱스를 함께 돌려줌 |

## dtype

- 실수는 `float32`, 정수는 `int32`만 계산에 씁니다. `int64` 데이터는 int32로, `float64`는 float32로 바뀝니다.
  `.long()`도 int32입니다.
- `uint8`, `int8`은 저장용입니다(산술 전에 `.float()`).
- bool dtype이 없어서 비교 결과(`a > b`)는 정수 0/1입니다. 마스크 인수(`masked_fill`, attention 마스크)는 이 정수
  텐서를 받고, 0과 1의 의미는 PyTorch의 False/True와 같습니다. bool 인덱싱(`x[mask]`)은 에러입니다.
- 정수 텐서끼리의 산술(`a + 1`)은 에러입니다. 결과가 float이 되는 경우(`a / 2`, `a * 0.5`)는 PyTorch와 같이 됩니다.

## 장치

CPU만 있습니다. `.to("cpu")`, `.cpu()`, `device="cpu"`, `x.device`는 되고, 다른 장치는 에러입니다.

## 지원하지 않는 것 (에러)

| 분류 | 항목 |
|---|---|
| 인덱싱 | 인덱스 대입 `x[i] = v`, bool 마스크, 인덱스 리스트 둘 이상, 음수 step |
| 텐서 | in-place 연산(`add_` 등), `max`/`min`, `zeros_like` 같은 `*_like`, `arange` |
| 레이어 | `TransformerDecoder`, `nn.Transformer`, LSTM `proj_size`, `Embedding(padding_idx, max_norm)`, `padding_mode≠"zeros"`, 풀링 `dilation`/`ceil_mode` |
| 손실 | `weight`, `ignore_index`(기본값 외), `reduction="none"`(cross_entropy) |
| 옵티마이저 | `nesterov`, `dampening`, `amsgrad`, `centered`, `maximize`, 파라미터 그룹 여러 개 |
| 데이터 | `num_workers`, `sampler`, `batch_sampler`, `pin_memory` |
| autograd | `backward(gradient=...)`, `create_graph`(고차 미분) |

## 결과가 PyTorch와 다를 수 있는 것

- **난수:** 분포는 같지만 생성기가 달라서, 같은 시드라도 값(초기 가중치, dropout 마스크)은 다릅니다.
- **부동소수점:** 연산 순서가 달라 마지막 자리가 다를 수 있습니다. 대신 newch 안에서는 스레드 수와 무관하게
  결과가 비트 단위로 같습니다.

전체 대응표는 [docs/PROGRESS.md](https://github.com/dandmadp/newch/blob/main/docs/PROGRESS.md)의 "이름 규칙"에 있습니다.
