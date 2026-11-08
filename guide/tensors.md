# 텐서

`newch.Tensor`는 C 엔진이 가진 다차원 배열을 가리킵니다. 만들기, 연산, 인덱싱은 PyTorch와 같은 이름과 규칙을 따릅니다.

```python
import newch as torch
import newch.net.functional as F
```

## 만들기

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])      # 정수 -> int32
b = torch.tensor([1.5, 2.5])                  # 실수 -> float32
z = torch.zeros(2, 3)
o = torch.ones(2, 3, dtype=torch.int32)
f = torch.full((2, 2), 7.0)
r = torch.rand(3, 4)                          # U[0, 1)
n = torch.randn(3, 4, requires_grad=True)     # N(0, 1)
print(a.dtype, b.dtype, a.shape, n.requires_grad)
```

생성 함수는 모두 `requires_grad=`와 `device=`를 받습니다. 장치는 CPU뿐이라 `device="cpu"`나 `device=x.device`만 됩니다.

### dtype

| dtype | 용도 |
|---|---|
| `torch.float32` | 계산과 학습. 모든 미분 가능한 연산은 float32 |
| `torch.int32` | 정수 데이터, 클래스 번호, 인덱스, 비교 결과 |
| `torch.uint8`, `torch.int8` | 저장용 1바이트 데이터(이미지 픽셀, 작은 라벨). 계산 전에 `.float()` |

```python
pixels = torch.tensor([0, 128, 255], dtype=torch.uint8)
print((pixels / 255).tolist())    # 정수 / 수 -> float32 (torch와 같은 승격 규칙)
labels = torch.tensor([3, 1, 2]).long()   # long()도 int32: 엔진에 int64가 없습니다
```

정수 텐서끼리의 산술(`a + 1`, `a * a`)은 torch라면 정수 결과가 나오지만, 엔진이 float32로 계산하므로 지원하지 않고
`.float()`로 바꾸라는 에러를 냅니다. 결과가 float이 되는 경우(`a / 2`, `a * 0.5`, `a + 실수 텐서`)는 됩니다.

## 연산

```python
x = torch.randn(4, 3)
y = (x * 2 + 1).relu()
s = x.sum(), x.mean(dim=0), x.amax(dim=1), x.sum((0, 1))
p = x.softmax(dim=-1)
w = torch.randn(3, 5)
h = x @ w                         # 행렬곱. 3차원 이상은 배치 행렬곱 (배치 차원 브로드캐스트)
print(h.shape, p.sum(-1).tolist())
```

| 분류 | 연산 |
|---|---|
| 산술 | `+ - * / **`(스칼라 지수), `@`, 비교 `== != < <= > >=` |
| 수학 | `exp log sqrt relu sigmoid tanh softmax log_softmax` |
| 축소 | `sum mean amax argmax` (`dim`은 정수나 튜플, `keepdim`) |
| 모양 | `reshape view transpose permute flatten squeeze unsqueeze contiguous .T .mT` |
| 기타 | `masked_fill index_select detach to float int item tolist numpy`, `torch.cat`, `torch.stack` |

활성화 함수와 신경망 연산(합성곱, 정규화, 손실, attention 등)은 `newch.net.functional`(`F`)에 있습니다.

`max`/`min`은 torch에서 값과 인덱스를 함께 돌려주는 경우가 있어 이름을 바꿨습니다. 최댓값만 필요하면 `amax`,
위치는 `argmax`를 쓰세요.

## 인덱싱과 뷰

numpy·torch와 같은 기본 인덱싱을 모든 축에서 지원합니다. 정수와 슬라이스는 데이터를 복사하지 않는 **뷰**입니다.

```python
m = torch.tensor([[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]])
print(m[1].tolist())              # 행
print(m[:, 2].tolist())           # 열 (뷰)
print(m[-1, 1:3].tolist())        # 음수 인덱스, 슬라이스
print(m[..., ::2].shape, m[None].shape)
print(m[[2, 0]].tolist())         # 정수 리스트 (복사, 그래디언트도 흐름)
print(m[:, [1, 1, 3]].shape)
```

- 인덱스 리스트·배열·int32 텐서는 한 표현식에 하나까지입니다 (`x[[0, 1], [2, 3]]` 같은 다중 고급 인덱싱은 에러).
- bool 마스크 인덱싱(`x[x > 0]`)은 지원하지 않습니다. bool dtype이 없어서, 정수 0/1을 인덱스로 잘못 읽는 일이 없도록
  에러를 냅니다. 마스크로 값을 바꾸려면 `masked_fill`을 씁니다.
- 음수 step(`x[::-1]`)은 torch와 같이 에러입니다.
- 인덱스 대입(`x[0] = 1`)은 없습니다.

```python
scores = torch.randn(2, 4)
mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])          # 1 = 가림
print(scores.masked_fill(mask, float("-inf")).softmax(-1).tolist())
```

`reshape`과 `view`는 메모리 배치가 허락하면 복사하지 않습니다. `transpose`, `permute`, `.T`도 뷰입니다.

## numpy 연동

```python
import numpy as np

t = torch.tensor(np.arange(6, dtype=np.float32).reshape(2, 3))   # numpy -> 텐서 (복사)
arr = t.numpy()                                                  # 텐서 -> numpy (복사)
print(np.asarray(t).sum(), (np.ones(3, np.float32) + t).shape)   # numpy와 섞어 써도 결과는 Tensor
```

numpy의 `uint8`/`int8`은 1바이트 그대로, 다른 정수는 int32, 실수는 float32가 됩니다.

## autograd

```python
w = torch.randn(3, requires_grad=True)
x = torch.tensor([1.0, 2.0, 3.0])
loss = ((w * x).sum() - 1.0) ** 2
loss.backward()
print(w.grad.tolist())
```

- `backward()`는 스칼라에서만 부릅니다. 그래디언트는 잎 텐서의 `.grad`에 누적됩니다.
- 역전파는 다 쓴 중간값을 바로 해제하므로, 같은 그래프로 두 번 역전파하려면 첫 번째에 `retain_graph=True`를 줍니다.
- `with torch.no_grad():` 안에서는 기록하지 않습니다. `x.detach()`는 그래프에서 떼어 낸 뷰입니다.

```python
torch.step_end()
```
