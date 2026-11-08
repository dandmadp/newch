# 시작하기

이 페이지는 텐서를 만들고, 그래디언트를 구하고, 작은 모델을 학습해 저장하는 데까지를 한 번에 보여 줍니다.
PyTorch를 써 봤다면 대부분 익숙한 코드입니다.

## import

```python
import newch as torch                 # 텐서, 수학, 메모리 (torch)
import newch.net as nn                # 모듈 (torch.nn)
import newch.net.functional as F      # 활성화, 합성곱, 손실 (torch.nn.functional)
```

이 문서는 PyTorch 코드와 나란히 보기 쉽도록 `torch`라는 이름으로 import합니다. `import newch as nc`처럼 써도 됩니다.

## 텐서와 autograd

```python
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = (a * a).sum()        # 스칼라
b.backward()
print(a.grad.tolist())   # d(sum a^2)/da = 2a -> [[2.0, 4.0], [6.0, 8.0]]
```

정수 데이터는 `int32`, 실수 데이터는 `float32`가 됩니다. 엔진에 `int64`와 `float64`가 없기 때문입니다
(자세한 내용은 [PyTorch와 다른 점](pytorch.md)).

## 모델 정의

`nn.Module`을 상속하고 `forward`를 쓰면 됩니다. 파라미터는 속성으로 넣기만 하면 자동으로 찾습니다.

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

model = Net()
print(model)
print(model.num_parameters(), "parameters")
```

## 학습 루프

나선 3개를 분류하는 예입니다.

```python
import math

def spiral(n_per_class=100):
    xs, ys = [], []
    for c in range(3):
        for i in range(n_per_class):
            r = i / n_per_class
            t = c * 4 + 4 * r + 0.2 * math.sin(i)
            xs.append([r * math.sin(t), r * math.cos(t)])
            ys.append(c)
    return torch.tensor(xs), torch.tensor(ys)

torch.manual_seed(0)
with torch.persistent():          # 여러 스텝에 걸쳐 쓸 데이터는 영구 텐서로 만듭니다
    x, y = spiral()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for step in range(500):
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(step, round(loss.item(), 4))
    torch.step_end()
```

PyTorch와 다른 점은 두 가지입니다.

- **`torch.step_end()`:** 이 스텝에서 만든 임시 텐서(활성값, 그래디언트 계산용 값)를 한 번에 해제합니다.
  매 스텝 끝에 부르세요.
- **`torch.persistent()`:** 이 블록 안에서 만든 텐서는 `step_end()`가 지나도 남습니다. 여러 스텝에 걸쳐 쓰는
  데이터는 이렇게 만듭니다. 파라미터는 원래 영구 텐서이므로 따로 할 일이 없습니다.

자세한 규칙은 [메모리 모델](memory.md)에 있습니다.

## 평가

`no_grad()` 안에서는 그래디언트를 기록하지 않고, 다 쓴 중간값을 바로 해제합니다.

```python
model.eval()
with torch.no_grad():
    pred = model(x).argmax(1)
    accuracy = (pred == y).float().mean().item()
print(f"정확도 {accuracy:.3f}")
torch.step_end()
```

비교 연산(`==`, `>` 등)의 결과는 정수 0/1 텐서입니다(newch에는 bool dtype이 없습니다).

## 저장과 불러오기

```python
torch.save_tensors(model.state_dict(), "spiral.ncw")

restored = Net()
restored.load_state_dict(torch.load_tensors("spiral.ncw"))
```

파일 형식은 단순한 바이너리(NCW1)이고, C API의 `nc_load`로도 읽을 수 있습니다.

## 짧게 쓰고 싶다면: `fit()`

`nn.Sequential`(또는 `nn.Model`을 상속한 모델)에는 Keras식 학습 루프가 붙어 있습니다.

```python
model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 3))
model.compile(optimizer="adam", loss="cross_entropy", metrics=["accuracy"], lr=0.01)
history = model.fit(x, y, epochs=30, batch_size=32, verbose=0)
print(history.history["accuracy"][-1])
```

`fit()`은 `step_end()`를 알아서 부릅니다. 자세한 사용법은 [학습](training.md)을 보세요.
