# 데이터

`newch.data`는 `torch.utils.data`와 같은 `Dataset`, `TensorDataset`, `DataLoader`에 더해, RAM보다 큰 데이터를
디스크에서 배치 단위로 읽는 `FileArray`를 제공합니다.

```python
import numpy as np
import newch as torch
from newch.data import DataLoader, Dataset, FileArray, TensorDataset
```

## TensorDataset과 DataLoader

```python
features = np.random.rand(1000, 16).astype(np.float32)
labels = np.random.randint(0, 4, 1000)

ds = TensorDataset(features, labels)          # numpy, 리스트, 텐서, FileArray
loader = DataLoader(ds, batch_size=128, shuffle=True, drop_last=False)
print(len(ds), len(loader))
for xb, yb in loader:
    print(xb.shape, xb.dtype, yb.shape, yb.dtype)
    torch.step_end()
    break
```

- 배치는 이번 스텝의 임시 텐서입니다. 다음 `step_end()`까지 씁니다.
- `ds[i]`는 행 하나(숫자나 리스트)를 돌려주고, 범위 밖 인덱스는 `IndexError`입니다.
- `shuffle`의 난수는 `generator=random.Random(seed)`로 고정할 수 있습니다.
- `sampler`, `batch_sampler`, `num_workers≠0`, `pin_memory`는 지원하지 않습니다(에러).

## 직접 만든 Dataset

`__len__`과 `__getitem__`만 쓰면 됩니다. 샘플이 `(x, y)` 튜플이면 필드별로 묶고, 텐서 샘플은 `stack`합니다.

```python
class Squares(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, i):
        return torch.tensor([float(i)]), i * i

for xb, yb in DataLoader(Squares(), batch_size=10):
    print(xb.shape, yb.tolist()[:3])
    torch.step_end()
    break
```

`collate_fn`으로 묶는 방법을 직접 정할 수도 있습니다.

## RAM보다 큰 데이터: FileArray

샘플을 행 단위 바이너리 파일로 저장하고 `FileArray`로 열면, 배치를 만들 때 필요한 행만 mmap으로 읽습니다.
데이터셋 크기와 상관없이 메모리는 배치 하나 정도만 씁니다.

```python
def rows(n):                       # 메모리에 전부 올리지 않고 한 행씩 만듭니다
    rng = np.random.default_rng(0)
    for _ in range(n):
        yield rng.integers(0, 256, (1, 8, 8), dtype=np.uint8)

x = FileArray.create("x.u8", rows(5000), row_shape=(1, 8, 8), dtype=torch.uint8)
y = FileArray.create("y.u8", (np.uint8(i % 10) for i in range(5000)), row_shape=(), dtype=torch.uint8)

loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
for xb, yb in loader:
    xb = xb.float() / 255          # uint8로 저장하면 float32의 1/4 크기, 배치마다 변환
    print(xb.shape, xb.dtype, yb.dtype)
    torch.step_end()
    break
x.close(); y.close()
```

- 이미 있는 파일은 `FileArray(path, shape=(rows, ...), dtype=...)`로 엽니다. `offset`으로 헤더를 건너뛸 수 있습니다.
- `fit()`에 `FileArray`를 넘기면 자동으로 스트리밍합니다.
- 분류 손실(`cross_entropy`, `nll_loss`)은 uint8·int8 라벨을 받아서 int32로 넓혀 씁니다.

```python
torch.step_end()
```
