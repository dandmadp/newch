# 학습

학습은 두 가지 방식으로 할 수 있습니다. PyTorch식으로 루프를 직접 쓰거나, Keras식 `fit()`에 맡기는 것입니다.
옵티마이저, 스케줄러, 체크포인트는 두 방식이 같은 것을 씁니다.

```python
import newch as torch
import newch.net as nn
import newch.net.functional as F

torch.manual_seed(0)
with torch.persistent():
    x = torch.randn(512, 8)
    y = (x[:, :4].sum(1) > 0).long()
```

## 학습 루프 (PyTorch식)

```python
from newch.data import DataLoader, TensorDataset

model = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 2))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)

for epoch in range(10):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = loss.item()        # step_end() 전에 값을 꺼냅니다
        torch.step_end()
    scheduler.step()
print("lr", scheduler.get_last_lr(), "loss", round(last_loss, 4))
```

## 옵티마이저

| 클래스 | 상태 메모리 | 메모 |
|---|---|---|
| `optim.SGD(params, lr, momentum, dampening, weight_decay, nesterov)` | 0x (momentum이면 1x) | `dampening≠0`, `nesterov`는 에러 |
| `optim.Adam(params, lr, betas, eps, weight_decay, amsgrad)` | 2x | weight_decay는 L2 항. `amsgrad`는 에러 |
| `optim.AdamW(...)` | 2x | 분리된 weight decay, 기본값 0.01 |
| `optim.RMSprop(params, lr, alpha, eps, weight_decay, momentum, centered)` | 1x~2x | `centered`는 에러 |

인수 검사(음수 lr, `[0, 1)` 밖의 beta 등)와 갱신식이 PyTorch와 같습니다. 상태는 엔진의 영구 메모리에 있습니다.

**메모리가 작을 때 큰 배치처럼 학습:** `gradient_accumulation_steps=n`이면 `step()`을 n번 부를 때마다 한 번, 그동안
모은 그래디언트의 평균으로 갱신합니다. 배치 16으로 n=4이면 배치 64와 같은 갱신을 배치 16의 활성 메모리로 합니다.

```python
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, gradient_accumulation_steps=4)
```

## 학습률 스케줄러

`optim.lr_scheduler`에 `StepLR`, `ExponentialLR`, `CosineAnnealingLR`, `LambdaLR`, `ReduceLROnPlateau`가 있습니다.
파라미터 그룹은 하나이므로 `get_last_lr()`은 원소 하나짜리 리스트입니다.

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
for val_loss in [1.0, 0.9, 0.95, 0.96, 0.97]:
    plateau.step(val_loss)
print(plateau.get_last_lr())
```

## fit() (Keras식)

`nn.Sequential`과 `nn.Model`을 상속한 모델에는 `compile` / `fit` / `evaluate` / `predict`가 있습니다.

```python
from newch import callbacks

model = nn.Sequential(nn.Linear(8, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 2))
model.compile(optimizer="adam", loss="cross_entropy", metrics=["accuracy"], lr=3e-3)
history = model.fit(
    x, y, batch_size=32, epochs=20, validation_split=0.2, verbose=0,
    callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
)
loss, acc = model.evaluate(x, y, verbose=0)
probs = model.predict(x[:5])
print(len(history.history["loss"]), round(acc, 3), len(probs))
```

- `optimizer`는 이름(`"adam"`, `"adamw"`, `"sgd"`, `"rmsprop"`: 해당 클래스를 기본값으로 만듦)이나 옵티마이저 객체입니다.
- `loss`는 `"cross_entropy"`, `"mse"`, 또는 `(출력, 타깃) -> 손실` 함수입니다.
- `x`, `y`는 텐서, numpy 배열, 리스트, `FileArray`입니다. `DataLoader`(또는 `(x, y)` 배치를 내는 반복자)를 `x`로 넘기면
  배치 단위로 스트리밍합니다.
- 콜백: `EarlyStopping`, `ModelCheckpoint`(가중치만 저장하므로 `save_weights_only=True` 필요), `LambdaCallback`,
  `History`. 스케줄러는 `LambdaCallback(on_epoch_end=lambda epoch, logs: sched.step(logs["val_loss"]))`처럼 연결합니다.

## 체크포인트와 학습 재개

`save_tensors`는 중첩된 dict/list/tuple과 숫자·문자열을 함께 저장할 수 있어서, 체크포인트 전체가 파일 하나입니다.
옵티마이저와 스케줄러의 `state_dict()`는 PyTorch와 같은 구조입니다.

```python
model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5)

torch.save_tensors({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": 7}, "ckpt.ncw")

ckpt = torch.load_tensors("ckpt.ncw")
model.load_state_dict(ckpt["model"])
opt.load_state_dict(ckpt["optimizer"])
sched.load_state_dict(ckpt["scheduler"])
print("resume from epoch", ckpt["epoch"] + 1)
```

끊지 않고 학습한 것과 저장 후 재개한 것의 결과가 비트 단위로 같다는 것을 테스트로 확인합니다.
손상되었거나 조작된 파일은 프로세스를 죽이지 않고 `NcError`를 냅니다.

```python
torch.step_end()
```
