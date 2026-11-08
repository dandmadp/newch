# newch

저사양 기기에서도 **학습**까지 할 수 있는 작은 머신러닝 라이브러리입니다.

- **PyTorch와 같은 코드:** 텐서, autograd, `nn` 모듈, 옵티마이저, `DataLoader`를 PyTorch와 같은 이름과 인수로 씁니다.
  import 줄만 바꾸면 PyTorch 모델 코드가 그대로 돌아갑니다.
- **작은 엔진:** 계산은 C11 엔진이 합니다. 네이티브 라이브러리는 수백 KB이고, Python 쪽 필수 의존성은 없습니다
  (numpy는 있으면 함께 씁니다).
- **적은 메모리:** 한 학습 스텝의 임시 텐서를 한 번에 돌려주는 스텝 아레나, 역전파가 다 쓴 값을 바로 해제하는 구조,
  디스크에서 배치 단위로 읽는 `FileArray`, int8 양자화가 있습니다.
- **어디서나:** Windows, Linux, macOS와 ARM·RISC-V 싱글보드 컴퓨터에서 돌아갑니다. x86은 AVX2, ARM은 NEON 커널을 씁니다.

## 설치

```bash
pip install newch            # Windows x86-64는 바로 설치, 그 밖의 플랫폼은 설치 중에 엔진을 컴파일합니다
pip install "newch[numpy]"   # numpy 연동까지
```

소스에서 설치할 때도 시스템 C 컴파일러는 필요 없습니다. pip 패키지 `ziglang`이 컴파일러 역할을 합니다.

## 30초 예제

```python
import newch as torch
import newch.net as nn
import newch.net.functional as F

torch.manual_seed(0)
with torch.persistent():                     # 학습 내내 쓸 데이터는 영구 텐서로
    x = torch.randn(256, 2)
    y = (x[:, 0] * x[:, 1] > 0).long()       # 1, 3사분면이면 1

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for step in range(300):
    opt.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    torch.step_end()                         # newch만의 한 줄: 이 스텝의 임시 텐서를 한 번에 해제

accuracy = (model(x).argmax(1) == y).float().mean().item()
print(f"정확도 {accuracy:.2f}")
```

`torch.step_end()` 한 줄을 빼면 PyTorch 코드와 같습니다. 이 줄이 하는 일은 [메모리 모델](memory.md)에서 설명합니다.

## 다음으로 읽을 것

| 하고 싶은 일 | 페이지 |
|---|---|
| 처음 써 본다 | [시작하기](quickstart.md) |
| 텐서 연산, 인덱싱, numpy 연동 | [텐서](tensors.md) |
| `step_end()`, 영구 텐서, 메모리 측정 | [메모리 모델](memory.md) |
| CNN, RNN, 트랜스포머 만들기 | [모델 만들기](models.md) |
| 옵티마이저, 스케줄러, `fit()`, 체크포인트 | [학습](training.md) |
| 큰 데이터셋, 디스크 스트리밍 | [데이터](data.md) |
| 저장, 양자화, 다른 기기로 옮기기, C API | [배포와 추론](deploy.md) |
| PyTorch 코드를 옮길 때 주의할 점 | [PyTorch와 다른 점](pytorch.md) |
