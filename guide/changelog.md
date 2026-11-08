# 변경 기록

## 0.4.0

0.3.0 이후 처음 내는 버전입니다. PyTorch 코드를 그대로 돌릴 수 있는 범위가 크게 넓어졌고, 0.3.0 코드 일부는 바꿔야
합니다(아래 "바뀐 API").

### 새 기능

- **PyTorch식 모델 코드:** `newch.net`이 패키지가 되어 `newch.net.functional`(F)과 `newch.net.init`이 생겼습니다.
  `ModuleList`, `ModuleDict`, `Identity`, 손실 모듈, `nll_loss`, `cat`, `stack`, 비교 연산, dtype 변환이 추가되었습니다.
- **트랜스포머:** 배치 행렬곱, `LayerNorm`, `masked_fill`, `F.scaled_dot_product_attention`, `MultiheadAttention`,
  `TransformerEncoderLayer`, `TransformerEncoder`.
- **순환 신경망:** `RNN`, `LSTM`, `GRU`(다층, 양방향, `batch_first`)와 `RNNCell`, `LSTMCell`, `GRUCell`.
- **합성곱:** `groups`(depthwise 전용 커널), `dilation`, `padding="same"`/`"valid"`, 배치 없는 `[C, H, W]` 입력.
- **인덱싱:** 모든 축의 정수·슬라이스·`None`·`...`(복사 없는 뷰)와 정수 인덱스 리스트, 음수 인덱스.
- **텐서:** 모든 축 `softmax`, 여러 축 `sum`/`mean`/`amax`, 정수→실수 승격(`a / 2`), `uint8` dtype, `x / 0`,
  `int(t)`, `x.device`, 생성 함수의 `device=`·`requires_grad=`, numpy와 섞은 연산이 Tensor를 유지.
- **손실:** `cross_entropy`의 확률 타깃, `[N, C, d1, ...]` 입력, 배치 없는 입력.
- **학습 재개:** 옵티마이저·스케줄러 `state_dict()`(PyTorch 구조)와 중첩 체크포인트를 한 파일에 저장하는 `save_tensors`.
- **성능:** 역전파 중 메모리 해제(학습 스텝 메모리 약 절반), 추론 중 즉시 해제, 원소별 연산·합성곱 변환·풀링 멀티스레드,
  Python 호출 오버헤드 감소. 결과는 여전히 스레드 수와 무관하게 비트 단위로 같습니다.
- **사용법 안내 사이트**(이 사이트). 예제 코드는 CI에서 실제로 실행해 검사합니다.

### 바뀐 API

| 0.3.0 | 0.4.0 |
|---|---|
| `nc.relu`, `nc.conv2d`, `nc.cross_entropy` 등 신경망 함수 | `import newch.net.functional as F` 후 `F.relu` 등 |
| `model.save_weights(p)` / `load_weights(p)` | `nc.save_tensors(model.state_dict(), p)` / `model.load_state_dict(nc.load_tensors(p))` |
| `compile("adam")`이 Keras 기본값 | `nc.optim.Adam`을 그 기본값으로 만듦 |
| `nc.optim._KerasRMSprop`, `Optimizer.learning_rate` | `nc.optim.RMSprop`, `Optimizer.lr` |
| 콜백 `LearningRateScheduler`, `ReduceLROnPlateau` | `nc.optim.lr_scheduler.*` |
| `compile(loss="mean_squared_error")` | `loss="mse"` |
| `newch.ml` (scikit-learn식) | 삭제 |
| `nc.rand(..., low, high)`, `nc.randn(..., std)` | `nc.rand(...) * (high - low) + low`, `nc.randn(...) * std` |
| `tensor.zero_grad()` | `model.zero_grad()` / `optimizer.zero_grad()` |
| 1차원 `FileArray`의 `fa[i]`가 `[값]` | 값 (numpy와 같음) |

### 고친 버그

- **보안:** 데이터 크기가 모양과 맞지 않는 체크포인트를 열면 프로세스가 죽거나(Python) 힙 버퍼를 넘어 쓸 수 있던(C
  `nc_load`) 문제. 0.3.0을 쓰고 있다면 신뢰할 수 없는 `.ncw` 파일을 열지 마세요.
- 원소 수가 int64를 넘는 모양이 작은 버퍼로 할당되던 문제.
- `TensorDataset`/`FileArray`의 범위 밖 인덱스가 첫 행으로 돌아가 `for x in ds`가 끝나지 않던 문제.
- `ndarray + tensor`가 numpy 배열을 돌려주어 autograd가 끊기던 문제.
- `tensor([np.float32(1.5)])`가 int32가 되던 문제, bool 리스트를 인덱스로 읽던 문제.
- 빈 축의 평균이 NaN이 아니라 0이던 문제.
- `Adam(betas=(0, ...))`, `RMSprop(alpha=0)`의 0이 기본값으로 바뀌던 문제.
- `EarlyStopping(restore_best_weights=True)`가 BatchNorm 통계를 되돌리지 않던 문제.
- 버린 `Context`가 엔진 메모리와 워커 스레드를 해제하지 않던 문제.
- 벡터 `exp`가 `-inf`에서 0이 아니던 문제(attention 마스크), 88.4 근처에서 `inf`가 되던 문제.
- 그 밖에 옵티마이저 인수 검사, `cat` 입력이 많을 때의 속도, 잘못된 dtype의 에러 메시지 등.

## 0.3.0

PyPI에 처음 올린 버전입니다.
