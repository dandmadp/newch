# 모델 만들기

모듈은 `newch.net`(`torch.nn`에 해당)에 있습니다. 이름, 인수 순서, 기본값, 파라미터 이름과 모양, 초기화가 PyTorch와
같아서 PyTorch의 `state_dict`를 numpy로 바꾸면 그대로 불러올 수 있습니다.

```python
import newch as torch
import newch.net as nn
import newch.net.functional as F
```

## Module

```python
class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(sizes, sizes[1:])])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.dropout(F.gelu(x))
        return x

model = MLP([4, 64, 64, 3])
print([name for name, _ in model.named_parameters()])
model.summary()
```

- 파라미터, 버퍼, 하위 모듈은 속성으로 넣기만 하면 찾습니다 (`parameters()`, `named_modules()`, `state_dict()`).
- `train()` / `eval()`은 Dropout과 BatchNorm의 동작을 바꾸고 하위 모듈에 전파됩니다.
- 직접 파라미터를 만들 때는 `torch.parameter(텐서)`, 버퍼는 `self.register_buffer(이름, 텐서)`를 씁니다.
- 컨테이너: `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`, `nn.Identity`.

## 레이어 목록

| 분류 | 모듈 | 함수형 (`F`) |
|---|---|---|
| 선형 | `Linear`, `Embedding` | `linear` |
| 합성곱 | `Conv2d` (groups, depthwise, dilation, `padding="same"`) | `conv2d` |
| 풀링 | `MaxPool2d`, `AvgPool2d` | `max_pool2d`, `avg_pool2d` |
| 정규화 | `BatchNorm1d`, `BatchNorm2d`, `LayerNorm` | `batch_norm`, `layer_norm` |
| 순환 | `RNN`, `LSTM`, `GRU`, `RNNCell`, `LSTMCell`, `GRUCell` | |
| 트랜스포머 | `MultiheadAttention`, `TransformerEncoderLayer`, `TransformerEncoder` | `scaled_dot_product_attention` |
| 활성화 | `ReLU`, `LeakyReLU`, `ReLU6`, `Hardsigmoid`, `Hardswish`, `SiLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax` | 같은 이름의 소문자 함수 |
| 기타 | `Dropout`, `Flatten` | `dropout` |
| 손실 | `CrossEntropyLoss`, `NLLLoss`, `MSELoss` | `cross_entropy`, `nll_loss`, `mse_loss` |

## CNN

```python
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding="same")
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):                        # x [N, 1, 28, 28]
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.relu(self.conv2(x))
        return self.fc(torch.flatten(x, 1))

print(SmallCNN()(torch.randn(2, 1, 28, 28)).shape)
```

**MobileNet식 depthwise-separable 블록:** `groups=in_channels`이면 채널별(depthwise) 합성곱입니다. newch는 이 경우
im2col을 거치지 않는 전용 커널을 써서, 같은 크기의 일반 합성곱보다 몇 배 빠르고 메모리도 적게 씁니다.

```python
def separable(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU6(),
        nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU6(),
    )

block = separable(16, 32, stride=2)
print(block(torch.randn(2, 16, 32, 32)).shape)
```

합성곱과 풀링은 배치 없는 `[C, H, W]` 입력도 받습니다.

## 순환 신경망

`RNN`, `LSTM`, `GRU`는 다층, 양방향, `batch_first`, 층 사이 dropout을 지원합니다. 반환값도 PyTorch와 같습니다.

```python
lstm = nn.LSTM(input_size=8, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
seq = torch.randn(4, 20, 8)                       # [batch, time, feature]
out, (h_n, c_n) = lstm(seq)
print(out.shape, h_n.shape)                       # [4, 20, 64], [4, 4, 32]

class SequenceClassifier(nn.Module):
    def __init__(self, vocab, classes):
        super().__init__()
        self.embed = nn.Embedding(vocab, 32)
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.head = nn.Linear(64, classes)

    def forward(self, tokens):                    # tokens int32 [N, T]
        _, h = self.gru(self.embed(tokens))       # Embedding은 어떤 모양의 인덱스든 받습니다
        return self.head(h[-1])

clf = SequenceClassifier(100, 5)
print(clf(torch.tensor([[1, 5, 9, 2], [3, 3, 7, 0]])).shape)
```

LSTM의 `proj_size`는 지원하지 않습니다(에러).

## 트랜스포머

```python
layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
encoder = nn.TransformerEncoder(layer, num_layers=2)

T = 10
causal = torch.tensor(list(range(T))) > torch.tensor(list(range(T))).reshape(T, 1)   # 1 = 미래 위치는 가림
h = encoder(torch.randn(2, T, 64), mask=causal, is_causal=True)
print(h.shape)
```

- bool dtype이 없으므로 마스크에는 비교 결과(정수 0/1)를 씁니다. 0과 1의 의미는 각 함수에서 PyTorch의
  False/True와 같습니다. 예를 들어 `nn.MultiheadAttention`의 `attn_mask`와 `key_padding_mask`에서 1은 "가림"이고,
  `F.scaled_dot_product_attention`의 정수 `attn_mask`에서 1은 "볼 수 있음"입니다. float 마스크는 점수에 더해집니다.
- 디코더(`TransformerDecoder`, `nn.Transformer`)와 KV 캐시는 아직 없습니다.

문자 단위 언어 모델 전체 예제는
[examples/tiny_transformer.py](https://github.com/dandmadp/newch/blob/main/examples/tiny_transformer.py)에 있습니다.

## 가중치 초기화

기본 초기화는 PyTorch와 같습니다. 바꾸려면 `nn.init`을 씁니다.

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)
```

`uniform_`, `normal_`, `constant_`, `zeros_`, `ones_`, `xavier_uniform_`, `xavier_normal_`, `kaiming_uniform_`,
`kaiming_normal_`, `calculate_gain`이 있습니다. 분포와 공식은 PyTorch와 같지만 난수는 newch의 생성기에서 나오므로,
같은 시드라도 값은 PyTorch와 다릅니다.

## PyTorch 가중치 가져오기

```py
model.load_state_dict({k: v.numpy() for k, v in torch_model.state_dict().items()})
```

레이어 이름과 텐서 레이아웃이 같아서, 같은 구조로 만든 newch 모델에 그대로 들어갑니다.

```python
torch.step_end()
```
