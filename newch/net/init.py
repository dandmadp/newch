"""torch.nn.init counterpart: in-place weight initialization (``nn.init.kaiming_uniform_(w)``).

Values come from the engine RNG (``nc.manual_seed``), so they differ from
PyTorch's for the same seed while following the same distributions.
"""

from __future__ import annotations

import math

from ..tensor import Tensor, full, rand, randn

__all__ = [
    "calculate_gain", "constant_", "kaiming_normal_", "kaiming_uniform_", "normal_", "ones_", "uniform_",
    "xavier_normal_", "xavier_uniform_", "zeros_",
]


def calculate_gain(nonlinearity: str, param=None) -> float:
    """Recommended scale for weights feeding the given activation."""
    if nonlinearity in ("linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d",
                        "conv_transpose3d", "sigmoid"):
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        slope = 0.01 if param is None else float(param)
        return math.sqrt(2.0 / (1.0 + slope * slope))
    if nonlinearity == "selu":
        return 0.75
    raise ValueError(f"no gain defined for {nonlinearity!r}")


def _fill(tensor: Tensor, src: Tensor) -> Tensor:
    return tensor.copy_(src)


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    return _fill(tensor, rand(*tensor.shape, low=a, high=b, ctx=tensor.ctx))


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    return _fill(tensor, randn(*tensor.shape, std=std, ctx=tensor.ctx) + float(mean))


def constant_(tensor: Tensor, val: float) -> Tensor:
    return _fill(tensor, full(tensor.shape, float(val), ctx=tensor.ctx))


def zeros_(tensor: Tensor) -> Tensor:
    return constant_(tensor, 0.0)


def ones_(tensor: Tensor) -> Tensor:
    return constant_(tensor, 1.0)


def _fans(tensor: Tensor) -> tuple:
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError(f"fan-in/fan-out need a tensor of at least 2 dimensions, got shape {shape}")
    receptive = 1
    for d in shape[2:]:
        receptive *= d
    return shape[1] * receptive, shape[0] * receptive


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _fans(tensor)
    bound = math.sqrt(3.0) * gain * math.sqrt(2.0 / (fan_in + fan_out))
    return uniform_(tensor, -bound, bound)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _fans(tensor)
    return normal_(tensor, 0.0, gain * math.sqrt(2.0 / (fan_in + fan_out)))


def _kaiming_std(tensor: Tensor, a, mode: str, nonlinearity: str) -> float:
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode!r}")
    fan_in, fan_out = _fans(tensor)
    return calculate_gain(nonlinearity, a) / math.sqrt(fan_in if mode == "fan_in" else fan_out)


def kaiming_uniform_(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Tensor:
    bound = math.sqrt(3.0) * _kaiming_std(tensor, a, mode, nonlinearity)
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Tensor:
    return normal_(tensor, 0.0, _kaiming_std(tensor, a, mode, nonlinearity))
