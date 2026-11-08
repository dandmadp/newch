"""torch.nn.functional counterpart: ``import newch.net.functional as F``.

Every function runs as an engine (C) op; the names and arguments follow
torch.nn.functional, and arguments that are not implemented raise when set.
"""

from ..tensor import (Tensor, _unsupported, avg_pool2d, batch_norm, conv2d, cross_entropy, dropout, gelu, hardsigmoid,
                      hardswish, layer_norm, leaky_relu, linear, log_softmax, max_pool2d, mse_loss, nll_loss, relu6,
                      scaled_dot_product_attention, silu, softmax)

__all__ = [
    "avg_pool2d", "batch_norm", "conv2d", "cross_entropy", "dropout", "gelu", "hardsigmoid", "hardswish",
    "layer_norm", "leaky_relu", "linear", "log_softmax", "max_pool2d", "mse_loss", "nll_loss", "relu", "relu6",
    "scaled_dot_product_attention", "sigmoid", "silu", "softmax", "tanh",
]


def relu(input: Tensor, inplace: bool = False) -> Tensor:
    _unsupported(inplace=(inplace, False))
    return input.relu()


def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()


def tanh(input: Tensor) -> Tensor:
    return input.tanh()
