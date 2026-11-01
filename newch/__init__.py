"""newch: a small machine-learning library for low-spec machines.

PyTorch-style tensors and autograd, Keras-style ``compile``/``fit``, backed
by a portable C engine. No Python dependencies (numpy is used if present).

    import newch as nc

    model = nc.Sequential([nc.net.Linear(2, 32), nc.net.ReLU(), nc.net.Linear(32, 3)])
    model.compile(optimizer="adam", loss="cross_entropy", metrics=["accuracy"])
    model.fit(x, y, epochs=20, batch_size=32)
"""

from . import callbacks, data, net, optim
from .net import Model, Module, Sequential, quantize_
from .tensor import (Context, NcError, Tensor, avg_pool2d, batch_norm, conv2d, cross_entropy, default_context,
                     dropout, exp, float32, full, int8, int32, linear, load_tensors, log, log_softmax, manual_seed,
                     matmul, max_pool2d, mem_stats, mse_loss, no_grad, ones, parameter, persistent, rand, randn,
                     release, relu, save_tensors, set_default_context, sigmoid, softmax, sqrt, step_end, tanh, tensor,
                     zeros)

__version__ = "0.2.0"

__all__ = [
    "Context", "Model", "Module", "NcError", "Sequential", "Tensor", "avg_pool2d", "batch_norm", "callbacks",
    "conv2d", "cross_entropy", "data", "default_context", "dropout", "exp", "float32", "full", "int8", "int32",
    "linear", "load_tensors", "log", "log_softmax", "manual_seed", "matmul", "max_pool2d", "mem_stats", "mse_loss",
    "net", "no_grad", "ones", "optim", "parameter", "persistent", "quantize_", "rand", "randn", "release", "relu",
    "save_tensors", "set_default_context", "sigmoid", "softmax", "sqrt", "step_end", "tanh", "tensor", "zeros",
]
