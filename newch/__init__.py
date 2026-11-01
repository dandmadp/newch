"""newch: a small machine-learning library for low-spec machines.

PyTorch-style tensors, autograd and modules on a portable C engine, plus a
compact ``fit()`` loop. No Python dependencies (numpy is used if present).

    import newch as nc
    import newch.net as nn
    import newch.net.functional as F

    model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 3))
    model.compile(optimizer="adam", loss="cross_entropy", metrics=["accuracy"])
    model.fit(x, y, epochs=20, batch_size=32)

``nc.*`` holds tensors (creation, math, shape) and memory control; neural
network functions (activations, layers, losses) are in ``newch.net.functional``.
"""

from . import callbacks, data, net, optim
from .net import Model, Module, Sequential, quantize_
from .tensor import (Context, NcError, Tensor, argmax, cat, default_context, exp, flatten, float32, full, int8, int32,
                     load_tensors, log, manual_seed, matmul, mem_stats, no_grad, ones, parameter, persistent, rand,
                     randn, release, save_tensors, set_default_context, sigmoid, sqrt, stack, step_end, tanh, tensor,
                     zeros)

__version__ = "0.3.1"

__all__ = [
    "Context", "Model", "Module", "NcError", "Sequential", "Tensor", "argmax", "callbacks", "cat", "data",
    "default_context", "exp", "flatten", "float32", "full", "int8", "int32", "load_tensors", "log", "manual_seed",
    "matmul", "mem_stats", "net", "no_grad", "ones", "optim", "parameter", "persistent", "quantize_", "rand",
    "randn", "release", "save_tensors", "set_default_context", "sigmoid", "sqrt", "stack", "step_end", "tanh",
    "tensor", "zeros",
]
