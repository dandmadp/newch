"""Modules (PyTorch style) and Model/Sequential with compile/fit (Keras style)."""

from __future__ import annotations

import math
from collections import OrderedDict, namedtuple

from . import _lib as L
from ._lib import lib
from .tensor import (Context, Tensor, _is_numpy, _to_buffer, avg_pool2d, batch_norm, conv2d, default_context, dropout,
                     full, int32, linear, load_tensors, max_pool2d, parameter, persistent, rand, randn, save_tensors,
                     tensor, zeros)

IncompatibleKeys = namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])


class Module:
    """Base class. Parameters, buffers and sub-modules are found through
    attributes, so a custom model only has to assign them in ``__init__`` and
    define ``forward``. Parameters are persistent tensors that require grad;
    buffers are the other persistent tensors (e.g. BatchNorm running stats)."""

    training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.forward")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ---- traversal -------------------------------------------------------
    def _items(self):
        for name, v in vars(self).items():
            if name.startswith("_"):
                continue
            if isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    yield f"{name}.{i}", item
            else:
                yield name, v

    def children(self) -> list["Module"]:
        return [v for _, v in self._items() if isinstance(v, Module)]

    def modules(self):
        """This module and every sub-module, depth first."""
        yield self
        for c in self.children():
            yield from c.modules()

    def _collect(self, prefix, out, seen, want):
        for name, v in self._items():
            full_name = f"{prefix}{name}"
            if isinstance(v, Tensor) and v.is_persistent and v._p not in seen and want(v):
                seen.add(v._p)
                out.append((full_name, v))
            elif isinstance(v, Module):
                v._collect(full_name + ".", out, seen, want)

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Tensor]]:
        out: list = []
        self._collect(prefix, out, set(), lambda t: t.requires_grad)
        return out

    def named_buffers(self, prefix: str = "") -> list[tuple[str, Tensor]]:
        out: list = []
        self._collect(prefix, out, set(), lambda t: not t.requires_grad)
        return out

    def parameters(self) -> list[Tensor]:
        return [p for _, p in self.named_parameters()]

    def buffers(self) -> list[Tensor]:
        return [b for _, b in self.named_buffers()]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---- state -----------------------------------------------------------
    def state_dict(self) -> "OrderedDict[str, Tensor]":
        """Parameters and buffers by dotted name, in definition order (as in PyTorch)."""
        out: list = []
        self._collect("", out, set(), lambda t: True)
        return OrderedDict(out)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> IncompatibleKeys:
        """Copies values in by name. Values may be Tensors, numpy arrays or
        nested lists, so ``{k: v.numpy() for k, v in torch_model.state_dict().items()}``
        loads into an equivalent newch model."""
        own = self.state_dict()
        missing = [k for k in own if k not in state_dict]
        unexpected = [k for k in state_dict if k not in own]
        if strict and (missing or unexpected):
            raise RuntimeError(f"load_state_dict: missing keys {missing}, unexpected keys {unexpected}")
        for name, t in own.items():
            if name not in state_dict:
                continue
            src = state_dict[name]
            shape = tuple(src.shape) if isinstance(src, Tensor) or _is_numpy(src) else _to_buffer(src, t.dtype)[2]
            if tuple(shape) != t.shape:
                raise RuntimeError(f"load_state_dict: size mismatch for '{name}': "
                                   f"checkpoint {tuple(shape)}, model {t.shape}")
            t.copy_(src)
        return IncompatibleKeys(missing, unexpected)

    def save_weights(self, path) -> None:
        """Writes ``state_dict()`` to an NCW1 file (Keras' save_weights)."""
        save_tensors(self.state_dict(), path)

    def load_weights(self, path) -> None:
        """Loads a file written by save_weights (strict: names and shapes must match)."""
        self.load_state_dict(load_tensors(path, ctx=self._context()))

    def _context(self) -> Context:
        sd = self.state_dict()
        return next(iter(sd.values())).ctx if sd else default_context()

    # ---- modes -----------------------------------------------------------
    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for c in self.children():
            c.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    # ---- display ---------------------------------------------------------
    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        kids = [(n, v) for n, v in self._items() if isinstance(v, Module)]
        head = f"{type(self).__name__}({self.extra_repr()}"
        if not kids:
            return head + ")"
        lines = [head]
        for n, v in kids:
            lines.append("  (" + n + "): " + repr(v).replace("\n", "\n  "))
        return "\n".join(lines) + "\n)"

    def summary(self) -> None:
        """Keras-style table of layers and parameter counts."""
        rows = [(n, type(m).__name__, m.num_parameters()) for n, m in self._items() if isinstance(m, Module)]
        rows = rows or [("", type(self).__name__, self.num_parameters())]
        w = max(len(f"{n} ({t})") for n, t, _ in rows) + 2
        print(f"{'Layer':<{w}}{'Params':>10}")
        print("-" * (w + 10))
        for n, t, c in rows:
            print(f"{f'{n} ({t})':<{w}}{c:>10,}")
        print("-" * (w + 10))
        print(f"{'Total':<{w}}{self.num_parameters():>10,}")


def _uniform_param(shape, bound: float, ctx: Context) -> Tensor:
    return parameter(rand(*shape, low=-bound, high=bound, ctx=ctx), ctx=ctx)


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------


class Linear(Module):
    """y = x @ weight.T + bias with weight [out, in], bias [out] (same as torch.nn.Linear)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, ctx: Context | None = None):
        ctx = ctx or default_context()
        self.in_features, self.out_features = in_features, out_features
        bound = 1.0 / math.sqrt(in_features)  # PyTorch's default init
        self.weight = _uniform_param((out_features, in_features), bound, ctx)
        self.bias = _uniform_param((out_features,), bound, ctx) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Conv2d(Module):
    """NCHW convolution, weight [out, in, kh, kw] (torch.nn.Conv2d without dilation/groups)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, bias: bool = True,
                 ctx: Context | None = None):
        from .tensor import _pair

        ctx = ctx or default_context()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride, self.padding = _pair(stride, "stride"), _pair(padding, "padding")
        bound = 1.0 / math.sqrt(in_channels * self.kernel_size[0] * self.kernel_size[1])
        self.weight = _uniform_param((out_channels, in_channels, *self.kernel_size), bound, ctx)
        self.bias = _uniform_param((out_channels,), bound, ctx) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}")


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, x): return max_pool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride or self.kernel_size}, padding={self.padding}"


class AvgPool2d(MaxPool2d):
    """Padded cells count toward the average (torch's count_include_pad=True)."""

    def forward(self, x): return avg_pool2d(x, self.kernel_size, self.stride, self.padding)


class Dropout(Module):
    """Zeroes inputs with probability p during training; identity in eval mode."""

    def __init__(self, p: float = 0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = p

    def forward(self, x): return dropout(x, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class _BatchNorm(Module):
    _input_ndims: tuple = ()

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True,
                 ctx: Context | None = None):
        ctx = ctx or default_context()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        with persistent(ctx):
            self.weight = parameter(full((num_features,), 1.0, ctx=ctx), ctx=ctx) if affine else None
            self.bias = parameter((num_features,), ctx=ctx) if affine else None
            self.running_mean = zeros(num_features, ctx=ctx)
            self.running_var = full((num_features,), 1.0, ctx=ctx)
            self.num_batches_tracked = zeros(dtype=int32, ctx=ctx)
        self._ctx = ctx

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim not in self._input_ndims:
            raise ValueError(f"{type(self).__name__} expects {' or '.join(map(str, self._input_ndims))}-D input, "
                             f"got {x.ndim}-D")
        if self.training:
            self.num_batches_tracked.copy_([self.num_batches_tracked.item() + 1])
        return batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training,
                          self.momentum, self.eps)

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.weight is not None}"


class BatchNorm1d(_BatchNorm):
    """Normalizes [N, C] or [N, C, L] over the batch (and length)."""
    _input_ndims = (2, 3)


class BatchNorm2d(_BatchNorm):
    """Normalizes [N, C, H, W] per channel."""
    _input_ndims = (4,)


class Embedding(Module):
    """Lookup table: int32 indices [B] -> rows [B, embedding_dim]; weight ~ N(0, 1)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, ctx: Context | None = None):
        ctx = ctx or default_context()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.weight = parameter(randn(num_embeddings, embedding_dim, ctx=ctx), ctx=ctx)

    def forward(self, idx) -> Tensor:
        return self.weight.index_select(0, idx)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"


class ReLU(Module):
    def forward(self, x): return x.relu()


class Sigmoid(Module):
    def forward(self, x): return x.sigmoid()


class Tanh(Module):
    def forward(self, x): return x.tanh()


class Softmax(Module):
    def __init__(self, dim: int = -1):
        self.dim = dim

    def forward(self, x): return x.softmax(self.dim)


class LogSoftmax(Module):
    def __init__(self, dim: int = -1):
        self.dim = dim

    def forward(self, x): return x.log_softmax(self.dim)


class Flatten(Module):
    """[B, d1, d2, ...] -> [B, d1*d2*...] with the default start_dim=1 (as in torch.nn.Flatten)."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        self.start_dim, self.end_dim = start_dim, end_dim

    def forward(self, x): return x.flatten(self.start_dim, self.end_dim)


# --------------------------------------------------------------------------
# int8 dynamic quantization (inference)
# --------------------------------------------------------------------------


class QuantizedLinear(Module):
    """Linear with int8 weights (per output row scales) and activations
    quantized at runtime. Inference only; ~4x smaller weights."""

    def __init__(self, weight_int8: Tensor, scale: Tensor, bias: Tensor | None, in_features: int,
                 out_features: int):
        self.in_features, self.out_features = in_features, out_features
        self.weight, self.scale, self.bias = weight_int8, scale, bias

    @classmethod
    def from_float(cls, lin: Linear) -> "QuantizedLinear":
        ctx = lin.weight.ctx
        scale_p = L.ptr_array([None])
        with persistent(ctx):
            q = ctx.wrap(lib.nc_quantize_rows(lin.weight.ptr, scale_p))
            scale = ctx.wrap(scale_p[0])
            bias = tensor(lin.bias, ctx=ctx) if lin.bias is not None else None  # copy without grad
        return cls(q, scale, bias, lin.in_features, lin.out_features)

    def forward(self, x: Tensor) -> Tensor:
        return x._w(lib.nc_linear_int8(x.ptr, self.weight.ptr, self.scale.ptr,
                                       self.bias.ptr if self.bias is not None else None))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, dtype=int8"


def quantize_(model: Module) -> Module:
    """Replaces every Linear in ``model`` with a QuantizedLinear, in place
    (the trailing underscore marks in-place, as in PyTorch). Returns model."""

    def convert(v):
        return QuantizedLinear.from_float(v) if isinstance(v, Linear) else v

    for m in list(model.modules()):
        for name, v in list(vars(m).items()):
            if isinstance(v, Linear):
                setattr(m, name, convert(v))
            elif isinstance(v, list) and any(isinstance(i, Linear) for i in v):
                setattr(m, name, [convert(i) for i in v])
            elif isinstance(v, tuple) and any(isinstance(i, Linear) for i in v):
                setattr(m, name, tuple(convert(i) for i in v))
    model.eval()
    return model


# --------------------------------------------------------------------------
# Model (Keras-style training API on any Module)
# --------------------------------------------------------------------------


class Model(Module):
    """A Module with ``compile`` / ``fit`` / ``evaluate`` / ``predict``.

    Subclass it instead of Module to get the high-level training loop for a
    custom model; Sequential already is one.
    """

    _compiled = None
    stop_training = False

    def compile(self, optimizer="adam", loss="cross_entropy", metrics=None, lr: float | None = None):
        from .training import compile_model

        compile_model(self, optimizer, loss, metrics, lr)
        return self

    def fit(self, x, y=None, epochs: int = 1, batch_size: int = 32, validation_data=None,
            validation_split: float = 0.0, shuffle: bool = True, callbacks=None, verbose: int = 1, seed=None):
        """Trains on arrays (x, y), or streams from a DataLoader / iterable of
        (x, y) batches passed as x with y=None (as Keras does for datasets)."""
        from .training import fit

        return fit(self, x, y, epochs, batch_size, validation_data, validation_split, shuffle, callbacks, verbose,
                   seed)

    def evaluate(self, x, y=None, batch_size: int = 256, return_dict: bool = False):
        """Loss (and metrics) on x, y. Like Keras: a float when there are no
        metrics, else ``[loss, accuracy]``; a dict with ``return_dict=True``."""
        from .training import evaluate

        logs = evaluate(self, x, y, batch_size)
        if return_dict:
            return logs
        values = list(logs.values())
        return values[0] if len(values) == 1 else values

    def predict(self, x, batch_size: int = 256):
        from .training import predict

        return predict(self, x, batch_size)


class Sequential(Model):
    """``Sequential([Linear(2, 16), ReLU(), Linear(16, 3)])`` or ``Sequential(a, b, c)``."""

    def __init__(self, *layers):
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = tuple(layers[0])
        self.layers = list(layers)

    def add(self, layer: Module) -> "Sequential":
        self.layers.append(layer)
        return self

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self) -> int:
        return len(self.layers)

    def _items(self):
        for i, layer in enumerate(self.layers):
            yield str(i), layer
