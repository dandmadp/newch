"""Modules (PyTorch style) and Model/Sequential with compile/fit (Keras style)."""

from __future__ import annotations

import math
from collections import OrderedDict, namedtuple

from . import _lib as L
from ._lib import lib
from .tensor import (Context, Tensor, _is_numpy, _pair, _to_buffer, _unsupported, _window, avg_pool2d, batch_norm,
                     conv2d, default_context, dropout, full, gelu, hardsigmoid, hardswish, int32, leaky_relu, linear,
                     load_tensors, log_softmax, max_pool2d, parameter, persistent, rand, randn, relu6, save_tensors,
                     silu, softmax, tensor, zeros)

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

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Resets the grads; with set_to_none (default, as in PyTorch) they read as None until the next backward."""
        for p in self.parameters():
            (lib.nc_clear_grad if set_to_none else lib.nc_zero_grad)(p.ptr)

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

    def __init__(self, in_features: int, out_features: int, bias: bool = True, *, ctx: Context | None = None):
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
    """NCHW convolution, weight [out, in, kh, kw] (torch.nn.Conv2d; dilation and groups must be 1)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1,
                 groups: int = 1, bias: bool = True, padding_mode: str = "zeros", *, ctx: Context | None = None):
        _unsupported(dilation=(_pair(dilation, "dilation"), (1, 1)), groups=(groups, 1),
                     padding_mode=(padding_mode, "zeros"))
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


class _Pool2d(Module):
    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride or self.kernel_size}, padding={self.padding}"


class MaxPool2d(_Pool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices: bool = False,
                 ceil_mode: bool = False):
        _unsupported(dilation=(_pair(dilation, "dilation"), (1, 1)), return_indices=(return_indices, False),
                     ceil_mode=(ceil_mode, False))
        _window(kernel_size, stride, padding)  # validate now, not at the first forward
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, x): return max_pool2d(x, self.kernel_size, self.stride, self.padding)


class AvgPool2d(_Pool2d):
    """Padded cells count toward the average (torch's count_include_pad=True)."""

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode: bool = False, count_include_pad: bool = True,
                 divisor_override=None):
        _unsupported(ceil_mode=(ceil_mode, False), count_include_pad=(count_include_pad, True),
                     divisor_override=(divisor_override, None))
        _window(kernel_size, stride, padding)
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, x): return avg_pool2d(x, self.kernel_size, self.stride, self.padding)


class Dropout(Module):
    """Zeroes inputs with probability p during training; identity in eval mode."""

    def __init__(self, p: float = 0.5, inplace: bool = False):
        _unsupported(inplace=(inplace, False))
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = p

    def forward(self, x): return dropout(x, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class _BatchNorm(Module):
    _input_ndims: tuple = ()

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float | None = 0.1, affine: bool = True,
                 track_running_stats: bool = True, *, ctx: Context | None = None):
        _unsupported(track_running_stats=(track_running_stats, True))
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
        momentum = self.momentum
        if self.training:
            tracked = self.num_batches_tracked.item() + 1
            self.num_batches_tracked.copy_([tracked])
            if momentum is None:  # cumulative moving average, as in PyTorch
                momentum = 1.0 / tracked
        return batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training,
                          momentum if momentum is not None else 0.0, self.eps)

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

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None, max_norm=None, *,
                 ctx: Context | None = None):
        _unsupported(padding_idx=(padding_idx, None), max_norm=(max_norm, None))
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


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        _unsupported(inplace=(inplace, False))
        self.negative_slope = negative_slope

    def forward(self, x): return leaky_relu(x, self.negative_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class _Elementwise(Module):
    _fn = None

    def __init__(self, inplace: bool = False):
        _unsupported(inplace=(inplace, False))

    def forward(self, x): return type(self)._fn(x)


class ReLU6(_Elementwise):
    _fn = staticmethod(relu6)


class Hardsigmoid(_Elementwise):
    _fn = staticmethod(hardsigmoid)


class Hardswish(_Elementwise):
    """x * relu6(x + 3) / 6: SiLU-like curve without exp (MobileNetV3)."""
    _fn = staticmethod(hardswish)


class SiLU(_Elementwise):
    """x * sigmoid(x) (also called swish)."""
    _fn = staticmethod(silu)


class GELU(Module):
    def __init__(self, approximate: str = "none"):
        if approximate not in ("none", "tanh"):
            raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
        self.approximate = approximate

    def forward(self, x): return gelu(x, self.approximate)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}"


class Softmax(Module):
    def __init__(self, dim: int | None = None):
        self.dim = dim

    def forward(self, x): return softmax(x, self.dim)


class LogSoftmax(Module):
    def __init__(self, dim: int | None = None):
        self.dim = dim

    def forward(self, x): return log_softmax(x, self.dim)


class Flatten(Module):
    """[B, d1, d2, ...] -> [B, d1*d2*...] with the default start_dim=1 (as in torch.nn.Flatten)."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        self.start_dim, self.end_dim = start_dim, end_dim

    def forward(self, x): return x.flatten(self.start_dim, self.end_dim)


# --------------------------------------------------------------------------
# Gradient utilities (torch.nn.utils)
# --------------------------------------------------------------------------


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> Tensor:
    """Scales grads in place so their joint 2-norm is at most max_norm
    (torch.nn.utils.clip_grad_norm_). Returns the norm before clipping."""
    if norm_type != 2.0:
        raise NotImplementedError("clip_grad_norm_ supports only norm_type=2")
    params = [parameters] if isinstance(parameters, Tensor) else list(parameters)
    if not params:
        return tensor(0.0)
    ptrs = L.ptr_array(p.ptr for p in params)
    total = lib.nc_clip_grad_norm(ptrs, len(params), float(max_norm))
    return tensor(total, ctx=params[0].ctx)


class utils:  # noqa: N801  (namespace, as torch.nn.utils)
    clip_grad_norm_ = staticmethod(clip_grad_norm_)


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

    @property
    def optimizer(self):
        """The optimizer set by compile() (Keras' model.optimizer), else None."""
        return self._compiled.optimizer if self._compiled is not None else None

    def compile(self, optimizer="rmsprop", loss=None, loss_weights=None, metrics=None, *, lr: float | None = None):
        """Keras' compile. ``optimizer`` is a name ("rmsprop", "adam", "adamw", "sgd" with Keras' default
        hyperparameters) or an optimizer instance; ``lr`` overrides the learning rate of a named optimizer."""
        from .training import compile_model

        _unsupported(loss_weights=(loss_weights, None))
        compile_model(self, optimizer, loss, metrics, lr)
        return self

    def fit(self, x=None, y=None, batch_size=None, epochs: int = 1, verbose="auto", callbacks=None,
            validation_split: float = 0.0, validation_data=None, shuffle: bool = True, class_weight=None,
            sample_weight=None, initial_epoch: int = 0, steps_per_epoch=None, *, seed=None):
        """Trains on arrays (x, y), or streams from a DataLoader / iterable of
        (x, y) batches passed as x with y=None (as Keras does for datasets)."""
        from .training import fit

        _unsupported(class_weight=(class_weight, None), sample_weight=(sample_weight, None),
                     steps_per_epoch=(steps_per_epoch, None))
        return fit(self, x, y, epochs, 32 if batch_size is None else batch_size, validation_data, validation_split,
                   shuffle, callbacks, 1 if verbose == "auto" else verbose, seed, initial_epoch)

    def evaluate(self, x=None, y=None, batch_size=None, verbose="auto", sample_weight=None, steps=None,
                 callbacks=None, return_dict: bool = False):
        """Loss (and metrics) on x, y. Like Keras: a float when there are no
        metrics, else ``[loss, accuracy]``; a dict with ``return_dict=True``."""
        from .training import evaluate

        _unsupported(sample_weight=(sample_weight, None), steps=(steps, None), callbacks=(callbacks, None))
        logs = evaluate(self, x, y, 32 if batch_size is None else batch_size)
        if return_dict:
            return logs
        values = list(logs.values())
        return values[0] if len(values) == 1 else values

    def predict(self, x, batch_size=None, verbose="auto", steps=None, callbacks=None):
        from .training import predict

        _unsupported(steps=(steps, None), callbacks=(callbacks, None))
        return predict(self, x, 32 if batch_size is None else batch_size)


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
