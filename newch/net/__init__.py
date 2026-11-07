"""Modules (PyTorch style) and Model/Sequential with compile/fit (Keras style)."""

from __future__ import annotations

import copy
import math
from collections import OrderedDict, namedtuple

from .. import _lib as L
from .._lib import lib
from ..tensor import persistent as _persistent_mode
from ..tensor import (Context, Tensor, _check_device, _is_device, _is_numpy, _pair, _to_buffer, _unsupported, _window, avg_pool2d,
                     batch_norm, cross_entropy, mse_loss, nll_loss,
                     conv2d, default_context, dropout, full, gelu, hardsigmoid, hardswish, int32, leaky_relu, linear,
                     load_tensors, log_softmax, max_pool2d, parameter, persistent, randn, relu6, save_tensors,
                     silu, softmax, tensor, zeros)

from ..tensor import _apply_mask, _conv_padding, _rand_uniform, layer_norm

__all__ = [
    "AvgPool2d", "BatchNorm1d", "BatchNorm2d", "Conv2d", "CrossEntropyLoss", "Dropout", "Embedding", "Flatten", "GELU",
    "Hardsigmoid", "Hardswish", "Identity", "LayerNorm", "LeakyReLU", "Linear", "LogSoftmax", "MSELoss", "MaxPool2d",
    "Model", "Module", "ModuleDict", "ModuleList", "MultiheadAttention", "NLLLoss", "QuantizedLinear", "ReLU", "ReLU6",
    "Sequential", "SiLU", "Sigmoid", "Softmax", "Tanh", "TransformerEncoder", "TransformerEncoderLayer",
    "clip_grad_norm_", "functional", "init", "quantize_", "utils",
]

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

    def named_children(self):
        for name, v in self._items():
            if isinstance(v, Module):
                yield name, v

    def modules(self):
        """This module and every sub-module, depth first."""
        yield self
        for c in self.children():
            yield from c.modules()

    def named_modules(self, prefix: str = ""):
        yield prefix, self
        for name, c in self.named_children():
            yield from c.named_modules(f"{prefix}.{name}" if prefix else name)

    def apply(self, fn) -> "Module":
        """Calls fn on every sub-module, then on self (e.g. weight initialization)."""
        for c in self.children():
            c.apply(fn)
        fn(self)
        return self

    def register_buffer(self, name: str, tensor_, persistent: bool = True) -> None:
        """Stores a non-trainable tensor that is saved in state_dict (e.g. running statistics)."""
        _unsupported(persistent=(persistent, True))
        if tensor_ is not None and not (tensor_.is_persistent and not tensor_.requires_grad):
            with _persistent_mode(tensor_.ctx):
                tensor_ = tensor(tensor_)  # owned copy that survives step_end()
        setattr(self, name, tensor_)

    def register_parameter(self, name: str, param) -> None:
        if param is not None and not (param.is_persistent and param.requires_grad):
            raise TypeError(f"'{name}' must be a parameter (nc.parameter(...)) or None")
        setattr(self, name, param)

    def to(self, *args, **kwargs) -> "Module":
        """Device moves are no-ops on the CPU; changing the dtype of parameters is not supported."""
        dtype = kwargs.pop("dtype", None)
        device = kwargs.pop("device", None)
        kwargs.pop("non_blocking", None)
        if kwargs:
            raise TypeError(f"to() got unexpected arguments {sorted(kwargs)}")
        for a in args:
            if a in L.DTYPE_CODES or not _is_device(a):
                dtype = a
            else:
                device = a
        _check_device(device)
        _unsupported(dtype=(dtype if dtype != "float32" else None, None))
        return self

    def cpu(self) -> "Module":
        return self

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
    return parameter(_rand_uniform(shape, -bound, bound, ctx), ctx=ctx)


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
    """NCHW convolution, weight [out, in/groups, kh, kw] (torch.nn.Conv2d). groups=in_channels gives a
    depthwise convolution (MobileNet); padding may be "same" or "valid"."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1,
                 groups: int = 1, bias: bool = True, padding_mode: str = "zeros", *, ctx: Context | None = None):
        _unsupported(padding_mode=(padding_mode, "zeros"))
        if groups < 1 or in_channels % groups or out_channels % groups:
            raise ValueError(f"in_channels ({in_channels}) and out_channels ({out_channels}) must be divisible "
                             f"by groups ({groups})")
        ctx = ctx or default_context()
        self.in_channels, self.out_channels, self.groups = in_channels, out_channels, groups
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride, self.dilation = _pair(stride, "stride"), _pair(dilation, "dilation")
        self.padding = padding if isinstance(padding, str) else _pair(padding, "padding")
        _conv_padding(self.padding, self.kernel_size, self.dilation, self.stride)  # validate now
        fan_in = in_channels // groups * self.kernel_size[0] * self.kernel_size[1]
        bound = 1.0 / math.sqrt(fan_in)  # torch's default (kaiming_uniform with a=sqrt(5))
        self.weight = _uniform_param((out_channels, in_channels // groups, *self.kernel_size), bound, ctx)
        self.bias = _uniform_param((out_channels,), bound, ctx) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        s = (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
             f"stride={self.stride}, padding={self.padding}")
        if self.dilation != (1, 1):
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        return s


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
    """Lookup table: int32 indices of any shape [*] -> rows [*, embedding_dim]; weight ~ N(0, 1)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None, max_norm=None, *,
                 ctx: Context | None = None):
        _unsupported(padding_idx=(padding_idx, None), max_norm=(max_norm, None))
        ctx = ctx or default_context()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.weight = parameter(randn(num_embeddings, embedding_dim, ctx=ctx), ctx=ctx)

    def forward(self, idx) -> Tensor:
        if not isinstance(idx, Tensor):
            idx = tensor(idx, dtype=int32, ctx=self.weight.ctx)
        if idx.ndim == 1:
            return self.weight.index_select(0, idx)
        flat = idx.reshape(-1)  # bound: must outlive the call
        return self.weight.index_select(0, flat).reshape(*idx.shape, self.embedding_dim)

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
            elif isinstance(v, dict) and any(isinstance(i, Linear) for i in v.values()):  # ModuleDict
                setattr(m, name, {k: convert(i) for k, i in v.items()})
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

    def compile(self, optimizer="adam", loss=None, metrics=None, *, lr: float | None = None):
        """Sets up fit()/evaluate(). ``optimizer`` is an nc.optim instance or a name ("adam", "adamw",
        "sgd", "rmsprop") that builds that class with its defaults; ``lr`` overrides the rate of a name.
        ``loss`` is "cross_entropy", "mse" or any callable(output, target)."""
        from ..training import compile_model

        compile_model(self, optimizer, loss, metrics, lr)
        return self

    def fit(self, x=None, y=None, batch_size=None, epochs: int = 1, verbose="auto", callbacks=None,
            validation_split: float = 0.0, validation_data=None, shuffle: bool = True, class_weight=None,
            sample_weight=None, initial_epoch: int = 0, steps_per_epoch=None, *, seed=None):
        """Trains on arrays (x, y), or streams from a DataLoader / iterable of
        (x, y) batches passed as x with y=None (as Keras does for datasets)."""
        from ..training import fit

        _unsupported(class_weight=(class_weight, None), sample_weight=(sample_weight, None),
                     steps_per_epoch=(steps_per_epoch, None))
        return fit(self, x, y, epochs, 32 if batch_size is None else batch_size, validation_data, validation_split,
                   shuffle, callbacks, 1 if verbose == "auto" else verbose, seed, initial_epoch)

    def evaluate(self, x=None, y=None, batch_size=None, verbose="auto", sample_weight=None, steps=None,
                 callbacks=None, return_dict: bool = False):
        """Loss (and metrics) on x, y. Like Keras: a float when there are no
        metrics, else ``[loss, accuracy]``; a dict with ``return_dict=True``."""
        from ..training import evaluate

        _unsupported(sample_weight=(sample_weight, None), steps=(steps, None), callbacks=(callbacks, None))
        logs = evaluate(self, x, y, 32 if batch_size is None else batch_size)
        if return_dict:
            return logs
        values = list(logs.values())
        return values[0] if len(values) == 1 else values

    def predict(self, x, batch_size=None, verbose="auto", steps=None, callbacks=None):
        from ..training import predict

        _unsupported(steps=(steps, None), callbacks=(callbacks, None))
        return predict(self, x, 32 if batch_size is None else batch_size)


class ModuleList(Module):
    """A list of sub-modules whose parameters are registered (torch.nn.ModuleList)."""

    def __init__(self, modules=None):
        self._list: list = []
        for m in modules or []:
            self.append(m)

    def _items(self):
        for i, m in enumerate(self._list):
            yield str(i), m

    def append(self, module: Module) -> "ModuleList":
        if not isinstance(module, Module):
            raise TypeError(f"ModuleList holds modules, got {type(module).__name__}")
        self._list.append(module)
        return self

    def extend(self, modules) -> "ModuleList":
        for m in modules:
            self.append(m)
        return self

    def insert(self, index: int, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError(f"ModuleList holds modules, got {type(module).__name__}")
        self._list.insert(index, module)

    def __getitem__(self, i):
        return ModuleList(self._list[i]) if isinstance(i, slice) else self._list[i]

    def __len__(self) -> int:
        return len(self._list)

    def __iter__(self):
        return iter(self._list)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleList has no forward(); iterate over it in your own forward()")


class ModuleDict(Module):
    """A dict of sub-modules whose parameters are registered (torch.nn.ModuleDict)."""

    def __init__(self, modules=None):
        self._dict: dict = {}
        for k, m in dict(modules or {}).items():
            self[k] = m

    def _items(self):
        yield from self._dict.items()

    def __setitem__(self, key: str, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError(f"ModuleDict holds modules, got {type(module).__name__}")
        self._dict[key] = module

    def __getitem__(self, key: str) -> Module:
        return self._dict[key]

    def __contains__(self, key) -> bool:
        return key in self._dict

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleDict has no forward(); pick modules from it in your own forward()")


class Identity(Module):
    """Returns its input; ignores constructor arguments (torch.nn.Identity)."""

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, x):
        return x


# Losses as modules (criterion = nn.CrossEntropyLoss(); loss = criterion(output, target))
class CrossEntropyLoss(Module):
    def __init__(self, weight=None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = "mean",
                 label_smoothing: float = 0.0):
        _unsupported(weight=(weight, None), size_average=(size_average, None), reduce=(reduce, None),
                     ignore_index=(ignore_index, -100))
        self.reduction, self.label_smoothing = reduction, label_smoothing

    def forward(self, input, target):
        return cross_entropy(input, target, reduction=self.reduction, label_smoothing=self.label_smoothing)


class NLLLoss(Module):
    def __init__(self, weight=None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = "mean"):
        _unsupported(weight=(weight, None), size_average=(size_average, None), reduce=(reduce, None),
                     ignore_index=(ignore_index, -100))
        self.reduction = reduction

    def forward(self, input, target):
        return nll_loss(input, target, reduction=self.reduction)


class MSELoss(Module):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean"):
        _unsupported(size_average=(size_average, None), reduce=(reduce, None))
        self.reduction = reduction

    def forward(self, input, target):
        return mse_loss(input, target, reduction=self.reduction)



# --------------------------------------------------------------------------
# Transformer blocks (torch.nn names, arguments and parameter layout)
# --------------------------------------------------------------------------


class LayerNorm(Module):
    """torch.nn.LayerNorm: normalizes over the trailing ``normalized_shape`` dims; weight 1, bias 0."""

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True, *,
                 ctx: Context | None = None):
        ctx = ctx or default_context()
        shape = (int(normalized_shape),) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.normalized_shape, self.eps, self.elementwise_affine = shape, eps, elementwise_affine
        with persistent(ctx):
            self.weight = parameter(full(shape, 1.0, ctx=ctx), ctx=ctx) if elementwise_affine else None
            self.bias = parameter(shape, ctx=ctx) if elementwise_affine and bias else None

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class MultiheadAttention(Module):
    """torch.nn.MultiheadAttention: packed ``in_proj_weight`` [3E, E] / ``in_proj_bias`` and ``out_proj``,
    so torch state_dicts load as they are. Inputs are (L, N, E), or (N, L, E) with batch_first.

    Masks follow torch: an integer 0/1 ``attn_mask`` / ``key_padding_mask`` marks positions that may
    NOT be attended (torch's True; newch has no bool dtype), a float mask is added to the scores.
    ``kdim``/``vdim`` other than embed_dim, ``add_bias_kv`` and ``add_zero_attn`` raise.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True,
                 add_bias_kv: bool = False, add_zero_attn: bool = False, kdim=None, vdim=None,
                 batch_first: bool = False, *, ctx: Context | None = None):
        _unsupported(add_bias_kv=(add_bias_kv, False), add_zero_attn=(add_zero_attn, False),
                     kdim=(kdim if kdim != embed_dim else None, None), vdim=(vdim if vdim != embed_dim else None, None))
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        ctx = ctx or default_context()
        self.embed_dim, self.num_heads, self.dropout, self.batch_first = embed_dim, num_heads, dropout, batch_first
        self.head_dim = embed_dim // num_heads
        bound = math.sqrt(6.0 / (embed_dim + 3 * embed_dim))  # xavier_uniform_, as torch
        self.in_proj_weight = _uniform_param((3 * embed_dim, embed_dim), bound, ctx)
        self.in_proj_bias = parameter((3 * embed_dim,), ctx=ctx) if bias else None
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, ctx=ctx)
        if bias:
            self.out_proj.bias.copy_([0.0] * embed_dim)

    def _project(self, query, key, value):
        E, W, b = self.embed_dim, self.in_proj_weight, self.in_proj_bias
        if query is key and key is value:  # self-attention: one GEMM for q, k and v
            L, N = query.shape[0], query.shape[1]
            qkv = linear(query, W, b).reshape(L * N, 3, E).transpose(0, 1)  # [3, L*N, E]
            return [qkv[i].reshape(L, N, E) for i in range(3)]
        out = []
        for i, x in enumerate((query, key, value)):
            rows = list(range(i * E, (i + 1) * E))
            out.append(linear(x, W[rows], b[rows] if b is not None else None))
        return out

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask=None, need_weights: bool = True,
                attn_mask=None, average_attn_weights: bool = True, is_causal: bool = False):
        if is_causal and attn_mask is None:
            raise RuntimeError("Need attn_mask if specifying the is_causal hint")  # as torch
        unbatched = query.ndim == 2
        if unbatched:
            query, key, value = (x.unsqueeze(1) for x in (query, key, value))
            if key_padding_mask is not None and isinstance(key_padding_mask, Tensor):
                key_padding_mask = key_padding_mask.unsqueeze(0)
        elif self.batch_first:
            same = query is key and key is value
            query = query.transpose(0, 1)
            key, value = (query, query) if same else (key.transpose(0, 1), value.transpose(0, 1))
        L, N, E = query.shape
        S, H, D = key.shape[0], self.num_heads, self.head_dim
        q, k, v = self._project(query, key, value)
        q = q.reshape(L, N * H, D).transpose(0, 1) * (1.0 / math.sqrt(D))  # [N*H, L, D]
        k = k.reshape(S, N * H, D).transpose(0, 1)
        v = v.reshape(S, N * H, D).transpose(0, 1)
        scores = q @ k.transpose(1, 2)  # [N*H, L, S]
        if attn_mask is not None:
            scores = _apply_mask(scores, attn_mask, keep=False)
        if key_padding_mask is not None:
            pad = key_padding_mask if isinstance(key_padding_mask, Tensor) else tensor(key_padding_mask, ctx=q.ctx)
            scores = _apply_mask(scores.reshape(N, H, L, S), pad.reshape(N, 1, 1, S), keep=False).reshape(N * H, L, S)
        attn = scores.softmax(-1)
        if self.training and self.dropout > 0.0:
            attn = dropout(attn, self.dropout, True)
        out = (attn @ v).transpose(0, 1).reshape(L, N, E)  # heads back side by side
        out = self.out_proj(out)
        if unbatched:
            out = out.squeeze(1)
        elif self.batch_first:
            out = out.transpose(0, 1)
        if not need_weights:
            return out, None
        w = attn.reshape(N, H, L, S)
        w = w.mean(1) if average_attn_weights else w
        return out, (w.squeeze(0) if unbatched else w)

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, batch_first={self.batch_first}"


_ACTIVATIONS = {"relu": lambda x: x.relu(), "gelu": lambda x: gelu(x)}


class TransformerEncoderLayer(Module):
    """torch.nn.TransformerEncoderLayer: self-attention + feed-forward, each with a residual and LayerNorm
    (post-norm by default, ``norm_first=True`` for pre-norm). Same sub-module names as torch."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation="relu", layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, *, ctx: Context | None = None):
        if isinstance(activation, str):
            if activation not in _ACTIVATIONS:
                raise RuntimeError(f"activation should be relu/gelu, not {activation}")
            activation = _ACTIVATIONS[activation]
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first,
                                            ctx=ctx)
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, ctx=ctx)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, ctx=ctx)
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, ctx=ctx)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, ctx=ctx)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = activation

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False,
                           is_causal=is_causal)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, src: Tensor, src_mask=None, src_key_padding_mask=None, is_causal: bool = False) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x


class TransformerEncoder(Module):
    """torch.nn.TransformerEncoder: ``num_layers`` independent copies (copy.deepcopy) of encoder_layer,
    then an optional final norm. enable_nested_tensor and mask_check only tune torch's fast path."""

    def __init__(self, encoder_layer: Module, num_layers: int, norm: Module | None = None,
                 enable_nested_tensor: bool = True, mask_check: bool = True):
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask=None, src_key_padding_mask=None, is_causal=None) -> Tensor:
        x = src
        for layer in self.layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=bool(is_causal))
        return self.norm(x) if self.norm is not None else x


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
        return Sequential(self.layers[i]) if isinstance(i, slice) else self.layers[i]

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def _items(self):
        for i, layer in enumerate(self.layers):
            yield str(i), layer


from . import functional, init  # noqa: E402  (torch.nn.functional / torch.nn.init)
