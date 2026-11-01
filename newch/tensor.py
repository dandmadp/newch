"""Tensor, Context and functional ops: the PyTorch-style layer."""

from __future__ import annotations

import array
import contextlib
import ctypes
from typing import Sequence

from . import _lib as L
from ._lib import lib

float32 = "float32"
int32 = "int32"
int8 = "int8"


class NcError(RuntimeError):
    """Raised when the engine reports an error (shape mismatch, bad index, ...)."""


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------


class Context:
    """Owns all engine memory. Most code uses the default context.

    Temporary tensors live until :meth:`step_end`. Persistent tensors
    (parameters, tensors created inside :func:`persistent`) are freed when
    their last Python reference goes away, or when the context is closed.

    threads: worker threads including the caller (0 = all cores).
    """

    def __init__(self, seed: int = 0, threads: int = 0, arena_chunk_mb: float = 1,
                 force_reference: bool = False):
        cfg = L.Config(int(arena_chunk_mb * (1 << 20)), seed, int(force_reference), int(threads))
        self._p = lib.nc_init(ctypes.byref(cfg))
        if not self._p:
            raise NcError("nc_init failed (out of memory?)")
        self.step_id = 0

    def step_end(self) -> None:
        """Frees every temporary tensor at once (O(1)) and the autograd tape."""
        lib.nc_step_end(self._p)
        self.step_id += 1

    def trim(self) -> None:
        """Returns unused step memory to the OS (after an unusually large step)."""
        lib.nc_arena_trim(self._p)

    @property
    def threads(self) -> int:
        return lib.nc_num_threads(self._p)

    def close(self) -> None:
        if self._p:
            lib.nc_free_ctx(self._p)
            self._p = None

    def raise_error(self) -> None:
        msg = lib.nc_error(self._p).decode(errors="replace") or "unknown engine error"
        lib.nc_clear_error(self._p)
        raise NcError(msg)

    def wrap(self, ptr, base: "Tensor | None" = None) -> "Tensor":
        """Wraps a fresh engine result. Persistent results are owned (freed on GC)."""
        if not ptr:
            self.raise_error()
        t = Tensor._from_ptr(ptr, self, base=base)
        t._owner = t._step is None
        return t

    @property
    def backend(self) -> str:
        return lib.nc_kernel_backend(self._p).decode()

    @property
    def cpu(self) -> dict:
        c = lib.nc_cpu(self._p).contents
        info = {name: getattr(c, name) for name, _ in L.CpuInfo._fields_}
        info["arch"] = c.arch.decode()
        return info

    def mem_stats(self) -> dict:
        s = lib.nc_get_mem_stats(self._p)
        return {name: getattr(s, name) for name, _ in L.MemStats._fields_}

    def reset_peak(self) -> None:
        lib.nc_reset_peak(self._p)


_default: Context | None = None


def default_context() -> Context:
    global _default
    if _default is None:
        _default = Context()
    return _default


def set_default_context(ctx: Context) -> None:
    global _default
    _default = ctx


def manual_seed(seed: int, ctx: Context | None = None) -> None:
    """Reseeds the random generator (like torch.manual_seed); tensors are kept."""
    lib.nc_manual_seed((ctx or default_context())._p, int(seed))


def step_end(ctx: Context | None = None) -> None:
    (ctx or default_context()).step_end()


def mem_stats(ctx: Context | None = None) -> dict:
    return (ctx or default_context()).mem_stats()


@contextlib.contextmanager
def no_grad(ctx: Context | None = None):
    ctx = ctx or default_context()
    prev = lib.nc_set_grad_enabled(ctx._p, 0)
    try:
        yield
    finally:
        lib.nc_set_grad_enabled(ctx._p, prev)


@contextlib.contextmanager
def persistent(ctx: Context | None = None):
    """Tensors created inside survive step_end() (no autograd recording)."""
    ctx = ctx or default_context()
    lib.nc_persistent_begin(ctx._p)
    try:
        yield
    finally:
        lib.nc_persistent_end(ctx._p)


# --------------------------------------------------------------------------
# Data conversion
# --------------------------------------------------------------------------


def _flatten(data, shape: list, out: list, depth: int = 0) -> None:
    if isinstance(data, (list, tuple)):
        if len(shape) <= depth:
            shape.append(len(data))
        elif shape[depth] != len(data):
            raise ValueError("ragged nested sequence: all rows must have the same length")
        for item in data:
            _flatten(item, shape, out, depth + 1)
    else:
        if depth != len(shape):
            raise ValueError("ragged nested sequence: inconsistent depth")
        out.append(data)


def _is_numpy(obj) -> bool:
    return type(obj).__module__.split(".")[0] == "numpy"


def _to_buffer(data, dtype: str | None):
    """Returns (keepalive, address, shape, dtype) for contiguous host data."""
    if _is_numpy(data):
        import numpy as np

        if dtype is None:
            dtype = int32 if np.issubdtype(data.dtype, np.integer) or data.dtype == np.bool_ else float32
        arr = np.ascontiguousarray(data, dtype={float32: np.float32, int32: np.int32, int8: np.int8}[dtype])
        return arr, arr.ctypes.data, arr.shape, dtype
    if isinstance(data, array.array):
        if dtype is None:
            dtype = float32 if data.typecode in "fd" else int8 if data.typecode == "b" else int32
        code = L.ARRAY_CODE[dtype]
        buf = data if data.typecode == code else array.array(code, data)
        return buf, (buf.buffer_info()[0] if len(buf) else None), (len(buf),), dtype

    shape: list = []
    flat: list = []
    _flatten(data, shape, flat)
    if dtype is None:
        dtype = float32 if any(isinstance(v, float) for v in flat) else int32
        if not flat:
            dtype = float32
    buf = array.array(L.ARRAY_CODE[dtype], flat if dtype == float32 else (int(v) for v in flat))
    addr = buf.buffer_info()[0] if len(buf) else None
    return buf, addr, tuple(shape), dtype


def _unflatten(flat: Sequence, shape: Sequence[int]):
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    step = len(flat) // shape[0] if shape[0] else 0
    return [_unflatten(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


# --------------------------------------------------------------------------
# Tensor
# --------------------------------------------------------------------------


Number = (int, float)


class Tensor:
    """A view of an engine tensor.

    Temporary tensors become invalid after ``step_end()``; touching one then
    raises NcError instead of reading freed memory.
    """

    __slots__ = ("_p", "_ctx", "_step", "_owner", "_base")

    @classmethod
    def _from_ptr(cls, ptr, ctx: Context, base: "Tensor | None" = None) -> "Tensor":
        t = object.__new__(cls)
        t._p = ptr
        t._ctx = ctx
        t._step = None if lib.nc_is_persistent(ptr) else ctx.step_id
        t._owner = False  # only Context.wrap hands out ownership
        t._base = base    # keeps the tensor whose memory this one uses alive
        return t

    def __del__(self):
        # Persistent tensors own their memory exclusively (the engine never
        # makes persistent views), so the owning wrapper can free it.
        try:
            if self._owner and self._p and self._ctx._p:
                lib.nc_release(self._p)
        except Exception:  # interpreter shutdown
            pass

    def __init__(self, *args, **kwargs):
        raise TypeError("use newch.tensor(...), newch.zeros(...) etc. to create tensors")

    @property
    def ptr(self):
        if self._step is not None and self._step != self._ctx.step_id:
            raise NcError(
                "tensor was freed by step_end(); create it inside nc.persistent() "
                "or copy the values out (tolist(), item()) before ending the step"
            )
        return self._p

    @property
    def ctx(self) -> Context:
        return self._ctx

    def _w(self, ptr) -> "Tensor":
        return self._ctx.wrap(ptr)

    def _view(self, ptr) -> "Tensor":
        """Result that may alias self's memory: keeps self alive."""
        return self._ctx.wrap(ptr, base=self)

    def _other(self, x) -> "Tensor":
        if isinstance(x, Tensor):
            return x
        if isinstance(x, Number):
            return self._w(lib.nc_scalar(self._ctx._p, float(x)))
        return tensor(x, ctx=self._ctx)

    # ---- metadata --------------------------------------------------------
    @property
    def shape(self) -> tuple:
        p = self.ptr
        n = lib.nc_ndim(p)
        s = lib.nc_shape(p)
        return tuple(s[i] for i in range(n))

    @property
    def ndim(self) -> int:
        return lib.nc_ndim(self.ptr)

    @property
    def dtype(self) -> str:
        return L.DTYPE_NAMES.get(lib.nc_get_dtype(self.ptr), "unknown")

    def numel(self) -> int:
        return lib.nc_numel(self.ptr)

    def size(self, dim: int | None = None):
        """Shape tuple, or the size of one dimension (as in PyTorch)."""
        return self.shape if dim is None else self.shape[dim]

    def __len__(self) -> int:
        shape = self.shape
        if not shape:
            raise TypeError("len() of a 0-d tensor")
        return shape[0]

    @property
    def is_persistent(self) -> bool:
        return self._step is None

    @property
    def requires_grad(self) -> bool:
        return bool(lib.nc_requires_grad(self.ptr))

    @requires_grad.setter
    def requires_grad(self, on: bool) -> None:
        lib.nc_set_requires_grad(self.ptr, int(on))
        if not lib.nc_ok(self._ctx._p):
            self._ctx.raise_error()

    @property
    def grad(self) -> "Tensor | None":
        g = lib.nc_grad(self.ptr)
        return Tensor._from_ptr(g, self._ctx, base=self) if g else None  # freed with self

    # ---- data out --------------------------------------------------------
    def item(self) -> float:
        if self.numel() != 1:
            raise ValueError(f"item() needs exactly one element, tensor has {self.numel()}")
        v = lib.nc_item(self.ptr)
        return int(v) if self.dtype == int32 else v

    def _flat(self) -> array.array:
        dt = self.dtype
        buf = array.array(L.ARRAY_CODE[dt], bytes(L.DTYPE_SIZE[dt] * self.numel()))
        if buf:
            lib.nc_copy_to(self.ptr, buf.buffer_info()[0], len(buf) * buf.itemsize)
        return buf

    def copy_(self, src) -> "Tensor":
        """Overwrites this (contiguous) tensor's values in place, like torch's copy_."""
        dt = self.dtype
        if isinstance(src, Tensor):
            src = src._flat()
        keep, addr, shape, _ = _to_buffer(src, dt)
        n = 1
        for d in shape:
            n *= d
        if n != self.numel():
            raise ValueError(f"copy_: source has {n} elements, tensor has {self.numel()}")
        if not lib.nc_copy_from(self.ptr, addr, n * L.DTYPE_SIZE[dt]):
            self._ctx.raise_error()
        del keep
        return self

    def tolist(self):
        return _unflatten(self._flat(), self.shape)

    def numpy(self):
        import numpy as np

        np_dtype = {float32: np.float32, int32: np.int32, int8: np.int8}[self.dtype]
        return np.frombuffer(self._flat(), dtype=np_dtype).reshape(self.shape).copy()

    def __repr__(self) -> str:
        try:
            shape = self.shape
        except NcError:
            return "tensor(<freed>)"
        flags = ", requires_grad=True" if self.requires_grad else ""
        if self.numel() > 64:
            body = f"<{self.numel()} values>"
        else:
            body = repr(self.tolist())
        return f"tensor({body}, shape={shape}, dtype={self.dtype}{flags})"

    def __float__(self) -> float:
        return float(self.item())

    def __bool__(self) -> bool:
        return bool(self.item())

    # ---- arithmetic ------------------------------------------------------
    def __add__(self, o):
        if isinstance(o, Number):
            return self._w(lib.nc_add_scalar(self.ptr, float(o)))
        return self._w(lib.nc_add(self.ptr, self._other(o).ptr))

    __radd__ = __add__

    def __sub__(self, o):
        if isinstance(o, Number):
            return self._w(lib.nc_add_scalar(self.ptr, -float(o)))
        return self._w(lib.nc_sub(self.ptr, self._other(o).ptr))

    def __rsub__(self, o):
        return self._other(o) - self

    def __mul__(self, o):
        if isinstance(o, Number):
            return self._w(lib.nc_scale(self.ptr, float(o)))
        return self._w(lib.nc_mul(self.ptr, self._other(o).ptr))

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, Number):
            return self._w(lib.nc_scale(self.ptr, 1.0 / float(o)))
        return self._w(lib.nc_div(self.ptr, self._other(o).ptr))

    def __rtruediv__(self, o):
        return self._other(o) / self

    def __neg__(self):
        return self._w(lib.nc_neg(self.ptr))

    def __pow__(self, p):
        if not isinstance(p, Number):
            raise TypeError("only scalar exponents are supported")
        return self._w(lib.nc_pow_scalar(self.ptr, float(p)))

    def __matmul__(self, o):
        return self._w(lib.nc_matmul(self.ptr, self._other(o).ptr))

    def __rmatmul__(self, o):
        return self._other(o) @ self

    # ---- math ------------------------------------------------------------
    def exp(self): return self._w(lib.nc_exp(self.ptr))
    def log(self): return self._w(lib.nc_log(self.ptr))
    def sqrt(self): return self._w(lib.nc_sqrt(self.ptr))
    def relu(self): return self._w(lib.nc_relu(self.ptr))
    def sigmoid(self): return self._w(lib.nc_sigmoid(self.ptr))
    def tanh(self): return self._w(lib.nc_tanh(self.ptr))
    def _last_dim(self, dim: int, what: str) -> None:
        if dim not in (-1, self.ndim - 1):
            raise NotImplementedError(f"{what} is only implemented over the last dimension (dim=-1)")

    def softmax(self, dim: int = -1):
        self._last_dim(dim, "softmax")
        return self._w(lib.nc_softmax(self.ptr))

    def log_softmax(self, dim: int = -1):
        self._last_dim(dim, "log_softmax")
        return self._w(lib.nc_log_softmax(self.ptr))

    # Reductions take PyTorch's `dim`; numpy's `axis` is accepted as an alias.
    def _reduce(self, fn, dim, keepdim, axis):
        dim = axis if dim is None else dim
        return self._w(fn(self.ptr, L.ALL_AXES if dim is None else int(dim), int(keepdim)))

    def sum(self, dim=None, keepdim=False, *, axis=None): return self._reduce(lib.nc_sum, dim, keepdim, axis)
    def mean(self, dim=None, keepdim=False, *, axis=None): return self._reduce(lib.nc_mean, dim, keepdim, axis)

    def amax(self, dim=None, keepdim=False, *, axis=None):
        """Maximum values (torch.amax). The gradient goes to the first maximum."""
        return self._reduce(lib.nc_max, dim, keepdim, axis)

    def argmax(self, dim=None, *, axis=None):
        """Index of the maximum; flattened index when dim is None (as in PyTorch)."""
        dim = axis if dim is None else dim
        return self._w(lib.nc_argmax(self.ptr, L.ALL_AXES if dim is None else int(dim)))

    # ---- shape -----------------------------------------------------------
    def reshape(self, *shape):
        """Returns a view when the layout allows it, otherwise a copy (as in PyTorch)."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        n, arr = L.shape_arg(shape)
        return self._view(lib.nc_reshape(self.ptr, n, arr))

    def flatten(self, start_dim: int = 0, end_dim: int = -1):
        s = self.shape
        if not s:
            return self.reshape(1)
        start, end = start_dim % len(s), end_dim % len(s)
        merged = 1
        for d in s[start:end + 1]:
            merged *= d
        return self.reshape(*s[:start], merged, *s[end + 1:])

    def transpose(self, dim0: int, dim1: int):
        return self._view(lib.nc_transpose(self.ptr, dim0, dim1))

    @property
    def T(self):
        """Transpose of a 0-2 dimensional tensor. Use .mT for batches of matrices."""
        if self.ndim < 2:
            return self
        if self.ndim > 2:
            raise NotImplementedError(".T reverses all dims in PyTorch; use .mT to swap the last two")
        return self.transpose(0, 1)

    @property
    def mT(self):
        """Swaps the last two dimensions (torch.Tensor.mT)."""
        return self.transpose(-2, -1)

    def contiguous(self): return self._view(lib.nc_contiguous(self.ptr))
    def detach(self): return self._view(lib.nc_detach(self.ptr))

    def index_select(self, dim: int, index):
        """Rows along ``dim`` (only dim=0 is implemented); grads scatter-add back."""
        if dim not in (0, -self.ndim):
            raise NotImplementedError("index_select is only implemented for dim=0")
        if not isinstance(index, Tensor):
            index = tensor(index, dtype=int32, ctx=self._ctx)
        return self._w(lib.nc_index_select(self.ptr, index.ptr))

    def __getitem__(self, key):
        """Indexing along axis 0: int, slice, list of ints or an int32 tensor."""
        if isinstance(key, int):
            n = self.shape[0]
            if not -n <= key < n:
                raise IndexError(f"index {key} out of range for axis 0 of size {n}")
            row = self.index_select(0, [key % n])
            return row.reshape(*self.shape[1:])
        if isinstance(key, slice):
            return self.index_select(0, list(range(*key.indices(self.shape[0]))))
        if isinstance(key, (list, Tensor)) or _is_numpy(key):
            return self.index_select(0, key)
        raise TypeError("only axis-0 indexing (int, slice, list, int32 tensor) is supported")

    # ---- autograd --------------------------------------------------------
    def backward(self) -> None:
        if lib.nc_backward(self.ptr) != 0:
            self._ctx.raise_error()

    def zero_grad(self) -> None:
        lib.nc_zero_grad(self.ptr)


# --------------------------------------------------------------------------
# Creation
# --------------------------------------------------------------------------


def _ctx(ctx):
    return ctx or default_context()


def _shape(shape) -> tuple:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        return tuple(shape[0])
    return tuple(shape)


def tensor(data, dtype: str | None = None, requires_grad: bool = False, ctx: Context | None = None) -> Tensor:
    """Creates a tensor by copying ``data`` (nested lists, array.array, numpy, Tensor).

    Integer data becomes int32, anything with floats becomes float32.
    """
    if isinstance(data, Tensor):
        if data.dtype == float32 and dtype in (None, float32):
            t = data.ctx.wrap(lib.nc_add_scalar(data.detach().ptr, 0.0))  # engine-side copy
            if requires_grad:
                t.requires_grad = True
            return t
        ctx = ctx or data.ctx
        data = data.tolist()
    ctx = _ctx(ctx)
    if isinstance(data, Number):
        dtype = dtype or (float32 if isinstance(data, float) else int32)
        keep, addr, shape, dtype = _to_buffer([data], dtype)
        shape = ()
    else:
        keep, addr, shape, dtype = _to_buffer(data, dtype)
    if dtype not in L.DTYPE_CODES:
        raise ValueError(f"unsupported dtype {dtype!r}; use float32 or int32")
    n, arr = L.shape_arg(shape)
    t = ctx.wrap(lib.nc_from_data(ctx._p, L.DTYPE_CODES[dtype], addr, n, arr))
    del keep
    if requires_grad:
        t.requires_grad = True
    return t


def zeros(*shape, dtype: str = float32, ctx: Context | None = None) -> Tensor:
    ctx = _ctx(ctx)
    n, arr = L.shape_arg(_shape(shape))
    return ctx.wrap(lib.nc_zeros(ctx._p, L.DTYPE_CODES[dtype], n, arr))


def full(shape, value: float, ctx: Context | None = None) -> Tensor:
    ctx = _ctx(ctx)
    n, arr = L.shape_arg(shape if isinstance(shape, (list, tuple)) else (shape,))
    return ctx.wrap(lib.nc_full(ctx._p, float(value), n, arr))


def ones(*shape, ctx: Context | None = None) -> Tensor:
    return full(_shape(shape), 1.0, ctx)


def randn(*shape, std: float = 1.0, ctx: Context | None = None) -> Tensor:
    ctx = _ctx(ctx)
    n, arr = L.shape_arg(_shape(shape))
    return ctx.wrap(lib.nc_randn(ctx._p, float(std), n, arr))


def rand(*shape, low: float = 0.0, high: float = 1.0, ctx: Context | None = None) -> Tensor:
    ctx = _ctx(ctx)
    n, arr = L.shape_arg(_shape(shape))
    return ctx.wrap(lib.nc_rand_uniform(ctx._p, float(low), float(high), n, arr))


def parameter(init, ctx: Context | None = None) -> Tensor:
    """A trainable persistent tensor. ``init`` is a Tensor/array-like or a shape tuple (zeros)."""
    ctx = _ctx(ctx)
    if isinstance(init, tuple) and all(isinstance(s, int) for s in init):
        n, arr = L.shape_arg(init)
        return ctx.wrap(lib.nc_parameter(ctx._p, None, n, arr))
    src = init if isinstance(init, Tensor) else tensor(init, dtype=float32, ctx=ctx)
    n, arr = L.shape_arg(src.shape)
    return ctx.wrap(lib.nc_parameter(ctx._p, src.ptr, n, arr))


def release(t: Tensor) -> None:
    """Frees a persistent tensor now instead of when it is garbage collected."""
    if t.is_persistent and t._p:
        lib.nc_release(t._p)
        t._p = None
        t._owner = False
        t._step = -1  # any further access raises


def tensor_from_bytes(buf: bytes, shape, dtype: str, ctx: Context | None = None) -> Tensor:
    """Tensor from raw little-endian row-major bytes (one copy, into the engine)."""
    ctx = _ctx(ctx)
    n, arr = L.shape_arg(shape)
    addr = ctypes.cast(ctypes.c_char_p(buf), ctypes.c_void_p) if buf else None
    return ctx.wrap(lib.nc_from_data(ctx._p, L.DTYPE_CODES[dtype], addr, n, arr))


# --------------------------------------------------------------------------
# Functional
# --------------------------------------------------------------------------


def matmul(a: Tensor, b: Tensor) -> Tensor: return a @ b
def relu(x: Tensor) -> Tensor: return x.relu()
def sigmoid(x: Tensor) -> Tensor: return x.sigmoid()
def tanh(x: Tensor) -> Tensor: return x.tanh()
def exp(x: Tensor) -> Tensor: return x.exp()
def log(x: Tensor) -> Tensor: return x.log()
def sqrt(x: Tensor) -> Tensor: return x.sqrt()
def softmax(x: Tensor, dim: int = -1) -> Tensor: return x.softmax(dim)
def log_softmax(x: Tensor, dim: int = -1) -> Tensor: return x.log_softmax(dim)


def _pair(v, name: str) -> tuple:
    if isinstance(v, str):
        raise NotImplementedError(f"string {name} ({v!r}) is not supported; pass an int or (h, w)")
    if isinstance(v, int):
        return v, v
    v = tuple(int(i) for i in v)
    if len(v) != 2:
        raise ValueError(f"{name} must be an int or a pair")
    return v


def conv2d(x: Tensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0) -> Tensor:
    """NCHW convolution (torch.nn.functional.conv2d without dilation/groups)."""
    (sh, sw), (ph, pw) = _pair(stride, "stride"), _pair(padding, "padding")
    return x._w(lib.nc_conv2d(x.ptr, weight.ptr, bias.ptr if bias is not None else None, sh, sw, ph, pw))


def max_pool2d(x: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    (kh, kw) = _pair(kernel_size, "kernel_size")
    (sh, sw) = _pair(stride if stride is not None else kernel_size, "stride")
    (ph, pw) = _pair(padding, "padding")
    return x._w(lib.nc_max_pool2d(x.ptr, kh, kw, sh, sw, ph, pw))


def avg_pool2d(x: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    """Average pooling; padded cells count toward the average (count_include_pad=True)."""
    (kh, kw) = _pair(kernel_size, "kernel_size")
    (sh, sw) = _pair(stride if stride is not None else kernel_size, "stride")
    (ph, pw) = _pair(padding, "padding")
    return x._w(lib.nc_avg_pool2d(x.ptr, kh, kw, sh, sw, ph, pw))


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training:
        return x
    return x._w(lib.nc_dropout(x.ptr, float(p)))


def batch_norm(x: Tensor, running_mean: Tensor, running_var: Tensor, weight: Tensor | None = None,
               bias: Tensor | None = None, training: bool = False, momentum: float = 0.1,
               eps: float = 1e-5) -> Tensor:
    """torch.nn.functional.batch_norm. In training mode the running buffers are updated in place."""
    c = running_mean.shape[0]
    if weight is None:
        weight = full((c,), 1.0, ctx=x.ctx)
    if bias is None:
        bias = zeros(c, ctx=x.ctx)
    return x._w(lib.nc_batch_norm(x.ptr, weight.ptr, bias.ptr, running_mean.ptr, running_var.ptr,
                                  int(training), float(momentum), float(eps)))


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """y = x @ weight.T + bias, weight [out, in] (torch.nn.functional.linear)."""
    return x._w(lib.nc_linear_fn(x.ptr, weight.ptr, bias.ptr if bias is not None else None))


def cross_entropy(logits: Tensor, labels) -> Tensor:
    """Mean cross-entropy; logits [N, C] float32, labels [N] int32."""
    if not isinstance(labels, Tensor):
        labels = tensor(labels, dtype=int32, ctx=logits.ctx)
    return logits._w(lib.nc_cross_entropy(logits.ptr, labels.ptr))


def mse_loss(pred: Tensor, target) -> Tensor:
    return pred._w(lib.nc_mse_loss(pred.ptr, pred._other(target).ptr))


# --------------------------------------------------------------------------
# Tensor files (NCW1)
# --------------------------------------------------------------------------


def save_tensors(tensors: dict, path) -> None:
    """Writes {name: Tensor} to an NCW1 file (format: docs/FORMAT.md)."""
    if not tensors:
        raise ValueError("save_tensors: nothing to save")
    items = list(tensors.items())
    ctx = items[0][1].ctx
    st = lib.nc_save(ctx._p, str(path).encode("utf-8"), len(items), L.name_array(k for k, _ in items),
                     L.ptr_array(t.ptr for _, t in items))
    if st != 0:
        ctx.raise_error()


def load_tensors(path, ctx: Context | None = None) -> dict:
    """Reads an NCW1 file into {name: persistent Tensor}."""
    import struct

    ctx = _ctx(ctx)
    out = {}
    with open(path, "rb") as f:
        if f.read(4) != b"NCW1":
            raise NcError(f"load_tensors: '{path}' is not an NCW1 tensor file")
        version, count = struct.unpack("<II", f.read(8))
        if version != 1:
            raise NcError(f"load_tensors: unsupported version {version}")
        with persistent(ctx):
            for _ in range(count):
                (name_len,) = struct.unpack("<I", f.read(4))
                name = f.read(name_len).decode("utf-8")
                code, ndim = struct.unpack("<II", f.read(8))
                shape = struct.unpack(f"<{ndim}q", f.read(8 * ndim))
                (nbytes,) = struct.unpack("<Q", f.read(8))
                data = f.read(nbytes)
                if len(data) != nbytes or code not in L.DTYPE_NAMES:
                    raise NcError(f"load_tensors: '{path}' is truncated or corrupt")
                out[name] = tensor_from_bytes(data, shape, L.DTYPE_NAMES[code], ctx)
    return out
