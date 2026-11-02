"""Tensor, Context and functional ops: the PyTorch-style layer."""

from __future__ import annotations

import array
import contextlib
import ctypes
import json
import warnings
from typing import Sequence

from . import _lib as L
from ._lib import lib

float32 = "float32"
int32 = "int32"
int8 = "int8"
uint8 = "uint8"  # compact storage for raw data (pixels, small labels); convert with .float()


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
        self._pdepth = 0  # nesting of persistent(): results are persistent while > 0
        self._grad_on = True  # False inside no_grad(): temporaries are freed when Python drops them
        self._live: dict = {}  # this step's temporaries that Python still references: ptr -> count

    def step_end(self) -> None:
        """Frees every temporary tensor at once (O(1)) and the autograd tape."""
        lib.nc_step_end(self._p)
        self.step_id += 1
        self._live = {}

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

    def wrap(self, ptr, base: "Tensor | None" = None, persistent: "bool | None" = None) -> "Tensor":
        """Wraps a fresh engine result. Persistent results are owned (freed on GC).

        Engine results are persistent exactly while persistent() is active, so the
        flag comes from Python; pass persistent=True for nc_parameter, which always is.
        """
        if not ptr:
            self.raise_error()
        t = Tensor._from_ptr(ptr, self, base, self._pdepth > 0 if persistent is None else persistent)
        t._owner = t._step is None
        t._free = t._step is not None and not self._grad_on
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
    prev_py, ctx._grad_on = ctx._grad_on, False
    try:
        yield
    finally:
        ctx._grad_on = prev_py
        lib.nc_set_grad_enabled(ctx._p, prev)


@contextlib.contextmanager
def persistent(ctx: Context | None = None):
    """Tensors created inside survive step_end() (no autograd recording)."""
    ctx = ctx or default_context()
    lib.nc_persistent_begin(ctx._p)
    ctx._pdepth += 1
    try:
        yield
    finally:
        ctx._pdepth -= 1
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


def _numpy_dtype(arr) -> str:
    """newch dtype for a numpy array: uint8/int8 stay 1 byte, other integers and bools become int32."""
    import numpy as np

    if arr.dtype == np.uint8:
        return uint8
    if arr.dtype == np.int8:
        return int8
    return int32 if np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_ else float32


def _to_buffer(data, dtype: str | None):
    """Returns (keepalive, address, shape, dtype) for contiguous host data."""
    if _is_numpy(data):
        import numpy as np

        if dtype is None:
            dtype = _numpy_dtype(data)
        arr = np.ascontiguousarray(data, dtype=dtype)  # newch dtype names are numpy names
        return arr, arr.ctypes.data, arr.shape, dtype
    if isinstance(data, array.array):
        if dtype is None:
            dtype = {"f": float32, "d": float32, "b": int8, "B": uint8}.get(data.typecode, int32)
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
_new_tensor = object.__new__


class Tensor:
    """A view of an engine tensor.

    Temporary tensors become invalid after ``step_end()``; touching one then
    raises NcError instead of reading freed memory.
    """

    __slots__ = ("_p", "_ctx", "_step", "_owner", "_base", "_free")

    @classmethod
    def _from_ptr(cls, ptr, ctx: Context, base: "Tensor | None" = None, persistent: "bool | None" = None) -> "Tensor":
        t = object.__new__(cls)
        t._p = ptr
        t._ctx = ctx
        if persistent is None:
            persistent = bool(lib.nc_is_persistent(ptr))
        if persistent:
            t._step = None
        else:
            # backward() releases graph memory except tensors Python still holds
            t._step = ctx.step_id
            live = ctx._live
            live[ptr] = live.get(ptr, 0) + 1
        t._owner = False  # only Context.wrap hands out ownership
        t._base = base    # keeps the tensor whose memory this one uses alive
        t._free = False   # set by Context.wrap / _w for no_grad results
        return t

    def __del__(self):
        # Persistent tensors own their memory exclusively (the engine never
        # makes persistent views), so the owning wrapper can free it.
        # Consequence for callers: never pass `make_tensor().ptr` straight
        # into a C call. CPython drops the temporary right after `.ptr` is
        # read, before the call runs; bind it to a name first.
        try:
            step = self._step
            if step is None:
                if self._owner and self._p and self._ctx._p:
                    lib.nc_release(self._p)
            elif step == self._ctx.step_id:
                live = self._ctx._live
                c = live.get(self._p, 0) - 1
                if c > 0:
                    live[self._p] = c
                else:
                    live.pop(self._p, None)
                    if self._free:
                        # inference: nobody holds this result any more; the engine
                        # keeps it anyway if backward needs it or it is a view
                        lib.nc_release_temp(self._p)
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
        """Wraps an op result (the hot path: same as Context.wrap, without the extra calls)."""
        ctx = self._ctx
        if not ptr:
            ctx.raise_error()
        t = _new_tensor(Tensor)
        t._p = ptr
        t._ctx = ctx
        t._base = None
        if ctx._pdepth:
            t._step = None
            t._owner = True
            t._free = False
        else:
            t._step = ctx.step_id
            t._owner = False
            t._free = not ctx._grad_on
            live = ctx._live
            live[ptr] = live.get(ptr, 0) + 1
        return t

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
        return v if self.dtype == float32 else int(v)

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

        return np.frombuffer(self._flat(), dtype=self.dtype).reshape(self.shape).copy()

    def __array__(self, dtype=None, copy=None):
        """numpy interop: np.asarray(t), np.array(t), np.stack([...]) (always a copy)."""
        if copy is False:
            raise ValueError("newch tensors cannot be shared with numpy without a copy")
        arr = self.numpy()
        return arr if dtype is None else arr.astype(dtype, copy=False)

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
        n = self.numel()
        if n != 1:  # also what `t in [a, b]` and `if t:` hit, as in PyTorch
            raise RuntimeError(f"the truth value of a tensor with {n} elements is ambiguous; "
                               "reduce it first, e.g. (t != 0).sum().item() > 0")
        return bool(self.item())

    # ---- comparison (elementwise, int32 0/1: there is no bool dtype) -----
    __hash__ = object.__hash__  # tensors stay usable as dict keys / in sets, as in PyTorch

    def _cmp(self, o, op: int):
        if isinstance(o, bool):
            o = int(o)
        if isinstance(o, int):
            o = tensor(o, dtype=int32, ctx=self._ctx)
        elif isinstance(o, float):
            o = tensor(o, dtype=float32, ctx=self._ctx)
        elif not isinstance(o, Tensor):
            return NotImplemented
        return self._w(lib.nc_compare(self.ptr, o.ptr, op))

    def _cmp_method(op: int):
        def method(self, other):
            out = self._cmp(other, op)
            if out is NotImplemented:
                raise TypeError(f"cannot compare a Tensor with {type(other).__name__}")
            return out
        return method

    def _cmp_operator(op: int):
        return lambda self, other: self._cmp(other, op)

    # methods raise for unsupported types; operators return NotImplemented, so `t == None` stays False
    eq, ne, lt, le, gt, ge = _cmp_method(0), _cmp_method(1), _cmp_method(2), _cmp_method(3), _cmp_method(4),         _cmp_method(5)
    __eq__, __ne__, __lt__, __le__, __gt__, __ge__ = _cmp_operator(0), _cmp_operator(1), _cmp_operator(2),         _cmp_operator(3), _cmp_operator(4), _cmp_operator(5)
    del _cmp_method, _cmp_operator

    # ---- dtype / device ---------------------------------------------------
    def to(self, *args, **kwargs) -> "Tensor":
        """to(device), to(dtype), to(device, dtype) or to(other). Only the CPU device exists."""
        dtype = kwargs.pop("dtype", None)
        device = kwargs.pop("device", None)
        kwargs.pop("non_blocking", None)
        _unsupported(copy=(kwargs.pop("copy", False), False))
        if kwargs:
            raise TypeError(f"to() got unexpected arguments {sorted(kwargs)}")
        for a in args:
            if isinstance(a, Tensor):
                dtype = a.dtype
            elif a in L.DTYPE_CODES:
                dtype = a
            else:
                device = a
        _check_device(device)
        if dtype is None or dtype == self.dtype:
            return self
        if dtype not in L.DTYPE_CODES:
            raise ValueError(f"unsupported dtype {dtype!r}")
        return self._w(lib.nc_to_dtype(self.ptr, L.DTYPE_CODES[dtype]))

    def float(self) -> "Tensor": return self.to(float32)
    def int(self) -> "Tensor": return self.to(int32)

    def long(self) -> "Tensor":
        """int32: newch has no int64 (library-wide rule)."""
        return self.to(int32)

    def cpu(self) -> "Tensor": return self

    def dim(self) -> int: return self.ndim

    # ---- arithmetic ------------------------------------------------------
    def __add__(self, o):
        if type(o) is Tensor:  # fast path: no conversion needed
            return self._w(lib.nc_add(self.ptr, o.ptr))
        if isinstance(o, Number):
            return self._w(lib.nc_add_scalar(self.ptr, float(o)))
        o = self._other(o)  # keep the operand alive across the call
        return self._w(lib.nc_add(self.ptr, o.ptr))

    __radd__ = __add__

    def __sub__(self, o):
        if type(o) is Tensor:
            return self._w(lib.nc_sub(self.ptr, o.ptr))
        if isinstance(o, Number):
            return self._w(lib.nc_add_scalar(self.ptr, -float(o)))
        o = self._other(o)  # keep the operand alive across the call
        return self._w(lib.nc_sub(self.ptr, o.ptr))

    def __rsub__(self, o):
        return self._other(o) - self

    def __mul__(self, o):
        if type(o) is Tensor:
            return self._w(lib.nc_mul(self.ptr, o.ptr))
        if isinstance(o, Number):
            return self._w(lib.nc_scale(self.ptr, float(o)))
        o = self._other(o)  # keep the operand alive across the call
        return self._w(lib.nc_mul(self.ptr, o.ptr))

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, Number):
            return self._w(lib.nc_scale(self.ptr, 1.0 / float(o)))
        o = self._other(o)  # keep the operand alive across the call
        return self._w(lib.nc_div(self.ptr, o.ptr))

    def __rtruediv__(self, o):
        return self._other(o) / self

    def __neg__(self):
        return self._w(lib.nc_neg(self.ptr))

    def __pow__(self, p):
        if not isinstance(p, Number):
            raise TypeError("only scalar exponents are supported")
        return self._w(lib.nc_pow_scalar(self.ptr, float(p)))

    def __matmul__(self, o):
        o = self._other(o)  # keep the operand alive across the call
        return self._w(lib.nc_matmul(self.ptr, o.ptr))

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

    def softmax(self, dim: int):
        self._last_dim(dim, "softmax")
        return self._w(lib.nc_softmax(self.ptr))

    def log_softmax(self, dim: int):
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

    def argmax(self, dim=None, keepdim: bool = False, *, axis=None):
        """Index of the maximum; flattened index when dim is None (as in PyTorch)."""
        dim = axis if dim is None else dim
        if keepdim and dim is None:
            raise NotImplementedError("argmax(keepdim=True) needs a dim")
        out = self._w(lib.nc_argmax(self.ptr, L.ALL_AXES if dim is None else int(dim)))
        if keepdim:
            shape = list(self.shape)
            shape[int(dim) % len(shape)] = 1
            out = out.reshape(*shape)
        return out

    # ---- shape -----------------------------------------------------------
    def reshape(self, *shape):
        """Returns a view when the layout allows it, otherwise a copy (as in PyTorch)."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        n, arr = L.shape_arg(shape)
        return self._view(lib.nc_reshape(self.ptr, n, arr))

    def view(self, *shape):
        """Same memory, new shape. Needs a contiguous tensor (else use reshape), as in PyTorch."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        if len(shape) == 1 and isinstance(shape[0], str):
            raise NotImplementedError("view(dtype) is not supported; use to(dtype)")
        if not lib.nc_is_contiguous(self.ptr):
            raise RuntimeError("view() needs a contiguous tensor; use reshape() or contiguous().view()")
        out = self.reshape(*shape)
        if out.is_persistent:
            raise NotImplementedError("view() inside persistent() would copy; use reshape()")
        return out

    def view_as(self, other: "Tensor"):
        return self.view(*other.shape)

    def unsqueeze(self, dim: int):
        shape = list(self.shape)
        d = dim + len(shape) + 1 if dim < 0 else dim
        if not 0 <= d <= len(shape):
            raise IndexError(f"unsqueeze dim {dim} out of range for {len(shape)}-d tensor")
        shape.insert(d, 1)
        return self.reshape(*shape)

    def squeeze(self, dim=None):
        shape = list(self.shape)
        if dim is None:
            keep = [s for s in shape if s != 1]
        else:
            dims = {d % len(shape) for d in (dim if isinstance(dim, (list, tuple)) else (dim,))} if shape else set()
            keep = [s for i, s in enumerate(shape) if not (i in dims and s == 1)]
        return self.reshape(*keep) if len(keep) != len(shape) else self

    def permute(self, *dims):
        """Reorders dimensions as a view (a series of transposes, no copy)."""
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        n = self.ndim
        dims = tuple(d % n for d in dims) if n else ()
        if sorted(dims) != list(range(n)):
            raise ValueError(f"permute needs a permutation of {n} dims, got {dims}")
        out, cur = self, list(range(n))
        for i, want in enumerate(dims):
            j = cur.index(want)
            if j != i:
                out = out.transpose(i, j)
                cur[i], cur[j] = cur[j], cur[i]
        return out

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
    def backward(self, gradient=None, retain_graph=None, create_graph: bool = False, inputs=None) -> None:
        """Backpropagates from this scalar. Like PyTorch, the graph's memory is released as
        it goes (activations nobody holds, intermediate grads), so a second backward through
        the same graph needs retain_graph=True on the first one. Tensors you still reference
        are never released."""
        _unsupported(create_graph=(create_graph, False), inputs=(inputs, None))
        if gradient is not None:
            raise NotImplementedError("backward(gradient=...) is not supported; call it on a scalar loss")
        ctx = self._ctx
        if retain_graph:
            st = lib.nc_backward_ex(self.ptr, 2, None, 0)
        else:
            live = list(ctx._live)
            st = lib.nc_backward_ex(self.ptr, 1, L.ptr_array(live), len(live))
        if st != 0:
            ctx.raise_error()

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
        if data.dtype == float32 and dtype in (None, float32) and ctx in (None, data.ctx):
            src = data.detach()  # must outlive the call: a persistent temp is freed on GC
            t = data.ctx.wrap(lib.nc_add_scalar(src.ptr, 0.0))  # engine-side copy
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


def full(size, fill_value, *, dtype: str | None = None, ctx: Context | None = None) -> Tensor:
    """Tensor filled with fill_value. dtype follows the value when not given (int -> int32)."""
    if isinstance(fill_value, bool):
        raise TypeError("bool tensors are not supported; use 0 / 1")
    ctx = _ctx(ctx)
    size = tuple(size) if isinstance(size, (list, tuple)) else (size,)
    if dtype is None:
        dtype = int32 if isinstance(fill_value, int) else float32
    if dtype == float32:
        n, arr = L.shape_arg(size)
        return ctx.wrap(lib.nc_full(ctx._p, float(fill_value), n, arr))
    count = 1
    for d in size:
        count *= int(d)
    buf = array.array(L.ARRAY_CODE[dtype], [int(fill_value)]) * count
    return tensor_from_bytes(buf.tobytes(), size, dtype, ctx)


def ones(*size, dtype: str = float32, ctx: Context | None = None) -> Tensor:
    return full(_shape(size), 1, dtype=dtype, ctx=ctx)


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
        return ctx.wrap(lib.nc_parameter(ctx._p, None, n, arr), persistent=True)
    src = init if isinstance(init, Tensor) else tensor(init, dtype=float32, ctx=ctx)
    n, arr = L.shape_arg(src.shape)
    return ctx.wrap(lib.nc_parameter(ctx._p, src.ptr, n, arr), persistent=True)


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
# Activations (torch.nn.functional names and arguments). Piecewise-linear ones
# (leaky_relu, relu6, hardsigmoid, hardswish) need no exp, which matters on
# small cores; silu and gelu cost one exp / erf / tanh per element.
def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    _unsupported(inplace=(inplace, False))
    return input._w(lib.nc_leaky_relu(input.ptr, float(negative_slope)))


def relu6(input: Tensor, inplace: bool = False) -> Tensor:
    _unsupported(inplace=(inplace, False))
    return input._w(lib.nc_relu6(input.ptr))


def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    _unsupported(inplace=(inplace, False))
    return input._w(lib.nc_hardsigmoid(input.ptr))


def hardswish(input: Tensor, inplace: bool = False) -> Tensor:
    _unsupported(inplace=(inplace, False))
    return input._w(lib.nc_hardswish(input.ptr))


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    _unsupported(inplace=(inplace, False))
    return input._w(lib.nc_silu(input.ptr))


def gelu(input: Tensor, approximate: str = "none") -> Tensor:
    """x * Phi(x): exact (erf) by default, approximate="tanh" for the tanh form."""
    if approximate not in ("none", "tanh"):
        raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
    return input._w(lib.nc_gelu(input.ptr, int(approximate == "tanh")))


def _implicit_dim(x: Tensor, dim, name: str) -> int:
    if dim is not None:
        return dim
    # PyTorch's legacy rule when dim is omitted (it warns as well)
    warnings.warn(f"{name}() without dim picks an axis implicitly; pass dim explicitly", stacklevel=3)
    return 0 if x.ndim in (0, 1, 3) else 1


def _check_device(device) -> None:
    if device is not None and str(device).split(":")[0] != "cpu":
        raise NotImplementedError(f"device {device!r} is not available; newch runs on the CPU only")


def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    return input.flatten(start_dim, end_dim)


def argmax(input: Tensor, dim=None, keepdim: bool = False) -> Tensor:
    return input.argmax(dim, keepdim)


def cat(tensors, dim: int = 0) -> Tensor:
    """Concatenates along dim (engine op; differentiable for float32)."""
    tensors = list(tensors)
    if not tensors:
        raise ValueError("cat() needs at least one tensor")
    ctx = tensors[0].ctx
    while True:  # the engine op takes up to 6 inputs per call
        group, rest = tensors[:6], tensors[6:]
        ptrs = L.ptr_array(t.ptr for t in group)
        out = ctx.wrap(lib.nc_cat(len(group), ptrs, int(dim)))
        if not rest:
            return out
        tensors = [out] + rest


def stack(tensors, dim: int = 0) -> Tensor:
    """Joins equal-shaped tensors along a new dim."""
    return cat([t.unsqueeze(dim) for t in tensors], dim)


def softmax(input: Tensor, dim=None) -> Tensor: return input.softmax(_implicit_dim(input, dim, "softmax"))
def log_softmax(input: Tensor, dim=None) -> Tensor: return input.log_softmax(_implicit_dim(input, dim, "log_softmax"))


def _unsupported(**kwargs) -> None:
    """Raises for arguments that exist in the original API but are not implemented here."""
    for name, (value, default) in kwargs.items():
        if value != default:
            raise NotImplementedError(f"{name}={value!r} is not supported (only {default!r})")


def _pair(v, name: str) -> tuple:
    if isinstance(v, str):
        raise NotImplementedError(f"string {name} ({v!r}) is not supported; pass an int or (h, w)")
    if isinstance(v, int):
        return v, v
    v = tuple(int(i) for i in v)
    if len(v) != 2:
        raise ValueError(f"{name} must be an int or a pair")
    return v


def conv2d(input: Tensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1,
           groups: int = 1) -> Tensor:
    """NCHW convolution (torch.nn.functional.conv2d); dilation and groups must be 1."""
    _unsupported(dilation=(_pair(dilation, "dilation"), (1, 1)), groups=(groups, 1))
    (sh, sw), (ph, pw) = _pair(stride, "stride"), _pair(padding, "padding")
    return input._w(lib.nc_conv2d(input.ptr, weight.ptr, bias.ptr if bias is not None else None, sh, sw, ph, pw))


def _window(kernel_size, stride, padding):
    (kh, kw) = _pair(kernel_size, "kernel_size")
    (sh, sw) = _pair(stride if stride not in (None, (), []) else kernel_size, "stride")
    (ph, pw) = _pair(padding, "padding")
    return kh, kw, sh, sw, ph, pw


def max_pool2d(input: Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode: bool = False,
               return_indices: bool = False) -> Tensor:
    _unsupported(dilation=(_pair(dilation, "dilation"), (1, 1)), ceil_mode=(ceil_mode, False),
                 return_indices=(return_indices, False))
    return input._w(lib.nc_max_pool2d(input.ptr, *_window(kernel_size, stride, padding)))


def avg_pool2d(input: Tensor, kernel_size, stride=None, padding=0, ceil_mode: bool = False,
               count_include_pad: bool = True, divisor_override=None) -> Tensor:
    """Average pooling; padded cells count toward the average (count_include_pad=True)."""
    _unsupported(ceil_mode=(ceil_mode, False), count_include_pad=(count_include_pad, True),
                 divisor_override=(divisor_override, None))
    return input._w(lib.nc_avg_pool2d(input.ptr, *_window(kernel_size, stride, padding)))


def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    _unsupported(inplace=(inplace, False))
    if not training:
        return input
    return input._w(lib.nc_dropout(input.ptr, float(p)))


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


def cross_entropy(input: Tensor, target, weight=None, size_average=None, ignore_index: int = -100, reduce=None,
                  reduction: str = "mean", label_smoothing: float = 0.0) -> Tensor:
    """Cross-entropy of logits input [N, C] and int32 class indices target [N].

    reduction "mean" or "sum"; label_smoothing mixes in the uniform distribution.
    Labels outside [0, C) raise (so ignore_index only works as the default -100).
    """
    _unsupported(weight=(weight, None), size_average=(size_average, None), reduce=(reduce, None),
                 ignore_index=(ignore_index, -100))
    if reduction not in ("mean", "sum"):
        raise NotImplementedError(f"reduction={reduction!r} is not supported (use 'mean' or 'sum')")
    target = _class_indices(target, input)
    loss = input._w(lib.nc_cross_entropy(input.ptr, target.ptr))
    if label_smoothing:
        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError("label_smoothing must be in [0, 1]")
        uniform = (input.log_softmax(-1).mean(-1)).mean() * -1.0  # mean over classes and rows
        loss = loss * (1.0 - label_smoothing) + uniform * label_smoothing
    return loss * float(target.shape[0]) if reduction == "sum" else loss


def nll_loss(input: Tensor, target, weight=None, size_average=None, ignore_index: int = -100, reduce=None,
             reduction: str = "mean") -> Tensor:
    """Negative log-likelihood of log-probabilities input [N, C] and class indices target [N]."""
    _unsupported(weight=(weight, None), size_average=(size_average, None), reduce=(reduce, None),
                 ignore_index=(ignore_index, -100))
    if reduction not in ("mean", "sum"):
        raise NotImplementedError(f"reduction={reduction!r} is not supported (use 'mean' or 'sum')")
    target = _class_indices(target, input)
    return input._w(lib.nc_nll_loss(input.ptr, target.ptr, int(reduction == "sum")))


def _class_indices(target, input: Tensor) -> Tensor:
    """int32 class indices; 1-byte integer labels (uint8 datasets) are widened."""
    if not isinstance(target, Tensor):
        return tensor(target, dtype=int32, ctx=input.ctx)
    return target.int() if target.dtype in (uint8, int8) else target


def mse_loss(input: Tensor, target, size_average=None, reduce=None, reduction: str = "mean") -> Tensor:
    """Squared error with reduction "mean", "sum" or "none"."""
    _unsupported(size_average=(size_average, None), reduce=(reduce, None))
    target = input._other(target)
    if reduction == "mean":
        return input._w(lib.nc_mse_loss(input.ptr, target.ptr))
    d = input - target
    if reduction == "sum":
        return (d * d).sum()
    if reduction == "none":
        return d * d
    raise ValueError(f"unknown reduction {reduction!r}")


# --------------------------------------------------------------------------
# Tensor files (NCW1)
# --------------------------------------------------------------------------


_TREE = "__newch_tree__"  # entry holding the layout of a nested checkpoint (UTF-8 JSON, uint8)


def _tree_encode(obj, path: str, out: list, names: set):
    if isinstance(obj, Tensor):
        name, k = path or "tensor", 1
        while name in names:  # keys containing dots can produce the same path twice
            name, k = f"{path}#{k}", k + 1
        names.add(name)
        out.append((name, obj))
        return {"T": name}
    sub = (lambda key: f"{path}.{key}" if path else str(key))
    if isinstance(obj, dict):
        items = []
        for key, v in obj.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise TypeError(f"save_tensors: dict keys must be str or int, got {key!r} at '{path}'")
            items.append([key, _tree_encode(v, sub(key), out, names)])
        return {"d": items}
    if isinstance(obj, (list, tuple)):
        return {"l" if isinstance(obj, list) else "t": [_tree_encode(v, sub(i), out, names) for i, v in enumerate(obj)]}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return {"v": obj}
    raise TypeError(f"save_tensors: cannot store {type(obj).__name__} at '{path}' "
                    "(Tensors, dicts, lists, tuples, numbers, strings and None only)")


def _tree_decode(node, tensors: dict):
    ((kind, val),) = node.items()
    if kind == "T":
        return tensors[val]
    if kind == "d":
        return {key: _tree_decode(v, tensors) for key, v in val}
    if kind in ("l", "t"):
        items = [_tree_decode(v, tensors) for v in val]
        return items if kind == "l" else tuple(items)
    return val


def save_tensors(tensors: dict, path) -> None:
    """Writes {name: Tensor} to an NCW1 file (format: docs/FORMAT.md).

    Whole checkpoints work too: nested dicts, lists and tuples of Tensors, numbers, strings and None,
    e.g. ``{"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": 3}``.
    load_tensors returns the same structure.
    """
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("save_tensors: expected a non-empty dict")
    if all(isinstance(k, str) and isinstance(v, Tensor) for k, v in tensors.items()):
        if _TREE in tensors:
            raise ValueError(f"save_tensors: the name {_TREE!r} is reserved")
        items, tree = list(tensors.items()), None
    else:
        items = []
        tree = json.dumps(_tree_encode(tensors, "", items, {_TREE})).encode("utf-8")
    ctx = items[0][1].ctx if items else default_context()
    if tree is not None:
        layout = tensor_from_bytes(tree, (len(tree),), uint8, ctx)  # bound: must outlive nc_save
        items.append((_TREE, layout))
    st = lib.nc_save(ctx._p, str(path).encode("utf-8"), len(items), L.name_array(k for k, _ in items),
                     L.ptr_array(t.ptr for _, t in items))
    if st != 0:
        ctx.raise_error()


def load_tensors(path, ctx: Context | None = None) -> dict:
    """Reads an NCW1 file into {name: persistent Tensor}, or the nested structure save_tensors wrote."""
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
    if _TREE in out:
        return _tree_decode(json.loads(out.pop(_TREE)._flat().tobytes()), out)
    return out
