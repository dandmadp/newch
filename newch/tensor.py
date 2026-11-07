"""Tensor, Context and functional ops: the PyTorch-style layer."""

from __future__ import annotations

import array
import contextlib
import ctypes
import json
import math
import numbers
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


def _dtype_code(dtype) -> int:
    try:
        return L.DTYPE_CODES[dtype]
    except (KeyError, TypeError):
        raise ValueError(f"unsupported dtype {dtype!r}; newch has float32, int32, int8 and uint8") from None


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

    def __deepcopy__(self, memo):
        return self  # copied modules keep using the same engine context

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
    elif isinstance(data, Tensor) or (_is_numpy(data) and getattr(data, "ndim", 0)):
        _flatten(data.tolist(), shape, out, depth)  # e.g. tensor([t0, t1]) or a list of numpy rows
    else:
        if depth != len(shape):
            raise ValueError("ragged nested sequence: inconsistent depth")
        out.append(data)


def _is_int(v) -> bool:
    """Python ints and bools, numpy integer and bool scalars."""
    return isinstance(v, numbers.Integral) or (_is_numpy(v) and getattr(v, "dtype", None) is not None
                                               and v.dtype.kind in "biu")


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
        arr = np.asarray(data, dtype=dtype, order="C")  # newch dtype names are numpy names; keeps 0-d
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
        dtype = float32 if any(type(v) is not int and not _is_int(v) for v in flat) else int32
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


Number = numbers.Number  # int, float, bool and numpy scalars
_new_tensor = object.__new__


class Tensor:
    """A view of an engine tensor.

    Temporary tensors become invalid after ``step_end()``; touching one then
    raises NcError instead of reading freed memory.
    """

    __slots__ = ("_p", "_ctx", "_step", "_owner", "_base", "_free")
    # numpy defers to Tensor for `ndarray + tensor` etc. (as for torch), so the result stays a Tensor
    __array_priority__ = 1000

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

    def __deepcopy__(self, memo):
        """copy.deepcopy (e.g. of a Module): a new tensor with the same values, persistent if
        this one is, and still a parameter if this one requires grad. Grads are not copied."""
        if self.is_persistent:
            with persistent(self._ctx):
                if self.requires_grad:
                    return parameter(self, ctx=self._ctx)
                return tensor(self, ctx=self._ctx)
        return tensor(self.detach(), ctx=self._ctx)

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

    def __int__(self) -> int:
        return int(self.item())

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d tensor")
        return (self[i] for i in range(self.shape[0]))

    def __bool__(self) -> bool:
        n = self.numel()
        if n != 1:  # also what `t in [a, b]` and `if t:` hit, as in PyTorch
            raise RuntimeError(f"the truth value of a tensor with {n} elements is ambiguous; "
                               "reduce it first, e.g. (t != 0).sum().item() > 0")
        return bool(self.item())

    # ---- comparison (elementwise, int32 0/1: there is no bool dtype) -----
    __hash__ = object.__hash__  # tensors stay usable as dict keys / in sets, as in PyTorch

    def _cmp(self, o, op: int):
        if isinstance(o, Tensor):
            pass
        elif type(o) is float:  # fast paths before the numbers ABCs
            o = tensor(o, dtype=float32, ctx=self._ctx)
        elif type(o) is int:
            o = tensor(o, dtype=int32, ctx=self._ctx)
        elif _is_int(o):
            o = tensor(int(o), dtype=int32, ctx=self._ctx)
        elif isinstance(o, numbers.Real):
            o = tensor(float(o), dtype=float32, ctx=self._ctx)
        elif _is_numpy(o):
            o = tensor(o, ctx=self._ctx)
        else:
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
            elif a in L.DTYPE_CODES or not _is_device(a):
                dtype = a
            else:
                device = a
        _check_device(device)
        if dtype is None or dtype == self.dtype:
            return self
        return self._w(lib.nc_to_dtype(self.ptr, _dtype_code(dtype)))

    def float(self) -> "Tensor": return self.to(float32)
    def int(self) -> "Tensor": return self.to(int32)

    def long(self) -> "Tensor":
        """int32: newch has no int64 (library-wide rule)."""
        return self.to(int32)

    def cpu(self) -> "Tensor": return self

    def dim(self) -> int: return self.ndim

    # ---- arithmetic ------------------------------------------------------
    # The engine computes in float32. When it refuses an integer operand (the NULL result below),
    # _promoted applies torch's type promotion: int with float, and true division, give float32;
    # int with int would give an integer tensor in torch, which is not supported and raises.
    def __add__(self, o):
        if type(o) is Tensor:  # fast path: no conversion needed
            p = lib.nc_add(self.ptr, o.ptr)
        elif isinstance(o, Number):
            p = lib.nc_add_scalar(self.ptr, float(o))
        else:
            o = self._other(o)  # keep the operand alive across the call
            p = lib.nc_add(self.ptr, o.ptr)
        return self._w(p) if p else self._promoted(Tensor.__add__, o)

    __radd__ = __add__

    def __sub__(self, o):
        if type(o) is Tensor:
            p = lib.nc_sub(self.ptr, o.ptr)
        elif isinstance(o, Number):
            p = lib.nc_add_scalar(self.ptr, -float(o))
        else:
            o = self._other(o)  # keep the operand alive across the call
            p = lib.nc_sub(self.ptr, o.ptr)
        return self._w(p) if p else self._promoted(Tensor.__sub__, o)

    def __rsub__(self, o):
        if isinstance(o, Number) and _is_int(o) and self.dtype != float32:
            self._int_arith_error()
        return self._other(o) - self

    def __mul__(self, o):
        if type(o) is Tensor:
            p = lib.nc_mul(self.ptr, o.ptr)
        elif isinstance(o, Number):
            p = lib.nc_scale(self.ptr, float(o))
        else:
            o = self._other(o)  # keep the operand alive across the call
            p = lib.nc_mul(self.ptr, o.ptr)
        return self._w(p) if p else self._promoted(Tensor.__mul__, o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, Number) and abs(float(o)) >= 1e-38:  # 1/o fits in float32
            p = lib.nc_scale(self.ptr, 1.0 / float(o))
        else:
            o = self._other(o)  # keep the operand alive across the call
            p = lib.nc_div(self.ptr, o.ptr)
        return self._w(p) if p else self._promoted(Tensor.__truediv__, o, true_div=True)

    def __rtruediv__(self, o):
        return self._other(o) / self

    def __neg__(self):
        p = lib.nc_neg(self.ptr)
        return self._w(p) if p else self._promoted(lambda a, _: -a, None)

    def __pow__(self, p):
        if not isinstance(p, Number):
            raise TypeError("only scalar exponents are supported")
        r = lib.nc_pow_scalar(self.ptr, float(p))
        return self._w(r) if r else self._promoted(Tensor.__pow__, p)

    def __matmul__(self, o):
        o = self._other(o)  # keep the operand alive across the call
        p = lib.nc_matmul(self.ptr, o.ptr)
        return self._w(p) if p else self._promoted(Tensor.__matmul__, o)

    def _promoted(self, op, o, true_div: bool = False) -> "Tensor":
        """Called after the engine refused the operands: retry in float32 where torch promotes."""
        a_int = self.dtype != float32
        if isinstance(o, Tensor):
            b_int = o.dtype != float32
        else:
            b_int = o is None or _is_int(o)
        if not a_int and not b_int:
            self._ctx.raise_error()  # not a dtype problem: the engine's own message
        lib.nc_clear_error(self._ctx._p)
        if b_int and a_int and not true_div:
            self._int_arith_error()
        a = self.float() if a_int else self
        b = o.float() if isinstance(o, Tensor) and b_int else o
        return op(a, b)

    def _int_arith_error(self):
        raise NcError(f"arithmetic on {self.dtype} tensors is not supported (torch would keep an integer "
                      "result; the engine computes in float32): convert with .float() first")

    def __rmatmul__(self, o):
        return self._other(o) @ self

    # ---- math ------------------------------------------------------------
    def exp(self): return self._w(lib.nc_exp(self.ptr))
    def log(self): return self._w(lib.nc_log(self.ptr))
    def sqrt(self): return self._w(lib.nc_sqrt(self.ptr))
    def relu(self): return self._w(lib.nc_relu(self.ptr))
    def sigmoid(self): return self._w(lib.nc_sigmoid(self.ptr))
    def tanh(self): return self._w(lib.nc_tanh(self.ptr))
    def _over_dim(self, dim: int, fn) -> "Tensor":
        """Runs a last-dim engine op over any dim (moved last and back: transposes are views)."""
        n = self.ndim
        if not -max(n, 1) <= dim < max(n, 1):
            raise IndexError(f"dim {dim} out of range for a {n}-d tensor")
        d = dim % n if n else 0
        if d == n - 1 or n == 0:
            return self._w(fn(self.ptr))
        moved = self.transpose(d, -1)
        return moved._w(fn(moved.ptr)).transpose(d, -1)

    def softmax(self, dim: int):
        return self._over_dim(dim, lib.nc_softmax)

    def log_softmax(self, dim: int):
        return self._over_dim(dim, lib.nc_log_softmax)

    # Reductions take PyTorch's `dim`; numpy's `axis` is accepted as an alias.
    def _reduce(self, fn, dim, keepdim, axis):
        dim = axis if dim is None else dim
        if isinstance(dim, (tuple, list)):
            return self._reduce_dims(fn, dim, keepdim)
        return self._w(fn(self.ptr, L.ALL_AXES if dim is None else int(dim), int(keepdim)))

    def _reduce_dims(self, fn, dims, keepdim):
        """dim=(d0, d1, ...): one reduction per dim, last dim first (a mean of means is the mean)."""
        n = self.ndim
        if not dims:
            return self._reduce(fn, None, keepdim, None)
        for d in dims:
            if not -max(n, 1) <= int(d) < max(n, 1):
                raise IndexError(f"dim {d} out of range for a {n}-d tensor")
        norm = [int(d) % n if n else 0 for d in dims]
        if len(set(norm)) != len(norm):
            raise RuntimeError(f"dim {dims} lists a dimension more than once")
        out = self
        for d in sorted(norm, reverse=True):
            out = out._w(fn(out.ptr, d, 1))
        if keepdim:
            return out
        return out.reshape(*[s for i, s in enumerate(self.shape) if i not in norm])

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

    def masked_fill(self, mask, value) -> "Tensor":
        """Copy with ``value`` where ``mask`` is set (torch.Tensor.masked_fill). mask: an integer 0/1
        tensor such as ``a == b`` gives (newch has no bool dtype), broadcast onto this tensor;
        value may be ``float("-inf")`` for attention masks."""
        if not isinstance(mask, Tensor):
            mask = tensor(mask, dtype=int32, ctx=self._ctx)
        elif mask.dtype == float32:
            raise TypeError("masked_fill needs an integer 0/1 mask (e.g. from ==, <), got float32")
        return self._w(lib.nc_masked_fill(self.ptr, mask.ptr, float(value)))

    def index_select(self, dim: int, index):
        """Rows along ``dim`` (only dim=0 is implemented); grads scatter-add back."""
        if dim not in (0, -self.ndim):
            raise NotImplementedError("index_select is only implemented for dim=0")
        if not isinstance(index, Tensor):
            index = tensor(index, dtype=int32, ctx=self._ctx)
        return self._w(lib.nc_index_select(self.ptr, index.ptr))

    def __getitem__(self, key):
        """torch/numpy basic indexing: ints, slices (step >= 1), None, ``...``, plus one integer index
        list / numpy array / int32 tensor. Ints and slices give views (no copy); an index list copies."""
        if type(key) is int and self.ndim:  # fast path: x[i]
            return self._select(0, key)
        if not isinstance(key, tuple):
            key = (key,)
        key = [int(k) if isinstance(k, Tensor) and k.ndim == 0 and k.dtype != float32 else k for k in key]
        if sum(k is Ellipsis for k in key) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        real = sum(1 for k in key if k is not None and k is not Ellipsis)
        if real > self.ndim:
            raise IndexError(f"too many indices for a {self.ndim}-d tensor: {real} were given")
        if Ellipsis in key:
            i = key.index(Ellipsis)
            key[i:i + 1] = [slice(None)] * (self.ndim - real)
        out, d = self, 0
        adv = None          # (axis in out, index) of the one index list / tensor
        picks = []          # positions in key of ints and the index list (numpy's "advanced" indices)
        for pos, k in enumerate(key):
            if k is None:
                out = out.unsqueeze(d)
                d += 1
            elif isinstance(k, slice):
                size = out.shape[d]
                start, stop, step = k.indices(size)
                if step < 1:
                    raise ValueError("step must be greater than zero")
                length = len(range(start, stop, step))
                if (start, length, step) != (0, size, 1):
                    out = out._view(lib.nc_slice(out.ptr, d, start, length, step))
                d += 1
            elif isinstance(k, bool) or type(k).__name__ == "bool_":
                raise IndexError("boolean indices are not supported; use masked_fill or integer indices")
            elif _is_int(k):
                out = out._select(d, int(k))
                picks.append(pos)
            elif isinstance(k, (list, Tensor)) or _is_numpy(k):
                if adv is not None:
                    raise NotImplementedError("only one index list / tensor per indexing expression is supported")
                adv = (d, k)
                picks.append(pos)
                d += 1
            else:
                raise TypeError(f"unsupported index type {type(k).__name__}")
        if adv is None:
            return out
        a, k = adv
        n = out.ndim
        moved = out if a == 0 else out.permute(a, *[i for i in range(n) if i != a])
        res = moved._take_rows(k)
        if a == 0:
            return res
        # numpy rule: the index dims stay in place when ints and the list are adjacent in the key,
        # otherwise they go first
        if picks != list(range(picks[0], picks[0] + len(picks))):
            return res
        k_nd = res.ndim - (n - 1)
        rest = list(range(k_nd, res.ndim))
        return res.permute(*rest[:a], *range(k_nd), *rest[a:])

    def _select(self, dim: int, i: int) -> "Tensor":
        """x.select(dim, i): a view without axis dim."""
        size = self.shape[dim]
        if not -size <= i < size:
            raise IndexError(f"index {i} out of range for dim {dim} of size {size}")
        part = self._view(lib.nc_slice(self.ptr, dim, i % size, 1, 1))
        shape = list(self.shape)
        del shape[dim]
        return part.reshape(*shape)

    def _take_rows(self, key) -> "Tensor":
        """self[key] for integer index lists/arrays/tensors of any shape; negative indices count from the end."""
        n = self.shape[0] if self.ndim else 0
        if isinstance(key, Tensor):
            if key.dtype != int32:
                raise IndexError(f"tensors used as indices must be int32, got {key.dtype} "
                                 "(boolean masks are not supported)")
            shape, flat = key.shape, key._flat()
        else:
            shape, flat = _to_buffer(key, None)[2], []
            _flatten(key.tolist() if _is_numpy(key) else key, [], flat)
            if any(isinstance(v, bool) or type(v).__name__ == "bool_" for v in flat):
                raise IndexError("boolean masks are not supported as indices; use masked_fill or integer indices")
            if any(not _is_int(v) for v in flat):
                raise IndexError("indices must be integers")
        if any(v < 0 for v in flat):
            if any(v < -n for v in flat):
                raise IndexError(f"index {min(flat)} out of range for axis 0 of size {n}")
            flat = [v + n if v < 0 else v for v in flat]
            key = None
        if key is None or len(shape) != 1 or not isinstance(key, Tensor):
            key = tensor(list(flat), dtype=int32, ctx=self._ctx)
        out = self.index_select(0, key)
        return out if len(shape) == 1 else out.reshape(*shape, *self.shape[1:])

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
    if dtype is not None:
        _dtype_code(dtype)  # clear error for float64, bool, ...
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
        dtype = dtype or (int32 if _is_int(data) else float32)
        keep, addr, shape, dtype = _to_buffer([data], dtype)
        shape = ()
    else:
        keep, addr, shape, dtype = _to_buffer(data, dtype)
    code = _dtype_code(dtype)
    n, arr = L.shape_arg(shape)
    t = ctx.wrap(lib.nc_from_data(ctx._p, code, addr, n, arr))
    del keep
    if requires_grad:
        t.requires_grad = True
    return t


def zeros(*shape, dtype: str = float32, ctx: Context | None = None) -> Tensor:
    ctx = _ctx(ctx)
    n, arr = L.shape_arg(_shape(shape))
    return ctx.wrap(lib.nc_zeros(ctx._p, _dtype_code(dtype), n, arr))


def full(size, fill_value, *, dtype: str | None = None, ctx: Context | None = None) -> Tensor:
    """Tensor filled with fill_value. dtype follows the value when not given (int -> int32)."""
    if isinstance(fill_value, bool):
        raise TypeError("bool tensors are not supported; use 0 / 1")
    ctx = _ctx(ctx)
    size = tuple(size) if isinstance(size, (list, tuple)) else (size,)
    if dtype is None:
        dtype = int32 if _is_int(fill_value) else float32
    _dtype_code(dtype)
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


def _randn(shape, std: float, ctx: Context | None) -> Tensor:
    ctx = _ctx(ctx)
    n, arr = L.shape_arg(shape)
    return ctx.wrap(lib.nc_randn(ctx._p, float(std), n, arr))


def _rand_uniform(shape, low: float, high: float, ctx: Context | None) -> Tensor:
    ctx = _ctx(ctx)
    n, arr = L.shape_arg(shape)
    return ctx.wrap(lib.nc_rand_uniform(ctx._p, float(low), float(high), n, arr))


def _random(t: Tensor, dtype: str, requires_grad: bool) -> Tensor:
    if dtype != float32:
        raise NotImplementedError(f"dtype={dtype!r} is not supported for random tensors (only float32)")
    if requires_grad:
        t.requires_grad = True
    return t


def randn(*size, dtype: str = float32, requires_grad: bool = False, ctx: Context | None = None) -> Tensor:
    """Samples from N(0, 1) (torch.randn). Other spreads: ``randn(...) * std + mean``."""
    return _random(_randn(_shape(size), 1.0, ctx), dtype, requires_grad)


def rand(*size, dtype: str = float32, requires_grad: bool = False, ctx: Context | None = None) -> Tensor:
    """Samples from U[0, 1) (torch.rand). Other ranges: ``rand(...) * (high - low) + low``."""
    return _random(_rand_uniform(_shape(size), 0.0, 1.0, ctx), dtype, requires_grad)


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
    code = _dtype_code(dtype)
    n, arr = L.shape_arg(shape)
    addr = ctypes.cast(ctypes.c_char_p(buf), ctypes.c_void_p) if buf else None
    return ctx.wrap(lib.nc_from_data(ctx._p, code, addr, n, arr))


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


_DEVICES = {"cpu", "cuda", "mps", "xpu", "xla", "meta", "hip", "hpu", "ipu", "mtia", "vulkan", "privateuseone"}


def _is_device(a) -> bool:
    """A device string/object (cpu, cuda:0, torch.device(...)); other strings are dtype names."""
    return not isinstance(a, str) or a.split(":")[0] in _DEVICES


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
    return _unbatched(input, lambda x: x._w(lib.nc_conv2d(x.ptr, weight.ptr, bias.ptr if bias is not None else None,
                                                          sh, sw, ph, pw)))


def _unbatched(input: Tensor, fn) -> Tensor:
    """Runs a 4-D [N, C, H, W] op on [C, H, W] input too (as torch does)."""
    if input.ndim != 3:
        return fn(input)
    out = fn(input.unsqueeze(0))
    return out.reshape(*out.shape[1:])


def _window(kernel_size, stride, padding):
    (kh, kw) = _pair(kernel_size, "kernel_size")
    (sh, sw) = _pair(stride if stride not in (None, (), []) else kernel_size, "stride")
    (ph, pw) = _pair(padding, "padding")
    return kh, kw, sh, sw, ph, pw


def max_pool2d(input: Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode: bool = False,
               return_indices: bool = False) -> Tensor:
    _unsupported(dilation=(_pair(dilation, "dilation"), (1, 1)), ceil_mode=(ceil_mode, False),
                 return_indices=(return_indices, False))
    win = _window(kernel_size, stride, padding)
    return _unbatched(input, lambda x: x._w(lib.nc_max_pool2d(x.ptr, *win)))


def avg_pool2d(input: Tensor, kernel_size, stride=None, padding=0, ceil_mode: bool = False,
               count_include_pad: bool = True, divisor_override=None) -> Tensor:
    """Average pooling; padded cells count toward the average (count_include_pad=True)."""
    _unsupported(ceil_mode=(ceil_mode, False), count_include_pad=(count_include_pad, True),
                 divisor_override=(divisor_override, None))
    win = _window(kernel_size, stride, padding)
    return _unbatched(input, lambda x: x._w(lib.nc_avg_pool2d(x.ptr, *win)))


def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    _unsupported(inplace=(inplace, False))
    if not training:
        return input
    return input._w(lib.nc_dropout(input.ptr, float(p)))


def batch_norm(input: Tensor, running_mean: Tensor, running_var: Tensor, weight: Tensor | None = None,
               bias: Tensor | None = None, training: bool = False, momentum: float = 0.1,
               eps: float = 1e-5) -> Tensor:
    """torch.nn.functional.batch_norm. In training mode the running buffers are updated in place."""
    c = running_mean.shape[0]
    if weight is None:
        weight = full((c,), 1.0, ctx=input.ctx)
    if bias is None:
        bias = zeros(c, ctx=input.ctx)
    return input._w(lib.nc_batch_norm(input.ptr, weight.ptr, bias.ptr, running_mean.ptr, running_var.ptr,
                                      int(training), float(momentum), float(eps)))


def layer_norm(input: Tensor, normalized_shape, weight: Tensor | None = None, bias: Tensor | None = None,
               eps: float = 1e-5) -> Tensor:
    """Normalizes over the trailing ``normalized_shape`` dims (torch.nn.functional.layer_norm)."""
    shape = (int(normalized_shape),) if isinstance(normalized_shape, int) else tuple(int(d) for d in normalized_shape)
    if not shape or tuple(input.shape[len(input.shape) - len(shape):]) != shape or len(shape) > input.ndim:
        raise RuntimeError(f"layer_norm: input {input.shape} does not end with normalized_shape {shape}")
    return input._w(lib.nc_layer_norm(input.ptr, len(shape), weight.ptr if weight is not None else None,
                                      bias.ptr if bias is not None else None, float(eps)))


def _causal_keep(L: int, S: int, ctx) -> Tensor:
    """int32 [L, S], 1 where key j <= query i (torch's top-left aligned causal mask)."""
    rows = tensor(list(range(L)), dtype=int32, ctx=ctx).reshape(L, 1)
    cols = tensor(list(range(S)), dtype=int32, ctx=ctx)
    return cols <= rows


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, attn_mask=None, dropout_p: float = 0.0,
                                 is_causal: bool = False, scale: float | None = None,
                                 enable_gqa: bool = False) -> Tensor:
    """softmax(query @ key^T * scale + mask) @ value over the last two dims
    (torch.nn.functional.scaled_dot_product_attention), batch dims broadcast.

    attn_mask: an integer 0/1 tensor (nonzero = may attend, like torch's bool mask; newch has no
    bool dtype) or a float tensor added to the scores. A row with nothing to attend gives NaN, as in torch.
    """
    _unsupported(enable_gqa=(enable_gqa, False))
    L, S = query.shape[-2], key.shape[-2]
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    scores = (query @ key.transpose(-2, -1)) * float(scale)
    if is_causal:
        if attn_mask is not None:
            raise ValueError("scaled_dot_product_attention: pass attn_mask or is_causal, not both")
        attn_mask = _causal_keep(L, S, query.ctx)
    if attn_mask is not None:
        scores = _apply_mask(scores, attn_mask, keep=True)
    attn = scores.softmax(-1)
    if dropout_p > 0.0:
        attn = dropout(attn, dropout_p, True)
    return attn @ value


def _apply_mask(scores: Tensor, mask, keep: bool) -> Tensor:
    """Float masks are added; integer masks block with -inf where they are 0 (keep=True, SDPA's
    convention) or where they are nonzero (keep=False, nn.MultiheadAttention's)."""
    if not isinstance(mask, Tensor):
        mask = tensor(mask, ctx=scores.ctx)
    if mask.dtype == float32:
        return scores + mask
    return scores.masked_fill(mask == 0 if keep else mask, float("-inf"))


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """y = input @ weight.T + bias, weight [out, in] (torch.nn.functional.linear)."""
    return input._w(lib.nc_linear_fn(input.ptr, weight.ptr, bias.ptr if bias is not None else None))


def cross_entropy(input: Tensor, target, weight=None, size_average=None, ignore_index: int = -100, reduce=None,
                  reduction: str = "mean", label_smoothing: float = 0.0) -> Tensor:
    """Cross-entropy of logits input [N, C] (also [C] and [N, C, d1, ...]) and either int32 class
    indices target [N] ([], [N, d1, ...]) or class probabilities with input's shape (float32).

    reduction "mean" or "sum"; label_smoothing mixes in the uniform distribution.
    Labels outside [0, C) raise (so ignore_index only works as the default -100).
    """
    _unsupported(weight=(weight, None), size_average=(size_average, None), reduce=(reduce, None),
                 ignore_index=(ignore_index, -100))
    if reduction not in ("mean", "sum"):
        raise NotImplementedError(f"reduction={reduction!r} is not supported (use 'mean' or 'sum')")
    if not 0.0 <= label_smoothing <= 1.0:
        raise ValueError("label_smoothing must be in [0, 1]")
    if not isinstance(target, Tensor):
        target = tensor(target, ctx=input.ctx)
    if target.dtype == float32:  # class probabilities
        input, target = _class_rows(input, target, probs=True)
        if label_smoothing:
            target = target * (1.0 - label_smoothing) + label_smoothing / input.shape[-1]
        per_row = (target * input.log_softmax(-1)).sum(-1) * -1.0
        return per_row.sum() if reduction == "sum" else per_row.mean()
    input, target = _class_rows(input, _class_indices(target, input), probs=False)
    loss = input._w(lib.nc_cross_entropy(input.ptr, target.ptr))
    if label_smoothing:
        uniform = (input.log_softmax(-1).mean(-1)).mean() * -1.0  # mean over classes and rows
        loss = loss * (1.0 - label_smoothing) + uniform * label_smoothing
    if reduction == "sum":
        n = target.shape[0]
        return loss * float(n) if n else input.sum()  # torch: 0 for no rows (the mean is NaN)
    return loss


def nll_loss(input: Tensor, target, weight=None, size_average=None, ignore_index: int = -100, reduce=None,
             reduction: str = "mean") -> Tensor:
    """Negative log-likelihood of log-probabilities input [N, C] (also [C], [N, C, d1, ...]) and class
    indices target [N] ([], [N, d1, ...])."""
    _unsupported(weight=(weight, None), size_average=(size_average, None), reduce=(reduce, None),
                 ignore_index=(ignore_index, -100))
    if reduction not in ("mean", "sum"):
        raise NotImplementedError(f"reduction={reduction!r} is not supported (use 'mean' or 'sum')")
    input, target = _class_rows(input, _class_indices(target, input), probs=False)
    return input._w(lib.nc_nll_loss(input.ptr, target.ptr, int(reduction == "sum")))


def _class_rows(input: Tensor, target: Tensor, probs: bool):
    """torch's loss layouts as rows [M, C]: unbatched [C] -> [1, C]; [N, C, d1, ...] -> [N*d1*..., C]
    (class axis moved last). Targets follow: indices [M], probabilities [M, C]."""
    if input.ndim == 1:
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)
    elif input.ndim > 2:
        c, n = input.shape[1], input.ndim
        want = input.shape if probs else input.shape[:1] + input.shape[2:]
        if target.shape != want:
            raise ValueError(f"target {target.shape} does not match input {input.shape} (expected {want})")
        order = (0, *range(2, n), 1)
        input = input.permute(*order).reshape(-1, c)
        target = target.permute(*order).reshape(-1, c) if probs else target.reshape(-1)
    if probs and target.shape != input.shape:
        raise ValueError(f"probability target {target.shape} must have the input's shape {input.shape}")
    return input, target


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
