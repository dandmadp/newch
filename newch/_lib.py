"""ctypes bindings to the C engine (nc.h). Internal: use the public API."""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, Structure, c_char_p, c_float, c_int, c_int64, c_size_t, c_uint64, c_void_p
from pathlib import Path

ALL_AXES = -128
F32, I32, I8 = 0, 1, 3
DTYPE_NAMES = {F32: "float32", I32: "int32", I8: "int8"}
DTYPE_CODES = {"float32": F32, "int32": I32, "int8": I8}
DTYPE_SIZE = {"float32": 4, "int32": 4, "int8": 1}
ARRAY_CODE = {"float32": "f", "int32": "i", "int8": "b"}


class Config(Structure):
    _fields_ = [("arena_chunk_bytes", c_size_t), ("seed", c_uint64), ("force_reference", c_int), ("threads", c_int)]


class CpuInfo(Structure):
    _fields_ = [
        ("arch", c_char_p),
        ("has_neon", c_int),
        ("has_dotprod", c_int),
        ("has_fp16", c_int),
        ("has_avx2", c_int),
        ("has_fma", c_int),
        ("has_rvv", c_int),
        ("n_cores", c_int),
        ("n_big_cores", c_int),
    ]


class MemStats(Structure):
    _fields_ = [
        ("persistent_bytes", c_size_t),
        ("arena_bytes_in_use", c_size_t),
        ("arena_bytes_reserved", c_size_t),
        ("arena_peak_bytes", c_size_t),
    ]


class OptimOpts(Structure):
    _fields_ = [(n, c_float) for n in ("lr", "momentum", "weight_decay", "beta1", "beta2", "eps")]


def _lib_names() -> list[str]:
    if sys.platform == "win32":
        return ["nc.dll"]
    if sys.platform == "darwin":
        return ["libnc.dylib"]
    return ["libnc.so"]


def _find_library() -> str:
    env = os.environ.get("NEWCH_LIB")
    if env:
        return env
    here = Path(__file__).resolve().parent
    for d in (here, here.parent / "build" / "host"):
        for name in _lib_names():
            if (d / name).exists():
                return str(d / name)
    raise OSError(
        "newch: native library not found. Build it with `python tools/build.py` "
        "or set NEWCH_LIB to the path of nc.dll / libnc.so."
    )


lib = ctypes.CDLL(_find_library())

P = c_void_p
I64P = POINTER(c_int64)

_SIGNATURES = {
    # context
    "nc_init": (P, [POINTER(Config)]),
    "nc_free_ctx": (None, [P]),
    "nc_manual_seed": (None, [P, c_uint64]),
    "nc_linear_fn": (P, [P, P, P]),
    "nc_step_end": (None, [P]),
    "nc_arena_trim": (None, [P]),
    "nc_num_threads": (c_int, [P]),
    "nc_persistent_begin": (None, [P]),
    "nc_persistent_end": (None, [P]),
    "nc_set_grad_enabled": (c_int, [P, c_int]),
    "nc_ok": (c_int, [P]),
    "nc_status_code": (c_int, [P]),
    "nc_error": (c_char_p, [P]),
    "nc_clear_error": (None, [P]),
    "nc_get_mem_stats": (MemStats, [P]),
    "nc_reset_peak": (None, [P]),
    "nc_cpu": (POINTER(CpuInfo), [P]),
    "nc_kernel_backend": (c_char_p, [P]),
    # tensors
    "nc_empty": (P, [P, c_int, c_int, I64P]),
    "nc_zeros": (P, [P, c_int, c_int, I64P]),
    "nc_full": (P, [P, c_float, c_int, I64P]),
    "nc_randn": (P, [P, c_float, c_int, I64P]),
    "nc_rand_uniform": (P, [P, c_float, c_float, c_int, I64P]),
    "nc_from_data": (P, [P, c_int, c_void_p, c_int, I64P]),
    "nc_scalar": (P, [P, c_float]),
    "nc_parameter": (P, [P, P, c_int, I64P]),
    "nc_ndim": (c_int, [P]),
    "nc_shape": (I64P, [P]),
    "nc_numel": (c_int64, [P]),
    "nc_get_dtype": (c_int, [P]),
    "nc_is_contiguous": (c_int, [P]),
    "nc_requires_grad": (c_int, [P]),
    "nc_set_requires_grad": (None, [P, c_int]),
    "nc_is_persistent": (c_int, [P]),
    "nc_release": (None, [P]),
    "nc_grad": (P, [P]),
    "nc_item": (c_float, [P]),
    "nc_data_f32": (c_void_p, [P]),
    "nc_copy_to": (c_int, [P, c_void_p, c_size_t]),
    "nc_copy_from": (c_int, [P, c_void_p, c_size_t]),
    # ops
    **{f"nc_{n}": (P, [P, P]) for n in ("add", "sub", "mul", "div", "matmul", "cross_entropy", "mse_loss",
                                         "index_select")},
    **{f"nc_{n}": (P, [P]) for n in ("neg", "exp", "log", "sqrt", "relu", "sigmoid", "tanh", "contiguous",
                                      "log_softmax", "softmax", "detach")},
    **{f"nc_{n}": (P, [P, c_float]) for n in ("scale", "add_scalar", "pow_scalar")},
    **{f"nc_{n}": (P, [P, c_int, c_int]) for n in ("sum", "mean", "max", "transpose")},
    "nc_argmax": (P, [P, c_int]),
    "nc_nll_loss": (P, [P, P, c_int]),
    "nc_compare": (P, [P, P, c_int]),
    "nc_cat": (P, [c_int, POINTER(c_void_p), c_int]),
    "nc_to_dtype": (P, [P, c_int]),
    **{f"nc_{n}": (P, [P]) for n in ("relu6", "hardsigmoid", "hardswish", "silu")},
    "nc_leaky_relu": (P, [P, c_float]),
    "nc_gelu": (P, [P, c_int]),
    "nc_conv2d": (P, [P, P, P, c_int, c_int, c_int, c_int]),
    "nc_max_pool2d": (P, [P, c_int, c_int, c_int, c_int, c_int, c_int]),
    "nc_avg_pool2d": (P, [P, c_int, c_int, c_int, c_int, c_int, c_int]),
    "nc_dropout": (P, [P, c_float]),
    "nc_batch_norm": (P, [P, P, P, P, P, c_int, c_float, c_float]),
    "nc_quantize_rows": (P, [P, POINTER(c_void_p)]),
    "nc_linear_int8": (P, [P, P, P, P]),
    "nc_save": (c_int, [P, c_char_p, c_int, POINTER(c_char_p), POINTER(c_void_p)]),
    "nc_load": (c_int, [P, c_char_p, c_int, POINTER(c_char_p), POINTER(c_void_p)]),
    "nc_reshape": (P, [P, c_int, I64P]),
    # autograd
    "nc_backward": (c_int, [P]),
    "nc_backward_ex": (c_int, [P, c_int, POINTER(c_void_p), c_int]),
    "nc_release_temp": (None, [P]),
    "nc_zero_grad": (None, [P]),
    "nc_clear_grad": (None, [P]),
    # optimizers
    "nc_sgd": (P, [P, POINTER(c_void_p), c_int, OptimOpts]),
    "nc_adam": (P, [P, POINTER(c_void_p), c_int, OptimOpts]),
    "nc_adamw": (P, [P, POINTER(c_void_p), c_int, OptimOpts]),
    "nc_optim_set_accumulation": (None, [P, c_int]),
    "nc_clip_grad_norm": (c_float, [POINTER(c_void_p), c_int, c_float]),
    "nc_optim_zero_grad": (None, [P]),
    "nc_optim_clear_grad": (None, [P]),
    "nc_rmsprop": (P, [P, POINTER(c_void_p), c_int, OptimOpts, c_int]),
    "nc_optim_step": (None, [P]),
    "nc_optim_set_lr": (None, [P, c_float]),
}

for _name, (_res, _args) in _SIGNATURES.items():
    _fn = getattr(lib, _name)
    _fn.restype = _res
    _fn.argtypes = _args


def shape_arg(shape) -> tuple[int, ctypes.Array]:
    shape = tuple(int(s) for s in shape)
    return len(shape), (c_int64 * max(len(shape), 1))(*shape)


def ptr_array(ptrs) -> ctypes.Array:
    ptrs = list(ptrs)
    return (c_void_p * max(len(ptrs), 1))(*ptrs)


def name_array(names) -> ctypes.Array:
    names = [n.encode("utf-8") for n in names]
    return (c_char_p * max(len(names), 1))(*names)
