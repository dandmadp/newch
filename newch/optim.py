"""Optimizers. State lives in the engine next to the parameters
(SGD: 0x or 1x parameter memory, Adam: 2x)."""

from __future__ import annotations

from . import _lib as L
from ._lib import lib
from .tensor import Tensor


class Optimizer:
    def __init__(self, params, factory, opts: L.OptimOpts):
        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")
        if not all(isinstance(p, Tensor) for p in params):
            raise TypeError("optimizer parameters must be newch Tensors")
        self.params = params  # keeps the Python wrappers alive
        self._ctx = params[0].ctx
        self._lr = opts.lr
        self._p = factory(self._ctx._p, L.ptr_array(p.ptr for p in params), len(params), opts)
        if not self._p:
            self._ctx.raise_error()

    def zero_grad(self) -> None:
        lib.nc_optim_zero_grad(self._p)

    def step(self) -> None:
        lib.nc_optim_step(self._p)

    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, value: float) -> None:
        self._lr = float(value)
        lib.nc_optim_set_lr(self._p, self._lr)


class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params, lib.nc_sgd, L.OptimOpts(lr, momentum, weight_decay, 0, 0, 0))


class Adam(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(params, lib.nc_adam, L.OptimOpts(lr, 0.0, weight_decay, betas[0], betas[1], eps))


# Names used by compile(); default learning rates follow Keras.
_BY_NAME = {"sgd": (SGD, 0.01), "adam": (Adam, 1e-3)}


def get(optimizer, params, lr: float | None = None) -> Optimizer:
    """Optimizer from a name ("adam", "sgd") or pass an instance through."""
    if isinstance(optimizer, Optimizer):
        return optimizer
    try:
        cls, default_lr = _BY_NAME[optimizer.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"unknown optimizer {optimizer!r}; use one of {sorted(_BY_NAME)} or an instance")
    return cls(params, lr=default_lr if lr is None else lr)
