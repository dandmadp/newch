"""Optimizers and learning-rate schedulers. Optimizer state lives in the
engine next to the parameters (SGD: 0x or 1x parameter memory, Adam/AdamW: 2x).

Low-memory training: ``gradient_accumulation_steps=n`` (as in Keras 3)
updates the weights every n-th ``step()`` with the averaged gradient, so a
batch of B trains like a batch of n*B at the activation memory of B.
"""

from __future__ import annotations

import math

from . import _lib as L
from ._lib import lib
from .tensor import Tensor, _unsupported


class Optimizer:
    def __init__(self, params, factory, opts: L.OptimOpts, gradient_accumulation_steps: int | None = None):
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
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if gradient_accumulation_steps is not None:
            if int(gradient_accumulation_steps) < 1:
                raise ValueError("gradient_accumulation_steps must be >= 1")
            lib.nc_optim_set_accumulation(self._p, int(gradient_accumulation_steps))

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Resets the grads (kept while gradient accumulation is in progress). With set_to_none
        (default, as in PyTorch) they read as None and the optimizer skips parameters that the
        next backward does not reach."""
        (lib.nc_optim_clear_grad if set_to_none else lib.nc_optim_zero_grad)(self._p)

    def step(self) -> None:
        lib.nc_optim_step(self._p)

    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, value: float) -> None:
        self._lr = float(value)
        lib.nc_optim_set_lr(self._p, self._lr)

    # Keras' name for the same value (used by Keras-style callbacks)
    learning_rate = lr


class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.0, dampening: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False, *, maximize: bool = False,
                 gradient_accumulation_steps: int | None = None):
        _unsupported(dampening=(dampening, 0.0), nesterov=(nesterov, False), maximize=(maximize, False))
        super().__init__(params, lib.nc_sgd, L.OptimOpts(lr, momentum, weight_decay, 0, 0, 0),
                         gradient_accumulation_steps)


class Adam(Optimizer):
    """torch.optim.Adam: weight_decay is an L2 term added to the gradient."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, amsgrad: bool = False, *, maximize: bool = False,
                 gradient_accumulation_steps: int | None = None):
        _unsupported(amsgrad=(amsgrad, False), maximize=(maximize, False))
        super().__init__(params, lib.nc_adam, L.OptimOpts(lr, 0.0, weight_decay, betas[0], betas[1], eps),
                         gradient_accumulation_steps)


class AdamW(Optimizer):
    """torch.optim.AdamW: decoupled weight decay (w *= 1 - lr * weight_decay), default 0.01."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2, amsgrad: bool = False, *, maximize: bool = False,
                 gradient_accumulation_steps: int | None = None):
        _unsupported(amsgrad=(amsgrad, False), maximize=(maximize, False))
        super().__init__(params, lib.nc_adamw, L.OptimOpts(lr, 0.0, weight_decay, betas[0], betas[1], eps),
                         gradient_accumulation_steps)


class RMSprop(Optimizer):
    """torch.optim.RMSprop: v = alpha v + (1 - alpha) g^2, w -= lr g / (sqrt(v) + eps)."""

    _eps_inside_sqrt = 0

    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0.0, momentum: float = 0.0, centered: bool = False, *,
                 maximize: bool = False, gradient_accumulation_steps: int | None = None):
        _unsupported(centered=(centered, False), maximize=(maximize, False))
        inside = self._eps_inside_sqrt
        super().__init__(params, lambda c, p, n, o: lib.nc_rmsprop(c, p, n, o, inside),
                         L.OptimOpts(lr, momentum, weight_decay, 0.0, alpha, eps), gradient_accumulation_steps)


class _KerasRMSprop(RMSprop):
    """compile(optimizer="rmsprop"): Keras' variant, w -= lr g / sqrt(v + epsilon)."""

    _eps_inside_sqrt = 1


# compile() names with Keras' default hyperparameters for each.
_BY_NAME = {
    "rmsprop": lambda p, lr: _KerasRMSprop(p, lr=1e-3 if lr is None else lr, alpha=0.9, eps=1e-7),
    "adam": lambda p, lr: Adam(p, lr=1e-3 if lr is None else lr, eps=1e-7),
    "adamw": lambda p, lr: AdamW(p, lr=1e-3 if lr is None else lr, eps=1e-7, weight_decay=0.004),
    "sgd": lambda p, lr: SGD(p, lr=1e-2 if lr is None else lr),
}


def get(optimizer, params, lr: float | None = None) -> Optimizer:
    """Optimizer from a compile() name ("rmsprop", "adam", "adamw", "sgd") or an instance."""
    if isinstance(optimizer, Optimizer):
        return optimizer
    try:
        make = _BY_NAME[optimizer.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"unknown optimizer {optimizer!r}; use one of {sorted(_BY_NAME)} or an instance")
    return make(params, lr)


# --------------------------------------------------------------------------
# Learning-rate schedulers (torch.optim.lr_scheduler). One parameter group,
# so get_last_lr() returns a one-element list as in PyTorch.
# --------------------------------------------------------------------------


class LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        if last_epoch != -1:
            raise NotImplementedError("resuming a scheduler (last_epoch != -1) is not supported")
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = -1
        self.step()

    def get_lr(self) -> float:
        raise NotImplementedError

    def step(self) -> None:
        self.last_epoch += 1
        self.optimizer.lr = self.get_lr()

    def get_last_lr(self) -> list:
        return [self.optimizer.lr]


class LambdaLR(LRScheduler):
    """lr = base_lr * lr_lambda(epoch)."""

    def __init__(self, optimizer: Optimizer, lr_lambda, last_epoch: int = -1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * self.lr_lambda(self.last_epoch)


class StepLR(LRScheduler):
    """lr = base_lr * gamma ** (epoch // step_size)."""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.step_size, self.gamma = step_size, gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * self.gamma ** (self.last_epoch // self.step_size)


class ExponentialLR(LRScheduler):
    """lr = base_lr * gamma ** epoch."""

    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * self.gamma ** self.last_epoch


class CosineAnnealingLR(LRScheduler):
    """lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2 (closed form)."""

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.T_max, self.eta_min = T_max, eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2


class _SchedulerNamespace:
    """``nc.optim.lr_scheduler.StepLR`` etc., the same path as in PyTorch."""

    LRScheduler = LRScheduler
    LambdaLR = LambdaLR
    StepLR = StepLR
    ExponentialLR = ExponentialLR
    CosineAnnealingLR = CosineAnnealingLR


lr_scheduler = _SchedulerNamespace()
