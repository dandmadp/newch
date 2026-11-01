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

    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0.0, momentum: float = 0.0, centered: bool = False, *,
                 maximize: bool = False, gradient_accumulation_steps: int | None = None):
        _unsupported(centered=(centered, False), maximize=(maximize, False))
        super().__init__(params, lambda c, p, n, o: lib.nc_rmsprop(c, p, n, o, 0),
                         L.OptimOpts(lr, momentum, weight_decay, 0.0, alpha, eps), gradient_accumulation_steps)


# Names accepted by compile(): the same classes, with their own defaults.
_BY_NAME = {"sgd": SGD, "adam": Adam, "adamw": AdamW, "rmsprop": RMSprop}


def get(optimizer, params, lr: float | None = None) -> Optimizer:
    """Optimizer from a compile() name ("adam", "adamw", "sgd", "rmsprop") or an instance.

    A name builds the same class as nc.optim.<Name>(params) with its defaults; lr overrides the rate.
    """
    if isinstance(optimizer, Optimizer):
        return optimizer
    try:
        cls = _BY_NAME[optimizer.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"unknown optimizer {optimizer!r}; use one of {sorted(_BY_NAME)} or an instance")
    return cls(params) if lr is None else cls(params, lr=lr)


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


class ReduceLROnPlateau:
    """Multiplies the rate by ``factor`` once ``metrics`` (passed to step) stops improving for
    more than ``patience`` epochs (torch.optim.lr_scheduler.ReduceLROnPlateau)."""

    def __init__(self, optimizer: Optimizer, mode: str = "min", factor: float = 0.1, patience: int = 10,
                 threshold: float = 1e-4, threshold_mode: str = "rel", cooldown: int = 0, min_lr: float = 0.0,
                 eps: float = 1e-8):
        if factor >= 1.0:
            raise ValueError("factor should be < 1.0")
        if mode not in ("min", "max") or threshold_mode not in ("rel", "abs"):
            raise ValueError("mode must be 'min'/'max' and threshold_mode 'rel'/'abs'")
        self.optimizer, self.mode, self.factor, self.patience = optimizer, mode, factor, patience
        self.threshold, self.threshold_mode, self.cooldown, self.min_lr, self.eps = (threshold, threshold_mode,
                                                                                    cooldown, min_lr, eps)
        self.best = math.inf if mode == "min" else -math.inf
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.last_epoch = 0

    def _better(self, a: float) -> bool:
        t, best = self.threshold, self.best
        if self.mode == "min":
            return a < (best * (1.0 - t) if self.threshold_mode == "rel" else best - t)
        return a > (best * (1.0 + t) if self.threshold_mode == "rel" else best + t)

    def step(self, metrics) -> None:
        current = float(metrics)
        self.last_epoch += 1
        if self._better(current):
            self.best, self.num_bad_epochs = current, 0
        else:
            self.num_bad_epochs += 1
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        if self.num_bad_epochs > self.patience:
            old = self.optimizer.lr
            new = max(old * self.factor, self.min_lr)
            if old - new > self.eps:
                self.optimizer.lr = new
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def get_last_lr(self) -> list:
        return [self.optimizer.lr]


class _SchedulerNamespace:
    """``nc.optim.lr_scheduler.StepLR`` etc., the same path as in PyTorch."""

    LRScheduler = LRScheduler
    LambdaLR = LambdaLR
    StepLR = StepLR
    ExponentialLR = ExponentialLR
    CosineAnnealingLR = CosineAnnealingLR
    ReduceLROnPlateau = ReduceLROnPlateau


lr_scheduler = _SchedulerNamespace()
