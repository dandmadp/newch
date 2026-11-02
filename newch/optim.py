"""Optimizers and learning-rate schedulers. Optimizer state lives in the
engine next to the parameters (SGD: 0x or 1x parameter memory, Adam/AdamW: 2x).

Low-memory training: ``gradient_accumulation_steps=n`` (as in Keras 3)
updates the weights every n-th ``step()`` with the averaged gradient, so a
batch of B trains like a batch of n*B at the activation memory of B.

Resuming: ``state_dict()`` / ``load_state_dict()`` use PyTorch's layout, and
``nc.save_tensors`` stores a whole checkpoint in one file:

    nc.save_tensors({"model": model.state_dict(), "optimizer": opt.state_dict()}, "ckpt.ncw")
"""

from __future__ import annotations

import ctypes
import math

from . import _lib as L
from ._lib import lib
from .tensor import Tensor, _to_buffer, _unsupported, float32, persistent, tensor_from_bytes


class Optimizer:
    # per subclass: names of the engine's state slots 0 and 1 (torch's state keys), whether the
    # state carries torch's "step", and the torch arguments that must keep their default
    _STATE_NAMES: tuple = (None, None)
    _HAS_STEP = False
    _FIXED: dict = {}

    def __init__(self, params, factory, hyper: dict, gradient_accumulation_steps: int | None = None):
        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")
        if not all(isinstance(p, Tensor) for p in params):
            raise TypeError("optimizer parameters must be newch Tensors")
        self.params = params  # keeps the Python wrappers alive
        self._ctx = params[0].ctx
        self._p = None
        self._hyper = dict(hyper)
        self._p = factory(self._ctx._p, L.ptr_array(p.ptr for p in params), len(params), self._opts(self._hyper))
        if not self._p:
            self._ctx.raise_error()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if gradient_accumulation_steps is not None:
            if int(gradient_accumulation_steps) < 1:
                raise ValueError("gradient_accumulation_steps must be >= 1")
            lib.nc_optim_set_accumulation(self._p, int(gradient_accumulation_steps))

    def __del__(self):
        # the engine state (up to 2x the parameters) goes with the Python object
        try:
            if self._p and self._ctx._p:
                lib.nc_optim_free(self._p)
                self._p = None
        except Exception:  # interpreter shutdown
            pass

    @staticmethod
    def _opts(h: dict) -> L.OptimOpts:
        raise NotImplementedError

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Resets the grads (kept while gradient accumulation is in progress). With set_to_none
        (default, as in PyTorch) they read as None and the optimizer skips parameters that the
        next backward does not reach."""
        (lib.nc_optim_clear_grad if set_to_none else lib.nc_optim_zero_grad)(self._p)

    def step(self) -> None:
        lib.nc_optim_step(self._p)

    @property
    def lr(self) -> float:
        return self._hyper["lr"]

    @lr.setter
    def lr(self, value: float) -> None:
        self._hyper["lr"] = float(value)
        lib.nc_optim_set_lr(self._p, self._hyper["lr"])

    # ---- checkpoints (torch.optim.Optimizer layout) ------------------------
    def state_dict(self) -> dict:
        """``{"state": {i: {name: Tensor, "step": float}}, "param_groups": [{hyperparameters, "params"}]}``
        as in PyTorch (one parameter group). The state Tensors are persistent copies, not live buffers."""
        step = float(lib.nc_optim_get_step(self._p))
        state = {}
        with persistent(self._ctx):
            for i, p in enumerate(self.params):
                entry = {"step": step} if self._HAS_STEP else {}
                for slot, name in enumerate(self._STATE_NAMES):
                    buf = lib.nc_optim_state(self._p, i, slot)
                    if buf:
                        entry[name] = tensor_from_bytes(ctypes.string_at(buf, 4 * p.numel()), p.shape, float32,
                                                        self._ctx)
                if entry:
                    state[i] = entry
        group = {**self._hyper, **self._FIXED, "params": list(range(len(self.params)))}
        return {"state": state, "param_groups": [group]}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restores hyperparameters, state buffers and the step count from ``state_dict()`` output
        (also PyTorch's, with tensors converted to numpy). Checks everything before changing anything."""
        groups = state_dict["param_groups"]
        if len(groups) != 1:
            raise ValueError(f"load_state_dict: newch optimizers have one parameter group, got {len(groups)}")
        group = groups[0]
        ids = list(group["params"])
        if len(ids) != len(self.params):
            raise ValueError(f"load_state_dict: {len(ids)} parameters in the checkpoint, {len(self.params)} here")
        _unsupported(**{k: (group.get(k, v), v) for k, v in self._FIXED.items()})
        missing = [k for k in self._hyper if k not in group]
        if missing:
            raise ValueError(f"load_state_dict: parameter group lacks {missing}")
        hyper = {k: tuple(float(b) for b in group[k]) if k == "betas" else float(group[k]) for k in self._hyper}
        if "momentum" in hyper and (hyper["momentum"] != 0) != (self._hyper["momentum"] != 0):
            raise ValueError("load_state_dict: momentum cannot switch between zero and non-zero "
                             f"({self._hyper['momentum']} -> {hyper['momentum']}); create the optimizer with it")
        allowed = {n for n in self._STATE_NAMES if n} | ({"step"} if self._HAS_STEP else set())
        states = state_dict.get("state", {})
        writes, steps = [], set()
        for i, (pid, p) in enumerate(zip(ids, self.params)):
            entry = states.get(pid, {})
            extra = set(entry) - allowed
            if extra:
                raise ValueError(f"load_state_dict: parameter {i} has unexpected state {sorted(extra)}")
            for slot, name in enumerate(self._STATE_NAMES):
                buf = lib.nc_optim_state(self._p, i, slot)
                if not buf:
                    continue
                if name not in entry:  # never stepped in the source: starts from zero, as fresh state
                    writes.append((buf, None, 4 * p.numel()))
                    continue
                src = entry[name]
                keep, addr, shape, _ = _to_buffer(src._flat() if isinstance(src, Tensor) else src, float32)
                n = math.prod(shape)
                if n != p.numel():
                    raise ValueError(f"load_state_dict: '{name}' of parameter {i} has {n} elements, "
                                     f"the parameter {p.numel()}")
                writes.append((buf, (keep, addr), 4 * n))
            if "step" in entry:
                s = entry["step"]
                steps.add(float(s.item() if hasattr(s, "item") else s))
        if len(steps) > 1:
            raise NotImplementedError(f"load_state_dict: parameters have different step counts {sorted(steps)}; "
                                      "newch keeps one count per optimizer")
        lib.nc_optim_set_opts(self._p, self._opts(hyper))
        if not lib.nc_ok(self._ctx._p):
            self._ctx.raise_error()
        self._hyper = hyper
        for buf, src, nbytes in writes:
            if src is None:
                ctypes.memset(buf, 0, nbytes)
            elif nbytes:
                ctypes.memmove(buf, src[1], nbytes)
        lib.nc_optim_set_step(self._p, int(steps.pop()) if steps else 0)


class SGD(Optimizer):
    _STATE_NAMES = ("momentum_buffer", None)
    _FIXED = {"dampening": 0.0, "nesterov": False, "maximize": False}

    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.0, dampening: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False, *, maximize: bool = False,
                 gradient_accumulation_steps: int | None = None):
        _unsupported(dampening=(dampening, 0.0), nesterov=(nesterov, False), maximize=(maximize, False))
        super().__init__(params, lib.nc_sgd, {"lr": lr, "momentum": momentum, "weight_decay": weight_decay},
                         gradient_accumulation_steps)

    @staticmethod
    def _opts(h: dict) -> L.OptimOpts:
        return L.OptimOpts(h["lr"], h["momentum"], h["weight_decay"], 0.0, 0.0, 0.0)


class Adam(Optimizer):
    """torch.optim.Adam: weight_decay is an L2 term added to the gradient."""

    _STATE_NAMES = ("exp_avg", "exp_avg_sq")
    _HAS_STEP = True
    _FIXED = {"amsgrad": False, "maximize": False}
    _FACTORY = "nc_adam"

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, amsgrad: bool = False, *, maximize: bool = False,
                 gradient_accumulation_steps: int | None = None):
        _unsupported(amsgrad=(amsgrad, False), maximize=(maximize, False))
        super().__init__(params, getattr(lib, self._FACTORY),
                         {"lr": lr, "betas": tuple(betas), "eps": eps, "weight_decay": weight_decay},
                         gradient_accumulation_steps)

    @staticmethod
    def _opts(h: dict) -> L.OptimOpts:
        return L.OptimOpts(h["lr"], 0.0, h["weight_decay"], h["betas"][0], h["betas"][1], h["eps"])


class AdamW(Adam):
    """torch.optim.AdamW: decoupled weight decay (w *= 1 - lr * weight_decay), default 0.01."""

    _FACTORY = "nc_adamw"

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2, amsgrad: bool = False, *, maximize: bool = False,
                 gradient_accumulation_steps: int | None = None):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, maximize=maximize,
                         gradient_accumulation_steps=gradient_accumulation_steps)


class RMSprop(Optimizer):
    """torch.optim.RMSprop: v = alpha v + (1 - alpha) g^2, w -= lr g / (sqrt(v) + eps)."""

    _STATE_NAMES = ("momentum_buffer", "square_avg")
    _HAS_STEP = True
    _FIXED = {"centered": False, "maximize": False}

    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0.0, momentum: float = 0.0, centered: bool = False, *,
                 maximize: bool = False, gradient_accumulation_steps: int | None = None):
        _unsupported(centered=(centered, False), maximize=(maximize, False))
        super().__init__(params, lambda c, p, n, o: lib.nc_rmsprop(c, p, n, o, 0),
                         {"lr": lr, "alpha": alpha, "eps": eps, "weight_decay": weight_decay, "momentum": momentum},
                         gradient_accumulation_steps)

    @staticmethod
    def _opts(h: dict) -> L.OptimOpts:
        return L.OptimOpts(h["lr"], h["momentum"], h["weight_decay"], 0.0, h["alpha"], h["eps"])


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

    def state_dict(self) -> dict:
        """Everything but the optimizer (and LambdaLR's function), as in PyTorch."""
        return {k: v for k, v in self.__dict__.items() if k not in ("optimizer", "lr_lambda")}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)


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

    def state_dict(self) -> dict:
        """Everything but the optimizer (and LambdaLR's function), as in PyTorch."""
        return {k: v for k, v in self.__dict__.items() if k not in ("optimizer", "lr_lambda")}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)


class _SchedulerNamespace:
    """``nc.optim.lr_scheduler.StepLR`` etc., the same path as in PyTorch."""

    LRScheduler = LRScheduler
    LambdaLR = LambdaLR
    StepLR = StepLR
    ExponentialLR = ExponentialLR
    CosineAnnealingLR = CosineAnnealingLR
    ReduceLROnPlateau = ReduceLROnPlateau


lr_scheduler = _SchedulerNamespace()
