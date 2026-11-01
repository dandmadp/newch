"""Keras-style training callbacks."""

from __future__ import annotations

import ctypes
import math
import warnings


class Callback:
    model = None

    def set_model(self, model) -> None:
        self.model = model

    def on_train_begin(self, logs: dict) -> None: ...
    def on_epoch_begin(self, epoch: int, logs: dict) -> None: ...
    def on_epoch_end(self, epoch: int, logs: dict) -> None: ...
    def on_train_end(self, logs: dict) -> None: ...


class History(Callback):
    """Returned by fit(): ``history.history["loss"]`` is one value per epoch."""

    def on_train_begin(self, logs):
        self.epoch: list[int] = []
        self.history: dict[str, list[float]] = {}

    def on_epoch_end(self, epoch, logs):
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


def _mode(monitor: str, mode: str) -> str:
    """'min' or 'max'; 'auto' maximizes accuracy-like metrics and minimizes the rest."""
    if mode == "auto":
        mode = "max" if "acc" in monitor else "min"
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'auto', 'min' or 'max', got {mode!r}")
    return mode


def _value(cb, logs: dict):
    """The monitored value, or None with a warning when the logs do not have it."""
    value = logs.get(cb.monitor)
    if value is None:
        warnings.warn(f"{type(cb).__name__}: no '{cb.monitor}' in this epoch's logs "
                      f"(have: {', '.join(sorted(logs)) or 'nothing'})", stacklevel=3)
    return value


def _optimizer(model):
    opt = getattr(model, "optimizer", None)
    if opt is None:
        raise RuntimeError(f"{type(model).__name__} has no optimizer yet; compile() it first")
    return opt


def _weights(model) -> list:
    return [p._flat() for p in model.parameters()]


def _set_weights(model, saved: list) -> None:
    from ._lib import lib

    for p, buf in zip(model.parameters(), saved):
        ctypes.memmove(lib.nc_data_f32(p.ptr), buf.buffer_info()[0], len(buf) * buf.itemsize)


class EarlyStopping(Callback):
    """Stops training once ``monitor`` has gone ``patience`` epochs without improving by ``min_delta``.

    ``baseline``: a value the monitor must beat for the patience counter to reset.
    ``restore_best_weights``: put back the weights of the best epoch when training ends.
    ``start_from_epoch``: epochs to let pass before watching (warm-up).
    """

    def __init__(self, monitor: str = "val_loss", min_delta: float = 0.0, patience: int = 0, verbose: int = 0,
                 mode: str = "auto", baseline: float | None = None, restore_best_weights: bool = False,
                 start_from_epoch: int = 0):
        self.monitor, self.min_delta, self.patience, self.verbose = monitor, abs(min_delta), patience, verbose
        self.mode = _mode(monitor, mode)
        self.baseline, self.restore_best_weights, self.start_from_epoch = baseline, restore_best_weights, \
            start_from_epoch
        self.stopped_epoch = 0

    def _improved(self, value, reference) -> bool:
        if reference is None:
            return True
        if self.mode == "min":
            return value < reference - self.min_delta
        return value > reference + self.min_delta

    def on_train_begin(self, logs):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs):
        value = _value(self, logs)
        if value is None or epoch < self.start_from_epoch:
            return
        if self.restore_best_weights and self.best_weights is None:
            self.best_weights = _weights(self.model)  # fallback when nothing ever improves
        self.wait += 1
        if self._improved(value, self.best):
            self.best, self.best_epoch = value, epoch
            if self.restore_best_weights:
                self.best_weights = _weights(self.model)
            if self.baseline is None or self._improved(value, self.baseline):
                self.wait = 0
            return
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs):
        if self.stopped_epoch > 0 and self.verbose:
            print(f"early stop at epoch {self.stopped_epoch + 1} "
                  f"(best {self.monitor} {self.best:.4f} at epoch {self.best_epoch + 1})")
        if self.restore_best_weights and self.best_weights is not None:
            _set_weights(self.model, self.best_weights)


class LambdaCallback(Callback):
    """A callback from plain functions: on_epoch_begin(epoch, logs), on_epoch_end(epoch, logs),
    on_train_begin(logs), on_train_end(logs)."""

    def __init__(self, on_epoch_begin=None, on_epoch_end=None, on_train_begin=None, on_train_end=None):
        self._fns = {"epoch_begin": on_epoch_begin, "epoch_end": on_epoch_end, "train_begin": on_train_begin,
                     "train_end": on_train_end}

    def _call(self, name, *args):
        fn = self._fns[name]
        if fn is not None:
            fn(*args)

    def on_epoch_begin(self, epoch, logs): self._call("epoch_begin", epoch, logs)
    def on_epoch_end(self, epoch, logs): self._call("epoch_end", epoch, logs)
    def on_train_begin(self, logs): self._call("train_begin", logs)
    def on_train_end(self, logs): self._call("train_end", logs)


class LearningRateScheduler(Callback):
    """Before each epoch, sets the learning rate to ``schedule(epoch, current_lr)``."""

    def __init__(self, schedule, verbose: int = 0):
        self.schedule, self.verbose = schedule, verbose

    def on_epoch_begin(self, epoch, logs):
        opt = _optimizer(self.model)
        opt.lr = float(self.schedule(epoch, opt.lr))
        if self.verbose:
            print(f"epoch {epoch + 1}: lr = {opt.lr:.6g}")


class ReduceLROnPlateau(Callback):
    """Scales the learning rate by ``factor`` when ``monitor`` stalls for ``patience`` epochs.

    After a reduction, ``cooldown`` epochs pass before the counter runs again; the rate never
    goes below ``min_lr``.
    """

    def __init__(self, monitor: str = "val_loss", factor: float = 0.1, patience: int = 10, verbose: int = 0,
                 mode: str = "auto", min_delta: float = 1e-4, cooldown: int = 0, min_lr: float = 0.0):
        if not 0.0 < factor < 1.0:
            raise ValueError(f"factor must be in (0, 1) so the rate goes down, got {factor}")
        self.monitor, self.factor, self.patience, self.verbose = monitor, factor, patience, verbose
        self.mode = _mode(monitor, mode)
        self.min_delta, self.cooldown, self.min_lr = min_delta, cooldown, min_lr

    def _better(self, a, b):
        return a < b - self.min_delta if self.mode == "min" else a > b + self.min_delta

    def on_train_begin(self, logs):
        self.best = math.inf if self.mode == "min" else -math.inf
        self.wait = 0
        self.cooldown_counter = 0

    def on_epoch_end(self, epoch, logs):
        current = _value(self, logs)
        if current is None:
            return
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
        if self._better(current, self.best):
            self.best, self.wait = current, 0
        elif self.cooldown_counter <= 0:
            self.wait += 1
            if self.wait >= self.patience:
                opt = _optimizer(self.model)
                if opt.lr > self.min_lr:
                    opt.lr = max(opt.lr * self.factor, self.min_lr)
                    if self.verbose:
                        print(f"epoch {epoch + 1}: {self.monitor} stalled, lr -> {opt.lr:.6g}")
                    self.cooldown_counter = self.cooldown
                    self.wait = 0


class ModelCheckpoint(Callback):
    """Writes the weights (NCW1 file) after every epoch, or only when ``monitor`` improves.

    ``filepath`` may use ``{epoch}`` and any log name, e.g. ``"w-{epoch:02d}-{val_loss:.3f}.ncw"``.
    Only weights can be saved, so pass ``save_weights_only=True``; rebuild the model in code and
    call ``load_weights()`` to restore it.
    """

    def __init__(self, filepath, monitor: str = "val_loss", verbose: int = 0, save_best_only: bool = False,
                 save_weights_only: bool = False, mode: str = "auto", save_freq="epoch",
                 initial_value_threshold: float | None = None):
        if not save_weights_only:
            raise NotImplementedError("newch checkpoints hold weights only: pass save_weights_only=True")
        if save_freq != "epoch":
            raise NotImplementedError("only save_freq='epoch' is supported")
        self.filepath, self.monitor, self.verbose = str(filepath), monitor, verbose
        self.save_best_only, self.save_weights_only = save_best_only, save_weights_only
        self.mode = _mode(monitor, mode)
        self.initial_value_threshold = initial_value_threshold

    def on_train_begin(self, logs):
        if self.initial_value_threshold is not None:
            self.best = self.initial_value_threshold
        else:
            self.best = math.inf if self.mode == "min" else -math.inf

    def on_epoch_end(self, epoch, logs):
        if self.save_best_only:
            current = _value(self, logs)
            if current is None:
                return
            if not (current < self.best if self.mode == "min" else current > self.best):
                return
            self.best = current
        path = self.filepath.format(epoch=epoch + 1, **logs)
        self.model.save_weights(path)
        if self.verbose:
            print(f"epoch {epoch + 1}: weights -> {path}")
