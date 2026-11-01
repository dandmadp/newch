"""Keras-style training callbacks."""

from __future__ import annotations

import ctypes
import math


class Callback:
    model = None

    def set_model(self, model) -> None:
        self.model = model

    def on_train_begin(self, logs: dict) -> None: ...
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


class EarlyStopping(Callback):
    """Stops when ``monitor`` has not improved by ``min_delta`` for ``patience`` epochs."""

    def __init__(self, monitor: str = "val_loss", min_delta: float = 0.0, patience: int = 0,
                 mode: str = "auto", restore_best_weights: bool = False):
        self.monitor, self.min_delta, self.patience = monitor, abs(min_delta), patience
        if mode == "auto":
            mode = "max" if "acc" in monitor else "min"
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'auto', 'min' or 'max'")
        self.sign = 1.0 if mode == "min" else -1.0
        self.restore_best_weights = restore_best_weights
        self.stopped_epoch = None

    def on_train_begin(self, logs):
        self.best = math.inf
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = None

    def on_epoch_end(self, epoch, logs):
        if self.monitor not in logs:
            raise KeyError(f"EarlyStopping: '{self.monitor}' not in logs (available: {sorted(logs)})")
        value = self.sign * logs[self.monitor]
        if value < self.best - self.min_delta:
            self.best, self.wait = value, 0
            if self.restore_best_weights:
                self.best_weights = [p._flat() for p in self.model.parameters()]
            return
        self.wait += 1
        if self.wait >= self.patience:
            self.model.stop_training = True
            self.stopped_epoch = epoch

    def on_train_end(self, logs):
        if self.restore_best_weights and self.best_weights is not None:
            from ._lib import lib

            for p, saved in zip(self.model.parameters(), self.best_weights):
                ctypes.memmove(lib.nc_data_f32(p.ptr), saved.buffer_info()[0], len(saved) * 4)


class LambdaCallback(Callback):
    """Keyword-only, with Keras' argument names: on_epoch_end(epoch, logs), ..."""

    def __init__(self, *, on_epoch_end=None, on_train_begin=None, on_train_end=None):
        self._e, self._b, self._t = on_epoch_end, on_train_begin, on_train_end

    def on_train_begin(self, logs):
        if self._b:
            self._b(logs)

    def on_epoch_end(self, epoch, logs):
        if self._e:
            self._e(epoch, logs)

    def on_train_end(self, logs):
        if self._t:
            self._t(logs)
