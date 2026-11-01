"""The compile/fit/evaluate/predict loop behind net.Model.

Two input modes share one loop:
- arrays (x, y): copied once into persistent engine memory, then each batch
  is gathered with index_select; the copies are released when the call ends.
- streaming: a DataLoader (or any iterable of (x, y) batches) passed as x
  with y=None. Only one batch is in memory at a time. FileArray inputs are
  streamed automatically.
"""

from __future__ import annotations

import array
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable

from . import optim
from .callbacks import Callback, History
from .data import DataLoader, FileArray, TensorDataset
from .tensor import (Context, Tensor, _is_numpy, cross_entropy, float32, int32, mse_loss, no_grad, persistent,
                     release, tensor)

# cross_entropy takes logits like torch. Keras' "sparse_categorical_crossentropy"
# expects probabilities by default, so it is deliberately not an alias.
LOSSES: dict[str, Callable] = {
    "cross_entropy": cross_entropy,
    "mse": mse_loss,
    "mean_squared_error": mse_loss,
}


@dataclass
class Compiled:
    ctx: Context
    optimizer: optim.Optimizer
    loss: Callable
    classification: bool
    accuracy: bool


def compile_model(model, optimizer, loss, metrics, lr) -> None:
    params = model.parameters()
    if not params:
        raise ValueError("model has no trainable parameters")
    if isinstance(loss, str):
        try:
            loss_fn = LOSSES[loss]
        except KeyError:
            raise ValueError(f"unknown loss {loss!r}; use one of {sorted(LOSSES)} or a callable")
    else:
        loss_fn = loss
    metrics = [m.lower() for m in (metrics or [])]
    unknown = set(metrics) - {"accuracy", "acc"}
    if unknown:
        raise ValueError(f"unsupported metrics {sorted(unknown)}; only 'accuracy' for now")
    classification = loss_fn is cross_entropy
    if metrics and not classification:
        raise ValueError("'accuracy' needs a classification loss (cross_entropy)")
    model._compiled = Compiled(params[0].ctx, optim.get(optimizer, params, lr), loss_fn, classification,
                               bool(metrics))


def _require(model) -> Compiled:
    if model._compiled is None:
        raise RuntimeError("call model.compile(...) before fit/evaluate")
    return model._compiled


def _is_stream(x, y) -> bool:
    return y is None and (isinstance(x, DataLoader) or (hasattr(x, "__iter__") and not isinstance(
        x, (list, tuple, Tensor, array.array)) and not _is_numpy(x)))


def _loader_for(x, y, batch_size, shuffle, seed):
    """FileArray inputs are streamed instead of copied into memory."""
    if isinstance(x, FileArray) or isinstance(y, FileArray):
        ds = TensorDataset(x) if y is None else TensorDataset(x, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=random.Random(seed))
    return None


class _Data:
    """Persistent copies of host data; released at the end of the call."""

    def __init__(self, ctx: Context):
        self.ctx, self.owned = ctx, []

    def put(self, data, dtype: str) -> Tensor:
        if isinstance(data, Tensor) and data.is_persistent and data.dtype == dtype:
            return data
        with persistent(self.ctx):
            t = tensor(data, dtype=dtype, ctx=self.ctx)
        self.owned.append(t)
        return t

    def release(self) -> None:
        for t in self.owned:
            release(t)
        self.owned.clear()


def _array_batches(ctx: Context, X: Tensor, Y: Tensor | None, rows, batch_size: int):
    for s in range(0, len(rows), batch_size):
        it = tensor(array.array("i", rows[s:s + batch_size]), dtype=int32, ctx=ctx)
        yield X.index_select(0, it), (Y.index_select(0, it) if Y is not None else None)


def _stream_batches(ctx: Context, loader, label_dtype: str | None):
    for item in loader:
        if isinstance(item, Tensor):
            xb, yb = item, None
        else:
            xb, yb = item[0], (item[1] if len(item) > 1 else None)
        if not isinstance(xb, Tensor):
            xb = tensor(xb, dtype=float32, ctx=ctx)
        if yb is not None and (not isinstance(yb, Tensor) or (label_dtype and yb.dtype != label_dtype)):
            yb = tensor(yb if not isinstance(yb, Tensor) else yb.tolist(), dtype=label_dtype, ctx=ctx)
        yield xb, yb


def _match_target(cfg: Compiled, out: Tensor, yb: Tensor) -> Tensor:
    # [B, 1] predictions against [B] targets would silently broadcast to [B, B]
    if not cfg.classification and out.ndim == yb.ndim + 1 and out.shape[-1] == 1:
        return yb.reshape(*yb.shape, 1)
    return yb


def _count_correct(out: Tensor, yb: Tensor) -> int:
    pred = out.argmax(-1)._flat()
    return sum(1 for a, b in zip(pred, yb._flat()) if a == b)


def _run_epoch(model, cfg: Compiled, batches, train: bool) -> dict:
    total, correct, seen = 0.0, 0, 0
    for xb, yb in batches:
        if yb is None:
            raise ValueError("training and evaluation need targets: pass y, or a loader yielding (x, y)")
        out = model(xb)
        loss = cfg.loss(out, _match_target(cfg, out, yb))
        if train:
            cfg.optimizer.zero_grad()
            loss.backward()
            cfg.optimizer.step()
        b = len(xb)
        total += loss.item() * b
        seen += b
        if cfg.accuracy:
            correct += _count_correct(out, yb)
        cfg.ctx.step_end()
    logs = {"loss": total / max(seen, 1)}
    if cfg.accuracy:
        logs["accuracy"] = correct / max(seen, 1)
    return logs


def _evaluate_batches(model, cfg: Compiled, batches) -> dict:
    was_training = model.training
    model.eval()
    try:
        with no_grad(cfg.ctx):
            return _run_epoch(model, cfg, batches, train=False)
    finally:
        model.train(was_training)


def _fmt(logs: dict) -> str:
    return " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())


def fit(model, x, y, epochs, batch_size, validation_data, validation_split, shuffle, callbacks, verbose, seed):
    cfg = _require(model)
    ctx = cfg.ctx
    label_dtype = int32 if cfg.classification else float32
    data = _Data(ctx)
    rng = random.Random(seed)
    try:
        loader = _loader_for(x, y, batch_size, shuffle, seed)
        streaming = loader is not None or _is_stream(x, y)
        if streaming:
            loader = loader or x
            if validation_split:
                raise ValueError("validation_split needs array inputs; pass validation_data for streamed data")

            def train_batches():
                return _stream_batches(ctx, loader, label_dtype)
        else:
            if y is None:
                raise ValueError("fit(x, y): y is required for array inputs")
            X, Y = data.put(x, float32), data.put(y, label_dtype)
            n = len(X)
            if len(Y) != n:
                raise ValueError(f"x has {n} samples but y has {len(Y)}")
            train_rows = list(range(n))
            val_rows: list = []
            if validation_split:
                if not 0.0 < validation_split < 1.0:
                    raise ValueError("validation_split must be in (0, 1)")
                cut = int(n * (1.0 - validation_split))  # last fraction, as in Keras
                train_rows, val_rows = train_rows[:cut], train_rows[cut:]

            def train_batches():
                order = train_rows[:]
                if shuffle:
                    rng.shuffle(order)
                return _array_batches(ctx, X, Y, order, batch_size)

        val_batches = None
        if not streaming and validation_split:
            def val_batches():
                return _array_batches(ctx, X, Y, val_rows, batch_size)
        elif validation_data is not None:
            if isinstance(validation_data, DataLoader) or _is_stream(validation_data, None):
                vloader = validation_data

                def val_batches():
                    return _stream_batches(ctx, vloader, label_dtype)
            else:
                vx, vy = validation_data
                vloader = _loader_for(vx, vy, batch_size, False, seed)
                if vloader is not None:
                    def val_batches():
                        return _stream_batches(ctx, vloader, label_dtype)
                else:
                    VX, VY = data.put(vx, float32), data.put(vy, label_dtype)
                    vrows_all = list(range(len(VX)))

                    def val_batches():
                        return _array_batches(ctx, VX, VY, vrows_all, batch_size)

        history = History()
        cbs: list[Callback] = [history, *(callbacks or [])]
        for cb in cbs:
            cb.set_model(model)
            cb.on_train_begin({})
        model.stop_training = False
        logs: dict = {}
        for epoch in range(epochs):
            model.train()
            t0 = time.perf_counter()
            logs = _run_epoch(model, cfg, train_batches(), train=True)
            if val_batches is not None:
                logs.update({f"val_{k}": v for k, v in _evaluate_batches(model, cfg, val_batches()).items()})
            for cb in cbs:
                cb.on_epoch_end(epoch, logs)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - {time.perf_counter() - t0:.2f}s - {_fmt(logs)}",
                      file=sys.stdout, flush=True)
            if model.stop_training:
                break
        for cb in cbs:
            cb.on_train_end(logs)
        return history
    finally:
        data.release()


def evaluate(model, x, y, batch_size) -> dict:
    cfg = _require(model)
    label_dtype = int32 if cfg.classification else float32
    loader = _loader_for(x, y, batch_size, False, None)
    if loader is not None or _is_stream(x, y):
        return _evaluate_batches(model, cfg, _stream_batches(cfg.ctx, loader or x, label_dtype))
    if y is None:
        raise ValueError("evaluate(x, y): y is required for array inputs")
    data = _Data(cfg.ctx)
    try:
        X, Y = data.put(x, float32), data.put(y, label_dtype)
        return _evaluate_batches(model, cfg, _array_batches(cfg.ctx, X, Y, list(range(len(X))), batch_size))
    finally:
        data.release()


def predict(model, x, batch_size):
    """Model outputs for x. Returns a numpy array for numpy input, else nested lists."""
    ctx = model._context()
    loader = _loader_for(x, None, batch_size, False, None)
    data = _Data(ctx)
    rows: list = []
    was_training = model.training
    model.eval()
    try:
        if loader is not None or _is_stream(x, None):
            batches = _stream_batches(ctx, loader or x, None)
        else:
            X = data.put(x, float32)
            batches = _array_batches(ctx, X, None, list(range(len(X))), batch_size)
        with no_grad(ctx):
            for xb, _ in batches:
                rows.extend(model(xb).tolist())
                ctx.step_end()
    finally:
        data.release()
        model.train(was_training)
    if _is_numpy(x):
        import numpy as np

        return np.asarray(rows, dtype=np.float32)
    return rows
