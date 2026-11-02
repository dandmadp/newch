"""Datasets and a streaming DataLoader (torch.utils.data style).

For data larger than RAM, store samples in a raw binary file and open it
with FileArray: rows are read through mmap one batch at a time, so memory use
stays around one batch regardless of the dataset size.

    FileArray.create("x.bin", rows, row_shape=(28, 28))     # write once
    ds = TensorDataset(FileArray("x.bin", (60000, 28, 28)), labels)
    for xb, yb in DataLoader(ds, batch_size=64, shuffle=True): ...
"""

from __future__ import annotations

import array
import mmap
import random
from pathlib import Path

from . import _lib as L
from .tensor import (Context, Tensor, _flatten, _is_numpy, _numpy_dtype, _unsupported, default_context, float32,
                     int32, tensor, tensor_from_bytes)


__all__ = ["DataLoader", "Dataset", "FileArray", "TensorDataset"]


class Dataset:
    """Map-style dataset: implement ``__len__`` and ``__getitem__``."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


# --------------------------------------------------------------------------
# Row sources: array-likes that can hand out a batch of rows as raw bytes
# --------------------------------------------------------------------------


class FileArray:
    """A read-only array stored row-major in a raw binary file (like numpy.memmap).

    shape includes the number of rows; dtype is "float32", "int32", "int8" or "uint8"
    (uint8 keeps raw pixels at a quarter of the float32 size: convert batches with ``.float()``).
    """

    def __init__(self, path, shape, dtype: str = float32, offset: int = 0):
        if dtype not in L.DTYPE_CODES:
            raise ValueError(f"unsupported dtype {dtype!r}")
        self.path, self.shape, self.dtype, self.offset = Path(path), tuple(int(s) for s in shape), dtype, offset
        self.row_shape = self.shape[1:]
        n = 1
        for d in self.row_shape:
            n *= d
        self.row_bytes = n * L.DTYPE_SIZE[dtype]
        need = offset + self.shape[0] * self.row_bytes
        size = self.path.stat().st_size
        if size < need:
            raise ValueError(f"{path}: {size} bytes, but shape {self.shape} {dtype} needs {need}")
        self._f = open(self.path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ) if need else None

    @classmethod
    def create(cls, path, rows, row_shape, dtype: str = float32) -> "FileArray":
        """Writes an iterable of rows (sequences / numpy rows) without holding them all in memory."""
        code = L.ARRAY_CODE[dtype]
        count = 0
        with open(path, "wb") as f:
            for row in rows:
                if _is_numpy(row):
                    import numpy as np

                    f.write(np.ascontiguousarray(row, dtype=dtype).tobytes())
                else:
                    flat: list = []
                    _flatten(row, [], flat) if isinstance(row, (list, tuple)) else flat.append(row)
                    f.write(array.array(code, flat if dtype == float32 else map(int, flat)).tobytes())
                count += 1
        return cls(path, (count, *row_shape), dtype)

    def __len__(self) -> int:
        return self.shape[0]

    def rows(self, indices) -> bytes:
        rb, off, mm = self.row_bytes, self.offset, self._mm
        if not indices:
            return b""
        first = indices[0]
        if indices == list(range(first, first + len(indices))):  # contiguous: one slice
            return mm[off + first * rb: off + (first + len(indices)) * rb]
        return b"".join(mm[off + i * rb: off + (i + 1) * rb] for i in indices)

    def __getitem__(self, i: int):
        return array.array(L.ARRAY_CODE[self.dtype], self.rows([i % len(self)])).tolist()

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        self._f.close()


class _MemoryRows:
    """In-memory row source built once from lists, array.array, numpy or a Tensor."""

    def __init__(self, data):
        if isinstance(data, Tensor):
            self.dtype, shape = data.dtype, data.shape
            flat = data._flat()
            self.buf = flat.tobytes()
        elif _is_numpy(data):
            import numpy as np

            kind = _numpy_dtype(data)
            arr = np.ascontiguousarray(data, dtype=kind)
            self.dtype, shape, self.buf = kind, arr.shape, arr.tobytes()
        else:
            shape_l: list = []
            flat_l: list = []
            _flatten(list(data), shape_l, flat_l)
            self.dtype = float32 if any(isinstance(v, float) for v in flat_l) else int32
            self.buf = array.array(L.ARRAY_CODE[self.dtype],
                                   flat_l if self.dtype == float32 else map(int, flat_l)).tobytes()
            shape = tuple(shape_l)
        if not shape:
            raise ValueError("dataset arrays need at least one dimension (the sample axis)")
        self.shape = tuple(shape)
        self.row_shape = self.shape[1:]
        n = 1
        for d in self.row_shape:
            n *= d
        self.row_bytes = n * L.DTYPE_SIZE[self.dtype]
        self._mv = memoryview(self.buf)

    def __len__(self) -> int:
        return self.shape[0]

    def rows(self, indices) -> bytes:
        rb, mv = self.row_bytes, self._mv
        if not indices:
            return b""
        first = indices[0]
        if indices == list(range(first, first + len(indices))):
            return bytes(mv[first * rb:(first + len(indices)) * rb])
        return b"".join(mv[i * rb:(i + 1) * rb] for i in indices)

    def __getitem__(self, i: int):
        return array.array(L.ARRAY_CODE[self.dtype], self.rows([i % len(self)])).tolist()


def _row_source(data):
    return data if isinstance(data, FileArray) else _MemoryRows(data)


class TensorDataset(Dataset):
    """Samples are rows of each array (torch.utils.data.TensorDataset).

    Arrays may be nested lists, array.array, numpy arrays, Tensors or FileArray
    (disk-backed). numpy uint8/int8 data keeps its 1-byte dtype, other integer
    data becomes int32, floating data float32.
    """

    def __init__(self, *arrays):
        if not arrays:
            raise ValueError("TensorDataset needs at least one array")
        self.sources = [_row_source(a) for a in arrays]
        n = len(self.sources[0])
        if any(len(s) != n for s in self.sources):
            raise ValueError("all arrays must have the same number of rows")

    def __len__(self) -> int:
        return len(self.sources[0])

    def __getitem__(self, index):
        return tuple(s[index] for s in self.sources)

    def get_batch(self, indices, ctx: Context) -> list[Tensor]:
        return [tensor_from_bytes(s.rows(indices), (len(indices), *s.row_shape), s.dtype, ctx)
                for s in self.sources]


class DataLoader:
    """Iterates a dataset in batches (torch.utils.data.DataLoader).

    Yields a list with one temporary Tensor per field, valid until the next
    ``step_end()``. TensorDataset batches are gathered directly as bytes; other
    datasets are collated from ``__getitem__``.
    """

    def __init__(self, dataset, batch_size: int = 1, shuffle: bool | None = None, sampler=None, batch_sampler=None,
                 num_workers: int = 0, collate_fn=None, pin_memory: bool = False, drop_last: bool = False, *,
                 generator: random.Random | None = None, ctx: Context | None = None):
        _unsupported(sampler=(sampler, None), batch_sampler=(batch_sampler, None), num_workers=(num_workers, 0),
                     pin_memory=(pin_memory, False))
        if batch_size is None or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.dataset, self.batch_size, self.shuffle, self.drop_last = dataset, batch_size, bool(shuffle), drop_last
        self.collate_fn = collate_fn
        self.generator = generator or random.Random()
        self._ctx = ctx

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        ctx = self._ctx or default_context()
        order = list(range(len(self.dataset)))
        if self.shuffle:
            self.generator.shuffle(order)
        bs = self.batch_size
        stop = len(order) - (len(order) % bs if self.drop_last else 0)
        for s in range(0, stop, bs):
            idx = order[s:s + bs]
            if self.collate_fn is not None:
                yield self.collate_fn([self.dataset[i] for i in idx])
            elif hasattr(self.dataset, "get_batch"):
                yield self.dataset.get_batch(idx, ctx)
            else:
                yield _collate([self.dataset[i] for i in idx], ctx)


def _stack(items, ctx: Context) -> Tensor:
    if _is_numpy(items[0]):
        import numpy as np

        return tensor(np.stack(items), ctx=ctx)
    return tensor(list(items), ctx=ctx)


def _collate(samples, ctx: Context):
    """Tuple samples -> list of batched fields (as torch's default_collate); else one Tensor."""
    if isinstance(samples[0], tuple):
        return [_stack([s[k] for s in samples], ctx) for k in range(len(samples[0]))]
    return _stack(samples, ctx)
