"""MNIST with a small CNN, streamed from disk.

    python examples/mnist_cnn.py [--epochs 2] [--data ~/.cache/newch/mnist]

The first run downloads MNIST (~11 MB) and converts it to raw FileArray
files; training then reads one batch at a time, so peak memory stays near
one batch plus the model, whatever the dataset size.
"""

from __future__ import annotations

import argparse
import array
import gzip
import os
import struct
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import newch as nc  # noqa: E402
from newch import net  # noqa: E402
from newch.data import DataLoader, FileArray, TensorDataset  # noqa: E402

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 60000),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 10000),
}


def fetch(data_dir: Path, name: str) -> Path:
    path = data_dir / name
    if not path.exists():
        print(f"downloading {name} ...", flush=True)
        urllib.request.urlretrieve(MIRROR + name, path)
    return path


def prepare(data_dir: Path, split: str):
    """IDX (gzip) -> float32 images [N, 1, 28, 28] in [0, 1] and int32 labels, as FileArrays."""
    img_gz, lab_gz, n = FILES[split]
    x_bin, y_bin = data_dir / f"{split}_x.f32", data_dir / f"{split}_y.i32"
    if not (x_bin.exists() and y_bin.exists()):
        with gzip.open(fetch(data_dir, img_gz), "rb") as f:
            _, count, rows, cols = struct.unpack(">IIII", f.read(16))
            with open(x_bin, "wb") as out:
                for _ in range(count):  # one image at a time: no full copy in memory
                    px = f.read(rows * cols)
                    out.write(array.array("f", (p / 255.0 for p in px)).tobytes())
        with gzip.open(fetch(data_dir, lab_gz), "rb") as f:
            f.read(8)
            y_bin.write_bytes(array.array("i", list(f.read())).tobytes())  # uint8 labels -> int32
    return FileArray(x_bin, (n, 1, 28, 28)), FileArray(y_bin, (n,), dtype=nc.int32)


def build_model() -> nc.Sequential:
    return nc.Sequential(
        net.Conv2d(1, 16, 3, padding=1), net.BatchNorm2d(16), net.ReLU(), net.MaxPool2d(2),
        net.Conv2d(16, 32, 3, padding=1), net.BatchNorm2d(32), net.ReLU(), net.MaxPool2d(2),
        net.Flatten(), net.Dropout(0.25), net.Linear(32 * 7 * 7, 10),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training samples")
    ap.add_argument("--data", default=str(Path.home() / ".cache" / "newch" / "mnist"))
    args = ap.parse_args()

    data_dir = Path(args.data)
    data_dir.mkdir(parents=True, exist_ok=True)
    xtr, ytr = prepare(data_dir, "train")
    xte, yte = prepare(data_dir, "test")
    train = TensorDataset(xtr, ytr)
    if args.limit:
        class Head(nc.data.Dataset):  # first N samples, still streamed
            def __len__(self): return args.limit
            def get_batch(self, idx, ctx): return train.get_batch(idx, ctx)
        train = Head()

    ctx = nc.default_context()
    print(f"backend {ctx.backend}, {ctx.threads} threads, {len(train)} training samples")
    model = build_model()
    model.summary()
    model.compile(optimizer="adam", loss="cross_entropy", metrics=["accuracy"], lr=1e-3)

    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(xte, yte), batch_size=256)
    t0 = time.perf_counter()
    model.fit(loader, epochs=args.epochs, validation_data=test_loader)
    print(f"trained in {time.perf_counter() - t0:.1f}s, "
          f"arena peak {nc.mem_stats()['arena_peak_bytes'] / 2**20:.1f} MiB")

    weights = data_dir / "mnist_cnn.ncw"
    model.save_weights(weights)
    float_size = os.path.getsize(weights)
    nc.quantize_(model)
    loss, acc = model.evaluate(test_loader)
    q_path = data_dir / "mnist_cnn_int8.ncw"
    model.save_weights(q_path)
    print(f"int8 Linear: test accuracy {acc:.4f}, weights {float_size / 1024:.0f} KiB -> "
          f"{os.path.getsize(q_path) / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
