"""MNIST with a small CNN, streamed from disk.

    python examples/mnist_cnn.py [--epochs 2] [--data ~/.cache/newch/mnist]

The first run downloads MNIST (~11 MB) and unpacks it to raw uint8 FileArray
files (55 MB instead of 220 MB as float32); training then reads one batch at a
time and converts it to float, so peak memory stays near one batch plus the
model, whatever the dataset size.
"""

from __future__ import annotations

import argparse
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
    """IDX (gzip) -> uint8 images [N, 1, 28, 28] and uint8 labels, as FileArrays (the IDX payload as is)."""
    img_gz, lab_gz, n = FILES[split]
    x_bin, y_bin = data_dir / f"{split}_x.u8", data_dir / f"{split}_y.u8"
    if not (x_bin.exists() and y_bin.exists()):
        with gzip.open(fetch(data_dir, img_gz), "rb") as f:
            _, count, rows, cols = struct.unpack(">IIII", f.read(16))
            with open(x_bin, "wb") as out:
                for _ in range(count):  # one image at a time: no full copy in memory
                    out.write(f.read(rows * cols))
        with gzip.open(fetch(data_dir, lab_gz), "rb") as f:
            f.read(8)
            y_bin.write_bytes(f.read())
    return FileArray(x_bin, (n, 1, 28, 28), dtype=nc.uint8), FileArray(y_bin, (n,), dtype=nc.uint8)


class Scaled(nc.data.Dataset):
    """Batches of a uint8 dataset as float32 pixels in [0, 1] (fit() feeds x to the model as is)."""

    def __init__(self, ds, limit: int = 0):
        self.ds, self.limit = ds, limit

    def __len__(self):
        return self.limit or len(self.ds)

    def get_batch(self, idx, ctx):
        x, y = self.ds.get_batch(idx, ctx)
        return [x.float() * (1 / 255), y]


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
    train = Scaled(TensorDataset(xtr, ytr), limit=args.limit)  # --limit: first N samples, still streamed

    ctx = nc.default_context()
    print(f"backend {ctx.backend}, {ctx.threads} threads, {len(train)} training samples")
    model = build_model()
    model.summary()
    model.compile(optimizer="adam", loss="cross_entropy", metrics=["accuracy"], lr=1e-3)

    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Scaled(TensorDataset(xte, yte)), batch_size=256)
    t0 = time.perf_counter()
    model.fit(loader, epochs=args.epochs, validation_data=test_loader)
    print(f"trained in {time.perf_counter() - t0:.1f}s, "
          f"arena peak {nc.mem_stats()['arena_peak_bytes'] / 2**20:.1f} MiB")

    weights = data_dir / "mnist_cnn.ncw"
    nc.save_tensors(model.state_dict(), weights)
    float_size = os.path.getsize(weights)
    nc.quantize_(model)
    loss, acc = model.evaluate(test_loader)
    q_path = data_dir / "mnist_cnn_int8.ncw"
    nc.save_tensors(model.state_dict(), q_path)
    print(f"int8 Linear: test accuracy {acc:.4f}, weights {float_size / 1024:.0f} KiB -> "
          f"{os.path.getsize(q_path) / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
