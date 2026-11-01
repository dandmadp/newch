"""Compares activation functions: element throughput and MNIST MLP accuracy (float and int8).

    python tools/bench_act.py                 # full run (needs MNIST, see examples/mnist_cnn.py)
    python tools/bench_act.py --speed-only    # throughput only, no data needed
    python tools/bench_act.py --seeds 1 --epochs 1

Prints Markdown tables. Elementwise ops are single-threaded, so throughput is per core.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import newch as nc  # noqa: E402
from newch import net  # noqa: E402

ACTS = {
    "relu": (nc.relu, net.ReLU),
    "leaky_relu": (nc.leaky_relu, net.LeakyReLU),
    "relu6": (nc.relu6, net.ReLU6),
    "hardswish": (nc.hardswish, net.Hardswish),
    "hardsigmoid": (nc.hardsigmoid, net.Hardsigmoid),
    "silu": (nc.silu, net.SiLU),
    "gelu": (nc.gelu, net.GELU),
    "gelu_tanh": (lambda x: nc.gelu(x, approximate="tanh"), lambda: net.GELU("tanh")),
    "tanh": (nc.tanh, net.Tanh),
    "sigmoid": (nc.sigmoid, net.Sigmoid),
}


def speed(n: int, reps: int) -> dict:
    out = {}
    for name, (fn, _) in ACTS.items():
        x = nc.parameter(nc.randn(n))
        fwd, both = [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            y = fn(x)
            t1 = time.perf_counter()
            y.sum().backward()
            t2 = time.perf_counter()
            fwd.append(t1 - t0)
            both.append(t2 - t0)
            nc.step_end()
        out[name] = (n / statistics.median(fwd) / 1e6, n / statistics.median(both) / 1e6)
        nc.release(x)
    return out


def accuracy(names, seeds: int, epochs: int, data_dir: Path) -> dict:
    from mnist_cnn import prepare  # noqa: E402  (downloads and converts once)
    from newch.data import DataLoader, TensorDataset

    xtr, ytr = prepare(data_dir, "train")
    xte, yte = prepare(data_dir, "test")

    class Flat(nc.data.Dataset):  # [N, 1, 28, 28] rows -> [B, 784] batches
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def get_batch(self, idx, ctx):
            xb, yb = self.ds.get_batch(idx, ctx)
            return [xb.reshape(len(idx), 784), yb]

    train, test = Flat(TensorDataset(xtr, ytr)), Flat(TensorDataset(xte, yte))
    results = {}
    for name in names:
        layer = ACTS[name][1]
        accs, q_accs, times = [], [], []
        for seed in range(seeds):
            nc.manual_seed(seed)
            model = nc.Sequential(net.Linear(784, 256), layer(), net.Linear(256, 128), layer(),
                                  net.Linear(128, 10))
            model.compile(optimizer=nc.optim.Adam(model.parameters(), lr=1e-3), loss="cross_entropy",
                          metrics=["accuracy"])
            import random

            loader = DataLoader(train, batch_size=64, shuffle=True, generator=random.Random(seed))
            t0 = time.perf_counter()
            model.fit(loader, epochs=epochs, verbose=0)
            times.append((time.perf_counter() - t0) / epochs)
            test_loader = DataLoader(test, batch_size=256)
            accs.append(model.evaluate(test_loader)[1])
            nc.quantize_(model)
            q_accs.append(model.evaluate(test_loader)[1])
        results[name] = (accs, q_accs, times)
        print(f"  {name}: {statistics.mean(accs):.4f}", file=sys.stderr, flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed-only", action="store_true")
    ap.add_argument("--n", type=int, default=1 << 20)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--data", default=str(Path.home() / ".cache" / "newch" / "mnist"))
    args = ap.parse_args()

    ctx = nc.default_context()
    print(f"backend {ctx.backend}, arch {ctx.cpu['arch']}, {ctx.threads} threads\n")
    sp = speed(args.n, args.reps)
    base_f, base_b = sp["relu"]
    print(f"### Throughput ({args.n} elements, median of {args.reps}, M elements/s)\n")
    print("| activation | forward | fwd+bwd | fwd+bwd vs relu |")
    print("|---|---:|---:|---:|")
    for name, (f, b) in sp.items():
        print(f"| {name} | {f:.0f} | {b:.0f} | {b / base_b:.2f}x |")
    if args.speed_only:
        return 0

    print(f"\n### MNIST MLP 784-256-128-10, Adam 1e-3, batch 64, {args.epochs} epochs, {args.seeds} seeds\n",
          flush=True)
    res = accuracy(list(ACTS), args.seeds, args.epochs, Path(args.data))
    print("| activation | test acc (mean ± sd) | int8 acc | int8 change (pt) | s/epoch |")
    print("|---|---:|---:|---:|---:|")
    for name, (accs, q, t) in res.items():
        sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
        change = statistics.mean(b - a for a, b in zip(accs, q))
        print(f"| {name} | {100 * statistics.mean(accs):.2f} ± {100 * sd:.2f} | {100 * statistics.mean(q):.2f} "
              f"| {100 * change:+.2f} | {statistics.mean(t):.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
