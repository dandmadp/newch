"""Tests for conv/pool/bn/dropout, save/load, quantization, streaming data and memory ownership."""

import gc
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import newch as nc  # noqa: E402
import newch.net.functional as F  # noqa: E402
from newch import net  # noqa: E402
from newch.data import DataLoader, FileArray, TensorDataset  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def bars(n, size=8, seed=0):
    """1x8x8 images with a horizontal (0) or vertical (1) bar plus noise."""
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        label = rng.randrange(2)
        pos = rng.randrange(size)
        img = [[rng.gauss(0, 0.2) for _ in range(size)] for _ in range(size)]
        for k in range(size):
            if label == 0:
                img[pos][k] += 1.0
            else:
                img[k][pos] += 1.0
        xs.append([img])
        ys.append(label)
    return xs, ys


def cnn():
    return nc.Sequential(
        net.Conv2d(1, 8, 3, padding=1), net.BatchNorm2d(8), net.ReLU(), net.MaxPool2d(2),
        net.Flatten(), net.Dropout(0.1), net.Linear(8 * 4 * 4, 2))


class Base(unittest.TestCase):
    def setUp(self):
        nc.manual_seed(7)
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        nc.step_end()

    def path(self, name):
        return os.path.join(self.tmp, name)


class TestLayers(Base):
    def test_cnn_trains(self):
        xs, ys = bars(400)
        model = cnn().compile(optimizer="adam", loss="cross_entropy", metrics=["accuracy"], lr=0.01)
        hist = model.fit(xs, ys, epochs=6, batch_size=32, validation_split=0.25, verbose=0, seed=1)
        self.assertGreater(hist.history["val_accuracy"][-1], 0.95)

    def test_shapes_and_modes(self):
        x = nc.randn(2, 3, 9, 9)
        self.assertEqual(net.Conv2d(3, 4, 3, stride=2, padding=1)(x).shape, (2, 4, 5, 5))
        self.assertEqual(net.Conv2d(3, 4, (3, 1))(x).shape, (2, 4, 7, 9))
        self.assertEqual(net.MaxPool2d(3, stride=2)(x).shape, (2, 3, 4, 4))
        self.assertEqual(net.AvgPool2d(2)(x).shape, (2, 3, 4, 4))
        drop = net.Dropout(0.5)
        drop.eval()
        self.assertIs(drop(x), x)  # identity in eval mode
        with self.assertRaises(NotImplementedError):
            net.Conv2d(3, 4, 3, padding="same")

    def test_batchnorm_buffers(self):
        bn = net.BatchNorm1d(2, momentum=1.0)
        x = nc.tensor([[1.0, 10.0], [3.0, 30.0]])
        bn(x)
        self.assertEqual(bn.running_mean.tolist(), [2.0, 20.0])
        self.assertEqual(bn.num_batches_tracked.item(), 1)
        self.assertEqual([n for n, _ in bn.named_buffers()], ["running_mean", "running_var", "num_batches_tracked"])
        bn.eval()
        y = bn(nc.tensor([[2.0, 20.0]]))  # equals the running mean -> 0
        self.assertAlmostEqual(y.tolist()[0][0], 0.0, places=5)
        with self.assertRaises(ValueError):
            net.BatchNorm2d(2)(x)


class TestSaveLoad(Base):
    def test_roundtrip_predicts_identically(self):
        xs, _ = bars(16)
        model = cnn()
        model.eval()
        before = model.predict(xs)
        p = self.path("cnn.ncw")
        nc.save_tensors(model.state_dict(), p)
        other = cnn()
        other.eval()
        other.load_state_dict(nc.load_tensors(p))
        self.assertEqual(other.predict(xs), before)
        self.assertEqual(list(nc.load_tensors(p)), list(model.state_dict()))
        self.assertEqual(list(model.state_dict())[:2], ["0.weight", "0.bias"])  # torch naming

    def test_strict_and_mismatch(self):
        model = nc.Sequential(net.Linear(3, 2))
        with self.assertRaises(RuntimeError):
            model.load_state_dict({"0.weight": [[0.0] * 3] * 2})  # missing 0.bias
        res = model.load_state_dict({"0.weight": [[1.0] * 3] * 2, "extra": [1.0]}, strict=False)
        self.assertEqual(res.unexpected_keys, ["extra"])
        self.assertEqual(model[0].weight.tolist(), [[1.0] * 3] * 2)
        with self.assertRaises(RuntimeError):
            model.load_state_dict({"0.weight": [[0.0] * 2] * 3, "0.bias": [0.0, 0.0]})

    @unittest.skipIf(np is None, "numpy not installed")
    def test_torch_style_state_dict(self):
        # what {k: v.numpy() for k, v in torch_model.state_dict().items()} produces
        sd = {"0.weight": np.ones((4, 3), np.float32), "0.bias": np.zeros(4, np.float32),
              "1.weight": np.full(4, 2.0, np.float32), "1.bias": np.zeros(4, np.float32),
              "1.running_mean": np.zeros(4, np.float32), "1.running_var": np.ones(4, np.float32),
              "1.num_batches_tracked": np.array(5, np.int64)}
        model = nc.Sequential(net.Linear(3, 4), net.BatchNorm1d(4))
        model.load_state_dict(sd)
        self.assertEqual(model[1].num_batches_tracked.item(), 5)
        model.eval()
        out = model(nc.tensor([[1.0, 1.0, 1.0]])).tolist()[0]
        self.assertAlmostEqual(out[0], 2.0 * 3.0 / (1 + 1e-5) ** 0.5, places=4)


class TestQuantize(Base):
    def test_quantize_matches_float(self):
        nc.step_end()
        gc.collect()
        model = nc.Sequential(net.Linear(64, 128), net.ReLU(), net.Linear(128, 10))
        x = nc.randn(8, 64)
        with nc.persistent():
            xp = nc.tensor(x)
        ref = model(xp).tolist()
        before = nc.mem_stats()["persistent_bytes"]
        nc.quantize_(model)
        gc.collect()
        after = nc.mem_stats()["persistent_bytes"]
        self.assertIsInstance(model[0], net.QuantizedLinear)
        got = model(xp).tolist()
        err = max(abs(a - b) for ra, rb in zip(got, ref) for a, b in zip(ra, rb))
        scale = max(abs(v) for r in ref for v in r)
        self.assertLess(err / scale, 0.05)
        self.assertLess(after, before * 0.5)  # float weights freed, int8 kept
        p = self.path("q.ncw")
        nc.save_tensors(model.state_dict(), p)
        self.assertEqual(nc.load_tensors(p)["0.weight"].dtype, nc.int8)


class TestData(Base):
    def test_file_array_streaming_fit(self):
        xs, ys = bars(300)
        fx = FileArray.create(self.path("x.bin"), (row for row in xs), row_shape=(1, 8, 8))
        fy = FileArray.create(self.path("y.bin"), ([y] for y in ys), row_shape=(), dtype=nc.int32)
        self.assertEqual(fx.shape, (300, 1, 8, 8))
        self.assertEqual(fy[5], [ys[5]])
        loader = DataLoader(TensorDataset(fx, fy), batch_size=50, shuffle=True)
        self.assertEqual(len(loader), 6)
        xb, yb = next(iter(loader))
        self.assertEqual((xb.shape, yb.shape, yb.dtype), ((50, 1, 8, 8), (50,), nc.int32))
        nc.step_end()

        model = cnn().compile(optimizer="adam", loss="cross_entropy", metrics=["accuracy"], lr=0.01)
        before = nc.mem_stats()["persistent_bytes"]
        hist = model.fit(loader, epochs=5, verbose=0)  # streamed, never fully in memory
        self.assertGreater(hist.history["accuracy"][-1], 0.9)
        hist2 = model.fit(fx, fy, epochs=1, batch_size=50, verbose=0)  # FileArray auto-streams
        self.assertIn("loss", hist2.history)
        grads = sum(p.numel() * 4 for p in model.parameters())
        self.assertLessEqual(nc.mem_stats()["persistent_bytes"] - before, grads + 4096)  # only grad buffers
        self.assertGreater(model.evaluate(loader)[1], 0.9)
        self.assertEqual(len(model.predict(DataLoader(TensorDataset(fx), batch_size=64))), 300)
        fx.close()
        fy.close()

    def test_generic_dataset_collate(self):
        class Squares(nc.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, i):
                return [float(i)], i * i

        batches = list(DataLoader(Squares(), batch_size=4))
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0][1].tolist(), [0, 1, 4, 9])
        self.assertEqual(batches[2][0].shape, (2, 1))
        nc.step_end()


class TestMemory(Base):
    def test_persistent_tensors_freed_on_gc(self):
        nc.step_end()
        gc.collect()
        base = nc.mem_stats()["persistent_bytes"]
        for _ in range(50):
            with nc.persistent():
                t = nc.zeros(256, 256)
            del t
        lin = net.Linear(512, 512)
        g = None
        lin(nc.randn(2, 512)).sum().backward()
        g = lin.weight.grad
        del lin
        gc.collect()
        self.assertEqual(g.shape, (512, 512))  # grad keeps its parameter alive
        del g
        gc.collect()
        nc.step_end()
        self.assertEqual(nc.mem_stats()["persistent_bytes"], base)

    def test_persistent_temporaries_survive_calls(self):
        # Temporaries made inside persistent() are owned and freed on GC; they
        # must stay alive until the C call that uses them returns. Run with
        # NC_DEBUG_POISON=1 (tools/build.py does) so a regression reads garbage.
        base = nc.randn(3)
        base_values = base.tolist()
        with nc.persistent():
            p = nc.tensor([1.0, 2.0, 3.0])
            copy = nc.tensor(base)                  # detach() temporary
            added = p + [10.0, 20.0, 30.0]          # list operand temporary
            prod = p * nc.tensor([2.0, 2.0, 2.0])
            rsub = [1.0, 1.0, 1.0] - p
            mse = F.mse_loss(p, [1.0, 2.0, 5.0])
        nc.step_end()
        self.assertEqual(copy.tolist(), base_values)
        self.assertEqual(added.tolist(), [11.0, 22.0, 33.0])
        self.assertEqual(prod.tolist(), [2.0, 4.0, 6.0])
        self.assertEqual(rsub.tolist(), [0.0, -1.0, -2.0])
        self.assertAlmostEqual(mse.item(), 4.0 / 3.0, places=5)

    def test_threads_and_trim(self):
        a, b = [], []
        for threads, out in ((1, a), (4, b)):
            ctx = nc.Context(seed=3, threads=threads)
            self.assertEqual(ctx.threads, threads)
            x = nc.randn(200, 300, ctx=ctx)
            w = nc.randn(300, 100, ctx=ctx)
            out.append((x @ w).tolist())
            ctx.step_end()
            ctx.trim()
            self.assertEqual(ctx.mem_stats()["arena_bytes_reserved"], 0)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
