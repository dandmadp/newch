"""Python API tests (stdlib unittest; numpy tests are skipped without numpy)."""

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import newch as nc  # noqa: E402
import newch.net.functional as F  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def spiral(per_class=100, seed=0):
    rng = random.Random(seed)
    xs, ys = [], []
    for c in range(3):
        for i in range(per_class):
            r = i / per_class
            th = c * 4.0 + r * 4.0 + rng.gauss(0, 0.2)
            xs.append([r * math.sin(th), r * math.cos(th)])
            ys.append(c)
    order = list(range(len(xs)))
    rng.shuffle(order)  # validation_split takes the tail, so classes must be mixed
    return [xs[i] for i in order], [ys[i] for i in order]


class Base(unittest.TestCase):
    def setUp(self):
        nc.manual_seed(1234)

    def tearDown(self):
        nc.step_end()


class TestTensor(Base):
    def test_create_and_roundtrip(self):
        t = nc.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(t.shape, (2, 3))
        self.assertEqual(t.dtype, nc.float32)
        self.assertEqual(t.tolist(), [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(nc.tensor([1, 2, 3]).dtype, nc.int32)
        self.assertEqual(nc.tensor(2.5).shape, ())
        self.assertAlmostEqual(nc.tensor(2.5).item(), 2.5)
        with self.assertRaises(ValueError):
            nc.tensor([[1, 2], [3]])

    def test_arithmetic_and_broadcast(self):
        a = nc.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = nc.tensor([10.0, 20.0, 30.0])
        self.assertEqual((a + b).tolist()[1], [14, 25, 36])
        self.assertEqual((a * 2).tolist()[0], [2, 4, 6])
        self.assertEqual((1 - a).tolist()[0], [0, -1, -2])
        self.assertAlmostEqual((6 / a).tolist()[0][2], 2.0, places=6)
        self.assertAlmostEqual((a ** 2).sum().item(), 91.0, places=4)
        self.assertEqual((-a).tolist()[0][0], -1)
        self.assertEqual(a.sum(axis=0).tolist(), [5, 7, 9])
        self.assertEqual(a.amax(dim=1).tolist(), [3, 6])
        self.assertEqual(a.sum(axis=1).tolist(), [6, 15])  # numpy alias
        self.assertEqual(a.argmax(-1).tolist(), [2, 2])

    def test_torch_semantics(self):
        a = nc.tensor([[1.0, 9.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(a.size(), (2, 3))
        self.assertEqual(a.size(1), 3)
        self.assertEqual(a.argmax().item(), 1)  # flattened by default
        self.assertEqual(a.flatten().shape, (6,))
        self.assertEqual(a.T.shape, (3, 2))
        b = nc.zeros(4, 2, 3)
        self.assertEqual(b.mT.shape, (4, 3, 2))
        self.assertEqual(nc.net.Flatten()(b).shape, (4, 6))
        with self.assertRaises(NotImplementedError):
            b.T
        with self.assertRaises(NotImplementedError):
            a.softmax(dim=0)
        with self.assertRaises(NotImplementedError):
            a.index_select(1, [0])
        self.assertEqual(a.view(3, 2).shape, (3, 2))  # never copies, as in PyTorch ...
        with self.assertRaises(RuntimeError):
            a.T.view(6)  # ... so a non-contiguous tensor needs reshape()
        self.assertFalse(hasattr(a, "max"))

    def test_manual_seed_reseeds(self):
        nc.manual_seed(7)
        first = nc.randn(4).tolist()
        kept = nc.parameter((2,))
        nc.manual_seed(7)
        self.assertEqual(nc.randn(4).tolist(), first)
        self.assertEqual(kept.tolist(), [0, 0])  # existing tensors survive

    def test_matmul_and_views(self):
        a = nc.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual((a @ a.T).tolist(), [[14, 32], [32, 77]])
        self.assertEqual(a.reshape(3, 2).shape, (3, 2))
        self.assertEqual(a.reshape(-1).shape, (6,))
        self.assertEqual(a[1].tolist(), [4, 5, 6])
        self.assertEqual(a[[1, 0, 1]].shape, (3, 3))
        self.assertEqual(a[0:1].tolist(), [[1, 2, 3]])

    def test_autograd(self):
        x = nc.parameter([1.0, -2.0, 3.0])
        y = (x * x).sum() + (x * 3).sum()
        y.backward()
        self.assertEqual(x.grad.tolist(), [5, -1, 9])  # 2x + 3
        x.zero_grad()
        with nc.no_grad():
            z = x * x
        self.assertFalse(z.requires_grad)

    def test_freed_tensor_raises(self):
        t = nc.tensor([1.0, 2.0])
        nc.step_end()
        with self.assertRaises(nc.NcError):
            t.tolist()
        with nc.persistent():
            keep = nc.tensor([1.0, 2.0])
        nc.step_end()
        self.assertEqual(keep.tolist(), [1, 2])
        nc.release(keep)
        with self.assertRaises(nc.NcError):
            keep.tolist()

    def test_engine_errors(self):
        with self.assertRaises(nc.NcError) as cm:
            nc.zeros(2, 3) + nc.zeros(4, 3)
        self.assertIn("broadcastable", str(cm.exception))
        self.assertEqual((nc.zeros(2) + 1).tolist(), [1, 1])  # context recovered
        with self.assertRaises(nc.NcError):
            F.cross_entropy(nc.zeros(2, 3), [0, 5])

    @unittest.skipIf(np is None, "numpy not installed")
    def test_numpy(self):
        arr = np.arange(6, dtype=np.float64).reshape(2, 3)
        t = nc.tensor(arr)
        self.assertEqual(t.dtype, nc.float32)
        np.testing.assert_allclose(t.numpy(), arr)
        self.assertEqual(nc.tensor(np.array([1, 2])).dtype, nc.int32)


class TestModules(Base):
    def test_parameter_discovery(self):
        class Net(nc.net.Module):
            def __init__(self):
                self.body = nc.Sequential(nc.net.Linear(2, 4), nc.net.ReLU())
                self.head = nc.net.Linear(4, 3, bias=False)
                self.scale = nc.parameter((1,))

            def forward(self, x):
                return self.head(self.body(x)) * self.scale

        net = Net()
        names = [n for n, _ in net.named_parameters()]
        self.assertEqual(names, ["body.0.weight", "body.0.bias", "head.weight", "scale"])
        self.assertEqual(net.num_parameters(), 2 * 4 + 4 + 4 * 3 + 1)
        self.assertEqual(net(nc.randn(5, 2)).shape, (5, 3))

    def test_linear_matches_torch_layout(self):
        lin = nc.net.Linear(3, 2)
        self.assertEqual(lin.weight.shape, (2, 3))  # [out, in]
        self.assertEqual(lin.bias.shape, (2,))
        x = nc.randn(5, 3)
        manual = x @ lin.weight.T + lin.bias
        for got, want in zip(lin(x).tolist(), manual.tolist()):
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, places=5)
        self.assertEqual(F.linear(x, lin.weight).shape, (5, 2))
        self.assertIn("in_features=3", repr(lin))

    def test_embedding(self):
        emb = nc.net.Embedding(10, 4)
        out = emb([1, 1, 7])
        self.assertEqual(out.shape, (3, 4))
        out.sum().backward()
        g = emb.weight.grad.tolist()
        self.assertEqual(g[1], [2, 2, 2, 2])
        self.assertEqual(g[0], [0, 0, 0, 0])

    def test_pytorch_style_loop(self):
        xs, ys = spiral(60)
        model = nc.Sequential(nc.net.Linear(2, 32), nc.net.ReLU(), nc.net.Linear(32, 3))
        opt = nc.optim.Adam(model.parameters(), lr=0.02)
        with nc.persistent():
            X, Y = nc.tensor(xs), nc.tensor(ys)
        first = None
        for _ in range(200):
            loss = F.cross_entropy(model(X), Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
            last = loss.item()
            nc.step_end()
        self.assertLess(last, first * 0.4)


class TestKeras(Base):
    def make(self):
        model = nc.Sequential([nc.net.Linear(2, 32), nc.net.ReLU(), nc.net.Linear(32, 32), nc.net.ReLU(),
                               nc.net.Linear(32, 3)])
        return model.compile(optimizer="adam", loss="cross_entropy", metrics=["accuracy"], lr=0.01)

    def test_fit_evaluate_predict(self):
        xs, ys = spiral(100)
        model = self.make()
        hist = model.fit(xs, ys, epochs=40, batch_size=32, validation_split=0.2, verbose=0, seed=0)
        self.assertEqual(len(hist.history["loss"]), 40)
        self.assertIn("val_accuracy", hist.history)
        self.assertGreater(hist.history["accuracy"][-1], 0.9)
        # dataset copies are released after fit, so repeated fits do not grow memory
        before = nc.mem_stats()["persistent_bytes"]
        model.fit(xs, ys, epochs=1, verbose=0)
        self.assertEqual(nc.mem_stats()["persistent_bytes"], before)

        loss, acc = model.evaluate(xs, ys)  # Keras default: [loss, metric]
        self.assertGreater(acc, 0.9)
        self.assertEqual(set(model.evaluate(xs, ys, return_dict=True)), {"loss", "accuracy"})
        pred = model.predict(xs[:5])
        self.assertEqual((len(pred), len(pred[0])), (5, 3))

    def test_early_stopping(self):
        xs, ys = spiral(30)
        model = self.make()
        stop = nc.callbacks.EarlyStopping(monitor="loss", min_delta=10.0, patience=1)
        hist = model.fit(xs, ys, epochs=50, verbose=0, callbacks=[stop])
        self.assertEqual(len(hist.history["loss"]), 2)  # never "improves" by 10
        self.assertEqual(stop.stopped_epoch, 1)

    def test_regression_mse(self):
        rng = random.Random(0)
        xs = [[rng.uniform(-1, 1)] for _ in range(256)]
        ys = [3 * x[0] + 1 for x in xs]  # 1-D targets vs [B, 1] output
        model = nc.Sequential(nc.net.Linear(1, 1)).compile(optimizer="sgd", loss="mse", lr=0.1)
        model.fit(xs, ys, epochs=30, batch_size=32, verbose=0)
        w, b = model.parameters()
        self.assertAlmostEqual(w.item(), 3.0, places=2)
        self.assertAlmostEqual(b.item(), 1.0, places=2)

    def test_compile_errors(self):
        model = nc.Sequential(nc.net.Linear(1, 1))
        with self.assertRaises(ValueError):
            model.compile(loss="mse", metrics=["accuracy"])
        with self.assertRaises(ValueError):  # expects probabilities in Keras; not an alias
            model.compile(loss="sparse_categorical_crossentropy")
        with self.assertRaises(RuntimeError):
            nc.Sequential(nc.net.Linear(1, 1)).fit([[0.0]], [0.0], verbose=0)


if __name__ == "__main__":
    unittest.main()
