"""Tests for training extras (AdamW, gradient accumulation, clipping, schedulers,
Keras callbacks), numpy interop and the scikit-learn style newch.ml module."""

import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import newch as nc  # noqa: E402
from newch import ml, net  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


class Base(unittest.TestCase):
    def setUp(self):
        nc.manual_seed(11)

    def tearDown(self):
        nc.step_end()


class TestOptimExtras(Base):
    def test_adamw_first_step(self):
        w = nc.parameter([1.0, -2.0])
        opt = nc.optim.AdamW([w], lr=0.1, weight_decay=0.5)
        (w * 3.0).sum().backward()
        opt.step()
        # decay first, then the Adam step of size lr (m / sqrt(v) = 1 at t = 1)
        for got, w0 in zip(w.tolist(), [1.0, -2.0]):
            self.assertAlmostEqual(got, w0 * 0.95 - 0.1, places=5)

    def test_gradient_accumulation_matches_big_batch(self):
        xs = [[random.Random(i).uniform(-1, 1) for _ in range(3)] for i in range(8)]
        ys = [sum(r) for r in xs]

        def run(batch, accum):
            nc.manual_seed(5)
            model = nc.Sequential(net.Linear(3, 1))
            opt = nc.optim.SGD(model.parameters(), lr=0.1, gradient_accumulation_steps=accum)
            for s in range(0, 8, batch):
                loss = nc.mse_loss(model(nc.tensor(xs[s:s + batch])), nc.tensor([[y] for y in ys[s:s + batch]]))
                opt.zero_grad()
                loss.backward()
                opt.step()
                nc.step_end()
            return [p.tolist() for p in model.parameters()]

        big, small = run(8, None), run(2, 4)  # 4 x batch 2 == one batch of 8 (mean losses)
        for a, b in zip(big, small):
            for x, y in zip(sum(a, []) if isinstance(a[0], list) else a, sum(b, []) if isinstance(b[0], list) else b):
                self.assertAlmostEqual(x, y, places=5)
        with self.assertRaises(ValueError):
            nc.optim.Adam([nc.parameter((1,))], gradient_accumulation_steps=0)

    def test_clip_grad_norm(self):
        p = nc.parameter([0.0, 0.0])
        (p * nc.tensor([3.0, 4.0])).sum().backward()
        total = nc.net.utils.clip_grad_norm_([p], max_norm=1.0)
        self.assertAlmostEqual(total.item(), 5.0, places=4)
        self.assertAlmostEqual(p.grad.tolist()[1], 0.8, places=5)
        with self.assertRaises(NotImplementedError):
            net.clip_grad_norm_([p], 1.0, norm_type=1)

    def test_schedulers(self):
        opt = nc.optim.SGD([nc.parameter((1,))], lr=1.0)
        sched = nc.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        lrs = []
        for _ in range(5):
            lrs.append(sched.get_last_lr()[0])
            sched.step()
        self.assertEqual(lrs, [1.0, 1.0, 0.5, 0.5, 0.25])
        opt2 = nc.optim.SGD([nc.parameter((1,))], lr=1.0)
        cos = nc.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=4, eta_min=0.0)
        for _ in range(2):
            cos.step()
        self.assertAlmostEqual(opt2.lr, 0.5, places=6)
        opt3 = nc.optim.SGD([nc.parameter((1,))], lr=2.0)
        nc.optim.lr_scheduler.LambdaLR(opt3, lambda e: 0.1).step()
        self.assertAlmostEqual(opt3.learning_rate, 0.2, places=6)


class TestCallbacks(Base):
    def data(self):
        rng = random.Random(0)
        xs = [[rng.uniform(-1, 1)] for _ in range(64)]
        return xs, [2 * x[0] for x in xs]

    def test_lr_scheduler_and_plateau(self):
        xs, ys = self.data()
        model = nc.Sequential(net.Linear(1, 1)).compile(optimizer="sgd", loss="mse", lr=0.1)
        seen = []
        model.fit(xs, ys, epochs=3, verbose=0, callbacks=[
            nc.callbacks.LearningRateScheduler(lambda epoch, lr: 0.1 / (epoch + 1)),
            nc.callbacks.LambdaCallback(on_epoch_end=lambda e, logs: seen.append(model.optimizer.lr))])
        for got, want in zip(seen, [0.1, 0.05, 0.1 / 3]):
            self.assertAlmostEqual(got, want, places=6)

        model.optimizer.lr = 0.1
        plateau = nc.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=1, min_delta=100.0)
        model.fit(xs, ys, epochs=3, verbose=0, callbacks=[plateau])
        # first epoch sets the best value; the next two never improve by 100 -> two halvings
        self.assertAlmostEqual(model.optimizer.lr, 0.025, places=6)

    def test_model_checkpoint(self):
        xs, ys = self.data()
        model = nc.Sequential(net.Linear(1, 1)).compile(optimizer="sgd", loss="mse", lr=0.1)
        tmp = tempfile.mkdtemp()
        pattern = os.path.join(tmp, "w-{epoch:02d}.ncw")
        model.fit(xs, ys, epochs=2, verbose=0, callbacks=[
            nc.callbacks.ModelCheckpoint(pattern, monitor="loss", save_weights_only=True)])
        self.assertTrue(os.path.exists(os.path.join(tmp, "w-02.ncw")))
        best = os.path.join(tmp, "best.ncw")
        model.fit(xs, ys, epochs=2, verbose=0, callbacks=[
            nc.callbacks.ModelCheckpoint(best, monitor="loss", save_best_only=True, save_weights_only=True)])
        other = nc.Sequential(net.Linear(1, 1))
        other.load_weights(best)
        with self.assertRaises(NotImplementedError):
            nc.callbacks.ModelCheckpoint(best)  # Keras default save_weights_only=False: whole-model save


class TestApiCompat(Base):
    """Argument order and defaults follow the original libraries; unsupported values raise."""

    def test_fit_positional_order_is_keras(self):
        xs = [[float(i % 3)] for i in range(12)]
        ys = [2 * x[0] for x in xs]
        model = nc.Sequential(net.Linear(1, 1)).compile(optimizer="sgd", loss="mse")
        hist = model.fit(xs, ys, 4, 3, 0)  # batch_size, epochs, verbose
        self.assertEqual(len(hist.history["loss"]), 3)
        hist = model.fit(xs, ys, epochs=5, initial_epoch=3, verbose=0)
        self.assertEqual(hist.epoch, [3, 4])
        with self.assertRaises(NotImplementedError):
            model.fit(xs, ys, verbose=0, sample_weight=[1.0] * 12)

    def test_compile_defaults(self):
        model = nc.Sequential(net.Linear(1, 1)).compile()
        self.assertEqual(type(model.optimizer).__name__, "_KerasRMSprop")
        self.assertAlmostEqual(model.optimizer.lr, 1e-3)
        with self.assertRaises(ValueError):  # no loss configured
            model.fit([[0.0]], [0.0], verbose=0)
        with self.assertRaises(NotImplementedError):
            model.compile(loss="mse", loss_weights=[1.0])

    def test_rmsprop_variants(self):
        def first_step(opt_factory):
            w = nc.parameter([1.0])
            opt = opt_factory([w])
            (w * 2.0).sum().backward()
            opt.step()
            return w.item()

        a = 0.9
        torch_form = first_step(lambda p: nc.optim.RMSprop(p, lr=0.1, alpha=a, eps=0.5))
        keras_form = first_step(lambda p: nc.optim._KerasRMSprop(p, lr=0.1, alpha=a, eps=0.5))
        v = (1 - a) * 4.0
        self.assertAlmostEqual(torch_form, 1.0 - 0.1 * 2.0 / (math.sqrt(v) + 0.5), places=5)
        self.assertAlmostEqual(keras_form, 1.0 - 0.1 * 2.0 / math.sqrt(v + 0.5), places=5)

    def test_optimizer_positional_order_is_torch(self):
        p = [nc.parameter((2,))]
        with self.assertRaises(NotImplementedError):
            nc.optim.SGD(p, 0.1, 0.9, 0.5)  # 4th positional is dampening, as in torch
        nc.optim.SGD(p, 0.1, 0.9, 0.0, 1e-4)  # weight_decay is 5th
        with self.assertRaises(NotImplementedError):
            nc.optim.Adam(p, 1e-3, (0.9, 0.999), 1e-8, 0.0, True)  # amsgrad
        with self.assertRaises(TypeError):
            nc.optim.Adam(p, 1e-3, (0.9, 0.999), 1e-8, 0.0, False, 4)  # accumulation is keyword-only

    def test_zero_grad_set_to_none(self):
        lin = net.Linear(2, 1)
        opt = nc.optim.SGD(lin.parameters(), lr=0.1, weight_decay=1.0)
        lin(nc.randn(3, 2)).sum().backward()
        opt.zero_grad()
        self.assertIsNone(lin.weight.grad)
        before = lin.weight.tolist()
        opt.step()  # skipped: no gradient, so no weight decay either
        self.assertEqual(lin.weight.tolist(), before)
        lin(nc.randn(3, 2)).sum().backward()
        lin.zero_grad(set_to_none=False)
        self.assertEqual(lin.weight.grad.tolist(), [[0.0, 0.0]])

    def test_layers_reject_unsupported(self):
        with self.assertRaises(NotImplementedError):
            net.Conv2d(3, 8, 3, 1, 1, 2)  # 6th positional is dilation, as in torch
        net.Conv2d(3, 8, 3, 1, 1, 1, 1, False)  # bias is 8th
        with self.assertRaises(NotImplementedError):
            net.MaxPool2d(2, ceil_mode=True)
        with self.assertRaises(NotImplementedError):
            net.Embedding(10, 4, padding_idx=0)
        with self.assertRaises(TypeError):
            net.Linear(2, 3, True, nc.default_context())  # ctx is keyword-only

    def test_batchnorm_cumulative_momentum(self):
        bn = net.BatchNorm1d(1, momentum=None)
        bn(nc.tensor([[0.0], [2.0]]))  # batch mean 1
        bn(nc.tensor([[4.0], [6.0]]))  # batch mean 5 -> running mean = average of batches = 3
        self.assertAlmostEqual(bn.running_mean.item(), 3.0, places=5)

    def test_functional_losses(self):
        logits = nc.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        mean = nc.cross_entropy(logits, [0, 1]).item()
        self.assertAlmostEqual(nc.cross_entropy(logits, [0, 1], reduction="sum").item(), 2 * mean, places=5)
        smooth = nc.cross_entropy(logits, [0, 1], label_smoothing=0.1).item()
        uniform = -sum(sum(r) / 3 for r in logits.log_softmax(-1).tolist()) / 2
        self.assertAlmostEqual(smooth, 0.9 * mean + 0.1 * uniform, places=5)
        with self.assertRaises(NotImplementedError):
            nc.cross_entropy(logits, [0, 1], ignore_index=0)
        with self.assertRaises(NotImplementedError):
            nc.cross_entropy(logits, [0, 1], reduction="none")
        a, b = nc.tensor([1.0, 2.0]), nc.tensor([0.0, 0.0])
        self.assertEqual(nc.mse_loss(a, b, reduction="none").tolist(), [1.0, 4.0])
        self.assertAlmostEqual(nc.mse_loss(a, b, reduction="sum").item(), 5.0)

    def test_creation_and_reductions(self):
        self.assertEqual(nc.full((2,), 3).dtype, nc.int32)  # dtype follows the value
        self.assertEqual(nc.full((2,), 3.0).tolist(), [3.0, 3.0])
        self.assertEqual(nc.ones(2, dtype=nc.int32).tolist(), [1, 1])
        t = nc.tensor([[1.0, 5.0], [7.0, 2.0]])
        self.assertEqual(t.argmax(1, keepdim=True).shape, (2, 1))
        with self.assertWarns(UserWarning):
            self.assertEqual(nc.softmax(t).shape, (2, 2))  # implicit dim 1 for 2-D
        with self.assertRaises(TypeError):
            t.softmax()  # Tensor.softmax needs dim, as in torch

    def test_activations(self):
        x = nc.tensor([-4.0, -1.0, 0.5, 4.0, 7.0])
        sig = [1 / (1 + math.exp(-v)) for v in x.tolist()]
        phi = [0.5 * (1 + math.erf(v / math.sqrt(2))) for v in x.tolist()]
        want = {
            net.LeakyReLU(0.1): [-0.4, -0.1, 0.5, 4.0, 7.0],
            net.ReLU6(): [0.0, 0.0, 0.5, 4.0, 6.0],
            net.Hardsigmoid(): [0.0, 1 / 3, 0.5 / 6 + 0.5, 1.0, 1.0],
            net.Hardswish(): [0.0, -1 / 3, 0.5 * 3.5 / 6, 4.0, 7.0],
            net.SiLU(): [v * s for v, s in zip(x.tolist(), sig)],
            net.GELU(): [v * p for v, p in zip(x.tolist(), phi)],
            net.Sigmoid(): sig,
            net.Tanh(): [math.tanh(v) for v in x.tolist()],
        }
        for layer, values in want.items():
            for got, w in zip(layer(x).tolist(), values):
                self.assertAlmostEqual(got, w, places=5, msg=repr(layer))
        approx = nc.gelu(x, approximate="tanh").tolist()[1]
        self.assertAlmostEqual(approx, -0.158808, places=5)
        with self.assertRaises(ValueError):
            net.GELU("fast")
        with self.assertRaises(NotImplementedError):
            nc.silu(x, inplace=True)
        p = nc.parameter([0.5, -2.0])
        nc.silu(p).sum().backward()  # d/dx x sigmoid(x) = s (1 + x (1 - s))
        s = 1 / (1 + math.exp(-0.5))
        self.assertAlmostEqual(p.grad.tolist()[0], s * (1 + 0.5 * (1 - s)), places=5)

    def test_dataloader_and_callbacks(self):
        ds = nc.data.TensorDataset([[1.0], [2.0], [3.0]])
        with self.assertRaises(NotImplementedError):
            nc.data.DataLoader(ds, 2, False, [0, 1])  # 4th positional is sampler
        batches = list(nc.data.DataLoader(ds, batch_size=2, collate_fn=lambda items: len(items)))
        self.assertEqual(batches, [2, 1])
        stop = nc.callbacks.EarlyStopping("loss", 0, 1, 1)  # 4th positional is verbose
        self.assertEqual((stop.verbose, stop.mode), (1, "min"))
        began = []
        cb = nc.callbacks.LambdaCallback(lambda epoch, logs: began.append(epoch))
        model = nc.Sequential(net.Linear(1, 1)).compile(optimizer="sgd", loss="mse")
        model.fit([[0.0], [1.0]], [0.0, 1.0], epochs=2, verbose=0, callbacks=[cb])
        self.assertEqual(began, [0, 1])
        with self.assertWarns(UserWarning):  # missing monitor warns, like the other callbacks
            model.fit([[0.0], [1.0]], [0.0, 1.0], epochs=1, verbose=0,
                      callbacks=[nc.callbacks.EarlyStopping(monitor="val_loss")])


@unittest.skipIf(np is None, "numpy not installed")
class TestNumpyInterop(Base):
    def test_array_protocol(self):
        t = nc.tensor([[1.0, 2.0], [3.0, 4.0]])
        a = np.asarray(t)
        self.assertEqual(a.shape, (2, 2))
        self.assertAlmostEqual(float(np.asarray(t).mean()), 2.5)
        self.assertEqual(np.asarray(t, dtype=np.float64).dtype, np.float64)


class TestML(Base):
    def test_standard_scaler(self):
        X = [[1.0, 10.0, 5.0], [3.0, 30.0, 5.0]]
        sc = ml.StandardScaler().fit(X)
        self.assertEqual(sc.mean_, [2.0, 20.0, 5.0])
        self.assertEqual(sc.scale_, [1.0, 10.0, 1.0])  # constant feature: scale 1
        self.assertEqual(sc.transform(X), [[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]])
        back = sc.inverse_transform(sc.transform(X))
        for r, w in zip(back, X):
            for a, b in zip(r, w):
                self.assertAlmostEqual(a, b, places=4)
        with self.assertRaises(ValueError):
            sc.transform([[1.0, 2.0]])
        with self.assertRaises(ml.NotFittedError):
            ml.StandardScaler().transform(X)

    def test_minmax_scaler(self):
        X = [[0.0, -1.0], [5.0, 1.0], [10.0, 3.0]]
        sc = ml.MinMaxScaler(feature_range=(-1, 1)).fit(X)
        out = sc.transform(X)
        self.assertEqual([r[0] for r in out], [-1.0, 0.0, 1.0])
        self.assertEqual(sc.data_range_, [10.0, 4.0])

    def test_train_test_split(self):
        X, y = list(range(10)), [v * 2 for v in range(10)]
        xtr, xte, ytr, yte = ml.train_test_split(X, y, test_size=0.3, random_state=0)
        self.assertEqual((len(xtr), len(xte)), (7, 3))
        self.assertEqual(sorted(xtr + xte), X)
        self.assertEqual(ytr, [v * 2 for v in xtr])
        a, b = ml.train_test_split(X, shuffle=False)
        self.assertEqual((a, b), (list(range(7)), [7, 8, 9]))  # ceil(0.25 * 10) = 3 test samples

    def test_linear_and_ridge(self):
        rng = random.Random(1)
        X = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(200)]
        y = [3 * a - 2 * b + 0.5 for a, b in X]
        lr = ml.LinearRegression().fit(X, y)
        self.assertAlmostEqual(lr.coef_[0], 3.0, places=3)
        self.assertAlmostEqual(lr.coef_[1], -2.0, places=3)
        self.assertAlmostEqual(lr.intercept_, 0.5, places=3)
        self.assertGreater(lr.score(X, y), 0.9999)
        with self.assertRaises(ValueError):  # collinear columns
            ml.LinearRegression().fit([[a, 2 * a] for a, _ in X], y)
        # ridge on centered data: w = (Xc^T Xc + alpha)^-1 Xc^T yc, for one feature a closed form
        X1 = [[1.0], [2.0], [3.0]]
        y1 = [1.0, 2.0, 3.0]
        r = ml.Ridge(alpha=2.0).fit(X1, y1)
        self.assertAlmostEqual(r.coef_[0], 2.0 / (2.0 + 2.0), places=5)
        self.assertAlmostEqual(r.intercept_, 2.0 - 0.5 * 2.0, places=5)
        multi = ml.Ridge(alpha=0.0).fit(X1, [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
        self.assertEqual(len(multi.coef_), 2)  # (n_targets, n_features)
        self.assertAlmostEqual(multi.predict([[4.0]])[0][1], -4.0, places=4)

    def test_kmeans(self):
        rng = random.Random(3)
        blobs = [(0.0, 0.0), (5.0, 5.0), (0.0, 6.0)]
        X = [[cx + rng.gauss(0, 0.3), cy + rng.gauss(0, 0.3)] for cx, cy in blobs for _ in range(50)]
        km = ml.KMeans(n_clusters=3, random_state=0).fit(X)
        centers = sorted((round(a), round(b)) for a, b in km.cluster_centers_)
        self.assertEqual(centers, [(0, 0), (0, 6), (5, 5)])
        self.assertEqual(len(set(km.labels_[:50])), 1)
        self.assertEqual(km.predict([[5.1, 4.9]])[0], km.labels_[60])
        self.assertAlmostEqual(km.score(X), -km.inertia_, places=2)
        rnd = ml.KMeans(n_clusters=3, init="random", random_state=1).fit(X)
        self.assertLess(rnd.inertia_, 100.0)

    def test_metrics(self):
        self.assertEqual(ml.accuracy_score([0, 1, 1, 0], [0, 1, 0, 0]), 0.75)
        self.assertAlmostEqual(ml.mean_squared_error([1.0, 2.0], [1.0, 4.0]), 2.0)
        self.assertAlmostEqual(ml.r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)

    @unittest.skipIf(np is None, "numpy not installed")
    def test_numpy_in_numpy_out(self):
        X = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
        sc = ml.StandardScaler().fit(X)
        self.assertIsInstance(sc.mean_, np.ndarray)
        out = sc.transform(X)
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_allclose(out.mean(axis=0), [0.0, 0.0], atol=1e-6)
        t = nc.tensor(X)
        self.assertIsInstance(sc.transform(t), nc.Tensor)


if __name__ == "__main__":
    unittest.main()
