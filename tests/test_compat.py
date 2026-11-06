"""Edge cases where newch has to match PyTorch: indexing, numpy interop, dtype inference, empty
reductions, unbatched inputs, dataset indexing, optimizer argument checks.

Each test pins a bug that once gave a different answer (or a misleading error) than torch.
"""

import math
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import newch as torch  # noqa: E402
import newch.net as nn  # noqa: E402
import newch.net.functional as F  # noqa: E402
from newch.data import DataLoader, Dataset, FileArray, TensorDataset  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


class Base(unittest.TestCase):
    def setUp(self):
        torch.set_default_context(torch.Context(seed=0, threads=1))

    def tearDown(self):
        torch.step_end()


class TestIndexing(Base):
    def setUp(self):
        super().setUp()
        self.x = torch.tensor([[0., 1.], [2., 3.], [4., 5.]], requires_grad=True)

    def test_negative_indices(self):
        self.assertEqual(self.x[[-1, 0]].tolist(), [[4., 5.], [0., 1.]])
        self.assertEqual(self.x[torch.tensor([-3, 2])].tolist(), [[0., 1.], [4., 5.]])
        with self.assertRaises(IndexError):
            self.x[[-4]]

    def test_negative_index_grad(self):
        self.x[[-1, -1, 0]].sum().backward()
        self.assertEqual(self.x.grad.tolist(), [[1., 1.], [0., 0.], [2., 2.]])

    def test_index_shape(self):
        self.assertEqual(self.x[[[0, 1], [2, -1]]].shape, (2, 2, 2))
        self.assertEqual(self.x[[]].shape, (0, 2))

    def test_bool_masks_raise(self):
        # torch treats these as masks; reading them as indices 1/0 would be silently wrong
        with self.assertRaises(IndexError):
            self.x[[True, False, True]]
        with self.assertRaises(IndexError):
            self.x[[0.5]]

    def test_iteration(self):
        self.assertEqual([r.tolist() for r in self.x], [[0., 1.], [2., 3.], [4., 5.]])
        with self.assertRaises(TypeError):
            iter(torch.tensor(1.0))

    def test_int(self):
        self.assertEqual(int(torch.tensor([7.9])), 7)
        self.assertEqual(int(torch.tensor(-3)), -3)


class TestArithmetic(Base):
    def test_divide_by_zero(self):
        a = torch.tensor([1., -1., 0.], requires_grad=True)
        out = (a / 0).tolist()
        self.assertEqual(out[:2], [math.inf, -math.inf])
        self.assertTrue(math.isnan(out[2]))
        self.assertEqual((torch.tensor([1.]) / -0.0).tolist(), [-math.inf])
        self.assertAlmostEqual((torch.tensor([1e-5]) / 1e-40).item() / 1e35, 1.0, places=4)

    def test_reduce_over_several_dims(self):
        v = [[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]
        x = torch.tensor(v, requires_grad=True)
        self.assertEqual(x.sum((0, 2)).tolist(), [30., 48.])
        self.assertEqual(x.sum((0, -1), keepdim=True).shape, (1, 2, 1))
        self.assertAlmostEqual(x.mean((0, 1, 2)).item(), 6.5, places=5)
        self.assertEqual(x.amax((1, 2)).tolist(), [6., 12.])
        self.assertEqual(x.sum(()).item(), 78.)
        x.mean((0, 1)).sum().backward()
        self.assertAlmostEqual(x.grad.tolist()[1][1][2], 0.25)
        with self.assertRaises(RuntimeError):
            x.sum((0, -3))
        with self.assertRaises(IndexError):
            x.sum((0, 3))

    def test_mean_of_nothing_is_nan(self):
        self.assertTrue(math.isnan(torch.zeros(0, 3).mean().item()))
        self.assertTrue(all(math.isnan(v) for v in torch.zeros(0, 3).mean(0).tolist()))
        no_labels = torch.zeros(0, dtype=torch.int32)
        self.assertTrue(math.isnan(F.cross_entropy(torch.zeros(0, 3), no_labels).item()))
        self.assertTrue(math.isnan(F.nll_loss(torch.zeros(0, 3), no_labels).item()))
        self.assertTrue(math.isnan(F.mse_loss(torch.zeros(0), torch.zeros(0)).item()))
        self.assertEqual(F.cross_entropy(torch.zeros(0, 3), no_labels, reduction="sum").item(), 0.0)
        self.assertEqual(F.nll_loss(torch.zeros(0, 3), no_labels, reduction="sum").item(), 0.0)

    def test_dtype_errors(self):
        for make in (lambda: torch.zeros(2, dtype="float64"), lambda: torch.ones(2, dtype="int64"),
                     lambda: torch.tensor([1], dtype="bool"), lambda: torch.tensor([1.]).to("float64"),
                     lambda: nn.Linear(1, 1).to("float16")):
            with self.assertRaises((ValueError, NotImplementedError)):
                make()
        with self.assertRaises(NotImplementedError):
            torch.tensor([1.]).to("cuda:0")


@unittest.skipIf(np is None, "numpy not installed")
class TestNumpy(Base):
    def test_numpy_on_the_left_keeps_autograd(self):
        t = torch.tensor([1., 2., 3.], requires_grad=True)
        for out in (np.ones(3, np.float32) + t, np.ones(3, np.float32) * t, np.float32(2) * t,
                    np.float64(1) - t, t * np.int64(2)):
            self.assertIsInstance(out, torch.Tensor)
            self.assertTrue(out.requires_grad)
        (np.full(3, 2, np.float32) * t).sum().backward()
        self.assertEqual(t.grad.tolist(), [2., 2., 2.])

    def test_compare_with_numpy(self):
        t = torch.tensor([1., 2., 3.])
        self.assertEqual((t > np.float32(2)).tolist(), [0, 0, 1])
        self.assertEqual((t == np.int64(2)).tolist(), [0, 1, 0])
        self.assertIsInstance(t >= np.array([1., 3., 3.], np.float32), torch.Tensor)

    def test_dtype_inference(self):
        self.assertEqual(torch.tensor([np.float32(1.5)]).tolist(), [1.5])
        self.assertEqual(torch.tensor(np.float32(2.5)).shape, ())
        self.assertEqual(torch.tensor(np.array(3)).shape, ())
        self.assertEqual(torch.tensor([np.int64(3)]).dtype, torch.int32)
        self.assertEqual(torch.tensor([np.bool_(True)]).dtype, torch.int32)
        self.assertEqual(torch.tensor([np.array([1, 2]), np.array([3, 4])]).tolist(), [[1, 2], [3, 4]])
        self.assertEqual(torch.full((2,), np.int64(3)).dtype, torch.int32)

    def test_tensors_inside_lists(self):
        self.assertEqual(torch.tensor([torch.tensor(1.5), torch.tensor(2.)]).tolist(), [1.5, 2.])


class TestLayers(Base):
    def test_embedding_any_index_shape(self):
        emb = nn.Embedding(4, 3)
        self.assertEqual(emb(torch.tensor([[1, 2], [3, 0]])).shape, (2, 2, 3))
        self.assertEqual(emb(torch.tensor(2)).shape, (3,))
        self.assertEqual(emb([[1, 2]]).shape, (1, 2, 3))
        self.assertEqual(emb(torch.tensor([[1, 2]])).tolist()[0][1], emb.weight.tolist()[2])
        emb(torch.tensor([[1, 1], [3, 1]])).sum().backward()
        self.assertEqual(emb.weight.grad.tolist(), [[0.] * 3, [3.] * 3, [0.] * 3, [1.] * 3])

    def test_unbatched_conv_and_pool(self):
        conv = nn.Conv2d(2, 3, 3, padding=1)
        x = torch.randn(2, 5, 5)
        batched = conv(x.unsqueeze(0))
        single = conv(x)
        self.assertEqual(single.shape, (3, 5, 5))
        self.assertEqual(single.tolist(), batched.tolist()[0])
        self.assertEqual(F.max_pool2d(torch.ones(2, 4, 4), 2).shape, (2, 2, 2))
        self.assertEqual(nn.AvgPool2d(2)(torch.ones(2, 4, 4)).shape, (2, 2, 2))
        w = torch.parameter(torch.ones(1, 2, 3, 3))
        F.conv2d(x, w).sum().backward()
        self.assertEqual(w.grad.shape, (1, 2, 3, 3))


class TestData(Base):
    def test_dataset_index_range(self):
        ds = TensorDataset([[1., 2.], [3., 4.], [5., 6.]], [0, 1, 2])
        self.assertEqual(ds[-1], ([5., 6.], 2))
        with self.assertRaises(IndexError):
            ds[3]
        self.assertEqual(len(list(ds)), 3)  # iteration used to wrap around forever

    def test_file_array_index_range(self):
        with tempfile.TemporaryDirectory() as d:
            fa = FileArray.create(os.path.join(d, "x.bin"), [[1, 2], [3, 4]], (2,), torch.int32)
            self.assertEqual(fa[-2], [1, 2])
            with self.assertRaises(IndexError):
                fa[2]
            fa.close()

    def test_collate_tensor_samples(self):
        class Pairs(Dataset):
            def __len__(self):
                return 5

            def __getitem__(self, i):
                return torch.tensor([float(i), 1.]), i

        batches = list(DataLoader(Pairs(), batch_size=2))
        self.assertEqual(len(batches), 3)
        xb, yb = batches[0]
        self.assertEqual(xb.tolist(), [[0., 1.], [1., 1.]])
        self.assertEqual(yb.tolist(), [0, 1])


class TestOptimizer(Base):
    def param(self):
        return torch.parameter(torch.tensor([1.]))

    def test_argument_checks(self):
        bad = [lambda p: torch.optim.SGD(p, lr=-0.1), lambda p: torch.optim.SGD(p, lr=0.1, momentum=-1),
               lambda p: torch.optim.Adam(p, betas=(1.0, 0.999)), lambda p: torch.optim.Adam(p, betas=(0.9, -0.1)),
               lambda p: torch.optim.AdamW(p, eps=-1e-8), lambda p: torch.optim.RMSprop(p, alpha=-0.5),
               lambda p: torch.optim.Adam(p, weight_decay=-1), lambda p: torch.optim.SGD(p, lr=float("nan"))]
        for make in bad:
            with self.assertRaises(ValueError):
                make([self.param()])

    def test_load_checks_hyperparameters(self):
        opt = torch.optim.SGD([self.param()], lr=0.1)
        sd = opt.state_dict()
        sd["param_groups"][0]["lr"] = -1.0
        with self.assertRaises(ValueError):
            opt.load_state_dict(sd)
        self.assertEqual(opt.lr, 0.1)

    def test_duplicate_parameters_warn(self):
        p = self.param()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.optim.SGD([p, p], lr=0.1)
        self.assertTrue(any("duplicate" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
