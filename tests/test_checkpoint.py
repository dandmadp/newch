"""Optimizer checkpoints (state_dict / load_state_dict, nested save_tensors) and the uint8 dtype."""

import array
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import newch as nc  # noqa: E402
from newch import net  # noqa: E402
from newch.data import DataLoader, FileArray, TensorDataset  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

OPTIMIZERS = {
    "adam": lambda p: nc.optim.Adam(p, lr=0.01, weight_decay=0.01),
    "adamw": lambda p: nc.optim.AdamW(p, lr=0.01),
    "sgd": lambda p: nc.optim.SGD(p, lr=0.05, momentum=0.9),
    "rmsprop": lambda p: nc.optim.RMSprop(p, lr=0.01, momentum=0.5),
}


def make_model(seed):
    nc.manual_seed(seed)
    return nc.Sequential(net.Linear(4, 8), net.ReLU(), net.Linear(8, 3))


def train_steps(model, opt, n, offset=0):
    for s in range(n):
        x = nc.tensor([[(i * 7 + j * 3 + s + offset) % 5 / 5.0 for j in range(4)] for i in range(6)])
        y = nc.tensor([(i + s + offset) % 3 for i in range(6)])
        opt.zero_grad()
        loss = net.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        nc.step_end()


def flat_params(model):
    return [v for p in model.parameters() for v in p._flat()]


class TestOptimizerCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "ckpt.ncw")

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_matches_uninterrupted_training(self):
        for name, make_opt in OPTIMIZERS.items():
            with self.subTest(name):
                ref = make_model(1)
                ref_opt = make_opt(ref.parameters())
                train_steps(ref, ref_opt, 3)
                nc.save_tensors({"model": ref.state_dict(), "optimizer": ref_opt.state_dict(), "epoch": 3}, self.path)
                train_steps(ref, ref_opt, 3, offset=3)

                model = make_model(2)  # different start: everything must come from the file
                opt = make_opt(model.parameters())
                ckpt = nc.load_tensors(self.path)
                self.assertEqual(ckpt["epoch"], 3)
                model.load_state_dict(ckpt["model"])
                opt.load_state_dict(ckpt["optimizer"])
                train_steps(model, opt, 3, offset=3)
                self.assertEqual(flat_params(model), flat_params(ref))  # bit-identical

    def test_state_dict_layout_is_torch_like(self):
        model = make_model(0)
        opt = nc.optim.Adam(model.parameters(), lr=0.01)
        train_steps(model, opt, 2)
        sd = opt.state_dict()
        self.assertEqual(sorted(sd["state"][0]), ["exp_avg", "exp_avg_sq", "step"])
        self.assertEqual(sd["state"][0]["step"], 2.0)
        self.assertEqual(sd["state"][0]["exp_avg"].shape, model[0].weight.shape)
        self.assertTrue(sd["state"][0]["exp_avg"].is_persistent)  # survives step_end
        g = sd["param_groups"][0]
        self.assertEqual((g["lr"], g["betas"], g["amsgrad"]), (0.01, (0.9, 0.999), False))
        self.assertEqual(g["params"], [0, 1, 2, 3])
        self.assertEqual(nc.optim.SGD(model.parameters()).state_dict()["state"], {})  # no momentum: no state
        before = sd["state"][0]["exp_avg"].tolist()
        train_steps(model, opt, 1)
        self.assertEqual(sd["state"][0]["exp_avg"].tolist(), before)  # a copy, not the live buffer

    def test_load_restores_hyperparameters(self):
        model = make_model(0)
        src = nc.optim.SGD(model.parameters(), lr=0.5, momentum=0.8, weight_decay=0.1)
        dst = nc.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        dst.load_state_dict(src.state_dict())
        self.assertEqual(dst.lr, 0.5)
        self.assertEqual(dst.state_dict()["param_groups"][0]["weight_decay"], 0.1)

    def test_load_rejects_mismatches_without_changing_anything(self):
        model = make_model(0)
        opt = nc.optim.Adam(model.parameters(), lr=0.01)
        train_steps(model, opt, 1)
        good = opt.state_dict()
        before = opt.state_dict()["state"][0]["exp_avg"].tolist()

        bad = {"state": dict(good["state"]), "param_groups": good["param_groups"]}
        bad["state"][3] = dict(good["state"][3], exp_avg=nc.zeros(5))  # wrong size, last param
        bad["state"][0] = dict(good["state"][0], exp_avg=nc.ones(*model[0].weight.shape))
        with self.assertRaises(ValueError):
            opt.load_state_dict(bad)
        self.assertEqual(opt.state_dict()["state"][0]["exp_avg"].tolist(), before)

        with self.assertRaises(ValueError):
            opt.load_state_dict({"state": {}, "param_groups": [dict(good["param_groups"][0], params=[0, 1])]})
        steps = {"state": {i: dict(e, step=float(i)) for i, e in good["state"].items()},
                 "param_groups": good["param_groups"]}
        with self.assertRaises(NotImplementedError):
            opt.load_state_dict(steps)
        with self.assertRaises(NotImplementedError):
            opt.load_state_dict({"state": {}, "param_groups": [dict(good["param_groups"][0], amsgrad=True)]})
        sgd = nc.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaises(ValueError):  # momentum buffer would have to appear
            sgd.load_state_dict(nc.optim.SGD(model.parameters(), lr=0.1, momentum=0.9).state_dict())

    @unittest.skipIf(np is None, "numpy not installed")
    def test_numpy_state_loads(self):
        model = make_model(0)
        opt = nc.optim.Adam(model.parameters())
        train_steps(model, opt, 2)
        sd = opt.state_dict()
        as_np = {"state": {i: {k: (v.numpy() if isinstance(v, nc.Tensor) else np.float32(v)) for k, v in e.items()}
                           for i, e in sd["state"].items()},
                 "param_groups": sd["param_groups"]}
        other = nc.optim.Adam(model.parameters())
        other.load_state_dict(as_np)
        self.assertEqual(other.state_dict()["state"][2]["exp_avg_sq"].tolist(), sd["state"][2]["exp_avg_sq"].tolist())

    def test_scheduler_state_dict(self):
        opt = nc.optim.SGD(make_model(0).parameters(), lr=1.0)
        sched = nc.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        for _ in range(3):
            sched.step()
        opt2 = nc.optim.SGD(make_model(0).parameters(), lr=1.0)
        sched2 = nc.optim.lr_scheduler.StepLR(opt2, step_size=2, gamma=0.5)
        sched2.load_state_dict(sched.state_dict())
        opt2.load_state_dict(opt.state_dict())
        sched.step()
        sched2.step()
        self.assertEqual(sched2.get_last_lr(), sched.get_last_lr())
        self.assertNotIn("optimizer", sched.state_dict())


class TestNestedSave(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "t.ncw")

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        a, b = nc.tensor([1.0, 2.0]), nc.tensor([[3, 4]])
        obj = {"x": {0: a, "a.b": (b, None, "s")}, "x.0": a, "n": [1, 2.5, True, float("inf")]}
        nc.save_tensors(obj, self.path)
        back = nc.load_tensors(self.path)
        self.assertEqual(set(back), {"x", "x.0", "n"})
        self.assertEqual(back["x"][0].tolist(), [1.0, 2.0])
        self.assertEqual(back["x.0"].tolist(), [1.0, 2.0])  # same path as x -> 0: stored under its own name
        t, none, s = back["x"]["a.b"]
        self.assertEqual((t.tolist(), t.dtype, none, s), ([[3, 4]], "int32", None, "s"))
        self.assertEqual(back["n"], [1, 2.5, True, float("inf")])

    def test_flat_dict_unchanged(self):
        nc.save_tensors({"w": nc.ones(2)}, self.path)
        self.assertEqual(list(nc.load_tensors(self.path)), ["w"])
        with self.assertRaises(ValueError):
            nc.save_tensors({"__newch_tree__": nc.ones(2)}, self.path)
        with self.assertRaises(TypeError):
            nc.save_tensors({"f": object()}, self.path)
        with self.assertRaises(TypeError):
            nc.save_tensors({"k": {1.5: nc.ones(1)}}, self.path)


class TestCorruptFiles(unittest.TestCase):
    """A damaged or hostile checkpoint raises NcError; it must never read or write past a buffer
    (a data size larger than the shape once crashed the process)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def entry(name, code, shape, data, nbytes=None):
        import struct
        nb = name if isinstance(name, bytes) else name.encode()
        return (struct.pack("<I", len(nb)) + nb + struct.pack("<II", code, len(shape))
                + struct.pack(f"<{len(shape)}q", *shape) + struct.pack("<Q", len(data) if nbytes is None else nbytes)
                + data)

    def write(self, body, count):
        import struct
        path = os.path.join(self.tmp.name, "bad.ncw")
        with open(path, "wb") as f:
            f.write(b"NCW1" + struct.pack("<II", 1, count) + body)
        return path

    def test_python_loader(self):
        import struct
        e = self.entry
        cases = {
            "shape bigger than data": (e("a", 0, (1000000,), b"\x00" * 4), 1),
            "shape smaller than data": (e("a", 0, (1,), b"\x00" * 64), 1),
            "negative dim": (e("a", 0, (-1,), b""), 1),
            "huge ndim": (struct.pack("<I", 1) + b"a" + struct.pack("<II", 0, 10**9), 1),
            "huge nbytes": (e("a", 0, (1,), b"\x00" * 4, nbytes=2**62), 1),
            "huge name": (struct.pack("<I", 2**31), 1),
            "more entries announced": (e("a", 0, (1,), b"\x00" * 4), 5),
            "bad utf-8 name": (e(b"\xff", 0, (1,), b"\x00" * 4), 1),
            "too many dims": (e("a", 0, (1,) * 9, b"\x00" * 4), 1),
            "unknown dtype": (e("a", 77, (1,), b"\x00" * 4), 1),
            "name twice": (e("a", 0, (1,), b"\x00" * 4) * 2, 2),
            "bad layout json": (e("__newch_tree__", 4, (3,), b"{x}"), 1),
            "layout points nowhere": (e("__newch_tree__", 4, (11,), b'{"T": "zz"}'), 1),
        }
        for what, (body, count) in cases.items():
            with self.subTest(what), self.assertRaises(nc.NcError):
                nc.load_tensors(self.write(body, count))

    def test_engine_loader(self):
        """nc_load (C) checks the byte count against the destination tensor."""
        from newch import _lib as L
        from newch._lib import lib
        path = self.write(self.entry("w", 0, (1,), b"A" * 64), 1)
        w = nc.parameter(nc.zeros(1))
        ctx = w.ctx
        st = lib.nc_load(ctx._p, path.encode(), 1, L.name_array(["w"]), L.ptr_array([w.ptr]))
        self.assertNotEqual(st, 0)
        lib.nc_clear_error(ctx._p)
        self.assertEqual(w.tolist(), [0.0])

    def test_tensor_from_bytes_checks_length(self):
        from newch.tensor import tensor_from_bytes
        for buf, shape in ((b"\x00" * 3, (1,)), (b"\x00" * 4, (-1,)), (b"\x00" * 8, (1,)), (b"", (1,) * 7)):
            with self.assertRaises(ValueError):
                tensor_from_bytes(buf, shape, "float32")
        self.assertEqual(tensor_from_bytes(b"", (0, 3), "float32").shape, (0, 3))


class TestUint8(unittest.TestCase):
    def test_tensor_basics(self):
        t = nc.tensor(array.array("B", [0, 7, 255]))
        self.assertEqual(t.dtype, nc.uint8)
        self.assertEqual(t.tolist(), [0, 7, 255])
        self.assertEqual(t[2].item(), 255)
        self.assertIsInstance(t[1].item(), int)
        self.assertEqual((t.float() * (1 / 255)).tolist()[2], 1.0)
        self.assertEqual(nc.tensor([-5.0, 3.7, 300.0]).to(nc.uint8).tolist(), [0, 3, 255])  # saturates
        self.assertEqual(nc.tensor([1, 2], dtype=nc.uint8).int().dtype, nc.int32)

    @unittest.skipIf(np is None, "numpy not installed")
    def test_numpy_keeps_one_byte(self):
        img = np.arange(12, dtype=np.uint8).reshape(3, 4)
        t = nc.tensor(img)
        self.assertEqual(t.dtype, "uint8")
        self.assertTrue((t.numpy() == img).all() and t.numpy().dtype == np.uint8)
        self.assertEqual(nc.tensor(np.array([-1, 2], dtype=np.int8)).dtype, "int8")
        self.assertEqual(nc.tensor(np.array([1, 2], dtype=np.int64)).dtype, "int32")
        batch = next(iter(DataLoader(TensorDataset(img, np.array([0, 1, 2], dtype=np.uint8)), batch_size=3)))
        self.assertEqual([b.dtype for b in batch], ["uint8", "uint8"])

    def test_file_array_and_training(self):
        with tempfile.TemporaryDirectory() as d:
            n = 24
            xs = FileArray.create(os.path.join(d, "x.u8"), ([(i * 37 + j * 11) % 256 for j in range(16)]
                                                           for i in range(n)), (16,), dtype=nc.uint8)
            ys = FileArray.create(os.path.join(d, "y.u8"), (i % 3 for i in range(n)), (), dtype=nc.uint8)
            self.assertEqual(os.path.getsize(os.path.join(d, "x.u8")), n * 16)  # one byte per pixel
            loader = DataLoader(TensorDataset(xs, ys), batch_size=8)
            nc.manual_seed(0)
            model = nc.Sequential(net.Linear(16, 3))
            opt = nc.optim.SGD(model.parameters(), lr=0.1)
            for xb, yb in loader:
                self.assertEqual((xb.dtype, yb.dtype), ("uint8", "uint8"))
                opt.zero_grad()
                loss = net.functional.cross_entropy(model(xb.float() / 255), yb)  # uint8 labels accepted
                loss.backward()
                opt.step()
                nc.step_end()
            xs.close()
            ys.close()

    def test_fit_with_uint8_labels(self):
        nc.manual_seed(0)
        model = nc.Sequential(net.Linear(2, 2))
        model.compile(optimizer="sgd", loss="cross_entropy", metrics=["accuracy"], lr=0.1)
        x = [[0.0, 1.0], [1.0, 0.0]] * 4
        y = nc.tensor(array.array("B", [0, 1] * 4))
        h = model.fit(DataLoader(TensorDataset(x, y), batch_size=4), epochs=1, verbose=0)
        self.assertIn("accuracy", h.history)


if __name__ == "__main__":
    unittest.main()
