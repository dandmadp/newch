"""Typical PyTorch model code runs with only the imports changed:

    import newch as torch; import newch.net as nn; import newch.net.functional as F
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import newch as torch  # noqa: E402
import newch.net as nn  # noqa: E402
import newch.net.functional as F  # noqa: E402
from test_features import bars  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)

    def tearDown(self):
        torch.step_end()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(16 * 2 * 2, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.25, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


class TestTrainingLoop(Base):
    def test_pytorch_example_structure(self):
        xs, ys = bars(320, seed=1)
        loader = torch.data.DataLoader(torch.data.TensorDataset(xs, ys), batch_size=32, shuffle=True)
        model = Net().to("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        for _epoch in range(4):
            model.train()
            for data, target in loader:
                data, target = data.to("cpu"), target.to("cpu")
                optimizer.zero_grad()
                loss = F.nll_loss(model(data), target)
                loss.backward()
                optimizer.step()
                torch.step_end()
            scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
                torch.step_end()
        self.assertIsInstance(correct, int)
        self.assertGreater(correct / total, 0.9)


class TestTensorMethods(Base):
    def test_shapes(self):
        x = torch.randn(2, 3, 4)
        self.assertEqual(x.view(x.size(0), -1).shape, (2, 12))
        self.assertEqual(x.unsqueeze(0).shape, (1, 2, 3, 4))
        self.assertEqual(x.unsqueeze(-1).shape, (2, 3, 4, 1))
        self.assertEqual(torch.zeros(1, 3, 1).squeeze().shape, (3,))
        self.assertEqual(torch.zeros(1, 3, 1).squeeze(0).shape, (3, 1))
        p = x.permute(2, 0, 1)
        self.assertEqual(p.shape, (4, 2, 3))
        self.assertEqual(p.tolist()[3][1][2], x.tolist()[1][2][3])
        self.assertEqual(x.dim(), 3)

    def test_cat_stack(self):
        a, b = torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])
        self.assertEqual(torch.cat([a, b]).tolist(), [[1, 2], [3, 4]])
        self.assertEqual(torch.cat([a, b], dim=1).tolist(), [[1, 2, 3, 4]])
        self.assertEqual(torch.stack([a[0], b[0]], dim=1).tolist(), [[1, 3], [2, 4]])
        many = torch.cat([torch.tensor([float(i)]) for i in range(9)])  # more than one engine call
        self.assertEqual(many.tolist(), [float(i) for i in range(9)])
        w = torch.parameter([1.0, 2.0])
        (torch.cat([w, w * 3.0]) * torch.tensor([1.0, 1.0, 2.0, 2.0])).sum().backward()
        self.assertEqual(w.grad.tolist(), [7.0, 7.0])  # 1 + 3 * 2

    def test_comparisons(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual((t == 2).tolist(), [0, 1, 0])
        self.assertEqual((t > 1.5).tolist(), [0, 1, 1])
        self.assertEqual((2 <= t).tolist(), [0, 1, 1])
        self.assertEqual(t.ne(torch.tensor([1, 0, 3])).tolist(), [0, 1, 0])  # float vs int32
        self.assertEqual((t >= 2).sum().item(), 2)
        self.assertFalse(t == None)  # noqa: E711  (identity comparison, not elementwise)
        self.assertEqual({t: 1}[t], 1)  # still hashable
        with self.assertRaises(TypeError):
            t.eq("x")

    def test_dtype_conversion(self):
        t = torch.tensor([-1.7, 2.9])
        self.assertEqual(t.int().tolist(), [-1, 2])
        self.assertEqual(t.long().dtype, torch.int32)  # no int64 in newch
        self.assertEqual(torch.tensor([1, 2]).float().tolist(), [1.0, 2.0])
        self.assertEqual(t.to(torch.int32).dtype, torch.int32)
        self.assertIs(t.to("cpu"), t)
        with self.assertRaises(NotImplementedError):
            t.to("cuda")
        self.assertEqual(torch.tensor([3e9, -3e9, float("nan")]).int().tolist(), [2 ** 31 - 1, -2 ** 31, 0])

    def test_truth_value(self):
        a, b = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
        with self.assertRaises(RuntimeError):
            bool(a)
        with self.assertRaises(RuntimeError):
            b in [a, b]  # elementwise ==, as in PyTorch
        self.assertTrue(torch.tensor([2.0]) > 1)

    def test_view_rules(self):
        x = torch.randn(3, 4)
        with self.assertRaises(RuntimeError):
            x.T.view(12)
        self.assertEqual(x.T.reshape(12).shape, (12,))


class TestModules(Base):
    def test_containers(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
                self.heads = nn.ModuleDict({"a": nn.Linear(4, 1), "b": nn.Linear(4, 2)})
                self.skip = nn.Identity()
                self.register_buffer("scale", torch.ones(4))

            def forward(self, x, head="a"):
                for layer in self.layers:
                    x = F.relu(layer(x)) + self.skip(x)
                return self.heads[head](x * self.scale)

        m = Block()
        self.assertEqual(m(torch.randn(3, 4), "b").shape, (3, 2))
        names = [n for n, _ in m.named_parameters()]
        self.assertEqual(names[:2], ["layers.0.weight", "layers.0.bias"])
        self.assertIn("heads.b.weight", names)
        self.assertIn("scale", m.state_dict())
        torch.step_end()
        self.assertEqual(m.scale.tolist(), [1.0] * 4)  # registered buffers survive step_end
        self.assertEqual([n for n, _ in m.named_modules()][:3], ["", "layers", "layers.0"])
        self.assertEqual(len(m.layers), 2)
        torch.quantize_(m)  # Linear inside ModuleList and ModuleDict
        self.assertIsInstance(m.layers[0], nn.QuantizedLinear)
        self.assertIsInstance(m.heads["b"], nn.QuantizedLinear)
        self.assertEqual(len(nn.Sequential(nn.ReLU(), nn.Tanh(), nn.ReLU())[1:]), 2)

    def test_losses(self):
        out, target = torch.randn(4, 3), torch.tensor([0, 2, 1, 2])
        self.assertAlmostEqual(nn.CrossEntropyLoss()(out, target).item(), F.cross_entropy(out, target).item(),
                               places=6)
        self.assertAlmostEqual(nn.NLLLoss()(F.log_softmax(out, dim=1), target).item(),
                               F.cross_entropy(out, target).item(), places=5)
        self.assertAlmostEqual(nn.MSELoss(reduction="sum")(out, out * 0.0).item(), (out * out).sum().item(),
                               places=4)
        with self.assertRaises(NotImplementedError):
            nn.NLLLoss(ignore_index=0)

    def test_init_and_apply(self):
        model = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

        model.apply(init)
        w = model[0].weight.tolist()
        bound = math.sqrt(3.0) * math.sqrt(2.0) / math.sqrt(256)
        flat = [v for row in w for v in row]
        self.assertLessEqual(max(abs(v) for v in flat), bound)
        var = sum(v * v for v in flat) / len(flat)
        self.assertAlmostEqual(var, bound * bound / 3, delta=0.1 * bound * bound / 3)  # uniform variance
        self.assertEqual(model[0].bias.tolist(), [0.0] * 128)
        self.assertAlmostEqual(nn.init.calculate_gain("tanh"), 5 / 3)
        nn.init.constant_(model[2].bias, 0.5)
        self.assertEqual(model[2].bias.tolist(), [0.5] * 10)
        with self.assertRaises(ValueError):
            nn.init.xavier_uniform_(model[2].bias)  # needs >= 2 dims

    def test_device_moves(self):
        m = Net()
        self.assertIs(m.to("cpu"), m)
        self.assertIs(m.cpu(), m)
        with self.assertRaises(NotImplementedError):
            m.to("cuda:0")
        with self.assertRaises(NotImplementedError):
            m.to(torch.int32)



class TestBackwardMemory(Base):
    def make(self):
        torch.manual_seed(5)
        return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(8 * 16 * 16, 4))

    def step(self, model, retain):
        ctx = torch.default_context()
        ctx.step_end()
        ctx.reset_peak()
        torch.manual_seed(9)
        x, y = torch.randn(32, 1, 16, 16), torch.tensor([i % 4 for i in range(32)])
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward(retain_graph=retain)
        grads = [p.grad.tolist() for p in model.parameters()]
        return out, loss, grads, torch.mem_stats()["arena_peak_bytes"]

    def test_release_keeps_results_and_lowers_peak(self):
        model = self.make()
        _, _, g_keep, peak_keep = self.step(model, True)
        model.zero_grad()
        out, loss, g_free, peak_free = self.step(model, None)
        self.assertEqual(g_free, g_keep)  # bit-identical gradients
        self.assertLess(peak_free, peak_keep)
        self.assertEqual(out.shape, (32, 4))  # tensors still referenced survive backward
        self.assertTrue(math.isfinite(out.sum().item()) and math.isfinite(loss.item()))

    def test_second_backward_needs_retain_graph(self):
        model = self.make()
        x = torch.randn(4, 1, 16, 16)
        hidden = model[1](model[0](x))
        loss = model[3](model[2](hidden)).sum()
        del hidden  # nothing holds the intermediate activations any more
        loss.backward()
        with self.assertRaises(torch.NcError):
            loss.backward()
        torch.step_end()
        loss = model(torch.randn(4, 1, 16, 16)).sum()
        loss.backward(retain_graph=True)
        loss.backward()  # allowed: the first call kept the graph
        torch.step_end()

    def test_intermediate_grad_is_none(self):
        w = torch.parameter([1.0, 2.0])
        h = w * 3.0
        h.sum().backward()
        self.assertIsNone(h.grad)  # non-leaf, as in PyTorch
        self.assertEqual(w.grad.tolist(), [3.0, 3.0])



class TestInferenceMemory(Base):
    def model(self):
        torch.manual_seed(2)
        return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                             nn.Flatten(), nn.Linear(16 * 28 * 28, 10)).eval()

    def test_no_grad_forwards_do_not_pile_up(self):
        model, ctx = self.model(), torch.default_context()
        ctx.step_end()
        x = torch.randn(64, 1, 28, 28)
        with torch.no_grad():
            first = model(x).tolist()
            ctx.reset_peak()
            for _ in range(5):
                out = model(x)
            peak = torch.mem_stats()["arena_peak_bytes"]
            self.assertEqual(out.tolist(), first)  # same values with memory reuse
        one_activation = 64 * 16 * 28 * 28 * 4
        self.assertLess(peak, 6 * one_activation)  # five forwards would hold ~20 activations otherwise

    def test_held_tensors_survive(self):
        model = self.model()
        with torch.no_grad():
            x = torch.randn(4, 1, 28, 28)
            h = model[1](model[0](x))  # kept by the caller
            flat = h.view(4, -1)       # a view keeps its base alive
            del h
            out = model[5](model[4](model[3](model[2](flat.view(4, 16, 28, 28)))))
        self.assertEqual(out.shape, (4, 10))
        self.assertTrue(math.isfinite(flat.sum().item()))

    def test_no_grad_tensor_later_used_for_training(self):
        lin = nn.Linear(8, 2)
        with torch.no_grad():
            feats = torch.randn(4, 8) * 2.0   # made under no_grad ...
        loss = F.cross_entropy(lin(feats), torch.tensor([0, 1, 0, 1]))
        del feats                              # ... dropped, but the tape still needs it
        loss.backward()
        self.assertTrue(all(math.isfinite(v) for row in lin.weight.grad.tolist() for v in row))


class TestApiSurface(Base):
    def test_torch_signatures(self):
        x, w, b = torch.randn(2, 3), torch.randn(4, 3), torch.zeros(4)
        self.assertEqual(F.linear(input=x, weight=w, bias=b).shape, (2, 4))
        rm, rv = torch.zeros(3), torch.ones(3)
        self.assertEqual(F.batch_norm(input=x, running_mean=rm, running_var=rv).shape, (2, 3))
        r = torch.rand(2, 3, requires_grad=True)
        self.assertTrue(r.requires_grad)
        self.assertTrue(all(0.0 <= v < 1.0 for row in r.tolist() for v in row))
        self.assertEqual(torch.randn((2, 5)).shape, (2, 5))
        with self.assertRaises(NotImplementedError):
            torch.rand(2, dtype=torch.int32)
        with self.assertRaises(TypeError):  # torch.rand has no low/high, randn no std
            torch.rand(2, low=-1.0)
        with self.assertRaises(TypeError):
            torch.randn(2, std=2.0)
        self.assertFalse(hasattr(r, "zero_grad"))  # torch tensors have none

    def test_no_stray_public_names(self):
        import newch.callbacks
        import newch.data
        import newch.optim

        for mod in (nn, F, torch.optim, newch.data, newch.callbacks, nn.init):
            for name in mod.__all__:
                self.assertTrue(hasattr(mod, name), f"{mod.__name__}.{name}")
        self.assertNotIn("annotations", dir(F))
        self.assertFalse(hasattr(torch.optim, "get"))
        self.assertNotIn("Context", nn.__all__)


if __name__ == "__main__":
    unittest.main()
