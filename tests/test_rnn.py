"""Recurrent layers against numpy re-implementations of torch's formulas (torch is not a test dependency)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import newch as torch  # noqa: E402
import newch.net as nn  # noqa: E402
import newch.net.functional as F  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def np_cell(mode, x, h, c, w_ih, w_hh, b_ih, b_hh):
    gi = x @ w_ih.T + (b_ih if b_ih is not None else 0)
    gh = h @ w_hh.T + (b_hh if b_hh is not None else 0)
    H = h.shape[-1]
    if mode == "LSTM":
        s = gi + gh
        i, f, g, o = s[:, :H], s[:, H:2 * H], s[:, 2 * H:3 * H], s[:, 3 * H:]
        c = sig(f) * c + sig(i) * np.tanh(g)
        return sig(o) * np.tanh(c), c
    if mode == "GRU":
        r = sig(gi[:, :H] + gh[:, :H])
        z = sig(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        return (1 - z) * n + z * h, None
    pre = gi + gh
    return (np.tanh(pre) if mode == "RNN_TANH" else np.maximum(pre, 0)), None


def np_rnn(mode, sd, x, layers, bidir, h0=None, c0=None):
    """x [L, N, I] -> out [L, N, D*H], h_n, c_n (torch.nn.RNN/LSTM/GRU, no dropout)."""
    D = 2 if bidir else 1
    L, N = x.shape[:2]
    H = sd["weight_hh_l0"].shape[1]
    hn, cn = [], []
    for layer in range(layers):
        outs = []
        for d in range(D):
            sfx = f"_l{layer}" + ("_reverse" if d else "")
            k = layer * D + d
            h = h0[k] if h0 is not None else np.zeros((N, H))
            c = c0[k] if c0 is not None else np.zeros((N, H))
            ys = [None] * L
            for t in (range(L - 1, -1, -1) if d else range(L)):
                h, c = np_cell(mode, x[t], h, c, sd["weight_ih" + sfx], sd["weight_hh" + sfx],
                               sd.get("bias_ih" + sfx), sd.get("bias_hh" + sfx))
                ys[t] = h
            outs.append(np.stack(ys))
            hn.append(h)
            cn.append(c)
        x = np.concatenate(outs, -1)
    return x, np.stack(hn), (np.stack(cn) if mode == "LSTM" else None)


@unittest.skipIf(np is None, "numpy not installed")
class TestRecurrent(unittest.TestCase):
    def setUp(self):
        torch.set_default_context(torch.Context(seed=0, threads=1))

    def tearDown(self):
        torch.step_end()

    CASES = [  # (class, mode, kwargs)
        (nn.RNN, "RNN_TANH", {}),
        (nn.RNN, "RNN_RELU", {"nonlinearity": "relu", "num_layers": 2}),
        (nn.LSTM, "LSTM", {}),
        (nn.LSTM, "LSTM", {"num_layers": 2, "bidirectional": True}),
        (nn.GRU, "GRU", {"bidirectional": True}),
        (nn.GRU, "GRU", {"num_layers": 2, "bias": False, "batch_first": True}),
    ]

    def test_against_numpy(self):
        rs = np.random.RandomState(0)
        for cls, mode, kw in self.CASES:
            with self.subTest(cls=cls.__name__, **kw):
                m = cls(3, 4, **kw)
                sd = {k: v.numpy().astype(np.float64) for k, v in m.state_dict().items()}
                layers, D = kw.get("num_layers", 1), 2 if kw.get("bidirectional") else 1
                x = rs.randn(5, 2, 3)
                h0 = rs.randn(layers * D, 2, 4)
                c0 = rs.randn(layers * D, 2, 4)
                want, wh, wc = np_rnn(mode, sd, x, layers, D == 2, h0, c0 if mode == "LSTM" else None)
                xin = x.transpose(1, 0, 2) if kw.get("batch_first") else x
                h0t, c0t = torch.tensor(h0.astype(np.float32)), torch.tensor(c0.astype(np.float32))
                out, hn = m(torch.tensor(xin.astype(np.float32)), (h0t, c0t) if mode == "LSTM" else h0t)
                if mode == "LSTM":
                    hn, cn = hn
                    np.testing.assert_allclose(cn.numpy(), wc, rtol=1e-4, atol=1e-5)
                got = out.numpy().transpose(1, 0, 2) if kw.get("batch_first") else out.numpy()
                np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
                np.testing.assert_allclose(hn.numpy(), wh, rtol=1e-4, atol=1e-5)
                torch.step_end()

    def test_gradients(self):
        """Directional derivatives of the numpy reference vs newch's grads, for every parameter and the input."""
        rs = np.random.RandomState(1)
        for cls, mode, kw in self.CASES:
            with self.subTest(cls=cls.__name__, **kw):
                m = cls(3, 4, **{k: v for k, v in kw.items() if k != "batch_first"})
                sd = {k: v.numpy().astype(np.float64) for k, v in m.state_dict().items()}
                layers, D = kw.get("num_layers", 1), 2 if kw.get("bidirectional") else 1
                x = rs.randn(4, 2, 3)
                wo = rs.randn(4, 2, D * 4)
                wh = rs.randn(layers * D, 2, 4)

                def loss(sd_, x_):
                    out, hn, _ = np_rnn(mode, sd_, x_, layers, D == 2)
                    return (out * wo).sum() + (hn * wh).sum()
                xt = torch.tensor(x.astype(np.float32), requires_grad=True)
                out, hn = m(xt)
                hn = hn[0] if mode == "LSTM" else hn
                ((out * torch.tensor(wo.astype(np.float32))).sum()
                 + (hn * torch.tensor(wh.astype(np.float32))).sum()).backward()
                eps = 1e-5
                for name, p in list(m.state_dict().items()) + [("input", xt)]:
                    v = x if name == "input" else sd[name]
                    dv = rs.randn(*v.shape)
                    if name == "input":
                        num = (loss(sd, x + eps * dv) - loss(sd, x - eps * dv)) / (2 * eps)
                    else:
                        num = (loss({**sd, name: v + eps * dv}, x) - loss({**sd, name: v - eps * dv}, x)) / (2 * eps)
                    got = float((p.grad.numpy() * dv).sum())
                    self.assertAlmostEqual(got, num, delta=2e-3 * (1 + abs(num)), msg=name)
                torch.step_end()

    def test_shapes_and_api(self):
        lstm = nn.LSTM(3, 5, num_layers=2, bidirectional=True, batch_first=True)
        self.assertEqual(list(lstm.state_dict())[:5],
                         ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0", "weight_ih_l0_reverse"])
        self.assertEqual(lstm.weight_ih_l1.shape, (20, 10))
        out, (h, c) = lstm(torch.randn(2, 7, 3))
        self.assertEqual((out.shape, h.shape, c.shape), ((2, 7, 10), (4, 2, 5), (4, 2, 5)))
        out, h = nn.GRU(3, 4)(torch.randn(6, 3))  # unbatched
        self.assertEqual((out.shape, h.shape), ((6, 4), (1, 4)))
        self.assertEqual(nn.LSTMCell(3, 4)(torch.randn(3))[1].shape, (4,))
        self.assertEqual(nn.GRUCell(3, 4)(torch.randn(2, 3)).shape, (2, 4))
        self.assertEqual(nn.RNNCell(3, 4, nonlinearity="relu")(torch.randn(2, 3), torch.zeros(2, 4)).shape, (2, 4))
        self.assertEqual(repr(lstm), "LSTM(3, 5, num_layers=2, batch_first=True, bidirectional=True)")
        for bad in (lambda: nn.LSTM(3, 4, proj_size=2), lambda: nn.RNN(3, 4, nonlinearity="gelu"),
                    lambda: nn.GRU(3, 4, dropout=1.5)):
            with self.assertRaises((ValueError, NotImplementedError)):
                bad()
        with self.assertRaises(ValueError):
            nn.LSTM(3, 4)(torch.randn(5, 2, 2))
        with self.assertRaises(RuntimeError):
            nn.GRU(3, 4)(torch.randn(5, 2, 3), torch.zeros(1, 3, 4))

    def test_cells_match_layers(self):
        """A one-layer RNN is its cell applied step by step (same weights)."""
        for layer_cls, cell_cls in ((nn.LSTM, nn.LSTMCell), (nn.GRU, nn.GRUCell), (nn.RNN, nn.RNNCell)):
            layer, cell = layer_cls(3, 4), cell_cls(3, 4)
            cell.load_state_dict({k[:-3]: v for k, v in layer.state_dict().items()})
            x = torch.randn(5, 2, 3)
            out, _ = layer(x)
            hx = None
            for t in range(5):
                hx = cell(x[t], hx)
            h = hx[0] if layer_cls is nn.LSTM else hx
            np.testing.assert_allclose(h.numpy(), out[4].numpy(), rtol=1e-5, atol=1e-6)

    def test_dropout_between_layers(self):
        m = nn.LSTM(3, 8, num_layers=2, dropout=0.5)
        x = torch.randn(4, 2, 3)
        a = m(x)[0].numpy()
        m.eval()
        b, c = m(x)[0].numpy(), m(x)[0].numpy()
        np.testing.assert_array_equal(b, c)
        self.assertFalse(np.array_equal(a, b))

    def test_lstm_remembers(self):
        """Classify the sign of the first value of a noisy 12-step sequence: needs memory."""
        torch.manual_seed(0)
        rs = np.random.RandomState(0)

        class Net(nn.Module):
            def __init__(self):
                self.lstm = nn.LSTM(1, 16, batch_first=True)
                self.head = nn.Linear(16, 2)

            def forward(self, x):
                out, (h, c) = self.lstm(x)
                return self.head(h[-1])

        def batch(n):
            x = rs.randn(n, 12, 1).astype(np.float32) * 0.5
            y = rs.randint(0, 2, n)
            x[:, 0, 0] = np.where(y == 1, 1.0, -1.0)
            return torch.tensor(x), torch.tensor(y)
        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=0.02)
        for _ in range(150):
            x, y = batch(32)
            opt.zero_grad()
            F.cross_entropy(net(x), y).backward()
            opt.step()
            torch.step_end()
        x, y = batch(500)
        acc = (net(x).argmax(1) == y).float().mean().item()
        self.assertGreater(acc, 0.95)


if __name__ == "__main__":
    unittest.main()
