"""Transformer building blocks: batched matmul, layer_norm, masked_fill, attention, encoder layers.

References are numpy re-implementations of torch's formulas (torch itself is not a test dependency).
"""

import copy
import math
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

NEG_INF = float("-inf")


def softmax(s):
    m = s.max(-1, keepdims=True)
    e = np.exp(s - m)
    return e / e.sum(-1, keepdims=True)


def np_mha(q, k, v, W, b, Wo, bo, H, attn_mask=None, kpm=None):
    """torch.nn.functional.multi_head_attention_forward for kdim == vdim == E, no bias_kv / zero_attn."""
    L, N, E = q.shape
    S, D = k.shape[0], E // H
    Q = q @ W[:E].T + b[:E]
    K = k @ W[E:2 * E].T + b[E:2 * E]
    V = v @ W[2 * E:].T + b[2 * E:]
    Q = Q.reshape(L, N * H, D).transpose(1, 0, 2) / math.sqrt(D)
    K = K.reshape(S, N * H, D).transpose(1, 0, 2)
    V = V.reshape(S, N * H, D).transpose(1, 0, 2)
    s = Q @ K.transpose(0, 2, 1)
    if attn_mask is not None:
        s = np.where(attn_mask, -np.inf, s) if attn_mask.dtype == bool else s + attn_mask
    if kpm is not None:
        s = np.where(kpm[:, None, None, :], -np.inf, s.reshape(N, H, L, S)).reshape(N * H, L, S)
    a = softmax(s)
    o = (a @ V).transpose(1, 0, 2).reshape(L, N, E) @ Wo.T + bo
    return o, a.reshape(N, H, L, S)


def np_layer_norm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    var = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(var + eps) * w + b


def mha_params(m):
    return (m.in_proj_weight.numpy(), m.in_proj_bias.numpy(), m.out_proj.weight.numpy(), m.out_proj.bias.numpy())


class Base(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def tearDown(self):
        torch.step_end()

    def assertClose(self, a, b, tol=1e-5):
        a, b = np.asarray(a), np.asarray(b)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b, atol=tol, rtol=tol), f"max diff {np.abs(a - b).max()}")


@unittest.skipIf(np is None, "numpy not installed")
class TestOps(Base):
    def test_batched_matmul_and_grads(self):
        cases = [((3, 4, 5), (3, 5, 2)), ((2, 1, 4, 5), (3, 5, 6)), ((4, 5), (2, 5, 3)), ((2, 3, 4, 5), (5, 2))]
        for sa, sb in cases:
            with self.subTest(sa=sa, sb=sb):
                a = torch.parameter(torch.randn(*sa))
                b = torch.parameter(torch.randn(*sb))
                c = a @ b
                self.assertClose(c.numpy(), a.numpy() @ b.numpy())
                (c * c).sum().backward()  # dL/dc = 2c
                g = 2 * (a.numpy() @ b.numpy())
                ga = g @ np.swapaxes(b.numpy(), -1, -2)
                gb = np.swapaxes(a.numpy(), -1, -2) @ g
                # reduce broadcast batch dims back to each operand's shape
                while ga.ndim > a.ndim:
                    ga = ga.sum(0)
                while gb.ndim > b.ndim:
                    gb = gb.sum(0)
                ga = ga.sum(tuple(i for i, d in enumerate(sa) if d == 1), keepdims=True) if 1 in sa[:-2] else ga
                gb = gb.sum(tuple(i for i, d in enumerate(sb) if d == 1), keepdims=True) if 1 in sb[:-2] else gb
                self.assertClose(a.grad.numpy(), ga, 1e-4)
                self.assertClose(b.grad.numpy(), gb, 1e-4)

    def test_matmul_with_transposed_view(self):
        q, k = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 6, 8)
        self.assertClose((q @ k.transpose(-2, -1)).numpy(), q.numpy() @ np.swapaxes(k.numpy(), -1, -2))
        with self.assertRaises(torch.NcError):
            torch.randn(2, 3, 4) @ torch.randn(3, 4, 5)  # batch 2 vs 3

    def test_layer_norm(self):
        x = torch.randn(2, 3, 6)
        w, b = torch.rand(6), torch.randn(6)
        self.assertClose(F.layer_norm(x, (6,), w, b).numpy(), np_layer_norm(x.numpy(), w.numpy(), b.numpy()))
        flat = np_layer_norm(x.numpy().reshape(2, 18), 1.0, 0.0).reshape(2, 3, 6)
        self.assertClose(F.layer_norm(x, [3, 6]).numpy(), flat)
        with self.assertRaises(RuntimeError):
            F.layer_norm(x, 5)
        ln = nn.LayerNorm(6)
        self.assertEqual(list(ln.state_dict()), ["weight", "bias"])
        self.assertEqual(ln.weight.tolist(), [1.0] * 6)
        self.assertIsNone(nn.LayerNorm(6, bias=False).bias)
        self.assertEqual(list(nn.LayerNorm(6, elementwise_affine=False).state_dict()), [])

    def test_masked_fill(self):
        x = torch.parameter(torch.randn(2, 3))
        y = x.masked_fill(x > 0, 0.0)  # mask from a comparison
        self.assertClose(y.numpy(), np.where(x.numpy() > 0, 0.0, x.numpy()))
        y.sum().backward()
        self.assertEqual(x.grad.tolist(), (x <= 0).float().tolist())  # no grad where filled
        with self.assertRaises(TypeError):
            x.masked_fill(torch.ones(2, 3), 1.0)
        self.assertEqual(torch.tensor([[1.0, 2.0]]).masked_fill([1, 0], NEG_INF).tolist(), [[NEG_INF, 2.0]])


@unittest.skipIf(np is None, "numpy not installed")
class TestAttention(Base):
    def test_sdpa_matches_reference(self):
        q, k, v = torch.randn(2, 4, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 3)
        ref = softmax(q.numpy() @ np.swapaxes(k.numpy(), -1, -2) / math.sqrt(8)) @ v.numpy()
        self.assertClose(F.scaled_dot_product_attention(q, k, v).numpy(), ref)
        keep = np.array([[1, 1, 0, 1, 0]] * 4, dtype=np.int32)  # 1 = may attend (torch's True)
        s = q.numpy() @ np.swapaxes(k.numpy(), -1, -2) / math.sqrt(8)
        ref = softmax(np.where(keep == 1, s, -np.inf)) @ v.numpy()
        self.assertClose(F.scaled_dot_product_attention(q, k, v, attn_mask=torch.tensor(keep)).numpy(), ref)
        bias = np.random.default_rng(0).normal(size=(4, 5)).astype(np.float32)
        ref = softmax(s + bias) @ v.numpy()
        self.assertClose(F.scaled_dot_product_attention(q, k, v, attn_mask=torch.tensor(bias)).numpy(), ref)
        sq = q.numpy() @ np.swapaxes(q.numpy(), -1, -2) * 0.5
        ref = softmax(np.where(np.tril(np.ones((4, 4))) == 1, sq, -np.inf)) @ q.numpy()
        self.assertClose(F.scaled_dot_product_attention(q, q, q, is_causal=True, scale=0.5).numpy(), ref)
        with self.assertRaises(ValueError):
            F.scaled_dot_product_attention(q, q, q, attn_mask=torch.tensor(keep[:, :4]), is_causal=True)

    def test_multihead_attention_matches_reference(self):
        L, S, N, E, H = 4, 6, 3, 8, 2
        m = nn.MultiheadAttention(E, H)
        W, b, Wo, bo = mha_params(m)
        self.assertEqual([k for k in m.state_dict()], ["in_proj_weight", "in_proj_bias", "out_proj.weight",
                                                       "out_proj.bias"])
        x = torch.randn(L, N, E)
        out, w = m(x, x, x)  # self-attention (packed projection)
        ro, rw = np_mha(x.numpy(), x.numpy(), x.numpy(), W, b, Wo, bo, H)
        self.assertClose(out.numpy(), ro)
        self.assertClose(w.numpy(), rw.mean(1))

        q, kv = torch.randn(L, N, E), torch.randn(S, N, E)  # cross-attention with masks
        block = np.zeros((L, S), dtype=bool)
        block[:, -1] = True
        pad = np.zeros((N, S), dtype=bool)
        pad[1, :2] = True
        out, w = m(q, kv, kv, key_padding_mask=torch.tensor(pad.astype(np.int32)),
                   attn_mask=torch.tensor(block.astype(np.int32)), average_attn_weights=False)
        ro, rw = np_mha(q.numpy(), kv.numpy(), kv.numpy(), W, b, Wo, bo, H, attn_mask=block, kpm=pad)
        self.assertClose(out.numpy(), ro)
        self.assertClose(w.numpy(), rw)
        fmask = np.random.default_rng(1).normal(size=(L, S)).astype(np.float32)
        out, _ = m(q, kv, kv, attn_mask=torch.tensor(fmask), need_weights=False)
        self.assertClose(out.numpy(), np_mha(q.numpy(), kv.numpy(), kv.numpy(), W, b, Wo, bo, H, attn_mask=fmask)[0])

    def test_batch_first_and_unbatched(self):
        m = nn.MultiheadAttention(8, 4, batch_first=True)
        W, b, Wo, bo = mha_params(m)
        x = torch.randn(2, 5, 8)  # (N, L, E)
        out, w = m(x, x, x)
        xs = x.numpy().transpose(1, 0, 2)
        ro, rw = np_mha(xs, xs, xs, W, b, Wo, bo, 4)
        self.assertClose(out.numpy(), ro.transpose(1, 0, 2))
        u = torch.randn(5, 8)  # unbatched (L, E)
        out, w = m(u, u, u)
        self.assertEqual((out.shape, w.shape), ((5, 8), (5, 5)))
        with self.assertRaises(ValueError):
            nn.MultiheadAttention(10, 3)
        with self.assertRaises(NotImplementedError):
            nn.MultiheadAttention(8, 2, kdim=4)
        with self.assertRaises(RuntimeError):
            m(x, x, x, is_causal=True)  # torch needs attn_mask with the hint

    def test_multihead_attention_gradients(self):
        m = nn.MultiheadAttention(4, 2)
        x = torch.randn(3, 2, 4)
        with torch.persistent():
            xp = torch.tensor(x)

        def loss():
            out, _ = m(xp, xp, xp, need_weights=False)
            return (out * out).sum()

        loss().backward()
        W = m.in_proj_weight
        ana = W.grad.numpy().ravel()
        flat = W.numpy().ravel()
        for j in (0, 5, 17, 30, 47):  # q, k and v rows
            for sgn, store in ((1, "p"), (-1, "m")):
                d = flat.copy()
                d[j] += sgn * 1e-2
                W.copy_(d.reshape(W.shape))
                with torch.no_grad():
                    val = loss().item()
                if store == "p":
                    fp = val
                else:
                    fm = val
            W.copy_(flat.reshape(W.shape))
            self.assertAlmostEqual((fp - fm) / 2e-2, ana[j], delta=2e-2 * max(1.0, abs(ana[j])))


@unittest.skipIf(np is None, "numpy not installed")
class TestEncoder(Base):
    def _np_layer(self, layer, x, norm_first):
        W, b, Wo, bo = mha_params(layer.self_attn)
        ln = lambda t, n: np_layer_norm(t, n.weight.numpy(), n.bias.numpy())  # noqa: E731
        ff = lambda t: np.maximum(t @ layer.linear1.weight.numpy().T + layer.linear1.bias.numpy(), 0) \
            @ layer.linear2.weight.numpy().T + layer.linear2.bias.numpy()  # noqa: E731
        sa = lambda t: np_mha(t, t, t, W, b, Wo, bo, layer.self_attn.num_heads)[0]  # noqa: E731
        if norm_first:
            x = x + sa(ln(x, layer.norm1))
            return x + ff(ln(x, layer.norm2))
        x = ln(x + sa(x), layer.norm1)
        return ln(x + ff(x), layer.norm2)

    def test_encoder_layer_matches_reference(self):
        x = torch.randn(5, 2, 8)
        for norm_first in (False, True):
            with self.subTest(norm_first=norm_first):
                layer = nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, norm_first=norm_first)
                self.assertClose(layer(x).numpy(), self._np_layer(layer, x.numpy(), norm_first), 1e-4)
        names = list(nn.TransformerEncoderLayer(8, 2, dim_feedforward=16).state_dict())
        self.assertEqual(names[:4], ["self_attn.in_proj_weight", "self_attn.in_proj_bias", "self_attn.out_proj.weight",
                                     "self_attn.out_proj.bias"])
        self.assertEqual(names[4:], ["linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias",
                                     "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"])

    def test_encoder_copies_are_independent(self):
        layer = nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0)
        enc = nn.TransformerEncoder(layer, num_layers=2)
        a, b = enc.layers[0].linear1.weight, enc.layers[1].linear1.weight
        self.assertIsNot(a, b)
        self.assertEqual(a.tolist(), b.tolist())  # copies of the same layer
        self.assertTrue(a.requires_grad and a.is_persistent)
        opt = torch.optim.SGD(enc.layers[0].parameters(), lr=0.1)
        enc(torch.randn(4, 2, 8)).sum().backward()
        opt.step()
        self.assertNotEqual(a.tolist(), b.tolist())
        self.assertEqual(b.tolist(), layer.linear1.weight.tolist())  # the template is untouched
        x = torch.randn(4, 2, 8)
        ref = x.numpy()
        for lyr in enc.layers:
            ref = self._np_layer(lyr, ref, False)
        self.assertClose(enc(x).numpy(), ref, 1e-4)

    def test_deepcopy_module(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
        clone = copy.deepcopy(model)
        self.assertEqual(clone.state_dict()["0.weight"].tolist(), model.state_dict()["0.weight"].tolist())
        clone[0].weight.copy_([[0.0] * 3] * 4)
        self.assertNotEqual(model[0].weight.tolist(), clone[0].weight.tolist())
        self.assertEqual(clone[1].num_batches_tracked.dtype, "int32")

    def test_tiny_transformer_learns(self):
        """Classify whether token 1 appears more often than token 2 in a sequence (needs attention)."""
        torch.manual_seed(1)
        rng = np.random.default_rng(0)
        L, V, E = 8, 4, 16
        seqs = rng.integers(0, V, size=(256, L))
        labels = ((seqs == 1).sum(1) > (seqs == 2).sum(1)).astype(np.int32)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.tok = nn.Embedding(V, E)
                self.pos = nn.Embedding(L, E)
                self.enc = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(E, 2, dim_feedforward=32, dropout=0.0, batch_first=True), 1)
                self.head = nn.Linear(E, 2)

            def forward(self, idx):  # idx int32 [N, L]
                n = idx.shape[0]
                h = self.tok(idx.reshape(-1)).reshape(n, L, E) + self.pos(torch.tensor(list(range(L))))
                return self.head(self.enc(h).mean(1))

        model = Net()
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(60):
            for s in range(0, 256, 32):
                opt.zero_grad()
                loss = F.cross_entropy(model(torch.tensor(seqs[s:s + 32])), torch.tensor(labels[s:s + 32]))
                loss.backward()
                opt.step()
                torch.step_end()
        model.eval()
        with torch.no_grad():
            acc = (model(torch.tensor(seqs)).argmax(1) == torch.tensor(labels)).float().mean().item()
        self.assertGreater(acc, 0.9)


if __name__ == "__main__":
    unittest.main()
