"""A character-level language model with nn.TransformerEncoder (causal), written the PyTorch way.

    python examples/tiny_transformer.py [--steps 300] [--threads 0]

Trains on a short built-in text (about 8 s for 300 steps on a 12-thread x86 PC), then samples from it.
Only the imports differ from a PyTorch script.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import newch as torch  # noqa: E402
import newch.net as nn  # noqa: E402
import newch.net.functional as F  # noqa: E402

TEXT = (
    "small machines can learn too. a model that fits in a few megabytes can still read, "
    "count and remember. it trains one batch at a time, frees what it no longer needs, "
    "and keeps going when the power comes back. small machines can learn too. "
) * 8


class CharLM(nn.Module):
    def __init__(self, vocab: int, block: int, d_model: int = 64, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(block, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        rows = torch.tensor(list(range(block))).reshape(block, 1)
        with torch.persistent():
            self.causal = torch.tensor(list(range(block))) > rows  # 1 = future position: not attended

    def forward(self, idx):  # idx int32 [N, block]
        n, t = idx.shape
        h = self.tok(idx) + self.pos(torch.tensor(list(range(t))))
        h = self.encoder(h, mask=self.causal, is_causal=True)
        return self.head(self.norm(h))  # [N, block, vocab]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    torch.set_default_context(torch.Context(seed=1, threads=args.threads))

    chars = sorted(set(TEXT))
    stoi = {c: i for i, c in enumerate(chars)}
    data = [stoi[c] for c in TEXT]
    rng = random.Random(0)

    model = CharLM(len(chars), args.block)
    print(f"{model.num_parameters():,} parameters, {torch.default_context().threads} threads")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        starts = [rng.randrange(len(data) - args.block - 1) for _ in range(args.batch_size)]
        x = torch.tensor([data[s:s + args.block] for s in starts])
        y = torch.tensor([data[s + 1:s + args.block + 1] for s in starts])
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == 1:
            print(f"step {step:4d}  loss {loss.item():.3f}  {time.perf_counter() - t0:.1f}s", flush=True)
        torch.step_end()
    peak = torch.mem_stats()["arena_peak_bytes"] / 2**20
    print(f"trained in {time.perf_counter() - t0:.1f}s, arena peak {peak:.1f} MiB")

    model.eval()
    out = [stoi[c] for c in "small "]
    with torch.no_grad():
        while len(out) < 120:
            ctx = out[-args.block:]
            if len(ctx) < args.block:  # left-pad with spaces to a full block
                ctx = [stoi[" "]] * (args.block - len(ctx)) + ctx
            logits = model(torch.tensor([ctx]))
            out.append(int(logits[0][args.block - 1].argmax().item()))
            torch.step_end()
    print("sample:", "".join(chars[i] for i in out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
