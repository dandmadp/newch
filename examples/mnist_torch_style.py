"""MNIST written the way PyTorch training scripts usually are: an nn.Module
subclass, functional activations, an explicit train/test loop.

    python examples/mnist_torch_style.py [--epochs 1] [--batch-size 64] [--lr 1e-3] [--resume]

A checkpoint (model, optimizer, scheduler, epoch) is written after every epoch;
--resume continues from it exactly where it stopped.

Only the imports differ from a PyTorch script (newch -> torch, newch.net -> torch.nn).
Data comes from examples/mnist_cnn.py's on-disk FileArrays (downloaded on first run).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import newch as torch  # noqa: E402
import newch.net as nn  # noqa: E402
import newch.net.functional as F  # noqa: E402
from mnist_cnn import prepare  # noqa: E402
from newch.data import DataLoader, TensorDataset  # noqa: E402


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


def train(model, loader, optimizer, epoch: int, log_every: int = 200) -> None:
    model.train()
    for batch, (data, target) in enumerate(loader):
        data = data.float() / 255  # uint8 pixels -> [0, 1]
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), target)
        loss.backward()
        optimizer.step()
        if batch % log_every == 0:
            print(f"epoch {epoch} [{batch * len(data)}/{len(loader.dataset)}] loss {loss.item():.4f}", flush=True)
        torch.step_end()  # newch: frees this step's temporaries at once


def test(model, loader) -> float:
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data.float() / 255)
            loss_sum += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            torch.step_end()
    n = len(loader.dataset)
    print(f"test: average loss {loss_sum / n:.4f}, accuracy {correct}/{n} ({100.0 * correct / n:.2f}%)")
    return correct / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", default=str(Path.home() / ".cache" / "newch" / "mnist"))
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data_dir = Path(args.data)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(TensorDataset(*prepare(data_dir, "train")), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(*prepare(data_dir, "test")), batch_size=256)

    model = Net().to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    ckpt_path = data_dir / "mnist_torch_style_ckpt.ncw"
    start = 1
    if args.resume and ckpt_path.exists():
        ckpt = torch.load_tensors(ckpt_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start = ckpt["epoch"] + 1
        print(f"resumed after epoch {ckpt['epoch']}")
    for epoch in range(start, args.epochs + 1):
        t0 = time.perf_counter()
        train(model, train_loader, optimizer, epoch)
        print(f"epoch {epoch}: {time.perf_counter() - t0:.1f}s")
        test(model, test_loader)
        scheduler.step()
        torch.save_tensors({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "epoch": epoch}, ckpt_path)
    torch.save_tensors(model.state_dict(), data_dir / "mnist_torch_style.ncw")
    return 0


if __name__ == "__main__":
    sys.exit(main())
