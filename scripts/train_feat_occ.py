#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skytracert_poc.model_feat import TinyFeatOccNet


class FeatDataset(Dataset):
    def __init__(self, npz_path: str):
        z = np.load(npz_path, allow_pickle=True)
        self.X = z["X_feat"].astype(np.float32)
        self.y = z["y_occ"].astype(np.float32)
        self.meta = json.loads(str(z["meta_json"]))

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = pick_device()
    print("device:", device)

    ds = FeatDataset(args.npz)
    F = int(ds.X.shape[1])

    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TinyFeatOccNet(freq_bins=F, hidden=args.hidden).to(device)

    pos = float(ds.y.mean())
    pos = max(1e-4, min(1.0 - 1e-4, pos))
    pos_weight = torch.tensor([(1.0 - pos) / pos], device=device)
    print(f"pos_frac={pos:.6f} => pos_weight={float(pos_weight.item()):.3f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    use_amp = device.type in ("mps", "cuda")

    def run(loader, train: bool):
        model.train(train)
        tot = 0.0
        n = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            tot += float(loss.detach().cpu()) * x.shape[0]
            n += x.shape[0]
        return tot / max(n, 1)

    best = float("inf")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        tr = run(train_loader, True)
        va = run(val_loader, False)
        print(f"epoch {ep:03d} train_loss={tr:.4f} val_loss={va:.4f}")
        if va < best:
            best = va
            ckpt = {
                "model_state_dict": model.state_dict(),
                "model": {"freq_bins": F, "hidden": args.hidden},
                "meta": ds.meta,
                "best_val": best,
                "epoch": ep,
            }
            torch.save(ckpt, out)

    print("saved ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
