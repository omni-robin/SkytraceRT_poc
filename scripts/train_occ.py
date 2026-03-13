#!/usr/bin/env python3
"""Train a tiny raw-IQ -> frequency occupancy model (Apple Silicon friendly).

Defaults are chosen for MPS (Metal):
- uses torch MPS if available
- uses mixed precision autocast on MPS (fp16) for speed

Input: NPZ from scripts/build_windows_npz.py
Output: a torch checkpoint (.pt) containing model + meta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import sys

# Allow running from repo root without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skytracert_poc.dataset import NPZWindowsDataset
from skytracert_poc.model import TinyIQOccNet


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
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--width", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # Small global knobs that help on Apple Silicon.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = pick_device()
    print("device:", device)

    ds = NPZWindowsDataset(args.npz)
    freq_bins = int(ds.meta.freq_bins)

    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    # MPS: num_workers>0 can be counterproductive; keep it 0 for now.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TinyIQOccNet(freq_bins=freq_bins, width=args.width).to(device)

    # BCEWithLogitsLoss because model outputs logits.
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    use_amp = device.type in ("mps", "cuda")

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss = 0.0
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
            total_loss += float(loss.detach().cpu()) * x.shape[0]
            n += x.shape[0]
        return total_loss / max(n, 1)

    best_val = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(train_loader, train=True)
        va = run_epoch(val_loader, train=False)
        print(f"epoch {ep:03d}  train_loss={tr:.4f}  val_loss={va:.4f}")

        if va < best_val:
            best_val = va
            ckpt = {
                "model_state_dict": model.state_dict(),
                "model": {"freq_bins": freq_bins, "width": args.width},
                "meta": ds.meta.__dict__,
                "best_val": best_val,
                "epoch": ep,
            }
            torch.save(ckpt, out_path)

    print("saved best ckpt ->", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
