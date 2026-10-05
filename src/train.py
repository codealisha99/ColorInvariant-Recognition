"""Train the color-invariant saree encoder.

Each batch holds P designs and K views of each. Views are random crops,
and with --color-aug (the default) most of them are Lab hue rotations of
the same photo. The loss pulls those views together and pushes other
designs apart.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data import GALLERY_HUE, QUERY_HUE, iter_epoch, list_images, split_designs
from src.loss import supcon_loss
from src.metrics import identification
from src.model import SareeEncoder


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_of() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def embed_split(model: SareeEncoder, paths: list[Path], degrees: float, region: str, device: torch.device, batch_size: int = 32) -> torch.Tensor:
    from src.data import load_rgb, recolored_region, tensor_from_region

    model.eval()
    outputs = []
    batch: list[torch.Tensor] = []
    for path in paths:
        batch.append(tensor_from_region(recolored_region(load_rgb(path), degrees, region=region)))
        if len(batch) == batch_size:
            outputs.append(model(torch.stack(batch).to(device)).cpu())
            batch = []
    if batch:
        outputs.append(model(torch.stack(batch).to(device)).cpu())
    return torch.cat(outputs)


def cross_palette_rank1(model: SareeEncoder, paths: list[Path], device: torch.device) -> float:
    gallery = embed_split(model, paths, GALLERY_HUE, "gallery", device)
    query = embed_split(model, paths, QUERY_HUE, "query", device)
    return identification(query, gallery)["rank1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sarees"))
    parser.add_argument("--out", type=Path, default=Path("runs/color"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--designs-per-batch", type=int, default=16)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color-aug", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = device_of()
    args.out.mkdir(parents=True, exist_ok=True)

    paths = list_images(args.data)
    split = split_designs(paths, seed=args.seed)
    print(f"device={device} designs={len(paths)} train={len(split.train)} val={len(split.val)} test={len(split.test)}")
    print(f"color_aug={args.color_aug} views={args.views} designs_per_batch={args.designs_per_batch}")

    rng = np.random.default_rng(args.seed)
    steps = len(split.train) // args.designs_per_batch
    if steps == 0:
        raise RuntimeError("Training batch is larger than the train split. Lower --designs-per-batch.")

    model = SareeEncoder(embed_dim=args.embed_dim, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_rank1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        batches = iter_epoch(split.train, args.designs_per_batch, args.views, args.color_aug, rng)
        for images, labels in tqdm(batches, total=steps, desc=f"epoch {epoch}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = supcon_loss(model(images), labels, temperature=args.temperature)
            if not torch.isfinite(loss):
                print("non-finite loss, skipping step")
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        schedule.step()
        rank1 = cross_palette_rank1(model, split.val, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_cross_palette_rank1": rank1}
        history.append(row)
        print(f"epoch {epoch:02d}  loss {row['loss']:.4f}  val rank-1 {rank1:.3f}")
        torch.save({"model": model.state_dict(), "args": vars(args) | {"out": str(args.out), "data": str(args.data)}, "epoch": epoch}, args.out / "last.pt")
        if rank1 >= best_rank1:
            best_rank1 = rank1
            torch.save({"model": model.state_dict(), "args": vars(args) | {"out": str(args.out), "data": str(args.data)}, "epoch": epoch, "val_rank1": rank1}, args.out / "best.pt")

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val rank-1 {best_rank1:.3f}  saved {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
