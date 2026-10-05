"""Score one query photo against a folder of known designs.

python -m src.infer --query path/to/photo.jpg --gallery data/sarees --topk 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data import list_images, load_rgb, recolored_region, tensor_from_region
from src.eval import device_of, load_model


@torch.no_grad()
def embed_photo(model, image, device: torch.device) -> torch.Tensor:
    tensor = tensor_from_region(recolored_region(image, region="center")).unsqueeze(0).to(device)
    return model(tensor).cpu().squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/color/best.pt"))
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, default=Path("data/sarees"))
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    device = device_of()
    model, _saved = load_model(args.checkpoint, device)
    query = embed_photo(model, load_rgb(args.query), device)

    paths = list_images(args.gallery)
    scores = []
    for path in paths:
        if path.resolve() == args.query.resolve():
            continue
        gallery = embed_photo(model, load_rgb(path), device)
        scores.append((float(query @ gallery), path))
    scores.sort(reverse=True, key=lambda item: item[0])
    print(f"query {args.query}")
    for rank, (score, path) in enumerate(scores[: args.topk], start=1):
        print(f"{rank:2d}  {score:.3f}  {path}")


if __name__ == "__main__":
    main()
