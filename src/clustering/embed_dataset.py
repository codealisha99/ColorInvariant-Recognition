"""STEP 2: embed every source image with the frozen trained encoder.

Reads all images under --data (via the existing src.data.list_images),
embeds each with the same center-crop + ImageNet-normalize pipeline used by
src/infer.py, and writes:

  runs/clustering/embeddings.npy           (N, 128) float32, row i <-> image i
  runs/clustering/image_paths.json         ordered list of image paths (strings)
  runs/clustering/embedding_metadata.json  checkpoint / dim / seed / timestamp

Does not train or modify the model. Read-only use of src.data / src.eval.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch

from src.data import list_images, load_rgb, recolored_region, tensor_from_region
from src.eval import device_of, load_model


@torch.no_grad()
def build_embeddings(
    checkpoint: Path,
    data_root: Path,
    output_dir: Path,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[str]]:
    device = device_of()
    model, _saved = load_model(checkpoint, device)
    model.eval()

    paths = list_images(data_root)
    embs: list[np.ndarray] = []
    batch: list[torch.Tensor] = []
    for path in paths:
        try:
            image = load_rgb(path)
        except Exception as exc:  # failure safety: report, don't silently skip
            raise RuntimeError(f"Cannot load image {path}: {exc}") from exc
        batch.append(tensor_from_region(recolored_region(image, region="center")))
        if len(batch) == batch_size:
            out = model(torch.stack(batch).to(device)).cpu().numpy()
            embs.append(out)
            batch = []
    if batch:
        out = model(torch.stack(batch).to(device)).cpu().numpy()
        embs.append(out)
    embeddings = np.concatenate(embs, axis=0).astype(np.float32)
    return embeddings, [str(p) for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/color/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/sarees"))
    parser.add_argument("--output", type=Path, default=Path("runs/clustering"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    args.output.mkdir(parents=True, exist_ok=True)
    embeddings, image_paths = build_embeddings(
        args.checkpoint, args.data, args.output, batch_size=args.batch_size
    )

    np.save(args.output / "embeddings.npy", embeddings)
    (args.output / "image_paths.json").write_text(json.dumps(image_paths, indent=2))
    metadata = {
        "num_images": len(image_paths),
        "embedding_dim": int(embeddings.shape[1]),
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data),
        "seed": args.seed,
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "pipeline": "center 224 crop, ImageNet normalize, SareeEncoder 128-d L2-normalized",
    }
    (args.output / "embedding_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Number of images: {len(image_paths)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Wrote {args.output / 'embeddings.npy'} shape {embeddings.shape}")


if __name__ == "__main__":
    main()
