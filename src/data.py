"""Design-level splits and the training views.

The Drive release has no colorway labels, so each source photo is one
design. Other palettes of that design are produced by Lab chroma rotation
while the photo is loaded. Splits are by photo, so a crop and a recoloring
of the same file never sit on opposite sides of the split.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from src.color import chroma_rotate

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Fixed angles so gallery and query of one design never share a palette.
GALLERY_HUE = 40.0
QUERY_HUE = 160.0


def list_images(root: Path) -> list[Path]:
    paths = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    if not paths:
        raise FileNotFoundError(f"No images under {root}. Run: python -m src.download")
    return sorted(paths)


@dataclass
class Split:
    train: list[Path]
    val: list[Path]
    test: list[Path]


def split_designs(paths: list[Path], seed: int = 42, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Split:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    n_train = int(round(len(paths) * train_ratio))
    n_val = int(round(len(paths) * val_ratio))
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return Split(
        train=[paths[i] for i in train_idx],
        val=[paths[i] for i in val_idx],
        test=[paths[i] for i in test_idx],
    )


_IMAGE_CACHE: dict[Path, Image.Image] = {}


def load_rgb(path: Path, max_side: int = 768) -> Image.Image:
    cached = _IMAGE_CACHE.get(path)
    if cached is not None:
        return cached
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = max_side / max(width, height)
    if scale < 1:
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BICUBIC)
    _IMAGE_CACHE[path] = image
    return image


def crop_region(image: Image.Image, region: str) -> Image.Image:
    """Two views of the same field, not two different parts of the saree.

    Opposite corners of one photo are often the border and the body, so they
    are not a positive pair. Both crops sit on the center of the cloth.
    The query crop is shifted 48px and mirrored, so the pixels do not line
    up with the gallery crop and a hue-only change is not enough to match.
    """
    image = T.Resize(360)(image)
    width, height = image.size
    if region == "gallery":
        left = (width - 224) // 2
        top = (height - 224) // 2
        mirror = False
    elif region == "query":
        left = min(width - 224, (width - 224) // 2 + 48)
        top = min(height - 224, (height - 224) // 2 + 48)
        mirror = True
    elif region == "center":
        left = (width - 224) // 2
        top = (height - 224) // 2
        mirror = False
    else:
        raise ValueError(f"unknown region {region}")
    crop = image.crop((left, top, left + 224, top + 224))
    if mirror:
        crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return crop


def recolored_region(image: Image.Image, degrees: float = 0.0, sat_scale: float = 1.0, region: str = "center") -> Image.Image:
    if abs(degrees) > 1e-6 or abs(sat_scale - 1.0) > 1e-6:
        image = chroma_rotate(image, degrees, sat_scale)
    return crop_region(image, region)


def tensor_from_region(image: Image.Image) -> torch.Tensor:
    """image is already a 224 crop."""
    return _NORMALIZE(T.ToTensor()(image))


_CROP = T.RandomResizedCrop(224, scale=(0.55, 1.0), interpolation=T.InterpolationMode.BICUBIC)
_BLUR = T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
_NORMALIZE = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def _spatial_view(image: Image.Image) -> Image.Image:
    view = _CROP(image)
    if random.random() < 0.5:
        view = T.functional.hflip(view)
    if random.random() < 0.2:
        view = _BLUR(view)
    return view


def sample_batch(
    paths: list[Path],
    indices: np.ndarray,
    views: int,
    color_aug: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """P designs, K views each.

    View 0 of every design shares one hue, so the batch contains
    same-palette negatives. The other views use independent hues, so
    positives of one design do not share a palette.
    """
    shared_hue = random.uniform(0.0, 360.0)
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for label, index in enumerate(indices.tolist()):
        image = load_rgb(paths[int(index)])
        for view_index in range(views):
            view = _spatial_view(image)
            if color_aug:
                if view_index == 0:
                    degrees, sat_scale = shared_hue, random.uniform(0.85, 1.15)
                elif random.random() < 0.15:
                    degrees, sat_scale = random.uniform(-20.0, 20.0), random.uniform(0.85, 1.15)
                else:
                    degrees = random.uniform(0.0, 360.0)
                    sat_scale = random.uniform(0.6, 1.4)
                view = chroma_rotate(view, degrees, sat_scale)
            images.append(_NORMALIZE(T.ToTensor()(view)))
            labels.append(label)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def iter_epoch(paths: list[Path], designs_per_batch: int, views: int, color_aug: bool, rng: np.random.Generator):
    order = rng.permutation(len(paths))
    for start in range(0, len(order) - designs_per_batch + 1, designs_per_batch):
        yield sample_batch(paths, order[start : start + designs_per_batch], views, color_aug)
