"""Evaluation protocol.

The release has no real multi-colorway pairs, so colorways are synthetic
and the protocol says so.

Split: by source photo, seed 42, 60% train / 20% val / 20% test.
Train photos are never embedded in the reported numbers.

Cross-palette identification
  Gallery: center 224 crop, Lab hue +40°.
  Query: the center crop shifted 48px, mirrored, Lab hue +160°.
  Both crops are the field of the cloth, not the border versus the body.
  The shift and the mirror keep the pixels from lining up.
  Metrics: Rank-1, Rank-5, mAP. One relevant gallery item per query,
  so AP is 1 / rank of that item.

Verification
  Positive pairs: query_i with gallery_i.
  Negative pairs: query_i with gallery_j, j != i.
  Metrics: ROC-AUC, equal-error rate, TAR at FAR = 1%.
  The decision threshold is chosen on val (EER) and applied to test.

Same-palette stress test
  Every gallery photo is recolored so its dominant hue matches that
  query. Rank-1 here is the claim that the motif, not the palette, is
  what matched.

An unedited-photo check embeds the shifted, mirrored crop with its
original palette against the recolored gallery crop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.color import palette_match
from src.data import (
    GALLERY_HUE,
    QUERY_HUE,
    list_images,
    load_rgb,
    recolored_region,
    split_designs,
    tensor_from_region,
)
from src.metrics import identification, pair_scores, verification
from src.model import SareeEncoder


def device_of() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: Path, device: torch.device) -> tuple[SareeEncoder, dict]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = saved["args"]
    model = SareeEncoder(embed_dim=args["embed_dim"], pretrained=False)
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    return model, saved


@torch.no_grad()
def embed_images(
    model: SareeEncoder,
    images,
    device: torch.device,
    batch_size: int = 32,
    backbone_only: bool = False,
) -> torch.Tensor:
    encode = model.forward_backbone if backbone_only else model
    outputs = []
    batch = []
    for image in images:
        batch.append(tensor_from_region(image))
        if len(batch) == batch_size:
            outputs.append(encode(torch.stack(batch).to(device)).cpu())
            batch = []
    if batch:
        outputs.append(encode(torch.stack(batch).to(device)).cpu())
    return torch.cat(outputs)


def region_views(paths: list[Path], degrees: float, region: str):
    return [recolored_region(load_rgb(path), degrees, region=region) for path in paths]


def report_split(
    name: str,
    model,
    paths: list[Path],
    device,
    threshold: float | None = None,
    backbone_only: bool = False,
) -> dict:
    gallery_images = region_views(paths, GALLERY_HUE, "gallery")
    query_images = region_views(paths, QUERY_HUE, "query")
    originals = region_views(paths, 0.0, "query")
    gallery = embed_images(model, gallery_images, device, backbone_only=backbone_only)
    query = embed_images(model, query_images, device, backbone_only=backbone_only)
    original = embed_images(model, originals, device, backbone_only=backbone_only)

    cross = identification(query, gallery)
    original_as_query = identification(original, gallery)
    positive, negative = pair_scores(query, gallery)
    pairs = verification(positive, negative)
    if threshold is not None:
        pairs["tar_at_val_threshold"] = float((positive >= threshold).mean())
        pairs["far_at_val_threshold"] = float((negative >= threshold).mean())
        pairs["val_threshold"] = threshold

    # Same-palette impostors: recolor every gallery item toward this query.
    stress_ranks = []
    for index, query_image in enumerate(query_images):
        matched = [palette_match(gallery_image, query_image) for gallery_image in gallery_images]
        matched_emb = embed_images(model, matched, device, backbone_only=backbone_only)
        query_emb = embed_images(model, [query_image], device, backbone_only=backbone_only)
        similarity = (query_emb @ matched_emb.T).squeeze(0)
        order = torch.argsort(similarity, descending=True)
        rank = int(order.eq(index).nonzero(as_tuple=False)[0].item()) + 1
        stress_ranks.append(rank)
    stress = torch.tensor(stress_ranks)
    return {
        "split": name,
        "designs": len(paths),
        "cross_palette_identification": cross,
        "original_query_identification": original_as_query,
        "cross_palette_verification": pairs,
        "same_palette_stress": {
            "rank1": float((stress == 1).float().mean()),
            "rank5": float((stress <= 5).float().mean()),
            "mAP": float((1.0 / stress.float()).mean()),
        },
    }


def load_imagenet_baseline(device: torch.device) -> SareeEncoder:
    """ImageNet MobileNetV3-Small. Embeddings are pooled features, not the projection."""
    model = SareeEncoder(pretrained=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/color/best.pt"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--imagenet",
        action="store_true",
        help="Frozen ImageNet backbone. L2-normalized pooled features, no trained projection.",
    )
    args = parser.parse_args()

    device = device_of()
    if args.imagenet:
        model = load_imagenet_baseline(device)
        saved = {"epoch": None, "args": {"data": "data/sarees", "embed_dim": model.feat_dim}}
        data_root = Path(args.data or "data/sarees")
        out = args.out or Path("runs/imagenet/eval.json")
        representation = "ImageNet MobileNetV3-Small pooled features, L2-normalized, no fine-tuning and no projection"
    else:
        model, saved = load_model(args.checkpoint, device)
        data_root = Path(args.data or saved["args"]["data"])
        out = args.out or (args.checkpoint.parent / "eval.json")
        representation = f"fine-tuned projection, {model.embed_dim}-d"
    split = split_designs(list_images(data_root), seed=args.seed)

    # Threshold is selected on val. Test numbers below use the val EER threshold
    # only as a reported operating point; AUC and EER are threshold-free.
    val_report = report_split("val", model, split.val, device, backbone_only=args.imagenet)
    val_threshold = val_report["cross_palette_verification"]["eer_threshold"]
    test_report = report_split("test", model, split.test, device, threshold=val_threshold, backbone_only=args.imagenet)
    payload = {
        "checkpoint": "imagenet" if args.imagenet else str(args.checkpoint),
        "epoch": saved.get("epoch"),
        "representation": representation,
        "embedding_dim": model.feat_dim if args.imagenet else model.embed_dim,
        "protocol": {
            "unit": "one source photo = one design",
            "colorways": "synthetic Lab chroma rotation; the corpus has no colorway labels",
            "gallery_hue_deg": GALLERY_HUE,
            "query_hue_deg": QUERY_HUE,
            "gallery_crop": "center 224 of a 360px resize, hue +40",
            "query_crop": "center 224 shifted 48px and mirrored, hue +160",
            "split": "60/20/20 by photo, seed 42",
            "corpus": "Drive handloom_sarees only (165 photos). normal_sarees was empty in the public folder. Kaggle weave-class set is not design-level and was not used.",
            "verification_threshold": "EER threshold fit on val",
            "val_eer_threshold": val_report["cross_palette_verification"]["eer_threshold"],
            "baseline": args.imagenet,
        },
        "val": val_report,
        "test": test_report,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"val": val_report, "test": test_report}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
