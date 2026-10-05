"""STEP 3+4: cluster frozen embeddings with cosine distance.

Default algorithm is DBSCAN (no pre-set k; finds noise). Agglomerative
clustering with a distance threshold is offered as an alternative, also
without a pre-set k. Writes clusters.json + cluster_summary.csv and prints
the human-readable discovered-cluster summary.

Metrics here measure embedding-space separation only, NOT motif accuracy
(there are no ground-truth motif labels).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _require_sklearn():
    try:
        from sklearn.cluster import AgglomerativeClustering, DBSCAN
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for clustering. Install it with: "
            "pip install scikit-learn"
        ) from exc
    return DBSCAN, AgglomerativeClustering


def cluster_embeddings(
    embeddings: np.ndarray,
    algorithm: str = "dbscan",
    eps: float = 0.25,
    min_samples: int = 3,
    distance_threshold: float = 0.35,
) -> np.ndarray:
    """Return integer labels, -1 = noise. Embeddings must be L2-normalized."""
    DBSCAN, AgglomerativeClustering = _require_sklearn()
    if algorithm == "dbscan":
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(embeddings)
    elif algorithm == "agglomerative":
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        ).fit_predict(embeddings)
    else:
        raise ValueError(f"Unknown algorithm {algorithm!r}; use 'dbscan' or 'agglomerative'.")
    return np.asarray(labels, dtype=int)


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Remap raw cluster ids to size rank: 0 = largest group. Noise stays -1.

    Keeps clusters.json / CSV / sheets / console summary on one naming scheme.
    Ties break by smaller raw id for determinism.
    """
    ids = sorted(set(labels.tolist()) - {-1}, key=lambda c: (-int((labels == c).sum()), c))
    mapping = {old: new for new, old in enumerate(ids)}
    out = labels.copy()
    for old, new in mapping.items():
        out[labels == old] = new
    return out


def clusters_to_dict(labels: np.ndarray, image_paths: list[str]) -> dict:
    clusters: dict[str, dict] = {}
    for cluster_id in sorted(set(labels.tolist())):
        if cluster_id == -1:
            continue
        members = [p for p, lab in zip(image_paths, labels.tolist()) if lab == cluster_id]
        clusters[f"cluster_{cluster_id}"] = {"size": len(members), "images": members}
    return clusters


def write_outputs(labels: np.ndarray, image_paths: list[str], output_dir: Path) -> dict:
    """Write clusters.json + cluster_summary.csv (size-descending). Return clusters dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters = clusters_to_dict(labels, image_paths)
    ordered = sorted(clusters.items(), key=lambda kv: kv[1]["size"], reverse=True)
    (output_dir / "clusters.json").write_text(
        json.dumps(dict(ordered), indent=2)
    )
    with open(output_dir / "cluster_summary.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cluster_id", "cluster_size", "image_path"])
        for name, info in ordered:
            for image in info["images"]:
                writer.writerow([name, info["size"], image])
    return dict(ordered)


def print_summary(labels: np.ndarray, image_paths: list[str]) -> None:
    total = len(image_paths)
    noise = int((labels == -1).sum())
    ranked = sorted(
        ((c, int((labels == c).sum())) for c in set(labels.tolist()) if c != -1),
        key=lambda t: (-t[1], t[0]),
    )
    assert noise + sum(s for _, s in ranked) == total, "assigned + noise must equal total images"
    print("=" * 50)
    print("DISCOVERED MOTIF CLUSTERS")
    print("=" * 50)
    for cid, size in ranked:
        print(f"Cluster {cid}: {size} images")
    print(f"Noise: {noise} images")
    print()
    print(f"Total images: {total}")
    print(f"Clustered images: {total - noise}")
    print(f"Noise/outliers: {noise}")
    print("=" * 50)


def load_inputs(output_dir: Path) -> tuple[np.ndarray, list[str]]:
    embeddings = np.load(output_dir / "embeddings.npy")
    image_paths = json.loads((output_dir / "image_paths.json").read_text())
    if len(image_paths) != embeddings.shape[0]:
        raise RuntimeError("embeddings.npy and image_paths.json disagree on N.")
    return embeddings, image_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/color/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/sarees"))
    parser.add_argument("--output", type=Path, default=Path("runs/clustering"))
    parser.add_argument("--algorithm", default="dbscan", choices=["dbscan", "agglomerative"])
    parser.add_argument("--eps", type=float, default=0.25)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--distance-threshold", type=float, default=0.35)
    args = parser.parse_args()

    # If embeddings are missing, build them first via the sibling module.
    if not (args.output / "embeddings.npy").exists():
        from src.clustering.embed_dataset import build_embeddings

        args.output.mkdir(parents=True, exist_ok=True)
        embeddings, image_paths = build_embeddings(args.checkpoint, args.data, args.output)
        np.save(args.output / "embeddings.npy", embeddings)
        (args.output / "image_paths.json").write_text(json.dumps(image_paths, indent=2))
    else:
        embeddings, image_paths = load_inputs(args.output)

    labels = cluster_embeddings(
        embeddings,
        algorithm=args.algorithm,
        eps=args.eps,
        min_samples=args.min_samples,
        distance_threshold=args.distance_threshold,
    )
    if args.algorithm == "dbscan":
        labels = relabel_by_size(labels)
    write_outputs(labels, image_paths, args.output)
    print_summary(labels, image_paths)


if __name__ == "__main__":
    main()
