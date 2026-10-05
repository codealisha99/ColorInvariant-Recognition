"""STEP 11: run the full clustering experiment end to end.

    python -m src.clustering.run --checkpoint runs/color/best.pt --data data/sarees --output runs/clustering

Pipeline: embed -> parameter sweep -> select config -> final cluster ->
metrics -> cluster JSON/CSV -> 2D map -> contact sheets -> representatives ->
README. Never touches train/eval/infer. All outputs under --output only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/color/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/sarees"))
    parser.add_argument("--output", type=Path, default=Path("runs/clustering"))
    parser.add_argument("--algorithm", default="dbscan", choices=["dbscan", "agglomerative"])
    parser.add_argument("--eps", type=float, default=None,
                        help="DBSCAN eps. If omitted, chosen from the sweep.")
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--distance-threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from src.clustering import cluster as C
    from src.clustering import embed_dataset as E
    from src.clustering import report as R
    from src.clustering import visualize as V

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # 1. embeddings
    args.output.mkdir(parents=True, exist_ok=True)
    embeddings, image_paths = E.build_embeddings(args.checkpoint, args.data, args.output)
    np.save(args.output / "embeddings.npy", embeddings)
    (args.output / "image_paths.json").write_text(json.dumps(image_paths, indent=2))
    (args.output / "embedding_metadata.json").write_text(json.dumps({
        "num_images": len(image_paths),
        "embedding_dim": int(embeddings.shape[1]),
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data),
        "seed": args.seed,
    }, indent=2))
    print(f"Number of images: {len(image_paths)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Checkpoint: {args.checkpoint}")

    # 2. sweep (dbscan only; agglomerative uses the given threshold directly)
    if args.algorithm == "dbscan":
        rows = R.run_sweep(embeddings)
        R.write_sweep_csv(rows, args.output / "cluster_sweep.csv")
        best, reason = R.select_config(rows)
        print(reason)
        eps = args.eps if args.eps is not None else float(best["eps"])
        ms = args.min_samples if args.min_samples is not None else int(best["min_samples"])
        if args.eps is not None or args.min_samples is not None:
            reason = f"User override: eps={eps}, min_samples={ms}."
        params: dict = {"eps": eps, "min_samples": ms}
    else:
        params = {"distance_threshold": args.distance_threshold}
        reason = (f"Agglomerative requested with distance_threshold={args.distance_threshold}; "
                  "no eps sweep applies.")
        eps, ms = 0.0, 0  # unused

    # 3. final cluster (relabel so cluster_0 is always the largest group)
    labels = C.cluster_embeddings(
        embeddings, algorithm=args.algorithm, eps=eps,
        min_samples=ms, distance_threshold=args.distance_threshold,
    )
    labels = C.relabel_by_size(labels)
    clusters = C.write_outputs(labels, image_paths, args.output)
    C.print_summary(labels, image_paths)

    # 4. metrics
    metrics = R.unsupervised_metrics(embeddings, labels)
    (args.output / "cluster_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # 5. 2D map
    xy, method_2d = V.project_2d(embeddings, seed=args.seed)
    V.make_embedding_map(xy, labels, image_paths, args.output / "embedding_map.png", method_2d)

    # 6. contact sheets
    sheets = V.make_contact_sheets(labels, image_paths, args.output / "contact_sheets")

    # 7. representatives
    reps = V.centroid_representatives(embeddings, labels)
    (args.output / "representatives.json").write_text(json.dumps(
        {k: [{"index": i, "image": image_paths[i]} for i in v] for k, v in reps.items()},
        indent=2,
    ))
    V.make_representatives_fig(reps, image_paths, args.output / "cluster_representatives.png")

    # 8. README
    R.write_readme(
        args.output / "README.md",
        checkpoint=str(args.checkpoint),
        embedding_dim=int(embeddings.shape[1]),
        num_images=len(image_paths),
        algorithm=f"{args.algorithm} (cosine distance)",
        params=params,
        selection_reason=reason,
        metrics=metrics,
        method_2d=method_2d,
        contact_sheets=[str(Path("contact_sheets") / p.name) for p in sheets],
        sweep_path="cluster_sweep.csv",
    )
    print(f"Done. All outputs under {args.output}/")


if __name__ == "__main__":
    main()
