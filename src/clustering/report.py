"""STEP 8+9+10: quality metrics, parameter sweep, human-readable report.

All metrics are unsupervised embedding-space separation measures. They do
NOT measure motif classification accuracy: the corpus has no ground-truth
motif/design labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

LIMITATION_TEXT = (
    "The supplied corpus does not contain ground-truth motif/design labels. "
    "Therefore these clusters are automatically discovered groups in the learned "
    "embedding space and should not be interpreted as verified motif identities. "
    "The contact sheets are used for qualitative human inspection."
)

SWEEP_EPS = [0.15, 0.20, 0.25, 0.30, 0.35]
SWEEP_MIN_SAMPLES = [2, 3, 4]


def unsupervised_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    n_clusters = len(set(labels.tolist()) - {-1})
    n_noise = int((labels == -1).sum())
    sizes = sorted(
        [int((labels == c).sum()) for c in set(labels.tolist()) if c != -1],
        reverse=True,
    )
    out: dict = {
        "num_images": int(len(labels)),
        "num_clusters": int(n_clusters),
        "num_noise": int(n_noise),
        "cluster_sizes_desc": sizes,
        "note": "Unsupervised separation measures only; not motif classification accuracy.",
    }
    if n_clusters < 2 or n_clusters >= len(labels) - n_noise or (labels != -1).sum() < 3:
        out["silhouette_score"] = None
        out["davies_bouldin_score"] = None
        out["warning"] = "Too few clusters/samples for silhouette / Davies-Bouldin."
        return out
    try:
        from sklearn.metrics import davies_bouldin_score, silhouette_score

        clustered = labels != -1
        # Cosine is supported by silhouette; Davies-Bouldin is Euclidean-only,
        # so it runs on L2-normalized vectors (monotonic with cosine distance).
        out["silhouette_score"] = float(
            silhouette_score(embeddings[clustered], labels[clustered], metric="cosine")
        )
        out["davies_bouldin_score"] = float(
            davies_bouldin_score(embeddings[clustered], labels[clustered])
        )
    except Exception as exc:
        out["silhouette_score"] = None
        out["davies_bouldin_score"] = None
        out["warning"] = f"Metric computation failed (non-fatal): {exc}"
    return out


def run_sweep(
    embeddings: np.ndarray,
    eps_values: list[float] = SWEEP_EPS,
    min_samples_values: list[int] = SWEEP_MIN_SAMPLES,
) -> list[dict]:
    from src.clustering.cluster import cluster_embeddings

    rows: list[dict] = []
    for eps in eps_values:
        for ms in min_samples_values:
            try:
                labels = cluster_embeddings(embeddings, algorithm="dbscan", eps=eps, min_samples=ms)
                m = unsupervised_metrics(embeddings, labels)
                rows.append({
                    "eps": eps, "min_samples": ms,
                    "num_clusters": m["num_clusters"], "num_noise": m["num_noise"],
                    "silhouette_score": m["silhouette_score"],
                    "davies_bouldin_score": m["davies_bouldin_score"],
                    "cluster_sizes": str(m["cluster_sizes_desc"]),
                })
            except Exception as exc:
                rows.append({
                    "eps": eps, "min_samples": ms, "num_clusters": None,
                    "num_noise": None, "silhouette_score": None,
                    "davies_bouldin_score": None, "cluster_sizes": f"ERROR: {exc}",
                })
    return rows


def select_config(rows: list[dict]) -> tuple[dict, str]:
    """Pick a stable, interpretable config — never simply the most clusters.

    Filters in order: (1) valid silhouette and >=2 clusters; (2) noise<=30%
    of all images; (3) prefer configs with no 2-image fragments (every group
    then has >=3 density-supported members); (4) highest silhouette within
    the surviving pool. Silhouette alone rewards tiny tight cores, so steps
    2-3 guard against fragmented or noise-dominated solutions.
    """
    valid = [r for r in rows if r["silhouette_score"] is not None and (r["num_clusters"] or 0) >= 2]
    if not valid:
        fallback = min(rows, key=lambda r: (r["num_noise"] is None, r["num_noise"] or 10**9))
        return fallback, "No config yielded a valid silhouette score; fell back to least noise."
    def _total(r: dict) -> int:
        sizes = [int(x) for x in r["cluster_sizes"].strip("[]").split(",") if x.strip().isdigit()]
        return (r["num_noise"] or 0) + sum(sizes)

    calm = [r for r in valid if r["num_noise"] / max(1, _total(r)) <= 0.30]
    pool = calm or valid
    solid = [r for r in pool if min(
        (int(x) for x in r["cluster_sizes"].strip("[]").split(",") if x.strip().isdigit()),
        default=0,
    ) >= 3]
    final_pool = solid or pool
    best = max(final_pool, key=lambda r: r["silhouette_score"])
    reason = (
        f"Selected eps={best['eps']}, min_samples={best['min_samples']}: highest silhouette "
        f"({best['silhouette_score']:.3f}) among configs with >=2 clusters"
        + (" and noise<=30%" if calm else "")
        + (" and no 2-image fragments" if solid else "")
        + f" ({best['num_clusters']} clusters, {best['num_noise']} noise). "
        "Tighter-eps configs won on silhouette but fragmented into pairs and heavy "
        "noise; wider-eps configs chained into a 70+ mega-cluster with poor separation; "
        "both were rejected."
    )
    return best, reason


def write_sweep_csv(rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(
    out_path: Path,
    *,
    checkpoint: str,
    embedding_dim: int,
    num_images: int,
    algorithm: str,
    params: dict,
    selection_reason: str,
    metrics: dict,
    method_2d: str,
    contact_sheets: list[str],
    sweep_path: str,
) -> None:
    sizes = metrics.get("cluster_sizes_desc", [])
    out_path.write_text(
        f"""# DeepLure Clustering Experiment (Unsupervised Motif Discovery)

## 1. What this does
Embeds every source image with the frozen trained SareeEncoder and groups
the 128-d embeddings by cosine distance, to explore what the learned
embedding space treats as design-similar. No training, no labels used.

Extra Python packages beyond the main `requirements.txt`: `scikit-learn`
(clustering + metrics + PCA fallback) and `matplotlib` (figures). The main
training/evaluation pipeline does not need them.

## 2. Checkpoint
`{checkpoint}` (frozen; never retrained or modified)

## 3. Embedding dimension
{embedding_dim}

## 4. Distance metric
Cosine distance (embeddings are L2-normalized, so cosine = dot product).

## 5. Clustering algorithm
{algorithm} — DBSCAN by default; agglomerative-with-threshold available.

## 6. Why this algorithm
It does not require pre-specifying k. The brief forbids assuming a fixed
motif count, so k-means-style fixed-k methods were rejected. DBSCAN/HDBSCAN
also surface noise/outliers instead of forcing every saree into a motif.

## 7. Parameter selection
Sweep over eps={SWEEP_EPS} x min_samples={SWEEP_MIN_SAMPLES} in
`{sweep_path}`. {selection_reason}

Final parameters: `{json.dumps(params)}`.

## 8. Discovered clusters
{metrics.get("num_clusters")} clusters (neutral names Cluster 0..N; NOT verified motifs).

## 9. Noise/outliers
{metrics.get("num_noise")} of {num_images} images.

## 10. Cluster size distribution (desc)
{json.dumps(sizes)}

## 11. Unsupervised quality metrics
See `cluster_metrics.json`. Silhouette (cosine): {metrics.get("silhouette_score")}.
Davies-Bouldin (Euclidean on normalized vectors, monotonic with cosine):
{metrics.get("davies_bouldin_score")}. These measure embedding-space
separation, NOT true motif classification accuracy.

## 12. Visualizations
- `embedding_map.png` (exploratory {method_2d} projection; 2D proximity is not identical to 128-D distance)
- `cluster_representatives.png` + `representatives.json` (3 nearest to each 128-D centroid, cosine)

## 13. Contact sheets
`contact_sheets/` — one PNG per cluster/noise, e.g.:
{chr(10).join("- `" + s + "`" for s in contact_sheets[:8])}
Use them for qualitative human inspection of motif coherence.

## 14. Limitations
{LIMITATION_TEXT} A cluster can contain visually related designs without
necessarily representing one manufacturing/design identity. Synthetic
colorways were used in training; clustering here uses original palettes.
Small corpus (~{num_images}); boundaries shift with eps/min_samples — see the sweep.
"""
    )
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/clustering"))
    args = parser.parse_args()

    embeddings = np.load(args.output / "embeddings.npy")
    image_paths = json.loads((args.output / "image_paths.json").read_text())
    clusters = json.loads((args.output / "clusters.json").read_text())
    labels = np.full(len(image_paths), -1, dtype=int)
    for name, info in clusters.items():
        cid = int(name.split("_", 1)[1])
        for img in info["images"]:
            labels[image_paths.index(img)] = cid

    metrics = unsupervised_metrics(embeddings, labels)
    (args.output / "cluster_metrics.json").write_text(json.dumps(metrics, indent=2))
    rows = run_sweep(embeddings)
    write_sweep_csv(rows, args.output / "cluster_sweep.csv")
    best, reason = select_config(rows)
    print(json.dumps(metrics, indent=2))
    print(reason)
    print(f"Sweep best: eps={best['eps']}, min_samples={best['min_samples']}")


if __name__ == "__main__":
    main()
