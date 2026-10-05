"""STEP 5+6+7 (figures): 2D embedding map, contact sheets, representatives grid.

2D map is exploratory only: UMAP if installed, else PCA fallback. A 2D
projection is NOT mathematically identical to the 128-D space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _require_mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for visualization. Install it with: pip install matplotlib"
        ) from exc
    return plt


def project_2d(embeddings: np.ndarray, seed: int = 42) -> tuple[np.ndarray, str]:
    """Return (N,2) points and the method name used."""
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=seed)
        return reducer.fit_transform(embeddings), "UMAP (cosine)"
    except ImportError:
        pass  # documented fallback below
    except Exception as exc:
        print(f"UMAP failed ({exc}); falling back to PCA.")
    from sklearn.decomposition import PCA

    print("UMAP unavailable; using PCA fallback for the 2D map.")
    return PCA(n_components=2, random_state=seed).fit_transform(embeddings), "PCA fallback"


def make_embedding_map(
    xy: np.ndarray,
    labels: np.ndarray,
    image_paths: list[str],
    out_path: Path,
    method: str,
) -> None:
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=(10, 8))
    cluster_ids = sorted(c for c in set(labels.tolist()) if c != -1)
    cmap = plt.get_cmap("tab20" if len(cluster_ids) <= 20 else "hsv")
    for i, cid in enumerate(cluster_ids):
        mask = labels == cid
        color = cmap(i / max(1, len(cluster_ids) - 1)) if len(cluster_ids) > 1 else cmap(0)
        ax.scatter(
            xy[mask, 0], xy[mask, 1], s=36, color=color, label=f"Cluster {cid}",
            edgecolors="k", linewidths=0.4, zorder=2,
        )
    noise = labels == -1
    if noise.any():
        ax.scatter(
            xy[noise, 0], xy[noise, 1], s=30, color="black", marker="x",
            label="Noise/outliers", zorder=3,
        )
    # Index labels only when readable; otherwise skip to avoid clutter.
    if len(image_paths) <= 60:
        for idx, (x, y) in enumerate(xy):
            ax.annotate(str(idx), (x, y), fontsize=6, alpha=0.7)
    ax.set_title("DeepLure Saree Embedding Space - Discovered Clusters")
    ax.set_xlabel("Embedding Dimension 1")
    ax.set_ylabel("Embedding Dimension 2")
    ax.text(
        0.01, 0.01,
        f"Exploratory {method} projection; 2D proximity \u2249 128-D distance.",
        transform=ax.transAxes, fontsize=7, alpha=0.7,
    )
    ax.legend(markerscale=1.5, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path} ({method})")


def make_contact_sheets(
    labels: np.ndarray,
    image_paths: list[str],
    out_dir: Path,
    thumb: int = 224,
    cols: int = 5,
) -> list[Path]:
    from PIL import Image, ImageDraw

    plt = _require_mpl()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Purge stale sheets from earlier runs with different cluster counts.
    for stale in sorted(out_dir.glob("cluster_*.png")):
        stale.unlink()
    written: list[Path] = []
    for cid in sorted(set(labels.tolist())):
        members = [(i, p) for i, (p, lab) in enumerate(zip(image_paths, labels.tolist())) if lab == cid]
        if not members:
            continue
        rows = (len(members) + cols - 1) // cols
        fig_w, fig_h = cols * 2.4, rows * 2.9 + 0.7
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
        tag = "noise" if cid == -1 else f"{cid:02d}"
        title = f"Noise/outliers - {len(members)} images" if cid == -1 else f"Cluster {cid} - {len(members)} images"
        fig.suptitle(title, fontsize=14)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (idx, path) in zip(axes.flat, members):
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((thumb, thumb))
            except Exception as exc:
                print(f"WARNING: cannot load {path}: {exc}; leaving blank.")
                continue
            ax.imshow(img)
            ax.set_title(f"{idx}: {Path(path).name}", fontsize=7)
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        out_path = out_dir / f"cluster_{tag}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        written.append(out_path)
        # keep linters quiet about the ImageDraw import path existing
        _ = ImageDraw
    print(f"Wrote {len(written)} contact sheets to {out_dir}/")
    return written


def centroid_representatives(
    embeddings: np.ndarray, labels: np.ndarray, per_cluster: int = 3
) -> dict[str, list[int]]:
    """Indices of the `per_cluster` nearest images to each centroid (cosine)."""
    reps: dict[str, list[int]] = {}
    norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    for cid in sorted(set(labels.tolist())):
        if cid == -1:
            continue
        idx = np.flatnonzero(labels == cid)
        centroid = norms[idx].mean(axis=0)
        centroid = centroid / max(1e-12, np.linalg.norm(centroid))
        dist = 1.0 - norms[idx] @ centroid
        order = idx[np.argsort(dist)][:per_cluster]
        reps[f"cluster_{cid}"] = order.tolist()
    return reps


def make_representatives_fig(
    reps: dict[str, list[int]],
    image_paths: list[str],
    out_path: Path,
    thumb: int = 224,
) -> None:
    from PIL import Image

    plt = _require_mpl()
    names = sorted(reps, key=lambda n: len(reps[n]), reverse=True)
    if not names:
        print("No clusters; skipping representatives figure.")
        return
    per = max(len(v) for v in reps.values())
    fig, axes = plt.subplots(len(names), per, figsize=(3 * per, 3 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        for col in range(per):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(reps[name]):
                idx = reps[name][col]
                try:
                    img = Image.open(image_paths[idx]).convert("RGB")
                    img.thumbnail((thumb, thumb))
                    ax.imshow(img)
                    ax.set_title(f"{name} rep{col} [{idx}]", fontsize=8)
                except Exception as exc:
                    ax.set_title(f"unreadable [{idx}]: {exc}", fontsize=7)
    fig.suptitle("Cluster representatives (closest to 128-D centroid, cosine)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/clustering"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    embeddings = np.load(args.output / "embeddings.npy")
    image_paths = json.loads((args.output / "image_paths.json").read_text())
    clusters = json.loads((args.output / "clusters.json").read_text())
    # Rebuild label array from clusters.json (+ noise = images not listed).
    listed = {img for info in clusters.values() for img in info["images"]}
    labels = np.full(len(image_paths), -1, dtype=int)
    name_to_id = {name: int(name.split("_", 1)[1]) for name in clusters}
    for name, info in clusters.items():
        for img in info["images"]:
            labels[image_paths.index(img)] = name_to_id[name]
    # Images absent from every cluster are noise (kept as -1).
    assert len(listed) + int((labels == -1).sum()) == len(image_paths)

    xy, method = project_2d(embeddings, seed=args.seed)
    make_embedding_map(xy, labels, image_paths, args.output / "embedding_map.png", method)
    make_contact_sheets(labels, image_paths, args.output / "contact_sheets")
    reps = centroid_representatives(embeddings, labels)
    (args.output / "representatives.json").write_text(
        json.dumps(
            {k: [{"index": i, "image": image_paths[i]} for i in v] for k, v in reps.items()},
            indent=2,
        )
    )
    make_representatives_fig(reps, image_paths, args.output / "cluster_representatives.png")


if __name__ == "__main__":
    main()
