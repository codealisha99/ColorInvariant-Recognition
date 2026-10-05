"""Closed-set identification and pair verification.

Gallery row i and query row i are two colorways of the same design.
"""

from __future__ import annotations

import numpy as np
import torch


def identification(query: torch.Tensor, gallery: torch.Tensor) -> dict[str, float]:
    """query and gallery are L2-normalized [N, D], aligned by design index."""
    similarity = query @ gallery.T
    order = similarity.argsort(dim=1, descending=True)
    targets = torch.arange(query.size(0))
    match = order.eq(targets[:, None]).float().argmax(dim=1) + 1
    reciprocal = 1.0 / match.float()
    return {
        "rank1": float((match == 1).float().mean()),
        "rank5": float((match <= 5).float().mean()),
        "mAP": float(reciprocal.mean()),
    }


def pair_scores(query: torch.Tensor, gallery: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """One positive per design, and every other gallery row as a negative."""
    similarity = (query @ gallery.T).cpu().numpy()
    positive = np.diag(similarity).copy()
    negative = similarity[~np.eye(similarity.shape[0], dtype=bool)]
    return positive, negative


def verification(positive: np.ndarray, negative: np.ndarray) -> dict[str, float]:
    auc = _roc_auc(positive, negative)
    eer, threshold = _eer(positive, negative)
    tar = _tar_at_far(positive, negative, far_target=0.01)
    return {"roc_auc": auc, "eer": eer, "eer_threshold": threshold, "tar_at_far_1pct": tar}


def _roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    scores = np.concatenate([positive, negative])
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    order = np.argsort(-scores)
    labels = labels[order]
    n_pos = max(1.0, float((labels == 1).sum()))
    n_neg = max(1.0, float((labels == 0).sum()))
    tpr = np.concatenate([[0.0], np.cumsum(labels == 1) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(labels == 0) / n_neg])
    return float(np.trapezoid(tpr, fpr))


def _eer(positive: np.ndarray, negative: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(-1.0, 1.0, 401)
    best_gap = 1e9
    eer = 1.0
    chosen = 0.0
    for threshold in thresholds:
        far = float((negative >= threshold).mean()) if len(negative) else 0.0
        frr = float((positive < threshold).mean()) if len(positive) else 0.0
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            eer = 0.5 * (far + frr)
            chosen = float(threshold)
    return eer, chosen


def _tar_at_far(positive: np.ndarray, negative: np.ndarray, far_target: float) -> float:
    if len(negative) == 0:
        return 1.0
    # Highest threshold that still lets at most far_target of negatives through.
    threshold = float(np.quantile(negative, 1.0 - far_target))
    return float((positive >= threshold).mean())
