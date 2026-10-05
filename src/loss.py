"""Supervised contrastive loss.

Positives are the other views of the same design in the batch, including
recolorings. Every other design is a negative, which is why same-palette
recolorings of different motifs have to be in the batch too.
"""

from __future__ import annotations

import torch


def supcon_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """embeddings are L2-normalized, shape [N, D]. labels shape [N].

    The self-comparison is dropped by multiplying it out of the denominator
    rather than by writing -inf into the logits. -inf makes log_softmax return
    NaN on MPS.
    """
    logits = embeddings @ embeddings.T / temperature
    logits = logits - logits.detach().amax(dim=1, keepdim=True)
    self_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))

    positive = labels.view(-1, 1).eq(labels.view(1, -1)) & ~self_mask
    positive_count = positive.sum(dim=1).clamp(min=1)
    loss = -(log_prob * positive.float()).sum(dim=1) / positive_count
    return loss.mean()
