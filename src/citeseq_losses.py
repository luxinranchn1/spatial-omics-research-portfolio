"""Portable partial-label loss excerpt; explicit parent mapping replaces private config."""
import torch
from torch.nn import functional as F

def classification_loss(logits: torch.Tensor, fine: torch.Tensor, coarse: torch.Tensor, arm: str, leaf_parent: torch.Tensor) -> torch.Tensor:
    """Mean over supervised cells; undefined cells contribute no class gradient."""
    resolved = fine >= 0
    if arm == 'hard':
        if not resolved.any():
            raise ValueError('An all-unlabelled hard-arm batch cannot use official two-task GradNorm')
        return F.cross_entropy(logits[resolved], fine[resolved], label_smoothing=0)
    coarse_only = (fine < 0) & (coarse >= 0)
    supervised = resolved | coarse_only
    if not supervised.any():
        raise ValueError('An all-unlabelled partial-arm batch cannot use official two-task GradNorm')
    per_cell = logits.new_zeros(logits.shape[0])
    if resolved.any():
        per_cell[resolved] = F.cross_entropy(logits[resolved], fine[resolved], reduction='none', label_smoothing=0)
    if coarse_only.any():
        logp = F.log_softmax(logits[coarse_only], dim=1)
        parents = leaf_parent.to(logits.device)
        c = coarse[coarse_only]
        compatible = parents[None, :] == c[:, None]
        per_cell[coarse_only] = -torch.logsumexp(logp.masked_fill(~compatible, -torch.inf), dim=1)
    return per_cell[supervised].mean()
