"""Loss modules for property-JEPA (stage 2).

Follows the same base-class convention as stage 1
(`models/crystal_structure_jepa/JEPA/losses.py`): a small ABC with a
`.metrics(**items)` helper that detaches/scalar-izes tensors for logging,
reimplemented locally (not imported from stage 1) to keep the two stages
independently importable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class losses(nn.Module, ABC):
    """Base class shared by property-JEPA loss modules."""

    def __init__(self, name: str | None = None, weight: float = 1.0):
        super().__init__()
        self.name = self.__class__.__name__ if name is None else name
        self.weight = weight

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Compute the loss."""

    @staticmethod
    def metrics(**items):
        return {
            key: value.detach().item() if torch.is_tensor(value) else value
            for key, value in items.items()
        }


def property_jepa_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
) -> torch.Tensor:
    """Property-JEPA prediction loss: mean-squared error in latent space.

    Unlike stage 1's `weighted_contrastive_loss` (an InfoNCE-style objective
    over a *batch* of crystal embeddings, needed because there is no
    natural per-sample "correct value" for a masked spatial block beyond
    "the other branch's embedding of the same crystal"), property-JEPA's
    target for a given `(material, queried property)` pair is well defined
    on its own: the ground-truth scalar value of that property, embedded by
    `TargetPropertyEncoder`. A direct per-sample regression loss (MSE
    between predicted and target embeddings) is therefore the natural
    analogue of I-JEPA's L2 prediction loss, without needing a batch-level
    contrastive term.

    Args:
        z_pred: `[B, H]`, from `PropertyPredictor`.
        z_target: `[B, H]`, from `TargetPropertyEncoder` (already detached
            by the caller; this function does not call `.detach()` itself
            so it stays usable in contexts where the caller wants to
            backprop into both, e.g. a unit test).

    Returns:
        Scalar loss.
    """
    if z_pred.shape != z_target.shape:
        raise ValueError(
            f"property_jepa_loss expected matching shapes, got {tuple(z_pred.shape)} "
            f"and {tuple(z_target.shape)}"
        )
    if z_pred.dim() != 2:
        raise ValueError(f"property_jepa_loss expects [B, H] tensors, got {tuple(z_pred.shape)}")
    return F.mse_loss(z_pred, z_target)


class PropertyJEPALoss(losses):
    """Module wrapper for `property_jepa_loss`."""

    def __init__(self, name: str | None = None):
        super().__init__(name="PropertyJEPALoss" if name is None else name)

    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        return property_jepa_loss(z_pred, z_target)
