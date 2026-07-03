from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class losses(nn.Module, ABC):
    """Base class shared by project loss modules."""

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


def weighted_contrastive_loss(
    context: torch.Tensor,
    target: torch.Tensor,
    ef_per_atom: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Crys-JEPA weighted contrastive objective over global crystal embeddings."""

    if context.shape != target.shape:
        raise ValueError(
            f"weighted_contrastive_loss expected matching shapes, got "
            f"{tuple(context.shape)} and {tuple(target.shape)}"
        )
    if context.dim() != 2:
        raise ValueError(
            f"weighted_contrastive_loss expects [B, D] tensors, got "
            f"{tuple(context.shape)}"
        )
    if context.size(0) < 2:
        raise ValueError("weighted_contrastive_loss requires batch size >= 2")
    if ef_per_atom.reshape(-1).shape[0] != context.size(0):
        raise ValueError(
            f"ef_per_atom must contain one value per sample, got "
            f"{ef_per_atom.reshape(-1).shape[0]} for batch size {context.size(0)}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    context_norm = F.normalize(context, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    sim = context_norm @ target_norm.t()

    ef = ef_per_atom.reshape(-1).to(device=context.device, dtype=context.dtype)
    weight = 1.0 - torch.exp(-torch.abs(ef.unsqueeze(0) - ef.unsqueeze(1)))
    diag = torch.eye(context.size(0), dtype=context.dtype, device=context.device)
    weight = weight + diag

    logits = sim * weight / temperature
    labels = torch.arange(context.size(0), device=context.device)
    return F.cross_entropy(logits, labels)


class WeightedContrastiveLoss(losses):
    """Module wrapper for the Crys-JEPA weighted contrastive objective."""

    def __init__(self, temperature: float = 0.1, name: str | None = None):
        super().__init__(name="WeightedContrastiveLoss" if name is None else name)
        self.temperature = temperature

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        ef_per_atom: torch.Tensor,
    ) -> torch.Tensor:
        return weighted_contrastive_loss(
            context,
            target,
            ef_per_atom,
            temperature=self.temperature,
        )
