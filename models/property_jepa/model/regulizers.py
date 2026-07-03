"""Optional embedding regularizers for property-JEPA (stage 2).

Reimplements stage 1's variance-covariance regularizer
(`models/crystal_structure_jepa/JEPA/regulizers.py`, `VCLoss`) locally, to
keep the two stages independently importable. Useful here for the same
reason as stage 1: with a pure MSE prediction loss (`losses.py`,
`property_jepa_loss`) the crystal/property embedding spaces can collapse to
a low-variance, highly-correlated solution that trivially minimizes MSE
without encoding useful structure; VICReg-style variance/covariance
penalties discourage that collapse.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class regulizers(nn.Module, ABC):
    """Base class shared by property-JEPA regularizers."""

    def __init__(self, name: str | None = None, weight: float = 1.0):
        super().__init__()
        self.name = self.__class__.__name__ if name is None else name
        self.weight = weight

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Compute the regularization objective."""

    @staticmethod
    def metrics(**items):
        return {
            key: value.detach().item() if torch.is_tensor(value) else value
            for key, value in items.items()
        }


class HingeStdLoss(regulizers):
    """Encourage each embedding feature to keep at least a target standard deviation."""

    def __init__(self, std_margin: float = 1.0):
        super().__init__(name="HingeStdLoss")
        self.std_margin = std_margin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        return torch.mean(F.relu(self.std_margin - std))


class CovarianceLoss(regulizers):
    """Penalize off-diagonal covariance terms to decorrelate embedding features."""

    def __init__(self):
        super().__init__(name="CovarianceLoss")

    @staticmethod
    def off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        if n != m:
            raise ValueError(f"expected a square matrix, got {tuple(x.shape)}")
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"CovarianceLoss expects [B, D], got {tuple(x.shape)}")
        if x.size(0) < 2:
            return x.new_tensor(0.0)

        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (x.size(0) - 1)
        return self.off_diagonal(cov).pow(2).mean()


class VCLoss(regulizers):
    """Variance-covariance regularizer for property-JEPA embeddings."""

    def __init__(
        self,
        std_coeff: float,
        cov_coeff: float,
        proj: nn.Module | None = None,
    ):
        super().__init__(name="VCLoss")
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.proj = nn.Identity() if proj is None else proj
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 2:
            x = x.reshape(-1, x.shape[-1])
        fx = self.proj(x)

        std_loss = self.std_loss_fn(fx)
        cov_loss = self.cov_loss_fn(fx)
        loss = self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return {
            "loss": loss,
            "std_loss": std_loss,
            "cov_loss": cov_loss,
        }
