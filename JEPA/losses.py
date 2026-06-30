from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regulizers import (
    CovarianceLoss,
    HingeStdLoss
)


class losses(nn.Module, ABC):
    """Abstract base class shared by project loss modules.

    The lowercase name follows the user-facing API requested for this project.
    Subclasses can use ``metrics`` to expose scalar logging values without
    leaking tensors that still require gradients.
    """

    def __init__(self, name=None, weight=1.0, reduction="mean"):
        super().__init__()
        self.name = self.__class__.__name__ if name is None else name
        self.weight = weight
        self.reduction = reduction

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Compute the loss."""

    @staticmethod
    def metrics(**items):
        return {
            key: value.detach().item() if torch.is_tensor(value) else value
            for key, value in items.items()
        }

    def extra_repr(self):
        return f"name={self.name}, weight={self.weight}, reduction={self.reduction}"


def sq_loss(x, y, reduction="mean"):
    """Simple square loss (MSE)."""
    return nn.functional.mse_loss(x, y, reduction=reduction)


def square_cost_seq(state, predi):
    """Square loss between two [B, C, T, H, W] sequences."""
    return sq_loss(state, predi)


class SquareLossSeq(nn.Module):
    """Square loss over a sequence [B, C, T, H, W] (feature dim at dim 1)."""

    def __init__(self, proj=None):
        super().__init__()
        self.proj = nn.Identity() if proj is None else proj

    def forward(self, state, predi):
        state = self.proj(state.transpose(0, 1).flatten(1).transpose(0, 1))
        predi = self.proj(predi.transpose(0, 1).flatten(1).transpose(0, 1))
        return square_cost_seq(state, predi)


class VICRegLoss(nn.Module):
    """VICReg loss combining invariance, variance (std), and covariance terms."""

    def __init__(self, std_coeff=1.0, cov_coeff=1.0):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, z1, z2):
        """Compute VICReg loss.

        Args:
            z1: [B, D] - First projection tensor
            z2: [B, D] - Second projection tensor

        Returns:
            dict with keys: loss, invariance_loss, var_loss, cov_loss
        """
        # Invariance loss (similarity)
        sim_loss = F.mse_loss(z1, z2)

        # Variance loss (applied to both views and summed)
        var_loss = self.std_loss_fn(z1) + self.std_loss_fn(z2)

        # Covariance loss (applied to both views and summed)
        cov_loss = self.cov_loss_fn(z1) + self.cov_loss_fn(z2)

        total_loss = sim_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss

        return {
            "loss": total_loss,
            "invariance_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }


######################################################
# BCS (Batched Characteristic Slicing) loss for SIGReg


def all_reduce(x, op):
    """All-reduce operation for distributed training."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        op = dist.ReduceOp.__dict__[op]
        dist.all_reduce(x, op=op)
        return x
    else:
        return x


def epps_pulley(x, t_min=-3, t_max=3, n_points=10):
    """Epps-Pulley test statistic for Gaussianity."""
    # integration points
    t = torch.linspace(t_min, t_max, n_points, device=x.device)
    # theoretical CF for N(0, 1)
    exp_f = torch.exp(-0.5 * t**2)
    # ECF
    x_t = x.unsqueeze(2) * t  # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)
    ecf = all_reduce(ecf, op="AVG")
    # weighted L2 distance
    err = exp_f * (ecf - exp_f).abs() ** 2
    T = torch.trapz(err, t, dim=1)
    return T


class BCS(nn.Module):
    """BCS (Batched Characteristic Slicing) loss for SIGReg."""

    def __init__(self, num_slices=256, lmbd=10.0):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0
        self.lmbd = lmbd

    def forward(self, z1, z2):
        with torch.no_grad():
            dev = z1.device
            g = torch.Generator(device=dev)
            g.manual_seed(self.step)
            proj_shape = (z1.size(1), self.num_slices)
            A = torch.randn(proj_shape, device=dev, generator=g)
            A /= A.norm(p=2, dim=0)
        view1 = z1 @ A
        view2 = z2 @ A

        self.step += 1
        bcs = (epps_pulley(view1).mean() + epps_pulley(view2).mean()) / 2
        invariance_loss = F.mse_loss(z1, z2).mean()
        total_loss = invariance_loss + self.lmbd * bcs
        return {"loss": total_loss, "bcs_loss": bcs, "invariance_loss": invariance_loss}


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
