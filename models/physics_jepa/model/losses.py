"""Loss for stage-3 physics-head fine-tuning: normalized-space MSE on both
`lambda_ep` and `omega_log`, plus a soft Allen-Dynes physical-validity
penalty on the DENORMALIZED `lambda_ep` prediction (see
`physics_utils.allen_dynes_validity_penalty` for the physical rationale).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..data.dataset import PhysicsTargetStats
from .physics_utils import DEFAULT_MU_STAR, DEFAULT_VALIDITY_EPS, allen_dynes_validity_penalty


@dataclass
class PhysicsLossOutput:
    loss: torch.Tensor
    loss_lambda: torch.Tensor
    loss_omega: torch.Tensor
    loss_validity: torch.Tensor


class PhysicsHeadsLoss(nn.Module):
    """`loss = loss_lambda + omega_weight * loss_omega + validity_weight * loss_validity`.

    Args:
        stats: `PhysicsTargetStats` (train-fit) used to denormalize
            `lambda_pred_norm` back to physical units before evaluating the
            Allen-Dynes validity penalty (the penalty is only meaningful in
            physical units, not z-scored log-space).
        omega_weight: Relative weight on the `omega_log` MSE term --
            `lambda_ep`'s normalized-log-space MSE and `omega_log`'s are
            each O(1) by construction (both z-scored), so 1.0 (equal
            weighting) is the default; expose it for tuning either target's
            relative priority.
        validity_weight: Weight on the soft Allen-Dynes validity penalty,
            small by default (0.05) since it is a physical-plausibility
            regularizer, not the primary supervised signal.
        mu_star: Coulomb pseudopotential (see `physics_utils`).
    """

    def __init__(
        self,
        stats: PhysicsTargetStats,
        omega_weight: float = 1.0,
        validity_weight: float = 0.05,
        mu_star: float = DEFAULT_MU_STAR,
        validity_eps: float = DEFAULT_VALIDITY_EPS,
    ):
        super().__init__()
        self.stats = stats
        self.omega_weight = omega_weight
        self.validity_weight = validity_weight
        self.mu_star = mu_star
        self.validity_eps = validity_eps

    def forward(
        self,
        lambda_pred_norm: torch.Tensor,
        omega_pred_norm: torch.Tensor,
        lambda_target: torch.Tensor,
        omega_target: torch.Tensor,
    ) -> PhysicsLossOutput:
        z_lambda_target, z_omega_target = self.stats.normalize(lambda_target, omega_target)

        loss_lambda = torch.mean((lambda_pred_norm - z_lambda_target) ** 2)
        loss_omega = torch.mean((omega_pred_norm - z_omega_target) ** 2)

        lambda_pred_phys, _ = self.stats.denormalize(lambda_pred_norm, omega_pred_norm)
        loss_validity = allen_dynes_validity_penalty(
            lambda_pred_phys, mu_star=self.mu_star, eps=self.validity_eps
        ).mean()

        loss = loss_lambda + self.omega_weight * loss_omega + self.validity_weight * loss_validity
        return PhysicsLossOutput(
            loss=loss, loss_lambda=loss_lambda, loss_omega=loss_omega, loss_validity=loss_validity
        )


__all__ = ["PhysicsHeadsLoss", "PhysicsLossOutput"]
