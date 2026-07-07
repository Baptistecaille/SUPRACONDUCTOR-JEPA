"""Energy-aware weighted InfoNCE loss for `foundation_jepa`, porting the
official Crys-JEPA "embedding screening" objective
(`components/jepa/frame/jepa.py::JEPA.cal_weight` / `JEPA.get_loss`,
arXiv:2605.14759).

Mechanism
---------
Within a training batch of `B` crystals, let `sim[i, j]` be the cosine
similarity between crystal `i`'s context embedding and crystal `j`'s target
embedding. Standard InfoNCE would only reward `sim[i, i]` (the true
context/target pair for the SAME crystal) relative to all `sim[i, j]`,
`j != i`, treating every other crystal in the batch as an equally-hard
negative regardless of how physically similar it is to crystal `i`.

The paper's insight (the "embedding screening" property that lets stability
comparisons later be done by embedding distance alone) is to make that
implicit repulsion energy-aware: crystals with a similar formation energy
per atom should NOT be pushed apart as hard as crystals with very different
formation energies, because part of what the embedding space is meant to
encode is a notion of energetic proximity. This is implemented via a
per-pair weight

    w[i, j] = 1 - exp(-|E_i - E_j|)     (-> 0 as E_i, E_j coincide,
                                            -> 1 as they diverge)

added to the diagonal (`w[i, i] = 1 + 1 = 2` after adding the identity, per
the official code) before scaling the similarity matrix and taking a
row-wise softmax cross-entropy against the diagonal (the true positive
pair). A near-zero weight for a similar-energy negative SHRINKS its
contribution to the softmax denominator relative to a large-weight
dissimilar-energy negative -- i.e. the loss stops penalizing the model for
placing similar-energy crystals near each other in embedding space, exactly
the desired "embedding screening" behavior.

Numerical-stability additions beyond the official code (justified since
this repo trains on 312k structures spanning a much wider formation-energy
range -- including some outliers -- than the paper's curated MP-only
corpus):
  * `temperature` is a named argument (official code hardcodes `0.1`).
  * an `eps` floor inside the two `log(... + eps)` calls, and clamping the
    cosine-similarity denominator away from exactly zero (guards a
    zero-norm embedding, which would otherwise NaN the whole batch).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def energy_aware_weight(formation_energy_peratom: torch.Tensor) -> torch.Tensor:
    """Pairwise weight `w[i, j] = 1 - exp(-|E_i - E_j|)`, port of official `cal_weight`.

    Args:
        formation_energy_peratom: `(B,)` per-crystal formation energy per atom.

    Returns:
        `(B, B)` symmetric weight matrix, `w[i,i] = 0` (no self-repulsion
        before the identity term is added by the caller).
    """
    e = formation_energy_peratom.reshape(-1)
    return 1.0 - torch.exp(-torch.abs(e.unsqueeze(0) - e.unsqueeze(1)))


@dataclass
class FoundationJEPALossOutput:
    loss: torch.Tensor
    loss_infonce: torch.Tensor
    loss_reg: torch.Tensor


class EnergyAwareInfoNCELoss(nn.Module):
    """Energy-aware weighted InfoNCE between context and target crystal embeddings.

    Port of the official `JEPA.get_loss`, with a configurable `temperature`
    and numerical-stability epsilon floors (see module docstring).
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        formation_energy_peratom: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context: `(B, H)` predictor output (context embedding + augmentation
                conditioning, passed through the predictor MLP).
            target: `(B, H)` target-branch crystal embedding (no augmentation).
            formation_energy_peratom: `(B,)`.

        Returns:
            Scalar loss (mean negative log-probability of the true
            context/target pair under the energy-aware-weighted softmax).
        """
        b = context.shape[0]
        if target.shape[0] != b:
            raise ValueError(
                f"context batch {b} must match target batch {target.shape[0]}"
            )
        if formation_energy_peratom.numel() != b:
            raise ValueError(
                f"formation_energy_peratom must have {b} entries, got "
                f"{formation_energy_peratom.numel()}"
            )

        context_norm = torch.norm(context, dim=-1, keepdim=True).clamp_min(self.eps)
        target_norm = torch.norm(target, dim=-1, keepdim=True).clamp_min(self.eps)
        dot_numerator = context @ target.t()
        dot_denominator = context_norm @ target_norm.t()
        sim = dot_numerator / dot_denominator

        weight = energy_aware_weight(formation_energy_peratom)
        diag = torch.eye(b, dtype=sim.dtype, device=sim.device)
        weight = weight + diag

        scaled = torch.exp(sim * weight / self.temperature)
        probs = scaled / (scaled.sum(dim=-1, keepdim=True) + self.eps)
        return -torch.log(probs.diagonal() + self.eps).mean()


class VarianceCovarianceRegularizer(nn.Module):
    """VICReg-style anti-collapse regularizer on the target-branch embedding.

    Reuses the same std-hinge + covariance-decorrelation formulation as
    `models/property_jepa/model/regulizers.py::VCLoss` (itself a port of
    stage 1's `VCLoss`) rather than the paper's SIGReg, since SIGReg's
    kernel-Stein-discrepancy formulation is not specified precisely enough
    in the paper text to port faithfully, and this repo already has a
    working, tested VICReg-style regularizer with the same purpose
    (preventing the embedding space from collapsing to a low-variance,
    highly-correlated solution that trivially minimizes the weighted
    InfoNCE loss above via a partial/degenerate solution).
    """

    def __init__(self, std_coeff: float = 25.0, cov_coeff: float = 1.0, std_margin: float = 1.0):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_margin = std_margin

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        if n != m:
            raise ValueError(f"expected a square matrix, got {tuple(x.shape)}")
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 2:
            x = x.reshape(-1, x.shape[-1])
        if x.size(0) < 2:
            zero = x.new_tensor(0.0)
            return {"loss": zero, "std_loss": zero, "cov_loss": zero}

        x_centered = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x_centered.var(dim=0) + 1e-4)
        std_loss = F.relu(self.std_margin - std).mean()

        cov = (x_centered.T @ x_centered) / (x.size(0) - 1)
        cov_loss = self._off_diagonal(cov).pow(2).mean()

        loss = self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return {"loss": loss, "std_loss": std_loss, "cov_loss": cov_loss}


class FoundationJEPALoss(nn.Module):
    """Combined energy-aware InfoNCE + anti-collapse regularization loss."""

    def __init__(
        self,
        temperature: float = 0.1,
        reg_weight: float = 0.01,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
    ):
        super().__init__()
        self.infonce = EnergyAwareInfoNCELoss(temperature=temperature)
        self.regularizer = VarianceCovarianceRegularizer(std_coeff=std_coeff, cov_coeff=cov_coeff)
        self.reg_weight = reg_weight

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        formation_energy_peratom: torch.Tensor,
    ) -> FoundationJEPALossOutput:
        loss_infonce = self.infonce(context, target, formation_energy_peratom)
        reg_out = self.regularizer(target)
        loss_reg = reg_out["loss"]
        loss = loss_infonce + self.reg_weight * loss_reg
        return FoundationJEPALossOutput(loss=loss, loss_infonce=loss_infonce, loss_reg=loss_reg)


__all__ = [
    "energy_aware_weight",
    "EnergyAwareInfoNCELoss",
    "VarianceCovarianceRegularizer",
    "FoundationJEPALoss",
    "FoundationJEPALossOutput",
]
