"""Top-level property-JEPA (stage 2) model: wires the four architectural pieces together.

    RotationInvariantCrystalEncoder   (crystal_encoder.py)
        -> z_crystal [B, H]
    PropertyEncoder (online)          (property_encoder.py)
        -> known-property context, mean-pooled over each material's other
           known properties -> [B, H]
    PropertyPredictor                 (predictor.py)
        -> z_pred [B, H] = f(z_crystal, known_property_context, query_type)
    TargetPropertyEncoder (EMA)       (property_encoder.py)
        -> z_target [B, H] = g(query_type, ground_truth_value), no grad
    PropertyJEPALoss                  (losses.py)
        -> MSE(z_pred, z_target.detach())

This is the stage-2 analogue of stage 1's `CrystalJEPA`
(`models/crystal_structure_jepa/JEPA/crystal_jepa.py`): a context/target
asymmetric pair with an EMA-updated, gradient-free target branch, a
predictor conditioned on an auxiliary description of what to predict (there:
"which spatial box was masked" via `mask_box`; here: "which property type is
being queried" via `query_property_type_id`), and an optional
variance-covariance regularizer on the predicted embedding to discourage
representation collapse. It differs in the object being predicted: stage 1
predicts a masked *spatial region of the same crystal*; stage 2 predicts a
held-out *scalar material property*, using the frozen, JEPA-pretrained
`RotationInvariantCrystalEncoder` output as the crystal representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .crystal_encoder import RotationInvariantCrystalEncoder
from .losses import PropertyJEPALoss
from .predictor import PropertyPredictor
from .property_encoder import PropertyEncoder, TargetPropertyEncoder, masked_mean_pool


@dataclass
class PropertyJEPAOutput:
    loss: torch.Tensor
    loss_pred: torch.Tensor
    loss_reg: torch.Tensor
    z_pred: torch.Tensor
    z_target: torch.Tensor
    metrics: dict[str, float]


class PropertyJEPA(nn.Module):
    """Predicts a queried scalar material property from crystal structure + known properties."""

    def __init__(
        self,
        crystal_encoder: RotationInvariantCrystalEncoder,
        property_encoder: PropertyEncoder,
        target_property_encoder: TargetPropertyEncoder,
        predictor: PropertyPredictor,
        pred_loss: nn.Module | None = None,
        regularizer: nn.Module | None = None,
        reg_weight: float = 0.0,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        if crystal_encoder.hidden_dim != property_encoder.hidden_dim:
            raise ValueError(
                "crystal_encoder and property_encoder must share hidden_dim, got "
                f"{crystal_encoder.hidden_dim} vs {property_encoder.hidden_dim}"
            )
        if property_encoder.hidden_dim != target_property_encoder.hidden_dim:
            raise ValueError(
                "property_encoder and target_property_encoder must share hidden_dim, "
                f"got {property_encoder.hidden_dim} vs {target_property_encoder.hidden_dim}"
            )
        self.crystal_encoder = crystal_encoder
        self.property_encoder = property_encoder
        self.target_property_encoder = target_property_encoder
        self.predictor = predictor
        self.pred_loss = PropertyJEPALoss() if pred_loss is None else pred_loss
        self.regularizer = regularizer
        self.reg_weight = reg_weight
        self.ema_decay = ema_decay
        for parameter in self.target_property_encoder.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _tensor(batch: dict, key: str) -> torch.Tensor:
        value = batch[key]
        if not torch.is_tensor(value):
            raise ValueError(f"batch[{key!r}] must be a tensor")
        return value

    def forward(self, batch: dict[str, torch.Tensor]) -> dict:
        """Run one JEPA prediction step.

        Expected `batch` keys:
            atom_tokens: `[B, N, 103]` (from `crystal_encoder.build_atom_tokens`, padded).
            atom_mask: `[B, N]` boolean.
            lattice_features: `[B, 12]`.
            known_property_type_id: `[B, K]` long, `K` = max known properties in
                this batch (padded).
            known_property_value: `[B, K]` float, normalized values.
            known_property_mask: `[B, K]` boolean, True where that slot is a
                real known property (not padding).
            query_property_type_id: `[B]` long, the property being predicted.
            query_property_value: `[B]` float, ground-truth normalized value
                for the queried property (used only to build the EMA target;
                never seen by the predictor).
        """
        atom_tokens = self._tensor(batch, "atom_tokens")
        atom_mask = self._tensor(batch, "atom_mask")
        lattice_features = self._tensor(batch, "lattice_features")
        known_type = self._tensor(batch, "known_property_type_id")
        known_value = self._tensor(batch, "known_property_value")
        known_mask = self._tensor(batch, "known_property_mask")
        query_type = self._tensor(batch, "query_property_type_id")
        query_value = self._tensor(batch, "query_property_value")

        z_crystal = self.crystal_encoder(atom_tokens, atom_mask, lattice_features)

        b, k = known_type.shape
        if k > 0:
            known_embed = self.property_encoder(known_type, known_value)
            known_context = masked_mean_pool(known_embed, known_mask)
        else:
            known_context = z_crystal.new_zeros(b, self.property_encoder.hidden_dim)

        z_pred = self.predictor(z_crystal, known_context, query_type)
        with torch.no_grad():
            z_target = self.target_property_encoder(query_type, query_value)

        loss_pred = self.pred_loss(z_pred, z_target.detach())
        loss_reg = z_pred.new_tensor(0.0)
        reg_metrics: dict[str, float] = {}
        if self.regularizer is not None and self.reg_weight > 0:
            reg_out = self.regularizer(z_pred)
            if isinstance(reg_out, dict):
                loss_reg = reg_out["loss"]
                reg_metrics = {
                    f"reg_{key}": float(value.detach().cpu())
                    for key, value in reg_out.items()
                    if torch.is_tensor(value) and value.dim() == 0
                }
            else:
                loss_reg = reg_out
        loss = loss_pred + self.reg_weight * loss_reg
        if not torch.isfinite(loss):
            raise ValueError("PropertyJEPA loss is NaN or infinite")

        return {
            "loss": loss,
            "loss_pred": loss_pred,
            "loss_reg": loss_reg,
            "z_pred": z_pred,
            "z_target": z_target,
            "metrics": {
                "loss": float(loss.detach().cpu()),
                "loss_pred": float(loss_pred.detach().cpu()),
                "loss_reg": float(loss_reg.detach().cpu()),
                **reg_metrics,
            },
        }

    @torch.no_grad()
    def update_ema(self, decay: float | None = None) -> None:
        """Update `target_property_encoder` towards `property_encoder` by EMA.

        Mirrors `CrystalJEPA.update_ema`
        (`models/crystal_structure_jepa/JEPA/crystal_jepa.py`): the whole
        state dict is EMA-blended (floating point) or copied (non-float
        buffers), rather than special-casing individual submodules, since
        `PropertyEncoder` / `TargetPropertyEncoder` share an identical
        structure end to end (unlike stage 1's context/target encoders,
        which differ by the extra `family_proj` branch).
        """
        decay = self.ema_decay if decay is None else decay
        target_state = self.target_property_encoder.state_dict()
        source_state = self.property_encoder.state_dict()
        for name, target_value in target_state.items():
            source_value = source_state[name]
            if target_value.shape != source_value.shape:
                raise ValueError(
                    f"EMA shape mismatch for {name}: "
                    f"{tuple(target_value.shape)} vs {tuple(source_value.shape)}"
                )
            if torch.is_floating_point(target_value):
                target_value.mul_(decay).add_(source_value, alpha=1.0 - decay)
            else:
                target_value.copy_(source_value)
