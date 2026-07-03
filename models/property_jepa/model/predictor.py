"""MLP predictor bridging the crystal-embedding space to the property-latent space.

Mirrors stage 1's `MaskConditionedPredictor` (predicts the target-branch
embedding of a masked spatial block from the context embedding + a
description of *which* block was masked) but conditions on *which property
type is being queried* instead of *which spatial box was masked*, and
additionally conditions on a pooled embedding of whatever other properties
of the same material are already known (see `PropertyEncoder`,
`masked_mean_pool`) -- e.g. predicting Tc can use a known formation energy
or band gap as auxiliary context, which the crystal embedding alone does not
carry.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.property_schema import NUM_PROPERTY_TYPES
from .layers import MLP


class PropertyPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_property_types: int = NUM_PROPERTY_TYPES,
        layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_property_types = num_property_types
        self.query_type_embedding = nn.Embedding(num_property_types, hidden_dim)
        self.predictor = MLP(hidden_dim, hidden_dim, hidden_dim, layers=layers)
        nn.init.trunc_normal_(self.query_type_embedding.weight, std=0.02)

    def forward(
        self,
        z_crystal: torch.Tensor,
        known_property_context: torch.Tensor,
        query_property_type_id: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the target-space embedding of the queried property.

        Args:
            z_crystal: `[B, hidden_dim]` crystal embedding (from
                `RotationInvariantCrystalEncoder`).
            known_property_context: `[B, hidden_dim]` pooled embedding of the
                material's other known properties (from `masked_mean_pool`
                over online `PropertyEncoder` outputs; an all-zero vector
                for materials with no other known property).
            query_property_type_id: `[B]` long tensor, the canonical property
                type (see `data/property_schema.py`) being predicted for
                each sample.

        Returns:
            Tensor `[B, hidden_dim]`: predicted embedding, to be compared
            (via a JEPA regression loss) against
            `TargetPropertyEncoder(query_property_type_id, ground_truth_value)`.
        """
        if z_crystal.dim() != 2 or z_crystal.size(-1) != self.hidden_dim:
            raise ValueError(
                f"z_crystal must be [B, {self.hidden_dim}], got {tuple(z_crystal.shape)}"
            )
        if known_property_context.shape != z_crystal.shape:
            raise ValueError(
                f"known_property_context must match z_crystal shape "
                f"{tuple(z_crystal.shape)}, got {tuple(known_property_context.shape)}"
            )
        if query_property_type_id.shape != (z_crystal.size(0),):
            raise ValueError(
                f"query_property_type_id must be [B], got "
                f"{tuple(query_property_type_id.shape)}"
            )
        query_embed = self.query_type_embedding(query_property_type_id.long())
        combined = z_crystal + known_property_context + query_embed
        return self.predictor(combined)
