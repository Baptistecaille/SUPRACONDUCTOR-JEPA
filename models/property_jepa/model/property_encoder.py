"""Property encoder: embeds a (property type, scalar value) pair into a latent space.

Property-JEPA treats each scalar material property (formation energy, band
gap, magnetic moment, Tc, ...) not as a raw regression target but as
something to be represented in a learned latent space -- exactly as I-JEPA /
Crys-JEPA represent masked *spatial* content in latent space rather than
predicting raw pixel/atom values. `PropertyEncoder` is the module that maps a
`(property_type_id, normalized_value)` pair to that latent space; it is used
twice, in the same asymmetric online/EMA-target pattern as stage 1's
`CrystalTransformerEncoder` / `TargetCrystalTransformerEncoder`
(`models/crystal_structure_jepa/JEPA/crystal_jepa.py`):

- The online `PropertyEncoder` embeds *known* properties of a material (e.g.
  "this row also has a measured band gap of 1.2 eV") so the predictor can
  condition on them -- it receives gradient during training.
- `TargetPropertyEncoder` is a frozen, EMA-updated copy used to embed the
  *queried* (masked-out, to-be-predicted) property's ground-truth value; it
  supplies the JEPA prediction target and never receives a direct gradient
  (`update_ema` in `models/property_jepa/model/jepa.py` copies the online
  encoder's weights into it, mirroring `CrystalJEPA.update_ema`).

Both instances share the same `property_type` vocabulary
(`models/property_jepa/data/property_schema.py`, currently 14 canonical
property types); an unknown/absent value is represented by a boolean mask at
the call site, never by a magic numeric sentinel.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.property_schema import NUM_PROPERTY_TYPES
from .layers import MLP


class PropertyEncoder(nn.Module):
    """Embed `(property_type_id, value)` pairs into a `hidden_dim` latent space."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_property_types: int = NUM_PROPERTY_TYPES,
        layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_property_types = num_property_types
        self.type_embedding = nn.Embedding(num_property_types, hidden_dim)
        self.value_proj = MLP(1, hidden_dim, hidden_dim, layers=layers)
        self.refine = MLP(hidden_dim, hidden_dim, hidden_dim, layers=layers)
        nn.init.trunc_normal_(self.type_embedding.weight, std=0.02)

    def _validate_inputs(
        self, property_type_id: torch.Tensor, value: torch.Tensor
    ) -> None:
        if property_type_id.shape != value.shape:
            raise ValueError(
                f"property_type_id {tuple(property_type_id.shape)} and value "
                f"{tuple(value.shape)} must have the same shape"
            )
        if property_type_id.numel() > 0:
            if int(property_type_id.min()) < 0 or int(
                property_type_id.max()
            ) >= self.num_property_types:
                raise ValueError(
                    f"property_type_id must be in [0, {self.num_property_types}), "
                    f"got range [{int(property_type_id.min())}, "
                    f"{int(property_type_id.max())}]"
                )

    def forward(
        self, property_type_id: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Encode a batch (of arbitrary leading shape) of property observations.

        Args:
            property_type_id: Long tensor of any shape `S`, values in
                `[0, num_property_types)`.
            value: Float tensor of shape `S`, matching `property_type_id`
                (normalized scalar property values; normalization is a
                data-loading concern, e.g. per-property z-score, not handled
                here).

        Returns:
            Tensor of shape `S + (hidden_dim,)`.
        """
        self._validate_inputs(property_type_id, value)
        type_embed = self.type_embedding(property_type_id.long())
        value_embed = self.value_proj(value.float().unsqueeze(-1))
        return self.refine(type_embed + value_embed)


class TargetPropertyEncoder(PropertyEncoder):
    """Frozen, EMA-updated counterpart of `PropertyEncoder` (JEPA prediction target)."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_property_types: int = NUM_PROPERTY_TYPES,
        layers: int = 2,
    ):
        super().__init__(
            hidden_dim=hidden_dim, num_property_types=num_property_types, layers=layers
        )
        for parameter in self.parameters():
            parameter.requires_grad = False


def masked_mean_pool(
    embeddings: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool `[B, K, H]` embeddings over the `K` axis, respecting a `[B, K]` mask.

    Rows with zero valid (`mask=True`) slots return an all-zero vector
    (there is simply no known-property context to condition on for that
    sample) instead of raising or producing NaNs.
    """
    if embeddings.dim() != 3:
        raise ValueError(f"embeddings must be [B, K, H], got {tuple(embeddings.shape)}")
    if mask.shape != embeddings.shape[:2]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match {tuple(embeddings.shape[:2])}"
        )
    mask_f = mask.to(dtype=embeddings.dtype).unsqueeze(-1)
    summed = (embeddings * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1.0)
    return summed / counts
