"""Rotation-invariant crystal-structure encoder for property-JEPA (stage 2).

Per-atom tokens are `[one_hot(atomic_number, 100) ‖ frac_coords (3)]` -- both
rotation-invariant (fractional coordinates are expressed in the lattice's own
basis, and the one-hot species vector obviously carries no orientation
information). The unit-cell shape is injected as a *global* rotation-invariant
descriptor (see `models/property_jepa/data/lattice_features.py`,
`lattice_invariant_features`, 12D) added to the CLS token before the
transformer stack, rather than as an extra per-atom feature -- the lattice is
a property of the whole cell, not of any individual atom.

This deliberately mirrors the stage-1 `CrystalTransformerEncoder`
(`models/crystal_structure_jepa/JEPA/crystal_jepa.py`) in overall shape
(atom-token projection + CLS + learned positional embedding + pre-norm
`TransformerEncoder` + final LayerNorm, CLS token pooled as the crystal
embedding) but is a fully independent implementation: stage 2 additionally
consumes the lattice tensor (stage 1 discards it entirely and only ever sees
fractional coordinates), and stage 2 has no masking/EMA-target machinery
(there is a single encoder, not a context/target pair) since its output feeds
a supervised property predictor rather than a self-supervised JEPA context
target.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.lattice_features import lattice_invariant_features
from .layers import MLP

NUM_ATOM_CLASSES = 100
LATTICE_FEATURE_DIM = 12


def build_atom_tokens(
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    """Build rotation-invariant per-atom tokens `[one_hot(Z, 100) ‖ frac_coords (3)]`.

    Args:
        frac_coords: Fractional coordinates, shape `(N, 3)`.
        atomic_numbers: 1-indexed atomic numbers, shape `(N,)`, in `[1, 100]`.

    Returns:
        Tensor of shape `(N, 103)`.
    """
    if frac_coords.dim() != 2 or frac_coords.size(-1) != 3:
        raise ValueError(f"frac_coords must be [N, 3], got {tuple(frac_coords.shape)}")
    if atomic_numbers.shape != frac_coords.shape[:1]:
        raise ValueError(
            f"atomic_numbers must be [N], got {tuple(atomic_numbers.shape)} for "
            f"{frac_coords.shape[0]} atoms"
        )
    one_hot = F.one_hot((atomic_numbers - 1).long(), num_classes=NUM_ATOM_CLASSES).float()
    return torch.cat([one_hot, frac_coords.float()], dim=-1)


class RotationInvariantCrystalEncoder(nn.Module):
    """Transformer encoder over atom tokens + a global rotation-invariant lattice descriptor.

    Rotation invariance holds end-to-end: atom tokens are built from
    fractional coordinates and species identity (both invariant), and the
    lattice contributes only through `lattice_invariant_features` (the 12D
    metric-tensor + reduced-cell-parameter descriptor, itself proven
    invariant in `lattice_features.py`). No raw Cartesian lattice matrix or
    Cartesian atomic position ever enters the network.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        layers: int = 6,
        attn_heads: int = 8,
        dropout: float = 0.0,
        max_atoms: int = 500,
        atom_token_dim: int = NUM_ATOM_CLASSES + 3,
        lattice_feature_dim: int = LATTICE_FEATURE_DIM,
    ):
        super().__init__()
        if hidden_dim % attn_heads != 0:
            raise ValueError("hidden_dim must be divisible by attn_heads")
        self.hidden_dim = hidden_dim
        self.max_atoms = max_atoms
        self.atom_token_dim = atom_token_dim
        self.lattice_feature_dim = lattice_feature_dim

        self.atom_proj = MLP(atom_token_dim, hidden_dim, hidden_dim)
        self.lattice_proj = MLP(lattice_feature_dim, hidden_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(max_atoms + 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attn_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def _validate_inputs(
        self,
        atom_tokens: torch.Tensor,
        atom_mask: torch.Tensor,
        lattice_features: torch.Tensor,
    ) -> None:
        if atom_tokens.dim() != 3:
            raise ValueError(f"atom_tokens must be [B, N, F], got {tuple(atom_tokens.shape)}")
        if atom_tokens.size(-1) != self.atom_token_dim:
            raise ValueError(
                f"expected atom token dim {self.atom_token_dim}, got {atom_tokens.size(-1)}"
            )
        if atom_mask.shape != atom_tokens.shape[:2]:
            raise ValueError(
                f"atom_mask shape {tuple(atom_mask.shape)} must match token shape "
                f"{tuple(atom_tokens.shape[:2])}"
            )
        if not atom_mask.any(dim=1).all():
            raise ValueError("each crystal must contain at least one visible atom")
        if atom_tokens.size(1) > self.max_atoms:
            raise ValueError(
                f"sequence length {atom_tokens.size(1)} exceeds max_atoms {self.max_atoms}"
            )
        if lattice_features.shape != (atom_tokens.size(0), self.lattice_feature_dim):
            raise ValueError(
                f"lattice_features must be [B, {self.lattice_feature_dim}], got "
                f"{tuple(lattice_features.shape)}"
            )

    def forward(
        self,
        atom_tokens: torch.Tensor,
        atom_mask: torch.Tensor,
        lattice_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of crystals into a single embedding vector per crystal.

        Args:
            atom_tokens: `[B, N, 103]`, from `build_atom_tokens` (padded).
            atom_mask: `[B, N]` boolean, True where the atom is real (not padding).
            lattice_features: `[B, 12]`, from `lattice_invariant_features`.

        Returns:
            Tensor `[B, hidden_dim]`: the pooled (CLS) crystal embedding.
        """
        self._validate_inputs(atom_tokens, atom_mask, lattice_features)
        b, n, _ = atom_tokens.shape

        tokens = self.atom_proj(atom_tokens)
        cls = self.cls_token.expand(b, 1, -1) + self.lattice_proj(lattice_features).unsqueeze(1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embedding[: n + 1].unsqueeze(0)

        cls_mask = torch.ones(b, 1, dtype=torch.bool, device=atom_mask.device)
        full_mask = torch.cat([cls_mask, atom_mask.bool()], dim=1)

        encoded = self.transformer(tokens, src_key_padding_mask=~full_mask)
        encoded = self.norm(encoded)
        return encoded[:, 0]


__all__ = [
    "NUM_ATOM_CLASSES",
    "LATTICE_FEATURE_DIM",
    "build_atom_tokens",
    "RotationInvariantCrystalEncoder",
    "lattice_invariant_features",
]
