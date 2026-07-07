"""Crystal encoder for `foundation_jepa`, porting the official Crys-JEPA
per-atom tokenization + CLS-token transformer encoding
(`components/jepa/frame/jepa.py::JEPA.encode`, arXiv:2605.14759).

Per-atom tokens are `[frac_coords (3) | one_hot(atomic_number, 100) |
lattice_triu6 (6)]` = 109-dim, matching the official `pre_backbone =
MLP(109, hidden_dim, hidden_dim)`. Unlike `property_jepa`'s
`RotationInvariantCrystalEncoder` (which injects the lattice as one global
12D *exactly rotation-invariant* descriptor added to the CLS token), the
official Crys-JEPA scheme BROADCASTS the (polar-decomposition-symmetrized,
scaled, z-scored) lattice descriptor onto every atom token -- and that
descriptor is NOT rotation-invariant end-to-end by itself; rotation
invariance is instead something the JEPA context/target self-supervised
task is meant to teach the encoder (see `model/augmentation.py`), not a
property baked into the input featurization by construction. This is a
faithful port of the official design, not a design choice made in this
port.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import MLP, FoundationTransformer

NUM_ATOM_CLASSES = 100
LATTICE_TRIU6_DIM = 6
ATOM_TOKEN_DIM = 3 + NUM_ATOM_CLASSES + LATTICE_TRIU6_DIM  # 109


def build_atom_tokens(
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    lattice_triu6: torch.Tensor,
) -> torch.Tensor:
    """Build per-atom tokens `[frac_coords(3) | one_hot(Z,100) | lattice_triu6(6)]`.

    Args:
        frac_coords: `(B, N, 3)` fractional coordinates (already padded).
        atomic_numbers: `(B, N)` 1-indexed atomic numbers in `[1, 100]`
            (padding positions may hold any in-range placeholder; they are
            masked out downstream and never influence the pooled output).
        lattice_triu6: `(B, 6)` per-crystal scaled+z-scored lattice
            descriptor (`data.lattice.matrix_to_triu6`, scaler-transformed),
            broadcast to every atom in that crystal.

    Returns:
        `(B, N, 109)` tensor.
    """
    if frac_coords.dim() != 3 or frac_coords.size(-1) != 3:
        raise ValueError(f"frac_coords must be [B, N, 3], got {tuple(frac_coords.shape)}")
    if atomic_numbers.shape != frac_coords.shape[:2]:
        raise ValueError(
            f"atomic_numbers must be [B, N], got {tuple(atomic_numbers.shape)}"
        )
    if lattice_triu6.shape != (frac_coords.size(0), LATTICE_TRIU6_DIM):
        raise ValueError(
            f"lattice_triu6 must be [B, {LATTICE_TRIU6_DIM}], got {tuple(lattice_triu6.shape)}"
        )
    one_hot = F.one_hot((atomic_numbers - 1).clamp(min=0).long(), num_classes=NUM_ATOM_CLASSES).float()
    n = frac_coords.size(1)
    lattice_bcast = lattice_triu6.unsqueeze(1).expand(-1, n, -1).to(frac_coords.dtype)
    return torch.cat([frac_coords.float(), one_hot, lattice_bcast], dim=-1)


class FoundationCrystalEncoder(nn.Module):
    """CLS-token transformer over `[frac_coords | one_hot(Z) | lattice_triu6]` atom tokens.

    Faithful port of the official `JEPA.encode` (pre_backbone MLP + Transformer
    + CLS pooling), operating on this repo's dense-batch/mask convention
    instead of `torch_geometric.utils.to_dense_batch`.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        layers: int = 8,
        attn_heads: int = 16,
        dropout: float = 0.0,
        max_atoms: int = 500,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        if hidden_dim % attn_heads != 0:
            raise ValueError("hidden_dim must be divisible by attn_heads")
        self.hidden_dim = hidden_dim
        self.max_atoms = max_atoms

        self.pre_backbone = MLP(ATOM_TOKEN_DIM, hidden_dim, hidden_dim)
        self.backbone = FoundationTransformer(
            hidden_dim,
            layers,
            attn_heads,
            dropout,
            max_len=max_atoms + 1,
            grad_checkpointing=grad_checkpointing,
        )

    def forward(self, atom_tokens: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_tokens: `[B, N, 109]`, from `build_atom_tokens` (padded).
            atom_mask: `[B, N]` boolean, True where the atom is real.

        Returns:
            `[B, hidden_dim]`: the pooled (CLS) crystal embedding.
        """
        if atom_tokens.dim() != 3 or atom_tokens.size(-1) != ATOM_TOKEN_DIM:
            raise ValueError(
                f"atom_tokens must be [B, N, {ATOM_TOKEN_DIM}], got {tuple(atom_tokens.shape)}"
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

        x = self.pre_backbone(atom_tokens)
        encoded = self.backbone(x, atom_mask.bool())
        return encoded[:, 0]


__all__ = [
    "NUM_ATOM_CLASSES",
    "LATTICE_TRIU6_DIM",
    "ATOM_TOKEN_DIM",
    "build_atom_tokens",
    "FoundationCrystalEncoder",
]
