"""Top-level `foundation_jepa` model: wires encoder + predictor + augmentation
+ energy-aware loss together, porting the official
`components/jepa/frame/jepa.py::JEPA` end-to-end.

Context/target construction (faithful to the official `JEPA.forward` /
`JEPA.aug_batch`):
  * TARGET branch: the crystal's OWN (unaugmented) frac_coords + the RAW
    (polar-decomposition-symmetrized) lattice matrix -> scaled by
    `num_atoms ** (1/3)` -> z-scored by the (train-fit, frozen)
    `MatrixMeanStdScaler` -> encoded once.
  * CONTEXT branch: frac_coords translated by a random per-sample vector,
    the RAW lattice matrix rotated by a random per-sample SO(3) matrix
    FIRST, THEN put through the identical scale-by-num_atoms + z-score
    pipeline (order matters: rotating an already-z-scored 6D vector is not
    a physical rotation) -> encoded, then the augmentation parameters
    (translation vector + 3 quaternion draws, 6D total) are embedded and
    ADDED to the context embedding before the predictor MLP -- i.e. the
    predictor is explicitly told "the view you're seeing was perturbed by
    exactly this much" and must undo that to match the target.

Both branches share the SAME encoder weights (no EMA target branch here,
matching the official code -- unlike `crystal_structure_jepa`'s
`CrystalJEPA`, which does use an EMA target encoder).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .augmentation import rotate_lattice, translate_dense
from .backbone import MLP
from .encoder import ATOM_TOKEN_DIM, FoundationCrystalEncoder, build_atom_tokens
from .losses import FoundationJEPALoss, FoundationJEPALossOutput
from ..data.lattice import MatrixMeanStdScaler, matrix_to_triu6


@dataclass
class FoundationJEPAOutput:
    loss: torch.Tensor
    loss_infonce: torch.Tensor
    loss_reg: torch.Tensor
    context_embedding: torch.Tensor
    target_embedding: torch.Tensor


class FoundationJEPA(nn.Module):
    """Energy-aware self-supervised crystal foundation model (Crys-JEPA port).

    Args:
        hidden_dim: Embedding width (paper default 512; a smaller value is
            used for the compute-bounded local pretraining run -- see
            `training/train_foundation_jepa.py`).
        layers: Transformer depth (paper default 8).
        attn_heads: Attention heads (paper default 16; must divide `hidden_dim`).
        dropout: Dropout probability (paper default 0.0).
        max_atoms: Max atoms per crystal covered by the positional embedding table.
        temperature: InfoNCE temperature (paper default 0.1).
        reg_weight: Weight on the anti-collapse regularizer relative to InfoNCE.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        layers: int = 8,
        attn_heads: int = 16,
        dropout: float = 0.0,
        max_atoms: int = 500,
        temperature: float = 0.1,
        reg_weight: float = 0.01,
        matrix_scaler: MatrixMeanStdScaler | None = None,
    ):
        super().__init__()
        self.encoder = FoundationCrystalEncoder(
            hidden_dim=hidden_dim,
            layers=layers,
            attn_heads=attn_heads,
            dropout=dropout,
            max_atoms=max_atoms,
        )
        self.predictor = MLP(hidden_dim, hidden_dim, hidden_dim)
        # 6D augmentation conditioning: 3D translation vector + 3 quaternion draws.
        self.cond_emb = MLP(6, hidden_dim, hidden_dim)
        self.loss_fn = FoundationJEPALoss(temperature=temperature, reg_weight=reg_weight)
        self.hidden_dim = hidden_dim
        # Fit on the TRAIN split only (scripts/../training/train_foundation_jepa.py);
        # stored on the model so it travels with the checkpoint and is available
        # unchanged at fine-tune/inference time.
        self.matrix_scaler = matrix_scaler

    def _scaled_lattice_triu6(self, raw_lattice_matrix: torch.Tensor, num_atoms: torch.Tensor) -> torch.Tensor:
        """RAW (symmetrized) `[B,3,3]` lattice -> scaled+z-scored `[B,6]` descriptor.

        Port of official `add_scaled_matrix` (`matrix / num_atoms**(1/3)`)
        composed with `matrix_scaler.transform`, applied to the triu6
        flattening of the (possibly just-rotated) matrix.
        """
        if self.matrix_scaler is None:
            raise RuntimeError(
                "matrix_scaler must be set (fit on the train split) before calling "
                "the model -- see training/train_foundation_jepa.py"
            )
        b = raw_lattice_matrix.shape[0]
        triu6 = torch.stack(
            [matrix_to_triu6(raw_lattice_matrix[i]) for i in range(b)], dim=0
        )
        triu6_scaled = triu6 / num_atoms.reshape(-1, 1).to(triu6.dtype) ** (1 / 3)
        return self.matrix_scaler.transform(triu6_scaled)

    def encode(
        self,
        frac_coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        raw_lattice_matrix: torch.Tensor,
        num_atoms: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of crystals (no augmentation) into pooled CLS embeddings.

        This is the method downstream fine-tuning heads (lambda/omega_log)
        call at inference/fine-tune time -- there is no JEPA
        context/augmentation machinery involved in ordinary encoding, only
        during self-supervised pretraining (`forward`).

        Args:
            frac_coords: `[B, N, 3]` (padded).
            atomic_numbers: `[B, N]` 1-indexed, in `[1, 100]` (padded).
            raw_lattice_matrix: `[B, 3, 3]` symmetrized (polar-decomposition)
                lattice matrix, in PHYSICAL (unscaled) units.
            num_atoms: `[B]` atom count per crystal (for the `num_atoms**(1/3)`
                size normalization).
            atom_mask: `[B, N]` boolean, True = real atom.

        Returns:
            `[B, hidden_dim]`.
        """
        lattice_triu6 = self._scaled_lattice_triu6(raw_lattice_matrix, num_atoms)
        tokens = build_atom_tokens(frac_coords, atomic_numbers, lattice_triu6)
        return self.encoder(tokens, atom_mask)

    def forward(
        self,
        frac_coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        raw_lattice_matrix: torch.Tensor,
        num_atoms: torch.Tensor,
        atom_mask: torch.Tensor,
        formation_energy_peratom: torch.Tensor,
    ) -> FoundationJEPAOutput:
        """One JEPA pretraining step: build context/target views, encode both, compute loss.

        Args:
            frac_coords: `[B, N, 3]` (padded), the crystal's own coordinates.
            atomic_numbers: `[B, N]` (padded).
            raw_lattice_matrix: `[B, 3, 3]`, PHYSICAL (unscaled) symmetrized
                lattice matrix.
            num_atoms: `[B]`.
            atom_mask: `[B, N]` boolean.
            formation_energy_peratom: `[B]`, feeds the energy-aware loss weight.
        """
        # --- target branch (no augmentation) ---
        target_emb = self.encode(
            frac_coords, atomic_numbers, raw_lattice_matrix, num_atoms, atom_mask
        )

        # --- context branch (translate frac_coords + rotate RAW lattice, then scale) ---
        frac_aug, translation_vec = translate_dense(frac_coords, atom_mask)
        lattice_aug_raw, quat_params = rotate_lattice(raw_lattice_matrix)
        context_raw = self.encode(
            frac_aug, atomic_numbers, lattice_aug_raw, num_atoms, atom_mask
        )

        aug_params = torch.cat([translation_vec, quat_params], dim=-1)  # [B, 6]
        cond = self.cond_emb(aug_params)
        context_emb = self.predictor(context_raw + cond)

        loss_out: FoundationJEPALossOutput = self.loss_fn(
            context_emb, target_emb, formation_energy_peratom
        )
        return FoundationJEPAOutput(
            loss=loss_out.loss,
            loss_infonce=loss_out.loss_infonce,
            loss_reg=loss_out.loss_reg,
            context_embedding=context_emb.detach(),
            target_embedding=target_emb.detach(),
        )


__all__ = ["FoundationJEPA", "FoundationJEPAOutput", "ATOM_TOKEN_DIM"]
