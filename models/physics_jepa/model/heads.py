"""Physics fine-tuning heads: `lambda_ep` (electron-phonon coupling) and
`omega_log` (logarithmic-average phonon frequency), predicted jointly from
a `foundation_jepa` crystal embedding.

Architecture: a (typically pretrained) `FoundationJEPA` instance produces a
`[B, hidden_dim]` CLS embedding via its own `.encode(...)` method (the
documented entry point for downstream fine-tuning heads -- see
`foundation_jepa/model/jepa.py::FoundationJEPA.encode`'s docstring, which
already carries the `matrix_scaler` needed to build atom tokens, so
`PhysicsHeads` never has to touch `build_atom_tokens` or the lattice
scaler itself); `PhysicsHeads` is a small 2-layer MLP trunk shared between
the two targets, followed by two separate linear output layers -- one
scalar per target, in NORMALIZED (log1p + z-scored, see
`data/dataset.py::PhysicsTargetStats`) space. A shared trunk (rather than
two fully independent MLPs) is used because `lambda_ep` and `omega_log` are
physically coupled (both derive from the same phonon spectrum / a2F(w)),
so sharing early layers lets the head exploit that correlation while still
letting the two final linear layers specialize.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...foundation_jepa.model.backbone import MLP
from ...foundation_jepa.model.jepa import FoundationJEPA


@dataclass
class PhysicsHeadsOutput:
    lambda_pred_norm: torch.Tensor
    omega_pred_norm: torch.Tensor


class PhysicsHeads(nn.Module):
    """Shared-trunk MLP heads for `lambda_ep` and `omega_log`, on top of a
    (typically pretrained) `FoundationJEPA` embedding.

    Args:
        foundation_model: A `FoundationJEPA` instance -- pass one loaded
            from a `foundation_jepa` checkpoint (with its fitted
            `matrix_scaler` already attached) to fine-tune from pretrained
            weights, or a freshly constructed one (matching
            hyperparameters, with a `matrix_scaler` fit on the SAME train
            split) to train `physics_jepa` from scratch as an ablation
            baseline.
        hidden_dim: Trunk MLP hidden width (defaults to the encoder's own
            `hidden_dim`, keeping heads small relative to the encoder).
        freeze_encoder: If True, the encoder's parameters are frozen
            (`requires_grad_(False)`) and only the head trunk + output
            layers are trained -- linear-probe-style evaluation of the
            pretrained representation. If False, the encoder is fine-tuned
            jointly with the heads (typically at a lower learning rate,
            set by the training script's `--encoder-lr`).
    """

    def __init__(
        self,
        foundation_model: FoundationJEPA,
        hidden_dim: int | None = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.foundation_model = foundation_model
        encoder_hidden_dim = foundation_model.hidden_dim
        trunk_hidden = hidden_dim if hidden_dim is not None else encoder_hidden_dim
        self.trunk = MLP(encoder_hidden_dim, trunk_hidden, trunk_hidden)
        self.lambda_head = nn.Linear(trunk_hidden, 1)
        self.omega_head = nn.Linear(trunk_hidden, 1)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.foundation_model.encoder.parameters():
                p.requires_grad_(False)

    def encoder_parameters(self):
        return self.foundation_model.encoder.parameters()

    def head_parameters(self):
        return list(self.trunk.parameters()) + list(self.lambda_head.parameters()) + list(
            self.omega_head.parameters()
        )

    def forward(
        self,
        frac_coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        raw_lattice_matrix: torch.Tensor,
        num_atoms: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> PhysicsHeadsOutput:
        if self.freeze_encoder:
            with torch.no_grad():
                embedding = self.foundation_model.encode(
                    frac_coords, atomic_numbers, raw_lattice_matrix, num_atoms, atom_mask
                )
        else:
            embedding = self.foundation_model.encode(
                frac_coords, atomic_numbers, raw_lattice_matrix, num_atoms, atom_mask
            )
        trunk_out = self.trunk(embedding)
        lambda_pred = self.lambda_head(trunk_out).squeeze(-1)
        omega_pred = self.omega_head(trunk_out).squeeze(-1)
        return PhysicsHeadsOutput(lambda_pred_norm=lambda_pred, omega_pred_norm=omega_pred)


__all__ = ["PhysicsHeads", "PhysicsHeadsOutput"]
