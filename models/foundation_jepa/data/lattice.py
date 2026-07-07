"""Lattice-matrix preprocessing for `foundation_jepa`, porting the official
Crys-JEPA pipeline (`utils/crys_utils.py::compute_lattice_polar_decomposition`,
`components/jepa/dataloader.py::add_scaled_matrix`,
`utils/utils.py::Scaler_mean_std`).

Design note -- why this differs from `property_jepa.data.lattice_features`
---------------------------------------------------------------------------
`property_jepa`'s `lattice_invariant_features` is EXACTLY invariant under
rotation (metric tensor `G = L @ L.T`), by construction: rotating the input
crystal changes nothing about the feature the network sees. That is the
right choice for a supervised property head that should simply never be
confused by orientation.

Here the goal is different: `foundation_jepa`'s self-supervised objective
(faithfully reproducing the official paper) is precisely to TEACH rotation
invariance via the JEPA context/target task -- the predictor must recover
the target (unrotated) embedding from the context (randomly rotated)
embedding plus the explicit rotation parameters it was given. For that task
to be non-trivial, the lattice representation fed to the encoder MUST
actually change when the crystal is rotated -- an exactly-invariant
descriptor would make the "augmentation" a no-op and the predictor's job
trivial. So we port the official scheme instead:

1. `polar_decomposition_symmetrize`: canonicalizes away the arbitrary
   orientation pymatgen happens to hand back for a given CIF (reflections /
   coordinate-frame conventions differ across data sources), producing a
   symmetric 3x3 matrix that depends only on the crystal's own geometry, not
   on which orthonormal frame the parser chose to express it in.
2. That symmetric matrix's upper triangle (6 values) is stored, scaled by
   `num_atoms ** (1/3)` (a simple size-normalization, matching the official
   code) and z-scored with `MatrixMeanStdScaler` fit on the TRAIN split only.
3. At training time (see `model/augmentation.py`), a genuinely random SO(3)
   rotation is applied to the (reconstructed) 3x3 matrix to build the
   "context" view fed to the encoder -- so the encoder's input tensor really
   does change with orientation, and the predictor is given the rotation
   parameters to compensate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TRIU_ROWS, TRIU_COLS = torch.triu_indices(3, 3)


def polar_decomposition_symmetrize(lattice_matrix: torch.Tensor) -> torch.Tensor:
    """Canonicalize a (3, 3) lattice matrix via polar decomposition.

    Port of the official `compute_lattice_polar_decomposition`: computes
    `L = W @ diag(S) @ V^T` (SVD), then the symmetric factor
    `P = V @ diag(S) @ V^T` conjugated back by `U = W @ V^T`,
    `P' = U @ P @ U^T`. `P'` depends only on the crystal's intrinsic shape
    (singular values of `L`), not on the arbitrary orthonormal frame in
    which the CIF happened to express it.

    Args:
        lattice_matrix: `(3, 3)` lattice matrix (rows = lattice vectors).

    Returns:
        `(3, 3)` symmetric matrix with entries below `1e-5` in magnitude
        zeroed out (matches official numerical cleanup).
    """
    w, s, v_t = torch.linalg.svd(lattice_matrix)
    v = v_t.transpose(0, 1)
    u = w @ v_t
    p = v @ torch.diag_embed(s) @ v_t
    p_prime = u @ p @ u.transpose(0, 1)
    p_prime = torch.where(p_prime.abs() < 1e-5, torch.zeros_like(p_prime), p_prime)
    return p_prime


def matrix_to_triu6(matrix: torch.Tensor) -> torch.Tensor:
    """Flatten a symmetric `(3, 3)` matrix to its 6 upper-triangular entries."""
    return matrix[TRIU_ROWS, TRIU_COLS]


def triu6_to_matrix(vector: torch.Tensor) -> torch.Tensor:
    """Reconstruct a symmetric `(3, 3)` matrix from its 6 upper-triangular entries."""
    a = torch.zeros(3, 3, dtype=vector.dtype, device=vector.device)
    a[0, 0] = vector[0]
    a[0, 1] = a[1, 0] = vector[1]
    a[0, 2] = a[2, 0] = vector[2]
    a[1, 1] = vector[3]
    a[1, 2] = a[2, 1] = vector[4]
    a[2, 2] = vector[5]
    return a


def batched_triu6_to_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """Batched version of `triu6_to_matrix`. `vectors`: `(B, 6)` -> `(B, 3, 3)`."""
    b = vectors.shape[0]
    a = torch.zeros(b, 3, 3, dtype=vectors.dtype, device=vectors.device)
    a[:, 0, 0] = vectors[:, 0]
    a[:, 0, 1] = a[:, 1, 0] = vectors[:, 1]
    a[:, 0, 2] = a[:, 2, 0] = vectors[:, 2]
    a[:, 1, 1] = vectors[:, 3]
    a[:, 1, 2] = a[:, 2, 1] = vectors[:, 4]
    a[:, 2, 2] = vectors[:, 5]
    return a


@dataclass
class MatrixMeanStdScaler:
    """Per-feature mean/std scaler for the 6D lattice descriptor, fit on train only.

    Port of the official `Scaler_mean_std`, restricted to what this repo
    needs (no `.pack()`/`.load()` checkpoint round-trip -- mean/std are saved
    directly as plain floats in the model checkpoint dict instead).
    """

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, matrices_6d: torch.Tensor) -> "MatrixMeanStdScaler":
        mean = matrices_6d.mean(dim=0, keepdim=True)
        std = matrices_6d.std(dim=0, keepdim=True)
        std = torch.where(std > 1e-8, std, torch.ones_like(std))
        return cls(mean=mean, std=std)

    def transform(self, matrices_6d: torch.Tensor) -> torch.Tensor:
        return (matrices_6d - self.mean.to(matrices_6d.device)) / self.std.to(matrices_6d.device)

    def inverse_transform(self, matrices_6d: torch.Tensor) -> torch.Tensor:
        return matrices_6d * self.std.to(matrices_6d.device) + self.mean.to(matrices_6d.device)


__all__ = [
    "polar_decomposition_symmetrize",
    "matrix_to_triu6",
    "triu6_to_matrix",
    "batched_triu6_to_matrix",
    "MatrixMeanStdScaler",
]
