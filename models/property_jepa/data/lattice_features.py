"""Rotation-invariant featurisation of the crystal lattice (unit-cell) tensor.

This module is specific to `property_jepa` (stage 2). Unlike stage 1
(`models/crystal_structure_jepa`), which only ever consumes fractional atomic
coordinates (already rotation-invariant, since they are expressed in the
lattice's own basis) and discards the lattice matrix entirely, property-JEPA
needs the lattice as an explicit input: unit-cell shape and volume carry
information relevant to several target properties (elastic moduli, magnetic
ordering distance, Tc via pressure/density proxies, etc.).

A crystal lattice is conventionally stored as a (3, 3) matrix of row vectors
`L = [a; b; c]` in a fixed Cartesian frame. Applying a rigid rotation `R` to
the whole crystal (atoms + cell) transforms the lattice as `L' = L @ R.T`
(each row vector a, b, c is rotated by R). `L'` is numerically different from
`L`, even though the crystal is physically identical -- so `L` itself is NOT
a safe network input if the network should be invariant to how the crystal
happens to be oriented in the input file.

Two rotation-invariant featurisations are provided here:

1. `metric_tensor` / `metric_tensor_6d`: the Gram (metric) tensor
   `G = L @ L.T`. Since `G' = L' @ L'.T = L @ R.T @ R @ L.T = L @ L.T = G`
   for any orthogonal `R` (R.T @ R = I), `G` is exactly invariant under
   rotation of the crystal. `G` is symmetric positive-definite; its 6
   upper-triangular entries (`Gxx, Gxy, Gxz, Gyy, Gyz, Gzz`) are a
   non-redundant, differentiable, exactly-invariant 6D descriptor. This is
   analogous in spirit to the polar-SVD 6D lattice descriptor used in stage
   1's (currently disabled) `compute_lattice_6d`, but derived independently
   here via the metric tensor -- simpler to prove invariant and cheaper to
   compute (no SVD).

2. `reduced_cell_parameters`: the classic crystallographic scalars
   `(a, b, c, alpha, beta, gamma)` -- cell edge lengths and inter-axial
   angles. These are rotation invariant by definition (they are computed
   from dot products of the lattice vectors) and additionally match the
   conventional crystallographic description, which can help interpretability
   and downstream debugging.

Both featurisations are provided so property-JEPA can use either or
concatenate them; `metric_tensor_6d` is the recommended default (smoother,
avoids the branch-cut/degenerate-angle issues that trigonometric parameters
can have for near-degenerate cells).
"""

from __future__ import annotations

import torch


def metric_tensor(lattice_matrix: torch.Tensor) -> torch.Tensor:
    """Compute the Gram (metric) tensor `G = L @ L.T` of a lattice matrix.

    Args:
        lattice_matrix: Lattice vectors as a (3, 3) float tensor (rows = a, b, c),
            in a fixed Cartesian frame.

    Returns:
        Tensor of shape (3, 3): the symmetric positive-definite metric tensor.
        `G[i, j] = v_i . v_j` where v_0, v_1, v_2 are the lattice row vectors.
        Invariant under any rotation (or reflection) applied to the whole
        crystal, since such a transform acts as `L -> L @ R.T` with `R`
        orthogonal, and `(L @ R.T) @ (L @ R.T).T == L @ L.T`.
    """
    if lattice_matrix.shape != (3, 3):
        raise ValueError(
            f"metric_tensor expects a (3, 3) lattice matrix, got {tuple(lattice_matrix.shape)}"
        )
    return lattice_matrix @ lattice_matrix.T


def metric_tensor_6d(
    lattice_matrix: torch.Tensor,
    num_atoms: int | None = None,
) -> torch.Tensor:
    """Rotation-invariant 6D descriptor of the lattice: upper triangle of `G = L @ L.T`.

    Args:
        lattice_matrix: Lattice vectors as a (3, 3) float tensor (rows = a, b, c).
        num_atoms: If given, the descriptor is normalised by `num_atoms ** (1/3)`
            so that cell-volume scale is roughly decoupled from the number of
            atoms per cell (larger cells with more atoms are not trivially
            "bigger" in this feature). Pass `None` to skip normalisation.

    Returns:
        Tensor of shape (6,): `[Gxx, Gxy, Gxz, Gyy, Gyz, Gzz]`, i.e. the upper
        triangle (row-major) of the metric tensor, optionally normalised.
    """
    g = metric_tensor(lattice_matrix)
    tri = torch.triu_indices(3, 3)
    g6 = g[tri[0], tri[1]]
    if num_atoms is not None:
        g6 = g6 / (float(num_atoms) ** (1.0 / 3.0))
    return g6


def reduced_cell_parameters(lattice_matrix: torch.Tensor) -> torch.Tensor:
    """Rotation-invariant crystallographic cell parameters `(a, b, c, alpha, beta, gamma)`.

    Args:
        lattice_matrix: Lattice vectors as a (3, 3) float tensor (rows = a, b, c).

    Returns:
        Tensor of shape (6,): `[a, b, c, alpha, beta, gamma]` where `a, b, c`
        are the row-vector norms (cell edge lengths) and `alpha, beta, gamma`
        are the inter-axial angles in degrees, defined as:
          alpha = angle(b, c), beta = angle(a, c), gamma = angle(a, b).
        All six quantities are computed from dot products / norms of the
        lattice row vectors and are therefore exactly invariant under any
        rotation applied to the whole crystal.
    """
    if lattice_matrix.shape != (3, 3):
        raise ValueError(
            f"reduced_cell_parameters expects a (3, 3) lattice matrix, got "
            f"{tuple(lattice_matrix.shape)}"
        )
    a_vec, b_vec, c_vec = lattice_matrix[0], lattice_matrix[1], lattice_matrix[2]
    a = torch.linalg.norm(a_vec)
    b = torch.linalg.norm(b_vec)
    c = torch.linalg.norm(c_vec)

    def _angle_deg(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        cos_theta = torch.dot(u, v) / (torch.linalg.norm(u) * torch.linalg.norm(v))
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        return torch.rad2deg(torch.arccos(cos_theta))

    alpha = _angle_deg(b_vec, c_vec)
    beta = _angle_deg(a_vec, c_vec)
    gamma = _angle_deg(a_vec, b_vec)

    return torch.stack([a, b, c, alpha, beta, gamma])


def lattice_invariant_features(
    lattice_matrix: torch.Tensor,
    num_atoms: int | None = None,
) -> torch.Tensor:
    """Concatenate both invariant featurisations into a single 12D descriptor.

    Args:
        lattice_matrix: Lattice vectors as a (3, 3) float tensor (rows = a, b, c).
        num_atoms: Forwarded to `metric_tensor_6d` for normalisation; the
            `reduced_cell_parameters` half is never normalised by atom count
            (lengths/angles are physical scalars independent of cell content).

    Returns:
        Tensor of shape (12,): `[metric_tensor_6d (6) ‖ reduced_cell_parameters (6)]`.
    """
    return torch.cat(
        [
            metric_tensor_6d(lattice_matrix, num_atoms=num_atoms),
            reduced_cell_parameters(lattice_matrix),
        ]
    )
