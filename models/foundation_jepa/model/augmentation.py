"""Rotation/translation augmentation for `foundation_jepa`'s JEPA context view.

Faithful port of the official `components/jepa/augmentation.py` (`translate`,
`_random_rotation_matrix_so3`, `rotate`), adapted from the ragged
`(N_total, F)` + `batch`-index representation (`torch_geometric`-style) to
this repo's dense `(B, N_max, F)` + boolean-mask representation. The maths
(uniform SO(3) sampling via random unit quaternions, per-sample uniform
fractional-coordinate shift) are unchanged; only the batching mechanics
differ.
"""

from __future__ import annotations

import torch


def translate_dense(frac_coords: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an independent random fractional-coordinate shift per sample.

    Args:
        frac_coords: `(B, N, 3)` fractional coordinates.
        mask: `(B, N)` boolean atom mask (True = real atom).

    Returns:
        `(frac_translated, translation_vec)`: `(B, N, 3)` (padding positions
        left at 0, harmless since they are masked out downstream) and
        `(B, 3)` the per-sample translation vector used.
    """
    b = frac_coords.shape[0]
    vec_t = torch.rand(b, 3, device=frac_coords.device, dtype=frac_coords.dtype)
    frac_translated = (frac_coords + vec_t.unsqueeze(1)) % 1.0
    frac_translated = frac_translated * mask.unsqueeze(-1)
    return frac_translated, vec_t


def random_rotation_matrix_so3(batch: int, device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform random SO(3) rotation matrices via random unit quaternions.

    Port of the official `_random_rotation_matrix_so3`.

    Returns:
        `(R, quat_params)`: `R` is `(batch, 3, 3)`; `quat_params` is
        `(batch, 3)` the three uniform random draws `(u1, u2, u3)` used to
        build the quaternion (fed to the predictor's conditioning MLP,
        matching the official 3-element `vec_r`).
    """
    u1 = torch.rand(batch, device=device, dtype=dtype)
    u2 = torch.rand(batch, device=device, dtype=dtype)
    u3 = torch.rand(batch, device=device, dtype=dtype)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)  # w

    x, y, z, w = q1, q2, q3, q4
    r = torch.empty((batch, 3, 3), device=device, dtype=dtype)

    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)

    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)

    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return r, torch.stack([u1, u2, u3], dim=-1)


def rotate_lattice(lattice: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an independent random SO(3) rotation to each sample's lattice matrix.

    Args:
        lattice: `(B, 3, 3)` lattice matrices.

    Returns:
        `(lattice_rotated, quat_params)`: `(B, 3, 3)` and `(B, 3)`.
    """
    b = lattice.shape[0]
    r, quat_params = random_rotation_matrix_so3(b, device=lattice.device, dtype=lattice.dtype)
    lattice_rotated = lattice @ r.transpose(-1, -2)
    return lattice_rotated, quat_params


__all__ = ["translate_dense", "random_rotation_matrix_so3", "rotate_lattice"]
