import torch

from models.foundation_jepa.model.augmentation import (
    random_rotation_matrix_so3,
    rotate_lattice,
    translate_dense,
)


def test_translate_dense_masks_padding():
    torch.manual_seed(0)
    frac_coords = torch.rand(2, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[0, 3:] = False

    translated, vec_t = translate_dense(frac_coords, mask)
    assert translated.shape == frac_coords.shape
    assert vec_t.shape == (2, 3)
    assert torch.all(translated[~mask] == 0)


def test_translate_dense_wraps_to_unit_cell():
    torch.manual_seed(1)
    frac_coords = torch.rand(4, 6, 3)
    mask = torch.ones(4, 6, dtype=torch.bool)
    translated, _ = translate_dense(frac_coords, mask)
    assert (translated >= 0).all()
    assert (translated < 1.0).all()


def test_random_rotation_matrix_is_orthogonal_and_proper():
    torch.manual_seed(2)
    r, quat_params = random_rotation_matrix_so3(8)
    assert r.shape == (8, 3, 3)
    assert quat_params.shape == (8, 3)

    identity = torch.eye(3).unsqueeze(0).expand(8, -1, -1)
    assert torch.allclose(r @ r.transpose(-1, -2), identity, atol=1e-4)
    dets = torch.linalg.det(r)
    assert torch.allclose(dets, torch.ones(8), atol=1e-3)


def test_rotate_lattice_preserves_gram_matrix_singular_values():
    """Rotating a lattice matrix by an orthogonal R should not change its
    singular values (R is an isometry)."""
    torch.manual_seed(3)
    lattice = torch.eye(3).unsqueeze(0).expand(4, -1, -1).clone() + 0.1 * torch.randn(4, 3, 3)
    rotated, quat_params = rotate_lattice(lattice)
    assert rotated.shape == lattice.shape
    assert quat_params.shape == (4, 3)

    s_before = torch.linalg.svdvals(lattice)
    s_after = torch.linalg.svdvals(rotated)
    assert torch.allclose(s_before.sort().values, s_after.sort().values, atol=1e-3)


def test_rotations_are_independent_per_sample():
    torch.manual_seed(4)
    r, _ = random_rotation_matrix_so3(2)
    assert not torch.allclose(r[0], r[1])
