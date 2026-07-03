"""Rotation-invariance regression tests for property_jepa's lattice featurisation.

These tests are the primary safeguard that property-JEPA's lattice input is
mathematically invariant to how the crystal is oriented in Cartesian space --
this module is intentionally independent of stage 1
(`models.crystal_structure_jepa`); it re-derives invariance from first
principles (the metric tensor and crystallographic cell parameters) rather
than reusing stage 1's disabled SVD-based descriptor.
"""

from __future__ import annotations

import pytest
import torch

from models.property_jepa.data.lattice_features import (
    lattice_invariant_features,
    metric_tensor,
    metric_tensor_6d,
    reduced_cell_parameters,
)


def _random_rotation_matrix(generator: torch.Generator) -> torch.Tensor:
    """Sample a uniformly random rotation matrix in SO(3) via QR decomposition."""
    random_matrix = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(random_matrix)
    # Fix signs so det(Q) == +1 (a proper rotation, not a reflection).
    d = torch.diagonal(r)
    q = q * torch.sign(d).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


@pytest.fixture
def rng() -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(0)
    return generator


@pytest.fixture
def sample_lattice() -> torch.Tensor:
    # A generic (non-cubic, non-degenerate) triclinic-like lattice matrix.
    return torch.tensor(
        [
            [4.1, 0.3, 0.2],
            [0.1, 3.7, -0.4],
            [-0.2, 0.5, 5.2],
        ],
        dtype=torch.float64,
    )


def test_metric_tensor_shape_and_symmetry(sample_lattice: torch.Tensor) -> None:
    g = metric_tensor(sample_lattice)
    assert g.shape == (3, 3)
    assert torch.allclose(g, g.T, atol=1e-10)
    # Positive-definite: all eigenvalues > 0 for a non-degenerate lattice.
    eigvals = torch.linalg.eigvalsh(g)
    assert (eigvals > 0).all()


def test_metric_tensor_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        metric_tensor(torch.eye(2))


@pytest.mark.parametrize("trial", range(20))
def test_metric_tensor_6d_invariant_to_random_rotation(
    trial: int, sample_lattice: torch.Tensor, rng: torch.Generator
) -> None:
    baseline = metric_tensor_6d(sample_lattice, num_atoms=8)
    rotation = _random_rotation_matrix(rng)
    rotated_lattice = sample_lattice @ rotation.T
    rotated = metric_tensor_6d(rotated_lattice, num_atoms=8)
    assert torch.allclose(baseline, rotated, atol=1e-8), (
        f"trial {trial}: metric_tensor_6d changed under rotation "
        f"(baseline={baseline}, rotated={rotated})"
    )


@pytest.mark.parametrize("trial", range(20))
def test_reduced_cell_parameters_invariant_to_random_rotation(
    trial: int, sample_lattice: torch.Tensor, rng: torch.Generator
) -> None:
    baseline = reduced_cell_parameters(sample_lattice)
    rotation = _random_rotation_matrix(rng)
    rotated_lattice = sample_lattice @ rotation.T
    rotated = reduced_cell_parameters(rotated_lattice)
    assert torch.allclose(baseline, rotated, atol=1e-6), (
        f"trial {trial}: reduced_cell_parameters changed under rotation "
        f"(baseline={baseline}, rotated={rotated})"
    )


def test_reduced_cell_parameters_matches_expected_values_for_orthogonal_cell() -> None:
    # A cubic cell with side 2.0: a=b=c=2.0, all angles 90 degrees.
    lattice = torch.eye(3, dtype=torch.float64) * 2.0
    params = reduced_cell_parameters(lattice)
    expected = torch.tensor([2.0, 2.0, 2.0, 90.0, 90.0, 90.0], dtype=torch.float64)
    assert torch.allclose(params, expected, atol=1e-6)


@pytest.mark.parametrize("trial", range(10))
def test_lattice_invariant_features_shape_and_invariance(
    trial: int, sample_lattice: torch.Tensor, rng: torch.Generator
) -> None:
    baseline = lattice_invariant_features(sample_lattice, num_atoms=4)
    assert baseline.shape == (12,)
    rotation = _random_rotation_matrix(rng)
    rotated_lattice = sample_lattice @ rotation.T
    rotated = lattice_invariant_features(rotated_lattice, num_atoms=4)
    assert torch.allclose(baseline, rotated, atol=1e-6)


def test_metric_tensor_6d_changes_under_anisotropic_rescaling(
    sample_lattice: torch.Tensor,
) -> None:
    """Sanity check: the descriptor is invariant to rotation but NOT to a
    genuine physical change of the cell (e.g. non-uniform scaling) -- this
    guards against a degenerate implementation that is trivially constant.
    """
    baseline = metric_tensor_6d(sample_lattice, num_atoms=None)
    rescaled_lattice = sample_lattice.clone()
    rescaled_lattice[0] *= 1.5  # stretch only the 'a' axis
    rescaled = metric_tensor_6d(rescaled_lattice, num_atoms=None)
    assert not torch.allclose(baseline, rescaled, atol=1e-3)
