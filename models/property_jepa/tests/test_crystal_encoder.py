import torch

from models.property_jepa.data.lattice_features import lattice_invariant_features
from models.property_jepa.model.crystal_encoder import (
    RotationInvariantCrystalEncoder,
    build_atom_tokens,
)


def _random_lattice(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    # Keep the lattice well-conditioned (avoid near-singular matrices).
    return torch.eye(3) * 5.0 + 0.1 * torch.randn(3, 3, generator=generator)


def test_build_atom_tokens_shape_and_dtype():
    frac_coords = torch.rand(4, 3)
    atomic_numbers = torch.tensor([1, 6, 26, 100])

    tokens = build_atom_tokens(frac_coords, atomic_numbers)

    assert tokens.shape == (4, 103)
    assert tokens.dtype == torch.float32


def test_build_atom_tokens_rejects_shape_mismatch():
    frac_coords = torch.rand(4, 3)
    atomic_numbers = torch.tensor([1, 6, 26])

    try:
        build_atom_tokens(frac_coords, atomic_numbers)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_encoder_outputs_hidden_dim_cls_embedding():
    encoder = RotationInvariantCrystalEncoder(hidden_dim=32, layers=1, attn_heads=4)
    atom_tokens = torch.randn(3, 5, 103)
    atom_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
            [True, False, False, False, False],
        ]
    )
    lattice_features = torch.randn(3, 12)

    out = encoder(atom_tokens, atom_mask, lattice_features)

    assert out.shape == (3, 32)
    assert torch.isfinite(out).all()


def test_encoder_rejects_wrong_lattice_feature_dim():
    encoder = RotationInvariantCrystalEncoder(hidden_dim=32, layers=1, attn_heads=4)
    atom_tokens = torch.randn(2, 4, 103)
    atom_mask = torch.ones(2, 4, dtype=torch.bool)
    lattice_features = torch.randn(2, 6)

    try:
        encoder(atom_tokens, atom_mask, lattice_features)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_encoder_rejects_fully_masked_crystal():
    encoder = RotationInvariantCrystalEncoder(hidden_dim=32, layers=1, attn_heads=4)
    atom_tokens = torch.randn(1, 3, 103)
    atom_mask = torch.zeros(1, 3, dtype=torch.bool)
    lattice_features = torch.randn(1, 12)

    try:
        encoder(atom_tokens, atom_mask, lattice_features)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_encoder_output_is_deterministic_in_eval_mode():
    """Same inputs must give bit-for-bit-stable output in eval mode (no dropout
    randomness), across two separate forward calls.

    Note: atom order is NOT permutation-invariant by this architecture -- like
    stage 1's `CrystalTransformerEncoder`
    (`models/crystal_structure_jepa/JEPA/crystal_jepa.py`), atoms receive a
    learned positional embedding by sequence position, so re-ordering atom
    tokens does change the output. The invariance guarantee this encoder
    provides is *rotation* invariance (see
    `test_lattice_invariant_features_are_rotation_invariant_end_to_end`
    below), not permutation invariance over atom order.
    """
    encoder = RotationInvariantCrystalEncoder(hidden_dim=32, layers=2, attn_heads=4)
    encoder.eval()
    atom_tokens = torch.randn(1, 6, 103)
    atom_mask = torch.ones(1, 6, dtype=torch.bool)
    lattice_features = torch.randn(1, 12)

    with torch.no_grad():
        out_first = encoder(atom_tokens, atom_mask, lattice_features)
        out_second = encoder(atom_tokens, atom_mask, lattice_features)

    assert torch.allclose(out_first, out_second, atol=1e-6)


def test_lattice_invariant_features_are_rotation_invariant_end_to_end():
    """Rotating the lattice (a proxy for rotating the whole crystal) must not
    change `lattice_invariant_features`, and therefore must not change the
    resulting crystal embedding when fed through the encoder."""
    lattice = _random_lattice()
    # A concrete rotation matrix (about the z-axis).
    theta = 0.7
    rotation = torch.tensor(
        [
            [torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta)), 0.0],
            [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta)), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_lattice = lattice @ rotation.T

    features = lattice_invariant_features(lattice)
    rotated_features = lattice_invariant_features(rotated_lattice)

    assert torch.allclose(features, rotated_features, atol=1e-4)
