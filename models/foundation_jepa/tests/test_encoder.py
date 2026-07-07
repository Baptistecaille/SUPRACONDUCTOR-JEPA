import torch

from models.foundation_jepa.model.encoder import (
    ATOM_TOKEN_DIM,
    FoundationCrystalEncoder,
    build_atom_tokens,
)


def test_atom_token_dim_is_109():
    assert ATOM_TOKEN_DIM == 109


def test_build_atom_tokens_shape_and_onehot():
    b, n = 2, 5
    frac_coords = torch.rand(b, n, 3)
    atomic_numbers = torch.randint(1, 100, (b, n))
    lattice_triu6 = torch.randn(b, 6)

    tokens = build_atom_tokens(frac_coords, atomic_numbers, lattice_triu6)
    assert tokens.shape == (b, n, 109)

    # one-hot block sums to 1 per atom
    one_hot_block = tokens[:, :, 3:103]
    assert torch.allclose(one_hot_block.sum(dim=-1), torch.ones(b, n))

    # lattice block broadcast identically across atoms within a sample
    lattice_block = tokens[:, :, 103:]
    for i in range(b):
        for j in range(1, n):
            assert torch.allclose(lattice_block[i, 0], lattice_block[i, j])


def test_build_atom_tokens_rejects_bad_shapes():
    frac_coords = torch.rand(2, 5, 3)
    atomic_numbers = torch.randint(1, 100, (2, 5))
    bad_lattice = torch.randn(2, 5)  # wrong last dim
    try:
        build_atom_tokens(frac_coords, atomic_numbers, bad_lattice)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_foundation_crystal_encoder_forward_shape():
    torch.manual_seed(0)
    encoder = FoundationCrystalEncoder(hidden_dim=16, layers=2, attn_heads=4, max_atoms=20)
    b, n = 3, 6
    frac_coords = torch.rand(b, n, 3)
    atomic_numbers = torch.randint(1, 100, (b, n))
    lattice_triu6 = torch.randn(b, 6)
    atom_mask = torch.ones(b, n, dtype=torch.bool)
    atom_mask[0, 4:] = False

    tokens = build_atom_tokens(frac_coords, atomic_numbers, lattice_triu6)
    out = encoder(tokens, atom_mask)
    assert out.shape == (b, 16)
    assert torch.isfinite(out).all()


def test_foundation_crystal_encoder_rejects_fully_masked_sample():
    encoder = FoundationCrystalEncoder(hidden_dim=8, layers=1, attn_heads=2, max_atoms=10)
    frac_coords = torch.rand(1, 4, 3)
    atomic_numbers = torch.randint(1, 100, (1, 4))
    lattice_triu6 = torch.randn(1, 6)
    atom_mask = torch.zeros(1, 4, dtype=torch.bool)

    tokens = build_atom_tokens(frac_coords, atomic_numbers, lattice_triu6)
    try:
        encoder(tokens, atom_mask)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_foundation_crystal_encoder_gradient_flows():
    torch.manual_seed(1)
    encoder = FoundationCrystalEncoder(hidden_dim=16, layers=2, attn_heads=4, max_atoms=20)
    frac_coords = torch.rand(2, 5, 3, requires_grad=False)
    atomic_numbers = torch.randint(1, 100, (2, 5))
    lattice_triu6 = torch.randn(2, 6)
    atom_mask = torch.ones(2, 5, dtype=torch.bool)

    tokens = build_atom_tokens(frac_coords, atomic_numbers, lattice_triu6)
    out = encoder(tokens, atom_mask)
    out.sum().backward()
    grad = encoder.pre_backbone.map[0].weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0
