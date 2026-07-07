import torch

from models.foundation_jepa.data.lattice import (
    MatrixMeanStdScaler,
    matrix_to_triu6,
    polar_decomposition_symmetrize,
)
from models.foundation_jepa.model.jepa import FoundationJEPA


def _toy_batch(seed=0, b=4, n=6):
    torch.manual_seed(seed)
    frac_coords = torch.rand(b, n, 3)
    atomic_numbers = torch.randint(1, 90, (b, n))
    atom_mask = torch.ones(b, n, dtype=torch.bool)
    atom_mask[0, 4:] = False
    atom_mask[1, 5:] = False
    num_atoms = atom_mask.sum(dim=1)
    raw_lattices = torch.stack(
        [polar_decomposition_symmetrize(torch.eye(3) + 0.1 * torch.randn(3, 3)) for _ in range(b)]
    )
    formation_energy = torch.randn(b)
    return frac_coords, atomic_numbers, raw_lattices, num_atoms, atom_mask, formation_energy


def _fit_scaler(raw_lattices, num_atoms):
    triu6 = torch.stack(
        [matrix_to_triu6(raw_lattices[i]) / (num_atoms[i].float() ** (1 / 3)) for i in range(raw_lattices.shape[0])]
    )
    return MatrixMeanStdScaler.fit(triu6)


def _build_model(**kwargs):
    frac_coords, atomic_numbers, raw_lattices, num_atoms, atom_mask, formation_energy = _toy_batch()
    scaler = _fit_scaler(raw_lattices, num_atoms)
    model = FoundationJEPA(hidden_dim=32, layers=2, attn_heads=4, max_atoms=10, matrix_scaler=scaler, **kwargs)
    return model, (frac_coords, atomic_numbers, raw_lattices, num_atoms, atom_mask, formation_energy)


def test_forward_output_shapes_and_finiteness():
    model, batch = _build_model()
    out = model(*batch)
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.loss_infonce)
    assert torch.isfinite(out.loss_reg)
    assert out.context_embedding.shape == (4, 32)
    assert out.target_embedding.shape == (4, 32)


def test_forward_requires_matrix_scaler():
    frac_coords, atomic_numbers, raw_lattices, num_atoms, atom_mask, formation_energy = _toy_batch()
    model = FoundationJEPA(hidden_dim=32, layers=2, attn_heads=4, max_atoms=10, matrix_scaler=None)
    try:
        model(frac_coords, atomic_numbers, raw_lattices, num_atoms, atom_mask, formation_energy)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_backward_updates_encoder_and_predictor_and_cond_emb():
    model, batch = _build_model()
    out = model(*batch)
    out.loss.backward()

    assert model.encoder.pre_backbone.map[0].weight.grad is not None
    assert model.encoder.pre_backbone.map[0].weight.grad.abs().sum() > 0
    assert model.predictor.map[0].weight.grad is not None
    assert model.cond_emb.map[0].weight.grad is not None


def test_encode_is_deterministic_without_augmentation():
    model, batch = _build_model()
    frac_coords, atomic_numbers, raw_lattices, num_atoms, atom_mask, _ = batch
    model.eval()
    with torch.no_grad():
        emb1 = model.encode(frac_coords, atomic_numbers, raw_lattices, num_atoms, atom_mask)
        emb2 = model.encode(frac_coords, atomic_numbers, raw_lattices, num_atoms, atom_mask)
    assert torch.allclose(emb1, emb2)


def test_context_and_target_embeddings_differ_due_to_augmentation():
    """Context view is translated+rotated, so its raw encoding (before the
    predictor tries to compensate) should generally differ from the target."""
    model, batch = _build_model()
    frac_coords, atomic_numbers, raw_lattices, num_atoms, atom_mask, _ = batch
    model.eval()
    with torch.no_grad():
        target_emb = model.encode(frac_coords, atomic_numbers, raw_lattices, num_atoms, atom_mask)
        from models.foundation_jepa.model.augmentation import rotate_lattice, translate_dense

        frac_aug, _ = translate_dense(frac_coords, atom_mask)
        lattice_aug, _ = rotate_lattice(raw_lattices)
        context_raw = model.encode(frac_aug, atomic_numbers, lattice_aug, num_atoms, atom_mask)
    assert not torch.allclose(target_emb, context_raw)


def test_hidden_dim_must_be_divisible_by_heads():
    try:
        FoundationJEPA(hidden_dim=17, layers=1, attn_heads=4)
        assert False, "expected ValueError"
    except ValueError:
        pass
