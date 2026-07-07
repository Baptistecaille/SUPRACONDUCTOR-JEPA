import torch

from models.foundation_jepa.data.lattice import (
    MatrixMeanStdScaler,
    batched_triu6_to_matrix,
    matrix_to_triu6,
    polar_decomposition_symmetrize,
    triu6_to_matrix,
)


def test_polar_decomposition_symmetrize_is_symmetric():
    torch.manual_seed(0)
    m = torch.eye(3) + 0.3 * torch.randn(3, 3)
    sym = polar_decomposition_symmetrize(m)
    assert torch.allclose(sym, sym.T, atol=1e-4)


def test_polar_decomposition_identity_stays_identity():
    sym = polar_decomposition_symmetrize(torch.eye(3))
    assert torch.allclose(sym, torch.eye(3), atol=1e-4)


def test_polar_decomposition_rotation_invariant_up_to_reflection():
    """Rotating the input lattice by an orthogonal matrix should leave the
    symmetrized descriptor's singular values (hence its diagonal-in-eigenbasis
    content) unchanged -- polar decomposition strips the applied rotation."""
    torch.manual_seed(1)
    m = torch.eye(3) + 0.2 * torch.randn(3, 3)
    q, _ = torch.linalg.qr(torch.randn(3, 3))  # random orthogonal matrix
    rotated = q @ m

    sym1 = polar_decomposition_symmetrize(m)
    sym2 = polar_decomposition_symmetrize(rotated)

    eig1 = torch.linalg.eigvalsh(sym1).sort().values
    eig2 = torch.linalg.eigvalsh(sym2).sort().values
    assert torch.allclose(eig1, eig2, atol=1e-3)


def test_triu6_roundtrip():
    torch.manual_seed(2)
    a = torch.randn(3, 3)
    sym = (a + a.T) / 2
    vec = matrix_to_triu6(sym)
    assert vec.shape == (6,)
    recon = triu6_to_matrix(vec)
    assert torch.allclose(recon, sym, atol=1e-6)


def test_batched_triu6_to_matrix_matches_single():
    torch.manual_seed(3)
    vecs = torch.randn(5, 6)
    batched = batched_triu6_to_matrix(vecs)
    for i in range(5):
        assert torch.allclose(batched[i], triu6_to_matrix(vecs[i]))


def test_matrix_scaler_fit_transform_roundtrip():
    torch.manual_seed(4)
    data = torch.randn(100, 6) * 3 + 5
    scaler = MatrixMeanStdScaler.fit(data)
    scaled = scaler.transform(data)
    assert scaled.mean(dim=0).abs().max() < 1e-4
    assert (scaled.std(dim=0) - 1.0).abs().max() < 1e-4
    recon = scaler.inverse_transform(scaled)
    assert torch.allclose(recon, data, atol=1e-3)


def test_matrix_scaler_handles_zero_variance_feature():
    data = torch.zeros(10, 6)
    data[:, 0] = 1.0  # constant feature, std would be 0
    scaler = MatrixMeanStdScaler.fit(data)
    scaled = scaler.transform(data)
    assert torch.isfinite(scaled).all()
