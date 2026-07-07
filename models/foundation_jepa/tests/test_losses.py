import torch

from models.foundation_jepa.model.losses import (
    EnergyAwareInfoNCELoss,
    FoundationJEPALoss,
    VarianceCovarianceRegularizer,
    energy_aware_weight,
)


def test_energy_aware_weight_symmetric_and_zero_diagonal():
    e = torch.tensor([0.0, 1.0, 2.0, 2.0])
    w = energy_aware_weight(e)
    assert w.shape == (4, 4)
    assert torch.allclose(w, w.T)
    assert torch.allclose(torch.diagonal(w), torch.zeros(4))
    # identical formation energies (index 2, 3) -> weight 0 between them
    assert torch.allclose(w[2, 3], torch.tensor(0.0))
    # weight increases monotonically with |ΔE|
    assert w[0, 1] < w[0, 2]


def test_infonce_loss_lower_for_aligned_embeddings():
    torch.manual_seed(0)
    loss_fn = EnergyAwareInfoNCELoss(temperature=0.1)
    b, h = 6, 16
    target = torch.randn(b, h)
    formation_energy = torch.randn(b)

    aligned_context = target + 0.01 * torch.randn(b, h)
    misaligned_context = torch.randn(b, h)

    loss_aligned = loss_fn(aligned_context, target, formation_energy)
    loss_misaligned = loss_fn(misaligned_context, target, formation_energy)
    assert loss_aligned.item() < loss_misaligned.item()


def test_infonce_loss_rejects_batch_mismatch():
    loss_fn = EnergyAwareInfoNCELoss()
    context = torch.randn(4, 8)
    target = torch.randn(5, 8)
    formation_energy = torch.randn(4)
    try:
        loss_fn(context, target, formation_energy)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_infonce_loss_finite_with_zero_norm_embedding():
    """A degenerate (all-zero) embedding must not NaN the loss (eps floor)."""
    loss_fn = EnergyAwareInfoNCELoss()
    context = torch.zeros(3, 8)
    target = torch.zeros(3, 8)
    formation_energy = torch.randn(3)
    loss = loss_fn(context, target, formation_energy)
    assert torch.isfinite(loss)


def test_variance_covariance_regularizer_penalizes_collapse():
    reg = VarianceCovarianceRegularizer()
    collapsed = torch.ones(10, 16)  # zero variance across the batch
    healthy = torch.randn(10, 16) * 3

    out_collapsed = reg(collapsed)
    out_healthy = reg(healthy)
    assert out_collapsed["loss"].item() > out_healthy["loss"].item()


def test_foundation_jepa_loss_combines_terms_and_backprops():
    torch.manual_seed(1)
    loss_fn = FoundationJEPALoss(temperature=0.1, reg_weight=0.05)
    context = torch.randn(5, 12, requires_grad=True)
    target = torch.randn(5, 12, requires_grad=True)
    formation_energy = torch.randn(5)

    out = loss_fn(context, target, formation_energy)
    assert torch.isfinite(out.loss)
    expected = out.loss_infonce + 0.05 * out.loss_reg
    assert torch.allclose(out.loss, expected, atol=1e-5)

    out.loss.backward()
    assert context.grad is not None
    assert target.grad is not None
