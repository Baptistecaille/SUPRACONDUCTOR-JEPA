import pytest
import torch

from JEPA.losses import WeightedContrastiveLoss, weighted_contrastive_loss


def test_weighted_contrastive_loss_returns_scalar():
    context = torch.eye(4, 512)[:4]
    target = torch.eye(4, 512)[:4]
    ef_per_atom = torch.tensor([-1.0, -2.0, -0.5, -3.0])

    loss = weighted_contrastive_loss(context, target, ef_per_atom)

    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_weighted_contrastive_loss_requires_batch_size_two():
    context = torch.randn(1, 512)
    target = torch.randn(1, 512)
    ef_per_atom = torch.tensor([-1.0])

    with pytest.raises(ValueError, match="batch size"):
        weighted_contrastive_loss(context, target, ef_per_atom)


def test_weighted_contrastive_loss_rejects_shape_mismatch():
    context = torch.randn(3, 512)
    target = torch.randn(4, 512)
    ef_per_atom = torch.randn(3)

    with pytest.raises(ValueError, match="matching shapes"):
        weighted_contrastive_loss(context, target, ef_per_atom)


def test_weighted_contrastive_loss_module_matches_function():
    context = torch.randn(3, 512)
    target = torch.randn(3, 512)
    ef_per_atom = torch.tensor([-1.0, -2.0, -0.5])

    fn_loss = weighted_contrastive_loss(context, target, ef_per_atom, temperature=0.1)
    module_loss = WeightedContrastiveLoss(temperature=0.1)(
        context, target, ef_per_atom
    )

    assert torch.allclose(fn_loss, module_loss)
