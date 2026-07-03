import torch

from models.property_jepa.model.losses import PropertyJEPALoss, property_jepa_loss
from models.property_jepa.model.regulizers import CovarianceLoss, HingeStdLoss, VCLoss


def test_property_jepa_loss_returns_scalar_and_zero_for_identical_inputs():
    z = torch.randn(4, 16)

    loss = property_jepa_loss(z, z.clone())

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_property_jepa_loss_rejects_shape_mismatch():
    z_pred = torch.randn(4, 16)
    z_target = torch.randn(3, 16)

    try:
        property_jepa_loss(z_pred, z_target)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_property_jepa_loss_module_matches_function():
    z_pred = torch.randn(3, 8)
    z_target = torch.randn(3, 8)

    fn_loss = property_jepa_loss(z_pred, z_target)
    module_loss = PropertyJEPALoss()(z_pred, z_target)

    assert torch.allclose(fn_loss, module_loss)


def test_hinge_std_loss_is_zero_when_std_meets_margin():
    x = torch.eye(4) * 10.0  # large per-feature spread

    loss = HingeStdLoss(std_margin=0.1)(x)

    assert loss.item() == 0.0


def test_covariance_loss_returns_zero_for_batch_size_one():
    x = torch.randn(1, 8)

    loss = CovarianceLoss()(x)

    assert torch.allclose(loss, torch.tensor(0.0))


def test_vc_loss_returns_dict_with_finite_components():
    vc = VCLoss(std_coeff=1.0, cov_coeff=1.0)
    x = torch.randn(8, 16)

    out = vc(x)

    assert set(out.keys()) == {"loss", "std_loss", "cov_loss"}
    assert torch.isfinite(out["loss"])
