import torch

from models.property_jepa.model.crystal_encoder import RotationInvariantCrystalEncoder
from models.property_jepa.model.jepa import PropertyJEPA
from models.property_jepa.model.predictor import PropertyPredictor
from models.property_jepa.model.property_encoder import PropertyEncoder, TargetPropertyEncoder
from models.property_jepa.model.regulizers import VCLoss

HIDDEN_DIM = 16
NUM_PROPERTY_TYPES = 14


def _make_model(reg_weight: float = 0.0) -> PropertyJEPA:
    return PropertyJEPA(
        crystal_encoder=RotationInvariantCrystalEncoder(
            hidden_dim=HIDDEN_DIM, layers=1, attn_heads=4
        ),
        property_encoder=PropertyEncoder(
            hidden_dim=HIDDEN_DIM, num_property_types=NUM_PROPERTY_TYPES
        ),
        target_property_encoder=TargetPropertyEncoder(
            hidden_dim=HIDDEN_DIM, num_property_types=NUM_PROPERTY_TYPES
        ),
        predictor=PropertyPredictor(
            hidden_dim=HIDDEN_DIM, num_property_types=NUM_PROPERTY_TYPES
        ),
        regularizer=VCLoss(std_coeff=0.1, cov_coeff=0.1) if reg_weight > 0 else None,
        reg_weight=reg_weight,
    )


def _make_batch(batch_size=3, num_atoms=5, num_known=2) -> dict:
    return {
        "atom_tokens": torch.randn(batch_size, num_atoms, 103),
        "atom_mask": torch.ones(batch_size, num_atoms, dtype=torch.bool),
        "lattice_features": torch.randn(batch_size, 12),
        "known_property_type_id": torch.randint(
            0, NUM_PROPERTY_TYPES, (batch_size, num_known)
        ),
        "known_property_value": torch.randn(batch_size, num_known),
        "known_property_mask": torch.ones(batch_size, num_known, dtype=torch.bool),
        "query_property_type_id": torch.randint(0, NUM_PROPERTY_TYPES, (batch_size,)),
        "query_property_value": torch.randn(batch_size),
    }


def test_property_jepa_forward_returns_finite_loss_and_metrics():
    model = _make_model()
    batch = _make_batch()

    out = model(batch)

    assert out["loss"].shape == ()
    assert torch.isfinite(out["loss"])
    assert out["z_pred"].shape == (3, HIDDEN_DIM)
    assert out["z_target"].shape == (3, HIDDEN_DIM)
    assert "loss_pred" in out["metrics"]


def test_property_jepa_forward_with_regularizer():
    model = _make_model(reg_weight=0.5)
    batch = _make_batch()

    out = model(batch)

    assert torch.isfinite(out["loss"])
    assert "reg_std_loss" in out["metrics"]
    assert "reg_cov_loss" in out["metrics"]


def test_target_property_encoder_has_no_gradients_after_backward():
    model = _make_model()
    batch = _make_batch()

    out = model(batch)
    out["loss"].backward()

    assert any(p.grad is not None for p in model.crystal_encoder.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())
    assert any(p.grad is not None for p in model.property_encoder.parameters())
    assert all(p.grad is None for p in model.target_property_encoder.parameters())


def test_update_ema_moves_target_towards_online_encoder():
    model = _make_model()
    target_before = model.target_property_encoder.type_embedding.weight.detach().clone()

    with torch.no_grad():
        model.property_encoder.type_embedding.weight.add_(1.0)
    model.update_ema(decay=0.5)

    target_after = model.target_property_encoder.type_embedding.weight.detach()
    assert not torch.allclose(target_before, target_after)


def test_property_jepa_handles_batch_with_no_known_properties():
    model = _make_model()
    batch = _make_batch(num_known=0)

    out = model(batch)

    assert torch.isfinite(out["loss"])


def test_property_jepa_rejects_mismatched_hidden_dims():
    try:
        PropertyJEPA(
            crystal_encoder=RotationInvariantCrystalEncoder(
                hidden_dim=16, layers=1, attn_heads=4
            ),
            property_encoder=PropertyEncoder(hidden_dim=8, num_property_types=NUM_PROPERTY_TYPES),
            target_property_encoder=TargetPropertyEncoder(
                hidden_dim=8, num_property_types=NUM_PROPERTY_TYPES
            ),
            predictor=PropertyPredictor(hidden_dim=8, num_property_types=NUM_PROPERTY_TYPES),
        )
        assert False, "expected ValueError"
    except ValueError:
        pass
