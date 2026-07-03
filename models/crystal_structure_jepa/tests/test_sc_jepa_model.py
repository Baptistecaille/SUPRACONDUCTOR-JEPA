import torch

from JEPA.crystal_jepa import (
    CrystalJEPA,
    CrystalTransformerEncoder,
    MaskConditionedPredictor,
    TargetCrystalTransformerEncoder,
)


def test_context_encoder_outputs_512_dim_cls():
    encoder = CrystalTransformerEncoder(input_dim=103)
    atom_features = torch.randn(3, 5, 103)
    atom_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
            [True, False, False, False, False],
        ]
    )

    out = encoder(atom_features, atom_mask)

    assert out.shape == (3, 512)


def test_target_encoder_accepts_110_dim_tokens():
    encoder = TargetCrystalTransformerEncoder()
    atom_features = torch.randn(2, 4, 110)
    atom_mask = torch.ones(2, 4, dtype=torch.bool)

    out = encoder(atom_features, atom_mask)

    assert out.shape == (2, 512)


def test_predictor_outputs_512_dim_embedding():
    predictor = MaskConditionedPredictor()
    z_context = torch.randn(4, 512)
    mask_box = torch.rand(4, 4)

    out = predictor(z_context, mask_box)

    assert out.shape == (4, 512)


def _make_batch(batch_size=3, context_atoms=4, target_atoms=5):
    return {
        "encoder1_atom_features": torch.randn(batch_size, context_atoms, 103),
        "encoder1_atom_mask": torch.ones(batch_size, context_atoms, dtype=torch.bool),
        "encoder2_atom_features": torch.randn(batch_size, target_atoms, 110),
        "encoder2_atom_mask": torch.ones(batch_size, target_atoms, dtype=torch.bool),
        "mask_box": torch.rand(batch_size, 4),
        "ef_per_atom": torch.linspace(-3.0, -1.0, batch_size),
    }


def test_crystal_jepa_forward_returns_finite_loss_and_metrics():
    model = CrystalJEPA(
        context_encoder=CrystalTransformerEncoder(input_dim=103, layers=1, attn_heads=8),
        target_encoder=TargetCrystalTransformerEncoder(layers=1, attn_heads=8),
        predictor=MaskConditionedPredictor(),
    )
    batch = _make_batch()

    out = model(batch)

    assert out["loss"].shape == ()
    assert torch.isfinite(out["loss"])
    assert out["z_pred"].shape == (3, 512)
    assert out["z_target"].shape == (3, 512)
    assert "loss_pred" in out["metrics"]


def test_target_encoder_parameters_have_no_gradients_after_backward():
    model = CrystalJEPA(
        context_encoder=CrystalTransformerEncoder(input_dim=103, layers=1, attn_heads=8),
        target_encoder=TargetCrystalTransformerEncoder(layers=1, attn_heads=8),
        predictor=MaskConditionedPredictor(),
    )
    batch = _make_batch()

    out = model(batch)
    out["loss"].backward()

    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_update_ema_changes_target_atom_projection_but_not_family_projection():
    model = CrystalJEPA(
        context_encoder=CrystalTransformerEncoder(input_dim=103, layers=1, attn_heads=8),
        target_encoder=TargetCrystalTransformerEncoder(layers=1, attn_heads=8),
        predictor=MaskConditionedPredictor(),
    )
    family_before = {
        name: param.detach().clone()
        for name, param in model.target_encoder.family_proj.named_parameters()
    }
    target_before = model.target_encoder.atom_proj.net[0].weight.detach().clone()

    with torch.no_grad():
        model.context_encoder.atom_proj.net[0].weight.add_(1.0)
    model.update_ema(decay=0.5)

    target_after = model.target_encoder.atom_proj.net[0].weight.detach()
    assert not torch.allclose(target_before, target_after)
    for name, param in model.target_encoder.family_proj.named_parameters():
        assert torch.allclose(family_before[name], param.detach())


def test_sc_jepa_public_api_importable():
    from JEPA import (
        CrystalJEPA,
        CrystalTransformerEncoder,
        MaskConditionedPredictor,
        TargetCrystalTransformerEncoder,
        WeightedContrastiveLoss,
        weighted_contrastive_loss,
    )

    assert CrystalJEPA is not None
    assert CrystalTransformerEncoder is not None
    assert TargetCrystalTransformerEncoder is not None
    assert MaskConditionedPredictor is not None
    assert WeightedContrastiveLoss is not None
    assert callable(weighted_contrastive_loss)
