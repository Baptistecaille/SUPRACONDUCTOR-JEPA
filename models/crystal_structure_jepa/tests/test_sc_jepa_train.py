import torch

from train_sc_jepa import build_model, move_batch_to_device, train_one_step


def test_train_one_step_updates_context_and_writes_finite_loss():
    device = torch.device("cpu")
    model = build_model(layers=1, attn_heads=8).to(device)
    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=1e-4,
    )
    batch = {
        "encoder1_atom_features": torch.randn(3, 4, 103),
        "encoder1_atom_mask": torch.ones(3, 4, dtype=torch.bool),
        "encoder2_atom_features": torch.randn(3, 5, 110),
        "encoder2_atom_mask": torch.ones(3, 5, dtype=torch.bool),
        "mask_box": torch.rand(3, 4),
        "ef_per_atom": torch.tensor([-1.0, -2.0, -0.5]),
        "material_id": ["a", "b", "c"],
    }
    batch = move_batch_to_device(batch, device)

    metrics = train_one_step(model, batch, optimizer)

    assert metrics["loss"] > 0
    assert metrics["loss_pred"] > 0


def test_build_model_can_enable_vc_regularizer():
    model = build_model(
        layers=1,
        attn_heads=8,
        regularizer="vc",
        reg_weight=0.01,
    )
    batch = {
        "encoder1_atom_features": torch.randn(3, 4, 103),
        "encoder1_atom_mask": torch.ones(3, 4, dtype=torch.bool),
        "encoder2_atom_features": torch.randn(3, 5, 110),
        "encoder2_atom_mask": torch.ones(3, 5, dtype=torch.bool),
        "mask_box": torch.rand(3, 4),
        "ef_per_atom": torch.tensor([-1.0, -2.0, -0.5]),
    }

    out = model(batch)

    assert torch.isfinite(out["loss"])
    assert out["metrics"]["loss_reg"] > 0
    assert "reg_std_loss" in out["metrics"]
    assert "reg_cov_loss" in out["metrics"]
