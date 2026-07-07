import torch

from models.foundation_jepa.data.lattice import MatrixMeanStdScaler
from models.foundation_jepa.model.jepa import FoundationJEPA
from models.physics_jepa.data.dataset import PhysicsTargetStats
from models.physics_jepa.model.heads import PhysicsHeads, PhysicsHeadsOutput
from models.physics_jepa.model.losses import PhysicsHeadsLoss, PhysicsLossOutput


def _make_foundation_model(hidden_dim=16, max_atoms=20):
    scaler = MatrixMeanStdScaler(mean=torch.zeros(1, 6), std=torch.ones(1, 6))
    return FoundationJEPA(
        hidden_dim=hidden_dim, layers=2, attn_heads=4, max_atoms=max_atoms, matrix_scaler=scaler
    )


def _make_batch(batch_size=4, n_atoms=3, max_atoms=20):
    frac_coords = torch.rand(batch_size, n_atoms, 3)
    atomic_numbers = torch.randint(1, 100, (batch_size, n_atoms))
    raw_lattice_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1) * 5.0
    num_atoms = torch.full((batch_size,), n_atoms, dtype=torch.long)
    atom_mask = torch.ones(batch_size, n_atoms, dtype=torch.bool)
    lambda_ep = torch.rand(batch_size) * 0.5 + 0.2
    omega_log = torch.rand(batch_size) * 100 + 150
    return {
        "frac_coords": frac_coords,
        "atomic_numbers": atomic_numbers,
        "raw_lattice_matrix": raw_lattice_matrix,
        "num_atoms": num_atoms,
        "atom_mask": atom_mask,
        "lambda_ep": lambda_ep,
        "omega_log": omega_log,
    }


class TestPhysicsHeads:
    def test_forward_returns_correct_shapes(self):
        fm = _make_foundation_model()
        heads = PhysicsHeads(fm)
        batch = _make_batch()
        out = heads(
            batch["frac_coords"], batch["atomic_numbers"], batch["raw_lattice_matrix"],
            batch["num_atoms"], batch["atom_mask"],
        )
        assert isinstance(out, PhysicsHeadsOutput)
        assert out.lambda_pred_norm.shape == (4,)
        assert out.omega_pred_norm.shape == (4,)
        assert torch.isfinite(out.lambda_pred_norm).all()
        assert torch.isfinite(out.omega_pred_norm).all()

    def test_freeze_encoder_stops_encoder_gradients(self):
        fm = _make_foundation_model()
        heads = PhysicsHeads(fm, freeze_encoder=True)
        for p in heads.encoder_parameters():
            assert not p.requires_grad

    def test_unfrozen_encoder_keeps_gradients_enabled(self):
        fm = _make_foundation_model()
        heads = PhysicsHeads(fm, freeze_encoder=False)
        assert all(p.requires_grad for p in heads.encoder_parameters())

    def test_head_parameters_excludes_encoder(self):
        fm = _make_foundation_model()
        heads = PhysicsHeads(fm)
        head_param_ids = {id(p) for p in heads.head_parameters()}
        encoder_param_ids = {id(p) for p in heads.encoder_parameters()}
        assert head_param_ids.isdisjoint(encoder_param_ids)

    def test_backward_updates_head_params_when_frozen(self):
        fm = _make_foundation_model()
        heads = PhysicsHeads(fm, freeze_encoder=True)
        batch = _make_batch()
        out = heads(
            batch["frac_coords"], batch["atomic_numbers"], batch["raw_lattice_matrix"],
            batch["num_atoms"], batch["atom_mask"],
        )
        loss = out.lambda_pred_norm.sum() + out.omega_pred_norm.sum()
        loss.backward()
        assert heads.lambda_head.weight.grad is not None
        assert heads.trunk.map[0].weight.grad is not None
        for p in heads.encoder_parameters():
            assert p.grad is None


class TestPhysicsHeadsLoss:
    def test_forward_returns_correct_output_type(self):
        stats = PhysicsTargetStats(0.0, 1.0, 0.0, 1.0)
        loss_fn = PhysicsHeadsLoss(stats)
        lambda_pred = torch.zeros(4)
        omega_pred = torch.zeros(4)
        lambda_target = torch.tensor([0.3, 0.5, 0.4, 0.6])
        omega_target = torch.tensor([180.0, 200.0, 220.0, 210.0])
        out = loss_fn(lambda_pred, omega_pred, lambda_target, omega_target)
        assert isinstance(out, PhysicsLossOutput)
        assert torch.isfinite(out.loss)
        assert out.loss.item() > 0

    def test_perfect_prediction_gives_near_zero_regression_loss(self):
        stats = PhysicsTargetStats(0.0, 1.0, 0.0, 1.0)
        lambda_target = torch.tensor([0.3, 0.5])
        omega_target = torch.tensor([180.0, 220.0])
        z_l, z_o = stats.normalize(lambda_target, omega_target)
        loss_fn = PhysicsHeadsLoss(stats, validity_weight=0.0)
        out = loss_fn(z_l, z_o, lambda_target, omega_target)
        assert out.loss_lambda.item() < 1e-6
        assert out.loss_omega.item() < 1e-6

    def test_validity_weight_zero_drops_penalty_contribution(self):
        stats = PhysicsTargetStats(0.0, 1.0, 0.0, 1.0)
        lambda_pred = torch.tensor([-5.0, -5.0])  # deep invalid region in physical units
        omega_pred = torch.tensor([0.0, 0.0])
        lambda_target = torch.tensor([0.3, 0.5])
        omega_target = torch.tensor([180.0, 220.0])
        loss_fn = PhysicsHeadsLoss(stats, validity_weight=0.0)
        out = loss_fn(lambda_pred, omega_pred, lambda_target, omega_target)
        assert out.loss_validity.item() > 0  # penalty still computed/reported
        assert abs(out.loss.item() - (out.loss_lambda + out.loss_omega).item()) < 1e-5

    def test_omega_weight_scales_omega_contribution(self):
        stats = PhysicsTargetStats(0.0, 1.0, 0.0, 1.0)
        lambda_pred = torch.zeros(2)
        omega_pred = torch.ones(2)
        lambda_target = torch.tensor([0.0, 0.0])
        omega_target = torch.tensor([0.0, 0.0])
        loss_fn_low = PhysicsHeadsLoss(stats, omega_weight=0.1, validity_weight=0.0)
        loss_fn_high = PhysicsHeadsLoss(stats, omega_weight=10.0, validity_weight=0.0)
        out_low = loss_fn_low(lambda_pred, omega_pred, lambda_target, omega_target)
        out_high = loss_fn_high(lambda_pred, omega_pred, lambda_target, omega_target)
        assert out_high.loss.item() > out_low.loss.item()
