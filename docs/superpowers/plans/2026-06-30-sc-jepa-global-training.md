# SC-JEPA Global Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first SC-JEPA training path that predicts a complete-crystal global embedding from a spatially corrupted superconductor crystal using the Crys-JEPA weighted contrastive loss.

**Architecture:** Add crystal-specific modules alongside the existing temporal JEPA code. Encoder 1 consumes corrupted 103-dim atom tokens, Encoder 2 consumes complete 110-dim atom tokens with a factorized target projection, the predictor outputs a 512-dim embedding, and target-compatible weights are updated by EMA. Training uses the Crys-JEPA batch contrastive objective weighted by formation-energy differences.

**Tech Stack:** Python 3.12, PyTorch, pymatgen, pandas, pytest.

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| Modify | `crystal_matrices.py` | Guarantee spatial masking leaves at least one visible atom and masks at least one atom when `num_atoms >= 2`. |
| Modify | `JEPA/losses.py` | Add Crys-JEPA weighted contrastive loss for `[B, 512]` embeddings and `ef_per_atom`. |
| Create | `JEPA/crystal_jepa.py` | Add `CrystalTransformerEncoder`, `TargetCrystalTransformerEncoder`, `MaskConditionedPredictor`, `CrystalJEPA`, and EMA helpers. |
| Modify | `JEPA/__init__.py` | Export the new SC-JEPA classes and loss. |
| Create | `train_sc_jepa.py` | Minimal training entrypoint with `max_batches`, checkpointing, and no remote/push behavior. |
| Create | `tests/test_sc_jepa_loss.py` | Unit tests for weighted contrastive loss. |
| Create | `tests/test_sc_jepa_model.py` | Unit tests for encoders, predictor, wrapper, EMA, and one training step. |
| Modify | `tests/test_crystal_matrices.py` | Add a regression test for guaranteed actual masking. |

Do not run `git push`. Local commits in the steps are optional if the user wants them; skip commit steps when working in a no-commit mode.

---

### Task 1: Guarantee Spatial Masking Actually Masks an Atom

**Files:**
- Modify: `crystal_matrices.py`
- Modify: `tests/test_crystal_matrices.py`

- [ ] **Step 1: Add a failing test for guaranteed masked atoms**

Append this test to `tests/test_crystal_matrices.py`:

```python
def test_sample_spatial_block_masks_one_atom_when_possible():
    frac_coords = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
        ],
        dtype=torch.float32,
    )
    visible, mask_box = sample_spatial_block(
        frac_coords,
        center=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
        box_size=0.01,
    )

    assert visible.shape == (2,)
    assert mask_box.shape == (4,)
    assert visible.any()
    assert (~visible).any()
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
python -m pytest tests/test_crystal_matrices.py::test_sample_spatial_block_masks_one_atom_when_possible -v
```

Expected: FAIL because the tiny box masks no atoms.

- [ ] **Step 3: Update `sample_spatial_block`**

In `crystal_matrices.py`, replace the end of `sample_spatial_block` from `visible = spatial_block_visible_mask(...)` through the return with:

```python
    visible = spatial_block_visible_mask(frac_coords, center, size)
    if frac_coords.numel() > 0 and not visible.any():
        toric_delta = torch.remainder(frac_coords - center + 0.5, 1.0) - 0.5
        farthest_atom = torch.norm(toric_delta, dim=-1).argmax()
        visible[farthest_atom] = True

    if frac_coords.shape[0] >= 2 and visible.all():
        toric_delta = torch.remainder(frac_coords - center + 0.5, 1.0) - 0.5
        closest_atom = torch.norm(toric_delta, dim=-1).argmin()
        visible[closest_atom] = False

    mask_box = torch.cat([center, size.reshape(1)])
    return visible, mask_box
```

- [ ] **Step 4: Run masking tests**

Run:

```bash
python -m pytest tests/test_crystal_matrices.py -v
```

Expected: PASS.

- [ ] **Step 5: Optional local commit**

Run only if local commits are desired:

```bash
git add crystal_matrices.py tests/test_crystal_matrices.py
git commit -m "fix: ensure spatial block masks atoms when possible"
```

Expected: local commit succeeds. Do not push.

---

### Task 2: Add Crys-JEPA Weighted Contrastive Loss

**Files:**
- Modify: `JEPA/losses.py`
- Create: `tests/test_sc_jepa_loss.py`

- [ ] **Step 1: Write failing loss tests**

Create `tests/test_sc_jepa_loss.py`:

```python
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
    module_loss = WeightedContrastiveLoss(temperature=0.1)(context, target, ef_per_atom)

    assert torch.allclose(fn_loss, module_loss)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_sc_jepa_loss.py -v
```

Expected: FAIL with import error for `WeightedContrastiveLoss`.

- [ ] **Step 3: Implement the loss**

Append this to `JEPA/losses.py`:

```python
def weighted_contrastive_loss(
    context: torch.Tensor,
    target: torch.Tensor,
    ef_per_atom: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Crys-JEPA weighted contrastive objective over global crystal embeddings."""
    if context.shape != target.shape:
        raise ValueError(
            f"weighted_contrastive_loss expected matching shapes, got "
            f"{tuple(context.shape)} and {tuple(target.shape)}"
        )
    if context.dim() != 2:
        raise ValueError(
            f"weighted_contrastive_loss expects [B, D] tensors, got {tuple(context.shape)}"
        )
    if context.size(0) < 2:
        raise ValueError("weighted_contrastive_loss requires batch size >= 2")
    if ef_per_atom.reshape(-1).shape[0] != context.size(0):
        raise ValueError(
            f"ef_per_atom must contain one value per sample, got "
            f"{ef_per_atom.reshape(-1).shape[0]} for batch size {context.size(0)}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    context_norm = F.normalize(context, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    sim = context_norm @ target_norm.t()

    ef = ef_per_atom.reshape(-1).to(device=context.device, dtype=context.dtype)
    weight = 1.0 - torch.exp(-torch.abs(ef.unsqueeze(0) - ef.unsqueeze(1)))
    diag = torch.eye(context.size(0), dtype=context.dtype, device=context.device)
    weight = weight + diag

    logits = sim * weight / temperature
    labels = torch.arange(context.size(0), device=context.device)
    return F.cross_entropy(logits, labels)


class WeightedContrastiveLoss(losses):
    """Module wrapper for the Crys-JEPA weighted contrastive objective."""

    def __init__(self, temperature: float = 0.1, name: str | None = None):
        super().__init__(name="WeightedContrastiveLoss" if name is None else name)
        self.temperature = temperature

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        ef_per_atom: torch.Tensor,
    ) -> torch.Tensor:
        return weighted_contrastive_loss(
            context,
            target,
            ef_per_atom,
            temperature=self.temperature,
        )
```

- [ ] **Step 4: Run loss tests**

Run:

```bash
python -m pytest tests/test_sc_jepa_loss.py -v
```

Expected: PASS.

- [ ] **Step 5: Optional local commit**

Run only if local commits are desired:

```bash
git add JEPA/losses.py tests/test_sc_jepa_loss.py
git commit -m "feat: add Crys-JEPA weighted contrastive loss"
```

Expected: local commit succeeds. Do not push.

---

### Task 3: Add Crystal Encoders and Mask-Conditioned Predictor

**Files:**
- Create: `JEPA/crystal_jepa.py`
- Create: `tests/test_sc_jepa_model.py`

- [ ] **Step 1: Write failing model shape tests**

Create `tests/test_sc_jepa_model.py`:

```python
import torch

from JEPA.crystal_jepa import (
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_sc_jepa_model.py::test_context_encoder_outputs_512_dim_cls tests/test_sc_jepa_model.py::test_target_encoder_accepts_110_dim_tokens tests/test_sc_jepa_model.py::test_predictor_outputs_512_dim_embedding -v
```

Expected: FAIL with import error for `JEPA.crystal_jepa`.

- [ ] **Step 3: Implement encoder and predictor modules**

Create `JEPA/crystal_jepa.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .losses import WeightedContrastiveLoss


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int = 2,
    ):
        super().__init__()
        if layers < 2:
            raise ValueError("MLP requires layers >= 2")
        modules: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(layers - 2):
            modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        modules.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrystalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        layers: int = 8,
        attn_heads: int = 16,
        dropout: float = 0.0,
        max_atoms: int = 500,
    ):
        super().__init__()
        if hidden_dim != 512:
            raise ValueError("SC-JEPA fixes hidden_dim to 512")
        if hidden_dim % attn_heads != 0:
            raise ValueError("hidden_dim must be divisible by attn_heads")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_atoms = max_atoms
        self.atom_proj = MLP(input_dim, hidden_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(max_atoms + 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attn_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def _validate_inputs(self, atom_features: torch.Tensor, atom_mask: torch.Tensor) -> None:
        if atom_features.dim() != 3:
            raise ValueError(f"atom_features must be [B, N, F], got {tuple(atom_features.shape)}")
        if atom_features.size(-1) != self.input_dim:
            raise ValueError(f"expected feature dim {self.input_dim}, got {atom_features.size(-1)}")
        if atom_mask.shape != atom_features.shape[:2]:
            raise ValueError(
                f"atom_mask shape {tuple(atom_mask.shape)} must match token shape "
                f"{tuple(atom_features.shape[:2])}"
            )
        if not atom_mask.any(dim=1).all():
            raise ValueError("each crystal must contain at least one visible atom")
        if atom_features.size(1) > self.max_atoms:
            raise ValueError(f"sequence length {atom_features.size(1)} exceeds max_atoms {self.max_atoms}")

    def project_tokens(self, atom_features: torch.Tensor) -> torch.Tensor:
        return self.atom_proj(atom_features)

    def forward(self, atom_features: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(atom_features, atom_mask)
        b, n, _ = atom_features.shape
        tokens = self.project_tokens(atom_features)
        cls = self.cls_token.expand(b, 1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embedding[: n + 1].unsqueeze(0)
        cls_mask = torch.ones(b, 1, dtype=torch.bool, device=atom_mask.device)
        full_mask = torch.cat([cls_mask, atom_mask.bool()], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=~full_mask)
        encoded = self.norm(encoded)
        return encoded[:, 0]


class TargetCrystalTransformerEncoder(CrystalTransformerEncoder):
    def __init__(self, hidden_dim: int = 512, **kwargs):
        super().__init__(input_dim=103, hidden_dim=hidden_dim, **kwargs)
        self.family_proj = MLP(7, hidden_dim, hidden_dim)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def _validate_inputs(self, atom_features: torch.Tensor, atom_mask: torch.Tensor) -> None:
        if atom_features.dim() != 3:
            raise ValueError(f"atom_features must be [B, N, F], got {tuple(atom_features.shape)}")
        if atom_features.size(-1) != 110:
            raise ValueError(f"target encoder expects feature dim 110, got {atom_features.size(-1)}")
        if atom_mask.shape != atom_features.shape[:2]:
            raise ValueError(
                f"atom_mask shape {tuple(atom_mask.shape)} must match token shape "
                f"{tuple(atom_features.shape[:2])}"
            )
        if not atom_mask.any(dim=1).all():
            raise ValueError("each crystal must contain at least one atom")
        if atom_features.size(1) > self.max_atoms:
            raise ValueError(f"sequence length {atom_features.size(1)} exceeds max_atoms {self.max_atoms}")

    def project_tokens(self, atom_features: torch.Tensor) -> torch.Tensor:
        atom_part = atom_features[..., :103]
        family_part = atom_features[..., 103:]
        return self.atom_proj(atom_part) + self.family_proj(family_part)


class MaskConditionedPredictor(nn.Module):
    def __init__(self, hidden_dim: int = 512, layers: int = 2):
        super().__init__()
        if hidden_dim != 512:
            raise ValueError("SC-JEPA fixes hidden_dim to 512")
        self.mask_mlp = MLP(4, hidden_dim, hidden_dim, layers=layers)
        self.predictor = MLP(hidden_dim, hidden_dim, hidden_dim, layers=layers)

    def forward(self, z_context: torch.Tensor, mask_box: torch.Tensor) -> torch.Tensor:
        if z_context.dim() != 2 or z_context.size(-1) != 512:
            raise ValueError(f"z_context must be [B, 512], got {tuple(z_context.shape)}")
        if mask_box.shape != (z_context.size(0), 4):
            raise ValueError(f"mask_box must be [B, 4], got {tuple(mask_box.shape)}")
        return self.predictor(z_context + self.mask_mlp(mask_box.to(z_context.dtype)))
```

- [ ] **Step 4: Run model shape tests**

Run:

```bash
python -m pytest tests/test_sc_jepa_model.py::test_context_encoder_outputs_512_dim_cls tests/test_sc_jepa_model.py::test_target_encoder_accepts_110_dim_tokens tests/test_sc_jepa_model.py::test_predictor_outputs_512_dim_embedding -v
```

Expected: PASS.

- [ ] **Step 5: Optional local commit**

Run only if local commits are desired:

```bash
git add JEPA/crystal_jepa.py tests/test_sc_jepa_model.py
git commit -m "feat: add SC-JEPA crystal encoder modules"
```

Expected: local commit succeeds. Do not push.

---

### Task 4: Add CrystalJEPA Wrapper and EMA Updates

**Files:**
- Modify: `JEPA/crystal_jepa.py`
- Modify: `tests/test_sc_jepa_model.py`

- [ ] **Step 1: Add failing wrapper and EMA tests**

Append these tests to `tests/test_sc_jepa_model.py`:

```python
from JEPA.crystal_jepa import CrystalJEPA


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_sc_jepa_model.py::test_crystal_jepa_forward_returns_finite_loss_and_metrics tests/test_sc_jepa_model.py::test_target_encoder_parameters_have_no_gradients_after_backward tests/test_sc_jepa_model.py::test_update_ema_changes_target_atom_projection_but_not_family_projection -v
```

Expected: FAIL with import error for `CrystalJEPA`.

- [ ] **Step 3: Implement `CrystalJEPA` and EMA**

Append this to `JEPA/crystal_jepa.py`:

```python
@dataclass
class CrystalJEPAOutput:
    loss: torch.Tensor
    loss_pred: torch.Tensor
    loss_reg: torch.Tensor
    z_pred: torch.Tensor
    z_target: torch.Tensor
    metrics: dict[str, float]


class CrystalJEPA(nn.Module):
    def __init__(
        self,
        context_encoder: CrystalTransformerEncoder,
        target_encoder: TargetCrystalTransformerEncoder,
        predictor: MaskConditionedPredictor,
        pred_loss: nn.Module | None = None,
        regularizer: nn.Module | None = None,
        reg_weight: float = 0.0,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.pred_loss = WeightedContrastiveLoss() if pred_loss is None else pred_loss
        self.regularizer = regularizer
        self.reg_weight = reg_weight
        self.ema_decay = ema_decay
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

    def _tensor(self, batch: dict, key: str) -> torch.Tensor:
        value = batch[key]
        if not torch.is_tensor(value):
            raise ValueError(f"batch[{key!r}] must be a tensor")
        return value

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, float]]:
        encoder1_features = self._tensor(batch, "encoder1_atom_features")
        encoder1_mask = self._tensor(batch, "encoder1_atom_mask")
        encoder2_features = self._tensor(batch, "encoder2_atom_features")
        encoder2_mask = self._tensor(batch, "encoder2_atom_mask")
        mask_box = self._tensor(batch, "mask_box")
        ef_per_atom = self._tensor(batch, "ef_per_atom")

        z_context = self.context_encoder(encoder1_features, encoder1_mask)
        with torch.no_grad():
            z_target = self.target_encoder(encoder2_features, encoder2_mask)
        z_pred = self.predictor(z_context, mask_box)

        loss_pred = self.pred_loss(z_pred, z_target.detach(), ef_per_atom)
        loss_reg = z_pred.new_tensor(0.0)
        if self.regularizer is not None and self.reg_weight > 0:
            reg_out = self.regularizer(z_pred)
            loss_reg = reg_out["loss"] if isinstance(reg_out, dict) else reg_out
        loss = loss_pred + self.reg_weight * loss_reg
        if not torch.isfinite(loss):
            raise ValueError("SC-JEPA loss is NaN or infinite")

        return {
            "loss": loss,
            "loss_pred": loss_pred,
            "loss_reg": loss_reg,
            "z_pred": z_pred,
            "z_target": z_target,
            "metrics": {
                "loss": float(loss.detach().cpu()),
                "loss_pred": float(loss_pred.detach().cpu()),
                "loss_reg": float(loss_reg.detach().cpu()),
            },
        }

    @torch.no_grad()
    def update_ema(self, decay: float | None = None) -> None:
        decay = self.ema_decay if decay is None else decay
        self._ema_module(self.target_encoder.atom_proj, self.context_encoder.atom_proj, decay)
        self._ema_parameter(self.target_encoder.cls_token, self.context_encoder.cls_token, decay)
        self._ema_parameter(self.target_encoder.pos_embedding, self.context_encoder.pos_embedding, decay)
        self._ema_module(self.target_encoder.transformer, self.context_encoder.transformer, decay)
        self._ema_module(self.target_encoder.norm, self.context_encoder.norm, decay)

    @staticmethod
    @torch.no_grad()
    def _ema_parameter(target: nn.Parameter, source: nn.Parameter, decay: float) -> None:
        if target.shape != source.shape:
            raise ValueError(f"EMA shape mismatch: {tuple(target.shape)} vs {tuple(source.shape)}")
        target.data.mul_(decay).add_(source.data, alpha=1.0 - decay)

    @staticmethod
    @torch.no_grad()
    def _ema_module(target: nn.Module, source: nn.Module, decay: float) -> None:
        target_state = target.state_dict()
        source_state = source.state_dict()
        for name, target_value in target_state.items():
            source_value = source_state[name]
            if target_value.shape != source_value.shape:
                raise ValueError(
                    f"EMA shape mismatch for {name}: "
                    f"{tuple(target_value.shape)} vs {tuple(source_value.shape)}"
                )
            if torch.is_floating_point(target_value):
                target_value.mul_(decay).add_(source_value, alpha=1.0 - decay)
            else:
                target_value.copy_(source_value)
```

- [ ] **Step 4: Run wrapper tests**

Run:

```bash
python -m pytest tests/test_sc_jepa_model.py -v
```

Expected: PASS.

- [ ] **Step 5: Optional local commit**

Run only if local commits are desired:

```bash
git add JEPA/crystal_jepa.py tests/test_sc_jepa_model.py
git commit -m "feat: add SC-JEPA wrapper with EMA updates"
```

Expected: local commit succeeds. Do not push.

---

### Task 5: Export Public SC-JEPA API

**Files:**
- Modify: `JEPA/__init__.py`
- Modify: `tests/test_sc_jepa_model.py`

- [ ] **Step 1: Add failing public import test**

Append this to `tests/test_sc_jepa_model.py`:

```python
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
```

- [ ] **Step 2: Run public import test**

Run:

```bash
python -m pytest tests/test_sc_jepa_model.py::test_sc_jepa_public_api_importable -v
```

Expected: FAIL because the new API is not exported.

- [ ] **Step 3: Update `JEPA/__init__.py` imports**

Add to the loss imports:

```python
    WeightedContrastiveLoss,
    weighted_contrastive_loss,
```

Add after the existing `from .jepa import ...` line:

```python
from .crystal_jepa import (
    CrystalJEPA,
    CrystalTransformerEncoder,
    MaskConditionedPredictor,
    TargetCrystalTransformerEncoder,
)
```

Add these names to `__all__`:

```python
    "CrystalJEPA",
    "CrystalTransformerEncoder",
    "MaskConditionedPredictor",
    "TargetCrystalTransformerEncoder",
    "WeightedContrastiveLoss",
    "weighted_contrastive_loss",
```

- [ ] **Step 4: Run public import test**

Run:

```bash
python -m pytest tests/test_sc_jepa_model.py::test_sc_jepa_public_api_importable -v
```

Expected: PASS.

- [ ] **Step 5: Optional local commit**

Run only if local commits are desired:

```bash
git add JEPA/__init__.py tests/test_sc_jepa_model.py
git commit -m "feat: export SC-JEPA training API"
```

Expected: local commit succeeds. Do not push.

---

### Task 6: Add Minimal Training Entrypoint

**Files:**
- Create: `train_sc_jepa.py`
- Create: `tests/test_sc_jepa_train.py`

- [ ] **Step 1: Write a failing training-step test with synthetic batch**

Create `tests/test_sc_jepa_train.py`:

```python
from pathlib import Path

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
```

- [ ] **Step 2: Run training-step test to verify it fails**

Run:

```bash
python -m pytest tests/test_sc_jepa_train.py -v
```

Expected: FAIL with import error for `train_sc_jepa`.

- [ ] **Step 3: Implement `train_sc_jepa.py`**

Create `train_sc_jepa.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from crystal_matrices import create_dataloader
from JEPA import (
    CrystalJEPA,
    CrystalTransformerEncoder,
    MaskConditionedPredictor,
    TargetCrystalTransformerEncoder,
)


def build_model(
    layers: int = 8,
    attn_heads: int = 16,
    dropout: float = 0.0,
    ema_decay: float = 0.996,
) -> CrystalJEPA:
    context_encoder = CrystalTransformerEncoder(
        input_dim=103,
        layers=layers,
        attn_heads=attn_heads,
        dropout=dropout,
    )
    target_encoder = TargetCrystalTransformerEncoder(
        layers=layers,
        attn_heads=attn_heads,
        dropout=dropout,
    )
    predictor = MaskConditionedPredictor()
    model = CrystalJEPA(
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        ema_decay=ema_decay,
    )
    model.update_ema(decay=0.0)
    return model


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def train_one_step(
    model: CrystalJEPA,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(batch)
    loss = out["loss"]
    loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
            grad_clip_norm,
        )
    optimizer.step()
    model.update_ema()
    return dict(out["metrics"])


def save_checkpoint(
    path: Path,
    model: CrystalJEPA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    dataloader = create_dataloader(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    model = build_model(
        layers=args.layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        ema_decay=args.ema_decay,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    global_step = 0
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch = move_batch_to_device(batch, device)
            metrics = train_one_step(model, batch, optimizer)
            global_step += 1
            if global_step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={metrics['loss']:.6f} loss_pred={metrics['loss_pred']:.6f}"
                )

    checkpoint_path = Path(args.checkpoint_path)
    save_checkpoint(checkpoint_path, model, optimizer, args.epochs - 1, global_step)
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SC-JEPA global embedding model")
    parser.add_argument("--csv-path", default="data/jepa/mp.csv.gz")
    parser.add_argument("--checkpoint-path", default="checkpoints/sc_jepa.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--attn-heads", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
```

- [ ] **Step 4: Run training-step test**

Run:

```bash
python -m pytest tests/test_sc_jepa_train.py -v
```

Expected: PASS.

- [ ] **Step 5: Optional local commit**

Run only if local commits are desired:

```bash
git add train_sc_jepa.py tests/test_sc_jepa_train.py
git commit -m "feat: add SC-JEPA training entrypoint"
```

Expected: local commit succeeds. Do not push.

---

### Task 7: Run Full Unit Suite and Smoke Train

**Files:**
- Read: `data/jepa/mp.csv.gz`
- Write: `checkpoints/sc_jepa_smoke.pt`

- [ ] **Step 1: Run focused test suite**

Run:

```bash
python -m pytest tests/test_crystal_matrices.py tests/test_sc_jepa_loss.py tests/test_sc_jepa_model.py tests/test_sc_jepa_train.py -v
```

Expected: PASS.

- [ ] **Step 2: Run a real-data smoke train**

Run:

```bash
python train_sc_jepa.py --device cpu --batch-size 4 --epochs 1 --max-batches 2 --layers 1 --attn-heads 8 --checkpoint-path checkpoints/sc_jepa_smoke.pt
```

Expected output includes finite `loss=` and `loss_pred=` values, and the command exits with status 0.

- [ ] **Step 3: Verify checkpoint exists**

Run:

```bash
python -c "from pathlib import Path; p=Path('checkpoints/sc_jepa_smoke.pt'); print(p.exists(), p.stat().st_size if p.exists() else 0)"
```

Expected output starts with `True` and a positive file size.

- [ ] **Step 4: Inspect git status without pushing**

Run:

```bash
git status --short
```

Expected: local modified/untracked files only. Do not push.

- [ ] **Step 5: Optional final local commit**

Run only if local commits are desired:

```bash
git add crystal_matrices.py JEPA/losses.py JEPA/crystal_jepa.py JEPA/__init__.py train_sc_jepa.py tests/test_crystal_matrices.py tests/test_sc_jepa_loss.py tests/test_sc_jepa_model.py tests/test_sc_jepa_train.py
git commit -m "feat: add SC-JEPA global embedding training"
```

Expected: local commit succeeds. Do not push.

---

## Self-Review

Spec coverage:

- Global embedding `[B, 512]`: Tasks 3 and 4.
- Encoder 1 `103` and Encoder 2 `110`: Tasks 3 and 4.
- Factorized target projection with fixed `family_proj_7`: Tasks 3 and 4.
- Predictor output `[B, 512]`: Task 3.
- Weighted contrastive Crys-JEPA loss with `ef_per_atom` and `temperature = 0.1`: Task 2.
- EMA target updates only for compatible parameters: Task 4.
- Spatial masking guarantees: Task 1.
- Minimal training loop with `max_batches` and checkpointing: Task 6.
- Smoke train: Task 7.

Placeholder scan:

- No `TBD`, `TODO`, or unspecified implementation steps.
- All tests include explicit code.
- All commands include expected outcomes.

Type consistency:

- Encoder outputs are `[B, 512]`.
- Predictor input and output are `[B, 512]`.
- Loss inputs are `context [B, 512]`, `target [B, 512]`, `ef_per_atom [B]`.
- Batch keys match the existing `collate_crystals` output.
