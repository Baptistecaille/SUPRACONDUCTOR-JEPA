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

    def _validate_inputs(
        self, atom_features: torch.Tensor, atom_mask: torch.Tensor
    ) -> None:
        if atom_features.dim() != 3:
            raise ValueError(
                f"atom_features must be [B, N, F], got {tuple(atom_features.shape)}"
            )
        if atom_features.size(-1) != self.input_dim:
            raise ValueError(
                f"expected feature dim {self.input_dim}, got {atom_features.size(-1)}"
            )
        if atom_mask.shape != atom_features.shape[:2]:
            raise ValueError(
                f"atom_mask shape {tuple(atom_mask.shape)} must match token shape "
                f"{tuple(atom_features.shape[:2])}"
            )
        if not atom_mask.any(dim=1).all():
            raise ValueError("each crystal must contain at least one visible atom")
        if atom_features.size(1) > self.max_atoms:
            raise ValueError(
                f"sequence length {atom_features.size(1)} exceeds max_atoms "
                f"{self.max_atoms}"
            )

    def project_tokens(self, atom_features: torch.Tensor) -> torch.Tensor:
        return self.atom_proj(atom_features)

    def forward(
        self, atom_features: torch.Tensor, atom_mask: torch.Tensor
    ) -> torch.Tensor:
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

    def _validate_inputs(
        self, atom_features: torch.Tensor, atom_mask: torch.Tensor
    ) -> None:
        if atom_features.dim() != 3:
            raise ValueError(
                f"atom_features must be [B, N, F], got {tuple(atom_features.shape)}"
            )
        if atom_features.size(-1) != 110:
            raise ValueError(
                f"target encoder expects feature dim 110, got {atom_features.size(-1)}"
            )
        if atom_mask.shape != atom_features.shape[:2]:
            raise ValueError(
                f"atom_mask shape {tuple(atom_mask.shape)} must match token shape "
                f"{tuple(atom_features.shape[:2])}"
            )
        if not atom_mask.any(dim=1).all():
            raise ValueError("each crystal must contain at least one atom")
        if atom_features.size(1) > self.max_atoms:
            raise ValueError(
                f"sequence length {atom_features.size(1)} exceeds max_atoms "
                f"{self.max_atoms}"
            )

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

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | dict[str, float]]:
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
        self._ema_parameter(
            self.target_encoder.pos_embedding, self.context_encoder.pos_embedding, decay
        )
        self._ema_module(self.target_encoder.transformer, self.context_encoder.transformer, decay)
        self._ema_module(self.target_encoder.norm, self.context_encoder.norm, decay)

    @staticmethod
    @torch.no_grad()
    def _ema_parameter(target: nn.Parameter, source: nn.Parameter, decay: float) -> None:
        if target.shape != source.shape:
            raise ValueError(
                f"EMA shape mismatch: {tuple(target.shape)} vs {tuple(source.shape)}"
            )
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
