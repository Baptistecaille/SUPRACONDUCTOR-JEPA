"""
model.py — Architecture SUPRA-JEPA

Trois modules :
  1. CrystalEmbedding   : tokens cristallins → vecteurs continus
  2. CrystalEncoder     : Transformer encodeur (contexte + target via EMA)
  3. JEPAPredictor      : prédit les embeddings masqués depuis le contexte
  4. SupraClassifier    : tête de classification binaire (fine-tuning)

Principe JEPA (LeCun 2022, Assran et al. 2023 I-JEPA) :
  - Context encoder  θ  : traite les tokens visibles
  - Target encoder   θ̄  : EMA de θ, traite tous les tokens (stop-gradient)
  - Predictor        φ  : prédit ẑ_target = f_φ(z_context, pos_mask)
  - Loss             ℒ  : ||ẑ_target - sg(z_target)||²  en espace normalisé
                          (Barlow-style : pas de négatives)
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


# ── 1. Embedding des tokens cristallins ──────────────────────────────────────

class CrystalEmbedding(nn.Module):
    """Chaque position de la séquence → vecteur d_model.

    Position 0  = token CLS : encode sg_id + paramètres de maille
    Positions 1… = tokens atomiques : element + wyckoff + coords fractionnelles

    Inspired by S2SNet (Liu et al. 2023) :
      atom_embedding = embed(element) + embed(wyckoff) + proj(coords)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model

        # Tokens atomiques
        self.element_emb  = nn.Embedding(cfg.n_elements,    d, padding_idx=0)
        self.wyckoff_emb  = nn.Embedding(cfg.n_wyckoff,     d, padding_idx=0)
        self.coords_proj  = nn.Linear(3, d, bias=False)

        # Token CLS (global crystal)
        self.sg_emb       = nn.Embedding(cfg.n_space_groups, d, padding_idx=0)
        self.lattice_proj = nn.Linear(6, d, bias=False)

        # Token de masque appris (remplace les tokens masqués dans le predictor)
        self.mask_token   = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        element_ids:  torch.Tensor,   # (B, L)
        wyckoff_ids:  torch.Tensor,   # (B, L)
        frac_coords:  torch.Tensor,   # (B, L, 3)
        sg_id:        torch.Tensor,   # (B,)
        lattice_feat: torch.Tensor,   # (B, 6)
        mask:         torch.Tensor | None = None,   # (B, L) bool — positions masquées
        use_mask_token: bool = False,
    ) -> torch.Tensor:                # (B, L, d_model)

        B, L = element_ids.shape

        # Tokens atomiques (positions 1…L-1 ; position 0 = CLS)
        x = (
            self.element_emb(element_ids)
            + self.wyckoff_emb(wyckoff_ids)
            + self.coords_proj(frac_coords)
        )  # (B, L, d)

        # Surcharge position 0 avec infos globales (CLS)
        cls_vec = self.sg_emb(sg_id).unsqueeze(1) + self.lattice_proj(lattice_feat).unsqueeze(1)
        x[:, 0:1, :] = cls_vec

        # Remplacement des positions masquées par le mask_token appris
        if use_mask_token and mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            mask_tok = self.mask_token.expand(B, L, -1)
            x = torch.where(mask_expanded, mask_tok, x)

        return self.drop(self.norm(x))


# ── 2. Encodeur Transformer ────────────────────────────────────────────────────

class CrystalEncoder(nn.Module):
    """Transformer standard sur la séquence de tokens cristallins.

    Utilisé deux fois :
      - context encoder θ  : reçoit uniquement les tokens NON masqués
      - target encoder  θ̄  : reçoit tous les tokens (EMA de θ, stop-gradient)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm (plus stable, recommandé)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers_encoder
        )

    def forward(
        self,
        x: torch.Tensor,              # (B, L, d_model)
        padding_mask: torch.Tensor,   # (B, L) bool — True = position valide
    ) -> torch.Tensor:                # (B, L, d_model)
        # PyTorch TransformerEncoder attend src_key_padding_mask=True là où on ignore
        key_pad = ~padding_mask       # True = ignorer (PAD)
        return self.transformer(x, src_key_padding_mask=key_pad)


# ── 3. Predictor JEPA ─────────────────────────────────────────────────────────

class JEPAPredictor(nn.Module):
    """Prédit les embeddings cibles (masqués) depuis le contexte encodé.

    Architecture : petit Transformer (n_layers_predictor couches).
    Entrée : tokens de contexte + mask_tokens aux positions masquées.
    Sortie : représentations prédites pour les positions masquées uniquement.

    Référence : I-JEPA (Assran et al. 2023), Graph-JEPA (Skenderi et al. 2025)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        pred_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 2,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            pred_layer, num_layers=cfg.n_layers_predictor
        )
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(
        self,
        context_enc: torch.Tensor,    # (B, L, d) — sortie du context encoder
        padding_mask: torch.Tensor,   # (B, L) bool
    ) -> torch.Tensor:                # (B, L, d)
        key_pad = ~padding_mask
        x = self.transformer(context_enc, src_key_padding_mask=key_pad)
        return self.out_proj(x)


# ── 4. Tête de classification (fine-tuning) ───────────────────────────────────

class SupraClassifier(nn.Module):
    """Classifieur binaire sur le token CLS.

    Utilisé en fine-tuning sur labels {0=non-SC, 1=SC}.
    BCEWithLogitsLoss avec pos_weight pour le déséquilibre de classes.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """cls_token : (B, d_model) — premier token de la sortie encodeur."""
        return self.head(cls_token).squeeze(-1)   # (B,) logits


# ── 5. Modèle complet SUPRA-JEPA ─────────────────────────────────────────────

class SupraJEPA(nn.Module):
    """Modèle complet : embedding + context encoder + EMA target + predictor.

    Usage :
      Prétraining  : forward_pretrain(batch) → loss JEPA
      Fine-tuning  : forward_classify(batch) → logits
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.embedding      = CrystalEmbedding(cfg)
        self.context_encoder = CrystalEncoder(cfg)
        self.target_encoder  = copy.deepcopy(self.context_encoder)
        self.predictor       = JEPAPredictor(cfg)
        self.classifier      = SupraClassifier(cfg)

        # Le target encoder ne reçoit pas de gradients directs
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    @torch.no_grad()
    def ema_update(self, decay: float | None = None):
        """Mise à jour EMA du target encoder (θ̄ ← τ θ̄ + (1-τ) θ)."""
        tau = decay if decay is not None else self.cfg.ema_decay
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_tgt.data.mul_(tau).add_(p_ctx.data, alpha=1.0 - tau)

    # ── Prétraining ───────────────────────────────────────────────────────────

    def forward_pretrain(self, batch: dict) -> torch.Tensor:
        """Loss JEPA : MSE entre prédiction et cible EMA normalisée.

        Seules les positions masquées contribuent à la loss
        (comme I-JEPA, pas comme MAE qui reconstruit tous les tokens).
        """
        element_ids  = batch["element_ids"]
        wyckoff_ids  = batch["wyckoff_ids"]
        frac_coords  = batch["frac_coords"]
        sg_id        = batch["sg_id"]
        lattice_feat = batch["lattice_feat"]
        padding_mask = batch["padding_mask"]
        mask         = batch["mask"]          # (B, L) — positions masquées

        # ── Target : encodage COMPLET, stop-gradient ─────────────────────────
        with torch.no_grad():
            x_full = self.embedding(
                element_ids, wyckoff_ids, frac_coords, sg_id, lattice_feat,
                mask=None, use_mask_token=False,
            )
            z_target = self.target_encoder(x_full, padding_mask)   # (B, L, d)

            if self.cfg.normalize_targets:
                z_target = F.normalize(z_target, dim=-1)

        # ── Contexte : tokens masqués remplacés par mask_token ────────────────
        x_ctx = self.embedding(
            element_ids, wyckoff_ids, frac_coords, sg_id, lattice_feat,
            mask=mask, use_mask_token=True,
        )
        # On passe TOUTE la séquence (avec mask_tokens) à l'encodeur de contexte
        z_ctx = self.context_encoder(x_ctx, padding_mask)          # (B, L, d)

        # ── Prédiction ────────────────────────────────────────────────────────
        z_pred = self.predictor(z_ctx, padding_mask)               # (B, L, d)

        if self.cfg.normalize_targets:
            z_pred = F.normalize(z_pred, dim=-1)

        # ── Loss sur les positions masquées et valides uniquement ─────────────
        active = mask & padding_mask                               # (B, L)
        loss = F.mse_loss(z_pred[active], z_target[active].detach())
        return loss

    # ── Fine-tuning ───────────────────────────────────────────────────────────

    def forward_classify(self, batch: dict) -> torch.Tensor:
        """Retourne les logits de classification binaire (B,)."""
        element_ids  = batch["element_ids"]
        wyckoff_ids  = batch["wyckoff_ids"]
        frac_coords  = batch["frac_coords"]
        sg_id        = batch["sg_id"]
        lattice_feat = batch["lattice_feat"]
        padding_mask = batch["padding_mask"]

        x = self.embedding(
            element_ids, wyckoff_ids, frac_coords, sg_id, lattice_feat,
            mask=None, use_mask_token=False,
        )
        z = self.context_encoder(x, padding_mask)   # (B, L, d)
        cls_token = z[:, 0, :]                       # token CLS = position 0
        return self.classifier(cls_token)
