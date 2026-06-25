from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Tokenisation ──────────────────────────────────────────────────────────
    max_atoms: int = 64          # atoms per unit cell (padding / truncation)
    n_elements: int = 119        # 0 = PAD, 1-118 = H→Og
    n_wyckoff: int = 27          # 0 = PAD, 1-26 = a→z
    n_space_groups: int = 231    # 1-230 (0 = PAD)

    # ── Architecture ──────────────────────────────────────────────────────────
    d_model: int = 256
    n_heads: int = 8
    n_layers_encoder: int = 6    # context encoder (and EMA target encoder)
    n_layers_predictor: int = 2  # lightweight predictor
    ffn_dim: int = 1024
    dropout: float = 0.1

    # ── JEPA prétraining ──────────────────────────────────────────────────────
    mask_ratio: float = 0.40     # fraction of atom tokens masqués
    ema_decay: float = 0.996     # momentum du target encoder (EMA)
    # Normalisation des embeddings cibles avant loss (comme VICReg / Barlow)
    normalize_targets: bool = True

    # ── Prétraining ───────────────────────────────────────────────────────────
    epochs_pretrain: int = 100
    batch_size_pretrain: int = 64
    lr_pretrain: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 1e-4

    # ── Fine-tuning ───────────────────────────────────────────────────────────
    epochs_finetune: int = 50
    batch_size_finetune: int = 32
    lr_finetune: float = 2e-5
    # Déséquilibre de classes : ~16k SC vs ~600k non-SC
    # pos_weight = n_neg / n_pos ≈ 37.5 (recalculé dynamiquement dans train)
    pos_weight: float = 37.5

    # ── Chemins ───────────────────────────────────────────────────────────────
    data_dir: str = "data"
    checkpoint_pretrain: str = "checkpoints/jepa_pretrained.pt"
    checkpoint_finetune: str = "checkpoints/jepa_finetune.pt"

    # ── Reproductibilité ──────────────────────────────────────────────────────
    seed: int = 42
