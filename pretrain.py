"""
pretrain.py — Boucle de prétraining JEPA

Loss : MSE dans l'espace latent normalisé entre la prédiction du predictor
       et les embeddings du target encoder (EMA, stop-gradient).

Pas de paires négatives. Pas de reconstruction pixel-level.
Stabilité assurée par :
  - Normalisation des embeddings cibles (normalize_targets=True)
  - EMA du target encoder (collapse impossible : le target bouge lentement)
  - Warmup du learning rate

Références :
  - I-JEPA (Assran et al. 2023) pour le schéma EMA + predictor
  - Barlow Twins (Zbontar et al. 2021) pour l'esprit non-contrastif
"""

import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import Config
from model import SupraJEPA


def get_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """LR warmup linéaire puis décroissance cosinus."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def ema_decay_schedule(step: int, total_steps: int, base: float = 0.996) -> float:
    """EMA decay augmente progressivement vers 1.0 (comme dans I-JEPA)."""
    return 1.0 - (1.0 - base) * (math.cos(math.pi * step / total_steps) + 1) / 2


def pretrain(
    model: SupraJEPA,
    train_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> SupraJEPA:

    model = model.to(device)
    model.train()

    # Le target encoder est frozen (mis à jour par EMA uniquement)
    params = [
        p for n, p in model.named_parameters()
        if "target_encoder" not in n
    ]
    optimizer = AdamW(params, lr=cfg.lr_pretrain, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs_pretrain * len(train_loader)
    scheduler = get_warmup_cosine_scheduler(optimizer, cfg.warmup_steps, total_steps)

    os.makedirs(os.path.dirname(cfg.checkpoint_pretrain), exist_ok=True)
    best_loss = float("inf")
    global_step = 0

    for epoch in range(cfg.epochs_pretrain):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            loss = model.forward_pretrain(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # EMA update avec decay schedule
            decay = ema_decay_schedule(global_step, total_steps, cfg.ema_decay)
            model.ema_update(decay)

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        lr_now = scheduler.get_last_lr()[0]
        print(f"[Pretrain] Epoch {epoch+1:03d}/{cfg.epochs_pretrain} | "
              f"loss={avg_loss:.4f} | lr={lr_now:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), cfg.checkpoint_pretrain)
            print(f"  ✓ checkpoint sauvegardé (loss={best_loss:.4f})")

    print(f"\nPrétraining terminé. Meilleure loss : {best_loss:.4f}")
    return model


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data import load_supercon_dataset, CrystalDataset
    from torch.utils.data import DataLoader
    import random

    torch.manual_seed(42)
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Chargement des données (structures non labelisées pour le prétraining)
    structures, labels = load_supercon_dataset(cfg.data_dir)
    # Pour le prétraining : on ignore les labels, on masque tout
    dataset = CrystalDataset(structures, [-1] * len(structures), cfg.max_atoms, cfg.mask_ratio)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size_pretrain, shuffle=True, num_workers=4)

    model = SupraJEPA(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables : {n_params:,}")

    pretrain(model, loader, cfg, device)
