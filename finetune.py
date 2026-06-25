"""
finetune.py — Fine-tuning pour classification binaire supraconducteur / non-SC

Stratégie :
  1. Charger le contexte encoder prétrainé
  2. Fine-tuner en full (encoder + tête) avec lr basse
  3. Gérer le déséquilibre via BCEWithLogitsLoss(pos_weight)
  4. Métriques : AUROC, F1, précision, rappel (pertinent pour données déséquilibrées)

Référence BEE-NET (NeurIPS ML4PS 2024) : true-negative-rate de 99.4% est la
cible sur ce type de tâche (screening de supraconducteurs rares).
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix,
)
import numpy as np

from config import Config
from model import SupraJEPA


def compute_pos_weight(labels: list[int], device: torch.device) -> torch.Tensor:
    """pos_weight = n_négatifs / n_positifs (pour BCEWithLogitsLoss)."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    w = n_neg / max(n_pos, 1)
    print(f"  pos_weight calculé : {w:.1f}  (n_pos={n_pos}, n_neg={n_neg})")
    return torch.tensor([w], device=device)


@torch.no_grad()
def evaluate(model: SupraJEPA, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model.forward_classify(batch)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"].cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1 / (1 + np.exp(-logits))   # sigmoid
    preds  = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics = {
        "auroc":     roc_auc_score(labels, probs),
        "ap":        average_precision_score(labels, probs),
        "f1":        f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "tnr":       tn / max(tn + fp, 1),   # true-negative-rate (BEE-NET metric)
        "tpr":       tp / max(tp + fn, 1),   # true-positive-rate / recall
    }
    return metrics


def finetune(
    model: SupraJEPA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    all_train_labels: list[int],
    cfg: Config,
    device: torch.device,
) -> SupraJEPA:

    model = model.to(device)

    # Charge les poids prétrainés si disponibles
    if os.path.exists(cfg.checkpoint_pretrain):
        state = torch.load(cfg.checkpoint_pretrain, map_location=device)
        # On charge uniquement context_encoder + embedding (pas le classifier)
        model_keys = {k: v for k, v in state.items() if "classifier" not in k}
        model.load_state_dict(model_keys, strict=False)
        print(f"✓ Poids prétrainés chargés depuis {cfg.checkpoint_pretrain}")
    else:
        print("⚠ Pas de checkpoint prétrainé trouvé — fine-tuning from scratch")

    pos_weight = compute_pos_weight(all_train_labels, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)

    os.makedirs(os.path.dirname(cfg.checkpoint_finetune), exist_ok=True)
    best_auroc = 0.0

    for epoch in range(cfg.epochs_finetune):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            labels = batch["label"].float()

            optimizer.zero_grad()
            logits = model.forward_classify(batch)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"[Finetune] Epoch {epoch+1:03d}/{cfg.epochs_finetune} | "
            f"loss={avg_loss:.4f} | "
            f"AUROC={val_metrics['auroc']:.4f} | "
            f"F1={val_metrics['f1']:.4f} | "
            f"TNR={val_metrics['tnr']:.4f} | "
            f"TPR={val_metrics['tpr']:.4f}"
        )

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), cfg.checkpoint_finetune)
            print(f"  ✓ checkpoint sauvegardé (AUROC={best_auroc:.4f})")

    # ── Évaluation finale sur test set ───────────────────────────────────────
    print("\n── Évaluation finale (test set) ──")
    model.load_state_dict(torch.load(cfg.checkpoint_finetune, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    for k, v in test_metrics.items():
        print(f"  {k:12s} : {v:.4f}")

    return model


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data import load_supercon_dataset, make_dataloaders

    torch.manual_seed(42)
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    structures, labels = load_supercon_dataset(cfg.data_dir)
    train_dl, val_dl, test_dl = make_dataloaders(structures, labels, cfg)

    # Labels d'entraînement pour calculer pos_weight
    n     = len(structures)
    n_val = int(n * 0.15)
    n_tst = int(n * 0.15)
    train_labels = labels[n_val + n_tst:]

    model = SupraJEPA(cfg)
    finetune(model, train_dl, val_dl, test_dl, train_labels, cfg, device)
