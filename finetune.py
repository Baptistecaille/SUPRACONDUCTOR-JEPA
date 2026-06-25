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
def predict_probs(
    model: SupraJEPA,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
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
    return probs, labels


def metrics_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics = {
        "auroc":     roc_auc_score(labels, probs),
        "ap":        average_precision_score(labels, probs),
        "f1":        f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "tnr":       tn / max(tn + fp, 1),   # true-negative-rate (BEE-NET metric)
        "tpr":       tp / max(tp + fn, 1),   # true-positive-rate / recall
        "threshold":  float(threshold),
    }
    return metrics


@torch.no_grad()
def evaluate(
    model: SupraJEPA,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    probs, labels = predict_probs(model, loader, device)
    return metrics_from_probs(probs, labels, threshold)


def choose_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    min_tnr: float | None = None,
) -> tuple[float, dict]:
    """Choisit un seuil sur validation.

    Si min_tnr est fourni, on garde le meilleur TPR parmi les seuils qui
    atteignent ce TNR. Sinon, on maximise F1.
    """
    thresholds = np.unique(np.concatenate(([0.0, 1.0], probs)))
    best_threshold = 0.5
    best_metrics = metrics_from_probs(probs, labels, best_threshold)
    best_score = (-1.0, -1.0, -1.0)

    for threshold in thresholds:
        metrics = metrics_from_probs(probs, labels, float(threshold))
        if min_tnr is not None and metrics["tnr"] < min_tnr:
            continue
        if min_tnr is None:
            score = (metrics["f1"], metrics["tnr"], metrics["tpr"])
        else:
            score = (metrics["tpr"], metrics["f1"], metrics["tnr"])
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def _print_metrics(title: str, metrics: dict):
    print(title)
    for k, v in metrics.items():
        print(f"  {k:12s} : {v:.4f}")


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

    if hasattr(train_loader.dataset, "labels"):
        all_train_labels = train_loader.dataset.labels
    pos_weight = compute_pos_weight(all_train_labels, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)

    os.makedirs(os.path.dirname(cfg.checkpoint_finetune), exist_ok=True)
    best_score = (-1.0, -1.0, -1.0, float("-inf"))

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

        score = (
            round(val_metrics["auroc"], 6),
            round(val_metrics["f1"], 6),
            round(val_metrics["tnr"], 6),
            -avg_loss,
        )
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), cfg.checkpoint_finetune)
            print(
                "  ✓ checkpoint sauvegardé "
                f"(AUROC={val_metrics['auroc']:.4f}, "
                f"F1={val_metrics['f1']:.4f}, TNR={val_metrics['tnr']:.4f})"
            )

    # ── Évaluation finale sur test set ───────────────────────────────────────
    model.load_state_dict(torch.load(cfg.checkpoint_finetune, map_location=device))

    val_probs, val_labels = predict_probs(model, val_loader, device)
    best_f1_threshold, best_f1_val_metrics = choose_threshold(val_probs, val_labels)

    threshold_path = os.path.join(os.path.dirname(cfg.checkpoint_finetune), "thresholds.json")
    thresholds = {"best_f1": best_f1_threshold}
    for target_tnr in (0.95, 0.99):
        threshold, metrics = choose_threshold(val_probs, val_labels, min_tnr=target_tnr)
        thresholds[f"tnr_{target_tnr:.2f}"] = threshold
        print(
            f"Seuil validation TNR≥{target_tnr:.2f}: {threshold:.4f} "
            f"(TNR={metrics['tnr']:.4f}, TPR={metrics['tpr']:.4f}, F1={metrics['f1']:.4f})"
        )

    import json
    with open(threshold_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"✓ Seuils sauvegardés dans {threshold_path}")

    print("\n── Calibration seuil validation (best F1) ──")
    _print_metrics("", best_f1_val_metrics)

    print("\n── Évaluation finale (test set, seuil=0.5) ──")
    test_probs, test_labels = predict_probs(model, test_loader, device)
    _print_metrics("", metrics_from_probs(test_probs, test_labels, threshold=0.5))

    print("\n── Évaluation finale (test set, seuil validation best F1) ──")
    _print_metrics("", metrics_from_probs(test_probs, test_labels, threshold=best_f1_threshold))

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

    train_labels = train_dl.dataset.labels

    model = SupraJEPA(cfg)
    finetune(model, train_dl, val_dl, test_dl, train_labels, cfg, device)
