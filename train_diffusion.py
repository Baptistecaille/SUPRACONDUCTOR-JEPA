"""
train_diffusion.py — Entraîne le modèle de diffusion cristallin.

Exemples :
  python train_diffusion.py
  python train_diffusion.py --positives-only
"""

import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import Config
from data import CrystalDataset, load_supercon_dataset
from diffusion import CrystalDDPM, batch_to_diffusion_vector


def train_diffusion(
    model: CrystalDDPM,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> CrystalDDPM:
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr_diffusion, weight_decay=cfg.weight_decay)

    os.makedirs(os.path.dirname(cfg.checkpoint_diffusion), exist_ok=True)
    best_loss = float("inf")

    for epoch in range(cfg.epochs_diffusion):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            x_0 = batch_to_diffusion_vector(batch, cfg)

            optimizer.zero_grad()
            loss = model.loss(x_0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(
            f"[Diffusion] Epoch {epoch + 1:03d}/{cfg.epochs_diffusion} | "
            f"loss={avg_loss:.4f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "best_loss": best_loss,
                },
                cfg.checkpoint_diffusion,
            )
            print(f"  checkpoint sauvegardé (loss={best_loss:.4f})")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positives-only",
        action="store_true",
        help="entraîne le générateur uniquement sur les supraconducteurs connus",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(42)
    cfg = Config()
    if args.epochs is not None:
        cfg.epochs_diffusion = args.epochs
    if args.batch_size is not None:
        cfg.batch_size_diffusion = args.batch_size
    if args.checkpoint is not None:
        cfg.checkpoint_diffusion = args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    structures, labels = load_supercon_dataset(cfg.data_dir)
    if args.positives_only:
        pairs = [(s, y) for s, y in zip(structures, labels) if y == 1]
        structures = [s for s, _ in pairs]
        labels = [y for _, y in pairs]
        print(f"Entraînement diffusion sur positifs uniquement : {len(structures)} structures")
    else:
        print(f"Entraînement diffusion sur toutes les structures : {len(structures)} structures")

    dataset = CrystalDataset(structures, labels, cfg.max_atoms, mask_ratio=None)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size_diffusion,
        shuffle=True,
        num_workers=2,
    )

    model = CrystalDDPM(cfg)
    train_diffusion(model, loader, cfg, device)


if __name__ == "__main__":
    main()
