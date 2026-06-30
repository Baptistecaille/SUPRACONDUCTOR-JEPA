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
                    f"loss={metrics['loss']:.6f} "
                    f"loss_pred={metrics['loss_pred']:.6f}"
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
