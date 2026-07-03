from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from crystal_matrices import create_dataloader
from JEPA import (
    CosineWithWarmup,
    CrystalJEPA,
    CrystalTransformerEncoder,
    MaskConditionedPredictor,
    TargetCrystalTransformerEncoder,
    VCLoss,
)


def build_model(
    layers: int = 8,
    attn_heads: int = 16,
    dropout: float = 0.0,
    ema_decay: float = 0.996,
    regularizer: str = "none",
    reg_weight: float = 0.0,
    reg_std_coeff: float = 1.0,
    reg_cov_coeff: float = 1.0,
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
    if regularizer == "none":
        regularizer_module = None
    elif regularizer == "vc":
        regularizer_module = VCLoss(std_coeff=reg_std_coeff, cov_coeff=reg_cov_coeff)
    else:
        raise ValueError(f"unknown regularizer: {regularizer}")
    model = CrystalJEPA(
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        regularizer=regularizer_module,
        reg_weight=reg_weight,
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
    scheduler: Any | None = None,
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
    if scheduler is not None:
        scheduler.step()
    model.update_ema()
    return dict(out["metrics"])


def save_checkpoint(
    path: Path,
    model: CrystalJEPA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    scheduler: Any | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {} if config is None else config,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def estimate_total_steps(
    dataloader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
) -> int:
    batches_per_epoch = len(dataloader)
    if args.max_batches is not None:
        batches_per_epoch = min(batches_per_epoch, args.max_batches)
    return max(1, batches_per_epoch * args.epochs)


def train(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    dataloader = create_dataloader(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        deduplicate=args.deduplicate,
        drop_last=True,
    )
    model = build_model(
        layers=args.layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        ema_decay=args.ema_decay,
        regularizer=args.regularizer,
        reg_weight=args.reg_weight,
        reg_std_coeff=args.reg_std_coeff,
        reg_cov_coeff=args.reg_cov_coeff,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineWithWarmup(
            optimizer,
            total_steps=estimate_total_steps(dataloader, args),
            warmup_ratio=args.warmup_ratio,
            min_lr=args.min_lr,
        )

    global_step = 0
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch = move_batch_to_device(batch, device)
            metrics = train_one_step(model, batch, optimizer, scheduler=scheduler)
            global_step += 1
            if global_step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={metrics['loss']:.6f} "
                    f"loss_pred={metrics['loss_pred']:.6f} "
                    f"loss_reg={metrics['loss_reg']:.6f} "
                    f"lr={lr:.6g}"
                )

    checkpoint_path = Path(args.checkpoint_path)
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        args.epochs - 1,
        global_step,
        scheduler=scheduler,
        config=vars(args),
    )
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
    parser.add_argument(
        "--no-deduplicate",
        dest="deduplicate",
        action="store_false",
        help=(
            "Disable material_id deduplication. On by default: data/jepa/mp.csv.gz "
            "is deduplicated at the source, but this flag defends against a future "
            "unclean CSV without requiring the caller to remember --deduplicate."
        ),
    )
    parser.set_defaults(deduplicate=True)
    parser.add_argument("--regularizer", choices=["none", "vc"], default="none")
    parser.add_argument("--reg-weight", type=float, default=0.0)
    parser.add_argument("--reg-std-coeff", type=float, default=1.0)
    parser.add_argument("--reg-cov-coeff", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
