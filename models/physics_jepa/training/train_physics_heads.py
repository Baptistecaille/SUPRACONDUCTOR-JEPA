"""Training entry point for stage-3 physics-head fine-tuning
(`lambda_ep` + `omega_log`) on top of a pretrained `foundation_jepa`
checkpoint.

Mirrors `models/foundation_jepa/training/train_foundation_jepa.py` /
`models/property_jepa/training/train_property_jepa.py` in overall shape
(`build_model` / `train_one_step` / `evaluate` / `save_checkpoint` /
`train` / CLI). Reads `data/processed/physics_jepa_{train,val,test}.csv.gz`
(`scripts/build_physics_dataset.py`) instead of the pretraining split, and
loads the `foundation_jepa` checkpoint produced by
`train_foundation_jepa.py` (`models/checkpoints/foundation_jepa.pt` by
default) to initialize the encoder before attaching `PhysicsHeads`.

Usage:
    python -m models.physics_jepa.training.train_physics_heads \\
        --foundation-checkpoint models/checkpoints/foundation_jepa.pt \\
        --train-csv data/processed/physics_jepa_train.csv.gz \\
        --val-csv data/processed/physics_jepa_val.csv.gz \\
        --checkpoint-path models/checkpoints/physics_jepa.pt \\
        --report-path docs/audit/physics_jepa_finetune_report.json \\
        --epochs 30 --batch-size 32 --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from ...foundation_jepa.data.lattice import MatrixMeanStdScaler
from ...foundation_jepa.model.jepa import FoundationJEPA
from ..data.dataset import PhysicsTargetStats, create_physics_dataloader
from ..model.heads import PhysicsHeads
from ..model.losses import PhysicsHeadsLoss
from ..model.physics_utils import allen_dynes_tc
from .schedulers import CosineWithWarmup


def load_foundation_model(checkpoint_path: str | Path, device: torch.device) -> tuple[FoundationJEPA, dict]:
    """Load a `foundation_jepa` checkpoint (see `train_foundation_jepa.py::save_checkpoint`)
    and rebuild the `FoundationJEPA` + its fitted `matrix_scaler` from the
    saved config + tensors."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    matrix_scaler = MatrixMeanStdScaler(
        mean=checkpoint["matrix_scaler_mean"], std=checkpoint["matrix_scaler_std"]
    )
    model = FoundationJEPA(
        hidden_dim=config["hidden_dim"],
        layers=config["layers"],
        attn_heads=config["attn_heads"],
        dropout=config.get("dropout", 0.0),
        max_atoms=config["max_atoms"],
        temperature=config.get("temperature", 0.1),
        reg_weight=config.get("reg_weight", 0.01),
        matrix_scaler=matrix_scaler,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), config


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def train_one_step(
    heads: PhysicsHeads,
    loss_fn: PhysicsHeadsLoss,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    grad_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    heads.train()
    optimizer.zero_grad(set_to_none=True)
    out = heads(
        batch["frac_coords"],
        batch["atomic_numbers"],
        batch["raw_lattice_matrix"],
        batch["num_atoms"],
        batch["atom_mask"],
    )
    loss_out = loss_fn(out.lambda_pred_norm, out.omega_pred_norm, batch["lambda_ep"], batch["omega_log"])
    loss_out.loss.backward()
    if grad_clip_norm is not None:
        trainable = [p for p in heads.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {
        "loss": loss_out.loss.item(),
        "loss_lambda": loss_out.loss_lambda.item(),
        "loss_omega": loss_out.loss_omega.item(),
        "loss_validity": loss_out.loss_validity.item(),
    }


@torch.no_grad()
def evaluate(
    heads: PhysicsHeads,
    loss_fn: PhysicsHeadsLoss,
    dataloader,
    device: torch.device,
    stats: PhysicsTargetStats,
) -> dict[str, float]:
    """Loss metrics (normalized space) + physical-unit MAE for `lambda_ep`,
    `omega_log`, and the Allen-Dynes Tc these two jointly imply."""
    heads.eval()
    totals = {"loss": 0.0, "loss_lambda": 0.0, "loss_omega": 0.0, "loss_validity": 0.0}
    abs_err = {"lambda_ep": 0.0, "omega_log": 0.0, "tc": 0.0}
    n_batches = 0
    n_samples = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        out = heads(
            batch["frac_coords"],
            batch["atomic_numbers"],
            batch["raw_lattice_matrix"],
            batch["num_atoms"],
            batch["atom_mask"],
        )
        loss_out = loss_fn(out.lambda_pred_norm, out.omega_pred_norm, batch["lambda_ep"], batch["omega_log"])
        for k in totals:
            totals[k] += getattr(loss_out, k).item()
        n_batches += 1

        lambda_pred, omega_pred = stats.denormalize(out.lambda_pred_norm, out.omega_pred_norm)
        abs_err["lambda_ep"] += (lambda_pred - batch["lambda_ep"]).abs().sum().item()
        abs_err["omega_log"] += (omega_pred - batch["omega_log"]).abs().sum().item()
        tc_pred = allen_dynes_tc(lambda_pred, omega_pred)
        tc_target = allen_dynes_tc(batch["lambda_ep"], batch["omega_log"])
        abs_err["tc"] += (tc_pred - tc_target).abs().sum().item()
        n_samples += batch["lambda_ep"].shape[0]

    if n_batches == 0:
        return {}
    result = {k: v / n_batches for k, v in totals.items()}
    result.update({f"mae_{k}": v / max(1, n_samples) for k, v in abs_err.items()})
    return result


def save_checkpoint(
    path: Path,
    heads: PhysicsHeads,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    stats: PhysicsTargetStats,
    scheduler: Any | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "heads_state_dict": heads.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "target_stats": stats.to_dict(),
        "config": {} if config is None else config,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    t_start = time.time()

    print(f"[physics_jepa] loading foundation_jepa checkpoint from {args.foundation_checkpoint} ...")
    foundation_model, foundation_config = load_foundation_model(args.foundation_checkpoint, device)

    print(f"[physics_jepa] fitting target stats from {args.train_csv} ...")
    stats = PhysicsTargetStats.fit(args.train_csv)

    print(f"[physics_jepa] loading + parsing up to {args.max_train_rows} train rows ...")
    train_loader = create_physics_dataloader(
        csv_path=args.train_csv,
        batch_size=args.batch_size,
        shuffle=True,
        max_atoms=foundation_config["max_atoms"],
        max_rows=args.max_train_rows,
        drop_last=True,
        cache_parsed=True,
    )
    val_loader = None
    if args.val_csv is not None:
        print(f"[physics_jepa] loading + parsing up to {args.max_val_rows} val rows ...")
        val_loader = create_physics_dataloader(
            csv_path=args.val_csv,
            batch_size=args.batch_size,
            shuffle=False,
            max_atoms=foundation_config["max_atoms"],
            max_rows=args.max_val_rows,
            drop_last=False,
            cache_parsed=True,
        )

    heads = PhysicsHeads(
        foundation_model, hidden_dim=args.head_hidden_dim, freeze_encoder=args.freeze_encoder
    ).to(device)
    loss_fn = PhysicsHeadsLoss(
        stats, omega_weight=args.omega_weight, validity_weight=args.validity_weight
    )

    if args.freeze_encoder:
        param_groups = [{"params": heads.head_parameters(), "lr": args.lr}]
    else:
        param_groups = [
            {"params": heads.head_parameters(), "lr": args.lr},
            {"params": heads.encoder_parameters(), "lr": args.encoder_lr},
        ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineWithWarmup(
            optimizer, total_steps=total_steps, warmup_ratio=args.warmup_ratio, min_lr=args.min_lr
        )

    history: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(args.epochs):
        epoch_losses = {"loss": 0.0, "loss_lambda": 0.0, "loss_omega": 0.0, "loss_validity": 0.0}
        n_batches = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            metrics = train_one_step(heads, loss_fn, batch, optimizer, scheduler=scheduler)
            for k in epoch_losses:
                epoch_losses[k] += metrics[k]
            n_batches += 1
            global_step += 1
        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_batches)

        val_metrics = (
            evaluate(heads, loss_fn, val_loader, device, stats) if val_loader is not None else None
        )
        lr = optimizer.param_groups[0]["lr"]
        log_line = (
            f"epoch={epoch} step={global_step} "
            f"train_loss={epoch_losses['loss']:.6f} "
            f"train_lambda={epoch_losses['loss_lambda']:.6f} "
            f"train_omega={epoch_losses['loss_omega']:.6f} lr={lr:.6g}"
        )
        if val_metrics:
            log_line += (
                f" val_loss={val_metrics['loss']:.6f} "
                f"val_mae_lambda={val_metrics['mae_lambda_ep']:.4f} "
                f"val_mae_omega={val_metrics['mae_omega_log']:.4f} "
                f"val_mae_tc={val_metrics['mae_tc']:.4f}"
            )
        print(log_line)

        record = {"epoch": epoch, "step": global_step, "lr": lr, **{f"train_{k}": v for k, v in epoch_losses.items()}}
        if val_metrics:
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(record)

        checkpoint_path = Path(args.checkpoint_path)
        if args.save_every_epochs > 0 and (
            (epoch + 1) % args.save_every_epochs == 0 or epoch == args.epochs - 1
        ):
            # Periodic checkpointing so a long unattended GPU run doesn't
            # lose all progress to a Colab session timeout/preemption.
            save_checkpoint(
                checkpoint_path,
                heads,
                optimizer,
                epoch,
                global_step,
                stats=stats,
                scheduler=scheduler,
                config=vars(args),
            )
            print(f"[physics_jepa] checkpoint saved to {checkpoint_path} (epoch={epoch})")

    elapsed = time.time() - t_start
    # A "bounded smoke run" caveat only applies when the upstream foundation
    # checkpoint/this run itself actually looks like one (CPU device or few
    # epochs); a full-scale GPU fine-tune should not carry that label.
    is_bounded_smoke_run = args.device == "cpu" or args.epochs < 20
    status = "bounded_local_cpu_finetune_run" if is_bounded_smoke_run else "full_scale_finetune_run"
    note = (
        (
            "Physics-head fine-tuning on a compute-bounded local CPU machine, "
            "on top of a foundation_jepa checkpoint that was itself only a "
            "bounded local pretraining smoke run (see "
            "docs/audit/foundation_jepa_pretrain_report.json). Metrics here "
            "validate the fine-tuning pipeline end-to-end; they are not "
            "representative of what a fully pretrained + fully fine-tuned "
            "model would achieve."
        )
        if is_bounded_smoke_run
        else (
            f"Fine-tuned with epochs={args.epochs}, batch_size={args.batch_size}, "
            f"freeze_encoder={args.freeze_encoder} on device={args.device}, on top "
            f"of foundation checkpoint {args.foundation_checkpoint}."
        )
    )
    report = {
        "status": status,
        "note": note,
        "elapsed_seconds": elapsed,
        "foundation_checkpoint": str(args.foundation_checkpoint),
        "config": vars(args),
        "history": history,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_metrics": {k: v for k, v in history[-1].items() if k.startswith("val_")} if history else None,
        "checkpoint_path": str(checkpoint_path),
    }
    if args.report_path is not None:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune physics_jepa heads (lambda, omega_log)")
    parser.add_argument("--foundation-checkpoint", default="models/checkpoints/foundation_jepa.pt")
    parser.add_argument("--train-csv", default="data/processed/physics_jepa_train.csv.gz")
    parser.add_argument("--val-csv", default="data/processed/physics_jepa_val.csv.gz")
    parser.add_argument("--checkpoint-path", default="models/checkpoints/physics_jepa.pt")
    parser.add_argument("--report-path", default="docs/audit/physics_jepa_finetune_report.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--head-hidden-dim", type=int, default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--omega-weight", type=float, default=1.0)
    parser.add_argument("--validity-weight", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=5,
        help=(
            "Overwrite --checkpoint-path every N epochs (and always on the "
            "final epoch), so a long unattended GPU run isn't lost to a "
            "session timeout/preemption."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
