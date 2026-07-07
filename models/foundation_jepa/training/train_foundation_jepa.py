"""Training entry point for `foundation_jepa` self-supervised pretraining.

Mirrors `models/property_jepa/training/train_property_jepa.py` in overall
shape (`build_model` / `train_one_step` / `save_checkpoint` / `train` / CLI),
reading the SAME leak-free splits
(`data/processed/property_jepa_{train,val,test}.csv.gz`) but optimizing
`models.foundation_jepa.model.FoundationJEPA`'s energy-aware InfoNCE
objective instead of stage 2's supervised property-prediction objective.

Compute-bounded local run
-------------------------
The official Crys-JEPA config (`external/Crys_JEPA/configs/jepa/mp.yml`)
trains `hidden_dim=512, layers=8, attn_head=16` for 2000 epochs at
batch_size=2048 on a GPU cluster. This machine has 8 CPUs / 16GB RAM / no
GPU. This script defaults to a MUCH smaller configuration
(`hidden_dim=128, layers=4, attn_heads=8`) and a `--max-train-rows`-bounded
subsample (default 6000, cached in memory after first-epoch CIF parsing via
`cache_parsed=True`) so a run completes in single-digit minutes on this
hardware. The resulting checkpoint is explicitly a smoke-scale /
non-converged pretraining run, not a production foundation model -- see
`docs/audit/foundation_jepa_pretrain_report.json`'s `"status"` field and the
recommendation to resume training on a GPU host for a real run.

Usage:
    python -m models.foundation_jepa.training.train_foundation_jepa \\
        --train-csv data/processed/property_jepa_train.csv.gz \\
        --val-csv data/processed/property_jepa_val.csv.gz \\
        --checkpoint-path models/checkpoints/foundation_jepa.pt \\
        --report-path docs/audit/foundation_jepa_pretrain_report.json \\
        --max-train-rows 6000 --max-val-rows 800 \\
        --epochs 15 --batch-size 32 --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from ..data.dataset import create_foundation_dataloader, fit_matrix_scaler
from ..data.lattice import MatrixMeanStdScaler
from ..model.jepa import FoundationJEPA
from .schedulers import CosineWithWarmup


def build_model(
    hidden_dim: int = 128,
    layers: int = 4,
    attn_heads: int = 8,
    dropout: float = 0.0,
    max_atoms: int = 200,
    temperature: float = 0.1,
    reg_weight: float = 0.01,
    matrix_scaler: MatrixMeanStdScaler | None = None,
) -> FoundationJEPA:
    return FoundationJEPA(
        hidden_dim=hidden_dim,
        layers=layers,
        attn_heads=attn_heads,
        dropout=dropout,
        max_atoms=max_atoms,
        temperature=temperature,
        reg_weight=reg_weight,
        matrix_scaler=matrix_scaler,
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def train_one_step(
    model: FoundationJEPA,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    grad_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(
        batch["frac_coords"],
        batch["atomic_numbers"],
        batch["raw_lattice_matrix"],
        batch["num_atoms"],
        batch["atom_mask"],
        batch["formation_energy_peratom"],
    )
    out.loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {
        "loss": out.loss.item(),
        "loss_infonce": out.loss_infonce.item(),
        "loss_reg": out.loss_reg.item(),
    }


@torch.no_grad()
def evaluate(model: FoundationJEPA, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "loss_infonce": 0.0, "loss_reg": 0.0}
    n_batches = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        out = model(
            batch["frac_coords"],
            batch["atomic_numbers"],
            batch["raw_lattice_matrix"],
            batch["num_atoms"],
            batch["atom_mask"],
            batch["formation_energy_peratom"],
        )
        totals["loss"] += out.loss.item()
        totals["loss_infonce"] += out.loss_infonce.item()
        totals["loss_reg"] += out.loss_reg.item()
        n_batches += 1
    if n_batches == 0:
        return {k: float("nan") for k in totals}
    return {k: v / n_batches for k, v in totals.items()}


def save_checkpoint(
    path: Path,
    model: FoundationJEPA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    matrix_scaler: MatrixMeanStdScaler,
    scheduler: Any | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "matrix_scaler_mean": matrix_scaler.mean,
        "matrix_scaler_std": matrix_scaler.std,
        "config": {} if config is None else config,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    t_start = time.time()

    print(f"[foundation_jepa] fitting lattice matrix scaler from {args.train_csv} ...")
    matrix_scaler = fit_matrix_scaler(
        args.train_csv, n_samples=args.scaler_fit_rows, max_atoms=args.max_atoms
    )

    print(f"[foundation_jepa] loading + parsing up to {args.max_train_rows} train rows ...")
    train_loader = create_foundation_dataloader(
        csv_path=args.train_csv,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        max_atoms=args.max_atoms,
        max_rows=args.max_train_rows,
        drop_last=True,
        cache_parsed=True,
    )
    val_loader = None
    if args.val_csv is not None:
        print(f"[foundation_jepa] loading + parsing up to {args.max_val_rows} val rows ...")
        val_loader = create_foundation_dataloader(
            csv_path=args.val_csv,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            max_atoms=args.max_atoms,
            max_rows=args.max_val_rows,
            drop_last=False,
            cache_parsed=True,
        )

    model = build_model(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        max_atoms=args.max_atoms,
        temperature=args.temperature,
        reg_weight=args.reg_weight,
        matrix_scaler=matrix_scaler,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineWithWarmup(
            optimizer, total_steps=total_steps, warmup_ratio=args.warmup_ratio, min_lr=args.min_lr
        )

    history: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(args.epochs):
        epoch_losses = {"loss": 0.0, "loss_infonce": 0.0, "loss_reg": 0.0}
        n_batches = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            metrics = train_one_step(model, batch, optimizer, scheduler=scheduler)
            for k in epoch_losses:
                epoch_losses[k] += metrics[k]
            n_batches += 1
            global_step += 1
        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_batches)

        val_metrics = evaluate(model, val_loader, device) if val_loader is not None else None
        lr = optimizer.param_groups[0]["lr"]
        log_line = (
            f"epoch={epoch} step={global_step} "
            f"train_loss={epoch_losses['loss']:.6f} "
            f"train_infonce={epoch_losses['loss_infonce']:.6f} "
            f"train_reg={epoch_losses['loss_reg']:.6f} lr={lr:.6g}"
        )
        if val_metrics is not None:
            log_line += f" val_loss={val_metrics['loss']:.6f}"
        print(log_line)

        record = {"epoch": epoch, "step": global_step, "lr": lr, **{f"train_{k}": v for k, v in epoch_losses.items()}}
        if val_metrics is not None:
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(record)

        checkpoint_path = Path(args.checkpoint_path)
        if args.save_every_epochs > 0 and (
            (epoch + 1) % args.save_every_epochs == 0 or epoch == args.epochs - 1
        ):
            # Periodic checkpointing so a long unattended GPU run (Colab
            # session timeout, preemption, etc.) doesn't lose all progress:
            # every `save_every_epochs` epochs (and always on the last one)
            # we overwrite `checkpoint_path` with the current weights.
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                global_step,
                matrix_scaler=matrix_scaler,
                scheduler=scheduler,
                config=vars(args),
            )
            print(f"[foundation_jepa] checkpoint saved to {checkpoint_path} (epoch={epoch})")

    elapsed = time.time() - t_start
    # Only label the run a "bounded smoke run" when it actually looks like one
    # (small model / few epochs / CPU device). A full-scale GPU run (e.g. the
    # official hidden_dim=512, layers=8, batch_size=2048, epochs=2000 config
    # run on Colab) should not carry that caveat in the saved report.
    is_bounded_smoke_run = (
        args.device == "cpu" or args.hidden_dim < 256 or args.epochs < 100
    )
    if is_bounded_smoke_run:
        status = "bounded_local_cpu_smoke_run_not_converged"
        note = (
            "Trained on a compute-bounded local CPU machine (8 CPUs, 16GB RAM, "
            "no GPU), NOT the paper's config (hidden_dim=512, layers=8, "
            "batch_size=2048, 2000 epochs on GPU). This checkpoint demonstrates "
            "the pipeline end-to-end and is a starting point for embeddings, "
            "not a converged foundation model. Recommend resuming/re-running "
            "at full scale on a GPU host before using embeddings for anything "
            "beyond pipeline validation."
        )
    else:
        status = "full_scale_run"
        note = (
            f"Trained with hidden_dim={args.hidden_dim}, layers={args.layers}, "
            f"batch_size={args.batch_size}, epochs={args.epochs} on device="
            f"{args.device}."
        )
    report = {
        "status": status,
        "note": note,
        "elapsed_seconds": elapsed,
        "n_params": n_params,
        "config": vars(args),
        "history": history,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1].get("val_loss") if history else None,
        "checkpoint_path": str(checkpoint_path),
    }
    if args.report_path is not None:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain foundation_jepa (Crys-JEPA port)")
    parser.add_argument("--train-csv", default="data/processed/property_jepa_train.csv.gz")
    parser.add_argument("--val-csv", default="data/processed/property_jepa_val.csv.gz")
    parser.add_argument("--checkpoint-path", default="models/checkpoints/foundation_jepa.pt")
    parser.add_argument("--report-path", default="docs/audit/foundation_jepa_pretrain_report.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max-train-rows", type=int, default=6000)
    parser.add_argument("--max-val-rows", type=int, default=800)
    parser.add_argument("--scaler-fit-rows", type=int, default=1500)
    parser.add_argument("--max-atoms", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--attn-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--reg-weight", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)
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
            "session timeout/preemption. Set to 1 for maximum safety at the "
            "cost of extra I/O, or a large number to only save at the end."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
