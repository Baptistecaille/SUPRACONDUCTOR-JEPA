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

Usage (CPU smoke run):
    python -m models.foundation_jepa.training.train_foundation_jepa \\
        --train-csv data/processed/property_jepa_train.csv.gz \\
        --val-csv data/processed/property_jepa_val.csv.gz \\
        --checkpoint-path models/checkpoints/foundation_jepa.pt \\
        --report-path docs/audit/foundation_jepa_pretrain_report.json \\
        --max-train-rows 6000 --max-val-rows 800 \\
        --epochs 15 --batch-size 32 --device cpu

Usage (full-scale single-GPU run, e.g. a 40GB A100):
    The paper config (hidden_dim=512, layers=8, batch_size=2048) does NOT
    fit in a single forward+backward on a 40GB GPU without
    --grad-checkpointing and --amp-dtype bf16 (see backbone.py's
    `MaskedMHA`/`FoundationTransformer` docstrings for the memory math). Add
    --grad-accum-steps only if the above two still don't fit -- it further
    shrinks the per-forward tensor size at the cost of fewer InfoNCE
    negatives per step.
    python -m models.foundation_jepa.training.train_foundation_jepa \\
        --train-csv data/processed/property_jepa_train.csv.gz \\
        --val-csv data/processed/property_jepa_val.csv.gz \\
        --checkpoint-path models/checkpoints/foundation_jepa.pt \\
        --report-path docs/audit/foundation_jepa_pretrain_report.json \\
        --device cuda --batch-size 2048 --epochs 2000 \\
        --hidden-dim 512 --layers 8 --attn-heads 16 \\
        --max-train-rows 249719 --max-val-rows 31215 \\
        --lr 1e-4 --weight-decay 1e-4 \\
        --scheduler cosine --warmup-ratio 0.1 --min-lr 1e-5 \\
        --grad-checkpointing --amp-dtype bf16 \\
        --save-every-epochs 5
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
    grad_checkpointing: bool = False,
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
        grad_checkpointing=grad_checkpointing,
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def chunk_batch(batch: dict[str, Any], n_chunks: int) -> list[dict[str, Any]]:
    """Split a batch dict into `n_chunks` sub-batches along dim 0 (for gradient
    accumulation): each sub-batch is forwarded/backwarded separately so the
    largest tensor that ever hits the GPU is roughly `batch_size // n_chunks`
    instead of the full `batch_size`, while the optimizer still takes one
    step per full logical batch (see `train_one_step`'s `grad_accum_steps`).
    Uses `torch.tensor_split`, which tolerates a batch size not evenly
    divisible by `n_chunks` (some chunks get one extra row) instead of
    raising like `torch.chunk` can for edge cases.

    NOTE: this model's loss is an in-batch InfoNCE, whose negatives come from
    whatever is in ONE forward pass -- chunking the batch this way reduces
    the number of negatives seen per forward to ~`batch_size // n_chunks`,
    it does NOT reconstruct the original batch's full negative set. This is
    the standard memory/negative-count trade-off for accumulated contrastive
    training; prefer `--grad-checkpointing`/`--amp-dtype bf16` first and only
    reach for `--grad-accum-steps > 1` if those alone don't fit.
    """
    if n_chunks <= 1:
        return [batch]
    tensor_keys = [k for k, v in batch.items() if torch.is_tensor(v)]
    other_keys = [k for k in batch if k not in tensor_keys]
    split_per_key = {k: torch.tensor_split(batch[k], n_chunks, dim=0) for k in tensor_keys}
    chunks = []
    for i in range(n_chunks):
        piece = {k: split_per_key[k][i] for k in tensor_keys}
        if piece[tensor_keys[0]].shape[0] == 0:
            continue
        piece.update({k: batch[k] for k in other_keys})
        chunks.append(piece)
    return chunks


def train_one_step(
    model: FoundationJEPA,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    grad_clip_norm: float | None = 1.0,
    amp_dtype: torch.dtype | None = None,
    grad_accum_steps: int = 1,
    accum_step_idx: int = 0,
) -> dict[str, float]:
    """Run one micro-batch of forward+backward, optionally under autocast and
    accumulating gradients over `grad_accum_steps` micro-batches before an
    optimizer step.

    Args:
        amp_dtype: If set (`torch.bfloat16` or `torch.float16`), wraps the
            forward pass in `torch.autocast` -- roughly halves activation
            memory vs. fp32 on top of whatever `--grad-checkpointing` saves.
            bf16 needs no loss scaling on Ampere+ (A100); fp16 would need a
            `GradScaler`, which this helper does not wire up -- prefer bf16
            on an A100.
        grad_accum_steps: Number of micro-batches to accumulate gradients
            over before `optimizer.step()`. The per-micro-batch loss is
            divided by this so the effective (accumulated) gradient matches
            training at the full logical batch size, letting you keep
            `--batch-size` at the paper's value for the loss's InfoNCE
            negatives while shrinking the ACTUAL tensor size that hits the
            GPU per forward/backward to `batch_size // grad_accum_steps`.
        accum_step_idx: 0-indexed position of this micro-batch within the
            current accumulation window; only the last one (`==
            grad_accum_steps - 1`) triggers `optimizer.step()` / `zero_grad()`
            / `scheduler.step()`.
    """
    model.train()
    is_first_in_window = accum_step_idx == 0
    is_last_in_window = accum_step_idx == grad_accum_steps - 1
    if is_first_in_window:
        optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type=batch["frac_coords"].device.type,
        dtype=amp_dtype,
        enabled=amp_dtype is not None,
    ):
        out = model(
            batch["frac_coords"],
            batch["atomic_numbers"],
            batch["raw_lattice_matrix"],
            batch["num_atoms"],
            batch["atom_mask"],
            batch["formation_energy_peratom"],
        )
        loss = out.loss / grad_accum_steps
    loss.backward()

    if is_last_in_window:
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
def evaluate(
    model: FoundationJEPA,
    dataloader,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "loss_infonce": 0.0, "loss_reg": 0.0}
    n_batches = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
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
        grad_checkpointing=args.grad_checkpointing,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    amp_dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp_dtype]
    if args.amp_dtype == "fp16":
        print(
            "[foundation_jepa] WARNING: --amp-dtype fp16 has no GradScaler wired up here "
            "(risk of silent NaN/inf gradients); prefer bf16 on an A100/H100."
        )

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
            micro_batches = chunk_batch(batch, args.grad_accum_steps)
            n_micro = len(micro_batches)
            for i, micro_batch in enumerate(micro_batches):
                metrics = train_one_step(
                    model,
                    micro_batch,
                    optimizer,
                    scheduler=scheduler,
                    amp_dtype=amp_dtype,
                    grad_accum_steps=n_micro,
                    accum_step_idx=i,
                )
            for k in epoch_losses:
                epoch_losses[k] += metrics[k]
            n_batches += 1
            global_step += 1
        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_batches)

        val_metrics = (
            evaluate(model, val_loader, device, amp_dtype=amp_dtype)
            if val_loader is not None
            else None
        )
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
    parser.add_argument(
        "--grad-checkpointing",
        action="store_true",
        help=(
            "Wrap each transformer block in activation checkpointing "
            "(recompute-in-backward instead of keeping every block's "
            "activations resident). ~8x cut to transformer activation "
            "memory at the paper's layers=8 config for ~20-30%% more "
            "compute. Recommended ON for any full-scale GPU run."
        ),
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["none", "bf16", "fp16"],
        default="none",
        help=(
            "Run the forward pass (and loss) under torch.autocast in this "
            "dtype. 'bf16' is recommended on Ampere+ (A100/H100): no loss "
            "scaling needed, roughly halves activation memory vs fp32. "
            "'fp16' needs a GradScaler that this script does not wire up -- "
            "avoid unless you add one. 'none' keeps fp32 (default, matches "
            "prior behavior)."
        ),
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help=(
            "Split each --batch-size batch into this many micro-batches, "
            "run forward/backward on each separately, and only step the "
            "optimizer after the last one (gradients accumulate). Lets you "
            "keep --batch-size at the paper's value (which sets the number "
            "of InfoNCE negatives) while shrinking the actual tensor size "
            "that hits the GPU per forward/backward to roughly "
            "batch_size // grad_accum_steps. NOTE: this also shrinks the "
            "number of in-batch negatives seen per forward pass -- prefer "
            "--grad-checkpointing / --amp-dtype bf16 first and only "
            "increase this if the model still doesn't fit."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
