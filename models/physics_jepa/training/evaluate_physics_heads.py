"""Evaluate a trained physics_jepa checkpoint (`PhysicsHeads`: lambda_ep +
omega_log) on a held-out split and derive Tc via the Allen-Dynes formula.

Reports both the bare Allen-Dynes interpolation formula and the f1
strong-coupling-corrected variant (see `models/physics_jepa/model/physics_utils.py`),
matching `docs/audit/physics_jepa_evaluation_report.md`'s methodology.

Usage:
    python -m models.physics_jepa.training.evaluate_physics_heads \\
        --checkpoint-path models/checkpoints/physics_jepa.pt \\
        --foundation-checkpoint models/checkpoints/foundation_jepa.pt \\
        --csv-path data/processed/physics_jepa_test.csv.gz \\
        --output-json docs/audit/physics_jepa_test_eval_report.json \\
        --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.dataset import PhysicsTargetStats, create_physics_dataloader
from ..model.heads import PhysicsHeads
from ..model.losses import PhysicsHeadsLoss
from ..model.physics_utils import allen_dynes_tc
from .train_physics_heads import evaluate, load_foundation_model, move_batch_to_device


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    foundation_checkpoint: str | Path,
    csv_path: str | Path,
    device: torch.device,
    batch_size: int = 32,
    save_arrays_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load `checkpoint_path`'s `PhysicsHeads` and score it on `csv_path`.

    Returns a metrics dict with `evaluate()`'s aggregate loss/MAE fields plus
    `mae_tc_bare_formula` / `mae_tc_f1_corrected_formula` (Tc MAE under the
    bare vs. f1-corrected Allen-Dynes formula). If `save_arrays_path` is
    given, also dumps the per-row true/pred arrays (lambda, omega, both Tc
    variants) to that `.npz` path for downstream plotting/inspection.
    """
    foundation_model, foundation_config = load_foundation_model(foundation_checkpoint, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    stats = PhysicsTargetStats.from_dict(ckpt["target_stats"])
    cfg = ckpt.get("config", {})
    heads = PhysicsHeads(
        foundation_model,
        hidden_dim=cfg.get("head_hidden_dim"),
        freeze_encoder=cfg.get("freeze_encoder", False),
    ).to(device)
    heads.load_state_dict(ckpt["heads_state_dict"])
    heads.eval()

    loss_fn = PhysicsHeadsLoss(
        stats,
        omega_weight=cfg.get("omega_weight", 1.0),
        validity_weight=cfg.get("validity_weight", 0.05),
    )
    loader = create_physics_dataloader(
        csv_path=csv_path,
        batch_size=batch_size,
        shuffle=False,
        max_atoms=foundation_config["max_atoms"],
        max_rows=None,
        drop_last=False,
        cache_parsed=True,
    )
    metrics = evaluate(heads, loss_fn, loader, device, stats)

    lam_true, lam_pred, om_true, om_pred = [], [], [], []
    tc_true_bare, tc_pred_bare, tc_true_corr, tc_pred_corr = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            out = heads(
                batch["frac_coords"],
                batch["atomic_numbers"],
                batch["raw_lattice_matrix"],
                batch["num_atoms"],
                batch["atom_mask"],
            )
            lp, op = stats.denormalize(out.lambda_pred_norm, out.omega_pred_norm)
            lt, ot = batch["lambda_ep"], batch["omega_log"]
            lam_true.append(lt)
            lam_pred.append(lp)
            om_true.append(ot)
            om_pred.append(op)
            tc_pred_bare.append(allen_dynes_tc(lp, op, apply_strong_coupling_correction=False))
            tc_true_bare.append(allen_dynes_tc(lt, ot, apply_strong_coupling_correction=False))
            tc_pred_corr.append(allen_dynes_tc(lp, op, apply_strong_coupling_correction=True))
            tc_true_corr.append(allen_dynes_tc(lt, ot, apply_strong_coupling_correction=True))

    arrs = {
        "lambda_true": torch.cat(lam_true).numpy(),
        "lambda_pred": torch.cat(lam_pred).numpy(),
        "omega_true": torch.cat(om_true).numpy(),
        "omega_pred": torch.cat(om_pred).numpy(),
        "tc_true_bare": torch.cat(tc_true_bare).numpy(),
        "tc_pred_bare": torch.cat(tc_pred_bare).numpy(),
        "tc_true_corrected": torch.cat(tc_true_corr).numpy(),
        "tc_pred_corrected": torch.cat(tc_pred_corr).numpy(),
    }
    metrics["mae_tc_bare_formula"] = float(np.mean(np.abs(arrs["tc_true_bare"] - arrs["tc_pred_bare"])))
    metrics["mae_tc_f1_corrected_formula"] = float(
        np.mean(np.abs(arrs["tc_true_corrected"] - arrs["tc_pred_corrected"]))
    )
    metrics.pop("mae_tc", None)
    metrics["n_rows"] = int(len(arrs["lambda_true"]))

    if save_arrays_path is not None:
        Path(save_arrays_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_arrays_path, **arrs)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained physics_jepa checkpoint")
    parser.add_argument("--checkpoint-path", default="models/checkpoints/physics_jepa.pt")
    parser.add_argument("--foundation-checkpoint", default="models/checkpoints/foundation_jepa.pt")
    parser.add_argument("--csv-path", default="data/processed/physics_jepa_test.csv.gz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--save-arrays", default=None, help="Optional .npz path for per-row arrays")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint_path,
        foundation_checkpoint=args.foundation_checkpoint,
        csv_path=args.csv_path,
        device=device,
        batch_size=args.batch_size,
        save_arrays_path=args.save_arrays,
    )
    print(
        " ".join(
            f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
    )
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2))
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
