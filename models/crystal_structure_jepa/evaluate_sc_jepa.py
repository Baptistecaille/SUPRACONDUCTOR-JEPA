from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from crystal_matrices import create_dataloader
from train_sc_jepa import build_model, move_batch_to_device


def _append_embeddings(
    store: dict[str, list[Any]],
    batch: dict[str, Any],
    out: dict[str, Any],
) -> None:
    store["material_id"].extend(batch["material_id"])
    store["ef_per_atom"].append(batch["ef_per_atom"].detach().cpu())
    store["z_pred"].append(out["z_pred"].detach().cpu())
    store["z_target"].append(out["z_target"].detach().cpu())
    if "tc" in batch:
        store.setdefault("tc", []).append(batch["tc"].detach().cpu())


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = build_model(
        layers=args.layers,
        attn_heads=args.attn_heads,
        dropout=0.0,
        ema_decay=args.ema_decay,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataloader = create_dataloader(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    totals = {"loss": 0.0, "loss_pred": 0.0, "loss_reg": 0.0}
    seen_batches = 0
    seen_samples = 0
    embeddings: dict[str, list[Any]] | None = None
    if args.save_embeddings is not None:
        embeddings = {"material_id": [], "ef_per_atom": [], "z_pred": [], "z_target": []}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            if len(batch["material_id"]) < 2:
                continue

            batch = move_batch_to_device(batch, device)
            out = model(batch)
            metrics = out["metrics"]
            batch_size = len(batch["material_id"])
            for key in totals:
                totals[key] += float(metrics[key]) * batch_size
            seen_batches += 1
            seen_samples += batch_size

            if embeddings is not None:
                _append_embeddings(embeddings, batch, out)

    if seen_samples == 0:
        raise ValueError("No evaluable samples found; use batch_size >= 2.")

    results = {key: value / seen_samples for key, value in totals.items()}
    results["batches"] = float(seen_batches)
    results["samples"] = float(seen_samples)

    if embeddings is not None:
        save_path = Path(args.save_embeddings)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "material_id": embeddings["material_id"],
                "ef_per_atom": torch.cat(embeddings["ef_per_atom"], dim=0),
                "z_pred": torch.cat(embeddings["z_pred"], dim=0),
                "z_target": torch.cat(embeddings["z_target"], dim=0),
                **(
                    {"tc": torch.cat(embeddings["tc"], dim=0)}
                    if "tc" in embeddings
                    else {}
                ),
            },
            save_path,
        )
        print(f"Saved embeddings to {save_path}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SC-JEPA checkpoint")
    parser.add_argument("--checkpoint-path", default="checkpoints/sc_jepa.pt")
    parser.add_argument("--csv-path", default="data/jepa/mp.csv.gz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--attn-heads", type=int, default=16)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--save-embeddings", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    metrics = evaluate(parse_args())
    print(
        " ".join(
            [
                f"loss={metrics['loss']:.6f}",
                f"loss_pred={metrics['loss_pred']:.6f}",
                f"loss_reg={metrics['loss_reg']:.6f}",
                f"batches={int(metrics['batches'])}",
                f"samples={int(metrics['samples'])}",
            ]
        )
    )
