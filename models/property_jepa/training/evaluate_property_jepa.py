"""Evaluate a trained property-JEPA checkpoint on a held-out split.

The training collate function samples one query property at random per
material. Evaluation instead expands each material into one example per
available property, using all other available properties as context. This
makes the aggregate and per-property JEPA losses deterministic for a fixed
checkpoint and split.

Usage:
    python -m models.property_jepa.training.evaluate_property_jepa \
        --checkpoint-path models/checkpoints/property_jepa.pt \
        --csv-path data/processed/property_jepa_test.csv.gz \
        --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from ..data.dataset import PropertyJEPADataset, PropertyStats
from ..data.property_schema import PROPERTY_TYPES
from ..model.crystal_encoder import build_atom_tokens
from .train_property_jepa import build_model, move_batch_to_device


def collate_all_property_queries(samples: list[dict]) -> dict[str, Any]:
    """Build one JEPA query example for every available property in each sample."""
    atom_tokens = []
    atom_counts = []
    lattice_features = []
    known_type_ids = []
    known_values = []
    known_masks = []
    query_type_ids = []
    query_values = []
    composition_keys: list[str] = []

    for sample in samples:
        sample_atom_tokens = build_atom_tokens(sample["frac_coords"], sample["atomic_numbers"])
        n_props = int(sample["property_type_ids"].shape[0])

        for query_idx in range(n_props):
            known_idx = [i for i in range(n_props) if i != query_idx]

            atom_tokens.append(sample_atom_tokens)
            atom_counts.append(sample["frac_coords"].shape[0])
            lattice_features.append(sample["lattice_features"])
            query_type_ids.append(sample["property_type_ids"][query_idx])
            query_values.append(sample["property_values"][query_idx])
            composition_keys.append(sample["composition_key"])

            if known_idx:
                known_type_ids.append(sample["property_type_ids"][known_idx])
                known_values.append(sample["property_values"][known_idx])
            else:
                known_type_ids.append(torch.zeros(0, dtype=torch.long))
                known_values.append(torch.zeros(0, dtype=torch.float32))
            known_masks.append(torch.ones(len(known_idx), dtype=torch.bool))

    if not atom_tokens:
        raise ValueError("No property-query examples found in batch.")

    num_atoms = torch.tensor(atom_counts, dtype=torch.long)
    padded_atom_tokens = pad_sequence(atom_tokens, batch_first=True)
    max_atoms = int(num_atoms.max().item())
    atom_mask = torch.arange(max_atoms).unsqueeze(0) < num_atoms.unsqueeze(1)

    padded_known_type = pad_sequence(known_type_ids, batch_first=True)
    padded_known_value = pad_sequence(known_values, batch_first=True)
    padded_known_mask = pad_sequence(known_masks, batch_first=True)
    if padded_known_type.shape[1] == 0:
        b = len(atom_tokens)
        padded_known_type = torch.zeros(b, 1, dtype=torch.long)
        padded_known_value = torch.zeros(b, 1, dtype=torch.float32)
        padded_known_mask = torch.zeros(b, 1, dtype=torch.bool)

    return {
        "composition_key": composition_keys,
        "atom_tokens": padded_atom_tokens,
        "atom_mask": atom_mask,
        "lattice_features": torch.stack(lattice_features),
        "known_property_type_id": padded_known_type,
        "known_property_value": padded_known_value,
        "known_property_mask": padded_known_mask,
        "query_property_type_id": torch.stack(query_type_ids),
        "query_property_value": torch.stack(query_values),
    }


def _checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config") or {}
    return {
        "hidden_dim": int(config.get("hidden_dim", 256)),
        "layers": int(config.get("layers", 6)),
        "attn_heads": int(config.get("attn_heads", 8)),
        "dropout": float(config.get("dropout", 0.0)),
        "ema_decay": float(config.get("ema_decay", 0.996)),
        "regularizer": str(config.get("regularizer", "none")),
        "reg_weight": float(config.get("reg_weight", 0.0)),
        "reg_std_coeff": float(config.get("reg_std_coeff", 1.0)),
        "reg_cov_coeff": float(config.get("reg_cov_coeff", 1.0)),
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    stats = PropertyStats.from_dict(checkpoint["property_stats"])
    model_config = _checkpoint_config(checkpoint)
    model = build_model(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = PropertyJEPADataset(
        args.csv_path,
        stats=stats,
        min_properties=args.min_properties,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_all_property_queries,
    )

    totals = {"loss": 0.0, "loss_pred": 0.0, "loss_reg": 0.0}
    by_property = {
        name: {"loss": 0.0, "loss_pred": 0.0, "loss_reg": 0.0, "queries": 0}
        for name in PROPERTY_TYPES
    }
    seen_batches = 0
    seen_materials = 0
    seen_queries = 0
    embeddings: dict[str, list[Any]] | None = None
    if args.save_embeddings is not None:
        embeddings = {
            "composition_key": [],
            "query_property_type_id": [],
            "query_property_value": [],
            "z_pred": [],
            "z_target": [],
        }

    for batch_idx, batch in enumerate(dataloader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        seen_materials += len(set(batch["composition_key"]))
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        query_count = int(batch["query_property_type_id"].shape[0])
        seen_batches += 1
        seen_queries += query_count

        for key in totals:
            totals[key] += float(out[key].detach().cpu()) * query_count

        per_query_loss = ((out["z_pred"] - out["z_target"]) ** 2).mean(dim=1).detach().cpu()
        query_type_ids = batch["query_property_type_id"].detach().cpu()
        for property_id, property_name in enumerate(PROPERTY_TYPES):
            mask = query_type_ids == property_id
            count = int(mask.sum().item())
            if count == 0:
                continue
            loss_pred_sum = float(per_query_loss[mask].sum().item())
            loss_reg_sum = float(out["loss_reg"].detach().cpu()) * count
            by_property[property_name]["loss"] += (
                loss_pred_sum + model_config["reg_weight"] * loss_reg_sum
            )
            by_property[property_name]["loss_pred"] += loss_pred_sum
            by_property[property_name]["loss_reg"] += loss_reg_sum
            by_property[property_name]["queries"] += count

        if embeddings is not None:
            embeddings["composition_key"].extend(batch["composition_key"])
            embeddings["query_property_type_id"].append(query_type_ids)
            embeddings["query_property_value"].append(batch["query_property_value"].detach().cpu())
            embeddings["z_pred"].append(out["z_pred"].detach().cpu())
            embeddings["z_target"].append(out["z_target"].detach().cpu())

    if seen_queries == 0:
        raise ValueError("No evaluable property queries found.")

    results: dict[str, Any] = {
        key: value / seen_queries for key, value in totals.items()
    }
    results.update(
        {
            "batches": seen_batches,
            "materials": seen_materials,
            "queries": seen_queries,
            "by_property": {},
        }
    )
    for property_name, property_totals in by_property.items():
        queries = property_totals["queries"]
        if queries == 0:
            continue
        results["by_property"][property_name] = {
            "loss": property_totals["loss"] / queries,
            "loss_pred": property_totals["loss_pred"] / queries,
            "loss_reg": property_totals["loss_reg"] / queries,
            "queries": queries,
        }

    if embeddings is not None:
        save_path = Path(args.save_embeddings)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "composition_key": embeddings["composition_key"],
                "query_property_type_id": torch.cat(embeddings["query_property_type_id"], dim=0),
                "query_property_value": torch.cat(embeddings["query_property_value"], dim=0),
                "z_pred": torch.cat(embeddings["z_pred"], dim=0),
                "z_target": torch.cat(embeddings["z_target"], dim=0),
            },
            save_path,
        )
        print(f"Saved embeddings to {save_path}")

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Saved metrics to {output_path}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained property-JEPA checkpoint")
    parser.add_argument("--checkpoint-path", default="models/checkpoints/property_jepa.pt")
    parser.add_argument("--csv-path", default="data/processed/property_jepa_test.csv.gz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-properties", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--save-embeddings", default=None)
    return parser.parse_args()


def _format_metrics(metrics: dict[str, Any]) -> str:
    return " ".join(
        [
            f"loss={metrics['loss']:.6f}",
            f"loss_pred={metrics['loss_pred']:.6f}",
            f"loss_reg={metrics['loss_reg']:.6f}",
            f"batches={metrics['batches']}",
            f"materials={metrics['materials']}",
            f"queries={metrics['queries']}",
        ]
    )


if __name__ == "__main__":
    metrics = evaluate(parse_args())
    print(_format_metrics(metrics))
    for property_name, property_metrics in metrics["by_property"].items():
        print(
            f"{property_name}: "
            f"loss={property_metrics['loss']:.6f} "
            f"queries={property_metrics['queries']}"
        )
