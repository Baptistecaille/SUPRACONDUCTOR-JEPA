"""
generate_materials.py — Génère des candidats puis les filtre avec SUPRA-JEPA.

Exemple :
  python generate_materials.py --n-samples 128 --top-k 20
"""

import argparse
import json
import os
from collections import Counter

import torch
from pymatgen.core import Element, Lattice, Structure

from config import Config
from diffusion import CrystalDDPM, diffusion_vector_to_batch
from model import SupraJEPA


IDX_TO_ELEMENT = {el.Z: el.symbol for el in Element}
IDX_TO_WYCKOFF = {idx + 1: letter for idx, letter in enumerate("abcdefghijklmnopqrstuvwxyz")}


def load_diffusion(
    cfg: Config,
    device: torch.device,
    allow_legacy_decode: bool = False,
) -> tuple[CrystalDDPM, dict | None]:
    model = CrystalDDPM(cfg).to(device)
    checkpoint = torch.load(cfg.checkpoint_diffusion, map_location=device)
    metadata = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
    if metadata is None or (
        isinstance(checkpoint, dict)
        and checkpoint.get("representation") != "bounded_features_v4_correlated_decode"
    ):
        message = (
            "Ce checkpoint diffusion ne contient pas le decodeur empirique correle v4. "
            "Re-entraine train_diffusion.py pour eviter les candidats artificiels "
            "du type H/Og, coordonnees 0/1 ou mailles extremes."
        )
        if not allow_legacy_decode:
            raise ValueError(message)
        print("Attention:", message)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model, metadata


def load_jepa(cfg: Config, device: torch.device) -> SupraJEPA:
    model = SupraJEPA(cfg).to(device)
    checkpoint_path = cfg.checkpoint_finetune
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"{checkpoint_path} introuvable. Lance finetune.py avant le screening génératif."
        )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def score_candidates(model: SupraJEPA, batch: dict) -> torch.Tensor:
    logits = model.forward_classify(batch)
    return torch.sigmoid(logits)


def candidate_to_record(batch: dict, probs: torch.Tensor, idx: int) -> dict:
    valid = batch["padding_mask"][idx, 1:].detach().cpu()
    element_ids = batch["element_ids"][idx, 1:].detach().cpu()
    wyckoff_ids = batch["wyckoff_ids"][idx, 1:].detach().cpu()
    coords = batch["frac_coords"][idx, 1:].detach().cpu()

    sites = []
    for site_idx in torch.where(valid)[0].tolist():
        z = int(element_ids[site_idx])
        wyck = int(wyckoff_ids[site_idx])
        sites.append(
            {
                "element": IDX_TO_ELEMENT.get(z, f"Z{z}"),
                "Z": z,
                "wyckoff": IDX_TO_WYCKOFF.get(wyck, "a"),
                "frac_coords": [float(x) for x in coords[site_idx].tolist()],
            }
        )

    lattice = batch["lattice_feat"][idx].detach().cpu()
    return {
        "p_superconductor": float(probs[idx].detach().cpu()),
        "space_group": int(batch["sg_id"][idx].detach().cpu()),
        "lattice": {
            "a": float(lattice[0] * 20.0),
            "b": float(lattice[1] * 20.0),
            "c": float(lattice[2] * 20.0),
            "alpha": float(lattice[3] * 180.0),
            "beta": float(lattice[4] * 180.0),
            "gamma": float(lattice[5] * 180.0),
        },
        "n_sites": len(sites),
        "sites": sites,
    }


def plausibility_issues(record: dict, min_distance: float = 0.7) -> list[str]:
    """Retourne les raisons de rejet géométrique/chimique les plus évidentes."""
    issues = []
    lattice = record["lattice"]
    lengths = [lattice["a"], lattice["b"], lattice["c"]]
    angles = [lattice["alpha"], lattice["beta"], lattice["gamma"]]

    if any(x <= 0.5 or x >= 60.0 for x in lengths):
        issues.append("lattice_length_out_of_range")
    if any(x <= 20.0 or x >= 175.0 for x in angles):
        issues.append("lattice_angle_out_of_range")
    if record["n_sites"] < 1:
        issues.append("empty_structure")
        return issues

    species = [site["element"] for site in record["sites"]]
    frac_coords = [site["frac_coords"] for site in record["sites"]]

    rounded_sites = {
        (species[i], tuple(round(float(v) % 1.0, 4) for v in frac_coords[i]))
        for i in range(len(species))
    }
    if len(rounded_sites) < len(species):
        issues.append("duplicate_sites")

    try:
        structure = Structure(
            Lattice.from_parameters(
                lattice["a"],
                lattice["b"],
                lattice["c"],
                lattice["alpha"],
                lattice["beta"],
                lattice["gamma"],
            ),
            species,
            frac_coords,
            coords_are_cartesian=False,
            to_unit_cell=True,
        )
        if len(structure) > 1:
            distances = structure.distance_matrix
            distances[distances == 0.0] = float("inf")
            if float(distances.min()) < min_distance:
                issues.append("too_short_interatomic_distance")
    except Exception as exc:
        issues.append(f"pymatgen_invalid:{type(exc).__name__}")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--min-atoms", type=int, default=1)
    parser.add_argument("--diffusion-checkpoint", type=str, default=None)
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--allow-legacy-decode", action="store_true")
    parser.add_argument("--no-validity-filter", action="store_true")
    parser.add_argument("--min-distance", type=float, default=0.7)
    parser.add_argument("--out", type=str, default="generated_candidates.json")
    args = parser.parse_args()

    cfg = Config()
    if args.diffusion_checkpoint is not None:
        cfg.checkpoint_diffusion = args.diffusion_checkpoint
    if args.jepa_checkpoint is not None:
        cfg.checkpoint_finetune = args.jepa_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    diffusion, metadata = load_diffusion(cfg, device, allow_legacy_decode=args.allow_legacy_decode)
    jepa = load_jepa(cfg, device)

    records = []
    rejected = 0
    issue_counts: Counter[str] = Counter()

    for batch_idx in range(args.max_batches):
        vectors = diffusion.sample(args.n_samples, device=device)
        batch = diffusion_vector_to_batch(vectors, cfg, min_atoms=args.min_atoms, metadata=metadata)
        batch = {k: v.to(device) for k, v in batch.items()}
        probs = score_candidates(jepa, batch)

        order = torch.argsort(probs, descending=True).tolist()
        for idx in order:
            record = candidate_to_record(batch, probs, idx)
            issues = plausibility_issues(record, min_distance=args.min_distance)
            record["validity_issues"] = issues
            if issues and not args.no_validity_filter:
                rejected += 1
                issue_counts.update(issues)
                continue
            records.append(record)
            if len(records) >= args.top_k:
                break

        print(
            f"Batch generation {batch_idx + 1}/{args.max_batches}: "
            f"{len(records)}/{args.top_k} candidats valides"
        )
        if len(records) >= args.top_k:
            break

    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Candidats sauvegardés dans {args.out}")
    if not args.no_validity_filter:
        print(f"Candidats rejetés par filtre de validité : {rejected}")
        if issue_counts:
            print("Raisons principales de rejet:")
            for issue, count in issue_counts.most_common():
                print(f"  {issue}: {count}")
        if not records:
            print(
                "Aucun candidat valide trouvé. Augmente --n-samples/--max-batches, "
                "ou inspecte les rejets avec --no-validity-filter."
            )
    for rank, record in enumerate(records[:10], start=1):
        print(
            f"#{rank:02d} P(SC)={record['p_superconductor']:.4f} "
            f"SG={record['space_group']} sites={record['n_sites']}"
        )


if __name__ == "__main__":
    main()
