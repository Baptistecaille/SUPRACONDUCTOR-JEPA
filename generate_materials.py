"""
generate_materials.py — Génère des candidats puis les filtre avec SUPRA-JEPA.

Exemple :
  python generate_materials.py --n-samples 128 --top-k 20
"""

import argparse
import json
import os

import torch
from pymatgen.core import Element

from config import Config
from diffusion import CrystalDDPM, diffusion_vector_to_batch
from model import SupraJEPA


IDX_TO_ELEMENT = {el.Z: el.symbol for el in Element}
IDX_TO_WYCKOFF = {idx + 1: letter for idx, letter in enumerate("abcdefghijklmnopqrstuvwxyz")}


def load_diffusion(cfg: Config, device: torch.device) -> CrystalDDPM:
    model = CrystalDDPM(cfg).to(device)
    checkpoint = torch.load(cfg.checkpoint_diffusion, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-atoms", type=int, default=1)
    parser.add_argument("--diffusion-checkpoint", type=str, default=None)
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="generated_candidates.json")
    args = parser.parse_args()

    cfg = Config()
    if args.diffusion_checkpoint is not None:
        cfg.checkpoint_diffusion = args.diffusion_checkpoint
    if args.jepa_checkpoint is not None:
        cfg.checkpoint_finetune = args.jepa_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    diffusion = load_diffusion(cfg, device)
    jepa = load_jepa(cfg, device)

    vectors = diffusion.sample(args.n_samples, device=device)
    batch = diffusion_vector_to_batch(vectors, cfg, min_atoms=args.min_atoms)
    batch = {k: v.to(device) for k, v in batch.items()}
    probs = score_candidates(jepa, batch)

    order = torch.argsort(probs, descending=True)
    keep = order[: min(args.top_k, args.n_samples)].tolist()
    records = [candidate_to_record(batch, probs, idx) for idx in keep]

    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Candidats sauvegardés dans {args.out}")
    for rank, record in enumerate(records[:10], start=1):
        print(
            f"#{rank:02d} P(SC)={record['p_superconductor']:.4f} "
            f"SG={record['space_group']} sites={record['n_sites']}"
        )


if __name__ == "__main__":
    main()
