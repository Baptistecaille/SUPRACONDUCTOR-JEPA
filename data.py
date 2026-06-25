"""
data.py — Tokenisation des structures cristallines + datasets PyTorch

Représentation d'un cristal :
  [CLS]  global_token(space_group, lattice)
  tok_1  atom_token(element, wyckoff, frac_coords)
  tok_2  …
  …
  [PAD]  padding jusqu'à max_atoms + 1 tokens

Références :
  - S2SNet (Liu et al. 2023) : tokenisation par site atomique + MLM
  - Crystalformer (Taniai et al. 2024) : encodage périodique des cristaux
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pymatgen.core import Structure, Element


# ── Constantes ────────────────────────────────────────────────────────────────

ELEMENT_TO_IDX: dict[str, int] = {
    el.symbol: el.Z for el in Element
}  # H→1 … Og→118, 0 réservé pour PAD

WYCKOFF_TO_IDX: dict[str, int] = {
    letter: idx + 1 for idx, letter in enumerate("abcdefghijklmnopqrstuvwxyz")
}  # a→1 … z→26, 0 réservé pour PAD


# ── Tokeniseur ────────────────────────────────────────────────────────────────

class CrystalTokenizer:
    """Convertit une pymatgen.Structure en tenseurs de tokens.

    Chaque site atomique → un token discret + coordonnées fractionnelles.
    Le premier token [CLS] encode les informations globales du cristal
    (groupe d'espace + paramètres de maille normalisés).
    """

    def __init__(self, max_atoms: int = 64):
        self.max_atoms = max_atoms  # hors CLS

    def tokenize(self, structure: Structure, label: int = -1) -> dict:
        """
        Returns
        -------
        dict avec :
          element_ids   : (max_atoms+1,)  int   — 0=PAD, position 0=CLS (sg)
          wyckoff_ids   : (max_atoms+1,)  int
          frac_coords   : (max_atoms+1, 3) float — coords normalisées [0,1]
          lattice_feat  : (6,)            float — (a,b,c,α,β,γ) normalisés
          sg_id         : ()              int   — numéro de groupe d'espace
          padding_mask  : (max_atoms+1,)  bool  — True = position valide
          label         : ()              int   — 1=SC, 0=non-SC, -1=inconnu
        """
        n = len(structure)
        n_tok = min(n, self.max_atoms)

        element_ids = np.zeros(self.max_atoms + 1, dtype=np.int64)
        wyckoff_ids = np.zeros(self.max_atoms + 1, dtype=np.int64)
        frac_coords = np.zeros((self.max_atoms + 1, 3), dtype=np.float32)
        padding_mask = np.zeros(self.max_atoms + 1, dtype=bool)

        # ── Token CLS (index 0) ───────────────────────────────────────────────
        try:
            sg = structure.get_space_group_info()[1]  # numéro 1-230
        except Exception:
            sg = 1
        element_ids[0] = 0       # réservé CLS
        wyckoff_ids[0] = 0
        frac_coords[0] = 0.0
        padding_mask[0] = True

        # ── Tokens atomiques (indices 1…n_tok) ────────────────────────────────
        try:
            analyzer = structure.get_symmetry_dataset()
            wyckoff_letters = analyzer.get("wyckoff_letters", ["a"] * n)
        except Exception:
            wyckoff_letters = ["a"] * n

        for i in range(n_tok):
            site = structure[i]
            el = site.specie.symbol if hasattr(site.specie, "symbol") else str(site.specie)
            element_ids[i + 1] = ELEMENT_TO_IDX.get(el, 1)
            wyckoff_ids[i + 1] = WYCKOFF_TO_IDX.get(
                wyckoff_letters[i] if i < len(wyckoff_letters) else "a", 1
            )
            frac_coords[i + 1] = np.mod(site.frac_coords, 1.0)
            padding_mask[i + 1] = True

        # ── Paramètres de maille normalisés ──────────────────────────────────
        lat = structure.lattice
        # Normalisation simple : a,b,c en Å /20, angles /180
        lattice_feat = np.array(
            [lat.a / 20.0, lat.b / 20.0, lat.c / 20.0,
             lat.alpha / 180.0, lat.beta / 180.0, lat.gamma / 180.0],
            dtype=np.float32,
        )

        return {
            "element_ids":  torch.from_numpy(element_ids),
            "wyckoff_ids":  torch.from_numpy(wyckoff_ids),
            "frac_coords":  torch.from_numpy(frac_coords),
            "lattice_feat": torch.from_numpy(lattice_feat),
            "sg_id":        torch.tensor(sg, dtype=torch.long),
            "padding_mask": torch.from_numpy(padding_mask),
            "label":        torch.tensor(label, dtype=torch.long),
        }


# ── Stratégie de masquage JEPA ────────────────────────────────────────────────

def random_mask(padding_mask: torch.Tensor, mask_ratio: float = 0.40) -> torch.Tensor:
    """Masque aléatoire sur les tokens atomiques (hors CLS à position 0).

    Returns
    -------
    mask : BoolTensor (seq_len,) — True = token masqué
    """
    seq_len = padding_mask.shape[0]
    valid_indices = torch.where(padding_mask[1:])[0] + 1  # exclut CLS
    n_mask = max(1, int(len(valid_indices) * mask_ratio))
    perm = torch.randperm(len(valid_indices))[:n_mask]
    masked_positions = valid_indices[perm]
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[masked_positions] = True
    return mask


# ── Datasets ──────────────────────────────────────────────────────────────────

class CrystalDataset(Dataset):
    """Dataset générique : liste de (Structure, label).

    Pour le prétraining, label = -1 (non utilisé).
    Pour le fine-tuning, label ∈ {0, 1}.
    """

    def __init__(
        self,
        structures: list[Structure],
        labels: list[int],
        max_atoms: int = 64,
        mask_ratio: float | None = None,  # None = pas de masquage (fine-tuning)
    ):
        self.structures = structures
        self.labels = labels
        self.tokenizer = CrystalTokenizer(max_atoms=max_atoms)
        self.mask_ratio = mask_ratio

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(self, idx: int) -> dict:
        tokens = self.tokenizer.tokenize(self.structures[idx], self.labels[idx])
        if self.mask_ratio is not None:
            tokens["mask"] = random_mask(tokens["padding_mask"], self.mask_ratio)
        else:
            tokens["mask"] = torch.zeros_like(tokens["padding_mask"])
        return tokens


# ── Chargement des données ────────────────────────────────────────────────────

def load_supercon_dataset(data_dir: str = "data"):
    """Charge le dataset SuperCon via matminer.

    Retourne deux listes : structures pymatgen, labels (1=SC, 0=non-SC).

    Nécessite :
      pip install matminer mp-api pymatgen spglib

    Pour les non-supraconducteurs, on utilise un échantillon du Materials Project
    sans supraconductivité connue. La fonction attend :
      data_dir/supercon_structures.json   (exporté depuis MP)
      data_dir/nonsc_structures.json
    """
    import json

    def _load_json(path):
        with open(path) as f:
            data = json.load(f)
        structs = [Structure.from_dict(d["structure"]) for d in data]
        labels = [d["label"] for d in data]
        return structs, labels

    sc_path = os.path.join(data_dir, "supercon_structures.json")
    nonsc_path = os.path.join(data_dir, "nonsc_structures.json")

    if not os.path.exists(sc_path):
        raise FileNotFoundError(
            f"{sc_path} introuvable.\n"
            "Lance d'abord scripts/prepare_data.py pour télécharger les structures."
        )

    sc_structs, sc_labels = _load_json(sc_path)
    nonsc_structs, nonsc_labels = _load_json(nonsc_path)

    structures = sc_structs + nonsc_structs
    labels = sc_labels + nonsc_labels
    return structures, labels


def make_dataloaders(
    structures, labels, config, val_ratio=0.15, test_ratio=0.15
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split train/val/test et retourne les DataLoaders."""
    import random
    random.seed(config.seed)

    n = len(structures)
    indices = list(range(n))
    random.shuffle(indices)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = indices[:n_test]
    val_idx = indices[n_test: n_test + n_val]
    train_idx = indices[n_test + n_val:]

    def _subset(idxs, mask_ratio):
        s = [structures[i] for i in idxs]
        l = [labels[i] for i in idxs]
        return CrystalDataset(s, l, config.max_atoms, mask_ratio)

    train_ds = _subset(train_idx, config.mask_ratio)
    val_ds = _subset(val_idx, None)
    test_ds = _subset(test_idx, None)

    train_dl = DataLoader(train_ds, batch_size=config.batch_size_pretrain, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=config.batch_size_finetune, shuffle=False, num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=config.batch_size_finetune, shuffle=False, num_workers=2)

    return train_dl, val_dl, test_dl
