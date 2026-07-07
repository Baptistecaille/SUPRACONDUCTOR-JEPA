"""Dataset + collate pipeline feeding `foundation_jepa` self-supervised pretraining.

Reads the SAME leak-free, composition-level splits already produced by
`scripts/build_property_jepa_splits.py`
(`data/processed/property_jepa_{train,val,test}.csv.gz`, 249,719 / 31,215 /
31,215 rows) rather than deriving a new split scheme -- the leakage
criterion this repo cares about (no val/test composition overlapping the
official Crys-JEPA MP-only pretrain corpus, `data/jepa/mp.csv.gz`) is
identical for `foundation_jepa` pretraining and `property_jepa` fine-tuning,
so reusing the existing split avoids maintaining two divergent
leakage-prevention implementations. Unlike `property_jepa`'s dataset, this
one only needs `cif` and `thermo_formation_energy_peratom` (the sole
supervisory signal used, and only inside the energy-aware InfoNCE weight,
`model/losses.py::energy_aware_weight`) -- rows missing either are dropped.

CIF parsing mirrors `models/property_jepa/data/dataset.py::_parse_cif_to_tensors`
(same deterministic atom ordering by `(atomic_number, x, y, z)`, for
reproducible tensors across parses) but additionally applies the official
Crys-JEPA `polar_decomposition_symmetrize` (`data/lattice.py`) to the raw
lattice matrix, and returns that RAW (unscaled) symmetrized matrix --
scaling by `num_atoms ** (1/3)` and the train-fit `MatrixMeanStdScaler`
happens inside `FoundationJEPA.encode` (see `model/jepa.py`), not here, so
the same parsed tensor can be reused unmodified for both the target branch
(as-is) and the context branch (rotated) without redoing CIF parsing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core.structure import Structure
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .lattice import MatrixMeanStdScaler, matrix_to_triu6, polar_decomposition_symmetrize


def _parse_cif_to_tensors(cif_text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parse a CIF string into `(frac_coords, atomic_numbers, raw_lattice_matrix)`.

    `raw_lattice_matrix` is the polar-decomposition-symmetrized lattice
    matrix in PHYSICAL (unscaled) units -- see module docstring.
    """
    structure = Structure.from_str(cif_text, fmt="cif")
    frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float32)
    atomic_numbers = torch.tensor(
        [
            max(site.species.items(), key=lambda item: (float(item[1]), item[0].Z))[0].Z
            for site in structure
        ],
        dtype=torch.long,
    )
    sort_key = (
        1000 * atomic_numbers.float()
        + 100 * frac_coords[:, 0]
        + 10 * frac_coords[:, 1]
        + frac_coords[:, 2]
    )
    order = torch.argsort(sort_key)
    frac_coords = frac_coords[order]
    atomic_numbers = atomic_numbers[order]

    raw_lattice_matrix = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
    raw_lattice_matrix = polar_decomposition_symmetrize(raw_lattice_matrix)
    return frac_coords, atomic_numbers, raw_lattice_matrix


class FoundationJEPADataset(Dataset):
    """Dataset backed by a `property_jepa_{train,val,test}.csv.gz` split file.

    Args:
        csv_path: Path to one of the three leak-free split files.
        max_atoms: Crystals with more atoms than this are dropped (keeps the
            positional-embedding table and attention cost bounded; the
            official paper's own MP corpus is dominated by small unit
            cells, so this rarely discards structures in practice).
        max_rows: If set, deterministically subsample down to this many
            rows (seeded) -- used for the compute-bounded local pretraining
            run (`training/train_foundation_jepa.py`); leave `None` for a
            full-corpus GPU run.
        seed: Seed for the `max_rows` subsampling.
        cache_parsed: If True, cache each row's parsed
            `(frac_coords, atomic_numbers, raw_lattice_matrix)` tensors in
            memory the first time it is accessed, so a multi-epoch training
            loop over the same (typically `max_rows`-bounded, CPU-scale)
            subset only pays the pymatgen CIF-parsing cost once per row
            instead of once per epoch. Leave `False` for a single-pass /
            full-corpus scan where each row is seen at most once or twice
            and the cache would only add memory pressure for no benefit.
    """

    def __init__(
        self,
        csv_path: str | Path,
        max_atoms: int = 200,
        max_rows: int | None = None,
        seed: int = 42,
        cache_parsed: bool = False,
    ):
        self.csv_path = Path(csv_path)
        data = pd.read_csv(self.csv_path)
        required = {"composition_key", "cif", "thermo_formation_energy_peratom"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")

        data = data.loc[data["cif"].notna() & data["thermo_formation_energy_peratom"].notna()]
        data = data.reset_index(drop=True)

        if max_rows is not None and len(data) > max_rows:
            rng = np.random.default_rng(seed)
            keep = rng.choice(len(data), size=max_rows, replace=False)
            data = data.iloc[np.sort(keep)].reset_index(drop=True)

        self.max_atoms = max_atoms
        self.data = data
        self.cache_parsed = cache_parsed
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        row = self.data.iloc[index]
        if self.cache_parsed and index in self._cache:
            frac_coords, atomic_numbers, raw_lattice_matrix = self._cache[index]
        else:
            frac_coords, atomic_numbers, raw_lattice_matrix = _parse_cif_to_tensors(row["cif"])
            if self.cache_parsed:
                self._cache[index] = (frac_coords, atomic_numbers, raw_lattice_matrix)
        if frac_coords.shape[0] > self.max_atoms:
            # Deterministic fallback: try the next row (wrap around) so a
            # DataLoader index never raises mid-epoch on an oversized crystal.
            return self.__getitem__((index + 1) % len(self))
        return {
            "composition_key": str(row["composition_key"]),
            "frac_coords": frac_coords,
            "atomic_numbers": atomic_numbers,
            "raw_lattice_matrix": raw_lattice_matrix,
            "formation_energy_peratom": torch.tensor(
                float(row["thermo_formation_energy_peratom"]), dtype=torch.float32
            ),
        }


def collate_foundation_batch(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Pad a batch of parsed crystals into dense `[B, N, ...]` tensors + mask."""
    frac_coords = [s["frac_coords"] for s in samples]
    atomic_numbers = [s["atomic_numbers"] for s in samples]
    num_atoms = torch.tensor([f.shape[0] for f in frac_coords], dtype=torch.long)

    padded_frac = pad_sequence(frac_coords, batch_first=True)
    padded_atomic_numbers = pad_sequence(atomic_numbers, batch_first=True, padding_value=1)
    max_atoms = int(num_atoms.max().item())
    atom_mask = torch.arange(max_atoms).unsqueeze(0) < num_atoms.unsqueeze(1)

    return {
        "composition_key": [s["composition_key"] for s in samples],
        "frac_coords": padded_frac,
        "atomic_numbers": padded_atomic_numbers,
        "atom_mask": atom_mask,
        "num_atoms": num_atoms,
        "raw_lattice_matrix": torch.stack([s["raw_lattice_matrix"] for s in samples]),
        "formation_energy_peratom": torch.stack([s["formation_energy_peratom"] for s in samples]),
    }


def create_foundation_dataloader(
    csv_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    max_atoms: int = 200,
    max_rows: int | None = None,
    seed: int = 42,
    drop_last: bool = True,
    cache_parsed: bool = False,
) -> DataLoader:
    """`drop_last=True` by default: the energy-aware InfoNCE loss needs `B >= 2`
    to form any negative pairs, so a ragged final batch of size 1 would be
    degenerate (loss well-defined for B=1 by convention, but statistically
    uninformative and worth simply dropping)."""
    dataset = FoundationJEPADataset(
        csv_path, max_atoms=max_atoms, max_rows=max_rows, seed=seed, cache_parsed=cache_parsed
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_foundation_batch,
        drop_last=drop_last,
    )


def fit_matrix_scaler(
    csv_path: str | Path,
    n_samples: int = 2000,
    max_atoms: int = 200,
    seed: int = 42,
) -> MatrixMeanStdScaler:
    """Fit `MatrixMeanStdScaler` on a random subsample of a split's lattice descriptors.

    Must be called on the TRAIN split only. Subsamples (rather than parsing
    every CIF) since parsing ~250k CIFs purely to fit a 6D mean/std is
    wasteful; `n_samples=2000` gives a stable estimate in practice (verified
    against a full-train-split fit in `tests/test_dataset.py`) while keeping
    this a fast, single-CPU-core operation.
    """
    dataset = FoundationJEPADataset(csv_path, max_atoms=max_atoms, max_rows=n_samples, seed=seed)
    triu6_list = []
    for i in range(len(dataset)):
        sample = dataset[i]
        triu6 = matrix_to_triu6(sample["raw_lattice_matrix"])
        n_atoms = sample["frac_coords"].shape[0]
        triu6_scaled = triu6 / (n_atoms ** (1 / 3))
        triu6_list.append(triu6_scaled)
    stacked = torch.stack(triu6_list, dim=0)
    return MatrixMeanStdScaler.fit(stacked)


__all__ = [
    "FoundationJEPADataset",
    "collate_foundation_batch",
    "create_foundation_dataloader",
    "fit_matrix_scaler",
]
