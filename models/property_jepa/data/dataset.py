"""Dataset + collate pipeline feeding `models.property_jepa.model.PropertyJEPA`.

Reads one of the leak-free splits produced by
`scripts/build_property_jepa_splits.py`
(`data/processed/property_jepa_{train,val,test}.csv.gz`) and turns each row
(a `composition_key` with a canonical CIF plus whichever per-source property
columns are non-null -- see `docs/audit/consolidation_report.json`) into:

1. A rotation-invariant crystal representation: per-atom tokens
   (`models.property_jepa.model.crystal_encoder.build_atom_tokens`) plus the
   12D `lattice_invariant_features` descriptor
   (`models.property_jepa.data.lattice_features`) -- computed once in
   `__getitem__` from the parsed CIF.
2. The set of canonical properties (`data/property_schema.py`) that happen
   to be non-null for that row, each normalized to a per-property z-score
   using statistics fit on the *training* split only (`PropertyStats`,
   `compute_property_stats`) to avoid leaking val/test distribution
   information into normalization.

Which single property is *queried* (the JEPA prediction target) versus
*known* (context fed to the predictor) for a given training step is decided
per-sample, per-batch, inside `collate_property_batch` -- mirroring stage
1's `collate_crystals` (`models/crystal_structure_jepa/crystal_matrices.py`),
which likewise samples its random masked spatial block inside the collate
function rather than in `Dataset.__getitem__`. This means the same material
can serve as a training example for predicting any of its available
properties across different epochs/batches, rather than being pinned to one
fixed query property for its whole dataset lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from pymatgen.core.structure import Structure
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .lattice_features import lattice_invariant_features
from .property_schema import PROPERTY_TYPES
from ..model.crystal_encoder import build_atom_tokens

# Canonical property type -> the single consolidated_properties.csv.gz column
# it is read from. Each canonical type maps to exactly one raw column so that
# a queried/known property value is never a silent mixture of DFT
# functionals/codes (see `dft_heterogeneity_warning` in
# docs/audit/consolidation_report.json); `formation_energy_peratom` and
# `e_above_hull` are the two exceptions, deliberately reading the
# already-documented "best available" fallback columns (thermo_* built via
# combine_first across MP/JARVIS/Alexandria), same as `electronic_bandgap_best`
# and `magnetic_total_moment_best` below.
PROPERTY_COLUMNS: dict[str, str] = {
    "formation_energy_peratom": "thermo_formation_energy_peratom",
    "e_above_hull": "thermo_e_above_hull",
    "e_phase_separation": "thermo_e_phase_separation_alexandria",
    "bandgap_hse": "electronic_bandgap_hse_jarvis",
    "dos_ef": "electronic_dos_ef_alexandria",
    "magmom_total": "magnetic_total_moment_best",
    "tc_experimental": "tc_experimental",
    "tc_dft_predicted": "tc_dft_predicted",
}
# bandgap_optb88vdw, bandgap_mbj, bulk_modulus_kv, shear_modulus_gv,
# poisson_ratio, elastic_tensor_max retired -- see property_schema.py note.

assert set(PROPERTY_COLUMNS) == set(PROPERTY_TYPES), (
    "PROPERTY_COLUMNS must cover exactly the canonical types in property_schema.PROPERTY_TYPES"
)


@dataclass
class PropertyStats:
    """Per-property-type normalization statistics (mean/std), fit on train only."""

    mean: dict[str, float]
    std: dict[str, float]

    def normalize(self, property_name: str, value: float) -> float:
        std = self.std.get(property_name, 1.0)
        std = std if std > 1e-8 else 1.0
        return (value - self.mean.get(property_name, 0.0)) / std

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, float]]) -> "PropertyStats":
        return cls(mean=dict(data["mean"]), std=dict(data["std"]))


def compute_property_stats(df: pd.DataFrame) -> PropertyStats:
    """Fit per-property mean/std over whichever rows have a non-null value.

    Intended to be called on the *train* split only; the resulting
    `PropertyStats` should then be reused (not recomputed) for val/test to
    avoid leaking their distribution into normalization.
    """
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for name, column in PROPERTY_COLUMNS.items():
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(values) == 0:
            mean[name] = 0.0
            std[name] = 1.0
            continue
        mean[name] = float(values.mean())
        computed_std = float(values.std(ddof=0))
        std[name] = computed_std if computed_std > 1e-8 else 1.0
    return PropertyStats(mean=mean, std=std)


def _parse_cif_to_tensors(cif_text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parse a CIF string into `(frac_coords, atomic_numbers, lattice_matrix)`.

    Atoms are sorted deterministically by (atomic_number, x, y, z), matching
    stage 1's `structure_to_tensors`
    (`models/crystal_structure_jepa/crystal_matrices.py`), so that two
    parses of the same structure always produce the same atom ordering.
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
    lattice_matrix = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
    return frac_coords, atomic_numbers, lattice_matrix


class PropertyJEPADataset(Dataset):
    """Dataset backed by a `property_jepa_{train,val,test}.csv.gz` split file."""

    def __init__(
        self,
        csv_path: str | Path,
        stats: PropertyStats,
        min_properties: int = 1,
    ):
        self.csv_path = Path(csv_path)
        self.stats = stats
        data = pd.read_csv(self.csv_path)
        required = {"composition_key", "cif"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")

        available_counts = self._count_available_properties(data)
        self.data = data.loc[available_counts >= min_properties].reset_index(drop=True)

    @staticmethod
    def _count_available_properties(data: pd.DataFrame) -> pd.Series:
        counts = pd.Series(0, index=data.index)
        for column in PROPERTY_COLUMNS.values():
            counts = counts + pd.to_numeric(data[column], errors="coerce").notna().astype(int)
        return counts

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        row = self.data.iloc[index]
        frac_coords, atomic_numbers, lattice_matrix = _parse_cif_to_tensors(row["cif"])
        lattice_features = lattice_invariant_features(lattice_matrix)

        property_type_ids: list[int] = []
        property_values: list[float] = []
        for name, column in PROPERTY_COLUMNS.items():
            raw_value = row[column]
            if pd.isna(raw_value):
                continue
            property_type_ids.append(PROPERTY_TYPES.index(name))
            property_values.append(self.stats.normalize(name, float(raw_value)))

        if not property_type_ids:
            raise ValueError(
                f"row {index} ({row['composition_key']!r}) has no available properties; "
                "construct the dataset with min_properties >= 1"
            )

        return {
            "composition_key": str(row["composition_key"]),
            "frac_coords": frac_coords,
            "atomic_numbers": atomic_numbers,
            "lattice_features": lattice_features,
            "property_type_ids": torch.tensor(property_type_ids, dtype=torch.long),
            "property_values": torch.tensor(property_values, dtype=torch.float32),
        }


def collate_property_batch(
    samples: list[dict],
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Pad a batch and, per sample, split its available properties into query + known.

    One available property is chosen uniformly at random as the JEPA query
    target; every other available property for that sample becomes "known"
    context (padded across the batch, with `known_property_mask` marking
    real vs. padding slots -- a sample with only one available property gets
    an all-`False` known-property row, which `PropertyJEPA.forward` handles
    via `masked_mean_pool`'s all-zero-row fallback).
    """
    atom_tokens = []
    atom_counts = []
    known_type_ids = []
    known_values = []
    known_masks = []
    query_type_ids = []
    query_values = []

    for sample in samples:
        atom_tokens.append(build_atom_tokens(sample["frac_coords"], sample["atomic_numbers"]))
        atom_counts.append(sample["frac_coords"].shape[0])

        n_props = sample["property_type_ids"].shape[0]
        query_idx = int(torch.randint(0, n_props, (1,), generator=generator).item())
        known_idx = [i for i in range(n_props) if i != query_idx]

        query_type_ids.append(sample["property_type_ids"][query_idx])
        query_values.append(sample["property_values"][query_idx])
        if known_idx:
            known_type_ids.append(sample["property_type_ids"][known_idx])
            known_values.append(sample["property_values"][known_idx])
        else:
            known_type_ids.append(torch.zeros(0, dtype=torch.long))
            known_values.append(torch.zeros(0, dtype=torch.float32))
        known_masks.append(torch.ones(len(known_idx), dtype=torch.bool))

    num_atoms = torch.tensor(atom_counts, dtype=torch.long)
    padded_atom_tokens = pad_sequence(atom_tokens, batch_first=True)
    max_atoms = int(num_atoms.max().item())
    atom_mask = torch.arange(max_atoms).unsqueeze(0) < num_atoms.unsqueeze(1)

    max_known = max((t.shape[0] for t in known_type_ids), default=0)
    max_known = max(max_known, 1)  # keep a well-formed [B, >=1] tensor even if all K=0
    padded_known_type = pad_sequence(known_type_ids, batch_first=True) if max_known else None
    padded_known_value = pad_sequence(known_values, batch_first=True) if max_known else None
    padded_known_mask = pad_sequence(known_masks, batch_first=True)
    # pad_sequence pads to the true max length across the batch; if that max is 0
    # (every sample has zero known properties) pad up to `max_known` (>= 1) by hand.
    if padded_known_type is None or padded_known_type.shape[1] == 0:
        b = len(samples)
        padded_known_type = torch.zeros(b, max_known, dtype=torch.long)
        padded_known_value = torch.zeros(b, max_known, dtype=torch.float32)
        padded_known_mask = torch.zeros(b, max_known, dtype=torch.bool)

    return {
        "composition_key": [sample["composition_key"] for sample in samples],
        "atom_tokens": padded_atom_tokens,
        "atom_mask": atom_mask,
        "lattice_features": torch.stack([sample["lattice_features"] for sample in samples]),
        "known_property_type_id": padded_known_type,
        "known_property_value": padded_known_value,
        "known_property_mask": padded_known_mask,
        "query_property_type_id": torch.stack(query_type_ids),
        "query_property_value": torch.stack(query_values),
    }


def create_property_dataloader(
    csv_path: str | Path,
    stats: PropertyStats,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    min_properties: int = 1,
    drop_last: bool = False,
) -> DataLoader:
    dataset = PropertyJEPADataset(csv_path, stats=stats, min_properties=min_properties)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_property_batch,
        drop_last=drop_last,
    )


__all__ = [
    "PROPERTY_COLUMNS",
    "PropertyStats",
    "compute_property_stats",
    "PropertyJEPADataset",
    "collate_property_batch",
    "create_property_dataloader",
]
