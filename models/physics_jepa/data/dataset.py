"""Dataset + collate pipeline feeding stage-3 physics-head fine-tuning.

Reads `data/processed/physics_jepa_{train,val,test}.csv.gz`
(`scripts/build_physics_dataset.py`; 7569 / 952 / 951 rows, 80/10/10 split
by whole composition, stratified on log10(Tc) deciles, seed=42) and
supervises on the two electron-phonon coupling constants needed to compute
Tc via the Allen-Dynes formula:

  * `lambda_ep`: electron-phonon coupling constant (dimensionless; train
    range ~[-0.05, 15.8], heavily right-skewed, median ~0.37).
  * `omega_log`: logarithmic-average phonon frequency (Kelvin; train range
    [0, 1165], median ~213).

CIF parsing reuses `foundation_jepa.data.dataset._parse_cif_to_tensors`
(same deterministic atom ordering + polar-decomposition lattice
symmetrization) so a `PhysicsJEPADataset` sample is tensor-for-tensor
compatible with `FoundationJEPA.encode(...)` -- the whole point being to
load the pretrained encoder and fine-tune small heads on top of it without
reimplementing CIF parsing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ...foundation_jepa.data.dataset import _parse_cif_to_tensors

TARGET_COLUMNS = ("lambda_ep", "omega_log")


class PhysicsJEPADataset(Dataset):
    """Dataset backed by a `physics_jepa_{train,val,test}.csv.gz` split file.

    Args:
        csv_path: Path to one of the three leak-free physics split files.
        max_atoms: Crystals with more atoms than this are dropped.
        max_rows: If set, deterministically subsample down to this many
            rows (seeded) -- used for a compute-bounded local fine-tuning
            run; leave `None` for a full-split GPU run.
        seed: Seed for the `max_rows` subsampling.
        cache_parsed: Cache each row's parsed CIF tensors in memory after
            first access (see `FoundationJEPADataset` for the rationale --
            identical here since fine-tuning also revisits the same rows
            across epochs).
    """

    def __init__(
        self,
        csv_path: str | Path,
        max_atoms: int = 200,
        max_rows: int | None = None,
        seed: int = 42,
        cache_parsed: bool = True,
    ):
        self.csv_path = Path(csv_path)
        data = pd.read_csv(self.csv_path)
        required = {"cif", *TARGET_COLUMNS}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")

        mask = data["cif"].notna()
        for col in TARGET_COLUMNS:
            mask &= data[col].notna()
        data = data.loc[mask].reset_index(drop=True)

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
            return self.__getitem__((index + 1) % len(self))
        return {
            "source_id": str(row.get("source_id", index)),
            "frac_coords": frac_coords,
            "atomic_numbers": atomic_numbers,
            "raw_lattice_matrix": raw_lattice_matrix,
            "lambda_ep": torch.tensor(float(row["lambda_ep"]), dtype=torch.float32),
            "omega_log": torch.tensor(float(row["omega_log"]), dtype=torch.float32),
        }


def collate_physics_batch(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Pad a batch of parsed crystals + physics targets into dense tensors + mask."""
    frac_coords = [s["frac_coords"] for s in samples]
    atomic_numbers = [s["atomic_numbers"] for s in samples]
    num_atoms = torch.tensor([f.shape[0] for f in frac_coords], dtype=torch.long)

    padded_frac = pad_sequence(frac_coords, batch_first=True)
    padded_atomic_numbers = pad_sequence(atomic_numbers, batch_first=True, padding_value=1)
    max_atoms = int(num_atoms.max().item())
    atom_mask = torch.arange(max_atoms).unsqueeze(0) < num_atoms.unsqueeze(1)

    return {
        "source_id": [s["source_id"] for s in samples],
        "frac_coords": padded_frac,
        "atomic_numbers": padded_atomic_numbers,
        "atom_mask": atom_mask,
        "num_atoms": num_atoms,
        "raw_lattice_matrix": torch.stack([s["raw_lattice_matrix"] for s in samples]),
        "lambda_ep": torch.stack([s["lambda_ep"] for s in samples]),
        "omega_log": torch.stack([s["omega_log"] for s in samples]),
    }


def create_physics_dataloader(
    csv_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    max_atoms: int = 200,
    max_rows: int | None = None,
    seed: int = 42,
    drop_last: bool = False,
    cache_parsed: bool = True,
) -> DataLoader:
    dataset = PhysicsJEPADataset(
        csv_path, max_atoms=max_atoms, max_rows=max_rows, seed=seed, cache_parsed=cache_parsed
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_physics_batch,
        drop_last=drop_last,
    )


class PhysicsTargetStats:
    """Per-target mean/std of `log1p(lambda_ep)` and `log(omega_log + 1)`.

    Both targets are strictly positive (up to one train-split outlier,
    `lambda_ep`'s min is -0.0471 -- see module docstring in
    `scripts/build_physics_dataset.py`'s report) and heavily right-skewed
    (`lambda_ep` max/median ~ 43x, `omega_log` max/median ~ 5.5x), so heads
    are trained in log space and predictions exponentiated back for
    evaluation/Tc computation. `log1p` (rather than plain `log`) tolerates
    the handful of values at or near zero without producing `-inf`.
    """

    def __init__(self, lambda_mean: float, lambda_std: float, omega_mean: float, omega_std: float):
        self.lambda_mean = lambda_mean
        self.lambda_std = lambda_std
        self.omega_mean = omega_mean
        self.omega_std = omega_std

    @classmethod
    def fit(cls, csv_path: str | Path) -> "PhysicsTargetStats":
        df = pd.read_csv(csv_path)
        log_lambda = np.log1p(df["lambda_ep"].clip(lower=0.0))
        log_omega = np.log1p(df["omega_log"].clip(lower=0.0))
        return cls(
            lambda_mean=float(log_lambda.mean()),
            lambda_std=float(log_lambda.std() + 1e-8),
            omega_mean=float(log_omega.mean()),
            omega_std=float(log_omega.std() + 1e-8),
        )

    def normalize(self, lambda_ep: torch.Tensor, omega_log: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_lambda = torch.log1p(lambda_ep.clamp(min=0.0))
        log_omega = torch.log1p(omega_log.clamp(min=0.0))
        z_lambda = (log_lambda - self.lambda_mean) / self.lambda_std
        z_omega = (log_omega - self.omega_mean) / self.omega_std
        return z_lambda, z_omega

    def denormalize(self, z_lambda: torch.Tensor, z_omega: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_lambda = z_lambda * self.lambda_std + self.lambda_mean
        log_omega = z_omega * self.omega_std + self.omega_mean
        lambda_ep = torch.expm1(log_lambda)
        omega_log = torch.expm1(log_omega)
        return lambda_ep, omega_log

    def to_dict(self) -> dict[str, float]:
        return {
            "lambda_mean": self.lambda_mean,
            "lambda_std": self.lambda_std,
            "omega_mean": self.omega_mean,
            "omega_std": self.omega_std,
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "PhysicsTargetStats":
        return cls(d["lambda_mean"], d["lambda_std"], d["omega_mean"], d["omega_std"])


__all__ = [
    "TARGET_COLUMNS",
    "PhysicsJEPADataset",
    "collate_physics_batch",
    "create_physics_dataloader",
    "PhysicsTargetStats",
]
