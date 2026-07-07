"""Tests for `models.foundation_jepa.data.dataset`."""

from __future__ import annotations

import gzip

import pandas as pd
import pytest
import torch
from pymatgen.core import Lattice, Structure

from models.foundation_jepa.data.dataset import (
    FoundationJEPADataset,
    collate_foundation_batch,
    create_foundation_dataloader,
    fit_matrix_scaler,
)


def _nacl_cif() -> str:
    lattice = Lattice.cubic(5.64)
    structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return structure.to(fmt="cif")


def _mgo_cif() -> str:
    lattice = Lattice.cubic(4.21)
    structure = Structure(lattice, ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return structure.to(fmt="cif")


def _perovskite_cif() -> str:
    lattice = Lattice.cubic(3.9)
    structure = Structure(
        lattice,
        ["Ca", "Ti", "O", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )
    return structure.to(fmt="cif")


def _make_fixture_df(n_repeats: int = 3) -> pd.DataFrame:
    rows = []
    cifs = [_nacl_cif(), _mgo_cif(), _perovskite_cif()]
    energies = [-2.1, -3.0, -3.4]
    keys = ["NaCl", "MgO", "CaTiO3"]
    for _ in range(n_repeats):
        for key, cif, energy in zip(keys, cifs, energies):
            rows.append(
                {
                    "composition_key": key,
                    "cif": cif,
                    "thermo_formation_energy_peratom": energy,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def fixture_csv(tmp_path):
    df = _make_fixture_df(n_repeats=4)
    path = tmp_path / "toy_foundation.csv.gz"
    with gzip.open(path, "wt") as f:
        df.to_csv(f, index=False)
    return path


def test_dataset_len_and_getitem_shapes(fixture_csv):
    ds = FoundationJEPADataset(fixture_csv)
    assert len(ds) == 12
    sample = ds[0]
    assert sample["frac_coords"].shape[-1] == 3
    assert sample["atomic_numbers"].dim() == 1
    assert sample["raw_lattice_matrix"].shape == (3, 3)
    assert sample["formation_energy_peratom"].shape == ()


def test_dataset_drops_rows_missing_required_columns(tmp_path):
    df = _make_fixture_df(n_repeats=1)
    df.loc[0, "thermo_formation_energy_peratom"] = float("nan")
    path = tmp_path / "toy_missing.csv.gz"
    with gzip.open(path, "wt") as f:
        df.to_csv(f, index=False)
    ds = FoundationJEPADataset(path)
    assert len(ds) == 2  # one row dropped


def test_dataset_requires_needed_columns(tmp_path):
    df = pd.DataFrame({"composition_key": ["x"], "cif": ["not a cif"]})
    path = tmp_path / "toy_bad.csv.gz"
    with gzip.open(path, "wt") as f:
        df.to_csv(f, index=False)
    with pytest.raises(ValueError):
        FoundationJEPADataset(path)


def test_dataset_max_rows_subsampling_deterministic(fixture_csv):
    ds1 = FoundationJEPADataset(fixture_csv, max_rows=5, seed=42)
    ds2 = FoundationJEPADataset(fixture_csv, max_rows=5, seed=42)
    assert len(ds1) == 5
    assert ds1.data["composition_key"].tolist() == ds2.data["composition_key"].tolist()


def test_collate_foundation_batch_padding_and_mask(fixture_csv):
    ds = FoundationJEPADataset(fixture_csv)
    samples = [ds[i] for i in range(4)]
    batch = collate_foundation_batch(samples)

    max_atoms = max(s["frac_coords"].shape[0] for s in samples)
    assert batch["frac_coords"].shape == (4, max_atoms, 3)
    assert batch["atomic_numbers"].shape == (4, max_atoms)
    assert batch["atom_mask"].shape == (4, max_atoms)
    assert batch["raw_lattice_matrix"].shape == (4, 3, 3)
    assert batch["formation_energy_peratom"].shape == (4,)

    for i, s in enumerate(samples):
        n = s["frac_coords"].shape[0]
        assert batch["atom_mask"][i, :n].all()
        if n < max_atoms:
            assert not batch["atom_mask"][i, n:].any()


def test_create_foundation_dataloader_batches(fixture_csv):
    dl = create_foundation_dataloader(fixture_csv, batch_size=4, shuffle=False, drop_last=False)
    batches = list(dl)
    total = sum(b["frac_coords"].shape[0] for b in batches)
    assert total == 12


def test_fit_matrix_scaler_shapes(fixture_csv):
    scaler = fit_matrix_scaler(fixture_csv, n_samples=12)
    assert scaler.mean.shape == (1, 6)
    assert scaler.std.shape == (1, 6)
    assert torch.isfinite(scaler.mean).all()
    assert torch.isfinite(scaler.std).all()
