"""Tests for `models.property_jepa.data.dataset`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from pymatgen.core import Lattice, Structure

from models.property_jepa.data.dataset import (
    PROPERTY_COLUMNS,
    PropertyJEPADataset,
    PropertyStats,
    collate_property_batch,
    compute_property_stats,
    create_property_dataloader,
)
from models.property_jepa.data.property_schema import PROPERTY_TYPES


def _cif(structure: Structure) -> str:
    return structure.to(fmt="cif")


def _nacl_cif() -> str:
    lattice = Lattice.cubic(5.64)
    structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return _cif(structure)


def _mgo_cif() -> str:
    lattice = Lattice.cubic(4.21)
    structure = Structure(lattice, ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return _cif(structure)


def _empty_row(composition_key: str, cif: str) -> dict:
    row = {column: np.nan for column in PROPERTY_COLUMNS.values()}
    row["composition_key"] = composition_key
    row["cif"] = cif
    return row


def _make_fixture_df() -> pd.DataFrame:
    """Three rows: two with several properties, one with exactly one."""
    rows = []

    row = _empty_row("NaCl", _nacl_cif())
    row["thermo_formation_energy_peratom"] = -2.1
    row["thermo_e_above_hull"] = 0.0
    row["electronic_bandgap_hse_jarvis"] = 5.0
    rows.append(row)

    row = _empty_row("MgO", _mgo_cif())
    row["thermo_formation_energy_peratom"] = -3.0
    row["magnetic_total_moment_best"] = 160.0
    rows.append(row)

    row = _empty_row("NaCl2", _nacl_cif())
    row["tc_experimental"] = 10.0
    rows.append(row)

    return pd.DataFrame(rows)


def _write_fixture_csv(tmp_path) -> str:
    df = _make_fixture_df()
    path = tmp_path / "fixture_split.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    return str(path)


def _unit_stats() -> PropertyStats:
    return PropertyStats(mean={name: 0.0 for name in PROPERTY_TYPES}, std={name: 1.0 for name in PROPERTY_TYPES})


def test_property_columns_cover_schema():
    assert set(PROPERTY_COLUMNS) == set(PROPERTY_TYPES)


def test_compute_property_stats_fits_mean_std():
    df = _make_fixture_df()
    stats = compute_property_stats(df)
    assert stats.mean["formation_energy_peratom"] == pytest.approx((-2.1 + -3.0) / 2)
    # a property with zero observed rows falls back to mean=0, std=1
    assert stats.mean["e_phase_separation"] == 0.0
    assert stats.std["e_phase_separation"] == 1.0


def test_property_stats_normalize_roundtrip():
    stats = PropertyStats(mean={"tc_experimental": 5.0}, std={"tc_experimental": 2.0})
    assert stats.normalize("tc_experimental", 9.0) == pytest.approx(2.0)


def test_property_stats_normalize_guards_zero_std():
    stats = PropertyStats(mean={"tc_experimental": 5.0}, std={"tc_experimental": 0.0})
    # std collapses to 1.0 instead of dividing by (near) zero
    assert stats.normalize("tc_experimental", 6.0) == pytest.approx(1.0)


def test_property_stats_to_dict_from_dict_roundtrip():
    stats = PropertyStats(mean={"tc_experimental": 1.0}, std={"tc_experimental": 2.0})
    restored = PropertyStats.from_dict(stats.to_dict())
    assert restored.mean == stats.mean
    assert restored.std == stats.std


def test_dataset_filters_by_min_properties(tmp_path):
    csv_path = _write_fixture_csv(tmp_path)
    stats = _unit_stats()

    dataset_all = PropertyJEPADataset(csv_path, stats=stats, min_properties=1)
    assert len(dataset_all) == 3

    dataset_strict = PropertyJEPADataset(csv_path, stats=stats, min_properties=2)
    assert len(dataset_strict) == 2  # NaCl2 has exactly one available property


def test_dataset_missing_columns_raises(tmp_path):
    df = pd.DataFrame({"composition_key": ["A"], "not_cif": ["x"]})
    path = tmp_path / "bad.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    with pytest.raises(ValueError):
        PropertyJEPADataset(str(path), stats=_unit_stats())


def test_dataset_getitem_shapes(tmp_path):
    csv_path = _write_fixture_csv(tmp_path)
    dataset = PropertyJEPADataset(csv_path, stats=_unit_stats(), min_properties=1)
    sample = dataset[0]
    assert sample["composition_key"] == "NaCl"
    assert sample["frac_coords"].shape == (2, 3)
    assert sample["atomic_numbers"].shape == (2,)
    assert sample["lattice_features"].shape == (12,)
    assert sample["property_type_ids"].shape == sample["property_values"].shape
    assert sample["property_type_ids"].numel() == 3  # NaCl row has 3 non-null properties


def test_dataset_single_property_row(tmp_path):
    csv_path = _write_fixture_csv(tmp_path)
    dataset = PropertyJEPADataset(csv_path, stats=_unit_stats(), min_properties=1)
    # NaCl2 (index 2) has exactly one available property: tc_experimental
    sample = dataset[2]
    assert sample["property_type_ids"].numel() == 1
    assert PROPERTY_TYPES[sample["property_type_ids"][0].item()] == "tc_experimental"


def test_collate_batch_shapes_and_query_known_split(tmp_path):
    csv_path = _write_fixture_csv(tmp_path)
    dataset = PropertyJEPADataset(csv_path, stats=_unit_stats(), min_properties=1)
    samples = [dataset[i] for i in range(len(dataset))]
    generator = torch.Generator().manual_seed(0)
    batch = collate_property_batch(samples, generator=generator)

    b = len(samples)
    max_atoms = max(s["frac_coords"].shape[0] for s in samples)
    assert batch["atom_tokens"].shape[0] == b
    assert batch["atom_tokens"].shape[1] == max_atoms
    assert batch["atom_mask"].shape == (b, max_atoms)
    assert batch["lattice_features"].shape == (b, 12)
    assert batch["query_property_type_id"].shape == (b,)
    assert batch["query_property_value"].shape == (b,)
    assert batch["known_property_type_id"].shape[0] == b
    assert batch["known_property_mask"].dtype == torch.bool


def test_collate_batch_single_property_sample_has_empty_known(tmp_path):
    csv_path = _write_fixture_csv(tmp_path)
    dataset = PropertyJEPADataset(csv_path, stats=_unit_stats(), min_properties=1)
    # Isolate the single-property sample (NaCl2) in its own batch.
    sample = dataset[2]
    batch = collate_property_batch([sample])
    assert batch["known_property_mask"].shape == (1, 1)
    assert not batch["known_property_mask"][0].any()
    # the lone available property must have been used as the query, not dropped
    assert PROPERTY_TYPES[batch["query_property_type_id"][0].item()] == "tc_experimental"


def test_collate_batch_all_samples_zero_known_properties():
    # Two synthetic single-property samples: every sample's only property
    # becomes the query, so known_property_mask must be all-False for all rows.
    samples = []
    for i in range(2):
        samples.append(
            {
                "composition_key": f"X{i}",
                "frac_coords": torch.zeros(1, 3),
                "atomic_numbers": torch.tensor([1]),
                "lattice_features": torch.zeros(12),
                "property_type_ids": torch.tensor([0]),
                "property_values": torch.tensor([0.5]),
            }
        )
    batch = collate_property_batch(samples)
    assert batch["known_property_mask"].shape == (2, 1)
    assert not batch["known_property_mask"].any()
    assert batch["known_property_type_id"].shape == (2, 1)


def test_create_property_dataloader_iterates(tmp_path):
    csv_path = _write_fixture_csv(tmp_path)
    stats = _unit_stats()
    loader = create_property_dataloader(csv_path, stats=stats, batch_size=2, shuffle=False, num_workers=0)
    batches = list(loader)
    assert len(batches) == 2  # 3 samples, batch_size=2, drop_last=False -> batches of 2 and 1
    assert batches[0]["atom_tokens"].shape[0] == 2
    assert batches[1]["atom_tokens"].shape[0] == 1
