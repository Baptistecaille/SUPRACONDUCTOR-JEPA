import gzip
import textwrap

import numpy as np
import pandas as pd
import pytest
import torch

from models.physics_jepa.data.dataset import (
    PhysicsJEPADataset,
    PhysicsTargetStats,
    collate_physics_batch,
    create_physics_dataloader,
)

NACL_CIF = textwrap.dedent(
    """\
    data_NaCl
    _cell_length_a 5.6
    _cell_length_b 5.6
    _cell_length_c 5.6
    _cell_angle_alpha 90
    _cell_angle_beta 90
    _cell_angle_gamma 90
    _symmetry_space_group_name_H-M 'P 1'
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Na1 Na 0.0 0.0 0.0
    Cl1 Cl 0.5 0.5 0.5
    """
)


def _make_csv(tmp_path, n_rows=6):
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "source_id": f"mat-{i}",
                "cif": NACL_CIF,
                "lambda_ep": 0.2 + 0.1 * i,
                "omega_log": 150.0 + 10.0 * i,
            }
        )
    df = pd.DataFrame(rows)
    path = tmp_path / "physics_train.csv.gz"
    with gzip.open(path, "wt") as f:
        df.to_csv(f, index=False)
    return path


def test_dataset_getitem_shapes(tmp_path):
    csv_path = _make_csv(tmp_path)
    dataset = PhysicsJEPADataset(csv_path)
    assert len(dataset) == 6
    sample = dataset[0]
    assert sample["frac_coords"].shape == (2, 3)
    assert sample["atomic_numbers"].shape == (2,)
    assert sample["raw_lattice_matrix"].shape == (3, 3)
    assert sample["lambda_ep"].item() == pytest.approx(0.2)
    assert sample["omega_log"].item() == pytest.approx(150.0)


def test_dataset_drops_rows_missing_targets(tmp_path):
    csv_path = _make_csv(tmp_path)
    df = pd.read_csv(csv_path)
    df.loc[0, "lambda_ep"] = np.nan
    with gzip.open(csv_path, "wt") as f:
        df.to_csv(f, index=False)
    dataset = PhysicsJEPADataset(csv_path)
    assert len(dataset) == 5


def test_dataset_max_rows_subsamples_deterministically(tmp_path):
    csv_path = _make_csv(tmp_path, n_rows=6)
    d1 = PhysicsJEPADataset(csv_path, max_rows=3, seed=0)
    d2 = PhysicsJEPADataset(csv_path, max_rows=3, seed=0)
    assert d1.data["source_id"].tolist() == d2.data["source_id"].tolist()
    assert len(d1) == 3


def test_dataset_cache_parsed_reuses_tensors(tmp_path):
    csv_path = _make_csv(tmp_path)
    dataset = PhysicsJEPADataset(csv_path, cache_parsed=True)
    first = dataset[0]
    assert 0 in dataset._cache
    second = dataset[0]
    assert torch.equal(first["frac_coords"], second["frac_coords"])


def test_collate_and_dataloader_batch_shapes(tmp_path):
    csv_path = _make_csv(tmp_path)
    dl = create_physics_dataloader(csv_path, batch_size=4, shuffle=False, drop_last=False)
    batch = next(iter(dl))
    assert batch["frac_coords"].shape[0] == 4
    assert batch["atom_mask"].dtype == torch.bool
    assert batch["lambda_ep"].shape == (4,)
    assert batch["omega_log"].shape == (4,)


def test_missing_columns_raises(tmp_path):
    df = pd.DataFrame({"cif": [NACL_CIF], "lambda_ep": [0.3]})
    path = tmp_path / "bad.csv.gz"
    with gzip.open(path, "wt") as f:
        df.to_csv(f, index=False)
    with pytest.raises(ValueError):
        PhysicsJEPADataset(path)


class TestPhysicsTargetStats:
    def test_fit_and_normalize_roundtrip(self, tmp_path):
        csv_path = _make_csv(tmp_path)
        stats = PhysicsTargetStats.fit(csv_path)
        lambda_ep = torch.tensor([0.3, 0.5])
        omega_log = torch.tensor([180.0, 220.0])
        z_l, z_o = stats.normalize(lambda_ep, omega_log)
        lam_back, om_back = stats.denormalize(z_l, z_o)
        assert torch.allclose(lam_back, lambda_ep, atol=1e-4)
        assert torch.allclose(om_back, omega_log, atol=1e-3)

    def test_to_from_dict_roundtrip(self):
        stats = PhysicsTargetStats(1.0, 2.0, 3.0, 4.0)
        restored = PhysicsTargetStats.from_dict(stats.to_dict())
        assert restored.lambda_mean == stats.lambda_mean
        assert restored.omega_std == stats.omega_std

    def test_handles_negative_and_zero_targets(self, tmp_path):
        csv_path = _make_csv(tmp_path)
        df = pd.read_csv(csv_path)
        df.loc[0, "lambda_ep"] = -0.05
        df.loc[1, "omega_log"] = 0.0
        with gzip.open(csv_path, "wt") as f:
            df.to_csv(f, index=False)
        stats = PhysicsTargetStats.fit(csv_path)
        assert np.isfinite(stats.lambda_mean)
        assert np.isfinite(stats.omega_mean)
