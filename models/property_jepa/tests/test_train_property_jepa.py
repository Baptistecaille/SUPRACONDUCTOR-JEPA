"""Tests for `models.property_jepa.training.train_property_jepa`.

Covers `build_model`, `train_one_step`, `save_checkpoint`, and a small
end-to-end smoke test of `train()` against a synthetic fixture CSV (no real
`data/processed/property_jepa_*.csv.gz` split required).
"""

from __future__ import annotations

import argparse

import pandas as pd
import pytest
import torch
from pymatgen.core import Lattice, Structure

from models.property_jepa.data.dataset import PROPERTY_COLUMNS, compute_property_stats, create_property_dataloader
from models.property_jepa.model.jepa import PropertyJEPA
from models.property_jepa.training.schedulers import CosineWithWarmup
from models.property_jepa.training.train_property_jepa import (
    build_model,
    estimate_total_steps,
    move_batch_to_device,
    save_checkpoint,
    train,
    train_one_step,
)


def _cif(structure: Structure) -> str:
    return structure.to(fmt="cif")


def _fixture_df(n_rows: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        lattice = Lattice.cubic(4.0 + 0.1 * i)
        structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        row = {column: float("nan") for column in PROPERTY_COLUMNS.values()}
        row["composition_key"] = f"Mat{i}"
        row["cif"] = _cif(structure)
        row["thermo_formation_energy_peratom"] = -1.0 - 0.1 * i
        row["magnetic_total_moment_best"] = 100.0 + i
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def fixture_csv(tmp_path):
    path = tmp_path / "fixture_train.csv.gz"
    _fixture_df().to_csv(path, index=False, compression="gzip")
    return str(path)


def test_build_model_returns_property_jepa():
    model = build_model(hidden_dim=32, layers=2, attn_heads=2)
    assert isinstance(model, PropertyJEPA)


def test_build_model_with_vc_regularizer():
    model = build_model(hidden_dim=32, layers=2, attn_heads=2, regularizer="vc", reg_weight=0.5)
    assert model.reg_weight == 0.5
    assert model.regularizer is not None


def test_build_model_rejects_unknown_regularizer():
    with pytest.raises(ValueError):
        build_model(regularizer="not-a-real-regularizer")


def test_train_one_step_updates_parameters(fixture_csv):
    df = pd.read_csv(fixture_csv)
    stats = compute_property_stats(df)
    loader = create_property_dataloader(fixture_csv, stats=stats, batch_size=3, shuffle=False)
    batch = next(iter(loader))

    model = build_model(hidden_dim=32, layers=2, attn_heads=2)
    optimizer = torch.optim.AdamW(
        list(model.crystal_encoder.parameters())
        + list(model.property_encoder.parameters())
        + list(model.predictor.parameters()),
        lr=1e-3,
    )
    before = [p.clone() for p in model.crystal_encoder.parameters()]
    metrics = train_one_step(model, batch, optimizer)
    after = list(model.crystal_encoder.parameters())

    assert "loss" in metrics and "loss_pred" in metrics
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_move_batch_to_device_is_noop_on_cpu(fixture_csv):
    df = pd.read_csv(fixture_csv)
    stats = compute_property_stats(df)
    loader = create_property_dataloader(fixture_csv, stats=stats, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    moved = move_batch_to_device(batch, torch.device("cpu"))
    assert moved["atom_tokens"].device.type == "cpu"
    assert moved["composition_key"] == batch["composition_key"]


def test_save_checkpoint_roundtrip(tmp_path):
    model = build_model(hidden_dim=32, layers=2, attn_heads=2)
    optimizer = torch.optim.AdamW(model.crystal_encoder.parameters(), lr=1e-3)
    stats = compute_property_stats(_fixture_df())
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, epoch=0, step=1, stats=stats, config={"foo": "bar"})

    assert path.exists()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 0
    assert checkpoint["step"] == 1
    assert checkpoint["config"] == {"foo": "bar"}
    assert "model_state_dict" in checkpoint
    assert checkpoint["property_stats"]["mean"] == stats.mean


def test_estimate_total_steps_respects_max_batches(fixture_csv):
    df = pd.read_csv(fixture_csv)
    stats = compute_property_stats(df)
    loader = create_property_dataloader(fixture_csv, stats=stats, batch_size=2, shuffle=False)
    args = argparse.Namespace(max_batches=1, epochs=3)
    assert estimate_total_steps(loader, args) == 3

    args_no_cap = argparse.Namespace(max_batches=None, epochs=2)
    assert estimate_total_steps(loader, args_no_cap) == len(loader) * 2


def test_train_smoke_end_to_end(fixture_csv, tmp_path):
    checkpoint_path = tmp_path / "smoke.pt"
    args = argparse.Namespace(
        train_csv=fixture_csv,
        checkpoint_path=str(checkpoint_path),
        device="cpu",
        batch_size=2,
        epochs=1,
        max_batches=2,
        num_workers=0,
        min_properties=1,
        hidden_dim=32,
        layers=2,
        attn_heads=2,
        dropout=0.0,
        ema_decay=0.9,
        lr=1e-3,
        weight_decay=0.0,
        log_every=1,
        regularizer="none",
        reg_weight=0.0,
        reg_std_coeff=1.0,
        reg_cov_coeff=1.0,
        scheduler="none",
        warmup_ratio=0.1,
        min_lr=1e-5,
    )
    returned_path = train(args)
    assert returned_path == checkpoint_path
    assert checkpoint_path.exists()
    stats_path = checkpoint_path.with_suffix(".property_stats.json")
    assert stats_path.exists()


def test_train_smoke_with_cosine_scheduler_and_regularizer(fixture_csv, tmp_path):
    checkpoint_path = tmp_path / "smoke_vc.pt"
    args = argparse.Namespace(
        train_csv=fixture_csv,
        checkpoint_path=str(checkpoint_path),
        device="cpu",
        batch_size=2,
        epochs=1,
        max_batches=2,
        num_workers=0,
        min_properties=1,
        hidden_dim=32,
        layers=2,
        attn_heads=2,
        dropout=0.0,
        ema_decay=0.9,
        lr=1e-3,
        weight_decay=0.0,
        log_every=1,
        regularizer="vc",
        reg_weight=0.1,
        reg_std_coeff=1.0,
        reg_cov_coeff=1.0,
        scheduler="cosine",
        warmup_ratio=0.1,
        min_lr=1e-5,
    )
    returned_path = train(args)
    assert returned_path == checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "scheduler_state_dict" in checkpoint


def test_cosine_with_warmup_importable_and_steps():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    scheduler = CosineWithWarmup(optimizer, total_steps=10, warmup_ratio=0.2, min_lr=1e-4)
    scheduler.step()
    assert scheduler.get_last_lr()[0] > 0
