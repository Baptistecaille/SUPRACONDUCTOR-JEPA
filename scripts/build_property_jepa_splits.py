"""Build a composition-level, leak-free train/val/test split of the
consolidated multi-property corpus for property-JEPA (stage 2).

Method
------
1. Any composition present in the Materials Project pretraining corpus
   (data/jepa/mp.csv.gz, matched via the same composition_key used by
   scripts/consolidate_properties.py) is confined to a train-eligible-only
   pool -- it may appear in train but is EXCLUDED from val/test. This
   mirrors the composition-level leakage logic in scripts/build_tc_splits.py
   and prevents property-JEPA from being evaluated on structures the
   stage-1 crystal encoder (and any property model built on top of it) may
   already have seen during pretraining.
2. From the remaining leak-free pool, val and test (10% of the TOTAL
   corpus each) are drawn by proportional stratified sampling on
   (has_experimental_Tc x has_mechanical_props x cif_source), seed=42, so
   that rare labels (experimental Tc, mechanical properties) are
   represented in both eval splits roughly in proportion to their overall
   prevalence rather than concentrated entirely in train.

Usage:
    python scripts/build_property_jepa_splits.py \
        --input data/processed/consolidated_properties.csv.gz \
        --output-dir data/processed \
        --report docs/audit/property_jepa_split_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_splits(consolidated: pd.DataFrame, seed: int = 42, val_frac: float = 0.10, test_frac: float = 0.10):
    df = consolidated.copy()
    df["has_mechanical"] = df["mechanical_bulk_modulus_kv"].notna()
    df["has_tc_experimental_flag"] = df["tc_experimental"].notna()
    df["leaked_vs_pretrain"] = df["has_materials_project"]
    df["strat_key"] = (
        df["has_tc_experimental_flag"].astype(int).astype(str) + "_" +
        df["has_mechanical"].astype(int).astype(str) + "_" +
        df["cif_source"].astype(str)
    )

    leak_free_df = df.loc[~df["leaked_vs_pretrain"]]
    n_total = len(df)
    n_val_target = int(round(val_frac * n_total))
    n_test_target = int(round(test_frac * n_total))
    val_frac_of_leakfree = n_val_target / len(leak_free_df)
    test_frac_of_leakfree = n_test_target / len(leak_free_df)

    rng = np.random.default_rng(seed)
    val_ids, test_ids = [], []
    for _, grp in leak_free_df.groupby("strat_key"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_val = min(int(round(n * val_frac_of_leakfree)), n)
        n_test = min(int(round(n * test_frac_of_leakfree)), n - n_val)
        val_ids.extend(idx[:n_val])
        test_ids.extend(idx[n_val:n_val + n_test])

    val_ids, test_ids = set(val_ids), set(test_ids)
    df["split"] = "train"
    df.loc[list(val_ids), "split"] = "val"
    df.loc[list(test_ids), "split"] = "test"
    return df, val_ids, test_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/consolidated_properties.csv.gz")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--report", default="docs/audit/property_jepa_split_report.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    consolidated = pd.read_csv(args.input)
    assert consolidated["composition_key"].is_unique, "input must be one row per composition_key"

    df, val_ids, test_ids = build_splits(consolidated, seed=args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_cols = [c for c in df.columns if c not in ("strat_key", "leaked_vs_pretrain", "has_tc_experimental_flag", "has_mechanical")]
    for split in ["train", "val", "test"]:
        sub = df.loc[df["split"] == split, keep_cols]
        sub.to_csv(out_dir / f"property_jepa_{split}.csv.gz", index=False, compression="gzip")

    train_ck = set(df.loc[df["split"] == "train", "composition_key"])
    val_ck = set(df.loc[df["split"] == "val", "composition_key"])
    test_ck = set(df.loc[df["split"] == "test", "composition_key"])

    report = {
        "description": "Composition-level, leak-free train/val/test split of the consolidated multi-property corpus.",
        "seed": args.seed,
        "split_sizes": {k: int(v) for k, v in df["split"].value_counts().items()},
        "leak_check": {
            "val_rows_in_mp_pretrain_corpus": int(df.loc[df["split"] == "val", "leaked_vs_pretrain"].sum()),
            "test_rows_in_mp_pretrain_corpus": int(df.loc[df["split"] == "test", "leaked_vs_pretrain"].sum()),
            "train_val_test_composition_overlap": {
                "train_and_val": len(train_ck & val_ck),
                "train_and_test": len(train_ck & test_ck),
                "val_and_test": len(val_ck & test_ck),
            },
        },
        "per_split_property_coverage": {
            split: {
                "n_rows": int((df["split"] == split).sum()),
                "tc_experimental": int(df.loc[df["split"] == split, "tc_experimental"].notna().sum()),
                "mechanical_bulk_modulus_kv": int(df.loc[df["split"] == split, "mechanical_bulk_modulus_kv"].notna().sum()),
                "thermo_formation_energy_peratom": int(df.loc[df["split"] == split, "thermo_formation_energy_peratom"].notna().sum()),
                "electronic_bandgap_best": int(df.loc[df["split"] == split, "electronic_bandgap_best"].notna().sum()),
                "magnetic_total_moment_best": int(df.loc[df["split"] == split, "magnetic_total_moment_best"].notna().sum()),
            }
            for split in ["train", "val", "test"]
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["split_sizes"], indent=2))
    print(f"Wrote splits to {out_dir}/property_jepa_{{train,val,test}}.csv.gz")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
