"""Deduplicate data/jepa/mp.csv.gz at the source (keep first occurrence per material_id).

The pretraining corpus originally contained 92,762 rows but only 80,769 unique
material_id values -- 11,993 material_id groups were duplicated (23,986 rows,
all with identical CIF text and ef_per_atom within each group, confirmed by
audit). This script removes the redundant rows in-place so every downstream
consumer (dataloaders, analysis, split-building) sees a corpus with exactly
one row per material_id, and the `deduplicate=True` flag on
SuperconductorDataset becomes a no-op safety net rather than a required step.

Usage:
    python scripts/dedup_mp_corpus.py --csv-path data/jepa/mp.csv.gz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def dedup_mp_corpus(csv_path: Path, report_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    n_before = len(df)
    n_unique = df["material_id"].nunique()

    # Sanity check: duplicate groups must be exact duplicates (identical cif +
    # ef_per_atom), never silently drop rows that differ.
    dup_mask = df.duplicated("material_id", keep=False)
    dup_groups = df[dup_mask].groupby("material_id")
    n_ambiguous = int((dup_groups["ef_per_atom"].nunique() > 1).sum())
    if n_ambiguous:
        raise ValueError(
            f"{n_ambiguous} material_id groups have differing ef_per_atom "
            "across duplicate rows -- refusing to silently drop; inspect manually."
        )

    deduped = df.drop_duplicates("material_id", keep="first").reset_index(drop=True)
    n_after = len(deduped)

    deduped.to_csv(csv_path, index=False, compression="gzip")

    report = {
        "rows_before": n_before,
        "rows_after": n_after,
        "unique_material_id": n_unique,
        "duplicate_rows_removed": n_before - n_after,
        "ambiguous_groups_found": n_ambiguous,
    }
    report_path.write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", default="data/jepa/mp.csv.gz")
    parser.add_argument("--report-path", default="data/processed/mp_dedup_report.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = dedup_mp_corpus(Path(args.csv_path), Path(args.report_path))
    print(json.dumps(result, indent=2))
