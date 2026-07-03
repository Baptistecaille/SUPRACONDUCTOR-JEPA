"""Build a rigorous, composition-level leak-free train/val/test split of the
Tc-labeled subset, and a matching pretraining-set exclusion list.

Rationale (see docs/data_audit_report.md for full detail):
  * The previous "no_train_leak" file (401 rows) only excluded rows whose
    *mp_material_id* appeared verbatim in data/jepa/mp.csv.gz. It missed rows
    whose crystal has the SAME normalized composition as a pretraining
    structure under a DIFFERENT material_id (different polymorph/relaxation)
    -- 43/401 rows in that file in fact leak by composition.
  * This script instead starts from the full superband_mp_matched.csv
    (4,013 rows, before any leak filtering) and removes every row whose
    normalized composition (derived straight from each row's own CIF
    `_chemical_formula_sum`, not the raw SuperBand formula string) appears
    anywhere in the deduplicated pretraining corpus (data/jepa/mp.csv.gz,
    80,769 unique material_ids). This is strictly stronger than the old
    filter and yields far more usable rows (~2,850 vs 401), because most of
    the composition overlap was concentrated in the 3DSC_MP-sourced CIFs
    that ARE literally MP entries already in the corpus, which the old
    filter happened to catch by id -- but 43 rows leaked via a different id
    for the same composition, and those are now correctly removed too.
  * Within the leak-free pool, rows sharing an identical composition are
    deduplicated (keep first by superband_id) so no composition can appear
    twice -- a prerequisite for a composition-disjoint train/val/test split.
  * The remaining rows are split 70/15/15 by stratifying on log10(Tc)
    deciles, with a fixed random seed, guaranteeing (a) no composition is
    shared across folds and (b) each fold has a similar Tc distribution.

Usage:
    python scripts/build_tc_splits.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split

FORMULA_SUM_RE = re.compile(r"_chemical_formula_sum\s+'?([^\n']+)'?")


def extract_formula_sum(cif: str) -> str | None:
    m = FORMULA_SUM_RE.search(cif)
    return m.group(1).strip() if m else None


def composition_key(formula: object) -> str | None:
    if pd.isna(formula):
        return None
    try:
        comp = Composition(str(formula)).fractional_composition
    except Exception:
        return None
    parts = [f"{el}:{amt:.6g}" for el, amt in sorted(comp.get_el_amt_dict().items())]
    return "|".join(parts)


def build_splits(
    matched_path: Path,
    pretrain_path: Path,
    out_dir: Path,
    report_path: Path,
    exclusion_list_path: Path,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict:
    matched = pd.read_csv(matched_path)
    pretrain = pd.read_csv(pretrain_path)

    matched["cif_formula_sum"] = matched["cif"].map(extract_formula_sum)
    matched["composition_key"] = matched["cif_formula_sum"].map(composition_key)

    pretrain["formula_sum"] = pretrain["cif"].map(extract_formula_sum)
    pretrain["composition_key"] = pretrain["formula_sum"].map(composition_key)
    pretrain_ckeys = set(pretrain["composition_key"].dropna())

    n_total = len(matched)
    matched["leaks_composition_vs_pretrain"] = matched["composition_key"].isin(
        pretrain_ckeys
    )
    n_leak = int(matched["leaks_composition_vs_pretrain"].sum())

    clean = matched[~matched["leaks_composition_vs_pretrain"]].copy()
    clean = clean.dropna(subset=["composition_key", "tc"])

    n_clean_with_dups = len(clean)
    clean = clean.sort_values("superband_id").drop_duplicates(
        "composition_key", keep="first"
    )
    n_clean_dedup = len(clean)

    # Stratify by log10(Tc) decile so each fold sees a comparable Tc range.
    clean["_log_tc"] = np.log10(clean["tc"].clip(lower=1e-3))
    clean["_tc_bin"] = pd.qcut(clean["_log_tc"], q=10, labels=False, duplicates="drop")

    train_val, test = train_test_split(
        clean,
        test_size=val_frac,
        stratify=clean["_tc_bin"],
        random_state=seed,
    )
    relative_val_frac = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_frac,
        stratify=train_val["_tc_bin"],
        random_state=seed,
    )

    keep_cols = [
        "material_id",
        "cif",
        "ef_per_atom",
        "tc",
        "superband_id",
        "superband_formula",
        "superband_cif_id",
        "mp_material_id",
        "cif_source",
        "composition_key",
    ]
    train_out = train[keep_cols].reset_index(drop=True)
    val_out = val[keep_cols].reset_index(drop=True)
    test_out = test[keep_cols].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(out_dir / "tc_train.csv", index=False)
    val_out.to_csv(out_dir / "tc_val.csv", index=False)
    test_out.to_csv(out_dir / "tc_test.csv", index=False)

    # Composition-disjointness sanity check.
    train_ck = set(train_out["composition_key"])
    val_ck = set(val_out["composition_key"])
    test_ck = set(test_out["composition_key"])
    assert not (train_ck & val_ck), "train/val composition overlap!"
    assert not (train_ck & test_ck), "train/test composition overlap!"
    assert not (val_ck & test_ck), "val/test composition overlap!"

    # Pretraining-set exclusion list: any pretraining material_id whose
    # composition matches a composition now used in the Tc-labeled splits,
    # so a future pretraining run can optionally hold these out.
    tc_ckeys = train_ck | val_ck | test_ck
    exclusion = pretrain[pretrain["composition_key"].isin(tc_ckeys)][
        ["material_id", "composition_key"]
    ].reset_index(drop=True)
    exclusion.to_csv(exclusion_list_path, index=False)

    report = {
        "input_rows_superband_mp_matched": n_total,
        "rows_removed_composition_leak_vs_pretrain": n_leak,
        "rows_after_leak_filter": n_clean_with_dups,
        "duplicate_composition_rows_removed": n_clean_with_dups - n_clean_dedup,
        "unique_compositions_final_pool": n_clean_dedup,
        "split_sizes": {
            "train": len(train_out),
            "val": len(val_out),
            "test": len(test_out),
        },
        "split_tc_stats": {
            "train": train_out["tc"].describe().to_dict(),
            "val": val_out["tc"].describe().to_dict(),
            "test": test_out["tc"].describe().to_dict(),
        },
        "composition_disjoint_verified": True,
        "pretraining_exclusion_list_rows": len(exclusion),
        "random_seed": seed,
        "stratify_variable": "log10(tc) decile",
        "note_supersedes": (
            "Supersedes data/processed/superband_mp_matched_no_train_leak.csv "
            "(401 rows, id-only leak filter) and "
            "superband_mp_matched_no_3dsc_formula.csv (128 rows, exact-formula-"
            "string filter, self-reported incomplete)."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matched-path", default="data/processed/superband_mp_matched.csv"
    )
    parser.add_argument("--pretrain-path", default="data/jepa/mp.csv.gz")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument(
        "--report-path", default="data/processed/tc_split_report.json"
    )
    parser.add_argument(
        "--exclusion-list-path",
        default="data/processed/pretrain_exclusion_list.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build_splits(
        Path(args.matched_path),
        Path(args.pretrain_path),
        Path(args.out_dir),
        Path(args.report_path),
        Path(args.exclusion_list_path),
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, default=str))
