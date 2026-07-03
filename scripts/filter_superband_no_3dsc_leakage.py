from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pymatgen.core import Composition


def composition_key(formula: object) -> str | None:
    if pd.isna(formula):
        return None
    try:
        comp = Composition(str(formula).strip()).fractional_composition
    except Exception:
        return None
    return "|".join(
        f"{element}:{float(amount):.8g}"
        for element, amount in sorted(comp.get_el_amt_dict().items())
    )


def add_leakage_flags(matched: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    train_formulas = set(train["formula_sc"].dropna().astype(str))
    train_mp_ids = set(train["material_id_2"].dropna().astype(str))
    train_compositions = set(train["formula_sc"].map(composition_key).dropna())

    flagged = matched.copy()
    flagged["_composition_key"] = flagged["superband_formula"].map(composition_key)
    flagged["leaks_exact_3dsc_formula"] = flagged["superband_formula"].astype(
        str
    ).isin(train_formulas)
    flagged["leaks_3dsc_mp_material_id"] = flagged["mp_material_id"].astype(str).isin(
        train_mp_ids
    )
    flagged["leaks_3dsc_composition"] = flagged["_composition_key"].isin(
        train_compositions
    )
    return flagged


def filter_dataset(
    matched_path: Path,
    train_path: Path,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    matched = pd.read_csv(matched_path)
    train = pd.read_csv(train_path, comment="#")
    flagged = add_leakage_flags(matched, train)

    if mode == "exact_formula":
        keep_mask = ~flagged["leaks_exact_3dsc_formula"]
        description = (
            "Removes rows whose SuperBand formula exactly appears in 3DSC_MP. "
            "This is useful for a quick split, but it is not a strict no-leak split."
        )
    elif mode == "mp_material_id":
        keep_mask = ~flagged["leaks_3dsc_mp_material_id"]
        description = "Removes rows whose matched MP material id appears in 3DSC_MP."
    elif mode == "composition":
        keep_mask = ~flagged["leaks_3dsc_composition"]
        description = "Removes rows whose normalized composition appears in 3DSC_MP."
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    filtered = flagged.loc[keep_mask].drop(columns=["_composition_key"]).reset_index(
        drop=True
    )
    report = {
        "mode": mode,
        "description": description,
        "input_rows": int(len(flagged)),
        "output_rows": int(len(filtered)),
        "removed_rows": int(len(flagged) - len(filtered)),
        "leak_counts_before_filter": {
            "exact_3dsc_formula": int(flagged["leaks_exact_3dsc_formula"].sum()),
            "mp_material_id": int(flagged["leaks_3dsc_mp_material_id"].sum()),
            "composition": int(flagged["leaks_3dsc_composition"].sum()),
        },
        "remaining_leak_counts_after_filter": {
            "exact_3dsc_formula": int(filtered["leaks_exact_3dsc_formula"].sum()),
            "mp_material_id": int(filtered["leaks_3dsc_mp_material_id"].sum()),
            "composition": int(filtered["leaks_3dsc_composition"].sum()),
        },
    }
    return filtered, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter matched SuperBand/MP rows against 3DSC_MP training data."
    )
    parser.add_argument(
        "--matched-path",
        default="data/processed/superband_mp_matched.csv",
    )
    parser.add_argument("--train-path", default="data/raw/3DSC_MP.csv")
    parser.add_argument(
        "--mode",
        choices=["exact_formula", "mp_material_id", "composition"],
        default="exact_formula",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/superband_mp_matched_no_3dsc_formula.csv",
    )
    parser.add_argument(
        "--report-path",
        default="data/processed/superband_mp_matched_no_3dsc_formula_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    report_path = Path(args.report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    filtered, report = filter_dataset(
        matched_path=Path(args.matched_path),
        train_path=Path(args.train_path),
        mode=args.mode,
    )
    filtered.to_csv(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(filtered)} rows to {output_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
