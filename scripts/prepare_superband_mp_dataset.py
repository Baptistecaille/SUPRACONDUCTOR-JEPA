from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pymatgen.core import Composition


MODEL_COLUMNS = ["material_id", "cif", "ef_per_atom", "tc"]


def composition_key(formula: object) -> str | None:
    """Return a stable composition key for formula matching."""
    if pd.isna(formula):
        return None
    text = str(formula).strip()
    if not text:
        return None
    try:
        comp = Composition(text).fractional_composition
    except Exception:
        return None
    parts = []
    for element, amount in sorted(comp.get_el_amt_dict().items()):
        parts.append(f"{element}:{float(amount):.8g}")
    return "|".join(parts)


def cif_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    try:
        number = float(value)
    except Exception:
        return str(value).strip()
    if number.is_integer():
        return str(int(number))
    return str(value).strip()


def read_cif_value(base_csv_path: Path, value: object) -> str | None:
    """Read a CIF string from either inline CIF text or a path from a CSV value."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.startswith("data_") or "\n" in text:
        return text

    candidates = [
        Path(text),
        base_csv_path.parent / Path(text),
        base_csv_path.parent / "cifs" / Path(text).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text()
    return None


def read_local_superband_cif(cif_root: Path, value: object) -> str | None:
    cid = cif_id(value)
    if cid is None:
        return None
    for suffix in (".cif", ".CIF"):
        candidate = cif_root / f"{cid}{suffix}"
        if candidate.exists():
            return candidate.read_text()
    return None


def build_reference_table(raw_3dsc_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(raw_3dsc_path, comment="#")
    required = {
        "formula_sc",
        "material_id_2",
        "cif",
        "formation_energy_per_atom_2",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{raw_3dsc_path} is missing columns: {', '.join(missing)}")

    ref = raw.copy()
    ref["_composition_key"] = ref["formula_sc"].map(composition_key)
    ref["_source_cif_text"] = ref["cif"].map(
        lambda value: read_cif_value(raw_3dsc_path, value)
    )
    ref = ref.dropna(
        subset=[
            "_composition_key",
            "material_id_2",
            "_source_cif_text",
            "formation_energy_per_atom_2",
        ]
    ).copy()

    if "e_above_hull_2" in ref.columns:
        ref["_sort_e_above_hull"] = pd.to_numeric(ref["e_above_hull_2"], errors="coerce")
    else:
        ref["_sort_e_above_hull"] = pd.NA
    ref["_sort_ef"] = pd.to_numeric(
        ref["formation_energy_per_atom_2"], errors="coerce"
    )
    ref = ref.sort_values(
        by=["_composition_key", "_sort_e_above_hull", "_sort_ef", "material_id_2"],
        na_position="last",
    )

    key_counts = ref.groupby("_composition_key").size()
    report = {
        "reference_rows": int(len(ref)),
        "reference_unique_compositions": int(ref["_composition_key"].nunique()),
        "reference_ambiguous_compositions": int((key_counts > 1).sum()),
    }

    ref = ref.drop_duplicates("_composition_key", keep="first")
    return ref, report


def prepare_superband_mp_dataset(
    superband_path: Path,
    raw_3dsc_path: Path,
    cif_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    superband = pd.read_csv(superband_path)
    required = {"id", "formula", "Tc", "cif"}
    missing = sorted(required.difference(superband.columns))
    if missing:
        raise ValueError(f"{superband_path} is missing columns: {', '.join(missing)}")

    ref, ref_report = build_reference_table(raw_3dsc_path)

    sb = superband.copy()
    sb["_composition_key"] = sb["formula"].map(composition_key)
    sb["_local_cif_text"] = sb["cif"].map(
        lambda value: read_local_superband_cif(cif_root, value)
    )

    merged = sb.merge(
        ref[
            [
                "_composition_key",
                "material_id_2",
                "_source_cif_text",
                "formation_energy_per_atom_2",
            ]
        ],
        on="_composition_key",
        how="inner",
        validate="many_to_one",
    )
    merged = merged.dropna(subset=["Tc", "formation_energy_per_atom_2"]).copy()
    merged["_chosen_cif"] = merged["_local_cif_text"].where(
        merged["_local_cif_text"].notna(), merged["_source_cif_text"]
    )
    merged = merged.dropna(subset=["_chosen_cif"]).copy()

    out = pd.DataFrame(
        {
            "material_id": "superband-" + merged["id"].astype(str),
            "cif": merged["_chosen_cif"],
            "ef_per_atom": pd.to_numeric(
                merged["formation_energy_per_atom_2"], errors="coerce"
            ),
            "tc": pd.to_numeric(merged["Tc"], errors="coerce"),
            "superband_id": merged["id"],
            "superband_formula": merged["formula"],
            "superband_cif_id": merged["cif"].map(cif_id),
            "mp_material_id": merged["material_id_2"],
            "cif_source": merged["_local_cif_text"].notna().map(
                {True: "superband_local", False: "3dsc_mp"}
            ),
        }
    )
    out = out.dropna(subset=MODEL_COLUMNS).reset_index(drop=True)

    report = {
        **ref_report,
        "superband_rows": int(len(superband)),
        "superband_rows_with_composition_key": int(sb["_composition_key"].notna().sum()),
        "superband_rows_with_local_cif": int(sb["_local_cif_text"].notna().sum()),
        "matched_rows_before_tc_filter": int(len(merged)),
        "output_rows": int(len(out)),
        "output_unique_superband_ids": int(out["superband_id"].nunique()),
        "output_unique_mp_material_ids": int(out["mp_material_id"].nunique()),
        "output_cif_sources": out["cif_source"].value_counts().to_dict(),
    }
    return out, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match SuperBand rows to 3DSC/Materials Project data."
    )
    parser.add_argument("--superband-path", default="data/raw/superband.csv")
    parser.add_argument("--raw-3dsc-path", default="data/raw/3DSC_MP.csv")
    parser.add_argument("--cif-root", default="data/SuperBand/cif")
    parser.add_argument(
        "--output-path",
        default="data/processed/superband_mp_matched.csv",
    )
    parser.add_argument(
        "--report-path",
        default="data/processed/superband_mp_matched_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    report_path = Path(args.report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    matched, report = prepare_superband_mp_dataset(
        superband_path=Path(args.superband_path),
        raw_3dsc_path=Path(args.raw_3dsc_path),
        cif_root=Path(args.cif_root),
    )
    matched.to_csv(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(matched)} rows to {output_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
