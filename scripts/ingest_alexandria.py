"""Ingest the Alexandria materials database (PBEsol, 3D all) into
data/raw/alexandria_pbesol.csv.gz.

Alexandria (https://alexandria.icams.rub.de/) is used here as the
scale-up source: it contributes ~415k structures at a single, consistent
DFT level of theory (PBEsol), roughly 4.4x the size of JARVIS-DFT. This
script deliberately downloads the PBEsol subset rather than the ~5M-row
PBE "all" file: the PBEsol set is already a >4x scale-up over JARVIS with
full property coverage, and 5M CIF conversions would cost far more compute
and storage for diminishing returns on this project's near-term dataset
size. If more scale is needed later, alex_pbe_3d_all is available via the
same jarvis-tools figshare data() loader (dataset="alex_pbe_3d_all").

Extracted properties (all 100% populated in the PBEsol file):
- thermo: formation_energy_peratom (e_form), e_above_hull, e_phase_separation,
  energy_total
- electronic: band_gap_ind, band_gap_dir, dos_ef
- magnetic: total_mag

Usage:
    python scripts/ingest_alexandria.py \
        --output data/raw/alexandria_pbesol.csv.gz \
        --report docs/audit/alexandria_schema_report.json

Notes:
- Requires network access to ndownloader.figshare.com (cached locally by
  jarvis-tools after the first run).
- All rows share the PBEsol functional (a single, consistent level of
  theory) -- distinct from JARVIS's OptB88vdW/TBmBJ/HSE06 and from
  Materials Project's PBE/PBE+U/r2SCAN. The dft_functional column is kept
  on every row so downstream merges never silently mix theory levels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_alexandria_dataframe(dataset: str = "alex_pbesol_3d_all") -> tuple[pd.DataFrame, int]:
    """Download Alexandria and return (dataframe, n_cif_conversion_failures)."""
    from jarvis.core.atoms import Atoms as JarvisAtoms
    from jarvis.db.figshare import data as jarvis_data

    raw_entries = jarvis_data(dataset=dataset)

    functional = "PBEsol" if "pbesol" in dataset else ("PBE" if "pbe" in dataset else "SCAN")

    rows = []
    n_cif_fail = 0
    for entry in raw_entries:
        try:
            jats = JarvisAtoms.from_dict(entry["atoms"])
            pmg_struct = jats.pymatgen_converter()
            cif_str = pmg_struct.to(fmt="cif")
        except Exception:
            n_cif_fail += 1
            continue

        rows.append(
            {
                "alexandria_id": entry.get("mat_id"),
                "formula": entry.get("formula"),
                "spg_number": entry.get("spg"),
                "cif": cif_str,
                "nsites": entry.get("nsites"),
                "formation_energy_peratom": entry.get("e_form"),
                "e_above_hull": entry.get("e_above_hull"),
                "e_phase_separation": entry.get("e_phase_separation"),
                "band_gap_ind": entry.get("band_gap_ind"),
                "band_gap_dir": entry.get("band_gap_dir"),
                "total_mag": entry.get("total_mag"),
                "dos_ef": entry.get("dos_ef"),
                "energy_total": entry.get("energy_total"),
                "dft_functional": functional,
            }
        )

    return pd.DataFrame(rows), n_cif_fail


_COLUMN_DOMAIN_MAP = {
    "formation_energy_peratom": "thermo",
    "e_above_hull": "thermo",
    "e_phase_separation": "thermo",
    "energy_total": "thermo",
    "band_gap_ind": "electronic",
    "band_gap_dir": "electronic",
    "dos_ef": "electronic",
    "total_mag": "magnetic",
}

_NON_PROPERTY_COLUMNS = {"alexandria_id", "formula", "cif", "spg_number", "nsites", "dft_functional"}


def build_schema_report(df: pd.DataFrame, n_cif_fail: int, dataset: str) -> dict:
    coverage = {}
    for col in df.columns:
        if col in _NON_PROPERTY_COLUMNS:
            continue
        n_valid = int(df[col].notna().sum())
        coverage[col] = {
            "n_valid": n_valid,
            "pct_valid": round(100 * n_valid / len(df), 2) if len(df) else 0.0,
        }
    return {
        "source": f"Alexandria materials database ({dataset})",
        "reference": "https://alexandria.icams.rub.de/",
        "total_rows": len(df),
        "cif_conversion_failures": n_cif_fail,
        "dft_functional_note": (
            "All rows in this file share one DFT functional (see dft_functional "
            "column) -- distinct from JARVIS's OptB88vdW/TBmBJ/HSE06 and from "
            "Materials Project's PBE/PBE+U/r2SCAN. Keep the functional column "
            "when merging across sources."
        ),
        "property_coverage": coverage,
        "column_domain_map": _COLUMN_DOMAIN_MAP,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="alex_pbesol_3d_all",
        help="jarvis-tools dataset key (default: alex_pbesol_3d_all; see module docstring for alternatives)",
    )
    parser.add_argument("--output", default="data/raw/alexandria_pbesol.csv.gz")
    parser.add_argument("--report", default="docs/audit/alexandria_schema_report.json")
    args = parser.parse_args()

    df, n_cif_fail = build_alexandria_dataframe(dataset=args.dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, compression="gzip")

    report = build_schema_report(df, n_cif_fail, args.dataset)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {len(df)} rows to {output_path}")
    print(f"Wrote schema report to {report_path}")


if __name__ == "__main__":
    main()
