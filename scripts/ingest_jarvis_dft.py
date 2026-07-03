"""Ingest the JARVIS-DFT (dft_3d) dataset into data/raw/jarvis_dft.csv.gz.

Downloads the ~94k-material JARVIS-DFT 3D dataset (NIST, via jarvis-tools),
converts every entry's atomic structure to a CIF string (through
pymatgen), and extracts the properties relevant to property-JEPA: thermo
(formation energy, e_above_hull), electronic (OptB88vdW / TBmBJ / HSE06 band
gaps), magnetic (OSZICAR / OUTCAR total magnetic moment), mechanical (bulk
modulus, shear modulus, Poisson ratio, elastic tensor max), and Tc_supercon.

Usage:
    python scripts/ingest_jarvis_dft.py \
        --output data/raw/jarvis_dft.csv.gz \
        --report docs/audit/jarvis_schema_report.json

Notes:
- Requires network access to ndownloader.figshare.com and its S3 redirect
  target for the initial download (cached locally afterwards by
  jarvis-tools under $ATOMGPTLAB_CACHE, default ~/.cache/atomgptlab).
- optb88vdw_bandgap, mbj_bandgap, and hse_gap are three DIFFERENT DFT
  functionals' band-gap estimates and are NOT interchangeable -- they are
  kept as separate columns rather than merged into one "bandgap" field.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _clean_value(value):
    """Convert JARVIS's 'na' string placeholder (and empty strings) to None."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("na", ""):
        return None
    return value


def _scalar_from_tensor(value, kind: str = "max"):
    """Reduce a nested list/tensor field (e.g. elastic_tensor) to one scalar summary."""
    value = _clean_value(value)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        try:
            arr = np.array(value, dtype=float)
            return float(arr.max()) if kind == "max" else float(arr.mean())
        except (TypeError, ValueError):
            return None
    return value


def build_jarvis_dataframe(dataset: str = "dft_3d") -> tuple[pd.DataFrame, int]:
    """Download JARVIS-DFT and return (dataframe, n_cif_conversion_failures)."""
    # Imported lazily so this module can be inspected/tested without the
    # (fairly heavy) jarvis-tools + pymatgen import chain if unused.
    from jarvis.core.atoms import Atoms as JarvisAtoms
    from jarvis.db.figshare import data as jarvis_data

    raw_entries = jarvis_data(dataset=dataset)

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
                "jarvis_id": entry.get("jid"),
                "formula": entry.get("formula"),
                "spg_number": _clean_value(entry.get("spg_number")),
                "cif": cif_str,
                "nat": _clean_value(entry.get("nat")),
                "formation_energy_peratom": _clean_value(entry.get("formation_energy_peratom")),
                "e_above_hull": _clean_value(entry.get("ehull")),
                "optb88vdw_bandgap": _clean_value(entry.get("optb88vdw_bandgap")),
                "mbj_bandgap": _clean_value(entry.get("mbj_bandgap")),
                "hse_gap": _clean_value(entry.get("hse_gap")),
                "magmom_oszicar": _clean_value(entry.get("magmom_oszicar")),
                "magmom_outcar": _clean_value(entry.get("magmom_outcar")),
                "bulk_modulus_kv": _clean_value(entry.get("bulk_modulus_kv")),
                "shear_modulus_gv": _clean_value(entry.get("shear_modulus_gv")),
                "poisson_ratio": _clean_value(entry.get("poisson")),
                "elastic_tensor_max": _scalar_from_tensor(entry.get("elastic_tensor"), kind="max"),
                "tc_supercon": _clean_value(entry.get("Tc_supercon")),
                "dft_functional": entry.get("func"),
                "mp_reference": entry.get("reference"),
            }
        )

    return pd.DataFrame(rows), n_cif_fail


_COLUMN_DOMAIN_MAP = {
    "formation_energy_peratom": "thermo",
    "e_above_hull": "thermo",
    "optb88vdw_bandgap": "electronic",
    "mbj_bandgap": "electronic",
    "hse_gap": "electronic",
    "magmom_oszicar": "magnetic",
    "magmom_outcar": "magnetic",
    "bulk_modulus_kv": "mechanical",
    "shear_modulus_gv": "mechanical",
    "poisson_ratio": "mechanical",
    "elastic_tensor_max": "mechanical",
    "tc_supercon": "superconducting",
}

_NON_PROPERTY_COLUMNS = {
    "jarvis_id", "formula", "cif", "spg_number", "nat", "dft_functional", "mp_reference",
}


def build_schema_report(df: pd.DataFrame, n_cif_fail: int) -> dict:
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
        "source": "JARVIS-DFT (dft_3d), NIST, via jarvis-tools",
        "reference": "https://doi.org/10.1016/j.commatsci.2025.114063",
        "total_rows": len(df),
        "cif_conversion_failures": n_cif_fail,
        "dft_functional_note": (
            "optb88vdw_bandgap, mbj_bandgap, and hse_gap are three different "
            "DFT functionals' band-gap estimates and are NOT interchangeable "
            "-- kept as separate columns rather than merged."
        ),
        "property_coverage": coverage,
        "column_domain_map": _COLUMN_DOMAIN_MAP,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dft_3d", help="jarvis-tools dataset key (default: dft_3d)")
    parser.add_argument(
        "--output",
        default="data/raw/jarvis_dft.csv.gz",
        help="Output CSV.gz path (default: data/raw/jarvis_dft.csv.gz)",
    )
    parser.add_argument(
        "--report",
        default="docs/audit/jarvis_schema_report.json",
        help="Output schema/coverage report JSON path",
    )
    args = parser.parse_args()

    df, n_cif_fail = build_jarvis_dataframe(dataset=args.dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, compression="gzip")

    report = build_schema_report(df, n_cif_fail)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {len(df)} rows to {output_path}")
    print(f"Wrote schema report to {report_path}")


if __name__ == "__main__":
    main()
