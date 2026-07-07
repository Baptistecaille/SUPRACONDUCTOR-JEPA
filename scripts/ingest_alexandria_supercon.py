"""Ingest the Alexandria_SuperConDB dataset into
data/raw/alexandria_supercon.csv.gz.

Alexandria_SuperConDB (8253 materials, via jarvis-tools `alex_supercon`) is
a much larger DFPT electron-phonon-coupling labelled set than JARVIS's own
supercon_3d/supercon_2d (see ingest_jarvis_supercon.py), but with a
narrower per-material payload: it carries the McMillan-style scalar
quantities directly, not the full alpha^2F(omega) spectrum.

Extracted properties (100% populated in this file):
- dos_ef      -- electronic density of states at the Fermi level
- debye_freq  -- Debye frequency (Kelvin) -- NOTE this is a genuinely
                 different quantity from omega_log (the logarithmic-
                 averaged frequency the Allen-Dynes formula consumes); do
                 not conflate the two across sources.
- lambda_ep   -- electron-phonon coupling constant, lambda
- omega_log   -- logarithmic-averaged phonon frequency (Kelvin), same
                 definition/units as JARVIS's wlog
- tc_allen_dynes -- Allen-Dynes Tc computed by the source from
                 (lambda_ep, omega_log, an assumed mu*)

No alpha^2F(omega) spectrum is available here -- the DOS-phonon auxiliary
loss (see Tc-module design) can only be trained on the JARVIS
supercon_3d/supercon_2d subset (ingest_jarvis_supercon.py), not on this
one.

Usage:
    python scripts/ingest_alexandria_supercon.py \
        --output data/raw/alexandria_supercon.csv.gz \
        --report docs/audit/alexandria_supercon_schema_report.json

Notes:
- Requires network access to ndownloader.figshare.com (cached locally by
  jarvis-tools after the first run).
- `id` values (e.g. "agm003769282") share Alexandria's "agm..." id
  namespace, but only ~52% overlap with the ids already present in
  data/raw/alexandria_pbesol.csv.gz (alex_pbesol_3d_all) was observed --
  treat this as a separate source to merge by composition_key, not assume
  every row here already has a PBEsol counterpart ingested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_alexandria_supercon_dataframe() -> tuple[pd.DataFrame, int]:
    """Download Alexandria_SuperConDB and return (dataframe, n_cif_conversion_failures)."""
    from jarvis.core.atoms import Atoms as JarvisAtoms
    from jarvis.db.figshare import data as jarvis_data

    raw_entries = jarvis_data(dataset="alex_supercon")

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
                "alexandria_id": entry.get("id"),
                "cif": cif_str,
                "dos_ef": entry.get("dosef"),
                "debye_freq": entry.get("debye"),
                "lambda_ep": entry.get("la"),
                "omega_log": entry.get("wlog"),
                "tc_allen_dynes": entry.get("Tc"),
            }
        )

    return pd.DataFrame(rows), n_cif_fail


_PROPERTY_COLUMNS = ["dos_ef", "debye_freq", "lambda_ep", "omega_log", "tc_allen_dynes"]
_NON_PROPERTY_COLUMNS = {"alexandria_id", "cif"}


def build_schema_report(df: pd.DataFrame, n_cif_fail: int) -> dict:
    coverage = {}
    for col in _PROPERTY_COLUMNS:
        n_valid = int(df[col].notna().sum())
        coverage[col] = {
            "n_valid": n_valid,
            "pct_valid": round(100 * n_valid / len(df), 2) if len(df) else 0.0,
        }
    return {
        "source": "Alexandria_SuperConDB, via jarvis-tools (dataset=alex_supercon)",
        "reference": "https://doi.org/10.1002/adfm.202404043",
        "total_rows": len(df),
        "cif_conversion_failures": n_cif_fail,
        "property_coverage": coverage,
        "no_a2f_spectrum_note": (
            "This source has no alpha^2F(omega) spectrum -- only the scalar "
            "lambda_ep/omega_log/dos_ef/debye_freq/tc_allen_dynes quantities. "
            "The DOS-phonon auxiliary loss can only draw on "
            "data/raw/jarvis_supercon.csv.gz (JARVIS supercon_3d/supercon_2d), "
            "which does carry the full spectrum."
        ),
        "debye_vs_omega_log_note": (
            "debye_freq (Debye frequency) and omega_log (logarithmic-averaged "
            "frequency) are DIFFERENT quantities -- the Allen-Dynes formula "
            "consumes omega_log, not debye_freq. Both are kept as separate "
            "columns; do not substitute one for the other."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/alexandria_supercon.csv.gz")
    parser.add_argument("--report", default="docs/audit/alexandria_supercon_schema_report.json")
    args = parser.parse_args()

    df, n_cif_fail = build_alexandria_supercon_dataframe()

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
