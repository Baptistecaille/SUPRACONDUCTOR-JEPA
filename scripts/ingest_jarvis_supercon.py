"""Ingest JARVIS's DFPT electron-phonon superconductor datasets into
data/raw/jarvis_supercon.csv.gz.

Downloads the `supercon_3d` (1058 materials) and `supercon_2d` (161
materials) figshare datasets via jarvis-tools. Unlike `dft_3d`
(`ingest_jarvis_dft.py`), these datasets carry actual DFPT electron-phonon
coupling quantities -- not just the McMillan/Allen-Dynes Tc that falls out
of them:

  * lamb   -- electron-phonon coupling constant, lambda (dimensionless)
  * wlog   -- logarithmic-averaged phonon frequency, omega_log (Kelvin,
              as stored by JARVIS -- this is what the Allen-Dynes formula
              actually consumes, NOT the Debye frequency)
  * Tc     -- Allen-Dynes Tc computed by JARVIS from (lamb, wlog, mu*)
  * a2F    -- Eliashberg spectral function alpha^2F(omega), resampled by
              JARVIS onto a fixed 100-point grid (0 to omega_max)
  * a2F_original_x / a2F_original_y -- the ORIGINAL (unresampled) DFPT
              alpha^2F(omega) spectrum, 50 points each. The frequency grid
              (a2F_original_x) is NOT fixed across materials -- it can
              include NEGATIVE frequencies, which signal dynamically
              unstable (imaginary) phonon modes at that q-point. Any
              downstream consumer must interpolate onto a common grid
              itself; this script stores the raw per-material arrays
              verbatim (JSON-encoded) rather than resampling them again.

This is the primary labelled dataset for property-JEPA's future
lambda/omega_log/Tc "physics module" heads -- see
docs/superpowers/plans (Tc module design discussion) for how these feed
a differentiable Allen-Dynes layer.

Usage:
    python scripts/ingest_jarvis_supercon.py \
        --output data/raw/jarvis_supercon.csv.gz \
        --report docs/audit/jarvis_supercon_schema_report.json

Notes:
- Requires network access to ndownloader.figshare.com (cached locally by
  jarvis-tools after the first run, under $ATOMGPTLAB_CACHE).
- Every supercon_3d/supercon_2d entry's `jid` is drawn from the SAME
  JARVIS-DFT id space as `dft_3d` (see ingest_jarvis_dft.py) -- these rows
  are a strict subset of jarvis_dft.csv.gz by jarvis_id, so downstream
  consolidation can join on jarvis_id rather than composition_key when
  exact-structure alignment matters.
- supercon_3d and supercon_2d have disjoint jid sets (no overlap observed);
  both are ingested into one file with a `dimensionality` column.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _clean_value(value):
    """Convert JARVIS's 'na' string placeholder (and empty strings) to None."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("na", ""):
        return None
    return value


def build_jarvis_supercon_dataframe() -> pd.DataFrame:
    """Download supercon_3d + supercon_2d and return one concatenated dataframe."""
    # Imported lazily so this module can be inspected/tested without the
    # (fairly heavy) jarvis-tools + pymatgen import chain if unused.
    from jarvis.core.atoms import Atoms as JarvisAtoms
    from jarvis.db.figshare import data as jarvis_data

    frames = []
    for dataset, dimensionality in [("supercon_3d", "3D"), ("supercon_2d", "2D")]:
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
                    "dimensionality": dimensionality,
                    "cif": cif_str,
                    "stability": _clean_value(entry.get("stability")),
                    "press_gpa": _clean_value(entry.get("press")),
                    "lambda_ep": _clean_value(entry.get("lamb")),
                    "omega_log": _clean_value(entry.get("wlog")),
                    "tc_allen_dynes": _clean_value(entry.get("Tc")),
                    "a2f_resampled": json.dumps(_clean_value(entry.get("a2F"))),
                    "a2f_original_x": json.dumps(_clean_value(entry.get("a2F_original_x"))),
                    "a2f_original_y": json.dumps(_clean_value(entry.get("a2F_original_y"))),
                }
            )
        frame = pd.DataFrame(rows)
        frame.attrs[f"n_cif_fail_{dataset}"] = n_cif_fail
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined.attrs["n_cif_fail_total"] = sum(
        f.attrs.get(f"n_cif_fail_{ds}", 0) for f, ds in zip(frames, ["supercon_3d", "supercon_2d"])
    )
    return combined


_PROPERTY_COLUMNS = ["lambda_ep", "omega_log", "tc_allen_dynes", "a2f_resampled", "a2f_original_x", "a2f_original_y"]
_NON_PROPERTY_COLUMNS = {"jarvis_id", "dimensionality", "cif", "stability", "press_gpa"}


def build_schema_report(df: pd.DataFrame) -> dict:
    coverage = {}
    for col in _PROPERTY_COLUMNS:
        n_valid = int(df[col].notna().sum() - (df[col] == "null").sum())
        coverage[col] = {
            "n_valid": n_valid,
            "pct_valid": round(100 * n_valid / len(df), 2) if len(df) else 0.0,
        }
    return {
        "source": "JARVIS supercon_3d + supercon_2d (DFPT electron-phonon coupling), NIST, via jarvis-tools",
        "reference": "https://www.nature.com/articles/s41524-022-00933-1 (3D); https://doi.org/10.1021/acs.nanolett.2c04420 (2D)",
        "total_rows": len(df),
        "rows_by_dimensionality": {k: int(v) for k, v in df["dimensionality"].value_counts().items()},
        "property_coverage": coverage,
        "a2f_original_x_note": (
            "The alpha^2F(omega) frequency grid (a2f_original_x) is NOT a fixed set of bins "
            "shared across materials -- it varies per material and can include negative "
            "frequencies (dynamically unstable / imaginary phonon modes). Consumers must "
            "interpolate onto a common grid themselves; values are stored as JSON-encoded "
            "lists, not fixed-width columns."
        ),
        "wlog_units_note": (
            "omega_log (wlog) is stored by JARVIS in Kelvin, not THz/cm^-1 -- this is the "
            "logarithmic-averaged phonon frequency the Allen-Dynes formula consumes directly, "
            "distinct from a Debye frequency."
        ),
        "overlap_with_dft_3d_note": (
            "Every jarvis_id here is drawn from the same id space as dft_3d "
            "(see ingest_jarvis_dft.py / data/raw/jarvis_dft.csv.gz) -- these rows are a "
            "strict subset by jarvis_id, so downstream joins can use jarvis_id directly "
            "rather than falling back to composition_key."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/raw/jarvis_supercon.csv.gz",
        help="Output CSV.gz path (default: data/raw/jarvis_supercon.csv.gz)",
    )
    parser.add_argument(
        "--report",
        default="docs/audit/jarvis_supercon_schema_report.json",
        help="Output schema/coverage report JSON path",
    )
    args = parser.parse_args()

    df = build_jarvis_supercon_dataframe()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, compression="gzip")

    report = build_schema_report(df)
    report["cif_conversion_failures"] = int(df.attrs.get("n_cif_fail_total", 0))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {len(df)} rows to {output_path}")
    print(f"Wrote schema report to {report_path}")


if __name__ == "__main__":
    main()
