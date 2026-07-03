"""Consolidate 4 raw data sources into one composition-level property table.

Sources merged (see docs/audit/*_schema_report.json for each source's own
ingestion audit):
  * data/raw/superband.csv + data/processed/superband_mp_matched.csv
    (experimental Tc, via SuperBand + 3DSC_MP composition matching)
  * data/jepa/mp.csv.gz (Materials Project pretraining corpus, formation
    energy only)
  * data/raw/jarvis_dft.csv.gz (JARVIS-DFT: thermo/electronic/magnetic/
    mechanical/Tc_supercon, OptB88vdW + TBmBJ + HSE06)
  * data/raw/alexandria_pbesol.csv.gz (Alexandria PBEsol: thermo/electronic/
    magnetic, scale-up source)

Deduplication strategy
-----------------------
A material is identified by its COMPOSITION, not by any source-specific ID:
each CIF's `_chemical_formula_sum` is parsed with pymatgen and normalized to
a fractional composition (`Composition.fractional_composition`), then
rendered as a sorted "el:frac" string. This is robust to different sources
using different formula-string conventions for the same composition, and
deliberately coarser than polymorph/relaxation identity -- see
docs/audit/consolidation_report.json ("dedup_method") for the full
rationale, which mirrors the composition-level leakage logic already used
in scripts/build_tc_splits.py.

Within each source, rows sharing a composition_key are collapsed to one
representative row (lowest e_above_hull / formation energy as the
stability tie-break for DFT sources; lowest superband_id for the
experimental Tc source, since Tc there is a measurement, not a relaxation
artifact to minimize).

A canonical CIF per composition is chosen by source priority: superband
(experimental) > materials_project > jarvis_dft > alexandria_pbesol.

Property domains covered: thermo, electronic, magnetic, mechanical,
superconducting (Tc, both experimental and DFT-predicted). Per-source
columns are always retained alongside "best available" convenience
columns, because the sources span at least 4 different levels of DFT
theory (OptB88vdW/TBmBJ/HSE06, PBEsol, PBE/PBE+U, and experiment) that
should never be silently mixed in a single training target -- see the
"dft_heterogeneity_warning" in the JSON report.

Usage:
    python scripts/consolidate_properties.py \
        --output data/processed/consolidated_properties.csv.gz \
        --report docs/audit/consolidation_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

_FORMULA_SUM_RE = re.compile(r"_chemical_formula_sum\s+'?([^\n']+)'?")


def extract_formula_sum(cif) -> str | None:
    if not isinstance(cif, str):
        return None
    m = _FORMULA_SUM_RE.search(cif)
    return m.group(1).strip() if m else None


def composition_key(formula) -> str | None:
    if formula is None or (isinstance(formula, float) and pd.isna(formula)):
        return None
    try:
        comp = Composition(str(formula)).fractional_composition
    except Exception:
        return None
    parts = [f"{el}:{amt:.6g}" for el, amt in sorted(comp.get_el_amt_dict().items())]
    return "|".join(parts)


def collapse_to_composition(df: pd.DataFrame, ck_col: str, stability_col: str | None, id_col: str) -> pd.DataFrame:
    """Keep one row per composition_key: lowest stability_col, else first by id_col."""
    d = df.dropna(subset=[ck_col]).copy()
    if stability_col is not None and stability_col in d.columns:
        d = d.sort_values([ck_col, stability_col], ascending=[True, True], na_position="last")
    else:
        d = d.sort_values([ck_col, id_col])
    return d.drop_duplicates(ck_col, keep="first").reset_index(drop=True)


def build_consolidated_table(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    mp_df = pd.read_csv(data_dir / "jepa" / "mp.csv.gz")
    jarvis_df = pd.read_csv(data_dir / "raw" / "jarvis_dft.csv.gz")
    alex_df = pd.read_csv(data_dir / "raw" / "alexandria_pbesol.csv.gz")
    sb_matched = pd.read_csv(data_dir / "processed" / "superband_mp_matched.csv")

    mp_df["composition_key"] = mp_df["cif"].map(extract_formula_sum).map(composition_key)
    jarvis_df["composition_key"] = jarvis_df["formula"].map(composition_key)
    alex_df["composition_key"] = alex_df["formula"].map(composition_key)
    sb_matched["composition_key"] = sb_matched["cif"].map(extract_formula_sum).map(composition_key)

    jarvis_tc = jarvis_df[jarvis_df["tc_supercon"].notna()]

    jarvis_comp = collapse_to_composition(jarvis_df, "composition_key", "e_above_hull", "jarvis_id")
    alex_comp = collapse_to_composition(alex_df, "composition_key", "e_above_hull", "alexandria_id")
    mp_comp = collapse_to_composition(mp_df, "composition_key", "ef_per_atom", "material_id")
    sb_comp = collapse_to_composition(sb_matched, "composition_key", None, "superband_id")
    jarvis_tc_comp = collapse_to_composition(jarvis_tc, "composition_key", "e_above_hull", "jarvis_id")

    jarvis_ck = set(jarvis_df["composition_key"].dropna())
    alex_ck = set(alex_df["composition_key"].dropna())
    mp_ck = set(mp_df["composition_key"].dropna())
    sb_ck = set(sb_matched["composition_key"].dropna())
    jarvis_tc_ck = set(jarvis_tc["composition_key"].dropna())

    all_ck = jarvis_ck | alex_ck | mp_ck | sb_ck
    base = pd.DataFrame({"composition_key": sorted(all_ck)})

    sb_sel = sb_comp[["composition_key", "cif", "superband_id", "tc", "mp_material_id"]].rename(
        columns={"cif": "cif_superband", "tc": "tc_experimental_superband", "mp_material_id": "mp_material_id_superband"}
    )
    mp_sel = mp_comp[["composition_key", "cif", "material_id", "ef_per_atom"]].rename(
        columns={"cif": "cif_mp", "material_id": "mp_material_id", "ef_per_atom": "thermo_formation_energy_peratom_mp"}
    )
    jarvis_sel = jarvis_comp[[
        "composition_key", "cif", "jarvis_id",
        "formation_energy_peratom", "e_above_hull",
        "optb88vdw_bandgap", "mbj_bandgap", "hse_gap",
        "magmom_oszicar", "magmom_outcar",
        "bulk_modulus_kv", "shear_modulus_gv", "poisson_ratio", "elastic_tensor_max",
        "tc_supercon",
    ]].rename(columns={
        "cif": "cif_jarvis",
        "formation_energy_peratom": "thermo_formation_energy_peratom_jarvis",
        "e_above_hull": "thermo_e_above_hull_jarvis",
        "optb88vdw_bandgap": "electronic_bandgap_optb88vdw_jarvis",
        "mbj_bandgap": "electronic_bandgap_mbj_jarvis",
        "hse_gap": "electronic_bandgap_hse_jarvis",
        "magmom_oszicar": "magnetic_magmom_oszicar_jarvis",
        "magmom_outcar": "magnetic_magmom_outcar_jarvis",
        "bulk_modulus_kv": "mechanical_bulk_modulus_kv_jarvis",
        "shear_modulus_gv": "mechanical_shear_modulus_gv_jarvis",
        "poisson_ratio": "mechanical_poisson_ratio_jarvis",
        "elastic_tensor_max": "mechanical_elastic_tensor_max_jarvis",
        "tc_supercon": "tc_dft_jarvis",
    })
    alex_sel = alex_comp[[
        "composition_key", "cif", "alexandria_id",
        "formation_energy_peratom", "e_above_hull", "e_phase_separation",
        "band_gap_ind", "band_gap_dir", "dos_ef", "total_mag",
    ]].rename(columns={
        "cif": "cif_alexandria",
        "formation_energy_peratom": "thermo_formation_energy_peratom_alexandria",
        "e_above_hull": "thermo_e_above_hull_alexandria",
        "e_phase_separation": "thermo_e_phase_separation_alexandria",
        "band_gap_ind": "electronic_bandgap_ind_alexandria",
        "band_gap_dir": "electronic_bandgap_dir_alexandria",
        "dos_ef": "electronic_dos_ef_alexandria",
        "total_mag": "magnetic_total_mag_alexandria",
    })

    merged = (
        base.merge(sb_sel, on="composition_key", how="left")
        .merge(mp_sel, on="composition_key", how="left")
        .merge(jarvis_sel, on="composition_key", how="left")
        .merge(alex_sel, on="composition_key", how="left")
    )

    def pick_canonical(row):
        for col, src in [
            ("cif_superband", "superband"),
            ("cif_mp", "materials_project"),
            ("cif_jarvis", "jarvis_dft"),
            ("cif_alexandria", "alexandria_pbesol"),
        ]:
            v = row[col]
            if isinstance(v, str) and len(v) > 0:
                return pd.Series([v, src])
        return pd.Series([None, None])

    canon = merged.apply(pick_canonical, axis=1)
    canon.columns = ["cif", "cif_source"]
    merged["cif"] = canon["cif"]
    merged["cif_source"] = canon["cif_source"]

    merged["thermo_formation_energy_peratom"] = (
        merged["thermo_formation_energy_peratom_jarvis"]
        .combine_first(merged["thermo_formation_energy_peratom_alexandria"])
        .combine_first(merged["thermo_formation_energy_peratom_mp"])
    )
    merged["thermo_e_above_hull"] = merged["thermo_e_above_hull_jarvis"].combine_first(
        merged["thermo_e_above_hull_alexandria"]
    )
    merged["thermo_functional"] = np.select(
        [
            merged["thermo_formation_energy_peratom_jarvis"].notna(),
            merged["thermo_formation_energy_peratom_alexandria"].notna(),
            merged["thermo_formation_energy_peratom_mp"].notna(),
        ],
        ["OptB88vdW(JARVIS)", "PBEsol(Alexandria)", "PBE(MaterialsProject,unknown_hull)"],
        default=None,
    )
    merged["electronic_bandgap_best"] = merged["electronic_bandgap_optb88vdw_jarvis"].combine_first(
        merged["electronic_bandgap_ind_alexandria"]
    )
    merged["electronic_bandgap_best_functional"] = np.select(
        [merged["electronic_bandgap_optb88vdw_jarvis"].notna(), merged["electronic_bandgap_ind_alexandria"].notna()],
        ["OptB88vdW(JARVIS)", "PBEsol_indirect(Alexandria)"],
        default=None,
    )
    merged["magnetic_total_moment_best"] = merged["magnetic_magmom_outcar_jarvis"].combine_first(
        merged["magnetic_total_mag_alexandria"]
    )
    merged["mechanical_bulk_modulus_kv"] = merged["mechanical_bulk_modulus_kv_jarvis"]
    merged["mechanical_shear_modulus_gv"] = merged["mechanical_shear_modulus_gv_jarvis"]
    merged["mechanical_poisson_ratio"] = merged["mechanical_poisson_ratio_jarvis"]
    merged["tc_experimental"] = merged["tc_experimental_superband"]
    merged["tc_dft_predicted"] = merged["tc_dft_jarvis"]

    for src, col in [
        ("superband", "cif_superband"),
        ("materials_project", "cif_mp"),
        ("jarvis_dft", "cif_jarvis"),
        ("alexandria_pbesol", "cif_alexandria"),
    ]:
        merged[f"has_{src}"] = merged[col].notna()
    n_sources = merged[
        ["has_superband", "has_materials_project", "has_jarvis_dft", "has_alexandria_pbesol"]
    ].sum(axis=1)
    merged["n_sources"] = n_sources

    final_cols = [
        "composition_key", "cif", "cif_source", "n_sources",
        "has_superband", "has_materials_project", "has_jarvis_dft", "has_alexandria_pbesol",
        "superband_id", "mp_material_id", "jarvis_id", "alexandria_id",
        "thermo_formation_energy_peratom", "thermo_e_above_hull", "thermo_functional",
        "thermo_formation_energy_peratom_jarvis", "thermo_e_above_hull_jarvis",
        "thermo_formation_energy_peratom_alexandria", "thermo_e_above_hull_alexandria",
        "thermo_e_phase_separation_alexandria", "thermo_formation_energy_peratom_mp",
        "electronic_bandgap_best", "electronic_bandgap_best_functional",
        "electronic_bandgap_optb88vdw_jarvis", "electronic_bandgap_mbj_jarvis", "electronic_bandgap_hse_jarvis",
        "electronic_bandgap_ind_alexandria", "electronic_bandgap_dir_alexandria", "electronic_dos_ef_alexandria",
        "magnetic_total_moment_best",
        "magnetic_magmom_oszicar_jarvis", "magnetic_magmom_outcar_jarvis", "magnetic_total_mag_alexandria",
        "mechanical_bulk_modulus_kv", "mechanical_shear_modulus_gv", "mechanical_poisson_ratio",
        "mechanical_elastic_tensor_max_jarvis",
        "tc_experimental", "tc_dft_predicted",
    ]
    consolidated = merged[final_cols].copy()

    report = {
        "description": "Composition-level consolidation of 4 raw sources into one wide property table.",
        "sources": {
            "superband": {"raw_rows": int(len(sb_matched)), "unique_compositions_used": int(len(sb_comp)),
                          "role": "experimental Tc (highest priority CIF + Tc label)"},
            "materials_project": {"raw_rows": int(len(mp_df)), "unique_compositions_used": int(len(mp_comp)),
                                   "role": "pretraining corpus CIFs / formation energy fallback"},
            "jarvis_dft": {"raw_rows": int(len(jarvis_df)), "unique_compositions_used": int(len(jarvis_comp)),
                           "role": "electronic/magnetic/mechanical/Tc_DFT (OptB88vdW + TBmBJ + HSE06)"},
            "alexandria_pbesol": {"raw_rows": int(len(alex_df)), "unique_compositions_used": int(len(alex_comp)),
                                   "role": "thermo/electronic/magnetic scale-up (PBEsol)"},
        },
        "dedup_method": (
            "Composition key = sorted, fractional-amount-normalized element:fraction string "
            "derived from each CIF's own _chemical_formula_sum (pymatgen Composition."
            "fractional_composition), NOT the raw source formula string. Within each source, "
            "rows sharing a composition_key are collapsed to one representative row (lowest "
            "e_above_hull / ef_per_atom as the stability tie-break; superband/3DSC kept by "
            "lowest superband_id since Tc is an experimental measurement, not DFT-derived)."
        ),
        "canonical_cif_priority": ["superband (experimental)", "materials_project", "jarvis_dft", "alexandria_pbesol"],
        "total_unique_compositions": int(len(consolidated)),
        "n_sources_histogram": {str(k): int(v) for k, v in n_sources.value_counts().sort_index().items()},
        "cif_source_histogram": {k: int(v) for k, v in merged["cif_source"].value_counts().items()},
        "cross_source_composition_overlap": {
            "jarvis_and_alexandria": len(jarvis_ck & alex_ck),
            "jarvis_and_mp": len(jarvis_ck & mp_ck),
            "alexandria_and_mp": len(alex_ck & mp_ck),
            "all_three_dft_sources": len(jarvis_ck & alex_ck & mp_ck),
            "jarvis_tc_and_superband": len(jarvis_tc_ck & sb_ck),
        },
        "dft_heterogeneity_warning": (
            "This table mixes AT LEAST 4 distinct levels of theory: OptB88vdW/TBmBJ/HSE06 "
            "(JARVIS-DFT), PBEsol (Alexandria), PBE/PBE+U (Materials Project, functional not "
            "tracked per-row in the original mp.csv.gz export), and experimental measurement "
            "(SuperBand Tc). The 'best available' columns pick ONE value per row by "
            "source-priority fallback for convenience, but per-source columns are always kept "
            "so downstream training/eval code can filter to a single consistent functional "
            "when precision matters."
        ),
        "known_limitations": [
            "mechanical_* coverage is only ~6-7% -- exclusively from JARVIS-DFT.",
            "tc_experimental coverage is ~1.3% vs. 95%+ for thermo/electronic -- severe "
            "volumetric imbalance across property domains.",
            "tc_dft_predicted (JARVIS Tc_supercon) overlaps the experimental Tc set in only "
            "a small fraction of compositions -- a weak, mostly-independent signal.",
        ],
    }

    return consolidated, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="data/processed/consolidated_properties.csv.gz")
    parser.add_argument("--report", default="docs/audit/consolidation_report.json")
    args = parser.parse_args()

    consolidated, report = build_consolidated_table(Path(args.data_dir))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated.to_csv(output_path, index=False, compression="gzip")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {len(consolidated)} rows to {output_path}")
    print(f"Wrote consolidation report to {report_path}")


if __name__ == "__main__":
    main()
