"""Build the structure-level, leak-free train/val/test split for property-JEPA's
lambda/omega_log/Tc "physics module" (Allen-Dynes fine-tuning heads).

Unlike `consolidate_properties.py` (which collapses every source to one
row per COMPOSITION, on the grounds that formation energy / band gap /
etc. are dominated by the most-stable polymorph), this script keeps ONE
ROW PER STRUCTURE. Electron-phonon coupling (lambda), the log-averaged
phonon frequency (omega_log), and the resulting Tc are structure-dependent,
not composition-dependent: two polymorphs of the same composition can have
wildly different lambda/Tc (e.g. one N:Zr polymorph in JARVIS supercon_3d
has Tc=2.4K, another has Tc=29.5K). Collapsing them to a single
"representative" row would throw away exactly the structure-sensitivity
this module needs to learn -- the model is expected to learn this
invariance/covariance itself from the lattice tensor + atom positions, not
have it pre-decided by a dedup heuristic.

Sources merged (see docs/audit/*_supercon_schema_report.json):
  * data/raw/jarvis_supercon.csv.gz     (1219 rows: JARVIS supercon_3d +
    supercon_2d -- has the full alpha^2F(omega) spectrum)
  * data/raw/alexandria_supercon.csv.gz (8253 rows: Alexandria_SuperConDB --
    scalar lambda/omega_log/dos_ef/debye_freq only, no spectrum)

No exact-CIF duplicates were found within or across these two sources, so
every row is kept.

Leakage control
----------------
Structures whose NORMALIZED COMPOSITION (parsed from each row's own CIF
`_chemical_formula_sum`, mirroring `build_tc_splits.py` /
`build_property_jepa_splits.py`) appears anywhere in the stage-1
pretraining corpus (data/jepa/mp.csv.gz) are confined to a train-eligible-
only pool, exactly as in `build_property_jepa_splits.py`.

Split granularity is DELIBERATELY THE COMPOSITION, not the individual
structure row: every polymorph of a given composition is assigned to the
SAME split (train, val, or xor test). This still allows the model to see
many distinct structures per composition within a split (the whole point
of keeping polymorphs unmerged), while preventing one polymorph of a
composition from leaking into train and a different polymorph of the SAME
composition from leaking into val/test (which would let the model
memorize composition-level shortcuts and inflate held-out performance).

Usage:
    python scripts/build_physics_dataset.py \
        --jarvis-supercon data/raw/jarvis_supercon.csv.gz \
        --alexandria-supercon data/raw/alexandria_supercon.csv.gz \
        --pretrain-corpus data/jepa/mp.csv.gz \
        --output-dir data/processed \
        --report docs/audit/physics_dataset_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

FORMULA_SUM_RE = re.compile(r"_chemical_formula_sum\s+'?([^\n']+)'?")


def extract_formula_sum(cif: str) -> str | None:
    if not isinstance(cif, str):
        return None
    m = FORMULA_SUM_RE.search(cif)
    return m.group(1).strip() if m else None


def composition_key(formula: object) -> str | None:
    if formula is None or (isinstance(formula, float) and pd.isna(formula)):
        return None
    try:
        comp = Composition(str(formula)).fractional_composition
    except Exception:
        return None
    parts = [f"{el}:{amt:.6g}" for el, amt in sorted(comp.get_el_amt_dict().items())]
    return "|".join(parts)


def load_jarvis_supercon(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source"] = "jarvis_supercon"
    df["source_id"] = df["jarvis_id"]
    df["has_a2f_spectrum"] = True
    df["dos_ef"] = np.nan
    df["debye_freq"] = np.nan
    return df[
        [
            "source",
            "source_id",
            "dimensionality",
            "cif",
            "stability",
            "press_gpa",
            "lambda_ep",
            "omega_log",
            "tc_allen_dynes",
            "dos_ef",
            "debye_freq",
            "has_a2f_spectrum",
            "a2f_resampled",
            "a2f_original_x",
            "a2f_original_y",
        ]
    ]


def load_alexandria_supercon(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source"] = "alexandria_supercon"
    df["source_id"] = df["alexandria_id"]
    df["dimensionality"] = "3D"
    df["stability"] = None
    df["press_gpa"] = None
    df["has_a2f_spectrum"] = False
    df["a2f_resampled"] = None
    df["a2f_original_x"] = None
    df["a2f_original_y"] = None
    return df[
        [
            "source",
            "source_id",
            "dimensionality",
            "cif",
            "stability",
            "press_gpa",
            "lambda_ep",
            "omega_log",
            "tc_allen_dynes",
            "dos_ef",
            "debye_freq",
            "has_a2f_spectrum",
            "a2f_resampled",
            "a2f_original_x",
            "a2f_original_y",
        ]
    ]


def build_combined(jarvis_path: str, alexandria_path: str) -> pd.DataFrame:
    jsc = load_jarvis_supercon(jarvis_path)
    asc = load_alexandria_supercon(alexandria_path)
    combined = pd.concat([jsc, asc], ignore_index=True)

    n_before = len(combined)
    combined = combined.drop_duplicates(subset=["cif"], keep="first").reset_index(drop=True)
    n_exact_cif_dups_removed = n_before - len(combined)

    combined["formula_sum"] = combined["cif"].map(extract_formula_sum)
    combined["composition_key"] = combined["formula_sum"].map(composition_key)
    combined["row_id"] = combined["source"] + ":" + combined["source_id"].astype(str)
    combined = combined.dropna(subset=["composition_key"]).reset_index(drop=True)

    return combined, n_exact_cif_dups_removed


def build_splits(
    combined: pd.DataFrame,
    pretrain_ckeys: set[str],
    seed: int = 42,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> tuple[pd.DataFrame, dict]:
    df = combined.copy()
    df["leaked_vs_pretrain"] = df["composition_key"].isin(pretrain_ckeys)

    # log10(Tc) decile as the stratification key -- mirrors build_tc_splits.py's
    # rationale (keep the Tc distribution comparable across folds), computed
    # per COMPOSITION GROUP (using each group's max Tc) since the split unit
    # is the composition, not the row.
    comp_groups = df.groupby("composition_key")
    comp_max_tc = comp_groups["tc_allen_dynes"].max()
    log_tc = np.log10(comp_max_tc.clip(lower=1e-6))
    try:
        tc_decile = pd.qcut(log_tc, q=10, labels=False, duplicates="drop")
    except ValueError:
        tc_decile = pd.Series(0, index=log_tc.index)
    comp_strat = tc_decile.astype(str)

    comp_info = pd.DataFrame(
        {
            "composition_key": comp_strat.index,
            "strat_key": comp_strat.values,
            "leaked_vs_pretrain": comp_groups["leaked_vs_pretrain"].first().values,
            "n_rows": comp_groups.size().values,
        }
    )

    leak_free_comps = comp_info.loc[~comp_info["leaked_vs_pretrain"]]
    n_total_rows = len(df)
    n_val_target = int(round(val_frac * n_total_rows))
    n_test_target = int(round(test_frac * n_total_rows))
    n_leak_free_rows = leak_free_comps["n_rows"].sum()
    val_frac_of_leakfree = n_val_target / n_leak_free_rows if n_leak_free_rows else 0.0
    test_frac_of_leakfree = n_test_target / n_leak_free_rows if n_leak_free_rows else 0.0

    rng = np.random.default_rng(seed)
    val_comps: set[str] = set()
    test_comps: set[str] = set()
    for _, grp in leak_free_comps.groupby("strat_key"):
        ck_arr = grp["composition_key"].to_numpy().copy()
        rng.shuffle(ck_arr)
        n_rows_cum = grp.set_index("composition_key")["n_rows"]
        # Greedily assign whole compositions to val/test until each hits its
        # row-count target for this stratum, biggest-first is avoided by the
        # shuffle above (keeps composition-size effects from concentrating).
        target_val = val_frac_of_leakfree * grp["n_rows"].sum()
        target_test = test_frac_of_leakfree * grp["n_rows"].sum()
        running_val, running_test = 0, 0
        for ck in ck_arr:
            rows_here = int(n_rows_cum[ck])
            if running_val < target_val:
                val_comps.add(ck)
                running_val += rows_here
            elif running_test < target_test:
                test_comps.add(ck)
                running_test += rows_here
            # else: falls through to train

    df["split"] = "train"
    df.loc[df["composition_key"].isin(val_comps), "split"] = "val"
    df.loc[df["composition_key"].isin(test_comps), "split"] = "test"

    report = {
        "n_total_rows": int(n_total_rows),
        "n_total_compositions": int(comp_info.shape[0]),
        "n_leaked_vs_pretrain_compositions": int(comp_info["leaked_vs_pretrain"].sum()),
        "split_row_counts": df["split"].value_counts().to_dict(),
        "split_composition_counts": {
            split: int(df.loc[df["split"] == split, "composition_key"].nunique())
            for split in ["train", "val", "test"]
        },
        "polymorph_rows_per_composition_max": int(comp_info["n_rows"].max()),
        "compositions_with_multiple_structures": int((comp_info["n_rows"] > 1).sum()),
    }
    return df, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jarvis-supercon", default="data/raw/jarvis_supercon.csv.gz")
    parser.add_argument("--alexandria-supercon", default="data/raw/alexandria_supercon.csv.gz")
    parser.add_argument("--pretrain-corpus", default="data/jepa/mp.csv.gz")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--report", default="docs/audit/physics_dataset_report.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    combined, n_exact_cif_dups_removed = build_combined(args.jarvis_supercon, args.alexandria_supercon)

    pretrain = pd.read_csv(args.pretrain_corpus, usecols=["material_id", "cif"])
    pretrain["formula_sum"] = pretrain["cif"].map(extract_formula_sum)
    pretrain["composition_key"] = pretrain["formula_sum"].map(composition_key)
    pretrain_ckeys = set(pretrain["composition_key"].dropna())

    df, split_report = build_splits(combined, pretrain_ckeys, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        out_path = output_dir / f"physics_jepa_{split}.csv.gz"
        df.loc[df["split"] == split].drop(columns=["split"]).to_csv(
            out_path, index=False, compression="gzip"
        )

    report = {
        "sources": ["data/raw/jarvis_supercon.csv.gz", "data/raw/alexandria_supercon.csv.gz"],
        "dedup_method": (
            "STRUCTURE-level (one row per polymorph); only exact-CIF duplicates "
            "are removed. Composition is used only to keep every polymorph of "
            "a given composition within the same train/val/test split (and to "
            "exclude pretrain-corpus-leaked compositions), not to collapse rows."
        ),
        "n_exact_cif_duplicates_removed": int(n_exact_cif_dups_removed),
        **split_report,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {len(df)} total rows across train/val/test to {output_dir}/physics_jepa_{{train,val,test}}.csv.gz")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
