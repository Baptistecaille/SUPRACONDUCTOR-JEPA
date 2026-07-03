# Data Manifest — SUPRA-JEPA

Generated as part of the 2026-07-02 data audit and reorganization. See
`docs/data_audit_report.md` for the full analysis (duplicate detection,
schema consistency, atom-count/Tc distributions, composition-level leakage).

Every file below was re-verified against disk at the time of writing. If you
regenerate a file, re-run the corresponding script in `scripts/` and update
the row count / provenance note here.

## Pretraining corpus (`data/jepa/`)

| File | Rows | Columns | Provenance |
|---|---|---|---|
| `mp.csv.gz` | **80,769** (one row per unique `material_id`) | `material_id`, `cif`, `ef_per_atom` | Materials Project export. Originally shipped with 92,762 rows / 11,993 duplicated `material_id` groups (23,986 duplicate rows, all exact duplicates — identical CIF + `ef_per_atom` within each group). Deduplicated in place by `scripts/dedup_mp_corpus.py` (keep-first); see `data/processed/mp_dedup_report.json` for the before/after counts. This is now the **single source of truth** for JEPA pretraining — `crystal_matrices.SuperconductorDataset`/`create_dataloader` default to `deduplicate=True` as a defensive no-op. |

The redundant plaintext `mp.csv` (107 MB, byte-identical row content to
`mp.csv.gz` before dedup) has been removed — `mp.csv.gz` is the only copy on
disk going forward.

## Raw source data (`data/raw/`)

| File / dir | Rows / count | Columns | Provenance |
|---|---|---|---|
| `3DSC_MP.csv` | 5,773 (after stripping the `#`-prefixed provenance comment line) | 92 columns incl. `formula_sc`, `material_id_2`, `cif`, `formation_energy_per_atom_2`, `tc` | 3DSC_MP superconductor reference dataset (Materials Project subset), used as the composition-matching bridge between SuperBand and MP. |
| `superband.csv` | 8,625 | `id`, `formula`, `Tc`, `Time`, `doi`, `cif`, `system`, `SG_num`, `SG_syb`, `density`, + 6 unnamed/empty trailing columns | Raw SuperBand experimental Tc measurements (github.com/ljcj007/SuperBand). |
| `cifs/` | 10,904 `.cif` files | — | Loose CIF files referenced by path from `3DSC_MP.csv`'s `cif` column (`data/final/MP/cifs/<formula>-MP-<mp_id>[-synth_doped].cif`). |
| `3DSC_MP/MP/` | 0 usable files | — | Effectively empty (only contained macOS AppleDouble junk, now removed). Not used by any script. |

## SuperBand submodule (`data/SuperBand/`)

Git clone of `github.com/ljcj007/SuperBand`. `cif/` contains 32 locally
available CIFs (a small subset of the 8,625 SuperBand entries) plus a
`README.md`. Used opportunistically by `scripts/prepare_superband_mp_dataset.py`
(`read_local_superband_cif`) — when a local CIF exists it is preferred over
the 3DSC_MP-sourced CIF (`cif_source == "superband_local"`, 25/4,013 rows in
the matched set); otherwise the 3DSC_MP CIF is used (`cif_source ==
"3dsc_mp"`, 3,988/4,013 rows).

## Processed / matched data (`data/processed/`)

| File | Rows | Columns | Provenance |
|---|---|---|---|
| `superband_mp_matched.csv` | 4,013 | `material_id`, `cif`, `ef_per_atom`, `tc`, `superband_id`, `superband_formula`, `superband_cif_id`, `mp_material_id`, `cif_source` | Built by `scripts/prepare_superband_mp_dataset.py`: joins SuperBand rows to 3DSC_MP/MP structures on normalized composition. **No leak filtering applied** — this is the raw matched pool. |
| `superband_mp_matched_report.json` | — | — | Build report for the row above (reference-table dedup stats). |
| `pretrain_exclusion_list.csv` | 0 (by construction) | `material_id`, `composition_key` | Emitted by `scripts/build_tc_splits.py`. Lists any pretraining-corpus `material_id` whose composition matches a composition used in the Tc train/val/test splits below. **Currently empty** because `build_tc_splits.py` already excludes every Tc row whose composition overlaps `mp.csv.gz` before building the splits — there is nothing left in the pretraining corpus to additionally hold out. Re-run this script if `mp.csv.gz` or the Tc splits are ever regenerated independently. |
| `tc_train.csv` | 1,995 | `material_id`, `cif`, `ef_per_atom`, `tc`, `superband_id`, `superband_formula`, `superband_cif_id`, `mp_material_id`, `cif_source`, `composition_key` | Leak-free Tc training split. See below. |
| `tc_val.csv` | 428 | (same schema as `tc_train.csv`) | Leak-free Tc validation split. |
| `tc_test.csv` | 428 | (same schema as `tc_train.csv`) | Leak-free Tc test split. |
| `tc_split_report.json` | — | — | Full report from `scripts/build_tc_splits.py`: leak counts, dedup counts, per-fold Tc distribution, composition-disjointness verification. |
| `mp_dedup_report.json` | — | — | Report from `scripts/dedup_mp_corpus.py` (before/after row counts for `mp.csv.gz`). |

**Removed (no longer on disk):** `superband_mp_matched_no_train_leak.csv` (401 rows, exact-`mp_material_id` filter — 43/401 rows still leaked by composition under a different id) and `superband_mp_matched_no_3dsc_formula.csv` + its report (128 rows, exact-formula-string filter, self-reported incomplete). Both were fully superseded by `tc_train/val/test.csv` (§ below) and were deleted during cleanup rather than kept as `legacy/` reference copies. `scripts/build_tc_splits.py` and `scripts/filter_superband_no_3dsc_leakage.py` retain code comments describing what these files were and why they were replaced, and `data/processed/superband_mp_matched.csv` (the unfiltered pool both were built from) is still on disk if either needs to be regenerated.

### How the Tc train/val/test split was built

`scripts/build_tc_splits.py`:

1. Start from `superband_mp_matched.csv` (4,013 rows, unfiltered).
2. Recompute each row's composition directly from **its own CIF's**
   `_chemical_formula_sum` field (not the raw SuperBand formula string —
   the two can disagree, e.g. `Ag0.02Ge2Pd1.98Sr1` vs. the actual MP
   structure `Sr1Ge2Pd2` for a doped/undoped mismatch), normalized via
   `pymatgen.core.Composition.fractional_composition`.
3. Drop any row whose composition appears anywhere in the deduplicated
   pretraining corpus (`data/jepa/mp.csv.gz`) — **1,155/4,013 rows removed**
   (28.8%). This is strictly stronger than the legacy `mp_material_id`-only
   filter.
4. Deduplicate the remaining 2,858 rows by composition (keep first by
   `superband_id`) — 7 rows removed, leaving **2,851 unique, leak-free
   compositions**.
5. Split 70/15/15 (train/val/test) stratified on `log10(Tc)` decile, fixed
   seed 42, via `sklearn.model_selection.train_test_split`. Verified
   programmatically that no composition is shared across any pair of folds.

## Directory cleanup performed

The following macOS filesystem clutter was removed from `data/` (moved to
Trash, not permanently deleted): `.DS_Store` files under `data/`,
`data/jepa/`, `data/raw/`, `data/raw/3DSC_MP/`; AppleDouble `._*` files
`data/raw/._cifs`, `data/raw/3DSC_MP/._MP`, `data/raw/3DSC_MP/MP/._cifs`;
and the repo-root `.DS_Store`. The redundant plaintext `data/jepa/mp.csv`
was also removed (see above).

**Second cleanup pass:** `data/processed/legacy/` (the archived superseded
Tc-filter files) and the empty `data/raw/3DSC_MP/MP/` subdirectory were
removed entirely, since neither is referenced by any active script and
the source pool they were built from (`superband_mp_matched.csv`) remains
on disk for reproducibility.

## Multi-source property corpus (2026-07-03 scale-up)

Two new bulk DFT sources were ingested and consolidated with the data above
into a single wide, composition-level property table for stage-2
("property-JEPA") training. Full analysis in
`docs/property_jepa_data_audit_report.md`; raw schema/consolidation/split
reports are the JSON files under `docs/audit/` named below.

The repository now has two independent model stages:

| Path | Role | Data consumed |
|---|---|---|
| `models/crystal_structure_jepa/` | Stage 1: crystal reconstruction / masked-structure JEPA. This is the original model moved under a dedicated module. | `data/jepa/mp.csv.gz` and the leak-free Tc downstream split when evaluating stage-1 embeddings. |
| `models/property_jepa/` | Stage 2: property-JEPA. New code only; it does not import from `models/crystal_structure_jepa/`. | `data/processed/property_jepa_{train,val,test}.csv.gz`, backed by `consolidated_properties.csv.gz`. |

The stage-2 architecture and training contract are documented in
`docs/property_jepa_design.md`.

### New raw sources (`data/raw/`)

| File | Rows | Provenance |
|---|---|---|
| `jarvis_dft.csv.gz` | 93,902 | JARVIS-DFT (`dft_3d`, NIST, via jarvis-tools). Every entry's structure converted to CIF (0 conversion failures). Contributes electronic (OptB88vdW/TBmBJ/HSE06 band gaps), magnetic, mechanical (bulk/shear modulus, Poisson ratio, elastic tensor max), and DFT-predicted Tc (`tc_supercon`) properties — the **only** source of mechanical properties in the consolidated corpus. Built by `scripts/ingest_jarvis_dft.py`; schema/coverage report at `docs/audit/jarvis_schema_report.json`. |
| `alexandria_pbesol.csv.gz` | 415,419 | Alexandria materials database, PBEsol functional, 3D-all subset (deliberately chosen over the ~5M-row PBE "all" file for compute/storage tradeoffs — see the audit report). Every extracted property (formation energy, `e_above_hull`, `e_phase_separation`, band gaps, `dos_ef`, magnetic moment) is 100% populated. Built by `scripts/ingest_alexandria.py`; schema report at `docs/audit/alexandria_schema_report.json`. |

### Consolidated property table (`data/processed/`)

| File | Rows | Provenance |
|---|---|---|
| `consolidated_properties.csv.gz` | 312,149 (one row per unique `composition_key`) | Composition-level merge of SuperBand + Materials Project + JARVIS-DFT + Alexandria PBEsol, built by `scripts/consolidate_properties.py`. `composition_key` is derived from each CIF's own `_chemical_formula_sum` (pymatgen fractional composition), not the raw source formula string. Canonical CIF per composition chosen by source priority: superband (experimental) > materials_project > jarvis_dft > alexandria_pbesol. Keeps **both** per-source raw columns and "best available" fallback columns per canonical property — see the DFT-heterogeneity warning in `docs/property_jepa_data_audit_report.md` §3 before using the fallback columns for anything precision-sensitive. Full stats in `docs/audit/consolidation_report.json`. |
| `property_jepa_train.csv.gz` | 249,719 | Leak-free stage-2 training split (see below). |
| `property_jepa_val.csv.gz` | 31,215 | Leak-free stage-2 validation split. |
| `property_jepa_test.csv.gz` | 31,215 | Leak-free stage-2 test split. |

Built by `scripts/build_property_jepa_splits.py`. Any composition present in
the MP pretraining corpus (`data/jepa/mp.csv.gz`) is confined to a
train-only-eligible pool, then val/test (10% each of the total corpus) are
drawn from the remaining leak-free pool via proportional stratified sampling
on `(has_experimental_Tc × has_mechanical_props × cif_source)`, seed 42.
Verified 0 composition overlap between any pair of {train, val, test, MP
pretraining corpus}. Full report (including per-split property coverage) at
`docs/audit/property_jepa_split_report.json`.

### Stage-2 model code consuming this data

`models/property_jepa/data/dataset.py` (`PropertyJEPADataset`,
`collate_property_batch`, `PropertyStats`, `create_property_dataloader`)
reads the three split files above and bridges them into
`PropertyJEPA.forward()`'s batch contract; each of the 14 canonical property
types in `models/property_jepa/data/property_schema.py` maps to exactly one
raw/best-available column (`PROPERTY_COLUMNS` in `dataset.py`) to avoid
mixing DFT functionals within a canonical property.

## Known limitations / caveats

- SuperBand CIFs are only locally available for 32/8,625 raw entries; the
  rest of the matched set relies on 3DSC_MP-sourced MP structures for the
  same composition, which is a reasonable proxy but not the exact
  experimental structure SuperBand measured Tc on.
- Composition-level leak filtering treats two structures as "the same" if
  they reduce to an identical normalized composition — it does **not**
  distinguish polymorphs (same composition, different structure/space
  group). This is the correct, conservative choice for a JEPA encoder that
  is expected to primarily learn composition- and coarse-structure-level
  representations, but it does mean a genuinely novel polymorph of an
  already-seen composition is still excluded from the Tc splits.
- With ~2,000 labeled training rows (and much fewer, ~400, under the old
  filter), still favor k-fold cross-validation and a low-capacity
  regression head over full fine-tuning of the 512-dim JEPA transformer.
- In the larger multi-source `consolidated_properties.csv.gz` corpus,
  mechanical properties remain ~6-7% covered (JARVIS-only) and experimental
  Tc remains the rarest label at 1.27% (3,964 compositions) even after the
  ~4x row-count scale-up from JARVIS + Alexandria — neither new source
  contributes meaningfully more experimental Tc labels. See
  `docs/property_jepa_data_audit_report.md` for the full breakdown and its
  implications for stage-2 batch sampling.
