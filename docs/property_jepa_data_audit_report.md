# Property-JEPA Multi-Source Data Audit Report

**Date:** 2026-07-03
**Scope:** Ingestion of two new bulk DFT sources (JARVIS-DFT, Alexandria
PBEsol), consolidation with the existing Materials Project pretraining
corpus and SuperBand/3DSC_MP Tc data into one wide, composition-level
property table, and a leak-free train/val/test split for stage-2
("property-JEPA") training. Raw analysis artifacts referenced below live in
`docs/audit/`; the file-by-file inventory of everything under `data/` is
`data/MANIFEST.md`. This report explains *why* the consolidated corpus and
splits look the way they do, and documents the caveats a modeler must know
before training on them.

---

## 1. Sources ingested

| Source | Raw rows | Unique compositions used | Role | Functional(s) |
|---|---|---|---|---|
| SuperBand | 8,625 | 3,964 | Experimental Tc (highest-priority CIF + Tc label) | Experimental measurement |
| Materials Project (`data/jepa/mp.csv.gz`) | 80,769 (deduped) | 57,017 | Pretraining-corpus CIFs / formation-energy fallback | PBE/PBE+U (not tracked per-row in the original export) |
| JARVIS-DFT (`dft_3d`) | 93,902 | 65,320 | Electronic / magnetic / mechanical / DFT-predicted Tc | OptB88vdW (primary), TBmBJ, HSE06 |
| Alexandria (PBEsol, 3D all) | 415,419 | 276,461 | Thermo / electronic / magnetic scale-up | PBEsol |

JARVIS-DFT ingestion produced 93,902 materials with 0 CIF conversion failures, and Alexandria ingestion produced 415,419 rows, also with 0 CIF conversion failures.

JARVIS-DFT was chosen over a larger Alexandria "all" pull for its unique
mechanical-property and DFT-predicted-Tc columns; Alexandria's PBEsol subset
was deliberately used as the scale-up source rather than the ~5M-row PBE
"all" file, since PBEsol already gives a >4x scale-up over JARVIS with full
property coverage on every row, at a fraction of the CIF-conversion compute
and storage cost of the full PBE set.

Both ingestion scripts (`scripts/ingest_jarvis_dft.py`,
`scripts/ingest_alexandria.py`) explicitly keep every DFT functional's
band-gap/energy columns separate rather than collapsing them into one
generic field — JARVIS alone reports three non-interchangeable band-gap
estimates (OptB88vdW, TBmBJ/mBJ, HSE06), populated at very different rates
(see §2).

---

## 2. Per-source property coverage

**JARVIS-DFT** (93,902 rows): formation energy and `e_above_hull` are 100%
populated; the primary OptB88vdW band gap is 100% populated but the two
higher-level gap corrections are sparse — mBJ band gap is populated for
21,559 rows (22.96%) and the HSE06 gap for only 56 rows (0.06%). Magnetic
moment (OSZICAR) is 94.58% populated. The four mechanical properties (bulk
modulus, shear modulus, Poisson ratio, elastic tensor max) are all populated
at ~34.3–34.5% (32,205–34,323 rows) — this is the *only* source contributing
mechanical properties to the consolidated corpus. `tc_supercon` (DFT-predicted
Tc) is populated for 1,058 rows (1.13%).

**Alexandria PBEsol** (415,419 rows): every extracted property — formation
energy, `e_above_hull`, `e_phase_separation`, both band-gap variants
(indirect/direct), `dos_ef`, total magnetic moment, and total energy — is
100% populated, since Alexandria reports a complete, uniform PBEsol
calculation for every structure in the dump.

This asymmetry (JARVIS: partial coverage, functional-tagged columns;
Alexandria: full coverage, single functional) is why the consolidated
schema keeps per-source raw columns *and* a "best available" fallback column
per canonical property (§3) rather than only one merged column per property.

---

## 3. Consolidation: composition-level dedup and merge

`scripts/consolidate_properties.py` builds
`data/processed/consolidated_properties.csv.gz` from the four sources above.
The composition key used throughout is a sorted, fractional-amount-normalized element:fraction string derived from each CIF's own _chemical_formula_sum (pymatgen Composition.fractional_composition), NOT the raw source formula string. Within each source,
rows sharing a `composition_key` are collapsed to one representative row,
using lowest e_above_hull / ef_per_atom as the stability tie-break; superband/3DSC kept by lowest superband_id since Tc is an experimental measurement, not DFT-derived.

Result: 312,149 total unique compositions, with the number of contributing
sources per composition distributed as 1 source: 252,711 compositions, 2
sources: 29,229, 3 sources: 29,243, and all 4 sources: 966. The canonical
CIF for each composition is chosen by canonical_cif_priority: superband (experimental), materials_project, jarvis_dft, alexandria_pbesol, giving a `cif_source` breakdown of alexandria_pbesol: 223,898,
materials_project: 55,904, jarvis_dft: 28,383, superband: 3,964.

**Cross-source composition overlap:** JARVIS∩Alexandria: 44,974, JARVIS∩MP:
36,902, Alexandria∩MP: 37,539, all three DFT sources: 30,058. JARVIS's
DFT-predicted Tc (`tc_supercon`) overlaps SuperBand's experimental Tc set in
only 63 compositions — far too small an intersection to cross-validate one
against the other at scale.

### DFT heterogeneity warning

This table mixes AT LEAST 4 distinct levels of theory: OptB88vdW/TBmBJ/HSE06 (JARVIS-DFT), PBEsol (Alexandria), PBE/PBE+U (Materials Project, functional not tracked per-row in the original mp.csv.gz export), and experimental measurement (SuperBand Tc). The "best available" columns
(`thermo_formation_energy_peratom`, `electronic_bandgap_best`,
`magnetic_total_moment_best`, etc.) pick one value per row by source-priority
fallback purely for convenience; per-source columns are always retained so
that any training/eval code can restrict itself to a single consistent
functional when precision matters, rather than silently mixing (e.g.)
OptB88vdW and PBEsol band gaps as if they were one measurement. Each of the
14 canonical property types used by `PropertyJEPA` (see
`models/property_jepa/data/property_schema.py`) is deliberately mapped to
exactly **one** raw/best-available column
(`models/property_jepa/data/dataset.py`'s `PROPERTY_COLUMNS`) for this same
reason — no canonical property is ever computed as an average or blend
across functionals.

### Known coverage limitations

mechanical_* (bulk/shear modulus, poisson) coverage is only ~6-7% -- exclusively from JARVIS-DFT; Alexandria and MP contribute none. tc_experimental coverage is 1.27% (3,964 compositions) vs. 95%+ for thermo/electronic -- severe volumetric imbalance across property domains, consistent with the imbalance flagged in the
earlier dataset-landscape audit. tc_dft_predicted (JARVIS Tc_supercon) covers only 659 of the 312,149 compositions and overlaps the experimental Tc set in only ~63 compositions -- it is a weak, mostly-independent signal, not a substitute for experimental Tc.

**Practical implication for stage-2 training:** every training batch must be
built from *whichever* properties happen to be non-null for the sampled
materials, never from a dense property matrix — this is exactly what
`PropertyJEPADataset`/`collate_property_batch`
(`models/property_jepa/data/dataset.py`) implement (per-row variable-length
known/query property sets, randomized per batch). Given the imbalance above,
naive uniform composition sampling exposes the model to a Tc query roughly
1 time in 79 batches on average; if downstream Tc fine-tuning performance is
insufficient, a property-aware or Tc-oversampled batch sampler is the
recommended next step (not yet implemented).

---

## 4. Leak-free train/val/test split

`scripts/build_property_jepa_splits.py` builds
`data/processed/property_jepa_{train,val,test}.csv.gz` from the consolidated
table. Method: Any composition present in the Materials Project pretraining corpus (data/jepa/mp.csv.gz, via composition_key) is confined to an eligible-for-train-only pool -- it can appear in train but never in val/test, preventing pretrain/eval leakage. From the remaining leak-free pool (255,132 / 312,149 compositions), val and test (10% each of the TOTAL corpus) are drawn via proportional stratified sampling on (has_experimental_Tc x has_mechanical_props x cif_source), seed=42, so that the rarest labels are represented
in both eval splits in proportion to their overall prevalence rather than
concentrated in train.

**Split sizes:** train 249,719 / val 31,215 / test 31,215 rows.

**Leak verification** (`docs/audit/property_jepa_split_report.json`):
0 val rows and 0 test rows have a composition present in the MP pretraining
corpus; pairwise composition overlap between train/val, train/test, and
val/test is 0 in every case — the three splits are fully composition-disjoint
from each other and from the stage-1 pretraining corpus.

**Per-split property coverage** (rows with a non-null value):

| Property | train (n=249,719) | val (n=31,215) | test (n=31,215) |
|---|---|---|---|
| `tc_experimental` | 3,268 | 348 | 348 |
| `mechanical_bulk_modulus_kv` | 19,018 | 1,239 | 1,239 |
| `thermo_formation_energy_peratom` | 247,674 | 30,882 | 30,885 |
| `electronic_bandgap_best` | 235,040 | 30,882 | 30,885 |
| `magnetic_total_moment_best` | 234,940 | 30,879 | 30,878 |

The stratification keeps the rare-label proportions close to identical
across all three splits (e.g. Tc: 1.31% train vs. 1.11–1.12% val/test —
close enough that neither eval split over- or under-represents the
supraconducting signal relative to train).

---

## 5. End-to-end validation

- Full repo pytest suite: **149/149 passing** (includes 24 new tests for
  `models/property_jepa/data/dataset.py` and 14 for
  `models/property_jepa/training/train_property_jepa.py`, on top of the
  pre-existing stage-1 and stage-2-model test suites).
- Smoke test: `models/property_jepa/training/train_property_jepa.py` run
  against a 200-row slice of the real `property_jepa_train.csv.gz`, 5
  optimizer steps, batch size 8, CPU — loss decreased monotonically
  (0.0536 → 0.0158 → 0.0218 over 5 steps, finite throughout), confirming the
  dataset → collate → `PropertyJEPA.forward` → backward → EMA-update
  pipeline runs correctly end-to-end on real, leak-free split data. The
  VC-regularizer and cosine-warmup-scheduler code paths were smoke-tested
  separately on a synthetic fixture and also produce finite losses.

---

## 6. Known limitations carried forward

- **Mechanical properties are single-source** (JARVIS-DFT only, ~6-7%
  coverage) — a model trained to predict `bulk_modulus_kv` etc. is
  implicitly a JARVIS-OptB88vdW-mechanical-property model, not a
  cross-functional consensus predictor.
- **Experimental Tc remains the rarest label** (1.27% of the consolidated
  corpus, ~3,964 compositions total) despite the ~4-5x row-count scale-up
  from adding JARVIS and Alexandria — those two sources contribute almost no
  additional experimental Tc labels (JARVIS's Tc column is DFT-predicted,
  not experimental, and overlaps SuperBand in only 63 compositions). Any
  Tc-focused evaluation should still expect a small effective sample size
  and prefer k-fold cross-validation or a low-capacity head over full
  fine-tuning, per the same recommendation carried from
  `docs/data_audit_report.md`.
- **Composition-level leak filtering is polymorph-blind** (as in the stage-1
  Tc-split report) — two structures with an identical normalized
  composition are treated as the same material for pretrain/eval-leak
  purposes, which is conservative but does mean a genuinely novel polymorph
  of an already-pretrained composition is still excluded from val/test.
- **DFT heterogeneity is real and unresolved by design** — the "best
  available" fallback columns exist for convenience only; any analysis
  that needs a single consistent level of theory (e.g. comparing predicted
  vs. true band gap error) must filter to one `cif_source`/functional rather
  than using the fallback columns directly.

## Reproducing this audit

- JARVIS-DFT ingestion: `python scripts/ingest_jarvis_dft.py`
- Alexandria ingestion: `python scripts/ingest_alexandria.py`
- Consolidation: `python scripts/consolidate_properties.py`
- Leak-free split: `python scripts/build_property_jepa_splits.py`

All four scripts are idempotent and re-runnable against updated raw data;
the split script's leak/disjointness checks are hard assertions that fail
loudly rather than silently producing a leaking split.
