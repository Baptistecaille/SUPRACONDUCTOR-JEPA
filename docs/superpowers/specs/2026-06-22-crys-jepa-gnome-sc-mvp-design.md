# Crys-JEPA-GNoME-SC MVP Design

## Goal

Build a Colab-compatible MVP research pipeline for screening GNoME-like crystal candidates and ranking plausible superconducting candidates. The MVP must be runnable without network access, learned checkpoints, or heavyweight crystal-graph dependencies, while leaving clean extension points for real Crys-JEPA, DINO-crystal, and DFT workflows later.

## Scope

The first version is a deterministic screening tool, not a full trained ML system. It will ingest small to medium CSV or JSON candidate tables, compute interpretable heuristic scores for each pipeline stage, rank the candidates, and write the result as CSV or JSON.

Included:

- Python package under `src/crys_jepa_gnome_sc`.
- CLI entry point via `python -m crys_jepa_gnome_sc screen`.
- Colab-friendly notebook scaffold in `notebooks/`.
- Example input data in `examples/`.
- Unit tests for formula parsing, filtering, scoring, ranking, and CLI behavior.

Excluded from this MVP:

- Training Crys-JEPA or DINO models.
- Downloading GNoME data automatically.
- Running DFT calculations.
- Requiring PyTorch Geometric, pymatgen, ASE, or Materials Project APIs.

## Architecture

The package is split into focused modules:

- `models.py`: typed dataclasses for raw candidates, stage scores, and ranked results.
- `io.py`: load and write candidates from CSV and JSON files.
- `chemistry.py`: parse chemical formulas, classify elements, and apply chemical/stability filters.
- `tc_model.py`: baseline deterministic superconducting transition temperature heuristic.
- `graph.py`: periodic crystal graph summary from available structure fields, with formula-derived fallback features.
- `surprise.py`: Crys-JEPA-inspired latent surprise heuristic.
- `motif.py`: DINO-crystal-inspired motif scoring heuristic.
- `ranking.py`: normalized EBRM-style multi-objective ranking.
- `pipeline.py`: orchestrates the end-to-end flow.
- `cli.py`: command-line interface.
- `__main__.py`: module execution entry point.

The modules communicate through dataclasses instead of free-form dictionaries where practical. IO accepts missing optional fields and the pipeline applies penalties rather than crashing when non-essential structure data is unavailable.

## Candidate Input Model

Each row should include at minimum:

- `material_id`
- `formula`

Optional fields:

- `formation_energy_per_atom`
- `energy_above_hull`
- `band_gap`
- `density`
- `space_group`
- `volume`
- `num_sites`
- `is_metallic`
- `metadata`

CSV values are parsed into Python types when possible. JSON input can be either a list of candidate objects or an object with a top-level `candidates` list.

## Pipeline Stages

1. Chemical and stability filtering:
   - Reject formulas that cannot be parsed.
   - Reject candidates containing radioactive or unsupported synthetic-only elements.
   - Penalize high `energy_above_hull`; by default filter out candidates above `0.20 eV/atom`.
   - Prefer metallic or low-band-gap candidates.

2. Baseline `Tc` prediction:
   - Compute an interpretable heuristic using composition families, metallicity, light elements, pressure-friendly hydrides, cuprate/iron-pnictide-like patterns, borides, carbides, nitrides, and stability.
   - Return a non-negative Kelvin estimate and explanation factors.

3. Crystal graph conversion:
   - Produce a compact graph summary with estimated nodes, edges, average coordination, periodicity confidence, and feature vector.
   - Use `num_sites`, `space_group`, `density`, and `volume` when available.
   - Fall back to formula-derived features when structure fields are missing.

4. Crys-JEPA latent surprise:
   - Score unusual but plausible chemistry and structural combinations.
   - Reward uncommon superconductivity-relevant element mixes.
   - Penalize unstable, unparseable, or sparse candidates.

5. DINO-crystal motif scoring:
   - Score motif families associated with superconductivity, including hydrides, borides, carbides/nitrides, cuprates, iron pnictides/chalcogenides, layered oxides, and heavy-element metallic systems.
   - Return a normalized score and motif labels.

6. EBRM ranking:
   - Normalize `Tc`, stability, surprise, motif, graph confidence, and synthesizability.
   - Combine them with configurable weights.
   - Output `rank`, `rank_score`, all stage scores, and explanation text.

## Error Handling

- Invalid formulas are not fatal; they receive a rejected filter status with an explanation.
- Missing optional values reduce confidence but do not stop processing.
- Empty input files raise a clear CLI error.
- Unsupported file extensions raise a clear CLI error.
- Output directory creation is handled by the CLI when needed.

## Testing Strategy

Tests will cover:

- Formula parsing with common formulas such as `MgB2`, `LaH10`, `YBa2Cu3O7`, and invalid inputs.
- Chemical filtering of stable, unstable, radioactive, and insulating candidates.
- `Tc`, surprise, motif, graph, and EBRM score ranges.
- Ranking order on a fixed fixture dataset.
- CLI end-to-end run from CSV input to CSV output.

The test suite must run with `pytest` and no network access.

## Success Criteria

- `python -m crys_jepa_gnome_sc screen examples/gnome_sample.csv --out ranked.csv` produces a ranked output file.
- `pytest` passes.
- The README explains setup, CLI usage, Python API usage, Colab usage, and how to replace heuristic components later.
- The notebook demonstrates loading the sample data, running the pipeline, and displaying the top candidates.
