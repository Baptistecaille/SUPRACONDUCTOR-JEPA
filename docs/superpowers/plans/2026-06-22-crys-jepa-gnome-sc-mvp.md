# Crys-JEPA-GNoME-SC MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, tested MVP pipeline for ranking GNoME-like crystal candidates for superconductivity.

**Architecture:** Implement a small Python package with focused modules for IO, chemistry, heuristic scoring, graph summaries, ranking, orchestration, and CLI execution. The MVP uses deterministic heuristics with typed dataclasses and produces explainable CSV/JSON outputs.

**Tech Stack:** Python 3.9+, standard library, pytest for tests, setuptools/pyproject packaging.

---

## File Structure

- Create `pyproject.toml`: package metadata, pytest config, CLI script.
- Create `README.md`: setup, usage, pipeline overview, extension notes.
- Create `src/crys_jepa_gnome_sc/__init__.py`: public exports and version.
- Create `src/crys_jepa_gnome_sc/__main__.py`: `python -m` entry point.
- Create `src/crys_jepa_gnome_sc/models.py`: dataclasses.
- Create `src/crys_jepa_gnome_sc/io.py`: CSV/JSON read/write.
- Create `src/crys_jepa_gnome_sc/chemistry.py`: formula parser and filters.
- Create `src/crys_jepa_gnome_sc/tc_model.py`: baseline Tc heuristic.
- Create `src/crys_jepa_gnome_sc/graph.py`: crystal graph summary.
- Create `src/crys_jepa_gnome_sc/surprise.py`: Crys-JEPA-inspired surprise score.
- Create `src/crys_jepa_gnome_sc/motif.py`: DINO-crystal-inspired motif score.
- Create `src/crys_jepa_gnome_sc/ranking.py`: EBRM ranking.
- Create `src/crys_jepa_gnome_sc/pipeline.py`: orchestration.
- Create `src/crys_jepa_gnome_sc/cli.py`: command-line interface.
- Create `examples/gnome_sample.csv`: representative fixture input.
- Create `notebooks/Crys_JEPA_GNoME_SC_MVP.ipynb`: Colab-oriented walkthrough.
- Create tests under `tests/`.

## Tasks

### Task 1: Chemistry Core

**Files:**
- Create: `tests/test_chemistry.py`
- Create: `src/crys_jepa_gnome_sc/models.py`
- Create: `src/crys_jepa_gnome_sc/chemistry.py`

- [x] Write failing tests for formula parsing and chemical filters.
- [x] Run the tests and verify missing implementation failures.
- [x] Implement dataclasses, formula parser, element classification, and filtering.
- [x] Run chemistry tests and verify they pass.

### Task 2: Stage Scoring Modules

**Files:**
- Create: `tests/test_scoring.py`
- Create: `src/crys_jepa_gnome_sc/tc_model.py`
- Create: `src/crys_jepa_gnome_sc/graph.py`
- Create: `src/crys_jepa_gnome_sc/surprise.py`
- Create: `src/crys_jepa_gnome_sc/motif.py`
- Create: `src/crys_jepa_gnome_sc/ranking.py`

- [x] Write failing tests for score ranges, motif labels, graph fallback, and ranking order.
- [x] Run the tests and verify missing implementation failures.
- [x] Implement deterministic scoring modules.
- [x] Run scoring tests and verify they pass.

### Task 3: IO, Pipeline, and CLI

**Files:**
- Create: `tests/test_pipeline_cli.py`
- Create: `src/crys_jepa_gnome_sc/io.py`
- Create: `src/crys_jepa_gnome_sc/pipeline.py`
- Create: `src/crys_jepa_gnome_sc/cli.py`
- Create: `src/crys_jepa_gnome_sc/__main__.py`
- Create: `src/crys_jepa_gnome_sc/__init__.py`
- Create: `pyproject.toml`
- Create: `examples/gnome_sample.csv`

- [x] Write failing end-to-end tests for loading candidates, running the pipeline, and CLI output.
- [x] Run the tests and verify missing implementation failures.
- [x] Implement IO, orchestration, CLI, package metadata, and sample data.
- [x] Run pipeline and CLI tests and verify they pass.

### Task 4: Documentation and Notebook

**Files:**
- Create: `README.md`
- Create: `notebooks/Crys_JEPA_GNoME_SC_MVP.ipynb`

- [x] Add README usage instructions and extension notes.
- [x] Add a minimal Colab-compatible notebook that installs the package, loads the sample CSV, runs screening, and displays top candidates.
- [x] Run full test suite.
- [x] Run example CLI command.
