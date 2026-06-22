# Crys-JEPA-GNoME-SC

Colab-compatible MVP research pipeline for discovering superconducting crystal candidates from GNoME-like candidate tables.

## Goal

This project screens crystal candidates and ranks them for superconductivity using:

1. Chemical and stability filtering
2. Baseline `Tc` prediction
3. Periodic crystal graph summary
4. Crys-JEPA-inspired latent surprise
5. DINO-crystal-inspired motif scoring
6. EBRM-style multi-objective ranking

This MVP is deterministic and offline. It is meant to produce useful ranked shortlists and clean extension points, not to replace trained Crys-JEPA, DINO-crystal, or DFT validation.

## Install

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install pytest
python -m pytest
```

## CLI Usage

Run the sample screen:

```bash
python -m crys_jepa_gnome_sc screen examples/gnome_sample.csv --out ranked.csv
```

The output includes rank, rank score, predicted `Tc`, stage scores, and semicolon-separated explanation reasons.

CSV and JSON inputs are supported. Each candidate needs at least:

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

## Python API

```python
from crys_jepa_gnome_sc.io import load_candidates, write_ranked_candidates
from crys_jepa_gnome_sc.pipeline import screen_candidates

candidates = load_candidates("examples/gnome_sample.csv")
ranked = screen_candidates(candidates)
write_ranked_candidates(ranked, "ranked.csv")
```

## Pipeline

```text
GNoME-like candidates
  |
  v
chemical + stability filtering
  |
  v
baseline Tc heuristic
  |
  v
crystal graph summary
  |
  v
Crys-JEPA-inspired surprise
  |
  v
DINO-crystal-inspired motifs
  |
  v
EBRM ranking
  |
  v
top candidates for DFT
```

## Extending The MVP

The heuristic modules are intentionally small:

- Replace `tc_model.predict_tc` with a trained baseline regressor.
- Replace `graph.build_graph_summary` with pymatgen, ASE, or PyG crystal graphs.
- Replace `surprise.score_surprise` with Crys-JEPA latent distance or energy.
- Replace `motif.score_motifs` with DINO-crystal embedding similarity.
- Tune `ranking.DEFAULT_WEIGHTS` for your validation objective.

Keep the dataclass interfaces stable and the CLI/notebook will continue to work.
