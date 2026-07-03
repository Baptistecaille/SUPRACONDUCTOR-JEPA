# Property-JEPA Design

**Date:** 2026-07-03
**Scope:** Stage-2 property-JEPA architecture, training objective, invariance
contract, and known limitations.

This document describes the second model stage under
`models/property_jepa/`. Stage 1 (`models/crystal_structure_jepa/`) is the
earlier crystal reconstruction / masked-structure JEPA and remains an
independent module. Stage 2 is written as new code and does not import stage-1
model or data utilities.

## 1. Goal

Property-JEPA learns a latent relationship between crystal structure and
scalar material properties. A sample contains:

- raw crystal structure: atom species, fractional coordinates, and lattice;
- any available scalar properties: thermodynamic, electronic, magnetic,
  mechanical, experimental Tc, and DFT-predicted Tc;
- a randomly selected query property to predict in latent space.

The model does not predict a raw scalar directly. It predicts the latent
embedding of the queried `(property_type, value)` pair, using the crystal
embedding plus any other known properties of the same material as context.

## 2. Data Contract

The training splits are:

| File | Rows | Purpose |
|---|---:|---|
| `data/processed/property_jepa_train.csv.gz` | 249,719 | Stage-2 training. |
| `data/processed/property_jepa_val.csv.gz` | 31,215 | Leak-free validation. |
| `data/processed/property_jepa_test.csv.gz` | 31,215 | Leak-free test. |

They are built from `data/processed/consolidated_properties.csv.gz`, a
composition-level merge of SuperBand, Materials Project, JARVIS-DFT, and
Alexandria PBEsol. The split method confines every composition seen in the
Materials Project pretraining corpus to train-only eligibility; validation
and test contain 0 MP-pretraining-overlapping compositions.

The canonical property vocabulary is defined in
`models/property_jepa/data/property_schema.py` and currently contains 14
property types:

| Domain | Property types |
|---|---|
| Thermodynamic | `formation_energy_peratom`, `e_above_hull`, `e_phase_separation` |
| Electronic | `bandgap_optb88vdw`, `bandgap_mbj`, `bandgap_hse`, `dos_ef` |
| Magnetic | `magmom_total` |
| Mechanical | `bulk_modulus_kv`, `shear_modulus_gv`, `poisson_ratio`, `elastic_tensor_max` |
| Superconducting | `tc_experimental`, `tc_dft_predicted` |

`models/property_jepa/data/dataset.py` maps each canonical property type to a
single source column. This avoids accidentally averaging or blending
different DFT functionals.

## 3. Rotation Invariance

The crystal encoder must be invariant to rigid rotations of the crystal in
Cartesian space. Stage 2 enforces this by never feeding raw Cartesian
positions or raw lattice orientation into the transformer.

Per-atom token:

```text
one_hot(atomic_number, 100) || fractional_coordinates(3)
```

Fractional coordinates are expressed in the crystal's own lattice basis, so a
global Cartesian rotation of the entire cell does not change them.

Lattice token:

`models/property_jepa/data/lattice_features.py` computes a 12D invariant
global descriptor:

```text
metric_tensor_6d(L) || reduced_cell_parameters(L)
```

The metric tensor is `G = L @ L.T`, where rows of `L` are the lattice vectors.
For a rigid rotation `R`, the rotated lattice is `L' = L @ R.T`, and:

```text
G' = L' @ L'.T = L @ R.T @ R @ L.T = L @ L.T = G
```

The 6D upper triangle of `G` is optionally normalized by
`num_atoms ** (1/3)`. The reduced cell parameters `(a, b, c, alpha, beta,
gamma)` provide an interpretable second invariant view of the same unit
cell. Tests cover equality after random SO(3) rotations.

## 4. Architecture

The implemented path is:

```text
atom tokens + lattice invariant features
    -> RotationInvariantCrystalEncoder
    -> z_crystal

known (property_type, value) pairs
    -> online PropertyEncoder
    -> masked mean pool
    -> known_property_context

z_crystal + known_property_context + query_property_type_embedding
    -> PropertyPredictor
    -> z_pred

query (property_type, ground_truth_value)
    -> EMA TargetPropertyEncoder
    -> z_target

loss = MSE(z_pred, stop_gradient(z_target)) + optional VC regularizer
```

Main modules:

| Module | Role |
|---|---|
| `model/crystal_encoder.py` | Transformer over atom tokens, with the invariant lattice descriptor injected into the CLS token. |
| `model/property_encoder.py` | MLP encoder for `(property_type_id, normalized_value)` pairs; online branch encodes known properties, EMA target branch encodes the queried property. |
| `model/predictor.py` | MLP predictor conditioned on the crystal embedding, pooled known-property context, and query property type. |
| `model/losses.py` | Latent-space MSE prediction loss. |
| `model/regulizers.py` | Optional variance/covariance anti-collapse regularizers. |
| `training/train_property_jepa.py` | CPU/GPU training entry point, checkpoint writer, and smoke-testable CLI. |

The design is close to I-JEPA/Crys-JEPA in spirit: the model predicts a
latent target representation rather than reconstructing raw labels. The
target property encoder is EMA-updated from the online property encoder, so
the target branch is stable and gradient-free.

## 5. Loss And Missing Labels

The corpus is sparse: most materials do not have every property. The dataset
therefore builds variable-length property sets per material.

For each batch sample:

1. List every non-null canonical property.
2. Randomly select one available property as the query.
3. Treat the remaining available properties as known context.
4. Pad known-property slots across the batch and provide a boolean mask.
5. Compute the JEPA loss only for the selected query property.

Rows with only one available property still train: their known-property
context is an all-zero vector, so prediction relies on crystal structure and
query type alone.

## 6. My Assessment

The architecture is a solid first stage-2 design. The strongest choices are:

- the rotation-invariant lattice featurization is simple, provable, and easy
  to test;
- the known/query property split is well matched to a sparse multi-property
  corpus;
- conditioning the predictor on property type is important because "predict
  band gap" and "predict Tc" are not the same task;
- keeping stage 2 independent from stage 1 makes experiments safer and avoids
  hidden coupling while the property objective is still evolving.

The main risk is that the current loss predicts an embedding produced from
the scalar property value itself. This can work as a JEPA-style latent target,
but it does not yet guarantee that the learned embedding is calibrated for
raw scalar prediction. For evaluation, the next practical step should be a
small decoding head from `z_pred` or `z_crystal` back to normalized scalar
values, with metrics such as MAE/RMSE for regression and AUROC/AP for
superconductor classification thresholds.

I would also watch the sampling policy. Uniform composition sampling will
under-train rare labels, especially experimental Tc and mechanical
properties. A property-balanced sampler, or a two-stage schedule that first
learns broad thermo/electronic structure and then oversamples rare Tc/mech
queries, is likely to matter more than making the transformer larger.

## 7. Known Limitations

- Experimental Tc is only 1.27% of the consolidated corpus. This is still a
  small effective training set for superconductivity.
- Mechanical labels are JARVIS-only and cover only ~6-7% of the corpus.
- The consolidated data mixes DFT functionals and experimental measurements.
  The code avoids silent column blending, but the user still needs to choose
  functional-consistent subsets for precision-sensitive evaluation.
- The leak split is composition-level and intentionally conservative. It does
  not distinguish polymorphs of the same reduced composition.
- Fractional coordinates plus a learned positional embedding are not
  permutation invariant by themselves. The dataset sorts atoms
  deterministically, which makes the representation stable, but a future
  graph/equivariant encoder could encode symmetry more naturally.
- The current model is invariant to global Cartesian rotation, but not
  explicitly invariant to all crystallographic equivalent cell choices,
  origin shifts, atom-list permutations beyond deterministic sorting, or
  supercell transformations.

## 8. Open Architecture Questions

1. Should the main downstream target be scalar regression, binary
   superconductivity classification, or both? If both, we should add an
   explicit evaluation head and task-specific metrics.
2. Should experimental Tc and DFT-predicted Tc share information, or should
   they remain fully separate property types? They overlap weakly and may
   encode different physics/noise.
3. Should property query sampling be uniform over available properties, or
   balanced over property types so rare labels appear often enough?
4. Should the lattice descriptor include volume/density features explicitly,
   or is `metric_tensor_6d || cell_parameters` enough?
5. Should the crystal encoder remain a transformer over sorted atom tokens,
   or move toward a symmetry-aware graph/equivariant model once the dataset
   and evaluation are stable?
6. Should the model pretrain on all property labels jointly, then fine-tune a
   Tc-specific head, or directly optimize a mixed JEPA + supervised Tc loss?

## 9. Verification Status

The stage-2 implementation currently has tests for lattice invariance,
crystal encoder shape/invariance behavior, property encoders, predictor,
loss/regularizers, dataset/collate, and the training script. The previously
recorded full-suite status after implementation was **149/149 passing**, with
a real-data CPU smoke test over 5 batches producing finite decreasing loss.

Re-run from the repository root with:

```bash
pytest
```

Smoke-test training with:

```bash
python -m models.property_jepa.training.train_property_jepa \
  --train-csv data/processed/property_jepa_train.csv.gz \
  --checkpoint-path checkpoints/property_jepa_smoke.pt \
  --epochs 1 \
  --max-batches 5 \
  --batch-size 8 \
  --device cpu
```
