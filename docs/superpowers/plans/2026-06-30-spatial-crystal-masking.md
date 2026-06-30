# Spatial Crystal Masking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement spatial block masking for Crys-JEPA so Encoder 1 sees incomplete atom tokens while Encoder 2 sees full tokens with crystal-family conditioning.

**Architecture:** Keep the implementation in `crystal_matrices.py`, where structures are already converted into tensors. Add explicit feature builders for Encoder 1 (`N x 103`) and Encoder 2 (`N x 110`), a crystal-family one-hot helper, toric spatial block masking, and collate support for padded full/corrupted views.

**Tech Stack:** Python 3.12, PyTorch, pymatgen, pytest.

---

### Task 1: Feature Builders

**Files:**
- Modify: `crystal_matrices.py`
- Modify: `tests/test_crystal_matrices.py`

- [x] Add tests for `build_encoder1_atom_feature_matrix` shape `(N, 103)`.
- [x] Add tests for `build_encoder2_atom_feature_matrix` shape `(N, 110)` with family one-hot broadcast.
- [x] Implement the builders using fractional coordinates plus 100-element atomic one-hot, without lattice values.

### Task 2: Crystal Family

**Files:**
- Modify: `crystal_matrices.py`
- Modify: `tests/test_crystal_matrices.py`

- [x] Add tests for cubic, hexagonal, tetragonal, orthorhombic, monoclinic, trigonal, and triclinic classification.
- [x] Implement lattice-parameter classification into the seven requested one-hot indices.
- [x] Return the family vector from `structure_to_tensors`.

### Task 3: Spatial Block Masking

**Files:**
- Modify: `crystal_matrices.py`
- Modify: `tests/test_crystal_matrices.py`

- [x] Add tests for toric box membership and guaranteed visible atoms.
- [x] Implement `spatial_block_visible_mask` and `sample_spatial_block`.
- [x] Filter only per-atom tensors for Encoder 1 and preserve per-crystal family features for Encoder 2.

### Task 4: Dataset/Collate Integration

**Files:**
- Modify: `crystal_matrices.py`
- Modify: `tests/test_crystal_matrices.py`

- [x] Update dataset samples to expose raw per-atom tensors, family one-hot, and Encoder 2 full features.
- [x] Update collate to return padded `encoder1_atom_features`, `encoder2_atom_features`, masks, and `mask_box`.
- [x] Run the test suite.
