# Crystal Matrices Crys-JEPA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implémenter dans `JEPA/crystal_matrices.py` les 4 fonctions de construction des matrices d'entrée du Crys-JEPA (arxiv 2605.14759), utilisables avec des `pymatgen.Structure`.

**Architecture:** Un seul fichier `JEPA/crystal_matrices.py` exporté depuis `JEPA/__init__.py`. Trois fonctions principales indépendantes (`compute_lattice_6d`, `build_atom_feature_matrix`, `build_energy_weight_matrix`) plus une fonction helper `structure_to_tensors` qui orchestre les deux premières à partir d'un `pymatgen.Structure`.

**Tech Stack:** Python 3.12, PyTorch ≥ 2.12, pymatgen (nouvelle dépendance).

## Global Constraints

- Python ≥ 3.12
- PyTorch ≥ 2.12.1
- La matrice d'atomes doit être exactement `(N, 109)` : 3 coords fractionnaires + 100 one-hot éléments + 6 lattice
- Le one-hot couvre 100 éléments (numéros atomiques 1–100)
- La lattice 6D est le triangle supérieur de la matrice symétrique issue de la décomposition polaire SVD, normalisé par `num_atoms^(1/3)`
- La matrice de poids énergie a `1` sur la diagonale et `1 - exp(-|Eᶠᵢ - Eᶠₖ|)` hors diagonale
- Aucune dépendance torch-geometric dans ce module
- Suivre le style existant du projet (pas de commentaires sauf WHY non-évident, pas de docstrings multi-lignes)

---

## Fichiers

| Action | Chemin | Responsabilité |
|---|---|---|
| Créer | `JEPA/crystal_matrices.py` | 4 fonctions de construction des matrices |
| Créer | `tests/test_crystal_matrices.py` | Tests unitaires TDD |
| Modifier | `JEPA/__init__.py` | Exporter les 4 fonctions |
| Modifier | `pyproject.toml` | Ajouter `pymatgen` en dépendance |

---

## Task 1 : Ajouter pymatgen comme dépendance

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: rien
- Produces: `pymatgen` disponible dans l'environnement

- [ ] **Step 1 : Ajouter pymatgen à pyproject.toml**

Modifier `pyproject.toml` pour ajouter pymatgen dans `dependencies` :

```toml
[project]
name = "supra-jepa"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.5.0",
    "pandas>=3.0.3",
    "pymatgen>=2024.1.1",
    "scikit-learn>=1.9.0",
    "torch>=2.12.1",
]
```

- [ ] **Step 2 : Installer la dépendance**

```bash
uv sync
```

Expected : résolution et installation de pymatgen sans erreur.

- [ ] **Step 3 : Vérifier l'import**

```bash
python -c "from pymatgen.core.structure import Structure; print('ok')"
```

Expected : `ok`

- [ ] **Step 4 : Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add pymatgen dependency for crystal matrices"
```

---

## Task 2 : `compute_lattice_6d` — décomposition polaire SVD

**Files:**
- Create: `JEPA/crystal_matrices.py`
- Create: `tests/test_crystal_matrices.py`

**Interfaces:**
- Consumes: rien
- Produces: `compute_lattice_6d(lattice_matrix: torch.Tensor, num_atoms: int) -> torch.Tensor` shape `(6,)`

- [ ] **Step 1 : Créer le fichier de test avec le test failing**

Créer `tests/test_crystal_matrices.py` :

```python
import torch
import pytest
from JEPA.crystal_matrices import compute_lattice_6d


def test_compute_lattice_6d_shape():
    L = torch.eye(3, dtype=torch.float32)
    out = compute_lattice_6d(L, num_atoms=4)
    assert out.shape == (6,)


def test_compute_lattice_6d_identity():
    # Identity lattice → symmetric matrix is identity → upper tri [1,0,0,1,0,1] / 4^(1/3)
    L = torch.eye(3, dtype=torch.float32)
    out = compute_lattice_6d(L, num_atoms=4)
    scale = 4 ** (1 / 3)
    expected = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]) / scale
    assert torch.allclose(out, expected, atol=1e-5)


def test_compute_lattice_6d_normalization():
    L = torch.eye(3, dtype=torch.float32)
    out8 = compute_lattice_6d(L, num_atoms=8)
    out1 = compute_lattice_6d(L, num_atoms=1)
    # More atoms → smaller values (divided by larger N^(1/3))
    assert (out8 < out1).all()
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
cd /Users/baptistecaillerie/Documents/SUPRA-JEPA && python -m pytest tests/test_crystal_matrices.py::test_compute_lattice_6d_shape -v
```

Expected : `ImportError: cannot import name 'compute_lattice_6d'`

- [ ] **Step 3 : Créer `JEPA/crystal_matrices.py` avec l'implémentation**

```python
import torch
import torch.nn.functional as F


def compute_lattice_6d(lattice_matrix: torch.Tensor, num_atoms: int) -> torch.Tensor:
    W, S, Vt = torch.linalg.svd(lattice_matrix)
    V = Vt.T
    U = W @ Vt
    P = V @ torch.diag(S ** 2) @ V.T
    P_prime = U @ P @ U.T
    P_prime[torch.abs(P_prime) < 1e-5] = 0.0
    tri = torch.triu_indices(3, 3)
    lattice_6d = P_prime[tri[0], tri[1]]
    return lattice_6d / (num_atoms ** (1 / 3))
```

- [ ] **Step 4 : Vérifier que les tests passent**

```bash
cd /Users/baptistecaillerie/Documents/SUPRA-JEPA && python -m pytest tests/test_crystal_matrices.py::test_compute_lattice_6d_shape tests/test_crystal_matrices.py::test_compute_lattice_6d_identity tests/test_crystal_matrices.py::test_compute_lattice_6d_normalization -v
```

Expected : 3 PASSED

- [ ] **Step 5 : Commit**

```bash
git add JEPA/crystal_matrices.py tests/test_crystal_matrices.py
git commit -m "feat: add compute_lattice_6d with polar SVD decomposition"
```

---

## Task 3 : `build_atom_feature_matrix` — matrice V ∈ ℝ^(N×109)

**Files:**
- Modify: `JEPA/crystal_matrices.py`
- Modify: `tests/test_crystal_matrices.py`

**Interfaces:**
- Consumes: `compute_lattice_6d` de la Task 2
- Produces: `build_atom_feature_matrix(frac_coords: torch.Tensor, atomic_numbers: torch.Tensor, lattice_6d: torch.Tensor) -> torch.Tensor` shape `(N, 109)`

- [ ] **Step 1 : Ajouter les tests failing**

Ajouter dans `tests/test_crystal_matrices.py` :

```python
from JEPA.crystal_matrices import build_atom_feature_matrix


def test_build_atom_feature_matrix_shape():
    N = 5
    frac_coords = torch.rand(N, 3)
    atomic_numbers = torch.tensor([1, 6, 8, 14, 26], dtype=torch.long)
    lattice_6d = torch.rand(6)
    out = build_atom_feature_matrix(frac_coords, atomic_numbers, lattice_6d)
    assert out.shape == (N, 109)


def test_build_atom_feature_matrix_one_hot():
    # Atom with atomic_number=6 (Carbon) should have one-hot[5]=1, rest=0
    frac_coords = torch.rand(1, 3)
    atomic_numbers = torch.tensor([6], dtype=torch.long)
    lattice_6d = torch.zeros(6)
    out = build_atom_feature_matrix(frac_coords, atomic_numbers, lattice_6d)
    one_hot_part = out[0, 3:103]  # indices 3..102 are the one-hot part
    assert one_hot_part[5].item() == 1.0
    assert one_hot_part.sum().item() == 1.0


def test_build_atom_feature_matrix_lattice_repeated():
    N = 3
    frac_coords = torch.rand(N, 3)
    atomic_numbers = torch.tensor([1, 2, 3], dtype=torch.long)
    lattice_6d = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    out = build_atom_feature_matrix(frac_coords, atomic_numbers, lattice_6d)
    # Last 6 values of each row should equal lattice_6d
    for i in range(N):
        assert torch.allclose(out[i, 103:], lattice_6d)


def test_build_atom_feature_matrix_frac_coords():
    N = 2
    frac_coords = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    atomic_numbers = torch.tensor([1, 2], dtype=torch.long)
    lattice_6d = torch.zeros(6)
    out = build_atom_feature_matrix(frac_coords, atomic_numbers, lattice_6d)
    # First 3 values of each row should equal frac_coords
    assert torch.allclose(out[:, :3], frac_coords)
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
cd /Users/baptistecaillerie/Documents/SUPRA-JEPA && python -m pytest tests/test_crystal_matrices.py::test_build_atom_feature_matrix_shape -v
```

Expected : `ImportError: cannot import name 'build_atom_feature_matrix'`

- [ ] **Step 3 : Implémenter `build_atom_feature_matrix`**

Ajouter dans `JEPA/crystal_matrices.py` :

```python
def build_atom_feature_matrix(
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    lattice_6d: torch.Tensor,
) -> torch.Tensor:
    N = frac_coords.shape[0]
    # atomic_numbers are 1-indexed (H=1..100), shift to 0-indexed for one_hot
    one_hot = F.one_hot((atomic_numbers - 1).long(), num_classes=100).float()
    lattice_expanded = lattice_6d.unsqueeze(0).expand(N, -1)
    return torch.cat([frac_coords.float(), one_hot, lattice_expanded], dim=-1)
```

- [ ] **Step 4 : Vérifier que les tests passent**

```bash
cd /Users/baptistecaillerie/Documents/SUPRA-JEPA && python -m pytest tests/test_crystal_matrices.py -k "atom_feature" -v
```

Expected : 4 PASSED

- [ ] **Step 5 : Commit**

```bash
git add JEPA/crystal_matrices.py tests/test_crystal_matrices.py
git commit -m "feat: add build_atom_feature_matrix (N x 109)"
```

---

## Task 4 : `structure_to_tensors` — helper pymatgen → tenseurs

**Files:**
- Modify: `JEPA/crystal_matrices.py`
- Modify: `tests/test_crystal_matrices.py`

**Interfaces:**
- Consumes: `compute_lattice_6d` de la Task 2
- Produces: `structure_to_tensors(structure: Structure) -> tuple[Tensor, Tensor, Tensor, int]`
  - `frac_coords: Tensor (N, 3)`
  - `atomic_numbers: Tensor (N,)` dtype long
  - `lattice_6d: Tensor (6,)`
  - `num_atoms: int`

- [ ] **Step 1 : Ajouter les tests failing**

Ajouter dans `tests/test_crystal_matrices.py` :

```python
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from JEPA.crystal_matrices import structure_to_tensors


def _make_test_structure():
    lattice = Lattice.cubic(4.0)
    species = ["Na", "Cl"]
    coords = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    return Structure(lattice, species, coords)


def test_structure_to_tensors_shapes():
    structure = _make_test_structure()
    frac_coords, atomic_numbers, lattice_6d, num_atoms = structure_to_tensors(structure)
    assert frac_coords.shape == (2, 3)
    assert atomic_numbers.shape == (2,)
    assert lattice_6d.shape == (6,)
    assert num_atoms == 2


def test_structure_to_tensors_frac_coords_range():
    structure = _make_test_structure()
    frac_coords, _, _, _ = structure_to_tensors(structure)
    assert (frac_coords >= 0.0).all() and (frac_coords <= 1.0).all()


def test_structure_to_tensors_atomic_numbers():
    structure = _make_test_structure()
    _, atomic_numbers, _, _ = structure_to_tensors(structure)
    # Na=11, Cl=17 — order may vary due to sorting, but both must be present
    nums = set(atomic_numbers.tolist())
    assert nums == {11, 17}


def test_structure_to_tensors_lattice_6d_shape():
    structure = _make_test_structure()
    _, _, lattice_6d, _ = structure_to_tensors(structure)
    assert lattice_6d.shape == (6,)
    assert lattice_6d.dtype == torch.float32
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
cd /Users/baptistecaillerie/Documents/SUPRA-JEPA && python -m pytest tests/test_crystal_matrices.py::test_structure_to_tensors_shapes -v
```

Expected : `ImportError: cannot import name 'structure_to_tensors'`

- [ ] **Step 3 : Implémenter `structure_to_tensors`**

Ajouter dans `JEPA/crystal_matrices.py` (après les imports existants, ajouter l'import pymatgen en tête de fichier) :

En tête du fichier, ajouter :
```python
from pymatgen.core.structure import Structure
```

Puis ajouter la fonction :

```python
def structure_to_tensors(
    structure: Structure,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float32)
    atomic_numbers = torch.tensor(structure.atomic_numbers, dtype=torch.long)
    num_atoms = len(atomic_numbers)

    # Sort atoms deterministically by (atomic_number, x, y, z)
    sort_key = (
        1000 * atomic_numbers.float()
        + 100 * frac_coords[:, 0]
        + 10 * frac_coords[:, 1]
        + frac_coords[:, 2]
    )
    order = torch.argsort(sort_key)
    frac_coords = frac_coords[order]
    atomic_numbers = atomic_numbers[order]

    lattice_matrix = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
    lattice_6d = compute_lattice_6d(lattice_matrix, num_atoms)

    return frac_coords, atomic_numbers, lattice_6d, num_atoms
```

- [ ] **Step 4 : Vérifier que les tests passent**

```bash
cd /Users/baptistecaillerie/Documents/SUPRA-JEPA && python -m pytest tests/test_crystal_matrices.py -k "structure_to_tensors" -v
```

Expected : 4 PASSED

- [ ] **Step 5 : Commit**

```bash
git add JEPA/crystal_matrices.py tests/test_crystal_matrices.py
git commit -m "feat: add structure_to_tensors helper for pymatgen Structure"
```

---

## Task 5 : `build_energy_weight_matrix` — matrice Ω ∈ ℝ^(B×B)

**Files:**
- Modify: `JEPA/crystal_matrices.py`
- Modify: `tests/test_crystal_matrices.py`

**Interfaces:**
- Consumes: rien
- Produces: `build_energy_weight_matrix(ef_per_atom: torch.Tensor) -> torch.Tensor` shape `(B, B)`

- [ ] **Step 1 : Ajouter les tests failing**

Ajouter dans `tests/test_crystal_matrices.py` :

```python
from JEPA.crystal_matrices import build_energy_weight_matrix


def test_build_energy_weight_matrix_shape():
    ef = torch.tensor([-1.0, -2.0, -0.5])
    out = build_energy_weight_matrix(ef)
    assert out.shape == (3, 3)


def test_build_energy_weight_matrix_diagonal_ones():
    ef = torch.tensor([-1.0, -2.0, -0.5])
    out = build_energy_weight_matrix(ef)
    assert torch.allclose(out.diagonal(), torch.ones(3))


def test_build_energy_weight_matrix_off_diagonal_range():
    ef = torch.tensor([-1.0, -2.0, -0.5])
    out = build_energy_weight_matrix(ef)
    diag_mask = ~torch.eye(3, dtype=torch.bool)
    off_diag = out[diag_mask]
    # Values must be in [0, 1)
    assert (off_diag >= 0.0).all() and (off_diag < 1.0).all()


def test_build_energy_weight_matrix_symmetry():
    ef = torch.tensor([-1.0, -2.0, -0.5])
    out = build_energy_weight_matrix(ef)
    assert torch.allclose(out, out.T)


def test_build_energy_weight_matrix_values():
    # Two crystals with energies 0 and 1: ω = 1 - exp(-1) ≈ 0.6321
    ef = torch.tensor([0.0, 1.0])
    out = build_energy_weight_matrix(ef)
    expected_off = 1.0 - torch.exp(torch.tensor(-1.0))
    assert torch.allclose(out[0, 1], expected_off, atol=1e-5)
    assert torch.allclose(out[1, 0], expected_off, atol=1e-5)
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
cd /Users/baptistecaillerie/Documents/SUPRA-JEPA && python -m pytest tests/test_crystal_matrices.py::test_build_energy_weight_matrix_shape -v
```

Expected : `ImportError: cannot import name 'build_energy_weight_matrix'`

- [ ] **Step 3 : Implémenter `build_energy_weight_matrix`**

Ajouter dans `JEPA/crystal_matrices.py` :

```python
def build_energy_weight_matrix(ef_per_atom: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(ef_per_atom.unsqueeze(0) - ef_per_atom.unsqueeze(1))
    weight = 1.0 - torch.exp(-diff)
    diag = torch.eye(len(ef_per_atom), dtype=torch.float32, device=ef_per_atom.device)
    return weight + diag
```

- [ ] **Step 4 : Vérifier que les tests passent**

```bash
cd /Users/baptistecaillerie/Documents/SUPRA-JEPA && python -m pytest tests/test_crystal_matrices.py -k "energy_weight" -v
```

Expected : 5 PASSED

- [ ] **Step 5 : Commit**

```bash
git add JEPA/crystal_matrices.py tests/test_crystal_matrices.py
git commit -m "feat: add build_energy_weight_matrix (B x B InfoNCE weights)"
```

---

## Task 6 : Exporter depuis `JEPA/__init__.py` + test de smoke complet

**Files:**
- Modify: `JEPA/__init__.py`
- Modify: `tests/test_crystal_matrices.py`

**Interfaces:**
- Consumes: toutes les fonctions des Tasks 2–5
- Produces: fonctions accessibles via `from JEPA import compute_lattice_6d, ...`

- [ ] **Step 1 : Ajouter le test d'import public**

Ajouter dans `tests/test_crystal_matrices.py` :

```python
def test_public_api_imports():
    from JEPA import (
        build_atom_feature_matrix,
        build_energy_weight_matrix,
        compute_lattice_6d,
        structure_to_tensors,
    )
    assert callable(compute_lattice_6d)
    assert callable(build_atom_feature_matrix)
    assert callable(structure_to_tensors)
    assert callable(build_energy_weight_matrix)


def test_end_to_end_pipeline():
    from pymatgen.core.structure import Structure
    from pymatgen.core.lattice import Lattice
    from JEPA import build_atom_feature_matrix, structure_to_tensors

    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

    frac_coords, atomic_numbers, lattice_6d, num_atoms = structure_to_tensors(structure)
    V = build_atom_feature_matrix(frac_coords, atomic_numbers, lattice_6d)

    assert V.shape == (num_atoms, 109)
    assert V.dtype == torch.float32
```

- [ ] **Step 2 : Vérifier que le test d'import échoue**

```bash
cd /Users/baptistecaillerie/Documents/SUPRA-JEPA && python -m pytest tests/test_crystal_matrices.py::test_public_api_imports -v
```

Expected : `ImportError`

- [ ] **Step 3 : Modifier `JEPA/__init__.py`**

Ajouter dans `JEPA/__init__.py`, dans les imports et dans `__all__` :

Après les imports existants, ajouter :
```python
from .crystal_matrices import (
    build_atom_feature_matrix,
    build_energy_weight_matrix,
    compute_lattice_6d,
    structure_to_tensors,
)
```

Dans `__all__`, ajouter :
```python
    "build_atom_feature_matrix",
    "build_energy_weight_matrix",
    "compute_lattice_6d",
    "structure_to_tensors",
```

- [ ] **Step 4 : Vérifier que tous les tests passent**

```bash
cd /Users/baptistecaillerie/Documents/SUPRA-JEPA && python -m pytest tests/test_crystal_matrices.py -v
```

Expected : tous PASSED (environ 20 tests)

- [ ] **Step 5 : Commit final**

```bash
git add JEPA/__init__.py tests/test_crystal_matrices.py
git commit -m "feat: export crystal matrices from public JEPA API"
```

---

## Self-Review

**Spec coverage :**
- ✅ `compute_lattice_6d` — polar SVD → 6D upper tri → normalisé → Task 2
- ✅ `build_atom_feature_matrix` → V ∈ ℝ^(N×109) — Task 3
- ✅ `structure_to_tensors` depuis pymatgen.Structure — Task 4
- ✅ `build_energy_weight_matrix` → Ω ∈ ℝ^(B×B) — Task 5
- ✅ Export public `JEPA/__init__.py` — Task 6
- ✅ Dépendance pymatgen — Task 1

**Placeholder scan :** aucun TBD, TODO ni "similar to Task N" — chaque étape a son code complet.

**Type consistency :**
- `compute_lattice_6d(lattice_matrix: Tensor, num_atoms: int) -> Tensor (6,)` — cohérent Tasks 2 et 4
- `build_atom_feature_matrix(frac_coords, atomic_numbers, lattice_6d) -> Tensor (N,109)` — cohérent Tasks 3 et 6
- `structure_to_tensors(structure) -> (Tensor(N,3), Tensor(N,), Tensor(6,), int)` — cohérent Tasks 4 et 6
- `build_energy_weight_matrix(ef_per_atom: Tensor) -> Tensor (B,B)` — cohérent Tasks 5 et 6
