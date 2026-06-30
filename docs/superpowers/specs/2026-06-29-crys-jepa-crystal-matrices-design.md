# Design : Matrices cristallines Crys-JEPA

**Date :** 2026-06-29  
**Référence :** arxiv 2605.14759 — *Crys-JEPA: Accelerating Crystal Discovery via Embedding Screening and Generative Refinement*  
**Repo source :** https://github.com/liun-online/Crys_JEPA  

## Objectif

Implémenter les matrices d'entrée du Crys-JEPA dans `JEPA/crystal_matrices.py` — fichier autonome dans la bibliothèque JEPA existante. L'utilisateur branche ses propres encodeurs, losses et dataloader par-dessus.

## Matrices à implémenter

### 1. Lattice 6D — `compute_lattice_6d(lattice_matrix, num_atoms)`

La maille cristalline `L ∈ ℝ^(3×3)` est symétrisée via décomposition polaire SVD :

```
W, S, Vᵀ = SVD(L)
U = W @ Vᵀ
P = V @ diag(S²) @ Vᵀ
P' = U @ P @ Uᵀ    # matrice symétrique 3×3
→ upper_tri(P') / num_atoms^(1/3)   # 6 scalaires
```

- **Entrée :** `lattice_matrix: torch.Tensor (3, 3)`, `num_atoms: int`
- **Sortie :** `torch.Tensor (6,)`
- **Source :** `CrystalDataset.compute_lattice_polar_decomposition()` dans le repo Crys-JEPA

### 2. Matrice d'atomes — `build_atom_feature_matrix(frac_coords, atomic_numbers, lattice_6d)`

Vecteur d'atome tel que défini dans le papier (§ 3.1, Eq. 2) :

```
vᵢ = [ Xᵢ (3) ∥ one_hot(Aᵢ, 100) ∥ L̂ (6) ]  ∈ ℝ^109
```

- `Xᵢ` : coordonnées fractionnaires de l'atome i  
- `one_hot(Aᵢ, 100)` : numéro atomique encodé en one-hot (100 éléments du tableau périodique)  
- `L̂` : lattice 6D normalisée, répétée pour chaque atome  

- **Entrée :** `frac_coords (N, 3)`, `atomic_numbers (N,)`, `lattice_6d (6,)`
- **Sortie :** `V ∈ ℝ^(N, 109)` — la matrice d'entrée de l'encodeur Transformer

### 3. Fonction helper — `structure_to_tensors(structure)`

Convertit un `pymatgen.core.structure.Structure` en tenseurs bruts prêts pour les deux fonctions ci-dessus.

- **Entrée :** `pymatgen.Structure`
- **Sortie :** `frac_coords (N, 3)`, `atomic_numbers (N,)`, `lattice_6d (6,)`, `num_atoms: int`
- Trie les atomes par (numéro atomique, coordonnées) pour un ordre déterministe

### 4. Matrice de poids énergie — `build_energy_weight_matrix(ef_per_atom)`

Pondère la loss InfoNCE pour que des cristaux avec des énergies de formation très différentes soient plus repoussés dans l'espace latent (§ 3.2, Eq. 8) :

```
Ωᵢₖ = 1 - exp(-|Eᶠᵢ - Eᶠₖ|)   pour i ≠ k
Ωᵢᵢ = 1
```

- **Entrée :** `ef_per_atom: torch.Tensor (B,)` — énergie de formation par atome pour chaque cristal du batch
- **Sortie :** `Ω ∈ ℝ^(B, B)`
- Optionnel pour l'utilisateur (fourni pour compléter le pipeline)

## Structure du fichier

```
JEPA/crystal_matrices.py
├── compute_lattice_6d(lattice_matrix, num_atoms) -> Tensor (6,)
├── structure_to_tensors(structure) -> (Tensor, Tensor, Tensor, int)
├── build_atom_feature_matrix(frac_coords, atomic_numbers, lattice_6d) -> Tensor (N, 109)
└── build_energy_weight_matrix(ef_per_atom) -> Tensor (B, B)
```

## Ce qui n'est PAS inclus

- Encodeur Transformer / predictor MLP (dans `JEPA/architectures.py` ou côté utilisateur)
- Loss InfoNCE pondérée (côté utilisateur)
- Augmentations (translation/rotation SO3)
- Dataloader / batching

## Dépendances

- `torch` (déjà dans le projet)
- `pymatgen` (pour `Structure`)
- Aucune nouvelle dépendance PyTorch Geometric requise pour ce module seul

## Flux de données

```
pymatgen.Structure
    ↓  structure_to_tensors()
frac_coords (N,3)  atomic_numbers (N,)  lattice_6d (6,)
    ↓  build_atom_feature_matrix()
V ∈ ℝ^(N×109)
    → encodeur Transformer → cls_token H ∈ ℝ^d

ef_per_atom (B,)
    ↓  build_energy_weight_matrix()
Ω ∈ ℝ^(B×B)  →  loss InfoNCE pondérée
```
