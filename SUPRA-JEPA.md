# SUPRA-JEPA

Modèle de classification binaire **supraconducteur / non-supraconducteur** basé sur une architecture JEPA (Joint Embedding Predictive Architecture) appliquée aux structures cristallines tokenisées.

---

## Objectif

Prédire si un matériau est supraconducteur à partir de sa seule structure cristalline (pas de calculs DFT, pas de propriétés physiques calculées en amont). Sortie : probabilité ∈ [0, 1], seuil à 0.5.

---

## Motivation du choix JEPA

Les approches classiques supervisées souffrent du manque de données labellisées (~16 000 SC connus). Le prétraining auto-supervisé sur des millions de structures non labellisées permet d'apprendre des représentations riches avant de fine-tuner sur la tâche cible.

JEPA vs MAE (Masked AutoEncoder) :
- MAE reconstruit en espace pixel → apprend les détails atomiques bas-niveau (positions exactes, densités)
- JEPA prédit en **espace latent** → biais implicite vers les features sémantiques (symétrie, motifs structuraux, environnements chimiques) — directement pertinents pour la supraconductivité

JEPA vs contrastif (CrystalCLR, SimCLR) :
- Pas de paires négatives → pas de besoin d'augmentations manuelles difficiles à définir sur des cristaux
- Stabilité assurée par EMA du target encoder (comme Barlow Twins dans l'esprit)

---

## Représentation d'entrée

Chaque cristal est une **séquence de tokens** de longueur fixe `max_atoms + 1`.

### Token CLS (position 0) — informations globales
- Embedding du groupe d'espace (1–230)
- Projection linéaire des paramètres de maille (a, b, c, α, β, γ)

### Tokens atomiques (positions 1…N) — un token par site
```
token_i = embed(element_Z) + embed(wyckoff_letter) + proj(frac_coords)
```
- `element_Z` : numéro atomique Z ∈ [1, 118]
- `wyckoff_letter` : lettre de position de Wyckoff (a→z)
- `frac_coords` : coordonnées fractionnelles (x, y, z) ∈ [0, 1]³

Les cristaux avec > `max_atoms` sites sont tronqués. Les plus courts sont paddés (mask d'attention).

---

## Architecture

```
Structure cristalline
        │
        ▼
 CrystalEmbedding         embed(Z) + embed(wyck) + proj(coords)
        │
        ├──────────────────────────────────────────┐
        │  [tokens visibles + mask_tokens]         │  [tous les tokens]
        ▼                                          ▼
 CrystalEncoder (θ)                    CrystalEncoder (θ̄, EMA)
  6 couches Transformer                 stop-gradient
  pre-norm, GELU                              │
        │                                    │ z_target (normalisé)
        ▼                                    │
 JEPAPredictor (φ)                           │
  2 couches Transformer                      │
        │                                    │
        ▼                                    ▼
     z_pred  ────── MSE loss ──────  z_target[positions masquées]

                    PRÉTRAINING

        │ (fine-tuning : token CLS uniquement)
        ▼
 SupraClassifier
  Linear → GELU → Dropout → Linear(1)
        │
        ▼
   logit → σ → P(supraconducteur)
```

### Hyperparamètres principaux

| Paramètre | Valeur |
|---|---|
| `d_model` | 256 |
| `n_heads` | 8 |
| `n_layers_encoder` | 6 |
| `n_layers_predictor` | 2 |
| `mask_ratio` | 40% |
| `ema_decay` | 0.996 → 1.0 (schedule) |
| `max_atoms` | 64 |

---

## Données

### Positifs — supraconducteurs
- **SuperCon database** (via matminer) : ~16 000 matériaux avec Tc > 0 K
- Structures cristallines récupérées depuis le **Materials Project** via `mp-api`

### Négatifs — non-supraconducteurs
- Structures stables du Materials Project (E_above_hull < 0.05 eV/atom) sans Tc connu
- ~15 000 échantillons → ratio ~1:1 pour le fine-tuning (ou pos_weight pour garder le déséquilibre réel)

### Split
- Train / Val / Test : 70 / 15 / 15 %
- Stratifié sur le label

---

## Entraînement

### Phase 1 — Prétraining JEPA (non-supervisé)

- Toutes les structures disponibles (labellisées ou non)
- 40% des tokens atomiques masqués aléatoirement à chaque batch
- Loss : MSE dans l'espace latent normalisé sur les positions masquées uniquement
- LR warmup linéaire → décroissance cosinus
- EMA decay augmente progressivement (schedule cosinus) pour éviter le collapse

### Phase 2 — Fine-tuning classification

- Charge les poids du context encoder prétrainé
- Entraînement full (encoder + tête) avec `lr_finetune` = 2e-5
- Loss : `BCEWithLogitsLoss` avec `pos_weight = n_neg / n_pos` (~37.5)
- Pas de masquage (le token CLS voit toute la structure)

---

## Métriques d'évaluation

Le déséquilibre extrême (supraconducteurs rares) impose des métriques adaptées :

| Métrique | Pourquoi |
|---|---|
| **AUROC** | Robuste au déséquilibre, seuil-indépendant |
| **AP (Average Precision)** | Résumé de la courbe précision/rappel |
| **TNR** (true-negative-rate) | Éviter les faux positifs coûteux (screening) |
| **TPR / Recall** | Ne pas manquer les SC réels |
| **F1** | Compromis précision/rappel |

Cible inspirée de **BEE-NET** (NeurIPS ML4PS 2024) : TNR ≥ 99%, TPR ≥ 80%.

---

## Références

- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence.* — JEPA original
- Assran et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* (I-JEPA) — [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
- Skenderi et al. (2025). *Graph-level Representation Learning with Joint-Embedding Predictive Architectures.* (Graph-JEPA) — [arXiv:2309.16014](https://arxiv.org/abs/2309.16014)
- Liu et al. (2023). *S2SNet: A Pretrained Neural Network for Superconductivity Discovery.* — [arXiv:2306.16270](https://arxiv.org/abs/2306.16270)
- Littwin et al. (2024). *How JEPA Avoids Noisy Features.* — [arXiv:2407.03475](https://arxiv.org/abs/2407.03475)
- Taniai et al. (2024). *Crystalformer.* — [arXiv:2403.11686](https://arxiv.org/abs/2403.11686)
- NeurIPS ML4PS 2024. *BEE-NET: Deep Learning for Superconductivity Prediction.*
