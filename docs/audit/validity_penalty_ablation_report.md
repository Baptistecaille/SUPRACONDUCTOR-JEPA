# Ablation : pénalité de validité Allen-Dynes (`validity_weight`)

## Question

Pourquoi contraindre `physics_jepa` avec une pénalité de validité Allen-Dynes plutôt que de
laisser les deux têtes (`lambda_ep`, `omega_log`) apprendre librement par simple MSE ?

## Ce qui est réellement contraint

`PhysicsHeadsLoss` n'impose PAS la formule d'Allen-Dynes comme cible d'apprentissage — le
signal principal reste une MSE en espace normalisé sur `lambda_ep` et `omega_log`
(poids 1.0, 100% piloté par les données). La seule composante physique est une pénalité
softplus douce (poids 0.05 par défaut, PAS un clamp dur) sur la quantité
`lambda - mu*(1+0.62*lambda)`, qui doit rester strictement positive pour que Tc soit
physiquement défini sous la formule d'Allen-Dynes.

## Motivation empirique

3.4% des lignes de `physics_jepa_train.csv.gz` (254 / 7569) ont un `lambda_ep` tel que
cette quantité soit déjà <= 0.01 dans les données elles-mêmes -- la zone où Tc diverge
ou devient négatif n'est pas une préoccupation purement théorique sur ce dataset.

## Protocole d'ablation

Deux runs de fine-tuning identiques (40 époques, `physics_jepa_{train,val}.csv.gz`
complets, `omega_weight=1.0`, `lr=1e-3`, `encoder_lr=1e-4`, cosine schedule), seule
différence : `validity_weight=0.05` (actuel) vs `validity_weight=0.0` (JEPA libre).

## Résultats (test set, 951 structures)

| métrique | validity_weight=0.05 | validity_weight=0.0 |
|---|---|---|
| MAE lambda_ep | 0.204 | 0.197 |
| MAE omega_log (K) | 36.5 | 35.8 |
| MAE Tc, formule bare (K) | 1.69 | 1.67 |
| MAE Tc, formule corrigée f1 (K) | 1.78 | 1.76 |
| Prédictions test dans la zone Tc-indéfinie | 0 / 951 | 0 / 951 |
| Marge minimale au seuil de validité (doit être > 0.01) | 0.056 | 0.022 |

## Interprétation

Sur ce jeu de test borné, retirer la pénalité **n'a pas** fait franchir la frontière
physique à une seule prédiction, et améliore même très légèrement les métriques de
régression brutes (la pénalité ne les aide pas). Mais la prédiction la plus proche de
la frontière s'en approche 2.5x plus sans la pénalité (0.022 vs 0.056) -- un signal
directionnel, pas une preuve, qu'aucune protection structurelle n'existe une fois la
pénalité retirée : les deux têtes sont des régressions scalaires indépendantes sans lien
intégré vers la contrainte physique en aval (division par cette quantité dans le calcul
de Tc). Sur un run plus long, un modèle plus grand, ou en extrapolation sur des
matériaux hors distribution, rien ne garantit que cette marge ne se referme pas.

## Recommandation

Garder une pénalité de poids faible (0.05) : son coût est négligeable (elle ne dégrade
pas mesurablement les MAE et n'agit pas comme un clamp dur qui bloquerait le gradient),
et elle sert de garde-fou pour la tête de dérivation de Tc en aval, plutôt que d'un
véritable a priori physique fort sur ce que le JEPA doit apprendre.
