# physics_jepa Fine-Tuning: Evaluation Report

## Summary

`physics_jepa` attaches two regression heads (`lambda_ep`, `omega_log`) on top of the
pretrained `foundation_jepa` encoder (`models/checkpoints/foundation_jepa.pt`) via a
shared MLP trunk, then derives `Tc` through the Allen-Dynes formula
(`mu_star=0.1`) with a soft (softplus) validity penalty for the physical constraint
`lambda - mu_star*(1+0.62*lambda) > eps`.

Two fine-tuning strategies were run to completion (40 epochs each, full
`physics_jepa_{train,val}.csv.gz` splits: 7569 train / 952 val rows, batch_size=32,
cosine LR schedule with warmup) on **compute-bounded local CPU hardware**:

1. **Joint fine-tuning** — encoder + heads both trainable (`lr=1e-3` heads,
   `encoder_lr=1e-4`).
2. **Linear probe** — `foundation_jepa` encoder frozen, only the heads trained
   (`lr=1e-3`).

Both used identical loss weights (`omega_weight=1.0`, `validity_weight=0.05`) and the
same `PhysicsTargetStats` log1p+z-score normalization fit on the train split.

## Held-out test-set metrics (`physics_jepa_test.csv.gz`, 951 structures)

| strategy | MAE lambda_ep | MAE omega_log (K) | MAE Tc, bare formula (K) | MAE Tc, f1-corrected (K) |
|---|---|---|---|---|
| Joint fine-tuning | 0.204 | 36.5 | 1.69 | 1.78 |
| Linear probe (frozen encoder) | 0.222 | 44.9 | 2.21 | 2.32 |

Joint fine-tuning outperforms the frozen-encoder linear probe on every metric,
consistent with the fact that `foundation_jepa`'s own pretraining here was itself only
a bounded local smoke run (900K params, 20 epochs, not the paper's GPU-cluster
config) — a not-fully-converged encoder benefits more from being allowed to keep
adapting during the physics fine-tuning stage than a fully pretrained one would.

### Allen-Dynes strong-coupling correction (`f1`, `f2`)

The bare Allen-Dynes interpolation used above is a McMillan-style formula that is only
accurate for weak-to-moderate coupling and is known to drift increasingly above
roughly `lambda ~ 1.5` — flagged during this evaluation because a non-trivial
fraction of `physics_jepa_train` rows have `lambda_ep > 1.5` (and some `> 3`).
`models/physics_jepa/model/physics_utils.py` now implements the two Allen & Dynes
(1975) correction factors, `f1` and `f2` (see the module docstring for the full
formulas), and `allen_dynes_tc(...)` applies `f1` by default
(`apply_strong_coupling_correction=True`) since it needs only `lambda_ep`/`mu_star`
and is always computable; `apply_strong_coupling_correction=False` reproduces the
original bare formula exactly for backward-compatible comparisons.

The "MAE Tc, f1-corrected" column above applies `f1` to both the predicted and the
ground-truth `(lambda_ep, omega_log)` before computing Tc and the MAE — it is *not*
directly comparable in absolute value to the bare-formula MAE column since both sides
of the comparison shift, but it is the number to track once the strong-coupling
correction becomes the standard formula. It is marginally higher than the bare-formula
MAE here, which is expected: `f1` amplifies absolute Tc values roughly proportionally
to `lambda`, so a fixed relative Tc-prediction error translates into a larger absolute
K error once `f1 > 1` is applied on both sides.

`f2` needs `omega2 = sqrt(<omega^2>)`, the second moment of the Eliashberg spectral
function `alpha^2F(omega)`, which the model does not predict (only ~14% of rows in
this dataset even have a stored raw spectrum to derive it from) — so `f2` is **not**
applied to model predictions in the table above. As an auxiliary, ground-truth-only
check, `omega2` was reconstructed from the raw `a2f_original_{x,y}` spectra on the 48
test rows with `has_a2f_spectrum=True`, keeping the 33 rows whose reconstructed
`lambda_check = 2*integral(a2F/w dw)` matched the stored `lambda_ep` within 30%. On
that subset, mean ground-truth Tc was ~4.4 K bare, ~4.7 K with `f1` alone, and ~2.0 K
with `f1*f2` — i.e. `f2` had a larger and opposite-signed effect than `f1` on these
particular materials (`omega2/omega_log` averaged ~0.21, well below the bare
formula's implicit assumption of 1.0). This is illustrative of `f2`'s potential
magnitude, not a validated correction, since it can only be checked on the small
spectrum-bearing subset. See `docs/audit/physics_jepa_test_eval_report.json` for the
full numbers (`f2_auxiliary_ground_truth_only` block).

## Training dynamics

The joint fine-tuning run's train loss keeps decreasing smoothly through 40 epochs
while validation loss plateaus/creeps up mildly after ~epoch 11 (see
`physics_jepa_training_curves.png`) — a mild overfitting signal on the *normalized
loss*. Despite that, validation MAE on Tc keeps trending down across the full 40
epochs, so the physically-relevant metric continues to improve even as the raw loss
shows early softening.

## Caveats (compute-bounded run)

- `foundation_jepa` itself is a bounded local CPU **smoke-pretraining** run (see
  `docs/audit/foundation_jepa_pretrain_report.json`), not the paper's GPU-cluster
  config (hidden_dim=512, 8 layers, batch 2048, 2000 epochs). All physics_jepa numbers
  above inherit that limitation.
- `mu_star=0.1` is a literature-standard uniform constant; the consolidated dataset
  does not carry a per-row mu_star.
- The `f2` strong-coupling correction is not applied to model predictions (no
  `omega2` prediction head exists); only `f1` is applied by default. See the
  "Allen-Dynes strong-coupling correction" section above.
- These runs validate the fine-tuning pipeline end-to-end (data loading, shared-trunk
  heads, physically-motivated loss, Allen-Dynes derivation, checkpointing) and give a
  directionally-correct comparison between fine-tuning strategies. They are **not**
  representative of the accuracy achievable once `foundation_jepa` is pretrained at
  the paper's GPU-cluster scale.

## Artifacts

- `models/checkpoints/physics_jepa.pt` — joint fine-tuning checkpoint (best strategy)
- `models/checkpoints/physics_jepa_frozen.pt` — linear-probe checkpoint
- `docs/audit/physics_jepa_finetune_report.json` — joint fine-tuning training report
- `docs/audit/physics_jepa_finetune_report_frozen.json` — linear-probe training report
- `physics_jepa_test_predictions.png` — test-set predicted-vs-true scatter (both strategies)
- `physics_jepa_training_curves.png` — training/validation loss and Tc MAE curves
