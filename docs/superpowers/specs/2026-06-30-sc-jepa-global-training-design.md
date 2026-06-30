# Design: SC-JEPA Global Embedding Training

Date: 2026-06-30

## Objective

Build the first trainable SC-JEPA pre-training pipeline for superconducting crystals. The model learns a latent representation of superconductors by predicting the complete-crystal embedding from a spatially corrupted crystal view.

This first version predicts a single global embedding per crystal, not atom-level embeddings. The goal is to make the JEPA training loop reliable before adding harder token-level reconstruction objectives.

The model hidden dimension is fixed to `512`, matching the public Crys-JEPA MP configuration.

## Dataset Scope

Training uses only superconducting crystals. The pre-training objective is not a binary classifier. It learns the geometry, symmetry, and chemistry distribution of known superconducting materials so the learned embedding can later support screening or downstream classification.

The existing dataset path and collate flow remain the base:

- `SuperconductorDataset` parses CIF structures.
- `collate_crystals` creates masked Encoder 1 inputs and full Encoder 2 inputs.
- Encoder 1 receives padded corrupted atom tokens.
- Encoder 2 receives padded complete atom tokens plus crystal-family conditioning.

## Crystal Representation

Each crystal is represented as a variable-length sequence of atom tokens.

Encoder 1 receives corrupted crystal tokens with 103 features per atom:

- fractional coordinates: 3
- atomic-number one-hot: 100

Encoder 2 receives complete crystal tokens with 110 features per atom:

- fractional coordinates: 3
- atomic-number one-hot: 100
- crystal-family one-hot: 7

Continuous lattice parameters are not used. The seven crystal families replace the continuous lattice matrix to reduce leakage through exact cell dimensions.

## Spatial Masking

Masking is applied only to Encoder 1 inputs. A toric box is sampled in fractional coordinate space, and atoms inside that box are removed from the corrupted sequence.

The collate step must guarantee:

- at least one visible atom remains for Encoder 1;
- at least one atom is masked whenever `num_atoms >= 2`;
- per-atom tensors are filtered for Encoder 1;
- per-crystal family information is kept only in Encoder 2.

The predictor receives the sampled mask box as `[center_x, center_y, center_z, box_size]`.

## Architecture

The implementation adds crystal-specific modules without replacing the existing temporal JEPA classes.

### CrystalTransformerEncoder

`CrystalTransformerEncoder` encodes padded atom-token sequences into one global crystal embedding.

Inputs:

- `atom_features: Tensor[B, N, F]`
- `atom_mask: BoolTensor[B, N]`, where `True` means a real atom token

Output:

- `cls_embedding: Tensor[B, 512]`

The encoder uses:

- an input projection to `512`;
- a learned CLS token prepended to the atom sequence;
- `nn.TransformerEncoder`;
- a key-padding mask derived from `atom_mask`;
- the final CLS token as the crystal embedding.

### Context Encoder

The context encoder receives Encoder 1 corrupted features.

Components:

- `atom_proj_103`: trainable by gradient;
- `cls_token`: trainable by gradient;
- `transformer`: trainable by gradient.

### Target Encoder EMA

The target encoder receives Encoder 2 complete features. It is not updated by backpropagation. Compatible parameters are updated by exponential moving average from the context encoder.

Components:

- `atom_proj_103`: EMA copy of context `atom_proj_103`;
- `family_proj_7`: fixed after initialization;
- `cls_token`: EMA copy of context `cls_token`;
- `transformer`: EMA copy of context `transformer`.

Because Encoder 1 has 103 input dimensions and Encoder 2 has 110, the target input projection is factorized:

```text
target_projection(token_110)
  = atom_proj_103(token_110[:103]) + family_proj_7(token_110[103:])
```

This preserves the asymmetric information design while allowing the atom projection and Transformer backbone to follow the context encoder through EMA.

## Predictor

`MaskConditionedPredictor` maps the corrupted-context embedding to the complete-target embedding.

Inputs:

- `z_context: Tensor[B, 512]`
- `mask_box: Tensor[B, 4]`

Output:

- `z_pred: Tensor[B, 512]`

The predictor has two branches:

- a mask MLP that projects the 4 mask-box scalars to `512`;
- a prediction MLP that receives `z_context + mask_embedding`.

The mask conditioning tells the predictor where the missing spatial block was located.

## Forward Pass

For each batch:

```text
z_context = context_encoder(
    encoder1_atom_features,
    encoder1_atom_mask,
)

with no_grad:
    z_target = target_encoder_ema(
        encoder2_atom_features,
        encoder2_atom_mask,
    )

z_pred = predictor(z_context, mask_box)

loss_pred = weighted_contrastive_loss(z_pred, z_target.detach(), ef_per_atom)
loss = loss_pred + reg_weight * loss_reg
```

The first smoke-training version sets `reg_weight = 0` by default and defines `loss_reg` as a zero tensor when no regularizer is configured. Anti-collapse regularization can then be enabled once the forward path, EMA update, and checkpointing are verified.

## Prediction Loss

The project uses the Crys-JEPA weighted contrastive objective instead of MSE.

Given:

- `context = z_pred: Tensor[B, 512]`
- `target = z_target.detach(): Tensor[B, 512]`
- `ef_per_atom: Tensor[B]`

Compute cosine similarities between every predicted context embedding and every target embedding:

```text
sim_ik = cosine(context_i, target_k)
```

Build the energy-difference weight matrix:

```text
W_ik = 1 - exp(-abs(ef_i - ef_k))  for i != k
W_ii = 1
```

Then apply the Crys-JEPA temperature-scaled batch objective:

```text
logits_ik = sim_ik * W_ik / temperature
prob_ik = exp(logits_ik) / sum_j exp(logits_ij)
loss = -mean_i log(prob_ii)
```

Default `temperature = 0.1`, matching the public Crys-JEPA implementation.

This keeps the predictor output in embedding space while making the training signal contrastive across the batch. Crystals with more different formation energies are repelled more strongly.

## EMA Update

After each optimizer step, compatible target parameters are updated as:

```text
target = ema_decay * target + (1 - ema_decay) * context
```

EMA applies to:

- `atom_proj_103`;
- `cls_token`;
- `transformer`.

EMA does not apply to:

- `family_proj_7`, because it has no context-side equivalent.

All target encoder parameters have `requires_grad = False`.

## Training Loop

Add a minimal training entrypoint for SC-JEPA pre-training.

Responsibilities:

- create dataloader from the superconducting crystal CSV;
- instantiate context encoder, target encoder, predictor, and SC-JEPA wrapper;
- move batch tensors to device;
- compute weighted contrastive prediction loss from `z_pred`, `z_target`, and `ef_per_atom`;
- run backward and optimizer step on context encoder plus predictor only;
- update EMA after optimizer step;
- log scalar losses;
- save checkpoints;
- support a `max_batches` argument for smoke runs.

The loop should remain small and explicit. Configuration can be simple function arguments or `argparse` flags.

## Error Handling

The implementation should fail early with clear messages when:

- atom feature dimensions are not 103 for Encoder 1 or 110 for Encoder 2;
- atom masks do not match token sequence lengths;
- a batch contains no visible context atoms;
- loss becomes NaN or infinite;
- batch size is less than 2 when using contrastive training;
- EMA update encounters incompatible parameter shapes outside the known projection asymmetry.

## Testing

Unit tests should cover:

- `CrystalTransformerEncoder` output shape `[B, 512]`;
- padding masks prevent padded atoms from changing the output materially;
- target encoder has no trainable gradients;
- EMA update changes compatible target parameters;
- `family_proj_7` is not overwritten by EMA;
- `MaskConditionedPredictor` output shape;
- weighted contrastive loss returns a finite scalar for a batch with `B >= 2`;
- full `CrystalJEPA.forward(batch)` returns finite scalar loss and metrics;
- a tiny training step produces gradients only for context encoder and predictor.

Smoke verification should run a short training command with `max_batches=2` and save a checkpoint.

## Out of Scope

This design does not include:

- atom-level or masked-token prediction;
- non-superconductor data;
- downstream binary classification;
- screening/ranking candidates;
- importing the full Crys-JEPA dependency stack;
- using continuous lattice parameters.

Those can be added after the global embedding pre-training loop is stable.

## Success Criteria

The implementation is ready when:

- a real dataloader batch runs through `CrystalJEPA.forward`;
- the returned loss is finite and scalar;
- `backward()` creates gradients for context encoder and predictor;
- target encoder parameters remain gradient-free;
- EMA updates compatible target weights;
- the weighted contrastive loss uses `ef_per_atom` and `temperature = 0.1`;
- a `max_batches=2` training run completes and writes a checkpoint.
