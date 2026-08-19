# DeSide PyTorch and Lightning migration implementation plan

## Summary

This plan executes the approved migration from TensorFlow/Keras to PyTorch and
PyTorch Lightning on branch `pytorch-dev`. The work is staged so DeSide remains
usable as an independent package while we align its training, checkpoint, and
prediction interfaces with VAEDecon conventions.

## Success criteria

- DeSide no longer depends on TensorFlow or Keras for model training or
  inference.
- `deside.DeSide` still exposes the main public training and prediction API.
- DeSide models save and load from PyTorch Lightning checkpoints.
- Existing preprocessing semantics remain intact.
- The DeSide package installs and runs in the same environment as VAEDecon.
- Documentation explains the major upgrade and how to use the legacy
  TensorFlow-based release.

## Phase 1: framework scaffold

### Code changes

1. Add `torch` and `lightning` dependencies to DeSide packaging and docs.
2. Create:
   - `deside/models/`
   - `deside/trainers/`
   - `deside/data/`
3. Implement the DeSide PyTorch MLP and optional pathway branch.
4. Implement a LightningModule for loss, metrics, optimizer, and checkpointing.
5. Add a lightweight Dataset class for DeSide training and prediction.

### Verification

- Import the new modules successfully.
- Instantiate the model with and without pathway input.
- Run one forward pass on synthetic tensors.

## Phase 2: public API migration

### Code changes

1. Rewrite `deside/decon_cf/deside.py` to remove TensorFlow/Keras usage.
2. Keep the `DeSide` class and public helper functions intact where practical.
3. Replace `.h5` loading and saving with `.ckpt` plus JSON metadata.
4. Preserve current preprocessing, gene alignment, and pathway-profile logic.
5. Preserve current prediction post-processing rules.

### Verification

- Train from `DeSide.train_model`.
- Save a checkpoint and reload it with `DeSide.get_model`.
- Predict from a saved model using `DeSide.predict`.

## Phase 3: regression and compatibility

### Code changes

1. Add tests for:
   - custom loss
   - forward pass shapes
   - checkpoint round-trip
   - inference post-processing
2. Add an integration test that imports DeSide in the VAEDecon environment.
3. Keep model artifacts and tensor layout compatible with later VAEDecon
   adapter work.

### Verification

- Run targeted pytest coverage for the migrated components.
- Confirm DeSide imports in the shared workspace without framework conflicts.

## Phase 4: documentation and release notes

### Code changes

1. Update `README.md`.
2. Update installation and usage docs.
3. Update changelog and explicitly mark this as a major upgrade.
4. Add legacy TensorFlow installation instructions and checkpoint migration
   notes.

### Verification

- Read through docs for consistency with the code.
- Confirm dependency declarations no longer require TensorFlow.

## Execution order for this branch

1. Framework scaffold
2. DeSide facade migration
3. Tests
4. Docs and dependency cleanup

## Current execution focus

Start with Phases 1 and 2 in the smallest useful vertical slice:

1. create the PyTorch model and Lightning trainer
2. wire `DeSide.train_model` and `DeSide.predict`
3. save and load `.ckpt` artifacts
4. verify with focused tests
