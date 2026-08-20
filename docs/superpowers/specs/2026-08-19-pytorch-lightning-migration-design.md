# DeSide PyTorch and Lightning migration design

## Summary

This design upgrades `DeSide` from TensorFlow/Keras to PyTorch and PyTorch
Lightning while keeping `DeSide` an independent Python package. The migration
keeps the existing public DeSide inference and training entry points as stable
as practical, replaces TensorFlow model training and checkpointing with
Lightning-based workflows, and aligns DeSide's internal conventions with
`VAEDecon` so that DeSide can later be integrated into the VAEDecon workflow as
the dedicated cell-proportion prediction component.

This is a major upgrade. The new implementation will not load arbitrary legacy
TensorFlow `.h5` checkpoints. Instead, it will define a new PyTorch checkpoint
format and document a legacy install path for users who need the older
TensorFlow release.

The recommended version-control strategy is to modify the existing DeSide
repository on a dedicated branch named `pytorch-dev`.

## Goals

- Replace TensorFlow and Keras usage in `DeSide` with PyTorch and PyTorch
  Lightning.
- Preserve DeSide model logic, preprocessing semantics, and user-facing API
  behavior wherever practical.
- Keep `DeSide` independently installable and usable outside `VAEDecon`.
- Align DeSide training, checkpoint, and data-loading patterns with the
  existing `VAEDecon` framework.
- Prepare for a later integration where `VAEDecon` can use DeSide as a
  cell-proportion prediction component.
- Update README, documentation, and dependency metadata for the new framework.
- Mark the change as a major upgrade in release notes and migration notes.

## Non-goals

- Supporting direct loading of arbitrary legacy TensorFlow `.h5` DeSide model
  files.
- Refactoring unrelated preprocessing, plotting, or statistics utilities that
  are already framework-agnostic.
- Coupling DeSide directly to VAEDecon internals during this upgrade.
- Rewriting DeSide into the full VAEDecon architecture.

## Current-state audit

### TensorFlow-dependent surface area

The TensorFlow dependency is concentrated in
`deside/decon_cf/deside.py`:

- `import tensorflow as tf`
- `from tensorflow import keras`
- Keras `Dense`, `Dropout`, `BatchNormalization`, `LayerNormalization`,
  `Activation`, `Input`, `Model`, and `concatenate`
- Keras optimizer `Adam`
- Keras losses and metrics
- Keras `EarlyStopping`
- `model.compile`, `model.fit`, `model.predict`, `model.save`
- `keras.models.load_model`
- custom loss `loss_fn_mae_rmse`

### Framework-agnostic areas

Most other DeSide modules remain framework-agnostic and can be preserved with
minimal change:

- input reading and preprocessing under `deside/utility`
- bulk and single-cell preprocessing helpers
- synthetic data generation
- result plotting and evaluation utilities
- documentation and example scripts that call DeSide's public API

### Packaging surface

TensorFlow is currently declared in:

- `pyproject.toml`
- `docs/requirements.txt`
- `README.md`

These must be replaced with PyTorch and Lightning dependencies, while docs must
also explain how to install the older TensorFlow-based line.

## High-level architecture

The migrated package keeps the current top-level public entry points, including:

- `deside.DeSide`
- `deside.predict_with_pretrained_model`
- existing preprocessing and plotting utilities

Internally, the implementation will be split into focused modules.

### 1. `deside/models/`

Purpose: define the migrated PyTorch network that reproduces the current DeSide
architecture.

Contents:

- `DeSideMLP` or similarly named `torch.nn.Module`
- a main GEP branch
- an optional pathway branch
- explicit support for:
  - hidden widths
  - dropout rates
  - batch normalization or layer normalization
  - `relu`
  - final output activation compatible with current DeSide options

Constraints:

- preserve current mathematical behavior as closely as possible
- accept tensors shaped `[batch, features]`
- keep pathway and non-pathway execution paths explicit

### 2. `deside/trainers/`

Purpose: wrap the PyTorch model in a Lightning training interface.

Contents:

- a LightningModule implementing:
  - `forward`
  - `training_step`
  - `validation_step`
  - `configure_optimizers`
- metric logging for loss, MAE, and RMSE
- checkpointing and early stopping behavior aligned with DeSide's current
  training intent and VAEDecon's conventions

Training semantics:

- custom loss remains `alpha * MAE + (1 - alpha) * RMSE`
- Adam optimizer remains the default
- validation-based early stopping remains available

### 3. `deside/data/`

Purpose: replace Keras fit-time numpy handling with explicit PyTorch Dataset and
DataLoader objects.

Contents:

- a dataset class for training samples
- deterministic train/validation splitting helpers
- collation compatible with DeSide's optional pathway input

Returned items:

- `gep`
- optional `pathway_profile`
- `cell_fraction`
- optional sample IDs for export and debugging

### 4. `deside/workflow/`

Purpose: provide train and inference orchestration in a style compatible with
the current package and conceptually aligned with VAEDecon.

Contents:

- training helper functions
- checkpoint loading helpers
- artifact path resolution
- prediction/export helpers

### 5. `deside/decon_cf/deside.py`

Purpose: remain the compatibility facade for current DeSide users.

Responsibilities:

- keep `DeSide` user-facing methods stable where practical
- delegate model creation, training, prediction, and checkpoint loading to the
  new PyTorch components
- preserve file outputs such as gene lists, cell types, metrics, and prediction
  CSVs

## Data flow and preprocessing design

The migrated training workflow preserves the current DeSide data semantics.

### Training flow

1. Read one or more training `.h5ad` files.
2. Extract bulk expression matrices `x` and cell-fraction targets `y`.
3. Merge training sources using the same inner-join logic on genes.
4. Apply current cell-type grouping and filtering behavior.
5. Remove invalid zero-sum target rows when current DeSide logic does so.
6. Apply existing expression transforms:
   - convert to TPM when needed
   - convert to `log2(TPM + 1)`
   - optional sample-wise scaling
   - optional scaling by constant
7. Build pathway profiles exactly as today for:
   - `add_to_end`
   - `convert`
8. Save feature and metadata files:
   - `celltypes.txt`
   - `genes.txt`
   - `genes_for_gep.txt`
   - `genes_for_pathway_profile.txt`
9. Split into training and validation sets deterministically.
10. Train with Lightning DataLoaders.

### Inference flow

1. Read input `.h5ad`, `.csv`, `.txt`, `.tsv`, or `DataFrame`.
2. Reproduce current feature alignment behavior against saved gene lists.
3. Rebuild pathway profiles if the model requires them.
4. Apply the same scaling logic as training/inference currently expects.
5. Load the Lightning checkpoint and reconstruct the PyTorch model from saved
   config.
6. Predict cell fractions.
7. Apply current post-processing rules:
   - optional `1 - alpha`
   - set fractions below `min_cell_fraction` to zero
   - renormalize rows whose sum exceeds `1`
   - fill `Cancer Cells` using `1-others` when needed
8. Save or return the prediction DataFrame.

## Model parity requirements

The PyTorch model must preserve the existing DeSide architecture and parameter
semantics:

- same hidden-layer widths
- same dropout positions and rates
- same normalization choices
- same activation ordering
- same optional pathway branch logic
- same branch concatenation semantics
- same output dimension and activation behavior

The current model uses a somewhat irregular normalization configuration via
`normalization_layer`. The migration must preserve this behavior explicitly
rather than silently simplifying it, because that would change the effective
architecture.

## Lightning training design

The Lightning module will follow the standard structure:

- `training_step`: forward pass, compute custom regression loss, log training
  metrics
- `validation_step`: compute validation loss and metrics without gradients
- `configure_optimizers`: create Adam optimizer and optional scheduler support

Checkpointing:

- model weights will be saved as Lightning `.ckpt`
- DeSide-specific metadata will be stored alongside checkpoints in the model
  directory
- the directory remains DeSide-owned and self-contained

Recommended artifact layout:

- `model_DeSide.ckpt` or `best_model.ckpt`
- `model_config.json`
- `training_config.json`
- `celltypes.txt`
- `genes.txt`
- `genes_for_gep.txt`
- `genes_for_pathway_profile.txt`
- `history_reg.csv` or equivalent exported metrics
- `key_params.txt` or a modernized JSON equivalent

Early stopping:

- preserve the intent of the current Keras `EarlyStopping` on validation loss
- use Lightning callbacks for early stopping and checkpointing

Validation split:

- replace Keras `validation_split=0.2` with an explicit deterministic split
- make the split configurable if needed, but default to current behavior

## Compatibility with VAEDecon

This migration should align with VAEDecon conventions without making DeSide a
submodule of VAEDecon.

### Required compatibility targets

- PyTorch tensor layout compatible with VAEDecon conventions
- Lightning-based checkpointing
- config-driven model reconstruction
- DataLoader-based training and inference
- stable prediction export as DataFrame and CSV
- predictable model artifact directory layout

### Integration boundary

During this upgrade, DeSide remains independent. The compatibility target is an
adapter-friendly interface, not direct code sharing everywhere.

Later VAEDecon integration should be able to:

- instantiate a DeSide predictor from a saved model directory
- call DeSide prediction on bulk expression input
- consume predicted cell proportions with shared tensor-shape expectations
- avoid runtime conflicts with DeSide dependencies

### Reuse strategy

The implementation should borrow patterns from VAEDecon's training, data, and
checkpoint flow, but avoid creating a hard dependency from DeSide onto
VAEDecon's internal modules in this phase.

## Migration boundaries and upgrade policy

This is a major upgrade and should be documented as such.

### Compatibility policy

- Public DeSide APIs should remain stable where practical.
- Saved model format changes are allowed.
- Generic legacy TensorFlow `.h5` loading is not supported in the new line.
- Legacy TensorFlow usage remains documented as an older install path.

### Legacy support documentation

Documentation must include:

- how to install the old TensorFlow-based DeSide version
- which release/tag corresponds to the legacy line
- that TensorFlow checkpoints are not compatible with the new PyTorch line
- guidance for users choosing between legacy and upgraded versions

## Validation plan

Validation will happen in three layers.

### 1. Unit and module tests

- verify model input and output shapes
- verify pathway and non-pathway branches
- verify custom loss calculation
- verify post-processing logic
- verify deterministic checkpoint save/load

### 2. Workflow tests

- train through DeSide's public training workflow
- load the saved PyTorch checkpoint
- predict through `DeSide.predict()`
- verify exported files and metadata
- verify package import and inference in the VAEDecon environment

### 3. Regression benchmarks

- run DeSide on benchmark datasets used by the legacy implementation
- compare MAE, RMSE, and convergence behavior
- require no material degradation in prediction quality

Exact floating-point identity across frameworks is not required, but the
migrated implementation must match the original model's behavior closely enough
to preserve intended functionality.

## Testing strategy

Add or update tests to cover:

- model forward pass with and without pathway inputs
- custom loss behavior
- preprocessing and gene alignment consistency
- deterministic train/validation split
- checkpoint round-trip
- pretrained-style prediction workflow
- DeSide package import within the VAEDecon environment

Where realistic benchmark datasets are too large for automated CI, include a
small regression-style fixture and document the larger manual benchmark
procedure.

## Documentation and packaging updates

Update the following:

- `README.md`
- `docs/installation.md`
- `docs/usage.md`
- `docs/functions.md` or relevant functional docs
- `docs/changelog.md`
- `pyproject.toml`
- `docs/requirements.txt`

Documentation changes must include:

- new PyTorch and Lightning installation requirements
- updated training and inference examples if CLI or artifact names change
- a migration note for checkpoint incompatibility
- a legacy-install section for the TensorFlow release
- a clear statement that this is a major upgrade

## Implementation phases

The migration is large enough that it should be executed in phases.

### Phase 1: framework scaffold

- add PyTorch and Lightning dependencies
- create model, trainer, and data module skeletons
- preserve current public API surface

### Phase 2: model and training parity

- port the current DeSide architecture
- port the custom loss
- port training loop semantics and callbacks

### Phase 3: inference and artifact migration

- implement checkpoint loading
- restore current prediction behavior
- preserve metadata and CSV outputs

### Phase 4: validation and docs

- add regression and integration tests
- update README and docs
- add legacy TensorFlow installation guidance
- mark release notes as a major upgrade

## Risks and mitigations

### Risk: silent architectural drift

Because the original Keras graph includes conditional normalization behavior,
the migration could accidentally simplify the model.

Mitigation:

- write explicit parity tests around layer shapes and branch behavior
- keep the architecture mapping table in implementation notes

### Risk: behavior drift from train/validation split changes

Keras `validation_split` is implicit, while Lightning expects explicit loaders.

Mitigation:

- use a deterministic split that matches the intended 80/20 behavior
- document any unavoidable difference

### Risk: checkpoint confusion for users

Users may assume `.h5` checkpoints still work.

Mitigation:

- document incompatibility prominently in README and installation docs
- name the new artifacts clearly as Lightning or PyTorch checkpoints

### Risk: premature coupling to VAEDecon

Trying to share too much code too early could destabilize DeSide as an
independent package.

Mitigation:

- align conventions first
- postpone deeper cross-repo reuse until after the migration is validated

## Open implementation notes

- Prefer matching `lightning==2.5.1` with the VAEDecon environment unless a
  documented DeSide-specific constraint requires a different version.
- Keep DeSide installable as a standalone package even if the VAEDecon
  environment is present in the same workspace.
- Preserve the current pre-trained-asset download workflow, but update it to
  resolve new checkpoint filenames and metadata where necessary.

## Recommended next step

Write a detailed implementation plan that breaks the migration into concrete
code changes, test tasks, and documentation tasks for phased execution on
branch `pytorch-dev`.
