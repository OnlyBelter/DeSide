# Auto-Bootstrap sctGEP for Bulk Simulation

## Goal

Extend the standalone bulk simulation workflow so it can generate the
single-cell-type GEP reference dataset automatically when
`input.sct_dataset_file_path` is empty, then reuse that generated artifact in
the existing mixed-bulk simulation pipeline.

This change must preserve the legacy simulation logic:

- use `SingleCellTypeGEPGenerator` for sctGEP generation
- use `BulkGEPGenerator` for mixed-bulk generation
- keep GEP-level filtering semantics unchanged
- keep gene-level filtering semantics unchanged
- keep the bulk simulation workflow independent from model training

## Scope

In scope:

- make `input.sct_dataset_file_path` optional in the standalone bulk-simulation
  config
- auto-generate an sctGEP `.h5ad` when that field is empty
- add a small config section for sctGEP bootstrap parameters
- pass the generated file into the current mixed-bulk workflow
- support skip-if-done for the generated sctGEP artifact
- write the resolved artifact path into the workflow summary

Out of scope:

- changing model-training configs or training workflow
- changing the underlying simulation math
- replacing the legacy generators
- adding a separate explicit `simulate-sct` CLI in this pass

## Recommended Approach

Use a two-stage orchestration inside the existing standalone workflow:

1. Resolve the sctGEP reference:
   - if `input.sct_dataset_file_path` is provided and exists, use it
   - if it is empty, generate the sctGEP dataset with
     `SingleCellTypeGEPGenerator`
   - if it is provided but missing, fail with a clear error
2. Run the current mixed-bulk simulation exactly as before, using the resolved
   sctGEP path.

This is the smallest change that gives the desired convenience without
changing the meaning of the legacy workflow.

## Config Design

Keep the existing field:

- `input.sct_dataset_file_path`

Interpretation:

- non-empty existing path: use it directly
- empty string: auto-generate the sctGEP dataset
- non-empty missing path: raise an error

Add a new top-level section:

```yaml
sct_generation:
  sct_dataset_name: 'mixed_sctGEP_nbase100'
  n_sample_each_cell_type: 10000
  n_base_for_positive_samples: 100
  sample_type: 'positive'
  sep_by_patient: false
  simu_method: 'ave'
  test_set: false
  minimum_n_base: 1
  ref_gene_list_file_path: ''
  output_file_path: ''
```

Notes:

- defaults should mirror the legacy `SingleCellTypeGEPGenerator` API
- `output_file_path` is optional; if empty, use the generator’s deterministic
  default path under `simu_bulk_dir`
- this section is only used when `input.sct_dataset_file_path` is empty

## Workflow

### Stage 1. sctGEP Resolution

The standalone workflow adds a helper that resolves the sctGEP reference path
before mixed-bulk generation.

Behavior:

- if `input.sct_dataset_file_path` is empty:
  - instantiate `SingleCellTypeGEPGenerator`
  - run `generate_samples(...)` with parameters from `sct_generation`
  - save or reuse the generated `.h5ad`
  - return the generated path
- otherwise:
  - validate the provided path exists
  - return the provided path

### Stage 2. Mixed-Bulk Simulation

The workflow then runs the existing `BulkGEPGenerator.generate_gep(...)`
pipeline unchanged, except that the `sct_dataset_file_path` it receives is the
resolved path from Stage 1.

### Stage 3. Existing Filtering

After mixed-bulk generation:

- keep the current GEP-level filtering flow unchanged
- keep the current gene-level filtering flow unchanged

## Error Handling

- empty `input.sct_dataset_file_path` with missing `merged_sc_dataset_file_path`
  should raise a clear error, because sctGEP bootstrap needs the merged
  single-cell dataset
- non-empty but nonexistent `input.sct_dataset_file_path` should raise a clear
  error rather than silently switching to generation
- invalid `sct_generation` values should fail during config validation

## Outputs

The workflow summary JSON should include:

- `resolved_sct_dataset_file`
- whether the file was reused or generated
- `generated_bulk_gep_file`
- optional `filtered_bulk_gep_file`
- optional `gene_list_file`

## Testing

Verification should cover:

1. config loads with `input.sct_dataset_file_path: ''`
2. bootstrap path resolution works
3. provided existing sctGEP path still works
4. provided missing sctGEP path fails clearly
5. skip-if-done reuses existing generated sctGEP
6. mixed-bulk generation still receives a valid sctGEP file path

## Risks

- the main risk is overlapping with unrelated in-progress edits in
  `deside/simulation/generate_data.py`
- to reduce that risk, the preferred implementation should keep changes
  concentrated in config loading and workflow orchestration, touching legacy
  generator code only if a minimal hook is required
