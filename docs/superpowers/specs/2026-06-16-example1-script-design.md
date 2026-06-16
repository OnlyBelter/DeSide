# Example 1 Runnable Script (Pretrained Model)
## Goal
Provide a runnable, non-notebook Example 1 entry point in this repository, using the new one-call API `deside.predict_with_pretrained_model(...)`.

## Non-Goals
- Do not bundle or auto-download the pretrained model or pathway assets.
- Do not change the underlying inference implementation (`DeSide.predict`).
- Do not add new dependencies.

## User Experience
- A single Python script runnable from the repo:
  - Location: `examples/example1_pretrained_model.py`
  - Intended usage:
    - `python examples/example1_pretrained_model.py --input-file path/xx_TPM.csv --output-file ./results/y_pred.csv`
  - Optional overrides:
    - `--model-dir` (default `./DeSide_model`)
    - `--dataset-dir` (default `./datasets`)

## Inputs / Outputs
- Inputs:
  - `--input-file`: bulk TPM (or log-space) GEP file path compatible with the existing `DeSide` input reader.
- Outputs:
  - `--output-file`: CSV file path written by `deside.predict_with_pretrained_model`.

## Asset Assumptions
The script assumes local Example 1 assets exist (same as `DeSide_mini_example`):
- `./DeSide_model/` contains the pre-trained model directory structure.
- `./datasets/gene_set/` contains pathway `.gmt` files.

## Implementation Notes
- Use `argparse` for CLI parameters.
- Call `deside.predict_with_pretrained_model(...)` directly.
- On success, exit code 0.
- On missing required assets, the helper will raise `FileNotFoundError` with an explicit message.

## Verification
- `python examples/example1_pretrained_model.py --help` prints help successfully.
- `python -m py_compile examples/example1_pretrained_model.py` succeeds.
- Documentation mention in `README.md` points users to the script.
