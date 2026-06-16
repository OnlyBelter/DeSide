Change Log
==========
The record of all notable changes to this project will be documented in this file.
***

## v1.3.3
June 16, 2026

- Add `predict_with_pretrained_model(...)` as a one-call entry point for Example 1.
- Add automatic downloading of essential Example 1 assets by default:
  - The pre-trained `model_DeSide.h5` is downloaded from Figshare (DOI: `10.6084/m9.figshare.25117862.v1`).
  - The remaining `DeSide_model/*` metadata files and `datasets/gene_set/*.gmt` files are downloaded from the
    `DeSide_mini_example` repository.
- Add explicit download logs for each downloaded file, including destination path, file size, and md5.
- Add a runnable script `examples/example1_pretrained_model.py` for non-notebook usage.
- Update documentation and README to reflect the new one-line workflow and auto-download behavior.

## v1.3.2 (revision)
September 15, 2024

- Update documentation.

## v1.3.1 (revision)
July 13, 2024

- Update Figure 1ab.
- Update workflow and plot functions during the revision.

## v1.2.2 (revision)
February 3, 2024

- Update the documentation.

## v1.2.0
January 16, 2024

- Add GEP-level filtering in the PCA space of TCGA samples.
- Add Dirichlet distribution to simulate the cellular proportions of bulk RNA-seq data.

## v1.1.0
October 1, 2023

- Update the workflow to include:
  - `alpha_total_rna_coefficient`: coefficient of total RNA for each cell type during bulk RNA-seq simulation.
  - `cell_type2subtypes`: mapping of cell types to subtypes.
- Update the DNN model to include pathway profiles.
- Update the documentation.

## v1.0.2 (bioRxiv)
June 2, 2023

- Update documentation.

## v0.9.8
November 19, 2021

- Build the whole workflow in one file.
- Record main parameters and running logs.

## v0.9.7
September 13, 2021

- Filter single cell data of CD4 T and CD8 T cells by the ratio of marker genes.
- Update marker gene list and re-annotate sub-clustering based on dot-plot figures of each single cell dataset.

## v0.9.6.1
July 16, 2021

- Set `total_cell_number` to 100 and `n_base` to 3 when simulating bulk cell data to better keep cell type diversity.

## v0.9.6
July 7, 2021

- Fix typos.
- Check whether `Cancer Cells` exists and update `bbknn` to v1.5.1.

## v0.9.5
July 2, 2021

- Fix typos.
- Update `Scanpy` to v1.8.0.
- Update `seaborn` to v0.11.1.

## v0.9.4
June 11, 2021

- Fix bugs.
- Update `Scanpy` to v1.7.2.
- Evaluate correlation between predicted cell fractions and mean marker-gene expression per cell type.

## v0.9.3
May 18, 2021

- Fix bugs.
- Add support for more tumor types.

## v0.9.2
May 12, 2021

- Build workflow and create documentation.

## v0.2
October 1, 2020

- First release.
