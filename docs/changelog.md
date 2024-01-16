Change Log
==========
The record of all notable changes to this project will be documented in this file.
***

## v0.2
1 Oct 2020, first release.

## v0.9.2
12 May 2021, build workflow and create documentation

## v0.9.3
18 May 2021, bug fix, deal with more tumor types

## v0.9.4
11 Jun 2021, bug fix, update `Scanpy` to v1.7.2, deal with the correlation between predicted cell fractions of each cell type and mean gene expression of corresponding marker genes

## v0.9.5
2 Jul 2021, typo fix, update `Scanpy` to v1.8.0 & update `seaborn` to v0.11.1

## v0.9.6
7 Jul 2021, typo fix, check if `Cancer Cells` existed and update `bbknn` to v1.5.1

## v0.9.6.1
16 Jul 2021, set `total_cell_number` to 100 and `n_base` to 3 when simulating bulk cell data for better keeping cell type diversity

## v0.9.7
13 Sep 2021
- Filtering single cell data of CD4 T and CD8 T cells by the ratio of marker genes
- Updating marker gene list and re-annotating sub-clustering based on dot-plot figure of each single cell dataset

## v0.9.8
19 Nov 2021
- Building the whole workflow in one file
- Recording main parameters and running logs

## v1.0.2 (bioRxiv)
2 Jun 2023
- Update documentation

## v1.1.0
1 Oct 2023
- Update the workflow to contain the following parameters:
  - `alpha_total_rna_coefficient`: the coefficient of total RNA for each cell type during the simulation of bulk RNA-seq data
  - `cell_type2subtypes`: the dictionary of cell types and their subtypes
- Update the DNN model to include pathway profiles
- Update the documentation


## v1.2.0
16 Jan 2024
- Add GEP-level filtering in the PCA space of TCGA samples
- Add Dirichlet distribution to simulate the cellular proportions of bulk RNA-seq data

