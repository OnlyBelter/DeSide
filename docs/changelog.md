Change Log
==========

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

## v0.9.9
21 Jan 2022
- Using marker gene ratios to compare the similarity of GEPs between simulated bulk cell dataset and TCGA dataset
- New method to filter simulated bulk cell GEPs depends on the nearest neighbors of each sample in TCGA dataset
- Filtering genes by the correlation between gene expression values and cell fraction of each cell type
- Removing genes depends on the loading matrix of PCA analysis by combining both simulated dataset and TCGA dataset

## v0.9.9.1
3 Mar 2022
- update for evaluation
