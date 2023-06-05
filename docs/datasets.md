Datasets
========

Datasets used in DeSide
***
## scRNA-seq datasets

| Dataset ID  | Journal              | DOI                         | Publish Date | Reported cells (total) | Organism               | Tissue                           | Data location                     | Sequencing method       | #patients |
|-------------|----------------------|-----------------------------|--------------|------------------|------------------------|----------------------------------|-----------------------------------|-------------------------|-----------|
| hnscc_cillo_01 | Immunity             | 10.1016/j.immuni.2019.11.014 | 20200107     | 131,224          | Human                  | Head and Neck Cancer (HNSC)      | GSE139324                         | 10x Single Cell 3' v2    | 26        |
| pdac_pengj_02 | Cell Res             | 10.1038/s41422-019-0195-y  | 20190704     | 57,530           | Human                  | Pancreatic Ductal Adenocarcinoma (PDAC)| [Link](https://bigd.big.ac.cn/gsa/browse/CRA001160) | 10x Single Cell 3' v2    | 22        |
| hnscc_puram_03 | Cell                 | 10.1016/j.cell.2017.10.044 | 20171130     | 5,902            | Human                  | Head and Neck Cancer (HNSC)      | GSE103322                         | Smart-seq2               | 16        |
| pdac_steele_04 | Nat Cancer           | 10.1038/s43018-020-00121-4 | 20201026     | 124,898          | Human                  | Pancreatic Ductal Adenocarcinoma (PDAC)| GSE155698                         | 10x Single Cell 3' v2    | 15        |
| luad_kim_05 | Nat Commun           | 10.1038/s41467-020-16164-1 | 20200508     | 208,506          | Human                  | Lung Adenocarcinoma (LUAD)       | GSE131907                         | 10x Single Cell 3' v2    | 13        |
| nsclc_guo_06 | Nature Medicine      | 10.1038/s41591-018-0045-3  | 20180625     | 12,346           | Human                  | Non-Small-Cell Lung Cancer (NSCLC) | GSE99254                          | Smart-Seq2               | 13        |
| pan_cancer_07 | Nat Genet            | 10.1038/s41588-020-00726-6 | 20201030     | 53,513           | Human                  | Cancer cell lines                | GSE157220                         | Illumina NextSeq 500    | -         |


- The number of **reported cells** may include cells that don't originate from solid tumors, which were removed during integrating.

## Merged datasets and Synthetic datasets

|              Dataset name              | #samples | Sampling method | Filtering | #cell types | #genes | Input dataset                  |      GEPs <br/>(type, fortmat)       |         Dataset type          |  Notation  |
|:--------------------------------------:|----------|-----------------|-----------|-------------|--------|--------------------------------|:-------------------------------:|:-----------------------------:|:----------:|
|                  TCGA                  | 7,699    | -               | -         | -           | 19,712 | -                              |           MCT, `TPM`            |     Downloaded from TCGA      |     DA     |
|          merged_7_sc_datasets          | 135,049  | -               | -         | 11          | 12,114 | 7 collected scRNA-seq datasets | Single cell, <br/>`log2(TPM+1)` |  Raw dataset from scRNA-seq   |     S0     |
|              SCT_POS_N10K              | 110,000  | n_base=100      | -         | 11          | 12,114 | S0                             |       SCT, `log2(TPM+1)`        | Used to simulate MCT datasets |     S1     |
|           Mixed_N100K_random           | 100,000  | Random          | No        | 11          | 12,114 | S1                             |       MCT, `log2(TPM+1)`        |         Training set          |     D0     |
|          Mixed_N100K_segment           | 100,000  | Segment         | Yes       | 11          | 6,168  | S1                             |       MCT, `log2(TPM+1)`        |         Training set          |     D1     |
| Mixed_N100K_segment_<br/>without_filtering  | 100,000  | Segment   | No        | 11          | 12,114 | S1                             |       MCT, `log2(TPM+1)`        |         Training set          |     D2     |
|            Test_set_random             | 3,000    | Random          | No        | 11          | 12,114 | S1                             |       MCT, `log2(TPM+1)`        |           Test set            |     T0     |
|               Test_set1                | 3,000    | Segment         | Yes       | 11          | 6,168  | S1                             |       MCT, `log2(TPM+1)`        |           Test set            |     T1     |
|               Test_set2                | 3,000    | Segment         | No        | 11          | 12,114 | S1                             |       MCT, `log2(TPM+1)`        |           Test set            |     T2     |
|              SCT_POS_N100              | 1100     | n_base=100      | -         | 11          | 12,114 | S0                             |       SCT, `log2(TPM+1)`        |           Test set            |     T3     |

- MCT: Bulk gene expression profile with multiple different cell types
- SCT: Bulk gene expression profile with single cell type (scGEP)
- GEPs: Gene expression profiles

## Download
- TCGA: [download link](https://figshare.com/articles/dataset/Merged_gene_expression_profiles_TPM_/23047547)
- merged_7_sc_datasets (S0): [download link](https://figshare.com/articles/dataset/Dataset_S0/23283908)
- SCT_POS_N10K (S1): [download link](https://figshare.com/articles/dataset/Dataset_S1/23043560)
- Mixed_N100K_random (D0): [download link](https://figshare.com/articles/dataset/Dataset_D0/23283932)
- Mixed_N100K_segment (D1): [download link](https://figshare.com/articles/dataset/Dataset_D1/23047391)
- Mixed_N100K_segment_without_filtering (D2): [download link](https://figshare.com/articles/dataset/Dataset_D2/23284256)
- All Test Sets: [download link](https://figshare.com/articles/dataset/All_Test_Sets/23283884)
  - Test_set_random (T0)
  - Test_set1 (T1)
  - Test_set2 (T2)
  - SCT_POS_N100 (T3)

`.h5ad` files can be opened by the function `scanpy.read_h5ad()` in [Scanpy](https://scanpy.readthedocs.io/en/stable/) or the class [`deside.utility.read_file.ReadH5AD`](https://deside.readthedocs.io/en/latest/func/utility.html#deside.utility.read_file.ReadH5AD) in DeSide.

