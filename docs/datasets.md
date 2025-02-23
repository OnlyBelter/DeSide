Datasets
========

Datasets used in DeSide
***

## Merged datasets and Synthetic datasets (Table S1)

|                Dataset name                | #samples    | Sampling method | Filtering | #cell types | #genes | Input dataset                   |    GEPs <br/>(type, fortmat)    |         Dataset type          | Notation |
|:------------------------------------------:|-------------|-----------------|-----------|-------------|--------|---------------------------------|:-------------------------------:|:-----------------------------:|:--------:|
|                    TCGA                    | 7,699       | -               | -         | -           | 19,712 | -                               |           MCT, `TPM`            |     Downloaded from TCGA      |    DA    |
|            merged_7_sc_datasets            | 325,474     | -               | -         | 19          | 17,834 | 12 collected scRNA-seq datasets | Single cell, <br/>`log2(TPM+1)` |  Raw dataset from scRNA-seq   |    S0    |
|                SCT_POS_N10K                | 10,000 x 16 | n_base=100      | -         | 16          | 17,834 | S0                              |       SCT, `log2(TPM+1)`        | Used to simulate MCT datasets |    S1    |
|             Mixed_N100K_random             | 100,000     | Random          | No        | 16          | 17,834 | S1                              |       MCT, `log2(TPM+1)`        |         Training set          |    D0    |
|            Mixed_N100K_segment             | 100,000     | Segment         | Yes       | 16          | 9,028  | S1                              |       MCT, `log2(TPM+1)`        |         Training set          |    D1    |
| Mixed_N100K_segment_<br/>without_filtering | 100,000     | Segment         | No        | 16          | 17,834 | S1                              |       MCT, `log2(TPM+1)`        |         Training set          |    D2    |
|              Test_set_random               | 3,000       | Random          | No        | 16          | 17,834 | S1                              |       MCT, `log2(TPM+1)`        |           Test set            |    T0    |
|                 Test_set1                  | 3,000       | Segment         | Yes       | 16          | 9,028  | S1                              |       MCT, `log2(TPM+1)`        |           Test set            |    T1    |
|                 Test_set2                  | 3,000       | Segment         | No        | 16          | 17,834 | S1                              |       MCT, `log2(TPM+1)`        |           Test set            |    T2    |
|                SCT_POS_N100                | 100 x 16    | n_base=100      | -         | 16          | 17,834 | S0                              |       SCT, `log2(TPM+1)`        |           Test set            |    T3    |

- MCT: Bulk gene expression profiles with multiple different cell types
- SCT: Bulk gene expression profiles with single cell type (sctGEPs)
- GEPs: Gene expression profiles

## Collected scRNA-seq datasets (Table S2)

| Dataset ID         | Journal         | DOI                          | Publish Date | Reported cells (total)* | Integrated cells (used) | Organism | Tissue                                  | Data location                                           | Sequencing method         | #patients** |
|--------------------|-----------------|------------------------------|--------------|-------------------------|-------------------------|----------|-----------------------------------------|---------------------------------------------------------|---------------------------|-------------|
| hnscc_cillo_01     | Immunity        | 10.1016/j.immuni.2019.11.014 | 20200107     | 131,224                 | 57,034                  | Human    | Head and Neck Cancer (HNSC)             | GSE139324                                               | 10x Single Cell 3' v2     | 26          |
| pdac_pengj_02      | Cell Res        | 10.1038/s41422-019-0195-y    | 20190704     | 57,530                  | 37,079                  | Human    | Pancreatic Ductal Adenocarcinoma (PDAC) | [Link](https://ngdc.cncb.ac.cn/gsa/browse/CRA001160)    | 10x Single Cell 3' v2     | 22          |
| hnscc_puram_03     | Cell            | 10.1016/j.cell.2017.10.044   | 20171130     | 5,902                   | 4,647                   | Human    | Head and Neck Cancer (HNSC)             | GSE103322                                               | Smart-seq2                | 16          |
| pdac_steele_04     | Nat Cancer      | 10.1038/s43018-020-00121-4   | 20201026     | 124,898                 | 32,062                  | Human    | Pancreatic Ductal Adenocarcinoma (PDAC) | GSE155698                                               | 10x Single Cell 3' v2     | 15          |
| luad_kim_05        | Nat Commun      | 10.1038/s41467-020-16164-1   | 20200508     | 208,506                 | 49,959                  | Human    | Lung Adenocarcinoma (LUAD)              | GSE131907                                               | 10x Single Cell 3' v2     | 13          |
| nsclc_guo_06       | Nature Medicine | 10.1038/s41591-018-0045-3    | 20180625     | 12,346                  | 4,050                   | Human    | Non-Small-Cell Lung Cancer (NSCLC)      | GSE99254                                                | Smart-Seq2                | 13          |
| pan_cancer_07      | Nat Genet       | 10.1038/s41588-020-00726-6   | 20201030     | 53,513                  | 30,681                  | Human    | Cancer cell lines                       | GSE157220                                               | Illumina NextSeq 500      | -           |
| prad_cheng_08      | Nat Cell Biol   | 10.1038/s41556-020-00613-6   | 20211108     | 36,424                  | 28,253                  | Human    | Prostate cancer (PRAD)                  | https://www.weizmann.ac.il/sites/3CA/prostate           | 10X Genomics	             | 12          |
| prad_dong_09	      | Commun Biol	    | 10.1038/s42003-020-01476-1   | 20201216     | 21,292                  | 16,472                  | Human    | Prostate cancer (PRAD)                  | https://www.weizmann.ac.il/sites/3CA/prostate           | 10X Genomics	             | 6           |
| hcc_sun_10         | Cell            | 10.1016/j.cell.2020.11.041   | 20201123     | 16,498                  | 11,365                  | Human    | Hepatocellular carcinoma (HCC)          | https://www.weizmann.ac.il/sites/3CA/liverbiliary       | 10X Genomics	             | 16          |
| gbm_neftel_11      | Cell            | 10.1016/j.cell.2019.06.024   | 20190618     | 24,131                  | 16,835                  | Human    | Glioblastoma multiforme (GBM)           | https://www.weizmann.ac.il/sites/3CA/brain (GSE131928)	 | 10X Genomics	             | 36          |
| gbm_abdelfattah_12 | Nat Commun      | 10.1038/s41467-022-28372-y   | 20220909     | 201,986                 | 37,037                  | Human    | Glioblastoma multiforme (GBM)           | GSE182109                                               | 10× Chromium / HiSeq 4000 | 8           |

- \* The number of **reported cells** may include cells that don't originate from solid tumors, which were removed during integrating.
- \*\* The count considered only the number of patients (samples) in the data that were integrated into the final dataset.


## Download
- TCGA (DA): [merged_tpm.csv.zip](https://doi.org/10.6084/m9.figshare.23047547.v2)
  - Contains 7699 samples from 19 cancer types
  - For sample-to-cancer type mapping, see: [tcga_sample_id2cancer_type.csv](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/datasets/TCGA/tpm/tcga_sample_id2cancer_type.csv)
- merged_12_sc_datasets (S0): [merged_12_sc_datasets_231003.h5ad](https://doi.org/10.6084/m9.figshare.23283908.v2)
- SCT_POS_N10K (S1): [simu_bulk_exp_SCT_N10K_S1_16sct.h5ad](https://doi.org/10.6084/m9.figshare.23043560.v2)
- Mixed_N100K_random (D0): [simu_bulk_exp_Mixed_N100K_random_log2cpm1p.h5ad](https://doi.org/10.6084/m9.figshare.23283932.v2)
- Mixed_N100K_segment (D1): [simu_bulk_exp_Mixed_N100K_D1.h5ad](https://doi.org/10.6084/m9.figshare.23047391.v2)
- Mixed_N100K_segment_without_filtering (D2): [simu_bulk_exp_Mixed_N100K_D2.h5ad](https://doi.org/10.6084/m9.figshare.23284256.v2)
- All Test Sets: [all_test_sets.zip](https://doi.org/10.6084/m9.figshare.23283884.v3)
  - Test_set_random (T0)
  - Test_set1 (T1)
  - Test_set2 (T2)
  - SCT_POS_N100 (T3)

`.h5ad` files can be opened by the function `scanpy.read_h5ad()` in [Scanpy](https://scanpy.readthedocs.io/en/stable/) or the class [`deside.utility.read_file.ReadH5AD`](https://deside.readthedocs.io/en/latest/func/utility.html#deside.utility.read_file.ReadH5AD) in DeSide.

