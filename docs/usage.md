Usage
=====

How to use DeSide



****

**DeSide have three steps to deconvolution. Here are two ways to use DeSide, one can skip the first two steps and use the provided well-trained model to predict cell fractions directly, if you do not want to train DeSide model by yourself. Another way is run the  program  step by step from the first step of DeSide (DeSide Simulation), during the process, you can reform the single cell dataset, retrain the model by yourself.**



The workflow of deconvolution by DeSide consists of three steps:

1.  DeSide Prediction
2.  DeSide Training
3.  DeSide Simulation

## DeSide prediction

Using the pre-trained model or self-trained model to predict cell type proportions in a new dataset. with function `deside_model.predict()`. [See details on this function]().

you can use the following command to perform the deconvolution:

### Example code

```python
from deside.decon_cf import DeSide

# read pre-trained DeSide model
model_dir = './DeSide_model/'
deside_model = DeSide(model_dir=model_dir)

# predict by pre-trained model
# - transpose=True, if the bulk_tpm_file is provided as genes by samples (rows by columns)
# - we used scaling_by_constant in the manuscript, Scaden used scaling_by_sample
deside_model.predict(input_file=bulk_tpm_file_path, output_file_path=y_pred_file_path, 
                     exp_type='TPM', transpose=True,
                     scaling_by_sample=False, scaling_by_constant=True)
```

### Input files

In this step, you can use the whole result directory of `result_dir` in `DeSide train` step as the input of  parameter `model_dir`, **[or you can download our best pre-trained model file from datasets file]() (location: datasets/well_trained_model/DeSide) and use the real directory in your computer.** Besides, Bulk gene expression profiles (bulk GEPs) need to be delivered which were given in transcripts per million (TPM) or log2(TPM + 1). It should be separated by ','  and saved in a `.csv` file. The shape of this file should be m×n, where m is the number of features (genes) and n is the number of samples. You can download an example file from [our datasets file]() (location: datasets\TCGA\tpm). If you have not this format of bulk GEPs file, DeSide can only provide functionality to convert TCGA read count data into the correct file format. Have a look at the [Data Processing]() section for instructions on how to use this function. 

- `model_dir`: The `result_dir` directory contains the well-trained model.
- input_file: Bulk GEPs in transcripts per million (TPM) or log2(TPM + 1) format, separated by ',', and saved in a `.csv` file.



### Output files

You will receive a `.csv` file containing the cell fraction for each cell type per sample, which will be saved in the directory specified by the `output_file_path` parameter.


Example files: You can download this file from [our results file]() (location: results/predicted_cell_fraction).     



## DeSide Training

If you wish to train a model from scranch, here is the function that we provide. Once you have set up the training set, , you can start training a DeSide model with function `DeSide().train_model()`.  [See details on this function]().  



### Example code

```python
from deside.utility import check_dir
from deside.decon_cf import DeSide

# create output directory
result_dir = './results/E2'
check_dir(result_dir)

# using D2 as the training set
training_set2file_path = {
    'D2': './datasets/simulated_bulk_cell_dataset/segment_7ds_0.95_n_base100_median_gep/simu_bulk_exp_Mixed_N100K_segment_log2cpm1p_filtered_by_high_corr_gene_and_quantile_range_q_5.0_q_95.0.h5ad',
}

all_cell_types = sorted_cell_types

# set the hyper-parameters
deside_parameters = {'architecture': ([100, 1000, 1000, 1000, 50],
                                      [0, 0, 0, 0.2, 0]),
                     'loss_function': 'mae+rmse',
                     'batch_normalization': False,
                     'last_layer_activation': 'sigmoid',
                     'learning_rate': 2e-5,
                     'batch_size': 128}

# remove cancer cell during training process
remove_cancer_cell = True


# set result dirtory to save DeSide model
model_dir = os.path.join(result_dir, 'DeSide_model')
log_file_path = os.path.join(result_dir, 'deside_running_log.txt')
deside_obj = DeSide(model_dir=model_dir, log_file_path=log_file_path)

# training DeSide
# - training_set_file_path is a list, multiple datasets will be combined together
deside_obj.train_model(training_set_file_path=[training_set2file_path['D2']], 
                       hyper_params=deside_parameters, cell_types=all_cell_types,
                       scaling_by_constant=True, scaling_by_sample=False,
                       remove_cancer_cell=remove_cancer_cell,
                       n_patience=100, n_epoch=3000, verbose=0)

```
### Input files

In this step, you can use the simulated bulk GEPs (training set) were saved in `.h5ad` file. This file usually generated from  `DeSide Simulation` step. The training set D2 we used can be downloaded from [our datasets file]() (location: datasets/simulated_bulk_cell_dataset/segment_7ds_0.95_n_base100_median_gep/simu_bulk_exp_Mixed_N100K_segment_log2cpm1p_filtered_by_high_corr_gene_and_quantile_range_q_5.0_q_95.0.h5ad)  

`xx` in the following file name is same as the parameter `dataset_name` in function `simulated_bulk_gep_generator`.    `?????`

- training_set_file_path: The simulated bulk GEPs (training set) were saved in `.h5ad` file that contains both GEPs (in log2(CPM+1) counts ) information and the fraction of cell types.
  

### Output files

You will get 5 files in this step and details are show below. The entire result directory (specified by the `training_set_file_path` parameter) can be utilized as the `model_dir` paramater in the `DeSide prediction` step. `xx` in the following file name is same as the parameter `model_name` in function  `dnn_training()`.

- celltypes.txt:	Cell types which were used in the training process.
- genes.txt:	Genes which were used in the training process.
- history_reg.csv: ...
- loss.png: ...
- model_xx.h5: Well-trained model.

Example files: You can download this file from [our datasets file]() (location: datasets/well_trained_model).     #目录指定到最好的模型目录？





## DeSide Simulation

### a. Using the single cell dataset we provided

In this step, you can generate simulated (GEPs) based on single cell RNA-seq (scRNA-seq) dataset. We provided 7 scRNA-seq datasets so far: 'hnscc_cillo_01', 'pdac_pengj_02', 'hnscc_puram_03', 'pdac_steele_04', 'luad_kim_05', 'nsclc_guo_06', 'pan_cancer_07'. You can use all (or part) of 7 scRNA-seq datasets by specified the sc_dataset_ids parameter when you call class `BulkGEPGenerator()`. Generated GEPs can be used as `training set` for training DeSide model.        ` or `test set` for testing the model performance after training finished.      ????`

### b. Preparing single cell dataset by yourself

If you want to use other scRNA-seq datasets to Simulate GEPs, you can follow our workflow to preprocess single cell datasets and merge them together. Python package `Scanpy` was used heavily in our workflow.

- Preprocessing a single dataset: [Here is an example of how i process my scRNA-seq dataset(s)]().
- Merging multiple datasets together: [Here is an example of how i process my scRNA-seq dataset(s)]().

Also read the section "1. Integrating single cell RNA-seq dataset" in supplementary material for more details.

[Explanation of parameters in this function can be found](). 

#### Example code for generating a training set

```python
from deside.simulation import simulated_bulk_gep_generator

n_base = 100  # averaging 100 GEPs sampled from S1 to synthesize 1 bulk GEP, used by S1 generation
gep_filtering_quantile = (0.0, 0.95)  # GEP-level filtering, L1-norm threshold
dataset_name = 'Mixed_N10K_segment'
# the list of single cell RNA-seq datasets
sc_dataset_ids = ['hnscc_cillo_01', 'pdac_pengj_02', 'hnscc_puram_03',
                  'pdac_steele_04', 'luad_kim_05', 'nsclc_guo_06', 'pan_cancer_07']

params={"sampling_method": "segment",
        "sc_dataset_ids": [
            "hnscc_cillo_01",
            "pdac_pengj_02",
            "hnscc_puram_03",
            "pdac_steele_04",
            "luad_kim_05",
            "nsclc_guo_06",
            "pan_cancer_07"
            ],
        "cell_types": [
            "B Cells",
            "CD4 T",
            "CD8 T",
            "Cancer Cells",
            "DC",
            "Endothelial Cells",
            "Fibroblasts",
            "Macrophages",
            "Mast Cells",
            "NK",
            "Neutrophils"],
        "n_samples": 10000,
        "sampling_method": "segment",
        "filtering": true,
        "filtering_ref_types": [
            "ACC",
            "BLCA",
            "BRCA",
            "GBM",
            "HNSC",
            "LGG",
            "LIHC",
            "LUAD",
            "PAAD",
            "PRAD",
            "CESC",
            "COAD",
            "KICH",
            "KIRC",
            "KIRP",
            "LUSC",
            "READ",
            "THCA",
            "UCEC"]
        }
sampling_method = params['sampling_method']
n_sc_datasets = len(sc_dataset_ids)
simu_bulk_exp_dir= os.path.join('results', 'E3', '{}_{}ds_{}_n_base{}_median_gep')
simu_bulk_exp_dir= simu_bulk_exp_dir.format(sampling_method, n_sc_datasets,
                                            gep_filtering_quantile[1], n_base)

sct_dataset_file_path='./datasets/generated_sc_dataset_7ds_n_base100/simu_bulk_exp_SCT_POS_N10K_log2cpm1p.h5ad'
tcga_data_dir = './datasets/TCGA/tpm/' 
tcga2cancer_type_file_path = os.path.join(tcga_data_dir, 'tcga_sample_id2cancer_type.csv')

bulk_generator = BulkGEPGenerator(simu_bulk_dir=simu_bulk_exp_dir,
                                    merged_sc_dataset_file_path=None,
                                    cell_types=params['cell_types'],
                                    sc_dataset_ids=params['sc_dataset_ids'],
                                    bulk_dataset_name=dataset_name,
                                    sct_dataset_file_path=sct_dataset_file_path,
                                    check_basic_info=False,
                                    tcga2cancer_type_file_path=tcga2cancer_type_file_path)

# GEP-filtering will be performed during this generation process

tcga_merged_tpm_file_path = os.path.join(tcga_data_dir, 'merged_tpm.csv')
log_file_path= './results/E3/DeSide_running_log.txt'
cell_prop_prior = None
bulk_generator.generate_gep(n_samples=params['n_samples'],
                            simu_method='mul',
                            sampling_method=params['sampling_method'],
                            reference_file=tcga_merged_tpm_file_path,
                            ref_exp_type='TPM',
                            filtering=params['filtering'],
                            filtering_ref_types=params['filtering_ref_types'],
                            gep_filtering_quantile=gep_filtering_quantile,
                            n_threads=5,
                            log_file_path=log_file_path,
                            show_filtering_info=False,
                            filtering_method='median_gep',
                            cell_prop_prior=cell_prop_prior)
```

#### Example code for generating a test set    ???

```python
from deside.simulation import simulated_bulk_gep_generator

simulated_bulk_gep_generator(n_per_gradient=5, 
                             result_dir="path/result/bulk_simulate/", n_sample_cell_type=5000, 
                             merged_sc_dataset_file_path="path/result/sc_datasets/file.h5ad ", 
                             cell_types=['Cancer Cells', 'CD4 T', 'CD8 T', 'Fibroblasts', 'DC', 'NK'], 
                             gradient_range={'Cancer Cells': (50, 100), 'CD8 T': (1, 30), 
                                             'Fibroblasts': (80, 100)},
                             dataset_name="Test_set", sc_dataset_id=["all"], 
                             generated_sc_dataset_dir="path/result/bulk_simulate/sc_data/")
```

### Input files

The provided single cell dataset, `simu_bulk_exp_SCT_POS_N10K_log2cpm1p.h5ad`,  is a `h5ad` file (see more details about this file format from [anndata](https://anndata.readthedocs.io/en/latest/index.html)) which contains 7 scRNA-seq datasets and 11 cell types. You can download this file from [our datasets file]() (location: './datasets/generated_sc_dataset_7ds_n_base100/simu_bulk_exp_SCT_POS_N10K_log2cpm1p.h5ad')

- simu_bulk_exp_SCT_POS_N10K_log2cpm1p.h5ad

  - `obs` contains the information of single cell samples such as sample id, cell type and dataset id.

  - `var` contains gene names of samples. 

  - `X` is a matrix of gene expression profiles (GEPs, log space) with the shape of n_sample  by n_gene.

  - 7 scRNA-seq dataset ids: "hnscc_cillo_01", "pdac_pengj_02", "hnscc_puram_03", "pdac_steele_04", "luad_kim_05", "nsclc_guo_06", "pan_cancer_07".

  - 11 cell types: B Cells, CD4 T, CD8 T, Cancer Cells, DC, Endothelial cells, Fibroblasts, Macrophages, Mast Cells, NK, Neutrophils.

    

    This file can be accessed by:

    ```python
    import anndata as an
    
    merged_sc_dataaset = an.read_h5ad('path/to/merged_6_sc_dataset.h5ad')
    print(merged_sc_dataset.X.shape)  # (67870, 11785)
    ```

    

### Output files

File `simu_bulk_exp_xx_log2cpm1p.h5ad` can be used as input data of function `dnn_training()` for training model or function `dnn_predict()` for testing model performance.

`xx` in the following file name is same as the parameter `dataset_name`.

- generated_11_cell_type_n5000_all.h5ad: this is an intermediate file before simulating bulk GEPs. This file contains the generated single cell dataset that contains 11 cell types from all 6 scRNA-seq datasets and each cell type has 5000 samples if the parameters was set by `sc_dataset_id = ['all'], n_sample_cell_type = 5000`;
- generated_frac_xx_.csv: Cell fraction of each cell type per simulated bulk sample;
- simu_bulk_exp_xx_selected_cell_id.csv: Cell id of cells in merged_6_sc_dataset which were used in this simulated bulk data;
- simu_bulk_exp_xx_CPM.txt [optional]: GEPs of simulated bulk data were given in counts per million (CPM) if `save_tpm=True`.
- simu_bulk_exp_xx_log2cpm1p.csv: GEPs of simulated bulk data were given in log2(CPM+1). 
- simu_bulk_exp_xx_log2cpm1p.h5ad: Contains the information of both gene expression ( given in  log2(CPM+1) ) and cell fraction of each cell type.

Example files: You can downloda this files from [our datasets file]() (location: datasets/simulated_bulk_cell_dataset).







## Data Processing

We provide function `read_counts2tpm()`  to create a file of correct format from TCGA read count data(s), [See details on this function]().

### Example code

```python
from deside.bulk_cell import read_counts2tpm

read_counts2tpm(read_counts_file_path='path/ACC_htseq.counts.csv', file_name_prefix='ACC',
                annotation_file_path='path/gencode.gene.info.v22.tsv', result_dir='path/result/bulk_GEPs/')
```



### Input files

In this step, TCGA read counts data (htseq.counts) in a .csv file (separated by ",")  or .txt file (separated by "\t") should be prepared. This file has the shape of m×n, where m is the number of features (genes) and n is the number of samples. You can download an example file from [our datasets file]() (location: datasets\TCGA\merged_data). Besides, file `gencode.gene.info.v22.tsv` for .. is also need, you can download this file from [our datasets file]() (location: datasets\TCGA\gencode.gene.info.v22.tsv)

- xx.csv: TCGA read counts data. 
- gencode.gene.info.v22.tsv:	?

### Output files

You will get 3 files transfer from TCGA read counts data, details show below. `xx` in the following file name is same as the parameter `file_name_prefix` in function  `read_counts2tpm()`.

- xx_htseq.counts.csv:	TCGA read counts data(?).
- xx_TPM.csv: GEPs given in TPM.
- xx_log2tpm1p.csv: GEPs given in log2(TPM + 1).

Example files: You can download this file from [our results file]() (location: datasets\TCGA\tpm).
     



  

##  Directory tree structure of `datasets`

We provided the datasets file with following directory tree structure, [you can download this file here](). Our datasets file contains all input datas and some intermediate output datas in our process by used DeSide. 

```
datasets/
├── cancer_purity/
│   └── cancer_purity.csv
├── simulated_bulk_cell_dataset/
│   ├── generated_frac_**.csv
│   ├── simu_bulk_exp_**_log2cpm1p.csv
│   ├── simu_bulk_exp_**_log2cpm1p.h5ad
│   ├── simu_bulk_exp_**_selected_cell_id.csv
│   ├── sc_dataset/
│   │   ├── generated_10_cell_type_n5000_pdac_pengj_02_pdac_steele_04.h5ad
│   │   ├── generated_11_cell_type_n5000_all.h5ad
│   │   ├── generated_8_cell_type_n5000_hnscc_cillo_01_hnscc_puram_03.h5ad
│   │   └── generated_9_cell_type_n5000_luad_kim_05.h5ad
│   └── test_set/
│       ├── generated_frac_**.csv
│       ├── simu_bulk_exp_**_CPM.txt
│       ├── simu_bulk_exp_**_log2cpm1p.csv
│       ├── simu_bulk_exp_**_log2cpm1p.h5ad
│       └── simu_bulk_exp_**_selected_cell_id.csv
├── single_cell/
│   ├── count_by_cell_type_and_dataset2.csv
│   ├── merged_6_sc_datasets.h5ad
│   ├── merged_6_sc_datasets.rar
│   └── merged_single_cell_dataset_sample_info.csv
├── TCGA/
│   ├── gdc_sample_sheet_10_tumors.tsv
│   ├── gencode.gene.info.v22.tsv
│   ├── merged_data/
│   │   └── **/
│   │       └── merged_**_htseq.counts.csv
│   └── tpm/
│       └── **/
│            ├── **_htseq.counts.csv
│            ├── **_log2tpm1p.csv
│            └── **_TPM.csv
└── well_trained_model/
    └── **/
        └── **/
            ├── celltypes.txt
            ├── genes.txt
            ├── history_reg.csv
            ├── loss.png
            └── model_**.h5
```





## Directory tree structure of `results`

We provided the results file with following directory tree structure, [you can download this file here](). Our results file contains final results in our process by used DeSide. 

```
results/
├── predicted_cell_fraction/
│   └── **/
│       ├── cell_fraction_by_DeSide_**/
│       │   ├── cancer_purity_merged_DeSide_**_predicted_result.csv
│       │   ├── CD8A_vs_predicted_CD8 T_proportion.png
│       │   ├── CPE_vs_predicted_1-others_proportion.png
│       │   ├── CPE_vs_predicted_Cancer Cells_proportion.png
│       │   ├── pred_cell_frac_before_decon.png
│       │   └── y_predicted_result.csv
│       └── cell_fraction_by_Scaden_**/
│           ├── cancer_purity_merged_Scaden_**_predicted_result.csv
│           ├── CD8A_vs_predicted_CD8 T_proportion.png
│           ├── CPE_vs_predicted_1-others_proportion.png
│           ├── CPE_vs_predicted_Cancer Cells_proportion.png
│           ├── pred_by_model_m1024.csv
│           ├── pred_by_model_m256.csv
│           ├── pred_by_model_m512.csv
│           ├── pred_cell_frac_before_decon.png
│           └── y_predicted_result.csv
└── test_set_pred/
    └── Test_set**/
        ├── DeSide_**/
        │   ├── Cancer Cells_true_vs_predicted_1-others_proportion.png
        │   ├── Cancer Cells_true_vs_predicted_Cancer Cells_pred_proportion.png
        │   ├── CD8 T_true_vs_predicted_CD8 T_pred_proportion.png
        │   ├── CD8A_vs_predicted_CD8 T_true_proportion.png
        │   ├── model_performance_evaluation.csv
        │   ├── y_predicted_result.csv
        │   ├── y_true_vs_absolute_error_deside.png
        │   ├── y_true_vs_absolute_error_deside_1-others.png
        │   ├── y_true_vs_y_pred_deside.png
        │   └── y_true_vs_y_pred_deside_1-others.png
        └── Scaden_**/
            ├── Cancer Cells_true_vs_predicted_Cancer Cells_pred_proportion.png
            ├── CD8 T_true_vs_predicted_CD8 T_pred_proportion.png
            ├── CD8A_vs_predicted_CD8 T_true_proportion.png
            ├── model_performance_evaluation.csv
            ├── pred_by_model_m1024.csv
            ├── pred_by_model_m256.csv
            ├── pred_by_model_m512.csv
            ├── y_predicted_result.csv
            ├── y_true_vs_absolute_error_average.png
            ├── y_true_vs_absolute_error_m1024.png
            ├── y_true_vs_absolute_error_m256.png
            ├── y_true_vs_absolute_error_m512.png
            ├── y_true_vs_y_pred_average.png
            ├── y_true_vs_y_pred_m1024.png
            ├── y_true_vs_y_pred_m256.png
            └── y_true_vs_y_pred_m512.png
```



