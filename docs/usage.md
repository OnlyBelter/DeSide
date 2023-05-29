Usage
=====

How to use DeSide



****
This package consists of three main modules:

1.  Model Prediction
2.  Model Training
3.  Dataset Simulation

There are two ways to use DeSide. Firstly, you can use the provided pre-trained model to directly predict cell proportions, 
eliminating the need to train the model by yourself. Alternatively, you can sequentially execute the `Dataset Simulation` and `Model Training` modules, training the model from scratch. 
Subsequently, you can use the self-trained model to predict cell proportions.


## Model Prediction

Using the pre-trained model or self-trained model, you can predict cell proportions in bulk gene expression profiles (bulk GEPs) by using the [`deside_model.predict()`]() function.

To perform the deconvolution, you can use the following command:


```python
import os
import pandas as pd
from deside.utility import check_dir
from deside.decon_cf import DeSide


# bulk gene expression profiles (GEPs) in TPM format
bulk_tpm_file_path = './datasets/TCGA/tpm/LUAD/LUAD_TPM.csv'
bulk_tpm = pd.read_csv(bulk_tpm_file_path, index_col=0)

# create output directory
result_dir = './results'
y_pred_file_path = os.path.join(result_dir, 'y_pred.csv')
check_dir(result_dir)

# read pre-trained DeSide model
model_dir = './DeSide_model/'
deside_model = DeSide(model_dir=model_dir) 

# predict by pre-trained model
deside_model.predict(input_file=bulk_tpm_file_path, 
                     output_file_path=y_pred_file_path, 
                     exp_type='TPM', transpose=True,
                     scaling_by_sample=False, scaling_by_constant=True)
```

### Input files

In this part, you can use the whole result directory of `result_dir` in `Model Training` module as the input parameter of `model_dir`, 
or you can download the pre-trained model from [DeSide_model]()  
and use the real directory in your computer.** Besides, bulk GEPs need to be delivered which were given in transcripts per million (TPM) or log2(TPM + 1). 
It should be separated by ','  and saved in a `.csv` file. The shape of this file should be m×n, where m is the number of features (genes) and n is the number of samples. 
You can download an example file from [our datasets file]() (location: datasets\TCGA\tpm). 
If you have not this format of bulk GEPs file, DeSide can only provide functionality to convert TCGA read count data into the correct file format. 
Have a look at the [Data Processing]() section for instructions on how to use this function. 
`Data Processing 待确认`

- model_dir: a folder instead of a single file
  - For self-trained model: the whole result folder of `result_dir` in `Model Training` module.
  - For pre-trained model: it can be downloaded from [DeSide_model](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/DeSide_model)
- input_file: Bulk GEPs in `transcripts per million` (TPM) or `log2(TPM + 1)` format, separated by ',', and saved in a `.csv` file. [Example file]()


### Output files

```text
results  # The output folder of this example
|-- CD8A_vs_predicted_CD8 T_proportion.png  # The figure depicting the predicted CD8 T cell proportions and the expression values of marker gene CD8A
|-- pred_cell_prop_before_decon.png         # The figure depicting the predicted cell proportions for all cell types
`-- y_pred.csv                              # The file containing the predicted cell proportions
```

## Model Training

```python
import os
from deside.utility import check_dir, sorted_cell_types
from deside.decon_cf import DeSide

# create output directory
result_dir = './results'
check_dir(result_dir)

# using dataset D1 as the training set
training_set2file_path = {
    'D1': './datasets/simulated_bulk_cell_dataset/simu_bulk_exp_Mixed_N100K_D1.h5ad',
}

all_cell_types = sorted_cell_types

# set hyper-parameters of the DNN model
deside_parameters = {'architecture': ([100, 1000, 1000, 1000, 50],
                                      [0, 0, 0, 0.2, 0]),
                     'loss_function': 'mae+rmse',
                     'batch_normalization': False,
                     'last_layer_activation': 'sigmoid',
                     'learning_rate': 2e-5,
                     'batch_size': 128}

# remove cancer cell during training process
remove_cancer_cell = True

# set result folder to save DeSide model
model_dir = os.path.join(result_dir, 'DeSide_model')
log_file_path = os.path.join(result_dir, 'deside_running_log.txt')
deside_obj = DeSide(model_dir=model_dir, log_file_path=log_file_path)

# training DeSide model
# - training_set_file_path is a list, multiple datasets will be combined as one training set
deside_obj.train_model(training_set_file_path=[training_set2file_path['D1']], 
                       hyper_params=deside_parameters, cell_types=all_cell_types,
                       scaling_by_constant=True, scaling_by_sample=False,
                       remove_cancer_cell=remove_cancer_cell,
                       n_patience=100, n_epoch=3000, verbose=0)
```
### Input files

In this module, you can use the simulated bulk GEPs (training set) with the pre-defined cell proportion matrix to train the DeSide model. 
This file usually generated from the `Dataset Simulation` module.

- training_set_file_path: the file path of training set in `.h5ad` format.

The training set `D1` we used in this example can be downloaded from [simu_bulk_exp_Mixed_N100K_D1.h5ad.zip](https://figshare.com/articles/dataset/Dataset_D1/23047391/1) 

### Output files

There are 7 files in the result folder. The entire result folder, `DeSide_model`, can be utilized as the `model_dir` parameter in the `Model Prediction` part. 
```text
results  # The output folder of this example
|-- DeSide_model
|   |-- celltypes.txt       # Cell types included in the training set
|   |-- genes.txt           # Gene list included in the training set
|   |-- history_reg.csv     # The history of recorded loss values during the training process
|   |-- key_params.txt      # Key parameters of the model
|   |-- loss.png            # The figure depicting loss values over epochs
|   `-- model_DeSide.h5     # Saved model after training
`-- deside_running_log.txt  # Log file
```

## DeSide Simulation

### a. Using the single cell dataset we provided

In this module, you can synthesize bulk tumors based on the dataset `S1`.
This example synthesized 10,000 samples of bulk tumors as a demonstration about the generation and filtering steps.

```python
import os
import pandas as pd
from deside.utility import (check_dir, sorted_cell_types)
from deside.utility.read_file import ReadExp, ReadH5AD
from deside.simulation import (BulkGEPGenerator, get_gene_list_for_filtering,
                               filtering_by_gene_list_and_pca_plot)

# the list of single cell RNA-seq datasets
sc_dataset_ids = ['hnscc_cillo_01', 'pdac_pengj_02', 'hnscc_puram_03',
                  'pdac_steele_04', 'luad_kim_05', 'nsclc_guo_06', 'pan_cancer_07']

# the list of cancer types in the TCGA dataset
cancer_types = ['ACC', 'BLCA', 'BRCA', 'GBM', 'HNSC', 'LGG', 'LIHC', 'LUAD', 'PAAD', 'PRAD',
                'CESC', 'COAD', 'KICH', 'KIRC', 'KIRP', 'LUSC', 'READ', 'THCA', 'UCEC']

# the list of cell types
all_cell_types = sorted_cell_types

# parameters
# for gene-level filtering
gene_list_type = 'high_corr_gene_and_quantile_range'
gene_quantile_range = [0.05, 0.5, 0.95]  # gene-level filtering

# for GEP-level filtering
gep_filtering_quantile = (0.0, 0.95)  # GEP-level filtering, L1-norm threshold
n_base = 100  # averaging 100 GEPs sampled from S1 to synthesize 1 bulk GEP, used by S1 generation

# optional, if set a prior cell proportion range for each cell type, the GEP-filtering step will be faster, default is (0, 1)
# It can be set as cell_prop_prior = {'B Cells': (0, 0.25), 'CD4 T': (0, 0.25), 'CD8 T': (0, 0.25),
#                    'DC': (0, 0.1), 'Mast Cells': (0, 0.1), 'NK': (0, 0.1), 'Neutrophils': (0, 0.1),
#                    'Endothelial Cells': (0, 0.5), 'Fibroblasts': (0, 0.5), 'Macrophages': (0, 0.5),
#                    'Cancer Cells': (0, 1)}
cell_prop_prior = None
dataset2parameters = {
    'Mixed_N10K_segment': {
        'sc_dataset_ids': sc_dataset_ids,
        'cell_types': all_cell_types,
        'n_samples': 10000,
        'sampling_method': 'segment', # or `random` used by Scaden
        'filtering': True,
    }
}

for ds, params in dataset2parameters.items():
    if ('filtering' in params) and ('filtering_ref_types' not in params):
        if params['filtering']:
            params['filtering_ref_types'] = cancer_types
        else:
            params['filtering_ref_types'] = None

# Using TCGA as reference GEPs to filter synthetic GEPs 
tcga_data_dir = r'./datasets/TCGA/tpm/'  # input
tcga_merged_tpm_file_path = os.path.join(tcga_data_dir, 'merged_tpm.csv')
tcga2cancer_type_file_path = os.path.join(tcga_data_dir, 'tcga_sample_id2cancer_type.csv')

# the file path of the dataset `S1`
sct_dataset_file_path = './datasets/simu_bulk_exp_SCT_N10K_S1.h5ad'

# naming the file of filtered bulk cell dataset
q_names = ['q_' + str(int(q * 1000)/10) for q in gene_quantile_range]
replace_by = f'_filtered_by_{gene_list_type}.h5ad'

if 'quantile_range' in gene_list_type:
    replace_by = f'_filtered_by_{gene_list_type}_{q_names[0]}_{q_names[2]}.h5ad'

sampling_method2dir = {
    'segment': os.path.join('results', 'E3', '{}_{}ds_{}_n_base{}_median_gep'),
}

n_sc_datasets = len(sc_dataset_ids)
dataset2path = {}
log_file_path = './results/E3/DeSide_running_log.txt'

# for gene filtering
d2_dir = './datasets/simulated_bulk_cell_dataset/D2/'
high_corr_gene_file_path = os.path.join(d2_dir, f'gene_list_filtered_by_high_corr_gene.csv')
high_corr_gene_list = pd.read_csv(high_corr_gene_file_path)
high_corr_gene_list = high_corr_gene_list['gene_name'].to_list()

for dataset_name, params in dataset2parameters.items():
    if 'SCT' in dataset_name:
        pass  # using `S1` directly, omit this step here
    else:
        sampling_method = params['sampling_method']
        # the folder of simulated bulk cells
        simu_bulk_exp_dir = sampling_method2dir[sampling_method]
        if sampling_method in ['segment']:
            simu_bulk_exp_dir = simu_bulk_exp_dir.format(sampling_method, n_sc_datasets,
                                                         gep_filtering_quantile[1], n_base)
        else:  # 'random'
            simu_bulk_exp_dir = simu_bulk_exp_dir.format(sampling_method, n_sc_datasets, n_base)
        check_dir(simu_bulk_exp_dir)
        bulk_generator = BulkGEPGenerator(simu_bulk_dir=simu_bulk_exp_dir,
                                          merged_sc_dataset_file_path=None,
                                          cell_types=params['cell_types'],
                                          sc_dataset_ids=params['sc_dataset_ids'],
                                          bulk_dataset_name=dataset_name,
                                          sct_dataset_file_path=sct_dataset_file_path,
                                          check_basic_info=False,
                                          tcga2cancer_type_file_path=tcga2cancer_type_file_path)
        # GEP-filtering will be performed during this generation process
        generated_bulk_gep_fp = bulk_generator.generated_bulk_gep_fp
        dataset2path[dataset_name] = generated_bulk_gep_fp
        if not os.path.exists(generated_bulk_gep_fp):
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

        # gene-level filtering that depends on the high correlation genes and quantile range (each dataset itself)
        if params['filtering']:
            filtered_file_path = generated_bulk_gep_fp.replace('.h5ad', replace_by)
            if not os.path.exists(filtered_file_path):
                gene_list = high_corr_gene_list.copy()
                # get gene list, filtering, PCA and plot
                current_result_dir = os.path.join(simu_bulk_exp_dir, dataset_name)
                check_dir(current_result_dir)
                # the gene list file for current dataset
                if 'quantile_range' in gene_list_type:
                    gene_list_file_path = os.path.join(simu_bulk_exp_dir, dataset_name, f'gene_list_filtered_by_{gene_list_type}.csv')
                    gene_list_file_path = gene_list_file_path.replace('.csv', f'_{q_names[0]}_{q_names[2]}.csv')
                    if not os.path.exists(gene_list_file_path):
                        print(f'Gene list of {dataset_name} will be saved in: {gene_list_file_path}')
                        quantile_gene_list = get_gene_list_for_filtering(bulk_exp_file=generated_bulk_gep_fp,
                                                                         filtering_type='quantile_range',
                                                                         tcga_file=tcga_merged_tpm_file_path,
                                                                         quantile_range=gene_quantile_range,
                                                                         result_file_path=gene_list_file_path,
                                                                         q_col_name=q_names)
                    else:
                        print(f'Gene list file existed: {gene_list_file_path}')
                        quantile_gene_list = pd.read_csv(gene_list_file_path)
                        quantile_gene_list = quantile_gene_list['gene_name'].to_list()
                    # get the intersection of the two gene lists (high correlation genes and within quantile range)
                    gene_list = [gene for gene in gene_list if gene in quantile_gene_list]
                bulk_exp_obj = ReadH5AD(generated_bulk_gep_fp)
                bulk_exp = bulk_exp_obj.get_df()
                bulk_exp_cell_frac = bulk_exp_obj.get_cell_fraction()
                tcga_exp = ReadExp(tcga_merged_tpm_file_path, exp_type='TPM').get_exp()
                pc_file_name = f'both_TCGA_and_simu_data_{dataset_name}'
                pca_model_file_path = os.path.join(current_result_dir, f'{pc_file_name}_PCA_{gene_list_type}.joblib')
                pca_data_file_path = os.path.join(current_result_dir, f'{dataset_name}_PCA_with_TCGA_{gene_list_type}.csv')
                # save GEPs data by filtered gene list
                print('Filtering by gene list and PCA plot')
                filtering_by_gene_list_and_pca_plot(bulk_exp=bulk_exp, tcga_exp=tcga_exp, gene_list=gene_list,
                                                    result_dir=current_result_dir, n_components=2,
                                                    simu_dataset_name=dataset_name,
                                                    pca_model_name_postfix=gene_list_type,
                                                    pca_model_file_path=pca_model_file_path,
                                                    pca_data_file_path=pca_data_file_path,
                                                    h5ad_file_path=filtered_file_path,
                                                    cell_frac_file=bulk_exp_cell_frac,
                                                    figsize=(5, 5))
```

#### Input files
- merged_tpm.csv: Gene expression profiles (GEPs) of 19 cancer types in TCGA (TPM format). Download link: https://doi.org/10.6084/m9.figshare.23047547.v1
- tcga_sample_id2cancer_type.csv: An annotation file that contains the cancer type for each sample id of above 19 cancer types. Download link: https://raw.githubusercontent.com/OnlyBelter/DeSide_mini_example/main/datasets/TCGA/tpm/tcga_sample_id2cancer_type.csv
- simu_bulk_exp_SCT_N10K_S1.h5ad: Dataset S1, which contains the synthesized single-cell GEPs (scGEPs). Download link: https://doi.org/10.6084/m9.figshare.23043560.v1

#### Output files
```text
results/E3  # the output folder of this example
|-- DeSide_running_log.txt  # log file
`-- segment_7ds_0.95_n_base100_median_gep
    |-- Mixed_N10K_segment
    |   |-- Mixed_N10K_segment_PCA_with_TCGA_high_corr_gene_and_quantile_range.csv  # Values of the first two principal components (PCs)
    |   |-- Mixed_N10K_segment_PCA_with_TCGA_high_corr_gene_and_quantile_range_PC0_PC1.png  # Visualization of PCA for the generated dataset and TCGA
    |   `-- gene_list_filtered_by_high_corr_gene_and_quantile_range_q_5.0_q_95.0.csv  # Gene list after filtering by correlation and quantile range
    |-- generated_frac_Mixed_N10K_segment.csv  # Cell proportion matrix
    |-- simu_bulk_exp_Mixed_N10K_segment_log2cpm1p.csv  # Generated bulk gene expression profiles (GEPs) without filtering (csv format)
    |-- simu_bulk_exp_Mixed_N10K_segment_log2cpm1p.h5ad  # Generated bulk GEPs without filtering (h5ad format)
    |-- simu_bulk_exp_Mixed_N10K_segment_log2cpm1p_filtered_by_high_corr_gene_and_quantile_range_q_5.0_q_95.0.h5ad  # Generated bulk GEPs after filtering (h5ad format)
    `-- simu_bulk_exp_Mixed_N10K_segment_sampled_sc_cell_id.csv  # Selected single-cell GEPs from dataset S1 during GEP sampling
```

### b. Preparing single cell dataset by yourself

If you want to use other scRNA-seq datasets to simulate GEPs, you can follow our workflow to preprocess single cell datasets and merge them together. The Python package `Scanpy` was used heavily in our workflow.

- Preprocessing a single dataset: [Here is an example of how i process my scRNA-seq dataset(s)]().
- Merging multiple datasets together: [Here is an example of how i process my scRNA-seq dataset(s)]().


## Data Processing

We provide function `read_counts2tpm()`  to create a file of correct format from TCGA read count data(s), [See details on this function]().

### Example code

```python
from deside.bulk_cell import read_counts2tpm

read_counts2tpm(read_counts_file_path='path/ACC_htseq.counts.csv', file_name_prefix='ACC',
                annotation_file_path='path/gencode.gene.info.v22.tsv', result_dir='path/result/bulk_GEPs/')
```



### Input files

In this step, TCGA read counts data (htseq.counts) in a .csv file (separated by ",")  or .txt file (separated by "\t") should be prepared. 
This file has the shape of `m×n`, where `m` is the number of features (genes) and `n` is the number of samples. 
You can download an example file from [our datasets file]() (location: datasets\TCGA\merged_data). Besides, file `gencode.gene.info.v22.tsv` for .. is also need, you can download this file from [our datasets file]() (location: datasets\TCGA\gencode.gene.info.v22.tsv)

- xx.csv: TCGA read counts data. 
- gencode.gene.info.v22.tsv:	?

### Output files

You will get 3 files transfer from TCGA read counts data, details show below. `xx` in the following file name is same as the parameter `file_name_prefix` in function  `read_counts2tpm()`.

- xx_htseq.counts.csv:	TCGA read counts data(?).
- xx_TPM.csv: GEPs given in TPM.
- xx_log2tpm1p.csv: GEPs given in log2(TPM + 1).

Example files: You can download this file from [our results file]() (location: datasets\TCGA\tpm).
