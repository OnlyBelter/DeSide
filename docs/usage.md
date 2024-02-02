Usage
=====
Usage of DeSide package is demonstrated.

***
This package consists of three main modules:

-  Utility
-  DeSide model
-  Dataset Simulation


## Utility
This module contains some utility functions, including:
- Bulk RNA-seq data pre-processing
- Single cell RNA-seq data pre-processing
- Read RNA-seq datasets with different formats and format conversion
- Plotting

### Bulk RNA-seq data pre-processing

We provide the function [`read_counts2tpm`](https://deside.readthedocs.io/en/latest/func/bulk_cell.html#deside.bulk_cell.read_counts2tpm)  to convert gene expression profiles (GEPs) to transcripts per million (TPM) format from read counts.

```python
from deside.bulk_cell import read_counts2tpm

read_counts2tpm(read_counts_file_path='path/xx_htseq.counts.csv', file_name_prefix='xx',
                annotation_file_path='path/gencode.gene.info.v22.tsv', result_dir='path/result/bulk_GEPs/')
```

#### Input files

- xx_htseq.counts.csv: read counts data downloaded from TCGA. [Example file](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/datasets/TCGA/tpm/LUAD/LUAD_TPM.csv) 
- gencode.gene.info.v22.tsv: the gene annotation file, which contains `exon_length` for each gene. [Download link](https://api.gdc.cancer.gov/data/b011ee3e-14d8-4a97-aed4-e0b10f6bbe82)

#### Output files

- xx_htseq.counts.csv: read counts data after filtering, only `protein_coding` genes are retained.
- xx_TPM.csv: GEPs given in TPM.
- xx_log2tpm1p.csv: GEPs given in `log2(TPM + 1)`.

### Single cell RNA-seq data pre-processing
The demonstration of single cell RNA-seq dataset pre-processing in this study shown in jupyter notebooks can be found: 
[single cell dataset integration](https://github.com/OnlyBelter/DeSide_mini_example/tree/main/single_cell_dataset_integration).

### Read RNA-seq datasets with different formats and format conversion
This package provides functions to read GEPs with different formats and convert between them.

- [`log_exp2cpm`](https://deside.readthedocs.io/en/latest/func/utility.html#deside.utility.log_exp2cpm): convert GEPs given in log2(CPM + 1) to CPM.
- [`non_log2cpm`](https://deside.readthedocs.io/en/latest/func/utility.html#deside.utility.non_log2cpm): normalize gene expression values to CPM / TPM from non-log space.
- [`non_log2log_cpm`](https://deside.readthedocs.io/en/latest/func/utility.html#deside.utility.non_log2log_cpm): convert gene expression values to log2(CPM + 1) from non-log space.
- [`read_data_from_h5ad`](https://deside.readthedocs.io/en/latest/func/utility.html#deside.utility.read_data_from_h5ad): read GEPs from `.h5ad` file.
- [`ReadExp`](https://deside.readthedocs.io/en/latest/func/utility.html#deside.utility.read_file.ReadExp): read GEPs from a `.csv` file or `pandas.DataFrame` and convert to specific format.
- [`ReadH5AD`](https://deside.readthedocs.io/en/latest/func/utility.html#deside.utility.read_file.ReadH5AD): read GEPs from a `.h5ad` file. Cell proportion matrix and gene expression values can be read separately from the simulated dataset.


`CPM` means counts per million and usually is used in scRNA-seq data, while `TPM` means transcripts per million and usually is used in bulk RNA-seq data.

#### An example of reading cell proportion matrix and gene expression values from a simulated dataset

```python
from deside.utility.read_file import ReadH5AD

h5ad_obj = ReadH5AD(file_path='path/xx.h5ad', show_info=True)
cell_proportion_matrix = h5ad_obj.get_cell_fraction()
gene_expression_values = h5ad_obj.get_df()
```
### Plotting

This package provides functions to plot the results of DeSide.
- [`plot_corr_two_columns`](https://deside.readthedocs.io/en/latest/func/plot.html#deside.plot.plot_corr_two_columns): plot the correlation between two columns in a dataframe.
- [`plot_predicted_result`](https://deside.readthedocs.io/en/latest/func/plot.html#deside.plot.plot_predicted_result): plot and evaluate the predicted cancer cell proportions for TCGA data by comparing with CPE values (Aran, D. et al., Nat Commun 6, 8971 (2015), Supplementary Data 1).



## DeSide model
There are two ways to use DeSide. 
Firstly, you can use the provided pre-trained model to predict cell proportions directly, 
eliminating the need to train the model by yourself. 
Alternatively, you can sequentially execute the `Dataset Simulation` and `Model Training` modules, training the model from scratch. 
Then use the self-trained model to predict cell proportions.

### Model Prediction

Using the pre-trained model or self-trained model, you can predict cell proportions in bulk gene expression profiles (bulk GEPs) by using the [`deside_model.predict`](https://deside.readthedocs.io/en/latest/func/deconvolution.html#deside.decon_cf.DeSide.predict) function.

```python
import os
import pandas as pd
from deside.utility import check_dir
from deside.decon_cf import DeSide
from deside.utility.read_file import read_gene_set

# bulk gene expression profiles (GEPs) in TPM format
bulk_tpm_file_path = 'path/xx_TPM.csv'
bulk_tpm = pd.read_csv(bulk_tpm_file_path, index_col=0)

# create output directory
result_dir = './results'
y_pred_file_path = os.path.join(result_dir, 'y_pred.csv')
check_dir(result_dir)
dataset_dir = './datasets/'

# hyper-parameters of the DNN model
deside_parameters = {
    'architecture': ([200, 2000, 2000, 2000, 50], [0.05, 0.05, 0.05, 0.2, 0]),
    'architecture_for_pathway_network': ([50, 500, 500, 500, 50], [0, 0, 0, 0, 0]),
    'loss_function_alpha': 0.5,  # alpha*mae + (1-alpha)*rmse, mae means mean absolute error
    'normalization': 'layer_normalization',  # batch_normalization / layer_normalization / None
     # 1 means to add a normalization layer, input | the first hidden layer | ... | output
    'normalization_layer': [0, 0, 1, 1, 1, 1],  # 1 more parameter than the number of hidden layers
    'pathway_network': True,  # using an independent pathway network
    'last_layer_activation': 'sigmoid',  # sigmoid / softmax
    'learning_rate': 1e-4,
    'batch_size': 128}

# read two gene sets as pathway mask
gene_set_file_path1 = os.path.join(dataset_dir, 'gene_set', 'c2.cp.kegg.v2023.1.Hs.symbols.gmt')
gene_set_file_path2 = os.path.join(dataset_dir, 'gene_set', 'c2.cp.reactome.v2023.1.Hs.symbols.gmt')
all_pathway_files = [gene_set_file_path1, gene_set_file_path2]
pathway_mask = read_gene_set(all_pathway_files)  # genes by pathways

# read pre-trained DeSide model
model_dir = './DeSide_model/'
deside_model = DeSide(model_dir=model_dir) 

# predict by pre-trained model
deside_model.predict(input_file=bulk_tpm_file_path, 
                     output_file_path=y_pred_file_path, 
                     exp_type='TPM', transpose=True,
                     scaling_by_sample=False, scaling_by_constant=True,
                     hyper_params=deside_parameters,
                     pathway_mask=pathway_mask)
```
- A complete example in jupyter notebook can be found: [E1 - Using pre-trained model.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/E1%20-%20Using%20pre-trained%20model.ipynb).

### Model Training

Training a model using the provided training set.
```python
import os
import pandas as pd
from deside.decon_cf import DeSide
from deside.utility import check_dir, sorted_cell_types
from deside.utility.read_file import read_gene_set

# create output directory
result_dir = './results'
check_dir(result_dir)
dataset_dir = './datasets/'

# using dataset D1 as the training set
training_set2file_path = {
    'D1': './datasets/simulated_bulk_cell_dataset/D1/simu_bulk_exp_Mixed_N100K_D1.h5ad',
}

cell_type2subtypes = {'B Cells': ['Non-plasma B cells', 'Plasma B cells'],
                      'CD4 T': ['CD4 T'], 'CD8 T': ['CD8 T (GZMK high)', 'CD8 T effector'],
                      'DC': ['DC'], 'Endothelial Cells': ['Endothelial Cells'],
                      'Cancer Cells': ['Cancer Cells'],
                      'Fibroblasts': ['CAFs', 'Myofibroblasts'], 'Macrophages': ['Macrophages'],
                      'Mast Cells': ['Mast Cells'], 'NK': ['NK'], 'Neutrophils': ['Neutrophils'],
                      'Double-neg-like T': ['Double-neg-like T'], 'Monocytes': ['Monocytes']}
all_cell_types = sorted([i for v in cell_type2subtypes.values() for i in v])
all_cell_types = [i for i in sorted_cell_types if i in all_cell_types]

# set hyper-parameters of the DNN model and other parameters for training
# hyper-parameters of the DNN model
deside_parameters = {
    'architecture': ([200, 2000, 2000, 2000, 50], [0.05, 0.05, 0.05, 0.2, 0]),
    'architecture_for_pathway_network': ([50, 500, 500, 500, 50], [0, 0, 0, 0, 0]),
    'loss_function_alpha': 0.5,  # alpha*mae + (1-alpha)*rmse, mae means mean absolute error
    'normalization': 'layer_normalization',  # batch_normalization / layer_normalization / None
     # 1 means to add a normalization layer, input | the first hidden layer | ... | output
    'normalization_layer': [0, 0, 1, 1, 1, 1],  # 1 more parameter than the number of hidden layers
    'pathway_network': True,  # using an independent pathway network
    'last_layer_activation': 'sigmoid',  # sigmoid / softmax
    'learning_rate': 1e-4,
    'batch_size': 128}

# read two gene sets as pathway mask
gene_set_file_path1 = os.path.join(dataset_dir, 'gene_set', 'c2.cp.kegg.v2023.1.Hs.symbols.gmt')
gene_set_file_path2 = os.path.join(dataset_dir, 'gene_set', 'c2.cp.reactome.v2023.1.Hs.symbols.gmt')
all_pathway_files = [gene_set_file_path1, gene_set_file_path2]
pathway_mask = read_gene_set(all_pathway_files)  # genes by pathways

# filtered gene list (gene-level filtering, filtered by correlation coefficients and quantiles)
filtered_gene_list = None  # for other datasets
if list(training_set2file_path.keys())[0] == 'D1':
    filtered_gene_file_path = os.path.join(dataset_dir, 'simulated_bulk_cell_dataset/D1/gene_list_filtered_by_high_corr_gene_and_quantile_range.csv')
    filtered_gene_list = pd.read_csv(filtered_gene_file_path, index_col=0).index.to_list()

# input gene list type for pathway profiles
input_gene_list = 'filtered_genes'

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
                       n_patience=100, n_epoch=3000, verbose=0,
                        pathway_mask=pathway_mask, method_adding_pathway='add_to_end', 
                        filtered_gene_list=filtered_gene_list, input_gene_list=input_gene_list)
```
- A complete example in jupyter notebook can be found: [E2 - Training a model from scratch.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/E2%20-%20Training%20a%20model%20from%20scratch.ipynb)

## Dataset Simulation

### a. Using the single cell dataset we provided

In this module, you can synthesize bulk tumors based on the dataset `S1`.

```python
import os
import pandas as pd
from deside.utility.read_file import ReadH5AD, ReadExp
from deside.utility import check_dir, sorted_cell_types
from deside.simulation import (BulkGEPGenerator, get_gene_list_for_filtering, 
                               filtering_by_gene_list_and_pca_plot)

# the list of single cell RNA-seq datasets
sc_dataset_ids = ['hnscc_cillo_01', 'pdac_pengj_02', 'hnscc_puram_03',
                  'pdac_steele_04', 'luad_kim_05', 'nsclc_guo_06', 
                  'pan_cancer_07', 'prad_cheng_08', 'prad_dong_09', 
                  'hcc_sun_10', 'gbm_neftel_11', 'gbm_abdelfattah_12']

# the list of cancer types in the TCGA dataset
cancer_types = ['ACC', 'BLCA', 'BRCA', 'GBM', 'HNSC', 'LGG', 'LIHC', 'LUAD', 'PAAD', 'PRAD',
                'CESC', 'COAD', 'KICH', 'KIRC', 'KIRP', 'LUSC', 'READ', 'THCA', 'UCEC']

cancer_types_for_filtering = cancer_types.copy()

# coefficient to correct the difference of total RNA abundance in different cell types
# There is no effect to the final results if all the coefficients are set to 1
alpha_total_rna_coefficient = {'B Cells': 1.0, 'CD4 T': 1.0, 'CD8 T': 1.0, 'DC': 1.0,
                               'Endothelial Cells': 1.0, 'Cancer Cells': 1.0, 'Fibroblasts': 1.0,
                               'Macrophages': 1.0, 'Mast Cells': 1.0, 'NK': 1.0, 'Neutrophils': 1.0,
                               'Double-neg-like T': 1.0, 'Monocytes': 1.0}

# cell types and the corresponding subtypes
cell_type2subtypes = {'B Cells': ['Non-plasma B cells', 'Plasma B cells'],
                      'CD4 T': ['CD4 T'], 'CD8 T': ['CD8 T (GZMK high)', 'CD8 T effector'],
                      'DC': ['DC'], 'Endothelial Cells': ['Endothelial Cells'],
                      'Cancer Cells': ['Cancer Cells'],
                      'Fibroblasts': ['CAFs', 'Myofibroblasts'], 'Macrophages': ['Macrophages'],
                      'Mast Cells': ['Mast Cells'], 'NK': ['NK'], 'Neutrophils': ['Neutrophils'],
                      'Double-neg-like T': ['Double-neg-like T'], 'Monocytes': ['Monocytes']}

# the list of cell types
all_cell_types = sorted([i for v in cell_type2subtypes.values() for i in v])
all_cell_types = [i for i in sorted_cell_types if i in all_cell_types]

# parameters
# for gene-level filtering
gene_list_type = 'high_corr_gene_and_quantile_range'
gene_quantile_range = [0.005, 0.5, 0.995]  # gene-level filtering

# for GEP-level filtering
gep_filtering_quantile = (0.0, 0.95)  # GEP-level filtering, L1-norm threshold
filtering_in_pca_space = True
pca_n_components = 0.9
n_base = 100  # averaging 100 GEPs sampled from S1 to synthesize 1 bulk GEP, used by S1 generation

cell_prop_prior = None
dataset2parameters = {
    'Mixed_N10K_segment': {
        'sc_dataset_ids': sc_dataset_ids,
        'cell_type2subtype': cell_type2subtypes,
        'n_samples': 8000,
        'sampling_method': 'segment', # or `random` used by Scaden
        'filtering': True,
    }
}

# skipped steps here ...
simu_bulk_exp_dir = './datasets/simulated_bulk_cell_dataset'
sct_dataset_file_path = 'path/to/simu_bulk_exp_SCT_N10K_S1_16sct.h5ad'
tcga2cancer_type_file_path = 'path/to/tcga_sample_id2cancer_type.csv'
tcga_merged_tpm_file_path = 'path/to/merged_tpm.csv'
high_corr_gene_file_path = 'path/to/gene_list_filtered_by_high_corr_gene.csv'
high_corr_gene_list = pd.read_csv(high_corr_gene_file_path)
high_corr_gene_list = high_corr_gene_list['gene_name'].to_list()
for dataset_name, params in dataset2parameters.items():
    # skipped steps here ...
    bulk_generator = BulkGEPGenerator(simu_bulk_dir=simu_bulk_exp_dir,
                                      merged_sc_dataset_file_path=None,
                                      cell_type2subtype=params['cell_type2subtype'],
                                      sc_dataset_ids=params['sc_dataset_ids'],
                                      bulk_dataset_name=dataset_name,
                                      sct_dataset_file_path=sct_dataset_file_path,
                                      check_basic_info=False,
                                      tcga2cancer_type_file_path=tcga2cancer_type_file_path,
                                      total_rna_coefficient=alpha_total_rna_coefficient,
                                      cell_type_col_name='cell_type',
                                      subtype_col_name='cell_type')
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
                                    cell_prop_prior=cell_prop_prior,
                                    filtering_in_pca_space=filtering_in_pca_space,
                                    norm_ord=1, pca_n_components=pca_n_components)

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
- This example synthesized 10,000 samples of bulk tumors as a demonstration about the generation and filtering steps. The complete example in jupyter notebook can be found: [E3 - Synthesizing bulk tumors.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/E3%20-%20Synthesizing%20bulk%20tumors.ipynb)

### b. Preparing single cell dataset by yourself

If you want to use other scRNA-seq datasets to simulate GEPs, you can follow our workflow to preprocess single cell datasets and merge them together. The Python package `Scanpy` was used heavily in our workflow.

- Preprocessing a single dataset: [03deal_with_Puram et al Cell.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/single_cell_dataset_integration/03deal_with_Puram%20et%20al%20Cell.ipynb).
- Merging multiple datasets together (part 1): [Merge_12_scRNA-seq_datasets_part1.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/single_cell_dataset_integration/Merge_12_scRNA-seq_datasets_part1.ipynb).
- Merging multiple datasets together (part 2-round1): [Merge_12_scRNA-seq_datasets_part2_first_round.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/single_cell_dataset_integration/Merge_12_scRNA-seq_datasets_part2_first_round.ipynb)
- Merging multiple datasets together (part 2-round2): [Merge_12_scRNA-seq_datasets_part2_second_round.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/single_cell_dataset_integration/Merge_12_scRNA-seq_datasets_part2_second_round.ipynb)
