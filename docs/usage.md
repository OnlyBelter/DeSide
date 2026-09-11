# Usage

Usage of DeSide package is demonstrated.

***

This package consists of three main modules:

- Utility
- DeSide model
- Dataset Simulation

> [!NOTE]
> As of the major upgrade on branch `pytorch-dev`, DeSide uses PyTorch and
> PyTorch Lightning internally instead of TensorFlow/Keras. Public training
> and prediction entry points remain stable, but saved model files change from
> `.h5` to a Lightning checkpoint directory. The TensorFlow-based 1.x line is
> still available as a legacy install documented in the installation page.

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

Using the pre-trained model or self-trained model, you can predict cell proportions in bulk gene expression profiles (bulk GEPs).
For Example 1, the recommended entry point is the one-call helper
[`predict_with_pretrained_model`](https://deside.readthedocs.io/en/latest/func/deconvolution.html#deside.decon_cf.predict_with_pretrained_model),
which wraps the existing `DeSide.predict()` workflow with the default hyper-parameters and pathway files used by the provided pre-trained model.

```python
import deside

# bulk gene expression profiles (GEPs) in TPM format
bulk_tpm_file_path = 'path/xx_TPM.csv'

deside.predict_with_pretrained_model(
    input_file=bulk_tpm_file_path,
    output_file_path='./results/y_pred.csv'
)
```

- The helper above expects the same Example 1 assets used in the mini example:
  - `./DeSide_model/` containing the pre-trained model files
  - `./datasets/gene_set/` containing the two pathway `.gmt` files
- By default, missing Example 1 assets will be downloaded automatically into `model_dir` and `dataset_dir`.
- Each download step will be printed explicitly, including destination path, file size, and md5.
- To disable auto-download (e.g. in offline environments), pass `auto_download=False`.
- If your files are stored elsewhere, pass `model_dir='path/DeSide_model'` and/or `dataset_dir='path/datasets'`.
- The helper uses the same defaults as the original notebook: `exp_type='TPM'`, `transpose=True`,
  `scaling_by_sample=False`, and `scaling_by_constant=True`.
- If you need full control over hyper-parameters or a custom pathway mask, you can still use
  [`DeSide.predict`](https://deside.readthedocs.io/en/latest/func/deconvolution.html#deside.decon_cf.DeSide.predict) directly.
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

This module now has a standalone, config-driven workflow for sctGEP generation,
mixed bulk GEP generation, and the two filtering stages. The recommended entry
point is the YAML configuration file
`deside/configs/example_bulk_simulation_config.yaml` together with the CLI
command `deside workflow filter-sim-data`.

### a. Using the single cell dataset we provided

This workflow reproduces the logic of the mini example while reducing the
manual setup. If `input.sct_dataset_file_path` is empty, DeSide first
bootstraps the single-cell-type reference dataset with
`SingleCellTypeGEPGenerator`, then reuses that generated `.h5ad` for the
existing mixed-bulk `BulkGEPGenerator` step.

The updated standalone workflow runs in three stages:

1. Resolve the sctGEP reference.
   - If `input.sct_dataset_file_path` is an existing `.h5ad` file, the
     workflow reuses it directly.
   - If `input.sct_dataset_file_path` is `''`, the workflow generates the
     sctGEP reference from `input.merged_sc_dataset_file_path` by using
     `sct_generation.*`.
   - If `input.sct_dataset_file_path` is set but the file is missing, the
     workflow raises an error.
2. Generate mixed bulk GEPs.
   - The workflow runs the legacy `BulkGEPGenerator.generate_gep(...)`
     pipeline with `simulation.*` and `gep_filtering.*`.
3. Apply gene-level filtering.
   - If `gene_filtering.enable: true`, the workflow derives the filtered gene
     list and optionally saves a filtered `.h5ad`, PCA outputs, and summary
     files.

To run the example workflow, make sure you have the required inputs in place:

1. Put the merged single-cell dataset at the path used by
   `input.merged_sc_dataset_file_path`.
2. Put the TCGA merged TPM matrix at the path used by
   `gep_filtering.reference_file` and `gene_filtering.tcga_file`.
3. Put the TCGA cancer-type annotation file at the path used by
   `input.tcga2cancer_type_file_path`.
4. Run the standalone workflow:

```bash
deside workflow filter-sim-data \
  --config deside/configs/example_bulk_simulation_config.yaml
```

The example config keeps `input.sct_dataset_file_path` empty:

```yaml
input:
  merged_sc_dataset_file_path: './datasets/generated_sc_dataset/merged_sc_dataset_log2cpm1p.h5ad'
  sct_dataset_file_path: ''
```

This means the workflow bootstraps the S1-style sctGEP dataset automatically
from the merged S0 dataset before generating the mixed bulk dataset
`Mixed_N10K_segment`.

<!-- prettier-ignore -->
> [!IMPORTANT]
> When `input.sct_dataset_file_path` is empty, `input.merged_sc_dataset_file_path`
> must point to an existing merged single-cell dataset. The workflow does not
> silently fall back to another source.

After the run completes, the workflow writes:

- The resolved sctGEP `.h5ad` file.
- The generated mixed bulk `.h5ad` file.
- The generated cell-fraction `.csv` file.
- The optional filtered bulk `.h5ad` file.
- The optional filtered gene-list `.csv` file.
- The summary JSON file `bulk_simulation_summary.json`.

### Example configuration

The file `deside/configs/example_bulk_simulation_config.yaml` is aligned with
the mini example notebook
[E3 - Synthesizing bulk tumors.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/E3%20-%20Synthesizing%20bulk%20tumors.ipynb),
but it uses the new auto-bootstrap workflow instead of requiring a prebuilt S1
file.

The current example configuration does the following:

- Uses the 12 single-cell datasets from the mini example.
- Generates an S1-style sctGEP dataset named `SCT_N10K_S1_16sct`.
- Generates `8000` mixed bulk samples with `sampling_method: 'segment'`.
- Uses `simu_method: 'mul'` for mixed-bulk synthesis.
- Applies TCGA-guided GEP-level filtering across 19 cancer types.
- Applies PCA-space filtering with `pca_n_components: 0.9` and `norm_ord: 1`.
- Applies gene-level filtering with
  `filtering_type: 'high_corr_gene_and_quantile_range'`.
- Uses the mini example quantile range `[0.005, 0.5, 0.995]`.

### Configuration reference

This section explains the main parameters in
`deside/configs/example_bulk_simulation_config.yaml` and highlights the common
alternatives you can use.

#### input

The `input` section defines the datasets and metadata needed before simulation
starts.

- `merged_sc_dataset_file_path`
  - Points to the merged single-cell reference dataset used for sctGEP
    bootstrapping.
  - This file is required when `sct_dataset_file_path: ''`.
- `sct_dataset_file_path`
  - Set this to `''` to auto-generate the sctGEP reference.
  - Set this to an existing `.h5ad` path to reuse a previously generated or
    downloaded sctGEP dataset.
- `tcga2cancer_type_file_path`
  - Points to the TCGA sample-to-cancer-type mapping file used by GEP-level
    filtering.
- `cell_type2subtype`
  - Defines which cell types and subtypes are included in simulation.
  - Keep a single subtype equal to the parent cell type, for example
    `CD4 T: ['CD4 T']`, when you don't want subtype splitting.
- `sc_dataset_ids`
  - Lists the merged single-cell datasets that can contribute cells during
    simulation.
  - Reduce this list if you want to restrict the reference to a smaller cohort.
- `total_rna_coefficient`
  - Sets optional per-cell-type RNA abundance correction factors.
  - Use all `1.0` values to keep the unadjusted legacy behavior.
- `cell_type_col_name` and `subtype_col_name`
  - Define which columns in the merged single-cell `obs` table hold the cell
    type and subtype labels.

#### output

The `output` section controls where workflow artifacts are written.

- `simu_bulk_dir`
  - Root directory for generated sctGEP, mixed bulk, and filtering outputs.
- `bulk_dataset_name`
  - Name used to build the mixed-bulk output filenames.
- `log_file_path`
  - Optional workflow log file.
- `summary_file_path`
  - Leave this empty to save the summary to
    `<gene_filtering_result_dir>/bulk_simulation_summary.json`.
- `gene_filtering_result_dir`
  - Leave this empty to save filtering outputs under
    `<simu_bulk_dir>/<bulk_dataset_name>/`.

#### simulation

The `simulation` section controls mixed-bulk generation.

- `n_samples`
  - Number of mixed bulk GEPs to generate.
- `sampling_method`
  - Supported choices are `'segment'`, `'random'`, `'fragment'`, and
    `'dirichlet'`.
  - The mini example uses `'segment'`.
- `sampling_range`
  - Optional per-cell-type sampling ranges. Leave this empty to use the legacy
    defaults of the selected sampling method.
- `n_threads`
  - Number of worker threads used during cell sampling.
- `simu_method`
  - `'mul'` uses the sctGEP reference for mixed-bulk generation.
  - Other legacy methods still exist, but the example workflow uses `'mul'`.
- `add_noise` and `noise_params`
  - Control optional noise injection into simulated GEPs.
  - Leave them disabled unless you need explicit perturbations.
- `cell_prop_prior`
  - Optionally constrains the cell-fraction search space.
  - This can speed up GEP-level filtering when you already know realistic
    ranges for specific cell types.

#### sct_generation

The `sct_generation` section is used only when
`input.sct_dataset_file_path: ''`.

- `sct_dataset_name`
  - Naming prefix for the generated sctGEP dataset.
- `n_sample_each_cell_type`
  - Number of positive sctGEP samples generated per cell type.
- `n_base_for_positive_samples`
  - Number of single cells averaged to create one positive sctGEP sample.
  - The mini example uses `100`.
- `sample_type`
  - `'positive'` generates one-cell-type samples.
  - `'negative'` generates mixed-cell-type samples for the sctGEP stage.
- `sep_by_patient`
  - Set this to `true` to sample cells from one patient at a time.
- `simu_method`
  - The example uses `'ave'` for the sctGEP stage.
- `test_set`
  - Set this to `true` to generate the legacy SCT test-set layout instead of
    the training-style dataset.
- `minimum_n_base`
  - Minimum number of cells required for a cell type to participate in
    sctGEP generation.
- `ref_gene_list_file_path`
  - Optional reference gene list used to align the merged single-cell dataset
    before sctGEP generation.
- `output_file_path`
  - Optional explicit path for the generated sctGEP `.h5ad` file.
  - Leave this empty to use the deterministic path under `simu_bulk_dir`.

#### gep_filtering

The `gep_filtering` section controls TCGA-guided sample-level filtering during
mixed-bulk generation.

- `enable`
  - Set this to `false` to skip GEP-level filtering entirely.
- `reference_file`
  - Points to the TCGA expression matrix used as the filtering reference.
- `ref_exp_type`
  - Expression scale of `reference_file`. The example uses `'TPM'`.
- `filtering_method`
  - Supported choices are `'median_gep'`, `'mean_gep'`, `'linear_mmd'`, and
    `'marker_ratio'`.
  - The mini example uses `'median_gep'`.
- `filtering_ref_types`
  - Lists the TCGA cancer types used as the reference cohort.
  - You can also set this to `['all']` to expand to every cancer type listed
    in `tcga2cancer_type_file_path`.
- `gep_filtering_quantile`
  - Lower and upper quantiles used to define the acceptance band during
    filtering.
  - Use `null` for an open lower bound when needed.
- `n_top`
  - Number of top features used by marker-ratio filtering.
- `show_filtering_info`
  - Prints more detailed filtering diagnostics when `true`.
- `high_corr_gene_list_file`
  - Optional precomputed high-correlation gene list used only as the feature
    space for GEP-level filtering.
- `filtering_by_gene_range`
  - Enables additional gene-range checks during GEP filtering.
- `min_percentage_within_gene_range`
  - Minimum fraction of genes that must stay within the TCGA range when
    `filtering_by_gene_range: true`.
- `gene_quantile_range`
  - Lower, center, and upper TCGA quantiles used by the gene-range check.
- `filtering_in_pca_space`
  - Set this to `true` to filter in PCA space instead of the original
    expression space.
- `pca_n_components`
  - Number of principal components if you set an integer, or explained
    variance ratio if you set a float.
- `norm_ord`
  - Norm order used for distance calculations. `1` means L1 distance, and `2`
    means Euclidean distance.

#### gene_filtering

The `gene_filtering` section controls post-generation gene-level filtering and
the optional filtered dataset.

- `enable`
  - Set this to `false` to keep the generated mixed-bulk dataset unchanged.
- `filtering_type`
  - Supported choices are `'high_corr_gene'`, `'quantile_range'`,
    `'all_genes'`, and `'high_corr_gene_and_quantile_range'`.
  - The example uses the intersection of high-correlation genes and
    quantile-range genes.
- `tcga_file`
  - Points to the TCGA TPM matrix used in gene filtering.
- `quantile_range`
  - Lower, center, and upper quantiles used to derive the quantile-range gene
    list.
- `q_col_name`
  - Column names corresponding to `quantile_range`.
  - Keep these names synchronized with the selected quantiles.
- `corr_threshold`
  - Correlation threshold used by the high-correlation gene filter.
- `n_gene_max`
  - Maximum number of genes selected per cell type during high-correlation
    filtering.
- `high_corr_gene_file`
  - Optional precomputed high-correlation gene list.
  - Leave this empty to compute the list inside the workflow.
- `save_filtered_h5ad`
  - Set this to `false` if you only want the filtered gene list.
- `filtered_dataset_postfix`
  - Suffix appended to the filtered dataset filename.
- `plot_pca`
  - Controls whether the workflow saves the PCA comparison plot for TCGA and
    simulated data.
- `pca_n_components` and `pca_figsize`
  - Control PCA output dimensionality and figure size.
- `gene_list_file`, `corr_result_file`, `pca_model_file`, `pca_data_file`,
  and `filtered_h5ad_file`
  - Optional explicit output paths. Leave them empty to use deterministic
    default filenames.

#### runtime

The `runtime` section controls workflow behavior rather than the simulation
design itself.

- `skip_if_done`
  - Reuses existing outputs when the expected files already exist.
- `check_basic_info`
  - Validates the single-cell reference metadata before generation starts.
- `zero_ratio_threshold`
  - Filters out single-cell profiles with too many zero-expression genes.
- `sc_dataset_gep_type`
  - Use `'log_space'` for log2-transformed single-cell inputs, or
    `'linear_space'` for non-log inputs.

### Legacy Python API

You can still call `SingleCellTypeGEPGenerator` and `BulkGEPGenerator`
directly from Python for custom workflows, but the config-driven standalone
workflow is now the recommended path for reproducing the mini example and for
keeping simulation separate from model training.

### b. Preparing single cell dataset by yourself

If you want to use other scRNA-seq datasets to simulate GEPs, you can follow our workflow to preprocess single cell datasets and merge them together. The Python package `Scanpy` was used heavily in our workflow.

- Preprocessing a single dataset: [03deal_with_Puram et al Cell.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/single_cell_dataset_integration/03deal_with_Puram%20et%20al%20Cell.ipynb).
- Merging multiple datasets together (part 1): [Merge_12_scRNA-seq_datasets_part1.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/single_cell_dataset_integration/Merge_12_scRNA-seq_datasets_part1.ipynb).
- Merging multiple datasets together (part 2-round1): [Merge_12_scRNA-seq_datasets_part2_first_round.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/single_cell_dataset_integration/Merge_12_scRNA-seq_datasets_part2_first_round.ipynb)
- Merging multiple datasets together (part 2-round2): [Merge_12_scRNA-seq_datasets_part2_second_round.ipynb](https://github.com/OnlyBelter/DeSide_mini_example/blob/main/single_cell_dataset_integration/Merge_12_scRNA-seq_datasets_part2_second_round.ipynb)
