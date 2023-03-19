import os
import csv
import json
import time
import umap
from typing import Union
import numpy as np
import pandas as pd
# import anndata as an
import seaborn as sns
import matplotlib as mpl
from pathlib import Path
import scipy.stats as stats
from joblib import dump, load
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from anndata import AnnData, read_h5ad
from sklearn.metrics import mean_squared_error, r2_score
import gzip
import shutil


default_core_marker_genes = {'Cancer Cells': ['KRT19', 'KRT18', 'KRT8', 'EPCAM'],
                             'CD4 T': ['BATF', 'ICOS', 'CD4', 'IL7R', 'FOXP3', 'TIGIT'],
                             'CD8 T': ['CD8A', 'CD8B'],
                             'T Cells': ['CD2', 'CD3D', 'CD3E'],
                             'B Cells': ['BANK1', 'CD79A', 'FCRL5', 'MS4A1'],
                             'DC': ['GZMB', 'CCR7', 'LAMP3', 'IRF7', 'IRF8'],
                             'Endothelial Cells': ['CLDN5', 'ENG', 'PLVAP', 'VWF'],
                             'Fibroblasts': ['COL1A1', 'COL1A2', 'COL3A1', 'MYL9'],
                             'Macrophages': ['AIF1', 'CD14', 'CD68', 'MS4A7'],
                             'Mast Cells': ['CPA3', 'HPGDS', 'GATA2'],
                             'NK': ['GNLY', 'NKG7', 'KLRD1'],
                             'Neutrophils': ['CSF3R', 'CXCR2', 'FPR1', 'SLC25A37']}


sorted_cell_types = ['B Cells', 'CD4 T', 'CD8 T', 'Cancer Cells', 'DC', 'Endothelial Cells',
                     'Fibroblasts', 'Macrophages', 'Mast Cells', 'NK', 'Neutrophils']


def get_inx2cell_type() -> dict:
    inx2cell_type = {_i: ct for _i, ct in enumerate(sorted_cell_types)}
    inx2cell_type[-1] = 'Neg'
    return inx2cell_type


def print_df(df):
    assert type(df) == pd.DataFrame
    print('  >>  <<  ')
    print(df.shape)
    print(df.head(2))


def set_fig_style(font_family=None, font_size=None):
    fig, ax = plt.subplots()
    sns.set_style("white")
    try:
        plt.style.use(['science', 'no-latex'])
    except:
        print('No science style')
        sns.set(palette='muted', font_scale=1.5)

    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    if font_family:
        mpl.rcParams['font.family'] = font_family
    if font_size:
        mpl.rcParams['font.size'] = font_size
    # print('figure.dpi will be set to', mpl.rcParams['figure.dpi'])
    plt.close('all')


def filter_gene_by_expression_log_mean(exp_df, min_exp_value=3, max_exp_value=10, min_exp_percent=0.8):
    """
    Ea(i)=log2(ave(TPM(i)1..k)+1), excluded genes with Ea<threshold

    "As genes with high expression can bias deconvolution results,
     we excluded the genes whose expression exceeded 700 TPM",
     Finotello, F. et al., Genome Med 11, 34 (2019). https://doi.org/10.1186/s13073-019-0638-6

    :param exp_df: expression values of each gene, non-log2 values, a dataframe
    :param min_exp_value: minimum log2 mean expression value of each gene across all samples
    :param max_exp_value: max log2 mean expression value of each gene across all samples
    :param min_exp_percent: minimum expression percentage of each gene in all samples
    :return: filtered genes in each sample
    """
    # assert type(exp_df) == pd.
    n_gene = exp_df.shape[0]
    n_sample = exp_df.shape[1]
    print('>>> Total gene: {}'.format(n_gene))
    log_mean_exp = np.log2(exp_df.mean(axis=1) + 1)
    above_min_exp_inx = log_mean_exp >= min_exp_value
    below_max_exp_inx = log_mean_exp <= max_exp_value
    exp_filtered_by_min_exp = exp_df.loc[above_min_exp_inx & below_max_exp_inx, :]
    print('>>> Number of gene below min_exp_value: {}'.format(n_gene - sum(above_min_exp_inx)))
    print('>>> Number of gene above max_exp_value: {}'.format(n_gene - sum(below_max_exp_inx)))
    # remove genes that does not express in high percentage samples
    keep_inx = (np.sum(exp_filtered_by_min_exp < min_exp_value, axis=1) / n_sample) < min_exp_percent
    return exp_filtered_by_min_exp.loc[keep_inx, :].copy()


def filter_gene_by_expression_min_max(exp_df, min_exp_value=3, max_exp_value=1000,
                                      min_exp_percent=0.8, high_exp_min_percent=0.2,
                                      min_num_sample=50):
    """
    exclude genes with E < min_exp_value among more than 20% (1-min_exp_percent) samples
    exclude genes with E > max_exp_value among less than 20% (high_exp_min_percent) samples

    "As genes with high expression can bias deconvolution results,
     we excluded the genes whose expression exceeded 700 TPM",
     Finotello, F. et al., Genome Med 11, 34 (2019). https://doi.org/10.1186/s13073-019-0638-6

    :param exp_df: expression values of each gene, non-log2 values, a dataframe
    :param min_exp_value: minimum expression value of each gene across all samples
    :param max_exp_value: max expression value of each gene across all samples
    :param min_exp_percent: float
        the min expression value of each gene should >=min_exp_value in all samples higher than this percent,
        otherwise will be exclude (treat as no expression in all samples), at least 0.8
    :param high_exp_min_percent: float
        the max expression value of each gene should >=max_exp_value in all samples higher than this percent,
        otherwise will be exclude (treat as only having extremely high expression in a few samples)
    :param min_num_sample: the minimum number of samples needs to fulfil above conditions instead of using percentage

    :return: filtered genes in each sample
    """
    n_gene = exp_df.shape[0]
    n_sample = exp_df.shape[1]
    print('>>> Total gene: {}'.format(n_gene))
    # log_mean_exp = np.log2(exp_df.mean(axis=1) + 1)
    # no expression in the most of samples
    if n_sample > min_num_sample / (1 - min_exp_percent):
        min_exp_percent = 1 - min_num_sample / n_sample  # bigger min_exp_percent in big n_sample
    nm_exp_inx = (np.sum(exp_df < min_exp_value, axis=1) / n_sample) > min_exp_percent
    # exist extremely high expression in a few samples
    if n_sample >= min_num_sample / high_exp_min_percent:
        high_exp_min_percent = min_num_sample / n_sample
    ehf_exp_inx = ((np.sum(exp_df > max_exp_value, axis=1) / n_sample) < high_exp_min_percent) & \
                  ((np.sum(exp_df > max_exp_value, axis=1) / n_sample) > 0)

    print('>>> Number of gene which no expression in the most of samples: {}'.format(sum(nm_exp_inx)))
    print('>>> Number of gene which extremely high expression in a few samples: {}'.format(sum(ehf_exp_inx)))
    return exp_df.loc[~(nm_exp_inx | ehf_exp_inx), :].copy()


def filter_sample_by_expression(exp_df, max_mean_exp=100, plot_cdf=True, fig_file_path=""):
    """
    remove samples have high mean expression, usually contain some extremely high expressed genes
    :param exp_df:
    :param max_mean_exp:
    :param plot_cdf: plot CDF of sample mean expression
    :param fig_file_path: where to save cdf figure
    """
    sample_mean_exp = exp_df.mean()
    n_sample = exp_df.shape[1]
    if plot_cdf:
        plt.figure(figsize=(8, 6))
        plt.hist(sample_mean_exp, density=True, cumulative=True, histtype='step', color='k')
        if fig_file_path:
            plt.savefig(fig_file_path, dpi=200)
        plt.show()
    below_max_inx = sample_mean_exp <= max_mean_exp
    print('>>> Number of sample above max_mean_exp: {}'.format(n_sample - sum(below_max_inx)))
    return exp_df.loc[:, below_max_inx].copy()


def log2_transform(df):
    """
    log2 transform expression values (plus 1)
    :param df:
    :return:
    """
    df = df.astype(np.float64)
    df = np.log2(df + 1)
    df = df.astype(np.float32)
    return df


def center_value(df, return_mean=False):
    """
    exp - mean(exp) for each genes
    :param df: expression dataframe, gene x sample
    :param return_mean: if return df_mean
    :return:
    """
    df_mean = df.mean(axis=1)
    if return_mean:
        return df - np.vstack(df_mean), df_mean.to_frame('gene_mean')
    return df - np.vstack(df_mean)


def filter_gene_by_variance(exp_df, threshold=0.1):
    """
    remove genes which variation is < threshold
    :param exp_df:
    :param threshold:
    :return:
    """
    n_gene_before_filter = exp_df.shape[0]
    exp_var = exp_df.var(axis=1)
    exp_df = exp_df.loc[exp_var >= threshold, :].copy()
    n_gene_after_filter = exp_df.shape[0]
    print('>>> there are {} removed by lower variation than {}'.format(n_gene_before_filter - n_gene_after_filter,
                                                                       threshold))
    return exp_df


def read_exp_from_hcluster(filtered_exp_file_path, reordered_ind2sample_name_file_path):
    """
    read expression profiles from the result of hcluster
    :param filtered_exp_file_path: gene by sample
        filtered expression profiles
    :param reordered_ind2sample_name_file_path: one of result files of hluster
    :return: {"filtered_exp": filtered_exp, 'cell_type2sample_name': cell_type2sample_name}
    """
    cell_type2sample_name = {}  # {'cancer_cell': ['sample1', 'sample8'], '': [], ...}
    filtered_exp = pd.read_csv(filtered_exp_file_path, index_col=0)
    reordered_ind2sample_name = pd.read_csv(reordered_ind2sample_name_file_path, index_col=0)
    reordered_ind2sample_name = reordered_ind2sample_name.loc[reordered_ind2sample_name['keep'] == 1, :]
    filtered_exp = filtered_exp.loc[:, reordered_ind2sample_name.index].copy()
    for sample_name in reordered_ind2sample_name.index:
        cell_type = reordered_ind2sample_name.loc[sample_name, 'subtype']
        if cell_type not in cell_type2sample_name.keys():
            cell_type2sample_name[cell_type] = []
        cell_type2sample_name[cell_type].append(sample_name)
    return {"filtered_exp": filtered_exp, 'cell_type2sample_name': cell_type2sample_name}


def read_cancer_purity(cancer_purity_file_path, sample_names: list):
    """
    Tumor purity estimates for TCGA samples
    Aran, D., Sirota, M. & Butte, A. Systematic pan-cancer analysis of tumour purity. Nat Commun 6, 8971 (2015).
    https://doi.org/10.1038/ncomms9971
    :param cancer_purity_file_path:
    :param sample_names: all sample names need to compare
    :return:
    """
    cancer_purity = pd.read_csv(cancer_purity_file_path, index_col=0)
    cancer_purity = cancer_purity.loc[~cancer_purity['CPE'].isnull()].copy()
    # print_df(cancer_purity)
    sample_name_mapping = {i[0:16]: i for i in sample_names}
    common_sample = set(list(sample_name_mapping.keys())) & set(cancer_purity.index)
    cancer_purity = cancer_purity.loc[common_sample, :].copy()
    cancer_purity.index = cancer_purity.index.map(sample_name_mapping)
    return cancer_purity


def cal_relative_error(y_true, y_pred, max_error=None, min_error=None):
    """
    calculate relative error between two dataFrame
    relative error = (y_pred - y_true) / y_true
    :param y_true: dataframe
    :param y_pred: dataframe, two dataframe have same index and columns (also same order)
    :param max_error: float
        all errors should <= this value
    :param min_error: float
        all errors should >= this value
    :return:
    """
    relative_error = (y_pred - y_true) / y_true
    if max_error:
        relative_error[relative_error > max_error] = max_error
    if min_error:
        relative_error[relative_error < min_error] = min_error
    return relative_error


def calculate_rmse(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    calculate the RMSE of each cell types by columns
    :param y_true: a dataFrame
        shape: number of samples x number of cell types
    :param y_pred: a dataFrame
    :return:
    """
    if y_true.shape[1] == 1:  # only one feature
        multioutput = 'uniform_average'
    else:  # multiple cell types
        multioutput = 'raw_values'
    return mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput=multioutput, squared=False)


def calculate_r2(y_true, y_pred):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    calculate the R^2 (coefficient of determination) of each cell types by columns
    :param y_true:
    :param y_pred:
    :return:
    """
    return r2_score(y_true=y_true, y_pred=y_pred, multioutput='raw_values')


def check_dir(path):
    """
    check if a path exist, create if not exist
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def parse_log_file(log_file_path, search_type='sample'):
    """
    parse log file, @sample xxx, ...
    :param log_file_path:
    :param search_type: search gene or sample in log file
    :return: list
        a list of sample names
    """
    sample_names = []
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            for i in f:
                i = i.strip()
                _sample_name = i.split(',')[0].replace(f'@{search_type} ', '')
                sample_names.append(_sample_name)
    return sample_names


def write_to_log(log_file_path, template, sample_names, n_round):
    """

    :param log_file_path:
    :param template: '@sample {}, removed, ..., round {}', only two {} for sample name and n_round
    :param sample_names:
    :param n_round:
    :return:
    """
    sample_names_in_log_file = parse_log_file(log_file_path)
    with open(log_file_path, 'a') as f:
        for s in sample_names:
            _info = template.format(s, n_round)
            if s not in sample_names_in_log_file:
                f.write(_info + '\n')
            print(_info)


def correct_gene_list(exp_profile: pd.DataFrame, gene_list: list) -> pd.DataFrame:
    """
    add 0 to expression profile for genes which not exist and sort gene according to the order in gene_list
    :param exp_profile: expression profiles of bulk data, sample by gene
    :param gene_list: a gene list with same order as training set (kNN scaden or Scaden scaden, single cell data)
    :return:
    """
    gene_not_in_bulk = [i for i in gene_list if i not in exp_profile.columns]
    n_not_in = len(gene_not_in_bulk)
    n_sample, _ = exp_profile.shape
    if n_not_in >= 1:
        # print(f'   > there are {n_not_in} genes not in gene list, 0 will be added for these genes')
        add_zero_expression = pd.DataFrame(data=np.zeros((n_sample, n_not_in)),
                                           index=exp_profile.index, columns=gene_not_in_bulk)
        exp_profile = pd.concat([exp_profile, add_zero_expression], axis=1)
    exp_profile = exp_profile.loc[:, gene_list]  # only use genes in gene_list
    return exp_profile


def extract_gz_file(file_dir: str, file_name: str):
    file_path = os.path.join(file_dir, file_name)
    file_path_out = os.path.join(file_dir, file_name.replace('.gz', ''))
    if os.path.exists(file_path) and file_name.endswith('.gz'):
        if not os.path.exists(file_path_out):
            with gzip.open(file_path, 'rb') as f_in:
                with open(file_path_out, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


def create_h5ad_dataset(simulated_bulk_exp_file_path, cell_fraction_file_path,
                        dataset_info, result_file_path: str,
                        filtering: bool = False, merge_t_cell: bool = False, gep_type='bulk'):
    """
    create .h5ad file according to cell fraction and simulated bulk expression profiles
    https://anndata.readthedocs.io/en/latest/index.html
    :param simulated_bulk_exp_file_path: simulated bulk expression profile, samples by genes
        .csv file or .h5ad file
    :param cell_fraction_file_path: .csv file, samples by cell types
    :param dataset_info: str
    :param result_file_path:
    :param filtering: if filtered by marker ratio
    :param merge_t_cell: whether merge the cell fraction of CD8 T and CD4 T cells
    :param gep_type: bulk (mixture contains >= 2 cell types) / sct (single cell type)
    :return:
    """
    try:
        if filtering:
            simulated_bulk_exp_file_path = simulated_bulk_exp_file_path.replace('_log2cpm1p.csv',
                                                                                '_log2cpm1p_filtered.csv')
            cell_fraction_file_path = cell_fraction_file_path.replace('.csv', '_filtered.csv')
        simu_bulk_exp_raw = read_df(simulated_bulk_exp_file_path)
    except UnicodeDecodeError:
        simu_bulk_exp_raw = read_h5ad(simulated_bulk_exp_file_path)
    # simu_bulk_exp.index = simu_bulk_exp.index.astype(str)
    cell_frac = read_df(cell_fraction_file_path)
    if merge_t_cell:
        if 'T Cells' not in cell_frac.columns:
            cell_frac['T Cells'] = cell_frac.loc[:, ['CD4 T', 'CD8 T']].sum(axis=1)
        cell_frac.drop(columns=['CD4 T', 'CD8 T'], inplace=True)
    sample_list = cell_frac.index.to_list()
    # cell_frac.index = cell_frac.index.astype(str)
    uns = {'cell_types': cell_frac.columns.to_list(),
           'dataset_info': dataset_info}
    simu_bulk_exp = pd.DataFrame()
    if type(simu_bulk_exp_raw) == pd.DataFrame:
        simu_bulk_exp = simu_bulk_exp_raw.loc[sample_list, :]
    elif type(simu_bulk_exp_raw) == AnnData:
        simu_bulk_exp_df = pd.DataFrame(simu_bulk_exp_raw.X, index=simu_bulk_exp_raw.obs.index,
                                        columns=simu_bulk_exp_raw.var.index)
        simu_bulk_exp_df = simu_bulk_exp_df.loc[sample_list, :]
        simu_bulk_exp = simu_bulk_exp_df.copy()
    if (cell_frac.shape[0] == simu_bulk_exp.shape[0]) and not np.all(simu_bulk_exp.index == cell_frac.index):
        cell_frac = cell_frac.loc[simu_bulk_exp.index, :].copy()
    elif cell_frac.shape[0] != simu_bulk_exp.shape[0]:
        print(f'   The shape of simu_bulk_exp: {simu_bulk_exp.shape}, the shape of cell_frac: {cell_frac.shape}')
        cell_frac = cell_frac[~cell_frac.index.duplicated(keep='first')].copy()
        simu_bulk_exp = simu_bulk_exp[~simu_bulk_exp.index.duplicated(keep='first')].copy()
        intersection_inx = [i for i in simu_bulk_exp.index if i in cell_frac.index]
        print(f'   The length of intersections in both cell_frac and simu_bulk_exp: {len(intersection_inx)}')
        cell_frac = cell_frac.loc[intersection_inx, :].copy()
        simu_bulk_exp = simu_bulk_exp.loc[intersection_inx, :].copy()
    # print(cell_frac.index)
    # print(simu_bulk_exp.index)
    if np.all(simu_bulk_exp.index == cell_frac.index):
        var = pd.DataFrame(index=simu_bulk_exp.columns, columns=[f'in_{gep_type}'])
        var[f'in_{gep_type}'] = 1
        adata = AnnData(X=simu_bulk_exp.values.astype('float32'), obs=cell_frac, uns=uns,
                        var=var, dtype=np.dtype('float32'))
        adata.write_h5ad(filename=Path(result_file_path), compression='gzip')
    else:
        raise KeyError('simu_bulk_exp and cell_frac file should have same sample order')


def read_data_from_h5ad(h5ad_file_path: str) -> dict:
    """
    Read simulated bulk gene expression profiles (GEPs) from .h5ad file

    :param h5ad_file_path: the file path of simulated bulk GEPs file (.h5ad)

    :return: a dict contains bulk GEPs (bulk_exp) and cell fractions (cell_frac) used to generate this dataset
    """
    raw_input = read_h5ad(h5ad_file_path)
    cell_fraction = raw_input.obs.round(3)
    if type(raw_input.X) == csr_matrix:
        x_data = raw_input.X.A.astype(np.float32)  # convert sparse matrix to dense matrix
    else:
        x_data = raw_input.X.astype(np.float32)  # samples x genes (eg: 6000 x 18863)
    bulk_exp = pd.DataFrame(data=x_data, index=raw_input.obs.index, columns=raw_input.var.index).round(3)
    return {'bulk_exp': bulk_exp, 'cell_frac': cell_fraction}


def ciber_exp(bulk_exp_fp, index_sample=True, log_transformed=True, result_fp=None):
    """
    generate expression file format of CIBERSORT, gene by sample, non-log space, tab-delimited format
    :param bulk_exp_fp: .csv file, log2(CPM + 1)
    :param index_sample: sample by gene if True, or gene by sample if False
    :param log_transformed: input dataset is log transformed if True
    :param result_fp:
    :return: gene by sample, non-log space values (CPM/TPM) and tab-delimited
    """
    bulk_exp = pd.read_csv(bulk_exp_fp, index_col=0)
    if index_sample:
        bulk_exp = bulk_exp.T
    if log_transformed:
        bulk_exp = log_exp2cpm(bulk_exp)
    if result_fp is not None:
        bulk_exp.round(2).to_csv(result_fp, sep='\t')


def log_exp2cpm(exp_df: Union[pd.DataFrame, np.array], log_base=2, correct=1) -> Union[pd.DataFrame, np.array]:
    """
    Convert log2(CPM + 1) to non-log space values (CPM / TPM)

    :param exp_df: samples by genes

    :param log_base: the base of log transform

    :param correct: plus 1 for avoiding log transform 0

    :return: counts per million (CPM) or transcript per million (TPM)
    """
    exp = np.power(log_base, exp_df) - correct
    # exp = exp.astype(np.float64)
    cpm = exp / np.vstack(exp.sum(axis=1)) * 1e6
    # cpm = cpm.astype(np.float32)
    return cpm


def non_log2log_cpm(input_file_path: Union[str, pd.DataFrame], result_file_path: str = None,
                    transpose: bool = True, correct: int = 1):
    """
    Convert non-log expression data to log2(CPM + 1) or log2(TPM + 1)

    :param input_file_path: non-log space expression file

    :param result_file_path: file path, samples by genes

    :param transpose: if input file is samples by genes, set to False, otherwise set to True

    :param correct: plus 1 for avoiding log transform 0

    :return: log2(CPM + 1) or save result to file
    """

    bulk_exp = pd.DataFrame()
    if type(input_file_path) is str:
        sep = get_sep(input_file_path)
        bulk_exp = pd.read_csv(input_file_path, index_col=0, sep=sep)
    elif type(input_file_path) is pd.DataFrame:
        bulk_exp = input_file_path
    if transpose:
        bulk_exp = bulk_exp.T  # transpose to samples by genes
    bulk_exp = non_log2cpm(bulk_exp)  # CPM/TPM
    bulk_exp = np.log2(bulk_exp + correct)
    if result_file_path is not None:
        bulk_exp.round(3).to_csv(result_file_path)
    else:
        return bulk_exp.round(3)


def non_log2cpm(exp_df, sum_exp=1e6) -> pd.DataFrame:
    """
    Normalize gene expression to CPM / TPM for non-log space

    :param exp_df: gene expression profile in non-log space, sample by gene

    :param sum_exp: sum of gene expression for each sample, default is 1e6

    :return: counts per million (CPM) or transcript per million (TPM)
    """
    return exp_df / np.vstack(exp_df.sum(axis=1)) * sum_exp


def get_corr(df_col1, df_col2, return_p_value=False) -> Union[float, tuple]:
    """
    calculate the correlation between two columns of dataframe
    :param df_col1: series, column1
    :param df_col2: series, column2
    :param return_p_value: if return p-value
    :return:
    """
    # correlation = np.corrcoef(df_col1, df_col2)
    corr, p_value = stats.pearsonr(df_col1, df_col2)
    if return_p_value:
        return corr, p_value
    else:
        return corr
    # return correlation[0, 1]


def get_sep(file_path, comment: str = None):
    """
    check the separater (`\t` or `,`) in this file
    :param file_path:
    :param comment: if remove comment lines start with ''
    :return:
    """
    sep = '\t'
    with open(file_path, 'r') as f_handle:
        # first_line = ''
        while True:
            first_line = f_handle.readline()
            if comment is not None:
                if first_line.startswith(comment):
                    continue
                else:
                    break
            break
        if ',' in first_line:
            sep = ','
    return sep


def read_marker_gene(marker_gene_file_path: str, include_t_cell: bool = False,
                     include_cd8_nk_marker: bool = False, use_cancer_cell: bool = False,
                     add_top_corr_gene: bool = False, corr_mean: bool = False) -> dict:
    """
    read marker genes for each cell type
    :param marker_gene_file_path: file path of selected marker genes for each cell type
    :param include_t_cell: if include the marker genes of T Cells for both CD4 and CD8 T
    :param include_cd8_nk_marker: if remain "CD8 T / NK" marker genes
    :param use_cancer_cell: if use "Cancer Cells" to replace "Epithelial Cells"
    :param add_top_corr_gene: add top correlated genes from the corr between cell fraction and gene expression value
    :param corr_mean: only two genes for CD4 T Cells and two genes for B Cells, others are same as core_marker
    :return: a dict of cell type to marker genes, {'': []}
    """
    cell_type2marker = {}
    marker_gene = pd.read_csv(marker_gene_file_path)
    if 'corr_mean' in marker_gene.columns and corr_mean:
        marker_gene = marker_gene.loc[marker_gene['corr_mean'] == 1, :].copy()
    if 'core_marker' in marker_gene.columns:
        if add_top_corr_gene:
            marker_gene = marker_gene.loc[marker_gene['core_marker'].isin([1, 2]), :].copy()
        else:
            marker_gene = marker_gene.loc[marker_gene['core_marker'] == 1, :].copy()
        if not include_cd8_nk_marker:
            marker_gene = marker_gene.loc[marker_gene['cell_type'] != 'CD8 T / NK', :].copy()
    cell_types = sorted(marker_gene['cell_type'].unique())
    for ct in cell_types:
        if include_t_cell:
            cell_type2marker[ct] = list(marker_gene.loc[marker_gene['cell_type'] == ct, 'marker_gene'].unique())
        else:
            if ct != 'T Cells':
                cell_type2marker[ct] = list(marker_gene.loc[marker_gene['cell_type'] == ct, 'marker_gene'].unique())
    if include_cd8_nk_marker and cell_type2marker.get('CD8 T / NK', ''):
        cell_type2marker['NK'] += cell_type2marker['CD8 T / NK']
        cell_type2marker['CD8 T'] += cell_type2marker['CD8 T / NK']
    if use_cancer_cell and 'Epithelial Cells' in cell_type2marker:
        cell_type2marker['Cancer Cells'] = cell_type2marker['Epithelial Cells']
        del cell_type2marker['Epithelial Cells']
    return cell_type2marker


def cal_exp_by_gene_list(exp_df, gene_list, min_exp_value=0, method='mean'):
    """
    calculate mean expression (or max) of a gene list (for single cell type)
    :param exp_df: sample by gene, usually in CPM / TPM
    :param gene_list: a list of genes which all are included in exp_df
    :param min_exp_value: min mean expression of the gene list
    :param method: mean or max
    :return: mean or max expression value of marker genes for each sample
    """
    for gene in gene_list:
        if gene not in exp_df.columns:
            raise KeyError(f'Gene {gene} not include in exp_df')
    current_exp = exp_df.loc[:, gene_list].copy()
    if method == 'mean':
        current_value = current_exp.mean(axis=1)
    elif method == 'max':
        current_value = current_exp.max(axis=1)
    else:
        raise KeyError('Only "mean" or "max" allowed')
    current_value[current_value < min_exp_value] = min_exp_value
    if exp_df.shape[0] == 1:
        return round(float(current_value), 3)  # float
    return current_value.round(3)  # pd.Series


def save_key_params(all_vars: dict, save_to_file_path=None):
    if save_to_file_path is None:
        k_params_path = os.path.join(all_vars['result_dir'], 'key_parames.txt')
    else:
        k_params_path = save_to_file_path
    if not os.path.exists(k_params_path):
        key_paths = ['result_dir', 'merged_sc_dataset_file_path', 'simu_bulk_exp_dir', 'generated_sc_dataset_dir',
                     'test_set_dir', 'tcga_data_dir', 'cancer_purity_file_path', 'marker_gene_file_path',
                     'pre_trained_model_dir', 'pred_cell_frac_tcga_dir']
        model_names = ['DeSide']
        log_file_path = all_vars.get('log_file_path', '')
        hyper_params = all_vars['deside_parameters']
        other_params = ['all_cell_types', 'dataset2parameters', 'cd4_high_in_cd8', 'n_base',
                        'total_cell_number', 'removed_cell_types', 'merge_t_cell', 'filter_simulated_bulk_cell',
                        'remove_cancer_cell_when_training', 'one_minus_alpha', 'remove_cancer_cell']
        key_paths_dict = {k: all_vars[k] for k in key_paths if k in all_vars}
        other_params_dict = {k: all_vars[k] for k in other_params if k in all_vars}

        all_key_params = {'model_names': model_names, 'key_paths_dict': key_paths_dict, 'hyper_params': hyper_params,
                          'other_params': other_params_dict, 'key_params_path': k_params_path,
                          'log_file_path': log_file_path}
        with open(k_params_path, 'w', encoding='utf-8') as f:
            json.dump(all_key_params, f, ensure_ascii=False, indent=4)


def print_msg(p_str, log_file_path=None):
    print()
    current_info = f'---->>> {p_str} <<<----'
    current_time = time.ctime()
    print(current_info)
    print(current_time)
    if log_file_path is not None:
        with open(log_file_path, 'a') as f_handle:
            f_handle.write(current_info + '\n')
            f_handle.write(current_time + '\n')
            f_handle.write('\n')


def read_xy(a: Union[str, pd.DataFrame], xy='cell_frac') -> pd.DataFrame:
    """
    read cell fraction or bulk expression
    :param a: a file path or a DataFrame
    :param xy: cell_frac or bulk_exp, only for .h5ad file
    """
    if (type(a) == str) and ('.h5ad' in a):
        raw_data = read_data_from_h5ad(a)
        exp = raw_data[xy]
    else:
        exp = read_df(a)
    return exp


def cal_corr_gene_exp_with_cell_frac(gene_exp: pd.DataFrame, cell_frac: pd.DataFrame,
                                     result_file_path=None, filtered_by_corr: float = None,
                                     filter_by_num: int = None):
    """
    calculate correlation between gene expression values and cell fractions for each cell type
    :param gene_exp: non-log space, tpm/cpm, samples by genes
    :param cell_frac: samples by cell types
    :param result_file_path:
    :param filtered_by_corr: threshold of correlation, only keep genes with higher corr than this value
    :param filter_by_num: the max number of genes to keep for each cell type
    :return: correlation, genes by cell types, filtered or all gene list
    """
    if not os.path.exists(result_file_path):
        _, n_cell_type = cell_frac.shape
        _, n_gene = gene_exp.shape
        print(f'The shape of cell_frac: {cell_frac.shape}')
        print(f'The shape of gene_exp: {gene_exp.shape}')
        corr = np.corrcoef(cell_frac.values, gene_exp.values, rowvar=False)
        corr_df = pd.DataFrame(data=corr[:n_cell_type, n_cell_type:].T,
                               columns=cell_frac.columns, index=gene_exp.columns)
    else:
        print(f'Previous correlation file will be used: {result_file_path}')
        corr_df = pd.read_csv(result_file_path, index_col=0)
    if (filtered_by_corr is not None) and (filter_by_num is not None):
        for cell_type in corr_df.columns:
            if corr_df.loc[corr_df[cell_type] >= filtered_by_corr, cell_type].shape[0] > filter_by_num:
                corr_df2 = corr_df.sort_values(by=[cell_type], ascending=False)
                # set to 0 if there are too many high corr genes for the cell fraction of this cell type
                corr_df.loc[corr_df.index.isin(corr_df2.iloc[filter_by_num:, :].index), cell_type] = 0
        corr_df[f'n_at_least_one'] = np.sum(corr_df >= filtered_by_corr, axis=1)
    elif filtered_by_corr is not None:
        corr_df[f'n_at_least_one'] = np.sum(corr_df >= filtered_by_corr, axis=1)
    elif filter_by_num is not None:
        for cell_type in corr_df.columns:
            corr_df2 = corr_df.sort_values(by=[cell_type], ascending=False)
            # set to 0 if there are too many genes than filter_by_num of this cell type
            corr_df.loc[corr_df.index.isin(corr_df2.iloc[filter_by_num:, :].index), cell_type] = 0
        corr_df[f'n_at_least_one'] = np.sum(corr_df > 0, axis=1)
    if (filtered_by_corr is not None) or (filter_by_num is not None):
        corr_df_filtered = corr_df.loc[corr_df[f'n_at_least_one'] >= 1,
                                       [i for i in corr_df.columns if i != 'n_at_least_one']].copy()
        corr_df_filtered.to_csv(result_file_path, float_format='%g')
        return corr_df_filtered
    else:
        corr_df.to_csv(result_file_path, float_format='%g')
        return corr_df


def read_df(df_file: Union[str, pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """
    check the type of df_file
    - if df_file is a file path, read this file
    - if df_file is a DataFrame, return directly
    """
    if type(df_file) == str:  # the file path of current file
        sep = get_sep(df_file)  # separated by '\t' or ','
        df = pd.read_csv(df_file, index_col=0, sep=sep)
    elif type(df_file) == pd.DataFrame:
        return df_file  # return DataFrame directly
    elif type(df_file) == np.ndarray:
        return df_file  # return np.ndarray directly
    else:
        raise TypeError(
            f'Only file path or pd.DataFrame was supported by df_file, {type(df_file)} is not supported.')
    return df


def aggregate_marker_gene_exp(exp_df: pd.DataFrame, marker_genes: dict = None,
                              agg_methods: dict = None, return_ratio: bool = False,
                              show_marker_gene: bool = True):
    """
    aggregate marker genes by the corresponding method in agg_methods (mean or max)
    :param exp_df: a DataFrame of gene expression profiles, TPM, samples by genes
    :param marker_genes:
    :param agg_methods: aggregate multiple marker genes of a cell type by `mean` or `max`
    :param return_ratio: return the ratio of aggregated marker gene exp for each cell type if True
    :param show_marker_gene: whether printing marker genes or not
    """
    if marker_genes is None:
        marker_genes = default_core_marker_genes.copy()
    if agg_methods is None:
        agg_methods = {}
    for k, _ in marker_genes.items():
        if k not in agg_methods:
            agg_methods[k] = 'mean'

    cell_type2marker_exp = {}
    for ct, marker in marker_genes.items():
        method = agg_methods[ct]
        current_tpm = exp_df.loc[:, exp_df.columns.isin(marker)].copy()  # sample by gene
        current_valid_marker = current_tpm.columns.to_list()
        if len(current_valid_marker) > 0:
            current_marker_str = ', '.join(current_valid_marker)
            if show_marker_gene:
                print(f'   Using {len(current_valid_marker)} marker genes for cell type {ct}: {current_marker_str}')
            # cell_type2mean_exp[ct + '_marker_mean'] = current_tpm.mean(axis=0)
            cell_type2marker_exp[ct + f'_marker_{method}'] = cal_exp_by_gene_list(exp_df=exp_df, method=method,
                                                                                  gene_list=current_valid_marker,
                                                                                  min_exp_value=1)
        else:
            Warning(f'No any marker genes in bulk cell TPM for cell type {ct}, '
                    f'this cell type will be ignored in later analysis.')
    marker_exp = pd.DataFrame.from_dict(cell_type2marker_exp, orient='columns')
    if return_ratio:
        return marker_exp / marker_exp.sum(axis=1).values.reshape(-1, 1)
    else:
        return marker_exp.round(3)


def cal_marker_ratio(bulk_exp_file_path: str, marker_genes: dict, marker_ratio_file_path: str,
                     agg_methods: dict = None, log_transformed: bool = True):
    """
    calculate marker ratio of each sample by gene expression values at large scale
    :param bulk_exp_file_path: log2cpm1p or TPM/CPM
    :param marker_genes:
    :param log_transformed: whether log2cpm1p or TPM/CPM
    :param marker_ratio_file_path: result file path
    :param agg_methods: aggregate multiple marker genes of a cell type by `mean` or `max`
    """
    if not os.path.exists(marker_ratio_file_path):
        # cell_frac = pd.read_csv(cell_frac_file_path, index_col=0)
        cell_types = list(marker_genes.keys())  # cell types in current dataset
        # only keep cell types in current dataset
        marker_genes = {k: v for k, v in marker_genes.items() if k in cell_types}
        # marker_ratio_dict = {}
        # col_names = None

        with open(marker_ratio_file_path, 'w') as write_file_handle:
            csv_writer = csv.writer(write_file_handle)
            csv_writer.writerow(['sample_id'] + list(marker_genes.keys()))
        with pd.read_csv(bulk_exp_file_path, chunksize=1000, index_col=0) as f_handle:
            for chunk in f_handle:
                if 'cancer_type' in chunk.columns:
                    chunk = chunk.drop(columns=['cancer_type'])
                if log_transformed:
                    current_exp = log_exp2cpm(chunk)  # convert log2cpm1p to CPM / TPM
                else:
                    current_exp = chunk
                current_marker_ratio = aggregate_marker_gene_exp(exp_df=current_exp, marker_genes=marker_genes,
                                                                 agg_methods=agg_methods, return_ratio=True)
                current_marker_ratio.to_csv(marker_ratio_file_path, mode='a', header=False, float_format='%g')
    else:
        print(f'   Previous result has existed: {marker_ratio_file_path}')


def get_cell_num(cell_type_frac, total_num=500):
    """
    calculate the number of cells to choose for each cell type based on the total number of cells
    and fraction of each cell type
    :param cell_type_frac: dataFrame
        the fraction of each cell type, sum to 1 for each row, samples by cell types
    :param total_num: the total number of cells contributing to one simulated RNA-seq sample
    :return:
    """
    assert type(cell_type_frac) is pd.DataFrame
    cell_num = cell_type_frac * total_num
    if total_num == 1:  # assign 1 for all cell types for matrix multiplication
        cell_num = cell_type_frac * 0 + 1
    elif total_num == 0:
        cell_num = np.ceil(cell_type_frac)  # assign 1 for the cell types with non-zero cell fractions
    return cell_num.round(0).astype(int)


def do_pca_analysis(exp_df, n_components=5, pca_result_fp=None, save_model: bool = False):
    """
    PCA analysis
    :param exp_df:
    :param n_components:
    :param pca_result_fp:
    :param save_model:
    :return: fitted PCA model
    """
    if os.path.exists(pca_result_fp):
        print(f'Loading PCA result from file: {pca_result_fp}')
        pca = load(pca_result_fp)
    else:
        pca = PCA(n_components=n_components)
        pca.fit(exp_df)
        if save_model:
            dump(pca, pca_result_fp)
    return pca


def do_umap_analysis(exp_df, n_components=5, n_neighbors=15, min_dist=0.1,
                     umap_model_result_fp=None, save_model: bool = False):
    """
    t-SNE analysis
    :param exp_df:
    :param n_components:
    :param n_neighbors:
    :param min_dist:
    :param umap_model_result_fp:
    :param save_model:
    :return:
    """
    if os.path.exists(umap_model_result_fp):
        print(f'Loading UMAP result from file: {umap_model_result_fp}')
        umap_model = load(umap_model_result_fp)
    else:
        umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
        umap_model.fit(exp_df)
        if save_model:
            dump(umap_model, umap_model_result_fp)
    return umap_model


def get_ccc(x, y):
    # Concordance Correlation Coefficient(CCC), https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    vx, cov_xy, cov_xy, vy = np.cov(x, y, bias=True).flatten()
    mx, my = x.mean(), y.mean()
    return 2*cov_xy / (vx + vy + (mx-my)**2)


if __name__ == '__main__':
    pass
