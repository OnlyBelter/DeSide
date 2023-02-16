import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from ..utility import (extract_gz_file, log_exp2cpm, cal_exp_by_gene_list)
import anndata as an
sc.settings.set_figure_params(dpi=200)
sns.set()


def check_by_stable_genes(sc_exp, stable_gene_list):
    """
    check single cell samples (each cell) by stable genes in bulk expression profiles
    :param sc_exp: pandas.DataFrame
        genes x cells of single cell, gene name as index of this df
    :param stable_gene_list: pandas.DataFrame
        gene with 95% confidence interval (95%_CI_left, 95%_CI_right)
    :return: percentage of genes in CI, cells x stable genes,
        -1: < 95%_CI_left,
        0: [95%_CI_left, 95%_CI_right]
        1: > 95%_CI_right
    """
    common_genes = [i for i in stable_gene_list.index if i in sc_exp.index]
    sc_exp = sc_exp.loc[common_genes, :].T
    stable_gene_list = stable_gene_list.loc[common_genes, :]
    result = pd.DataFrame(index=sc_exp.index, columns=sc_exp.columns)
    for sample in sc_exp.index:
        current_exp = sc_exp.loc[sample, :]
        for gene in current_exp.index:
            if current_exp[gene] > stable_gene_list.loc[gene, '95%_CI_right']:
                result.loc[sample, gene] = 1
            elif current_exp[gene] < stable_gene_list.loc[gene, '95%_CI_left']:
                result.loc[sample, gene] = -1
            else:
                result.loc[sample, gene] = 0
    result['in_CI_%'] = np.sum(result == 0, axis=1) / len(common_genes)
    result['<_CI_left_%'] = np.sum(result == -1, axis=1) / len(common_genes)
    result['>_CI_right_%'] = np.sum(result == 1, axis=1) / len(common_genes)
    return result


def get_10x_mtx(file_dir, prefix=None, min_genes=200, min_cells=3, result_dir=None,
                target_sum=1e4, log_base=None, max_n_genes_by_counts=2500, max_pct_counts_mt=5,
                max_total_counts=None, filter_genes: bool = True, fig_prefix=None, dense_file=False):
    """

    :param file_dir: contains three files: barcodes.tsv  genes.tsv  matrix.mtx
    :param prefix:
    :param min_genes: min genes in each cell
    :param min_cells: min cells for each gene expressed
    :param target_sum: normalized total reads
    :param log_base: logarithmize the data, usually is 2
    :param max_n_genes_by_counts: filtering n_genes_by_counts
    :param max_pct_counts_mt: filtering pct_counts_mt
    :param max_total_counts: filtering total_counts
    :param filter_genes: bool, if need to filter genes
    :param fig_prefix:
    :param result_dir:
    :param dense_file: bool, if input data is stored in dense format
    :return:
    """
    for _file in ['barcodes.tsv', 'genes.tsv', 'matrix.mtx']:
        file_name = prefix + _file + '.gz'
        # print(os.path.join(file_dir, file_name))
        if os.path.exists(os.path.join(file_dir, file_name)):
            # print('...')
            extract_gz_file(file_dir=file_dir, file_name=file_name)
    if dense_file:
        adata = sc.read_csv(file_dir)  # file_dir is file path now
    else:
        adata = sc.read_10x_mtx(file_dir, prefix=prefix, var_names='gene_symbols')
    adata = normalize_mtx(adata=adata, prefix=prefix, min_genes=min_genes, min_cells=min_cells,
                          result_dir=result_dir, target_sum=target_sum, log_base=log_base,
                          max_n_genes_by_counts=max_n_genes_by_counts, max_pct_counts_mt=max_pct_counts_mt,
                          max_total_counts=max_total_counts, filter_genes=filter_genes, fig_prefix=fig_prefix)
    return adata


def normalize_mtx(adata, prefix=None, min_genes=200, min_cells=3, result_dir=None,
                  target_sum=1e4, log_base=None, max_n_genes_by_counts=2500, max_pct_counts_mt=5,
                  max_total_counts=None, filter_genes: bool = True, fig_prefix=None):
    """

    :param adata: an AnnData object which contains a single sample data (cells by genes)
    :param prefix:
    :param min_genes: min genes in each cell
    :param min_cells: min cells for each gene expressed
    :param target_sum: normalized total reads
    :param log_base: logarithmize the data, usually is 2
    :param max_n_genes_by_counts: filtering n_genes_by_counts
    :param max_pct_counts_mt: filtering pct_counts_mt
    :param max_total_counts: filtering total_counts
    :param filter_genes: bool, if need to filter genes
    :param fig_prefix:
    :param result_dir:
    :return: log2(CPM + 1) after filtering
    """
    adata.var_names_make_unique()
    print('   The shape of adata before filtering is: {}'.format(adata.shape))
    # filtering cells and genes
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if filter_genes:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    # QC
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    if fig_prefix is not None:
        prefix = fig_prefix
    if result_dir is not None:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # plot QC metrics
        plt.figure(figsize=(15, 3))
        sc.settings.figdir = result_dir
        sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                     jitter=0.4, multi_panel=True, save=prefix + 'qc_metrics.png')

        # plot total counts vs percentage of counts in mitochondrial genes
        plt.figure(figsize=(8, 6))
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', save=prefix + 'total_counts_vs_pct_counts_mt.png')

        # plot total counts vs the number of genes expressed in the count matrix
        plt.figure(figsize=(8, 6))
        sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts',
                      save=prefix + 'total_counts_vs_n_genes_by_counts.png')

    # filtering cells
    adata = adata[adata.obs.n_genes_by_counts <= max_n_genes_by_counts, :]
    adata = adata[adata.obs.pct_counts_mt <= max_pct_counts_mt, :]
    if max_total_counts is not None:
        adata = adata[adata.obs.total_counts <= max_total_counts, :]
    if target_sum is not None:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    if log_base is not None and adata.shape[0] >= 1:
        sc.pp.log1p(adata, base=log_base)
        # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        # print('   The number of highly variable genes: {}'.format(sum(adata.var['highly_variable'])))
        # if result_dir is not None:
        #     plt.figure(figsize=(10, 4))
        #     sc.pl.highly_variable_genes(adata, save=prefix + 'highly_variable_genes.png')

    print('   The shape of adata after filtering is: {}'.format(adata.shape))
    return adata


def get_sample_id(obs_df, n, cell_type, n_base, class_by='leiden', sep_by_patient=False):
    """
    select sample id by random sampling
    :param obs_df: adata.obs in .h5ad file
    :param n: the number of cells to be generated
    :param cell_type: sampling only within this cell types (cell_type) or sub cell types (leiden)
    :param n_base: the number of single cells to average
    :param class_by: leiden or cell_type, the column name of cell type (or subtype) in .h5ad file
    :param sep_by_patient: only sampling from one patient in original dataset if True
    :return: a tuple of tuples  (('s1', 's2'), ('s3', 's5')), each tuple contains n_base ids for a single sample
    """
    if n_base == 0:
        return tuple()
    obs_df = obs_df.loc[obs_df[class_by] == cell_type, :].copy()
    all_samples_id = obs_df.index.to_list()
    groupby_patient_count = None
    if sep_by_patient:
        groupby_patient_count = obs_df.groupby(['sample_id']).count()
        groupby_patient_count = groupby_patient_count.loc[groupby_patient_count['leiden'] > n_base + 10, :].copy()
    selected_samples = {}
    while len(selected_samples) < n:
        _s = tuple()
        if (groupby_patient_count is not None) and (groupby_patient_count.shape[0] > 1):
            selected_patient_id = groupby_patient_count.sample(1).index.to_list()[0]
            all_samples_id = obs_df.loc[obs_df['sample_id'] == selected_patient_id, :].index.to_list()
        if len(all_samples_id) < n_base:
            print(cell_type, len(all_samples_id), n_base)
        _s = np.random.choice(all_samples_id, n_base, replace=False)
        _s = tuple(sorted(_s))
        if _s not in selected_samples:
            selected_samples[_s] = 1
    if n == 1:
        return tuple(selected_samples.keys())[0]  # return a tuple  ('s1', 's2', ...)
    return tuple(selected_samples.keys())  # return a tuple of tuples  (('s1', 's2'), ('s3', 's5'))


# def generate_sc(adata, n, cell_type, n_base=3, log_base=2, group_by='leiden',
#                 return_cell_id=False, filtering: bool = False, marker_genes=None,
#                 cell_type_scope: list = None):
#     """
#     Generate single cell expression data by averaging `n_base` cells from a specific `cell_type`
#
#     :param adata: AnnData object which contains multiple merged single cell datasets, GEPs stored by log2(CPM + 1),
#         CPM means counts per million, similar to TPM.
#
#     :param n: the number of cells to be generated
#
#     :param cell_type: sampling only within this cell types or sub cell types
#
#     :param n_base: the number of single cells to average
#
#     :param log_base: if None, don't do log transform
#
#     :param group_by: `leiden` (sub cell type) or `cell_type`, the column name of cell type in .h5ad file
#
#     :param return_cell_id: if return selected sample id directly without GEP
#
#     :param filtering: whether filter CD8 T and CD4 T based on the expression of corresponding marker genes
#
#     :param marker_genes: marker genes for each cell type, only for filtering
#
#     :param cell_type_scope: filter generated bulk cell expression profile only among these cell types
#
#     :return: a tuple of selected sample ids or an AnnData object with log2(CPM + 1) GEP
#     """
#     exp_threshold_low = 50
#     exp_threshold_up = 100
#     if marker_genes is None:
#         marker_genes = default_core_marker_genes
#     round_counter = 0
#     # only keep current cell type
#     adata = adata[adata.obs[group_by] == cell_type, :].copy()
#     adata_df = pd.DataFrame(adata.X.A, index=adata.obs.index, columns=adata.var.index)
#     adata_df = log_exp2cpm(adata_df)  # convert to CPM
#     g_sample2exp = {}
#     g_sample2id = {}
#     while len(g_sample2exp) < n:
#         # get sample_id for 100 samples each time
#         _selected_samples = get_sample_id(obs_df=adata.obs, n=100, n_base=n_base,
#                                           cell_type=cell_type, class_by=group_by)
#         for i, _s in enumerate(_selected_samples):
#             keep_this_exp = True
#             unique_inx = i + round_counter * 100
#             # expression profile of one generated single cell expression
#             exp_sc = adata_df.loc[_s, :].mean(axis=0)
#             # cell_type_abbr = ''
#             if cell_type in subcell_type2abbr:
#                 cell_type_abbr = subcell_type2abbr[cell_type]
#             elif cell_type in cell_type2abbr:
#                 cell_type_abbr = cell_type2abbr[cell_type]
#             else:
#                 raise KeyError('Unknown cell type')
#             gen_id = 'gen_' + cell_type_abbr + '_' + str(unique_inx)
#             if group_by == 'leiden':
#                 par_ct_name = cell_type_mapping[cell_type]  # the name of parent cell type
#             else:
#                 par_ct_name = cell_type
#             if filtering:
#                 exp_sc_df = exp_sc.to_frame(name=gen_id).T
#                 t_marker_mean_cpm = cal_exp_by_gene_list(exp_sc_df, gene_list=marker_genes['T Cells'], min_exp_value=1)
#                 if par_ct_name not in ['CD4 T', 'CD8 T']:
#                     if t_marker_mean_cpm >= exp_threshold_low:
#                         keep_this_exp = False
#                 else:
#                     if t_marker_mean_cpm < exp_threshold_up:
#                         keep_this_exp = False
#                 if not keep_this_exp:
#                     continue
#
#                 # filter by the expression value of marker genes
#                 marker_mean = {}
#                 if cell_type_scope is None:
#                     cell_type_scope = list(cell_type2abbr.keys())
#                 for _cell_type in cell_type_scope:
#                     if _cell_type in marker_genes.keys():  # non-cancer cell
#                         marker_mean[_cell_type] = cal_exp_by_gene_list(exp_sc_df, gene_list=marker_genes[_cell_type])
#                         if _cell_type == par_ct_name:
#                             # the mean expression of marker genes for current cell type should >= exp_threshold_up
#                             if marker_mean[_cell_type] < exp_threshold_up:
#                                 keep_this_exp = False
#                         else:  # the mean expression of marker genes for other cell types should < exp_threshold_low
#                             if marker_mean[_cell_type] >= exp_threshold_low:
#                                 keep_this_exp = False
#                         if not keep_this_exp:
#                             break
#
#             if keep_this_exp:
#                 g_sample2exp[gen_id] = exp_sc
#                 g_sample2id[gen_id] = _s
#         round_counter += 1
#     # selected_samples = get_sample_id(obs_df=adata.obs, n=n, n_base=n_base, cell_type=cell_type, class_by=group_by)
#     if return_cell_id:
#         return tuple(list(g_sample2id.values())[:n])
#
#     generated_samples = pd.DataFrame.from_dict(g_sample2exp, orient='index', columns=adata_df.columns)
#     if len(g_sample2exp) > n:
#         generated_samples = generated_samples.iloc[range(n), :].copy()
#     adata = an.AnnData(generated_samples)
#     if adata.shape[0] > 1:
#         adata.X = csr_matrix(adata.X)
#     adata.obs['dataset_id'] = 'generated'
#     adata.obs[group_by] = cell_type
#     g_sample2id = {i: ','.join(j) for i, j in g_sample2id.items()}
#     adata.obs['original_sample_ids'] = adata.obs.index.map(g_sample2id)
#     if group_by == 'leiden':
#         adata.obs['cell_type'] = cell_type_mapping.get(cell_type)
#     sc.pp.log1p(adata, base=log_base)
#     return adata


def merge_adata(adata_list: list, keep_index=True):
    """
    merge a list of AnnData objects
    :param adata_list:
    :param keep_index:
    :return:
    """
    # all_dataset_id = list(cell_type2dataset_n.keys())
    merged = None
    if keep_index:
        index_unique = None
    else:
        index_unique = "-"
    for i in range(len(adata_list)-1):
        # print('>>> deale with {}'.format(all_dataset_id[i]))
        next_dataset = adata_list[i+1]
        if merged is None:
            current_dataset = adata_list[i]
            merged = current_dataset.concatenate(next_dataset, index_unique=index_unique)
        else:
            merged = merged.concatenate(next_dataset, index_unique=index_unique)
    return merged


# def generate_sc_dataset2(merged_ds, n_samples: int = 5000, cell_types: list = None,
#                          result_file_path=None, group_by='leiden', n_base=3, filtering: bool = False,
#                          cd4_high_in_cd8: str = None):
#     """
#     Simple version of `generate_sc_dataset`,
#     generate single cell dataset by averaging the GEP of `n_base` (3 or 5) cells
#     from the same cell type (or sub cell type, leiden) from merged scRNA-seq dataset
#
#     :param merged_ds: all or parts of file 6_dataset_merged_after_batch_correction_filtered.h5ad
#
#     :param n_samples: number of samples for each cell type (or sub cell type)
#
#     :param cell_types:
#
#     :param result_file_path:
#
#     :param group_by: group by `cell_type` or `leiden` (sub cell type) when generating single cell dataset
#
#     :param n_base: the number of single cells to average
#
#     :param filtering: whether filter CD8 T and CD4 T based on the expression of corresponding marker genes
#
#     :param cd4_high_in_cd8: how to deal with the CD8 T cells that CD4 markers are highly expressed
#         (remove, CD4, CD8)
#
#     :return: a file of an AnnData object with log2(CPM + 1) GEP
#     """
#     if not os.path.exists(result_file_path):
#         cell_type2sc = {}
#         cells_cd4_high_in_cd8 = merged_ds.obs.loc[(merged_ds.obs['cell_type'] == 'CD8 T') &
#                                                   (merged_ds.obs['m_cd4/m_cd8 group'] == 'middle'), :].index.to_list()
#         if cd4_high_in_cd8 == 'remove':
#             merged_ds = merged_ds[~merged_ds.obs.index.isin(cells_cd4_high_in_cd8), :].copy()
#             print('   The CD8 T cells with highly expressed CD4 T markers will be removed...')
#         elif cd4_high_in_cd8 == 'CD4':
#             merged_ds.obs['leiden'].cat.add_categories(['CD4-high (1)', 'CD4-high (2)'], inplace=True)
#             merged_ds.obs.loc[merged_ds.obs.index.isin(cells_cd4_high_in_cd8), 'leiden'] = \
#                 merged_ds.obs.loc[merged_ds.obs.index.isin(cells_cd4_high_in_cd8),
#                                   'leiden'].replace({'CD8 T (1)': 'CD4-high (1)', 'CD8 T (2)': 'CD4-high (2)'})
#             merged_ds.obs.loc[merged_ds.obs.index.isin(cells_cd4_high_in_cd8), ['cell_type']] = \
#                 merged_ds.obs.loc[merged_ds.obs.index.isin(cells_cd4_high_in_cd8), 'cell_type'].replace('CD8 T', 'CD4 T')
#         print(merged_ds)
#         print(merged_ds.obs.loc[:, 'leiden'].value_counts())
#         # print(merged_ds.obs.loc[merged_ds.obs['cell_type'] == 'CD4 T', 'leiden'].value_counts())
#         for ct in cell_types:
#             print('   >>> deal with cell type {}'.format(ct))
#             if group_by == 'leiden':
#                 sub_cell_types = sorted(list(merged_ds.obs.loc[merged_ds.obs['cell_type'] == ct, 'leiden'].unique()))
#                 # print(sub_cell_types)
#                 for sct in sub_cell_types:
#                     print('       deal with sub cell type {}'.format(sct))
#                     current_adata = merged_ds[merged_ds.obs['leiden'] == sct, :].copy()
#                     generated = generate_sc(current_adata, n=n_samples, cell_type=sct, group_by=group_by,
#                                             n_base=n_base, filtering=filtering, cell_type_scope=cell_types)
#                     cell_type2sc[sct] = generated
#             elif group_by == 'cell_type':
#                 current_adata = merged_ds[merged_ds.obs['cell_type'] == ct, :].copy()
#                 generated = generate_sc(current_adata, n=n_samples, cell_type=ct, group_by=group_by,
#                                         n_base=n_base, filtering=filtering, cell_type_scope=cell_types)
#                 cell_type2sc[ct] = generated
#             else:
#                 raise ValueError('Unrecognized parameter group_by {}, group_by should be "leiden" for sub cell type '
#                                  'or "cell_type" for cell type'.format(group_by))
#         all_adata = list(cell_type2sc.values())
#         all_generated_data = merge_adata(all_adata)
#         print(all_generated_data.obs['cell_type'].value_counts())
#
#         # clustering
#         print('   >>> clustering generated single cell dataset...')
#         if 'leiden' in all_generated_data.obs.columns:
#             all_generated_data.obs['sub_cell_type'] = all_generated_data.obs['leiden'].copy()
#         # sc_file_dir = os.path.dirname(sc_file_path)
#         sc.tl.pca(all_generated_data, svd_solver='arpack', n_comps=100, use_highly_variable=False, zero_center=False)
#         # sc.pl.pca_variance_ratio(sc_dataset, n_pcs=30)
#         sc.pp.neighbors(all_generated_data, n_neighbors=10, n_pcs=40)
#         sc.tl.umap(all_generated_data)
#         sc.tl.leiden(all_generated_data)
#         all_generated_data.write(result_file_path)
#     else:
#         print(f'   Previous result has existed: {result_file_path}')


def filter_cells_by_marker_gene(single_cell_dataset, cell_types: list = None, cell_type2markers: dict = None,
                                exp_range: tuple = (50, 320), dataset_id: str = '', groupby: str = 'leiden',
                                max_exp: float = 0.0):
    """

    :param single_cell_dataset:
    :param cell_types:
    :param cell_type2markers:
    :param exp_range: (low_exp_threshold, middle_exp_threshold)
    :param dataset_id:
    :param groupby: index of groupby
    :param max_exp: max expression to plot
    :return:
    """
    cd4_cd8_ratio = 10
    assert type(single_cell_dataset) == an.AnnData
    all_removed_cells = []
    dataset2filter_info = pd.DataFrame(columns=['n_before_filtering', 'n_after_filtering', 'hit_cell_type'],
                                       index=list(single_cell_dataset.obs['leiden'].unique()))
    dataset2filter_info.index.name = 'leiden'
    cell_type_exp_value_collector_total = []
    for gi, group in single_cell_dataset.obs.groupby([groupby]):
        print(gi, group.shape)
        # cell_type2exp_value_collector_current_gi = []
        current_part_data = single_cell_dataset[single_cell_dataset.obs.index.isin(group.index), :].copy()
        gene_exp = pd.DataFrame(current_part_data.X.A, columns=current_part_data.var.index,
                                index=current_part_data.obs.index)
        gene_exp = log_exp2cpm(gene_exp)  # convert the expression values from log2(CPM + 1) to CPM

        cell_type_marker_exp = pd.DataFrame(index=gene_exp.index, columns=cell_types)  # sample by cell_type
        cell_type_quantile = pd.DataFrame(index=['q_10', 'q_25', 'q_50', 'q_75', 'q_90', 'q_95'],
                                          columns=cell_types)

        # cell_type2exp_flatten = {}

        for k, _cell_type in enumerate(cell_types):
            if _cell_type in cell_type2markers.keys():
                current_marker_gene = cell_type2markers[_cell_type]
                # cell_type2marker_exp[_cell_type] = gene_exp.loc[:, current_marker_gene].copy()  # a dataFrame
                method = 'mean'
                if _cell_type in ['B Cells', 'CD4 T']:
                    method = 'max'
                cell_type_marker_exp[_cell_type] = \
                    cal_exp_by_gene_list(gene_exp, gene_list=current_marker_gene, method=method, min_exp_value=1)
                _values = cell_type_marker_exp[_cell_type].values
                cell_type_quantile[_cell_type] = [np.quantile(_values, 0.1),
                                                  np.quantile(_values, 0.25),
                                                  np.quantile(_values, 0.50),
                                                  np.quantile(_values, 0.75),
                                                  np.quantile(_values, 0.90),
                                                  np.quantile(_values, 0.95)]
        if max_exp > 0:
            cell_type_marker_exp[cell_type_marker_exp > max_exp] = max_exp
        cell_type_marker_exp['m_max_cd4/mean_cd8'] = cell_type_marker_exp['CD4 T'] / cell_type_marker_exp['CD8 T']
        # ct2e_gi['m_max_cd4/mean_cd8'] = ct2e_gi['m_max_CD4 T'] / ct2e_gi['m_mean_CD8 T']
        cell_type_marker_exp['m_cd4/m_cd8 group'] = 'middle'
        cell_type_marker_exp.loc[cell_type_marker_exp['m_max_cd4/mean_cd8'] > cd4_cd8_ratio,
                                 'm_cd4/m_cd8 group'] = 'high'
        cell_type_marker_exp.loc[cell_type_marker_exp['m_max_cd4/mean_cd8'] < (1 / cd4_cd8_ratio),
                                 'm_cd4/m_cd8 group'] = 'low'
        cell_type_exp_value_collector_total.append(cell_type_marker_exp.copy())

        n_cells = gene_exp.shape[0]
        t_cell_marker = 'low'
        removed_cells = {}
        hit_cell_type = None
        dataset2filter_info.loc[gi, 'n_before_filtering'] = n_cells
        for k, _cell_type in enumerate(cell_types):
            if _cell_type in cell_type2markers.keys():
                q50_current_cell_type = cell_type_quantile.loc['q_50', _cell_type]
                # q90_current_cell_type = cell_type_quantile.loc['q_90', _cell_type]
                if q50_current_cell_type < exp_range[1]:  # Q50 in low or middle expression range
                    # 50% cells are expressed low as current cell type
                    if _cell_type not in ['CD4 T', 'CD8 T']:
                        # don't check CD4 T since the markers of CD4 may express in other cell types
                        removed_cells = update_removed_cells(removed_cells=removed_cells, cell_type=_cell_type,
                                                             cell_type_marker_exp=cell_type_marker_exp,
                                                             filter_upper=exp_range[1])
                    elif _cell_type == 'CD8 T':
                        if t_cell_marker != 'high':
                            # only filter CD8 T when t_cell_marker is not high
                            removed_cells = update_removed_cells(removed_cells=removed_cells, cell_type=_cell_type,
                                                                 cell_type_marker_exp=cell_type_marker_exp,
                                                                 filter_upper=exp_range[1])
                        else:  # t_cell_marker == 'high'
                            if (q50_current_cell_type >= exp_range[0]) and (q50_current_cell_type < exp_range[1]):
                                # Q50 quantile in middle expression range
                                # Treat this cell type as CD8 T (may be CD8 Tex)
                                # filtering by CD4 T / CD8 T
                                _result = filter_cd4_cd8(removed_cells=removed_cells, cell_type=_cell_type,
                                                         cell_type_marker_exp=cell_type_marker_exp,
                                                         cell_type_quantile=cell_type_quantile,
                                                         exp_range=exp_range)
                                removed_cells = update_removed_cells(removed_cells=_result['removed_cells'],
                                                                     cell_type=_cell_type,
                                                                     cell_type_marker_exp=cell_type_marker_exp,
                                                                     filter_lower=_result['filter_lower'],
                                                                     filter_upper=_result['filter_upper'])
                                hit_cell_type = 'CD8 T'
                                break

                else:  # the marker genes of current cell type are highly expressed (Q50 in high expression range)
                    if _cell_type == 'T Cells':
                        t_cell_marker = 'high'

                    if (n_cells - len(removed_cells)) >= 3000:  # remove 35% since too many cells in this cluster
                        filter_lower = cell_type_quantile.loc['q_25', _cell_type]
                        filter_upper = cell_type_quantile.loc['q_90', _cell_type]
                    else:  # remove 15% since a few cells
                        filter_lower = cell_type_quantile.loc['q_10', _cell_type]
                        filter_upper = cell_type_quantile.loc['q_95', _cell_type]

                    if t_cell_marker != 'high':  # filtering by quantile, t_cell_marker is low or middle
                        if _cell_type not in ['CD4 T', 'NK']:
                            removed_cells = update_removed_cells(removed_cells=removed_cells, cell_type=_cell_type,
                                                                 cell_type_marker_exp=cell_type_marker_exp,
                                                                 filter_lower=filter_lower, filter_upper=filter_upper)
                        elif _cell_type == 'NK':  # don't check CD4 T when t_cell_marker is low or middle
                            # filtering by CD4 T / CD8 T
                            _result = filter_cd4_cd8(removed_cells=removed_cells, cell_type=_cell_type,
                                                     cell_type_marker_exp=cell_type_marker_exp,
                                                     cell_type_quantile=cell_type_quantile,
                                                     exp_range=exp_range)
                            removed_cells = update_removed_cells(removed_cells=_result['removed_cells'],
                                                                 cell_type=_cell_type,
                                                                 cell_type_marker_exp=cell_type_marker_exp,
                                                                 filter_lower=_result['filter_lower'],
                                                                 filter_upper=_result['filter_upper'])
                        else:  # don't check CD4 T when t_cell_marker is low or middle
                            continue

                    else:  # t_cell_marker is high
                        _result = None
                        if _cell_type == 'T Cells':
                            pass
                        elif _cell_type in ['CD8 T', 'CD4 T', 'NK']:
                            # filtering by CD4 T / CD8 T
                            _result = filter_cd4_cd8(removed_cells=removed_cells, cell_type=_cell_type,
                                                     cell_type_marker_exp=cell_type_marker_exp,
                                                     cell_type_quantile=cell_type_quantile, exp_range=exp_range)
                            removed_cells = update_removed_cells(removed_cells=_result['removed_cells'],
                                                                 cell_type=_cell_type,
                                                                 cell_type_marker_exp=cell_type_marker_exp,
                                                                 filter_lower=_result['filter_lower'],
                                                                 filter_upper=_result['filter_upper'])

                    if _cell_type != 'T Cells':
                        hit_cell_type = _cell_type
                        break  # first hit except "T Cells", filtering by this cell type and stopped
        dataset2filter_info.loc[gi, 'n_after_filtering'] = n_cells - len(removed_cells)
        dataset2filter_info.loc[gi, 'hit_cell_type'] = hit_cell_type
        if len(removed_cells) > 0:
            _cell_ids = list(removed_cells.keys())
            all_removed_cells += _cell_ids
    cell_type_exp_value_collector_total_df = pd.concat(cell_type_exp_value_collector_total)
    cell_type_exp_value_collector_total_df = cell_type_exp_value_collector_total_df.loc[single_cell_dataset.obs.index, :]
    assert np.all(single_cell_dataset.obs.index == cell_type_exp_value_collector_total_df.index)
    single_cell_dataset.obs['m_max_cd4/mean_cd8'] = cell_type_exp_value_collector_total_df['m_max_cd4/mean_cd8']
    single_cell_dataset.obs['m_cd4/m_cd8 group'] = cell_type_exp_value_collector_total_df['m_cd4/m_cd8 group']
    filtered_dataset = single_cell_dataset[~single_cell_dataset.obs.index.isin(all_removed_cells), :].copy()
    dataset2filter_info['dataset_id'] = dataset_id
    return filtered_dataset, dataset2filter_info


def update_removed_cells(removed_cells: dict = None, cell_type_marker_exp: pd.DataFrame = None,
                         filter_upper: float = None, filter_lower: float = None, cell_type: str = '') -> dict:
    """

    :param removed_cells:
    :param cell_type_marker_exp:
    :param filter_upper: filter_upper > filter_lower
    :param filter_lower:
    :param cell_type
    :return:
    """
    if (filter_lower is not None) and (filter_upper is not None):
        _removed = cell_type_marker_exp.loc[(cell_type_marker_exp[cell_type] > filter_upper) |
                                            (cell_type_marker_exp[cell_type] <= filter_lower), :].index.to_list()
    elif filter_lower is None:
        _removed = cell_type_marker_exp.loc[(cell_type_marker_exp[cell_type] > filter_upper), :].index.to_list()
    elif filter_upper is None:
        _removed = cell_type_marker_exp.loc[(cell_type_marker_exp[cell_type] <= filter_lower), :].index.to_list()
    else:
        raise KeyError('Both filter_lower and filter_upper are None')
    if len(_removed) > 0:
        for _ in _removed:
            removed_cells[_] = 1

    return removed_cells


def filter_by_cd8_marker(removed_cells: dict = None, cell_type_marker_exp: pd.DataFrame = None,
                         cell_type_quantile: pd.DataFrame = None, cell_type: str = '',
                         exp_range: tuple = None):
    removed_cells = update_removed_cells(removed_cells=removed_cells, cell_type='CD8 T',
                                         cell_type_marker_exp=cell_type_marker_exp,
                                         filter_upper=exp_range[0])
    n_cells = cell_type_marker_exp.shape[0]
    # check n_cells - len(removed_cells) again
    if (n_cells - len(removed_cells)) >= 3000:  # remove 35% since too many cells
        filter_lower = cell_type_quantile.loc['q_25', cell_type]
        filter_upper = cell_type_quantile.loc['q_90', cell_type]
    else:  # remove 15% since a few cells
        filter_lower = cell_type_quantile.loc['q_10', cell_type]
        filter_upper = cell_type_quantile.loc['q_95', cell_type]
    return {'removed_cells': removed_cells, 'filter_lower': filter_lower, 'filter_upper': filter_upper}


def filter_by_cd4_marker(removed_cells: dict = None, cell_type_marker_exp: pd.DataFrame = None,
                         cell_type_quantile: pd.DataFrame = None, cell_type: str = '', exp_range: tuple = None):
    q50 = cell_type_quantile.loc['q_50', 'CD4 T']
    filter_upper = min(q50, exp_range[1])
    removed_cells = update_removed_cells(removed_cells=removed_cells, cell_type='CD4 T',
                                         cell_type_marker_exp=cell_type_marker_exp,
                                         filter_upper=filter_upper)
    n_cells = cell_type_marker_exp.shape[0]
    # check n_cells - len(removed_cells) again
    if (n_cells - len(removed_cells)) >= 3000:  # remove 35% since too many cells
        filter_lower = cell_type_quantile.loc['q_25', cell_type]
        filter_upper = cell_type_quantile.loc['q_90', cell_type]
    else:  # remove 15% since a few cells
        filter_lower = cell_type_quantile.loc['q_10', cell_type]
        filter_upper = cell_type_quantile.loc['q_95', cell_type]
    return {'removed_cells': removed_cells, 'filter_lower': filter_lower, 'filter_upper': filter_upper}


def filter_cd4_cd8(removed_cells: dict = None, cell_type: str = '', exp_range: tuple = (),
                   cell_type_marker_exp: pd.DataFrame = None, cell_type_quantile: pd.DataFrame = None):
    """
    filter cells by the ratio of marker genes expression between CD4 T and CD8 T (max CD4 T / mean CD8 T) or
    the expression value
    :param removed_cells:
    :param cell_type: current cell type with high expression of it's marker genes
    :param exp_range:
    :param cell_type_marker_exp:
    :param cell_type_quantile:
    :return:
    """
    # q50 = cell_type_quantile.loc['q_50', 'CD4 T']
    # filter_upper = min(q50, exp_range[1])
    # removed_cells = update_removed_cells(removed_cells=removed_cells, cell_type='CD4 T',
    #                                      cell_type_marker_exp=cell_type_marker_exp,
    #                                      filter_upper=filter_upper)
    _removed = []
    if cell_type == 'CD4 T':
        _removed = cell_type_marker_exp.loc[cell_type_marker_exp['m_cd4/m_cd8 group'] != 'high', :].index.to_list()
    elif cell_type == 'CD8 T':
        _removed = cell_type_marker_exp.loc[cell_type_marker_exp['m_cd4/m_cd8 group'] == 'high', :].index.to_list()
        # _removed = cell_type_marker_exp.loc[cell_type_marker_exp['m_cd4/m_cd8 group'] != 'low', :].index.to_list()
    elif cell_type == 'NK':  # both CD4 T marker and CD8 T marker should be low
        _removed = cell_type_marker_exp.loc[(cell_type_marker_exp['CD8 T'] > exp_range[1]) |
                                            (cell_type_marker_exp['CD4 T'] > exp_range[1]), :].index.to_list()

    if len(_removed) > 0:
        for _ in _removed:
            removed_cells[_] = 1

    n_cells = cell_type_marker_exp.shape[0]
    # check n_cells - len(removed_cells) again
    if (n_cells - len(removed_cells)) >= 3000:  # remove 35% since too many cells
        filter_lower = cell_type_quantile.loc['q_25', cell_type]
        filter_upper = cell_type_quantile.loc['q_90', cell_type]
    else:  # remove 15% since a few cells
        filter_lower = cell_type_quantile.loc['q_10', cell_type]
        filter_upper = cell_type_quantile.loc['q_95', cell_type]
    return {'removed_cells': removed_cells, 'filter_lower': filter_lower, 'filter_upper': filter_upper}


if __name__ == '__main__':
    pass
