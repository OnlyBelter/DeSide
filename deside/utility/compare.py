import os
import warnings
import pandas as pd
from scipy.stats import percentileofscore
from .pub_func import check_dir, read_marker_gene, read_df, sorted_cell_types
from ..utility import log_exp2cpm, non_log2cpm, aggregate_marker_gene_exp

name_mapping_CIBERSORT = {"B cells": 'B Cells', "T cells CD8": 'CD8 T', "T cells CD4": 'CD4 T',
                          "Macrophages": 'Macrophages', "Dendritic cells": 'DC',
                          "Mast cells": 'Mast Cells', "Myocytes": 'myocyte', "CAFs": 'Fibroblasts',
                          "Endothelial cells": 'Endothelial cells', "Malignant cells": 'Cancer Cells'}


def parse_result_from_cibersort(result_file_path):
    """

    :param result_file_path:
    :return:
    """
    name_mapping_reverse = {}
    for i, j in name_mapping_CIBERSORT.items():
        if j not in name_mapping_reverse:
            name_mapping_reverse[j] = []
        name_mapping_reverse[j].append(i)
    result_from_cibersort = pd.read_csv(result_file_path, index_col=0)
    result = pd.DataFrame(index=result_from_cibersort.index, columns=list(name_mapping_reverse.keys()))
    for col in result.columns:
        result[col] = result_from_cibersort.loc[:, name_mapping_reverse[col]].sum(axis=1)
    cell_types = [i for i in sorted_cell_types if i in result.columns]
    return result.loc[:, cell_types]


def _read_result(file_path, cell_type_name_mapping, cell_types, algo=None):
    """
    read a single result file from different algorithms for performance comparison ()
    :param file_path:
    :param cell_type_name_mapping: mapping from cell type names of current algo to cell type names in DeSide
    :param algo: the name of each algorithm, CIBERSORT, MuSiC, EPIC, Scaden, Scaden-web and DeSide
    :return:
    """
    predicted_result = read_df(file_path)
    predicted_result.rename(columns=cell_type_name_mapping, inplace=True)
    predicted_result.index = predicted_result.index.map(lambda x: x.replace('.', '-'))
    return predicted_result.loc[:, cell_types]


def read_and_merge_result(raw_result_dir: str, cell_type_name_mapping: dict, algo: str,
                          result_file_path=None, tcga_sample2cancer_type_file_path=None) -> pd.DataFrame:
    """
    read and merge predicted cell fractions of each algorithm
    :param raw_result_dir:
    :param cell_type_name_mapping: mapping from cell type names of current algo to cell type names in DeSide
    :param algo: EPIC, CIBERSORT, MuSiC, DeSide and Scaden
    :param result_file_path:
    :param tcga_sample2cancer_type_file_path: the file path of sample id to cancer type mapping file in TCGA
    :return:
    """
    cell_types = [i for i in sorted_cell_types if i in list(cell_type_name_mapping.values())]
    cancer_dataset2file_path = {}
    cancer_type = ''
    ds = ''  # reference dataset
    assert os.path.exists(raw_result_dir), '{} does not exist'.format(raw_result_dir)
    for root, dirs, files in os.walk(raw_result_dir):
        for file_name in files:
            if (('.txt' in file_name) or (f'.csv' in file_name)) and \
                    ('cancer_purity' not in file_name) and ('signature_score' not in file_name):
                if 'CIBERSORT' in algo:
                    _, _, _, cancer_type, ds, _ = file_name.split('_')
                elif algo == 'MuSiC':
                    cancer_type, ds, _ = file_name.split('_')
                    ds = ds.replace('LUAD1', 'LUAD')
                elif 'EPIC' in algo:
                    if 'NOref' in file_name:
                        _, cancer_type, ds = file_name.split('_')
                    else:
                        cancer_type, _ = file_name.split('.')
                    if algo == 'EPIC':
                        ds = '46sig'
                    else:  # EPIC_self_ref
                        ds = 'self'
                elif algo == 'Scaden_ascites':
                    cancer_type = file_name.split('_')[-1].replace('.txt', '')
                    ds = 'Ascites'
                elif algo == 'Scaden_simu_bulk':
                    cancer_type = file_name.split('_')[-1].replace('.txt', '')
                    ds = 'simu_bulk_2ds'
                elif 'DeSide' in algo:
                    cancer_type = root.split(os.path.sep)[-1]
                    if '/' in cancer_type:
                        cancer_type = cancer_type.split('/')[-1]
                    ds = 'simu_bulk_2ds'
                    if algo == 'DeSide_softmax':
                        ds = 'simu_bulk_2ds_softmax'
                elif algo == 'Kassandra_self':
                    cancer_type = 'all'
                    ds = 'self'
                if '.txt' in ds:
                    ds = ds.replace('.txt', '')
                cancer_type = cancer_type.replace('HNSCC', 'HNSC')
                ds = ds.replace('HNSCC', 'HNSC')
                assert os.path.exists(os.path.join(root, file_name)), f'{os.path.join(root, file_name)} does not exist'
                cancer_dataset2file_path[cancer_type + '-' + ds + '_ref'] = os.path.join(root, file_name)
    # print(cancer_dataset2file_path)
    sample2cell_frac = {}
    counter = 0
    columns = ['sample_id', 'cancer_type', 'reference_dataset'] + cell_types

    for cancer_dataset, file_path in cancer_dataset2file_path.items():
        # print(cancer_dataset, file_path)
        cancer_type, ref_dataset = cancer_dataset.split('-')
        if 'Scaden' in algo:
            ref_dataset = ref_dataset.replace('Scaden_', '')
        elif algo == 'DeSide':
            ref_dataset = ref_dataset.replace('DeSide_', '')
        if ref_dataset == 'ref':
            ref_dataset = 'Mixed_ref'
        current_result = _read_result(file_path, cell_type_name_mapping=cell_type_name_mapping,
                                      cell_types=cell_types, algo=algo)
        if algo == 'Kassandra_self':
            current_result = current_result / 100
        for row in current_result.iterrows():
            sample_id = row[0]
            sample2cell_frac[counter] = [sample_id, cancer_type, ref_dataset] + list(row[1].values)
            counter += 1
    merged_result = pd.DataFrame.from_dict(sample2cell_frac, orient='index', columns=columns)
    if algo == 'Kassandra_self':
        tcga_sample2cancer_type = pd.read_csv(tcga_sample2cancer_type_file_path, index_col=0)
        tcga_sample2cancer_type.index = tcga_sample2cancer_type.index.map(lambda x: x.replace('.', '-'))
        tcga_sample2cancer_type = tcga_sample2cancer_type.to_dict()['cancer_type']
        merged_result.drop(columns=['cancer_type'], inplace=True)
        merged_result['cancer_type'] = merged_result['sample_id'].map(lambda x: tcga_sample2cancer_type[x])
        merged_result = merged_result.loc[:, columns].copy()
    if result_file_path:
        if os.path.exists(result_file_path):
            merged_result.to_csv(result_file_path, mode='a', header=False, float_format='%.3f')
        else:
            merged_result.to_csv(result_file_path, float_format='%.3f')
    else:
        return merged_result


def mean_exp_of_marker_gene(marker_gene_file_path, bulk_tpm_file_path, result_file_path: str = None,
                            cancer_type: str = None, debug=False, trans=False, cell_types: list = None,
                            log_exp: bool = False, gene_list_in_model: list = None):
    """
    mean expression value of marker genes for each cell type (max expression value for B Cells and CD4 T Cells)
    :param marker_gene_file_path: file path of selected marker gene for each cell type
    :param bulk_tpm_file_path: file path of bulk cell TPM, should be gene by sample
    :param result_file_path
    :param cancer_type: tumor type / cancer type / test set name
    :param debug: if output intermediate result for debug
    :param trans: taking transposition if the bulk cell TPM is organized with sample by gene
    :param cell_types: cell types
    :param log_exp: TPM is need. If the expression values are log transformed, it should be transformed to TPM
    :param gene_list_in_model: filtering bulk_tpm and rescaling to TPM if not None
    :return: a dataframe, sample by cell type (mean expression of marker genes), and save to file
    """
    if result_file_path is not None:
        result_dir = os.path.dirname(result_file_path)
        check_dir(result_dir)
    else:
        result_dir = '.'
    temporal_dir = os.path.join(result_dir, 'debug_temp', cancer_type)
    if debug:
        check_dir(temporal_dir)
    include_t_cell = False
    if (cell_types is not None) and ('T Cells' in cell_types):
        include_t_cell = True
        cell_types += ['CD8 T', 'CD4 T']
    cell_type2marker = read_marker_gene(marker_gene_file_path, include_t_cell=include_t_cell,
                                        use_cancer_cell=True, corr_mean=True)
    cell_type2marker = {k: v for k, v in cell_type2marker.items() if k in cell_types}
    # if ('Cancer Cells' not in cell_type2marker) and ('Epithelial Cells' in cell_type2marker):
    #     cell_type2marker['Cancer Cells'] = cell_type2marker['Epithelial Cells']
    # sep = get_sep(bulk_tpm_file_path)
    # tpm = pd.read_csv(bulk_tpm_file_path, index_col=0, sep=sep)
    tpm = read_df(bulk_tpm_file_path)
    if trans:
        tpm = tpm.T
    if log_exp:
        tpm = log_exp2cpm(tpm)
    if gene_list_in_model is not None:
        intersection_genes = [i for i in gene_list_in_model if i in tpm.index]
        if len(intersection_genes) != len(gene_list_in_model):
            n_diff = len(gene_list_in_model) - len(intersection_genes)
            warnings.warn(f'There are {n_diff} genes don\'t include in current bulk TPM dataset')
        tpm = tpm.loc[tpm.index.isin(intersection_genes), :].copy()
        tpm = non_log2cpm(tpm.T).T  # rescaling to TPM after filtering by gene list in model
        # print(tpm.sum(axis=0))

    marker_exp = aggregate_marker_gene_exp(exp_df=tpm.T, marker_genes=cell_type2marker)
    return marker_exp


def cal_gene_signature_score(marker_gene_file_path, bulk_tpm_file_path, result_file_path: str = None,
                             cancer_type: str = None, trans=False, cell_types: list = None):
    """
    gene signature score for each cell type,
      ref to Combes et al., 2022, Cell185, 184–203 (METHOD DETAILS, Gene Signature Score)
    :param marker_gene_file_path: file path of selected marker gene for each cell type
    :param bulk_tpm_file_path: file path of bulk cell TPM, should be gene by sample
    :param result_file_path
    :param cancer_type: tumor type / cancer type / test set name
    :param trans: taking transposition if the bulk cell TPM is organized with sample by gene
    :param cell_types: cell types
    :return: a dataframe, sample by cell type (gene signature score for each cell type of each sample),
      and save to file
    """

    include_t_cell = False
    if (cell_types is not None) and ('T Cells' in cell_types):
        include_t_cell = True
        cell_types += ['CD8 T', 'CD4 T']
    cell_type2marker = read_marker_gene(marker_gene_file_path, include_t_cell=include_t_cell,
                                        use_cancer_cell=True)
    # sep = get_sep(bulk_tpm_file_path)
    # tpm = pd.read_csv(bulk_tpm_file_path, index_col=0, sep=sep)
    tpm = read_df(bulk_tpm_file_path)
    if trans:
        tpm = tpm.T

    cell_type2signature_score = {}
    if cell_types is None:
        cell_types = list(cell_type2marker.keys())
    for ct in cell_types:
        marker = cell_type2marker[ct]
        current_tpm = tpm.loc[tpm.index.isin(marker), :]  # gene by sample, E, m x n matrix of gene expression
        current_valid_marker = current_tpm.index.to_list()
        if len(current_valid_marker) > 0:
            current_marker_str = ', '.join(current_valid_marker)
            print(f'   Using {len(current_valid_marker)} marker genes for cell type {ct}: {current_marker_str}')
            # cell_type2mean_exp[ct + '_marker_mean'] = current_tpm.mean(axis=0)
            p = pd.DataFrame(index=current_tpm.index, columns=current_tpm.columns)
            for inx, _row in current_tpm.iterrows():
                p.loc[inx] = [percentileofscore(_row, g_exp) for g_exp in _row]
            p = p.mean(axis=0)
            cell_type2signature_score[ct + f'_gene_signature_score'] = \
                pd.Series(data=[percentileofscore(p, _) for _ in p], index=p.index)
        else:
            Warning(f'No any marker genes in bulk cell TPM for cell type {ct}, '
                    f'this cell type will be ignored in later analysis.')
    signature_score = pd.DataFrame.from_dict(cell_type2signature_score, orient='columns')
    if cancer_type is not None:
        signature_score['cancer_type'] = cancer_type
    if result_file_path is not None:
        signature_score.to_csv(result_file_path, float_format='%.3f')
    return signature_score.round(3)


if __name__ == '__main__':
    pass
    # algo2merged_file_path = {
    #     # 'CIBERSORT': './merged_result/CIBERSORT_predicted_cell_fraction.csv',
    #                          'DeSide': './merged_result/DeSide_predicted_cell_fraction.csv',
    #                          # 'EPIC': './merged_result/EPIC_predicted_cell_fraction.csv',
    #                          # 'Scaden': './merged_result/Scaden_predicted_cell_fraction.csv',
    #                          # 'MuSiC': './merged_result/MuSiC_predicted_cell_fraction.csv'
    # }
    # algo2raw_result_dir = {'CIBERSORT': r'D:\project001_data\DeSide_example\04predict_and_compare\CIBERSORT_result',
    #                        'DeSide': r'D:\project001_data\DeSide_example\04predict_and_compare\DeSide_result',
    #                        'EPIC': r'D:\project001_data\DeSide_example\04predict_and_compare\EPIC_result-sig-top-30',
    #                        'Scaden': r'D:\project001_data\DeSide_example\04predict_and_compare\Scaden_result',
    #                        'MuSiC': r'D:\project001_data\DeSide_example\04predict_and_compare\MuSiC_result'}
    # algo2cell_types = {'CIBERSORT': ['CD8 T', 'Cancer Cells'],
    #                    'DeSide': ['CD8 T', 'Cancer Cells', '1-others'],
    #                    'EPIC': ['CD8 T', 'Cancer Cells', 'otherCells'],
    #                    'Scaden': ['CD8 T', 'Cancer Cells'],
    #                    'MuSiC': ['CD8 T', 'Cancer Cells']}
    # for algo, m_fp in algo2merged_file_path.items():
    #     print(f'Merge the results of {algo}...')
    #     read_and_merge_result(raw_result_dir=algo2raw_result_dir[algo], algo=algo,
    #                           cell_type=algo2cell_types[algo])
