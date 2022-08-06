import os
import gzip
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..utility import print_df, center_value, log2_transform, check_dir, get_sep
# from ..utility import filter_gene_by_expression_min_max
# from ..plot import plot_hcluster


def get_data_from_firehose_raw(file_path, data_type):
    """
    get data from raw data file which is downloaded from firehose
    http://gdac.broadinstitute.org/
    :param file_path:
    :param data_type: raw_count	or scaled_estimate
    :return: a dataframe
    """
    print('>>> Start to read bulk expression file...')
    bulk_raw_data = pd.read_csv(file_path, sep='\t', header=[0, 1], index_col=0)
    idx = pd.IndexSlice
    if data_type not in ['raw_count', 'scaled_estimate']:
        raise TypeError('data_type should be raw_count or scaled_estimate')
    bulk_exp = bulk_raw_data.loc[:, idx[:, data_type]]  # genes by samples
    if data_type == 'scaled_estimate':
        bulk_exp = bulk_exp * 1e6
    bulk_exp.columns = bulk_exp.columns.get_level_values(0)
    # print_df(bulk_exp)
    return bulk_exp


def get_gene_len(file_path):
    """
    get gene length (only exon)
    gene info file downloaded from https://gdc.cancer.gov/about-data/gdc-data-processing/gdc-reference-files
    - file TCGA.hg19.June2011.gaf
    :param file_path:
    :return:
    """
    gene_info = pd.read_csv(file_path, sep='\t', index_col='FeatureID')
    gene_info['GeneLen'] = gene_info['FeatureCoordinates'].map(lambda x: int(x.split(',')[-1].split('-')[-1]))
    return gene_info.loc[:, ['GeneLen']]


def normalize_by_gene_len(bulk_exp, gene_len):
    """
    normalize bulk expression by gene length
    :param bulk_exp: a data frame contains all expression data of each sample
    :param gene_len: a data frame contains gene length information, 'GeneLen' is a column, bp
    :return: normalized bulk expression
    """
    # common_genes = list(set(bulk_exp.index.to_list()) & set(gene_len.index.to_list()))
    common_genes = [i for i in bulk_exp.index if i in gene_len.index]
    bulk_exp = bulk_exp.loc[common_genes, :]
    gene_len = gene_len.loc[common_genes, :]
    bulk_norm_by_gene_len = bulk_exp / np.vstack(gene_len['GeneLen']) * 1000
    return bulk_norm_by_gene_len


def split_cancer_normal_samples(bulk_exp_fp, sample_info_fp):
    """
    split samples by sample information to cancer group and normal group
    :param bulk_exp_fp:
    :param sample_info_fp: get from http://gdac.broadinstitute.org/runs/stddata__latest/samples_report/HNSC.html
    :return: dict
           only cancer and normal tissue bulk expression data
    """
    bulk_exp = pd.read_csv(bulk_exp_fp, index_col=0)
    sample_info = pd.read_csv(sample_info_fp, sep='\t')
    sample_info = sample_info.loc[sample_info['Protocol'] == 'RSEM_genes', :].copy()
    if '.' in bulk_exp.columns[0]:  # replaced '-' with '.' by R
        bulk_exp.columns = [i.replace('.', '-') for i in bulk_exp.columns]
    bulk_exp.index = bulk_exp.index.map(lambda x: x.split('|')[0] if x[0] != '?' else x)  # split and get gene name
    # print_df(bulk_exp)
    bulk_exp_normal_tissue = bulk_exp.loc[:, sample_info.loc[sample_info['sample type'] ==
                                                             'Solid Tissue Normal', 'TCGA Barcode']]
    # only keep Primary Solid Tumor
    bulk_exp_cancer = bulk_exp.loc[:, sample_info.loc[sample_info['sample type'].isin(['Primary Solid Tumor']),
                                                      'TCGA Barcode']]
    bulk_exp_cancer = bulk_exp_cancer.loc[~bulk_exp_cancer.index.duplicated(keep='first')]
    bulk_exp_normal_tissue = bulk_exp_normal_tissue.loc[~bulk_exp_normal_tissue.index.duplicated(keep='first')]
    return {'cancer': bulk_exp_cancer, 'normal_tissue': bulk_exp_normal_tissue}


def get_sample2subtype(bulk_exp, subtype_info_fp):
    """
    match subtype by the supplementary materials of HNSC marker paper
    set the type of samples that don't include in marker paper as 'New'
    :param bulk_exp: bulk expression dataframe, only use sample names in this bulk expression dataframe
    :param subtype_info_fp: a file comes from the supplementary materials of HNSC marker paper
    :return:
    """
    subtype = pd.read_excel(subtype_info_fp, index_col=0)
    subtype.index = subtype.index.map(lambda x: x.replace('.', '-'))
    sample2subtype = {}
    for s in bulk_exp.columns.to_list():
        _s = s[:12]
        if _s in subtype.index:
            _st = subtype.loc[_s, 'RNA']
        else:
            _st = 'New'
        sample2subtype[s] = _st
    sample2subtype_df = pd.DataFrame.from_dict(sample2subtype, orient='index')
    # print('  >>> Shape of sample2subtype: {}'.format(sample2subtype_df.shape))
    sample2subtype_df.rename(columns={0: 'subtype'}, inplace=True)
    return sample2subtype_df


def filter_by_corr_with_subclass(bulk_exp, subtype, corr_threshold=0.1):
    """
    filter samples by correlation with each subclass in marker paper

    :param bulk_exp: pd.DataFrame
        bulk cell expression value normalized by TMM (edgeR), don't need to log transform and center
    :param subtype: pd.DataFrame
        selected typical samples for each subtype, a dataframe with
                     ['reordered_ind', 'cluster_id', 'subtype', 'reclass'] as columns, and 'sample_name' as index
                     this information comes from the SI of marker paper
                     Nature 517, 576–582 (2015). https://doi.org/10.1038/nature14129
    :param corr_threshold:
    :return:
    """
    # df_c = df_c.loc[:, ~df_c.index.isin(subtype.index)].copy()
    mean_corr2each_subtype = pd.DataFrame(index=bulk_exp.columns)
    # print_df(df_c)
    _bulk_exp = bulk_exp.copy()
    bulk_exp = log2_transform(bulk_exp)
    bulk_exp = center_value(bulk_exp)
    sample_corr = bulk_exp.corr()
    for _st in subtype['subtype'].unique():
        if _st != 'New':
            print('current subtype: {}'.format(_st))
            sample_corr2_with_st = sample_corr.loc[:, subtype[subtype['subtype'] == _st].index].copy()
            print_df(sample_corr2_with_st)
            mean_corr2each_subtype['mean_corr2{}'.format(_st.lower()[:3])] = sample_corr2_with_st.mean(axis=1)
    samples_filtered = _bulk_exp.loc[:, np.sum(mean_corr2each_subtype.values >= corr_threshold, axis=1) == 1]
    return samples_filtered


def unzip_and_merge(sample_info_file_path: str, gdc_download_dir: str,
                    result_dir: str, log_file_path: str,
                    file_type='htseq.counts'):
    """
    Unzip and merge files downloaded from https://portal.gdc.cancer.gov/ by cancer types (Project ID).
        And only keep "Primary Tumor" (one of Sample Type).

    :param sample_info_file_path: gdc_sample_sheet, downloaded from GDC.
        A .tsv file contains File ID, File Name, Data, Category, Data Type, Project ID, Case ID, Sample ID, Sample Type

    :param gdc_download_dir: file folder contains downloaded data from GDC
        One sample one .htseq.counts.gz file.
    :param result_dir: where to save merged files

    :param log_file_path: log file path

    :param file_type: file type downloaded from GDC website

    :return: None
    """
    print('Start to unzip downloaded files...')
    check_dir(result_dir)
    sample_info = pd.read_csv(sample_info_file_path, sep='\t')
    # only keep primary tumor
    sample_info = sample_info.loc[sample_info['Sample Type'] == 'Primary Tumor', :]
    cancer_type2file_name = {}
    file_name_counter = {}
    file_name2new_path = {}
    for _, row in tqdm(sample_info.iterrows()):
        cancer_type = row['Project ID'].split('-')[1]
        dir1 = row['File ID']
        file_name = row['File Name']
        file_path = os.path.join(gdc_download_dir, dir1, file_name)

        if os.path.exists(file_path):
            new_file_dir = os.path.join(result_dir, cancer_type)
            check_dir(new_file_dir)
            new_file_name = row['Sample ID'] + f'.{file_type}'
            if new_file_name not in file_name_counter:
                file_name_counter[new_file_name] = 0
            file_name_counter[new_file_name] += 1
            new_file_path = os.path.join(new_file_dir, new_file_name)
            file_name2new_path[new_file_name] = new_file_path
            if not os.path.exists(new_file_path):
                with gzip.open(file_path, 'rb') as f_in:
                    with open(new_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            if cancer_type not in cancer_type2file_name:
                cancer_type2file_name[cancer_type] = []
            cancer_type2file_name[cancer_type].append(new_file_name)
        else:
            with open(log_file_path, 'a') as f_log:
                f_log.write(f'File path not found: {file_path}' + '\n')
    for file_name, count in file_name_counter.items():
        if count > 1:  # remove duplicate files (duplicate Sample ID)
            if os.path.exists(file_name2new_path[file_name]):
                os.remove(file_name2new_path[file_name])

    # merge
    print('Start to merge files by cancer type...')
    for cancer_type, file_names in cancer_type2file_name.items():
        merged_result_file_path = os.path.join(result_dir, cancer_type, f'merged_{cancer_type}_{file_type}.csv')
        if not os.path.exists(merged_result_file_path):
            current_files = []
            for fn in file_names:
                if file_name_counter[fn] == 1:
                    current_df = pd.read_csv(file_name2new_path[fn], index_col=0, header=None, sep='\t')
                    current_df.columns = [fn.split('.')[0]]
                    current_df.index.name = 'gene_id'
                    current_files.append(current_df)
            merged_df = pd.concat(current_files, axis=1)
            print(f'    {cancer_type}: {merged_df.shape}')
            merged_df.to_csv(merged_result_file_path)


def read_counts2tpm(read_counts_file_path: str, annotation_file_path: str,
                    result_dir: str, file_type='htseq.counts', file_name_prefix: str = ''):
    """
    Convert read counts (htseq.counts) to TPM (transcript per million)

    :param read_counts_file_path: the file path of merged read counts file (.htseq.counts),
        separated by tab or comma, gene by sample

    :param result_dir: the folder of saving result files

    :param annotation_file_path:  file path of gencode.gene.info.v22.tsv
        download from https://api.gdc.cancer.gov/data/b011ee3e-14d8-4a97-aed4-e0b10f6bbe82
        or other annotation files, gene_type, gene_name and exon_length should be included

    :param file_type: htseq.counts, raw data type downloaded from https://portal.gdc.cancer.gov/

    :param file_name_prefix: prefix of result file, only for naming

    :return: None
    """
    print(f'   Start to convert {file_type} to TPM...')
    # current_result_dir = os.path.dirname(result_dir)
    check_dir(result_dir)

    sep = get_sep(read_counts_file_path)
    read_counts = pd.read_csv(read_counts_file_path, index_col=0, sep=sep)
    index_type = 'gene_name'
    if 'ENSG' in read_counts.index[0]:
        index_type = 'gene_id'
    sep2 = get_sep(annotation_file_path, comment='#')
    anno = pd.read_csv(annotation_file_path, index_col=index_type, sep=sep2, comment='#')
    if index_type == 'gene_name':
        anno['gene_name'] = anno.index
    anno = anno.loc[anno['gene_type'] == 'protein_coding', ['gene_name', 'exon_length']]

    # only keep protein coding genes and convert Ensembl gene id to gene symbol
    filtered_read_counts_file_path = os.path.join(result_dir, f'{file_name_prefix}_{file_type}.csv')
    tpm_file_path = os.path.join(result_dir, f'{file_name_prefix}_TPM.csv')
    log2_trans_file_path = os.path.join(result_dir, f'{file_name_prefix}_log2tpm1p.csv')
    if not os.path.exists(tpm_file_path):
        merged_file = anno.merge(read_counts, left_index=True, right_index=True)
        merged_file.set_index('gene_name', inplace=True)
        duplicated_inx_bool = merged_file.index.duplicated(keep='first')
        if np.any(duplicated_inx_bool):
            print('   Remove duplicated genes by keeping the first row...')
            merged_file = merged_file[~duplicated_inx_bool]
        merged_file.iloc[:, 1:].to_csv(filtered_read_counts_file_path, float_format='%.3f')
        norm_by_gene_len = merged_file.iloc[:, 1:] / np.vstack(merged_file['exon_length']) * 1000
        tpm = norm_by_gene_len / np.hstack(norm_by_gene_len.sum(axis=0)) * 1e6
        n, m = tpm.shape  # gene by sample
        print('   There are {} genes and {} samples.'.format(n, m))
        tpm.to_csv(tpm_file_path, float_format='%.3f')
        tpm_t = tpm.T
        log2tpm1p = log2_transform(tpm_t)
        log2tpm1p.to_csv(log2_trans_file_path, float_format='%.3f')
        print('   Converting finished.')
    else:
        print('   Previous result has existed in file {}.'.format(tpm_file_path))
