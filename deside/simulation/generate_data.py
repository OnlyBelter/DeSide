import os
# import gc
import json
import numpy as np
import pandas as pd
from scipy import stats
# import scanpy as sc
from typing import Union
from tqdm import tqdm
import multiprocessing

from joblib import load, dump
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from ..utility import (create_h5ad_dataset, check_dir, cal_corr_gene_exp_with_cell_frac,
                       ExpObj, QueryNeighbors, log2_transform, print_msg, get_cell_num,
                       sorted_cell_types, do_pca_analysis, non_log2cpm)
from ..utility.read_file import ReadH5AD, read_single_cell_type_dataset, ReadExp
from ..single_cell import get_sample_id
from ..plot import plot_pca


def segment_generation_fraction(n_samples: int = None, max_value: int = 10000,
                                cell_types: list = None, sample_prefix: str = None,
                                cell_prop_prior: dict = None) -> pd.DataFrame:
    """
    Generate cell fractions by fixing a specific percentage (gradient) range (i.e., from 1% to 100%)
        for each specific cell type, and n samples for each gradient of each cell type

    :param n_samples: the number of samples needs to generate in total

    :param max_value: cell proportion will be sampled from U(0, max_value), and then scaled to [0, 1]

    :param cell_types: None or a list of cell types. Using all valid cell types if None. All valid cell types can be
        found by `list(deside.utility.cell_type2abbr.keys())`.

    :param sample_prefix: only for naming

    :param cell_prop_prior: the prior range of cell proportions for each cell type, {'cell_type': (0, 0.1), '': (0, 0.2), ...}

    :return: generated cell fraction, sample by cell type
    """
    if sample_prefix is None:
        sample_prefix = 'seg'

    n_cell_type = len(cell_types)
    # all_samples = []
    all_samples_tmp = []
    while len(all_samples_tmp) < n_samples:
        frag_for_one_sample = []
        current_max_value = max_value  # 100%
        current_sample_valid = True
        for j in range(n_cell_type - 1):
            if current_max_value > 1:  # > 1 left
                _frag = np.random.randint(current_max_value)
                current_max_value -= _frag
                frag_for_one_sample.append(_frag)
            elif current_max_value == 1:  # 1 left
                frag_for_one_sample.append(current_max_value)
                current_max_value = 0
            else:  # current_max_value == 0, 0 left
                frag_for_one_sample.append(0)
        frag_for_one_sample.append(current_max_value)  # for last fragment (0 or > 0)
        assert np.sum(frag_for_one_sample) == max_value
        np.random.shuffle(frag_for_one_sample)

        frag_for_one_sample = np.array(frag_for_one_sample) / max_value  # normalize to sum to 1
        cell_type2frag = dict(zip(cell_types, frag_for_one_sample))
        if cell_prop_prior is not None:
            for cell_type, frag in cell_type2frag.items():
                if cell_type in cell_prop_prior:
                    # check if the cell fraction is in the prior range
                    if (frag < cell_prop_prior[cell_type][0]) or (frag > cell_prop_prior[cell_type][1]):
                        current_sample_valid = False
                        break
                else:
                    Warning(
                        f'The prior range for cell type {cell_type} is not provided '
                        f'and will be ignored during the check.')
        if current_sample_valid:
            all_samples_tmp.append(cell_type2frag)

    index = [sample_prefix + '_' + str(i + 1) for i in range(len(all_samples_tmp))]
    all_samples_df = pd.DataFrame(all_samples_tmp, index=index)
    all_samples_df = all_samples_df.loc[:, cell_types]
    return all_samples_df.round(4)


def seg_random_generation_fraction(n_samples: int = None, cell_types: list = None,
                                   sample_prefix: str = None) -> pd.DataFrame:
    """
    Generate cell fraction by combining segment and random sampling method

    :param n_samples: the number of samples need to generate in total

    :param cell_types: None or a list of cell types. Using all valid cell types if None. All valid cell types can be
        found by `list(deside.utility.cell_type2abbr.keys())`.

    :param sample_prefix: only for naming

    :return: generated cell fraction, sample by cell type
    """
    if sample_prefix is None:
        sample_prefix = 'seg_random'

    # all_samples = []
    all_samples_tmp = []
    while len(all_samples_tmp) < n_samples:
        # get the first fraction
        first_segment = np.random.rand()
        others_fraction = np.random.rand(len(cell_types)-1)
        # scaling sum to 1 and then rescaling to (1-first_segment)
        others_fraction = others_fraction / others_fraction.sum() * (1 - first_segment)
        frag_for_one_sample = np.concatenate([np.array([first_segment]), others_fraction])
        frag_for_one_sample = frag_for_one_sample / frag_for_one_sample.sum()  # scaling again
        np.random.shuffle(frag_for_one_sample)
        all_samples_tmp.append(dict(zip(cell_types, frag_for_one_sample)))

    index = [sample_prefix + '_' + str(i + 1) for i in range(len(all_samples_tmp))]
    all_samples_df = pd.DataFrame(all_samples_tmp, index=index)
    all_samples_df = all_samples_df.loc[:, cell_types]
    # all_samples.append(current_df)
    return all_samples_df.round(4)


def fragment_generation_fraction(n_samples: int = None, cell_types: list = None,
                                 sample_prefix: str = None, reference_distribution: dict = None,
                                 bins: int = 10, minimal_prop: float = 0.005) -> pd.DataFrame:
    """
    Generate cell fraction by fragment sampling method

    :param n_samples: the number of samples need to generate in total

    :param cell_types: None or a list of cell types. Using all valid cell types if None. All valid cell types can be
        found by `list(deside.utility.cell_type2abbr.keys())`.

    :param sample_prefix: only for naming

    :param reference_distribution: reference distribution for each cell type, seperated to 10 bins for [0, 1]
        {'B Cells': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], '': [], '': [], ...}

    :param bins: the number of bins for each distribution

    :param minimal_prop: set to 0 if less than this value

    :return: generated cell fraction, sample by cell type
    """
    if sample_prefix is None:
        sample_prefix = 'fragment'
    bin_lower_edges = np.linspace(0, 1 - 1/bins, bins)  # array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ct2prop = {}
    if reference_distribution is None:
        reference_distribution = {}
        for ct in cell_types:
            reference_distribution[ct] = np.ones(bins) * (1 / bins)  # uniform distribution
    for ct in cell_types:
        # get cell proportions for each cell type
        current_ref_dis = reference_distribution[ct]
        current_prop = np.random.choice(bin_lower_edges, n_samples, p=current_ref_dis)  # sampling depending on ref dis
        # lower edge + x ~ U(0, 0.1) -> [lower_edge, lower_edge + 0.1]
        noise_for_each_prop = 0.1 * np.random.rand(n_samples)
        current_prop += noise_for_each_prop
        ct2prop[ct] = current_prop

    index = [sample_prefix + '_' + str(i + 1) for i in range(n_samples)]
    all_samples_df = pd.DataFrame.from_dict(ct2prop, orient='columns')
    all_samples_df.index = index
    # scaling sum to 1
    all_samples_df = all_samples_df / np.sum(all_samples_df.values, axis=1).reshape(-1, 1)
    all_samples_df[all_samples_df < minimal_prop] = 0
    all_samples_df = all_samples_df / np.sum(all_samples_df.values, axis=1).reshape(-1, 1)  # scale again
    all_samples_df = all_samples_df.loc[:, cell_types]
    # all_samples.append(current_df)
    return all_samples_df.round(4)


def _create_fractions(n_cell_types, fixed_range: dict = None):
    """
    generate (pure) random fractions
    :param n_cell_types: number of fractions to create
    :param fixed_range: the range of cell fraction for each cell type, {'cell_type': (0, 100), '': (), ...}
    :return: list of random fracs with the length n_cell_types
    """
    if (fixed_range is None) or (len(fixed_range) < n_cell_types):
        fracs = np.random.rand(n_cell_types)  # uniform distribution over [0, 1)
    else:
        fracs = []
        for cell_type, ct_range in fixed_range.items():
            _frac = np.random.randint(ct_range[0], ct_range[1]) / 100
            fracs.append(_frac)
        fracs = np.array(fracs)
    fracs_sum = np.sum(fracs)
    fracs = np.divide(fracs, fracs_sum)
    return fracs


def random_generation_fraction(n_samples: int = 100, cell_types: list = (),
                               sample_prefix: str = None, fixed_range: dict = None) -> pd.DataFrame:
    """
    Create pure random cell fractions, same as `Scaden`

    :param n_samples: number of samples to create

    :param cell_types: a list of cell types

    :param sample_prefix: prefix of sample names

    :param fixed_range: the range of cell fraction for each cell type, {'cell_type': (0, 100), '': (), ...}

    :return:  generated cell fraction, sample by cell type
    """
    if sample_prefix is None:
        sample_prefix = 's_random'
    n_cell_types = len(cell_types)
    id2cell_frac = {}

    for i in range(n_samples):
        sample_name = sample_prefix + '_' + str(i)
        id2cell_frac[sample_name] = _create_fractions(n_cell_types, fixed_range=fixed_range)
    generated_frac_df = pd.DataFrame.from_dict(id2cell_frac, orient='index', columns=cell_types)
    return generated_frac_df.round(2)


def map_cell_id2exp(sc_exp, selected_cell_id):
    """

    :param sc_exp: a AnnData object, log2(CPM + 1), samples by genes
    :param selected_cell_id: a dataFrame which contains cell_type, n_cell, selected_cell_id
    :return: a DataFrame, log2(CPM + 1), samples by genes
    """
    # sc_exp = an.read_h5ad(sc_exp)
    adata_df = pd.DataFrame(sc_exp.X.A, index=sc_exp.obs.index, columns=sc_exp.var.index)
    adata_df = np.power(2, adata_df) - 1  # convert to non-log values
    simulated_exp = {}

    for sample_id, group in selected_cell_id.groupby(by=selected_cell_id.index):
        cell_ids = []
        for i, row in group.iterrows():
            if row['n_cell'] > 0:
                cell_ids += row['selected_cell_id'].split(';')
        current_merged = adata_df.loc[cell_ids, :].copy()
        simulated_exp[sample_id] = current_merged.mean(axis=0)  # single simulated bulk expression profile

    simulated_exp_df = pd.DataFrame.from_dict(data=simulated_exp, orient='index')
    # simu_adata = an.AnnData(simulated_exp_df)
    # simu_adata.X = np.log2(simu_adata.X + 1)  # log2(CPM + 1)
    # simulated_exp_df.rename(index={i: j for i, j in enumerate(filtered_single_cell_exp.index)}, inplace=True)
    log2cpm = np.log2(simulated_exp_df.values + 1)
    return pd.DataFrame(log2cpm, index=simulated_exp_df.index, columns=sc_exp.var.index).round(2)


class BulkGEPGenerator(object):
    """
    Generate bulk GEPs from single cell datasets

    :param simu_bulk_dir: the directory to save simulated bulk cell GEPs
    :param merged_sc_dataset_file_path: the file path of pre-merged single cell datasets
    :param sct_dataset_file_path: the file path of single cell datasets (scGEP, dataset `S1`)
    :param cell_type2subtype: cell types used when generating bulk GEPs, {'cell_type': ['subtype1', 'subtype2', ...], ...}'
    :param sc_dataset_ids: single cell dataset id used when generating bulk GEPs
    :param bulk_dataset_name: the name of generated bulk dataset, only for naming
    :param check_basic_info: whether to check basic information of single cell datasets
    :param zero_ratio_threshold: the threshold of zero ratio of genes in single cell GEPs, remove the GEP if zero ratio > threshold
    :param sc_dataset_gep_type: the type of single cell GEPs, `log_space` or `linear_space`
    :param tcga2cancer_type_file_path: the file path of `tcga_sample_id2cancer_type.csv`, which contains the cancer type of TCGA samples
    :param total_rna_coefficient: the coefficient of total RNA, used to correct the difference of total RNA for each cell type
    :param subtype_col_name: the column name of subtype in single cell datasets
    :param cell_type_col_name: the column name of cell type in single cell datasets
    """
    def __init__(self, simu_bulk_dir, merged_sc_dataset_file_path, sct_dataset_file_path,
                 cell_type2subtype: dict, sc_dataset_ids: list, bulk_dataset_name: str = None,
                 check_basic_info: bool = True, zero_ratio_threshold: float = 0.97,
                 sc_dataset_gep_type: str = 'log_space', tcga2cancer_type_file_path: str = None,
                 total_rna_coefficient: dict = None, subtype_col_name: str = None, cell_type_col_name: str = None):
        """
        """
        self.simu_bulk_dir = simu_bulk_dir  # result dir
        check_dir(simu_bulk_dir)
        self.merged_sc_fp = merged_sc_dataset_file_path
        self.cell_type_in_sc = None  # cell types in merged single cell datasets
        self.cell_subtype_in_sc = None  # cell subtypes in merged single cell datasets
        self.dataset_in_sc = None  # single cell datasets were merged together
        self.cell_type2subtype = cell_type2subtype
        # self.cell_subtype_used only contains cell subtypes, not including cell types.
        # If cell subtypes were not needed, self.cell_subtype_used is []
        self.cell_type_used, self.cell_subtype_used = self.get_cell_type_used(cell_type2subtype)
        self.sc_dataset_used = sc_dataset_ids
        self.merged_sc_dataset = None
        self.generated_sc_fp = None  # the file path of generated single cell dataset
        # self.generated_sc_dataset = None
        self.generated_sc_dataset_obs = None
        self.generated_sc_dataset_df = None  # CPM GEPs of generated single cell dataset
        self.bulk_dataset_name = bulk_dataset_name
        self.n_samples = None  # the number of generated bulk cell GEPs
        self.generated_bulk_gep = None
        # self.generated_bulk_gep_fp = None  # .h5ad file path of generated bulk GEPs, log2(TPM+1)
        self.generated_bulk_gep_counter = 0
        self.n_round = 0
        self.generated_cell_frac = None
        self.total_cell_number = None  # N, average to form a bulk GEP
        self.q_dis_nn_ref_upper: float = 0  # the quantile of distance for each pair in reference dataset, such as TCGA
        self.q_dis_nn_ref_lower: float = 0  # the quantile of distance for each pair in reference dataset, such as TCGA
        self.filtering_quantile_upper = 0  # the upper quantile used to determine q_dis_nn_ref for sample filtering
        self.filtering_quantile_lower = None  # the lower quantile used to determine q_dis_nn_ref for sample filtering
        self.marker_ratio_ref = None  # the marker gene ratios of reference dataset
        prefix = f'simu_bulk_exp_{self.bulk_dataset_name}'
        self.generated_cell_fraction_fp = os.path.join(self.simu_bulk_dir,
                                                       f'generated_frac_{self.bulk_dataset_name}.csv')
        self.ref_neighbor_counter_fp = os.path.join(self.simu_bulk_dir, f'ref2n_neighbors_{self.bulk_dataset_name}.csv')
        self.generated_bulk_gep_csv_fp = os.path.join(self.simu_bulk_dir, prefix + f'_log2cpm1p.csv')
        self.sampled_sc_cell_id_file_path = os.path.join(self.simu_bulk_dir, prefix + '_sampled_sc_cell_id.csv')
        self.generated_bulk_gep_fp = self.generated_bulk_gep_csv_fp.replace('.csv', '.h5ad')  # final result
        self.n_neighbors_each_ref = 1  # control the distribution of marker ratio by reference dataset
        self.ref_neighbor_counter = {}
        self.merged_sc_dataset_obs = None
        self.merged_sc_dataset_df = None  # CPM GEPs of merged single cell dataset
        self.obs_df = None  # used for sampling
        self.sct_dataset_df = None  # only used in subclass "BulkGEPGeneratorSCT"
        self.zero_ratio_threshold = zero_ratio_threshold
        self.sct_dataset_file_path = sct_dataset_file_path
        self.sct_dataset_obs = None
        self.m_gep_ref = None  # median / mean GEP of reference dataset
        # unique expression values in scRNA-seq dataset saved in merged_7_sc_dataset_log2cpm1p.h5ad
        self.sc_dataset_gep_type = sc_dataset_gep_type
        self.tcga2cancer_type_file_path = tcga2cancer_type_file_path
        self.total_rna_coefficient = total_rna_coefficient
        self.subtype_col_name = subtype_col_name
        self.cell_type_col_name = cell_type_col_name
        if check_basic_info and not os.path.exists(self.generated_bulk_gep_fp):
            self._check_basic_info()

    def _generate_cell_fraction(self, sampling_method: str, n_cell_frac: int, sampling_range: dict = None,
                                sample_prefix: str = None, ref_distribution: dict = None,
                                random_n_cell_type: list = None, cell_prop_prior: dict = None):
        """
        Generate cell proportions for each simulated bulk GEP

        :param sampling_method: 'segment' or 'random'
        :param n_cell_frac: the number of GEPs to generate
        :param sampling_range: the range of sampling, such as {'cell_type1': [0.1, 0.9], 'cell_type2': [0.1, 0.9], ...}
        :param sample_prefix: the prefix of sample name, such as 'sample1', 'sample2', ...
        :param ref_distribution: the reference distribution of cell fractions, such as {'cell_type1': [0.1, 0.2, 0.3], ...}
        :param random_n_cell_type: the number of cell types to randomly select from reference distribution
        :param cell_prop_prior: the prior of cell proportions, such as {'cell_type1': 0.1, 'cell_type2': 0.2, ...}
        """
        cell_type_with_subtype = []
        cell_type_contain_subtype = []
        for k, v in self.cell_type2subtype.items():
            if len(v) == 1 and v[0] == k:
                cell_type_with_subtype.append(k)
            else:
                cell_type_with_subtype.extend(v)
                cell_type_contain_subtype.append(k)
        if sampling_method == 'segment':
            gen_cell_fracs = segment_generation_fraction(n_samples=n_cell_frac,
                                                         max_value=10000,
                                                         sample_prefix=sample_prefix,
                                                         cell_types=cell_type_with_subtype,
                                                         cell_prop_prior=cell_prop_prior)
            # if len(cell_type_contain_subtype) > 0:
            #     for cell_type in cell_type_contain_subtype:
            #         _cell_fracs = segment_generation_fraction(
            #             n_samples=n_cell_frac,
            #             max_value=10000,
            #             sample_prefix=sample_prefix,
            #             cell_types=self.cell_type2subtype[cell_type],
            #             cell_prop_prior=None
            #         )
            #         _cell_fracs = _cell_fracs * gen_cell_fracs[cell_type].values.reshape(-1, 1)
            #         for subtype in self.cell_type2subtype[cell_type]:
            #             gen_cell_fracs[subtype] = _cell_fracs[subtype]
            #         gen_cell_fracs.pop(cell_type)

        elif sampling_method == 'seg_random':
            gen_cell_fracs = seg_random_generation_fraction(n_samples=n_cell_frac,
                                                            sample_prefix=sample_prefix,
                                                            cell_types=cell_type_with_subtype)

        elif sampling_method == 'fragment':
            if random_n_cell_type is None:
                gen_cell_fracs = fragment_generation_fraction(n_samples=n_cell_frac,
                                                              sample_prefix=sample_prefix,
                                                              cell_types=cell_type_with_subtype,
                                                              reference_distribution=ref_distribution)
            else:
                gen_cell_frac_list = []
                n_sample_for_each_n_cell_type = int(n_cell_frac / len(random_n_cell_type))
                for n_cell_type in random_n_cell_type:
                    if 2 <= n_cell_type <= len(cell_type_with_subtype):
                        cell_types = cell_type_with_subtype[:n_cell_type]  # generate by fixed cell types, then shuffle
                        _gen_cell_fracs = fragment_generation_fraction(n_samples=n_sample_for_each_n_cell_type,
                                                                       sample_prefix=f'{sample_prefix}_{n_cell_type}',
                                                                       cell_types=cell_types,
                                                                       reference_distribution=ref_distribution)
                        gen_cell_frac_list.append(_gen_cell_fracs)
                    else:
                        raise ValueError(f'All numbers in "random_n_cell_type" should be >= 2 and '
                                         f'<= the number of used cell types,  {n_cell_type} got.')
                gen_cell_fracs_total = pd.concat(gen_cell_frac_list)
                gen_cell_fracs_total = gen_cell_fracs_total.fillna(0)
                # shuffle_inx = np.zeros(gen_cell_fracs.shape, dtype=np.int16)
                inx2cell_frac = {}
                for i in range(gen_cell_fracs_total.shape[0]):
                    shuffle_inx = shuffle(np.arange(gen_cell_fracs_total.shape[1]))
                    inx2cell_frac[i] = gen_cell_fracs_total.values[i, shuffle_inx]  # shuffle each row
                gen_cell_fracs = pd.DataFrame.from_dict(inx2cell_frac, orient='index',
                                                        columns=gen_cell_fracs_total.columns)
                gen_cell_fracs.index = gen_cell_fracs_total.index

        elif sampling_method == 'random':
            if sampling_range is not None:
                # gradient_range = sampling_range.copy()
                if len(sampling_range) < len(cell_type_with_subtype):
                    print('   * Since the length of gradient_range is less than cell types, gradient range '
                          'will not be used. Instead, [0, 1] range will be used for all cell types...')
                    sampling_range = None
            gen_cell_fracs = random_generation_fraction(n_samples=n_cell_frac,
                                                        cell_types=cell_type_with_subtype,
                                                        sample_prefix=sample_prefix,
                                                        fixed_range=sampling_range)
        elif sampling_method == 'dirichlet':
            _gen_cell_fracs = np.random.dirichlet(np.ones(len(cell_type_with_subtype)) / len(cell_type_with_subtype),
                                                  size=n_cell_frac)
            index = [sample_prefix + '_' + str(i + 1) for i in range(n_cell_frac)]
            gen_cell_fracs = pd.DataFrame(_gen_cell_fracs, index=index, columns=cell_type_with_subtype)
        else:
            raise ValueError(
                f'Only "segment", "dirichlet" and "random" were supported for parameter "sampling_method", '
                f'{sampling_method} is invalid')
        return gen_cell_fracs

    @staticmethod
    def get_cell_type_used(cell_type2subtype: dict):
        cell_type_used = list(cell_type2subtype.keys())
        sub_cell_type_used = [i for v in cell_type2subtype.values() for i in v if i not in cell_type_used]

        return cell_type_used, sub_cell_type_used

    def generate_gep(self, n_samples, sampling_range: dict = None, sampling_method: str = 'segment',
                     total_cell_number: int = 100, n_threads: int = 10, filtering: bool = True,
                     reference_file: Union[str, pd.DataFrame] = None, ref_exp_type: str = None,
                     gep_filtering_quantile: tuple = (None, 0.95), log_file_path: str = None,
                     n_top: int = 20, simu_method='mul', filtering_method='media_gep',
                     add_noise: bool = False, noise_params: tuple = (), filtering_ref_types: list = None,
                     show_filtering_info: bool = False, cell_prop_prior: dict = None,
                     high_corr_gene_list: list = None, filtering_by_gene_range: bool = False,
                     min_percentage_within_gene_range: float = 0.95, gene_quantile_range: list = None,
                     filtering_in_pca_space: bool = False, pca_n_components: Union[int, float] = 0.9, norm_ord=1):
        """
        Generating simulated bulk GEPs from scGEP dataset (`S1`)

        :param n_samples: the number of GEPs to generate
        :param total_cell_number: N, the total number of cells sampled from merged single cell dataset
                                      and averaged to simulate a single bulk RNA-seq sample
        :param sampling_method: segment or random, method to generate cell fractions
        :param sampling_range: the range of sampling, such as {'cell_type1': [0.1, 0.9], 'cell_type2': [0.1, 0.9], ...},
            optional, only used when sampling_method is `random`
        :param n_threads: number of threads used for parallel computing
        :param filtering: whether filtering generated GEPs
        :param reference_file: the file path of reference dataset for filtering
        :param ref_exp_type: the type of expression values in reference dataset, TPM / log_space
        :param gep_filtering_quantile: quantile of nearest distance of each pair in reference,
            smaller quantile gives smaller radius and fewer simulated GEPs will be kept
        :param log_file_path: the path of log file
        :param n_top: if too many neighbors were founded for one single sample, only keep n_top neighbors,
            used in marker ratio filtering
        :param simu_method: the method to generate simulated bulk GEPs,
            ave (average all selected single cell GEPs), mul (multiple GEP by cell fractions)
        :param filtering_method: marker_ratio (l2 distance with marker gene ratio) or
            median_gep (`l1` distance with median expression value for each gene) or
            linear_mmd (A. Gretton et al., J. Mach. Learn. Res. 13, 723–773 (2012)., also see:
            https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py)
        :param add_noise: whether add noise to generated bulk GEPs
        :param noise_params: parameters for noise generation, (f, max_sum),
            ref: Hao, Yuning, et al. PLoS Computational Biology, 2019
        :param filtering_ref_types: the cancer types used for filtering
        :param show_filtering_info: whether show filtering information
        :param cell_prop_prior: a prior range of cell proportions for each cell type in solid tumors
        :param high_corr_gene_list: a list of genes that the expression values have high correlation with
            the cell proportions for at least one cell type
        :param filtering_by_gene_range: whether filtering GEPs by gene expression range,
            the percentage of genes within a specific quantile range in TCGA
        :param min_percentage_within_gene_range: the minimal percentage of genes within a specific quantile range in TCGA
        :param gene_quantile_range: the quantile range of gene expression values in TCGA for gene based filtering
        :param filtering_in_pca_space: whether filtering GEPs in PCA space
        :param pca_n_components: the number of components used for PCA, or the explained variance ratio of PCA if float
        :param norm_ord: the order of norm used for filtering, 1 for l1 norm, 2 for l2 norm
        """
        n_total_cpus = multiprocessing.cpu_count()
        n_threads = min(n_total_cpus - 1, n_threads)
        self.n_samples = n_samples
        self.total_cell_number = total_cell_number
        self.filtering_quantile_lower, self.filtering_quantile_upper = gep_filtering_quantile
        if 'linear_mmd' in filtering_method:
            # calculate the linear MMD, square of L2 norm
            norm_ord = 2
        # print(f'   > filtering quantile is {self.filtering_quantile * 100}%')
        # simulating bulk cell GEP by mixing GEPs of different cell types
        if filtering:
            min_n_cell_frac = np.min([5000, n_samples])  # bigger number of samples for filtering
        else:
            min_n_cell_frac = np.min([1000, n_samples])  # smaller number of samples without filtering
        if not os.path.exists(self.generated_bulk_gep_fp):
            # read generated single cell dataset into self.generated_sc_dataset
            self._check_intermediate_generated_gep()
            # using merged single cell dataset directly
            obs_df = self.merged_sc_dataset_obs
            sc_dataset = 'merged_sc_dataset'
            gene_list_in_sc_ds = []
            if (self.merged_sc_fp is None) and (self.sct_dataset_file_path is None):
                raise FileNotFoundError('Either "merged_sc_dataset_file_path" or '
                                        '"sct_dataset_file_path" should be provided')
            if (self.merged_sc_fp is not None) and (self.merged_sc_dataset_df is None):
                if self.sc_dataset_gep_type == 'log_space':
                    self.merged_sc_dataset_df = ReadH5AD(self.merged_sc_fp).get_df(convert_to_tpm=True)
                else:  # non log space
                    self.merged_sc_dataset_df = ReadH5AD(self.merged_sc_fp).get_df()
                gene_list_in_sc_ds = self.merged_sc_dataset_df.columns.to_list()
            if simu_method == 'mul':  # mul, using either merged_sc_dataset or sct_dataset
                self.total_cell_number = 1  # only 1 sample for each cell type
                if (self.sct_dataset_file_path is not None) and (self.sct_dataset_df is None):
                    self.sct_dataset_obs, self.sct_dataset_df = self._read_sct_dataset()
                    sc_dataset = 'sct_dataset'
                    obs_df = self.sct_dataset_obs
                    gene_list_in_sc_ds = self.sct_dataset_df.columns.to_list()
            if sc_dataset == 'merged_sc_dataset':
                min_n_cell_frac = 300  # since some cell types only have a small number of cells
            # n_round = 0
            sample_id_for_filtering = []
            tcga_gene_info = None
            exp_ref_df = None
            pca_model = None
            gene_list_in_pca = None
            d = 0  # the number of PCA components that explain 90% of variance (pca_explained_variance)
            if filtering and filtering_ref_types is not None:
                s2c = pd.read_csv(self.tcga2cancer_type_file_path, index_col=0)  # sample id to cancer type in TCGA
                sample_id_for_filtering = s2c.loc[s2c['cancer_type'].isin(filtering_ref_types), :].index.to_list()
                _str = ', '.join(filtering_ref_types)
                print(f'   > {len(sample_id_for_filtering)} samples in {_str} are used '
                      f'for {filtering_method} filtering.')
            if not filtering:
                cell_prop_prior = None
            with tqdm(total=self.n_samples) as pbar:
                if self.generated_bulk_gep_counter != 0:
                    pbar.update(self.generated_bulk_gep_counter)
                while self.generated_bulk_gep_counter < self.n_samples:
                    generated_cell_frac = self._generate_cell_fraction(
                        sampling_method=sampling_method, n_cell_frac=min_n_cell_frac,
                        sampling_range=sampling_range, sample_prefix=f's_{sampling_method}_{self.n_round}',
                        cell_prop_prior=cell_prop_prior)
                    # setting step_size equals to n_cell_frac, so n_parts equals to 1
                    selected_cell_ids = self._sc_sampling(cell_frac=generated_cell_frac,
                                                          n_threads=n_threads, obs_df=obs_df, sc_dataset=sc_dataset)
                    simulated_gep = self._map_cell_id2exp(selected_cell_id=selected_cell_ids,
                                                          simu_method=simu_method,
                                                          sc_dataset=sc_dataset,
                                                          cell_frac=generated_cell_frac,
                                                          add_noise=add_noise, noise_params=noise_params)
                    simulated_gep_bak = simulated_gep.copy()  # TPM
                    if filtering:
                        if reference_file is None or ref_exp_type is None:
                            raise ValueError('Both "reference_file" and "ref_exp_type" should not be None '
                                             'when "filtering" is True')
                        if high_corr_gene_list is not None:
                            assert np.all([i in gene_list_in_sc_ds for i in high_corr_gene_list])
                            gene_list_in_sc_ds = high_corr_gene_list
                            print(f'   > {len(gene_list_in_sc_ds)} high corr genes are used for filtering.')
                            simulated_gep = simulated_gep.loc[:, gene_list_in_sc_ds]
                            simulated_gep = non_log2cpm(simulated_gep)
                        if exp_ref_df is None:
                            exp_obj_ref = ExpObj(exp_file=reference_file, exp_type=ref_exp_type)
                            # exp_obj_ref.align_with_gene_list(gene_list=gene_list_in_sc_ds, fill_not_exist=True)
                            exp_ref_df = exp_obj_ref.get_exp()  # TPM
                            exp_ref_df = exp_ref_df.loc[exp_ref_df.index.isin(sample_id_for_filtering), :]

                    if filtering and filtering_method == 'marker_ratio':
                        # print('   Filtering simulated bulk cell GEPs by marker gene ratio of TCGA...')
                        if self.marker_ratio_ref is None:
                            exp_obj_ref = ExpObj(exp_file=reference_file, exp_type=ref_exp_type)
                            exp_obj_ref.cal_marker_gene_ratio(agg_methods={'CD4 T': 'max', 'B Cells': 'max'},
                                                              cell_types=self.cell_type_used, show_marker_gene=True)
                            self.marker_ratio_ref = exp_obj_ref.get_marker_ratios()
                            n_ref = self.marker_ratio_ref.shape[0]
                            if n_ref < self.n_samples:
                                # skip some reference samples (1000) since non-epithelial cancers exist
                                self.n_neighbors_each_ref = int(np.ceil(self.n_samples / (n_ref - 1000)))
                            if not self.ref_neighbor_counter:
                                self.ref_neighbor_counter = {i: 0 for i in self.marker_ratio_ref.index}
                            n_top = min(n_top, self.n_neighbors_each_ref)

                        simulated_gep = self._filter_gep_by_reference(simulated_gep=simulated_gep,
                                                                      n_top=n_top)
                        if (simulated_gep is None) or (simulated_gep.shape[0] < 50):
                            if self.filtering_quantile_upper < 0.999:
                                # larger filtering_quantile to get more neighbors
                                self.filtering_quantile_upper += 0.001
                            else:
                                self.filtering_quantile_upper += 0.0001
                            qn1 = QueryNeighbors(df_file=self.marker_ratio_ref)
                            self.q_dis_nn_ref_upper = qn1.get_quantile_of_nn_distance(
                                quantile=self.filtering_quantile_upper)  # quantile of distance
                            print(f'   > Larger filtering_quantile will be used to get more neighbors.')
                            print(f'   > Quantile distance of {self.filtering_quantile_upper * 100}% is: {self.q_dis_nn_ref_upper}')
                    if filtering and filtering_by_gene_range:
                        if tcga_gene_info is None:
                            if gene_quantile_range is None:
                                quantile_range = [0.005, 0.5, 0.995]
                            else:
                                quantile_range = gene_quantile_range
                            q_col_name = ['q_' + str(int(q * 1000) / 10) for q in quantile_range]
                            tcga_gene_info = get_quantile(exp_ref_df, quantile_range=quantile_range,
                                                          col_name=q_col_name)
                        valid_gep_list = []
                        for inx, row in simulated_gep.iterrows():
                            valid = True
                            current_gene_list = \
                                get_gene_list_filtered_by_quantile_range(bulk_exp=row, tcga_exp=exp_ref_df,
                                                                         tcga_gene_info=tcga_gene_info,
                                                                         quantile_range=quantile_range,
                                                                         q_col_name=q_col_name)
                            if len(current_gene_list) / exp_ref_df.shape[1] < min_percentage_within_gene_range:
                                valid = False
                            valid_gep_list.append(valid)
                        if show_filtering_info:
                            print(f'   > {np.sum(valid_gep_list)} were kept after filtering by gene range.')
                        simulated_gep = simulated_gep.loc[valid_gep_list, :].copy()

                    if filtering and (filtering_method == 'median_gep' or
                                      filtering_method == 'mean_gep' or filtering_method == 'linear_mmd') and \
                            (simulated_gep is not None):
                        if filtering_in_pca_space:
                            # TODO, this gene list should be the same as the gene list in PCA model, not the gene list in sc_ds
                            if gene_list_in_pca is None:
                                gene_list_in_pca = []
                            pca_model_dir = os.path.dirname(os.path.dirname(reference_file))
                            pca_model_dir = os.path.join(pca_model_dir, f'pca_model_{pca_n_components}')
                            check_dir(pca_model_dir)
                            pca_model_path = os.path.join(pca_model_dir, 'tcga_pca_model_for_gep_filtering.pkl')
                            gene_list_in_pca_file_path = os.path.join(pca_model_dir, 'gene_list_for_pca.csv')
                            if pca_model is None:
                                if os.path.exists(pca_model_path) and os.path.exists(gene_list_in_pca_file_path):
                                    pca_model = load(pca_model_path)
                                    gene_list_in_pca = pd.read_csv(gene_list_in_pca_file_path,
                                                                   index_col='0').index.to_list()
                                    print(f'   > PCA model was loaded from {pca_model_path}, '
                                          f'and gene list was loaded from {gene_list_in_pca_file_path}')
                                    # align reference GEPs with the gene list in the PCA model
                                    exp_obj_ref = ReadExp(exp_file=exp_ref_df, exp_type=ref_exp_type)
                                    exp_obj_ref.align_with_gene_list(gene_list=gene_list_in_pca, fill_not_exist=True)
                                    exp_ref_df = exp_obj_ref.get_exp()  # TPM
                                else:
                                    pca_model = PCA(n_components=pca_n_components, random_state=42)
                                    # using the intersection of gene list in sc_ds and gene list in TCGA
                                    gene_list_in_pca = list(set(gene_list_in_sc_ds) & set(exp_ref_df.columns))
                                    exp_ref_df = exp_ref_df.loc[:, gene_list_in_pca].copy()
                                    exp_ref_df = non_log2cpm(exp_ref_df)  # TPM
                                    exp_ref_df_log = log2_transform(exp_ref_df)  # using log2(TPM+1) for PCA
                                    pca_model.fit(exp_ref_df_log)
                                    dump(pca_model, pca_model_path)
                                    # save the gene list for PCA
                                    pd.DataFrame(gene_list_in_pca).to_csv(gene_list_in_pca_file_path)
                                assert np.all(exp_ref_df.columns == gene_list_in_pca)
                                exp_ref_df = log2_transform(exp_ref_df)
                                exp_ref_df = pca_model.transform(exp_ref_df)
                                exp_ref_df = pd.DataFrame(exp_ref_df, index=range(exp_ref_df.shape[0]),
                                                          columns=range(exp_ref_df.shape[1]))
                                if not os.path.exists(os.path.join(pca_model_dir, 'tcga_pca_ref.csv')):
                                    exp_ref_df.to_csv(os.path.join(pca_model_dir, 'tcga_pca_ref.csv'))
                                cumsum = np.cumsum(pca_model.explained_variance_ratio_)
                                d = len(cumsum)
                                print(f'   > {d} dimensions are needed to explain '
                                      f'{cumsum.max() * 100}% variance.')  # 1515
                                # exp_ref_df = exp_ref_df.iloc[:, :d].copy()
                            # align simulated GEP with the gene list in the PCA model
                            simulated_gep_obj = ReadExp(exp_file=simulated_gep, exp_type='TPM')
                            simulated_gep_obj.align_with_gene_list(gene_list=gene_list_in_pca, fill_not_exist=True)
                            simulated_gep = simulated_gep_obj.get_exp()  # TPM
                            assert np.all(simulated_gep.columns == gene_list_in_pca)
                            simulated_gep = log2_transform(simulated_gep)
                            simulated_gep = pca_model.transform(simulated_gep)
                            simulated_gep = pd.DataFrame(simulated_gep, index=simulated_gep_bak.index,
                                                         columns=range(simulated_gep.shape[1]))
                            # simulated_gep = simulated_gep.iloc[:, :d].copy()
                        assert np.all(exp_ref_df.columns == simulated_gep.columns)
                        if self.m_gep_ref is None:
                            if ('mean_gep' in filtering_method) or ('linear_mmd' in filtering_method):
                                # The maximum mean discrepancy (MMD) using
                                # linear kernel is equivalent to "square of L2 norm"
                                self.m_gep_ref = exp_ref_df.mean(axis=0).values.reshape(1, -1)
                            elif 'median_gep' in filtering_method:
                                self.m_gep_ref = exp_ref_df.median(axis=0).values.reshape(1, -1)  # TPM / PCs
                            else:
                                raise ValueError(f'filtering_method {filtering_method} is invalid')
                            distance_with_center_ref = np.linalg.norm(exp_ref_df - self.m_gep_ref,
                                                                      ord=norm_ord, axis=1)
                            if 'linear_mmd' in filtering_method:
                                distance_with_center_ref = distance_with_center_ref ** 2
                            self.q_dis_nn_ref_upper = np.quantile(distance_with_center_ref,
                                                                  self.filtering_quantile_upper)
                        dis_ref_simu_gep = np.linalg.norm(simulated_gep.values - self.m_gep_ref,
                                                          ord=norm_ord, axis=1)
                        if 'linear_mmd' in filtering_method:
                            dis_ref_simu_gep = dis_ref_simu_gep ** 2
                        if self.filtering_quantile_lower is not None:
                            self.q_dis_nn_ref_lower = np.quantile(distance_with_center_ref,
                                                                  self.filtering_quantile_lower)
                            if show_filtering_info:
                                print(f'   > Quantile distance of {self.filtering_quantile_lower * 100}% is: '
                                      f'{self.q_dis_nn_ref_lower}, {np.sum(dis_ref_simu_gep < self.q_dis_nn_ref_lower)} were removed')
                                print(f'   > Quantile distance of {self.filtering_quantile_upper * 100}% is: '
                                      f'{self.q_dis_nn_ref_upper}, {np.sum(dis_ref_simu_gep > self.q_dis_nn_ref_upper)} were removed')
                            simulated_gep = simulated_gep.loc[(dis_ref_simu_gep <= self.q_dis_nn_ref_upper) &
                                                              (dis_ref_simu_gep >= self.q_dis_nn_ref_lower), :]
                        else:
                            simulated_gep = simulated_gep.loc[dis_ref_simu_gep <= self.q_dis_nn_ref_upper, :].copy()

                    if simulated_gep is not None:
                        simulated_gep = simulated_gep_bak.loc[simulated_gep.index, :].copy()
                        if (self.generated_bulk_gep_counter + simulated_gep.shape[0]) > self.n_samples:
                            n_last_part = self.n_samples - self.generated_bulk_gep_counter
                            simulated_gep = simulated_gep.iloc[range(n_last_part)].copy()
                        simulated_gep = log2_transform(simulated_gep)
                        self.generated_bulk_gep_counter += simulated_gep.shape[0]
                        pbar.update(simulated_gep.shape[0])
                        generated_cell_frac = generated_cell_frac.loc[simulated_gep.index, :].copy()
                        selected_cell_ids = selected_cell_ids.loc[simulated_gep.index, :].copy()
                        self._save_simulated_bulk_gep(gep=simulated_gep, cell_id=selected_cell_ids,
                                                      cell_fraction=generated_cell_frac)
                    self.n_round += 1
            msg = f'   > Got {self.generated_bulk_gep_counter} samples from {self.n_round * min_n_cell_frac}'
            if sampling_method in ['segment', 'seg_random']:
                q_dis = 'radius' if filtering_method == 'marker_ratio' else 'l1 distance'
                if self.q_dis_nn_ref_lower is not None:
                    msg = f'   > Got {self.generated_bulk_gep_counter} samples from {self.n_round * min_n_cell_frac} ' \
                          f'within {q_dis} between {self.q_dis_nn_ref_lower} and {self.q_dis_nn_ref_upper} ' \
                          f'by quantile {self.filtering_quantile_lower * 100}%-{self.filtering_quantile_upper * 100}%'
                else:
                    msg = f'   > Got {self.generated_bulk_gep_counter} samples from {self.n_round * min_n_cell_frac} ' \
                          f'within {q_dis} {self.q_dis_nn_ref_upper} by quantile {self.filtering_quantile_upper * 100}%'
            print(msg)
            data_info = f'Simulated {self.generated_bulk_gep_counter} bulk cell gene expression profiles ' \
                        f'by {sampling_method}, log2(TPM + 1)'
            create_h5ad_dataset(simulated_bulk_exp_file_path=self.generated_bulk_gep_csv_fp,
                                cell_fraction_file_path=self.generated_cell_fraction_fp,
                                dataset_info=data_info,
                                result_file_path=self.generated_bulk_gep_fp)
            if log_file_path is not None:
                obj_info = self.__str__()
                print_msg(obj_info, log_file_path=log_file_path)
        else:
            print(f'   Previous result existed: {self.generated_bulk_gep_fp}')
            print(self.__str__())

    def _filter_gep_by_reference(self, simulated_gep, n_top: int = None) -> Union[pd.DataFrame, None]:
        """
        Filtering generated GEP by reference dataset, such as TCGA

        :param simulated_gep: simulated GEPs that need to filter by reference, TPM
        :param n_top: if too many neighbors were founded for one single sample, only keep n_top neighbors
        """

        # print(marker_ratio_ref.head(2))
        exp_obj_simu_gep = ExpObj(exp_file=simulated_gep, exp_type='TPM')
        exp_obj_simu_gep.cal_marker_gene_ratio(agg_methods={'CD4 T': 'max', 'B Cells': 'max'},
                                               cell_types=self.cell_type_used)
        marker_ratio_simu_gep = exp_obj_simu_gep.get_marker_ratios()

        # the distance of nearest neighbor for each sample
        # quantile = 0.999
        if self.q_dis_nn_ref_upper == 0:
            qn1 = QueryNeighbors(df_file=self.marker_ratio_ref)
            # quantile of distance
            self.q_dis_nn_ref_upper = qn1.get_quantile_of_nn_distance(quantile=self.filtering_quantile_upper)
            print(f'   > Quantile distance of {self.filtering_quantile_upper * 100}% is: {self.q_dis_nn_ref_upper}')
        # nn_dis = qn1.get_nn()

        qn2 = QueryNeighbors(df_file=marker_ratio_simu_gep)
        current_ref_nc = sorted(self.ref_neighbor_counter.items(), key=lambda x: x[1])
        _keep_ref = [i for i, j in current_ref_nc if j < self.n_neighbors_each_ref]
        ref_neighbors_within_radius = qn2.get_neighbors_by_radius(
            radius=self.q_dis_nn_ref_upper, n_top=n_top, share_neighbors=False,
            q_df_file=self.marker_ratio_ref.loc[_keep_ref, :].copy(),
            )
        ref2n_neighbors = ref_neighbors_within_radius.groupby(ref_neighbors_within_radius.index).count()
        _keep_neighbors = []
        for i, n_n in ref2n_neighbors.iterrows():
            self.ref_neighbor_counter[i] += n_n['nn']
        # print(f'   There are {ref_neighbors_within_radius.shape[0]} simulated GEPs left after filtering.')
        if ref_neighbors_within_radius.shape[0] > 0:
            return simulated_gep.loc[ref_neighbors_within_radius['nn'], :].copy()
        else:
            return None

    def read_merged_single_cell_dataset(self):
        if self.merged_sc_dataset is None:
            self.merged_sc_dataset = ReadH5AD(self.merged_sc_fp).get_h5ad()

    def read_generated_single_cell_dataset(self):
        if self.generated_sc_fp is None or (not os.path.exists(self.generated_sc_fp)):
            raise FileNotFoundError('   Please generate single cell dataset first '
                                    'using function "generate_single_cell_dataset", and try again.')
        generated_sc_dataset_obj = ReadH5AD(self.generated_sc_fp)
        generated_sc_dataset = generated_sc_dataset_obj.get_h5ad()
        self.generated_sc_dataset_obs = generated_sc_dataset.obs.copy()
        self.generated_sc_dataset_df = generated_sc_dataset_obj.get_df(convert_to_tpm=True)

    def get_info_in_merged_single_cell_dataset(self, check_zero_ratio: bool = True,
                                               zero_ratio_threshold: float = 0.95):
        if self.merged_sc_dataset is None:
            self.read_merged_single_cell_dataset()
        self.cell_type_in_sc = list(self.merged_sc_dataset.obs[self.cell_type_col_name].unique())
        # remove nan in the cell type list
        self.cell_type_in_sc = [i for i in self.cell_type_in_sc if type(i) == str]
        if self.subtype_col_name is not None:
            self.cell_subtype_in_sc = list(self.merged_sc_dataset.obs[self.subtype_col_name].unique())
        self.dataset_in_sc = list(self.merged_sc_dataset.obs['dataset_id'].unique())
        # self.merged_sc_dataset_obs = self.merged_sc_dataset.obs.copy()
        self.merged_sc_dataset_obs = self.merged_sc_dataset.obs.loc[
            self.merged_sc_dataset.obs['dataset_id'].isin(self.sc_dataset_used), :].copy()
        # always removing this part: marker genes of CD4 T cells expressed high in CD8 T cells
        if 'm_cd4/m_cd8 group' in self.merged_sc_dataset_obs.columns:
            self.merged_sc_dataset_obs = \
                self.merged_sc_dataset_obs.loc[~((self.merged_sc_dataset_obs['cell_type'] == 'CD8 T')
                                                 & (self.merged_sc_dataset_obs['m_cd4/m_cd8 group'] == 'high')),
                                               :].copy()

        self.merged_sc_dataset_obs['sample_id'] = \
            self.merged_sc_dataset_obs['sample_id'].cat.add_categories('pan_cancer')
        # add sample_id for pan_cancer_07 dataset to use groupby later
        self.merged_sc_dataset_obs.loc[self.merged_sc_dataset_obs['dataset_id'] == 'pan_cancer_07',
                                       'sample_id'] = 'pan_cancer'
        # The unique gene expression values of each gene in different cell types (merged scRNA-seq dataset)
        self.merged_sc_dataset = None
        if check_zero_ratio:
            if self.sc_dataset_gep_type == 'log_space':
                self.merged_sc_dataset_df = ReadH5AD(self.merged_sc_fp).get_df(convert_to_tpm=True)
            else:
                self.merged_sc_dataset_df = ReadH5AD(self.merged_sc_fp).get_df()
            zero_ratio = np.sum(self.merged_sc_dataset_df < 1, axis=1) / self.merged_sc_dataset_df.shape[1]
            high_zero_ratio_cells = zero_ratio[zero_ratio > zero_ratio_threshold].index.to_list()
            # remove high zero ratio cells
            print(f'   The zero ratio of {len(high_zero_ratio_cells)} cells '
                  f'> {zero_ratio_threshold} and will be removed.')
            self.merged_sc_dataset_obs = self.merged_sc_dataset_obs.loc[
                                         ~self.merged_sc_dataset_obs.index.isin(high_zero_ratio_cells), :].copy()

    def _sc_sampling(self, cell_frac: pd.DataFrame, obs_df: pd.DataFrame, n_threads: int = 10,
                     total_cell_number: int = None, sep_by_patient=False, sc_dataset=None,
                     minimum_n_base: int =1):
        """
        Mix single cell expression profiles to simulated bulk expression profile according to `cell_frac`.

        :param cell_frac: dataFrame, generated cell fraction for each cell type, samples by cell types

        :param n_threads: how many thread to use

        :param obs_df: a dataFrame of sample info for sampling

        :param total_cell_number: N, total cell number to sample for each simulated bulk sample

        :param sep_by_patient: whether to separate samples by patient or not during sampling

        :param sc_dataset: which single cell dataset to use for sampling

        :return: sampled cell_ids
        """
        if total_cell_number is not None:
            self.total_cell_number = total_cell_number
        # if self.obs_df is None:
        #     self.obs_df = obs_df
        cell_num = get_cell_num(cell_type_frac=cell_frac, total_num=self.total_cell_number)
        # print('   Start to select cells randomly based on cell types for generating each bulk expression profile...')
        # all cell types of each cell only need to select one SCT from self.obs_df
        # all_cell_num_is_one = np.all(cell_num == 1)
        cell_num_flatten = []
        # n_samples = cell_frac.shape[0]
        for cell_type in cell_num.columns:
            _part = pd.DataFrame(index=cell_num.index)
            _part['cell_type'] = cell_type
            _part['n_cell'] = cell_num[cell_type]
            _part['class_by'] = self.cell_type_col_name
            if cell_type in self.cell_subtype_used:
                _part['class_by'] = self.subtype_col_name
            # if all_cell_num_is_one:
            #     _selected_cell_ids = self.obs_df.loc[self.obs_df['cell_type'] == cell_type,
            #                                          :].sample(n=n_samples).index.to_list()
            #     _part['selected_cell_id'] = _selected_cell_ids
            cell_num_flatten.append(_part)
        sampled_cell_ids = pd.concat(cell_num_flatten)
        # contains all cell types for each single simulated bulk expression profile
        # if not all_cell_num_is_one:
        paras = [(obs_df, 1, row['cell_type'], row['n_cell'], row['class_by'],
                  sep_by_patient, sc_dataset, minimum_n_base)
                 for i, row in sampled_cell_ids.iterrows()]
        n_threads = min(multiprocessing.cpu_count()-2, n_threads)
        # https://pythonspeed.com/articles/python-multiprocessing/
        with multiprocessing.get_context('spawn').Pool(n_threads) as p:
            results = p.starmap(get_sample_id, paras)
        # print(results)
        results_str = [';'.join(i) for i in results]
        sampled_cell_ids['selected_cell_id'] = results_str
        sampled_cell_ids.index.name = 'sample_id'
        sampled_cell_ids.sort_values(by=['sample_id', 'cell_type'], inplace=True)

        return sampled_cell_ids

    @staticmethod
    def _sample_noise(miu=0, s=566.1, f=0.25, n_samples=10000) -> np.ndarray:
        """
        Generate noise for each gene in one bulk GEP, modified from Hao, Yuning, et al. PLoS Computational Biology, 2019

        :param miu: mean of normal distribution
        :param s: the mean std of all samples in TCGA with TPM values, LGG and GBM were excluded
        """
        sigma = f * np.log2(s)
        norm_dis = stats.norm(miu, sigma)
        x = norm_dis.rvs(size=n_samples)
        return np.power(2, x)

    def _map_cell_id2exp(self, selected_cell_id, sc_dataset: str = 'merged_sc_dataset',
                         simu_method: str = 'ave', cell_frac: pd.DataFrame = None,
                         gep_type='MCT', add_noise: bool = False, noise_params: tuple = ()) -> pd.DataFrame:
        """
        mapping sampled cell_ids to the corresponding GEPs
        :param selected_cell_id: a dataFrame which contains cell_type, n_cell, selected_cell_id
        :param sc_dataset: merged_sc_dataset, generated_sc_dataset or sct_dataset
        :param simu_method: ave (average all selected single cell GEPs), mul (multiple GEP by cell fractions)
        :param gep_type: MCT means multiple cell types (bulk GEP), SCT means single cell type
        :return: a DataFrame, TPM, samples by genes
        """
        if sc_dataset == 'sct_dataset':
            sc_ds_df = self.sct_dataset_df
        elif sc_dataset == 'merged_sc_dataset':
            sc_ds_df = self.merged_sc_dataset_df
        else:  # generated single cell dataset
            sc_ds_df = self.generated_sc_dataset_df
        simulated_exp = {}
        if gep_type == 'SCT':  # each sample id only contains a single cell type
            selected_cell_id = selected_cell_id.loc[selected_cell_id['n_cell'] > 1, :].copy()
            # n_non_zero = 1000
            n_genes = sc_ds_df.shape[1]
            for cell_type, group in selected_cell_id.groupby('cell_type'):
                for sample_id, row in group.iterrows():
                    cell_ids = row['selected_cell_id'].split(';')
                    # simulated_exp[sample_id] = sc_ds_df.loc[cell_ids, :].mean(axis=0)  # average
                    current_gene_exp = sc_ds_df.loc[cell_ids, :].mean(axis=0)  # average
                    if simu_method == 'random_replacement':
                        # long_tail_noise_non_zero = np.random.random(n_non_zero) * 2
                        # n_zero = np.random.randint(n_non_zero/10, n_non_zero)
                        # long_tail_noise = np.append(long_tail_noise_non_zero, np.zeros(n_zero))
                        long_tail_noise = np.random.random(size=n_genes)
                        # replace the values < 1 with random selected values
                        mask = (current_gene_exp < 1).values.astype(int)
                        current_gene_exp = current_gene_exp + long_tail_noise * mask
                    simulated_exp[sample_id] = pd.Series(current_gene_exp.values, index=current_gene_exp.index)
        else:
            assert simu_method == 'mul', 'Only support matrix multiplication for generating MCT'
            subtype2cell_type = {i: k for k, v in self.cell_type2subtype.items() for i in v}
            ct2rna_coefficient = {}
            if self.total_rna_coefficient is not None:
                # consider the total RNA amount of each cell type
                all_cell_types = selected_cell_id['cell_type'].unique()
                for cell_type in all_cell_types:
                    if cell_type in self.total_rna_coefficient:
                        ct2rna_coefficient[cell_type] = self.total_rna_coefficient[cell_type]
                    elif (cell_type in subtype2cell_type) and \
                            (subtype2cell_type[cell_type] in self.total_rna_coefficient):
                        ct2rna_coefficient[cell_type] = self.total_rna_coefficient[subtype2cell_type[cell_type]]
                    else:
                        ct2rna_coefficient[cell_type] = 1.0
                print('   > The following total RNA coefficient will be used: ', ct2rna_coefficient)
            for sample_id, group in selected_cell_id.groupby(by=selected_cell_id.index):
                all_n_cell_is_one = np.all(group['n_cell'] == 1)
                assert all_n_cell_is_one, 'n_cell should be 1 for all cell types'
                cell_ids = group['selected_cell_id'].to_list()
                current_merged = sc_ds_df.loc[cell_ids, :].copy()
                # using merged single cell dataset directly

                # sort by cell types to make sure the correction of matrix multiplication
                _cell_types = group['cell_type'].to_list()
                current_cell_frac = cell_frac.loc[sample_id, _cell_types].copy().to_frame()
                if self.total_rna_coefficient is not None:
                    # consider the total RNA amount of each cell type
                    rna_coefficient = [ct2rna_coefficient[ct] for ct in _cell_types]
                    # multiply by total RNA coefficient for each cell type
                    current_merged = current_merged * np.array(rna_coefficient).reshape(-1, 1)
                    matrix_mul = current_merged.values.T @ current_cell_frac.values  # mix different cell types
                    matrix_mul = matrix_mul / np.sum(matrix_mul, axis=0) * 1e6  # convert to TPM
                else:
                    matrix_mul = current_merged.values.T @ current_cell_frac.values  # mix different cell types directly

                simulated_exp[sample_id] = pd.Series(matrix_mul.reshape(-1), index=current_merged.columns)
                if add_noise:
                    assert len(noise_params) == 2, 'noise_params should be a tuple of (f, total_max)'
                    noise = self._sample_noise(n_samples=len(simulated_exp[sample_id]), f=noise_params[0])
                    if noise.sum() > noise_params[1]:
                        noise = noise / np.sum(noise) * noise_params[1]
                    simulated_exp[sample_id] = simulated_exp[sample_id] + noise

        simulated_exp_df = pd.DataFrame.from_dict(data=simulated_exp, orient='index')
        simulated_exp_df = non_log2cpm(simulated_exp_df, sum_exp=1e6)  # convert to TPM
        return simulated_exp_df.round(3)

    def _save_simulated_bulk_gep(self, gep: pd.DataFrame, cell_id: pd.DataFrame, cell_fraction: pd.DataFrame = None):

        if not os.path.exists(self.generated_bulk_gep_csv_fp):
            gep.to_csv(self.generated_bulk_gep_csv_fp, float_format='%g')
        else:
            gep.to_csv(self.generated_bulk_gep_csv_fp, header=False, mode='a', float_format='%g')

        if not os.path.exists(self.sampled_sc_cell_id_file_path):
            cell_id.to_csv(self.sampled_sc_cell_id_file_path)
        else:
            cell_id.to_csv(self.sampled_sc_cell_id_file_path, header=False, mode='a')

        if cell_fraction is not None:
            if not os.path.exists(self.generated_cell_fraction_fp):
                cell_fraction.to_csv(self.generated_cell_fraction_fp, float_format='%g')
            else:
                cell_fraction.to_csv(self.generated_cell_fraction_fp, header=False, mode='a', float_format='%g')

        if self.ref_neighbor_counter:
            pd.DataFrame.from_dict(self.ref_neighbor_counter, orient='index').to_csv(self.ref_neighbor_counter_fp)

    def _check_intermediate_generated_gep(self):
        """
        if the generation process broke accidentally,
        generated intermediate result can be used to recovery generation process
        """
        if os.path.exists(self.generated_cell_fraction_fp):
            gen_cell_frac = pd.read_csv(self.generated_cell_fraction_fp, index_col=0)
            gen_cell_frac = gen_cell_frac[~gen_cell_frac.index.duplicated(keep='first')].copy()
            if os.path.exists(self.generated_bulk_gep_csv_fp):
                gen_bulk_gep = pd.read_csv(self.generated_bulk_gep_csv_fp, index_col=0, usecols=[0, 1])
                gen_bulk_gep = gen_bulk_gep[~gen_bulk_gep.index.duplicated(keep='first')].copy()
                common_inx = [i for i in gen_cell_frac.index if i in gen_bulk_gep.index]
                if common_inx and (len(common_inx) == gen_bulk_gep.shape[0]) and (len(common_inx) == gen_cell_frac.shape[0]):
                    self.generated_bulk_gep_counter = len(common_inx)
                    if 'seg_random' in gen_cell_frac.iloc[-1].name:
                        n_round = int(gen_cell_frac.iloc[-1].name.split('_')[3])
                    else:
                        n_round = int(gen_cell_frac.iloc[-1].name.split('_')[2])
                    self.n_round = n_round + 1
                    print(f'   The following intermediate generated result will be reused: \n'
                          f'{self.generated_cell_fraction_fp}, {self.generated_bulk_gep_csv_fp}.\n'
                          f'self.n_round will be reset to {self.n_round}, '
                          f'self.generated_bulk_gep_counter will be reset to {self.generated_bulk_gep_counter}')
                    if os.path.exists(self.ref_neighbor_counter_fp):
                        ref2n_neighbors = pd.read_csv(self.ref_neighbor_counter_fp, index_col=0).to_dict()['0']
                        self.ref_neighbor_counter = ref2n_neighbors.copy()
                else:
                    os.remove(self.generated_cell_fraction_fp)
                    os.remove(self.generated_bulk_gep_csv_fp)
                    os.remove(self.sampled_sc_cell_id_file_path)

    def _check_basic_info(self):
        """
        check cell types and dataset
        """
        self.get_info_in_merged_single_cell_dataset(zero_ratio_threshold=self.zero_ratio_threshold)
        cell_type_not_in_sc = [i for i in self.cell_type_used if i not in self.cell_type_in_sc]
        sub_cell_type_not_in_sc = []
        if self.cell_subtype_used:
            sub_cell_type_not_in_sc = [i for i in self.cell_subtype_used if i not in self.cell_subtype_in_sc]
        dataset_not_in_sc = [i for i in self.sc_dataset_used if i not in self.dataset_in_sc]
        if len(cell_type_not_in_sc) > 0:
            invalid_cell_types = ', '.join(cell_type_not_in_sc)
            valid_cell_types = ', '.join(self.cell_type_in_sc)
            raise ValueError(f'   Invalid cell types: {invalid_cell_types}, '
                             f'only the following cell types existed in single cell dataset: {valid_cell_types}')
        else:
            used_cell_types = ', '.join(self.cell_type_used)
            print(f'   The following cell types will be used: {used_cell_types}')

        if len(sub_cell_type_not_in_sc) > 0:
            invalid_sub_cell_types = ', '.join(sub_cell_type_not_in_sc)
            valid_sub_cell_types = ', '.join(self.cell_subtype_in_sc)
            raise ValueError(f'   Invalid sub cell types: {invalid_sub_cell_types}, '
                             f'only the following sub cell types existed in single cell dataset: {valid_sub_cell_types}')
        elif self.cell_subtype_used:
            used_sub_cell_types = ', '.join(self.cell_subtype_used)
            print(f'   The following sub cell types will be used: {used_sub_cell_types}')

        if len(dataset_not_in_sc) > 0:
            invalid_datasets = ', '.join(dataset_not_in_sc)
            valid_datasets = ', '.join(self.dataset_in_sc)
            raise ValueError(f'   Invalid datasets: {invalid_datasets}, '
                             f'only the following datasets existed in single cell dataset: {valid_datasets}')
        else:
            used_datasets = ', '.join(self.sc_dataset_used)
            print(f'   The following datasets will be used: {used_datasets}')

    def _read_sct_dataset(self, latent_z_nn_info_file=None):
        """
        positive samples of SCT (single cell type), generated by BulkGEPGeneratorSCT
        :param latent_z_nn_info_file: neighbor information of latent z for all samples, used for QC
        """
        sct_dataset_obs, sct_dataset_df = \
            read_single_cell_type_dataset(sct_dataset_file_path=self.sct_dataset_file_path,
                                          latent_z_nn_info_file=latent_z_nn_info_file)
        return sct_dataset_obs, sct_dataset_df

    def __str__(self):
        _dict = {k: v for k, v in self.__dict__.items() if type(v) in [list, str, int, float]}
        return str(self.__class__) + ':\n' + json.dumps(_dict, indent=2)


class SingleCellTypeGEPGenerator(BulkGEPGenerator):
    """
    Generating single cell type GEPs (sctGEPs)

    :param simu_bulk_dir: the directory to save simulated bulk cell GEPs
    :param merged_sc_dataset_file_path: the file path of pre-merged single cell datasets
    :param cell_type2subtype: cell types used when generating bulk GEPs,
        {cell_type: [sub_cell_type1, sub_cell_type2, ...], ...}
    :param sc_dataset_ids: single cell dataset id used when generating bulk GEPs
    :param bulk_dataset_name: the name of generated bulk dataset, only for naming
    :param zero_ratio_threshold: the threshold of zero ratio of genes in single cell GEPs,
        remove the GEP if zero ratio > threshold
    :param sc_dataset_gep_type: the type of single cell GEPs, `log_space` or `linear_space`
    """
    def __init__(self, merged_sc_dataset_file_path, cell_type2subtype, sc_dataset_ids,
                 simu_bulk_dir, bulk_dataset_name, zero_ratio_threshold: float = 0.97,
                 sc_dataset_gep_type: str = 'log_space', subtype_col_name: str = None,
                 cell_type_col_name: str = 'cell_type'):
        super().__init__(merged_sc_dataset_file_path=merged_sc_dataset_file_path, simu_bulk_dir=simu_bulk_dir,
                         cell_type2subtype=cell_type2subtype, sc_dataset_ids=sc_dataset_ids,
                         bulk_dataset_name=bulk_dataset_name,
                         zero_ratio_threshold=zero_ratio_threshold, sct_dataset_file_path=None,
                         sc_dataset_gep_type=sc_dataset_gep_type, subtype_col_name=subtype_col_name,
                         cell_type_col_name=cell_type_col_name)

    def generate_samples(self, n_sample_each_cell_type: int = 10000,
                         n_base_for_positive_samples: int = 100,
                         sample_type: str = 'positive', sep_by_patient=False,
                         simu_method='ave', test_set: bool = False,
                         minimum_n_base: int = 1, ref_gene_list_file_path: str = None):
        """
        :param n_sample_each_cell_type: the number of samples to generate for each cell type

        :param n_base_for_positive_samples: the number of single cells to average

        :param sample_type: positive means only 1 cell type is used, negative means more than 1 cell types are used

        :param sep_by_patient: only sampling from one patient in the original dataset if True

        :param simu_method: `ave`: averaging all GEPs, or `scale_by_mGEP`: scaling by the mean GEP of all samples in the TCGA dataset
            or `random_replacement`: replacing the gene expression value (<1) by another value within the same cell type selected randomly

        :param test_set: if True, generate a test set using the same cell types as the training set,
            otherwise generate data set with all cell types

        :param minimum_n_base: the minimum number of single cells to average for each bulk sample

        :param ref_gene_list_file_path: the file path of a reference gene list,
            the list of genes used for filtering the single cell dataset for generating SCT dataset,
            especially for using a customized scRNA-seq dataset

        """
        if not os.path.exists(self.generated_bulk_gep_fp):
            if test_set:
                self.cell_type_used = [k for k, v in self.cell_type2subtype.items() if k == v[0]]
            self.n_samples = n_sample_each_cell_type * (len(self.cell_type_used) + len(self.cell_subtype_used))
            if not os.path.exists(self.generated_cell_fraction_fp):
                print(f'   Generate cell proportions for single cell type (SCT) samples in {self.bulk_dataset_name}')
                generated_cell_frac = self.generate_frac_sc(
                    sample_type=sample_type, sample_prefix=f'{self.bulk_dataset_name}_{sample_type[:3]}'
                )
                generated_cell_frac.to_csv(self.generated_cell_fraction_fp, float_format='%g')
            else:
                print(f'   Previous result exists: {self.generated_cell_fraction_fp}')
            # DC has 543 cells, larger chunk_size can cause error if set 'replace=False' and 'n_base=1' while sampling
            chunk_size = int(n_sample_each_cell_type / 10)
            chunk_counter = 0
            # read existed GEPs
            existed_gep_ids = pd.Index([])
            if os.path.exists(self.generated_bulk_gep_csv_fp):
                existed_gep_ids = pd.read_csv(self.generated_bulk_gep_csv_fp, index_col=0, usecols=[0]).index
            if ref_gene_list_file_path is not None:
                ref_gene_list = pd.read_csv(ref_gene_list_file_path, index_col=0).index.to_list()
                sc_df_obj = ReadExp(self.merged_sc_dataset_df, exp_type='TPM')
                sc_df_obj.align_with_gene_list(ref_gene_list)
                # self.merged_sc_dataset_df = self.merged_sc_dataset_df.loc[:, ref_gene_list]
                self.merged_sc_dataset_df = sc_df_obj.get_exp()  # TPM
                print(f'   The number of genes in the reference gene list: {self.merged_sc_dataset_df.shape[1]} ',
                      f'will be used. The file path is: {ref_gene_list_file_path}')
            with pd.read_csv(self.generated_cell_fraction_fp, chunksize=chunk_size, index_col=0) as reader:
                simu_method = simu_method  # average single cell GEPs for both positive and negative sampling
                for rows in tqdm(reader):
                    rows = rows.loc[~rows.index.isin(existed_gep_ids), :].copy()  # filter existed samples
                    if rows.shape[0] > 0:
                        if sample_type == 'positive' and n_base_for_positive_samples > 1:
                            total_cell_number = n_base_for_positive_samples
                        else:  # negative sampling (multiple cell types are used) or positive sampling with n_base=1
                            total_cell_number = 0  # assign 1 for the cell types with non-zero cell fractions
                        if sample_type == 'positive':
                            # change subgroup for each chunk based on cell_type2subgroup_id
                            all_cell_types = rows.columns[np.argmax(rows.values, axis=1)].unique()
                            if len(all_cell_types) > 1:
                                raise ValueError(f'   More than one cell types are selected: {all_cell_types}')
                            cell_type = all_cell_types[0]
                            col_name = self.cell_type_col_name
                            if cell_type in self.cell_subtype_used:
                                col_name = self.subtype_col_name
                            assert col_name in self.merged_sc_dataset_obs.columns, \
                                f"Wrong column name for cell type or cell subtype '{col_name}', please check." \
                                f"Available columns are: {self.merged_sc_dataset_obs.columns}"
                            current_obs_df = self.merged_sc_dataset_obs.loc[
                                self.merged_sc_dataset_obs[col_name] == cell_type, :].copy()
                            if current_obs_df.shape[0] < minimum_n_base:
                                raise ValueError(f'   The number of single cells for cell type {cell_type} '
                                                 f'is less than {minimum_n_base}, please check or remove '
                                                 f'cell type {cell_type} to continue.')
                            elif current_obs_df.shape[0] <= n_base_for_positive_samples:
                                print(f'Warning: the number of cells in {cell_type} is '
                                      f'less than n_base {n_base_for_positive_samples},',
                                      f'all samples (n={current_obs_df.shape[0]}) will be used.')
                        else:
                            current_obs_df = self.merged_sc_dataset_obs.copy()
                        selected_cell_ids = self._sc_sampling(cell_frac=rows, total_cell_number=total_cell_number,
                                                              obs_df=current_obs_df,
                                                              sep_by_patient=sep_by_patient,
                                                              minimum_n_base=minimum_n_base)
                        simulated_gep = self._map_cell_id2exp(selected_cell_id=selected_cell_ids, simu_method=simu_method,
                                                              cell_frac=rows, sc_dataset='merged_sc_dataset', gep_type='SCT')
                        simulated_gep = log2_transform(simulated_gep)
                        self.generated_bulk_gep_counter += simulated_gep.shape[0]
                        # generated_cell_frac = generated_cell_frac.loc[simulated_gep.index, :].copy()
                        selected_cell_ids = selected_cell_ids.loc[simulated_gep.index, :].copy()
                        self._save_simulated_bulk_gep(gep=simulated_gep, cell_id=selected_cell_ids)
                    else:
                        print(f'   Previous result exists: {self.generated_bulk_gep_fp}, '
                              f'will skip this chunk {chunk_counter}')
                    chunk_counter += 1

            data_info = f'Simulated {self.generated_bulk_gep_counter} gene expression profiles ' \
                        f'for each cell type, log2(TPM + 1)'
            create_h5ad_dataset(simulated_bulk_exp_file_path=self.generated_bulk_gep_csv_fp,
                                cell_fraction_file_path=self.generated_cell_fraction_fp,
                                dataset_info=data_info,
                                result_file_path=self.generated_bulk_gep_fp)
        else:
            print(f'   Previous result exists: {self.generated_bulk_gep_fp}')

    def generate_frac_sc(self, sample_prefix: str = None, sample_type: str = 'positive') -> pd.DataFrame:
        """
        Generate cell fractions for single cell samples, positive samples only contain one specific cell type,
        negative samples contain >= 2 cell types with equal proportion

        :param sample_prefix: prefix of sample names

        :param sample_type: positive samples (one specific cell type) or negative samples (>= 2 cell types)

        :return:  generated cell fraction, sample by cell type
        """
        if sample_prefix is None:
            sample_prefix = f's_sc_{sample_type}'
        n_cell_types = len(self.cell_type_used) + len(self.cell_subtype_used)
        cols = self.cell_type_used + self.cell_subtype_used
        n_for_each_cell_type = int(self.n_samples / n_cell_types)
        generated_frac_df = \
            pd.DataFrame(index=[f'{sample_prefix}_{i}_{j}' for i in cols for j in range(n_for_each_cell_type)],
                         columns=cols, data=np.zeros((self.n_samples, n_cell_types)))
        if sample_type == 'positive':
            for i, cell_type in enumerate(cols):
                inx_start = i * n_for_each_cell_type
                inx_end = min((i+1) * n_for_each_cell_type, self.n_samples)
                generated_frac_df.iloc[inx_start:inx_end, i] = 1
        else:  # negative samples
            n_for_each_n_ct = int(self.n_samples / (n_cell_types - 1))
            for n_ct in range(2, n_cell_types+1):  # the number of cell types used (=2)
                inx_start = (n_ct-2) * n_for_each_n_ct
                inx_end = min((n_ct-1) * n_for_each_n_ct, self.n_samples)
                frac = 1 / n_ct
                for inx in range(inx_start, inx_end):
                    if n_ct < n_cell_types:
                        current_cell_type_inx = np.random.choice(range(n_cell_types), size=n_ct, replace=False)
                    else:
                        current_cell_type_inx = np.array(range(n_cell_types))
                    generated_frac_df.iloc[inx, current_cell_type_inx] = frac
        return generated_frac_df.round(4)


# class BulkGEPGeneratorSCT(BulkGEPGenerator):
#     """
#     generate bulk GEPs by cell proportion x single GEP of single cell type (SCT)
#     """
#     def __init__(self, sct_dataset_file_path, cell_type2subtype, simu_bulk_dir, bulk_dataset_name):
#         super().__init__(simu_bulk_dir=simu_bulk_dir, cell_type2subtype=cell_type2subtype, bulk_dataset_name=bulk_dataset_name,
#                          merged_sc_dataset_file_path=None, check_basic_info=False, sc_dataset_ids=[],
#                          sct_dataset_file_path=sct_dataset_file_path)
#         # self.sct_dataset_file_path = sct_dataset_file_path
#         # self.sct_dataset_obs = None
#         # self.sct_dataset_df = None
#
#     def generate_samples(self, n_samples, latent_z_nn_file_path: str = None,
#                          ref_distribution: dict = None, sampling_method: str = 'fragment',
#                          total_cell_number: int = 1, n_threads: int = 10,
#                          log_file_path: str = None, cell_prop_file_path: str = None, add_token_cell_type: bool = False,
#                          random_n_cell_type: list = None):
#         """
#         :param n_samples: the number of GEPs to generate
#         :param latent_z_nn_file_path: neighbor information of latent z for all samples, used for QC and sampling
#         :param total_cell_number: N, the total number of cells sampled from merged single cell dataset
#                                       and averaged to simulate a single bulk RNA-seq sample
#         :param sampling_method: segment or random, method to generate cell fractions
#         :param ref_distribution: reference distribution for each cell type, seperated to 10 bins for [0, 1]
#             {'B Cells': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], '': [], '': [], ...}
#             only used for fragment sampling method when call function self._generate_cell_fraction
#         :param n_threads: number of threads used for parallel computing
#         :param log_file_path:
#         :param cell_prop_file_path: file path of pre-generated cell proportion, use this file if provided
#         :param add_token_cell_type: add all cell types in generated_cell_frac.csv file,
#              even if some cell types take 0% in all samples, only for keeping same format for all dataset
#         :param random_n_cell_type: a list of the number of cell types (selected randomly) used for simulating bulk GEPs
#         """
#         n_total_cpus = multiprocessing.cpu_count()
#         n_threads = min(n_total_cpus - 1, n_threads)
#         self.n_samples = n_samples
#         self.total_cell_number = total_cell_number  # set 1 for all cell types
#         # simulating bulk cell GEP by mixing GEPs of different cell types
#         min_n_cell_frac = np.min([100, n_samples])  # smaller number of samples without filtering
#         if cell_prop_file_path is not None:
#             self.generated_cell_fraction_fp = cell_prop_file_path
#         if not os.path.exists(self.generated_bulk_gep_fp):
#             self.sct_dataset_obs, self.sct_dataset_df = self._read_sct_dataset(latent_z_nn_info_file=
#                                                                               latent_z_nn_file_path)
#
#             if not os.path.exists(self.generated_cell_fraction_fp):
#                 print(f'   Generate cell proportions for bulk samples in {self.bulk_dataset_name}')
#                 generated_cell_frac = self._generate_cell_fraction(
#                     sampling_method=sampling_method, n_cell_frac=self.n_samples,
#                     sample_prefix=f's_{self.bulk_dataset_name}_{sampling_method}_0',
#                     ref_distribution=ref_distribution,
#                     random_n_cell_type=random_n_cell_type,
#                 )
#                 if add_token_cell_type:
#                     for ct in sorted_cell_types:
#                         if ct not in generated_cell_frac.columns:
#                             generated_cell_frac[ct] = 0
#                     generated_cell_frac = generated_cell_frac.loc[:, sorted_cell_types]
#                 generated_cell_frac.to_csv(self.generated_cell_fraction_fp, float_format='%g')
#             else:
#                 print(f'   Previous result exists: {self.generated_cell_fraction_fp}')
#
#             chunk_size = int(self.n_samples / min_n_cell_frac)
#             chunk_size = max(chunk_size, 100)
#             chunk_counter = 0
#             with pd.read_csv(self.generated_cell_fraction_fp, chunksize=chunk_size, index_col=0) as reader:
#                 simu_method = 'mul'  # matrix multiplication
#                 for rows in tqdm(reader):
#
#                     selected_cell_ids = self._sc_sampling(cell_frac=rows,
#                                                           n_threads=n_threads, obs_df=self.sct_dataset_obs)
#                     simulated_gep = self._map_cell_id2exp(selected_cell_id=selected_cell_ids, cell_frac=rows,
#                                                           sc_dataset='sct_dataset', simu_method=simu_method)
#
#                     simulated_gep = log2_transform(simulated_gep)
#                     self.generated_bulk_gep_counter += simulated_gep.shape[0]
#                     # generated_cell_frac = generated_cell_frac.loc[simulated_gep.index, :].copy()
#                     selected_cell_ids = selected_cell_ids.loc[simulated_gep.index, :].copy()
#                     self._save_simulated_bulk_gep(gep=simulated_gep, cell_id=selected_cell_ids)
#                     chunk_counter += 1
#
#             data_info = f'Simulated {self.generated_bulk_gep_counter} bulk cell gene expression profiles ' \
#                         f'by sampling method: {sampling_method}, simulation method: {simu_method}, log2(TPM + 1)'
#             create_h5ad_dataset(simulated_bulk_exp_file_path=self.generated_bulk_gep_csv_fp,
#                                 cell_fraction_file_path=self.generated_cell_fraction_fp,
#                                 dataset_info=data_info,
#                                 result_file_path=self.generated_bulk_gep_fp)
#             if log_file_path is not None:
#                 obj_info = self.__str__()
#                 print_msg(obj_info, log_file_path=log_file_path)
#         else:
#             print(f'   Previous result existed: {self.generated_bulk_gep_fp}')
#             print(self.__str__())


# for gene-level filtering
def get_quantile(exp_df, quantile_range, col_name: list = None) -> pd.DataFrame:
    """
    Get quantile
    :param exp_df:
    :param quantile_range:
    :param col_name:
    :return:
    """
    quantile_df = pd.DataFrame(index=exp_df.columns, columns=col_name)
    quantile_df[col_name[0]] = exp_df.quantile(quantile_range[0], axis=0)
    quantile_df[col_name[1]] = exp_df.quantile(quantile_range[1], axis=0)
    quantile_df[col_name[2]] = exp_df.quantile(quantile_range[2], axis=0)
    return quantile_df


def get_gene_list_for_filtering(bulk_exp_file, tcga_file, result_file_path, q_col_name: list = None,
                                filtering_type: str = 'quantile_range',
                                corr_threshold: float = 0.3, n_gene_max: int = 1000,
                                corr_result_fp: str = None, quantile_range: list = None):
    """
    Gene-level filtering based on the filtering type
    :param bulk_exp_file: simulated bulk expression file, log2(TPM + 1)
    :param tcga_file:
    :param filtering_type: high_corr_gene, quantile_range, all_genes, high_corr_gene_and_quantile_range
      - high_corr_gene: expression values with high correlation with the cell proportions of any cell types
      - quantile_range: the median of expression values within the [q_5, q_95] quantile range
    :param corr_result_fp:
    :param quantile_range: median gene expression (quantile_range[1], expected as 0.5) of simulated bulk cell GEPs that
        is less than quantile_range[0] or greater than quantile_range[2] of the quantile expression value
        of the corresponding gene in TCGA dataset will be removed
    :param result_file_path:
    :param q_col_name:
    :param corr_threshold: correlation threshold for gene filtering
    :param n_gene_max: maximum number of genes for each cell type during gene filtering
    :return:
    """

    assert filtering_type in ['high_corr_gene', 'quantile_range', 'all_genes', 'high_corr_gene_and_quantile_range'], \
        f'filtering_type: {filtering_type} is not supported, only support high_corr_gene, quantile_range, all_genes, ' \
        f'high_corr_gene_and_quantile_range'
    if not os.path.exists(result_file_path):
        bulk_exp = ReadH5AD(bulk_exp_file).get_df(convert_to_tpm=True)
        tcga_obj = ReadExp(tcga_file, exp_type='TPM')
        tcga_obj.align_with_gene_list(gene_list=bulk_exp.columns.to_list())  # common genes will be kept
        tcga_obj.to_log2cpm1p()
        tcga = tcga_obj.get_exp()
        bulk_exp = bulk_exp.loc[:, tcga.columns].copy()
        bulk_exp = non_log2cpm(bulk_exp)
        bulk_exp = log2_transform(bulk_exp)
        assert (len(tcga.columns) == len(bulk_exp.columns)) and np.all(tcga.columns == bulk_exp.columns), \
            f'Columns of bulk_exp and tcga are not the same'
        gene_list = []
        if 'high_corr_gene' in filtering_type:
            h5_obj = ReadH5AD(bulk_exp_file)
            gene_exp = h5_obj.get_df(convert_to_tpm=True)
            cell_frac = h5_obj.get_cell_fraction()
            corr_df = cal_corr_gene_exp_with_cell_frac(gene_exp=gene_exp, cell_frac=cell_frac,
                                                       result_file_path=corr_result_fp, filtered_by_corr=corr_threshold,
                                                       filter_by_num=n_gene_max)
            del gene_exp
            gene_list = corr_df.index.to_list()
            print(f'{len(gene_list)} genes are selected by high correlation')
        if 'quantile_range' in filtering_type:
            # both bulk_exp and tcga are in log-space
            gene_list_qr = get_gene_list_filtered_by_quantile_range(bulk_exp=bulk_exp, tcga_exp=tcga,
                                                                    quantile_range=quantile_range,
                                                                    q_col_name=q_col_name)
            print(f'{len(gene_list_qr)} genes are selected by quantile range')
            if len(gene_list) > 0:  # if high-correlation genes exist, then filter by high-correlation genes
                gene_list = [gene for gene in gene_list if gene in gene_list_qr]
                print(f'{len(gene_list)} genes are selected by both high correlation and quantile range')
            else:
                gene_list = gene_list_qr
        if filtering_type == 'all_genes':
            gene_list = bulk_exp.columns.to_list()
            print(f'All {len(gene_list)} genes will be used')

        gene_list_df = pd.DataFrame(columns=['gene_name'])
        gene_list_df['gene_name'] = gene_list
        gene_list_df.to_csv(result_file_path, index=False)
    else:
        print(f'Loading the gene list from file: {result_file_path}')
        gene_list_df = pd.read_csv(result_file_path)
        gene_list = gene_list_df['gene_name'].to_list()
    return gene_list


def get_gene_list_filtered_by_quantile_range(bulk_exp, tcga_exp, quantile_range: list = None,
                                             q_col_name: list = None, tcga_gene_info=None):
    """
    Get gene list filtered by quantile range
    :param bulk_exp: bulk expression, TPM
    :param tcga_exp: TCGA expression data, TPM
    :param quantile_range: lower boundary, median, upper boundary, such as [0.025, 0.5, 0.975]
    :param q_col_name: column names for quantile range
    :param tcga_gene_info: gene quantile values in TCGA
    """
    if quantile_range is None:
        quantile_range = [0.025, 0.5, 0.975]
        q_col_name = ['q_' + str(int(q * 1000) / 10) for q in quantile_range]
    if tcga_gene_info is None:
        tcga_gene_info = get_quantile(tcga_exp, quantile_range=quantile_range, col_name=q_col_name)
    if type(bulk_exp) is pd.DataFrame and bulk_exp.shape[0] > 1:
        bulk_gene_info = get_quantile(bulk_exp, quantile_range=quantile_range, col_name=q_col_name)
        gene_inx = (bulk_gene_info[q_col_name[1]] >= tcga_gene_info[q_col_name[0]]) & \
                   (bulk_gene_info[q_col_name[1]] <= tcga_gene_info[q_col_name[2]])
        gene_list_qr = bulk_exp.loc[:, gene_inx].columns.to_list()
    else:  # if bulk_exp is a series
        gene_inx = (bulk_exp >= tcga_gene_info[q_col_name[0]]) & (bulk_exp <= tcga_gene_info[q_col_name[2]])
        gene_list_qr = bulk_exp.loc[gene_inx].index.to_list()
    return gene_list_qr


def cal_loading_by_pca(pca, gene_list, loading_matrix_file_path=None):
    """
    Cal loading by PCA
    :param pca:
    :param gene_list:
    :param loading_matrix_file_path:
    :return:
    """
    loading = pca.components_.T * np.sqrt(pca.explained_variance_)
    # loading = loading.loc[:, gene_list]
    com_matrix = pd.DataFrame(data=loading[:, 0:3], index=gene_list, columns=['PC1', 'PC2', 'PC3'])
    com_matrix['PC1_abs'] = com_matrix['PC1'].abs()
    com_matrix['PC2_abs'] = com_matrix['PC2'].abs()
    com_matrix['PC3_abs'] = com_matrix['PC3'].abs()
    com_matrix['PC_top3_sum'] = com_matrix.loc[:, ['PC1_abs', 'PC2_abs', 'PC3_abs']].sum(axis=1)
    if loading_matrix_file_path is not None:
        print(f'Saving loading matrix to file: {loading_matrix_file_path}')
        com_matrix.to_csv(loading_matrix_file_path, index_label='gene_name')
    return loading


def filtering_by_gene_list_and_pca_plot(bulk_exp, tcga_exp, gene_list, result_dir, simu_dataset_name,
                                        n_components=5, pca_model_name_postfix='', bulk_exp_type='log_space',
                                        tcga_exp_type='TPM', pca_model_file_path=None, pca_data_file_path=None,
                                        h5ad_file_path=None, cell_frac_file=None, figsize=(5, 5)):
    """
    Filtering by gene list and pca plot
    :param bulk_exp: log2cpm1p
    :param tcga_exp: TPM
    :param gene_list:
    :param result_dir:
    :param n_components:
    :param pca_model_name_postfix:
    :param bulk_exp_type:
    :param tcga_exp_type:
    :param pca_model_file_path:
    :param pca_data_file_path:
    :param simu_dataset_name:
    :param h5ad_file_path: save simulated bulk GEPs depending on filtered gene list to .h5ad file if not None
    :param cell_frac_file:
    :param figsize:
    :return:
    """
    bulk_obj = ReadExp(bulk_exp, exp_type=bulk_exp_type)
    bulk_obj.align_with_gene_list(gene_list=gene_list)
    bulk_exp = bulk_obj.get_exp()
    tcga_obj = ReadExp(tcga_exp, exp_type=tcga_exp_type)
    tcga_obj.align_with_gene_list(gene_list=gene_list)
    tcga_obj.to_log2cpm1p()
    tcga_exp = tcga_obj.get_exp()

    # PCA and plot
    check_dir(result_dir)
    assert n_components >= 2, 'n_components must be >= 2'
    if not os.path.exists(pca_data_file_path):
        # combine both simulated bulk cell GEPs and TCGA dataset together
        simu_bulk_with_tcga = pd.concat([bulk_exp, tcga_exp])
        pca_model = do_pca_analysis(exp_df=simu_bulk_with_tcga, n_components=n_components,
                                    pca_result_fp=pca_model_file_path)
        pcs = pca_model.transform(simu_bulk_with_tcga)
        if n_components >= 3:
            pca_df = pd.DataFrame(pcs[:, range(3)], index=simu_bulk_with_tcga.index, columns=['PC1', 'PC2', 'PC3'])
        else:
            pca_df = pd.DataFrame(pcs[:, range(2)], index=simu_bulk_with_tcga.index, columns=['PC1', 'PC2'])
        pca_df.to_csv(pca_data_file_path)
    else:
        print(f'{pca_data_file_path} already exists, skip PCA analysis')
        pca_df = pd.read_csv(pca_data_file_path, index_col=0)
        pca_model = load(pca_model_file_path)

    # plot
    title = f'{simu_dataset_name}_PCA_with_TCGA_{pca_model_name_postfix}'
    color_code = np.array([simu_dataset_name] * bulk_exp.shape[0] + ['TCGA'] * tcga_exp.shape[0])
    # cumsum = np.cumsum(pca_model.explained_variance_ratio_)
    # pca_df['class'] = color_code
    plot_pca(data=pca_df, figsize=figsize,
             result_fp=os.path.join(result_dir, title + '.png'),
             color_code=color_code, explained_variance_ratio=pca_model.explained_variance_ratio_)
    if h5ad_file_path is not None:
        assert cell_frac_file is not None, 'cell_frac_file should be provided if h5ad_file_path is not None'
        print(f'Saving filtered bulk exp to file: {h5ad_file_path}')
        if not os.path.exists(h5ad_file_path):
            print('Saving as .h5ad file...')
            dataset_info = f'Some genes were removed from this dataset for increasing the similarity ' \
                           f'between TCGA and simulated bulk cell GEPs. This similarity was evaluated by PCA analysis.'
            create_h5ad_dataset(simulated_bulk_exp_file_path=bulk_exp,
                                cell_fraction_file_path=cell_frac_file, dataset_info=dataset_info,
                                result_file_path=h5ad_file_path)


if __name__ == '__main__':
    # pass
    # gene_samples = gradient_generation_fraction(n=2)
    root_dir = r'F:\projects\001EMT-infiltration\analysis_result\014_test\nn_model'
    # cell_frac = pd.read_csv(os.path.join(root_dir, 'generated_frac_n1.csv'), index_col=0)
    # sc_fp = os.path.join(os.path.join(r'F:\projects\001EMT-infiltration\raw_data\single cells',
    #                                   'generated_11_cell_type_n1000.h5ad'))
    # simu_bulk_exp = simulate_bulk_expression(cell_frac=cell_frac, sc_exp_file_path=sc_fp)
    # simu_bulk_exp.write(os.path.join(root_dir, 'simu_bulk_exp_n1.h5ad'))
    # simu_bulk_n100_fp = os.path.join(root_dir, 'simu_bulk_exp_n100.h5ad')
    # prefix = 'simu_bulk_exp_n10'
    # simu_bulk_exp_fp = os.path.join(root_dir, prefix + '_log2cpm1p.csv')
    # sc_fp = os.path.join(os.path.join(r'F:\projects\001EMT-infiltration\raw_data\single cells',
    #                                   'generated_11_cell_type_n5000.h5ad'))
    # cell_frac_fp = os.path.join(root_dir, 'generated_frac_test_set_n10.csv')
    # if not os.path.exists(simu_bulk_exp_fp):
    #     cell_frac = pd.read_csv(cell_frac_fp, index_col=0)
    #     simulate_bulk_expression(cell_frac=cell_frac, sc_exp_file_path=sc_fp,
    #                              n_threads=4, result_dir=root_dir, prefix=prefix)
    #     simu_bulk_exp.to_csv(simu_bulk_n100_selected_id_fp)
