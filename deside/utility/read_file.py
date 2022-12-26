import os
import sys
import numpy as np
import pandas as pd
import anndata as an
from tqdm import tqdm
from typing import Union
from scipy.sparse import csr_matrix
from sklearn import preprocessing as pp
from .pub_func import (log_exp2cpm, read_df, non_log2log_cpm, non_log2cpm,
                       get_inx2cell_type, sorted_cell_types, log2_transform)


class ReadH5AD(object):
    """
    read .h5ad file, usually the values are log2 transformed
    """
    def __init__(self, file_path: str, show_info: bool = False):
        """
        :param file_path: the file path of .h5ad file, samples by genes, log2cpm1p format
        """
        self.dataset = an.read_h5ad(file_path)
        if show_info:
            print(self.dataset)

    def get_df(self, result_file_path: str = None, convert_to_tpm: bool = False,
               scaling_by_sample: bool = False) -> pd.DataFrame:
        """
        convert to DataFrame, samples by genes, log space (log2cpm1p)
        :param result_file_path:
        :param convert_to_tpm: whether to convert log2cpm1p to TPM
        :param scaling_by_sample: whether to scale the expression values of each sample to [0, 1] by 'min_max'
        """
        if type(self.dataset.X) == csr_matrix:
            x_data = self.dataset.X.A.astype(np.float32)  # convert sparse matrix to dense matrix
        else:
            x_data = self.dataset.X.astype(np.float32)

        if convert_to_tpm:
            x_data = log_exp2cpm(x_data)
        if scaling_by_sample:
            scaler = pp.MinMaxScaler(feature_range=(0, 1), copy=True)
            x_data = scaler.fit_transform(x_data.T).T

        df = pd.DataFrame(data=x_data, index=self.dataset.obs.index,
                          columns=self.dataset.var.index).round(3)
        if result_file_path is not None:
            df.to_csv(result_file_path, float_format='%.3f')
        return df

    def get_cell_fraction(self) -> Union[None, pd.DataFrame]:
        if self.dataset.obs.shape[1] > 0:
            return self.dataset.obs.round(3)
        else:
            print('   There is no cell fraction in this .h5ad file')
            return None

    def get_h5ad(self):
        return self.dataset


class ReadExp(object):
    """
    read gene expression file, and convert to specific format (TPM / CPM, log2cpm1p)
    - TPM: transcript per million
    - CPM: UMI reads per million (3' end sc-RNA seq), same as TPM in the full-length RNA-seq of bulk cells
    - log_space: log2(CPM + 1), or log2(TPM + 1)
    - non_log: non log space, could be normalized to TPM
    Data from full-length protocols may benefit from normalization methods that take into account gene length
    (e.g. Patelet al, 2014; Kowalczyket al,2015; Soneson & Robinson, 2018),
    while 3' enrichment data do not.
    A  commonly  used  normalization  method  for  full-length scRNA-seq data is TPM normalization (Liet al, 2009),
    which comes from bulk RNA-seq analysis. (Luecken, M. D. & Theis, F. J., Mol. Syst. Biol. 15, e8746 (2019))
    """
    def __init__(self, exp_file, exp_type='TPM', transpose: bool = False):
        """
        :param exp_file: file path or DataFrame, samples by genes
        :param exp_type: TPM / CPM, log_space, non_log
        :param transpose: transpose if exp_file formed as genes (index) by samples (columns)
        """
        assert exp_type in ['TPM', 'CPM', 'log_space', 'non_log']
        self.file_type = exp_type
        self.exp = read_df(exp_file)
        if transpose:
            self.exp = self.exp.T
        self.scaled_by_sample = False

    def to_tpm(self):
        if self.file_type == 'non_log':
            self.exp = non_log2cpm(self.exp)
        elif self.file_type == 'TPM' or self.file_type == 'CPM':
            pass
        elif self.file_type == 'log_space':
            self.exp = log_exp2cpm(self.exp)
        self.file_type = 'TPM'

    def to_log2cpm1p(self):
        if self.file_type != 'log_space':
            self.exp = non_log2log_cpm(self.exp, transpose=False)
        else:
            print('   This file has already log2 transformed.')
        self.file_type = 'log_space'

    def get_file_type(self):
        return self.file_type

    def get_exp(self):
        return self.exp.round(3)

    def save(self, file_path, sep=',', transpose: bool = False):
        if transpose:
            self.exp = self.exp.T.copy()
        self.exp.to_csv(file_path, sep=sep, float_format='%.3f')

    def do_scaling(self):
        """
        Scaling by sample to [0, 1], same as Scaden
        """
        if not self.scaled_by_sample:
            scaler = pp.MinMaxScaler(feature_range=(0, 1), copy=True)
            x_scaled = scaler.fit_transform(self.exp.T).T  # scaling by column (sample), so T is needed here
            self.scaled_by_sample = True
            self.exp = pd.DataFrame(data=x_scaled, index=self.exp.index,
                                    columns=self.exp.columns).round(3)
        # return self.exp

    def do_scaling_by_constant(self, divide_by=20):
        """
        Scaling by dividing a constant in log space, so all expression values are in [0, 1)
        """
        if self.file_type != 'log_space':
            raise ValueError('   This file is not in log space')
        if np.any(self.exp.values > 1.0):
            self.exp = self.exp / divide_by

    def align_with_gene_list(self, gene_list: list = None, fill_not_exist=False):
        common_genes = [i for i in gene_list if i in self.exp.columns]
        not_exist_in_exp = [i for i in gene_list if i not in common_genes]
        removed_genes = [i for i in self.exp.columns if i not in common_genes]
        print(f'{len(common_genes)} common genes will be used, {len(removed_genes)} genes will be removed.')
        self.exp = self.exp.loc[:, common_genes].copy()
        if fill_not_exist and (len(not_exist_in_exp) != 0):
            print(f'{len(not_exist_in_exp)} genes are not in current dataset, 0 will be filled')
            _not_exist_exp = pd.DataFrame(np.zeros((self.exp.shape[0], len(not_exist_in_exp))), index=self.exp.index,
                                          columns=not_exist_in_exp)
            self.exp = pd.concat([self.exp, _not_exist_exp], axis=1)
            self.exp = self.exp.loc[:, gene_list].copy()
        if self.file_type == 'log_space':  # scaling to TPM after alignment
            self.to_tpm()
            self.to_log2cpm1p()
        else:
            self.file_type = 'non_log'
            self.to_tpm()


class TrainingDatasetLoader(object):
    def __init__(self, pos_data_path, neg_data_path):

        print("Opening {} and {}".format(pos_data_path, neg_data_path))
        sys.stdout.flush()

        self.cache_pos = ReadH5AD(pos_data_path)
        self.cache_neg = ReadH5AD(neg_data_path)

        print("Loading data into memory...")
        sys.stdout.flush()
        self.gep_pos = self.cache_pos.get_df()  # log space, samples by genes
        self.cell_prop_pos = self.cache_pos.get_cell_fraction()  # samples by cell types
        self.gep_neg = self.cache_neg.get_df()
        self.cell_prop_neg = self.cache_neg.get_cell_fraction()

        # self.images = self.cache['images'][:]
        # self.labels = self.cache['labels'][:].astype(np.float32)
        # self.image_dims = self.images.shape
        # n_train_samples = self.cell_prop_pos.shape[0] + self.cell_prop_neg.shape[0]

        # self.train_inds = np.random.permutation(np.arange(n_train_samples))

        self.pos_train_inds = np.random.permutation(np.arange(self.cell_prop_pos.shape[0]))
        self.neg_train_inds = np.random.permutation(np.arange(self.cell_prop_neg.shape[0]))

    def get_train_size(self):
        return self.cell_prop_pos.shape[0] + self.cell_prop_neg.shape[0]

    def get_n_genes(self):
        # the number of genes in each GEP
        return self.gep_pos.shape[1]

    def get_n_cell_types(self):
        return self.cell_prop_pos.shape[1]

    # def get_train_steps_per_epoch(self, batch_size, factor=10):
    #     return self.get_train_size()//factor//batch_size

    def get_batch(self, n, only_faces=False, p_pos=None, p_neg=None):
        if only_faces:
            selected_inds = np.random.choice(self.pos_train_inds, size=n, replace=False, p=p_pos)
            train_gep = self.gep_pos.iloc[selected_inds, :].copy()
            train_cell_prop = self.cell_prop_pos.iloc[selected_inds, :].copy()
        else:
            # selected_pos_inds = tfp.distributions.Multinomial(total_count=n//2, logits=self.pos_train_inds, )
            selected_pos_inds = np.random.choice(self.pos_train_inds, size=n//2, replace=False, p=p_pos)
            selected_neg_inds = np.random.choice(self.neg_train_inds, size=n//2, replace=False, p=p_neg)
            t_gep1 = self.gep_pos.iloc[selected_pos_inds, :].copy()
            t_cell_prop1 = self.cell_prop_pos.iloc[selected_pos_inds, :].copy()
            t_gep2 = self.gep_neg.iloc[selected_neg_inds, :].copy()
            t_cell_prop2 = self.cell_prop_neg.iloc[selected_neg_inds, :].copy()
            train_gep = pd.concat([t_gep1, t_gep2])
            train_cell_prop = pd.concat([t_cell_prop1, t_cell_prop2])

        return train_gep.values, train_cell_prop.values

    # def get_n_most_prob_faces(self, prob, n):
    #     idx = np.argsort(prob)[::-1]
    #     most_prob_inds = self.pos_train_inds[idx[:10*n:10]]
    #     return (self.images[most_prob_inds,...]/255.).astype(np.float32)

    def get_all_pos(self) -> np.ndarray:
        return self.gep_pos.values


def read_single_cell_type_dataset(sct_dataset_file_path: str, latent_z_nn_info_file: Union[str, pd.DataFrame] = None):
    """
    positive samples of SCT (single cell type), generated by SingleCellTypeGEPGenerator
    :param sct_dataset_file_path: the file path of GEPs for single cell type (SCT), positive samples
    :param latent_z_nn_info_file: neighbor information of latent z for all samples, used for QC
    """
    sct_dataset_obj = ReadH5AD(sct_dataset_file_path)
    sct_dataset_df = sct_dataset_obj.get_df(convert_to_tpm=True)
    inx2cell_type = get_inx2cell_type()
    if (latent_z_nn_info_file is not None) and os.path.exists(latent_z_nn_info_file):
        latent_z_nn_info = read_df(latent_z_nn_info_file)
        latent_z_nn_info = \
            latent_z_nn_info.loc[
                (latent_z_nn_info['class'] != -1) &  # only positive samples
                (latent_z_nn_info['n_neighbor_class'] == 1) &  # all neighbors belong to the same cell type
                (latent_z_nn_info['class'] == latent_z_nn_info['pred_class']),  # predicted class is same as true class
                ['class']].copy()
        latent_z_nn_info['cell_type'] = latent_z_nn_info['class'].map(lambda x: inx2cell_type[x])
        sct_dataset_obs = latent_z_nn_info.copy()
    else:
        sct_dataset_obs = sct_dataset_obj.get_cell_fraction()
        sct_dataset_obs['class'] = sct_dataset_obs.values.argmax(axis=1)
        sct_dataset_obs['cell_type'] = sct_dataset_obs['class'].map(lambda x: inx2cell_type[x])
    return sct_dataset_obs, sct_dataset_df


def read_sct_dataset(sct_files: list, scaling_by_constant=False):
    """
    combine all sct dataset together
    :param sct_files: a list of file paths for all used SCT datasets, .h5ad format
    :param scaling_by_constant:
    :return:
    """
    sct_exp_list, sct_info_list = [], []
    for file_path in sct_files:
        sct_obj = ReadH5AD(file_path)
        sct_df = sct_obj.get_df().astype(np.float32)
        sct_cell_frac = sct_obj.get_cell_fraction()
        if scaling_by_constant:
            sct_obj = ReadExp(sct_df, exp_type='log_space')
            sct_obj.do_scaling_by_constant()
            sct_df = sct_obj.get_exp()
        sct_exp_list.append(sct_df)
        sct_info_list.append(sct_cell_frac)
    sct_exp = pd.concat(sct_exp_list, axis=0)
    sct_info = pd.concat(sct_info_list, axis=0)
    sample2inx_sct_ds = {sample_id: i for i, sample_id in enumerate(sct_info.index.to_list())}
    assert len(sample2inx_sct_ds) == sct_exp.shape[0], 'Duplicated sample ids are not allowed.'
    return sct_exp, sample2inx_sct_ds


def read_ds_two_parts(dataset2info: dict, dataset_name: str, sample2inx_sct_ds: dict, scaling_by_constant=False):
    """

    :param dataset2info:
    :param dataset_name:
    :param sample2inx_sct_ds: mapping sample id to index in SCT dataset (used in simulating bulk GEPs)
    :param scaling_by_constant:
    :return:
    """
    x_list, y_list = [], []
    col_name_sc_id = [f'{_}_sc_inx' for _ in sorted_cell_types]
    for k, v in dataset2info.items():
        path_to_exp_data = v['h5ad_file_path']
        dataset_type = v['dataset_type']
        x_obj = ReadH5AD(path_to_exp_data)
        cell_prop = x_obj.get_cell_fraction()
        x_df = x_obj.get_df().astype(np.float32)
        if dataset_type == 'sct':
            cell_prop[col_name_sc_id] = 0
            for r_id, _ in tqdm(cell_prop.iterrows()):
                cell_prop.loc[r_id, col_name_sc_id] = sample2inx_sct_ds[r_id]
        else:
            path_to_sc_id = os.path.join(v['file_dir'], v['sampled_sc_id_file'])
            sc_ids = pd.read_csv(path_to_sc_id, index_col=[0, 1])
            r_id2sc_inx = {}
            for r_id, row in tqdm(cell_prop.iterrows()):
                q_inx = list(zip([r_id] * len(sorted_cell_types), sorted_cell_types))
                selected_sc_id = sc_ids.loc[q_inx, 'selected_cell_id'].to_list()
                r_id2sc_inx[r_id] = [sample2inx_sct_ds[_] for _ in selected_sc_id]
            r_id2sc_inx_df = pd.DataFrame.from_dict(r_id2sc_inx, orient='index', columns=col_name_sc_id)
            cell_prop = cell_prop.merge(r_id2sc_inx_df, left_index=True, right_index=True)
            # cell_prop.loc[r_id, col_name_sc_id] =
        if scaling_by_constant:
            x_obj = ReadExp(x_df, exp_type='log_space')
            x_obj.do_scaling_by_constant()
            x_df = x_obj.get_exp()
        x_list.append(x_df.copy())
        y_list.append(cell_prop.copy())
    x = pd.concat(x_list)
    y = pd.concat(y_list)

    print(f'   There are {x.shape[0]} samples in {dataset_name}...')
    return x, y


def align_with_gene_list(dataset_file: Union[pd.DataFrame, str], gene_list, gene_exp_type) -> pd.DataFrame:
    """

    :param dataset_file: file path or DataFrame, samples by genes
    :param gene_list:
    :param gene_exp_type: TPM / CPM, log_space, non_log
    :return: log2tpm1p
    """
    dataset = ReadExp(dataset_file, exp_type=gene_exp_type).get_exp()
    gene_not_in_ds = [i for i in gene_list if i not in dataset.columns]
    dataset[gene_not_in_ds] = 0  # set expression values to 0 if a gene not exists
    dataset = dataset.loc[:, gene_list].copy()  # remove genes not contain in gene_list
    dataset = log_exp2cpm(dataset)
    dataset = log2_transform(dataset)
    return dataset
