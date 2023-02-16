import numpy as np
import pandas as pd
from typing import Union
from .read_file import ReadExp
from sklearn.neighbors import NearestNeighbors
from .pub_func import default_core_marker_genes, aggregate_marker_gene_exp, read_df


class ExpObj(ReadExp):
    """
    An object to save gene expression profiles
    """
    def __init__(self, exp_file, exp_type: str, transpose: bool = False):
        super().__init__(exp_file, exp_type=exp_type, transpose=transpose)
        self.marker_ratio = None

    def cal_marker_gene_ratio(self, cell_types: list, marker_genes: dict = None,
                              agg_methods: dict = None, show_marker_gene: bool = False):
        if marker_genes is None:
            marker_genes = {k: v for k, v in default_core_marker_genes.items() if k in cell_types}

        self.marker_ratio = aggregate_marker_gene_exp(exp_df=self.exp, marker_genes=marker_genes,
                                                      return_ratio=True,
                                                      agg_methods=agg_methods,
                                                      show_marker_gene=show_marker_gene)

    def get_marker_ratios(self):
        if self.marker_ratio is not None:
            return self.marker_ratio
        else:
            raise ValueError('"marker_ratio" is None, please calculate marker ratios first '
                             'by calling function "cal_marker_gene_ratio".')


class QueryNeighbors(object):
    def __init__(self, df_file: Union[str, pd.DataFrame]):
        """
        :param df_file: a file path or DataFrame of marker ratios for cell types, samples by cell types
        """
        self.df = read_df(df_file)

    def fit_nn_model(self, n_neighbors: int = 2, radius: float = None):
        """
        fit a NearestNeighbors model
        - if radius is None, fit a model with specific n_neighbors
        - if radius is not None, fit a model with specific radius
        """
        if radius is not None:
            nn_model = NearestNeighbors(radius=radius)
        else:
            nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
        nn_model.fit(self.df.values)
        return nn_model

    def get_nn(self, q_df_file: Union[str, pd.DataFrame] = None, n_neighbors: int = 1) -> pd.DataFrame:
        """
        query nearest neighbors (NN) from the NN model which is fitted on self.df
        :param q_df_file: query NN for all samples in this dataset
        - if None: query the distance of nearest neighbor for all samples in the same df, self.df
        :param n_neighbors: the number of nearest neighbors to query
        :return: the nearest neighbor and corresponding distance ['nn', 'nn_dis']
        """
        assert n_neighbors >= 1, 'n_neighbors should >= 1'
        nn_model = self.fit_nn_model(n_neighbors=n_neighbors + 1)
        if q_df_file is not None:
            q_df = read_df(q_df_file)
            nn_index = 0  # 0 is queried NN from fitted nn_model
        else:
            q_df = self.df.copy()
            nn_index = 1  # 0 is itself
        assert np.all(self.df.columns == q_df.columns), 'self.df and q_df should have the same columns with same order'
        dis, indices = nn_model.kneighbors(q_df.values)
        # the distance of nearest neighbor for each sample of q_df
        result = pd.DataFrame(index=q_df.index)
        if n_neighbors == 1:
            result['nn'] = self.df.iloc[indices[:, nn_index], 0].index.to_list()  # the index or sample_name of NN
            result['nn_dis'] = dis[:, nn_index]
        else:  # > 1
            nn_cols = [f'nn_{i}' for i in range(n_neighbors)]
            nn_dis_cols = [f'nn_dis_{i}' for i in range(n_neighbors)]
            for j in range(n_neighbors):
                result[nn_cols[j]] = self.df.iloc[indices[:, nn_index + j], 0].index.to_list()
                result[nn_dis_cols[j]] = np.round(dis[:, nn_index + j], 3)
        result.index.name = 'query_sample_id'
        return result

    def get_neighbors_by_radius(self, radius: float, q_df_file: Union[str, pd.DataFrame] = None,
                                share_neighbors: bool = False, n_top: int = None) -> pd.DataFrame:
        """
        query neighbors by a specific radius
        :param radius: only find neighbors within this radius
        :param q_df_file: query NN for all samples in this dataset
        :param share_neighbors: if share neighbors between different query samples
        - False: duplicated neighbors will be removed among different query samples
        - True: duplicated neighbors will be kept
        :param n_top: if too many neighbors were founded for one single sample, only keep n_top neighbors
        return: the nearest neighbors within the fixed radius and corresponding distance ['nn', 'nn_dis']
        """
        nn_model = self.fit_nn_model(radius=radius)
        if q_df_file is not None:
            q_df = read_df(q_df_file)
        else:
            q_df = self.df.copy()  # query in the same dataset, self.df

        t2s_neighbors = []
        inx_collector = {}
        inx2sample_id_df = {i: sample_id for i, sample_id in enumerate(self.df.index.to_list())}
        for q_inx in q_df.index:
            query_ratio = q_df.loc[[q_inx], :]
            dis, res_index = nn_model.radius_neighbors(query_ratio)
            res_inx2dis = dict(zip(res_index[0], dis[0]))
            if res_inx2dis:
                res_inx2dis = {k: v for k, v in res_inx2dis.items()}
                res_inx2dis = sorted(res_inx2dis.items(), key=lambda x: x[1])  # a list here
                if not share_neighbors:
                    res_inx2dis = [_ for _ in res_inx2dis if _[0] not in inx_collector]
                if (n_top is not None) and len(res_inx2dis) > n_top:
                    res_inx2dis = res_inx2dis[:n_top]
                current_neighbors_df = pd.DataFrame.from_dict(dict(
                    [(inx2sample_id_df[inx], dis) for inx, dis in res_inx2dis],
                ), orient='index', columns=['nn_dis'])
                current_neighbors_df['query_sample_id'] = q_inx
                t2s_neighbors.append(current_neighbors_df)
                for each_item in res_inx2dis:
                    inx_collector[each_item[0]] = 1
        if t2s_neighbors:
            results = pd.concat(t2s_neighbors)
            results.index.name = 'nn'
            results.reset_index(inplace=True)
            results.set_index('query_sample_id', inplace=True)
        else:
            results = pd.DataFrame()
        return results

    def get_quantile_of_nn_distance(self, quantile: float, q_df_file: Union[str, pd.DataFrame] = None):
        """
        get the quantile of NN distance
        """
        nn_distance = self.get_nn(q_df_file=q_df_file)
        q_dis = np.quantile(nn_distance['nn_dis'], quantile)
        # print(f'   {quantile} quantile distance is: {q_dis}')
        return q_dis
