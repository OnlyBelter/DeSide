import numpy as np
import pandas as pd


def get_core_zone_of_pca(pca_data: pd.DataFrame, col_x, col_y, q_lower, q_upper):
    """
    get core zone of TCGA in PCA plot
    :param pca_data: PCA data
    :param col_x: x axis column name
    :param col_y: y axis column name
    :param q_lower: lower quantile
    :param q_upper: upper quantile
    :return: coordinate_core_zone, n_tcga_in_core_zone, n_non_tcga_in_core_zone
    """
    assert 'TCGA' in pca_data['class'].unique()
    x_tcga = pca_data.loc[pca_data['class'] == 'TCGA', col_x]
    y_tcga = pca_data.loc[pca_data['class'] == 'TCGA', col_y]
    x_q_lower = np.quantile(x_tcga, q_lower)
    x_q_upper = np.quantile(x_tcga, q_upper)
    y_q_lower = np.quantile(y_tcga, q_lower)
    y_q_upper = np.quantile(y_tcga, q_upper)
    core_zone = pca_data.loc[(pca_data[col_x] >= x_q_lower) & (pca_data[col_x] <= x_q_upper) &
                             (pca_data[col_y] >= y_q_lower) & (pca_data[col_y] <= y_q_upper), :]
    coordinate_core_zone = {'x_lower': x_q_lower, 'x_upper': x_q_upper, 'y_lower': y_q_lower, 'y_upper': y_q_upper}
    n_tcga_in_core_zone = core_zone.loc[core_zone['class'] == 'TCGA', :].shape[0]
    n_non_tcga_in_core_zone = core_zone.loc[core_zone['class'] != 'TCGA', :].shape[0]
    return coordinate_core_zone, n_tcga_in_core_zone, n_non_tcga_in_core_zone
