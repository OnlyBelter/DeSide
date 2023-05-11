import os
import pandas as pd
import numpy as np
from typing import Union
import statsmodels.api as sm
from sklearn.metrics import median_absolute_error
from ..utility import (print_df, cal_relative_error, calculate_rmse, check_dir, get_corr,
                       read_xy, read_df, get_inx2cell_type, log2_transform, get_core_zone_of_pca,
                       get_ccc, read_cancer_purity, cancer_types)
# from ..utility.read_file import ReadExp
from .plot_nn import plot_corr_two_columns
import matplotlib.patches as patches
# import importlib
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# set_fig_style()
# sns.set(font_scale=1.5)
# sns.set_style('white')


class ScatterPlot(object):
    def __init__(self, x: Union[str, pd.DataFrame], y: Union[str, pd.DataFrame],
                 postfix: str = None, group_info: pd.DataFrame = None):
        """
        :param x:
        :param y:
        :param postfix: only for naming
        """
        self.x = read_xy(x)
        self.y = read_xy(y)
        common_inx = [i for i in self.x.index if i in self.y.index]
        self.postfix = postfix
        self.group_info = group_info
        if group_info is not None:
            common_inx = [i for i in group_info.index if i in common_inx]
            self.group_info = self.group_info.loc[common_inx, :].copy()
        self.show_columns = None
        self.x = self.x.loc[common_inx, :].copy()
        self.y = self.y.loc[common_inx, :].copy()
        assert np.all(self.x.index == self.y.index)

    def plot(self, show_columns: Union[list, dict], result_file_dir: str = None,
             x_label: str = None, y_label: str = None, show_corr: bool = True, show_rmse: bool = False,
             show_diag: bool = True, show_mae: bool = False, pred_by: str = None,
             fig_size=(8, 8), group_by: str = None, show_reg_line: bool = False, s=6, order=1,
             legend_loc: str = 'best'):
        """
        :param show_columns: a list of column names in both x and y, could be multiple common columns
            or a dict {'x': '', 'y': ''}, only one column allowed
        :param result_file_dir:
        :param x_label:
        :param y_label:
        :param show_corr:
        :param show_rmse:
        :param show_mae: media absolute error
        :param show_diag:
        :param pred_by: algorithm name, will be showed in ylabel
        :param fig_size:
        :param group_by: one of the column name in self.group_info
        :param show_reg_line: fit regression model
        :param s:
        :param legend_loc:
        :param order: 1 for linear regression; 2 for Polynomial Regressions, y = alpha + beta1*x + beta2*x^2
        """
        plt.figure(figsize=fig_size)
        ax = plt.axes()
        # f, ax = plt.subplots(figsize=fig_size)

        all_x = []
        all_y = []
        self.show_columns = show_columns
        if type(show_columns) == dict:
            self.x = self.x[show_columns['x']]
            self.y = self.y[show_columns['y']]
            all_x.append(self.x)
            all_y.append(self.y)
            if (self.group_info is not None) and (group_by in self.group_info.columns):
                inx = self.group_info[group_by] == 1
                plt.scatter(self.x[~inx], self.y[~inx], s=1, label='others')
                plt.scatter(self.x[inx], self.y[inx], s=5, label=group_by, marker='x')
            else:
                plt.scatter(self.x, self.y, s=s, label=show_columns['x'], alpha=.4)  # only 1 vs 1 column
            if show_reg_line:
                self.fit_reg_model(ax=ax, order=order)
        else:
            show_columns = [i for i in show_columns if i in self.y.columns]
            # for cell_type in show_columns:
            #     if cell_type not in y_true.columns:
            #         y_true[cell_type] = 0
            show_columns_str = ', '.join(show_columns)
            assert np.all([i in self.x.columns for i in show_columns]), \
                f'All of elements in show_columns ({show_columns_str}) should exist in ' \
                f'the columns of both x ({self.x.columns}) and y ({self.y.columns})'

            # y_true = self.x.loc[:, show_columns]
            # y_pred = self.y.loc[:, show_columns]
            # sns.set(font_scale=font_scale)
            # plt.figure(figsize=(8, 6))
            for i, col in enumerate(show_columns):
                _x = self.x.loc[:, col]
                _y = self.y.loc[:, col]
                all_x.append(_x)
                all_y.append(_y)
                plt.scatter(_x, _y, label=col, s=6, alpha=1 - 0.05 * i)
        x_left, x_right = plt.xlim()
        y_bottom, y_top = plt.ylim()
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        if show_diag:
            _ = max(x_right, y_top)
            plt.plot([0, _], [0, _], linestyle='--', color='tab:gray', linewidth=1, alpha=0.8)
        if show_corr:  # show metrics in test set
            corr = get_corr(all_x, all_y)
            plt.text(x_right * 0.60, y_top * 0.10, 'corr = {:.2f}'.format(corr))
        if show_mae:
            mae = median_absolute_error(y_true=all_x, y_pred=all_y)
            plt.text(x_right * 0.60, y_top * 0.07, 'MAE = {:.2f}'.format(mae))
        if show_rmse and (not show_mae):
            rmse = calculate_rmse(y_true=pd.DataFrame(all_x), y_pred=pd.DataFrame(all_y))
            plt.text(x_right * 0.60, y_top * 0.07, 'RMSE = {:.2f}'.format(rmse))
        if show_rmse and show_mae:
            rmse = calculate_rmse(y_true=pd.DataFrame(all_x), y_pred=pd.DataFrame(all_y))
            plt.text(x_right * 0.60, y_top * 0.04, 'RMSE = {:.2f}'.format(rmse))
        if x_label:
            plt.xlabel(x_label)
        else:
            plt.xlabel('y_true')
        if y_label:
            plt.ylabel(y_label)
        elif pred_by:
            plt.ylabel('Pred by {} (n={})'.format(pred_by, self.y.shape[0]))
        else:
            plt.ylabel('y_pred')
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(labels) > 1:
            plt.legend(handles[1:], labels[1:], loc=legend_loc)
        plt.tight_layout()
        if result_file_dir:
            plt.savefig(os.path.join(result_file_dir,
                                     'x_vs_y_{}.png'.format(self.postfix)), dpi=300)
            # plt.savefig(os.path.join(result_file_dir,
            #                          'x_vs_y_{}.svg'.format(self.postfix)), dpi=300)
        plt.close()

    def fit_reg_model(self, ax, alpha_ci=0.05, order=1):
        """
        only used 1vs1 comparing, show_columns should be a dict
        :param ax
        :param alpha_ci: 1 - alpha_ci confidence interval
        :param order: 1 for linear regression; 2 for Polynomial Regressions, y = alpha + beta1*x + beta2*x^2
        """

        if type(self.x) == pd.Series:
            self.x = self.x.to_frame()
        self.x['intercept'] = 1  # add 1 as intercept column to fit `intercept`
        x_col = self.show_columns['x']  # column name, a str
        x_col_square = f'{x_col}^2'
        if order == 2:
            self.x[x_col_square] = self.x[x_col] ** 2
            mod = sm.OLS(self.y, self.x.loc[:, ['intercept', x_col, x_col_square]])
        else:  # order == 1
            mod = sm.OLS(self.y, self.x.loc[:, ['intercept', x_col]])
        res = mod.fit()
        # print(res.summary())
        ci = res.conf_int(alpha_ci)  # 95%, +/- 2*SD
        x_lin = np.linspace(self.x[x_col].min(), self.x[x_col].max(), 20)
        beta1 = res.params[x_col]
        alpha = res.params['intercept']
        beta2 = 0
        if order == 2:
            beta2 = res.params[x_col_square]
        y_reg_line = x_lin * beta1 + alpha + np.power(x_lin, 2) * beta2
        if order == 2:
            y_lower_bound = x_lin * ci.loc[x_col, 0] + ci.loc['intercept', 0] + \
                np.power(x_lin, 2) * ci.loc[x_col_square, 0]
            y_upper_bound = x_lin * ci.loc[x_col, 1] + ci.loc['intercept', 1] + \
                np.power(x_lin, 2) * ci.loc[x_col_square, 1]
        else:
            y_lower_bound = x_lin * ci.loc[x_col, 0] + ci.loc['intercept', 0]
            y_upper_bound = x_lin * ci.loc[x_col, 1] + ci.loc['intercept', 1]
        xy = self.x.copy()
        xy['y_pred'] = self.x[x_col]
        xy['y_true'] = self.y
        sns.regplot(x='y_pred', y='y_true', data=xy, ax=ax, order=order,
                    x_estimator=np.mean,
                    scatter_kws={"s": 5}, color='tab:grey', x_bins=50,
                    line_kws={'color': 'tab:orange', 'lw': 1, 'alpha': 0})
        # p_value = res.pvalues[x_col]
        # r2 = res.rsquared
        # print(f'p_value: {p_value}', f'R^2: {r2}')
        if alpha > 0:
            if order == 2:
                plt.plot(x_lin, y_reg_line, c='tab:orange',
                         label=f'$y= {beta2: .2f}x^2 + {beta1: .2f}x + {alpha: .2f}$', linewidth=1)
            else:
                plt.plot(x_lin, y_reg_line, c='tab:orange',
                         label=f'$y={beta1: .2f}x + {alpha: .2f}$', linewidth=1)
        else:
            if order == 2:
                plt.plot(x_lin, y_reg_line, c='tab:orange',
                         label=f'$y= {beta2: .2f}x^2 + {beta1: .2f}x - {abs(alpha): .2f}$', linewidth=1)
            else:
                plt.plot(x_lin, y_reg_line, c='tab:orange',
                         label=f'$y={beta1: .2f}x - {abs(alpha): .2f}$', linewidth=1)
        plt.plot(x_lin, y_lower_bound, c='tab:brown', label=f'{100 - alpha_ci * 100}% CI', linewidth=1)
        plt.plot(x_lin, y_upper_bound, c='tab:brown', linewidth=1)


def compare_y_y_pred_plot(y_true: Union[str, pd.DataFrame], y_pred: Union[str, pd.DataFrame],
                          show_columns: list = None, result_file_dir=None, annotation: dict = None,
                          y_label=None, x_label=None, model_name='average',
                          show_metrics: bool = False, figsize: tuple = (8, 8)):
    """
    Plot y against y_pred to visualize the performance of prediction result

    :param y_true: this file contains the ground truth of cell fractions when it was simulated

    :param y_pred: this file contains the predicted value of y

    :param show_columns: this list contains the name of columns that want to plot in figure

    :param result_file_dir: where to save results

    :param annotation: annotations that need to show in figure, {anno_name: {col1: value1, col2: value2, ...}, ...}

    :param y_label: y label

    :param x_label: x label

    :param model_name: only for naming files

    :param show_metrics: show correlation and RMSE

    :param figsize: figure size

    :return: None
    """
    if show_columns is None:
        show_columns = []
    if annotation is None:
        annotation = {}
    y_true = read_xy(a=y_true, xy='cell_frac')
    y_pred = read_xy(a=y_pred, xy='cell_frac')
    if '1-others' in show_columns:
        if 'Cancer Cells' in y_true.columns:
            y_true['1-others'] = y_true['Cancer Cells']
        else:
            y_true['1-others'] = 0
    if ('T Cells' in y_pred.columns) and ('T Cells' not in y_true.columns):
        y_true['T Cells'] = y_true.loc[:, ['CD4 T', 'CD8 T']].sum(axis=1)
    # less cell type than show_columns for this dataset
    show_columns = [i for i in show_columns if i in y_true.columns]
    # for cell_type in show_columns:
    #     if cell_type not in y_true.columns:
    #         y_true[cell_type] = 0
    show_columns_str = ', '.join(show_columns)
    assert np.all([i in y_true.columns for i in show_columns]) and \
           np.all([i in y_pred.columns for i in show_columns]), \
           f'All of elements in show_columns ({show_columns_str}) should exist in ' \
           f'the columns of both y_true ({y_true.columns}) and y_pred ({y_pred.columns})'
    common_inx = [i for i in y_true.index if i in y_pred.index]

    y_true = y_true.loc[common_inx, show_columns]
    y_pred = y_pred.loc[common_inx, show_columns]
    # sns.set(font_scale=font_scale)
    plt.figure(figsize=figsize)
    all_x = []
    all_y = []
    for i, col in enumerate(show_columns):
        _x = y_true.loc[:, col]
        _y = y_pred.loc[:, col]
        all_x.append(_x)
        all_y.append(_y)
        plt.scatter(_x, _y, label=col, s=6, alpha=1 - 0.05 * i)
        if annotation:
            x_left, x_right = plt.xlim()
            y_bottom, y_top = plt.ylim()
            for k, v in annotation.items():
                plt.text(x_left * 1.5, y_top * 0.8, 'k ({.4f})'.format(v[col]))
    x_left, x_right = plt.xlim()
    y_bottom, y_top = plt.ylim()
    x_max = x_right + x_right * 0.01
    y_max = y_top + y_top * 0.01
    plt.plot([0, max(x_max, y_max)], [0, max(x_max, y_max)], linestyle='--', color='tab:gray')
    if show_metrics:  # show metrics in test set
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        corr = get_corr(all_x, all_y)
        rmse = calculate_rmse(y_true=pd.DataFrame(all_x), y_pred=pd.DataFrame(all_y))
        plt.text(0.70 * x_max, 0.16 * y_max, 'corr = {:.3f}'.format(corr))
        plt.text(0.70 * x_max, 0.10 * y_max, 'RMSE = {:.3f}'.format(rmse))
    if x_label:
        plt.xlabel(x_label)
    else:
        plt.xlabel('y_true')
    if y_label:
        plt.ylabel(y_label)
    else:
        plt.ylabel('y_predicted')
    plt.legend()
    plt.tight_layout()
    if result_file_dir:
        plt.savefig(os.path.join(result_file_dir, 'y_true_vs_y_pred_{}.png'.format(model_name)), dpi=200)
    plt.close()


def compare_exp_and_cell_fraction(merged_file_path, result_dir,
                                  cell_types: list, clustering_ct: list = None,
                                  outlier_file_path=None, predicted_by='DeSide', font_scale=1.5,
                                  signature_score_method: str = 'mean_exp', update_figures=False):
    """
    Comparing the mean expression value (or gene signature score) of marker genes for each cell type
      and the predicted cell fraction
    :param merged_file_path: the file path of merged mean expression value of marker genes and predicted cell fractions,
         sample by cell type, should contain `cancer_type` column to mark corresponding dataset
    :param result_dir: where to save results
    :param cell_types: all cell types used by DeSide
    :param clustering_ct: cell types used for clustering of cancer types
    :param outlier_file_path: the file path of outlier samples selected manually
    :param predicted_by: the name of prediction algorithm, DeSide or Scaden
    :param font_scale: font scaling
    :param signature_score_method:
    :param update_figures: if update figures
    :return:
    """
    check_dir(result_dir)
    # result_dir_scaled = result_dir + '_scaled'
    # check_dir(result_dir_scaled)
    cancer_type2corr_file_path = os.path.join(result_dir, 'cancer_type2corr.csv')
    # print(merged_file_path)
    merged_df = read_df(merged_file_path)
    # merged_df = pd.read_csv(merged_file_path, index_col=0)
    cancer_types = list(merged_df['cancer_type'].unique())
    if 'T Cells' in cell_types and 'T Cells' not in merged_df.columns:
        merged_df['T Cells'] = merged_df.loc[:, ['CD4 T', 'CD8 T']].sum(axis=1)
    if (not os.path.exists(cancer_type2corr_file_path)) or update_figures:
        if outlier_file_path is not None:
            outlier_samples = pd.read_csv(outlier_file_path, index_col=0)
            if outlier_samples.shape[0] > 0:  # remove outliers
                print(f'   {outlier_samples.shape[0]} outlier samples will be removed...')
                merged_df = merged_df.loc[~merged_df.index.isin(outlier_samples.index), :].copy()
        cancer_type2corr = {}
        for cancer_type in cancer_types:
            print('----------------------------------------------------')
            print(f'   Deal with cancer type: {cancer_type}...')
            current_df = merged_df.loc[merged_df['cancer_type'] == cancer_type, :]
            # print(current_df)
            # plot predicted cell fraction against corresponding mean expression value of marker genes
            current_result_dir = os.path.join(result_dir, cancer_type)
            # current_result_dir_scaled = os.path.join(result_dir_scaled, cancer_type)
            if cancer_type not in cancer_type2corr:
                cancer_type2corr[cancer_type] = {}
            for cell_type in cell_types:
                # if cell_type != 'Cancer Cells':
                if signature_score_method == 'mean_exp':
                    method = 'marker_mean'
                    if cell_type in ['B Cells'] and np.any(['max' in i for i in current_df.columns]):
                        method = 'marker_max'
                else:
                    method = signature_score_method
                col_name1 = cell_type + f'_{method}'
                col_name2 = cell_type
                cancer_type2corr[cancer_type][cell_type] = get_corr(current_df[col_name1], current_df[col_name2])
                plot_corr_two_columns(df=current_df, col_name1=col_name1, col_name2=col_name2,
                                      predicted_by=predicted_by, font_scale=font_scale, scale_exp=False,
                                      output_dir=current_result_dir, diagonal=False, cancer_type=cancer_type,
                                      update_figures=update_figures)

            gc.collect()
        cancer_type2corr_df = pd.DataFrame.from_dict(cancer_type2corr, orient='index')
        cancer_type2corr_df.fillna(0, inplace=True)
        cancer_type2corr_df.to_csv(cancer_type2corr_file_path, float_format='%.3f')
    else:
        print(f'   Using previous cancer_type2cor file from: {cancer_type2corr_file_path}.')
        cancer_type2corr_df = pd.read_csv(cancer_type2corr_file_path, index_col=0)
    # sns.set(font_scale=1.5)
    if clustering_ct is not None:
        c_ct = {'clustering_ct': clustering_ct}
        other_ct = [ct for ct in cell_types if ct not in (clustering_ct + ['Cancer Cells'])]
        if len(other_ct) >= 2:
            c_ct = {'clustering_ct': clustering_ct, 'other_ct': other_ct}
        for k, v in c_ct.items():
            plot_clustermap(data=cancer_type2corr_df, columns=v,
                            result_file_path=os.path.join(result_dir, f'cancer_type2corr_{k}.png'))


def plot_clustermap(data: pd.DataFrame, columns: list, result_file_path: str):
    """
    plot cluster map for correlation table or cell fraction table
    """
    # sns.set(font_scale=1.5)
    g = sns.clustermap(data.loc[:, columns], cmap="vlag")
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=40)
    plt.tight_layout()
    plt.savefig(result_file_path, dpi=200)
    # plt.show()
    plt.close('all')


def compare_cell_fraction_across_cancer_type(merged_cell_fraction: pd.DataFrame, result_dir='.', cell_type: str = '',
                                             xlabel: str = 'Cancer Type',
                                             ylabel: str = 'Tumor purity in each sample (CPE)',
                                             outlier_file_path: str = None, cell_type2max: float = 0.0):
    """
    Specific plotting for file cancer_purity.csv, downloaded from Aran, D., Sirota, M. & Butte,
    A. Systematic pan-cancer analysis of tumour purity. Nat Commun 6, 8971 (2015). https://doi.org/10.1038/ncomms9971

    And other predicted cell fractions across all cancer types can be plotted.

    :param merged_cell_fraction: merged cell fraction predicted by DeSide

    :param cell_type: current cell type to plot

    :param result_dir: where to save result

    :param xlabel: x label

    :param ylabel: y label

    :param outlier_file_path:

    :param cell_type2max: max cell fraction to keep when plotting

    :return: None
    """
    x = 'cancer_type'
    check_dir(result_dir)

    if outlier_file_path is not None:
        outlier_samples = pd.read_csv(outlier_file_path, index_col=0)
        if outlier_samples.shape[0] > 0:  # remove outliers
            print(f'   {outlier_samples.shape[0]} outlier samples will be removed...')
            merged_cell_fraction = merged_cell_fraction.loc[~merged_cell_fraction.index.isin(outlier_samples.index),
                                   :].copy()
    # sns.set(font_scale=font_scale)
    plt.figure(figsize=(10, 6))
    # Draw a nested boxplot to show bills by day and time
    # sns.set_color_codes('bright')
    # sample_labels = list(purity['Cancer type'].unique())
    current_cancer_type_frac = merged_cell_fraction.loc[:, [cell_type, 'cancer_type']]
    if cell_type2max > 0:
        current_cancer_type_frac.loc[current_cancer_type_frac[cell_type] > cell_type2max, cell_type] = cell_type2max
    # mean cell fraction of each cancer type
    mean_for_each_cancer_type = current_cancer_type_frac.groupby('cancer_type').mean().sort_values(by=cell_type)
    cancer_type_order = mean_for_each_cancer_type.index.to_list()
    # print(mean_for_each_cancer_type)
    ax = sns.boxplot(x=x, y=cell_type, palette=sns.color_palette("muted"), whis=[0, 100],
                     data=current_cancer_type_frac, showfliers=False, order=cancer_type_order)
    # ax.tick_params(labelsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    # Add in points to show each observation, http://seaborn.pydata.org/examples/horizontal_boxplot.html
    sns.stripplot(x=x, y=cell_type, data=current_cancer_type_frac,
                  size=2, color=".4", linewidth=0, dodge=True, order=cancer_type_order, ax=ax)
    ax.grid(True, axis='y')
    # remove the top and right ticks
    ax.tick_params(axis='x', which='both', top=False)
    ax.tick_params(axis='y', which='both', right=False)
    # sns.despine(offset=10, trim=True, left=True)

    # handles, labels = ax.get_legend_handles_labels()
    # n_half_label = int(len(labels)/2)
    # plt.legend(handles[0:n_half_label], labels[0:n_half_label], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    file_name = f'pred_{cell_type}_across_cancers.png'
    plt.savefig(os.path.join(result_dir, file_name), dpi=200)
    plt.close()


def plot_pca(data: pd.DataFrame, result_fp=None, color_code=None, s=5, figsize=(8, 8),
             color_code2label: dict = None, explained_variance_ratio: np.array = None, label_name='PC',
             show_legend=True, show_xy_labels=True, anno=None, show_core_zone_of_tcga=False):
    """
    plot PCA result of simulated bulk cell dataset
    :param data: PCA table, samples by PCs
    :param result_fp:
    :param color_code: a "np.array" to mark the label of each sample
    :param color_code2label:
    :param explained_variance_ratio: pca_model.explained_variance_ratio_
    :param label_name: label name for x-axis
    :param show_legend:
    :param show_xy_labels:
    :param anno: annotation for x-axis, which layer was used to generate the data
    :param show_core_zone_of_tcga: whether to show the core zone of TCGA data
    :return:
    """
    # sns.set_style('white')
    # sns.set(font_scale=1.5)
    if data.shape[1] >= 3:
        pc_comb = [(0, 1), (1, 2), (0, 2)]
    elif data.shape[1] == 2:
        pc_comb = [(0, 1)]
    else:
        raise IndexError(f'data should have >= 2 columns, but {data.shape[1]} got')
    if color_code is not None:
        data['class'] = color_code
    for pc1, pc2 in pc_comb:
        # plt.figure(figsize=figsize)
        if 'class' in data.columns:
            col_x = f'{label_name}{pc1 + 1}'
            col_y = f'{label_name}{pc2 + 1}'
            g = sns.jointplot(x=col_x, y=col_y, data=data, kind='scatter', hue='class',
                              s=s, space=0, height=figsize[1], alpha=0.5)
            ax = g.ax_joint
            n_tcga, n_non_tcga = 0, 0
            q_lower, q_upper = 0.1, 0.9
            if show_core_zone_of_tcga and 'TCGA' in data['class'].unique():
                coord, n_tcga, n_non_tcga = get_core_zone_of_pca(pca_data=data, col_x=col_x, col_y=col_y,
                                                                 q_lower=q_lower, q_upper=q_upper)
                width = coord['x_upper'] - coord['x_lower']
                height = coord['y_upper'] - coord['y_lower']
                rect = patches.Rectangle((coord['x_lower'], coord['y_lower']), width, height,
                                         fill=False, color='red', linewidth=1,
                                         linestyle='dashed', label='Core Zone of TCGA')
                ax.add_patch(rect)
            if show_xy_labels:
                x_label = col_x
                y_label = col_y
                if (explained_variance_ratio is not None) and (anno is not None):
                    x_label = col_x + f' ({explained_variance_ratio[pc1] * 100:.1f}%, {anno})'
                    y_label = col_y + f' ({explained_variance_ratio[pc2] * 100:.1f}%)'
                elif explained_variance_ratio is not None:
                    x_label = col_x + f' ({explained_variance_ratio[pc1] * 100:.1f}%)'
                    y_label = col_y + f' ({explained_variance_ratio[pc2] * 100:.1f}%)'
                elif anno is not None:
                    x_label = col_x + f' ({anno})'
                if show_core_zone_of_tcga:
                    q_range = f'$q_{{{q_lower * 100:.0f}}}-q_{{{q_upper * 100:.0f}}}$'
                    x_label += f'\nCore Zone ({q_range}): TCGA ({n_tcga}), Non-TCGA ({n_non_tcga})'
                ax.set(xlabel=x_label, ylabel=y_label)
            else:
                ax.set(xlabel=None, ylabel=None)
            # Put the legend out of the figure
            if show_legend:
                # g_legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2 - 0.1 * n_class), ncol=2)
                g_legend = ax.legend(loc='best', ncol=2)
                for _ in g_legend.legendHandles:
                    _.set_linewidth(1)
            else:
                ax.legend([], [], frameon=False)
            # remove the top and right ticks
            g.ax_marg_x.tick_params(axis='x', which='both', top=False)
            g.ax_marg_x.grid(False)
            g.ax_marg_y.tick_params(axis='y', which='both', right=False)
            g.ax_marg_y.grid(False)
        else:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            for i in np.unique(color_code)[::-1]:
                current_part = data.loc[color_code == i, :].copy()
                if color_code2label is None:
                    ax.scatter(current_part.iloc[:, pc1], current_part.iloc[:, pc2], label=i, alpha=.3)
                else:
                    ax.scatter(current_part.iloc[:, pc1], current_part.iloc[:, pc2],
                               label=color_code2label[i], alpha=.3)

            # plt.title(title, fontsize=18)
            if explained_variance_ratio is None:
                plt.xlabel(f'{label_name}{pc1 + 1}')
                plt.ylabel(f'{label_name}{pc2 + 1}')
            else:
                plt.xlabel(f'{label_name}{pc1 + 1} ({(explained_variance_ratio[pc1] * 100): .1f}%)')
                plt.ylabel(f'{label_name}{pc2 + 1} ({(explained_variance_ratio[pc2] * 100): .1f}%)')

            plt.legend()
            plt.tight_layout()
        if result_fp is not None:
            if '.png' in result_fp:
                plt.savefig(result_fp.replace('.png', f'_{label_name}{pc1}_{label_name}{pc2}.png'),
                            bbox_inches='tight', dpi=300)
            if '.pdf' in result_fp:
                plt.savefig(result_fp.replace('.pdf', f'_{label_name}{pc1}_{label_name}{pc2}.pdf'),
                            bbox_inches='tight', dpi=300)


def compare_mean_exp_with_cell_frac_across_algo(cancer_type: str, algo2merged_fp: dict, signature_score_fp: str,
                                                cell_type: str, inx2plot: dict,
                                                outliers_fp: str = None, cancer_type2max_frac=None,
                                                result_file_name_prefix: str = '', result_dir='./figures'):
    """
    compare predicted cell fraction of each cell type with corresponding mean expression value of marker genes in TPM
        one cancer type and one cell type, 2 x 3 plots, 6 different algorithms
    :param cancer_type:
    :param algo2merged_fp: file path of merged cell fractions for each algo
    :param signature_score_fp: file path of mean expression of marker genes for each cell type (all cancer types)
        samples by cell types
    :param cell_type: current cell type (CD8 T/ CD4 T/ B Cells)
    :param outliers_fp: outliers in each cancer type selected manually
    :param inx2plot:
    :param cancer_type2max_frac:
    :param result_file_name_prefix:
    :param result_dir:
    :return:
    """
    check_dir(result_dir)
    mean_exp = pd.read_csv(signature_score_fp, index_col=0)
    if outliers_fp is not None and os.path.exists(outliers_fp):
        outliers = pd.read_csv(outliers_fp, index_col=0)
        mean_exp = mean_exp.loc[~mean_exp.index.isin(outliers.index), :].copy()
    # mean_exp = mean_exp.loc[mean_exp['cancer_type'] == cancer_type, [f'{cell_type}_marker_mean']].copy()

    corr_list = [None] * len(inx2plot)
    max_cell_frac = 0
    fig, ax = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(3.5, 2.5), constrained_layout=True)
    n_sample = 0
    for i in range(2):
        for j in range(3):
            plot_target = inx2plot[(i, j)]
            if plot_target:
                algo, ref = plot_target.split('-')
                merged_result = pd.read_csv(algo2merged_fp[algo], index_col=0)
                if cell_type in merged_result.columns:
                    merged_result = merged_result.loc[(merged_result['reference_dataset'] == ref) &
                                                      (merged_result['cancer_type'] == cancer_type)].copy()
                    merged_result.set_index('sample_id', inplace=True)
                    if algo == 'EPIC' and 'otherCells' in merged_result.columns:
                        merged_result['Cancer Cells'] = merged_result.loc[:, ['Cancer Cells', 'otherCells']].sum(axis=1)
                    df = merged_result.merge(mean_exp, left_index=True, right_index=True)
                    n_sample = df.shape[0]
                    # print(df.shape, algo)
                    col_name1 = f'{cell_type}_marker_mean'  # mean of marker gene expression values
                    col_name2 = cell_type  # predicted cell fraction
                    corr = np.corrcoef(df[col_name1], df[col_name2])
                    if df[cell_type].max() > max_cell_frac:
                        max_cell_frac = df[cell_type].max()

                    if cancer_type2max_frac is not None:
                        ax[i, j].set_ylim([-0.01, cancer_type2max_frac[cancer_type] + 0.02])
                    else:
                        if cell_type in ['CD8 T', 'CD4 T', 'B Cells']:
                            _max_exp = 0.25
                        else:
                            _max_exp = 0.6
                        ax[i, j].set_ylim([-0.01, _max_exp + 0.02])
                        df.loc[df[cell_type] > _max_exp, cell_type] = _max_exp  # set max fraction to 0.25
                    # mae = median_absolute_error(y_true=df[col_name1], y_pred=df[col_name2])
                    ax[i, j].scatter(df[col_name1], df[col_name2], s=1, alpha=0.8)
                    # x_left, x_right = ax[i, j].get_xlim()
                    y_bottom, y_top = ax[i, j].get_ylim()
                    if 'CIBERSORT' in algo:
                        algo = 'C.SORT'
                    elif 'Scaden' in algo:
                        algo = 'Scaden'
                    elif 'EPIC' in algo:
                        algo = 'EPIC'
                    if 'simu_bulk' in ref:
                        ref = 'simu_2ds'
                    ax[i, j].set_xlabel('{} - {}'.format(algo, ref.replace('_ref', '')), fontsize=8)
                    ax[i, j].text(1, y_top * 0.8, 'corr = {:.2f}'.format(corr[0, 1]), fontsize=6)
                    corr_list[i * 3 + j] = round(corr[0, 1], 3)
    fig.supylabel('Predicted cell fraction of {}'.format(f'{cell_type}'))
    fig.supxlabel('mean expression of marker genes in {} (n={})'.format(cancer_type, n_sample))
    # plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{result_file_name_prefix}_in_{cancer_type}.png'), dpi=300)
    print('  Max cell fraction: {}'.format(max_cell_frac))
    return {'corr': corr_list}


def compare_y_y_pred_plot_cpe(y_true: pd.Series, y_pred: pd.Series, inx=tuple(), cancer_type='',
                              show_metrics: bool = False, ax=None):
    """
    Plot y against y_pred to visualize the performance of prediction result

    :param y_true: CPE

    :param y_pred: this file contains the predicted value of y

    :param inx: a tuple of two elements, the first element is the index of y_true, the second element is the index of y_pred

    :param cancer_type: cancer type

    :param show_metrics: show correlation and RMSE

    :param ax: matplotlib axis

    :return: None
    """
    # Use the pyplot interface to change just one subplot...
    plt.sca(ax)

    plt.scatter(y_pred, y_true, s=1, alpha=0.75, rasterized=True)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xticks([0, 1])
    plt.yticks([0, 0.5, 1])
    x_left, x_right = plt.xlim()
    y_bottom, y_top = plt.ylim()
    x_max = x_right
    y_max = y_top
    plt.plot([0, max(x_max, y_max)], [0, max(x_max, y_max)], linestyle='--', color='tab:gray')
    corr = 0
    rmse = 0
    ccc = 0
    if show_metrics:  # show metrics in test set
        corr = get_corr(y_pred, y_true)
        rmse = calculate_rmse(y_true=pd.DataFrame(y_true), y_pred=pd.DataFrame(y_pred))
        ccc = get_ccc(y_pred.values, y_true.values)
        plt.text(0.3 * x_max, 0.2 * y_max, 'corr = {:.2f}'.format(corr), fontsize=6)
        plt.text(0.3 * x_max, 0.1 * y_max, 'RMSE = {:.2f}'.format(rmse), fontsize=6)
        plt.text(0.3 * x_max, 0.0 * y_max, 'CCC = {:.2f}'.format(ccc), fontsize=6)
    if inx:
        plt.ylabel(f'{cancer_type} ({y_true.shape[0]})', fontsize=6)
    # if inx and inx[0] == 8:
    #     plt.xlabel(f'{algo}', fontsize=6)
    # plt.legend()
    return corr, rmse, ccc


def plot_pred_cell_prop_with_cpe(cpe_file_path, pred_cell_prop_file_path, result_dir, save_metrics: bool = True):
    all_cancer_types = sorted([i for i in cancer_types if i != 'PAAD'])
    fig, axes = plt.subplots(6, 3, sharex='all', sharey='all', figsize=(5, 6))
    pred_cell_prop = pd.read_csv(pred_cell_prop_file_path, index_col=0)
    cpe = read_cancer_purity(cpe_file_path, sample_names=pred_cell_prop.index)
    pred_cell_prop = pred_cell_prop.merge(cpe['CPE'], left_index=True, right_index=True)
    metrics_value = {}
    for j in range(3):
        for i in range(6):
            current_cancer_type = all_cancer_types[i + j * 6]
            current_data = pred_cell_prop.loc[pred_cell_prop['cancer_type'] == current_cancer_type, :]
            corr, rmse, ccc = compare_y_y_pred_plot_cpe(y_pred=current_data['Cancer Cells'], y_true=current_data['CPE'],
                                                        show_metrics=True, ax=axes[i, j], cancer_type=current_cancer_type,
                                                        inx=(i, j))
            metrics_value[current_cancer_type] = {'corr': corr, 'rmse': rmse, 'ccc': ccc}

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Predicted cancer cell proportions by DeSide', labelpad=5)
    plt.ylabel("CPE", labelpad=15)

    plt.tight_layout(h_pad=0.02, w_pad=0.15)
    plt.savefig(os.path.join(result_dir, 'pred_cancer_cell_prop_vs_cpe-deside.png'), dpi=300)
    if save_metrics:
        metrics_value_df = pd.DataFrame.from_dict(metrics_value, orient='index')
        metrics_value_df.to_csv(os.path.join(result_dir, 'pred_cancer_cell_prop_vs_cpe-deside-metrics.csv'))
