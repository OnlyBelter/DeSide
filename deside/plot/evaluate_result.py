import os
import pandas as pd
import numpy as np
from typing import Union
# from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import median_absolute_error
from ..utility import (print_df, cal_relative_error, calculate_rmse, check_dir,
                       get_corr, read_xy, read_df, get_inx2cell_type, log2_transform, set_fig_style)
# from ..utility.read_file import ReadExp
from .plot_nn import plot_corr_two_columns
# import matplotlib
# import importlib
import matplotlib.pyplot as plt
import seaborn as sns
import gc

set_fig_style()
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
             fig_size=(8, 8), group_by: str = None, show_reg_line: bool = False, s=6, order=1):
        """
        :param show_columns: a list of column names in both x and y, could be multiple common columns
            or a dict {'x': '', 'y': ’‘}, only one column allowed
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
                plt.scatter(self.x[~inx], self.y[~inx], s=6, label='others')
                plt.scatter(self.x[inx], self.y[inx], s=10, label=group_by, marker='x')
            else:
                plt.scatter(self.x, self.y, s=s, label=show_columns['x'])  # only 1 vs 1 column
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
            plt.plot([0, _], [0, _], linestyle='--', color='tab:gray')
        if show_corr:  # show metrics in test set
            corr = get_corr(all_x, all_y)
            plt.text(x_right * 0.70, y_top * 0.16, 'corr = {:.3f}'.format(corr))
        if show_mae:
            mae = median_absolute_error(y_true=all_x, y_pred=all_y)
            plt.text(x_right * 0.70, y_top * 0.10, 'MAE = {:.3f}'.format(mae))
        if show_rmse and (not show_mae):
            rmse = calculate_rmse(y_true=pd.DataFrame(all_x), y_pred=pd.DataFrame(all_y))
            plt.text(x_right * 0.70, y_top * 0.10, 'RMSE = {:.3f}'.format(rmse))
        if show_rmse and show_mae:
            rmse = calculate_rmse(y_true=pd.DataFrame(all_x), y_pred=pd.DataFrame(all_y))
            plt.text(x_right * 0.70, y_top * 0.04, 'RMSE = {:.3f}'.format(rmse))
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
        plt.legend()
        plt.tight_layout()
        if result_file_dir:
            plt.savefig(os.path.join(result_file_dir,
                                     'x_vs_y_{}.png'.format(self.postfix)), dpi=200)
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
        sns.regplot(x='y_pred', y='y_true', data=xy, ax=ax, x_estimator=np.mean, order=order)
        # p_value = res.pvalues[x_col]
        # r2 = res.rsquared
        # print(f'p_value: {p_value}', f'R^2: {r2}')
        if alpha > 0:
            if order == 2:
                plt.plot(x_lin, y_reg_line, c='r', label=f'$y= {beta2: .2f}x^2 + {beta1: .2f}x + {alpha: .2f}$')
            else:
                plt.plot(x_lin, y_reg_line, c='r', label=f'$y={beta1: .2f}x + {alpha: .2f}$')
        else:
            if order == 2:
                plt.plot(x_lin, y_reg_line, c='r', label=f'$y= {beta2: .2f}x^2 + {beta1: .2f}x - {abs(alpha): .2f}$')
            else:
                plt.plot(x_lin, y_reg_line, c='r', label=f'$y={beta1: .2f}x - {abs(alpha): .2f}$')
        plt.plot(x_lin, y_lower_bound, c='b', label=f'{100 - alpha_ci * 100}% CI')
        plt.plot(x_lin, y_upper_bound, c='b')


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


def plot_error(y_true, y_pred_file_path, show_columns=None, error_type='relative_error',
               result_file_dir=None, annotation=(('MAE', 0),), y_label=None, model_name='average'):
    """
    plot y against y_pred to visualize the performance of prediction scaden
    :param y_true: str | pd.DataFrame
        this file contains the ground truth of cell fractions when it was simulated
    :param y_pred_file_path: str
        this file contains the predicted value of y
    :param show_columns: list
        this list contains the name of columns which want to plot in figure
    :param error_type: relative_error or absolute_error
    :param result_file_dir: str
    :param annotation: tuple of tuples, MAE means mean absolute error
        annotations that need to show in figure, ((anno_name, value), (anno_name, value), ...)
    :param y_label:
    :param model_name: only for naming files
    :return:
    """
    check_dir(result_file_dir)
    result_file_path = os.path.join(result_file_dir, 'y_true_vs_{}_{}.png'.format(error_type, model_name))
    if not os.path.exists(result_file_path):
        y_true = read_xy(y_true, xy='cell_frac')
        y_pred = pd.read_csv(y_pred_file_path, index_col=0)
        if '1-others' in show_columns:
            if 'Cancer Cells' in y_true.columns:
                y_true['1-others'] = y_true['Cancer Cells']
            else:
                y_true['1-others'] = 0
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
        errors = cal_relative_error(y_true=y_true, y_pred=y_pred, max_error=1)
        if error_type == 'absolute_error':
            errors = y_pred - y_true
        plt.figure(figsize=(8, 6))
        for i, col in enumerate(show_columns):
            _x = y_true.loc[:, col]
            _y = errors.loc[:, col]
            plt.scatter(_x, _y, label=col, s=6, alpha=1 - 0.02 * i)
        if annotation:
            x_left, x_right = plt.xlim()
            y_bottom, y_top = plt.ylim()
            for k, v in annotation:
                if k == 'MAE':
                    v = median_absolute_error(y_true=y_true, y_pred=y_pred)
                plt.text(x_left + 0.05, y_top * 0.8, '{}: {:.3f}'.format(k, v))
        # plt.plot([0, 1], [0, 1], linestyle='--', color='tab:gray')
        plt.xlabel('y_true')
        if y_label:
            plt.ylabel(y_label)
        else:
            plt.ylabel('y_predicted')
        # plt.legend()
        plt.tight_layout()
        plt.savefig(result_file_path, dpi=200)
        plt.close()
    else:
        print(f'Using previous result: {result_file_path}')


def plot_min_rmse(decon_performance, file_path):
    """
    plot the relation between minimal rmse and the number of sample
    :param decon_performance: a dataFrame
         the result of improved_cibersortx algo
    :param file_path: the path to save result
    :return:
    """
    sample_groups = decon_performance.groupby(['ref_sample_name'])
    sample2min_rmse = {}
    best_model_info = decon_performance.loc[decon_performance['best_model'] == 1, :]
    for s, g in sample_groups:
        sample_name = s
        min_rmse = g['rmse'].min()
        n_sample_with_min_rmse = int(g.loc[g['rmse'] == min_rmse, 'n_sample'])
        sample2min_rmse[sample_name] = {'min_rmse': min_rmse, 'n_sample': n_sample_with_min_rmse}
    sample2min_rmse_df = pd.DataFrame.from_dict(sample2min_rmse, orient='index')
    plt.figure(figsize=(8, 6))
    plt.scatter(sample2min_rmse_df['min_rmse'], sample2min_rmse_df['n_sample'], label='Mini RMSE')
    plt.scatter(best_model_info['rmse'], best_model_info['n_sample'], marker='+', label='Best scaden RMSE')
    plt.xlabel('RMSE')
    plt.ylabel('n_sample')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path, dpi=200)


def plot_emt_score(emt_score_file_path: str, sample2label_file_path: str, result_dir: str, bulk_exp_file_path: str):
    """

    :param emt_score_file_path: .gct file format
        get this file from the scaden of ssGSEA in GenePattern
    :param sample2label_file_path:
    :param result_dir:
    :param bulk_exp_file_path:
    :return:
    """
    emt_score = pd.read_csv(emt_score_file_path, skiprows=[0, 1], sep='\t', index_col=0)
    emt_score.drop(columns=['Description'], inplace=True)
    emt_score_t = emt_score.T  # sample by gene set
    # emt_score_t = (emt_score_t - emt_score_t.mean()) / emt_score_t.std()
    print_df(emt_score_t)
    sample2label = pd.read_csv(sample2label_file_path, index_col=0)
    print_df(sample2label)
    # label2samples = sample2label.groupby('labels')

    bulk_exp = pd.read_csv(bulk_exp_file_path, index_col=0, sep='\t')
    bulk_exp = bulk_exp.loc[:, bulk_exp.columns.isin(emt_score_t.index)].copy()
    bulk_exp_sorted = bulk_exp.sort_values(by=['CD8A'], axis=1)
    emt_score_sorted = emt_score_t.loc[bulk_exp_sorted.columns, :]
    print_df(bulk_exp_sorted)
    print_df(emt_score_sorted)

    emt_gene_set = {'GenomeRes': ('GenomeRes_E', 'GenomeRes_M'), 'PNAS': ('PNAS_E', 'PNAS_M'),
                    'NG': ('NG_E', 'NG_M'), 'Integrated': ('AtLeast2_E_in_3_geneset', 'AtLeast2_M_in_4_geneset'),
                    'MIX': ('NG_E', 'MSigDB_EMT')}
    for i, geneset in emt_gene_set.items():
        plt.figure(figsize=(8, 6))
        # current_emt_score = emt_score_sorted.loc[:, geneset]
        # for label, current_g in label2samples:
        less_than_samples = bulk_exp_sorted.loc['CD8A'] <= 20
        emt_score_sorted = emt_score_sorted.loc[less_than_samples, :]
        plt.scatter(x=bulk_exp_sorted.loc['CD8A', less_than_samples],
                    y=emt_score_sorted.loc[:, geneset[1]] / emt_score_sorted.loc[:, geneset[0]])
        plt.xlabel('CD8A expression')
        plt.ylabel('{} / {}'.format(geneset[1], geneset[0]))
        # plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'emt_score_{}_less_than20.png'.format(i)), dpi=200)
        plt.close()


def plot_emt_score_from_gsva(score_file_path, bulk_exp_file_path, result_file_path, sample2label_file_path=None):
    """
    GSVA: Hänzelmann, S., Castelo, R. & Guinney, J. GSVA: gene set variation analysis for microarray and RNA-Seq data.
        BMC Bioinformatics 14, 7 (2013). https://doi.org/10.1186/1471-2105-14-7
    :param score_file_path:
    :param bulk_exp_file_path:
    :param result_file_path:
    :param sample2label_file_path:
    :return:
    """
    emt_score = pd.read_csv(score_file_path, index_col=0)
    emt_score.rename(columns={'V1': 'EMT_score'}, inplace=True)
    emt_score.index = [i.replace('.', '-') for i in emt_score.index]
    bulk_exp = pd.read_csv(bulk_exp_file_path, index_col=0, sep='\t')
    bulk_exp_sorted = bulk_exp.sort_values(by=['CD8A'], axis=1)
    emt_score_sorted = emt_score.loc[bulk_exp_sorted.columns, :]
    if sample2label_file_path is not None:
        sample2label = pd.read_csv(sample2label_file_path, index_col=0)
        sample2label_sorted = sample2label.loc[bulk_exp_sorted.columns, :]
        sample2label_sorted.sort_values(by=['labels'], ascending=False, inplace=True)
        emt_score_sorted = emt_score_sorted.loc[sample2label_sorted.index, :]
        bulk_exp_sorted = bulk_exp_sorted.loc[:, sample2label_sorted.index]
    plt.figure(figsize=(8, 6))
    plt.scatter(x=bulk_exp_sorted.loc['CD8A'], y=emt_score_sorted['EMT_score'])
    plt.xlabel('CD8A expression')
    plt.ylabel('EMT_score')
    # plt.legend()
    plt.tight_layout()
    if result_file_path:
        plt.savefig(result_file_path, dpi=200)
    plt.show()


def compare_cancer_purity(purity: pd.DataFrame, result_dir='.',
                          xlabel: str = 'Cancer Type', ylabel: str = 'Tumor purity in each sample (CPE)',
                          file_name: str = 'compare_cancer_purity_in_each_cancer.png'):
    """
    Specific plotting for file cancer_purity.csv, downloaded from Aran, D., Sirota, M. & Butte,
    A. Systematic pan-cancer analysis of tumour purity. Nat Commun 6, 8971 (2015). https://doi.org/10.1038/ncomms9971

    :param purity: sample by purity score (CPE), must contain group label ('labels') for each sample

    :param result_dir: where to save result

    :param xlabel: x label

    :param ylabel: y label

    :param file_name: file name

    :return: None
    """

    plt.figure(figsize=(10, 6))
    # Draw a nested boxplot to show bills by day and time
    # sns.set_color_codes('bright')
    # sample_labels = list(purity['Cancer type'].unique())
    ax = sns.boxplot(x="Cancer type", y="CPE", palette=sns.color_palette("muted"), data=purity, whis=[0, 1])
    ax.tick_params(labelsize=10)
    # Add in points to show each observation, http://seaborn.pydata.org/examples/horizontal_boxplot.html
    sns.stripplot(x="Cancer type", y="CPE", data=purity,
                  size=3, color=".3", linewidth=0, dodge=True)
    sns.despine(offset=10, trim=True, left=True)
    # handles, labels = ax.get_legend_handles_labels()
    # n_half_label = int(len(labels)/2)
    # plt.legend(handles[0:n_half_label], labels[0:n_half_label], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, file_name), dpi=200)
    plt.close()


def compare_y_y_pred_decon(sample_name: str, purified_gep_file_path: str, result_file_dir=None,
                           show_corr=True, x_label=None, y_label=None, font_scale=1.0,
                           log_transform=False, z_score_threshold=8,
                           rsd_z_score_threshold=0.9, all_sample_error=None):
    """
    Plot y against y_pred to visualize the performance of each model

    :param sample_name: sample name

    :param purified_gep_file_path: str
        this file contains the true value of y

    :param result_file_dir: str

    :param show_corr: if show correlation

    :param x_label:

    :param y_label:

    :param font_scale:

    :param log_transform:

    :param z_score_threshold: the threshold of z_score of absolute error, remove if |z_score| >= this value

    :param rsd_z_score_threshold: the threshold of relative error, remove if |relative error| >= this value

    :param all_sample_error: file all_sample_error_fp

    :return:
    """
    gep_i = pd.read_csv(purified_gep_file_path, index_col=0)
    # all_sample_error = pd.read_csv(all_sample_error_fp, index_col=0)
    gep = gep_i.copy()
    gep['error_code'] = all_sample_error['error_code']
    # y_y_pred = gep.loc[:, ['y', 'y_pred']].copy()
    if log_transform:
        gep['y'] = np.log2(gep['y'] + 1)
        gep['y_pred'] = np.log2(gep['y_pred'] + 1)
    # y_y_pred['error'] = gep['y_pred'] - gep['y']
    # y_y_pred['error_z_score'] = stats.zscore(y_y_pred['error'])
    # https://mathworld.wolfram.com/RelativeError.html
    # y_y_pred['relative_error'] = y_y_pred['error'] / (gep['y'] + 0.01)
    # y_y_pred['label'] = 2
    # y_y_pred.loc[gep['absolute_error_z_score'].abs() >= z_score_threshold, 'label'] = 3
    # y_y_pred.loc[gep['relative_error'].abs() >= relative_error_threshold, 'label'] += 5
    # y = y_y_pred.loc[:, ['y']]
    # y_pred = y_y_pred.loc[:, ['y_pred']]
    max_y = max(gep.loc[:, ['y', 'y_pred']].max())

    # sns.set(font_scale=font_scale)
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Show Chinese characters
    plt.figure(figsize=(8, 6))
    for i, group in gep.groupby('error_code'):
        if i == 2:
            label = f'|z-score of mean abs error| < {z_score_threshold} & |z-score of RSD| < {rsd_z_score_threshold}'
        elif i == 3:
            label = f'|z-score of mean abs error| >= {z_score_threshold} & |z-score of RSD| < {rsd_z_score_threshold}'
        elif i == 7:
            label = f'|z-score of mean abs error| < {z_score_threshold} & |z-score of RSD| >= {rsd_z_score_threshold}'
        else:
            label = f'|z-score of mean abs error| >= {z_score_threshold} & |z-score of RSD| >= {rsd_z_score_threshold}'
        plt.scatter(group['y'], group['y_pred'], s=6, label=label)

    plt.plot([0, max_y], [0, max_y], linestyle='--', color='tab:gray')
    plt.xlabel('y_true ({})'.format(sample_name))
    plt.ylabel('y_predicted')
    if x_label:
        plt.xlabel(x_label.format(sample_name))
    if y_label:
        plt.ylabel(y_label)

    if show_corr:
        x_left, x_right = plt.xlim()
        y_bottom, y_top = plt.ylim()
        corr = gep['y'].corr(gep['y_pred'])
        rmse = calculate_rmse(y_true=gep_i[['y']], y_pred=gep_i[['y_pred']])
        plt.text(x_left + 1.5, y_top * 0.75, 'corr = {:.3f}'.format(corr))
        plt.text(x_left + 1.5, y_top * 0.70, 'RMSE = {:.3f}'.format(rmse))
        # plt.title('{}, corr: {:.4f}, RMSE: {:.4f}'.format(sample_name, corr, rmse))
        # plt.text(x_left * 1.5, y_top * 0.7, ''.format(rmse))
    plt.legend()
    plt.tight_layout()
    if result_file_dir:
        plt.savefig(os.path.join(result_file_dir, 'y_true_vs_y_pred_{}.png'.format(sample_name)), dpi=200)
    plt.close()


def y_y_pred_error_hist_decon(sample_name, purified_gep_file_path, result_file_dir=None,
                              y_label=None, font_scale=1.0):
    """
    plot y against y_pred to visualize the performance of each model
    :param sample_name:
    :param purified_gep_file_path: str
        this file contains the true value of y
    :param result_file_dir: str
    :param y_label:
    :param font_scale:
    :return:
    """
    gep = pd.read_csv(purified_gep_file_path, index_col=0)

    # y = gep.loc[:, ['y']]
    # y_pred = gep.loc[:, ['y_pred']]

    # sns.set(font_scale=font_scale)
    plt.figure(figsize=(8, 6))
    # error_z_score = stats.zscore(gep['y'] - gep['y_pred'])
    plt.hist(gep['y'] - gep['y_pred'])

    # plt.plot([0, max_y], [0, max_y], linestyle='--', color='tab:gray')
    plt.xlabel('y_true - y_pred ({})'.format(sample_name))
    if y_label:
        plt.ylabel(y_label)
    else:
        plt.ylabel('number of samples')

    plt.tight_layout()
    if result_file_dir:
        plt.savefig(os.path.join(result_file_dir, 'y_true_y_pred_error_hist_{}.png'.format(sample_name)), dpi=200)
    plt.close()


def compare_cancer_cell_with_cpe(cancer_type: str, algo2merged_fp, cancer_purity_fp, inx2plot: dict,
                                 result_file_name_prefix, result_dir='./figures'):
    """
    comparing predicted cell fraction of cancer cells with CPE value for a specific cancer type
    :param cancer_type:
    :param algo2merged_fp:
    :param cancer_purity_fp:
    :param inx2plot:
    :param result_file_name_prefix:
    :param result_dir:
    :return:
    """
    check_dir(result_dir)
    cancer_purity = pd.read_csv(cancer_purity_fp, index_col=0)
    cancer_purity = cancer_purity.loc[(cancer_purity['Cancer type'] == cancer_type) &
                                      (cancer_purity['CPE'].notna()), :]
    corr_list = [None] * len(inx2plot)
    mae_list = [None] * len(inx2plot)
    n_sample = 0
    m, n = 2, 3
    if len(inx2plot) == 4:
        m, n = 2, 2
    if cancer_purity.shape[0] > 0:
        fig, ax = plt.subplots(m, n, sharex='col', sharey='row', figsize=(5*n, 5*m))
        for i in range(m):
            for j in range(n):
                plot_target = inx2plot[(i, j)]
                if plot_target:
                    algo, ref = plot_target.split('-')
                    merged_result = pd.read_csv(algo2merged_fp[algo], index_col=0)
                    merged_result = merged_result.loc[(merged_result['reference_dataset'] == ref) &
                                                      (merged_result['cancer_type'] == cancer_type)].copy()
                    merged_result.set_index('sample_id', inplace=True)
                    if algo == 'EPIC':
                        merged_result['Cancer Cells'] = merged_result.loc[:, ['Cancer Cells', 'otherCells']].sum(axis=1)
                    df = merged_result.merge(cancer_purity, left_index=True, right_index=True)
                    n_sample = df.shape[0]
                    # print(df.shape, algo)
                    col_name1 = 'Cancer Cells'
                    # if algo == 'DeSide':
                    #     col_name1 = '1-others'
                        # print(merged_result.head())
                    col_name2 = 'CPE'
                    corr = np.corrcoef(df[col_name1], df[col_name2])
                    mae = median_absolute_error(y_true=df[col_name1], y_pred=df[col_name2])
                    ax[i, j].scatter(df[col_name1], df[col_name2])
                    ax[i, j].set_xlabel('{} with {}'.format(algo, ref))
                    # plt.ylabel('{}预测值 (样本数={})'.format(predicted_by, df.shape[0]))
                    ax[i, j].text(0.45, 0.15, 'corr = {:.3f}'.format(corr[0, 1]))
                    ax[i, j].text(0.45, 0.05, '$MAE$ = {:.3f}'.format(mae))
                    ax[i, j].plot([0, 1], [0, 1], linestyle='--', color='tab:gray')
                    corr_list[i * n + j] = round(corr[0, 1], 3)
                    mae_list[i * n + j] = round(mae, 3)
        fig.supxlabel('Predicted cell fraction of {} in {} (n={})'.format('Cancer Cells', cancer_type, n_sample))
        fig.supylabel('CPE')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, '{}_{}.png'.format(result_file_name_prefix, cancer_type)), dpi=200)
    return {'corr': corr_list, 'mae': mae_list}


def compare_cd8t_with_cd8a(cancer_type, algo2merged_fp, tpm_fp, gene_list, inx2plot, cancer_type2max_frac=None,
                           result_file_name_prefix='', result_dir='./figures'):
    """
    compare predicted cell fraction of CD8+ T cells with CD8A expression value in TPM
    :param cancer_type:
    :param algo2merged_fp:
    :param tpm_fp:
    :param gene_list:
    :param inx2plot:
    :param cancer_type2max_frac:
    :param result_file_name_prefix:
    :param result_dir:
    :return:
    """
    check_dir(result_dir)
    tpm = pd.read_csv(tpm_fp, index_col=0).T  # convert to sample by gene
    tpm = tpm.loc[:, gene_list].copy()

    corr_list = [None] * len(inx2plot)
    if len(gene_list) > 0:
        max_cell_frac = 0
        fig, ax = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(15, 10))
        for i in range(2):
            for j in range(3):
                plot_target = inx2plot[(i, j)]
                if plot_target:
                    algo, ref = plot_target.split('-')
                    merged_result = pd.read_csv(algo2merged_fp[algo], index_col=0)
                    merged_result = merged_result.loc[(merged_result['reference_dataset'] == ref) &
                                                      (merged_result['cancer_type'] == cancer_type)].copy()
                    merged_result.set_index('sample_id', inplace=True)
                    if algo == 'EPIC':
                        merged_result['Cancer Cells'] = merged_result.loc[:, ['Cancer Cells', 'otherCells']].sum(axis=1)
                    df = merged_result.merge(tpm, left_index=True, right_index=True)
                    # print(df.shape, algo)
                    col_name1 = 'CD8A'  # marker gene expression
                    col_name2 = 'CD8 T'  # cell fraction
                    corr = np.corrcoef(df[col_name1], df[col_name2])
                    if df['CD8 T'].max() > max_cell_frac:
                        max_cell_frac = df['CD8 T'].max()

                    if cancer_type2max_frac:
                        ax[i, j].set_ylim([-0.01, cancer_type2max_frac[cancer_type] + 0.02])
                    else:
                        ax[i, j].set_ylim([-0.01, 0.32])
                        df.loc[df['CD8 T'] > 0.3, 'CD8 T'] = 0.3
                    # mae = median_absolute_error(y_true=df[col_name1], y_pred=df[col_name2])
                    ax[i, j].scatter(df[col_name1], df[col_name2])
                    # x_left, x_right = ax[i, j].get_xlim()
                    y_bottom, y_top = ax[i, j].get_ylim()
                    ax[i, j].set_xlabel('{} with {} (n={})'.format(algo, ref, df.shape[0]))
                    ax[i, j].text(1, y_top * 0.9, 'corr = {:.3f}'.format(corr[0, 1]))
                    corr_list[i * 3 + j] = round(corr[0, 1], 3)
        fig.supylabel('Predicted cell fraction of {}'.format('CD8+ T cells'))
        fig.supxlabel('TPM of CD8A in {}'.format(cancer_type))
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f'{result_file_name_prefix}_{cancer_type}.png'), dpi=200)
        print('  Max cell fraction: {}'.format(max_cell_frac))
    return {'corr': corr_list}


def deside_compare_cc_1_others(cancer_type, algo2merged_fp, cancer_purity_fp, inx2plot,
                               result_file_name_prefix, result_dir='./figures'):
    """
    compare predicted cell fraction of cancer cells and 1-others
    :param cancer_type:
    :param algo2merged_fp:
    :param cancer_purity_fp:
    :param inx2plot:
    :param result_file_name_prefix:
    :param result_dir:
    :return:
    """
    check_dir(result_dir)
    cancer_purity = pd.read_csv(cancer_purity_fp, index_col=0)
    cancer_purity = cancer_purity.loc[(cancer_purity['Cancer type'] == cancer_type) &
                                      (cancer_purity['CPE'].notna()), :]
    corr_list = [None] * 2
    mae_list = [None] * 2
    if cancer_purity.shape[0] > 0:
        fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10, 5))
        algo = ''
        df = pd.DataFrame()
        ref = ''
        for j in range(2):
            plot_target = inx2plot[j]
            if plot_target:
                algo, ref, col = plot_target.split('-')
                merged_result = pd.read_csv(algo2merged_fp[algo], index_col=0)
                merged_result = merged_result.loc[(merged_result['reference_dataset'] == ref) &
                                                  (merged_result['cancer_type'] == cancer_type)].copy()
                merged_result.set_index('sample_id', inplace=True)
                df = merged_result.merge(cancer_purity, left_index=True, right_index=True)
                if col == 'cancer_cells':
                    col_name1 = 'Cancer Cells'
                else:
                    col_name1 = '1-others'
                col_name2 = 'CPE'
                corr = np.corrcoef(df[col_name1], df[col_name2])
                mae = median_absolute_error(y_true=df[col_name1], y_pred=df[col_name2])
                ax[j].scatter(df[col_name1], df[col_name2])
                ax[j].set_xlabel('{}'.format(col_name1))
                ax[j].text(0.5, 0.15, 'corr = {:.3f}'.format(corr[0, 1]))
                ax[j].text(0.5, 0.05, '$MAE$ = {:.3f}'.format(mae))
                ax[j].plot([0, 1], [0, 1], linestyle='--', color='tab:gray')
                corr_list[j] = round(corr[0, 1], 3)
                mae_list[j] = round(mae, 3)
        fig.supxlabel('Predicted cell fraction by {} with {} (n={}) in {}'.format(algo, ref, df.shape[0], cancer_type))
        fig.supylabel('CPE')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, result_file_name_prefix + '_{}.png'.format(cancer_type)), dpi=200)
    return {'corr': corr_list, 'mae': mae_list}


def plot_line_across_cancers(inx2plot, values_df, result_file_path,
                             xlabel='', ylabel='', mark_point=0, comparing_type=''):
    """

    :param inx2plot:
    :param values_df:
    :param result_file_path:
    :param xlabel:
    :param ylabel:
    :param mark_point: horizontal line if not 0
    :param comparing_type: diff_algo, diff_dataset
    :return:
    """
    plt.figure(figsize=(12, 6))
    n_cancer_type = values_df.shape[0]
    for i in list(inx2plot.values()):
        line_width = 1.5
        marker = ''
        if i:
            if comparing_type == 'diff_algo':
                if 'DeSide' in i:
                    # line_width = 2.5
                    marker = '*'
            if comparing_type == 'diff_dataset':
                if 'Mixed' in i:
                    marker = '*'
                if 'HNSC' in i:
                    marker = '+'
                if 'LUAD' in i:
                    marker = 'o'

            plt.plot(range(n_cancer_type), values_df.loc[:, i], label=i, linewidth=line_width,
                     marker=marker, markersize=10)
    if mark_point != 0:
        xmin, xmax = plt.xlim()
        plt.hlines(y=mark_point, xmin=xmin, xmax=xmax, linestyles='dashed', colors='tab:gray')
    plt.xticks(range(n_cancer_type), values_df.index.to_list())
    # plt.xlabel('Cancer type')
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_file_path, dpi=200)


def plot_cell_fraction_hist(cell_fraction: pd.DataFrame, sampling_method='pure random',
                            result_dir='.', dataset=None, density=False, bins=20):
    """
    plot cell fraction hist for each cell type for comparing different sampling methods
    :param cell_fraction: sample by cell type
    :param sampling_method: pure random / gradient
    :param result_dir:
    :param dataset
    :param density: if plot density
    :param bins:
    :return:
    """
    file_path = os.path.join(result_dir, f'cell_fraction_sampled_by_{sampling_method}_hist.png')
    if dataset is not None:
        file_path = os.path.join(result_dir, f'cell_fraction_sampled_by_{sampling_method}_hist_{dataset}.png')
    if not os.path.exists(file_path):
        check_dir(result_dir)
        plt.figure(figsize=(8, 6))
        for ct in cell_fraction.columns:
            plt.hist(cell_fraction[ct], label=ct, alpha=1, bins=bins, histtype='step', density=density)
        plt.legend()
        plt.xlabel(f'Cell fraction by {sampling_method} sampling')
        if density:
            plt.ylabel('Density')
        else:
            plt.ylabel('Number of samples')
        plt.tight_layout()
        plt.savefig(file_path, dpi=200)
    else:
        print(f'File existed at {file_path}')


def plot_n_cell_type_hist(cell_frac, sampling_method, result_dir, dataset=None):
    """
    plot the hist of the number of cell types that the cell fraction is not zero in simulated dataset,
    - some samples may contains all cell types, but some samples may only contains 1 or 2 cell types (others are 0)
    """
    file_path = os.path.join(result_dir, f'n_cell_type_sampled_by_{sampling_method}_hist.png')
    if dataset is not None:
        file_path = os.path.join(result_dir, f'n_cell_type_sampled_by_{sampling_method}_hist_{dataset}.png')
    if not os.path.exists(file_path):
        check_dir(result_dir)
        plt.figure(figsize=(8, 6))
        plt.hist(np.sum(cell_frac != 0, axis=1))
        plt.xlabel(f'The number of cell types in each sample from {sampling_method} sampling')
        plt.ylabel('The number of samples')
        plt.tight_layout()
        plt.savefig(file_path, dpi=200)
    else:
        print(f'File existed at {file_path}')


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
             show_legend=True, show_xy_labels=True, anno=None):
    """
    plot PCA result of simulated bulk cell dataset
    :param data: PCA table, samples by PCs
    :param result_fp:
    :param color_code: an np.array to mark the label of each sample
    :param color_code2label:
    :param explained_variance_ratio: pca_model.explained_variance_ratio_
    :param label_name: label name for x axis
    :param show_legend:
    :param show_xy_labels:
    :param anno: annotation for x axis
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
    for pc1, pc2 in pc_comb:
        # plt.figure(figsize=figsize)
        if 'class' in data.columns:
            g = sns.jointplot(x=f'{label_name}{pc1 + 1}', y=f'{label_name}{pc2 + 1}',
                              data=data, kind='scatter', hue='class', s=s, space=0, height=figsize[1], alpha=0.5)
            ax = g.ax_joint
            if show_xy_labels:
                if (explained_variance_ratio is not None) and (anno is not None):
                    x_label = f'{label_name}{pc1 + 1} ({explained_variance_ratio[pc1] * 100:.1f}%, {anno})'
                    y_label = f'{label_name}{pc2 + 1} ({explained_variance_ratio[pc2] * 100:.1f}%)'
                elif explained_variance_ratio is not None:
                    x_label = f'{label_name}{pc1 + 1} ({explained_variance_ratio[pc1] * 100:.1f}%)'
                    y_label = f'{label_name}{pc2 + 1} ({explained_variance_ratio[pc2] * 100:.1f}%)'
                elif anno is not None:
                    x_label = f'{label_name}{pc1 + 1} ({anno})'
                    y_label = f'{label_name}{pc2 + 1}'
                else:
                    x_label = f'{label_name}{pc1 + 1}'
                    y_label = f'{label_name}{pc2 + 1}'
                ax.set(xlabel=x_label, ylabel=y_label)
            else:
                ax.set(xlabel=None, ylabel=None)
            # Put the legend out of the figure
            if show_legend:
                # g_legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2 - 0.1 * n_class), ncol=2)
                g_legend = ax.legend(loc='best', ncol=2)
                for _ in g_legend.legendHandles:
                    _.set_linewidth(2)
            else:
                ax.legend([], [], frameon=False)
            # remove the top and right ticks
            g.ax_marg_x.tick_params(axis='x', which='both', top=False)
            g.ax_marg_x.grid(False)
            g.ax_marg_y.tick_params(axis='y', which='both', right=False)
            g.ax_marg_y.grid(False)
        else:
            fig = plt.figure(figsize=(12, 8))
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
                            bbox_inches='tight', dpi=200)
            if '.pdf' in result_fp:
                plt.savefig(result_fp.replace('.pdf', f'_{label_name}{pc1}_{label_name}{pc2}.pdf'),
                            bbox_inches='tight', dpi=200)


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
    if outliers_fp is not None:
        outliers = pd.read_csv(outliers_fp, index_col=0)
        mean_exp = mean_exp.loc[~mean_exp.index.isin(outliers.index), :].copy()
    # mean_exp = mean_exp.loc[mean_exp['cancer_type'] == cancer_type, [f'{cell_type}_marker_mean']].copy()

    corr_list = [None] * len(inx2plot)
    max_cell_frac = 0
    fig, ax = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(15, 10))
    n_sample = 0
    for i in range(2):
        for j in range(3):
            plot_target = inx2plot[(i, j)]
            if plot_target:
                algo, ref = plot_target.split('-')
                merged_result = pd.read_csv(algo2merged_fp[algo], index_col=0)
                merged_result = merged_result.loc[(merged_result['reference_dataset'] == ref) &
                                                  (merged_result['cancer_type'] == cancer_type)].copy()
                merged_result.set_index('sample_id', inplace=True)
                if algo == 'EPIC':
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
                    _max_exp = 0.25
                    ax[i, j].set_ylim([-0.01, _max_exp + 0.02])
                    df.loc[df[cell_type] > _max_exp, cell_type] = _max_exp  # set max fraction to 0.25
                # mae = median_absolute_error(y_true=df[col_name1], y_pred=df[col_name2])
                ax[i, j].scatter(df[col_name1], df[col_name2])
                # x_left, x_right = ax[i, j].get_xlim()
                y_bottom, y_top = ax[i, j].get_ylim()
                ax[i, j].set_xlabel('{} with {}'.format(algo, ref))
                ax[i, j].text(1, y_top * 0.9, 'corr = {:.3f}'.format(corr[0, 1]))
                corr_list[i * 3 + j] = round(corr[0, 1], 3)
    fig.supylabel('Predicted cell fraction of {}'.format(f'{cell_type}'))
    fig.supxlabel('mean expression of marker genes in {} (n={})'.format(cancer_type, n_sample))
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{result_file_name_prefix}_in_{cancer_type}.png'), dpi=200)
    print('  Max cell fraction: {}'.format(max_cell_frac))
    return {'corr': corr_list}


def plot_latent_z(latent_z_file: pd.DataFrame, result_dir, file_name=None, label_type='cell_type'):
    """

    :param latent_z_file: latent_z with class
    :param result_dir:
    :param file_name:
    :param label_type:
    :return:
    """
    inx2cell_type = get_inx2cell_type()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cmap = plt.get_cmap('tab20')
    cmaplist = [cmap(15)] + [cmap(i) for i in range(1, 14, 2)] + [cmap(i) for i in [16, 17, 18, 19]]
    # cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 12)
    col_names = latent_z_file.columns.to_list()
    for inx in latent_z_file['class'].unique():
        if label_type == 'cell_type':
            label = inx2cell_type[inx]
        else:
            label = str(inx)
        current_part = latent_z_file.loc[latent_z_file['class'] == inx, :].copy()
        if (inx + 1) > len(cmaplist):
            inx = (inx % 7)
        ax.scatter(current_part[col_names[0]], current_part[col_names[1]],
                   color=cmaplist[inx + 1], label=label, alpha=.8)
    plt.legend()
    plt.xlabel(col_names[0])
    plt.ylabel(col_names[1])
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, file_name), dpi=200)


def plot_weights(w_file, selected_sample_ids: list, cell_types, gene_list,
                 para_const_obj, postfix, result_dir, sample_id, check_validity: bool = True):
    """
    compare the GEPs of each cell type before (ground truth) and after decon
    :param w_file: gene by cell type, TPM
    :param selected_sample_ids: selected sample ids during simulation bulk GEPs (random sampling)
    :param cell_types:
    :param gene_list:
    :param para_const_obj:
    :param postfix:
    :param result_dir:
    :param sample_id:
    :param check_validity
    :return:
    """
    w = read_df(w_file)
    w = w.loc[:, cell_types].copy()
    if check_validity:
        w = para_const_obj.check_validity(w=w.values.T, training=False)
    sc_after_decon = pd.DataFrame(w.T, index=gene_list, columns=cell_types)
    sc_before_decon = para_const_obj.sct_dataset_df.loc[selected_sample_ids, :].T
    sc_before_decon.columns = cell_types
    sc_before_decon = log2_transform(sc_before_decon)
    sc_after_decon = log2_transform(sc_after_decon)
    for ct in cell_types:
        plot_obj = ScatterPlot(x=sc_before_decon, y=sc_after_decon, postfix=postfix + '_' + ct)
        plot_obj.plot(show_columns={'x': ct, 'y': ct}, show_rmse=True, result_file_dir=result_dir,
                      x_label=f'Selected GEP of single cell type when simulation',
                      y_label=f'GEP after dcon ({sample_id}, {ct})')


def plot_weights2(w_file, selected_sample_ids: list, cell_types, gene_list,
                  sct_dataset_df, postfix, result_dir, sample_id, y: dict):
    """
    compare the GEPs of each cell type before (ground truth) and after decon
    :param w_file: gene by cell type, TPM
    :param selected_sample_ids: selected sample ids during simulation bulk GEPs (random sampling)
    :param cell_types:
    :param gene_list:
    :param postfix:
    :param result_dir:
    :param sample_id:
    :param sct_dataset_df: SCT (POS) dataset used when simulating bulk GEP, TPM
    :param y: cell proportion of each cell type in current sample, {'y_ture': {}, 'y_pred': {}}
    :return:
    """
    w = read_df(w_file)
    w = w.loc[:, cell_types].copy()
    sc_after_decon = pd.DataFrame(w, index=gene_list, columns=cell_types)
    sc_before_decon = sct_dataset_df.loc[selected_sample_ids, :].T
    sc_before_decon.columns = cell_types
    sc_before_decon = log2_transform(sc_before_decon)
    sc_after_decon = log2_transform(sc_after_decon)
    for ct in cell_types:
        y_true = y['y_true'][ct]
        y_pred = y['y_pred'][ct]
        plot_obj = ScatterPlot(x=sc_before_decon, y=sc_after_decon, postfix=postfix + '_' + ct)
        plot_obj.plot(show_columns={'x': ct, 'y': ct}, show_rmse=True, result_file_dir=result_dir,
                      x_label=f'Selected GEP of single cell type when simulation (y_true: {y_true:.3f}, {ct})',
                      y_label=f'GEP after dcon ({sample_id}, y_pred: {y_pred:.3f})')
