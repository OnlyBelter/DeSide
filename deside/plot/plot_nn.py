import os
# import importlib
# import umap
import numpy as np
import pandas as pd
import seaborn as sns
# import scipy.stats as stats
import matplotlib.pyplot as plt
# from joblib import dump, load
from .plot_gene import compare_exp_between_group
from ..utility import read_cancer_purity, check_dir, read_df, log2_transform, set_fig_style
from sklearn.metrics import median_absolute_error
# sns.set()
# sns.set(font_scale=1.5)
# plt.rcParams.update({'font.size': 20})
set_fig_style()


def plot_loss(history_df, output_dir=None, x_label='n_epoch', y_label='MSE', file_name=None):
    """
    :param history_df:
    :param output_dir:
    :param x_label:
    :param y_label:
    :param file_name:
    :return:
    """
    # sns.set(font_scale=1.5)
    plt.figure(figsize=(8, 6))
    if 'loss' in history_df.columns:
        plt.plot(history_df['epoch'], history_df['loss'], label='loss')
    if 'val_loss' in history_df.columns:
        plt.plot(history_df['epoch'], history_df['val_loss'], label='val_loss')
    if 'total_loss' in history_df.columns:
        plt.plot(history_df['epoch'], history_df['total_loss'], label='total_loss')
    if 'val_total_loss' in history_df.columns:
        plt.plot(history_df['epoch'], history_df['val_total_loss'], label='val_total_loss')
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label.upper())
    plt.tight_layout()
    if output_dir:
        if file_name is not None:
            plt.savefig(os.path.join(output_dir, file_name), dpi=200)
        else:
            plt.savefig(os.path.join(output_dir, 'loss.png'), dpi=200)
        plt.close()
    else:
        return plt


def plot_corr_two_columns(df: pd.DataFrame, output_dir: str, col_name1: str = 'CPE',
                          col_name2: str = 'cancer_cell', cancer_type: str = '', diagonal: bool = True,
                          predicted_by: str = None, font_scale: float = 1.5, scale_exp=False, update_figures=False):
    """
    Plot the relation between two columns in DataFrame `df`

    :param df: a dataFrame which contains CPE (cancer purity) and cancer_fraction

    :param output_dir: result folder

    :param col_name1: column name, such as CPE (cancer purity), x axis

    :param col_name2: column name, such as cancer cell fraction (predicted cancer purity), y axis

    :param cancer_type: mark x axis / y axis label

    :param diagonal: if plot diagonal

    :param predicted_by: model name

    :param font_scale: scale font size

    :param scale_exp: if scale all expression values to range [0, 10] by x_i/max(x) * 10

    :param update_figures: if update figures in output_dir

    :return: None
    """
    check_dir(output_dir)
    result_file_path = os.path.join(output_dir, '{}_vs_predicted_{}_proportion.png'.format(col_name1, col_name2))
    if (not os.path.exists(result_file_path)) or update_figures:
        # sns.set(font_scale=font_scale)
        # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # show Chinese characters
        plt.figure(figsize=(8, 8))
        df_col1 = df[col_name1].copy()
        df_col2 = df[col_name2].copy()
        if np.any(df_col1 > 2) and scale_exp:
            df_col1 = df_col1 / df_col1.max() * 10
        if np.any(df_col2 > 2) and scale_exp:
            df_col2 = df_col2 / df_col2.max() * 10

        corr = np.corrcoef(df_col1, df_col2)
        # print(corr)
        if np.isnan(corr[0, 1]):
            corr[0, 1] = 0  # when all predicted cell fraction are 0, set corr to 0
        # corr2 = stats.pearsonr(df[col_name1], df[col_name2])
        mae = median_absolute_error(y_true=df_col1, y_pred=df_col2)
        plt.scatter(df_col1, df_col2, s=8)
        plt.xlabel('{} ({})'.format(col_name1, cancer_type))
        # plt.xlabel('{} (头颈癌)'.format(col_name1))
        # plt.ylabel('预测 {} 比例 (样本数={})'.format(col_name2, df.shape[0]))
        x_left, x_right = plt.xlim()
        y_bottom, y_top = plt.ylim()
        if '_true' in col_name2:
            plt.ylabel('{} prop. (n={})'.format(col_name2, df.shape[0]))
        elif predicted_by:
            plt.ylabel('Predicted {} prop. by {} (n={})'.format(col_name2, predicted_by, df.shape[0]))
            # plt.ylabel('{}预测值 (样本数={})'.format(predicted_by, df.shape[0]))
        else:
            plt.ylabel('{} prop. (n={})'.format(col_name2, df.shape[0]))
        if 'CPE' in [col_name1, col_name2]:
            plt.text(0.05, 0.95, 'corr = {:.3f}'.format(corr[0, 1]))
            plt.text(0.05, 0.90, '$MAE$ = {:.3f}'.format(mae))
        elif ('CD8A' in [col_name1, col_name2]) or ('CD8A+CD8B' in [col_name1, col_name2]):
            plt.text(x_left + 1.5, y_top * 0.92, 'corr = {:.3f}'.format(corr[0, 1]))
        elif 'CD3E' in [col_name1, col_name2]:
            plt.text(x_left + 1.5, y_top * 0.92, 'corr = {:.3f}'.format(corr[0, 1]))
        elif 'y_pred' in [col_name1, col_name2]:
            plt.text(0.2, 0.92, 'corr = {:.3f}'.format(corr[0, 1]))
            plt.text(0.2, 0.85, '$MAE$ = {:.3f}'.format(mae))
            # plt.title()
        elif '_true' in col_name1 or '_true' in col_name2:
            plt.text(0.1, 0.92, 'corr = {:.3f}'.format(corr[0, 1]))
            plt.text(0.1, 0.85, '$MAE$ = {:.3f}'.format(mae))
        elif ('_marker_mean' in col_name1) or ('_marker_max' in col_name1):
            # compare mean expression of marker genes and predicted cell fraction
            plt.text(x_right * 0.05, y_top * 0.92, 'corr = {:.3f}'.format(corr[0, 1]))
        elif '_gene_signature_score' in col_name1:
            # compare mean expression of marker genes and predicted cell fraction
            plt.text(x_right * 0.05, y_top * 0.92, 'corr = {:.3f}'.format(corr[0, 1]))
        if diagonal:
            plt.plot([0, 1], [0, 1], linestyle='--', color='tab:gray')
        plt.tight_layout()
        plt.savefig(result_file_path, dpi=200)
        plt.close('all')
    else:
        print(f'   Using previous figure, {result_file_path}')


def plot_predicted_result(cell_frac_result_fp, bulk_exp_fp, cancer_type,
                          model_name, result_dir, cancer_purity_fp: str = None,
                          font_scale=2.0, update_figures=False):
    """
    Plot and evaluate predicted results of DeSide or Scaden model for TCGA data

    :param cell_frac_result_fp: the file path of predicted cell fraction

    :param bulk_exp_fp: the file path of bulk cell expression profile or pd.Dataframe, TPM, gene by sample

    :param cancer_type: only for naming or mark x / y label when plotting

    :param model_name: model name, DeSide or Scaden

    :param result_dir: where to save result

    :param cancer_purity_fp: estimated tumor purity for TCGA, download from
        Aran, D. et al., Nat Commun 6, 8971 (2015), Supplementary Data 1

    :param font_scale: scale font size

    :param update_figures: whether to update figures

    :return: None
    """
    y_pred = read_df(cell_frac_result_fp)  # cell fraction, sample by cell type
    # sep = get_sep(bulk_exp_fp)
    bulk_exp_cpm = read_df(bulk_exp_fp)
    # bulk_exp_cpm = log_exp2cpm(bulk_exp_log2cpm1p)

    # plot CD8 T cell fraction against CD8A expression value
    merged_df1 = y_pred.merge(bulk_exp_cpm.T, left_index=True, right_index=True)
    plot_corr_two_columns(df=merged_df1, col_name2='CD8 T', col_name1='CD8A',
                          predicted_by=model_name, font_scale=font_scale,
                          output_dir=result_dir, diagonal=False, cancer_type=cancer_type, update_figures=update_figures)

    if cancer_purity_fp is not None:
        # read cancer purity file
        cancer_purity = read_cancer_purity(cancer_purity_fp, sample_names=list(y_pred.index))
        merged_df = y_pred.merge(cancer_purity, left_index=True, right_index=True)
        # plot CPE vs cell fraction of cancer cell / 1-others
        if merged_df.shape[0] > 0:
            merged_df.to_csv(os.path.join(result_dir,
                                          f'cancer_purity_merged_{model_name}_predicted_result.csv'))
            plot_corr_two_columns(df=merged_df, col_name1='CPE', col_name2='Cancer Cells',
                                  output_dir=result_dir, font_scale=font_scale,
                                  cancer_type=cancer_type, predicted_by=model_name, update_figures=update_figures)
            # plot_corr_two_columns(df=merged_df, col_name1='CPE', col_name2='1-others',
            #                       output_dir=result_dir, font_scale=font_scale,
            #                       cancer_type=cancer_type, predicted_by=model_name)
        else:
            print('   There is no any samples in cancer purity about this cancer type ({})'.format(cancer_type))
    # plot cell fraction of each cell type before decon_cf
    y_pred['labels'] = 1
    # if 'DeSide' in model_name:
    #     y_pred.drop(columns='Cancer Cells', inplace=True)
    #     # remove labels and the last cell type
    #     cell_types = sorted(y_pred.columns.to_list())
    #     cell_types = [i for i in cell_types if i not in ['1-others', 'labels']]
    #     cell_types = cell_types + ['1-others']
    # else:
    #     y_pred.drop(columns='1-others', inplace=True)
    #     cell_types = sorted(y_pred.columns.to_list()[:-2]) + ['Cancer Cells']
    cell_types = sorted(y_pred.columns.to_list())
    cell_types = [i for i in cell_types if i not in ['1-others', 'labels']]
    # cell_types =
    print('   Cell types: ', ', '.join(cell_types))
    compare_exp_between_group(exp=y_pred, group_list=tuple(cell_types),
                              result_dir=result_dir, xlabel=f'Cell Type ({cancer_type})',
                              ylabel=f'Cell prop. predicted by {model_name}',
                              file_name='pred_cell_prop_before_decon.png', font_scale=font_scale - 0.4,
                              xticks_rotation=50)


def plot_paras(paras_file_path, vae_cla_model, latent_z_pos,
               current_cell_types, sampled_sc_id_file=None, sample_id: str = None, result_file=None):
    """
    plot parameters of regression model (deconvolved GEP) in latent z space
    :param paras_file_path: w, weights of regression model which represent valid GEPs for each cell type
    :param vae_cla_model:
    :param latent_z_pos: latent z for all training set of VAEClassifier model (encoder)
    :param current_cell_types
    :param sample_id: only provide if plot this sample
    :param sampled_sc_id_file: selected single cell id for each simulated GEP
    :param result_file:
    """

    latent_z_pos = read_df(latent_z_pos)
    paras = read_df(paras_file_path)
    if paras.shape[0] > paras.shape[1]:  # sample by cell type
        paras = paras.loc[:, current_cell_types].T
    else:
        paras = paras.loc[current_cell_types, :]
    paras = log2_transform(paras)

    _, _, latent_z_paras, _ = vae_cla_model.encoder_predict(paras.values)

    plt.figure(figsize=(8, 8))
    plt.scatter(latent_z_pos.loc[:, 'z1'], latent_z_pos.loc[:, 'z2'], color='gray')
    if sample_id is not None:  # the location of ground truth
        if sampled_sc_id_file is None:
            raise FileNotFoundError('sampled_sc_id_file should be provided with sample_id to plot this sample')
        sampled_sc_id_file = read_df(sampled_sc_id_file)
        current_sc_ids = sampled_sc_id_file.loc[sample_id, :].copy()
        sc_ids = dict(zip(current_sc_ids['cell_type'], current_sc_ids['selected_cell_id']))
        sc_id_list = [sc_ids[_] for _ in current_cell_types]
        plt.scatter(latent_z_pos.loc[sc_id_list, 'z1'], latent_z_pos.loc[sc_id_list, 'z2'], marker='x', color='red')
    plt.scatter(latent_z_paras[:, 0], latent_z_paras[:, 1], marker='*', color='green')
    if result_file is not None:
        plt.savefig(result_file, dpi=200)
    plt.close()


def plot_paras_all_cell_types(latent_z_paras_file, latent_z_pos_file, current_cell_types,
                              sampled_sc_id_file=None, sample_id: str = None, result_file=None):
    """
    plot parameters of regression model (deconvolved GEP) in latent z space
    :param latent_z_paras_file: latent z of all cell types which represent valid GEPs for each cell type
        - generated by VAEDecon model (encoder), n_cell_type x latent_dim
    :param latent_z_pos_file: latent z for all training set of VAEClassifier model (encoder)
    :param current_cell_types:
    :param sample_id: only provide if plot this sample
    :param sampled_sc_id_file: selected single cell id for each simulated GEP
    :param result_file:
    """

    latent_z_pos_file = read_df(latent_z_pos_file)
    if type(latent_z_paras_file) == str:
        paras = pd.read_csv(latent_z_paras_file, index_col=[0, 1])
    else:  # pd.Dataframe
        paras = latent_z_paras_file
    sample_inx = [(sample_id, ct) for ct in current_cell_types]
    latent_z_paras = paras.loc[sample_inx, :].copy()  # n_cell_type x latent_dim
    plt.figure(figsize=(8, 8))
    col_names = latent_z_pos_file.columns.to_list()
    plt.scatter(latent_z_pos_file.iloc[:, 0], latent_z_pos_file.iloc[:, 1], color='gray')
    if sample_id is not None:  # the location of ground truth
        if sampled_sc_id_file is None:
            raise FileNotFoundError('sampled_sc_id_file should be provided with sample_id to plot this sample')
        if type(sampled_sc_id_file) == str:
            sampled_sc_id_file = pd.read_csv(sampled_sc_id_file, index_col=[0, 1])
        # sample_inx = list(zip([sample_id] * len(current_cell_types), current_cell_types))
        current_sc_ids = sampled_sc_id_file.loc[sample_inx, 'selected_cell_id'].to_list()
        # sc_ids = dict(zip(current_sc_ids['cell_type'], current_sc_ids['selected_cell_id']))
        # sc_id_list = [sc_ids[_] for _ in current_cell_types]
        plt.scatter(latent_z_pos_file.loc[current_sc_ids, col_names[0]],
                    latent_z_pos_file.loc[current_sc_ids, col_names[1]],
                    marker='x', color='red')
    plt.scatter(latent_z_paras.loc[:, col_names[0]], latent_z_paras.loc[:, col_names[1]], marker='*', color='green')
    plt.xlabel(f'{col_names[0]} of latent space')
    plt.ylabel(f'{col_names[1]} of latent space')
    if result_file is not None:
        plt.savefig(result_file, dpi=200)
    plt.close()
