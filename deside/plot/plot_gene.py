import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as an
import seaborn as sns
import scipy.stats as stats
from ..utility import check_dir, cell_type2abbr, set_fig_style
from matplotlib import rcParams
import matplotlib.pyplot as plt
from ..utility.pub_func import log_exp2cpm, cal_exp_by_gene_list
# sns.set(palette='muted', font_scale=1.5)

set_fig_style()


def plot_single_gene_exp(gene_name, exp_sorted, output_dir='',
                         exp_type='', x_label='', y_label='gene_expression'):
    """
    plot single gene expression profile across all samples (sorted)
    :param gene_name:
    :param exp_sorted: sorted expression profile, gene x sample
    :param output_dir:
    :param exp_type:
    :param x_label:
    :param y_label:
    :return:
    """
    plt.figure(figsize=(8, 6))
    n_sample = exp_sorted.shape[1]
    if gene_name in exp_sorted.index:
        gene_exp = exp_sorted.loc[gene_name]
        # print(gene_exp)
        plt.plot(np.arange(n_sample), gene_exp.values)
        plt.title('{} expression'.format(gene_name))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, '{}_exp_{}.png'.format(gene_name, exp_type)), dpi=200)
        # plt.show()
        plt.close()
    else:
        print('gene not exist')


def plot_gene_pdf(gene_name, exp_df, output_dir=''):
    """
    plot PDF of a given gene across all samples
    :param gene_name:
    :param exp_df:
    :param output_dir:
    :return:
    """
    plt.figure(figsize=(8, 6))
    gene_exp = exp_df.loc[gene_name, :]
    norm_dis = stats.norm(np.mean(gene_exp), np.std(gene_exp))
    _x = np.linspace(gene_exp.min(), gene_exp.max(), 101)
    _pdf = norm_dis.pdf(_x)
    # plt.plot(_x, _pdf, 'b-', label='$\mu={},\ \sigma={}$'.format(np.mean(gene_exp), np.std(gene_exp)))
    plt.hist(gene_exp.values, bins=20, density=True)
    plt.plot(_x, _pdf, 'r--', label=r'$N(\mu={:.2f},\ \sigma={:.2f})$'.format(np.mean(gene_exp), np.std(gene_exp)))
    plt.title('PDF of {} expression'.format(gene_name))
    plt.xlabel('TMM after normalized by gene length')
    plt.ylabel('Prob')
    plt.legend()
    plt.tight_layout()
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if output_dir:
            plt.savefig(os.path.join(output_dir, '{}_exp_pdf.png'.format(gene_name)), dpi=200)
        # plt.show()
    else:
        return plt


def plot_emt_gene_exp(exp_df, output_dir, exp_type='before_deco', sorted_by='CD8A'):
    """
    plot 3 genes expression in sorted expression profile (all bulk or purified samples)
    :param exp_df: a expression dataframe, sorted
    :param output_dir: output dir
    :param exp_type: before purified (deconvolution) / after purified
    :param sorted_by: sort by expression values based on this gene
    """
    plot_single_gene_exp('CD8A', exp_sorted=exp_df, output_dir=output_dir, exp_type=exp_type,
                         x_label='Sorted by {}'.format(sorted_by), y_label='TMM after normalized by gene length')
    plot_single_gene_exp('VIM', exp_sorted=exp_df, output_dir=output_dir, exp_type=exp_type,
                         x_label='Sorted by {}'.format(sorted_by), y_label='TMM after normalized by gene length')
    if ('CDH1' in exp_df.index) and ('VIM' in exp_df.index):
        plot_single_gene_exp('CDH1', exp_sorted=exp_df, output_dir=output_dir, exp_type=exp_type,
                             x_label='Sorted by {}'.format(sorted_by), y_label='TMM after normalized by gene length')

        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(exp_df.shape[1]), exp_df.loc['CDH1'] / exp_df.loc['VIM'])
        plt.title('CDH1/VIM')
        plt.xlabel('Samples sorted by TMM of {}'.format(sorted_by))
        plt.ylabel('TMM after normalized by gene length')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '{}_exp_{}.png'.format('CDH1_VIM', exp_type)), dpi=200)
        plt.close()


def compare_exp_between_group(exp: pd.DataFrame, group_list: tuple, max_threshold=1000, result_dir='.',
                              xlabel: str = 'Gene name', ylabel: str = 'Gene expression of HNSCC in TCGA',
                              file_name: str = 'compare_gene_exp_in_groups.png', font_scale=1.2,
                              xticks_rotation=0):
    """
    comparing the expression value of multiple genes in all samples
    :param exp: sample by gene, must contain group label ('labels') for each sample
    :param group_list: genes or cell types need to plot
    :param max_threshold: max expression values to plot, set all values which >= this value to this value
    :param result_dir:
    :param xlabel
    :param ylabel
    :param file_name
    :param font_scale: scaling font size
    :param xticks_rotation:
    :return:
    """
    # sns.set(font_scale=font_scale)
    exp_one_by_one = []
    if 'CD8A+CD8B' in group_list:
        exp['CD8A+CD8B'] = exp['CD8A'] + exp['CD8B']
    for g_name in group_list:
        _current_exp = exp.loc[:, [g_name, 'labels']]
        _current_exp.rename(columns={g_name: 'g_exp'}, inplace=True)
        _current_exp['g_name'] = g_name
        exp_one_by_one.append(_current_exp)
    exp_flatten = pd.concat(exp_one_by_one)
    exp_flatten.loc[exp_flatten['g_exp'] > max_threshold, 'g_exp'] = max_threshold
    plt.figure(figsize=(8, 6))
    # Draw a nested boxplot to show bills by day and time
    # sns.set_color_codes('bright')
    sample_labels = list(exp['labels'].unique())
    hue = None
    if len(sample_labels) >= 2:
        hue = 'labels'
    ax = sns.boxplot(x="g_name", y="g_exp", palette=sns.color_palette("muted"),
                     hue=hue, data=exp_flatten, whis=[0, 100], hue_order=sample_labels, order=group_list)

    # Add in points to show each observation, http://seaborn.pydata.org/examples/horizontal_boxplot.html
    sns.stripplot(x="g_name", y="g_exp", data=exp_flatten,
                  size=3, color=".3", linewidth=0, hue=hue, dodge=True, hue_order=sample_labels, ax=ax)
    sns.despine(offset=10, trim=True, left=True)

    if len(sample_labels) >= 2:
        handles, labels = ax.get_legend_handles_labels()
        n_half_label = int(len(labels) / 2)
        plt.legend(handles[0:n_half_label], labels[0:n_half_label], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # remove the top and right ticks
    ax.tick_params(axis='x', which='both', top=False)
    ax.tick_params(axis='y', which='both', right=False)
    plt.xticks(rotation=xticks_rotation)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, file_name), dpi=200)
    plt.close('all')


def plot_cd8_marker(bulk_exp, cancer_type, markers=('CD8A', 'CD8B', 'TIGIT'), result_dir=None):
    """
    compare marker genes in different group (separate all samples by labels)
    :param bulk_exp: sample by gene, should contain 'labels' columns
    :param cancer_type:
    :param markers: marker genes for a specific cell type, such as CD8
    :param result_dir:
    :return:
    """
    cd8_related_exp = bulk_exp.loc[:, list(markers) + ['labels']].copy()
    cd8_related_exp['CD8A+CD8B'] = cd8_related_exp['CD8A'] + cd8_related_exp['CD8B']
    plt.figure(figsize=(8, 6))
    # labels = db.labels_
    for i, x_reduced in cd8_related_exp.groupby(['labels']):
        plt.scatter(x_reduced.loc[x_reduced['labels'] == i, 'CD8A+CD8B'],
                    x_reduced.loc[x_reduced['labels'] == i, 'TIGIT'], label=i)
    plt.xlabel('CD8A+CD8B')
    plt.ylabel('TIGIT')
    corr = cd8_related_exp.corr().loc['CD8A+CD8B', 'TIGIT']
    print(cd8_related_exp.corr())
    # plt.text(cd8_related_exp.loc[:, 'CD8A+CD8B'].max()*0.05,
    #          cd8_related_exp.loc[:, 'PRF1'].max()*0.95, 'corr={:.2f}'.format(corr))
    plt.title('cancer type: {}, corr={:.2f}'.format(cancer_type, corr))
    plt.legend()
    plt.tight_layout()
    if result_dir is not None:
        plt.savefig(os.path.join(result_dir, 'bulk_cd8_marker_2D.png'), dpi=200)
    plt.close()


def plot_gene_exp(gene_names, bulk_exp, output_dir='', sort_by='',
                  exp_type='', x_label='', y_label='gene_expression'):
    """
    plot multiple gene expression profiles across all samples (sorted)
    :param gene_names: a list of gene names
    :param bulk_exp: gene expression profiles (GEPs), gene x sample
    :param output_dir:
    :param sort_by: a gene name in GEPs, sort GEPs by the expression value of this gene
    :param exp_type: TPM / log2tpm1p
    :param x_label: sample id
    :param y_label: TPM of gene
    :return:
    """
    plt.figure(figsize=(8, 6))
    n_sample = bulk_exp.shape[1]
    if sort_by and sort_by in bulk_exp.index:
        bulk_exp = bulk_exp.sort_values(by=sort_by, axis=1)
    for gene_name in gene_names:
        if gene_name in bulk_exp.index:
            gene_exp = bulk_exp.loc[gene_name]
            # print(gene_exp)
            plt.plot(np.arange(n_sample), gene_exp.values, label=gene_name)
        else:
            print('Gene {} does not exist ...'.format(gene_name))
            # plt.title('{} expression'.format(gene_name))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    gene_name_str = '-'.join(gene_names)
    if output_dir:
        plt.savefig(os.path.join(output_dir, '{}_exp_{}.png'.format(gene_name_str, exp_type)), dpi=200)
    # plt.show()
    plt.close()


def plot_marker_gene_in_cell_type(n3000_dataset, cell_type_exp: str = '', cell_type_marker: str = '',
                                  cell_type2marker_genes: dict = None, result_dir: str = ''):
    """
    plot the expression values of marker genes (cell_type_marker) in specific cell type (cell_type_exp)
    :param n3000_dataset: N3000 dataset in .h5ad format
    :param cell_type_exp: to specify expression values of this cell type
    :param cell_type_marker: to specify marker genes of this cell type
    :param cell_type2marker_genes: a dict of cell type to corresponding marker genes {'': [], }
    :param result_dir:
    :return:
    """
    if result_dir is not None:
        check_dir(result_dir)
    assert type(n3000_dataset) == an.AnnData
    if cell_type2marker_genes is None:
        cell_type2marker_genes = {}
    # marker_genes = []
    if cell_type_marker in cell_type2marker_genes:
        marker_genes = cell_type2marker_genes[cell_type_marker]
    else:
        raise KeyError(f'Cell type {cell_type_marker} in "cell_type_marker" do not exist in "cell_type2marker_genes"')
    current_ds = None
    cell_types_in_n3000 = list(n3000_dataset.obs.cell_type.unique())
    if cell_type_exp in cell_types_in_n3000:
        current_ds = n3000_dataset[n3000_dataset.obs.cell_type == cell_type_exp, :].copy()
    print(current_ds)
    current_ds_df = pd.DataFrame(data=current_ds.X.A, index=current_ds.obs.index, columns=current_ds.var.index)
    current_ds_cpm = log_exp2cpm(current_ds_df)
    current_marker_exp = current_ds_cpm.loc[:, marker_genes].copy()
    current_marker_exp[current_marker_exp >= 9999] = 9999
    current_marker_exp.boxplot(figsize=(8, 6))
    plt.xlabel('Marker gene of {}'.format(cell_type_marker))
    plt.ylabel('Gene expression of {} (CPM)'.format(cell_type_exp))
    if result_dir:
        plt.savefig(os.path.join(result_dir, f'boxplot_marker_gene_of_{cell_type_marker}_in_{cell_type_exp}.png'),
                    dpi=200)
    plt.close()
    # t_cell_marker_exp = None
    # if cell_type_exp in ['CD8 T', 'CD4 T', 'NK']:
    t_cell_markers = cell_type2marker_genes['T Cells']
    t_cell_marker_exp = current_ds_cpm.loc[:, t_cell_markers].copy()
    t_cell_marker_mean = cal_exp_by_gene_list(exp_df=current_ds_cpm, gene_list=t_cell_markers)
    # t_cell_marker_mean[t_cell_marker_mean < 1] = 1

    current_marker_mean = cal_exp_by_gene_list(exp_df=current_ds_cpm, gene_list=marker_genes, min_exp_value=1)
    t_vs_current = t_cell_marker_mean / current_marker_mean
    plt.figure(figsize=(8, 6))
    _range = None
    # if (cell_type_exp in ['CD4 T', 'CD8 T']) and (cell_type_exp == cell_type_marker):
    #     _range = [0, 20]
    # elif (cell_type_exp == 'CD8 T') and (cell_type_marker == 'NK'):
    #     _range = [0, 20]
    plt.hist(t_vs_current, bins=50, range=_range)
    plt.xlabel(f'T Cell marker / {cell_type_marker} marker (mean exp) in {cell_type_exp}')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'hist_T_vs_{cell_type_marker}_in_{cell_type_exp}.png'.replace(' ', '_')),
                dpi=200)
    plt.close()

    if t_cell_marker_exp is not None:
        t_cell_marker_exp.boxplot(figsize=(8, 6))
        plt.xlabel('Marker gene of T Cells')
        plt.ylabel('Gene expression of {} (CPM)'.format(cell_type_exp))
        if result_dir:
            plt.savefig(os.path.join(result_dir, f'boxplot_marker_gene_of_T_Cells_in_{cell_type_exp}.png'),
                        dpi=200)
        plt.close()


def plot_marker_exp(dataset_id: str = '', single_cell_dataset: an.AnnData = None,
                    cell_type2markers: dict = None, max_exp: float = 0.0, exp_range: tuple = (50, 320),
                    cell_types: list = None, result_dir: str = None, groupby: str = 'leiden',
                    font_scale: float = 1.5) -> pd.DataFrame:
    """
    plot the boxplot of marker genes for each cell type (12 cell types) of each sub-cluster (leiden) in a sc dataset
    :param dataset_id: the index of current dataset
    :param single_cell_dataset: single cell dataset in .h5ad format
    :param cell_type2markers: {cell_type: [marker genes]}
    :param cell_types: 12 sorted cell types
    :param max_exp: max expression to plot
    :param exp_range: (low_exp_threshold, middle_exp_threshold)
    :param result_dir: datasets/single_cell/marker_gene_expression/before_filtering
    :param groupby: index of groupby, such as leiden
    :param font_scale:
    :return:
    """
    cd4_cd8_ratio = 10
    sns.set(palette='muted', font_scale=font_scale)
    assert result_dir is not None
    assert type(single_cell_dataset) == an.AnnData
    check_dir(result_dir)
    sc.settings.figdir = result_dir

    # print(single_cell_dataset)
    cell_type2exp_value_collector_total = []
    quantile_file_path = os.path.join(result_dir, f'{dataset_id}_quantile.csv')
    if not os.path.exists(quantile_file_path):
        current_dataset_quantile = []
        for gi, group in single_cell_dataset.obs.groupby([groupby]):
            print(gi, group.shape)
            cell_type2exp_value_collector_current_gi = []
            current_part_data = single_cell_dataset[single_cell_dataset.obs.index.isin(group.index), :].copy()
            gene_exp = pd.DataFrame(current_part_data.X.A, columns=current_part_data.var.index,
                                    index=current_part_data.obs.index)
            gene_exp = log_exp2cpm(gene_exp)  # convert the expression values from log2(CPM + 1) to CPM
            gi_name = str(gi).replace(' / ', '_')
            result_file_path = os.path.join(result_dir, 'marker_exp_' + gi_name.replace(' ', '_') + '.png')

            # plot for each sub-cluster
            cell_type2marker_exp = {}
            cell_type2exp_flatten = {}
            if cell_types is None:
                cell_types = list(cell_type2markers.keys())

            id2cell_type = {}
            cell_type2quantile = {}
            for k, _cell_type in enumerate(cell_types):
                if _cell_type in cell_type2markers.keys():
                    id2cell_type[k] = _cell_type
                    current_marker_gene = cell_type2markers[_cell_type]
                    cell_type2marker_exp[_cell_type] = gene_exp.loc[:, current_marker_gene].copy()  # a dataFrame
                    method = 'mean'
                    if _cell_type in ['B Cells', 'CD4 T']:
                        method = 'max'
                    cell_type2marker_exp[_cell_type][f'm_{method}'] = \
                        cal_exp_by_gene_list(gene_exp, gene_list=current_marker_gene, method=method, min_exp_value=1)
                    _values = cell_type2marker_exp[_cell_type][f'm_{method}'].values
                    cell_type2quantile[_cell_type] = [round(np.quantile(_values, 0.1), 2),
                                                      round(np.quantile(_values, 0.25), 2),
                                                      round(np.quantile(_values, 0.50), 2),
                                                      round(np.quantile(_values, 0.75), 2),
                                                      round(np.quantile(_values, 0.90), 2),
                                                      round(np.quantile(_values, 0.95), 2)]
                    gene2exp = {}
                    current_exp_value = cell_type2marker_exp[_cell_type]  # a dataFrame
                    if max_exp > 0:
                        current_exp_value[current_exp_value > max_exp] = max_exp
                    for col in current_exp_value.columns:
                        # for each columns in this dataFrame (marker genes and m_mean/m_max)
                        gene2exp[col] = current_exp_value.loc[:, [col]].copy()
                        gene2exp[col]['col_name'] = col
                        gene2exp[col].rename(columns={col: 'exp_value'}, inplace=True)
                    cell_type2exp_flatten[_cell_type] = pd.concat(list(gene2exp.values()))
                    current_exp_value.rename(columns={f'm_{method}': f'm_{method}_{_cell_type}'}, inplace=True)
                    cell_type2exp_value_collector_current_gi.append(current_exp_value.copy())
            ct2e_gi = pd.concat(cell_type2exp_value_collector_current_gi, axis=1)
            ct2e_gi['leiden'] = gi
            ct2e_gi['m_max_cd4/mean_cd8'] = ct2e_gi['m_max_CD4 T'] / ct2e_gi['m_mean_CD8 T']
            ct2e_gi['m_cd4/m_cd8 group'] = 'middle'
            ct2e_gi.loc[ct2e_gi['m_max_cd4/mean_cd8'] > cd4_cd8_ratio, 'm_cd4/m_cd8 group'] = 'high'
            ct2e_gi.loc[ct2e_gi['m_max_cd4/mean_cd8'] < (1 / cd4_cd8_ratio), 'm_cd4/m_cd8 group'] = 'low'
            cell_type2exp_value_collector_total.append(ct2e_gi.copy())
            if ('CD8 T' in gi) or ('CD4 T' in gi) or ('NK' in gi):
                t_exp = ct2e_gi.loc[:, ['m_max_CD4 T', 'm_mean_CD8 T', 
                                        'm_max_cd4/mean_cd8', 'm_cd4/m_cd8 group']].copy()

                g = sns.JointGrid()
                sns.scatterplot(x='m_mean_CD8 T', y='m_max_CD4 T', data=t_exp, s=5,
                                ax=g.ax_joint, hue='m_cd4/m_cd8 group',
                                hue_order=['high', 'middle', 'low'])

                sns.boxplot(x=t_exp['m_mean_CD8 T'], palette=sns.color_palette("muted"), whis=[0, 100],
                            showfliers=False, ax=g.ax_marg_x)
                sns.stripplot(x=t_exp['m_mean_CD8 T'], size=2, color=".4", linewidth=0, dodge=True, ax=g.ax_marg_x)
                sns.boxplot(y=t_exp['m_max_CD4 T'], palette=sns.color_palette("muted"), whis=[0, 100],
                            showfliers=False, ax=g.ax_marg_y)
                sns.stripplot(y=t_exp['m_max_CD4 T'], size=2, color=".4", linewidth=0, dodge=True, ax=g.ax_marg_y)
                # g.plot(sns.scatterplot, sns.boxplot, s=3, alpha=.6)
                g.ax_joint.axhline(exp_range[0], ls='--', linewidth=0.5, color='r')
                g.ax_joint.axhline(exp_range[1], ls='--', linewidth=0.5, color='r')
                g.ax_joint.axvline(exp_range[0], ls='--', linewidth=0.5, color='r')
                g.ax_joint.axvline(exp_range[1], ls='--', linewidth=0.5, color='r')

                g.ax_joint.set_ylabel('markers\' max of CD4 T')
                g.ax_joint.set_xlabel('markers\' mean of CD8 T')

                plt.savefig(os.path.join(result_dir, f'CD8_marker_vs_CD4_marker_in_{gi_name}.png'), dpi=200)
                plt.close()

                # plot the ratio of marker max of CD4 / marker mean of CD8
                current_part_data.obs['m_max_cd4/mean_cd8'] = t_exp['m_max_cd4/mean_cd8'].copy()
                current_part_data.obs.loc[t_exp['m_max_cd4/mean_cd8'] > 2, 'm_max_cd4/mean_cd8'] = 2
                rcParams['figure.figsize'] = 10, 10

                sc.pl.umap(current_part_data, color='m_max_cd4/mean_cd8', legend_loc='on data',
                           legend_fontsize='small',
                           title=f'm_max_cd4/mean_cd8 - {gi_name} in {dataset_id}',
                           frameon=False, save=f'_{gi_name}_in_{dataset_id}.png', show=False)
                plt.close('all')

            fig, axes = plt.subplots(4, 3, sharey='row', figsize=(15, 16))
            for i in range(4):
                for j in range(3):
                    current_cell_type = id2cell_type[i*3 + j]
                    ax = sns.boxplot(x='col_name', y='exp_value', palette=sns.color_palette("muted"), whis=[0, 100],
                                     data=cell_type2exp_flatten[current_cell_type], showfliers=False, ax=axes[i, j])
                    # ax.tick_params(labelsize=11)
                    # Add in points to show each observation, http://seaborn.pydata.org/examples/horizontal_boxplot.html
                    sns.stripplot(x='col_name', y='exp_value', data=cell_type2exp_flatten[current_cell_type],
                                  size=2, color=".4", linewidth=0, dodge=True, ax=axes[i, j])
                    _quantile_str = ', '.join([str(i) for i in cell_type2quantile[current_cell_type][1:-1]])
                    axes[i, j].set_xlabel(current_cell_type + f'({_quantile_str})')
            fig.supxlabel(f'leiden_cluster: {gi}, ({gene_exp.shape[0]} cells)')
            plt.tight_layout()
            if result_file_path is not None:
                plt.savefig(result_file_path, dpi=200)
            else:
                plt.show()
            plt.close('all')
            cell_type2quantile_df = pd.DataFrame.from_dict(cell_type2quantile, orient='index',
                                                           columns=['q_10', 'q_25', 'q_50', 'q_75', 'q_90', 'q_95'])
            cell_type2quantile_df['leiden'] = gi
            current_dataset_quantile.append(cell_type2quantile_df)
        current_dataset_marker_exp = pd.concat(cell_type2exp_value_collector_total)
        current_dataset_marker_exp['dataset_id'] = dataset_id
        current_dataset_marker_exp.to_csv(os.path.join(result_dir, f'{dataset_id}_marker_exp.csv'))

        # plot the ratio of marker max of CD4 / marker mean of CD8
        current_dataset_marker_exp.loc[current_dataset_marker_exp['m_max_cd4/mean_cd8'] > 2,
                                       'm_max_cd4/mean_cd8'] = 2
        # current_dataset_marker_exp = current_dataset_marker_exp.loc[single_cell_dataset.obs.index,
        #                                                             ['m_max_cd4/mean_cd8']]
        single_cell_dataset.obs['m_max_cd4/mean_cd8'] = current_dataset_marker_exp.loc[single_cell_dataset.obs.index,
                                                                                       'm_max_cd4/mean_cd8']
        rcParams['figure.figsize'] = 10, 10

        sc.pl.umap(single_cell_dataset, color='m_max_cd4/mean_cd8', legend_loc='on data',
                   legend_fontsize='small',
                   title=f'm_max_cd4/mean_cd8 in {dataset_id}',
                   frameon=False, save=f'_m_max_cd4_mean_cd8_in_{dataset_id}.png', show=False)
        sc.pl.umap(single_cell_dataset, color='leiden', legend_loc='on data',
                   legend_fontsize='small',
                   title=f'leiden (annotated sub-cluster) in {dataset_id}',
                   frameon=False, save=f'_leiden_in_{dataset_id}.png', show=False)
        plt.close('all')

        current_dataset_quantile_df = pd.concat(current_dataset_quantile)
        current_dataset_quantile_df['dataset_id'] = dataset_id
        current_dataset_quantile_df.to_csv(quantile_file_path, index_label='cell_type')
    else:
        print(f'   Using previous result: {quantile_file_path}')
        current_dataset_quantile_df = pd.read_csv(quantile_file_path, index_col=0)
    return current_dataset_quantile_df


def plot_marker_ratio(marker_ratio: pd.DataFrame, fig_file_dir: str, dataset_name: str):
    """
    plot the marker gene ratio of each cell type
    """
    _group_melted = pd.melt(marker_ratio, value_vars=list(marker_ratio.columns))
    if '_' in marker_ratio.columns.to_list()[0]:  # 'Cancer Cells_marker_mean'
        _group_melted['variable'] = _group_melted['variable'].map(lambda x: x.split('_')[0])
    _group_melted['variable'] = _group_melted['variable'].map(cell_type2abbr)
    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(x="variable", y="value", data=_group_melted, cut=0, scaler='width')
    plt.suptitle(f'Ratio of marker genes in {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_file_dir, f'marker_genes_{dataset_name}.png'), dpi=200)
