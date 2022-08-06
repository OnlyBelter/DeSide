import os
import numpy as np
import pandas as pd
import seaborn as sns
from .pub_func import print_df
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
# from emtdecode.plot import plot_single_gene_exp
sns.set()
# from emtdecode.utility import print_df


def sort_sample_before_deco(bulk_exp_file_path, cell_fraction_file_path=None,
                            sort_by='CD8A', result_dir=None,
                            two_groups=False, sorted_bulk_file_name=None):
    """
    sort bulk samples by a single gene before deconvolution
    :param bulk_exp_file_path: gene by sample file
    :param cell_fraction_file_path:
    :param sort_by: CD8A / t_cell_fraction
    :param two_groups: half is low, half is high after sorted by the expression of CD8A
    :param sorted_bulk_file_name
    :param result_dir:
    :return:
    """
    # bulk_exp_type = 'before_deco'  # before deconvolution
    # output_file_name_before_corrected = 'cancer_exp_before_deco_sorted.txt'
    before_corrected_sorted_file_path = os.path.join(result_dir, sorted_bulk_file_name)
    if not os.path.exists(before_corrected_sorted_file_path):
        # bulk_exp_file_path = os.path.join(current_output_dir, 'clustering', )
        sep = '\t'
        with open(bulk_exp_file_path, 'r') as h:
            first_line = h.readline()
            if ',' in first_line:
                sep = ','
        bulk_exp = pd.read_csv(bulk_exp_file_path, index_col=0, sep=sep)  # gene by sample
        # print('>>> bulk cancer expression before sorted info')
        # print_df(bulk_exp)
        cls_file_name = 'phenotype_labels.cls'

        sample_label = np.array(['middle'] * bulk_exp.shape[1])
        if sort_by.lower() == 'cd8a':
            # sort samples by the expression of CD8A
            cd8_exp_tmp = bulk_exp.loc[['CD8A', 'CD8B'], :].copy()
            # cd8_exp_tmp.loc['a+b'] = cd8_exp_tmp.loc['CD8A'] + cd8_exp_tmp.loc['CD8B']
            # from high to low
            cd8_exp_tmp_sorted = cd8_exp_tmp.sort_values(by=['CD8A'], axis=1, ascending=False)
            bulk_exp_sorted = bulk_exp.loc[:, cd8_exp_tmp_sorted.columns]
            sample_label[cd8_exp_tmp_sorted.loc['CD8A'] <= 1] = 'low'
            sample_label[cd8_exp_tmp_sorted.loc['CD8A'] >= 8] = 'high'
        elif sort_by.lower() == 'cd8a+cd8b':
            cd8_exp_tmp = bulk_exp.loc[['CD8A', 'CD8B'], :].copy()
            cd8_exp_tmp.loc['a+b'] = cd8_exp_tmp.loc['CD8A'] + cd8_exp_tmp.loc['CD8B']
            # from high to low
            cd8_exp_tmp_sorted = cd8_exp_tmp.sort_values(by=['a+b'], axis=1, ascending=False)
            bulk_exp_sorted = bulk_exp.loc[:, cd8_exp_tmp_sorted.columns]
            sample_label[cd8_exp_tmp_sorted.loc['a+b'] <= 1] = 'low'
            sample_label[cd8_exp_tmp_sorted.loc['a+b'] >= 8] = 'high'
        elif sort_by.lower() == 't_cell_fraction':
            cell_fraction = pd.read_csv(cell_fraction_file_path, index_col=0)
            # from high to low
            cell_fraction_sorted = cell_fraction.sort_values(by=['T cell'], ascending=False)
            bulk_exp_sorted = bulk_exp.loc[:, cell_fraction_sorted.index]
            sample_label[cell_fraction_sorted['T cell'] <= 0.03] = 'low'
            sample_label[cell_fraction_sorted['T cell'] >= 0.08] = 'high'
        else:
            raise ValueError('Only CD8A, CD8A+CD8B and T_CELL_FRACTION allowed')
        if two_groups:
            n_half_sample = int(np.ceil(np.median(range(bulk_exp.shape[1]))))
            sample_label[:n_half_sample] = 'high'
            sample_label[n_half_sample:] = 'low'
        # print('>>> bulk exp sorted info')
        # print_df(bulk_exp_sorted)

        # plot
        # plot_helper(exp_df=bulk_exp_sorted, output_dir=result_dir, exp_type=bulk_exp_type)

        bulk_exp_sorted2 = bulk_exp_sorted.copy()
        bulk_exp_sorted2['DESCRIPTION'] = 'NA'
        # bulk_exp_sorted2.head(2)
        bulk_exp_sorted2.to_csv(before_corrected_sorted_file_path, sep='\t',
                                columns=['DESCRIPTION'] + bulk_exp_sorted.columns.to_list(), index_label='NAME')

        # print(len(sample_label))
        cls_file_path = os.path.join(result_dir, cls_file_name)
        # if not os.path.exists(cls_file_path):
        with open(cls_file_path, 'w') as f_handle:
            n_label = len(np.unique(sample_label))
            f_handle.write(' '.join([str(i) for i in [len(sample_label), n_label, 1]]) + '\n')
            if two_groups:
                f_handle.write(' '.join(['#', 'high', 'low']) + '\n')
            else:
                f_handle.write(' '.join(['#', 'high', 'middle', 'low']) + '\n')
            f_handle.write(' '.join(sample_label) + '\n')
    else:
        cols = list(pd.read_csv(before_corrected_sorted_file_path, nrows=1, sep='\t'))
        bulk_exp_sorted = pd.read_csv(before_corrected_sorted_file_path, sep='\t',
                                      usecols=[i for i in cols if i != 'DESCRIPTION'], index_col=0)
    return bulk_exp_sorted


def sort_sample_after_deco(purified_exp_file_path, before_deco_sorted_file_path,
                           tmm_norm_factor_fp, result_file_path=None):
    """

    :param purified_exp_file_path: genes by samples, CPM (non-log space), separate by ","
    :param before_deco_sorted_file_path: sorted bulk expression profile and contains "DESCRIPTION" column
        genes by samples, separated by "\t"
    :param tmm_norm_factor_fp: TMM normalization factor file path, samples by factors
    :param result_file_path:
    :return:
    """

    if not os.path.exists(result_file_path):
        before_deco_sorted_bulk_exp = pd.read_csv(before_deco_sorted_file_path, index_col=0, sep='\t')
        purified_cancer_cell_exp = pd.read_csv(purified_exp_file_path, index_col=0, sep=',')
        # TMM normalization
        norm_factor = pd.read_csv(tmm_norm_factor_fp, index_col=0)
        norm_factor = norm_factor.loc[purified_cancer_cell_exp.columns, :].copy()
        purified_cancer_cell_tmm = purified_cancer_cell_exp / np.hstack(norm_factor['norm.factors'])
        purified_cancer_cell_tmm = purified_cancer_cell_tmm.round(3)
        purified_cancer_cell_tmm['DESCRIPTION'] = 'NA'
        purified_cancer_cell_tmm_sorted = purified_cancer_cell_tmm.loc[:, before_deco_sorted_bulk_exp.columns]
        purified_cancer_cell_tmm_sorted.to_csv(result_file_path, sep='\t', index_label='NAME')


def classify_sample_by_tsne_clustering(bulk_exp_file_path, markers: list,
                                       result_dir=None, tsne_file_name=''):
    """

    :param bulk_exp_file_path:
    :param markers: the marker genes of CD8
    :param result_dir:
    :param tsne_file_name:
    :return:
    """
    result_file_path = os.path.join(result_dir, tsne_file_name)
    if not os.path.exists(result_file_path):
        sep = '\t'
        with open(bulk_exp_file_path, 'r') as h:
            first_line = h.readline()
            if ',' in first_line:
                sep = ','
        bulk_exp = pd.read_csv(bulk_exp_file_path, index_col=0, sep=sep)

        cd8_related_exp = bulk_exp.loc[markers, :].T
        print(cd8_related_exp.shape)
        cd8_related_exp_scaled = StandardScaler().fit_transform(cd8_related_exp)

        tsne = TSNE(n_components=2, n_jobs=5, learning_rate=200, random_state=42,
                    n_iter=2000, init='pca', verbose=1, perplexity=7)  # 9
        x_reduced = tsne.fit_transform(cd8_related_exp_scaled)
        x_reduced = pd.DataFrame(data=x_reduced, index=cd8_related_exp.index)
        cd8_related_exp['tsne1'] = x_reduced.loc[:, 0]
        cd8_related_exp['tsne2'] = x_reduced.loc[:, 1]
        db = DBSCAN(eps=12, min_samples=8).fit(x_reduced)

        plt.figure(figsize=(8, 6))
        labels = db.labels_
        for i in np.unique(labels):
            plt.scatter(x_reduced.loc[labels == i, 0], x_reduced.loc[labels == i, 1], label=i)
        plt.xlabel('t-SNE1')
        plt.ylabel('t-SNE2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'bulk_cd8_marker_tsne.png'), dpi=200)

        cd8_related_exp['labels'] = labels
        cd8_related_exp.to_csv(result_file_path)
    else:
        cd8_related_exp = pd.read_csv(result_file_path, index_col=0)
    return cd8_related_exp


def classify_sample_by_2d_clustering(bulk_exp_file_path, markers: list = ('CD8A', 'CD8B', 'PRF1'),
                                     result_dir=None, result_file_name='',
                                     cancer_type='', replace=False):
    """
    classify samples by (CD8A + CD8B) and PRF1
    :param bulk_exp_file_path: gene by sample
    :param markers: the marker genes of CD8
    :param cancer_type:
    :param replace: rewrite previous result
    :param result_dir:
    :param result_file_name:
    :return:
    """
    result_file_path = os.path.join(result_dir, result_file_name)
    if replace or (not os.path.exists(result_file_path)):
        sep = '\t'
        with open(bulk_exp_file_path, 'r') as h:
            first_line = h.readline()
            if ',' in first_line:
                sep = ','
        bulk_exp = pd.read_csv(bulk_exp_file_path, index_col=0, sep=sep)

        cd8_related_exp = bulk_exp.loc[markers, :].T
        cd8_related_exp['CD8A+CD8B'] = cd8_related_exp['CD8A'] + cd8_related_exp['CD8B']
        cd8_related_exp['(CD8A+CD8B)*PRF1'] = cd8_related_exp['CD8A+CD8B'] * cd8_related_exp['PRF1']
        print(cd8_related_exp.shape)
        # cd8_related_exp_scaled = StandardScaler().fit_transform(cd8_related_exp)
        #
        # tsne = TSNE(n_components=2, n_jobs=5, learning_rate=200, random_state=42,
        #             n_iter=2000, init='pca', verbose=1, perplexity=7)  # 9
        # x_reduced = tsne.fit_transform(cd8_related_exp_scaled)
        # x_reduced = cd8_related_exp.loc[:, ['CD8A+CD8B', 'PRF1']]
        # cd8_related_exp['tsne1'] = x_reduced.loc[:, 0]
        # cd8_related_exp['tsne2'] = x_reduced.loc[:, 1]
        # db = DBSCAN(eps=1.8, min_samples=15).fit(x_reduced)
        # kmeans = KMeans(n_clusters=3, random_state=42).fit(cd8_related_exp['(CD8A+CD8B)*PRF1'].values.reshape(-1,1))
        # db = DBSCAN(eps=5, random_state=42).fit(cd8_related_exp['(CD8A+CD8B)*PRF1'].values.reshape(-1, 1))
        # cd8_related_exp['labels'] = kmeans.labels_
        cd8_related_exp['labels'] = 'middle'
        quantile_1_3 = cd8_related_exp['(CD8A+CD8B)*PRF1'].quantile(1/3)
        quantile_2_3 = cd8_related_exp['(CD8A+CD8B)*PRF1'].quantile(1/3)
        cd8_related_exp.loc[cd8_related_exp['(CD8A+CD8B)*PRF1'] > max(quantile_2_3, 120), 'labels'] = 'high'
        cd8_related_exp.loc[cd8_related_exp['(CD8A+CD8B)*PRF1'] < min(quantile_1_3, 10), 'labels'] = 'low'
        cd8_related_exp.loc[((cd8_related_exp['CD8A+CD8B'] > np.power(max(quantile_2_3, 120), 0.5)) |
                             (cd8_related_exp['PRF1'] > np.power(max(quantile_2_3, 120), 0.5))) &
                            (cd8_related_exp['labels'] == 'middle'), 'labels'] = 'high'
        cd8_related_exp.loc[((cd8_related_exp['CD8A+CD8B'] > np.power(max(quantile_1_3, 10), 0.25)) |
                             (cd8_related_exp['PRF1'] > np.power(max(quantile_1_3, 10), 0.75))) &
                            (cd8_related_exp['labels'] == 'low'), 'labels'] = 'middle'

        plt.figure(figsize=(8, 6))
        # labels = db.labels_
        for i, x_reduced in cd8_related_exp.groupby(['labels']):
            plt.scatter(x_reduced.loc[x_reduced['labels'] == i, 'CD8A+CD8B'],
                        x_reduced.loc[x_reduced['labels'] == i, 'PRF1'], label=i)
        plt.xlabel('CD8A+CD8B')
        plt.ylabel('PRF1')
        corr = cd8_related_exp.corr().loc['CD8A+CD8B', 'PRF1']
        print(cd8_related_exp.corr())
        # plt.text(cd8_related_exp.loc[:, 'CD8A+CD8B'].max()*0.05,
        #          cd8_related_exp.loc[:, 'PRF1'].max()*0.95, 'corr={:.2f}'.format(corr))
        plt.title('cancer type: {}, corr={:.2f}'.format(cancer_type, corr))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'bulk_cd8_marker_2D.png'), dpi=200)
        plt.close()
        # cd8_related_exp['labels'] = labels
        cd8_related_exp.to_csv(result_file_path)
    else:
        cd8_related_exp = pd.read_csv(result_file_path, index_col=0)
    return cd8_related_exp


def classify_by_markers_before_deco(bulk_exp_file_path, classified_file_path=None,
                                    result_dir=None, cls_file_name='phenotype_labels.cls',
                                    sorted_exp_file_name='cancer_exp_before_deco_sorted.txt',
                                    purified_cancer_cell_file_path=None):
    """
    sort bulk samples by a single gene before deconvolution
    :param bulk_exp_file_path:
    :param classified_file_path: classified by DBSCAN after t-SNE of only marker genes
    :param purified_cancer_cell_file_path:
        only for removing samples which have been removed in purified cancer cell files
    :param cls_file_name:
    :param sorted_exp_file_name:
    :param result_dir:
    :return:
    """
    # bulk_exp_type = 'before_deco'  # before deconvolution
    # cls_file_name = ''
    # output_file_name_before_corrected =
    before_corrected_sorted_file_path = os.path.join(result_dir, sorted_exp_file_name)

    if not os.path.exists(before_corrected_sorted_file_path):
        # bulk_exp_file_path = os.path.join(current_output_dir, 'clustering', )
        sep = '\t'
        with open(bulk_exp_file_path, 'r') as h:
            first_line = h.readline()
            if ',' in first_line:
                sep = ','
        bulk_exp = pd.read_csv(bulk_exp_file_path, index_col=0, sep=sep)
        purified_cc = None
        if os.path.exists(purified_cancer_cell_file_path):
            purified_cc = pd.read_csv(purified_cancer_cell_file_path, index_col=0, sep='\t')
            purified_cc.columns = [i.replace('.', '-') for i in purified_cc.columns]
        if purified_cc is not None:
            bulk_exp = bulk_exp.loc[:, bulk_exp.columns.isin(purified_cc.columns)]
        print('>>> bulk cancer expression before sorted info')
        print_df(bulk_exp)

        class_info = pd.read_csv(classified_file_path, index_col=0)
        class_info = class_info.loc[class_info['labels'] != -1, :].copy()
        if purified_cc is not None:
            class_info = class_info.loc[class_info.index.isin(purified_cc.columns), :]
        print_df(class_info)
        class_info = class_info.sort_values(by=['labels'])
        sample_label = class_info['labels'].values
        bulk_exp_sorted = bulk_exp.loc[:, class_info.index]  # gene by sample
        print('>>> bulk exp sorted info')
        print_df(bulk_exp_sorted)

        # plot
        # plot_helper(exp_df=bulk_exp_sorted, output_dir=result_dir, exp_type=bulk_exp_type)

        bulk_exp_sorted2 = bulk_exp_sorted.copy()
        bulk_exp_sorted2['DESCRIPTION'] = 'NA'
        # bulk_exp_sorted2.head(2)
        bulk_exp_sorted2.to_csv(before_corrected_sorted_file_path, sep='\t',
                                columns=['DESCRIPTION'] + bulk_exp_sorted.columns.to_list(), index_label='NAME')

        # print(len(sample_label))
        cls_file_path = os.path.join(result_dir, cls_file_name)
        # if not os.path.exists(cls_file_path):
        unique_labels = [str(i) for i in list(np.unique(sample_label))]
        with open(cls_file_path, 'w') as f_handle:
            f_handle.write(' '.join([str(i) for i in [len(sample_label), len(unique_labels), 1]]) + '\n')
            f_handle.write(' '.join(['#'] + unique_labels) + '\n')
            f_handle.write(' '.join([str(i) for i in sample_label]) + '\n')


def table2gct(table_file_path, result_file_path):
    """
    convert dataFrame to GCT format for expression profile
    http://software.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats
    :param table_file_path: gene by sample
    :param result_file_path:
    :return:
    """
    # first_line = None
    sep = '\t'
    with open(table_file_path, 'r') as f:
        first_line = f.readline()
        print(first_line)
        if ',' in first_line:
            sep = ','
    exp_df = pd.read_csv(table_file_path, index_col=0, sep=sep)
    with open(result_file_path, 'w') as f:
        f.write('#1.2' + '\n')
        f.write('{}\t{}\n'.format(exp_df.shape[0], exp_df.shape[1]))
    exp_df.index.name = 'NAME'
    sample_names = exp_df.columns.to_list().copy()
    exp_df['Description'] = 'NA'
    exp_df = exp_df.loc[:, ['Description'] + sample_names]
    exp_df.to_csv(result_file_path, mode='a', sep='\t', header=True)


if __name__ == '__main__':
    pass
